"""Realism-match runner: sim vs real, per instrument, FOMs 1-7.

Compares the image-library cohort against matched simulated frames on the
seven figures of merit:

1. Sky-region noise statistics (paired-difference sigma, noise vs signal,
   sky power spectrum), with the sky patches restricted to those on dark sky
   in both the real and the matched frame (:func:`pair_sky`).
2. Star-cutout radial profiles / encircled energy where the cohort has
   star frames.
3. Normalized limb profiles and 10-90% rise widths, binned by resolution
   and phase.
4. Ring-edge profiles and rise widths (Cassini).
5. Exposure-stratified dynamic-range statistics (never unstratified).
6. Artifact rates vs the catalog defaults.
7. Technique-diagnostic values on matched scene/frame pairs -- READ-ONLY:
   reported for the record and never a tuning target, because FOM 7 is
   built from the navigator's own outputs and tuning against it would
   re-admit circularity through parameter fitting.

Every real frame gets one matched sim frame (same instrument chain, same
exposure, same content class; see :mod:`tests.integration.sim_realism_scenes`),
and both sides run identical extraction (:mod:`tests.integration.sim_realism_support`).
The scalar divergence per figure of merit is the Wasserstein-1 distance (W1,
quantile-clipped, real-IQR-normalized); curve kinds (sky PSD, star / limb / ring
profiles) get the density-W1 between the frame-averaged curves.  Where a cohort cannot
support a statistic, the summary labels it rather than reporting a fake
distribution.

Run standalone (not under pytest)::

    python -m tests.integration.sim_realism [--instrument coiss_calib_nac]
        [--max-frames N] [--skip-fom7]

Requires ``PDS3_HOLDINGS_DIR`` (and star-catalog env vars for the star
frames).  Full-cohort runtime is dominated by the navigator's feature
extraction on the real frames; see the dev guide's realism section for the
measured figure.  Figures land in ``docs/simulator_report/_figures/`` and
the JSON summary in ``tests/integration/realism_results/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs import ObsSnapshotInst
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instrument_catalog import realism_cohort_instruments
from spindoctor.sim.instrument_navigator import resolve_sim_inst_config
from spindoctor.sim.instrument_resolvers import (
    resolve_calibrated_sky_if,
    resolve_dark_sky_ceiling_if,
    resolve_data_units,
)
from spindoctor.sim.realism.artifact_rates import (
    ArtifactRates,
    split_stationary_spikes,
)
from spindoctor.sim.realism.divergence import (
    W1Result,
    cohort_support,
    w1_between_densities,
    w1_divergence,
)
from spindoctor.sim.realism.dynamic_range import stratify_by_exposure
from spindoctor.sim.realism.noise import SkyPatch, dark_sky_patches, noisy_patches
from spindoctor.support.types import NDArrayFloatType
from tests.integration.checkout_guard import require_checkout_package
from tests.integration.sidecar import LibraryRoot, Sidecar, load_sidecar
from tests.integration.sim_realism_scenes import matched_scene
from tests.integration.sim_realism_support import (
    FrameSamples,
    add_sky_samples,
    extract_feature_samples,
    extract_pixel_samples,
    frame_settings_reader,
    prepare_frame_features,
)

__all__ = [
    'FrameRecord',
    'InstrumentComparison',
    'RealismResults',
    'SkyPair',
    'SkyPairStatus',
    'pair_sky',
    'pool_frame_pairs',
    'run_realism_match',
    'twins_have_sky',
]

# (mission, camera) -> sim instrument whose signal chain and units match the
# cohort's products: each instrument config entry's realism_cohort.
INSTRUMENT_FOR: dict[tuple[str, str], str] = realism_cohort_instruments()


def saturation_level(instrument: str) -> float:
    """The full-scale level in cohort units, from the navigator's instrument block.

    The level feeds the saturation statistics and the FOM 2 saturated-star filter.

    Parameters:
        instrument: The sim instrument a cohort is matched against.

    Returns:
        The block's ``noise.saturation_dn`` for a ``raw_dn`` product; NaN for a
        calibrated one, whose I/F units define no fixed level.
    """
    if resolve_data_units(instrument) != 'raw_dn':
        return math.nan
    return float(resolve_sim_inst_config(DEFAULT_CONFIG, instrument)['noise']['saturation_dn'])


# Scene classes -> NavModel globs for the feature FOMs (the sim-side models
# are registered as body_sim:/rings_sim:, so both spellings appear).  Classes
# absent here contribute pixel statistics only.  body_irregular and
# below_resolution_body are deliberately excluded from FOM 3: an ellipsoid
# model's limb against an irregular body measures shape-model error, not
# forward-model fidelity.
_STAR_MODELS = ['stars']
_BODY_MODELS = ['body:*', 'body_sim:*']
_RING_MODELS = ['rings:*', 'rings_sim:*']
_TITAN_MODELS = ['titan:*', 'titan_sim:*']
MODELS_FOR_CLASS: dict[str, list[str]] = {
    'star_dominated': _STAR_MODELS,
    'stars_plus_body': _STAR_MODELS,
    'faint_stars': _STAR_MODELS,
    'one_bright_star_no_body': _STAR_MODELS,
    'two_bright_stars_no_body': _STAR_MODELS,
    'body_full_fov': _BODY_MODELS,
    'body_partial_overflow': _BODY_MODELS,
    'high_phase_terminator': _BODY_MODELS,
    'multi_body': _BODY_MODELS,
    'ring_only_flat': _RING_MODELS,
    'ring_only_curved': _RING_MODELS,
    'ring_plus_body': _RING_MODELS + _BODY_MODELS,
    # Titan frames: the haze model's own envelope, so the limb FOM measures
    # forward-model fidelity on a hazy body rather than on a hard limb the
    # body model would predict for it.
    'titan_haze': _TITAN_MODELS,
}

# FOM 7 matched pairs: one frame per class per instrument, capped so the
# runner's navigation cost stays a handful of frames per instrument.
FOM7_CLASSES: tuple[str, ...] = ('star_dominated', 'body_full_fov', 'ring_only_flat')
FOM7_MAX_PAIRS_PER_INSTRUMENT = 3

# Which sample-kind and curve-kind prefixes belong to which figure of merit (for
# support labeling and reporting).
FOM_KIND_PREFIXES: dict[str, tuple[str, ...]] = {
    'fom1_noise': ('sky_sigma', 'sky_mean_minus_floor', 'signal_sigma', 'sky_psd'),
    'fom2_psf': ('star_ee50', 'star_ee80', 'star_profile'),
    'fom3_limb': ('limb_width', 'limb_profile'),
    'fom4_ring': ('ring_edge_width', 'ring_radial_profile'),
    'fom5_dynrange': ('frac_saturated', 'frac_near_floor', 'signal_p95', 'signal_p99'),
    'fom6_artifacts': ('artifact_',),
}


@dataclass
class FrameRecord:
    """One cohort frame's matching metadata.

    Parameters:
        image_id: Sidecar image_id.
        scene_class: Sidecar primary scene tag.
        instrument: Matched sim instrument name.
        exposure_sec: Frame exposure (sidecar, falling back to the label).
        stratum: Exposure-stratum label.
        offset_vu: Operator-verified ground-truth offset ``(dv, du)``.
        diameter_px: Largest body's predicted apparent diameter (limb
            frames; default where no body model was built).
        phase_angle_deg: That body's phase angle.
        frame_shape_vu: The frame's (v, u) shape.
        filter_name: The frame's filter combination (``'CL1+CL2'``) on a
            camera the sim's filter catalog covers, else None.
        gain_state: The frame's gain state when its mission's label reader
            reads one, else None.
        quantization: ``'sqrt_lut'`` for a frame downlinked through a
            companding table, which the camera's default downlink table models,
            else None.
    """

    image_id: str
    scene_class: str
    instrument: str
    exposure_sec: float
    stratum: str
    offset_vu: tuple[float, float]
    diameter_px: float = 150.0
    phase_angle_deg: float = 45.0
    frame_shape_vu: tuple[int, int] = (512, 512)
    filter_name: str | None = None
    gain_state: int | None = None
    quantization: str | None = None


class SkyPairStatus(StrEnum):
    """What became of one frame pair's FOM 1 sky patches.

    A frame's sky patches here are the darkest quarter of the uniform patches whose
    noise sigma is positive.

    ``COMPARED``: both frames kept dark-sky patches, which entered the pooled sky
    statistics.  ``DROPPED``: neither side added any, because a frame kept no
    dark-sky patch, or because the matched frame has no sky patches while other
    matched frames of the instrument have some.  ``NOT_JUDGED``: the instrument
    has no dark-sky ceiling (its products are raw DN, or its entry sets none), so
    every patch of both frames entered the pools.  ``NO_REAL_SKY``: the real frame
    has no sky patches, so neither side added any.  ``NO_SIM_SKY``: no matched
    frame of the instrument has sky patches, so the real frame's dark-sky patches
    (every sky patch, on an instrument without a ceiling) entered the real pool
    alone.
    """

    COMPARED = 'compared'
    DROPPED = 'dropped'
    NOT_JUDGED = 'not_judged'
    NO_REAL_SKY = 'no_real_sky'
    NO_SIM_SKY = 'no_sim_sky'


@dataclass(frozen=True)
class SkyPair:
    """How one real frame's FOM 1 sky patches compared with its matched frame's.

    Parameters:
        image_id: The real frame's image_id.
        quantization: The real frame's downlink quantization (``'sqrt_lut'`` for a
            frame downlinked through a companding table), else None.
        status: What became of the pair's sky patches.
        real_level: The median mean (native units) of the real frame's kept
            patches, or of all its sky patches when it kept none; NaN without sky
            patches.
        sim_level: The same for the matched frame.
        n_real_patches: The real frame's sky patches.
        n_sim_patches: The matched frame's sky patches.
        n_real_kept: The real sky patches kept: those at or below the dark-sky
            ceiling, or all of them on an instrument without one.  It is 0 when the
            pair is dropped because its matched frame has no sky patches.
        n_sim_kept: The matched frame's sky patches kept, likewise.  It is 0 when
            either frame has no sky patches.
        real_sigma: The median noise sigma of the real patches that entered the
            pool, NaN when none did.
        sim_sigma: The same for the matched frame.
    """

    image_id: str
    quantization: str | None
    status: SkyPairStatus
    real_level: float
    sim_level: float
    n_real_patches: int
    n_sim_patches: int
    n_real_kept: int
    n_sim_kept: int
    real_sigma: float
    sim_sigma: float

    @property
    def compared(self) -> bool:
        """Whether both frames kept dark-sky patches."""
        return self.status is SkyPairStatus.COMPARED

    @property
    def real_added(self) -> bool:
        """Whether the real frame added sky samples to the pool."""
        return self.status in (
            SkyPairStatus.COMPARED,
            SkyPairStatus.NOT_JUDGED,
            SkyPairStatus.NO_SIM_SKY,
        )

    @property
    def sim_added(self) -> bool:
        """Whether the matched frame added sky samples to the pool."""
        return self.status in (SkyPairStatus.COMPARED, SkyPairStatus.NOT_JUDGED)

    @property
    def sigma_ratio(self) -> float:
        """The simulated over real median sky sigma, NaN unless both sides entered."""
        if not (np.isfinite(self.sim_sigma) and np.isfinite(self.real_sigma)):
            return float('nan')
        if self.real_sigma <= 0.0:
            return float('nan')
        return self.sim_sigma / self.real_sigma


@dataclass
class InstrumentComparison:
    """Pooled real and sim samples plus divergences for one instrument.

    Parameters:
        instrument: Sim instrument name.
        records: The cohort frames compared.
        real: Pooled real-side samples and curves.
        sim: Pooled sim-side samples and curves.
        divergences: Per-sample-kind W1 results.
        curve_divergences: Per-curve-kind density-W1 results between the
            frame-averaged real and sim curves (sky PSD, star profile,
            limb / ring profiles).
        limb_bins_real_only: FOM 3 strata populated only on the real side
            (no sim counterpart; excluded from the pooled statistic and
            disclosed in the report).
        limb_bins_sim_only: FOM 3 strata populated only on the sim side.
        frame_kinds: Per side (``'real'``, ``'sim'``), each frame's sample and
            curve kinds once its sky has been paired, in the records' order.
        fom_frames: Per side, the number of frames contributing to each figure
            of merit.
        fom_support: Cohort-support label per figure of merit, from the side
            with fewer contributing frames.
        kind_frames: Per side, the number of frames contributing to each sample
            and curve kind.
        kind_support: Cohort-support label per sample and curve kind, from the
            side with fewer contributing frames.
        fom7_rows: Read-only technique-diagnostic rows from the matched
            navigation pairs.
        sky_pairs: Per frame, how its FOM 1 sky patches compared with the
            matched frame's.
    """

    instrument: str
    records: list[FrameRecord] = field(default_factory=list)
    real: FrameSamples = field(default_factory=FrameSamples)
    sim: FrameSamples = field(default_factory=FrameSamples)
    divergences: dict[str, W1Result] = field(default_factory=dict)
    curve_divergences: dict[str, W1Result] = field(default_factory=dict)
    limb_bins_real_only: list[str] = field(default_factory=list)
    limb_bins_sim_only: list[str] = field(default_factory=list)
    frame_kinds: dict[str, list[frozenset[str]]] = field(
        default_factory=lambda: {'real': [], 'sim': []}
    )
    fom_frames: dict[str, dict[str, int]] = field(default_factory=dict)
    fom_support: dict[str, str] = field(default_factory=dict)
    kind_frames: dict[str, dict[str, int]] = field(default_factory=dict)
    kind_support: dict[str, str] = field(default_factory=dict)
    fom7_rows: list[dict[str, Any]] = field(default_factory=list)
    sky_pairs: list[SkyPair] = field(default_factory=list)
    real_rates: list[ArtifactRates] = field(default_factory=list)
    sim_rates: list[ArtifactRates] = field(default_factory=list)
    spike_split_real: tuple[float, float] = (float('nan'), float('nan'))
    spike_split_sim: tuple[float, float] = (float('nan'), float('nan'))


@dataclass
class RealismResults:
    """The full realism-match output across instruments.

    Parameters:
        comparisons: Per-instrument comparison, keyed by instrument name.
        runtime_sec: Wall-clock runtime of the match.
    """

    comparisons: dict[str, InstrumentComparison]
    runtime_sec: float


def _resolve_image_url(url: str) -> Any:
    """Resolve a sidecar's opaque pds3:// URL against the holdings root."""
    from filecache import FCPath

    if url.startswith('pds3://'):
        holdings_root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
        return FCPath(f'{holdings_root}/{url[len("pds3://") :]}')
    return FCPath(url)


def _load_real_obs(sidecar: Sidecar) -> ObsSnapshotInst:
    """Load the real observation behind one sidecar."""
    from tests.integration.test_autonomous_nav import _MISSION_TO_OBS_CLASS

    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    return cast(ObsSnapshotInst, obs_class.from_file(_resolve_image_url(sidecar.image_url)))


def discover_cohort() -> dict[str, list[Sidecar]]:
    """Group the image library's sidecars by matched sim instrument."""
    cohort: dict[str, list[Sidecar]] = {}
    for path in LibraryRoot().discover_sidecar_paths():
        sidecar = load_sidecar(path)
        instrument = INSTRUMENT_FOR.get((sidecar.mission, sidecar.camera))
        if instrument is not None:
            cohort.setdefault(instrument, []).append(sidecar)
    for sidecars in cohort.values():
        sidecars.sort(key=lambda s: s.image_id)
    return cohort


def _stratum_label(exposure_sec: float | None) -> str:
    """The exposure-stratum label for one frame."""
    strata = stratify_by_exposure([exposure_sec])
    return next(iter(strata))


def _frame_samples_real(
    sidecar: Sidecar, instrument: str
) -> tuple[FrameSamples, FrameRecord, ArtifactRates]:
    """Extract every sample the FOMs need from one real frame."""
    obs = _load_real_obs(sidecar)
    exposure = sidecar.exposure_time_sec
    if exposure is None:
        exposure = float(getattr(obs, 'texp', 1.0))
    stratum = _stratum_label(exposure)
    scene_class = sidecar.primary_scene_tag
    offset = (sidecar.ground_truth.offset_dv_px, sidecar.ground_truth.offset_du_px)
    record = FrameRecord(
        image_id=sidecar.image_id,
        scene_class=scene_class,
        instrument=instrument,
        exposure_sec=float(exposure),
        stratum=stratum,
        offset_vu=offset,
        frame_shape_vu=(int(obs.data_shape_vu[0]), int(obs.data_shape_vu[1])),
    )
    reader = frame_settings_reader(sidecar.mission)
    if reader is not None:
        # Matching content, not tuning: the sim renders the frame's own filter
        # throughput, gain state, and downlink table, so its DN-to-I/F scale
        # and quantization step are the real product's.
        settings = reader(obs, image_id=sidecar.image_id, instrument=instrument)
        record.filter_name = settings.filter_name
        record.gain_state = settings.gain_state
        record.quantization = settings.quantization
    saturation = saturation_level(instrument)
    samples, rates = extract_pixel_samples(
        np.asarray(obs.data, dtype=np.float64),
        exposure_stratum=stratum,
        saturation_level=saturation,
    )
    only_models = MODELS_FOR_CLASS.get(scene_class)
    if only_models is not None:
        image, features, model_metadata = prepare_frame_features(obs, only_models=only_models)
        diameter, phase = _largest_body(model_metadata)
        if diameter is not None and phase is not None:
            record.diameter_px = diameter
            record.phase_angle_deg = phase
        samples.merge(
            extract_feature_samples(
                image,
                features,
                model_metadata,
                offset_vu=offset,
                extfov_margin_vu=obs.extfov_margin_vu,
                saturation_level=saturation,
            )
        )
    return samples, record, rates


def _largest_body(
    model_metadata: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None]:
    """Diameter and phase of the largest body model, if any."""
    best: tuple[float, float] | None = None
    for name, meta in model_metadata.items():
        if not name.startswith('body:'):
            continue
        diameter = meta.get('predicted_diameter_px')
        phase = meta.get('phase_angle_deg')
        if diameter is None or phase is None:
            continue
        if best is None or float(diameter) > best[0]:
            best = (float(diameter), float(phase))
    if best is None:
        return None, None
    return best


def _sim_obs_for_record(record: FrameRecord) -> tuple[ObsSim, dict[str, Any]]:
    """Render the matched sim frame for one cohort record."""
    scene = matched_scene(
        record.image_id,
        record.scene_class,
        record.instrument,
        record.exposure_sec,
        size_vu=record.frame_shape_vu,
        diameter_px=record.diameter_px,
        phase_angle_deg=record.phase_angle_deg,
        filter_name=record.filter_name,
        gain_state=record.gain_state,
        quantization=record.quantization,
    )
    obs = ObsSim.from_file(f'/tmp/sim_realism_{record.image_id}.yaml', sim_params=scene)
    return obs, scene


def _frame_samples_sim(record: FrameRecord) -> tuple[FrameSamples, ArtifactRates]:
    """Extract the matched sim frame's samples (same machinery as real)."""
    obs, _scene = _sim_obs_for_record(record)
    saturation = saturation_level(record.instrument)
    samples, rates = extract_pixel_samples(
        np.asarray(obs.data, dtype=np.float64),
        exposure_stratum=record.stratum,
        saturation_level=saturation,
    )
    only_models = MODELS_FOR_CLASS.get(record.scene_class)
    if only_models is not None:
        image, features, model_metadata = prepare_frame_features(obs, only_models=only_models)
        samples.merge(
            extract_feature_samples(
                image,
                features,
                model_metadata,
                offset_vu=(0.0, 0.0),  # matched scenes plant zero offset
                extfov_margin_vu=obs.extfov_margin_vu,
                saturation_level=saturation,
            )
        )
    return samples, rates


def _diagnostic_row(instrument: str, image_id: str, side: str, result: Any) -> list[dict[str, Any]]:
    """Flatten one NavResult's per-technique diagnostics into rows."""
    rows: list[dict[str, Any]] = []
    for tech in result.per_technique:
        row: dict[str, Any] = {
            'instrument': instrument,
            'image_id': image_id,
            'side': side,
            'technique': tech.technique_name,
            'confidence': float(tech.confidence),
        }
        diag = tech.diagnostics
        if dataclasses.is_dataclass(diag) and not isinstance(diag, type):
            for key, value in dataclasses.asdict(diag).items():
                if isinstance(value, (int, float, bool)) or value is None:
                    row[key] = value
        rows.append(row)
    return rows


def _run_fom7(
    records: list[FrameRecord], sidecars_by_id: dict[str, Sidecar]
) -> list[dict[str, Any]]:
    """Navigate a handful of matched pairs and collect diagnostics (read-only)."""
    rows: list[dict[str, Any]] = []
    chosen: list[FrameRecord] = []
    for scene_class in FOM7_CLASSES:
        for record in records:
            if record.scene_class == scene_class:
                chosen.append(record)
                break
        if len(chosen) >= FOM7_MAX_PAIRS_PER_INSTRUMENT:
            break
    for record in chosen:
        real_obs = _load_real_obs(sidecars_by_id[record.image_id])
        real_result = NavOrchestrator(build_models_for_obs(real_obs)).navigate(real_obs)
        rows.extend(_diagnostic_row(record.instrument, record.image_id, 'real', real_result))
        sim_obs, _scene = _sim_obs_for_record(record)
        sim_result = NavOrchestrator(build_models_for_obs(sim_obs)).navigate(sim_obs)
        rows.extend(_diagnostic_row(record.instrument, record.image_id, 'sim', sim_result))
    return rows


def _median_sigma(patches: list[SkyPatch]) -> float:
    """The median noise sigma of some sky patches, NaN for none."""
    if not patches:
        return float('nan')
    return float(np.median([p.sigma for p in patches]))


def _sky_level(kept: list[SkyPatch], patches: list[SkyPatch]) -> float:
    """A frame's reported sky level: the median mean of its kept patches.

    A frame that kept no patch reports the median mean of all its sky patches
    instead, and one without sky patches reports NaN.
    """
    chosen = kept or patches
    if not chosen:
        return float('nan')
    return float(np.median([p.mean for p in chosen]))


def _sky_patches(frame: FrameSamples) -> list[SkyPatch]:
    """A frame's sky patches whose noise sigma is positive."""
    return noisy_patches(frame.sky.patches) if frame.sky is not None else []


def twins_have_sky(sims: list[FrameSamples]) -> bool:
    """Whether any of an instrument's matched frames has sky patches.

    Parameters:
        sims: The instrument's matched frames' samples.

    Returns:
        True when at least one matched frame has a sky patch whose noise sigma is
        positive.
    """
    return any(_sky_patches(sim) for sim in sims)


def _dark_sky_ceiling_level(instrument: str) -> float | None:
    """The highest mean a dark-sky patch may have on ``instrument``, or None.

    The level is the calibrated sky residual plus the instrument's dark-sky ceiling.
    It is None on a raw-DN instrument and on a calibrated one that sets no ceiling.
    """
    if resolve_data_units(instrument) != 'calibrated_if':
        return None
    ceiling = resolve_dark_sky_ceiling_if(instrument)
    if ceiling is None:
        return None
    return resolve_calibrated_sky_if(instrument) + ceiling


def pair_sky(
    record: FrameRecord, real: FrameSamples, sim: FrameSamples, *, twins_with_sky: bool
) -> SkyPair:
    """Add one frame pair's FOM 1 sky samples from the patches on dark sky in both frames.

    A frame's sky patches are the darkest quarter of the uniform patches whose noise
    sigma is positive (a constant tile, such as zero-filled missing data, has a
    sigma of 0).  On a
    frame filled with rings or a body they lie on that content, and comparing them
    with the other frame's dark sky would set signal-dependent noise against sky
    noise.  So on an instrument that sets a dark-sky ceiling, a patch of either
    frame counts only when its mean lies at or below the calibrated sky residual
    plus that ceiling, and the pair adds sky samples only when both frames keep at
    least one patch.  On an instrument without a ceiling every patch is added.

    Parameters:
        record: The real frame's record.
        real: The real frame's samples; the sky samples that enter the pool are
            added to it.
        sim: The matched frame's samples, likewise.
        twins_with_sky: Whether any matched frame of the instrument has sky
            patches.  When none has, the real frame's dark-sky patches are added
            alone (all of its sky patches on an instrument without a ceiling);
            otherwise a pair whose matched frame has none is dropped.

    Returns:
        The pair's sky comparison.
    """
    real_patches = _sky_patches(real)
    sim_patches = _sky_patches(sim)
    ceiling_level = _dark_sky_ceiling_level(record.instrument)
    real_kept: list[SkyPatch] = []
    sim_kept: list[SkyPatch] = []
    added_real: list[SkyPatch] = []
    added_sim: list[SkyPatch] = []
    if not real_patches:
        status = SkyPairStatus.NO_REAL_SKY
    elif not sim_patches and not twins_with_sky:
        # No matched frame of the instrument has sky, so the real frame's sky is
        # recorded alone; it still has to lie on dark sky where a ceiling is set.
        real_kept = (
            real_patches if ceiling_level is None else dark_sky_patches(real_patches, ceiling_level)
        )
        if real_kept:
            status = SkyPairStatus.NO_SIM_SKY
            added_real = real_kept
        else:
            status = SkyPairStatus.DROPPED
    elif not sim_patches:
        status = SkyPairStatus.DROPPED
    elif ceiling_level is None:
        # A raw-DN frame's zero point is its bias pedestal, which the real product
        # may have removed while the matched frame keeps the simulator's, so the
        # levels can't be compared.  A calibrated camera whose entry sets no ceiling
        # is handled the same way.
        status = SkyPairStatus.NOT_JUDGED
        real_kept = added_real = real_patches
        sim_kept = added_sim = sim_patches
    else:
        # Dark sky is judged on both frames by level alone, against the calibrated
        # sky residual the simulator draws empty sky at.  Any noise sigma in the
        # rule would make the selection depend on the quantity being compared:
        # the matched frame's sigma shrinks the band when the twin is too quiet,
        # and the real frame's sigma grows on content.  The instrument config
        # holds the ceiling and the measurement behind it.
        real_kept = dark_sky_patches(real_patches, ceiling_level)
        sim_kept = dark_sky_patches(sim_patches, ceiling_level)
        if real_kept and sim_kept:
            status = SkyPairStatus.COMPARED
            added_real = real_kept
            added_sim = sim_kept
        else:
            status = SkyPairStatus.DROPPED
    add_sky_samples(real, added_real)
    add_sky_samples(sim, added_sim)
    return SkyPair(
        image_id=record.image_id,
        quantization=record.quantization,
        status=status,
        real_level=_sky_level(real_kept, real_patches),
        sim_level=_sky_level(sim_kept, sim_patches),
        n_real_patches=len(real_patches),
        n_sim_patches=len(sim_patches),
        n_real_kept=len(real_kept),
        n_sim_kept=len(sim_kept),
        real_sigma=_median_sigma(added_real),
        sim_sigma=_median_sigma(added_sim),
    )


def _fom_for_kind(kind: str) -> str | None:
    """Which figure of merit a sample kind belongs to."""
    for fom, prefixes in FOM_KIND_PREFIXES.items():
        if any(kind.startswith(prefix) for prefix in prefixes):
            return fom
    return None


def _mean_curve_density(
    curves: list[tuple[NDArrayFloatType, NDArrayFloatType]],
) -> NDArrayFloatType:
    """Frame-averaged curve as a nonnegative density over its axis.

    Bins that are NaN in every frame drop to zero mass, and negative
    excursions (background over-subtraction, normalized gap edges) are
    clipped: a density-W1 comparison needs nonnegative mass.
    """
    stack = np.stack([y for _x, y in curves])
    with warnings.catch_warnings():
        # A bin empty in every frame is legitimately all-NaN.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    return np.asarray(np.clip(np.nan_to_num(mean, nan=0.0), 0.0, None), dtype=np.float64)


def _pool_copopulated_limb_bins(comparison: InstrumentComparison) -> list[str]:
    """Build the FOM 3 pooled sample from strata populated on both sides.

    A raw pool over every stratum compares different scene mixtures
    whenever a stratum is empty on one side (the cohort or the sim
    emission path may leave one empty), and the pooled W1 then measures
    the mixture difference rather than the width difference.  Only
    co-populated strata pool into ``limb_width_copop``; one-sided strata
    are recorded on the comparison for disclosure.  Returns the co-populated
    strata's kinds.
    """
    bin_kinds = sorted(
        k
        for k in set(comparison.real.samples) | set(comparison.sim.samples)
        if k.startswith('limb_width_p')
    )
    copopulated = [
        k for k in bin_kinds if comparison.real.samples.get(k) and comparison.sim.samples.get(k)
    ]
    comparison.limb_bins_real_only = [
        k for k in bin_kinds if comparison.real.samples.get(k) and not comparison.sim.samples.get(k)
    ]
    comparison.limb_bins_sim_only = [
        k for k in bin_kinds if comparison.sim.samples.get(k) and not comparison.real.samples.get(k)
    ]
    if copopulated:
        comparison.real.samples['limb_width_copop'] = [
            x for k in copopulated for x in comparison.real.samples[k]
        ]
        comparison.sim.samples['limb_width_copop'] = [
            x for k in copopulated for x in comparison.sim.samples[k]
        ]
    return copopulated


def _count_frames(comparison: InstrumentComparison, copopulated_limb_bins: list[str]) -> None:
    """Count each side's contributing frames per kind and per figure of merit.

    A frame contributes to a kind when it added samples or a curve of that kind, and
    to ``limb_width_copop`` when it added samples to a co-populated limb stratum.  A
    frame contributes to a figure of merit when it contributes to any of its kinds.
    """
    copop = frozenset(copopulated_limb_bins)
    for side, kinds_per_frame in comparison.frame_kinds.items():
        kind_counts: dict[str, int] = {}
        fom_counts: dict[str, int] = {}
        for kinds in kinds_per_frame:
            frame_kinds = set(kinds)
            if kinds & copop:
                frame_kinds.add('limb_width_copop')
            for kind in frame_kinds:
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            foms = {fom for fom in map(_fom_for_kind, frame_kinds) if fom is not None}
            for fom in foms:
                fom_counts[fom] = fom_counts.get(fom, 0) + 1
        comparison.kind_frames[side] = kind_counts
        comparison.fom_frames[side] = fom_counts


def _two_sided_support(counts: dict[str, dict[str, int]], key: str) -> str:
    """The support label for ``key`` from the side with fewer contributing frames."""
    n_real = counts.get('real', {}).get(key, 0)
    n_sim = counts.get('sim', {}).get(key, 0)
    return cohort_support(min(n_real, n_sim)).value


def _aggregate(comparison: InstrumentComparison) -> None:
    """Compute divergences and support labels from the pooled samples.

    A statistic's support counts the frames on both sides and takes the smaller
    count, so a statistic with no simulated samples is labeled unsupported however
    many real frames it has.
    """
    copopulated = _pool_copopulated_limb_bins(comparison)
    _count_frames(comparison, copopulated)
    for kind, real_values in sorted(comparison.real.samples.items()):
        sim_values = comparison.sim.samples.get(kind, [])
        comparison.divergences[kind] = w1_divergence(
            np.asarray(real_values), np.asarray(sim_values)
        )
    # Curve kinds get the density-W1: the frame-averaged curve on each side
    # is treated as a density over its axis (frequency or distance), and the
    # W1 scalar is how far mass must move along that axis to turn one
    # shape into the other, normalized by the real density's IQR.
    for kind in sorted(set(comparison.real.curves) | set(comparison.sim.curves)):
        real_curves = comparison.real.curves.get(kind, [])
        sim_curves = comparison.sim.curves.get(kind, [])
        if not real_curves or not sim_curves:
            continue
        comparison.curve_divergences[kind] = w1_between_densities(
            real_curves[0][0],
            _mean_curve_density(real_curves),
            _mean_curve_density(sim_curves),
        )
    for fom in FOM_KIND_PREFIXES:
        comparison.fom_support[fom] = _two_sided_support(comparison.fom_frames, fom)
    for kind in sorted(set(comparison.divergences) | set(comparison.curve_divergences)):
        comparison.kind_support[kind] = _two_sided_support(comparison.kind_frames, kind)
    # FOM 7 support: pairs actually navigated.
    n_pairs = len({row['image_id'] for row in comparison.fom7_rows})
    comparison.fom_support['fom7_diagnostics'] = cohort_support(
        n_pairs, supported_min=3, limited_min=1
    ).value
    # FOM 6 cross-frame split: real and sim hot pixels alike recur at fixed
    # detector positions (the sim's map is keyed on the emulated detector), so
    # the stationary fractions compare directly.
    comparison.spike_split_real = split_stationary_spikes(comparison.real_rates)
    comparison.spike_split_sim = split_stationary_spikes(comparison.sim_rates)


def _kinds_of(frame: FrameSamples) -> frozenset[str]:
    """The sample and curve kinds one frame added."""
    return frozenset(frame.samples) | frozenset(frame.curves)


def pool_frame_pairs(
    comparison: InstrumentComparison,
    real_frames: list[FrameSamples],
    sim_frames: list[FrameSamples],
) -> None:
    """Pair each frame's sky with its matched frame's and pool both sides' samples.

    Each pair's sky samples are added to its frames before the frames are merged
    into the pools, so the pools carry the sky samples the pairing kept.

    Parameters:
        comparison: The instrument's comparison, holding one record per frame;
            it receives the sky pairs, the pooled samples, and each frame's
            kinds on each side.
        real_frames: The real frames' samples, in the records' order.
        sim_frames: The matched frames' samples, in the same order.
    """
    with_sky = twins_have_sky(sim_frames)
    for record, real_samples, sim_samples in zip(
        comparison.records, real_frames, sim_frames, strict=True
    ):
        comparison.sky_pairs.append(
            pair_sky(record, real_samples, sim_samples, twins_with_sky=with_sky)
        )
        comparison.real.merge(real_samples)
        comparison.sim.merge(sim_samples)
        comparison.frame_kinds['real'].append(_kinds_of(real_samples))
        comparison.frame_kinds['sim'].append(_kinds_of(sim_samples))


def run_realism_match(
    *,
    instruments: list[str] | None = None,
    max_frames_per_instrument: int | None = None,
    skip_fom7: bool = False,
) -> RealismResults:
    """Run the full realism match and return the per-instrument comparisons.

    Parameters:
        instruments: Restrict to these sim instrument names (None = all).
        max_frames_per_instrument: Cap the cohort per instrument (for smoke
            runs); None processes every frame.
        skip_fom7: Skip the matched-pair navigation runs.

    Returns:
        The aggregated results.
    """
    start = time.monotonic()
    cohort = discover_cohort()
    comparisons: dict[str, InstrumentComparison] = {}
    for instrument, sidecars in sorted(cohort.items()):
        if instruments is not None and instrument not in instruments:
            continue
        if max_frames_per_instrument is not None:
            sidecars = sidecars[:max_frames_per_instrument]
        comparison = InstrumentComparison(instrument=instrument)
        sidecars_by_id = {s.image_id: s for s in sidecars}
        real_frames: list[FrameSamples] = []
        for sidecar in sidecars:
            frame_samples, record, rates = _frame_samples_real(sidecar, instrument)
            comparison.records.append(record)
            real_frames.append(frame_samples)
            comparison.real_rates.append(rates)
        sim_frames: list[FrameSamples] = []
        for record in comparison.records:
            sim_samples, sim_frame_rates = _frame_samples_sim(record)
            sim_frames.append(sim_samples)
            comparison.sim_rates.append(sim_frame_rates)
        pool_frame_pairs(comparison, real_frames, sim_frames)
        if not skip_fom7:
            comparison.fom7_rows = _run_fom7(comparison.records, sidecars_by_id)
        _aggregate(comparison)
        comparisons[instrument] = comparison
    return RealismResults(comparisons=comparisons, runtime_sec=time.monotonic() - start)


def main() -> None:
    """CLI entry: run the match, write figures and the JSON summary."""
    from tests.integration.sim_realism_report import write_figures, write_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--instrument',
        action='append',
        help='restrict to a sim instrument (repeatable); default all',
    )
    parser.add_argument(
        '--max-frames', type=int, default=None, help='cap cohort frames per instrument'
    )
    parser.add_argument('--skip-fom7', action='store_true', help='skip the navigation pairs')
    args = parser.parse_args()
    results = run_realism_match(
        instruments=args.instrument,
        max_frames_per_instrument=args.max_frames,
        skip_fom7=args.skip_fom7,
    )
    write_figures(results)
    summary_path = write_summary(results)
    print(f'realism match complete in {results.runtime_sec:.1f} s; summary: {summary_path}')


if __name__ == '__main__':
    require_checkout_package(__file__)
    main()
