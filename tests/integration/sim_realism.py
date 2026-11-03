"""Section 7 realism-match runner: sim vs real, per instrument, FOMs 1-7.

Compares the WS-3 image-library cohort against matched simulated frames on
the seven figures of merit of the sim-realism plan's Section 7:

1. Sky-region noise statistics (paired-difference sigma, noise vs signal,
   sky power spectrum).
2. Star-cutout radial profiles / encircled energy where the cohort has
   star frames.
3. Normalized limb profiles and 10-90% rise widths, binned by resolution
   and phase.
4. Ring-edge profiles and rise widths (Cassini).
5. Exposure-stratified dynamic-range statistics (never unstratified).
6. Artifact incidence rates vs the catalog defaults.
7. Technique-diagnostic values on matched scene/frame pairs -- READ-ONLY:
   reported for the record and never a tuning target, because FOM 7 is
   built from the navigator's own outputs and tuning against it would
   re-admit circularity through parameter fitting.

Every real frame gets one matched sim frame (same instrument chain, same
exposure, same content class; see :mod:`tests.integration.sim_realism_scenes`),
and both sides run identical extraction (:mod:`tests.integration.sim_realism_support`).
The scalar divergence per figure of merit is the 15.10-H W1 (quantile-clipped,
real-IQR-normalized); curve kinds (sky PSD, star / limb / ring profiles) get
the density-W1 between the frame-averaged curves.  Where a cohort cannot
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
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.obs.obs_snapshot import ObsSnapshot
from spindoctor.sim.realism.artifact_incidence import (
    ArtifactIncidence,
    split_stationary_spikes,
)
from spindoctor.sim.realism.divergence import (
    W1Result,
    cohort_support,
    w1_between_densities,
    w1_divergence,
)
from spindoctor.sim.realism.dynamic_range import stratify_by_exposure
from spindoctor.support.types import NDArrayFloatType
from tests.integration.sidecar import LibraryRoot, Sidecar, load_sidecar
from tests.integration.sim_realism_scenes import matched_scene
from tests.integration.sim_realism_support import (
    FrameSamples,
    extract_feature_samples,
    extract_pixel_samples,
    prepare_frame_features,
)

__all__ = [
    'FrameRecord',
    'InstrumentComparison',
    'RealismResults',
    'run_realism_match',
]

# (mission, camera) -> sim instrument whose signal chain and units match the
# cohort's products (COISS CALIB and VGISS GEOMED are I/F; GOSSI REDR and
# LORRI sci are DN).
INSTRUMENT_FOR: dict[tuple[str, str], str] = {
    ('COISS', 'NAC'): 'coiss_calib_nac',
    ('COISS', 'WAC'): 'coiss_calib_wac',
    ('GOSSI', 'SSI'): 'gossi',
    ('VGISS', 'NA'): 'vgiss',
    ('VGISS', 'WA'): 'vgiss',
    ('NHLORRI', 'LORRI'): 'nhlorri',
}

# Full-scale level in cohort units for the saturation statistics and the
# FOM 2 saturated-star filter; NaN where the cohort's calibrated units do
# not define a fixed level (CALIB/GEOMED I/F).
SATURATION_LEVEL: dict[str, float] = {
    'coiss_calib_nac': float('nan'),
    'coiss_calib_wac': float('nan'),
    'gossi': 255.0,
    'nhlorri': 4095.0,
    'vgiss': float('nan'),
}

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

# Which sample-kind prefixes belong to which figure of merit (for support
# labeling and reporting).
FOM_KIND_PREFIXES: dict[str, tuple[str, ...]] = {
    'fom1_noise': ('sky_sigma', 'sky_mean_minus_floor', 'signal_sigma'),
    'fom2_psf': ('star_ee50', 'star_ee80'),
    'fom3_limb': ('limb_width',),
    'fom4_ring': ('ring_edge_width',),
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
        fom_frames: Contributing real-frame count per figure of merit.
        fom_support: Cohort-support label per figure of merit.
        fom7_rows: Read-only technique-diagnostic rows from the matched
            navigation pairs.
    """

    instrument: str
    records: list[FrameRecord] = field(default_factory=list)
    real: FrameSamples = field(default_factory=FrameSamples)
    sim: FrameSamples = field(default_factory=FrameSamples)
    divergences: dict[str, W1Result] = field(default_factory=dict)
    curve_divergences: dict[str, W1Result] = field(default_factory=dict)
    limb_bins_real_only: list[str] = field(default_factory=list)
    limb_bins_sim_only: list[str] = field(default_factory=list)
    fom_frames: dict[str, int] = field(default_factory=dict)
    fom_support: dict[str, str] = field(default_factory=dict)
    fom7_rows: list[dict[str, Any]] = field(default_factory=list)
    real_incidences: list[ArtifactIncidence] = field(default_factory=list)
    sim_incidences: list[ArtifactIncidence] = field(default_factory=list)
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


def _load_real_obs(sidecar: Sidecar) -> ObsSnapshot:
    """Load the real observation behind one sidecar."""
    from tests.integration.test_autonomous_nav import _MISSION_TO_OBS_CLASS

    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    return cast(ObsSnapshot, obs_class.from_file(_resolve_image_url(sidecar.image_url)))


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
) -> tuple[FrameSamples, FrameRecord, ArtifactIncidence]:
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
    saturation = SATURATION_LEVEL[instrument]
    samples, incidence = extract_pixel_samples(
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
    return samples, record, incidence


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
    )
    obs = ObsSim.from_file(f'/tmp/sim_realism_{record.image_id}.yaml', sim_params=scene)
    return obs, scene


def _frame_samples_sim(record: FrameRecord) -> tuple[FrameSamples, ArtifactIncidence]:
    """Extract the matched sim frame's samples (same machinery as real)."""
    obs, _scene = _sim_obs_for_record(record)
    saturation = SATURATION_LEVEL[record.instrument]
    samples, incidence = extract_pixel_samples(
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
    return samples, incidence


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


def _pool_copopulated_limb_bins(comparison: InstrumentComparison) -> None:
    """Build the FOM 3 pooled sample from strata populated on both sides.

    A raw pool over every stratum compares different scene mixtures
    whenever a stratum is empty on one side (the cohort or the sim
    emission path may leave one empty), and the pooled W1 then measures
    the mixture difference rather than the width difference.  Only
    co-populated strata pool into ``limb_width_copop``; one-sided strata
    are recorded on the comparison for disclosure.
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


def _aggregate(comparison: InstrumentComparison) -> None:
    """Compute divergences and support labels from the pooled samples."""
    _pool_copopulated_limb_bins(comparison)
    for kind, real_values in sorted(comparison.real.samples.items()):
        sim_values = comparison.sim.samples.get(kind, [])
        comparison.divergences[kind] = w1_divergence(
            np.asarray(real_values), np.asarray(sim_values)
        )
    # Curve kinds get the density-W1: the frame-averaged curve on each side
    # is treated as a density over its axis (frequency or distance), and the
    # 15.10-H scalar is how far mass must move along that axis to turn one
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
        n = comparison.fom_frames.get(fom, 0)
        comparison.fom_support[fom] = cohort_support(n).value
    # FOM 7 support: pairs actually navigated.
    n_pairs = len({row['image_id'] for row in comparison.fom7_rows})
    comparison.fom_support['fom7_diagnostics'] = cohort_support(
        n_pairs, supported_min=3, limited_min=1
    ).value
    # FOM 6 cross-frame split: real and sim hot pixels alike recur at fixed
    # detector positions (the sim's map is keyed on the emulated detector), so
    # the stationary fractions compare directly.
    comparison.spike_split_real = split_stationary_spikes(comparison.real_incidences)
    comparison.spike_split_sim = split_stationary_spikes(comparison.sim_incidences)


def _count_fom_frames(comparison: InstrumentComparison, frame: FrameSamples) -> None:
    """Increment per-FOM contributing-frame counts for one real frame."""
    seen: set[str] = set()
    for kind in frame.samples:
        fom = _fom_for_kind(kind)
        if fom is not None:
            seen.add(fom)
    for kind in frame.curves:
        if kind == 'sky_psd':
            seen.add('fom1_noise')
    for fom in seen:
        comparison.fom_frames[fom] = comparison.fom_frames.get(fom, 0) + 1


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
        for sidecar in sidecars:
            frame_samples, record, incidence = _frame_samples_real(sidecar, instrument)
            comparison.records.append(record)
            comparison.real.merge(frame_samples)
            comparison.real_incidences.append(incidence)
            _count_fom_frames(comparison, frame_samples)
        for record in comparison.records:
            sim_samples, sim_incidence = _frame_samples_sim(record)
            comparison.sim.merge(sim_samples)
            comparison.sim_incidences.append(sim_incidence)
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
    main()
