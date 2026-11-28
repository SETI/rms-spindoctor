"""Shared extraction machinery for the realism-match runner.

This module turns one frame -- real or simulated -- into the pooled sample
arrays the figures of merit compare.  Both sides run exactly the same code:
the same patch finder, the same profile sampler, the same feature machinery
(the navigator's ``NavOrchestrator.prepare``), so estimator bias cancels in
the comparison.  The only asymmetry is where the frame's true offset comes
from: the operator-verified sidecar for a real frame, zero by construction
for the matched sim frames (which plant no offset).

Sample kinds are string keys ('sky_sigma', 'star_ee50', 'limb_width_p1_r0',
...); the runner pools them across frames per instrument and hands each
real/sim kind pair to the W1 divergence.  Curve kinds ('sky_psd',
'star_profile', 'ring_radial_profile') accumulate per-frame curves that the
runner averages for the overlay figures and the density-W1 statistic.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.feature import NavFeatureType
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import LimbPolyline, RingEdgePolyline, StarGeometry
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs import ObsSnapshotInst
from spindoctor.sim.realism.artifact_rates import (
    ArtifactRates,
    measure_artifact_rates,
)
from spindoctor.sim.realism.dynamic_range import frame_dynamic_range
from spindoctor.sim.realism.noise import (
    SkyPatch,
    find_uniform_patches,
    noisy_patches,
    radial_power_spectrum,
)
from spindoctor.sim.realism.profiles import (
    edge_normal_profiles,
    edge_tracks_in_image,
    ee_radius,
    encircled_energy,
    profile_rise_width,
    radial_profile,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType
from tests.integration.sim_realism_limb_peak import (
    LIMB_PEAK_KIND_PREFIX,
    LimbPeak,
    LimbSelection,
    SilhouetteOffsets,
    limb_peak_offset,
)

__all__ = [
    'FrameSamples',
    'FrameSettings',
    'FrameSettingsReader',
    'FrameSky',
    'add_sky_samples',
    'diagnostic_rows',
    'extract_feature_samples',
    'extract_pixel_samples',
    'frame_settings_reader',
    'largest_body',
    'mean_curve_density',
    'median_patch_sigma',
    'prepare_frame_features',
    'register_frame_settings_reader',
    'reported_sky_level',
]

# Phase-angle bin edges (degrees) for the FOM 3 limb-width stratification.
PHASE_BIN_EDGES_DEG: tuple[float, ...] = (60.0, 120.0)
# Apparent-diameter bin edges (pixels) for the FOM 3 resolution strata.
DIAMETER_BIN_EDGES_PX: tuple[float, ...] = (100.0, 400.0)

# Sampling geometry for edge profiles: +-8 px at 4 samples per pixel.
_EDGE_HALF_LENGTH_PX = 8.0
_EDGE_N_SAMPLES = 65
_EDGE_SPACING_PX = 2.0 * _EDGE_HALF_LENGTH_PX / (_EDGE_N_SAMPLES - 1)

# Star cutout geometry.
_STAR_R_MAX_PX = 8.0
_STAR_N_BINS = 16
# Minimum star peak over local sky sigma for a usable FOM 2 cutout.
_STAR_MIN_PEAK_SNR = 20.0
# At most this many stars per frame contribute FOM 2 samples (the
# brightest qualifying ones), so both sides sample the same well-measured
# brightness regime regardless of their noise floors.
_MAX_STARS_PER_FRAME = 8

# Cap on per-frame profile vertices so one long limb cannot dominate the
# pooled distribution; vertices are decimated evenly to this count.
_MAX_PROFILES_PER_FEATURE = 200

# Flat-top (plateau-core) star guard: reject a cutout when at least this
# many pixels of the central 3x3 sit within a relative epsilon of the
# cutout maximum.  A central plateau carries no sub-pixel shape
# information for the encircled-energy radii, whatever produced it: a
# clipped core shares one value after the ADC ceiling (and a calibration
# that rescales the frame preserves that equality, so the test works on
# cohorts whose calibrated units define no fixed saturation level), and a
# faint core quantized onto very few output levels can tie the same way
# (observed on noise-free simulated 8-bit frames, where a 1-2 LSB core
# ties across the plateau).  Both plateau kinds are rejected; the test is
# a shape-information guard, not solely a clip signature.
_FLAT_TOP_MIN_PIXELS = 3
_FLAT_TOP_REL_EPS = 1e-6


@dataclass(frozen=True)
class FrameSettings:
    """A real frame's commanded settings its matched scene renders through.

    The reader also says which of the frame's pixels the camera clipped, which a
    calibrated product hides: calibration rescales each pixel, so a clipped disc no
    longer reads one exact level.

    Parameters:
        filter_name: The filter combination (``'CL1+CL2'``) when the matched
            instrument catalogs it, else None.
        gain_state: The frame's gain state, or None when its label carries none.
        quantization: The downlink quantization the frame went through, as a scene's
            ``detector.quantization`` spells it, or None for the instrument's own.
        clipped: Which of the frame's pixels read the camera's full scale in its raw
            product, in the frame's shape, or None when the reader cannot tell.
        silhouette_radii_km: Radii in km, longest first, keyed by body name, that the
            real silhouettes use in place of the radii the frame's kernels give.
    """

    filter_name: str | None = None
    gain_state: int | None = None
    quantization: str | None = None
    clipped: NDArrayBoolType | None = None
    silhouette_radii_km: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)


# A reader of a real frame's commanded settings: it takes the real observation and
# the keywords ``image_id``, ``instrument``, and ``image_path`` (the product's resolved
# path).
FrameSettingsReader = Callable[..., FrameSettings]

# The prefix of a module that reads one mission's frame labels and registers its
# reader here when imported.  The module is named for its mission.
_FRAME_SETTINGS_MODULE_PREFIX = 'sim_realism_frames_'

# The registered readers, by the sidecar's mission.
_FRAME_SETTINGS_READERS: dict[str, FrameSettingsReader] = {}


def register_frame_settings_reader(mission: str, reader: FrameSettingsReader) -> None:
    """Register the reader of one mission's real-frame settings.

    A mission-named module (``sim_realism_frames_<mission>.py`` beside this one)
    calls this when it is imported.

    Parameters:
        mission: The sidecar mission whose frames the reader reads.
        reader: The reader.
    """
    _FRAME_SETTINGS_READERS[mission] = reader


def frame_settings_reader(mission: str) -> FrameSettingsReader | None:
    """The reader of a mission's real-frame settings, or None if it has none.

    Every ``sim_realism_frames_*`` module beside this one is imported first, so each
    registers its reader without this module referring to any mission.  If a mission
    has no reader, its matched scenes render at the instrument's own settings.

    Parameters:
        mission: The sidecar's mission.

    Returns:
        The registered reader, or None.
    """
    directory = str(Path(__file__).resolve().parent)
    for module in pkgutil.iter_modules([directory]):
        if module.name.startswith(_FRAME_SETTINGS_MODULE_PREFIX):
            importlib.import_module(f'{__package__}.{module.name}')
    return _FRAME_SETTINGS_READERS.get(mission)


@dataclass(frozen=True)
class FrameSky:
    """One frame's FOM 1 sky patches, held until the runner pairs the frame.

    Parameters:
        patches: The frame's sky patches: the darkest quarter of its near-uniform
            patches whose noise sigma is positive.
        floor: The frame's floor (1st percentile) in native units.
        psd_freq: The spatial-frequency grid of the sky power spectra.
        psd_tiles: ``(cells, unit-total power)`` for up to four 64-px tiles, where
            ``cells`` are the four sky patches that make up the tile.
    """

    patches: list[SkyPatch]
    floor: float
    psd_freq: NDArrayFloatType
    psd_tiles: list[tuple[tuple[SkyPatch, ...], NDArrayFloatType]]


@dataclass
class FrameSamples:
    """Sample arrays and curves extracted from one frame.

    Parameters:
        samples: Scalar sample lists per kind (pooled by the runner).
        curves: Per-kind list of ``(x, y)`` curve pairs; the runner
            averages the y-arrays over frames on the shared x-grid.
        sky: The frame's FOM 1 sky patches.  The runner tests them against the
            matched frame's, then turns the kept ones into sky samples with
            :func:`add_sky_samples`.  :meth:`merge` doesn't copy this field.
        limb_peak: The gradient-peak measurement of the frame's measured limb, or None
            when the frame measured none.  :meth:`merge` doesn't copy this field.
        limb_width_directions_deg: The directions (see
            :class:`~tests.integration.sim_realism_limb_peak.LimbOrientation`) of the
            measured limb's vertices that gave a rise width, which the matched frame's
            widths are taken near.  :meth:`merge` doesn't copy this field.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)
    curves: dict[str, list[tuple[NDArrayFloatType, NDArrayFloatType]]] = field(default_factory=dict)
    sky: FrameSky | None = None
    limb_peak: LimbPeak | None = None
    limb_width_directions_deg: tuple[float, ...] = ()

    def add(self, kind: str, values: list[float] | NDArrayFloatType) -> None:
        """Append scalar samples under ``kind``, dropping non-finite values."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size:
            self.samples.setdefault(kind, []).extend(float(x) for x in finite)

    def add_curve(self, kind: str, x: NDArrayFloatType, y: NDArrayFloatType) -> None:
        """Append one curve under ``kind`` when it has finite content."""
        if np.any(np.isfinite(y)):
            self.curves.setdefault(kind, []).append(
                (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
            )

    def merge(self, other: FrameSamples) -> None:
        """Fold another frame's samples and curves into this collection."""
        for kind, values in other.samples.items():
            self.samples.setdefault(kind, []).extend(values)
        for kind, curves in other.curves.items():
            self.curves.setdefault(kind, []).extend(curves)


def _sky_and_signal_patches(image: NDArrayFloatType) -> tuple[list[SkyPatch], list[SkyPatch]]:
    """Split a frame's near-uniform 32-px patches into sky and signal strata.

    Only the patches whose noise sigma is measured and positive take part.  A tile
    of one constant value, such as zero-filled missing data, doesn't hold any noise.
    If it counted toward the darkest quarter, it could push the frame's real dark sky
    out of the sky stratum.

    Parameters:
        image: The frame in its native units.

    Returns:
        ``(sky, signal)``: the darkest quarter of the noisy patches by mean, and
        the rest.
    """
    patches = noisy_patches(find_uniform_patches(image, patch_size=32, max_mean_quantile=None))
    if not patches:
        return [], []
    cutoff = float(np.quantile(np.array([p.mean for p in patches]), 0.25))
    sky = [p for p in patches if p.mean <= cutoff]
    signal = [p for p in patches if p.mean > cutoff]
    return sky, signal


def _sky_psd_tiles(
    image: NDArrayFloatType, sky: list[SkyPatch]
) -> list[tuple[tuple[SkyPatch, ...], NDArrayFloatType]]:
    """Sky power spectra from up to four 64-px tiles made wholly of sky patches.

    The patches lie on a 32-px grid, so each 64-px tile covers four of them.  A
    tile is a candidate only when all four are sky patches, so that light from a
    ring or a limb beside one patch never reaches the spectrum.  The candidates
    are ranked by their brightest patch, and up to four of the darkest are
    measured.  Each spectrum is normalized to unit total power, because sky_sigma
    measures the amplitude.

    Parameters:
        image: The frame in its native units.
        sky: The frame's sky patches.

    Returns:
        ``(cells, unit-total power)`` per measured tile, where ``cells`` are the
        tile's four sky patches.
    """
    by_corner = {(p.v0, p.u0): p for p in sky}
    candidates: list[tuple[SkyPatch, ...]] = []
    for p in sky:
        corners = ((p.v0, p.u0), (p.v0, p.u0 + 32), (p.v0 + 32, p.u0), (p.v0 + 32, p.u0 + 32))
        if all(corner in by_corner for corner in corners):
            candidates.append(tuple(by_corner[corner] for corner in corners))
    candidates.sort(key=lambda cells: max(c.mean for c in cells))
    tiles: list[tuple[tuple[SkyPatch, ...], NDArrayFloatType]] = []
    for cells in candidates:
        v0, u0 = cells[0].v0, cells[0].u0
        _freq, power = radial_power_spectrum(image[v0 : v0 + 64, u0 : u0 + 64], n_bins=16)
        total = np.nansum(power)
        if np.isfinite(total) and total > 0.0:
            tiles.append((cells, power / total))
        if len(tiles) == 4:
            break
    return tiles


def extract_pixel_samples(
    image: NDArrayFloatType,
    *,
    exposure_stratum: str,
    saturation_level: float,
) -> tuple[FrameSamples, ArtifactRates]:
    """FOM 1, 5, and 6 samples from raw frame pixels.

    The FOM 1 sky patches are held on the returned samples' ``sky`` field, and
    they aren't added as samples here.  The runner adds them with
    :func:`add_sky_samples` after testing them against the matched frame's.

    Parameters:
        image: The frame in its native units (DN or I/F).
        exposure_stratum: Label of the frame's exposure stratum; the FOM 5
            sample kinds are suffixed with it so comparisons never cross
            strata.
        saturation_level: Full-scale value in the frame's units, or NaN
            when the cohort's units do not define one (calibrated I/F
            products); the saturated fraction is then not measured.

    Returns:
        ``(samples, rates)``: the frame's samples, plus the raw
        artifact rates (with spike positions) so the runner can do
        the cross-frame stationary/transient split.
    """
    out = FrameSamples()
    arr = np.asarray(image, dtype=np.float64)

    # FOM 1: near-uniform patches, sky and signal strata.  Sky-patch means
    # are reported relative to the frame's floor (1st percentile) so the
    # statistic is pedestal-invariant: some cohort products are
    # bias-subtracted (LORRI sci) while the matching sim chain carries its
    # bias pedestal, and the *level above floor* is the comparable part.
    # The sky patches stay on the frame (``FrameSamples.sky``) until the
    # runner has tested them against the matched frame's.  Only patches on
    # comparable content enter the pooled sky statistics (see
    # :func:`add_sky_samples`).
    frame_floor = float(np.percentile(arr[np.isfinite(arr)], 1.0)) if arr.size else 0.0
    sky, signal = _sky_and_signal_patches(arr)
    if sky:
        out.add('signal_sigma', [p.sigma for p in signal])
        psd_tiles = _sky_psd_tiles(arr, sky)
        freq, _ = radial_power_spectrum(arr[:64, :64], n_bins=16)
        out.sky = FrameSky(patches=sky, floor=frame_floor, psd_freq=freq, psd_tiles=psd_tiles)
        sky_sigma = float(np.median([p.sigma for p in sky]))
    else:
        sky_sigma = float('nan')

    # FOM 5: dynamic range within the exposure stratum.
    level = saturation_level if np.isfinite(saturation_level) else float('inf')
    stats = frame_dynamic_range(arr, saturation_level=level, noise_sigma=sky_sigma)
    if np.isfinite(saturation_level):
        out.add(f'frac_saturated_{exposure_stratum}', [stats.frac_saturated])
    out.add(f'frac_near_floor_{exposure_stratum}', [stats.frac_near_floor])
    # Signal percentiles relative to the frame median remove the pedestal
    # while keeping the stretch of the signal distribution comparable.
    p50 = stats.percentiles[3]
    if np.isfinite(p50):
        out.add(f'signal_p95_minus_p50_{exposure_stratum}', [stats.percentiles[5] - p50])
        out.add(f'signal_p99_minus_p50_{exposure_stratum}', [stats.percentiles[6] - p50])

    # FOM 6: artifact rates.
    rates = measure_artifact_rates(arr)
    out.add('artifact_missing_line_frac', [rates.missing_line_fraction])
    out.add('artifact_spike_frac', [rates.spike_fraction])
    return out, rates


def add_sky_samples(frame: FrameSamples, patches: list[SkyPatch]) -> None:
    """Add a frame's FOM 1 sky samples from the sky patches kept for its pair.

    Parameters:
        frame: The frame, whose ``sky`` holds its floor and power spectra.
        patches: The sky patches kept for the comparison (a subset of
            ``frame.sky.patches``).  A tile's power spectrum goes into the
            frame-averaged sky spectrum only when all four of its patches were kept.
    """
    sky = frame.sky
    if sky is None or not patches:
        return
    frame.add('sky_sigma', [p.sigma for p in patches])
    frame.add('sky_mean_minus_floor', [p.mean - sky.floor for p in patches])
    kept = set(patches)
    psd_curves = [power for cells, power in sky.psd_tiles if all(c in kept for c in cells)]
    if psd_curves:
        with warnings.catch_warnings():
            # A frequency bin empty in every tile is legitimately NaN.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_psd = np.nanmean(np.stack(psd_curves), axis=0)
        frame.add_curve('sky_psd', sky.psd_freq, mean_psd)


def largest_body(
    model_metadata: dict[str, dict[str, Any]],
) -> tuple[str | None, float | None, float | None]:
    """Name, diameter, and phase of the largest body model, if any.

    Parameters:
        model_metadata: The per-model metadata of one frame, keyed by model name
            (``body:DIONE``).

    Returns:
        The body's name after the colon, its predicted apparent diameter in pixels,
        and its phase angle in degrees, or three Nones when no body model states
        both.
    """
    best: tuple[str, float, float] | None = None
    for name, meta in model_metadata.items():
        if not name.startswith('body:'):
            continue
        diameter = meta.get('predicted_diameter_px')
        phase = meta.get('phase_angle_deg')
        if diameter is None or phase is None:
            continue
        if best is None or float(diameter) > best[1]:
            best = (name.split(':', 1)[1], float(diameter), float(phase))
    if best is None:
        return None, None, None
    return best


def prepare_frame_features(
    obs: ObsSnapshotInst, *, only_models: str | list[str]
) -> tuple[NDArrayFloatType, list[NavFeature], dict[str, dict[str, Any]]]:
    """Run the navigator's feature extraction on one frame.

    Both cohorts pass through here, so FOMs 2-4 measure the image around
    features the *same* extraction machinery placed.  The reliability gate
    is skipped: realism compares what the detector delivered, not what the
    gate kept.

    Parameters:
        obs: A loaded observation (real instrument snapshot or ObsSim).
        only_models: NavModel name globs to build ('stars', 'body:*',
            'rings:*').

    Returns:
        ``(image, features, model_metadata)``: the sensor-area image in
        native units, the extracted features with polyline/star geometry
        translated into sensor pixel coordinates, and per-model metadata.
    """
    models = build_models_for_obs(obs)
    # Measurement, not navigation: the simulated body model's LIMB_ARC
    # emission gates (minimum resolved diameter, maximum phase) are
    # navigation policy, and applying them here would leave the matched
    # high-phase and small-body scenes with no sim-side limb geometry --
    # their FOM 3 strata would silently compare different scene mixtures.
    # Gates off, the model emits the lit geometric limb (the real body
    # model's LIMB_ARC definition).  A real obs builds no simulated
    # models, so this changes only the sim side.
    for model in models:
        if isinstance(model, NavModelBodySimulated):
            model.apply_limb_emission_gates = False
    orchestrator = NavOrchestrator(models, only_models=only_models)
    prep = orchestrator.prepare(obs, apply_gate=False)
    margin_v, margin_u = obs.extfov_margin_vu
    shape_v, shape_u = obs.data_shape_vu
    image = np.asarray(
        prep.context.image_ext[margin_v : margin_v + shape_v, margin_u : margin_u + shape_u],
        dtype=np.float64,
    )
    return image, list(prep.features), dict(prep.model_metadata)


def _limb_bin_label(phase_deg: float, diameter_px: float) -> str:
    """The FOM 3 stratification label for one limb feature."""
    p_bin = int(np.searchsorted(np.asarray(PHASE_BIN_EDGES_DEG), phase_deg, side='right'))
    r_bin = int(np.searchsorted(np.asarray(DIAMETER_BIN_EDGES_PX), diameter_px, side='right'))
    return f'p{p_bin}_r{r_bin}'


def _normalized_mean_profile(profiles: NDArrayFloatType) -> NDArrayFloatType | None:
    """Mean edge profile normalized to [0, 1] between its plateaus."""
    if profiles.shape[0] == 0:
        return None
    mean = np.mean(profiles, axis=0)
    quarter = max(2, mean.size // 4)
    inside = float(np.median(mean[:quarter]))
    outside = float(np.median(mean[-quarter:]))
    span = inside - outside
    if abs(span) < 1e-12:
        return None
    return np.asarray((mean - outside) / span, dtype=np.float64)


def _decimate(
    vertices: NDArrayFloatType, normals: NDArrayFloatType, cap: int
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Evenly subsample a polyline to at most ``cap`` vertices."""
    n = vertices.shape[0]
    if n <= cap:
        return vertices, normals
    idx = np.linspace(0, n - 1, cap).astype(int)
    return vertices[idx], normals[idx]


def _has_flat_top(image: NDArrayFloatType, cv: int, cu: int) -> bool:
    """True when the star cutout at ``(cv, cu)`` has a clipped (flat) core.

    The cutout maximum is taken over the FOM 2 profile window; the test
    trips when at least ``_FLAT_TOP_MIN_PIXELS`` pixels of the central 3x3
    sit within ``_FLAT_TOP_REL_EPS`` (relative) of it.  Saturation-level
    independent, so it guards the calibrated cohorts too.

    Parameters:
        image: Sensor-area image in native units.
        cv: Rounded row of the star center (inside the image).
        cu: Rounded column of the star center (inside the image).

    Returns:
        Whether the core is a plateau.
    """
    half = int(np.ceil(_STAR_R_MAX_PX))
    cutout = image[max(0, cv - half) : cv + half + 1, max(0, cu - half) : cu + half + 1]
    finite = cutout[np.isfinite(cutout)]
    if finite.size == 0:
        return False
    cutout_max = float(finite.max())
    if cutout_max <= 0.0:
        return False
    core = image[max(0, cv - 1) : cv + 2, max(0, cu - 1) : cu + 2]
    threshold = cutout_max - _FLAT_TOP_REL_EPS * abs(cutout_max)
    n_at_max = int(np.count_nonzero(core >= threshold))
    return n_at_max >= _FLAT_TOP_MIN_PIXELS


def extract_feature_samples(
    image: NDArrayFloatType,
    features: list[NavFeature],
    model_metadata: dict[str, dict[str, Any]],
    *,
    offset_vu: tuple[float, float],
    extfov_margin_vu: tuple[int, int],
    saturation_level: float,
    matched_model: str | None = None,
    silhouette_offsets: SilhouetteOffsets | None = None,
    limb_peak_model: str | None = None,
    limb_selection: LimbSelection | None = None,
) -> FrameSamples:
    """FOM 2-4 samples around the frame's extracted features.

    Feature geometry arrives in extfov coordinates at the *predicted*
    positions; each vertex is translated to sensor coordinates and shifted
    by the frame's known offset so the samples straddle the actual edge.

    Parameters:
        image: Sensor-area image in native units (from
            :func:`prepare_frame_features`).
        features: Extracted features (gate skipped).
        model_metadata: Per-model metadata (phase angle, diameter) for the
            FOM 3 stratification.
        offset_vu: The frame's known ``(dv, du)`` offset: operator-verified
            ground truth for a real frame, the planted offset for a sim
            frame.
        extfov_margin_vu: The obs's extended-FOV margins, subtracted to
            translate feature coordinates onto ``image``.
        saturation_level: Full-scale value for the star saturation filter;
            NaN disables the explicit-level test (calibrated cohorts, where
            CALIB saturation is not represented by a fixed level).  The
            flat-top plateau guard runs regardless, so clipped cores are
            rejected on every cohort.
        matched_model: The one body model whose limb adds FOM 3 widths and a limb
            profile: the body the matched scene renders, so each width has a
            counterpart of the same body.  None, or no ``limb_selection``, adds no limb
            widths.
        silhouette_offsets: The frame's body silhouettes, which the gradient peak of
            ``limb_peak_model``'s limb is measured from.
        limb_peak_model: The one body model whose binned limb also adds its
            gradient-peak offset from the silhouette, as a sample of kind
            ``limb_peak_<bin>`` (see :mod:`tests.integration.sim_realism_limb_peak`).
            It is the body the matched scene renders, so each frame of a pair measures
            the same body.  None, or no silhouettes, adds no gradient peak.
        limb_selection: Which vertices of the measured limb may be used, and the
            body's orientation, which gives each vertex its direction.  On a real frame
            its track test also leaves out the width profiles whose 8 px tracks touch
            another body, the planet, or the planet's rings or shadow.  On a matched
            frame a width is taken only near the directions of the real widths.

    Returns:
        The frame's feature samples, with the gradient-peak measurement when one was
        made.
    """
    out = FrameSamples()
    dv, du = float(offset_vu[0]), float(offset_vu[1])
    mv, mu = extfov_margin_vu
    shift = np.array([dv - mv, du - mu])

    # Local sky sigma for the star SNR filter.
    sky_patches, _signal = _sky_and_signal_patches(image)
    sky_sigma = float(np.median([p.sigma for p in sky_patches])) if sky_patches else 0.0

    # Qualifying star cutouts collect first; only the brightest few per
    # frame contribute samples.  A per-side SNR threshold alone would admit
    # different brightness mixes on the two sides (whichever side is
    # noisier admits fewer faint stars), and faint noise-dominated cutouts
    # inflate the encircled-energy tail; ranking by peak and capping the
    # count is symmetric and keeps both samples in the well-measured
    # regime.
    star_candidates: list[tuple[float, NDArrayFloatType, NDArrayFloatType]] = []
    star_radius: NDArrayFloatType | None = None
    for feature in features:
        geometry = feature.geometry
        if isinstance(geometry, StarGeometry):
            center = (
                geometry.predicted_vu[0] + shift[0],
                geometry.predicted_vu[1] + shift[1],
            )
            radius, intensity = radial_profile(
                image, center, r_max=_STAR_R_MAX_PX, n_bins=_STAR_N_BINS
            )
            if not np.any(np.isfinite(intensity)):
                continue
            peak = float(np.nanmax(intensity))
            if peak <= 0.0 or (sky_sigma > 0.0 and peak < _STAR_MIN_PEAK_SNR * sky_sigma):
                continue
            # Contaminant guard: the profile must peak in its core, and the
            # far tail must stay well below the peak.  A hot pixel or cosmic
            # ray inside the cutout otherwise masquerades as PSF energy and
            # wrecks the encircled-energy radii.
            if int(np.nanargmax(intensity)) > 1:
                continue
            tail = intensity[radius > 3.0]
            if tail.size and np.any(np.isfinite(tail)) and float(np.nanmax(tail)) > 0.3 * peak:
                continue
            cv, cu = round(center[0]), round(center[1])
            if 0 <= cv < image.shape[0] and 0 <= cu < image.shape[1]:
                raw_peak = float(
                    image[
                        max(0, cv - 1) : cv + 2,
                        max(0, cu - 1) : cu + 2,
                    ].max()
                )
                if np.isfinite(saturation_level) and raw_peak >= saturation_level:
                    continue
                if _has_flat_top(image, cv, cu):
                    continue
            star_candidates.append((peak, radius, intensity))
            star_radius = radius
        elif isinstance(geometry, LimbPolyline):
            if feature.source_model != matched_model or limb_selection is None:
                continue
            vertices, normals = _decimate(
                np.asarray(geometry.vertices_vu, dtype=np.float64) + shift,
                np.asarray(geometry.normals_vu, dtype=np.float64),
                _MAX_PROFILES_PER_FEATURE,
            )
            keep = limb_selection.keeps(
                vertices, normals, _EDGE_HALF_LENGTH_PX, limb_selection.width_deg
            ) & edge_tracks_in_image(
                image.shape, vertices, normals, half_length_px=_EDGE_HALF_LENGTH_PX
            )
            vertices, normals = vertices[keep], normals[keep]
            profiles = edge_normal_profiles(
                image,
                vertices,
                normals,
                half_length_px=_EDGE_HALF_LENGTH_PX,
                n_samples=_EDGE_N_SAMPLES,
            )
            widths = [profile_rise_width(p, spacing_px=_EDGE_SPACING_PX) for p in profiles]
            meta = model_metadata.get(feature.source_model, {})
            phase_value = meta.get('phase_angle_deg')
            phase = float('nan') if phase_value is None else float(phase_value)
            diameter = float(meta.get('predicted_diameter_px') or float('nan'))
            if not np.isfinite(diameter):
                # Fallback for a model without diameter metadata: the
                # polyline's bbox extent is exact for a fully framed matched
                # body (and a fair lower bound otherwise).
                bbox = geometry.bbox_extfov_vu
                diameter = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            if np.isfinite(phase) and np.isfinite(diameter):
                # Only binned widths are emitted; the runner builds the
                # pooled FOM 3 statistic from strata populated on both
                # sides, so a raw cross-strata pool never appears.
                out.add(f'limb_width_{_limb_bin_label(phase, diameter)}', widths)
                measured = np.isfinite(np.asarray(widths, dtype=np.float64))
                out.limb_width_directions_deg += tuple(
                    float(d) for d in limb_selection.orientation.directions_deg(vertices[measured])
                )
                if silhouette_offsets is not None and feature.source_model == limb_peak_model:
                    out.limb_peak = limb_peak_offset(
                        image,
                        geometry,
                        feature.source_model,
                        shift_vu=shift,
                        silhouette_offsets=silhouette_offsets,
                        selection=limb_selection,
                    )
                    if out.limb_peak.offset_px is not None:
                        label = _limb_bin_label(phase, diameter)
                        out.add(f'{LIMB_PEAK_KIND_PREFIX}{label}', [out.limb_peak.offset_px])
            mean_profile = _normalized_mean_profile(profiles)
            if mean_profile is not None:
                taps = np.linspace(-_EDGE_HALF_LENGTH_PX, _EDGE_HALF_LENGTH_PX, _EDGE_N_SAMPLES)
                out.add_curve('limb_profile', taps, mean_profile)
        elif isinstance(geometry, RingEdgePolyline):
            vertices, normals = _decimate(
                np.asarray(geometry.vertices_vu, dtype=np.float64) + shift,
                np.asarray(geometry.normals_vu, dtype=np.float64),
                _MAX_PROFILES_PER_FEATURE,
            )
            profiles = edge_normal_profiles(
                image,
                vertices,
                normals,
                half_length_px=_EDGE_HALF_LENGTH_PX,
                n_samples=_EDGE_N_SAMPLES,
            )
            widths = [profile_rise_width(p, spacing_px=_EDGE_SPACING_PX) for p in profiles]
            out.add('ring_edge_width', widths)
            if profiles.shape[0]:
                taps = np.linspace(-_EDGE_HALF_LENGTH_PX, _EDGE_HALF_LENGTH_PX, _EDGE_N_SAMPLES)
                mean = np.mean(profiles, axis=0)
                peak = float(np.max(np.abs(mean)))
                if peak > 0.0:
                    out.add_curve('ring_radial_profile', taps, mean / peak)
        elif feature.feature_type is NavFeatureType.TERMINATOR_ARC:
            # Terminator realism is judged separately, by FOM 3 computed on the
            # terminator side; excluded from the FOM 3 limb distribution.
            continue
    if star_candidates and star_radius is not None:
        star_candidates.sort(key=lambda entry: entry[0], reverse=True)
        star_profiles: list[NDArrayFloatType] = []
        for peak, radius, intensity in star_candidates[:_MAX_STARS_PER_FRAME]:
            r_ee, ee = encircled_energy(radius, intensity)
            out.add('star_ee50', [ee_radius(r_ee, ee, 0.5)])
            out.add('star_ee80', [ee_radius(r_ee, ee, 0.8)])
            star_profiles.append(np.asarray(intensity, dtype=np.float64) / peak)
        with warnings.catch_warnings():
            # Outer radial bins may be NaN in every cutout.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_star = np.nanmean(np.stack(star_profiles), axis=0)
        out.add_curve('star_profile', star_radius, mean_star)
    return out


def diagnostic_rows(instrument: str, image_id: str, side: str, result: Any) -> list[dict[str, Any]]:
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


def median_patch_sigma(patches: list[SkyPatch]) -> float:
    """The median noise sigma of some sky patches, NaN for none."""
    if not patches:
        return float('nan')
    return float(np.median([p.sigma for p in patches]))


def reported_sky_level(kept: list[SkyPatch], patches: list[SkyPatch]) -> float:
    """A frame's reported sky level: the median mean of its kept patches.

    If the frame didn't keep any patch, the level is the median mean of all its sky
    patches.  If the frame has no sky patches, the level is NaN.
    """
    chosen = kept or patches
    if not chosen:
        return float('nan')
    return float(np.median([p.mean for p in chosen]))


def mean_curve_density(
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
