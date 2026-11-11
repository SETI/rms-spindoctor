"""Truth checks for the simulator's validation scenes: per-scene tolerances from measurement.

A validation scene plants a known pointing offset, so the navigator's answer can be
compared with the truth.  This module holds what each such comparison is held to, keyed
by ``(scene_name, technique)``: :data:`FUSED` for the orchestrator's fused answer, or a
technique name for one technique's answer.  Every check measures the error as a distance
in two dimensions, ``hypot(dv_error, du_error)``, never one axis at a time.

The measurement.  Each scene was rendered and navigated on its committed
``random_seed`` and on 15 more (101, 202, ..., 1515), so that each check covers the
spread over noise draws rather than one draw.  A scene that fails on some seeds is
measured over the seeds where it succeeds.

The two rules.

* **An accurate answer** is held to a tolerance: the largest error over the seeds plus a
  margin of max(0.05 px, a fifth of that largest error), rounded up to 0.01 px.  The
  0.05 px floor covers the drift a baseline already allows (0.02 px per axis) and the
  jitter between machines; the proportional part covers seeds the measurement did not
  sample.  An entry in :data:`TRUTH_TOLERANCES` records the error on the committed seed,
  the largest error, the spread (largest minus smallest), and the margin.
* **A known wrong answer** -- wrong for a reason an issue explains -- is held to a band
  around its known error, so a fix shows up as a deliberate change.  A known error is a
  bias and its spread over the seeds is its noise, so the margin scales with the spread:
  the band runs from the smallest error minus the margin to the largest plus the margin,
  where the margin is max(0.05 px, a fifth of the spread), and each end is rounded
  outward to 0.001 px.  Bands on one technique's answer or on one component live here
  (:data:`KNOWN_ERRORS`, :data:`RANK1_NORMAL_ERROR`); a fused band on a scene outside
  ``algorithmic_invariants`` lives in the scene's ``expected`` block
  (``known_offset_error_px`` / ``known_offset_error_tol_px``, asserted by
  :mod:`tests.integration.sim_expected`).

The fused answer is also checked against the sigma it reports: the error measured in
units of the reported covariance must not exceed :data:`SIGMA_MULTIPLE`.  For a round
covariance that measure is the error divided by the per-axis sigma.  Over the seeds every
accurate validation scene sits at or below 1.9 times its reported sigma; 3 leaves room
above that and still fails an answer that is confidently wrong.  A scene that sits above
3 for a known reason is held to a band on that measure (:data:`SIGMA_BANDS`) with a
margin of max(0.1, a fifth of its spread), so a fix leaves the band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.ensemble_observability import mixed_scale_pinvh
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene import iter_scene_paths, scene_class_for_path

FUSED = '*'
"""The technique key for the orchestrator's fused answer."""

SEEDS_MEASURED = 16
"""The number of noise seeds each check was measured over (the committed one and 15)."""

SIGMA_MULTIPLE = 3.0
"""The largest fused error, in units of the reported covariance, a validation scene may show."""

VALIDATION_CLASSES: frozenset[str] = frozenset(
    {
        'algorithmic_invariants',
        'artifact_sweep',
        'atmosphere',
        'expected_fail',
        'multi_body_geometry',
        'mutual_event',
        'regression',
        'ring_system',
        'star_confounder',
    }
)
"""The scene classes that check accuracy on a frame; the remaining classes are sweeps."""


@dataclass(frozen=True)
class TruthTolerance:
    """A measured error and the tolerance derived from it.

    Parameters:
        error_px: The error on the scene's committed seed.
        worst_px: The largest error over the seeds measured.
        spread_px: The largest minus the smallest error over the seeds measured.
        margin_px: The margin added to ``worst_px``.
    """

    error_px: float
    worst_px: float
    spread_px: float
    margin_px: float

    @property
    def tolerance_px(self) -> float:
        """The largest error that passes: the worst measured error plus the margin."""
        return self.worst_px + self.margin_px


@dataclass(frozen=True)
class KnownErrorBand:
    """A known error's measured range over the seeds and the band that holds it.

    The values are in pixels for an offset error and in sigmas for a band on the error
    measured in units of the reported covariance.

    Parameters:
        smallest: The smallest value measured over the seeds.
        largest: The largest value measured over the seeds.
        margin: The margin outside the measured range on each side.
        low: The band's lower end, ``smallest - margin`` rounded down.
        high: The band's upper end, ``largest + margin`` rounded up.
    """

    smallest: float
    largest: float
    margin: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        """Return True when ``value`` lies inside the band."""
        return self.low <= value <= self.high


def _t(error_px: float, worst_px: float, spread_px: float, margin_px: float) -> TruthTolerance:
    """Return a :class:`TruthTolerance` from one table row, positionally for brevity."""
    return TruthTolerance(
        error_px=error_px, worst_px=worst_px, spread_px=spread_px, margin_px=margin_px
    )


def _b(smallest: float, largest: float, margin: float, low: float, high: float) -> KnownErrorBand:
    """Return a :class:`KnownErrorBand` from one table row, positionally for brevity."""
    return KnownErrorBand(smallest=smallest, largest=largest, margin=margin, low=low, high=high)


# Columns: error on the committed seed, worst error over the seeds, spread over the
# seeds, margin.  All in pixels; the tolerance is worst + margin.
TRUTH_TOLERANCES: dict[tuple[str, str], TruthTolerance] = {
    # algorithmic_invariants, fused answer (disc and star scenes).  planted_offset_disc,
    # planted_offset_disc_if, and planted_offset_small carry known errors instead.
    ('planted_offset_irregular', FUSED): _t(0.001, 0.001, 0.000, 0.05),
    ('planted_offset_limb', FUSED): _t(0.067, 0.073, 0.009, 0.05),
    ('planted_offset_limb_mesh', FUSED): _t(0.076, 0.091, 0.027, 0.05),
    # The prediction is the smooth limit of the rendered lumpy mesh; this is the price
    # of that shape mismatch.
    ('planted_offset_shapemismatch', FUSED): _t(0.667, 0.675, 0.013, 0.14),
    ('planted_offset_star_field', FUSED): _t(0.018, 0.019, 0.003, 0.05),
    ('planted_offset_star_field_bright', FUSED): _t(0.040, 0.041, 0.001, 0.05),
    # algorithmic_invariants, one technique navigated alone (StarRefineNav read from
    # the fused run, since it needs the pass-1 prior).
    ('planted_offset_blob', 'BodyBlobNav'): _t(0.006, 0.009, 0.007, 0.05),
    ('planted_offset_blob_crescent', 'BodyBlobNav'): _t(0.253, 0.282, 0.062, 0.06),
    ('planted_offset_blob_crescent_displaced', 'BodyBlobNav'): _t(0.252, 0.278, 0.057, 0.06),
    ('planted_offset_blob_mesh_crescent', 'BodyBlobNav'): _t(0.486, 0.592, 0.114, 0.12),
    ('planted_offset_limb', 'BodyLimbNav'): _t(0.099, 0.099, 0.050, 0.05),
    ('planted_offset_limb_mesh', 'BodyLimbNav'): _t(0.175, 0.255, 0.127, 0.06),
    ('planted_offset_ring', 'RingEdgeNav'): _t(0.011, 0.016, 0.005, 0.05),
    ('planted_unique_star_single', 'StarUniqueMatchNav'): _t(0.054, 0.058, 0.006, 0.05),
    ('planted_refine_star_field', 'StarRefineNav'): _t(0.016, 0.019, 0.003, 0.05),
    # artifact_sweep.  The scene seed also places the lost lines, so the 16 seeds would
    # render 16 different losses.  These rows hold the placement at the committed seed
    # and vary only the noise draw: the telemetry stages derive their own sub-seeds
    # from the scene seed, and the measurement fixed those at the committed value.
    ('disc_structured_loss', FUSED): _t(0.080, 0.080, 0.004, 0.05),
    ('disc_structured_loss_adversarial', FUSED): _t(0.141, 0.141, 0.000, 0.05),
    ('star_field_structured_loss', FUSED): _t(0.067, 0.070, 0.004, 0.05),
    ('star_field_structured_loss_adversarial', FUSED): _t(0.133, 0.137, 0.005, 0.05),
    # atmosphere.  titan_haze_limb is a haze-blind navigator fitting a hard limb under
    # a soft haze limb.
    ('titan_haze', FUSED): _t(0.128, 0.137, 0.018, 0.05),
    ('titan_haze_limb', FUSED): _t(0.261, 0.269, 0.010, 0.06),
    # multi_body_geometry.
    ('two_moons', FUSED): _t(0.005, 0.005, 0.000, 0.05),
    # mutual_event.
    ('mutual_deep', FUSED): _t(0.164, 0.174, 0.012, 0.05),
    ('mutual_grazing', FUSED): _t(0.110, 0.117, 0.012, 0.05),
    ('mutual_half', FUSED): _t(0.315, 0.319, 0.014, 0.07),
    # regression, fused answer.  relief_psf_stray_light carries a known error in its
    # expected block instead (#791).
    ('banded_planet_transit', FUSED): _t(0.124, 0.131, 0.012, 0.05),
    ('disc_subpixel_offset', FUSED): _t(0.000, 0.000, 0.000, 0.05),
    ('low_phase_limb_misconverge', FUSED): _t(0.004, 0.004, 0.000, 0.05),
    ('star_equal_brightness', FUSED): _t(0.033, 0.033, 0.002, 0.05),
    ('star_zero_offset', FUSED): _t(0.044, 0.044, 0.003, 0.05),
    ('textured_moon_offset', FUSED): _t(0.261, 0.630, 0.526, 0.13),
    # regression, one technique navigated alone.
    ('disc_subpixel_offset', 'BodyDiscCorrelateNav'): _t(0.000, 0.000, 0.000, 0.05),
    ('star_equal_brightness', 'StarFieldFromCatalogNav'): _t(0.033, 0.033, 0.002, 0.05),
    ('star_zero_offset', 'StarFieldFromCatalogNav'): _t(0.041, 0.041, 0.003, 0.05),
    # ring_system.  orbit_error_ringlet carries a known error in its expected block.
    ('edge_wave_gap', FUSED): _t(0.003, 0.008, 0.005, 0.05),
    ('mmode_ringlet', FUSED): _t(0.003, 0.004, 0.003, 0.05),
    ('moonlet_propeller', FUSED): _t(0.003, 0.007, 0.005, 0.05),
    ('spoked_sheet', FUSED): _t(0.014, 0.015, 0.004, 0.05),
    # star_confounder.  double_star's error is the unresolved companion's pull on the
    # photocenter.  lock_single_dense fails on 4 of the 16 seeds; its numbers come from
    # the 12 where it succeeds.
    ('double_star', FUSED): _t(0.388, 0.390, 0.006, 0.08),
    ('lock_pair', FUSED): _t(0.039, 0.041, 0.004, 0.05),
    ('lock_single', FUSED): _t(0.029, 0.030, 0.003, 0.05),
    ('lock_single_dense', FUSED): _t(0.058, 0.059, 0.010, 0.05),
    ('lock_triangle', FUSED): _t(0.055, 0.056, 0.003, 0.05),
    ('saturated_star', FUSED): _t(0.019, 0.021, 0.003, 0.05),
}

# Columns: smallest and largest error over the seeds, margin, band low, band high; px.
# A FUSED key is checked in test_sim_algorithmic_invariants.  A technique key is read
# from the fused run's per-technique results in test_sim_truth_checks, which also
# asserts the answer is not flagged spurious, so flagging it shows up too.
KNOWN_ERRORS: dict[tuple[str, str], KnownErrorBand] = {
    # The disc correlation stops at the edge of its +-0.5 px refinement window, 1.5 px
    # for a planted 1.43 px (#791).
    ('planted_offset_disc', FUSED): _b(0.0705, 0.0705, 0.05, 0.020, 0.121),
    ('planted_offset_disc_if', FUSED): _b(0.0705, 0.0719, 0.05, 0.020, 0.122),
    ('planted_offset_small', FUSED): _b(0.0705, 0.0705, 0.05, 0.020, 0.121),
    # BodyBlobNav answers far from the truth without flagging them spurious (#975).
    ('two_moons', 'BodyBlobNav'): _b(36.293, 36.309, 0.05, 36.243, 36.359),
    ('mutual_deep', 'BodyBlobNav'): _b(6.970, 6.980, 0.05, 6.920, 7.031),
    ('mutual_half', 'BodyBlobNav'): _b(8.474, 8.741, 0.0533, 8.421, 8.795),
    ('mutual_grazing', 'BodyBlobNav'): _b(1.822, 1.834, 0.05, 1.772, 1.884),
    ('titan_crescent_horns', 'BodyBlobNav'): _b(28.230, 30.533, 0.4604, 27.770, 30.994),
    ('titan_crescent_horns_noiseless', 'BodyBlobNav'): _b(30.407, 30.408, 0.05, 30.357, 30.458),
    ('titan_haze_limb', 'BodyBlobNav'): _b(1.241, 1.347, 0.05, 1.191, 1.397),
    ('textured_moon_offset', 'BodyBlobNav'): _b(0.427, 2.440, 0.4023, 0.025, 2.842),
    # The band's lower end would fall below zero, so it stops at zero.
    ('relief_psf_stray_light', 'BodyBlobNav'): _b(0.193, 1.920, 0.3454, 0.000, 2.266),
    # RingEdgeNav lands 51.8 px off along the straight edge, the unobservable axis
    # (#755).
    ('planted_rank1_ring_flat', 'RingEdgeNav'): _b(51.753, 51.757, 0.05, 51.703, 51.807),
    # The roll is fitted about the extended field of view's corner, so the star
    # techniques' translations carry about 5 px (#434).
    ('planted_rotation_star_field', 'StarUniqueMatchNav'): _b(5.011, 5.071, 0.05, 4.961, 5.121),
    ('planted_rotation_star_field', 'StarRefineNav'): _b(4.780, 4.870, 0.05, 4.730, 4.920),
}

# The rank-1 flat-ring scene misses its planted shift along the one observable axis, the
# edge normal, by about 0.52 px (#873).  The band holds that component's magnitude.
RANK1_NORMAL_ERROR = _b(0.5158, 0.5206, 0.05, 0.465, 0.571)

# Columns as KNOWN_ERRORS, in sigmas: the fused error in units of the reported
# covariance, for the scenes that sit above SIGMA_MULTIPLE for a known reason.
SIGMA_BANDS: dict[str, KnownErrorBand] = {
    # The two-star fused sigma is about 0.01 px against a steady 0.04 px error (#788).
    'lock_pair': _b(3.186, 4.479, 0.2585, 2.927, 4.738),
    # The observable component misses by 0.52 px against a 0.16 px sigma (#873).
    'planted_rank1_ring_flat': _b(3.132, 3.288, 0.1, 3.032, 3.388),
    # The translation is fitted about the extended field of view's corner (#434).
    'planted_rotation_star_field': _b(16.428, 16.517, 0.1, 16.328, 16.617),
}

# The planted roll is recovered 0.018 to 0.020 deg short over the seeds; the margin is
# the larger of 0.01 deg and a fifth of the worst error.
ROLL_WORST_ERROR_DEG = 0.020
ROLL_MARGIN_DEG = 0.01
ROLL_TOLERANCE_DEG = ROLL_WORST_ERROR_DEG + ROLL_MARGIN_DEG


def offset_error_px(offset_px: tuple[float, float], scene: dict[str, Any]) -> float:
    """Return the distance between an offset and the scene's planted offset, in pixels."""
    return math.hypot(
        offset_px[0] - float(scene.get('offset_v', 0.0)),
        offset_px[1] - float(scene.get('offset_u', 0.0)),
    )


def assert_matches_truth(
    offset_px: tuple[float, float] | None, scene: dict[str, Any], technique: str
) -> None:
    """Assert an offset meets the scene's truth check for ``technique``.

    The check is the scene's tolerance when :data:`TRUTH_TOLERANCES` holds one, and its
    known-error band when :data:`KNOWN_ERRORS` does.

    Parameters:
        offset_px: The navigated ``(dv, du)``, or None when navigation gave no offset.
        scene: The validated scene mapping, which holds the planted offset.
        technique: :data:`FUSED` or the technique name the offset came from.

    Raises:
        AssertionError: When the offset is missing, outside the tolerance, or outside
            the band, or when the scene has no truth check for ``technique``.
    """
    name = scene['scene_name']
    key = (name, technique)
    assert offset_px is not None, f'{name} ({technique}): navigation gave no offset'
    error = offset_error_px(offset_px, scene)
    if key in KNOWN_ERRORS:
        band = KNOWN_ERRORS[key]
        assert band.contains(error), (
            f'{name} ({technique}): offset error {error:.3f} px left the known-error band '
            f'[{band.low}, {band.high}] px (measured {band.smallest} to {band.largest} px)'
        )
        return
    assert key in TRUTH_TOLERANCES, f'{name}: no truth check for {technique}'
    tolerance = TRUTH_TOLERANCES[key]
    assert error <= tolerance.tolerance_px, (
        f'{name} ({technique}): offset error {error:.3f} px exceeds the tolerance '
        f'{tolerance.tolerance_px:.3f} px (worst measured {tolerance.worst_px:.3f} px '
        f'plus margin {tolerance.margin_px:.2f} px)'
    )


def error_in_sigmas(result: NavResult, scene: dict[str, Any]) -> float:
    """Return the fused error measured in units of the reported covariance.

    This is the Mahalanobis distance ``sqrt(e^T C^+ e)`` of the error ``e`` under the
    fused 2-by-2 covariance ``C``.  The pseudo-inverse measures a rank-1 covariance
    along its one observable axis.  It is the combine step's own, which inverts a
    sentinel-sized unobservable variance on its own axis, so that variance cannot
    raise the eigenvalue cutoff past the observable axis and hide a miss along it.

    Parameters:
        result: A fused result that carries an offset.
        scene: The validated scene mapping, which holds the planted offset.
    """
    assert result.offset_px is not None
    error = np.array(
        [
            result.offset_px[0] - float(scene.get('offset_v', 0.0)),
            result.offset_px[1] - float(scene.get('offset_u', 0.0)),
        ]
    )
    cov = np.asarray(result.covariance_px2, dtype=np.float64)[:2, :2]
    return float(math.sqrt(max(0.0, float(error @ mixed_scale_pinvh(cov, rcond=1e-9) @ error))))


def validation_scene_paths(root: Path) -> list[Path]:
    """Return every catalog scene under ``root`` whose class is a validation class."""
    return [p for p in iter_scene_paths(root) if scene_class_for_path(p) in VALIDATION_CLASSES]
