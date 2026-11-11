"""Truth checks for the simulator's validation scenes: per-scene tolerances from measurement.

A validation scene plants a known pointing offset, so the navigator's answer can be
compared with the truth.  This module holds the tolerance each such comparison is held
to, keyed by ``(scene_name, technique)``: :data:`FUSED` for the orchestrator's fused
answer, or a technique name for a technique navigated alone (or read from the fused run's
per-technique results).  Every tolerance bounds the error as a distance in two dimensions,
``hypot(dv_error, du_error)``, never one axis at a time.

How a tolerance is set.  Each scene was rendered and navigated on its committed
``random_seed`` and on 15 more (101, 202, ..., 1515), so that a tolerance covers the
spread over noise draws rather than one draw.  An entry records the error on the
committed seed, the largest error over the seeds, and the spread (largest minus
smallest).  The margin is the larger of 0.05 px and a fifth of the largest error, rounded
up to 0.01 px, and the tolerance is the largest error plus the margin.  The 0.05 px floor
covers the drift a baseline already allows (0.02 px per axis) and the jitter between
machines; the proportional part covers seeds the measurement did not sample.  A scene
that fails on some seeds has its error measured over the seeds where it succeeds.

A scene whose answer is wrong for a known reason is held to a band around its known
error instead, so a fix shows up as a deliberate change rather than passing silently.
A fused known error lives in the scene's ``expected`` block
(``known_offset_error_px`` / ``known_offset_error_tol_px``, asserted by
:mod:`tests.integration.sim_expected`); a known error on one component lives here as a
:class:`KnownErrorBand`.  Each end of a band sits outside the measured extreme by the
same margin rule.

The fused answer is also checked against the sigma it reports: the error measured in
units of the reported covariance must not exceed :data:`SIGMA_MULTIPLE`.  For a round
covariance that measure is the error divided by the per-axis sigma.  Measured on the
committed seeds and the 15 others, every accurate validation scene sits at or below
1.9 times its reported sigma; 3 leaves room above that and still fails an answer that
is confidently wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene import iter_scene_paths, scene_class_for_path

FUSED = '*'
"""The technique key for the orchestrator's fused answer."""

SEEDS_MEASURED = 16
"""The number of noise seeds each tolerance was measured over (the committed one and 15)."""

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
    """A known error, its measured range over the seeds, and the band that holds it.

    Parameters:
        smallest_px: The smallest error measured over the seeds.
        largest_px: The largest error measured over the seeds.
        low_px: The band's lower end, a margin below ``smallest_px``.
        high_px: The band's upper end, a margin above ``largest_px``.
    """

    smallest_px: float
    largest_px: float
    low_px: float
    high_px: float

    def contains(self, error_px: float) -> bool:
        """Return True when ``error_px`` lies inside the band."""
        return self.low_px <= error_px <= self.high_px


def _t(error_px: float, worst_px: float, spread_px: float, margin_px: float) -> TruthTolerance:
    return TruthTolerance(
        error_px=error_px, worst_px=worst_px, spread_px=spread_px, margin_px=margin_px
    )


# Columns: error on the committed seed, worst error over the seeds, spread over the
# seeds, margin.  All in pixels; the tolerance is worst + margin.
TRUTH_TOLERANCES: dict[tuple[str, str], TruthTolerance] = {
    # algorithmic_invariants, fused answer (disc and star scenes).  The disc scenes'
    # 0.07 px is the correlation's answer stopping at the edge of its +-0.5 px
    # refinement window, 1.5 px for a planted 1.43 px (#791).
    ('planted_offset_disc', FUSED): _t(0.071, 0.071, 0.000, 0.05),
    ('planted_offset_disc_if', FUSED): _t(0.071, 0.072, 0.001, 0.05),
    ('planted_offset_irregular', FUSED): _t(0.001, 0.001, 0.000, 0.05),
    ('planted_offset_limb', FUSED): _t(0.067, 0.073, 0.009, 0.05),
    ('planted_offset_limb_mesh', FUSED): _t(0.076, 0.091, 0.027, 0.05),
    # The prediction is the smooth limit of the rendered lumpy mesh; this is the price
    # of that shape mismatch.
    ('planted_offset_shapemismatch', FUSED): _t(0.667, 0.675, 0.013, 0.14),
    ('planted_offset_small', FUSED): _t(0.071, 0.071, 0.000, 0.05),
    ('planted_offset_star_field', FUSED): _t(0.018, 0.019, 0.003, 0.05),
    ('planted_offset_star_field_bright', FUSED): _t(0.040, 0.041, 0.001, 0.05),
    # algorithmic_invariants, one technique.
    ('planted_offset_blob', 'BodyBlobNav'): _t(0.006, 0.009, 0.007, 0.05),
    ('planted_offset_blob_crescent', 'BodyBlobNav'): _t(0.253, 0.282, 0.062, 0.06),
    ('planted_offset_blob_crescent_displaced', 'BodyBlobNav'): _t(0.252, 0.278, 0.057, 0.06),
    ('planted_offset_blob_mesh_crescent', 'BodyBlobNav'): _t(0.486, 0.592, 0.114, 0.12),
    ('planted_offset_limb', 'BodyLimbNav'): _t(0.099, 0.099, 0.050, 0.05),
    ('planted_offset_limb_mesh', 'BodyLimbNav'): _t(0.175, 0.255, 0.127, 0.06),
    ('planted_offset_ring', 'RingEdgeNav'): _t(0.011, 0.016, 0.005, 0.05),
    ('planted_unique_star_single', 'StarUniqueMatchNav'): _t(0.054, 0.058, 0.006, 0.05),
    ('planted_refine_star_field', 'StarRefineNav'): _t(0.016, 0.019, 0.003, 0.05),
    # artifact_sweep.  The seed places the lost lines, so the spread is large.
    ('disc_structured_loss', FUSED): _t(0.080, 0.447, 0.439, 0.09),
    ('disc_structured_loss_adversarial', FUSED): _t(0.141, 0.908, 0.870, 0.19),
    ('star_field_structured_loss', FUSED): _t(0.067, 0.109, 0.094, 0.05),
    ('star_field_structured_loss_adversarial', FUSED): _t(0.133, 0.443, 0.426, 0.09),
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
    # regression, one technique.
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

# The rank-1 flat-ring scene misses its planted shift along the one observable axis, the
# edge normal, by about 0.52 px (#873).  The band holds that component's magnitude.
RANK1_NORMAL_ERROR = KnownErrorBand(smallest_px=0.516, largest_px=0.521, low_px=0.41, high_px=0.63)

# The planted roll is recovered 0.018 to 0.020 deg short over the seeds; the margin is
# the larger of 0.01 deg and a fifth of the worst error.
ROLL_WORST_ERROR_DEG = 0.020
ROLL_MARGIN_DEG = 0.01
ROLL_TOLERANCE_DEG = ROLL_WORST_ERROR_DEG + ROLL_MARGIN_DEG

# Fused answers known to sit far outside their reported sigma, each tied to the issue
# that explains it.  The sigma check expects these to fail, so a fix shows up.
SIGMA_EXCEPTIONS: dict[str, str] = {
    # The two-star fused sigma is about 0.01 px against a steady 0.04 px error (#788).
    'lock_pair': 'two-star fused sigma is too small',
    # The along-edge component is unobservable; the observable one misses by 0.52 px
    # against a 0.16 px sigma (#873).
    'planted_rank1_ring_flat': 'rank-1 ring misses its observable component',
    # The translation is fitted about the extended field of view's corner (#434).
    'planted_rotation_star_field': 'roll is fitted about the extended-FOV corner',
}


def tolerance_for(scene_name: str, technique: str) -> TruthTolerance:
    """Return the tolerance for one scene and technique, failing clearly when it has none."""
    key = (scene_name, technique)
    assert key in TRUTH_TOLERANCES, f'{scene_name}: no truth tolerance for {technique}'
    return TRUTH_TOLERANCES[key]


def offset_error_px(offset_px: tuple[float, float], scene: dict[str, Any]) -> float:
    """Return the distance between an offset and the scene's planted offset, in pixels."""
    return math.hypot(
        offset_px[0] - float(scene.get('offset_v', 0.0)),
        offset_px[1] - float(scene.get('offset_u', 0.0)),
    )


def assert_within_tolerance(
    offset_px: tuple[float, float] | None, scene: dict[str, Any], technique: str
) -> None:
    """Assert an offset lies within the scene's truth tolerance for ``technique``.

    Parameters:
        offset_px: The navigated ``(dv, du)``, or None when navigation gave no offset.
        scene: The validated scene mapping, which holds the planted offset.
        technique: :data:`FUSED` or the technique name the offset came from.

    Raises:
        AssertionError: When the offset is missing or lies outside the tolerance.
    """
    name = scene['scene_name']
    assert offset_px is not None, f'{name} ({technique}): navigation gave no offset'
    tolerance = tolerance_for(name, technique)
    error = offset_error_px(offset_px, scene)
    assert error <= tolerance.tolerance_px, (
        f'{name} ({technique}): offset error {error:.3f} px exceeds the tolerance '
        f'{tolerance.tolerance_px:.3f} px (worst measured {tolerance.worst_px:.3f} px '
        f'plus margin {tolerance.margin_px:.2f} px)'
    )


def error_in_sigmas(result: NavResult, scene: dict[str, Any]) -> float:
    """Return the fused error measured in units of the reported covariance.

    This is the Mahalanobis distance ``sqrt(e^T C^+ e)`` of the error ``e`` under the
    fused 2-by-2 covariance ``C``.  The pseudo-inverse measures a rank-1 covariance
    along its one observable axis.

    Parameters:
        result: A fused result that carries an offset.
        scene: The validated scene mapping, which holds the planted offset.

    Returns:
        The error in units of sigma.
    """
    assert result.offset_px is not None
    error = np.array(
        [
            result.offset_px[0] - float(scene.get('offset_v', 0.0)),
            result.offset_px[1] - float(scene.get('offset_u', 0.0)),
        ]
    )
    cov = np.asarray(result.covariance_px2, dtype=np.float64)[:2, :2]
    return float(math.sqrt(max(0.0, float(error @ np.linalg.pinv(cov, rcond=1e-9) @ error))))


def validation_scene_paths(root: Path) -> list[Path]:
    """Return every catalog scene under ``root`` whose class is a validation class."""
    return [p for p in iter_scene_paths(root) if scene_class_for_path(p) in VALIDATION_CLASSES]
