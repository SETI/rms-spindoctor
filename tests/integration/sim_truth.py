"""Truth checks for the simulator's validation scenes, with tolerances measured per scene.

A validation scene plants a known pointing offset, so the navigator's answer can be
compared with the truth.  This module holds the limit for each comparison.  The tables are
keyed by ``(scene_name, technique)``, where ``technique`` is :data:`FUSED` for the
orchestrator's fused answer or a technique name for one technique's answer.  Every check
measures the error as a distance in two dimensions, ``hypot(dv_error, du_error)``, never
one axis at a time.

The measurement.  Each scene was rendered and navigated on its committed
``random_seed`` and on 15 more (101, 202, ..., 1515), so that each check covers the
spread over noise draws rather than one draw.  A scene that fails on some seeds is
measured over the seeds where it succeeds.  Where a scene's ``random_seed`` also draws
part of the scene, the seeds varied that part too: the spokes of ``spoked_sheet``, the
confounder stars of ``lock_single_dense``, and the catalog scatter of ``wrong_catalog``.
``textured_moon_offset`` and ``relief_psf_stray_light`` were measured on 16 draws of the
detector noise alone instead: the committed draw and the scene key ``noise_seed`` at 101,
202, ..., 1515.  Those draws keep the albedo texture and the limb relief the test
renders, which a new ``random_seed`` would redraw.  Every check also holds on 9 draws of
the detector noise alone, ``noise_seed`` at 101 to 808.

The two rules.

* **An accurate answer** is held to a tolerance: the largest error over the seeds plus a
  margin of max(0.05 px, a fifth of that largest error), rounded up to 0.01 px.  The
  0.05 px floor covers the drift a baseline already allows (0.02 px per axis) and the
  jitter between machines; the proportional part covers seeds the measurement did not
  sample.  An entry in :data:`TRUTH_TOLERANCES` records the error on the committed seed,
  the largest error, the spread (largest minus smallest), and the margin.
* **A known wrong answer** is wrong for a reason an issue explains, or by design, where
  the scene plants an error the technique cannot see, such as a wrong body ephemeris.
  It is held to a band around its known error, so a fix, or a scene that loses its
  planted error, shows up as a deliberate change.  A known error is a
  bias, and its spread over the seeds is its noise, so the margin scales with the spread.
  The margin is max(0.05 px, a fifth of the spread).  The band runs from the smallest
  error minus the margin to the largest error plus the margin, with each end rounded
  outward to 0.001 px.  Bands on one technique's answer or on one component live here
  (:data:`KNOWN_ERRORS`, :data:`RANK1_NORMAL_ERROR`).  A fused band on a scene outside
  ``algorithmic_invariants`` lives in the scene's ``expected`` block, as
  ``known_offset_error_px`` and ``known_offset_error_tol_px``, and
  :mod:`tests.integration.sim_expected` asserts it.

The fused answer is also checked against the sigma it reports: the error measured in
units of the reported covariance must not exceed :data:`SIGMA_MULTIPLE`.  For a round
covariance that measure is the error divided by the per-axis sigma.  Over the seeds,
every accurate validation scene sits at or below 1.9 times its reported sigma.  A limit
of 3 leaves room above that and still fails an answer that is confidently wrong.  A scene
that sits above 3 for a known reason is held to a band on that measure
(:data:`SIGMA_BANDS`) with a margin of max(0.1, a fifth of its spread), so a fix leaves
the band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.ensemble_observability import mixed_scale_pinvh
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene.io import iter_scene_paths, scene_class_for_path

FUSED = '*'
"""The technique key for the orchestrator's fused answer."""

SEEDS_MEASURED = 16
"""The number of noise seeds each check covers: the committed seed and 15 others."""

SIGMA_MULTIPLE = 3.0
"""The largest fused error a validation scene may show, in sigmas of its covariance."""

VALIDATION_CLASSES: frozenset[str] = frozenset(
    {
        'algorithmic_invariants',
        'artifact_sweep',
        'atmosphere',
        'expected_fail',
        'full_frame_cameras',
        'mixed_content',
        'multi_body_geometry',
        'mutual_event',
        'regression',
        'ring_system',
        'star_confounder',
    }
)
"""The scene classes that check accuracy on a frame.  The remaining classes are sweeps."""

OWN_MODULE_CLASSES: frozenset[str] = frozenset({'full_frame_cameras', 'mixed_content'})
"""The validation classes that a test module of their own navigates.

These classes hold full-frame scenes.  Each class's module, ``test_sim_<class>.py``,
builds its tests with :func:`~tests.integration.sim_scene_checks.scene_class_tests`,
which navigates every scene once and applies all of the scene's checks to that one
result.  ``test_sim_truth_checks``, ``test_sim_expected``, and ``test_sim_baselines``
leave these classes out of their navigation and keep their checks that need none.
"""


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
    # algorithmic_invariants, fused answer (disc and star scenes).  The disc rows were
    # re-measured when the disc correlation's final pass moved to the high-pass
    # refinement, which reaches the planted 1.43 px where the half-pixel window around
    # the whole pixel stopped at 1.5 (#791).
    ('planted_offset_disc', FUSED): _t(0.001, 0.011, 0.010, 0.05),
    ('planted_offset_disc_if', FUSED): _t(0.001, 0.011, 0.010, 0.05),
    ('planted_offset_irregular', FUSED): _t(0.007, 0.011, 0.010, 0.05),
    ('planted_offset_limb', FUSED): _t(0.019, 0.025, 0.006, 0.05),
    ('planted_offset_limb_mesh', FUSED): _t(0.053, 0.058, 0.006, 0.05),
    # The prediction is the smooth limit of the rendered lumpy mesh.  This error is the
    # cost of that mismatch.  The high-pass disc refinement weighs the mismatched limb
    # more than the raw one did, so the error grew from 0.657 to 0.792 px (#791).
    ('planted_offset_shapemismatch', FUSED): _t(0.792, 0.798, 0.007, 0.16),
    ('planted_offset_small', FUSED): _t(0.001, 0.011, 0.010, 0.05),
    ('planted_offset_star_field', FUSED): _t(0.018, 0.019, 0.003, 0.05),
    ('planted_offset_star_field_bright', FUSED): _t(0.040, 0.041, 0.001, 0.05),
    # algorithmic_invariants, one technique navigated alone.  StarRefineNav is read from
    # the fused run, because it needs the pass-1 prior.
    ('planted_offset_blob', 'BodyBlobNav'): _t(0.002, 0.005, 0.004, 0.05),
    ('planted_offset_blob_crescent', 'BodyBlobNav'): _t(0.013, 0.017, 0.011, 0.05),
    ('planted_offset_blob_crescent_displaced', 'BodyBlobNav'): _t(0.006, 0.019, 0.013, 0.05),
    ('planted_offset_blob_mesh_crescent', 'BodyBlobNav'): _t(0.800, 0.805, 0.016, 0.17),
    ('planted_offset_limb', 'BodyLimbNav'): _t(0.197, 0.197, 0.022, 0.05),
    ('planted_offset_limb_mesh', 'BodyLimbNav'): _t(0.429, 0.433, 0.030, 0.09),
    ('planted_offset_ring', 'RingEdgeNav'): _t(0.029, 0.033, 0.005, 0.05),
    ('planted_unique_star_single', 'StarUniqueMatchNav'): _t(0.054, 0.058, 0.006, 0.05),
    ('planted_refine_star_field', 'StarRefineNav'): _t(0.016, 0.019, 0.003, 0.05),
    # artifact_sweep.  The scene seed also places the lost lines, so the 16 seeds would
    # render 16 different losses.  These rows keep the lost lines where the committed
    # seed puts them and vary only the noise draw.  The telemetry stages derive their own
    # sub-seeds from the scene seed, and the measurement fixed those at the committed
    # value.
    # The disc rows were re-measured for the high-pass disc refinement (#791).  On
    # disc_structured_loss a lost line across the disc pulls its maximum 0.12 px in v;
    # the raw refinement's maximum was as far off the other way, and its 0.080 px came
    # from stopping at its window edge.
    ('disc_structured_loss', FUSED): _t(0.134, 0.144, 0.010, 0.05),
    ('disc_structured_loss_adversarial', FUSED): _t(0.011, 0.011, 0.010, 0.05),
    ('star_field_structured_loss', FUSED): _t(0.067, 0.070, 0.004, 0.05),
    ('star_field_structured_loss_adversarial', FUSED): _t(0.133, 0.137, 0.005, 0.05),
    # atmosphere.  In titan_haze_limb, a haze-blind navigator fits a hard limb under a
    # soft haze limb, and the high-pass disc refinement ignores the haze's smooth glow
    # (#791).  titan_crescent_horns and its noiseless twin carry a known error in their
    # expected blocks instead (#1050).
    ('titan_haze', FUSED): _t(0.046, 0.063, 0.020, 0.05),
    ('titan_haze_limb', FUSED): _t(0.008, 0.014, 0.006, 0.05),
    # full_frame_cameras.  On these scenes a noise_seed draw gives the same answer as the
    # random_seed of the same value: random_seed reaches the noise and, apart from that,
    # only the fixed-pattern and sky streams, which change a few pixels of the render.
    # The unfit twist sets the larger errors on the star fields and the Galileo disc,
    # since a translation follows the turned stars or disc it measures.  The roll turns
    # the scene about the boresight, the frame center after the offset.  In
    # nhlorri_star_field the +0.19 deg twist moves the four measured stars' mean
    # position, about 280 px from the boresight, by about 0.94 px.  gossi_star_field
    # lands 0.05 px off with its twist and 0.02 px off without it, and gossi_disc
    # 0.17 px off with its twist and 0.04 px off without it.  coiss_wac_disc's error is
    # the limb fit's own: 0.16 px with the roll, 0.17 px without.  The three
    # short-exposure and large-offset scenes fail as their expected blocks declare
    # (#342, #785, #1153).
    ('coiss_calib_nac_disc', FUSED): _t(0.096, 0.099, 0.009, 0.05),
    ('coiss_calib_wac_star_field', FUSED): _t(0.015, 0.035, 0.028, 0.05),
    ('coiss_nac_star_field', FUSED): _t(0.017, 0.231, 0.220, 0.05),
    ('coiss_wac_disc', FUSED): _t(0.156, 0.166, 0.039, 0.05),
    ('gossi_disc', FUSED): _t(0.166, 0.182, 0.017, 0.05),
    ('gossi_star_field', FUSED): _t(0.054, 0.062, 0.018, 0.05),
    ('nhlorri_star_field', FUSED): _t(0.942, 0.962, 0.058, 0.20),
    # mixed_content.  In stars_body_ephemeris_error the body sits 1.81 px from where the
    # stars put it, and the fused answer lies between the two, 0.16 to 0.26 px from the
    # truth toward the body; the tolerance stays well below the body's own 1.81 px
    # error.  In crescent_stars the fused answer follows the stars, a little off the
    # disc's 0.01 px.  moon_over_ring_sheet is expected to conflict (#755).
    # saturn_rings_wac follows its ring edges.
    ('crescent_stars', FUSED): _t(0.013, 0.179, 0.166, 0.05),
    ('saturn_rings_wac', FUSED): _t(0.101, 0.101, 0.070, 0.05),
    ('stars_body_ephemeris_error', FUSED): _t(0.220, 0.258, 0.100, 0.06),
    # mixed_content, one technique read from the fused run.  StarRefineNav starts from
    # the body's pass-1 answer, which carries the planted 1.81 px error, and still lands
    # within 0.12 px of the truth, with the same answer on every draw as with the planted
    # error removed.  This row holds the star side of the scene: a refinement that
    # followed its start would land near 1.8 px off and fail here, while the fused
    # tolerance alone would pass one that followed it part way.
    ('stars_body_ephemeris_error', 'StarRefineNav'): _t(0.023, 0.113, 0.107, 0.05),
    # multi_body_geometry.
    ('two_moons', FUSED): _t(0.016, 0.016, 0.011, 0.05),
    # mutual_event.  All three scenes carry a known error in their expected blocks
    # instead.  In mutual_deep the far body's crescent, which the disc template leaves
    # out, pulls the disc correlation 0.24 to 0.26 px off (#1050).  In mutual_grazing and
    # mutual_half the composite disc template keeps a dark outline at the near body's
    # limb, which the image does not have, and it pulls the disc correlation 0.12 to
    # 0.13 px and 0.41 px off (#1099).
    # regression, fused answer.  relief_psf_stray_light carries a known error in its
    # expected block instead (#1050).
    ('banded_planet_transit', FUSED): _t(0.076, 0.084, 0.012, 0.05),
    ('disc_subpixel_offset', FUSED): _t(0.000, 0.008, 0.008, 0.05),
    ('low_phase_limb_misconverge', FUSED): _t(0.054, 0.054, 0.000, 0.05),
    ('star_equal_brightness', FUSED): _t(0.033, 0.033, 0.002, 0.05),
    ('star_zero_offset', FUSED): _t(0.044, 0.044, 0.003, 0.05),
    # Measured on the 16 noise draws, which keep the albedo texture: 0.363 to 0.368 px.
    ('textured_moon_offset', FUSED): _t(0.364, 0.368, 0.005, 0.08),
    # regression, one technique navigated alone.
    ('disc_subpixel_offset', 'BodyDiscCorrelateNav'): _t(0.000, 0.008, 0.008, 0.05),
    ('star_equal_brightness', 'StarFieldFromCatalogNav'): _t(0.033, 0.033, 0.002, 0.05),
    ('star_zero_offset', 'StarFieldFromCatalogNav'): _t(0.041, 0.041, 0.003, 0.05),
    # ring_system.  orbit_error_ringlet carries a known error in its expected block.
    ('edge_wave_gap', FUSED): _t(0.008, 0.012, 0.004, 0.05),
    ('mmode_ringlet', FUSED): _t(0.025, 0.026, 0.002, 0.05),
    ('moonlet_propeller', FUSED): _t(0.012, 0.015, 0.004, 0.05),
    ('spoked_sheet', FUSED): _t(0.026, 0.026, 0.004, 0.05),
    # star_confounder.  double_star's error is the unresolved companion's pull on the
    # photocenter.  lock_single_dense fails on 4 of the 16 seeds; its numbers come from
    # the 12 where it succeeds.
    ('double_star', FUSED): _t(0.388, 0.390, 0.006, 0.08),
    ('lock_pair', FUSED): _t(0.039, 0.041, 0.004, 0.05),
    ('lock_single', FUSED): _t(0.029, 0.030, 0.003, 0.05),
    ('lock_single_dense', FUSED): _t(0.055, 0.056, 0.006, 0.05),
    ('lock_triangle', FUSED): _t(0.055, 0.056, 0.003, 0.05),
    ('saturated_star', FUSED): _t(0.033, 0.035, 0.003, 0.05),
}

# Columns: smallest and largest error over the seeds, margin, band low, and band high,
# all in pixels.  test_sim_algorithmic_invariants checks a FUSED key.
# test_sim_truth_checks checks a technique key against that technique's result in the
# fused run.  It also asserts that the answer is not flagged spurious, so a change that
# flags it fails too.
KNOWN_ERRORS: dict[tuple[str, str], KnownErrorBand] = {
    # BodyBlobNav answers far from the truth and does not flag these answers spurious
    # (#975).
    ('two_moons', 'BodyBlobNav'): _b(36.293, 36.309, 0.05, 36.243, 36.359),
    ('mutual_deep', 'BodyBlobNav'): _b(6.720, 6.729, 0.05, 6.670, 6.780),
    ('mutual_half', 'BodyBlobNav'): _b(8.560, 8.807, 0.05, 8.509, 8.858),
    ('mutual_grazing', 'BodyBlobNav'): _b(3.443, 3.453, 0.05, 3.392, 3.503),
    ('titan_crescent_horns', 'BodyBlobNav'): _b(25.091, 25.123, 0.05, 25.041, 25.174),
    ('titan_crescent_horns_noiseless', 'BodyBlobNav'): _b(25.151, 25.151, 0.05, 25.100, 25.201),
    ('titan_haze_limb', 'BodyBlobNav'): _b(1.474, 1.482, 0.05, 1.424, 1.532),
    ('moon_over_ring_sheet', 'BodyBlobNav'): _b(0.942, 1.543, 0.1202, 0.821, 1.664),
    ('saturn_rings_wac', 'BodyBlobNav'): _b(6.863, 6.879, 0.05, 6.812, 6.929),
    # These two are measured on the 16 noise draws, which keep the albedo texture and
    # the limb relief.  The spreads are 0.006 px and 0.055 px.
    ('textured_moon_offset', 'BodyBlobNav'): _b(0.597, 0.604, 0.05, 0.547, 0.654),
    ('relief_psf_stray_light', 'BodyBlobNav'): _b(1.794, 1.850, 0.05, 1.744, 1.900),
    # RingEdgeNav lands 51.8 px off along the straight edge, the unobservable axis
    # (#755).
    ('planted_rank1_ring_flat', 'RingEdgeNav'): _b(51.753, 51.760, 0.05, 51.703, 51.810),
    # RingEdgeNav slides 8.4 to 9.3 px along gently curved arcs, with a sigma along them
    # near 0.1 px (#755).
    ('moon_over_ring_sheet', 'RingEdgeNav'): _b(8.432, 9.252, 0.1641, 8.267, 9.417),
    # These two are wrong by design.  The scene plants a spacecraft-ephemeris error
    # (spk_error) that moves the body 1.81 px from where the pointing puts it, and the
    # body techniques answer the pointing plus that error.  The bands hold the planted
    # error, so the scene fails if it stops reaching the body techniques: without it
    # they land 0.03 px and 0.09 to 0.10 px off.
    ('stars_body_ephemeris_error', 'BodyDiscCorrelateNav'): _b(1.811, 1.811, 0.05, 1.761, 1.862),
    ('stars_body_ephemeris_error', 'BodyLimbNav'): _b(1.831, 1.836, 0.05, 1.781, 1.887),
    # The roll is fitted about the extended field of view's corner, so the star
    # techniques' translations carry about 5 px of error (#434).
    ('planted_rotation_star_field', 'StarUniqueMatchNav'): _b(5.029, 5.088, 0.05, 4.979, 5.138),
    ('planted_rotation_star_field', 'StarRefineNav'): _b(4.604, 4.695, 0.05, 4.554, 4.745),
}

# The rank-1 flat-ring scene misses its planted shift along the one observable axis, the
# edge normal, by about 0.46 px (#873).  The render puts both edges on their radii.  The
# predicted edges' vertices sit at border-pixel centers (#1092), and along these straight
# edges, which run parallel to the v axis, the in-frame vertices the fit uses, 220 on
# each edge, sit 0.460 px toward lower u on average; the fit inherits that.  The band
# holds that component's magnitude.
RANK1_NORMAL_ERROR = _b(0.4581, 0.4624, 0.05, 0.408, 0.513)

# Columns as in KNOWN_ERRORS, but in sigmas: the fused error in units of the reported
# covariance.  These are the scenes that sit above SIGMA_MULTIPLE for a known reason.
SIGMA_BANDS: dict[str, KnownErrorBand] = {
    # The two-star fused sigma is about 0.01 px against a steady 0.04 px error (#788).
    'lock_pair': _b(3.186, 4.479, 0.2585, 2.927, 4.738),
    # The observable component misses by 0.46 px against a 0.095 px sigma (#873).
    'planted_rank1_ring_flat': _b(4.802, 4.902, 0.1, 4.702, 5.002),
    # The translation is fitted about the extended field of view's corner (#434).
    'planted_rotation_star_field': _b(16.333, 16.447, 0.1, 16.233, 16.547),
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
    along its one observable axis.  It is the one the combine step uses.  That
    pseudo-inverse inverts a sentinel-sized variance on an unobservable axis separately,
    so the variance cannot raise the eigenvalue cutoff above the observable axis and
    hide a miss along that axis.

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
