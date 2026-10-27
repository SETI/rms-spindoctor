"""Single-variable sweep invariants.

Each sweep drives one catalog scene by varying a single parameter and navigates
every step.  These tests assert how a navigation diagnostic *responds* to the
controlled change -- the verification layer a calibrated confidence formula
relies on.  Assertions are trends and bounds (a technique transition, a
degradation to failure, recovery within tolerance), not exact values, because the
underlying navigation carries sub-millipixel cross-process jitter.

Run in-process; heavier than the algorithmic invariants (each sweep navigates
several frames), so the module is ``@pytest.mark.integration`` -- the deliberate
tier, alongside the baselines.  The checks that only build and validate each sweep
step navigate nothing and run with the unit suite, in
``tests/spindoctor/sim/test_sim_sweep_build.py``.
"""

import copy
import functools
from pathlib import Path

import pytest

from tests.integration.sim_sweep import (
    SweepRow,
    build_sweep_params,
    load_sweep,
    run_sweep,
    run_sweep_steps,
)

pytestmark = pytest.mark.integration

_SWEEPS_ROOT = Path(__file__).parent / 'sim_sweeps'
_RECOVERY_TOLERANCE_PX = 0.5


def _rows(sweep_name: str) -> list[SweepRow]:
    """Load and run a sweep by name."""
    return run_sweep(load_sweep(_SWEEPS_ROOT / f'{sweep_name}.yaml'))


def test_noise_sweep_starts_clean() -> None:
    """At the lowest read noise the scene recovers the planted offset."""
    rows = _rows('noise_read_noise')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_noise_sweep_degrades_to_failure() -> None:
    """At the highest read noise navigation fails -- the navigability cliff."""
    rows = _rows('noise_read_noise')
    assert rows[-1].status == 'failed'


def test_noise_sweep_low_noise_all_succeed() -> None:
    """Every below-cliff step recovers within tolerance."""
    rows = _rows('noise_read_noise')
    for row in rows:
        if row.status != 'success':
            continue
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_phase_sweep_navigates_every_phase() -> None:
    """The resolved body navigates to success across the full phase range."""
    rows = _rows('phase_regular_body')
    for row in rows:
        assert row.status == 'success'


def test_phase_sweep_recovers_every_phase() -> None:
    """The recovered offset stays within tolerance across the full phase range."""
    rows = _rows('phase_regular_body')
    for row in rows:
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_range_sweep_largest_body_uses_resolved_body_technique() -> None:
    """The largest (well-resolved) body navigates by a resolved-body technique.

    Disc correlation and the limb DT fit both recover a clean resolved body;
    which of the two ranks primary is a confidence-ordering question the
    confidence calibration owns (the sim-anchored fit ranks the disc correlation's
    ~0.03 px recovery above the limb's ~0.1 px gradient-peak floor), so the
    assertion accepts either rather than pinning the ordering.
    """
    rows = _rows('range_body_size')
    assert rows[0].primary_technique in ('BodyDiscCorrelateNav', 'BodyLimbNav')


def test_range_sweep_smallest_body_fails() -> None:
    """The smallest body is unnavigable."""
    rows = _rows('range_body_size')
    assert rows[-1].status == 'failed'


def test_range_sweep_reaches_blob_regime() -> None:
    """A small-but-navigable body falls to the orientation-free BodyBlobNav."""
    rows = _rows('range_body_size')
    primaries = [row.primary_technique for row in rows]
    assert 'BodyBlobNav' in primaries


def test_range_sweep_transitions_technique() -> None:
    """The primary technique is not constant -- the range ladder transitions."""
    rows = _rows('range_body_size')
    distinct = {row.primary_technique for row in rows if row.primary_technique is not None}
    assert len(distinct) >= 2


_ROLL_TOLERANCE_DEG = 0.3


def test_star_rotation_sweep_recovers_roll() -> None:
    """The star field recovers every planted roll in the working window.

    The recovered roll is read from whichever technique reports it (the two-star
    path at +/-1 deg, the field matcher at the larger rolls), so the assertion is
    on the roll error, not the fused status.
    """
    rows = _rows('star_rotation')
    for row in rows:
        assert row.rotation_error_deg is not None
        assert row.rotation_error_deg < _ROLL_TOLERANCE_DEG


# The irregularity walk navigates one mesh realization, the base scene's
# mesh_seed (4, the median shape mismatch of mesh seeds 0 to 12; see the scene).
# Its single-seed tests are tripwires on this realization.  The properties that
# hold across shapes are asserted on the population of mesh seeds 0 to 12 at the
# walk's extreme step, lumpiness 0.7, below.
_IRREGULARITY_POPULATION_SEEDS = range(13)
_IRREGULARITY_EXTREME_LUMPINESS = 0.7
# Measured 2026-10-27: at lumpiness 0.7 the shape-lock veto reports 4 of the 13
# seeds conflicted (mesh seeds 0, 2, 5, and 12).
_IRREGULARITY_MIN_VETOED = 4
# A "confident wrong" result: success at this confidence or above, this far off.
_CONFIDENT = 0.9
_IRREGULARITY_CONFIDENT_WRONG_PX = 5.0
# Every one of mesh seeds 0 to 12 dips at least this far below the clean step's
# confidence somewhere on its walk (the smallest dip is seed 10's, 0.31).
_IRREGULARITY_MIN_CONFIDENCE_DROP = 0.3


@functools.cache
def _irregularity_population() -> tuple[SweepRow, ...]:
    """Navigate the irregularity walk's extreme step once per mesh seed 0 to 12.

    Each row's ``value`` is its mesh seed.  Cached, so the population's tests
    share one set of navigations.
    """
    spec = load_sweep(_SWEEPS_ROOT / 'irregularity_shape_mismatch.yaml')
    extreme = [
        params
        for value, params in build_sweep_params(spec)
        if value == _IRREGULARITY_EXTREME_LUMPINESS
    ]
    assert len(extreme) == 1
    steps = []
    for mesh_seed in _IRREGULARITY_POPULATION_SEEDS:
        params = copy.deepcopy(extreme[0])
        params['bodies'][0]['mesh_seed'] = mesh_seed
        steps.append((float(mesh_seed), params))
    return tuple(run_sweep_steps(steps, spec.technique))


def test_irregularity_sweep_starts_matched() -> None:
    """At zero relief the fused recovery stays sub-half-pixel.

    The fused offset precision-weights the disc correlation (~0.03 px
    recovery here) against the limb DT fit (which carries the documented
    ~0.1 px-class gradient-peak/model offset and errs a few tenths on
    this mesh).  With the #210 model-error floors both report
    comparable honest sigmas, so the fused value is their blend rather
    than whichever technique used to claim the tighter covariance; the
    bound reflects that blend, not a single technique's floor.
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 0.5


def test_irregularity_sweep_bias_grows() -> None:
    """Shape mismatch grows the recovered centroid bias as relief increases.

    This realization ends 5.2 px off.  Every one of mesh seeds 0 to 12 ends
    more than 2 px off (the least, seed 11, at 2.7 px), but the error never
    falls between steps on only 4 of them (seeds 4, 5, 9, and 10).
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[-1].offset_error_px is not None
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 1.0


def test_irregularity_sweep_confidence_drops() -> None:
    """Shape mismatch pulls the fused confidence well below the clean step's.

    Somewhere on the walk the confidence falls at least 0.3 below the clean
    (zero-relief) step's 0.99.  That holds on every one of mesh seeds 0 to 12,
    but it is asserted on this realization only, so the test is a tripwire on
    this walk rather than a check of the population.  The bound is the
    population's smallest dip (seed 10's, 0.31) less 0.01; this walk dips 0.59,
    to 0.40 at lumpiness 0.7, so a failure here means the walk changed, and the
    population is what to measure next.

    Falling below it at every mismatched step holds only on seeds 0, 8, and 11,
    and from lumpiness 0.2 on only on six of them, so neither is asserted: on
    many shapes a mismatched step still reports 0.99 (this realization does at
    lumpiness 0.1, 0.2, and 0.6).
    """
    rows = _rows('irregularity_shape_mismatch')
    lowest = min(row.confidence for row in rows[1:])
    assert lowest <= rows[0].confidence - _IRREGULARITY_MIN_CONFIDENCE_DROP


def test_irregularity_sweep_extreme_step_is_not_a_confident_success() -> None:
    """This realization's extreme step is not reported as a confident success.

    At lumpiness 0.7 this realization's disc and limb lock onto the mismatched
    shape and the pose-free blob carries the result, a 0.40-confidence success
    5.2 px off; the shape-lock veto does not fire here.  The veto's rate over
    shapes is asserted on the population below, and so is the confident-wrong
    case this tripwire guards on this one shape.
    """
    rows = _rows('irregularity_shape_mismatch')
    extreme = rows[-1]
    confident_success = extreme.status == 'success' and extreme.confidence >= _CONFIDENT
    assert not confident_success


def test_irregularity_sweep_clean_step_still_succeeds() -> None:
    """The zero-relief step navigates cleanly: the veto does not blanket the sweep."""
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].status == 'success'


def test_irregularity_population_veto_rate_at_extreme_mismatch() -> None:
    """At lumpiness 0.7 the shape-lock veto declines at least 4 of 13 mesh shapes.

    Over mesh seeds 0 to 12 the disc correlation and the limb fit lock onto the
    mismatched shape at the extreme step.  The veto reports a shape
    ``conflicted`` when the pose-free blob disagrees with that lock; on the
    other shapes the blob is displaced with the lock or carries the result
    itself.  The bound is the measured rate, 4 of 13 (seeds 0, 2, 5, and 12), so
    a change that weakens the veto on these shapes fails here.
    """
    vetoed = [row for row in _irregularity_population() if row.status == 'conflicted']
    assert len(vetoed) >= _IRREGULARITY_MIN_VETOED


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='#790: the shape-lock veto misses confident wrong solutions on irregular mesh '
    'bodies; mesh seed 10 ends a 0.91-confidence success 12.7 px off at lumpiness 0.7',
)
def test_irregularity_population_never_confidently_wrong() -> None:
    """No mesh shape at lumpiness 0.7 is a confident (>= 0.9) success more than 5 px off.

    A frame whose fit is several pixels off must not report a confident success,
    whatever the shape.  Mesh seed 10 does: its blob is displaced together with
    the disc lock, so the body-witness disagreement the veto reads never appears.
    """
    confident_wrong = [
        int(row.value)
        for row in _irregularity_population()
        if row.status == 'success'
        and row.confidence >= _CONFIDENT
        and row.offset_error_px is not None
        and row.offset_error_px > _IRREGULARITY_CONFIDENT_WRONG_PX
    ]
    assert confident_wrong == []


def test_pose_disagreement_starts_clean() -> None:
    """With the predicted pose agreeing, the limb recovers the planted offset."""
    rows = _rows('pose_disagreement')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 1.0


def test_pose_disagreement_limb_error_grows() -> None:
    """The limb fit error grows monotonically as the predicted pose drifts."""
    rows = _rows('pose_disagreement')
    first = rows[0].offset_error_px
    second = rows[1].offset_error_px
    last = rows[-1].offset_error_px
    assert first is not None
    assert second is not None
    assert last is not None
    assert second > first
    assert last > first + 1.0
    assert last > 2.0


def test_pose_disagreement_limb_is_confidently_wrong() -> None:
    """At the largest disagreement the limb is several pixels off yet still 'success'.

    Apart from the 55 deg step, where it self-flags spurious and the frame fails,
    the pinned limb does not self-flag across this tumble range -- it returns a
    confidently-wrong fix (a multi-pixel error at unchanged confidence), which is
    why the per-technique demote decision (test_sim_irregular_pose) cannot rely on
    the limb's own status and compares it against the pose-free blob instead.
    """
    rows = _rows('pose_disagreement')
    assert rows[-1].status == 'success'
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0


def test_artifact_sweep_starts_clean() -> None:
    """At zero missing-line expected_events the frame recovers the planted offset."""
    rows = _rows('artifact_missing_lines')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_artifact_sweep_confidence_degrades() -> None:
    """Structured loss drops the fused confidence below the clean baseline.

    The deliverable curve is navigation quality versus artifact severity: as
    more lines are lost the fused confidence falls below its clean (rows[0])
    value.  The assertion is on the confidence dip (the render-robust signal that
    loss degrades the navigation), not a monotone step-by-step drop, because the
    recovered offset near the cliff jitters across processes.
    """
    rows = _rows('artifact_missing_lines')
    degraded_min = min(row.confidence for row in rows[1:])
    assert degraded_min < rows[0].confidence


def test_artifact_sweep_degrades_to_failure() -> None:
    """At the highest expected count half the frame is gone and navigation fails."""
    rows = _rows('artifact_missing_lines')
    assert rows[-1].status == 'failed'


_CONFOUNDER_RECOVERY_TOLERANCE_PX = 0.5
# A success whose recovered offset lands farther than this from the planted
# truth is a confident wrong offset -- the failure mode the confounder work
# exists to rule out.  A true false lock is many pixels off, so a generous
# threshold keeps the invariant immune to the sub-pixel cross-process jitter.
_CONFIDENT_WRONG_PX = 1.0


def test_star_confounder_low_density_recovers() -> None:
    """At the lowest confounder density every seed recovers the planted offset."""
    rows = _rows('star_confounder_density')
    lowest = min(row.value for row in rows)
    low_rows = [row for row in rows if row.value == lowest]
    assert low_rows
    for row in low_rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _CONFOUNDER_RECOVERY_TOLERANCE_PX


def test_star_confounder_never_confidently_wrong() -> None:
    """No sweep point returns a confident wrong offset.

    The deliverable safety property: at every density the navigator either
    recovers the offset within tolerance or reports failure / low confidence --
    it never locks confidently onto a confounder.  A success farther than the
    confident-wrong threshold from the planted offset fails this test.
    """
    rows = _rows('star_confounder_density')
    for row in rows:
        if row.status == 'success':
            assert row.offset_error_px is not None
            assert row.offset_error_px < _CONFIDENT_WRONG_PX


def test_star_confounder_breaks_down_at_high_density() -> None:
    """Rising confounder density degrades the success rate to the failure regime.

    The curve's whole point is that it breaks: the high-density success rate sits
    below the clean floor, and at least one high-density realization fails
    outright.  The specific cliff location jitters across processes, so the
    assertion is on the aggregate degradation, not a per-point status.
    """
    rows = _rows('star_confounder_density')
    high_rows = [row for row in rows if row.value >= 200.0]
    assert high_rows
    high_success_rate = sum(row.status == 'success' for row in high_rows) / len(high_rows)
    assert high_success_rate < 1.0
    assert any(row.status != 'success' for row in high_rows)


def test_star_confounder_ensemble_replicates_each_point() -> None:
    """The ensemble mode navigates every density across the full seed population."""
    spec = load_sweep(_SWEEPS_ROOT / 'star_confounder_density.yaml')
    rows = run_sweep(spec)
    assert len(rows) == len(spec.values) * spec.ensemble_seeds
    for value in spec.values:
        seeds = {row.seed for row in rows if row.value == value}
        assert len(seeds) == spec.ensemble_seeds


def test_star_catalog_scatter_zero_recovers() -> None:
    """With a perfect catalog (scatter 0) every seed recovers the planted offset."""
    rows = _rows('star_catalog_scatter')
    clean = [row for row in rows if row.value == 0.0]
    assert clean
    for row in clean:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_star_catalog_scatter_wholesale_error_never_succeeds() -> None:
    """In the wholesale-catalog-error regime no seed ever reports success.

    Only the safety envelope is asserted: clean recovery at scatter 0 (above)
    and no success once the scatter reaches the wholesale-error regime the
    wrong_catalog expected_fail scene pins at 8 px.  The intermediate region is
    deliberately NOT asserted -- it is characterization raw material, recorded
    below for the confidence recalibration work.
    """
    rows = _rows('star_catalog_scatter')
    wholesale = [row for row in rows if row.value >= 6.0]
    assert wholesale
    for row in wholesale:
        assert row.status != 'success'


# Measured star_catalog_scatter curve (2026-07-16, seeds 7-9), recorded for the
# astrometric-residual characterization: this is observed behavior, not asserted
# correctness (#291).  The intermediate region degrades gracefully but the
# navigator does not yet self-flag the growing astrometric residual:
#   scatter 0.5-1.0 px: all seeds success, errors 0.16-0.67 px, high tiers;
#   scatter 1.5 px:     first outright failure (1 of 3 seeds);
#   scatter 2.0 px:     all seeds "success" at 0.6-1.3 px error (a medium-tier
#                       success at ~1.2 px error is the currently-observed
#                       behavior, never asserted as correct);
#   scatter 3.0 px:     2 of 3 seeds fail; the surviving success errs 0.7 px;
#   scatter 4.0 px:     population splits -- 2 of 3 seeds fail, one seed still
#                       reports a medium-tier (0.66) success at 2.8 px error,
#                       which is why the never-succeeds assertion starts at 6;
#   scatter 6.0-8.0 px: every seed fails (all techniques spurious).


# ---------------------------------------------------------------------------
# Section 8 model-mismatch axes: each walks a single render-vs-navigate mismatch
# from a self-consistency floor (value 0, or equality with the navigator's
# configuration) and asserts the recovery-error-vs-mismatch response.  The floor
# point recovers; the mismatched points degrade.  Absolute levels are
# sim-anchored, so the assertions are on the floor / degradation trend.
# ---------------------------------------------------------------------------


def test_psf_mismatch_floor_recovers() -> None:
    """At the navigator-matched PSF (sigma 0.54) the limb recovers the offset."""
    rows = _rows('psf_limb_mismatch')
    assert rows[0].value == 0.54
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_psf_mismatch_degrades_to_cliff() -> None:
    """A broad rendered PSF grows the limb error, then washes the edge out."""
    rows = _rows('psf_limb_mismatch')
    assert rows[0].offset_error_px is not None
    mismatched = [r.offset_error_px for r in rows[1:] if r.offset_error_px is not None]
    assert max(mismatched) > rows[0].offset_error_px + 0.2
    assert rows[-1].status == 'failed'


def test_psf_wings_floor_recovers() -> None:
    """At zero wing energy the kernel is the navigator-matched pure Gaussian."""
    rows = _rows('psf_limb_wings')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_psf_wings_bias_stays_gentle() -> None:
    """Unmodeled Moffat wing energy biases the limb gently, without a cliff.

    Unlike the core-sigma axis (whose blur washes the limb gradient out to
    outright failure), moving up to 60% of the kernel energy into the
    isotropic wing leaves the core sharp: every step still recovers within
    tolerance, and the error at the largest wing sits above the floor
    (measured ~0.08 px -> ~0.14 px).  The axis pins that measured shape --
    wing mismatch alone does not break the DT limb fit.
    """
    rows = _rows('psf_limb_wings')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX
    assert rows[-1].offset_error_px is not None
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px


def test_photometric_mismatch_floor_recovers() -> None:
    """Minnaert k=1 is Lambert, so the floor recovers the planted offset."""
    rows = _rows('photometric_minnaert_mismatch')
    assert rows[0].value == 1.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_photometric_mismatch_biases_recovery() -> None:
    """Walking off Lambert grows the disc-correlation centroid bias."""
    rows = _rows('photometric_minnaert_mismatch')
    assert rows[0].offset_error_px is not None
    mismatched = [r.offset_error_px for r in rows[1:] if r.offset_error_px is not None]
    assert max(mismatched) > rows[0].offset_error_px + 0.2


def test_spk_parallax_floor_recovers() -> None:
    """With no parallax shift the body sits at its catalog position."""
    rows = _rows('spk_parallax_error')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_spk_parallax_error_tracks_planted_shift() -> None:
    """The recovered-offset error grows with the planted parallax shift.

    A spacecraft-position error moves the whole body while the navigator predicts
    the unshifted catalog geometry, so the recovery absorbs the shift almost
    one-for-one -- the error at the largest planted shift approaches its
    magnitude.
    """
    rows = _rows('spk_parallax_error')
    assert rows[-1].value == 6.0
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 4.0


def test_differential_smear_floor_recovers() -> None:
    """With no star trail the sharp field recovers the planted offset."""
    rows = _rows('differential_smear')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_differential_smear_degrades_at_long_trail() -> None:
    """A long star trail drifts the centroids off their catalog positions."""
    rows = _rows('differential_smear')
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 0.05


def test_ring_orbit_error_floor_recovers() -> None:
    """With no radial orbit error the ring edge recovers the planted offset."""
    rows = _rows('ring_orbit_error')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_ring_orbit_error_absorbs_into_offset() -> None:
    """A radial ring-orbit error is absorbed into the recovered offset."""
    rows = _rows('ring_orbit_error')
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0


def test_atmosphere_haze_floor_recovers() -> None:
    """With no haze the hard limb recovers the planted offset."""
    rows = _rows('atmosphere_haze')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_atmosphere_haze_biases_limb() -> None:
    """A thickening haze lifts and blurs the limb, biasing the recovery."""
    rows = _rows('atmosphere_haze')
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 0.1


# The per-technique dense and wide offset sweeps (``*_offset_fine`` /
# ``*_offset_wide``) are characterization runs, not assertions: they are executed
# by ``sim_sweep_runner`` to produce the report's figures, and the specific defect
# they expose (the disc gradient-NCC sub-pixel bias) is guarded by the fast
# ``test_sim_regression`` case rather than by re-running the full sweep here.
