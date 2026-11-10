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
import itertools
from pathlib import Path

import numpy as np
import pytest

from spindoctor.sim.forward.atmosphere import DEFAULT_HAZE_ALBEDO
from spindoctor.sim.render import render_combined_model
from tests.integration.sim_sweep import (
    SweepRow,
    build_sweep_params,
    load_sweep,
    run_sweep,
    run_sweep_steps,
)

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SWEEPS_ROOT = Path(__file__).parent / 'sim_sweeps'
_RECOVERY_TOLERANCE_PX = 0.5


@functools.cache
def _cached_rows(sweep_name: str) -> tuple[SweepRow, ...]:
    """Load and run a sweep by name, once per test process."""
    return tuple(run_sweep(load_sweep(_SWEEPS_ROOT / f'{sweep_name}.yaml')))


def _rows(sweep_name: str) -> list[SweepRow]:
    """The rows of a sweep by name; several tests share one navigated run."""
    return list(_cached_rows(sweep_name))


def _sweep_values(sweep_name: str) -> list[float]:
    """The values a sweep's specification steps through, in order."""
    return list(load_sweep(_SWEEPS_ROOT / f'{sweep_name}.yaml').values)


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


def test_calibrated_noise_sweep_recovers_at_every_read_noise() -> None:
    """On the CALIB product the disc holds within tolerance across the whole sweep.

    The calibrated base renders through the full instrument chain, and its disc
    carries a charge comparable to the raw base's, yet the raw sweep's cliff at
    64 DN does not appear: every step recovers.
    """
    rows = _rows('noise_read_noise_calib')
    assert [row.value for row in rows] == _sweep_values('noise_read_noise_calib')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_calibrated_noise_sweep_raises_the_rendered_sky_noise_at_every_step() -> None:
    """Each read-noise step renders a noisier sky, so the sweep's recovery is not vacuous.

    The noise is read from paired horizontal differences in the frame's top-left
    32 x 32 px corner, which is sky in the base scene; the scene's seed is the same
    at every step.
    """
    steps = build_sweep_params(load_sweep(_SWEEPS_ROOT / 'noise_read_noise_calib.yaml'))
    sky_noise = []
    for _, sim_params in steps:
        img, _ = render_combined_model(sim_params)
        corner = np.asarray(img, dtype=np.float64)[:32, :32]
        sky_noise.append(float(np.std(np.diff(corner, axis=1)) / np.sqrt(2.0)))
    assert sky_noise == sorted(sky_noise)
    assert len(set(sky_noise)) == len(sky_noise)


def test_calibrated_exposure_sweep_starts_clean() -> None:
    """At the base exposure the CALIB product recovers the planted offset."""
    rows = _rows('noise_exposure_calib')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_calibrated_exposure_sweep_degrades_to_failure() -> None:
    """At the shortest exposure the disc sinks below the noise and navigation fails."""
    rows = _rows('noise_exposure_calib')
    assert rows[-1].status == 'failed'


def test_calibrated_exposure_sweep_fails_between_1_ms_and_0_3_ms() -> None:
    """Every step down to 1 ms recovers, and navigation fails at every step from 0.3 ms."""
    by_exposure = {row.value: row for row in _rows('noise_exposure_calib')}
    assert sorted(by_exposure) == sorted(_sweep_values('noise_exposure_calib'))
    for exposure in (0.1, 0.01, 0.001):
        recovered = by_exposure[exposure]
        assert recovered.status == 'success', exposure
        assert recovered.offset_error_px is not None, exposure
        assert recovered.offset_error_px < _RECOVERY_TOLERANCE_PX, exposure
    for exposure in (0.0003, 0.0001, 0.00003):
        assert by_exposure[exposure].status == 'failed', exposure


# The calibrated disc's steady sub-pixel error, and the bump it carries at offsets
# of 0.31, 0.382, and 0.45 px.  The steady bound sits above the measured maximum
# of 0.0062 px over 8 seeds x 23 steps, the raw twin's own level.  Each bump offset
# has its own bound above its measured error, which is the same on every seed:
# 0.190 and 0.118 px at 0.31 and 0.382 (16 seeds), 0.050 px at 0.45.  The bump is
# the correlation's sub-pixel refinement window, which spans half a pixel either
# side of the integer peak (#929).
_CALIB_DISC_STEADY_BIAS_PX = 0.008
_CALIB_DISC_BUMP_BOUNDS_PX = {0.31: 0.25, 0.382: 0.25, 0.45: 0.06}


def test_calibrated_disc_offset_sweep_recovers_every_offset() -> None:
    """The disc correlation recovers every sub-pixel offset on the CALIB product."""
    rows = _rows('disc_offset_fine_calib')
    assert [row.value for row in rows] == _sweep_values('disc_offset_fine_calib')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_calibrated_disc_offset_sweep_holds_its_bias() -> None:
    """Outside the 0.31-0.45 px bump the error stays below a hundredth of a pixel."""
    for row in _rows('disc_offset_fine_calib'):
        assert row.offset_error_px is not None
        bound = _CALIB_DISC_BUMP_BOUNDS_PX.get(row.value, _CALIB_DISC_STEADY_BIAS_PX)
        assert row.offset_error_px < bound, row.value


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


class _SmallestBodyNavigatedError(Exception):
    """The smallest range-sweep body navigated when it should have failed."""


# #941
@pytest.mark.xfail(
    strict=True,
    raises=_SmallestBodyNavigatedError,
    reason=(
        'With wrap-aware PSR the 4 px body navigates on a PSR computed from about five '
        'background cells, and nothing yet requires a minimum background size.'
    ),
)
def test_range_sweep_smallest_body_fails() -> None:
    """The smallest body is unnavigable.

    The expected failure is its own exception, so an assertion raised while the sweep
    runs is reported as a failure rather than absorbed by the marker.
    """
    rows = _rows('range_body_size')
    if rows[-1].status != 'failed':
        raise _SmallestBodyNavigatedError(
            f'the 4 px body ended {rows[-1].status} by {rows[-1].primary_technique}'
        )


def test_range_sweep_smallest_body_is_not_a_blob() -> None:
    """The 4 px body is below the blob floor, so BodyBlobNav never navigates it."""
    rows = _rows('range_body_size')
    assert rows[-1].primary_technique != 'BodyBlobNav'


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
# Measured 2026-11-10: at lumpiness 0.7 the shape-lock veto fires on 1 of the 13
# seeds (mesh seed 5).  Seed 2 also ends conflicted, but because its disc and limb
# disagree with each other; the veto plays no part.  On the noise draw before the
# detector drew shot noise per pixel, the veto fired on 3 seeds (0, 5, and 12),
# and seed 2 was conflicted then too, because its disc and limb disagreed.  One
# noise draw moved the rate that far, so this bound is one draw's measurement
# until the rate is measured over several noise seeds (#960).
_IRREGULARITY_MIN_VETOED = 1
# The status reason the shape-lock veto sets on the frames it declines.
_SHAPE_LOCK_REASON = 'body_shape_lock_suspect'
# A "confident wrong" result: success at this confidence or above, this far off.
_CONFIDENT = 0.9
_IRREGULARITY_CONFIDENT_WRONG_PX = 5.0
# Twelve of the 13 mesh seeds 0 to 12 dip at least this far below the clean
# step's confidence somewhere on their walks (the smallest of those dips is seed
# 9's, 0.37).  Seed 7 dips only 0.10, to 0.89 at lumpiness 0.7.  Measured
# 2026-11-10.
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

    The fused offset precision-weights the disc correlation (0.07 px off
    here) against the limb DT fit (which carries the documented ~0.1 px-class
    gradient-peak/model offset and is 0.05 px off here).  With the #210
    model-error floors both report comparable honest sigmas, so the fused
    value is their blend rather than whichever technique used to claim the
    tighter covariance; the bound reflects that blend, not a single
    technique's floor.
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 0.5


def test_irregularity_sweep_bias_grows() -> None:
    """Shape mismatch grows the recovered centroid bias as relief increases.

    This realization ends 5.2 px off.  Every one of mesh seeds 0 to 12 ends
    more than 2 px off (the least, seeds 6 and 11, both at 2.7 px), but the
    error never falls between steps on only 6 of them (seeds 2, 5, 7, 9, 10,
    and 12).  On this realization it holds at 2.89 px from lumpiness 0.4 to 0.6
    and falls by less than a millipixel at 0.6.
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[-1].offset_error_px is not None
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 1.0


def test_irregularity_sweep_confidence_drops() -> None:
    """Shape mismatch pulls the fused confidence well below the clean step's.

    Somewhere on the walk the confidence falls at least 0.3 below the clean
    (zero-relief) step's 0.99.  That holds on 12 of the 13 mesh seeds 0 to 12
    (seed 7 dips only 0.10), but it is asserted on this realization only, so the
    test is a tripwire on this walk rather than a check of the population.  The
    bound sits below the smallest of those 12 dips (seed 9's, 0.37); this walk
    dips 0.59, to 0.40 at lumpiness 0.7, so a failure here means the walk
    changed, and the population is what to measure next.

    Falling that far at every mismatched step holds on none of the seeds, and
    from lumpiness 0.2 on it holds only on seed 8, so neither is asserted: on
    many shapes a mismatched step still reports 0.99 (this realization does at
    lumpiness 0.1 and 0.2).
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
    """At lumpiness 0.7 the shape-lock veto declines at least 1 of 13 mesh shapes.

    Over mesh seeds 0 to 12 the disc correlation and the limb fit lock onto the
    mismatched shape at the extreme step.  When the pose-free blob disagrees with
    that lock, the veto reports the frame conflicted, with the status reason
    ``body_shape_lock_suspect``.  The test counts that reason, so a frame that is
    conflicted for another reason does not count.  On this noise draw the veto
    fires on seed 5 alone.  Seed 2 is conflicted too, but only because its disc
    and limb disagree with each other.  On the remaining shapes the blob moves
    with the lock or carries the result itself.  The bound is the rate measured on one
    noise draw, so a change that weakens the veto on these shapes fails here.
    """
    vetoed = [row for row in _irregularity_population() if row.status_reason == _SHAPE_LOCK_REASON]
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
    """The limb fit error grows as the predicted pose drifts.

    The check reads, in walk order, the steps where the pinned limb's result is
    not flagged spurious.  The second of them must be further off than the first,
    and the last must be more than 1 px further off than the first and more than
    2 px off.  On this noise draw the limb self-flags spurious at the 45 and 90
    deg steps (10 and 55 deg from the true pose), so the steps read are 35, 55,
    and 110 deg, at 0.17, 1.26, and 5.39 px off.  The test pins that list of
    steps first, so a change in which steps the limb self-flags at fails here.
    """
    # The filter keeps the steps where the pinned limb returned a result that is not
    # flagged spurious, whatever the fused status.  Which steps those are depends on
    # the noise draw, so the list pinned below is this draw's.  Measuring the growth
    # over several noise seeds is #960.
    reading = [
        (row.value, row.offset_error_px)
        for row in _rows('pose_disagreement')
        if row.offset_error_px is not None
    ]
    assert [value for value, _ in reading] == [35.0, 55.0, 110.0]
    errors = [error for _, error in reading]
    assert errors[1] > errors[0]
    assert errors[-1] > errors[0] + 1.0
    assert errors[-1] > 2.0


def test_pose_disagreement_limb_is_confidently_wrong() -> None:
    """At the largest disagreement the limb is several pixels off yet still 'success'.

    The pinned limb self-flags spurious at the 45 and 90 deg steps, and the frame
    fails there.  At the walk's other steps it does not self-flag, and at the last
    one it returns a confidently-wrong fix (5.4 px off at 110 deg, at 0.55 confidence
    against 0.62 at the true pose).  That is why the per-technique demote
    decision (test_sim_irregular_pose) cannot rely on the limb's own status and
    compares it against the pose-free blob instead.
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


# The scatter of the wrong_catalog expected_fail scene the sweep starts from.
_WHOLESALE_CATALOG_ERROR_PX = 8.0

# A success farther than this from the planted offset is a wrong answer.  Large
# scatter can still leave two stars near their catalog positions: at seed 8 the
# draws of stars 4 and 6 are 0.9 and 1.2 px at 6 px of scatter and 1.2 and
# 1.6 px at 8 px, the pair search pairs them, and the answer lands 0.66 and
# 0.83 px from the planted offset with a reported sigma near 0.9 px.  A
# coincidence between unrelated sources lands at the scatter's own scale, as
# the 4.9 px answer of #835 did.
_SCATTER_OFF_TARGET_PX = 2.0


def _off_target_successes(rows: list[SweepRow], *, min_scatter_px: float) -> list[SweepRow]:
    """Return the successes at ``min_scatter_px`` or more that miss the planted offset."""
    return [
        row
        for row in rows
        if row.value >= min_scatter_px
        and row.status == 'success'
        and (row.offset_error_px is None or row.offset_error_px > _SCATTER_OFF_TARGET_PX)
    ]


def test_star_catalog_scatter_wholesale_error_never_succeeds_off_target() -> None:
    """In the wholesale-catalog-error regime no seed reports a wrong success.

    Only the safety envelope is asserted: clean recovery at scatter 0 (above)
    and no success off the planted offset once the scatter reaches the
    wholesale-error regime the wrong_catalog expected_fail scene pins at 8 px.
    The intermediate region is deliberately NOT asserted -- it is
    characterization raw material, recorded below for the confidence
    recalibration work.
    """
    rows = _rows('star_catalog_scatter')
    assert _off_target_successes(rows, min_scatter_px=_WHOLESALE_CATALOG_ERROR_PX) == []


def test_star_catalog_scatter_six_px_never_succeeds_off_target() -> None:
    """At 6 px of scatter, short of the wholesale regime, no seed reports a wrong success."""
    rows = _rows('star_catalog_scatter')
    six_px = [row for row in rows if row.value == 6.0]
    assert six_px != []
    assert _off_target_successes(six_px, min_scatter_px=6.0) == []


# At 4 px of scatter, seed 9 pairs two stars whose draws agree and reports a
# medium-tier success 3.65 px from the planted offset (#835).
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason='a 4 px scatter seed succeeds 3.65 px off (#835)'
)
def test_star_catalog_scatter_four_px_never_succeeds_off_target() -> None:
    """At 4 px of scatter no seed reports a success off the planted offset."""
    rows = _rows('star_catalog_scatter')
    four_px = [row for row in rows if row.value == 4.0]
    assert four_px != []
    assert _off_target_successes(four_px, min_scatter_px=4.0) == []


def test_star_catalog_scatter_reaches_the_wholesale_error_regime() -> None:
    """The scatter sweep has steps at 8 px or more for the never-succeeds check to read."""
    rows = _rows('star_catalog_scatter')
    assert [row.value for row in rows if row.value >= _WHOLESALE_CATALOG_ERROR_PX] != []


# Measured star_catalog_scatter curve (2026-11-02, seeds 7-9), recorded for the
# astrometric-residual characterization: this is observed behavior, not asserted
# correctness (#291).  The navigator does not yet self-flag the growing
# astrometric residual:
#   scatter 0.0-1.0 px: all seeds success, errors 0.04-0.70 px, medium/high;
#   scatter 1.5 px:     seed 7 conflicted, seeds 8-9 success at 1.08/0.52 px;
#   scatter 2.0 px:     seed 7 fails, seed 9 conflicted, seed 8 success 1.39 px;
#   scatter 3.0 px:     2 of 3 seeds fail; seed 9 success at 2.0 px (medium);
#   scatter 4.0 px:     2 of 3 seeds fail; seed 9 reports a medium-tier (0.81)
#                       success 3.65 px from the planted offset, a pair of
#                       stars whose draws agree (#835, pinned below);
#   scatter 6.0-8.0 px: seeds 7 and 9 fail; seed 8 succeeds 0.66 and 0.83 px
#                       from the planted offset (see _SCATTER_OFF_TARGET_PX).


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
    """Unmodeled Moffat wing energy does not break the DT limb fit.

    Unlike the core-sigma axis (whose blur washes the limb gradient out to
    outright failure), moving up to 60% of the kernel energy into the
    isotropic wing leaves the core sharp: every step still recovers within
    tolerance.  Any bias the wings add is smaller than the sweep's scatter
    across noise seeds, several hundredths of a pixel, so no bound on it is
    asserted.
    """
    rows = _rows('psf_limb_wings')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX, row.value


def test_psf_wings_change_the_rendered_image() -> None:
    """The largest wing weight renders a different frame from the zero-wing floor.

    The two steps keep the sweep's scene and seed, with the noise switched off so that
    only the kernel can tell them apart.  If the optics stage ignored the wing weight,
    every step would render the floor's frame and the recovery checks above would pass
    without testing any wing energy.
    """
    steps = build_sweep_params(load_sweep(_SWEEPS_ROOT / 'psf_limb_wings.yaml'))
    frames = []
    for _, sim_params in (steps[0], steps[-1]):
        quiet = copy.deepcopy(sim_params)
        quiet['noise'] = {'poisson': False, 'read_noise_dn': 0.0}
        img, _ = render_combined_model(quiet)
        frames.append(np.asarray(img, dtype=np.float64))
    assert not np.array_equal(frames[0], frames[1])


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


def test_atmosphere_haze_albedo_floor_recovers() -> None:
    """An albedo-0 haze adds no light, so the bare limb recovers the offset."""
    rows = _rows('atmosphere_haze_albedo')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_atmosphere_haze_albedo_biases_limb() -> None:
    """A brighter haze of the same column lifts the limb, biasing the recovery."""
    rows = _rows('atmosphere_haze_albedo')
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 0.1


# The smallest rise in recovered-offset error between successive albedo steps up
# to the scene default.  The steps measure 0.06, 0.12, and 0.21 px.
_ALBEDO_STEP_RISE_PX = 0.05


def test_atmosphere_haze_albedo_bias_grows_up_to_the_default_albedo() -> None:
    """Each brighter haze up to the default albedo lifts the limb farther.

    Past the default the error levels off (0.42 px at 0.6, 0.40 px at 1.0), so
    the brighter rows are left unordered.
    """
    rows = [row for row in _rows('atmosphere_haze_albedo') if row.value <= DEFAULT_HAZE_ALBEDO]
    errors = [row.offset_error_px for row in rows]
    assert [row.value for row in rows] == [0.0, 0.15, 0.3, 0.6]
    rises = [
        later - earlier
        for earlier, later in itertools.pairwise(errors)
        if earlier is not None and later is not None
    ]
    assert len(rises) == len(rows) - 1
    assert min(rises) > _ALBEDO_STEP_RISE_PX


# The per-technique dense and wide offset sweeps (``*_offset_fine`` /
# ``*_offset_wide``) are characterization runs, not assertions: they are executed
# by ``sim_sweep_runner`` to produce the report's figures, and the specific defect
# they expose (the disc gradient-NCC sub-pixel bias) is guarded by the fast
# ``test_sim_regression`` case rather than by re-running the full sweep here.
