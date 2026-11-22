"""Single-variable sweep invariants.

Each sweep drives one catalog scene by varying a single parameter and navigates
every step.  These tests assert how a navigation diagnostic *responds* to the
controlled change -- the verification layer a calibrated confidence formula
relies on.  Assertions are trends and bounds (a technique transition, a
degradation to failure, recovery within tolerance), not exact values, because the
underlying navigation carries sub-millipixel cross-process jitter.

A check that holds on every noise draw stays a single-draw tripwire on the scene's own
draw, and its docstring records the spread measured over other draws.  A check whose
outcome moves with the noise draw is asserted on a population of draws instead: the
scene's own and four more, each drawn with the scene key ``noise_seed``, which redraws
only the detector's per-frame noise.  Three sweeps are read that way: the calibrated
exposure sweep, the pose-disagreement sweep, and the differential-smear sweep.  Their
own draw is the run the tripwires on the same sweeps read, so they stay in this module,
and the four extra draws of the three add about 26 s to the module's 118 s on one worker
(measured 2026-11-18 on a loaded machine).  The population of irregular shapes shares no
run with these sweeps and lives in ``test_sim_irregularity_population``.

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

from spindoctor.sim.render.bodies.atmosphere import DEFAULT_HAZE_ALBEDO
from spindoctor.sim.render.driver import render_scene
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


# The noise draws a population check reads: the scene's own draw (None) and four more.
# Each population bound below was measured on 25 draws, the scene's own and noise seeds
# 101, 202, ..., 2424, and holds for every choice of 5 of them.  Each tripwire's spread
# was measured on 9 draws, the scene's own and noise seeds 101 to 808 (#960).
_NOISE_DRAWS: tuple[int | None, ...] = (None, 101, 202, 303, 404)


@functools.cache
def _cached_noise_rows(sweep_name: str, noise_seed: int) -> tuple[SweepRow, ...]:
    """Run a sweep by name with every step's detector noise drawn from ``noise_seed``."""
    spec = load_sweep(_SWEEPS_ROOT / f'{sweep_name}.yaml')
    steps = [
        (value, {**params, 'noise_seed': noise_seed}) for value, params in build_sweep_params(spec)
    ]
    return tuple(run_sweep_steps(steps, spec.technique))


def _population(sweep_name: str) -> list[list[SweepRow]]:
    """The rows of a sweep on each noise draw of ``_NOISE_DRAWS``, in that order."""
    return [
        _rows(sweep_name) if seed is None else list(_cached_noise_rows(sweep_name, seed))
        for seed in _NOISE_DRAWS
    ]


def test_noise_sweep_starts_clean() -> None:
    """At the lowest read noise the scene recovers the planted offset.

    A single-draw tripwire: on each of 9 noise draws the error is below 1e-15 px.
    """
    rows = _rows('noise_read_noise')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_noise_sweep_degrades_to_failure() -> None:
    """At the highest read noise navigation fails -- the navigability cliff.

    A single-draw tripwire: the step fails on each of 9 noise draws.
    """
    rows = _rows('noise_read_noise')
    assert rows[-1].status == 'failed'


def test_noise_sweep_low_noise_all_succeed() -> None:
    """Every below-cliff step recovers within tolerance.

    A single-draw tripwire: over 9 noise draws the worst error is 0 to 0.018 px.
    """
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
    64 DN does not appear: every step recovers.  A single-draw tripwire: over 9 noise
    draws the worst error is 0 to 0.056 px.
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
    at every step.  A single-draw tripwire: the sky noise rises at every step on each
    of 9 noise draws.
    """
    steps = build_sweep_params(load_sweep(_SWEEPS_ROOT / 'noise_read_noise_calib.yaml'))
    sky_noise = []
    for _, sim_params in steps:
        img, _ = render_scene(sim_params)
        corner = np.asarray(img, dtype=np.float64)[:32, :32]
        sky_noise.append(float(np.std(np.diff(corner, axis=1)) / np.sqrt(2.0)))
    assert sky_noise == sorted(sky_noise)
    assert len(set(sky_noise)) == len(sky_noise)


def test_calibrated_exposure_sweep_starts_clean() -> None:
    """At the base exposure the CALIB product recovers the planted offset.

    A single-draw tripwire: on each of 25 noise draws the error is below 1e-15 px.
    """
    rows = _rows('noise_exposure_calib')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


# Measured on 25 noise draws.  At 30 us the frame fails on 24; on noise seed 707 it is a
# 0.63-confidence success 46.6 px off.
_EXPOSURE_SHORTEST_MIN_FAILED = 4


def _recovers(row: SweepRow) -> bool:
    """True when a step succeeds within the recovery tolerance."""
    return (
        row.status == 'success'
        and row.offset_error_px is not None
        and row.offset_error_px < _RECOVERY_TOLERANCE_PX
    )


def test_calibrated_exposure_sweep_degrades_to_failure() -> None:
    """At the shortest exposure the disc sinks below the noise and navigation fails.

    Asserted on the noise population: at 30 us at least 4 of the 5 draws fail.
    """
    failed = [rows[-1].status == 'failed' for rows in _population('noise_exposure_calib')]
    assert sum(failed) >= _EXPOSURE_SHORTEST_MIN_FAILED


# The exposure cliff over 25 noise draws.  At 0.1 s and 10 ms every draw recovers
# within 0.01 px, and at 0.1 ms every draw fails.  At 1 ms 23 draws recover; the
# other two succeed 1.11 and 0.71 px off at 0.40 confidence.  At 0.3 ms 22 draws
# fail; noise seeds 606 and 707 recover, and noise seed 2323 succeeds 0.60 px off at
# 0.74 confidence.  The 1 ms count below holds for every choice of 5 of the 25
# draws.  The 0.3 ms count holds for every choice except those that take all three
# of noise seeds 606, 707, and 2323, where only 2 of the 5 fail.
_EXPOSURE_ALWAYS_RECOVERS = (0.1, 0.01)
_EXPOSURE_ALWAYS_FAILS = 0.0001
_EXPOSURE_1_MS_MIN_RECOVERED = 2
_EXPOSURE_0_3_MS_MIN_FAILED = 3


def test_calibrated_exposure_sweep_cliff_lies_between_10_ms_and_0_1_ms() -> None:
    """Every draw recovers down to 10 ms and fails at 0.1 ms; 1 ms and 0.3 ms straddle it.

    Asserted on the noise population.  At 1 ms at least 2 of the 5 draws recover,
    and at 0.3 ms at least 3 of them fail.
    """
    population = [{row.value: row for row in rows} for rows in _population('noise_exposure_calib')]
    for by_exposure in population:
        assert sorted(by_exposure) == sorted(_sweep_values('noise_exposure_calib'))
        for exposure in _EXPOSURE_ALWAYS_RECOVERS:
            assert _recovers(by_exposure[exposure]), exposure
        assert by_exposure[_EXPOSURE_ALWAYS_FAILS].status == 'failed'
    recovered_1_ms = sum(_recovers(by_exposure[0.001]) for by_exposure in population)
    failed_0_3_ms = sum(by_exposure[0.0003].status == 'failed' for by_exposure in population)
    assert recovered_1_ms >= _EXPOSURE_1_MS_MIN_RECOVERED
    assert failed_0_3_ms >= _EXPOSURE_0_3_MS_MIN_FAILED


# The calibrated disc's steady sub-pixel error, and the bump it carries at offsets
# of 0.31, 0.382, and 0.45 px.  The steady bound sits above the measured maximum
# of 0.0062 px over 8 seeds x 23 steps, the raw twin's own level, and of 0.0070 px
# over 9 noise draws.  Each bump offset
# has its own bound above its measured error, which is the same on every seed:
# 0.190 and 0.118 px at 0.31 and 0.382 (16 seeds), 0.050 px at 0.45.  The bump is
# the correlation's sub-pixel refinement window, which spans half a pixel either
# side of the integer peak (#929).
_CALIB_DISC_STEADY_BIAS_PX = 0.008
_CALIB_DISC_BUMP_BOUNDS_PX = {0.31: 0.25, 0.382: 0.25, 0.45: 0.06}


def test_calibrated_disc_offset_sweep_recovers_every_offset() -> None:
    """The disc correlation recovers every sub-pixel offset on the CALIB product.

    A single-draw tripwire: on each of 9 noise draws the worst error is 0.190 px.
    """
    rows = _rows('disc_offset_fine_calib')
    assert [row.value for row in rows] == _sweep_values('disc_offset_fine_calib')
    for row in rows:
        assert row.status == 'success'
        assert row.offset_error_px is not None
        assert row.offset_error_px < _RECOVERY_TOLERANCE_PX


def test_calibrated_disc_offset_sweep_holds_its_bias() -> None:
    """Outside the 0.31-0.45 px bump the error stays below a hundredth of a pixel.

    A single-draw tripwire: over 9 noise draws the steady error is 0.0036 to 0.0070 px,
    and each bump's error is the same on every draw.
    """
    for row in _rows('disc_offset_fine_calib'):
        assert row.offset_error_px is not None
        bound = _CALIB_DISC_BUMP_BOUNDS_PX.get(row.value, _CALIB_DISC_STEADY_BIAS_PX)
        assert row.offset_error_px < bound, row.value


def test_phase_sweep_navigates_every_phase() -> None:
    """The resolved body navigates to success across the full phase range.

    A single-draw tripwire: every step succeeds on each of 9 noise draws.
    """
    rows = _rows('phase_regular_body')
    for row in rows:
        assert row.status == 'success'


def test_phase_sweep_recovers_every_phase() -> None:
    """The recovered offset stays within tolerance across the full phase range.

    A single-draw tripwire: over 9 noise draws the worst error is 0 to 0.008 px.
    """
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
    assertion accepts either rather than pinning the ordering.  A single-draw
    tripwire: the disc correlation ranks primary on each of 9 noise draws.
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
    runs is reported as a failure rather than absorbed by the marker.  A single-draw
    tripwire: the 4 px body ends a success on each of 9 noise draws.
    """
    rows = _rows('range_body_size')
    if rows[-1].status != 'failed':
        raise _SmallestBodyNavigatedError(
            f'the 4 px body ended {rows[-1].status} by {rows[-1].primary_technique}'
        )


def test_range_sweep_smallest_body_is_not_a_blob() -> None:
    """The 4 px body is below the blob floor, so BodyBlobNav never navigates it.

    A single-draw tripwire: the disc correlation carries it on each of 9 noise draws.
    """
    rows = _rows('range_body_size')
    assert rows[-1].primary_technique != 'BodyBlobNav'


def test_range_sweep_reaches_blob_regime() -> None:
    """A small-but-navigable body falls to the orientation-free BodyBlobNav.

    A single-draw tripwire: the blob ranks primary at two or three steps on each of 9
    noise draws.
    """
    rows = _rows('range_body_size')
    primaries = [row.primary_technique for row in rows]
    assert 'BodyBlobNav' in primaries


def test_range_sweep_transitions_technique() -> None:
    """The primary technique is not constant -- the range ladder transitions.

    A single-draw tripwire: the ladder transitions on each of 9 noise draws.
    """
    rows = _rows('range_body_size')
    distinct = {row.primary_technique for row in rows if row.primary_technique is not None}
    assert len(distinct) >= 2


_ROLL_TOLERANCE_DEG = 0.3


def test_star_rotation_sweep_recovers_roll() -> None:
    """The star field recovers every planted roll in the working window.

    The recovered roll is read from whichever technique reports it (the two-star
    path at +/-1 deg, the field matcher at the larger rolls), so the assertion is
    on the roll error, not the fused status.  A single-draw tripwire: over 9 noise
    draws the worst roll error is 0.030 to 0.033 deg.
    """
    rows = _rows('star_rotation')
    for row in rows:
        assert row.rotation_error_deg is not None
        assert row.rotation_error_deg < _ROLL_TOLERANCE_DEG


# The irregularity walk navigates one mesh realization, the base scene's
# mesh_seed (4, the median shape mismatch of mesh seeds 0 to 12; see the scene).
# Its single-seed tests are tripwires on this realization.  The properties that
# hold across shapes are asserted on the population of mesh seeds 0 to 12 at the
# walk's extreme step, lumpiness 0.7, in test_sim_irregularity_population.
# A "confident" result: success at this confidence or above.
_CONFIDENT = 0.9
# Twelve of the 13 mesh seeds 0 to 12 dip at least this far below the clean
# step's confidence somewhere on their walks (the smallest of those dips is seed
# 10's, 0.31).  Seed 7 dips only 0.28, to 0.71 at lumpiness 0.5.  Measured
# 2026-11-22.
_IRREGULARITY_MIN_CONFIDENCE_DROP = 0.3


# The per-technique model-error floors the docstring refers to are #210.
def test_irregularity_sweep_starts_matched() -> None:
    """At zero relief the fused recovery stays sub-half-pixel.

    The fused offset precision-weights the disc correlation (0.07 px off
    here) against the limb DT fit (which carries the documented gradient-peak/model
    offset and, fitting only the lit limb, is 0.39 px off here).  With the per-technique
    model-error floors, both report comparable sigmas, so the fused
    value is their blend rather than whichever technique used to claim the
    tighter covariance; the bound reflects that blend, not a single
    technique's floor.  A single-draw tripwire: over 9 noise draws the error is
    0.045 to 0.108 px.
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 0.5


def test_irregularity_sweep_bias_grows() -> None:
    """Shape mismatch grows the recovered centroid bias as relief increases.

    On the scene's own draw this realization ends 8.3 px off, where the shape-lock
    veto declines it.  Every one of mesh seeds 0 to 12 ends more than 2 px off (the
    least, seed 11, at 2.7 px), and the error never falls between steps on 7 of them
    (seeds 1, 2, 4, 6, 7, 9, and 10), this realization among them.  A single-draw
    tripwire: over 9 noise draws the last step is 5.262 to 5.265 px off on the 7
    draws where the blob carries it, and 8.27 to 8.29 px off on the 2 where the veto
    declines it.
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
    (seed 7 dips only 0.28), but it is asserted on this realization only, so the
    test is a tripwire on this walk rather than a check of the population.  The
    bound sits below the smallest of those 12 dips (seed 10's, 0.31); on the scene's
    own draw this walk dips 0.84, to 0.15 at lumpiness 0.7, where the shape-lock veto
    declines the step, so a failure here means the walk changed, and the population
    is what to measure next.  Over 9 noise draws the dip is 0.84 on the 2 draws where
    the veto declines the extreme step and 0.59 on the other 7.

    Falling that far at every mismatched step holds on none of the seeds, nor
    does it from lumpiness 0.2 on, so neither is asserted: on many shapes a
    mismatched step still reports 0.99 (this realization does from lumpiness 0.1
    to 0.6).
    """
    rows = _rows('irregularity_shape_mismatch')
    lowest = min(row.confidence for row in rows[1:])
    assert lowest <= rows[0].confidence - _IRREGULARITY_MIN_CONFIDENCE_DROP


def test_irregularity_sweep_extreme_step_is_not_a_confident_success() -> None:
    """This realization's extreme step is not reported as a confident success.

    At lumpiness 0.7 this realization's disc and limb lock onto the mismatched
    shape.  On the scene's own draw the shape-lock veto declines the step as
    conflicted, at 0.15 confidence.  The veto's rate over shapes is asserted on the
    population in ``test_sim_irregularity_population``, and so is the wrong lock
    this tripwire guards on this one shape.  Over 9 noise draws the step is declined
    on 2 and is a 0.40-confidence success 5.3 px off, carried by the pose-free blob,
    on the other 7.
    """
    rows = _rows('irregularity_shape_mismatch')
    extreme = rows[-1]
    confident_success = extreme.status == 'success' and extreme.confidence >= _CONFIDENT
    assert not confident_success


def test_irregularity_sweep_clean_step_still_succeeds() -> None:
    """The zero-relief step navigates cleanly: the veto does not blanket the sweep.

    A single-draw tripwire: the step succeeds on each of 9 noise draws.
    """
    rows = _rows('irregularity_shape_mismatch')
    assert rows[0].status == 'success'


def test_pose_disagreement_starts_clean() -> None:
    """With the predicted pose agreeing, the limb recovers the planted offset.

    A single-draw tripwire: over 25 noise draws the error is 0.22 to 0.26 px.
    """
    rows = _rows('pose_disagreement')
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < 1.0


# Measured on 25 noise draws.  The pinned limb never flags the 35, 45, and 55 deg
# steps, which read 0.22 to 0.26, 0.56 to 0.60, and 1.42 to 1.45 px off.  It flags
# the 90 deg step on 21 draws and the 110 deg step on 12; where it reads them, they
# are 0.95 to 0.96 and 0.78 to 0.81 px off, closer than the 55 deg step.
_POSE_ALWAYS_READ = (35.0, 55.0)


def _limb_reading(rows: list[SweepRow]) -> dict[float, float]:
    """The pinned limb's error at each step where its result is not flagged spurious.

    The pinned limb's error is None at a step where it returned no result or one
    flagged spurious, whatever the fused status.
    """
    return {row.value: row.offset_error_px for row in rows if row.offset_error_px is not None}


def test_pose_disagreement_limb_error_grows() -> None:
    """The limb fit error grows as the predicted pose drifts 20 deg from the true one.

    Asserted on the noise population: on every draw the limb reads the 35 and 55 deg
    steps (0 and 20 deg from the true pose), and the 55 deg step is further off.
    Farther from the true pose the error does not keep growing (see the measurement
    above), so no bound is asserted there.
    """
    for reading in (_limb_reading(rows) for rows in _population('pose_disagreement')):
        assert set(_POSE_ALWAYS_READ) <= set(reading)
        assert reading[55.0] > reading[35.0]


def test_artifact_sweep_starts_clean() -> None:
    """At zero missing-line expected_events the frame recovers the planted offset.

    A single-draw tripwire: on each of 9 noise draws the error is below 1e-15 px.
    """
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
    recovered offset near the cliff jitters across processes.  A single-draw
    tripwire: the noise draws keep the lost lines where they are, and the confidence
    falls from 0.84 to 0 on each of 9 of them.
    """
    rows = _rows('artifact_missing_lines')
    degraded_min = min(row.confidence for row in rows[1:])
    assert degraded_min < rows[0].confidence


def test_artifact_sweep_degrades_to_failure() -> None:
    """At the highest expected count half the frame is gone and navigation fails.

    A single-draw tripwire: the step fails on each of 9 noise draws.
    """
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
    rows = _rows('star_confounder_density')
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
    """At the navigator-matched PSF (sigma 0.54) the limb recovers the offset.

    A single-draw tripwire: over 9 noise draws the error is 0.181 to 0.201 px.
    """
    rows = _rows('psf_limb_mismatch')
    assert rows[0].value == 0.54
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_psf_mismatch_grows_the_limb_error() -> None:
    """A broad rendered PSF grows the limb error.

    The lit limb survives every blur the sweep walks: at a PSF sigma of 4 px it is
    still a success, 1.24 to 1.46 px off at 0.56 to 0.58 confidence over 9 noise
    draws.  A single-draw tripwire: over those draws the largest error rises 1.04 to
    1.26 px above the floor's.
    """
    rows = _rows('psf_limb_mismatch')
    assert rows[0].offset_error_px is not None
    mismatched = [r.offset_error_px for r in rows[1:] if r.offset_error_px is not None]
    assert max(mismatched) > rows[0].offset_error_px + 0.2


def test_psf_wings_floor_recovers() -> None:
    """At zero wing energy the kernel is the navigator-matched pure Gaussian.

    A single-draw tripwire: over 9 noise draws the error is 0.181 to 0.201 px.
    """
    rows = _rows('psf_limb_wings')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_psf_wings_bias_stays_gentle() -> None:
    """Unmodeled Moffat wing energy does not break the DT limb fit.

    Unlike the core-sigma axis (whose blur grows the limb error past a pixel),
    moving up to 60% of the kernel energy into the isotropic wing leaves the core
    sharp: every step still recovers within tolerance.  Any bias the wings add is
    smaller than the sweep's scatter across noise seeds, several hundredths of a
    pixel, so no bound on it is asserted.  A single-draw tripwire: over 9 noise draws
    the worst error is 0.241 to 0.276 px.
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
        img, _ = render_scene(quiet)
        frames.append(np.asarray(img, dtype=np.float64))
    assert not np.array_equal(frames[0], frames[1])


def test_photometric_mismatch_floor_recovers() -> None:
    """Minnaert k=1 is Lambert, so the floor recovers the planted offset.

    A single-draw tripwire: the error is 0.000 px on each of 9 noise draws.
    """
    rows = _rows('photometric_minnaert_mismatch')
    assert rows[0].value == 1.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_photometric_mismatch_biases_recovery() -> None:
    """Walking off Lambert grows the disc-correlation centroid bias.

    A single-draw tripwire: the largest error is 1.035 px above the floor's on each of
    9 noise draws.
    """
    rows = _rows('photometric_minnaert_mismatch')
    assert rows[0].offset_error_px is not None
    mismatched = [r.offset_error_px for r in rows[1:] if r.offset_error_px is not None]
    assert max(mismatched) > rows[0].offset_error_px + 0.2


def test_spk_parallax_floor_recovers() -> None:
    """With no parallax shift the body sits at its catalog position.

    A single-draw tripwire: on each of 9 noise draws the error is below 1e-15 px.
    """
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
    magnitude.  A single-draw tripwire: it is 6.000 px on each of 9 noise draws.
    """
    rows = _rows('spk_parallax_error')
    assert rows[-1].value == 6.0
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 4.0


def test_differential_smear_floor_recovers() -> None:
    """With no star trail the sharp field recovers the planted offset.

    A single-draw tripwire: over 25 noise draws the error is 0.004 to 0.008 px.
    """
    rows = _rows('differential_smear')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


# Measured on 25 noise draws: the 6 px trail ends 0.016 to 0.162 px further off than
# the floor, in three clusters (5 draws at 0.016 to 0.035 px, 7 near 0.08 px, and 13
# near 0.16 px), so how far the centroids drift depends on the noise draw.
_SMEAR_MIN_DRIFT_PX = 0.01


def test_differential_smear_degrades_at_long_trail() -> None:
    """A long star trail drifts the centroids off their catalog positions.

    Asserted on the noise population: on every draw the 6 px trail ends more than
    0.01 px further off than the floor.
    """
    for rows in _population('differential_smear'):
        assert rows[0].offset_error_px is not None
        assert rows[-1].offset_error_px is not None
        assert rows[-1].offset_error_px > rows[0].offset_error_px + _SMEAR_MIN_DRIFT_PX


def test_ring_orbit_error_floor_recovers() -> None:
    """With no radial orbit error the ring edge recovers the planted offset.

    A single-draw tripwire: over 9 noise draws the error is 0.006 to 0.009 px.
    """
    rows = _rows('ring_orbit_error')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_ring_orbit_error_absorbs_into_offset() -> None:
    """A radial ring-orbit error is absorbed into the recovered offset.

    A single-draw tripwire: over 9 noise draws the last step is 3.37 to 4.85 px off.
    """
    rows = _rows('ring_orbit_error')
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > 2.0


def test_atmosphere_haze_floor_recovers() -> None:
    """With no haze the hard limb recovers the planted offset.

    A single-draw tripwire: over 9 noise draws the error is 0.033 to 0.035 px.
    """
    rows = _rows('atmosphere_haze')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_atmosphere_haze_biases_limb() -> None:
    """A thickening haze lifts and blurs the limb, biasing the recovery.

    A single-draw tripwire: over 9 noise draws the last step is 0.313 to 0.318 px
    further off than the floor.
    """
    rows = _rows('atmosphere_haze')
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 0.1


def test_atmosphere_haze_albedo_floor_recovers() -> None:
    """An albedo-0 haze adds no light, so the bare limb recovers the offset.

    A single-draw tripwire: over 25 noise draws the error is 0.033 to 0.035 px.
    """
    rows = _rows('atmosphere_haze_albedo')
    assert rows[0].value == 0.0
    assert rows[0].status == 'success'
    assert rows[0].offset_error_px is not None
    assert rows[0].offset_error_px < _RECOVERY_TOLERANCE_PX


def test_atmosphere_haze_albedo_biases_limb() -> None:
    """A brighter haze of the same column lifts the limb, biasing the recovery.

    A single-draw tripwire: over 9 noise draws the last step is 0.268 to 0.271 px
    further off than the floor.
    """
    rows = _rows('atmosphere_haze_albedo')
    assert rows[0].offset_error_px is not None
    assert rows[-1].offset_error_px is not None
    assert rows[-1].offset_error_px > rows[0].offset_error_px + 0.1


# The smallest rise in recovered-offset error between successive albedo steps up
# to the scene default.  Over 25 noise draws the three rises measure 0.044 to
# 0.052 px, 0.107 to 0.116 px, and 0.160 to 0.168 px, so every draw clears the
# bound with at least 0.024 px to spare.  (The first rise measured 0.077 to 0.123 px
# while the haze and the night side were capped and floored, under a bound of 0.05.)
_ALBEDO_STEP_RISE_PX = 0.02


def test_atmosphere_haze_albedo_bias_grows_up_to_the_default_albedo() -> None:
    """Each brighter haze up to the default albedo lifts the limb farther.

    Past the default the error falls back (0.36 px at 0.6, 0.30 px at 1.0), so
    the brighter rows are left unordered.  A single-draw tripwire: on the scene's
    own draw the rises are 0.050, 0.112, and 0.166 px, and the smallest rise is
    above the bound on each of 25 noise draws.
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
