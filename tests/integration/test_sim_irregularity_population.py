"""The shape-lock veto over a population of irregular shapes and noise draws.

The irregularity sweep (``sim_sweeps/irregularity_shape_mismatch.yaml``) walks one mesh
realization, and ``test_sim_sweeps`` holds that walk as a tripwire.  The properties that
hold across shapes are asserted here, on the walk's extreme step, lumpiness 0.7, for mesh
seeds 0 to 12 on each of 5 noise draws: the scene's own draw and four more, each drawn
with the scene key ``noise_seed``, which redraws only the detector's per-frame noise.
Which shapes the veto catches moves with the noise, so a single draw cannot tell a
weaker veto from an unlucky one.

The population is 65 navigations (13 mesh seeds on 5 noise draws).  They share nothing
with the sweeps in ``test_sim_sweeps``, so the population lives in its own module, which
``--dist=loadfile`` can run on another worker.  The module takes about 60 s on one
worker (measured 2026-11-18 on a loaded machine).
"""

import copy
import functools
from pathlib import Path

import pytest

from tests.integration.sim_sweep import SweepRow, build_sweep_params, load_sweep, run_sweep_steps

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SWEEP_PATH = Path(__file__).parent / 'sim_sweeps' / 'irregularity_shape_mismatch.yaml'

# The noise draws the population reads: the scene's own draw (None) and four more.  Each
# bound below was measured on 25 draws, the scene's own and noise seeds 101, 202, ...,
# 2424, and holds for every choice of 5 of them (#960).
_NOISE_DRAWS: tuple[int | None, ...] = (None, 101, 202, 303, 404)
_POPULATION_MESH_SEEDS = range(13)
_EXTREME_LUMPINESS = 0.7
# Measured on 25 noise draws (2026-11-22): at lumpiness 0.7 the shape-lock veto fires
# on 1 to 8 of the 13 mesh seeds per draw, 116 of the 325 frames in all, on mesh seeds
# 1, 4, 5, 6, 8, 9, 10, 11, and 12.  Over any 5 of the 25 draws it fires 13 to 32
# times (median 23); over the 5 draws the test reads, 19 times.  Seed 2 also ends
# conflicted on about half the draws, but because its disc and limb disagree with
# each other; the veto plays no part.  A veto that stopped firing on these shapes
# would count 0.  The floor is the smallest count over any 5 of the 25 draws.
_MIN_VETOED = 13
# The status reason the shape-lock veto sets on the frames it declines.
_SHAPE_LOCK_REASON = 'body_shape_lock_suspect'
# A "confident wrong" result: success at this confidence or above, this far off.
_CONFIDENT = 0.9
_CONFIDENT_WRONG_PX = 5.0


@functools.cache
def _draw(noise_seed: int | None) -> tuple[SweepRow, ...]:
    """Navigate the irregularity walk's extreme step once per mesh seed 0 to 12.

    Each row's ``value`` is its mesh seed.  ``noise_seed`` redraws the detector
    noise, and None keeps the scene's own draw.  Cached, so the population's tests
    share one set of navigations.
    """
    spec = load_sweep(_SWEEP_PATH)
    extreme = [params for value, params in build_sweep_params(spec) if value == _EXTREME_LUMPINESS]
    assert len(extreme) == 1
    steps = []
    for mesh_seed in _POPULATION_MESH_SEEDS:
        params = copy.deepcopy(extreme[0])
        params['bodies'][0]['mesh_seed'] = mesh_seed
        if noise_seed is not None:
            params['noise_seed'] = noise_seed
        steps.append((float(mesh_seed), params))
    return tuple(run_sweep_steps(steps, spec.technique))


def _population() -> list[SweepRow]:
    """The extreme step's rows for every mesh seed on every noise draw of _NOISE_DRAWS."""
    return [row for seed in _NOISE_DRAWS for row in _draw(seed)]


def test_irregularity_population_veto_rate_at_extreme_mismatch() -> None:
    """At lumpiness 0.7 the shape-lock veto declines at least 13 of the 65 frames.

    Over mesh seeds 0 to 12 the disc correlation and the limb fit lock onto the
    mismatched shape at the extreme step.  When the pose-free blob disagrees with
    that lock, the veto reports the frame conflicted, with the status reason
    ``body_shape_lock_suspect``.  The test counts that reason, so a frame that is
    conflicted for another reason does not count.  The population is the 13 shapes
    on each of 5 noise draws, because which shapes the veto catches moves with the
    noise: it fires on 1 to 8 shapes per draw.  On the remaining shapes the blob
    moves with the lock or carries the result itself.  A change that weakens the
    veto on these shapes fails here.
    """
    vetoed = [row for row in _population() if row.status_reason == _SHAPE_LOCK_REASON]
    assert len(vetoed) >= _MIN_VETOED


# The frames of the population measurement below that end a confident wrong
# success: (mesh seed, noise seed).
_CONFIDENT_WRONG_FRAMES: tuple[tuple[int, int], ...] = ((9, 808), (9, 2424))


@functools.cache
def _frame(mesh_seed: int, noise_seed: int) -> SweepRow:
    """Navigate the extreme step for one mesh seed on one noise draw."""
    spec = load_sweep(_SWEEP_PATH)
    extreme = [params for value, params in build_sweep_params(spec) if value == _EXTREME_LUMPINESS]
    params = copy.deepcopy(extreme[0])
    params['bodies'][0]['mesh_seed'] = mesh_seed
    params['noise_seed'] = noise_seed
    return run_sweep_steps([(float(mesh_seed), params)], spec.technique)[0]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='#790: the shape-lock veto misses confident wrong solutions on irregular mesh '
    'bodies; mesh seed 9 ends a 0.91-confidence success 8.4 px off at lumpiness 0.7 on '
    'noise seeds 808 and 2424',
)
def test_irregularity_known_confident_wrong_frames() -> None:
    """No mesh shape at lumpiness 0.7 is a confident (>= 0.9) success more than 5 px off.

    A frame whose fit is several pixels off must not report a confident success,
    whatever the shape.  Over the 325 frames of mesh seeds 0 to 12 on 25 noise draws,
    two do: mesh seed 9 on noise seeds 808 and 2424, each a 0.91-confidence success
    8.4 px off.  Its blob is displaced together with the disc lock, so the
    body-witness disagreement the veto reads never appears.  The defect is that rare,
    and no 5 draws catch it reliably, so the check navigates those two frames.
    """
    confident_wrong = [
        frame
        for frame in _CONFIDENT_WRONG_FRAMES
        if (row := _frame(*frame)).status == 'success'
        and row.confidence >= _CONFIDENT
        and row.offset_error_px is not None
        and row.offset_error_px > _CONFIDENT_WRONG_PX
    ]
    assert confident_wrong == []
