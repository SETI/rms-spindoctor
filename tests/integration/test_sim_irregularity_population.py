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
# on 1 to 4 of the 13 mesh seeds per draw, 56 of the 325 frames in all, and only ever
# on mesh seeds 0, 1, 5, 10, and 12.  Over any 5 of the 25 draws it fires 5 to 16 times
# (median 11); over the 5 draws the test reads, 10 times.  Seed 2 also ends conflicted
# on most draws, but because its disc and limb disagree with each other; the veto
# plays no part.  A veto that stopped firing on these shapes would count 0.  The floor
# of 5 catches only the loss of about half the veto, because the smallest count over
# any 5 of the 25 draws is 5, against the 10 the test's own draws give.
_MIN_VETOED = 5
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
    """At lumpiness 0.7 the shape-lock veto declines at least 5 of the 65 frames.

    Over mesh seeds 0 to 12 the disc correlation and the limb fit lock onto the
    mismatched shape at the extreme step.  When the pose-free blob disagrees with
    that lock, the veto reports the frame conflicted, with the status reason
    ``body_shape_lock_suspect``.  The test counts that reason, so a frame that is
    conflicted for another reason does not count.  The population is the 13 shapes
    on each of 5 noise draws, because which shapes the veto catches moves with the
    noise: it fires on 1 to 4 shapes per draw.  On the remaining shapes the blob
    moves with the lock or carries the result itself.  A change that weakens the
    veto on these shapes fails here.
    """
    vetoed = [row for row in _population() if row.status_reason == _SHAPE_LOCK_REASON]
    assert len(vetoed) >= _MIN_VETOED


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='#790: the shape-lock veto misses confident wrong solutions on irregular mesh '
    'bodies; mesh seed 7 on noise seed 101 ends a 0.901-confidence success 5.2 px off at '
    'lumpiness 0.7',
)
def test_irregularity_population_never_confidently_wrong() -> None:
    """No mesh shape at lumpiness 0.7 is a confident (>= 0.9) success more than 5 px off.

    A frame whose fit is several pixels off must not report a confident success,
    whatever the shape.  Mesh seed 7 does on 3 of 25 noise draws (101, 1313, and
    2020), at 0.901 to 0.903 confidence and 5.2 to 5.3 px off.  On 17 others its
    confidence is 0.87 or above but under 0.9, so the defect sits at the edge of what
    the check counts: of the 5 draws the check reads, only noise seed 101 shows it.
    """
    confident_wrong = [
        int(row.value)
        for row in _population()
        if row.status == 'success'
        and row.confidence >= _CONFIDENT
        and row.offset_error_px is not None
        and row.offset_error_px > _CONFIDENT_WRONG_PX
    ]
    assert confident_wrong == []
