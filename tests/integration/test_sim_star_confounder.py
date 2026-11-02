"""Star-confounder lock regimes: recovery of the planted offset.

The one/two/three-star lock scenes plant a single navigable star, a navigable
pair, and a navigable triangle in fields of non-navigable confounders the
navigator never learns about, at a confounder density where the star techniques
should still win.  These tests navigate each scene and assert the planted offset
is recovered within tolerance -- the "1/2/3-star lock scenes navigate"
deliverable.  The saturated-star and double-star scenes plant a genuine centroid
bias (a clipped core, an unresolved companion) and assert recovery inside a
documented tolerance that absorbs it.

The expected status / confidence tier for each scene is asserted separately,
through the ``expected`` block and the :mod:`tests.integration.sim_expected`
machinery (``test_sim_expected``); this module owns the offset-recovery half.

Everything renders and navigates in-process (no holdings or SPICE), but the full
ensemble runs per scene, so the module is ``@pytest.mark.integration``.
"""

import math
from pathlib import Path

import pytest

from spindoctor.sim.scene import load_sim_scene
from tests.integration.sim_expected import navigate_scene

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes' / 'star_confounder'


def _offset_error(scene_name: str) -> float:
    """Navigate a star_confounder scene and return the recovered-vs-planted error."""
    scene = load_sim_scene(_SCENES_ROOT / f'{scene_name}.yaml')
    result = navigate_scene(scene)
    assert result.status == 'success', f'{scene_name}: expected success, got {result.status}'
    assert result.offset_px is not None
    dv = result.offset_px[0] - float(scene['offset_v'])
    du = result.offset_px[1] - float(scene['offset_u'])
    return math.hypot(dv, du)


# The clean lock regimes recover the planted offset to a small fraction of a
# pixel; the confounders do not perturb the fit.
_LOCK_TOLERANCE_PX = 0.5


def test_lock_single_recovers_offset() -> None:
    """The one-star lock recovers the planted offset within tolerance."""
    assert _offset_error('lock_single') < _LOCK_TOLERANCE_PX


def test_lock_pair_recovers_offset() -> None:
    """The two-star lock recovers the planted offset within tolerance."""
    assert _offset_error('lock_pair') < _LOCK_TOLERANCE_PX


def test_lock_triangle_recovers_offset() -> None:
    """The three-star triangle lock recovers the planted offset within tolerance."""
    assert _offset_error('lock_triangle') < _LOCK_TOLERANCE_PX


def test_lock_single_dense_recovers_offset() -> None:
    """The near-breakdown dense-confounder lock still recovers the offset."""
    assert _offset_error('lock_single_dense') < _LOCK_TOLERANCE_PX


def test_saturated_star_recovers_within_bias_tolerance() -> None:
    """A clipped bright star navigates; the tolerance absorbs its centroid bias.

    The measured recovered-vs-planted error at this geometry is ~0.61 px,
    dominated by the clipped-core centroid shift; the tolerance is 1.0 px.
    """
    assert _offset_error('saturated_star') < 1.0


def test_double_star_recovers_within_photocenter_tolerance() -> None:
    """A companion-bearing star navigates; the tolerance absorbs the photocenter pull.

    The measured recovered-vs-planted error is ~0.38 px (the unresolved
    companion's photocenter shift); the tolerance is 0.6 px.
    """
    assert _offset_error('double_star') < 0.6
