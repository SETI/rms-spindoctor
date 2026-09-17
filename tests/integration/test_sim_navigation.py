"""End-to-end navigation of simulated images.

Renders a sim scene into an ObsSim, builds its models, runs the orchestrator,
and checks the navigator succeeds and recovers a planted offset.  Everything is
in-process (no holdings or SPICE), so this runs in the default suite -- it is the
first rung of the planted-offset algorithmic-invariant layer (Phase T4).
"""

from typing import Any

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim


def _disc_scene(*, offset_v: float = 0.0, offset_u: float = 0.0) -> dict[str, Any]:
    """A clean centered disc scene with an optional planted offset."""
    return {
        'size_v': 128,
        'size_u': 128,
        'instrument': 'coiss_nac',
        'random_seed': 42,
        'offset_v': offset_v,
        'offset_u': offset_u,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 64.0,
                'center_u': 64.0,
                'axis1': 86.0,
                'axis2': 86.0,
                'axis3': 86.0,
                'illumination_angle': 18.0,
                'phase_angle': 30.0,
            }
        ],
    }


def _navigate(scene: dict[str, Any]) -> Any:
    obs = ObsSim.from_file('/tmp/nav.json', sim_params=scene)
    orchestrator = NavOrchestrator(build_models_for_obs(obs), only_models='*', only_techniques='*')
    return orchestrator.navigate(obs)


def test_sim_body_navigates_successfully() -> None:
    """A clean sim body frame navigates to a success status."""
    result = _navigate(_disc_scene())
    assert result.status == 'success'


def test_sim_navigation_classifies_clean() -> None:
    """The bias pedestal keeps the dark sky from reading as missing data."""
    result = _navigate(_disc_scene())
    assert result.image_classifier.image_class == 'clean'


def test_sim_recovers_planted_offset_v() -> None:
    """The navigator recovers the planted v offset within tolerance."""
    result = _navigate(_disc_scene(offset_v=3.0, offset_u=-2.0))
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - 3.0) < 1.0


def test_sim_recovers_planted_offset_u() -> None:
    """The navigator recovers the planted u offset within tolerance."""
    result = _navigate(_disc_scene(offset_v=3.0, offset_u=-2.0))
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - (-2.0)) < 1.0
