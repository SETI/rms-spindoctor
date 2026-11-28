"""Keys a scene leaves out take their defaults, and a key it sets is checked.

Each test leaves one key out, or sets one to a value of the wrong kind, and checks the
validator's answer: the bound that reads an absent key as its default, the entries of a
fixed-length list, and the names the navigator's view and the renderer give the bodies.
"""

import re

import pytest
from tests.spindoctor.sim.scenes import body_scene, frame_scene, ring_system_block

from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params


def test_a_non_number_pose_entry_is_refused() -> None:
    """An entry of ``pose_euler_deg`` that isn't a number is refused."""
    scene = body_scene(body={'shape_model': 'polyhedral_mesh', 'pose_euler_deg': [10.0, 'x', 5.0]})
    message = "bodies[0].pose_euler_deg[1]: expected a number, got str 'x'"
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_non_number_cloud_center_entry_is_refused() -> None:
    """An entry of a cloud's ``center_vu`` that isn't a number is refused."""
    blob = {'center_vu': ['x', 1.0], 'sigma_px': 2.0, 'amplitude': 0.2}
    atmosphere = {'scale_height_px': 2.0, 'tau_ref': 0.5, 'cloud_blobs': [blob]}
    scene = body_scene(body={'atmosphere': atmosphere})
    message = "bodies[0].atmosphere.cloud_blobs[0].center_vu[0]: expected a number, got str 'x'"
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_the_orbit_error_bound_reads_an_absent_ae_as_zero() -> None:
    """With ``ae_px`` left out of the orbit, the bound counts it as 0 px."""
    feature = {
        'name': 'F1',
        'kind': 'ringlet',
        'tau': 1.0,
        'width_px': 10.0,
        'orbit': {'a_px': 40.0},
        'orbit_error': {'delta_ae_px': 61.0},
    }
    scene = frame_scene(ring_system=ring_system_block(32.0, 32.0, [feature]))
    with pytest.raises(SimSceneValidationError, match=r'effective eccentric amplitude to 61\.0 px'):
        validate_sim_params(scene, source='probe')


def test_a_nav_override_name_that_collides_is_refused() -> None:
    """Two bodies whose navigator names match are refused, as for the bodies' own names."""
    scene = body_scene(body={'name': 'RHEA', 'nav_override': {'name': 'DIONE'}})
    scene['bodies'].append({**scene['bodies'][0], 'name': 'DIONE', 'center_v': 5.0})
    scene['bodies'][1].pop('nav_override')
    with pytest.raises(SimSceneValidationError, match=r'share the navigator-side effective name'):
        validate_sim_params(scene, source='probe')
