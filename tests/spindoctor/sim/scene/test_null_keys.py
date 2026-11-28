"""A scene key set to null is refused at any depth, with its full key path.

A scene leaves a key out to take its default, so a null means nothing anywhere in a
scene.  Each test plants one null and checks that validation refuses it with a message
that gives the key's path.
"""

import re
from pathlib import Path

import pytest
from tests.spindoctor.sim.scenes import body_scene, frame_scene, star_entry

from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.io import load_sim_scene
from spindoctor.sim.scene.validate import validate_sim_params

_LEAVE_OUT = 'is null; leave the key out to take its default'


def test_a_null_top_level_number_is_refused() -> None:
    """A null top-level number is refused with the way to take its default."""
    with pytest.raises(
        SimSceneValidationError, match=re.escape(f'probe: exposure_sec {_LEAVE_OUT}')
    ):
        validate_sim_params(frame_scene(exposure_sec=None), source='probe')


def test_a_null_key_in_a_block_is_refused() -> None:
    """A null key of a top-level block is refused."""
    with pytest.raises(SimSceneValidationError, match=re.escape(f'noise.poisson {_LEAVE_OUT}')):
        validate_sim_params(frame_scene(noise={'poisson': None}), source='probe')


def test_a_null_key_in_a_nested_block_is_refused() -> None:
    """A null noise value inside instrument_config is refused before a render reads it."""
    scene = frame_scene(instrument_config={'noise': {'saturation_dn': None}})
    path = 'instrument_config.noise.saturation_dn'
    with pytest.raises(SimSceneValidationError, match=re.escape(f'{path} {_LEAVE_OUT}')):
        validate_sim_params(scene, source='probe')


def test_a_null_key_in_a_list_entry_is_refused() -> None:
    """A null key of a mapping inside a list is refused, and its path gives the index."""
    scene = frame_scene(stars=[star_entry(20.0, 20.0, 8.0, catalog_error_v=None)])
    with pytest.raises(
        SimSceneValidationError, match=re.escape(f'stars[0].catalog_error_v {_LEAVE_OUT}')
    ):
        validate_sim_params(scene, source='probe')


def test_a_null_list_entry_is_refused() -> None:
    """A null entry of a list is refused, and its path gives the entry's index."""
    scene = body_scene(body={'pose_euler_deg': [10.0, None, 5.0]})
    message = 'bodies[0].pose_euler_deg[1] is null; give the list entry a value or remove it'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_null_key_in_a_nav_override_is_refused() -> None:
    """A null key inside a body's nav_override is refused like a null anywhere else."""
    scene = body_scene(body={'nav_override': {'mesh_lumpiness': None}})
    path = 'bodies[0].nav_override.mesh_lumpiness'
    with pytest.raises(SimSceneValidationError, match=re.escape(f'{path} {_LEAVE_OUT}')):
        validate_sim_params(scene, source='probe')


def test_a_null_inside_a_tuple_is_refused() -> None:
    """A null entry of a tuple is refused like one of a list, with the entry's path."""
    scene = body_scene(body={'pose_euler_deg': (10.0, None, 5.0)})
    message = 'bodies[0].pose_euler_deg[1] is null; give the list entry a value or remove it'
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(scene, source='probe')


def test_a_null_scene_name_in_a_file_is_refused_as_a_null(tmp_path: Path) -> None:
    """A file's null header key is refused as a null, before the header checks read it."""
    path = tmp_path / 'probe.yaml'
    path.write_text('schema_version: 2\nscene_name: null\ninstrument: coiss_nac\n')
    with pytest.raises(SimSceneValidationError, match=re.escape(f'scene_name {_LEAVE_OUT}')):
        load_sim_scene(path)
