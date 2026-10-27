"""The ellipsoid-only appearance keys a polyhedral-mesh body refuses are one table.

The scene validator refuses every key in the schema's ``MESH_UNSUPPORTED_KEYS``
on a ``shape_model: polyhedral_mesh`` body, with the sentence
``mesh_unsupported_message`` builds.  The sim editor reads the same table and the
same sentence (``test_sim_editor_mesh_keys``).
"""

from typing import Any

import pytest

from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params
from spindoctor.sim.scene_schema import MESH_UNSUPPORTED_KEYS, mesh_unsupported_message

# The full table, pinned so a key cannot slip in or out of it unnoticed.
_EXPECTED_KEYS = {
    'atmosphere',
    'photometric_law',
    'minnaert_k',
    'opposition_surge',
    'albedo_texture',
    'disc_texture',
    'transits',
    'crater_fill',
    'crater_min_radius',
    'crater_max_radius',
    'crater_power_law_exponent',
    'crater_relief_scale',
}

# A valid value for each refused key, so only the mesh rule can refuse it.
_VALID_VALUES: dict[str, Any] = {
    'atmosphere': {'scale_height_px': 4.0, 'tau_ref': 1.0},
    'photometric_law': 'minnaert',
    'minnaert_k': 0.6,
    'opposition_surge': {'amplitude': 0.4, 'width_deg': 5.0},
    'albedo_texture': {'rms': 0.05, 'corr_px': 10.0},
    'disc_texture': {'band_amplitude': 0.1},
    'transits': [{'moon': {'radius_px': 3.0}}],
    'crater_fill': 0.5,
    'crater_min_radius': 0.06,
    'crater_max_radius': 0.2,
    'crater_power_law_exponent': 2.5,
    'crater_relief_scale': 0.5,
}


def _scene(body: dict[str, Any]) -> dict[str, Any]:
    """A minimal valid scene holding ``body`` as its only body."""
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'bodies': [body],
    }


def _mesh_body(**extra: Any) -> dict[str, Any]:
    """A minimal valid polyhedral-mesh body entry plus ``extra`` keys."""
    return {
        'name': 'LUMPY',
        'center_v': 64.0,
        'center_u': 64.0,
        'shape_model': 'polyhedral_mesh',
        'axis1': 30.0,
        'axis2': 26.0,
        'axis3': 24.0,
        **extra,
    }


def test_the_table_holds_exactly_the_twelve_ellipsoid_only_keys() -> None:
    """MESH_UNSUPPORTED_KEYS is the pinned twelve-key table, without duplicates."""
    assert set(MESH_UNSUPPORTED_KEYS) == _EXPECTED_KEYS
    assert len(MESH_UNSUPPORTED_KEYS) == len(_EXPECTED_KEYS)


def test_every_refused_key_validates_on_an_ellipsoid() -> None:
    """Each table key, with its test value, is a valid ellipsoid key."""
    body = {key: value for key, value in _VALID_VALUES.items() if key != 'minnaert_k'}
    body.update(name='ROUND', center_v=64.0, center_u=64.0, axis1=30.0, minnaert_k=0.6)
    validate_sim_params(_scene(body))


@pytest.mark.parametrize('key', sorted(_EXPECTED_KEYS))
def test_the_validator_refuses_the_key_with_the_shared_sentence(key: str) -> None:
    """A mesh body carrying the key fails with mesh_unsupported_message's sentence."""
    scene = _scene(_mesh_body(**{key: _VALID_VALUES[key]}))
    with pytest.raises(SimSceneValidationError) as excinfo:
        validate_sim_params(scene, source='scene.yaml')
    assert str(excinfo.value) == f'scene.yaml: bodies[0]: {mesh_unsupported_message([key])}'


def test_the_sentence_lists_the_keys_sorted() -> None:
    """mesh_unsupported_message names its keys in sorted order."""
    message = mesh_unsupported_message(['photometric_law', 'minnaert_k'])
    assert message.startswith("['minnaert_k', 'photometric_law'] are not supported on")
