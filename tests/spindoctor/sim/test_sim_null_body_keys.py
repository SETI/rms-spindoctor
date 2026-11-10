"""A null body key reads as absent to the renderer, the navigator, and the validator.

The validator accepts null for every key of a ``bodies`` entry and for every
optional key of the body's sub-blocks, and reads it as absent.  The renderer
parses each body once into a typed spec, and the specs of a scene with a null
key and the scene without it are equal, so the two render the same frame.  The
navigator's filtered view leaves a null key out, so it reads the absent key too.
Atmosphere keys are covered in ``test_sim_atmosphere.py``.
"""

import copy
import dataclasses
import re
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.atmosphere import AtmosphereSpec, atmosphere_spec_from_params
from spindoctor.sim.forward.body_spec import BodySpec, body_spec_from_params
from spindoctor.sim.forward.body_texture import AlbedoTextureSpec, albedo_spec_from_params
from spindoctor.sim.forward.haze_structure import CLOUD_BLOB_KEYS
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import SimSceneValidationError, build_nav_params, validate_sim_params
from spindoctor.sim.scene_checks_body import (
    _ALBEDO_TEXTURE_KEYS,
    _DISC_TEXTURE_KEYS,
    _OPPOSITION_SURGE_KEYS,
    _POSE_SCATTER_KEYS,
    _SURFACE_SPOT_KEYS,
    _TRANSIT_MOON_KEYS,
    _TRANSIT_SHADOW_KEYS,
)
from spindoctor.sim.scene_schema import _BODY_IDEALIZED_KEYS, _BODY_KEYS, ELLIPSOID_UNSUPPORTED_KEYS

_SIZE = 32

# Body keys only the mesh renderer reads, so a null one is checked on a mesh body.
_MESH_PATH_KEYS = frozenset(
    {
        'mesh_lumpiness',
        'mesh_n_lat',
        'mesh_n_lon',
        'mesh_seed',
        'mesh_detail_octaves',
        'pose_euler_deg',
        *ELLIPSOID_UNSUPPORTED_KEYS,
    }
)

# A path step is a mapping key or a list index.
_Path = tuple[str | int, ...]


def _ellipsoid_body() -> dict[str, Any]:
    """An ellipsoid body that gives every key the ellipsoid renderer reads a value."""
    spot = {'lat_deg': 10.0, 'lon_deg': 80.0, 'radius_deg': 12.0, 'albedo_factor': 0.6}
    return {
        'name': 'RHEA',
        'shape_model': 'ellipsoid',
        'center_v': 15.0,
        'center_u': 17.0,
        'axis1': 14.0,
        'axis2': 12.0,
        'axis3': 11.0,
        'rotation_z': 20.0,
        'rotation_tilt': 10.0,
        'illumination_angle': 30.0,
        'phase_angle': 40.0,
        'range_km': 500000.0,
        'km_per_pixel': 6.0,
        'crater_fill': 0.3,
        'crater_min_radius': 0.06,
        'crater_max_radius': 0.2,
        'crater_power_law_exponent': 2.5,
        'crater_relief_scale': 0.5,
        'anti_aliasing': 0.8,
        'limb_relief_rms': 0.01,
        'limb_relief_corr_deg': 20.0,
        'photometric_law': 'minnaert',
        'minnaert_k': 0.7,
        'opposition_surge': {'amplitude': 0.2, 'width_deg': 5.0},
        'albedo_texture': {'rms': 0.05, 'corr_px': 4.0, 'spots': [dict(spot)]},
        'disc_texture': {
            'band_amplitude': 0.1,
            'band_wavenumber': 6.0,
            'band_phase_deg': 15.0,
            'storms': [dict(spot)],
        },
        'transits': [
            {
                'moon': {'dv_px': 1.0, 'du_px': -2.0, 'radius_px': 2.0, 'albedo_factor': 0.8},
                'shadow': {'dv_px': -1.0, 'du_px': 2.0, 'radius_px': 2.0, 'darkness': 0.7},
            }
        ],
        'atmosphere': {
            'scale_height_px': 2.0,
            'tau_ref': 0.5,
            'cloud_blobs': [{'center_vu': [1.0, -1.0], 'sigma_px': 2.0, 'amplitude': 0.2}],
        },
        'seed': 11,
        'nav_override': {'center_v': 15.5},
    }


def _mesh_body() -> dict[str, Any]:
    """A mesh body that gives every key the mesh renderer reads a value."""
    return {
        'name': 'HYPERION',
        'shape_model': 'polyhedral_mesh',
        'center_v': 16.0,
        'center_u': 16.0,
        'axis1': 14.0,
        'axis2': 11.0,
        'axis3': 10.0,
        'illumination_angle': 30.0,
        'phase_angle': 40.0,
        'mesh_lumpiness': 0.2,
        'mesh_n_lat': 12,
        'mesh_n_lon': 24,
        'mesh_seed': 3,
        'mesh_detail_octaves': 1,
        'pose_euler_deg': [10.0, 20.0, 5.0],
        'shading': 'gouraud',
        'pose_scatter': {'sigma_deg': 2.0},
        'limb_relief_rms': 0.01,
        'seed': 5,
    }


def _scene(body: dict[str, Any]) -> dict[str, Any]:
    """A small coiss_nac scene holding ``body``."""
    return {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 7,
        'bodies': [body],
    }


def _base_body(key: str) -> dict[str, Any]:
    """The body whose renderer reads ``key``."""
    return _mesh_body() if key in _MESH_PATH_KEYS else _ellipsoid_body()


def _render_inputs(
    body: dict[str, Any],
) -> tuple[BodySpec, AlbedoTextureSpec | None, AtmosphereSpec | None]:
    """The parsed body, with the sub-block specs the renderer builds from it.

    The spec keeps the albedo texture and the atmosphere as mappings, which the
    renderer parses when it renders; they are replaced by those parsed specs.
    """
    spec = body_spec_from_params(body, index=0)
    return (
        dataclasses.replace(spec, albedo_texture=None, atmosphere=None),
        albedo_spec_from_params(spec.albedo_texture, seed=0),
        atmosphere_spec_from_params(spec.atmosphere, oversample=1),
    )


@pytest.mark.parametrize('key', sorted(_BODY_KEYS))
def test_null_body_key_parses_as_absent(key: str) -> None:
    """A validated null body key gives the spec of a body without the key."""
    absent = _base_body(key)
    absent.pop(key, None)
    stated = {**absent, key: None}
    validate_sim_params(_scene(stated), source='null')
    assert body_spec_from_params(stated, index=0) == body_spec_from_params(absent, index=0)


def _nested_cases() -> list[tuple[str, _Path, str]]:
    """Every optional key of a body sub-block, with the body it is checked on."""
    paths: list[tuple[str, _Path, frozenset[str]]] = [
        ('ellipsoid', ('opposition_surge',), _OPPOSITION_SURGE_KEYS),
        ('ellipsoid', ('albedo_texture',), _ALBEDO_TEXTURE_KEYS),
        ('ellipsoid', ('albedo_texture', 'spots', 0), _SURFACE_SPOT_KEYS),
        ('ellipsoid', ('disc_texture',), _DISC_TEXTURE_KEYS),
        ('ellipsoid', ('disc_texture', 'storms', 0), _SURFACE_SPOT_KEYS),
        ('ellipsoid', ('transits', 0, 'moon'), _TRANSIT_MOON_KEYS),
        ('ellipsoid', ('transits', 0, 'shadow'), _TRANSIT_SHADOW_KEYS),
        (
            'ellipsoid',
            ('atmosphere', 'cloud_blobs', 0),
            CLOUD_BLOB_KEYS - {'sigma_px', 'amplitude'},
        ),
        ('mesh', ('pose_scatter',), _POSE_SCATTER_KEYS),
    ]
    return [(shape, path, key) for shape, path, keys in paths for key in sorted(keys)]


def _mapping(body: dict[str, Any], path: _Path) -> dict[str, Any]:
    """The mapping at ``path`` under the body."""
    node: Any = body
    for step in path:
        node = node[step]
    assert isinstance(node, dict)
    return node


@pytest.mark.parametrize(
    ('shape', 'path', 'key'),
    _nested_cases(),
    ids=lambda value: '.'.join(str(step) for step in value) if isinstance(value, tuple) else value,
)
def test_null_sub_block_key_parses_as_absent(shape: str, path: _Path, key: str) -> None:
    """A validated null sub-block key gives the render inputs of a body without it."""
    absent = _mesh_body() if shape == 'mesh' else _ellipsoid_body()
    _mapping(absent, path).pop(key, None)
    stated = copy.deepcopy(absent)
    _mapping(stated, path)[key] = None
    validate_sim_params(_scene(stated), source='null')
    assert _render_inputs(stated) == _render_inputs(absent)


@pytest.mark.parametrize('shape', ['ellipsoid', 'mesh'])
def test_null_body_keys_render_as_absent(shape: str) -> None:
    """A body whose optional keys are all null renders as the body without them.

    The body keeps its name and geometry, and every other key its renderer reads
    is null.
    """
    kept = {'name', 'shape_model', 'center_v', 'center_u', 'axis1', 'axis2', 'axis3'}
    full = _mesh_body() if shape == 'mesh' else _ellipsoid_body()
    absent = {key: value for key, value in full.items() if key in kept}
    stated = {**absent, **{key: None for key in full if key not in kept}}
    image, _ = render_combined_model(_scene(stated))
    expected, _ = render_combined_model(_scene(absent))
    np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize('key', sorted(_BODY_IDEALIZED_KEYS))
def test_null_body_key_reaches_the_navigator_as_absent(key: str) -> None:
    """The navigator's view of a body with a null key equals its view without the key.

    The body carries no ``nav_override``, so the key's own null is what is tested.
    """
    absent = {k: v for k, v in _base_body(key).items() if k not in {key, 'nav_override'}}
    stated = {**absent, key: None}
    validate_sim_params(_scene(stated), source='null')
    assert build_nav_params(_scene(stated)) == build_nav_params(_scene(absent))


def test_null_nav_override_key_leaves_the_scene_value() -> None:
    """A null nav_override key overrides nothing; the navigator sees the scene's value."""
    body = {**_ellipsoid_body(), 'nav_override': {'center_v': None}}
    validate_sim_params(_scene(body), source='null')
    assert build_nav_params(_scene(body))['bodies'][0]['center_v'] == 15.0


def test_null_body_name_stays_out_of_the_navigator_view() -> None:
    """A null name is left out, so the navigator takes its default rather than 'None'."""
    body = {**_ellipsoid_body(), 'name': None}
    assert 'name' not in build_nav_params(_scene(body))['bodies'][0]


def test_null_body_name_takes_the_positional_name() -> None:
    """A null body name keys the render truth by the positional default."""
    body = {'name': None, 'center_v': 16.0, 'center_u': 16.0, 'axis1': 10.0, 'axis2': 10.0}
    _, meta = render_combined_model(_scene(body))
    assert meta['order_near_to_far'] == ['SIM-BODY-1']


@pytest.mark.parametrize(
    ('body', 'message'),
    [
        (
            {**_mesh_body(), 'pose_euler_deg': [None, 20.0, 5.0]},
            'bodies[0].pose_euler_deg entries must be numbers; got null',
        ),
        (
            {
                **_ellipsoid_body(),
                'atmosphere': {
                    'scale_height_px': 2.0,
                    'tau_ref': 0.5,
                    'cloud_blobs': [{'center_vu': [None, 1.0], 'sigma_px': 2.0, 'amplitude': 0.2}],
                },
            },
            'bodies[0].atmosphere.cloud_blobs[0].center_vu entries must be numbers; got null',
        ),
    ],
    ids=['pose_euler_deg', 'center_vu'],
)
def test_a_null_list_entry_is_refused(body: dict[str, Any], message: str) -> None:
    """A fixed-length list has no default for one entry, so a null entry is refused."""
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        validate_sim_params(_scene(body), source='null')
