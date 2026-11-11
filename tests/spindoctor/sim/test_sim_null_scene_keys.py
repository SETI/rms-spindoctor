"""Null star, optics, sky, spk_error, detector, and artifacts keys read as absent.

The validator accepts null for every optional key of a ``stars`` entry and its
``companion``, of the ``optics`` block and its sub-blocks, of ``sky_counts``,
of ``spk_error``, of ``detector``, and of ``artifacts`` (apart from a mode's map),
and reads it as absent.  The renderer reads these keys the same way: a scene whose
optional key is null renders the same frame, and records the same star truth, as
the scene without the key.  The body, noise, and ring blocks have their
own null tests.
"""

import copy
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_checks import (
    _COMPANION_KEYS,
    _DETECTOR_KEYS,
    _DISTORTION_KEYS,
    _GHOST_KEYS,
    _OPTICS_KEYS,
    _PSF_KEYS,
    _SKY_COUNTS_KEYS,
    _SMEAR_ENTRY_KEYS,
    _SPK_ERROR_KEYS,
    _STRAY_LIGHT_KEYS,
)
from spindoctor.sim.scene_schema import _STAR_KEYS
from spindoctor.sim.star_records import star_record_from_params

_SIZE = 32

# A path step is a mapping key or a list index.
_Path = tuple[str | int, ...]


def _scene() -> dict[str, Any]:
    """A small scene that gives a value to every optional key of four blocks.

    The four blocks are ``stars``, ``sky_counts``, ``spk_error``, and ``optics``.
    """
    return {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 7,
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 10.0,
                'center_u': 10.0,
                'axis1': 8.0,
                'axis2': 8.0,
                'range_km': 500000.0,
            }
        ],
        'stars': [
            {
                'name': 'BRIGHT',
                'catalog_name': 'UCAC4',
                'spectral_class': 'K0',
                'v': 22.0,
                'u': 21.0,
                'vmag': 6.0,
                'move_v': 0.5,
                'move_u': -0.5,
                'psf_size': [9, 9],
                'psf_sigma': 1.2,
                'navigable': True,
                'catalog_error_v': 0.3,
                'catalog_error_u': -0.2,
                'delta_mag': 0.4,
                'companion': {'sep_px': 1.5, 'delta_mag': 1.0, 'angle_deg': 30.0},
            }
        ],
        'sky_counts': {'a': -3.0, 'b': 0.3, 'density_factor': 2.0, 'diffuse_e_per_px': 3.0},
        'spk_error': {'dv_px': 0.4, 'du_px': -0.3, 'reference_range_km': 400000.0},
        'optics': {
            'psf': {'sigma_v': 1.0, 'sigma_u': 1.2, 'w': 0.1, 'r0': 2.5, 'n': 3.5},
            'smear': [{'dv_px': 0.6, 'du_px': 0.2, 'object_class': 'all'}],
            'distortion': {
                'k1': 1e-6,
                'k2': 1e-10,
                'center_v': 15.0,
                'center_u': 17.0,
                'nonradial_rms_px': 0.1,
            },
            'ghosts': [{'dv_px': 3.0, 'du_px': -2.0, 'amplitude': 0.05, 'defocus_sigma': 1.5}],
            'stray_light': {
                'amplitude': 0.02,
                'direction_deg': 40.0,
                'model': 'linear',
                'center_v': 12.0,
                'center_u': 14.0,
            },
        },
    }


def _mapping(scene: dict[str, Any], path: _Path) -> dict[str, Any]:
    """The mapping at ``path`` in the scene."""
    node: Any = scene
    for step in path:
        node = node[step]
    assert isinstance(node, dict)
    return node


def _scenes(path: _Path, key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The scene with ``key`` at ``path`` set to null, and the scene without it.

    The null scene is validated, so each case checks a null the validator accepts.
    """
    absent = _scene()
    _mapping(absent, path).pop(key, None)
    stated = copy.deepcopy(absent)
    _mapping(stated, path)[key] = None
    validate_sim_params(stated, source='null')
    return stated, absent


def _assert_renders_alike(stated: dict[str, Any], absent: dict[str, Any]) -> None:
    """Both scenes render the same frame and record the same star truth."""
    image, meta = render_combined_model(stated)
    expected, expected_meta = render_combined_model(absent)
    np.testing.assert_array_equal(image, expected)
    assert meta['star_info'] == expected_meta['star_info']


def _cases(blocks: list[tuple[_Path, frozenset[str]]]) -> list[tuple[_Path, str]]:
    """Every (path, key) pair of the listed blocks."""
    return [(path, key) for path, keys in blocks for key in sorted(keys)]


def _case_id(value: Any) -> str:
    """A readable test id for a path or a key."""
    return '.'.join(str(step) for step in value) if isinstance(value, tuple) else str(value)


@pytest.mark.parametrize(
    ('path', 'key'),
    _cases([(('stars', 0), _STAR_KEYS), (('stars', 0, 'companion'), _COMPANION_KEYS)]),
    ids=_case_id,
)
def test_null_star_key_renders_as_absent(path: _Path, key: str) -> None:
    """A validated null star key gives the frame and star record of the absent key."""
    stated, absent = _scenes(path, key)
    _assert_renders_alike(stated, absent)
    centers = {'default_v': _SIZE / 2.0, 'default_u': _SIZE / 2.0}
    stated_record = star_record_from_params(stated['stars'][0], index=0, **centers)
    absent_record = star_record_from_params(absent['stars'][0], index=0, **centers)
    assert vars(stated_record) == vars(absent_record)


@pytest.mark.parametrize(
    ('path', 'key'),
    _cases(
        [
            (('optics',), _OPTICS_KEYS),
            # A null sigma_v or match_navigator is refused, so neither is a case.
            (('optics', 'psf'), _PSF_KEYS - {'sigma_v', 'match_navigator'}),
            # A null object_class is refused.
            (('optics', 'smear', 0), _SMEAR_ENTRY_KEYS - {'object_class'}),
            (('optics', 'distortion'), _DISTORTION_KEYS),
            (('optics', 'ghosts', 0), _GHOST_KEYS),
            (('optics', 'stray_light'), _STRAY_LIGHT_KEYS),
        ]
    ),
    ids=_case_id,
)
def test_null_optics_key_renders_as_absent(path: _Path, key: str) -> None:
    """A validated null optics key renders the frame the absent key renders."""
    _assert_renders_alike(*_scenes(path, key))


@pytest.mark.parametrize('key', sorted(_SKY_COUNTS_KEYS))
def test_null_sky_counts_key_renders_as_absent(key: str) -> None:
    """A validated null sky_counts key renders the frame the absent key renders."""
    _assert_renders_alike(*_scenes(('sky_counts',), key))


@pytest.mark.parametrize('key', sorted(_SPK_ERROR_KEYS))
def test_null_spk_error_key_renders_as_absent(key: str) -> None:
    """A validated null spk_error key renders the frame the absent key renders."""
    _assert_renders_alike(*_scenes(('spk_error',), key))


def test_null_detector_keys_render_as_absent() -> None:
    """A detector block whose every key is null renders as an empty detector block."""
    stated = {**_scene(), 'detector': dict.fromkeys(_DETECTOR_KEYS)}
    validate_sim_params(stated, source='null')
    _assert_renders_alike(stated, {**_scene(), 'detector': {}})


def test_null_artifacts_keys_render_as_absent() -> None:
    """Null artifacts switches and a null mode parameter render as the absent keys.

    A null ``instrument_defaults`` or ``adversarial`` acts like its False default
    whether the reader takes null as absent or as false, so only the null
    ``contiguous_run`` tests the reader.  A mode's map itself can't be null;
    validation refuses it.
    """
    stated = {
        **_scene(),
        'artifacts': {
            'instrument_defaults': None,
            'adversarial': None,
            'missing_lines': {'expected_events': 2.0, 'contiguous_run': None},
        },
    }
    validate_sim_params(stated, source='null')
    absent = {**_scene(), 'artifacts': {'missing_lines': {'expected_events': 2.0}}}
    _assert_renders_alike(stated, absent)
