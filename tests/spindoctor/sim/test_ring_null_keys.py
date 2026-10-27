"""A null optional ring key means unset, to every ring reader as to the validator.

The validator accepts null for every optional key in the ring block and reads it
as absent.  The renderer, the orbit parser the navigator-side model shares, the
validator's orbit-error bounds, and the sim editor read it the same way: a scene
whose optional ring key is null renders exactly the frame the scene without the
key renders.  The editor's side is in ``test_sim_editor_ring_nulls.py``.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model
from spindoctor.sim.ring_geometry import ring_orbit_from_mapping
from spindoctor.sim.scene import load_sim_scene, validate_sim_params
from spindoctor.sim.scene_schema import scene_value

_SCENES = Path(__file__).parents[2] / 'integration' / 'sim_scenes'

# A path step is a mapping key or a list index.
_Path = tuple[str | int, ...]


def _scene(name: str) -> dict[str, Any]:
    """A shipped ring scene, augmented so every optional ring key is exercised.

    ``spoked_sheet`` gains an azimuthal modulation and shadow, and
    ``orbit_error_ringlet`` an orbit error carrying all three deltas.
    """
    scene = load_sim_scene(_SCENES / 'ring_system' / f'{name}.yaml')
    ring_system = scene['ring_system']
    if name == 'spoked_sheet':
        ring_system['azimuthal']['modulation'] = {
            'amplitude': 0.2,
            'm': 2,
            'lam_peak_deg': 10.0,
        }
        ring_system['azimuthal']['shadow'] = {
            'start_deg': 10.0,
            'extent_deg': 30.0,
            'darkness': 0.5,
        }
    if name == 'orbit_error_ringlet':
        ring_system['features'][0]['orbit_error'] = {
            'delta_a_px': 0.5,
            'delta_ae_px': 0.2,
            'delta_long_peri_deg': 3.0,
        }
    return scene


def _mapping(scene: dict[str, Any], path: _Path) -> dict[str, Any]:
    """The mapping at ``path`` under the scene's ring system."""
    node: Any = scene['ring_system']
    for step in path:
        node = node[step]
    assert isinstance(node, dict)
    return node


_GEOMETRY: _Path = ('geometry',)
_FEATURE0: _Path = ('features', 0)
_ORBIT0: _Path = ('features', 0, 'orbit')

# (scene, mapping path under ring_system, optional key): every optional ring key
# the renderer reads, in a scene whose render consults it.
_CASES: list[tuple[str, _Path, str]] = [
    ('mmode_ringlet', _GEOMETRY, 'center_v'),
    ('mmode_ringlet', _GEOMETRY, 'center_u'),
    ('mmode_ringlet', _GEOMETRY, 'node_deg'),
    ('mmode_ringlet', (), 'phase_deg'),
    ('mmode_ringlet', _FEATURE0, 'albedo'),
    ('mmode_ringlet', _FEATURE0, 'phase_g'),
    ('mmode_ringlet', _ORBIT0, 'ae_px'),
    ('mmode_ringlet', _ORBIT0, 'long_peri_deg'),
    ('mmode_ringlet', _ORBIT0, 'rate_peri_deg_per_day'),
    ('mmode_ringlet', (*_ORBIT0, 'modes', 0), 'amp_px'),
    ('mmode_ringlet', (*_ORBIT0, 'modes', 0), 'peri_deg'),
    ('edge_wave_gap', _FEATURE0, 'side'),
    ('edge_wave_gap', ('features', 1, 'orbit', 'edge_wave'), 'amp_px'),
    ('edge_wave_gap', ('features', 1, 'orbit', 'edge_wave'), 'lam0_deg'),
    ('moonlet_propeller', ('moonlets', 0), 'radius_px'),
    ('moonlet_propeller', (), 'km_per_pixel'),
    ('orbit_error_ringlet', _ORBIT0, 'ae_px'),
    ('orbit_error_ringlet', ('features', 0, 'orbit_error'), 'delta_a_px'),
    ('orbit_error_ringlet', ('features', 0, 'orbit_error'), 'delta_ae_px'),
    ('orbit_error_ringlet', ('features', 0, 'orbit_error'), 'delta_long_peri_deg'),
    ('spoked_sheet', ('azimuthal', 'modulation'), 'm'),
    ('spoked_sheet', ('azimuthal', 'modulation'), 'lam_peak_deg'),
    ('spoked_sheet', ('azimuthal', 'shadow'), 'start_deg'),
]


def _case_id(case: tuple[str, _Path, str]) -> str:
    """A readable id: the scene and the key's path."""
    name, path, key = case
    return '.'.join([name, *(str(step) for step in path), key])


def _render(scene: dict[str, Any]) -> np.ndarray:
    """Validate ``scene`` and render it, returning the image."""
    image, _meta = render_combined_model(validate_sim_params(scene, source='case'))
    return np.asarray(image)


@pytest.mark.parametrize('case', _CASES, ids=[_case_id(case) for case in _CASES])
def test_a_null_ring_key_renders_as_the_absent_key(case: tuple[str, _Path, str]) -> None:
    """Null and absent give the same frame, and neither raises in a reader."""
    name, path, key = case
    with_null = _scene(name)
    _mapping(with_null, path)[key] = None
    without = _scene(name)
    _mapping(without, path).pop(key, None)
    np.testing.assert_array_equal(_render(with_null), _render(without))


def test_the_orbit_parser_reads_every_null_as_its_default() -> None:
    """The navigator's shared orbit parser gives a null-keyed orbit the bare orbit's values."""
    nulls = {
        'a_px': 60.0,
        'ae_px': None,
        'long_peri_deg': None,
        'rate_peri_deg_per_day': None,
        'modes': [{'m': 2, 'amp_px': None, 'peri_deg': None}],
        'edge_wave': {'amp_px': None, 'wavelength_px': 10.0, 'damping_rad': 0.6, 'lam0_deg': None},
    }
    bare = {
        'a_px': 60.0,
        'modes': [{'m': 2}],
        'edge_wave': {'wavelength_px': 10.0, 'damping_rad': 0.6},
    }
    assert ring_orbit_from_mapping(nulls) == ring_orbit_from_mapping(bare)


def test_orbit_error_bounds_read_a_null_ae_as_zero() -> None:
    """With ae_px null, the bound refuses the delta it refuses with ae_px absent."""
    scene = _scene('orbit_error_ringlet')
    feature = scene['ring_system']['features'][0]
    feature['orbit']['ae_px'] = None
    feature['orbit_error'] = {'delta_ae_px': 61.0}
    with pytest.raises(ValueError, match=r'effective eccentric amplitude to 61\.0 px'):
        validate_sim_params(copy.deepcopy(scene), source='case')


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(None, 7.0), (0.0, 0.0), (3.5, 3.5)],
    ids=['null', 'zero', 'value'],
)
def test_scene_value_reads_null_as_the_default(value: float | None, expected: float) -> None:
    """A null takes the default; a zero or any other value is kept."""
    assert scene_value({'k': value}, 'k', 7.0) == expected
