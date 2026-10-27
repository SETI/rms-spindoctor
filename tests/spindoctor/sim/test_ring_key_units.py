"""Ring scene keys name their units, and the per-kind shape-key rules are one table.

Every length and angle key in the ``ring_system`` block carries its unit in its
name (``a_px``, ``long_peri_deg``, ``rate_peri_deg_per_day``, ``damping_rad``,
and so on).  Each retired name is refused with a message that names its
replacement and what the replacement holds; nothing translates it.  The
retired-to-current map is the schema's ``RENAMED_RING_KEYS``.

Which radial shape keys a feature kind takes is the schema's
``RING_KIND_SHAPE_KEYS``: the validator enforces exactly that table (a required
key's absence fails, a rejected key's presence fails, an optional key validates
either way).
"""

import re
from typing import Any

import pytest

from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params
from spindoctor.sim.scene_checks_ring import (
    _RING_EDGE_WAVE_KEYS,
    _RING_FEATURE_ORBIT_KEYS,
    _RING_MODULATION_KEYS,
    _RING_MOONLET_KEYS,
    _RING_ORBIT_MODE_KEYS,
    _RING_SPOKES_KEYS,
)
from spindoctor.sim.scene_schema import (
    _RING_FEATURE_KEYS,
    RENAMED_RING_KEYS,
    RING_KIND_SHAPE_KEYS,
    RING_SHAPE_KEYS,
    RING_SIDE_DEFAULTS,
    ring_key_description,
)

# The full retired -> current map, pinned so a rename cannot slip in or out of
# the table unnoticed.
_EXPECTED_RENAMES: dict[str, dict[str, str]] = {
    'ring_system.features': {
        'width': 'width_px',
        'wavelength': 'wavelength_px',
        'damping': 'damping_px',
    },
    'ring_system.features.orbit': {
        'a': 'a_px',
        'ae': 'ae_px',
        'long_peri': 'long_peri_deg',
        'rate_peri': 'rate_peri_deg_per_day',
    },
    'ring_system.features.orbit.modes': {'amp': 'amp_px', 'peri': 'long_peri_deg'},
    'ring_system.features.orbit.edge_wave': {
        'amp': 'amp_px',
        'wavelength': 'wavelength_px',
        'damp': 'damping_rad',
        'lam0': 'lam0_deg',
    },
    'ring_system.azimuthal.modulation': {'phase_deg': 'lam_peak_deg'},
    'ring_system.azimuthal.spokes': {'r_inner': 'r_inner_px', 'r_outer': 'r_outer_px'},
    'ring_system.moonlets': {'a': 'a_px'},
}

# Each block's current key inventory, as the validator holds it.
_INVENTORIES: dict[str, frozenset[str]] = {
    'ring_system.features': _RING_FEATURE_KEYS,
    'ring_system.features.orbit': _RING_FEATURE_ORBIT_KEYS,
    'ring_system.features.orbit.modes': _RING_ORBIT_MODE_KEYS,
    'ring_system.features.orbit.edge_wave': _RING_EDGE_WAVE_KEYS,
    'ring_system.azimuthal.modulation': _RING_MODULATION_KEYS,
    'ring_system.azimuthal.spokes': _RING_SPOKES_KEYS,
    'ring_system.moonlets': _RING_MOONLET_KEYS,
}

# Where each block sits in _full_scene(), and its label in validation messages.
_BLOCK_PATHS: dict[str, tuple[tuple[str | int, ...], str]] = {
    'ring_system.features': (('features', 0), 'ring_system.features[0]'),
    'ring_system.features.orbit': (('features', 0, 'orbit'), 'ring_system.features[0].orbit'),
    'ring_system.features.orbit.modes': (
        ('features', 0, 'orbit', 'modes', 0),
        'ring_system.features[0].orbit.modes[0]',
    ),
    'ring_system.features.orbit.edge_wave': (
        ('features', 0, 'orbit', 'edge_wave'),
        'ring_system.features[0].orbit.edge_wave',
    ),
    'ring_system.azimuthal.modulation': (
        ('azimuthal', 'modulation'),
        'ring_system.azimuthal.modulation',
    ),
    'ring_system.azimuthal.spokes': (('azimuthal', 'spokes'), 'ring_system.azimuthal.spokes'),
    'ring_system.moonlets': (('moonlets', 0), 'ring_system.moonlets[0]'),
}

_RETIRED = [(block, old) for block, renames in _EXPECTED_RENAMES.items() for old in renames]


def _feature(kind: str) -> dict[str, Any]:
    """A feature of ``kind`` carrying exactly the shape keys the kind requires."""
    values = {'width_px': 8.0, 'wavelength_px': 6.0, 'damping_px': 12.0}
    feature: dict[str, Any] = {
        'name': 'F1',
        'kind': kind,
        'tau': 1.0,
        'orbit': {'a_px': 40.0},
    }
    for key, rule in RING_KIND_SHAPE_KEYS[kind].items():
        if rule == 'required':
            feature[key] = values[key]
    return feature


def _scene(ring_system: dict[str, Any]) -> dict[str, Any]:
    """A minimal scene around ``ring_system``."""
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'ring_system': ring_system,
    }


def _full_scene() -> dict[str, Any]:
    """A valid scene whose ring system carries every block that had a renamed key."""
    feature = _feature('ringlet')
    feature['orbit'] = {
        'a_px': 40.0,
        'ae_px': 1.0,
        'long_peri_deg': 10.0,
        'rate_peri_deg_per_day': 0.5,
        'modes': [{'m': 2, 'amp_px': 1.5, 'long_peri_deg': 30.0}],
        'edge_wave': {'amp_px': 1.0, 'wavelength_px': 8.0, 'damping_rad': 0.5, 'lam0_deg': 90.0},
    }
    return _scene(
        {
            'geometry': {'opening_deg_obs': 30.0, 'opening_deg_sun': 20.0},
            'phase_deg': 20.0,
            'features': [feature],
            'azimuthal': {
                'modulation': {'amplitude': 0.2, 'm': 2, 'lam_peak_deg': 30.0},
                'spokes': {
                    'count': 3,
                    'r_inner_px': 30.0,
                    'r_outer_px': 50.0,
                    'contrast': -0.5,
                    'width_deg': 10.0,
                },
            },
            'moonlets': [{'a_px': 45.0, 'lam_deg': 10.0, 'amplitude': 0.3}],
        }
    )


def _block(scene: dict[str, Any], block: str) -> dict[str, Any]:
    """The mapping at ``block``'s path in a scene built by :func:`_full_scene`."""
    node: Any = scene['ring_system']
    for step in _BLOCK_PATHS[block][0]:
        node = node[step]
    assert isinstance(node, dict)
    return node


def test_the_rename_table_is_the_expected_map() -> None:
    """RENAMED_RING_KEYS maps exactly the retired ring keys to their unit-named keys."""
    actual = {
        block: {old: new for old, (new, _) in renames.items()}
        for block, renames in RENAMED_RING_KEYS.items()
    }
    assert actual == _EXPECTED_RENAMES


def test_the_full_scene_validates() -> None:
    """The scene the refusal tests start from is valid with every current key."""
    scene = _full_scene()
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize(('block', 'old'), _RETIRED, ids=[f'{b}.{o}' for b, o in _RETIRED])
def test_a_retired_ring_key_is_refused_with_its_replacement(block: str, old: str) -> None:
    """A retired key fails validation with a message naming its replacement and unit."""
    scene = _full_scene()
    new, description = RENAMED_RING_KEYS[block][old]
    mapping = _block(scene, block)
    if new in mapping:
        mapping[old] = mapping.pop(new)
    else:
        mapping[old] = 1.0
    label = _BLOCK_PATHS[block][1]
    expected = f"{label}: '{old}' was renamed; use '{new}' ({description})"
    with pytest.raises(SimSceneValidationError, match=re.escape(expected)):
        validate_sim_params(scene)


@pytest.mark.parametrize(('block', 'old'), _RETIRED, ids=[f'{b}.{o}' for b, o in _RETIRED])
def test_a_retired_name_is_absent_from_its_block_inventory(block: str, old: str) -> None:
    """The validator's inventory holds the current name and drops the retired one."""
    new = _EXPECTED_RENAMES[block][old]
    assert old not in _INVENTORIES[block]
    assert new in _INVENTORIES[block]


def test_every_retired_key_in_one_mapping_is_reported() -> None:
    """Two retired keys in one orbit are both named, in sorted order."""
    scene = _full_scene()
    orbit = _block(scene, 'ring_system.features.orbit')
    orbit['a'] = orbit.pop('a_px')
    orbit['long_peri'] = orbit.pop('long_peri_deg')
    with pytest.raises(
        SimSceneValidationError, match=r"'a' was renamed; use 'a_px' .*; 'long_peri' was renamed"
    ):
        validate_sim_params(scene)


def test_ring_key_description_reads_the_rename_table() -> None:
    """A current key's description is the one recorded beside its rename."""
    assert ring_key_description('ring_system.features.orbit.edge_wave', 'damping_rad') == (
        'the azimuthal e-folding length, in radians of downstream longitude'
    )


def test_ring_key_description_refuses_a_key_that_replaced_nothing() -> None:
    """A key outside the rename table has no recorded description."""
    with pytest.raises(KeyError, match=r'ring_system\.features\.orbit\.modes\.m '):
        ring_key_description('ring_system.features.orbit.modes', 'm')


# ---------------------------------------------------------------------------
# The per-kind shape-key table
# ---------------------------------------------------------------------------

_KIND_KEY_CASES = [(kind, key) for kind in RING_KIND_SHAPE_KEYS for key in RING_SHAPE_KEYS]


def _shape_value(key: str) -> Any:
    """A valid value for shape key ``key``."""
    return 'out' if key == 'side' else 5.0


@pytest.mark.parametrize('kind', list(RING_KIND_SHAPE_KEYS))
def test_a_feature_with_its_required_shape_keys_validates(kind: str) -> None:
    """Each kind validates with exactly its required shape keys."""
    scene = _scene({'geometry': {'opening_deg_obs': 30.0, 'opening_deg_sun': 20.0}})
    scene['ring_system']['features'] = [_feature(kind)]
    assert validate_sim_params(scene) is scene


@pytest.mark.parametrize(
    ('kind', 'key'), _KIND_KEY_CASES, ids=[f'{k}-{s}' for k, s in _KIND_KEY_CASES]
)
def test_the_validator_enforces_the_kind_shape_table(kind: str, key: str) -> None:
    """Required keys must be present, rejected keys must be absent, optional keys may be either."""
    scene = _scene({'geometry': {'opening_deg_obs': 30.0, 'opening_deg_sun': 20.0}})
    feature = _feature(kind)
    scene['ring_system']['features'] = [feature]
    rule = RING_KIND_SHAPE_KEYS[kind].get(key)
    if rule == 'required':
        del feature[key]
        with pytest.raises(
            SimSceneValidationError, match=re.escape(f"{key} is required for kind '{kind}'")
        ):
            validate_sim_params(scene)
    elif rule == 'optional':
        feature[key] = _shape_value(key)
        assert validate_sim_params(scene) is scene
    else:
        feature[key] = _shape_value(key)
        with pytest.raises(
            SimSceneValidationError, match=re.escape(f"{key} is not allowed for kind '{kind}'")
        ):
            validate_sim_params(scene)


def test_every_side_default_belongs_to_a_kind_that_takes_side() -> None:
    """A side default exists exactly for the kinds that take side."""
    takers = {kind for kind, rules in RING_KIND_SHAPE_KEYS.items() if 'side' in rules}
    assert set(RING_SIDE_DEFAULTS) == takers


def test_an_m_mode_takes_m_of_at_least_2() -> None:
    """orbit.modes refuses m = 1: mode 1 is the base ellipse."""
    scene = _full_scene()
    _block(scene, 'ring_system.features.orbit.modes')['m'] = 1
    with pytest.raises(SimSceneValidationError, match=r'modes\[0\]\.m must be an integer >= 2'):
        validate_sim_params(scene)


@pytest.mark.parametrize(('m', 'valid'), [(0, False), (1, True)])
def test_a_brightness_modulation_takes_m_of_at_least_1(m: int, valid: bool) -> None:
    """azimuthal.modulation accepts m = 1 (one bright arc) and refuses m = 0."""
    scene = _full_scene()
    _block(scene, 'ring_system.azimuthal.modulation')['m'] = m
    if valid:
        assert validate_sim_params(scene) is scene
        return
    with pytest.raises(SimSceneValidationError, match=r'modulation\.m must be an integer >= 1'):
        validate_sim_params(scene)
