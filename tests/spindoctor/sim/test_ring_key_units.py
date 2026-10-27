"""Ring scene keys name their units, and the per-kind shape-key rules are one table.

Each ring-plane length and each angle in the ``ring_system`` block carries its
unit in its name (``a_px``, ``long_peri_deg``, ``rate_peri_deg_per_day``,
``damping_rad``, and so on), and the schema's ``RING_KEY_DESCRIPTIONS`` says
what each of the keys the sim editor describes holds.

Which radial shape keys a feature kind takes is one table, written out here: the
schema's ``RING_KIND_SHAPE_KEYS`` holds it, the validator enforces exactly it (a
required key's absence fails, a rejected key's presence fails, an optional key
validates either way), and the per-kind tables in the developer guide and the
scene README match it row for row.
"""

import re
from pathlib import Path
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
    RING_KEY_DESCRIPTIONS,
    RING_KIND_SHAPE_KEYS,
    RING_SHAPE_KEYS,
    RING_SIDE_DEFAULTS,
    ring_key_description,
)

_REPO = Path(__file__).resolve().parents[3]
_DEV_GUIDE = _REPO / 'docs' / 'dev_guide' / 'dev_guide_simulator.rst'
_SCENE_README = _REPO / 'tests' / 'integration' / 'sim_scenes' / 'README.txt'

# Each block's length and angle keys, each named with its unit, written out so
# the validator's inventories are checked against the intended names.
_UNIT_KEYS: dict[str, frozenset[str]] = {
    'ring_system.features': frozenset({'width_px', 'wavelength_px', 'damping_px'}),
    'ring_system.features.orbit': frozenset(
        {'a_px', 'ae_px', 'long_peri_deg', 'rate_peri_deg_per_day'}
    ),
    'ring_system.features.orbit.modes': frozenset({'amp_px', 'peri_deg'}),
    'ring_system.features.orbit.edge_wave': frozenset(
        {'amp_px', 'wavelength_px', 'damping_rad', 'lam0_deg'}
    ),
    'ring_system.azimuthal.modulation': frozenset({'lam_peak_deg'}),
    'ring_system.azimuthal.spokes': frozenset({'r_inner_px', 'r_outer_px', 'width_deg'}),
    'ring_system.moonlets': frozenset({'a_px', 'lam_deg', 'radius_px'}),
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

# Where each block sits in _full_scene().
_BLOCK_PATHS: dict[str, tuple[str | int, ...]] = {
    'ring_system.features': ('features', 0),
    'ring_system.features.orbit': ('features', 0, 'orbit'),
    'ring_system.features.orbit.modes': ('features', 0, 'orbit', 'modes', 0),
    'ring_system.features.orbit.edge_wave': ('features', 0, 'orbit', 'edge_wave'),
    'ring_system.azimuthal.modulation': ('azimuthal', 'modulation'),
    'ring_system.azimuthal.spokes': ('azimuthal', 'spokes'),
    'ring_system.moonlets': ('moonlets', 0),
}

# The per-kind shape-key table, written out so the validator is checked against
# the intended rules and not against the schema constant it reads.
_EXPECTED_KIND_TABLE: dict[str, dict[str, str]] = {
    'ringlet': {
        'width_px': 'required',
        'side': 'rejected',
        'wavelength_px': 'rejected',
        'damping_px': 'rejected',
    },
    'gap': {
        'width_px': 'required',
        'side': 'rejected',
        'wavelength_px': 'rejected',
        'damping_px': 'rejected',
    },
    'edge': {
        'width_px': 'rejected',
        'side': 'optional',
        'wavelength_px': 'rejected',
        'damping_px': 'rejected',
    },
    'ramp': {
        'width_px': 'required',
        'side': 'optional',
        'wavelength_px': 'rejected',
        'damping_px': 'rejected',
    },
    'wave': {
        'width_px': 'rejected',
        'side': 'rejected',
        'wavelength_px': 'required',
        'damping_px': 'required',
    },
}
_EXPECTED_SIDE_DEFAULTS: dict[str, str] = {'edge': 'in', 'ramp': 'out'}


def _feature(kind: str) -> dict[str, Any]:
    """A feature of ``kind`` carrying exactly the shape keys the kind requires."""
    values = {'width_px': 8.0, 'wavelength_px': 6.0, 'damping_px': 12.0}
    feature: dict[str, Any] = {
        'name': 'F1',
        'kind': kind,
        'tau': 1.0,
        'orbit': {'a_px': 40.0},
    }
    for key, rule in _EXPECTED_KIND_TABLE[kind].items():
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
    """A valid scene whose ring system carries every block in :data:`_BLOCK_PATHS`."""
    feature = _feature('ringlet')
    feature['orbit'] = {
        'a_px': 40.0,
        'ae_px': 1.0,
        'long_peri_deg': 10.0,
        'rate_peri_deg_per_day': 0.5,
        'modes': [{'m': 2, 'amp_px': 1.5, 'peri_deg': 30.0}],
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
    for step in _BLOCK_PATHS[block]:
        node = node[step]
    assert isinstance(node, dict)
    return node


def test_the_full_scene_validates() -> None:
    """The scene the block tests start from is valid."""
    scene = _full_scene()
    assert validate_sim_params(scene) is scene


def test_an_m_mode_refuses_the_mode_1_longitude_key() -> None:
    """A mode's phase is peri_deg; mode 1's long_peri_deg is not a mode key.

    The two longitudes differ: only mode 1's precesses, takes the planted
    delta_long_peri_deg, and has a declared sigma.
    """
    scene = _full_scene()
    mode = _block(scene, 'ring_system.features.orbit.modes')
    mode['long_peri_deg'] = mode.pop('peri_deg')
    with pytest.raises(
        SimSceneValidationError, match=re.escape("modes[0]: unknown keys: ['long_peri_deg']")
    ):
        validate_sim_params(scene)


@pytest.mark.parametrize('block', list(_UNIT_KEYS))
def test_each_length_and_angle_key_names_its_unit(block: str) -> None:
    """The validator's inventory for each block holds each unit-named length and angle key."""
    assert _UNIT_KEYS[block] - _INVENTORIES[block] == set()


def test_ring_key_description_reads_the_description_table() -> None:
    """A key's description is the one the table records for it."""
    assert ring_key_description('ring_system.features.orbit.edge_wave', 'damping_rad') == (
        'the azimuthal e-folding length, in radians of downstream longitude'
    )


def test_ring_key_description_refuses_an_undescribed_key() -> None:
    """A key outside the description table raises KeyError naming it."""
    with pytest.raises(KeyError, match=r'ring_system\.features\.orbit\.modes\.m '):
        ring_key_description('ring_system.features.orbit.modes', 'm')


@pytest.mark.parametrize('block', list(RING_KEY_DESCRIPTIONS))
def test_every_described_key_is_in_its_block_inventory(block: str) -> None:
    """The description table describes only keys the validator accepts in that block."""
    assert set(RING_KEY_DESCRIPTIONS[block]) - _INVENTORIES[block] == set()


# ---------------------------------------------------------------------------
# The per-kind shape-key table
# ---------------------------------------------------------------------------

_KIND_KEY_CASES = [(kind, key) for kind, rules in _EXPECTED_KIND_TABLE.items() for key in rules]


def _shape_value(key: str) -> Any:
    """A valid value for shape key ``key``."""
    return 'out' if key == 'side' else 5.0


def test_the_schema_kind_table_is_the_expected_table() -> None:
    """RING_KIND_SHAPE_KEYS lists each kind's required and optional keys, and nothing else."""
    expected = {
        kind: {key: rule for key, rule in rules.items() if rule != 'rejected'}
        for kind, rules in _EXPECTED_KIND_TABLE.items()
    }
    assert expected == RING_KIND_SHAPE_KEYS


def test_the_shape_keys_are_the_table_columns() -> None:
    """RING_SHAPE_KEYS names exactly the keys the expected table rules on."""
    assert set(RING_SHAPE_KEYS) == set(_EXPECTED_KIND_TABLE['ringlet'])


def test_the_side_defaults_are_the_expected_defaults() -> None:
    """An edge defaults to side 'in' and a ramp to side 'out'."""
    assert _EXPECTED_SIDE_DEFAULTS == RING_SIDE_DEFAULTS


@pytest.mark.parametrize('kind', list(_EXPECTED_KIND_TABLE))
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
    rule = _EXPECTED_KIND_TABLE[kind][key]
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


def _expected_table_rows() -> list[list[str]]:
    """The per-kind table's rows as whitespace-separated tokens."""
    kinds = list(_EXPECTED_KIND_TABLE)
    rows = [['key', *kinds]]
    for key in _EXPECTED_KIND_TABLE['ringlet']:
        rows.append([key, *(_EXPECTED_KIND_TABLE[kind][key] for kind in kinds)])
    rows.append(['default', 'side', *(_EXPECTED_SIDE_DEFAULTS.get(kind, 'n/a') for kind in kinds)])
    return rows


@pytest.mark.parametrize('path', [_DEV_GUIDE, _SCENE_README], ids=['dev_guide', 'scene_readme'])
def test_the_documented_kind_table_is_the_expected_table(path: Path) -> None:
    """The developer guide and the scene README carry the expected per-kind table."""
    lines = [line.replace('``', '').split() for line in path.read_text().splitlines()]
    for row in _expected_table_rows():
        assert row in lines, f'{path.name} is missing the per-kind table row {row}'


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
