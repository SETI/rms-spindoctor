"""Tests for ``spindoctor.nav_model.body_shape``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from spindoctor.nav_model.body_shape import (
    BODY_SHAPE_TABLE,
    DEFAULT_BODY_SHAPE,
    BodyShape,
    load_body_shape,
)


class _FakeConfig:
    """Stand-in exposing only the ``body_shape`` attribute the loader reads."""

    def __init__(self, body_shape: dict[str, Any] | None) -> None:
        self.body_shape = body_shape if body_shape is not None else {}


# ---------------------------------------------------------------------------
# Hard-coded fallback table (no YAML override)
# ---------------------------------------------------------------------------


def test_lookup_known_body_no_yaml_returns_table_entry() -> None:
    """A known body with empty YAML returns the hard-coded profile."""
    config = _FakeConfig({})
    assert load_body_shape('MIMAS', config=config) == BODY_SHAPE_TABLE['MIMAS']


def test_lookup_case_insensitive() -> None:
    """Body-name lookup ignores case."""
    config = _FakeConfig({})
    assert load_body_shape('mimas', config=config) == BODY_SHAPE_TABLE['MIMAS']


def test_lookup_unknown_body_falls_back_to_default() -> None:
    """Bodies absent from the table return ``DEFAULT_BODY_SHAPE``."""
    config = _FakeConfig({})
    assert load_body_shape('NEW MOON', config=config) == DEFAULT_BODY_SHAPE


def test_default_body_shape_values() -> None:
    """The default body shape has its documented numeric defaults."""
    assert DEFAULT_BODY_SHAPE.ellipsoid_rms_residual_km == 2.0
    assert DEFAULT_BODY_SHAPE.crater_scale_km == 5.0
    assert DEFAULT_BODY_SHAPE.albedo_variation == 0.15
    assert DEFAULT_BODY_SHAPE.spice_orbital_residual_km == 2.0
    assert DEFAULT_BODY_SHAPE.min_blob_diameter_px == 5.0
    assert DEFAULT_BODY_SHAPE.shape_class_hint == 'unknown'


def test_body_shape_dataclass_is_frozen() -> None:
    """``BodyShape`` is frozen — assignment raises ``FrozenInstanceError``."""
    shape = BodyShape(
        ellipsoid_rms_residual_km=1.0,
        crater_scale_km=2.0,
        albedo_variation=0.1,
        spice_orbital_residual_km=0.5,
    )
    with pytest.raises(dataclasses.FrozenInstanceError, match='ellipsoid_rms_residual_km'):
        shape.ellipsoid_rms_residual_km = 3.0  # type: ignore[misc]


def test_known_saturn_moons_share_shape_profile() -> None:
    """All major Saturn moons share the dedicated saturn-moon profile."""
    profile = BODY_SHAPE_TABLE['MIMAS']
    for body in ('MIMAS', 'ENCELADUS', 'TETHYS', 'DIONE', 'RHEA', 'IAPETUS', 'TITAN'):
        assert BODY_SHAPE_TABLE[body] is profile
    assert profile.shape_class_hint == 'regular'


def test_irregular_bodies_use_irregular_profile() -> None:
    """Hyperion and Phoebe share the irregular-shape profile."""
    assert BODY_SHAPE_TABLE['HYPERION'] is BODY_SHAPE_TABLE['PHOEBE']
    assert BODY_SHAPE_TABLE['HYPERION'].ellipsoid_rms_residual_km == 10.0
    assert BODY_SHAPE_TABLE['HYPERION'].shape_class_hint == 'highly_irregular'


def test_gas_giants_share_shape_profile() -> None:
    """All four giants share the gas-giant profile."""
    profile = BODY_SHAPE_TABLE['SATURN']
    for body in ('SATURN', 'JUPITER', 'URANUS', 'NEPTUNE'):
        assert BODY_SHAPE_TABLE[body] is profile
    assert profile.shape_class_hint == 'regular'


# ---------------------------------------------------------------------------
# YAML overrides (config_220_body_shape.yaml)
# ---------------------------------------------------------------------------


def test_yaml_overrides_individual_field() -> None:
    """A YAML value overwrites the hard-coded baseline for that field."""
    config = _FakeConfig({'MIMAS': {'ellipsoid_rms_residual_km': 0.42}})
    shape = load_body_shape('MIMAS', config=config)
    assert shape.ellipsoid_rms_residual_km == 0.42
    # Other fields untouched — they stay at the hard-coded SATURN-moon profile.
    assert shape.crater_scale_km == BODY_SHAPE_TABLE['MIMAS'].crater_scale_km
    assert shape.albedo_variation == BODY_SHAPE_TABLE['MIMAS'].albedo_variation


def test_yaml_supplies_shape_class_hint_only() -> None:
    """A YAML that only sets shape_class_hint still picks up that field."""
    config = _FakeConfig({'MIMAS': {'shape_class_hint': 'highly_irregular'}})
    shape = load_body_shape('MIMAS', config=config)
    assert shape.shape_class_hint == 'highly_irregular'
    assert shape.ellipsoid_rms_residual_km == BODY_SHAPE_TABLE['MIMAS'].ellipsoid_rms_residual_km


def test_yaml_overrides_unknown_body_against_default() -> None:
    """A YAML entry for a body absent from BODY_SHAPE_TABLE overrides DEFAULT."""
    config = _FakeConfig(
        {
            'NEW_MOON': {
                'ellipsoid_rms_residual_km': 0.5,
                'shape_class_hint': 'regular',
            }
        }
    )
    shape = load_body_shape('NEW_MOON', config=config)
    assert shape.ellipsoid_rms_residual_km == 0.5
    assert shape.shape_class_hint == 'regular'
    # All other fields fall back to DEFAULT_BODY_SHAPE values.
    assert shape.crater_scale_km == DEFAULT_BODY_SHAPE.crater_scale_km
    assert shape.spice_orbital_residual_km == DEFAULT_BODY_SHAPE.spice_orbital_residual_km


def test_yaml_unknown_field_is_ignored() -> None:
    """YAML entries with extra keys (e.g. ``radii_km``) do not break the loader."""
    config = _FakeConfig(
        {
            'MIMAS': {
                'radii_km': [200.0, 195.0, 191.0],
                'albedo_mean': 0.96,
                'shape_class_hint': 'regular',
            }
        }
    )
    shape = load_body_shape('MIMAS', config=config)
    assert shape == BODY_SHAPE_TABLE['MIMAS']


def test_load_body_shape_picks_up_yaml_override() -> None:
    """``load_body_shape`` merges the operator-curated YAML over the baseline."""
    config = _FakeConfig({'MIMAS': {'crater_scale_km': 0.7}})
    assert load_body_shape('MIMAS', config=config).crater_scale_km == 0.7
