"""Tests for ``spindoctor.nav_model.body_shape``."""

from __future__ import annotations

import dataclasses

import pytest

from spindoctor.config import Config
from spindoctor.config.sections_body_shape import BodyShapeEntry
from spindoctor.nav_model.body_shape import (
    BODY_SHAPE_TABLE,
    DEFAULT_BODY_SHAPE,
    BodyShape,
    load_body_shape,
)


def _config_with(body_shape: dict[str, BodyShapeEntry]) -> Config:
    """Return a configuration whose body-shape table is ``body_shape``.

    Parameters:
        body_shape: The table, keyed by SPICE body name.

    Returns:
        A configuration with every other section packaged.
    """
    config = Config()
    config._nav_config = dataclasses.replace(config.nav_config, body_shape=body_shape)
    return config


# ---------------------------------------------------------------------------
# Hard-coded fallback table (no configured entry)
# ---------------------------------------------------------------------------


def test_lookup_known_body_no_entry_returns_table_entry() -> None:
    """A known body without a configured entry returns the hard-coded profile."""
    config = _config_with({})
    assert load_body_shape('MIMAS', config=config) == BODY_SHAPE_TABLE['MIMAS']


def test_lookup_case_insensitive() -> None:
    """Body-name lookup ignores case."""
    config = _config_with({})
    assert load_body_shape('mimas', config=config) == BODY_SHAPE_TABLE['MIMAS']


def test_lookup_unknown_body_falls_back_to_default() -> None:
    """Bodies absent from the table return ``DEFAULT_BODY_SHAPE``."""
    config = _config_with({})
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
# Configured entries (config_220_body_shape.yaml)
# ---------------------------------------------------------------------------


def test_entry_overrides_only_the_keys_it_sets() -> None:
    """A configured value overwrites the hard-coded baseline for its key only."""
    entry = BodyShapeEntry(crater_scale_km=0.7, albedo_variation=0.04)
    config = _config_with({'MIMAS': entry})
    expected = dataclasses.replace(
        BODY_SHAPE_TABLE['MIMAS'], crater_scale_km=0.7, albedo_variation=0.04
    )
    assert load_body_shape('MIMAS', config=config) == expected


def test_entry_for_a_body_with_no_profile_overrides_the_default() -> None:
    """An entry for a body absent from ``BODY_SHAPE_TABLE`` overrides the default."""
    entry = BodyShapeEntry(ellipsoid_rms_residual_km=0.5, shape_class_hint='regular')
    config = _config_with({'NEW_MOON': entry})
    expected = dataclasses.replace(
        DEFAULT_BODY_SHAPE, ellipsoid_rms_residual_km=0.5, shape_class_hint='regular'
    )
    assert load_body_shape('NEW_MOON', config=config) == expected


def test_entry_sets_the_orbital_residual_and_blob_diameter() -> None:
    """An entry's SPK residual and blob diameter reach the loaded shape."""
    entry = BodyShapeEntry(spice_orbital_residual_km=5.0, min_blob_diameter_px=9.0)
    config = _config_with({'MIMAS': entry})
    expected = dataclasses.replace(
        BODY_SHAPE_TABLE['MIMAS'], spice_orbital_residual_km=5.0, min_blob_diameter_px=9.0
    )
    assert load_body_shape('MIMAS', config=config) == expected
