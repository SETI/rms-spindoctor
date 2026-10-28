"""Tests for ``spindoctor.config.visible_limb``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spindoctor.config import Config
from spindoctor.config.visible_limb import (
    VisibleLimbModel,
    validate_visible_limb_config,
    visible_limb_model_for,
)

_ENTRY: dict[str, Any] = {
    'altitude_km': [[0.0, 50.0], [60.0, 150.0], [90.0, 200.0]],
    'sigma_km': 80.0,
    'band_offsets_km': {'MT2': 90.0, 'mt3': 210.0},
    'km_per_px_range': [60.0, 260.0],
    'phase_deg_range': [10.0, 120.0],
}


@pytest.mark.parametrize(
    ('phase_deg', 'expected_km'),
    [(-5.0, 50.0), (0.0, 50.0), (30.0, 100.0), (90.0, 200.0), (120.0, 200.0)],
    ids=['below_table', 'first_knot', 'interpolated', 'last_knot', 'above_table'],
)
def test_altitude_interpolates_in_phase(phase_deg: float, expected_km: float) -> None:
    """The altitude is linear between knots and held at the ends.

    Parameters:
        phase_deg: Phase angle.
        expected_km: Expected altitude.
    """
    model = VisibleLimbModel.from_mapping('SATURN', _ENTRY)
    assert model.altitude_for_image_km(phase_deg, ['CL1', 'CL2']) == pytest.approx(expected_km)


def test_altitude_adds_the_largest_band_offset() -> None:
    """An image taken through absorption-band filters gets the largest listed offset."""
    model = VisibleLimbModel.from_mapping('SATURN', _ENTRY)
    assert model.altitude_for_image_km(0.0, ['mt2', 'MT3']) == pytest.approx(260.0)


def test_sigma_loads() -> None:
    """The altitude uncertainty comes through from the entry."""
    assert VisibleLimbModel.from_mapping('SATURN', _ENTRY).sigma_km == 80.0


@pytest.mark.parametrize(
    ('km_per_px', 'phase_deg', 'applies'),
    [
        (59.9, 50.0, False),
        (60.0, 50.0, True),
        (260.0, 50.0, True),
        (260.1, 50.0, False),
        (100.0, 9.9, False),
        (100.0, 10.0, True),
        (100.0, 120.0, True),
        (100.0, 120.1, False),
    ],
    ids=[
        'scale_below',
        'scale_low_end',
        'scale_high_end',
        'scale_above',
        'phase_below',
        'phase_low_end',
        'phase_high_end',
        'phase_above',
    ],
)
def test_model_applies_over_its_measured_ranges(
    km_per_px: float, phase_deg: float, applies: bool
) -> None:
    """The model applies only over the scales and phases it was measured on.

    Parameters:
        km_per_px: Limb scale.
        phase_deg: Phase angle.
        applies: Whether the model applies there.
    """
    model = VisibleLimbModel.from_mapping('SATURN', _ENTRY)
    assert model.applies_at(km_per_px, phase_deg) is applies


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'altitude_km': []}, 'at least one pair'),
        ({'altitude_km': 5.0}, 'list of pairs'),
        ({'altitude_km': [[0.0, 1.0, 2.0]]}, 'altitude_km must be two numbers'),
        ({'altitude_km': [[0.0, float('nan')]]}, 'altitude_km values must be finite'),
        ({'altitude_km': [[10.0, 1.0], [10.0, 2.0]]}, 'strictly increase'),
        ({'sigma_km': -1.0}, 'must be finite and >= 0'),
        ({'sigma_km': 'wide'}, 'sigma_km must be a number'),
        ({'band_offsets_km': [90.0]}, 'map filter names'),
        ({'band_offsets_km': {'MT2': float('inf')}}, 'band_offsets_km must be finite'),
        ({'km_per_px_range': [260.0, 60.0]}, '0 < min < max'),
        ({'km_per_px_range': 60.0}, 'km_per_px_range must be two numbers'),
        ({'phase_deg_range': [120.0, 10.0]}, 'phase_deg_range must satisfy min < max'),
    ],
    ids=[
        'empty',
        'scalar_table',
        'triple',
        'nan',
        'repeated_phase',
        'sigma_neg',
        'sigma_text',
        'offsets_list',
        'offset_inf',
        'range_order',
        'range_scalar',
        'phase_order',
    ],
)
def test_entries_are_validated(change: dict[str, Any], message: str) -> None:
    """A malformed entry fails with ValueError naming the field.

    Parameters:
        change: Fields replaced in a valid entry.
        message: Expected message fragment.
    """
    with pytest.raises(ValueError, match=message):
        VisibleLimbModel.from_mapping('SATURN', {**_ENTRY, **change})


@pytest.mark.parametrize('key', ['altitude_km', 'sigma_km', 'km_per_px_range', 'phase_deg_range'])
def test_required_keys(key: str) -> None:
    """Each required key's absence fails with ValueError naming it.

    Parameters:
        key: The key left out.
    """
    entry = {k: v for k, v in _ENTRY.items() if k != key}
    with pytest.raises(ValueError, match=f'{key} is required'):
        VisibleLimbModel.from_mapping('SATURN', entry)


def test_entry_must_be_a_mapping() -> None:
    """A non-mapping entry fails with ValueError."""
    with pytest.raises(ValueError, match='must be a mapping'):
        VisibleLimbModel.from_mapping('SATURN', [1, 2])


def test_model_for_reads_the_instrument_entry() -> None:
    """The instrument's entry for the body (any case) is the body's model."""
    model = visible_limb_model_for({'visible_limb_altitude': {'SATURN': _ENTRY}}, 'saturn')
    assert model == VisibleLimbModel.from_mapping('SATURN', _ENTRY)


@pytest.mark.parametrize(
    'inst_config',
    [None, {}, {'visible_limb_altitude': {'JUPITER': _ENTRY}}, {'visible_limb_altitude': None}],
    ids=['no_config', 'no_entry', 'other_body', 'null_entry'],
)
def test_model_for_is_none_without_an_entry(inst_config: dict[str, Any] | None) -> None:
    """Without an entry for the body there is no model.

    Parameters:
        inst_config: The instrument configuration.
    """
    assert visible_limb_model_for(inst_config, 'SATURN') is None


def test_validation_names_the_camera() -> None:
    """A malformed entry anywhere in the configuration fails, naming section and camera."""
    config = {'inst_x': {'cam': {'visible_limb_altitude': {'SATURN': {**_ENTRY, 'sigma_km': -1}}}}}
    with pytest.raises(ValueError, match=r'inst_x\.cam\.visible_limb_altitude: SATURN'):
        validate_visible_limb_config(config)


def test_validation_rejects_a_non_mapping_block() -> None:
    """A visible-limb block that is not a mapping fails."""
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_visible_limb_config({'inst_x': {'cam': {'visible_limb_altitude': [1]}}})


def test_validation_accepts_the_packaged_configuration() -> None:
    """The packaged configuration's entries are valid and Cassini ISS has Saturn's."""
    config = Config()
    config.read_config()
    model = visible_limb_model_for(config.category('cassini_iss_calib')['wac'], 'SATURN')
    assert model is not None


def test_a_malformed_override_fails_when_loaded(tmp_path: Path) -> None:
    """A user override with a malformed entry is refused when it is loaded."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        'cassini_iss:\n  wac:\n    visible_limb_altitude:\n      SATURN:\n        sigma_km: -5\n'
    )
    config = Config()
    config.read_config()
    with pytest.raises(ValueError, match='sigma_km must be finite'):
        config.update_config(override)
