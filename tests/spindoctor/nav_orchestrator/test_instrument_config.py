"""Tests for ``spindoctor.nav_orchestrator.instrument_config``."""

from __future__ import annotations

import math
from typing import Any

import pytest

from spindoctor.nav_orchestrator.image_classifier import ImageQualityThresholds
from spindoctor.nav_orchestrator.instrument_config import (
    InstrumentSettings,
    instrument_settings_from_obs,
)


class _FakeObs:
    """Minimal stand-in carrying an ``inst_config`` dict."""

    def __init__(self, inst_config: dict[str, Any] | None) -> None:
        self.inst_config = inst_config


def _raw_dn_block() -> dict[str, Any]:
    return {
        'data_units': 'raw_dn',
        'noise': {
            'saturation_dn': 4095.0,
            'full_well_dn': 4095.0,
            'expected_noise_dn': 4.0,
            'marker_value': 0,
            'read_noise_dn': 4.0,
        },
        'image_quality_thresholds': {
            'blank_max_dn': 5.0,
            'saturation_threshold_dn': 4095.0,
            'noisy_threshold_dn': 10.0,
            'max_missing_frac_clean': 0.30,
            'max_overexposed_frac_clean': 0.80,
        },
        'fit_camera_rotation': False,
        'max_rotation_deg': 5.0,
    }


def test_no_inst_config_returns_legacy_defaults() -> None:
    """An obs without inst_config yields legacy raw_dn defaults and no saturation_dn."""
    settings = instrument_settings_from_obs(_FakeObs(None))
    assert isinstance(settings, InstrumentSettings)
    assert settings.data_units == 'raw_dn'
    assert settings.saturation_dn is None
    assert isinstance(settings.thresholds, ImageQualityThresholds)
    assert settings.fit_camera_rotation is False


def test_raw_dn_block_loads_thresholds() -> None:
    """A populated raw_dn inst_config produces the expected thresholds."""
    obs = _FakeObs(_raw_dn_block())
    settings = instrument_settings_from_obs(obs)
    assert settings.data_units == 'raw_dn'
    assert settings.saturation_dn == 4095.0
    assert settings.thresholds.blank_max_dn == 5.0
    assert settings.thresholds.saturation_threshold_dn == 4095.0
    assert settings.thresholds.noisy_threshold == 10.0
    assert settings.thresholds.max_saturation_frac_clean == 0.80
    assert settings.thresholds.max_missing_frac_clean == 0.30


def test_raw_dn_ignores_stray_signal_scale_key() -> None:
    """A stray ``signal_dn_to_image_unit_scale`` key is silently ignored.

    The field was removed from ``InstrumentSettings`` when the star gate
    moved to a magnitude-based limit, so the loader no longer reads or
    validates it; a leftover key must not break loading.
    """
    block = _raw_dn_block()
    block['noise']['signal_dn_to_image_unit_scale'] = 0.5
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert settings.data_units == 'raw_dn'
    assert not hasattr(settings, 'signal_dn_to_image_unit_scale')


def test_calibrated_if_block_uses_if_thresholds() -> None:
    """calibrated_if reads I/F thresholds and disables the saturation gate."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {
            'marker_value': 'NaN',
        },
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'noisy_threshold_if': 0.005,
            'max_missing_frac_clean': 0.05,
            'max_overexposed_frac_clean': 0.80,
        },
        'fit_camera_rotation': False,
        'max_rotation_deg': 5.0,
    }
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert settings.data_units == 'calibrated_if'
    assert settings.saturation_dn is None
    assert settings.thresholds.blank_max_dn == 1.0e-3
    # Per Phase 10 §F: the saturation gate is intentionally off for
    # calibrated_if, so the classifier sees an inf threshold and
    # ``saturation_frac`` stays 0.0 regardless of pixel values.
    assert math.isinf(settings.thresholds.saturation_threshold_dn)
    assert settings.thresholds.noisy_threshold == 0.005


def test_calibrated_if_does_not_require_signal_scale() -> None:
    """calibrated_if loads cleanly without a DN-to-image-unit scale.

    The star gate is magnitude based now, so ``calibrated_if`` no longer
    needs (or carries) ``signal_dn_to_image_unit_scale``.
    """
    block = {
        'data_units': 'calibrated_if',
        'noise': {'marker_value': 'NaN'},
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'noisy_threshold_if': 0.005,
        },
    }
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert settings.data_units == 'calibrated_if'
    assert not hasattr(settings, 'signal_dn_to_image_unit_scale')


def test_calibrated_if_rejects_explicit_saturation_threshold() -> None:
    """A stale ``saturation_threshold_if`` field fails fast at load time."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {
            'marker_value': 'NaN',
        },
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'saturation_threshold_if': 10.0,
            'noisy_threshold_if': 0.005,
        },
    }
    with pytest.raises(ValueError, match='saturation_threshold_if'):
        instrument_settings_from_obs(_FakeObs(block))


def test_missing_data_units_raises() -> None:
    """An inst_config without data_units fails fast."""
    block = _raw_dn_block()
    del block['data_units']
    with pytest.raises(ValueError, match='data_units'):
        instrument_settings_from_obs(_FakeObs(block))


def test_unknown_data_units_raises() -> None:
    """A bogus data_units value fails fast."""
    block = _raw_dn_block()
    block['data_units'] = 'reflectance'
    with pytest.raises(ValueError, match='data_units'):
        instrument_settings_from_obs(_FakeObs(block))


def test_raw_dn_missing_noise_block_raises() -> None:
    """A raw_dn instrument without a noise block fails fast."""
    block = _raw_dn_block()
    del block['noise']
    with pytest.raises(ValueError, match='noise block'):
        instrument_settings_from_obs(_FakeObs(block))


def test_raw_dn_missing_saturation_dn_raises() -> None:
    """A raw_dn instrument without saturation_dn fails fast."""
    block = _raw_dn_block()
    del block['noise']['saturation_dn']
    with pytest.raises(ValueError, match='saturation_dn'):
        instrument_settings_from_obs(_FakeObs(block))


def test_max_rotation_must_be_positive() -> None:
    """A non-positive max_rotation_deg fails fast."""
    block = _raw_dn_block()
    block['max_rotation_deg'] = -1.0
    with pytest.raises(ValueError, match='max_rotation_deg'):
        instrument_settings_from_obs(_FakeObs(block))


def test_calibrated_if_default_marker_is_nan() -> None:
    """A calibrated_if instrument with an unset marker_value defaults to NaN."""
    block = {
        'data_units': 'calibrated_if',
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'noisy_threshold_if': 0.005,
        },
        # An empty noise block still defaults the marker to NaN.
        'noise': {},
    }
    settings = instrument_settings_from_obs(_FakeObs(block))

    assert math.isnan(settings.marker_value)


def test_shipped_inst_configs_load_cleanly() -> None:
    """Every shipped per-instrument config loads through the helper."""
    from spindoctor.config import Config

    config = Config()
    config.read_config()
    blocks = [
        config.category('cassini_iss').get('nac'),
        config.category('cassini_iss').get('wac'),
        config.category('newhorizons_lorri'),
        config.category('galileo_ssi'),
    ]
    for block in blocks:
        settings = instrument_settings_from_obs(_FakeObs(block))
        assert settings.data_units == 'raw_dn'
        assert settings.saturation_dn is not None
        assert settings.saturation_dn > 0.0


def test_shipped_calibrated_if_configs_load_cleanly() -> None:
    """Every shipped calibrated-IF block loads and disables the saturation gate."""
    from spindoctor.config import Config

    config = Config()
    config.read_config()
    blocks = [
        config.category('cassini_iss_calib').get('nac'),
        config.category('cassini_iss_calib').get('wac'),
        config.category('voyager_iss'),
    ]
    for block in blocks:
        settings = instrument_settings_from_obs(_FakeObs(block))
        assert settings.data_units == 'calibrated_if'
        assert settings.saturation_dn is None
        # Saturation gate disabled for calibrated_if (Phase 10 §F).
        assert math.isinf(settings.thresholds.saturation_threshold_dn)


def test_model_error_floors_absent_is_empty() -> None:
    """An instrument that declares no model_error_floor_px block has no camera floors."""
    settings = instrument_settings_from_obs(_FakeObs(_raw_dn_block()))
    assert dict(settings.model_error_floor_px) == {}


def test_model_error_floors_are_read() -> None:
    """A declared per-technique floor reaches the settings unchanged."""
    block = _raw_dn_block()
    block['model_error_floor_px'] = {'RingAnnulusNav': 0.9}
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert dict(settings.model_error_floor_px) == {'RingAnnulusNav': 0.9}


@pytest.mark.parametrize(
    ('floors', 'error', 'message'),
    [
        ({'RingAnulusNav': 0.9}, ValueError, 'does not read a per-camera floor'),
        ({'RingAnnulusNav': -0.1}, ValueError, 'must be >= 0'),
        ({'RingAnnulusNav': float('nan')}, ValueError, 'must be a finite numeric value'),
        ([0.9], TypeError, 'model_error_floor_px must be a mapping'),
    ],
    ids=['unknown-technique', 'negative', 'nan', 'not-a-mapping'],
)
def test_model_error_floors_reject_invalid_blocks(
    floors: Any, error: type[Exception], message: str
) -> None:
    """A misspelled technique, a negative or NaN floor, or a non-mapping block fails to load."""
    block = _raw_dn_block()
    block['model_error_floor_px'] = floors
    with pytest.raises(error, match=message):
        instrument_settings_from_obs(_FakeObs(block))


def test_shipped_cassini_wac_declares_annulus_floor() -> None:
    """Both Cassini WAC blocks carry the calibrated annulus floor; the NAC keeps the tuning's."""
    from spindoctor.config import Config

    config = Config()
    config.read_config()
    floors = {
        (section, camera): dict(
            instrument_settings_from_obs(
                _FakeObs(config.category(section).get(camera))
            ).model_error_floor_px
        )
        for section in ('cassini_iss', 'cassini_iss_calib')
        for camera in ('nac', 'wac')
    }
    assert floors == {
        ('cassini_iss', 'nac'): {},
        ('cassini_iss', 'wac'): {'RingAnnulusNav': 0.9},
        ('cassini_iss_calib', 'nac'): {},
        ('cassini_iss_calib', 'wac'): {'RingAnnulusNav': 0.9},
    }
