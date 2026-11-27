"""Tests for ``spindoctor.nav_orchestrator.instrument_config``."""

import math

from tests.camera_sections import calibrated_camera, raw_camera

from spindoctor.config.sections_camera import CameraBlock
from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs


class _FakeObs:
    """Minimal stand-in carrying a camera block."""

    def __init__(self, inst_config: CameraBlock) -> None:
        self.inst_config = inst_config


def test_raw_dn_camera_gives_dn_settings() -> None:
    """A raw_dn camera gives its saturation level and its DN thresholds."""
    camera = raw_camera()
    settings = instrument_settings_from_obs(_FakeObs(camera))
    assert settings.saturation_dn == camera.noise.saturation_dn
    assert settings.thresholds.saturation_threshold_dn == (
        camera.image_quality_thresholds.saturation_threshold_dn
    )
    assert settings.thresholds.blank_max_dn == camera.image_quality_thresholds.blank_max_dn


def test_calibrated_if_camera_has_no_saturation() -> None:
    """A calibrated_if camera gives its I/F thresholds and no saturation level."""
    camera = calibrated_camera()
    settings = instrument_settings_from_obs(_FakeObs(camera))
    assert settings.saturation_dn is None
    # The saturation limit is off for I/F images, so no finite pixel is saturated.
    assert math.isinf(settings.thresholds.saturation_threshold_dn)
    assert settings.thresholds.blank_max_dn == camera.image_quality_thresholds.blank_max_if
