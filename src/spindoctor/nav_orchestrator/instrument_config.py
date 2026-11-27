"""Per-instrument configuration consumed by the orchestrator.

Each observation carries its camera's block on ``obs.inst_config``
(:class:`~spindoctor.config.sections_camera.CameraBlock`, set when the observation is
read from its file).  This module gathers the values the orchestrator needs from it:
the units, the saturation level, the missing-data marker, the image-quality thresholds,
the rotation settings, and the filter applied to the image before navigation.

The returned :class:`InstrumentSettings` is a plain dataclass, so the orchestrator can
branch on ``data_units`` without reading the block again.
"""

from dataclasses import dataclass
from typing import Any

from spindoctor.config.sections_camera import CameraBlock, DataUnits, SourceImageFilter
from spindoctor.support.image_class import ImageQualityThresholds, image_quality_thresholds

__all__ = [
    'InstrumentSettings',
    'instrument_settings_from_obs',
]


@dataclass(frozen=True)
class InstrumentSettings:
    """Per-instrument settings the orchestrator needs at navigate time.

    Parameters:
        data_units: ``raw_dn`` for images in raw detector counts, or ``calibrated_if``
            for images in I/F.
        saturation_dn: The saturation level in DN; None for a ``calibrated_if``
            camera, which has no saturation mask.
        marker_value: The value a missing pixel carries.
        thresholds: The image-quality thresholds for the classifier, in DN for
            ``raw_dn`` and in I/F for ``calibrated_if``.
        fit_camera_rotation: Whether a technique's fit solves for a camera rotation.
        max_rotation_deg: The largest rotation, in degrees, a fit may find.
        source_image_filter: The filter applied to the image before navigation.
    """

    data_units: DataUnits
    saturation_dn: float | None
    marker_value: float
    thresholds: ImageQualityThresholds
    fit_camera_rotation: bool
    max_rotation_deg: float
    source_image_filter: SourceImageFilter


def instrument_settings_from_obs(obs: Any) -> InstrumentSettings:
    """Build the orchestrator's per-instrument settings from an observation.

    Parameters:
        obs: An observation, or a stand-in carrying the ``inst_config`` attribute.

    Returns:
        The settings.
    """
    camera: CameraBlock = obs.inst_config
    # A calibrated_if camera has no saturation mask: the same physical full-well DN
    # maps to a different I/F for every combination of exposure time, filter, and gain,
    # so its block leaves noise.saturation_dn out.  The orchestrator emits an empty
    # saturation mask (see ``_build_saturation_mask``) and the thresholds carry an
    # infinite saturation threshold, so the ``fully_overexposed`` early-out cannot fire.
    return InstrumentSettings(
        data_units=camera.data_units,
        saturation_dn=camera.noise.saturation_dn,
        marker_value=camera.noise.marker(),
        thresholds=image_quality_thresholds(camera),
        fit_camera_rotation=camera.fit_camera_rotation,
        max_rotation_deg=camera.max_rotation_deg,
        source_image_filter=camera.source_image_filter,
    )
