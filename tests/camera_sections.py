"""The packaged camera blocks, for tests that need a camera's values.

A camera block has no defaults, since every value lives in the packaged YAML.  A test
that needs a camera takes a packaged block from here and changes only the fields it is
about, so it never restates a value.
"""

import dataclasses
from typing import Any

from spindoctor.config.sections_camera import CameraBlock
from spindoctor.support.image_class import ImageQualityThresholds, image_quality_thresholds
from tests.packaged_sections import packaged_sections

# The sections of the packaged real cameras and the cameras inside them; None for a
# section that is one camera's block.
PACKAGED_CAMERAS: tuple[tuple[str, str | None], ...] = (
    ('cassini_iss', 'nac'),
    ('cassini_iss', 'wac'),
    ('cassini_iss_calib', 'nac'),
    ('cassini_iss_calib', 'wac'),
    ('galileo_ssi', None),
    ('newhorizons_lorri', None),
    ('voyager_iss', 'vgiss_1_nac'),
    ('voyager_iss', 'vgiss_1_wac'),
    ('voyager_iss', 'vgiss_2_nac'),
    ('voyager_iss', 'vgiss_2_wac'),
)


def packaged_camera(section: str, camera: str | None = None, **changes: Any) -> CameraBlock:
    """Return a packaged camera block, with the given fields changed.

    Parameters:
        section: The camera section.
        camera: The camera inside a section that holds several; None for a section that
            is one camera's block.
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    block = packaged_sections().camera_block(section, camera)
    return dataclasses.replace(block, **changes)


def raw_camera(**changes: Any) -> CameraBlock:
    """Return the packaged Cassini ISS narrow angle camera's block, which reads raw DN.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    return packaged_camera('cassini_iss', 'nac', **changes)


def calibrated_camera(**changes: Any) -> CameraBlock:
    """Return the packaged calibrated Cassini ISS narrow angle camera's block, in I/F.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The camera block.
    """
    return packaged_camera('cassini_iss_calib', 'nac', **changes)


def camera_mapping(block: CameraBlock) -> dict[str, Any]:
    """Return a camera block as the mapping a file gives, with the keys it leaves out omitted.

    Parameters:
        block: The camera block.

    Returns:
        The mapping.
    """

    def strip(value: Any) -> Any:
        """Drop the keys whose value is None from every mapping inside ``value``."""
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if item is not None}
        return value

    mapping: dict[str, Any] = strip(dataclasses.asdict(block))
    return mapping


def raw_thresholds(**changes: Any) -> ImageQualityThresholds:
    """Return the image-quality thresholds of :func:`raw_camera`, with fields changed.

    Parameters:
        **changes: Fields to change, by name.

    Returns:
        The thresholds.
    """
    return dataclasses.replace(image_quality_thresholds(raw_camera()), **changes)
