"""Per-instrument configuration consumed by the orchestrator.

Each ``ObsSnapshotInst`` carries its already-resolved per-camera config
mapping on ``obs.inst_config`` (set by ``ObsInst.from_file``).  This
module reads the ``data_units``, ``noise``, and
``image_quality_thresholds`` blocks defined in
``config_4N0_inst_*.yaml`` and returns the values the orchestrator
needs.

The returned ``InstrumentSettings`` is a plain dataclass so the
orchestrator can branch on ``data_units`` without re-reading raw YAML.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from spindoctor.support.image_class import (
    ImageQualityThresholds,
    image_quality_thresholds,
    missing_data_marker,
    required_finite_float,
)

__all__ = [
    'InstrumentSettings',
    'instrument_settings_from_obs',
]


DataUnits = Literal['raw_dn', 'calibrated_if']
"""The two recognized per-instrument data-unit kinds.

``raw_dn`` instruments expose pixels in raw analog-to-digital counts; their
saturation, blank, and noise thresholds are quoted in DN.  ``calibrated_if``
instruments expose pixels in I/F (incidence-corrected reflectance);
DN-keyed thresholds are unavailable and the I/F variants are used instead.
"""


@dataclass(frozen=True)
class InstrumentSettings:
    """Per-instrument settings the orchestrator needs at navigate time.

    Parameters:
        data_units: ``raw_dn`` or ``calibrated_if`` per
            ``config_4N0_inst_*.yaml``.
        saturation_dn: Per-instrument saturation DN; ``None`` for
            calibrated-IF instruments.
        marker_value: Missing-data sentinel value (``0`` for most raw
            instruments; ``NaN`` for calibrated-IF).
        thresholds: Image-quality thresholds for the classifier.  All
            values normalized to the appropriate units (DN for
            ``raw_dn``, I/F for ``calibrated_if``).
        fit_camera_rotation: Per-camera flag enabling 3-DoF technique
            fits.
        max_rotation_deg: Maximum allowed rotation magnitude when
            ``fit_camera_rotation`` is True.
    """

    data_units: DataUnits
    saturation_dn: float | None
    marker_value: float
    thresholds: ImageQualityThresholds
    fit_camera_rotation: bool
    max_rotation_deg: float


def instrument_settings_from_obs(obs: Any) -> InstrumentSettings:
    """Build the orchestrator's per-instrument settings from an obs.

    Reads ``obs.inst_config`` (an ``AttrDict`` populated by
    ``ObsInst.from_file``) and returns a frozen ``InstrumentSettings``
    instance.  When ``inst_config`` is ``None`` (test fixtures, simulated
    obs without per-instrument wiring), defaults appropriate for an
    untyped ``raw_dn`` instrument are returned: ``saturation_dn=None``
    (no saturation mask) and the standard ``ImageQualityThresholds``
    defaults so the legacy code path stays unchanged.

    Parameters:
        obs: An ``ObsSnapshotInst`` (or a stand-in carrying the
            ``inst_config`` attribute).

    Returns:
        An ``InstrumentSettings`` instance.

    Raises:
        ValueError: If ``inst_config`` is supplied but missing
            required fields, or carries an unrecognized ``data_units``
            value.
        TypeError: If ``inst_config`` or its ``noise`` entry is not a
            mapping.
    """
    inst_config = getattr(obs, 'inst_config', None)
    if inst_config is None:
        return InstrumentSettings(
            data_units='raw_dn',
            saturation_dn=None,
            marker_value=0.0,
            thresholds=ImageQualityThresholds(),
            fit_camera_rotation=False,
            max_rotation_deg=5.0,
        )
    if not isinstance(inst_config, Mapping):
        raise TypeError(
            f'obs.inst_config must be a mapping (dict / AttrDict); got {type(inst_config).__name__}'
        )
    # The thresholds and the marker come from the reader the simulated-image
    # editor's verdict shares (spindoctor.support.image_class); it also refuses an
    # unknown data_units, a missing image_quality_thresholds block, a noise entry
    # that is not a mapping, and a saturation_threshold_if on a calibrated block.
    thresholds = image_quality_thresholds(inst_config)
    marker_value = missing_data_marker(inst_config)
    data_units: DataUnits = cast(DataUnits, inst_config['data_units'])
    saturation_dn: float | None = None
    if data_units == 'raw_dn':
        noise = inst_config.get('noise')
        if noise is None:
            raise ValueError(
                "instrument config with data_units='raw_dn' missing required noise block"
            )
        saturation_dn = required_finite_float(noise, 'saturation_dn', location='noise')
    # A calibrated_if block has no saturation mask: the same physical full-well
    # DN maps to a different I/F for every combination of exposure time, filter,
    # and gain.  The orchestrator emits an empty saturation mask (see
    # ``_build_saturation_mask``) and the thresholds carry an infinite saturation
    # threshold, so the ``fully_overexposed`` early-out cannot fire.
    fit_rotation = bool(inst_config.get('fit_camera_rotation', False))
    max_rotation = float(inst_config.get('max_rotation_deg', 5.0))
    if not math.isfinite(max_rotation) or max_rotation <= 0.0:
        raise ValueError(f'max_rotation_deg must be finite and > 0; got {max_rotation!r}')
    return InstrumentSettings(
        data_units=data_units,
        saturation_dn=saturation_dn,
        marker_value=marker_value,
        thresholds=thresholds,
        fit_camera_rotation=fit_rotation,
        max_rotation_deg=max_rotation,
    )
