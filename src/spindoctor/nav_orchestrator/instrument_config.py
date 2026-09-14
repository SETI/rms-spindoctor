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

from spindoctor.nav_orchestrator.image_classifier import ImageQualityThresholds

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


def _coerce_marker_value(value: Any) -> float:
    """Map a YAML marker_value entry to a float (``NaN`` permitted)."""
    if value is None:
        return float('nan')
    if isinstance(value, str):
        if value.strip().lower() == 'nan':
            return float('nan')
        raise ValueError(f'noise.marker_value string {value!r} must be "NaN" or numeric')
    return float(value)


def _required_float(block: Any, key: str, *, location: str) -> float:
    """Read a required finite numeric field from a YAML mapping.

    Raises ``ValueError`` with a path-aware error message when the key
    is missing, non-numeric, or non-finite (NaN / +-Inf).  Non-finite
    config values would propagate into thresholds the orchestrator
    compares against pixel data, so they are rejected at load time.
    """
    if block is None or key not in block:
        raise ValueError(f'{location} missing required field {key!r}')
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{location}.{key} must be numeric; got {type(value).__name__}={value!r}')
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f'{location}.{key} must be a finite numeric value; got {value!r}')
    return coerced


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
    data_units_raw = inst_config.get('data_units')
    if not isinstance(data_units_raw, str) or data_units_raw not in (
        'raw_dn',
        'calibrated_if',
    ):
        raise ValueError(
            'instrument config missing required field data_units '
            f"(must be 'raw_dn' or 'calibrated_if'); got {data_units_raw!r}"
        )
    data_units: DataUnits = cast(DataUnits, data_units_raw)
    noise = inst_config.get('noise')
    iqt_block = inst_config.get('image_quality_thresholds')
    if not isinstance(iqt_block, Mapping):
        raise ValueError(
            'instrument config missing required image_quality_thresholds block '
            f'(or block is not a mapping); got {type(iqt_block).__name__}'
        )
    if noise is not None and not isinstance(noise, Mapping):
        raise TypeError(
            f'instrument config noise block must be a mapping; got {type(noise).__name__}'
        )
    if data_units == 'raw_dn':
        if noise is None:
            raise ValueError(
                "instrument config with data_units='raw_dn' missing required noise block"
            )
        saturation_dn: float | None = _required_float(noise, 'saturation_dn', location='noise')
        marker_value = _coerce_marker_value(noise.get('marker_value', 0))
        thresholds = ImageQualityThresholds(
            saturation_threshold_dn=_required_float(
                iqt_block,
                'saturation_threshold_dn',
                location='image_quality_thresholds',
            ),
            missing_data_marker_dn=marker_value,
            max_saturation_frac_clean=float(iqt_block.get('max_overexposed_frac_clean', 0.80)),
            max_missing_frac_clean=float(iqt_block.get('max_missing_frac_clean', 0.30)),
            partial_dropout_min_frac=float(iqt_block.get('partial_dropout_min_frac', 0.05)),
            blank_max_dn=_required_float(
                iqt_block, 'blank_max_dn', location='image_quality_thresholds'
            ),
            noisy_threshold=_required_float(
                iqt_block, 'noisy_threshold_dn', location='image_quality_thresholds'
            ),
        )
    else:
        # calibrated_if: DN-keyed fields are unavailable, and the
        # saturation gate is intentionally disabled.  An I/F-keyed
        # saturation threshold is meaningless — the same physical
        # full-well DN maps to a different I/F value for every
        # combination of exposure time, filter, and gain — so we never
        # compute one.  The orchestrator emits an empty saturation mask
        # (see ``_build_saturation_mask``) and the classifier is
        # handed an ``inf`` threshold so ``saturation_frac`` is always
        # 0.0 and the ``fully_overexposed`` early-out cannot fire.
        # Reject any explicit ``saturation_threshold_if`` so stale
        # configs surface immediately rather than silently no-op.
        marker_value = _coerce_marker_value(
            noise.get('marker_value', float('nan')) if noise is not None else float('nan')
        )
        saturation_dn = None
        if 'saturation_threshold_if' in iqt_block:
            raise ValueError(
                'calibrated_if instrument must not declare '
                'image_quality_thresholds.saturation_threshold_if; the '
                'saturation gate is off for calibrated images because I/F '
                'depends on exposure / filter / gain (see Phase 10 §F)'
            )
        thresholds = ImageQualityThresholds(
            saturation_threshold_dn=math.inf,
            missing_data_marker_dn=marker_value,
            max_saturation_frac_clean=float(iqt_block.get('max_overexposed_frac_clean', 0.80)),
            max_missing_frac_clean=float(iqt_block.get('max_missing_frac_clean', 0.30)),
            partial_dropout_min_frac=float(iqt_block.get('partial_dropout_min_frac', 0.05)),
            blank_max_dn=_required_float(
                iqt_block, 'blank_max_if', location='image_quality_thresholds'
            ),
            noisy_threshold=_required_float(
                iqt_block, 'noisy_threshold_if', location='image_quality_thresholds'
            ),
        )
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
