"""The schema of an instrument entry's detector block and readout modes.

An entry's ``detector`` block (:class:`Detector`) holds the values of its detector
chain, and each of its ``readout_modes`` (:class:`ReadoutMode`) holds the detector
values one binning changes.  A readout mode may state any detector key except the
three that belong to the detector as a whole, and each key it leaves out keeps the
unbinned value.  :meth:`Detector.with_readout_mode` merges one over the other.  The
rest of an entry's schema is in :mod:`spindoctor.sim.instruments.schema`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import AfterValidator

from spindoctor.schema import (
    SCHEMA_CONFIG,
    DigitInt,
    Float,
    NonNeg,
    NonNegInt,
    Positive,
    RuleError,
    Unit,
)
from spindoctor.schema.errors import RangeError
from spindoctor.sim.artifact_modes import DetectorModel, QuantizationMode

__all__ = [
    'Detector',
    'GainTable',
    'ReadoutMode',
    'ReadoutModes',
    'VidiconChain',
]

# The electron-chain keys a CCD entry carries.
_CCD_REQUIRED: tuple[str, ...] = (
    'full_well_e',
    'gain_e_per_dn_by_state',
    'default_gain_state',
    'read_noise_e',
    'dark_current_e_per_sec',
    'hot_pixel_fraction',
    'hot_pixel_amplitude_e',
    'hot_pixel_column_factor',
    'bias_pedestal_sigma_dn',
    'bias_row_gradient_dn',
    'bias_col_gradient_dn',
    'bloom_length',
)

# The detector keys a readout mode cannot change: a readout mode bins the same
# detector, so its model and its vidicon chain are the unbinned mode's.
_READOUT_FIXED_KEYS: tuple[str, ...] = ('detector_model', 'vidicon')


def _non_negative_key(value: int) -> int:
    """Return an integer key if it is at least zero, else raise the range error."""
    if value < 0:
        raise RangeError('a non-negative integer')
    return value


def _positive_key(value: int) -> int:
    """Return an integer key if it is above zero, else raise the range error."""
    if value <= 0:
        raise RangeError('a positive integer')
    return value


def _contiguous_states(table: dict[int, float]) -> dict[int, float]:
    """Return a gain table if its states run without a gap from the lowest one.

    Raises:
        ValueError: If a state is missing between the lowest and the highest.
    """
    states = sorted(table)
    if states and states != list(range(states[0], states[0] + len(states))):
        raise ValueError(f'the gain states {states} must be contiguous integers from {states[0]}')
    return table


def _has_unbinned_mode(modes: dict[int, Any]) -> dict[int, Any]:
    """Return the readout modes if binning 1 is among them.

    Raises:
        RuleError: If it is not.
    """
    if 1 not in modes:
        raise RuleError('binning 1 (the unbinned readout) is not defined', at=(), reads=((1,),))
    return modes


#: A gain table: electrons per DN, keyed by gain state.  The states are non-negative
#: integers that run without a gap from the lowest one.
GainTable = Annotated[
    dict[Annotated[DigitInt, AfterValidator(_non_negative_key)], Positive],
    AfterValidator(_contiguous_states),
]


@dataclass(frozen=True)
class VidiconChain:
    """The DN-domain noise of a vidicon's readout.

    Parameters:
        read_noise_line_dn: The per-line read noise, in DN.
        read_noise_pixel_dn: The per-pixel read noise, in DN.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    read_noise_line_dn: NonNeg
    read_noise_pixel_dn: NonNeg


@dataclass(frozen=True)
class Detector:
    """An entry's detector block: the values of its detector chain.

    Every entry states the first five keys.  A CCD entry also states its electron
    chain: the full well, gain table, default gain state, read noise, dark current,
    hot-pixel and bias-structure terms, and the bloom length.  A vidicon
    entry states its ``vidicon`` chain.  The other keys are optional.

    Parameters:
        detector_model: The chain the detector stage renders with.
        exposure_ref_sec: The reference exposure, in seconds, at which a signal of
            1.0 fills the scaled well.
        bias_dn: The bias pedestal, in DN.
        quantization: How the frame is quantized to DN.
        cosmic_ray_events_per_pixel_per_sec: The cosmic-ray rate.
        full_well_e: The full well, in electrons.
        gain_e_per_dn_by_state: Electrons per DN, keyed by gain state.
        default_gain_state: The gain state a scene gets when it sets none.
        read_noise_e: The read noise, in electrons.
        dark_current_e_per_sec: The dark current, in electrons per second.
        readout_dark_e: The dark charge collected during readout, in electrons at
            the last line.
        hot_pixel_fraction: The fraction of pixels that are hot.
        hot_pixel_amplitude_e: A hot pixel's charge, in electrons.
        hot_pixel_column_factor: The share of hot pixels that line up in columns.
        cosmic_ray_readout_dwell_sec: The time, in seconds, the frame sits exposed
            to cosmic rays during readout.
        cosmic_ray_amplitude_e: The smallest charge a cosmic ray deposits, in
            electrons.
        cosmic_ray_spread_px: The sigma, in native pixels, of a hit's diffusion.
        bias_pedestal_sigma_dn: The frame-to-frame jitter of the bias, in DN.
        bias_row_gradient_dn: The bias gradient along the rows, in DN.
        bias_col_gradient_dn: The bias gradient along the columns, in DN.
        bloom_length: How many pixels charge above the full well spills along the
            column.
        calibrated_sky_if: The calibrated sky residual, in I/F, of a calibrated
            frame.
        dark_sky_ceiling_if: The realism match's dark-sky ceiling, in I/F.
        calibration_column_residual_e: The calibration's residual column pattern, in
            electrons.
        calibration_pixel_residual_e: The calibration's residual per-pixel noise, in
            electrons.
        vidicon: The vidicon chain.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    detector_model: DetectorModel
    exposure_ref_sec: Positive
    bias_dn: NonNeg
    quantization: QuantizationMode
    cosmic_ray_events_per_pixel_per_sec: NonNeg
    full_well_e: Positive | None = None
    gain_e_per_dn_by_state: GainTable | None = None
    default_gain_state: NonNegInt | None = None
    read_noise_e: NonNeg | None = None
    dark_current_e_per_sec: NonNeg | None = None
    readout_dark_e: NonNeg | None = None
    hot_pixel_fraction: Unit | None = None
    hot_pixel_amplitude_e: NonNeg | None = None
    hot_pixel_column_factor: Unit | None = None
    cosmic_ray_readout_dwell_sec: NonNeg | None = None
    cosmic_ray_amplitude_e: NonNeg | None = None
    cosmic_ray_spread_px: NonNeg | None = None
    bias_pedestal_sigma_dn: NonNeg | None = None
    bias_row_gradient_dn: NonNeg | None = None
    bias_col_gradient_dn: NonNeg | None = None
    bloom_length: NonNegInt | None = None
    calibrated_sky_if: Float | None = None
    dark_sky_ceiling_if: Positive | None = None
    calibration_column_residual_e: NonNeg | None = None
    calibration_pixel_residual_e: NonNeg | None = None
    vidicon: VidiconChain | None = None

    def __post_init__(self) -> None:
        """Refuse a detector without its model's keys, or a default state not cataloged."""
        model = self.detector_model
        required = _CCD_REQUIRED if model == 'ccd' else ('vidicon',)
        missing = sorted(key for key in required if getattr(self, key) is None)
        if missing:
            raise RuleError(
                f'a {model} detector requires the keys {missing}',
                at=(),
                reads=(*missing, 'detector_model'),
            )
        if model != 'vidicon' and self.vidicon is not None:
            raise RuleError(
                f'a {model} detector has no vidicon chain to configure',
                at='vidicon',
                reads=('vidicon', 'detector_model'),
            )
        table = self.gain_e_per_dn_by_state or {}
        state = self.default_gain_state
        if state is not None and state not in table:
            raise RuleError(
                f'state {state} is not in gain_e_per_dn_by_state {sorted(table)}',
                at='default_gain_state',
                reads=('default_gain_state', ('gain_e_per_dn_by_state', state)),
            )

    def with_readout_mode(self, mode: ReadoutMode) -> Detector:
        """The detector's values in one readout mode.

        Parameters:
            mode: The readout mode.

        Returns:
            A copy of this detector in which each key the mode sets has the mode's
            value.
        """
        return dataclasses.replace(self, **mode.overrides())


@dataclass(frozen=True)
class ReadoutMode:
    """The detector values one binning changes.

    A b-by-b block of native pixels reads out as one pixel.  Each field is a
    :class:`Detector` key, optional, and the keys the mode leaves out keep the
    unbinned values.  A readout mode bins the same detector, so it cannot change
    ``detector_model`` or the ``vidicon`` chain, and the realism match reads the
    dark-sky ceiling from the detector block alone, so it cannot set
    ``dark_sky_ceiling_if``.

    Parameters:
        detector_model: Refused when set.
        exposure_ref_sec: As :class:`Detector`.
        bias_dn: As :class:`Detector`.
        quantization: As :class:`Detector`.
        cosmic_ray_events_per_pixel_per_sec: As :class:`Detector`.
        full_well_e: As :class:`Detector`.
        gain_e_per_dn_by_state: As :class:`Detector`.
        default_gain_state: As :class:`Detector`.
        read_noise_e: As :class:`Detector`.
        dark_current_e_per_sec: As :class:`Detector`.
        readout_dark_e: As :class:`Detector`.
        hot_pixel_fraction: As :class:`Detector`.
        hot_pixel_amplitude_e: As :class:`Detector`.
        hot_pixel_column_factor: As :class:`Detector`.
        cosmic_ray_readout_dwell_sec: As :class:`Detector`.
        cosmic_ray_amplitude_e: As :class:`Detector`.
        cosmic_ray_spread_px: As :class:`Detector`.
        bias_pedestal_sigma_dn: As :class:`Detector`.
        bias_row_gradient_dn: As :class:`Detector`.
        bias_col_gradient_dn: As :class:`Detector`.
        bloom_length: As :class:`Detector`.
        calibrated_sky_if: As :class:`Detector`.
        dark_sky_ceiling_if: Refused when set.
        calibration_column_residual_e: As :class:`Detector`.
        calibration_pixel_residual_e: As :class:`Detector`.
        vidicon: Refused when set.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    detector_model: DetectorModel | None = None
    exposure_ref_sec: Positive | None = None
    bias_dn: NonNeg | None = None
    quantization: QuantizationMode | None = None
    cosmic_ray_events_per_pixel_per_sec: NonNeg | None = None
    full_well_e: Positive | None = None
    gain_e_per_dn_by_state: GainTable | None = None
    default_gain_state: NonNegInt | None = None
    read_noise_e: NonNeg | None = None
    dark_current_e_per_sec: NonNeg | None = None
    readout_dark_e: NonNeg | None = None
    hot_pixel_fraction: Unit | None = None
    hot_pixel_amplitude_e: NonNeg | None = None
    hot_pixel_column_factor: Unit | None = None
    cosmic_ray_readout_dwell_sec: NonNeg | None = None
    cosmic_ray_amplitude_e: NonNeg | None = None
    cosmic_ray_spread_px: NonNeg | None = None
    bias_pedestal_sigma_dn: NonNeg | None = None
    bias_row_gradient_dn: NonNeg | None = None
    bias_col_gradient_dn: NonNeg | None = None
    bloom_length: NonNegInt | None = None
    calibrated_sky_if: Float | None = None
    dark_sky_ceiling_if: Positive | None = None
    calibration_column_residual_e: NonNeg | None = None
    calibration_pixel_residual_e: NonNeg | None = None
    vidicon: VidiconChain | None = None

    def __post_init__(self) -> None:
        """Refuse a key that belongs to the detector as a whole."""
        for key in _READOUT_FIXED_KEYS:
            if getattr(self, key) is not None:
                raise RuleError(
                    f'a readout mode bins the same detector, so it cannot change detector.{key}',
                    at=key,
                )
        if self.dark_sky_ceiling_if is not None:
            raise RuleError(
                'the realism comparison reads the dark-sky ceiling from the detector block, '
                'so a readout mode cannot set it',
                at='dark_sky_ceiling_if',
            )

    def overrides(self) -> dict[str, Any]:
        """The detector keys the mode sets.

        Returns:
            Each key the mode sets, with its value; empty for a mode that keeps every
            unbinned value.
        """
        return {
            item.name: getattr(self, item.name)
            for item in dataclasses.fields(self)
            if getattr(self, item.name) is not None
        }


#: The readout modes, keyed by binning factor.  Binning 1 is always present.
ReadoutModes = Annotated[
    dict[Annotated[DigitInt, AfterValidator(_positive_key)], ReadoutMode],
    AfterValidator(_has_unbinned_mode),
]
