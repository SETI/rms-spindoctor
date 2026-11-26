"""The scene's ``detector`` and ``noise`` blocks: the detector chain a frame is read by.

Both blocks are truth.  The ``detector`` block overrides the values of the camera's
instrument entry, and each key it leaves out keeps the camera's value (for a binned
frame, the value of the selected readout mode).  The ``noise`` block plants the
detector's noise.  Each noise key a scene leaves out is off, unless
``artifacts.instrument_defaults`` turns on the camera's own value; a key the scene sets
always wins.

A key that the camera's instrument entry also declares is typed by that entry's
declaration (:mod:`spindoctor.sim.instruments.schema_detector`), so the scene and the
catalog accept the same range for it.  The rules that need the instrument catalog,
such as whether the camera catalogs a gain state, are in
:mod:`spindoctor.sim.scene.rules_detector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import AfterValidator

from spindoctor.schema import SCHEMA_CONFIG, Bool, Int, NonNeg, PositiveInt, RuleError, Str
from spindoctor.sim.artifact_modes import DetectorModel, QuantizationMode
from spindoctor.sim.instruments.schema_detector import (
    BiasDn,
    CalibrationColumnResidualE,
    CalibrationPixelResidualE,
    CosmicRayAmplitudeE,
    CosmicRayEventsPerPixelPerSec,
    CosmicRayReadoutDwellSec,
    CosmicRaySpreadPx,
    DarkCurrentEPerSec,
    ExposureRefSec,
    FullWellDn,
    GainState,
    MissingDataRate,
    ReadNoiseLineDn,
    ReadNoisePixelDn,
    SaturationDn,
    SignalFullScaleFrac,
)
from spindoctor.sim.scene.declare import EditorHint, IntWithin, optional
from spindoctor.sim.scene.fields import DEFAULT_BINNING

__all__ = [
    'CONTOUR_STEP_MAX',
    'DEFAULT_CONTOUR_STEP',
    'ContourStep',
    'DetectorOverride',
    'Noise',
    'NoiseVidicon',
]

#: The largest step of the ``contour_8bit`` quantization: the top code of the 8-bit
#: word it posterizes.
CONTOUR_STEP_MAX = 255

#: The DN step of the ``contour_8bit`` quantization when a scene leaves
#: ``detector.contour_step`` out.
DEFAULT_CONTOUR_STEP = 8

#: The DN step of the ``contour_8bit`` quantization.
ContourStep = Annotated[Int, AfterValidator(IntWithin(1, CONTOUR_STEP_MAX))]

# The reason a key is off when a scene leaves it out, unless the camera's chain is on.
_CAMERA_DEFAULT = "the camera's value under instrument_defaults, else 0"


@dataclass(frozen=True, kw_only=True)
class NoiseVidicon:
    """The read noise of the vidicon chain, in DN.

    Parameters:
        read_noise_line_dn: The per-line read noise, in DN.
        read_noise_pixel_dn: The per-pixel read noise, in DN.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    read_noise_line_dn: ReadNoiseLineDn | None = optional(
        'truth',
        'The sigma of the read noise every pixel of one line shares, in DN.',
        when_absent=_CAMERA_DEFAULT,
    )
    read_noise_pixel_dn: ReadNoisePixelDn | None = optional(
        'truth',
        'The sigma of the read noise of each pixel, in DN.',
        when_absent=_CAMERA_DEFAULT,
    )


@dataclass(frozen=True, kw_only=True)
class Noise:
    """The scene's ``noise`` block: the detector noise the frame is rendered with.

    Parameters:
        poisson: Whether the electron image gets shot noise.
        read_noise_dn: The read-noise sigma, in DN.
        cosmic_ray_events_per_pixel_per_sec: The cosmic-ray rate.
        cosmic_ray_readout_dwell_sec: The time the frame collects cosmic rays during
            readout.
        cosmic_ray_amplitude_e: The smallest charge a cosmic ray deposits.
        cosmic_ray_spread_px: The sigma of a hit's diffusion.
        dark_current_e_per_sec: The dark current.
        calibration_column_residual_e: The calibration's residual column pattern.
        calibration_pixel_residual_e: The calibration's residual per-pixel noise.
        vidicon: The vidicon chain's read noise.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    poisson: Bool | None = optional(
        'truth',
        'Whether the CCD chain adds Poisson shot noise to its electron image.',
        when_absent='on under instrument_defaults, else off',
    )
    read_noise_dn: NonNeg | None = optional(
        'truth',
        "The CCD chain's Gaussian read-noise sigma, in DN, converted to electrons "
        "through the frame's gain.",
        when_absent="the camera's value in electrons under instrument_defaults, else 0",
        editor=EditorHint(minimum=0.0, maximum=50.0, decimals=2),
    )
    cosmic_ray_events_per_pixel_per_sec: CosmicRayEventsPerPixelPerSec | None = optional(
        'truth',
        'The expected cosmic-ray events per frame pixel per second. A frame draws its '
        'count from a Poisson distribution whose mean is the rate times the collection '
        'time, times the pixel count. On the CCD chain the collection time is the '
        "exposure, the camera's extra integration time, and the readout dwell together; "
        'on the vidicon chain it is the exposure alone. The '
        'camera states its rate per native pixel, so a binned frame multiplies the '
        "camera's rate by the native pixels in a frame pixel.",
        when_absent=_CAMERA_DEFAULT,
        editor=EditorHint(minimum=0.0, maximum=0.01, decimals=5, step=0.0001),
    )
    cosmic_ray_readout_dwell_sec: CosmicRayReadoutDwellSec | None = optional(
        'truth',
        'The time, in seconds, the frame collects cosmic rays outside its integration, '
        'while it waits to be read. Only the CCD chain reads it.',
        when_absent=_CAMERA_DEFAULT,
    )
    cosmic_ray_amplitude_e: CosmicRayAmplitudeE | None = optional(
        'truth',
        'The smallest charge a cosmic ray deposits, in electrons. A hit leaves this '
        'times one plus a standard lognormal draw. Only the CCD chain reads it: every '
        "vidicon hit fills the chain's full scale.",
        when_absent="the camera's value under instrument_defaults, else the full well",
    )
    cosmic_ray_spread_px: CosmicRaySpreadPx | None = optional(
        'truth',
        "The sigma of a hit's charge diffusion on the CCD chain, in frame pixels. The "
        'camera states it in native pixels, so a binned frame sees the camera value '
        'shrink by the binning.',
        when_absent=_CAMERA_DEFAULT,
    )
    dark_current_e_per_sec: DarkCurrentEPerSec | None = optional(
        'truth',
        "The CCD chain's dark current, in electrons per frame pixel per second. It is "
        'the same on every line, and it is added before the shot noise. The camera '
        "states it per native pixel, so a binned frame multiplies the camera's value by "
        'the native pixels in a frame pixel.',
        when_absent=_CAMERA_DEFAULT,
    )
    calibration_column_residual_e: CalibrationColumnResidualE | None = optional(
        'truth',
        "The sigma of a calibrated frame's fixed residual per column, in electrons per "
        'native pixel. Only a calibrated frame reads it.',
        when_absent=_CAMERA_DEFAULT,
    )
    calibration_pixel_residual_e: CalibrationPixelResidualE | None = optional(
        'truth',
        "The sigma of a calibrated frame's residual per pixel, in electrons per native "
        'pixel. Only a calibrated frame reads it.',
        when_absent=_CAMERA_DEFAULT,
    )
    vidicon: NoiseVidicon | None = optional(
        'truth',
        'The read noise of the vidicon chain, which only a frame rendered through it reads.',
        when_absent=_CAMERA_DEFAULT,
    )


@dataclass(frozen=True, kw_only=True)
class DetectorOverride:
    """The scene's ``detector`` block: the camera values a scene replaces.

    Parameters:
        gain_state: The electron chain's gain state.
        binning: The readout mode's binning.
        detector_model: The chain the frame is rendered through.
        bias_dn: The bias pedestal.
        saturation_dn: The ADC ceiling.
        full_well_dn: A native pixel's full well, in DN.
        signal_full_scale_frac: The fraction of the well a signal of 1.0 fills.
        exposure_ref_sec: The reference exposure.
        filter: The filter combination.
        quantization: How the frame is quantized to DN.
        downlink_table: The downlink table the ``sqrt_lut`` quantization decodes
            through.
        contour_step: The DN step of the ``contour_8bit`` quantization.
        missing_data_rate: The fraction of the pixels lost in the downlink.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    gain_state: GainState | None = optional(
        'truth',
        "The electron chain's gain state, which the camera must catalog.",
        when_absent="the camera's default gain state",
    )
    binning: PositiveInt | None = optional(
        'truth',
        'The readout mode: each frame pixel sums a block of this many native pixels on '
        "a side. The camera's instrument entry lists its readout modes. It is the one "
        "detector value the navigator's view carries, as its top-level ``binning``, "
        "because a real frame's label states its readout mode.",
        default=DEFAULT_BINNING,
    )
    detector_model: DetectorModel | None = optional(
        'truth',
        'The chain the frame is rendered through: ``ccd``, the electron chain, or '
        '``vidicon``, the DN chain.',
        when_absent="the camera's chain",
    )
    bias_dn: BiasDn | None = optional(
        'truth',
        'The additive bias pedestal, in DN. It lifts dark sky off zero, which is the '
        'missing-data marker, and a calibrated frame subtracts it again.',
        when_absent="the camera's value",
    )
    saturation_dn: SaturationDn | None = optional(
        'truth',
        'The ADC ceiling, in DN. The detector clips every pixel at it.',
        when_absent="the camera's value",
    )
    full_well_dn: FullWellDn | None = optional(
        'truth',
        "A native pixel's full well, in DN. The CCD chain converts it to electrons "
        "through the frame's gain; on the vidicon chain it is the DN full scale.",
        when_absent="the camera's value",
    )
    signal_full_scale_frac: SignalFullScaleFrac | None = optional(
        'truth',
        'The fraction of the well a signal of 1.0 fills at the reference exposure.',
        when_absent="the camera's value",
    )
    exposure_ref_sec: ExposureRefSec | None = optional(
        'truth',
        'The reference exposure, in seconds. A scene sets this or ``filter``, never both.',
        when_absent="the camera's value, or the one the filter sets",
    )
    filter: Str | None = optional(
        'truth',
        'The filter combination, written as the product label writes it, such as '
        '``CL1+CL2``. It sets the signal scale from the filter table of the camera, '
        'which must catalog it.',
        when_absent="the camera's nominal signal scale",
    )
    quantization: QuantizationMode | None = optional(
        'truth',
        'How the ADC quantizes the frame to DN.',
        when_absent="the camera's value for the readout mode",
    )
    downlink_table: Str | None = optional(
        'truth',
        'The downlink table the ``sqrt_lut`` quantization decodes through, one the camera '
        'lists. Only that quantization reads it.',
        when_absent="the camera's default table",
    )
    contour_step: ContourStep | None = optional(
        'truth',
        'The DN step of the ``contour_8bit`` quantization. Only that quantization reads it.',
        default=DEFAULT_CONTOUR_STEP,
    )
    missing_data_rate: MissingDataRate | None = optional(
        'truth',
        'The fraction of the pixels the telemetry stage sets to the missing-data marker.',
        when_absent="the camera's value",
    )

    def __post_init__(self) -> None:
        """Refuse settings the detector chain cannot honor together."""
        if self.detector_model == 'vidicon' and self.binning not in (None, 1):
            raise RuleError(
                f'binning {self.binning} is not supported with detector_model vidicon; the '
                'vidicon DN path has no binned readout',
                at='binning',
                reads=('binning', 'detector_model'),
            )
        if self.filter is None:
            return
        if self.exposure_ref_sec is not None:
            raise RuleError(
                'filter and exposure_ref_sec both set the signal scale; set one of them',
                at='filter',
                reads=('filter', 'exposure_ref_sec'),
            )
        if self.detector_model == 'vidicon':
            raise RuleError(
                'filter sets the CCD signal scale; the vidicon DN path has no reference '
                'exposure for it to set',
                at='filter',
                reads=('filter', 'detector_model'),
            )
