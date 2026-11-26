"""Resolved detector parameters for one scene render.

:class:`DetectorParams` collapses the emulated instrument's detector defaults
(:mod:`spindoctor.sim.instruments.resolvers`), the scene ``detector`` / ``noise``
blocks, and the ``artifacts.instrument_defaults`` switch into one flat, resolved view
the detector stage reads.  It reads nothing from the navigator's configuration, so a
scene's frame follows the instrument entry and the scene alone, whatever the navigator
believes about the camera.

A scene's ``detector`` key replaces the instrument entry's value.  The entry's gain,
bias, ADC ceiling, full well, signal scale, and missing-data rate apply whether or not
``instrument_defaults`` is on.  The noise terms resolve in this order, highest first: an
explicit scene ``noise`` key, then the detector default when ``instrument_defaults`` is
on, then the disabled floor (physical-chain artifacts default to zero so an unconfigured
scene renders a clean DN frame, per the stage-activation rule).
``instrument_defaults`` turns on the whole physical chain, including Poisson
shot noise, the artifact modes the instrument entry lists under
``artifact_modes.instrument_defaults``, and the detector defaults'
``cosmic_ray_events_per_pixel_per_sec`` (radiation on the detector is physics
of the environment, not a transmission defect; every CCD entry records either a
cohort-measured rate or a retained zero); the
missing-data loss modes are transmission artifacts, not physical-chain noise,
and stay off.

The ``noise.read_noise_dn`` key is a DN value, converted to electrons through
the resolved gain, so a scene that pins a DN read-noise level gets exactly that
DN-level behavior out of the electron chain.  ``signal_full_scale_frac`` is the
well fraction a signal of 1.0 fills at the reference exposure.  A scene
``detector.full_well_dn`` sets the well of either chain in DN; the CCD chain converts it
to electrons through the resolved gain.  A scene
``detector.filter`` replaces the reference exposure with the one the filter's
measured throughput implies (:mod:`spindoctor.sim.instruments.resolvers`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.sim.artifact_modes import (
    ARTIFACT_MODES,
    DETECTOR_MODE_ORDER,
    PROBABILITY,
)
from spindoctor.sim.instruments.resolvers import (
    FilterCatalogError,
    cataloged_binnings,
    instrument_default_modes,
    mode_available,
    resolve_calibrated_sky_if,
    resolve_data_units,
    resolve_detector_defaults,
    resolve_detector_id,
    resolve_downlink_table,
    resolve_filter_throughput,
    resolve_mode_with_catalog,
)
from spindoctor.sim.instruments.schema_detector import Detector
from spindoctor.sim.render.detector.sqrt_lut import DownlinkCurve, resolve_sqrt_lut
from spindoctor.sim.render.optics.stage import instrument_defaults_on
from spindoctor.sim.scene.fields import scene_binning
from spindoctor.sim.scene.idealized import DEFAULT_EXPOSURE_SEC
from spindoctor.sim.scene.truth_detector import DEFAULT_CONTOUR_STEP
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayFloatType

__all__ = ['CalibrationMap', 'DetectorParams', 'calibration_map', 'resolve_detector_params']


class DetectorParamError(ValueError):
    """Raised when a scene selects an unavailable detector configuration."""


@dataclass
class DetectorParams:
    """The flat, resolved detector view for one render.

    Parameters:
        detector_model: 'ccd' (electron chain) or 'vidicon' (DN chain).
        data_units: The frame's units, 'raw_dn' or 'calibrated_if', from the
            instrument entry.
        signal_full_scale_frac: Well fraction a signal of 1.0 fills at the
            reference exposure.
        full_well_e: Full well in electrons (CCD path).
        exposure_ref_sec: Exposure the well fraction references; a scene
            ``detector.filter`` derives it from the filter's throughput.
        exposure_sec: The scene exposure, as commanded.  The calibration divides by it;
            whatever the frame collects follows :attr:`integration_sec`.
        gain_e_per_dn: Resolved gain (electrons per DN) for the selected state.
        read_noise_e: Read-noise sigma in electrons (CCD path).
        bias_dn: Additive DN bias pedestal.
        saturation_dn: ADC clip ceiling in DN.
        full_well_dn: The DN full scale of the vidicon chain.
        quantization: ADC quantization sub-mode.
        poisson: Whether shot noise is applied.
        cosmic_ray_events_per_pixel_per_sec: Expected cosmic-ray events per pixel
            per second the detector collects.
        cosmic_ray_readout_dwell_sec: Effective seconds the detector collects
            cosmic rays outside the exposure, while the frame waits to be read.
        cosmic_ray_amplitude_e: Smallest cosmic-ray deposit per event, in
            electrons (the full well unless the scene or the detector defaults set
            one).
        cosmic_ray_spread_px: Cosmic-ray charge-diffusion sigma in frame pixels;
            0 keeps each deposit where it lands.
        dark_current_e_per_sec: Dark current (electrons / sec); 0 disables.
        vidicon: The vidicon DN-noise sub-parameters (vidicon path only).
        calibration_scale_dn_per_s_per_if: Derived I/F calibration scale.
        dark_dn: Dark pedestal in DN subtracted before the I/F divide.
        calibrated_sky_if: The calibrated sky residual in I/F, added after the
            I/F divide on a calibrated frame: the instrument's detector default,
            else the generic entry's (see
            :func:`~spindoctor.sim.instruments.resolvers.resolve_calibrated_sky_if`).
        missing_data_rate: The fraction of the pixels the downlink loses at random.
        random_seed: The scene seed for the detector's fixed-pattern streams and for
            each artifact mode's activation draw.
        noise_seed: The seed for the detector's per-frame random streams: the
            scene's ``noise_seed`` when it sets one, else its ``random_seed``.
        detector_id: The emulated detector the fixed hot-pixel map belongs to:
            the instrument entry's ``detector_id`` (a calibrated product's entry
            sets it to its raw camera), else the instrument name, and ``generic``
            for a generic scene.
        binning: The readout mode's binning factor ``b``: each frame pixel
            sums a ``b`` by ``b`` block of native pixels.  The detector defaults'
            native dark current and cosmic-ray rate are already scaled by
            ``b**2`` here, and the chain scales the extended signal and the
            saturation well the same way.
        instrument_defaults: Whether the physical-chain opt-in is on.
        calibration_column_residual_e: Sigma, in electrons per native pixel, of
            the calibrated product's fixed per-column residual; 0 disables.
        calibration_pixel_residual_e: Sigma, in electrons per native pixel, of
            the calibrated product's per-frame per-pixel residual; 0 disables.
        quantization_contour_step: The DN posterization step for the
            ``contour_8bit`` quantization sub-mode.
        sqrt_lut: The companding curve of the ``sqrt_lut`` quantization
            sub-mode: the downlink table the scene's ``detector.downlink_table``
            selects, else the instrument's default table, else (for an instrument
            that lists none) the plain square-root curve scaled to
            ``saturation_dn``; None selects that plain curve.
        adversarial_modes: The active detector modes that place their events on the
            navigation features.  Under the scene's ``adversarial`` switch these are
            the modes whose activation key the scene itself sets.  A mode that only
            the instrument defaults turn on keeps the camera's own placement, so the
            camera's hot pixels stay on its fixed map.
        detector_modes: The resolved config of every active detector-stage
            artifact mode (registry defaults filled from the catalog), keyed by
            mode name; the chain applies each at its physical point and records it.
        extra_integration_sec: The time the detector collects light beyond the
            commanded exposure, from the instrument entry (zero when it states none,
            and for the vidicon chain, whose signal does not scale with the exposure).
    """

    detector_model: str
    data_units: str
    signal_full_scale_frac: float
    full_well_e: float
    exposure_ref_sec: float
    exposure_sec: float
    gain_e_per_dn: float
    read_noise_e: float
    bias_dn: float
    saturation_dn: float
    full_well_dn: float
    quantization: str
    poisson: bool
    cosmic_ray_events_per_pixel_per_sec: float
    cosmic_ray_readout_dwell_sec: float
    cosmic_ray_amplitude_e: float
    cosmic_ray_spread_px: float
    dark_current_e_per_sec: float
    vidicon: dict[str, float]
    calibration_scale_dn_per_s_per_if: float
    dark_dn: float
    calibrated_sky_if: float
    missing_data_rate: float
    random_seed: int
    noise_seed: int
    detector_id: str = 'generic'
    binning: int = 1
    instrument_defaults: bool = False
    calibration_column_residual_e: float = 0.0
    calibration_pixel_residual_e: float = 0.0
    quantization_contour_step: int = DEFAULT_CONTOUR_STEP
    sqrt_lut: DownlinkCurve | None = None
    adversarial_modes: frozenset[str] = frozenset()
    detector_modes: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra_integration_sec: float = 0.0

    @property
    def integration_sec(self) -> float:
        """The time the frame collects light: the exposure plus the extra integration."""
        return self.exposure_sec + self.extra_integration_sec


@dataclass(frozen=True)
class CalibrationMap:
    """The affine map between a calibrated frame's DN and its I/F.

    The calibration inverse applies :meth:`to_if` to the digitized frame; a
    telemetry mode that acts on the DN the spacecraft transmitted (the lossy
    codec) maps the I/F frame back with :meth:`to_dn` first, so it behaves on a
    calibrated frame as it does on a raw one.

    Parameters:
        dn_per_if: DN per unit I/F at the scene exposure, the inverse's divisor.
        bias_dn: The bias pedestal the inverse subtracts, in DN.
        dark_dn: The dark pedestal the inverse subtracts, in DN.
        sky_if: The calibrated sky residual the inverse adds, in I/F.
    """

    dn_per_if: float
    bias_dn: float
    dark_dn: float
    sky_if: float

    def to_if(self, dn: NDArrayFloatType) -> NDArrayFloatType:
        """Map DN to I/F: subtract the pedestals, divide, add the sky residual."""
        result: NDArrayFloatType = (dn - self.bias_dn - self.dark_dn) / self.dn_per_if
        return result + self.sky_if

    def to_dn(self, i_over_f: NDArrayFloatType) -> NDArrayFloatType:
        """Map I/F back to DN, the inverse of :meth:`to_if`."""
        result: NDArrayFloatType = (i_over_f - self.sky_if) * self.dn_per_if
        return result + self.bias_dn + self.dark_dn


def calibration_map(dp: DetectorParams) -> CalibrationMap | None:
    """The DN-to-I/F map of a calibrated frame, or None for a raw-DN frame.

    The divisor is the derived calibration scale times the scene exposure (a
    non-positive exposure counts as 1 s).  A non-positive divisor also gives
    None: the frame stays in DN.

    Parameters:
        dp: The resolved detector parameters of the frame.

    Returns:
        The map, or None when the frame is raw DN or the divisor is not positive.
    """
    if dp.data_units == 'raw_dn':
        return None
    exposure = dp.exposure_sec if dp.exposure_sec > 0.0 else 1.0
    dn_per_if = dp.calibration_scale_dn_per_s_per_if * exposure
    if dn_per_if <= 0.0:
        return None
    return CalibrationMap(
        dn_per_if=dn_per_if, bias_dn=dp.bias_dn, dark_dn=dp.dark_dn, sky_if=dp.calibrated_sky_if
    )


def _default_or_zero(
    scene_noise: Mapping[str, Any],
    key: str,
    default: float | None,
    *,
    instrument_defaults: bool,
) -> float:
    """A physical-chain knob: scene override, else detector default (if on), else 0.

    Parameters:
        scene_noise: The scene's ``noise`` keys that carry a value.
        key: The knob's key in ``scene_noise``.
        default: The detector default, or None when the entry states none.
        instrument_defaults: Whether the scene turns the detector defaults on.

    Returns:
        The knob's value.
    """
    if key in scene_noise:
        return float(scene_noise[key])
    if instrument_defaults and default is not None:
        return float(default)
    return 0.0


def _cataloged_binning(params: Mapping[str, Any], *, instrument: str | None) -> int:
    """Read the scene's binning factor and refuse one the instrument does not catalog.

    Parameters:
        params: The full scene mapping, read by
            :func:`~spindoctor.sim.scene.fields.scene_binning`.
        instrument: The sim instrument name, for the catalog lookup.

    Returns:
        The binning factor, 1 when the scene leaves it out.

    Raises:
        DetectorParamError: If the binning isn't an integer (a bool counts as
            neither), or the instrument doesn't catalog it.
    """
    try:
        binning = scene_binning(params)
    except ValueError as exc:
        raise DetectorParamError(f'sim instrument {instrument!r}: {exc}') from exc
    available = cataloged_binnings(instrument)
    if binning not in available:
        raise DetectorParamError(
            f'sim instrument {instrument!r} has no cataloged binning {binning}; '
            f'available binnings: {list(available)}'
        )
    return binning


def _resolve_gain(
    detector_defaults: Detector,
    detector_block: Mapping[str, Any],
    *,
    instrument: str | None,
) -> float:
    """Select ``gain_e_per_dn`` from the detector defaults' table by the scene gain state.

    Parameters:
        detector_defaults: The instrument's detector defaults in the scene's readout
            mode (see
            :func:`~spindoctor.sim.instruments.resolvers.resolve_detector_defaults`).
        detector_block: The scene ``detector`` block (may set ``gain_state``).
        instrument: The sim instrument name, for error messages.

    Returns:
        The electrons-per-DN gain for the selected state.

    Raises:
        DetectorParamError: If the selected state has no catalog entry.
    """
    # A vidicon states no gain table: its chain is in DN, at unit gain.
    table = detector_defaults.gain_e_per_dn_by_state or {0: 1.0}
    state = detector_block.get('gain_state')
    if state is None:
        state = detector_defaults.default_gain_state or 0
    state = int(state)
    if state not in table:
        raise DetectorParamError(
            f'sim instrument {instrument!r} has no cataloged gain state {state}; '
            f'available states: {sorted(table)}'
        )
    return float(table[state])


def _mode_active(mode_name: str, cfg: Mapping[str, Any], random_seed: int) -> bool:
    """Whether a detector artifact mode is active this frame.

    A mode whose activation key is ``probability`` activates with that probability,
    drawn from the mode's own seeded stream so toggling one never perturbs another.
    Any other activation key (an expected event count or a fraction of the pixels) says
    how much of the mode the frame shows, so the mode is active whenever it is positive.
    """
    activation = ARTIFACT_MODES[mode_name].activation
    value = float(cfg.get(activation) or 0.0)
    if value <= 0.0:
        return False
    if activation != PROBABILITY:
        return True
    rng = np.random.default_rng(derive_effect_seed(random_seed, f'detector/{mode_name}/activate'))
    return bool(rng.random() < value)


def _resolve_detector_modes(
    artifacts: Mapping[str, Any],
    instrument: str | None,
    random_seed: int,
    *,
    instrument_defaults: bool,
) -> dict[str, dict[str, Any]]:
    """Resolve the active detector-stage artifact modes for a render.

    A mode the scene sets, or one ``instrument_defaults`` turns on for the instrument,
    is resolved by
    :func:`~spindoctor.sim.instruments.resolvers.resolve_mode_with_catalog` and
    returned keyed by name if the instrument offers it and its activation key makes it
    active.  A key the scene sets replaces the instrument's value for that key alone,
    so a scene under ``instrument_defaults`` can change one parameter of a mode the
    camera shows, or turn the mode off by setting its activation key to 0.
    """
    supplied = instrument_default_modes(instrument) if instrument_defaults else {}
    modes: dict[str, dict[str, Any]] = {}
    for name in DETECTOR_MODE_ORDER:
        if name not in artifacts and name not in supplied:
            continue
        raw = artifacts.get(name, {})
        if not isinstance(raw, Mapping) or not mode_available(name, instrument):
            continue
        cfg = resolve_mode_with_catalog(
            name, raw, instrument, instrument_defaults=instrument_defaults
        )
        if not _mode_active(name, cfg, random_seed):
            continue
        modes[name] = cfg
    return modes


def _filter_exposure_ref(
    detector_block: Mapping[str, Any], *, instrument: str | None, filter_name: str, well_e: float
) -> float:
    """The reference exposure at which a signal of 1.0 fills ``well_e`` through a filter.

    The filter's measured throughput is the electrons per second one native
    pixel collects at I/F 1.0, so the reference exposure is the well charge a
    signal of 1.0 stands for divided by that rate.  The forward chain and the
    derived calibration scale both read the reference exposure, so the raw DN
    and the calibrated DN-to-I/F scale follow the filter together.

    Parameters:
        detector_block: The scene ``detector`` block.
        instrument: The sim instrument name.
        filter_name: The scene's ``detector.filter``.
        well_e: ``signal_full_scale_frac * full_well_e``.

    Returns:
        The reference exposure in seconds.

    Raises:
        DetectorParamError: If the block also sets ``exposure_ref_sec``, selects
            the vidicon model, or the camera does not catalog the filter.
    """
    if detector_block.get('exposure_ref_sec') is not None:
        raise DetectorParamError(
            f'sim instrument {instrument!r}: detector.filter and detector.exposure_ref_sec '
            f'both set the signal scale; set one of them'
        )
    if detector_block.get('detector_model') == 'vidicon':
        raise DetectorParamError(
            f'sim instrument {instrument!r}: detector.filter sets the CCD signal scale; '
            f'the vidicon DN path has no reference exposure for it to set'
        )
    try:
        throughput = resolve_filter_throughput(instrument, filter_name)
    except FilterCatalogError as exc:
        raise DetectorParamError(str(exc)) from exc
    return well_e / throughput


def resolve_detector_params(params: Mapping[str, Any]) -> DetectorParams:
    """Collapse the scene and its instrument's detector defaults into one detector view.

    A scene ``noise``, ``noise.vidicon``, or ``detector`` key that the scene leaves out
    takes the detector default or the configuration default.

    Parameters:
        params: The scene.

    Returns:
        The resolved :class:`DetectorParams`.

    Raises:
        DetectorParamError: If the scene selects an unavailable gain state, sets
            a filter or a downlink table its camera does not catalog, sets both
            ``detector.filter`` and ``detector.exposure_ref_sec``, or compands the
            frame through a downlink table that decodes above the ADC ceiling.
    """
    instrument = params.get('instrument')
    detector_block = params.get('detector') or {}
    binning = _cataloged_binning(params, instrument=instrument)
    # The readout mode's published values replace the native ones it names.
    detector_defaults = resolve_detector_defaults(instrument, binning)
    # A binned pixel collects the charge of its binning**2 native pixels.
    pixels_per_frame_pixel = float(binning * binning)

    scene_noise = params.get('noise') or {}
    artifacts = params.get('artifacts') or {}
    instrument_defaults = instrument_defaults_on(params)

    detector_model = str(detector_block.get('detector_model', detector_defaults.detector_model))
    if detector_model == 'vidicon' and binning != 1:
        raise DetectorParamError(
            f'sim instrument {instrument!r} cannot use detector_model vidicon with '
            f'binning {binning}; the vidicon DN path has no binned readout'
        )
    data_units = resolve_data_units(instrument)
    exposure_sec = float(params.get('exposure_sec', DEFAULT_EXPOSURE_SEC))
    exposure_ref_sec = float(
        detector_block.get('exposure_ref_sec', detector_defaults.exposure_ref_sec)
    )
    quantization = str(detector_block.get('quantization', detector_defaults.quantization))

    signal_full_scale_frac = float(
        detector_block.get('signal_full_scale_frac', detector_defaults.signal_full_scale_frac)
    )
    bias_dn = float(detector_block.get('bias_dn', detector_defaults.bias_dn))
    saturation_dn = float(detector_block.get('saturation_dn', detector_defaults.saturation_dn))
    full_well_dn = float(detector_block.get('full_well_dn', detector_defaults.full_well_dn))
    missing_data_rate = float(
        detector_block.get('missing_data_rate', detector_defaults.missing_data_rate)
    )

    gain_e_per_dn = _resolve_gain(detector_defaults, detector_block, instrument=instrument)
    # A scene's well is in DN on both chains.  Without one, the CCD chain takes the
    # entry's well in electrons, if it states one.
    full_well_e = detector_defaults.full_well_e
    if full_well_e is None or 'full_well_dn' in detector_block:
        full_well_e = full_well_dn * gain_e_per_dn
    # The smallest cosmic-ray deposit: a scene value, else the detector default when
    # instrument_defaults is on, else the full well, so every hit clips.
    if 'cosmic_ray_amplitude_e' in scene_noise:
        cosmic_ray_amplitude = float(scene_noise['cosmic_ray_amplitude_e'])
    elif instrument_defaults and detector_defaults.cosmic_ray_amplitude_e is not None:
        cosmic_ray_amplitude = detector_defaults.cosmic_ray_amplitude_e
    else:
        cosmic_ray_amplitude = full_well_e
    filter_name = detector_block.get('filter')
    if filter_name is not None:
        exposure_ref_sec = _filter_exposure_ref(
            detector_block,
            instrument=instrument,
            filter_name=str(filter_name),
            well_e=signal_full_scale_frac * full_well_e,
        )

    # Read noise: a scene DN override wins (converted to electrons via gain);
    # otherwise the detector default in electrons when instrument_defaults is on;
    # otherwise zero (no read noise).
    if 'read_noise_dn' in scene_noise:
        read_noise_e = float(scene_noise['read_noise_dn']) * gain_e_per_dn
    elif instrument_defaults:
        read_noise_e = detector_defaults.read_noise_e or 0.0
    else:
        read_noise_e = 0.0

    # Poisson: an explicit noise key wins (noise: {poisson: false} turns it off
    # even under instrument_defaults); otherwise the physical-chain opt-in turns
    # shot noise on, and the floor is off.
    if 'poisson' in scene_noise:
        poisson = bool(scene_noise['poisson'])
    else:
        poisson = instrument_defaults
    # Cosmic rays (expected events per pixel per second): scene value, else the
    # detector defaults' rate when instrument_defaults is on, else none.
    cr_key = 'cosmic_ray_events_per_pixel_per_sec'
    if cr_key in scene_noise:
        cosmic_ray_rate = float(scene_noise[cr_key])
    elif instrument_defaults:
        rate = detector_defaults.cosmic_ray_events_per_pixel_per_sec
        cosmic_ray_rate = rate * pixels_per_frame_pixel
    else:
        cosmic_ray_rate = 0.0
    cosmic_ray_dwell = _default_or_zero(
        scene_noise,
        'cosmic_ray_readout_dwell_sec',
        detector_defaults.cosmic_ray_readout_dwell_sec,
        instrument_defaults=instrument_defaults,
    )
    cosmic_ray_spread = _default_or_zero(
        scene_noise,
        'cosmic_ray_spread_px',
        detector_defaults.cosmic_ray_spread_px,
        instrument_defaults=instrument_defaults,
    )
    # The detector defaults state the spread in native pixels, so a binned pixel sees
    # it shrink by the binning, as the PSF kernel does; a scene key states the
    # frame pixel's own value.
    if 'cosmic_ray_spread_px' not in scene_noise:
        cosmic_ray_spread /= binning

    dark_current = _default_or_zero(
        scene_noise,
        'dark_current_e_per_sec',
        detector_defaults.dark_current_e_per_sec,
        instrument_defaults=instrument_defaults,
    )
    if 'dark_current_e_per_sec' not in scene_noise:
        dark_current *= pixels_per_frame_pixel
    calibration_column_residual = _default_or_zero(
        scene_noise,
        'calibration_column_residual_e',
        detector_defaults.calibration_column_residual_e,
        instrument_defaults=instrument_defaults,
    )
    calibration_pixel_residual = _default_or_zero(
        scene_noise,
        'calibration_pixel_residual_e',
        detector_defaults.calibration_pixel_residual_e,
        instrument_defaults=instrument_defaults,
    )
    # The vidicon DN-noise sub-parameters activate on the vidicon path only, and
    # like every physical-chain artifact stay disabled until instrument_defaults
    # is on or the scene sets them explicitly (the stage-activation floor).
    default_vidicon = (
        dataclasses.asdict(detector_defaults.vidicon)
        if instrument_defaults and detector_defaults.vidicon is not None
        else {}
    )
    scene_vidicon = scene_noise.get('vidicon') or {}
    vidicon = {**default_vidicon, **{k: float(v) for k, v in scene_vidicon.items()}}

    # Derived I/F calibration scale.  Uses the image-side DN well (frac *
    # full_well_e / gain for the CCD path, frac * full_well_dn for the vidicon
    # path).  The shared inverse divides by scale * exposure_sec.  The CCD forward
    # chain scales electrons by the integration time over exposure_ref_sec, so its
    # scale carries 1 / exposure_ref_sec, and a noise-free signal of 1.0
    # round-trips to I/F (exposure_sec + extra_integration_sec) / exposure_sec:
    # 1.0 for a camera that states no extra integration time.  The vidicon forward
    # mapping has no exposure term at all, so its scale folds in 1 / exposure_sec,
    # the exposure cancels (the inverse divides by the DN full scale alone), and a
    # signal of 1.0 round-trips to I/F 1.0.
    if detector_model == 'vidicon':
        image_well_dn = signal_full_scale_frac * full_well_dn
        calibration_scale = image_well_dn / exposure_sec if exposure_sec > 0.0 else image_well_dn
    else:
        image_well_dn = signal_full_scale_frac * full_well_e / gain_e_per_dn
        image_well_dn *= pixels_per_frame_pixel
        calibration_scale = (
            image_well_dn / exposure_ref_sec if exposure_ref_sec > 0.0 else image_well_dn
        )
    dark_dn = 0.0 if detector_model == 'vidicon' else dark_current * exposure_sec / gain_e_per_dn

    detector_modes = _resolve_detector_modes(
        artifacts,
        instrument,
        int(params['random_seed']),
        instrument_defaults=instrument_defaults,
    )
    adversarial_modes = (
        frozenset(
            name
            for name in detector_modes
            if ARTIFACT_MODES[name].activation in artifacts.get(name, {})
        )
        if artifacts.get('adversarial', False)
        else frozenset()
    )
    try:
        downlink_table = resolve_downlink_table(instrument, detector_block.get('downlink_table'))
    except ValueError as exc:
        raise DetectorParamError(str(exc)) from exc
    sqrt_lut = resolve_sqrt_lut(downlink_table, saturation_dn=saturation_dn)
    if quantization == 'sqrt_lut' and downlink_table is not None:
        try:
            downlink_table.check_ceiling(saturation_dn)
        except ValueError as exc:
            raise DetectorParamError(f'sim instrument {instrument!r}: {exc}') from exc

    # A scene's noise_seed redraws the per-frame detector streams of the same scene.
    noise_seed = params['noise_seed'] if 'noise_seed' in params else params['random_seed']
    return DetectorParams(
        detector_model=detector_model,
        data_units=data_units,
        signal_full_scale_frac=signal_full_scale_frac,
        full_well_e=full_well_e,
        exposure_ref_sec=exposure_ref_sec,
        exposure_sec=exposure_sec,
        gain_e_per_dn=gain_e_per_dn,
        read_noise_e=read_noise_e,
        bias_dn=bias_dn,
        saturation_dn=saturation_dn,
        full_well_dn=full_well_dn,
        quantization=quantization,
        poisson=poisson,
        cosmic_ray_events_per_pixel_per_sec=cosmic_ray_rate,
        cosmic_ray_readout_dwell_sec=cosmic_ray_dwell,
        cosmic_ray_amplitude_e=cosmic_ray_amplitude,
        cosmic_ray_spread_px=cosmic_ray_spread,
        dark_current_e_per_sec=dark_current,
        vidicon=vidicon,
        calibration_scale_dn_per_s_per_if=calibration_scale,
        dark_dn=dark_dn,
        calibrated_sky_if=(
            resolve_calibrated_sky_if(instrument)
            if detector_defaults.calibrated_sky_if is None
            else detector_defaults.calibrated_sky_if
        ),
        missing_data_rate=missing_data_rate,
        random_seed=int(params['random_seed']),
        noise_seed=int(noise_seed),
        detector_id=resolve_detector_id(instrument),
        binning=binning,
        instrument_defaults=instrument_defaults,
        calibration_column_residual_e=calibration_column_residual,
        calibration_pixel_residual_e=calibration_pixel_residual,
        quantization_contour_step=int(detector_block.get('contour_step', DEFAULT_CONTOUR_STEP)),
        sqrt_lut=sqrt_lut,
        adversarial_modes=adversarial_modes,
        detector_modes=detector_modes,
        extra_integration_sec=(
            detector_defaults.extra_integration_sec if detector_model == 'ccd' else 0.0
        ),
    )
