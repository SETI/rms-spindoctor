"""The detector stage: composed signal to digitized detector counts.

This is the normative unit chain of the forward model.  For a CCD the composed
intensive signal is converted to electrons through the exposure, the point-source
electron plane is added, and the frame passes through Poisson shot noise,
electron-domain full-well bloom, read noise, coherent banding, gain to DN, bias
structure, quantization, and the ADC clip.  A camera's physical saturation
therefore emerges from ``full_well_e / gain_e_per_dn`` (below the ADC ceiling for
Cassini), not from the ADC clip.

The Voyager vidicon skips the electron conversion and applies its noise directly
in DN (line-correlated read noise plus a faint coherent component); its point
sources (stars) are already DN and are added onto the converted signal before
that DN noise.  A calibrated
(I/F) scene renders through the full DN chain and then inverts the calibration
transform, so calibrated products carry propagated shot/read noise and
quantization texture in I/F units.

The deterministic conversion (signal to DN) always runs -- it is what makes a DN
frame -- while the stochastic and structured sub-effects (shot noise, read noise,
cosmic rays, dark/hot pixels, banding, bias structure) activate only when their
scene block or ``instrument_defaults`` requests them, so a scene with no such
block renders a clean DN frame (the self-consistency floor).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.sim.forward.detector.electronics_stages import (
    add_bright_dark_pairs,
    add_coherent_banding,
    add_dark_ramp,
    add_fixed_pattern_dn,
    add_fixed_pattern_response,
    add_residual_image,
    add_serial_tail,
    apply_beam_bend,
    apply_exposure_shading,
    apply_frame_transfer_smear,
)
from spindoctor.sim.forward.detector.noise_stages import (
    add_banding,
    add_bias_structure,
    add_cosmic_rays,
    add_dark_current,
    add_hot_pixels,
    deposit_morphological_events,
)
from spindoctor.sim.forward.detector.params import DetectorParams, resolve_detector_params
from spindoctor.sim.forward.feature_loci import dilated_pixels, extract_feature_loci
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['apply_detector', 'apply_saturation', 'quantize_dn']


def _stage_rng(seed: int, effect: str) -> np.random.Generator:
    """A generator for one detector sub-effect, seeded independently by name."""
    return np.random.default_rng(derive_effect_seed(seed, f'detector/{effect}'))


def _feature_pool(frame: SimFrame) -> tuple[NDArrayIntType, NDArrayIntType]:
    """The dilated navigation-feature pixel pool for adversarial placement."""
    loci = extract_feature_loci(frame.truth, frame.signal.shape)
    return dilated_pixels(loci, radius=3, shape=frame.signal.shape)


def _hot_pixel_pool(
    frame: SimFrame, dp: DetectorParams
) -> tuple[NDArrayIntType, NDArrayIntType] | None:
    """The adversarial hot-pixel placement pool, or None for uniform placement.

    When the hot_pixels artifact mode requests adversarial placement, the
    population is biased onto the pixels on and beside the navigation features
    (limb / ring-edge arcs and star positions, dilated by a few pixels).
    """
    if not dp.hot_pixel_adversarial:
        return None
    return _feature_pool(frame)


def _mode_seed(dp: DetectorParams, mode_name: str) -> np.random.Generator:
    """A generator seeded independently for one detector artifact mode."""
    return _stage_rng(dp.random_seed, mode_name)


def _record_routed_modes(dp: DetectorParams, records: dict[str, Any]) -> None:
    """Record the routed-only detector modes (bias, bloom, quantization) as active.

    These fold their override into a flat DetectorParams field in the resolver,
    so the chain applies them through the generic mechanic; their truth record
    is the resolved config, marked active.
    """
    for name in (
        'bias_structure',
        'bloom',
        'quantization_lut',
        'quantization_ls8b',
        'contouring_8bit',
    ):
        if name in dp.detector_modes and name not in records:
            records[name] = {'active': True, **dp.detector_modes[name]}


def apply_saturation(
    electrons: NDArrayFloatType,
    *,
    full_well_e: float,
    bloom_length: int = 0,
) -> None:
    """Cap the electron image at the full well, optionally blooming along columns.

    Charge above ``full_well_e`` spills along the column (the v axis) up to
    ``bloom_length`` pixels each way, conserving the total excess, before every
    pixel is capped at the full well.  A saturated star therefore blooms into a
    vertical streak the way it does on cameras with column bleed, and the capped
    pixels read ``full_well_e / gain`` DN after conversion (below the ADC clip on
    an antiblooming-free camera such as Cassini's NAC).

    Parameters:
        electrons: The electron image, modified in place.
        full_well_e: The full-well ceiling in electrons.
        bloom_length: Column-bloom half-length in pixels; 0 disables bloom.
    """
    if bloom_length > 0:
        excess = np.maximum(electrons - full_well_e, 0.0)
        if float(excess.max()) > 0.0:
            width = 2 * bloom_length + 1
            spread = ndimage.uniform_filter1d(excess, size=width, axis=0, mode='constant')
            np.minimum(electrons, full_well_e, out=electrons)
            electrons += spread
    np.minimum(electrons, full_well_e, out=electrons)


def quantize_dn(
    dn: NDArrayFloatType,
    *,
    mode: str,
    saturation_dn: float,
    contour_step: int = 8,
) -> NDArrayFloatType:
    """Quantize a DN image by the selected ADC sub-mode.

    Parameters:
        dn: The DN image (float).
        mode: 'exact' (round to integer, uniform bins), '8bit' (integer bins
            with a hard 255 code ceiling), 'uneven_12bit' (integer bins with
            histogram spikes at the power-of-two bit boundaries), 'sqrt_lut'
            (square-root companding to 8 bits and back, leaving a
            signal-dependent residual), 'ls8b' (low 8 bits kept: values above
            255 wrap modulo 256, the banded wraparound on bright targets), or
            'contour_8bit' (8-bit output posterized to multiples of
            ``contour_step``, the uneven-bin contouring of a coarse ADC).
        saturation_dn: The ADC ceiling, used to scale the companding LUT.
        contour_step: The DN posterization step for the 'contour_8bit' mode.

    Returns:
        The quantized DN image.

    Raises:
        ValueError: If ``mode`` is not a known quantization sub-mode.
    """
    if mode == 'exact':
        return np.rint(dn)
    if mode == 'ls8b':
        # LS8B telemetry mode: only the low 8 bits are kept, so a value above
        # 255 wraps modulo 256 -- the banded wraparound a bright target shows.
        return np.mod(np.rint(dn), 256.0)
    if mode == 'contour_8bit':
        # A coarse ADC with uneven bins is worst at DN multiples of the step:
        # posterizing an 8-bit word to multiples of contour_step reproduces the
        # visible contouring of a smooth low-contrast scene.
        step = max(int(contour_step), 1)
        clipped = np.clip(np.rint(dn), 0.0, 255.0)
        return np.clip(np.round(clipped / step) * step, 0.0, 255.0)
    if mode == '8bit':
        # An 8-bit output word: integer DN clipped at the 255 code ceiling.
        # The ceiling is the word width, not the scene's saturation_dn, so an
        # 8-bit mode on a deeper detector still tops out at 255.
        return np.clip(np.rint(dn), 0.0, 255.0)
    if mode == 'uneven_12bit':
        # Uneven bit weights concentrate codes at the power-of-two carry
        # boundaries: values within one DN of a 2^m boundary snap to it, which
        # is the histogram-spike signature of an ADC with unequal bit weights.
        quantized = np.rint(dn)
        max_bit = int(np.log2(max(saturation_dn, 2.0))) + 1
        for m in range(1, max_bit):
            boundary = float(2**m)
            near = np.abs(quantized - boundary) <= 1.0
            quantized[near] = boundary
        return quantized
    if mode == 'sqrt_lut':
        # Square-root companding: encode to 8 bits through a sqrt LUT and decode
        # back, so the quantization step tracks the photon noise and the inverse
        # leaves a residual that grows with signal.
        ceiling = max(saturation_dn, 1.0)
        clipped = np.clip(dn, 0.0, ceiling)
        code = np.rint(255.0 * np.sqrt(clipped / ceiling))
        return ceiling * (code / 255.0) ** 2
    raise ValueError(f'unknown quantization mode {mode!r}')


def _convert_to_electrons(signal: NDArrayFloatType, dp: DetectorParams) -> NDArrayFloatType:
    """Convert the intensive [0, 1] signal to electrons through the exposure."""
    scale = (
        dp.signal_full_scale_frac
        * dp.full_well_e
        * (dp.exposure_sec / dp.exposure_ref_sec if dp.exposure_ref_sec > 0.0 else 1.0)
    )
    return np.clip(signal, 0.0, 1.0) * scale


def _apply_ccd(frame: SimFrame, dp: DetectorParams, rng: np.random.Generator) -> None:
    """Run the CCD electron unit chain, overwriting the signal plane with DN.

    The registry's detector artifact modes render at their physical point in the
    chain: the fixed-pattern response and the dark ramp are per-pixel / dark
    structure before Poisson; the frame-transfer smear is integrated charge
    before Poisson; the anti-blooming pairs, radiation transients, and column
    bloom follow the shot term; coherent banding is added in electrons before the
    gain divide; and the additive fixed pattern and the saturation serial tail
    ride on the digitized DN before quantization.
    """
    modes = dp.detector_modes
    records: dict[str, Any] = {}
    electrons = _convert_to_electrons(frame.signal, dp)
    # Point sources are already electrons; add them AFTER the intensive
    # conversion and BEFORE Poisson so stars never pass through the signal scale.
    electrons += frame.point_e
    if 'fixed_pattern' in modes:
        cfg = modes['fixed_pattern']
        records['fixed_pattern'] = add_fixed_pattern_response(
            electrons,
            prnu_rms=float(cfg.get('prnu_rms') or 0.0),
            vignetting_frac=float(cfg.get('vignetting_frac') or 0.0),
            dust_donut_count=int(cfg.get('dust_donut_count') or 0),
            rng=_mode_seed(dp, 'fixed_pattern'),
        )
    add_dark_current(
        electrons,
        rate_e_per_sec=dp.dark_current_e_per_sec,
        exposure_sec=dp.exposure_sec,
    )
    if 'dark_ramp' in modes:
        records['dark_ramp'] = _apply_dark_ramp(electrons, modes['dark_ramp'], dp)
    if 'frame_transfer_smear' in modes:
        cfg = modes['frame_transfer_smear']
        records['frame_transfer_smear'] = apply_frame_transfer_smear(
            electrons,
            t_scrub_sec=float(cfg.get('t_scrub_sec') or 0.0),
            t_transfer_sec=float(cfg.get('t_transfer_sec') or 0.0),
            exposure_sec=dp.exposure_sec,
        )
    if dp.poisson:
        electrons = rng.poisson(np.maximum(electrons, 0.0)).astype(np.float64)
    if 'bright_dark_pairs' in modes:
        records['bright_dark_pairs'] = _apply_bright_dark_pairs(
            electrons, modes['bright_dark_pairs'], frame, dp
        )
    hot_record = add_hot_pixels(
        electrons,
        fraction=dp.hot_pixel_fraction,
        amplitude_e=dp.hot_pixel_amplitude_e,
        column_factor=dp.hot_pixel_column_factor,
        rng=_stage_rng(dp.random_seed, 'hot_pixels'),
        candidate_pool=_hot_pixel_pool(frame, dp),
    )
    # The hot_pixels artifact mode is routed through the resolver's flat knobs,
    # so the chain records its realized population here; the generic noise-block
    # stress path stays unrecorded like the other generic knobs.
    if dp.hot_pixel_mode_active:
        records['hot_pixels'] = hot_record
    if 'radiation_transients' in modes:
        records['radiation_transients'] = _apply_radiation_transients(
            electrons, modes['radiation_transients'], dp
        )
    apply_saturation(electrons, full_well_e=dp.full_well_e, bloom_length=dp.bloom_length)
    # Cosmic rays deposit well above the full well AFTER the bloom cap, so they
    # reach the ADC ceiling and land on the orchestrator's masks.
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=dp.cosmic_ray_events_per_pixel_per_sec,
        exposure_sec=dp.exposure_sec,
        amplitude_e=dp.full_well_e,
        rng=_stage_rng(dp.random_seed, 'cosmic_rays'),
    )
    if dp.read_noise_e > 0.0:
        electrons += rng.normal(0.0, dp.read_noise_e, size=electrons.shape)
    if 'banding_coherent' in modes:
        cfg = modes['banding_coherent']
        records['banding_coherent'] = add_coherent_banding(
            electrons,
            amplitude_e=float(cfg.get('amplitude_e') or 0.0),
            period_px=float(cfg.get('period_px') or 0.0),
            orientation=str(cfg.get('orientation', 'horizontal')),
            freq_step_factor=float(cfg.get('freq_step_factor', 1.0)),
            dark_step_dn=float(cfg.get('dark_step_dn', 0.0)),
            gain_e_per_dn=dp.gain_e_per_dn,
            rng=_mode_seed(dp, 'banding_coherent'),
        )
    else:
        add_banding(
            electrons,
            amplitude_e=dp.banding_amplitude_e,
            period_px=dp.banding_period_px,
            rng=_stage_rng(dp.random_seed, 'banding'),
        )
    dn = electrons / dp.gain_e_per_dn + dp.bias_dn
    add_bias_structure(
        dn,
        pedestal_sigma_dn=dp.bias_pedestal_sigma_dn,
        row_gradient_dn=dp.bias_row_gradient_dn,
        col_gradient_dn=dp.bias_col_gradient_dn,
        rng=_stage_rng(dp.random_seed, 'bias_structure'),
    )
    if 'fixed_pattern' in modes:
        cfg = modes['fixed_pattern']
        dn_record = add_fixed_pattern_dn(
            dn,
            stitch_period_px=int(cfg.get('stitch_period_px') or 0),
            stitch_amplitude_dn=float(cfg.get('stitch_amplitude_dn') or 0.0),
            jail_bar_dn=float(cfg.get('jail_bar_dn') or 0.0),
            rng=_mode_seed(dp, 'fixed_pattern_dn'),
        )
        records['fixed_pattern'] = {**records.get('fixed_pattern', {}), **dn_record, 'active': True}
    if 'serial_tail' in modes:
        cfg = modes['serial_tail']
        records['serial_tail'] = add_serial_tail(
            dn,
            saturation_dn=dp.saturation_dn,
            saturation_frac=float(cfg.get('saturation_frac', 0.99)),
            amplitude_dn=float(cfg.get('amplitude_dn') or 0.0),
            length_px=int(cfg.get('length_px') or 0),
            direction=str(cfg.get('direction', 'right')),
        )
    dn = quantize_dn(
        dn,
        mode=dp.quantization,
        saturation_dn=dp.saturation_dn,
        contour_step=dp.quantization_contour_step,
    )
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn
    _record_routed_modes(dp, records)
    if records:
        frame.truth.setdefault('artifacts', {}).update(records)


def _apply_dark_ramp(
    electrons: NDArrayFloatType, cfg: dict[str, Any], dp: DetectorParams
) -> dict[str, Any]:
    """Apply the dark_ramp mode: an additive dark gradient or a shutter shading."""
    if str(cfg.get('kind', 'dark_gradient')) == 'exposure_shading':
        return apply_exposure_shading(
            electrons,
            top_factor=float(cfg.get('top_factor') or 0.0),
            bottom_factor=float(cfg.get('bottom_factor') or 0.0),
        )
    rbi = float(cfg.get('rbi_column_factor') or 0.0)
    hot_columns = None
    if rbi > 0.0:
        # The RBI flavor enhances the columns read out above hot pixels; a seeded
        # set of columns stands in for that population so the flavor is stable
        # per seed and independent of the hot-pixel realization.
        col_rng = _mode_seed(dp, 'dark_ramp_columns')
        n_cols = max(1, electrons.shape[1] // 32)
        hot_columns = col_rng.integers(0, electrons.shape[1], size=n_cols)
    return add_dark_ramp(
        electrons,
        amplitude_e=float(cfg.get('amplitude_e') or 0.0),
        nonlinear=float(cfg.get('nonlinear', 1.0)),
        rbi_column_factor=rbi,
        hot_columns=hot_columns,
    )


def _apply_bright_dark_pairs(
    electrons: NDArrayFloatType, cfg: dict[str, Any], frame: SimFrame, dp: DetectorParams
) -> dict[str, Any]:
    """Apply the bright_dark_pairs mode with exposure scaling and placement pool.

    ``expected_events_per_sec`` is a rate per second of exposure, so the frame's
    expected pair count is that rate times the exposure time.
    """
    rng = _mode_seed(dp, 'bright_dark_pairs')
    expected = float(cfg['expected_events_per_sec']) * dp.exposure_sec
    count = int(rng.poisson(expected)) if expected > 0.0 else 0
    pool = _feature_pool(frame) if dp.artifacts_adversarial else None
    return add_bright_dark_pairs(
        electrons,
        count=count,
        amplitude_e=float(cfg.get('amplitude_e') or 0.0),
        rng=rng,
        candidate_pool=pool,
    )


def _apply_radiation_transients(
    electrons: NDArrayFloatType, cfg: dict[str, Any], dp: DetectorParams
) -> dict[str, Any]:
    """Apply radiation_transients: cosmic morphology in the Galileo-scaled regime.

    ``expected_events_per_sec`` is a hit rate per second the detector is
    exposed to the environment, scaled by the environment factor.  The detector
    collects hits during the exposure and during the readout dwell -- the dwell
    adds transients in proportion to its length, the regime that forced
    Galileo's faster summation readout -- so the expected spike count is
    ``expected_events_per_sec * environment_factor * (exposure_sec +
    readout_dwell_sec)``.  A non-positive exposure contributes zero time, so
    such a frame collects only the dwell's hits (the scene validator refuses a
    non-positive ``exposure_sec``, so only a direct caller can pass one).
    """
    rate = float(cfg['expected_events_per_sec'])
    environment = float(cfg.get('environment_factor', 1.0))
    dwell = float(cfg.get('readout_dwell_sec', 0.0))
    exposure = max(dp.exposure_sec, 0.0)
    expected = rate * environment * (exposure + dwell)
    amplitude = cfg.get('amplitude_e')
    amplitude_e = float(amplitude) if amplitude is not None else dp.full_well_e
    rng = _mode_seed(dp, 'radiation_transients')
    n_events = int(rng.poisson(expected)) if expected > 0.0 else 0
    deposit_morphological_events(
        electrons,
        n_events=n_events,
        amplitude_e=amplitude_e,
        rng=rng,
        amplitude_dist='exponential',
    )
    return {'active': n_events > 0, 'expected': expected, 'events': n_events}


def _apply_vidicon(frame: SimFrame, dp: DetectorParams, rng: np.random.Generator) -> None:
    """Run the vidicon DN chain: no electron conversion, DN-domain noise.

    The vidicon is not photon-noise dominated, so its signal maps straight to
    the 8-bit DN full scale and the noise is applied in DN: a line-correlated
    read-noise term (a per-line offset plus a within-line white component) and a
    faint vertical coherent periodic component, then 8-bit quantization.  The
    chain lacks a cosmic-ray stage, so the scene validator refuses a
    non-zero ``noise.cosmic_ray_events_per_pixel_per_sec`` on this path.
    """
    modes = dp.detector_modes
    records: dict[str, Any] = {}
    signal_dn = np.clip(frame.signal, 0.0, 1.0) * dp.signal_full_scale_frac * dp.full_well_dn
    size_v, size_u = signal_dn.shape
    # Point sources on the vidicon are already DN (the vidicon has no electron
    # domain): add them onto the converted signal before the DN-domain noise, so
    # the read-noise and coherent terms ride on the star cores as they do on the
    # extended signal.  The optics PSF has already shaped them.
    dn = signal_dn
    dn += frame.point_e
    # The erase-cycle residual image is a pre-noise ghost of the prior frame.
    if 'residual_image' in modes:
        cfg = modes['residual_image']
        offset = cfg.get('offset_px') or [5, 5]
        records['residual_image'] = add_residual_image(
            dn,
            amplitude=float(cfg.get('amplitude') or 0.0),
            prior=str(cfg.get('prior', 'self_offset')),
            offset_v=int(offset[0]),
            offset_u=int(offset[1]),
        )
    vidicon = dp.vidicon
    line_sigma = float(vidicon.get('read_noise_line_dn', 0.0))
    pixel_sigma = float(vidicon.get('read_noise_pixel_dn', 0.0))
    if line_sigma > 0.0:
        dn = dn + rng.normal(0.0, line_sigma, size=(size_v,))[:, None]
    if pixel_sigma > 0.0:
        dn = dn + rng.normal(0.0, pixel_sigma, size=(size_v, size_u))
    coherent_amp = float(vidicon.get('coherent_amplitude_dn', 0.0))
    coherent_period = float(vidicon.get('coherent_period_px', 0.0))
    if coherent_amp > 0.0 and coherent_period > 0.0:
        phase = float(_stage_rng(dp.random_seed, 'vidicon_coherent').uniform(0.0, 2.0 * np.pi))
        cols = np.arange(size_u, dtype=np.float64)
        coherent = coherent_amp * np.sin(2.0 * np.pi * cols / coherent_period + phase)
        dn = dn + coherent[None, :]
    # The readout dark-current line ramp accumulates down the frame (DN domain).
    if 'dark_ramp' in modes:
        records['dark_ramp'] = _apply_dark_ramp(dn, modes['dark_ramp'], dp)
    # Explicit coherent banding rides on the DN signal (gain 1 in the DN domain).
    if 'banding_coherent' in modes:
        cfg = modes['banding_coherent']
        records['banding_coherent'] = add_coherent_banding(
            dn,
            amplitude_e=float(cfg.get('amplitude_e') or 0.0),
            period_px=float(cfg.get('period_px') or 0.0),
            orientation=str(cfg.get('orientation', 'horizontal')),
            freq_step_factor=float(cfg.get('freq_step_factor', 1.0)),
            dark_step_dn=float(cfg.get('dark_step_dn', 0.0)),
            gain_e_per_dn=1.0,
            rng=_mode_seed(dp, 'banding_coherent'),
        )
    if 'bias_structure' in modes:
        add_bias_structure(
            dn,
            pedestal_sigma_dn=dp.bias_pedestal_sigma_dn,
            row_gradient_dn=dp.bias_row_gradient_dn,
            col_gradient_dn=dp.bias_col_gradient_dn,
            rng=_stage_rng(dp.random_seed, 'bias_structure'),
        )
    # The brightness-dependent beam bend warps the formed image before digitizing.
    if 'beam_bend' in modes:
        records['beam_bend'] = apply_beam_bend(
            dn, amplitude_px=float(modes['beam_bend'].get('amplitude_px') or 0.0)
        )
    dn = dn + dp.bias_dn
    dn = quantize_dn(
        dn,
        mode=dp.quantization,
        saturation_dn=dp.saturation_dn,
        contour_step=dp.quantization_contour_step,
    )
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn
    _record_routed_modes(dp, records)
    if records:
        frame.truth.setdefault('artifacts', {}).update(records)


def _apply_calibration_inverse(frame: SimFrame, dp: DetectorParams) -> None:
    """Invert the calibration transform: DN to I/F, bias and dark subtracted first.

    Matches the real pipeline: the bias pedestal and dark pedestal are subtracted
    before the exposure divide, so a calibrated frame carries no spurious 1/
    exposure pedestal.  The calibration scale is derived so a noise-free signal
    of 1.0 at the reference exposure round-trips to I/F 1.0.  The calibrated sky
    residual (``calibrated_sky_if``) is then added to every pixel.  A real
    calibrated product's dark sky keeps a residual level, and the residual keeps
    the simulated dark sky off the missing-data marker, which is 0 on every
    calibrated instrument: without it a noise-free dark sky would land on exactly
    0 I/F, and the navigator would count it as missing data.

    A pixel that still lands exactly on a numeric marker moves to the next float
    above it, so no pixel the detector delivers reads as missing data.  The
    digitized DN are integers, so the calibrated values fall on a lattice, and
    for some exposures the subtracted dark pedestal puts a lattice point on the
    marker; a real calibrated product is continuous (its flat field and per-pixel
    dark model see to that) and never carries an exact 0 for a valid pixel.
    """
    exposure = dp.exposure_sec if dp.exposure_sec > 0.0 else 1.0
    denom = dp.calibration_scale_dn_per_s_per_if * exposure
    if denom <= 0.0:
        return
    calibrated = (frame.signal - dp.bias_dn - dp.dark_dn) / denom + dp.calibrated_sky_if
    if not np.isnan(dp.marker_value):
        calibrated[calibrated == dp.marker_value] = np.nextafter(dp.marker_value, np.inf)
    frame.signal[:] = calibrated


def apply_detector(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Detector stage: convert the composed signal to detector counts in place.

    All signal is composed before this stage runs, so the shot term sees the
    noise-free signal it should grow with.  Feature masks in ``frame.truth`` were
    derived from the noise-free signal and are unaffected.

    Parameters:
        frame: The frame whose signal plane is converted in place.
        params: The full scene mapping; resolved into detector parameters via
            :func:`spindoctor.sim.forward.detector.params.resolve_detector_params`.
        rng: The stage generator, used for the shot and read-noise streams; the
            structured sub-effects derive their own named streams from the scene
            seed.
    """
    dp = resolve_detector_params(params)
    if dp.detector_model == 'vidicon':
        _apply_vidicon(frame, dp, rng)
    else:
        _apply_ccd(frame, dp, rng)
    if dp.data_units != 'raw_dn':
        _apply_calibration_inverse(frame, dp)
