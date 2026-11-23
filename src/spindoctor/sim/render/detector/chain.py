"""The detector stage: composed signal to digitized detector counts.

This is the normative unit chain of the renderer.  For a CCD the composed
intensive signal is converted to electrons through the exposure, the point-source
electron plane is added, and the frame passes through Poisson shot noise,
electron-domain full-well bloom, read noise, banding, gain to DN, bias
structure, quantization, and the ADC clip.  A camera's physical saturation
therefore emerges from ``full_well_e / gain_e_per_dn`` (below the ADC ceiling for
Cassini), not from the ADC clip.

The shot noise is sampled per pixel
(:func:`~spindoctor.sim.render.detector.shot_noise.sample_poisson`): one uniform
per pixel, turned into a count through that pixel's Poisson distribution.  No
random draw in the chain takes a number of values that depends on the signal, so
changing the signal in one pixel changes the noise drawn for that pixel alone.

The Voyager vidicon skips the electron conversion and applies its noise directly
in DN (line-correlated read noise plus any banding); its point sources (stars) and
its cosmic-ray hits are already DN and are added onto the converted signal before
that DN noise.  A calibrated (I/F) scene renders through the full DN chain and then
inverts the calibration transform, so calibrated products carry propagated shot/read
noise and quantization texture in I/F units.

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

from spindoctor.sim.render.detector.banding import add_banding
from spindoctor.sim.render.detector.bias import add_bias_structure
from spindoctor.sim.render.detector.dark import (
    add_dark_current,
    add_readout_dark,
    readout_dark_profile,
)
from spindoctor.sim.render.detector.detector_maps import detector_column_map
from spindoctor.sim.render.detector.fixed_pattern import (
    add_fixed_pattern_dn,
    add_fixed_pattern_response,
)
from spindoctor.sim.render.detector.params import (
    DetectorParams,
    calibration_map,
    resolve_detector_params,
)
from spindoctor.sim.render.detector.pixel_defects import add_bright_dark_pairs, add_hot_pixels
from spindoctor.sim.render.detector.quantization import quantize_dn
from spindoctor.sim.render.detector.saturation import add_serial_tail, apply_saturation
from spindoctor.sim.render.detector.shot_noise import sample_poisson
from spindoctor.sim.render.detector.shutter import (
    apply_frame_transfer_smear,
    apply_shutter_shading,
)
from spindoctor.sim.render.detector.transients import (
    add_cosmic_rays,
    deposit_morphological_events,
)
from spindoctor.sim.render.detector.vidicon import add_residual_image, apply_beam_bend
from spindoctor.sim.render.feature_loci import dilated_pixels, extract_feature_loci
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['apply_detector', 'keep_off_marker']


# The detector streams that hold a property of the camera rather than of one frame.
# A real camera shows the same hot pixels, the same per-pixel response, the same
# column offsets, and the same warm columns in every frame, so these streams draw
# from the scene's random_seed even when the scene sets a noise_seed.  Every other
# detector stream is drawn fresh for each frame and follows noise_seed.
_CAMERA_STREAMS: frozenset[str] = frozenset(
    {'hot_pixels', 'fixed_pattern', 'fixed_pattern_dn', 'readout_dark_columns'}
)


def _stream_rng(dp: DetectorParams, effect: str) -> np.random.Generator:
    """A generator for one detector sub-effect, seeded independently by name.

    The stream is named ``detector/<effect>``.  A fixed-pattern stream derives from
    the scene's ``random_seed`` and every per-frame stream from its ``noise_seed``,
    so a scene that sets ``noise_seed`` draws new noise on the same camera.
    """
    seed = dp.random_seed if effect in _CAMERA_STREAMS else dp.noise_seed
    return np.random.default_rng(derive_effect_seed(seed, f'detector/{effect}'))


def _feature_pool(
    frame: SimFrame, dp: DetectorParams, mode_name: str
) -> tuple[NDArrayIntType, NDArrayIntType] | None:
    """The pixels a mode places its events among when it places them adversarially.

    Returns the dilated navigation-feature pixels if the mode is one of
    ``dp.adversarial_modes``, and None, for the mode's own placement, otherwise.
    """
    if mode_name not in dp.adversarial_modes:
        return None
    loci = extract_feature_loci(frame.truth, frame.signal.shape)
    return dilated_pixels(loci, radius=3, shape=frame.signal.shape)


def _convert_to_electrons(signal: NDArrayFloatType, dp: DetectorParams) -> NDArrayFloatType:
    """Convert the intensive [0, 1] signal to electrons through the exposure.

    A signal of 1.0 fills ``signal_full_scale_frac`` of a native pixel's well at
    the reference exposure; a binned frame pixel sums ``binning**2`` native
    pixels of it.
    """
    scale = (
        dp.signal_full_scale_frac
        * dp.full_well_e
        * dp.binning**2
        * (dp.exposure_sec / dp.exposure_ref_sec if dp.exposure_ref_sec > 0.0 else 1.0)
    )
    return np.clip(signal, 0.0, 1.0) * scale


def _apply_ccd(frame: SimFrame, dp: DetectorParams, rng: np.random.Generator) -> None:
    """Run the CCD electron unit chain, overwriting the signal plane with DN.

    The registry's detector artifact modes render at their physical point in the
    chain: the shutter shading and the fixed-pattern response scale the light, and
    the readout dark joins the dark signal, before Poisson; the frame-transfer smear
    is integrated charge before Poisson; the anti-blooming pairs, radiation
    transients, and column bloom follow the shot term; banding is added in electrons
    before the gain divide; and the additive fixed pattern and the saturation serial
    tail ride on the digitized DN before quantization.
    """
    modes = dp.detector_modes
    records: dict[str, Any] = {}
    electrons = _convert_to_electrons(frame.signal, dp)
    # Point sources are already electrons; add them AFTER the intensive
    # conversion and BEFORE Poisson so stars never pass through the signal scale.
    electrons += frame.point_e
    if 'shutter_shading' in modes:
        records['shutter_shading'] = _apply_shutter_shading(electrons, modes['shutter_shading'])
    if 'fixed_pattern' in modes:
        cfg = modes['fixed_pattern']
        records['fixed_pattern'] = add_fixed_pattern_response(
            electrons,
            prnu_rms=float(cfg.get('prnu_rms') or 0.0),
            vignetting_frac=float(cfg.get('vignetting_frac') or 0.0),
            dust_donut_count=int(cfg.get('dust_donut_count') or 0),
            rng=_stream_rng(dp, 'fixed_pattern'),
        )
    add_dark_current(
        electrons,
        rate_e_per_sec=dp.dark_current_e_per_sec,
        exposure_sec=dp.exposure_sec,
    )
    if 'readout_dark' in modes:
        records['readout_dark'] = _apply_readout_dark(electrons, modes['readout_dark'], dp)
    if 'frame_transfer_smear' in modes:
        cfg = modes['frame_transfer_smear']
        records['frame_transfer_smear'] = apply_frame_transfer_smear(
            electrons,
            t_scrub_sec=float(cfg.get('t_scrub_sec') or 0.0),
            t_transfer_sec=float(cfg.get('t_transfer_sec') or 0.0),
            exposure_sec=dp.exposure_sec,
        )
    if dp.poisson:
        # One uniform per pixel, whatever the means, so a change to one pixel's
        # mean (a zero-crossing convolution tail, a last-bit FFT difference)
        # changes only that pixel's count and leaves the read noise drawn after
        # it in place (#911).
        electrons = sample_poisson(rng, electrons)
    if 'bright_dark_pairs' in modes:
        records['bright_dark_pairs'] = _apply_bright_dark_pairs(
            electrons, modes['bright_dark_pairs'], frame, dp
        )
    if 'hot_pixels' in modes:
        records['hot_pixels'] = _apply_hot_pixels(electrons, modes['hot_pixels'], frame, dp)
    if 'radiation_transients' in modes:
        records['radiation_transients'] = _apply_radiation_transients(
            electrons, modes['radiation_transients'], dp
        )
    # A binned pixel holds its block's binning**2 native wells, the charge the
    # signal scale and the I/F calibration both assume.
    bloom_length = 0
    if 'bloom' in modes:
        bloom_length = int(modes['bloom']['bloom_length'] or 0)
        records['bloom'] = {'active': bloom_length > 0, 'bloom_length': bloom_length}
    apply_saturation(
        electrons, full_well_e=dp.full_well_e * dp.binning**2, bloom_length=bloom_length
    )
    # Cosmic rays deposit AFTER the bloom cap, so a full-well-scale hit reaches
    # the ADC ceiling and lands on the orchestrator's masks; a modest hit stays
    # below it, as most real ones do.
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=dp.cosmic_ray_events_per_pixel_per_sec,
        exposure_sec=dp.exposure_sec,
        amplitude_e=dp.cosmic_ray_amplitude_e,
        rng=_stream_rng(dp, 'cosmic_rays'),
        readout_dwell_sec=dp.cosmic_ray_readout_dwell_sec,
        spread_px=dp.cosmic_ray_spread_px,
    )
    if dp.read_noise_e > 0.0:
        electrons += rng.normal(0.0, dp.read_noise_e, size=electrons.shape)
    if 'banding' in modes:
        records['banding'] = _apply_banding(
            electrons, modes['banding'], dp, gain_e_per_dn=dp.gain_e_per_dn
        )
    dn = electrons / dp.gain_e_per_dn + dp.bias_dn
    if 'bias_structure' in modes:
        records['bias_structure'] = _apply_bias_structure(dn, modes['bias_structure'], dp)
    if 'fixed_pattern' in modes:
        cfg = modes['fixed_pattern']
        dn_record = add_fixed_pattern_dn(
            dn,
            stitch_period_px=int(cfg.get('stitch_period_px') or 0),
            stitch_amplitude_dn=float(cfg.get('stitch_amplitude_dn') or 0.0),
            jail_bar_dn=float(cfg.get('jail_bar_dn') or 0.0),
            rng=_stream_rng(dp, 'fixed_pattern_dn'),
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
        sqrt_lut=dp.sqrt_lut,
    )
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn
    if records:
        frame.truth.setdefault('artifacts', {}).update(records)


def _apply_banding(
    plane: NDArrayFloatType, cfg: dict[str, Any], dp: DetectorParams, *, gain_e_per_dn: float
) -> dict[str, Any]:
    """Apply the banding mode to a plane that holds ``gain_e_per_dn`` units per DN."""
    return add_banding(
        plane,
        amplitude=float(cfg['amplitude'] or 0.0),
        period_px=float(cfg['period_px'] or 0.0),
        orientation=str(cfg['orientation']),
        horizontal_pattern=str(cfg['horizontal_pattern']),
        freq_step_factor=float(cfg['freq_step_factor']),
        dark_step_dn=float(cfg['dark_step_dn']),
        ringing_amplitude=float(cfg['ringing_amplitude']),
        ringing_period_px=float(cfg['ringing_period_px']),
        ringing_decay_px=float(cfg['ringing_decay_px']),
        binning=dp.binning,
        gain_e_per_dn=gain_e_per_dn,
        rng=_stream_rng(dp, 'banding'),
    )


def _apply_bias_structure(
    dn: NDArrayFloatType, cfg: dict[str, Any], dp: DetectorParams
) -> dict[str, Any]:
    """Apply the bias_structure mode to the DN image."""
    return add_bias_structure(
        dn,
        pedestal_sigma_dn=float(cfg['pedestal_sigma_dn'] or 0.0),
        row_gradient_dn=float(cfg['row_gradient_dn'] or 0.0),
        col_gradient_dn=float(cfg['col_gradient_dn'] or 0.0),
        rng=_stream_rng(dp, 'bias_structure'),
    )


def _apply_shutter_shading(signal: NDArrayFloatType, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the shutter_shading mode to the collected light."""
    return apply_shutter_shading(
        signal,
        top_factor=float(cfg['top_factor'] or 0.0),
        bottom_factor=float(cfg['bottom_factor'] or 0.0),
    )


def _readout_dark_profile(size_v: int, dp: DetectorParams) -> NDArrayFloatType:
    """The readout dark per line of the frame, zeros when the mode is off."""
    cfg = dp.detector_modes.get('readout_dark')
    if cfg is None:
        return np.zeros(size_v, dtype=np.float64)
    return readout_dark_profile(
        size_v,
        amplitude=float(cfg['amplitude'] or 0.0),
        nonlinear=float(cfg['nonlinear']),
        binning=dp.binning,
    )


def _apply_readout_dark(
    plane: NDArrayFloatType, cfg: dict[str, Any], dp: DetectorParams
) -> dict[str, Any]:
    """Apply the readout_dark mode, with its enhanced columns drawn once per camera.

    The enhanced columns stand in for the ones read out above hot pixels.  They are a
    seeded set of one column in 32, stable per scene seed and independent of the hot
    pixels.
    """
    size_v, size_u = plane.shape
    rbi = float(cfg['rbi_column_factor'])
    rbi_columns = np.zeros(0, dtype=np.int64)
    if rbi > 0.0:
        col_rng = _stream_rng(dp, 'readout_dark_columns')
        rbi_columns = col_rng.integers(0, size_u, size=max(1, size_u // 32))
    return add_readout_dark(
        plane,
        _readout_dark_profile(size_v, dp),
        rbi_column_factor=rbi,
        rbi_columns=rbi_columns,
    )


def _apply_bright_dark_pairs(
    electrons: NDArrayFloatType, cfg: dict[str, Any], frame: SimFrame, dp: DetectorParams
) -> dict[str, Any]:
    """Apply the bright_dark_pairs mode with exposure scaling and placement pool.

    ``expected_events_per_sec`` is a rate per second of exposure, so the frame's
    expected pair count is that rate times the exposure time.
    """
    rng = _stream_rng(dp, 'bright_dark_pairs')
    expected = float(cfg['expected_events_per_sec']) * dp.exposure_sec
    count = int(rng.poisson(expected)) if expected > 0.0 else 0
    return add_bright_dark_pairs(
        electrons,
        count=count,
        amplitude_e=float(cfg.get('amplitude_e') or 0.0),
        rng=rng,
        candidate_pool=_feature_pool(frame, dp, 'bright_dark_pairs'),
    )


def _apply_hot_pixels(
    electrons: NDArrayFloatType, cfg: dict[str, Any], frame: SimFrame, dp: DetectorParams
) -> dict[str, Any]:
    """Apply the hot_pixels mode: the detector's fixed map, or an adversarial draw.

    If the scene sets the mode's ``pixel_fraction`` and turns its adversarial switch on,
    the population is drawn onto the pixels on and beside the navigation features (limb
    and ring-edge arcs and star positions, dilated by a few pixels).  The population the
    instrument defaults turn on stays on the detector's map.
    """
    return add_hot_pixels(
        electrons,
        fraction=float(cfg['pixel_fraction']),
        amplitude_e=float(cfg['amplitude_e'] or 0.0),
        column_factor=float(cfg['column_factor'] or 0.0),
        detector_id=dp.detector_id,
        rng=_stream_rng(dp, 'hot_pixels'),
        candidate_pool=_feature_pool(frame, dp, 'hot_pixels'),
        binning=dp.binning,
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
    rng = _stream_rng(dp, 'radiation_transients')
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
    read-noise term (a per-line offset plus a within-line white component) and the
    banding mode's bands, then 8-bit quantization.

    Cosmic rays (``noise.cosmic_ray_events_per_pixel_per_sec``, with the CCD
    chain's meaning: expected events per pixel per second of exposure) land on
    the photoconductive target, which stores the image as charge until the
    electron beam reads it.  A charged particle crossing the target discharges it
    the way light does, so a hit joins the stored image: after the erase-cycle
    residual (a ghost of the prior frame, which this frame's hits don't reach)
    and before the readout, so the beam's read noise, readout dark, banding, and
    bend all act on it.  Each hit deposits at least the DN full
    scale (``full_well_dn``) and the target saturates there, so the core of a hit
    holds exactly the full scale and digitizes at the ADC ceiling, as a CCD hit
    does.
    """
    modes = dp.detector_modes
    records: dict[str, Any] = {}
    signal_dn = np.clip(frame.signal, 0.0, 1.0) * dp.signal_full_scale_frac * dp.full_well_dn
    size_v, size_u = signal_dn.shape
    # Point sources on the vidicon are already DN (the vidicon has no electron
    # domain): add them onto the converted signal before the DN-domain noise, so
    # the read noise and the bands ride on the star cores as they do on the
    # extended signal.  The optics PSF has already shaped them.
    dn = signal_dn
    dn += frame.point_e
    if 'shutter_shading' in modes:
        records['shutter_shading'] = _apply_shutter_shading(dn, modes['shutter_shading'])
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
    hits = np.zeros_like(dn)
    add_cosmic_rays(
        hits,
        events_per_pixel_per_sec=dp.cosmic_ray_events_per_pixel_per_sec,
        exposure_sec=dp.exposure_sec,
        amplitude_e=dp.full_well_dn,
        rng=_stream_rng(dp, 'cosmic_rays'),
    )
    # The target holds at most its full-scale charge, so a hit saturates the
    # pixels it lands on.
    hit = hits > 0.0
    dn[hit] = np.minimum(dn[hit] + hits[hit], dp.full_well_dn)
    vidicon = dp.vidicon
    line_sigma = float(vidicon.get('read_noise_line_dn', 0.0))
    pixel_sigma = float(vidicon.get('read_noise_pixel_dn', 0.0))
    if line_sigma > 0.0:
        dn = dn + rng.normal(0.0, line_sigma, size=(size_v,))[:, None]
    if pixel_sigma > 0.0:
        dn = dn + rng.normal(0.0, pixel_sigma, size=(size_v, size_u))
    if 'readout_dark' in modes:
        records['readout_dark'] = _apply_readout_dark(dn, modes['readout_dark'], dp)
    if 'banding' in modes:
        records['banding'] = _apply_banding(dn, modes['banding'], dp, gain_e_per_dn=1.0)
    if 'bias_structure' in modes:
        records['bias_structure'] = _apply_bias_structure(dn, modes['bias_structure'], dp)
    # The brightness-dependent beam bend warps the formed image before digitizing.
    if 'beam_bend' in modes:
        records['beam_bend'] = apply_beam_bend(
            dn,
            amplitude_px=float(modes['beam_bend'].get('amplitude_px') or 0.0),
            full_brightness_dn=dp.signal_full_scale_frac * dp.full_well_dn,
        )
    dn = dn + dp.bias_dn
    dn = quantize_dn(
        dn,
        mode=dp.quantization,
        saturation_dn=dp.saturation_dn,
        contour_step=dp.quantization_contour_step,
        sqrt_lut=dp.sqrt_lut,
    )
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn
    if records:
        frame.truth.setdefault('artifacts', {}).update(records)


def keep_off_marker(values: NDArrayFloatType, marker: float) -> None:
    """Move every value exactly on a numeric marker to the next float above it.

    A calibrated frame's values come from whole DN through an affine map, so
    they fall on a lattice that can cross the marker exactly; a real calibrated
    product is continuous and never carries the marker on a valid pixel.  A NaN
    marker never equals a value, so it leaves ``values`` unchanged.

    Parameters:
        values: The calibrated values, modified in place.
        marker: The frame's missing-data marker, possibly NaN.
    """
    if np.isnan(marker):
        return
    values[values == marker] = np.nextafter(marker, np.inf)


def _subtract_calibration_residual(dn: NDArrayFloatType, dp: DetectorParams) -> None:
    """Subtract the calibrated product's residual structure from the DN in place.

    A real calibration pipeline subtracts per-pixel models (a dark model, a
    per-line bias) from integer DN, so its product is continuous, and the
    models' mismatch leaves structure the raw frame does not carry.  Two terms
    reproduce it, both in electrons per native pixel and converted by the gain:
    a column pattern fixed per detector
    (:func:`~spindoctor.sim.render.detector.detector_maps.detector_column_map`),
    with sigma ``calibration_column_residual_e``, and a per-frame Gaussian per
    pixel, with sigma ``calibration_pixel_residual_e``.  A binned pixel sums its
    native pixels: the ``b`` rows of a column add coherently and the ``b``
    columns independently, and the pixel term adds in quadrature over the
    ``b**2`` native pixels.  Both sigmas 0 is a no-op.

    Parameters:
        dn: The digitized DN image, modified in place.
        dp: The resolved detector parameters.
    """
    column_e = dp.calibration_column_residual_e
    pixel_e = dp.calibration_pixel_residual_e
    if column_e <= 0.0 and pixel_e <= 0.0:
        return
    b = dp.binning
    size_u = dn.shape[1]
    if column_e > 0.0:
        native = detector_column_map(
            size_u * b, effect='calibration_columns', detector_id=dp.detector_id
        )
        columns = native.reshape(size_u, b).sum(axis=1) * b
        dn -= columns[np.newaxis, :] * column_e / dp.gain_e_per_dn
    if pixel_e > 0.0:
        rng = _stream_rng(dp, 'calibration_residual')
        dn -= rng.normal(0.0, pixel_e * b / dp.gain_e_per_dn, size=dn.shape)


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

    Before the divide, the calibration residual
    (:func:`_subtract_calibration_residual`) takes the per-column and per-pixel
    structure out that a real pipeline's per-pixel corrections leave, so the
    calibrated values are continuous like a real product's rather than on the
    integer DN lattice.  A pixel that still lands exactly on a numeric marker
    moves to the next float above it, so no pixel the detector delivers reads as
    missing data: with the residual off, the calibrated values fall on a
    lattice, and for some exposures the subtracted dark pedestal puts a lattice
    point on the marker.
    """
    calibration = calibration_map(dp)
    if calibration is None:
        return
    # The pipeline's dark model removes the readout dark line by line; the enhanced
    # columns, and the shot noise the dark carried, stay in the product.  The
    # vidicon's readout dark is cataloged as what its calibrated product keeps, so
    # a vidicon frame has none to remove.
    if dp.detector_model == 'ccd':
        profile = _readout_dark_profile(frame.signal.shape[0], dp)
        frame.signal -= profile[:, np.newaxis] / dp.gain_e_per_dn
    _subtract_calibration_residual(frame.signal, dp)
    calibrated = calibration.to_if(frame.signal)
    keep_off_marker(calibrated, dp.marker_value)
    frame.signal[:] = calibrated


def apply_detector(frame: SimFrame, *, params: Mapping[str, Any]) -> None:
    """Detector stage: convert the composed signal to detector counts in place.

    All signal is composed before this stage runs, so the shot term sees the
    noise-free signal it should grow with.  Feature masks in ``frame.truth`` were
    derived from the noise-free signal and are unaffected.

    The shot and read noise draw from the stream ``detector``.  The CCD chain draws one
    uniform per pixel for the shot noise and then one normal per pixel for the read
    noise; the vidicon chain draws its line and pixel read noise.  The number of values
    drawn depends only on the frame's dimensions and the enabled terms, never on the
    signal.  Each structured sub-effect draws from its own ``detector/<effect>`` stream.
    The per-frame streams derive from the scene's ``noise_seed`` when it sets one and
    from its ``random_seed`` otherwise.  The fixed-pattern streams (hot pixels, the
    fixed pattern, and the readout dark's enhanced columns) and each artifact mode's
    activation draw always derive from ``random_seed``.

    Parameters:
        frame: The frame whose signal plane is converted in place.
        params: The full scene mapping; resolved into detector parameters via
            :func:`spindoctor.sim.render.detector.params.resolve_detector_params`.
    """
    dp = resolve_detector_params(params)
    # The stream carries the stage's own name and must never be renamed: it seeds the
    # shot and read noise of every committed scene that has them.
    rng = np.random.default_rng(derive_effect_seed(dp.noise_seed, 'detector'))
    if dp.detector_model == 'vidicon':
        _apply_vidicon(frame, dp, rng)
    else:
        _apply_ccd(frame, dp, rng)
    if dp.data_units != 'raw_dn':
        _apply_calibration_inverse(frame, dp)
