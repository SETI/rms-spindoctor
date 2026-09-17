"""Resolved detector parameters for one scene render.

:class:`DetectorParams` collapses the emulated instrument's config block, the
per-instrument catalog defaults (:mod:`spindoctor.sim.forward.artifacts_catalog`),
the scene ``detector`` / ``noise`` blocks, and the ``artifacts.instrument_defaults``
switch into one flat, resolved view the detector stage reads.

Resolution precedence, highest first: an explicit scene key (``detector`` block,
then ``noise`` block), then the catalog value when ``instrument_defaults`` is on,
then the disabled floor (physical-chain artifacts default to zero so an
unconfigured scene renders a clean DN frame, per the stage-activation rule).
``instrument_defaults`` turns on the whole physical chain, including Poisson
shot noise, the catalog's electron-domain full-well bloom, and the catalog's
cohort-measured cosmic-ray rate where one is recorded (radiation on the
detector is physics of the environment, not a transmission defect); the
missing-data loss modes are artifact incidences, not physical-chain noise,
and stay at zero.

The ``noise.read_noise_dn`` key is a DN value, converted to electrons through
the resolved gain, so a scene that pins a DN read-noise level gets exactly that
DN-level behavior out of the electron chain.  ``signal_full_scale_frac`` is the
well fraction a signal of 1.0 fills at the reference exposure; the image-side
DN well is derived (``full_well_e / gain_e_per_dn``) and the navigator-side
``full_well_dn`` config key is a separate published value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifact_modes import (
    DETECTOR_MODE_ORDER,
    mode_available,
    normalize_instrument,
    resolve_mode_config,
)
from spindoctor.sim.forward.artifacts_catalog import (
    resolve_detector_defaults,
    resolve_mode_with_catalog,
)
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.seeds import derive_effect_seed

__all__ = ['DetectorParams', 'resolve_detector_params']


class DetectorParamError(ValueError):
    """Raised when a scene selects an unavailable detector configuration."""


@dataclass
class DetectorParams:
    """The flat, resolved detector view for one render.

    Parameters:
        detector_model: 'ccd' (electron chain) or 'vidicon' (DN chain).
        data_units: 'raw_dn' or 'calibrated_if'.
        signal_full_scale_frac: Well fraction a signal of 1.0 fills at the
            reference exposure.
        full_well_e: Full well in electrons (CCD path).
        exposure_ref_sec: Exposure the well fraction references.
        exposure_sec: The scene exposure.
        gain_e_per_dn: Resolved gain (electrons per DN) for the selected state.
        read_noise_e: Read-noise sigma in electrons (CCD path).
        bias_dn: Additive DN bias pedestal.
        saturation_dn: ADC clip ceiling in DN.
        full_well_dn: Published ADC-referenced well (vidicon DN full scale).
        quantization: ADC quantization sub-mode.
        poisson: Whether shot noise is applied.
        bloom_length: Electron-domain full-well bloom half-length (0 disables).
        cosmic_ray_rate_per_sec: Cosmic-ray fluence (events / cm^2 / sec).
        pixel_area_cm2: Detector pixel area (scales the cosmic-ray count).
        dark_current_e_per_sec: Dark current (electrons / sec); 0 disables.
        hot_pixel_fraction: Fraction of pixels that are hot; 0 disables.
        hot_pixel_amplitude_e: Hot-pixel amplitude scale in electrons.
        hot_pixel_column_factor: Warm-column fraction bled from a hot pixel.
        banding_amplitude_e: Coherent-banding amplitude in electrons; 0 disables.
        banding_period_px: Coherent-banding spatial period in pixels.
        bias_pedestal_sigma_dn: Per-image bias-pedestal jitter (DN); 0 disables.
        bias_row_gradient_dn: Low-order row bias gradient span (DN).
        bias_col_gradient_dn: Low-order column bias gradient span (DN).
        vidicon: The vidicon DN-noise sub-parameters (vidicon path only).
        calibration_scale_dn_per_s_per_if: Derived I/F calibration scale.
        dark_dn: Dark pedestal in DN subtracted before the I/F divide.
        random_seed: The scene seed for the per-effect sub-streams.
        instrument_defaults: Whether the physical-chain opt-in is on.
        hot_pixel_adversarial: Whether the hot-pixel population is placed
            adversarially (biased onto the navigation features) rather than
            uniformly.
        hot_pixel_mode_active: Whether the ``hot_pixels`` artifact mode drove
            the hot-pixel knobs (in which case the chain records the realized
            population in the frame truth), as opposed to the generic
            noise-block / instrument_defaults path.
        quantization_contour_step: The DN posterization step for the
            ``contour_8bit`` quantization sub-mode.
        artifacts_adversarial: Whether stochastic detector artifact modes place
            their events adversarially onto the navigation features.
        detector_modes: The resolved config of every active detector-stage
            artifact mode (registry defaults filled from the catalog), keyed by
            mode name; the chain applies each at its physical point and records
            it.  Routed modes (bias, bloom, quantization) also appear here for
            the truth record, their override already folded into the flat fields.
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
    bloom_length: int
    cosmic_ray_rate_per_sec: float
    pixel_area_cm2: float
    dark_current_e_per_sec: float
    hot_pixel_fraction: float
    hot_pixel_amplitude_e: float
    hot_pixel_column_factor: float
    banding_amplitude_e: float
    banding_period_px: float
    bias_pedestal_sigma_dn: float
    bias_row_gradient_dn: float
    bias_col_gradient_dn: float
    vidicon: dict[str, float]
    calibration_scale_dn_per_s_per_if: float
    dark_dn: float
    random_seed: int = 42
    instrument_defaults: bool = False
    hot_pixel_adversarial: bool = False
    hot_pixel_mode_active: bool = False
    quantization_contour_step: int = 8
    artifacts_adversarial: bool = False
    detector_modes: dict[str, dict[str, Any]] = field(default_factory=dict)


def _default_or_zero(
    scene_noise: Mapping[str, Any],
    catalog: Mapping[str, Any],
    key: str,
    *,
    instrument_defaults: bool,
) -> float:
    """A physical-chain knob: scene override, else catalog (if on), else 0."""
    if key in scene_noise:
        return float(scene_noise[key])
    if instrument_defaults:
        return float(catalog.get(key, 0.0))
    return 0.0


def _resolve_gain(
    catalog: Mapping[str, Any], detector_block: Mapping[str, Any], *, instrument: str | None
) -> float:
    """Select ``gain_e_per_dn`` from the catalog table by the scene gain state.

    Parameters:
        catalog: The instrument's ``DETECTOR_DEFAULTS`` entry.
        detector_block: The scene ``detector`` block (may set ``gain_state``).
        instrument: The sim instrument name, for error messages.

    Returns:
        The electrons-per-DN gain for the selected state.

    Raises:
        DetectorParamError: If the selected state has no catalog entry.
    """
    table = catalog.get('gain_e_per_dn_by_state') or {0: 1.0}
    state = detector_block.get('gain_state')
    if state is None:
        state = int(catalog.get('default_gain_state', 0))
    state = int(state)
    if state not in table:
        raise DetectorParamError(
            f'sim instrument {instrument!r} has no cataloged gain state {state}; '
            f'available states: {sorted(table)}'
        )
    return float(table[state])


# Detector modes routed to an existing generic mechanic by overriding a flat
# DetectorParams field rather than by a dedicated chain call.
_ROUTED_ONLY_MODES: frozenset[str] = frozenset(
    {'bias_structure', 'bloom', 'quantization_lut', 'quantization_ls8b', 'contouring_8bit'}
)
# Detector modes whose incidence is an expected event count (drawn Poisson in
# the mechanic), not a per-frame activation probability.
_COUNT_DETECTOR_MODES: frozenset[str] = frozenset({'radiation_transients', 'bright_dark_pairs'})


def _mode_active(mode_name: str, incidence: float, random_seed: int) -> bool:
    """Whether a detector artifact mode is active this frame.

    Count modes are active whenever their incidence is positive (the mechanic
    draws the event count); the deterministic-shape modes activate with
    probability ``incidence``, drawn from the mode's own seeded stream so
    toggling one never perturbs another.
    """
    if incidence <= 0.0:
        return False
    if mode_name in _COUNT_DETECTOR_MODES:
        return True
    rng = np.random.default_rng(derive_effect_seed(random_seed, f'detector/{mode_name}/activate'))
    return bool(rng.random() < incidence)


def _resolve_detector_modes(
    artifacts: Mapping[str, Any],
    instrument: str | None,
    random_seed: int,
    *,
    instrument_defaults: bool,
) -> dict[str, dict[str, Any]]:
    """Resolve the active detector-stage artifact modes for a render.

    Each present, available, and active mode is resolved (scene value over
    catalog default over registry default) and returned keyed by name.  LORRI's
    ``frame_transfer_smear`` is the one physical-signal-chain member turned on
    under ``instrument_defaults``: when a LORRI scene has not set it, it is
    injected at incidence 1 with the catalog's nominal scrub/transfer times.
    The injection is LORRI-only; the instrument-agnostic generic block accepts
    every mode but emulates no frame-transfer camera, so it gets no injection.
    """
    modes: dict[str, dict[str, Any]] = {}
    for name in DETECTOR_MODE_ORDER:
        raw = artifacts.get(name)
        if raw is None:
            if (
                name == 'frame_transfer_smear'
                and instrument_defaults
                and normalize_instrument(instrument) == 'nhlorri'
            ):
                raw = {'incidence': 1.0}
            else:
                continue
        if not isinstance(raw, Mapping) or not mode_available(name, instrument):
            continue
        cfg = resolve_mode_with_catalog(name, raw, instrument)
        if not _mode_active(name, float(cfg.get('incidence', 0.0)), random_seed):
            continue
        modes[name] = cfg
    return modes


def resolve_detector_params(params: Mapping[str, Any]) -> DetectorParams:
    """Collapse the scene, config, and catalog into a resolved detector view.

    Parameters:
        params: The full scene ``sim_params`` mapping.

    Returns:
        The resolved :class:`DetectorParams`.

    Raises:
        DetectorParamError: If the scene selects an unavailable gain state.
    """
    instrument = params.get('instrument')
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, instrument, params.get('instrument_config')
    )
    inst_noise = inst_config.get('noise') or {}
    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    catalog = resolve_detector_defaults(instrument)

    detector_block = params.get('detector') or {}
    scene_noise = params.get('noise') or {}
    artifacts = params.get('artifacts') or {}
    instrument_defaults = bool(artifacts.get('instrument_defaults', False))

    detector_model = str(detector_block.get('detector_model', catalog['detector_model']))
    data_units = str(inst_config.get('data_units', 'raw_dn'))
    exposure_sec = float(params.get('exposure_sec', 1.0))
    exposure_ref_sec = float(
        detector_block.get('exposure_ref_sec', catalog.get('exposure_ref_sec', 1.0))
    )
    quantization = str(detector_block.get('quantization', catalog.get('quantization', 'exact')))

    signal_full_scale_frac = float(
        scene_noise.get(
            'signal_full_scale_frac',
            inst_noise.get('signal_full_scale_frac', sim_noise['signal_full_scale_frac']),
        )
    )
    bias_dn = float(
        scene_noise.get('bias_dn', catalog.get('bias_dn', sim_noise.get('bias_dn', 0.0)))
    )
    saturation_dn = float(inst_noise.get('saturation_dn', sim_noise['saturation_dn']))
    full_well_dn = float(inst_noise.get('full_well_dn', sim_noise['full_well_dn']))

    gain_e_per_dn = _resolve_gain(catalog, detector_block, instrument=instrument)
    full_well_e = float(catalog.get('full_well_e', full_well_dn * gain_e_per_dn))

    # Read noise: a scene DN override wins (converted to electrons via gain);
    # otherwise the catalog electrons value when instrument_defaults is on;
    # otherwise the honest floor (no read noise).
    if 'read_noise_dn' in scene_noise:
        read_noise_e = float(scene_noise['read_noise_dn']) * gain_e_per_dn
    elif instrument_defaults:
        read_noise_e = float(catalog.get('read_noise_e', 0.0))
    else:
        read_noise_e = 0.0

    # Poisson: an explicit noise key wins (noise: {poisson: false} turns it off
    # even under instrument_defaults); otherwise the physical-chain opt-in turns
    # shot noise on, and the floor is off.
    if 'poisson' in scene_noise:
        poisson = bool(scene_noise['poisson'])
    else:
        poisson = instrument_defaults
    # Full-well bloom follows the same precedence: explicit scene value, else
    # the catalog's electron-domain bloom when instrument_defaults is on, else
    # the disabled floor.
    if 'bloom_length' in scene_noise:
        bloom_length = int(scene_noise['bloom_length'])
    elif instrument_defaults:
        bloom_length = int(catalog.get('bloom_length', 0))
    else:
        bloom_length = int(sim_noise.get('bloom_length', 0))
    # Cosmic rays: scene value, else the catalog's cohort-measured rate when
    # instrument_defaults is on (the catalog records it per pixel per second
    # at the chain's unit pixel area), else the config floor.
    if 'cosmic_ray_rate_per_sec' in scene_noise:
        cosmic_ray_rate = float(scene_noise['cosmic_ray_rate_per_sec'])
    elif instrument_defaults and 'cosmic_ray_rate_per_sec' in catalog:
        cosmic_ray_rate = float(catalog['cosmic_ray_rate_per_sec'])
    else:
        cosmic_ray_rate = float(sim_noise.get('cosmic_ray_rate_per_sec', 0.0))
    pixel_area_cm2 = float(scene_noise.get('pixel_area_cm2', sim_noise.get('pixel_area_cm2', 1.0)))

    dark_current = _default_or_zero(
        scene_noise, catalog, 'dark_current_e_per_sec', instrument_defaults=instrument_defaults
    )
    hot_pixel_fraction = _default_or_zero(
        scene_noise, catalog, 'hot_pixel_fraction', instrument_defaults=instrument_defaults
    )
    banding_amplitude = _default_or_zero(
        scene_noise, catalog, 'banding_amplitude_e', instrument_defaults=instrument_defaults
    )
    bias_pedestal_sigma = _default_or_zero(
        scene_noise, catalog, 'bias_pedestal_sigma_dn', instrument_defaults=instrument_defaults
    )
    bias_row_gradient = _default_or_zero(
        scene_noise, catalog, 'bias_row_gradient_dn', instrument_defaults=instrument_defaults
    )
    bias_col_gradient = _default_or_zero(
        scene_noise, catalog, 'bias_col_gradient_dn', instrument_defaults=instrument_defaults
    )
    # Shape parameters (period, amplitude scales) track the catalog; they never
    # activate a stage on their own (the amplitude/fraction gate does that).
    hot_pixel_amplitude = float(
        scene_noise.get('hot_pixel_amplitude_e', catalog.get('hot_pixel_amplitude_e', 0.0))
    )
    hot_pixel_column_factor = float(
        scene_noise.get('hot_pixel_column_factor', catalog.get('hot_pixel_column_factor', 0.0))
    )
    banding_period_px = float(
        scene_noise.get('banding_period_px', catalog.get('banding_period_px', 64.0))
    )

    # The hot_pixels artifact mode (routed here from the registry) wins over the
    # noise-block / instrument_defaults hot-pixel knobs: an explicit mode is a
    # deliberate placement, so it overrides the generic stress path.  incidence
    # is the hot-pixel fraction; amplitude and column factor fall back to the
    # catalog when the mode leaves them unset.  The mode is honored only where
    # it is available (validation already rejects it elsewhere, e.g. LORRI).
    hot_pixel_adversarial = False
    hot_pixel_mode_active = False
    hot_cfg = artifacts.get('hot_pixels') if isinstance(artifacts, Mapping) else None
    if isinstance(hot_cfg, Mapping) and mode_available('hot_pixels', instrument):
        hot_pixel_mode_active = True
        resolved = resolve_mode_config('hot_pixels', hot_cfg)
        hot_pixel_fraction = float(resolved['incidence'])
        mode_amplitude = resolved.get('amplitude_e')
        if mode_amplitude is not None:
            hot_pixel_amplitude = float(mode_amplitude)
        elif hot_pixel_amplitude <= 0.0:
            hot_pixel_amplitude = float(catalog.get('hot_pixel_amplitude_e', 0.0))
        mode_column_factor = resolved.get('column_factor')
        if mode_column_factor is not None:
            hot_pixel_column_factor = float(mode_column_factor)
        elif hot_pixel_column_factor <= 0.0:
            hot_pixel_column_factor = float(catalog.get('hot_pixel_column_factor', 0.0))
        hot_pixel_adversarial = bool(artifacts.get('adversarial', False))

    # The vidicon DN-noise sub-parameters activate on the vidicon path only, and
    # like every physical-chain artifact stay disabled until instrument_defaults
    # is on or the scene sets them explicitly (the stage-activation floor).
    catalog_vidicon = dict(catalog.get('vidicon') or {}) if instrument_defaults else {}
    scene_vidicon = dict(scene_noise.get('vidicon') or {})
    vidicon = {**catalog_vidicon, **{k: float(v) for k, v in scene_vidicon.items()}}

    # Derived I/F calibration scale: a noise-free signal of 1.0 round-trips
    # through the inverse transform to I/F 1.0.  Uses the image-side DN well
    # (frac * full_well_e / gain for the CCD path, frac * full_well_dn for the
    # vidicon path).  The shared inverse divides by scale * exposure_sec; the
    # CCD forward chain scales electrons by exposure_sec / exposure_ref_sec, so
    # its scale carries 1 / exposure_ref_sec, while the vidicon forward mapping
    # has no exposure term at all, so its scale folds in 1 / exposure_sec and
    # the exposure cancels (the inverse divides by the DN full scale alone).
    if detector_model == 'vidicon':
        image_well_dn = signal_full_scale_frac * full_well_dn
        calibration_scale = image_well_dn / exposure_sec if exposure_sec > 0.0 else image_well_dn
    else:
        image_well_dn = signal_full_scale_frac * full_well_e / gain_e_per_dn
        calibration_scale = (
            image_well_dn / exposure_ref_sec if exposure_ref_sec > 0.0 else image_well_dn
        )
    dark_dn = 0.0 if detector_model == 'vidicon' else dark_current * exposure_sec / gain_e_per_dn

    # Detector-stage artifact modes.  An explicit mode wins over the generic
    # noise-block / instrument_defaults knob for the same mechanic (the
    # precedence hot_pixels already uses): the routed modes fold their override
    # into the flat field here, and banding_coherent silences the generic
    # banding so only the explicit family renders.
    artifacts_adversarial = bool(artifacts.get('adversarial', False))
    detector_modes = _resolve_detector_modes(
        artifacts,
        instrument,
        int(params.get('random_seed', 42)),
        instrument_defaults=instrument_defaults,
    )
    contour_step = 8
    if 'banding_coherent' in detector_modes:
        banding_amplitude = 0.0
    if 'bias_structure' in detector_modes:
        bias_mode = detector_modes['bias_structure']
        if bias_mode.get('pedestal_sigma_dn') is not None:
            bias_pedestal_sigma = float(bias_mode['pedestal_sigma_dn'])
        if bias_mode.get('row_gradient_dn') is not None:
            bias_row_gradient = float(bias_mode['row_gradient_dn'])
        if bias_mode.get('col_gradient_dn') is not None:
            bias_col_gradient = float(bias_mode['col_gradient_dn'])
    if 'bloom' in detector_modes:
        bloom_override = detector_modes['bloom'].get('bloom_length')
        if bloom_override is not None:
            bloom_length = int(bloom_override)
    if 'quantization_lut' in detector_modes:
        quantization = 'sqrt_lut'
    if 'quantization_ls8b' in detector_modes:
        quantization = 'ls8b'
    if 'contouring_8bit' in detector_modes:
        quantization = 'contour_8bit'
        contour_step = int(detector_modes['contouring_8bit'].get('step', 8))

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
        bloom_length=bloom_length,
        cosmic_ray_rate_per_sec=cosmic_ray_rate,
        pixel_area_cm2=pixel_area_cm2,
        dark_current_e_per_sec=dark_current,
        hot_pixel_fraction=hot_pixel_fraction,
        hot_pixel_amplitude_e=hot_pixel_amplitude,
        hot_pixel_column_factor=hot_pixel_column_factor,
        banding_amplitude_e=banding_amplitude,
        banding_period_px=banding_period_px,
        bias_pedestal_sigma_dn=bias_pedestal_sigma,
        bias_row_gradient_dn=bias_row_gradient,
        bias_col_gradient_dn=bias_col_gradient,
        vidicon=vidicon,
        calibration_scale_dn_per_s_per_if=calibration_scale,
        dark_dn=dark_dn,
        random_seed=int(params.get('random_seed', 42)),
        instrument_defaults=instrument_defaults,
        hot_pixel_adversarial=hot_pixel_adversarial,
        hot_pixel_mode_active=hot_pixel_mode_active,
        quantization_contour_step=contour_step,
        artifacts_adversarial=artifacts_adversarial,
        detector_modes=detector_modes,
    )
