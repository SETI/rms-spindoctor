"""Field-type validators for the sim-scene schema.

The ``_check_*`` / ``_require_*`` helpers here enforce the per-field types of
every scene block: the per-object stars entries, the optics sub-blocks (PSF,
smear, distortion, ghosts, stray light), the noise, detector, artifacts, and
spk_error blocks, plus the primitive scalar checks they all share.  Each
block's key inventory lives beside its checker (unknown keys fail validation,
so a typo cannot silently render an un-blurred or clean frame).  The
``bodies``-entry checkers, the schema's largest block, live in the sibling
:mod:`spindoctor.sim.scene_checks_body`, and the ``ring_system`` block's
checkers in :mod:`spindoctor.sim.scene_checks_ring`; both build on the same
primitives.

:func:`spindoctor.sim.scene.validate_sim_params` drives these helpers; every
violation raises :class:`spindoctor.sim.scene_schema.SimSceneValidationError`.
"""

from __future__ import annotations

import math
from typing import Any

from spindoctor.sim.forward.artifact_modes import (
    ARTIFACT_MODES,
    MODE_KEYS,
    ModeParam,
    mode_available,
    mode_unavailable_message,
)
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.status_reason import NavStatusReason

# The scene-level ``expected`` block's allowed vocabularies.  These mirror the
# image-library sidecar taxonomy (status / confidence tier / status_reason) but
# are defined independently here: a sim scene is not a sidecar, so the sim path
# owns its own copy rather than importing the sidecar module.  ``confidence_tier``
# admits the five navigation ranks plus null (assert the status only).
_EXPECTED_KEYS: frozenset[str] = frozenset(
    {
        'status',
        'status_reason',
        'confidence_tier',
        'known_offset_error_px',
        'known_offset_error_tol_px',
    }
)
_EXPECTED_STATUSES: frozenset[str] = frozenset({'success', 'failed', 'conflicted'})
_EXPECTED_TIERS: frozenset[str] = frozenset({'high', 'medium', 'low', 'failed', 'conflicted'})
_EXPECTED_STATUS_REASONS: frozenset[str] = frozenset(reason.value for reason in NavStatusReason)


def _check_star_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``stars`` entry's field types."""
    label = f'stars[{index}]'
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    _check_optional_str(obj.get('catalog_name'), f'{label}.catalog_name', source=source)
    _check_optional_str(obj.get('spectral_class'), f'{label}.spectral_class', source=source)
    for key in (
        'v',
        'u',
        'vmag',
        'move_v',
        'move_u',
        'catalog_error_v',
        'catalog_error_u',
        'delta_mag',
    ):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    _check_optional_positive_number(obj.get('psf_sigma'), f'{label}.psf_sigma', source=source)
    _check_optional_bool(obj.get('navigable'), f'{label}.navigable', source=source)
    _check_star_companion(obj.get('companion'), label=label, source=source)
    psf_size = obj.get('psf_size')
    if psf_size is not None:
        valid = isinstance(psf_size, (list, tuple)) and len(psf_size) == 2
        if not valid or any(isinstance(x, bool) or not isinstance(x, int) for x in psf_size):
            raise SimSceneValidationError(
                f'{source}: {label}.psf_size must be a list of 2 integers when present'
            )


# An unresolved binary: a second point source at ``sep_px`` from the primary
# along ``angle_deg``, ``delta_mag`` fainter.  Its photocenter pull off the
# catalog position is a physical catalog error the navigator cannot know.
_COMPANION_KEYS: frozenset[str] = frozenset({'sep_px', 'delta_mag', 'angle_deg'})


def _check_star_companion(value: Any, *, label: str, source: str) -> None:
    """Validate one star's ``companion`` map (unresolved-binary parameters)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {label}.companion must be a mapping when present')
    unknown = set(value) - _COMPANION_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.companion: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('sep_px'), f'{label}.companion.sep_px', source=source
    )
    _check_optional_number(value.get('delta_mag'), f'{label}.companion.delta_mag', source=source)
    _check_optional_number(value.get('angle_deg'), f'{label}.companion.angle_deg', source=source)


# Allowed keys inside the scene-level optics block and its sub-blocks.  Unknown
# keys fail, so a typo does not silently render an un-blurred frame.
_OPTICS_KEYS: frozenset[str] = frozenset({'psf', 'smear', 'distortion', 'ghosts', 'stray_light'})
_PSF_KEYS: frozenset[str] = frozenset({'match_navigator', 'sigma_v', 'sigma_u', 'w', 'r0', 'n'})
_SMEAR_ENTRY_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'object_class'})
_SMEAR_OBJECT_CLASSES: frozenset[str] = frozenset({'all', 'stars', 'bodies', 'rings'})
_DISTORTION_KEYS: frozenset[str] = frozenset(
    {'k1', 'k2', 'center_v', 'center_u', 'nonradial_rms_px'}
)
_GHOST_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'amplitude', 'defocus_sigma'})
_SPK_ERROR_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'reference_range_km'})
_STRAY_LIGHT_KEYS: frozenset[str] = frozenset(
    {'amplitude', 'direction_deg', 'model', 'center_v', 'center_u'}
)


def _check_optics(value: Any, *, source: str) -> None:
    """Validate the scene-level ``optics`` block and every sub-block.

    Parameters:
        value: The ``optics`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid optics field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics must be a mapping when present')
    unknown = set(value) - _OPTICS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: optics: unknown keys: {sorted(unknown)}')
    _check_psf_block(value.get('psf'), source=source)
    _check_smear_list(value.get('smear'), source=source)
    _check_distortion_block(value.get('distortion'), source=source)
    _check_ghosts_list(value.get('ghosts'), source=source)
    _check_stray_light_block(value.get('stray_light'), source=source)


def _check_psf_block(value: Any, *, source: str) -> None:
    """Validate ``optics.psf`` (either match-navigator form or explicit params)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics.psf must be a mapping when present')
    unknown = set(value) - _PSF_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: optics.psf: unknown keys: {sorted(unknown)}')
    if 'match_navigator' in value:
        _check_optional_bool(
            value.get('match_navigator'), 'optics.psf.match_navigator', source=source
        )
        if value.get('match_navigator') is not True:
            raise SimSceneValidationError(
                f'{source}: optics.psf.match_navigator must be true when present '
                f'(a false value authors a PSF block with no kernel; drop the '
                f'optics.psf block instead)'
            )
        extra = set(value) - {'match_navigator'}
        if extra:
            raise SimSceneValidationError(
                f'{source}: optics.psf.match_navigator is exclusive; drop {sorted(extra)}'
            )
        return
    # An explicit-parameter block needs its core width: the renderer builds
    # the kernel from sigma_v (sigma_u defaults to it), so a block without
    # one would crash at render time instead of failing here.
    if value.get('sigma_v') is None:
        raise SimSceneValidationError(
            f'{source}: optics.psf needs sigma_v (or the match_navigator form); '
            f'a psf block without a core width renders nothing it can build a '
            f'kernel from'
        )
    for key in ('sigma_v', 'sigma_u', 'r0'):
        _check_optional_positive_number(value.get(key), f'optics.psf.{key}', source=source)
    _check_optional_number(value.get('w'), 'optics.psf.w', source=source)
    _check_optional_number(value.get('n'), 'optics.psf.n', source=source)
    w = value.get('w')
    if w is not None and not 0.0 <= float(w) <= 1.0:
        raise SimSceneValidationError(f'{source}: optics.psf.w must lie in [0, 1]; got {w!r}')


def _check_smear_list(value: Any, *, source: str) -> None:
    """Validate ``optics.smear`` (a list of per-object-class motion entries)."""
    if value is None:
        return
    if not isinstance(value, list):
        raise SimSceneValidationError(f'{source}: optics.smear must be a list when present')
    for index, entry in enumerate(value):
        label = f'optics.smear[{index}]'
        if not isinstance(entry, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping')
        unknown = set(entry) - _SMEAR_ENTRY_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(entry.get('dv_px'), f'{label}.dv_px', source=source)
        _check_optional_number(entry.get('du_px'), f'{label}.du_px', source=source)
        object_class = entry.get('object_class', 'all')
        if object_class not in _SMEAR_OBJECT_CLASSES:
            raise SimSceneValidationError(
                f'{source}: {label}.object_class must be one of '
                f'{sorted(_SMEAR_OBJECT_CLASSES)}; got {object_class!r}'
            )


def _check_distortion_block(value: Any, *, source: str) -> None:
    """Validate ``optics.distortion`` (radial polynomial plus non-radial field)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics.distortion must be a mapping when present')
    unknown = set(value) - _DISTORTION_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: optics.distortion: unknown keys: {sorted(unknown)}'
        )
    for key in ('k1', 'k2', 'center_v', 'center_u'):
        _check_optional_number(value.get(key), f'optics.distortion.{key}', source=source)
    _check_optional_nonnegative_number(
        value.get('nonradial_rms_px'), 'optics.distortion.nonradial_rms_px', source=source
    )


def _check_ghosts_list(value: Any, *, source: str) -> None:
    """Validate ``optics.ghosts`` (a list of displaced defocused copies)."""
    if value is None:
        return
    if not isinstance(value, list):
        raise SimSceneValidationError(f'{source}: optics.ghosts must be a list when present')
    for index, entry in enumerate(value):
        label = f'optics.ghosts[{index}]'
        if not isinstance(entry, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping')
        unknown = set(entry) - _GHOST_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(entry.get('dv_px'), f'{label}.dv_px', source=source)
        _check_optional_number(entry.get('du_px'), f'{label}.du_px', source=source)
        _check_optional_number(entry.get('amplitude'), f'{label}.amplitude', source=source)
        _check_optional_nonnegative_number(
            entry.get('defocus_sigma'), f'{label}.defocus_sigma', source=source
        )


def _check_stray_light_block(value: Any, *, source: str) -> None:
    """Validate ``optics.stray_light`` (the smooth scattered-light ramp/bump)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: optics.stray_light must be a mapping when present'
        )
    unknown = set(value) - _STRAY_LIGHT_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: optics.stray_light: unknown keys: {sorted(unknown)}'
        )
    _check_optional_number(value.get('amplitude'), 'optics.stray_light.amplitude', source=source)
    _check_optional_number(
        value.get('direction_deg'), 'optics.stray_light.direction_deg', source=source
    )
    _check_optional_number(value.get('center_v'), 'optics.stray_light.center_v', source=source)
    _check_optional_number(value.get('center_u'), 'optics.stray_light.center_u', source=source)
    model = value.get('model')
    if model is not None and model not in ('linear', 'radial'):
        raise SimSceneValidationError(
            f"{source}: optics.stray_light.model must be 'linear' or 'radial'; got {model!r}"
        )


# The complete inventory of the truth-side noise block: the detector stage's
# stochastic / structured knobs plus the telemetry stage's missing-data rate.
# Unknown noise keys fail validation, so a typo cannot silently render the
# clean floor.  'poisson' is the only boolean; 'bloom_length' is an integer;
# 'vidicon' is a sub-mapping with its own inventory; everything else is a
# non-negative number.
_NOISE_BOOL_KEYS: frozenset[str] = frozenset({'poisson'})
_NOISE_INT_KEYS: frozenset[str] = frozenset({'bloom_length'})
_NOISE_NUMBER_KEYS: frozenset[str] = frozenset(
    {
        'read_noise_dn',
        'bias_dn',
        'cosmic_ray_rate_per_sec',
        'missing_data_rate',
        'signal_full_scale_frac',
        'pixel_area_cm2',
        'dark_current_e_per_sec',
        'hot_pixel_fraction',
        'hot_pixel_amplitude_e',
        'hot_pixel_column_factor',
        'banding_amplitude_e',
        'banding_period_px',
        'bias_pedestal_sigma_dn',
        'bias_row_gradient_dn',
        'bias_col_gradient_dn',
    }
)
_NOISE_KEYS: frozenset[str] = (
    _NOISE_BOOL_KEYS | _NOISE_INT_KEYS | _NOISE_NUMBER_KEYS | frozenset({'vidicon'})
)
_VIDICON_NOISE_KEYS: frozenset[str] = frozenset(
    {
        'read_noise_line_dn',
        'read_noise_pixel_dn',
        'coherent_amplitude_dn',
        'coherent_period_px',
    }
)

# The detector block selects the electron-chain gain state, the detector model
# (CCD electron chain or the Voyager vidicon DN path), the exposure the well
# fraction references, and the ADC quantization sub-mode.  Per-instrument
# defaults come from artifacts_catalog.py; scene keys override them.
_DETECTOR_KEYS: frozenset[str] = frozenset(
    {'gain_state', 'detector_model', 'exposure_ref_sec', 'quantization'}
)
_DETECTOR_MODELS: frozenset[str] = frozenset({'ccd', 'vidicon'})
# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC
# modes reproduce the documented histogram structure of each camera.
_QUANTIZATION_MODES: frozenset[str] = frozenset(
    {'exact', 'uneven_12bit', '8bit', 'sqrt_lut', 'ls8b', 'contour_8bit'}
)
# The artifacts block: the two switches (the physical-chain opt-in and the
# adversarial-placement flag) plus one map per artifact mode, keyed by exactly
# the mode-key registry.  Unknown keys fail.
_ARTIFACTS_SWITCH_KEYS: frozenset[str] = frozenset({'instrument_defaults', 'adversarial'})
_ARTIFACTS_KEYS: frozenset[str] = _ARTIFACTS_SWITCH_KEYS | MODE_KEYS


def _check_noise(value: Any, *, source: str) -> None:
    """Validate the scene-level ``noise`` block against its full key inventory.

    Parameters:
        value: The ``noise`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or mistyped noise field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: noise must be a mapping when present')
    unknown = set(value) - _NOISE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: noise: unknown keys: {sorted(unknown)}')
    for key in _NOISE_BOOL_KEYS:
        _check_optional_bool(value.get(key), f'noise.{key}', source=source)
    for key in _NOISE_INT_KEYS:
        _check_optional_nonnegative_int(value.get(key), f'noise.{key}', source=source)
    for key in _NOISE_NUMBER_KEYS:
        _check_optional_nonnegative_number(value.get(key), f'noise.{key}', source=source)
    vidicon = value.get('vidicon')
    if vidicon is None:
        return
    if not isinstance(vidicon, dict):
        raise SimSceneValidationError(f'{source}: noise.vidicon must be a mapping when present')
    unknown = set(vidicon) - _VIDICON_NOISE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: noise.vidicon: unknown keys: {sorted(unknown)}')
    for key in _VIDICON_NOISE_KEYS:
        _check_optional_nonnegative_number(vidicon.get(key), f'noise.vidicon.{key}', source=source)


def _check_detector(value: Any, *, instrument: str, source: str) -> None:
    """Validate the scene-level ``detector`` block's field types.

    Parameters:
        value: The ``detector`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name, used to
            check the selected gain state against the instrument's catalog.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid detector field,
            including a gain state the instrument does not catalog.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: detector must be a mapping when present')
    unknown = set(value) - _DETECTOR_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: detector: unknown keys: {sorted(unknown)}')
    if value.get('gain_state') is not None:
        gain_state = _require_int(
            {'gain_state': value['gain_state']}, 'gain_state', source=f'{source}: detector'
        )
        # A gain state the instrument does not catalog fails here, at
        # validation, with the cataloged alternatives in the message; the
        # render-time resolver keeps its own guard as a backstop for scenes
        # that bypass validation.
        from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults

        table = resolve_detector_defaults(instrument).get('gain_e_per_dn_by_state') or {0: 1.0}
        if gain_state not in table:
            raise SimSceneValidationError(
                f'{source}: detector.gain_state {gain_state} is not cataloged for '
                f'instrument {instrument!r}; available states: {sorted(table)}'
            )
    model = value.get('detector_model')
    if model is not None and model not in _DETECTOR_MODELS:
        raise SimSceneValidationError(
            f'{source}: detector.detector_model must be one of {sorted(_DETECTOR_MODELS)}; '
            f'got {model!r}'
        )
    _check_optional_positive_number(
        value.get('exposure_ref_sec'), 'detector.exposure_ref_sec', source=source
    )
    quantization = value.get('quantization')
    if quantization is not None and quantization not in _QUANTIZATION_MODES:
        raise SimSceneValidationError(
            f'{source}: detector.quantization must be one of {sorted(_QUANTIZATION_MODES)}; '
            f'got {quantization!r}'
        )


def _check_artifacts(value: Any, *, instrument: str, source: str) -> None:
    """Validate the scene-level ``artifacts`` block against the mode registry.

    Beyond the two switch keys, the block is keyed by exactly the artifact-mode
    registry (unknown keys fail).  A mode that has no implementation yet fails
    as not-yet-implemented, and a mode unavailable on the scene's instrument
    fails with the registry's message (the LORRI hot-pixel case carries a
    bespoke one).  Each present mode's parameters are then type-checked against
    its schema.

    Parameters:
        value: The ``artifacts`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name, used to
            check per-mode availability.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, unimplemented, unavailable, or
            mistyped artifacts field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: artifacts must be a mapping when present')
    unknown = set(value) - _ARTIFACTS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: artifacts: unknown keys: {sorted(unknown)}')
    _check_optional_bool(
        value.get('instrument_defaults'), 'artifacts.instrument_defaults', source=source
    )
    _check_optional_bool(value.get('adversarial'), 'artifacts.adversarial', source=source)
    for mode_name in set(value) & MODE_KEYS:
        _check_artifact_mode(mode_name, value[mode_name], instrument=instrument, source=source)


def _check_artifact_mode(mode_name: str, config: Any, *, instrument: str, source: str) -> None:
    """Validate one artifact-mode map: availability, implementation, and params."""
    label = f'artifacts.{mode_name}'
    if not isinstance(config, dict):
        raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
    mode = ARTIFACT_MODES[mode_name]
    if not mode.implemented:
        raise SimSceneValidationError(
            f'{source}: artifact mode {mode_name!r} is not yet implemented'
        )
    if not mode_available(mode_name, instrument):
        raise SimSceneValidationError(
            f'{source}: {mode_unavailable_message(mode_name, instrument)}'
        )
    param_map = mode.param_map
    unknown = set(config) - set(param_map)
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    for name, param in param_map.items():
        if name in config:
            _check_mode_param(config[name], param, key=f'{label}.{name}', source=source)


def _check_mode_param(value: Any, param: ModeParam, *, key: str, source: str) -> None:
    """Type-check one artifact-mode parameter value against its schema kind."""
    kind = param.kind
    if kind == 'bool':
        _check_optional_bool(value, key, source=source)
    elif kind == 'nonneg_number':
        _check_optional_nonnegative_number(value, key, source=source)
    elif kind == 'unit_interval':
        _check_optional_nonnegative_number(value, key, source=source)
        if value is not None and float(value) > 1.0:
            raise SimSceneValidationError(f'{source}: {key} must lie in [0, 1]; got {value!r}')
    elif kind == 'int':
        _check_optional_number(value, key, source=source)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise SimSceneValidationError(f'{source}: {key} must be an integer when present')
    elif kind == 'nonneg_int':
        _check_optional_nonnegative_int(value, key, source=source)
    elif kind == 'positive_int':
        _check_optional_positive_int(value, key, source=source)
    elif kind == 'enum':
        if param.choices is not None and value not in param.choices:
            raise SimSceneValidationError(
                f'{source}: {key} must be one of {list(param.choices)}; got {value!r}'
            )
    elif kind == 'int_list':
        _check_int_list(value, param.length, key=key, source=source)
    else:  # pragma: no cover - guards a registry authoring mistake
        raise SimSceneValidationError(f'{source}: {key} has unknown parameter kind {kind!r}')


def _check_int_list(value: Any, length: int | None, *, key: str, source: str) -> None:
    """Fail validation unless ``value`` is a list of ``length`` integers."""
    if value is None:
        return
    valid = isinstance(value, (list, tuple)) and (length is None or len(value) == length)
    if not valid or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise SimSceneValidationError(
            f'{source}: {key} must be a list of {length} integers when present'
        )


def _check_spk_error(value: Any, *, source: str) -> None:
    """Validate the scene-level ``spk_error`` block's field types."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: spk_error must be a mapping when present')
    unknown = set(value) - _SPK_ERROR_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: spk_error: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('dv_px'), 'spk_error.dv_px', source=source)
    _check_optional_number(value.get('du_px'), 'spk_error.du_px', source=source)
    _check_optional_positive_number(
        value.get('reference_range_km'), 'spk_error.reference_range_km', source=source
    )


# The background-sky star-count law: cumulative log10 N(<m) = a + b*m per square
# degree, a local-density multiplier, and an optional flat diffuse-sky floor.
# diffuse_e_per_px is detector-native despite the '_e_' in its name: electrons
# per pixel on a CCD, DN per pixel on the Voyager vidicon (which has no electron
# domain), matching the unit domain of the point-source plane it adds to.
_SKY_COUNTS_KEYS: frozenset[str] = frozenset({'a', 'b', 'density_factor', 'diffuse_e_per_px'})


def _check_sky_counts(value: Any, *, source: str) -> None:
    """Validate the scene-level ``sky_counts`` block's field types.

    Parameters:
        value: The ``sky_counts`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid sky_counts field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: sky_counts must be a mapping when present')
    unknown = set(value) - _SKY_COUNTS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: sky_counts: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('a'), 'sky_counts.a', source=source)
    _check_optional_number(value.get('b'), 'sky_counts.b', source=source)
    _check_optional_nonnegative_number(
        value.get('density_factor'), 'sky_counts.density_factor', source=source
    )
    _check_optional_nonnegative_number(
        value.get('diffuse_e_per_px'), 'sky_counts.diffuse_e_per_px', source=source
    )


def _check_star_catalog_scatter(value: Any, *, source: str) -> None:
    """Validate the scene-level ``star_catalog_scatter_px`` sigma.

    A non-negative per-star position-scatter sigma (detector pixels): every
    catalog star's RENDERED position is displaced by a seeded Gaussian draw of
    this sigma, added to any explicit per-star ``catalog_error_*``.

    Parameters:
        value: The ``star_catalog_scatter_px`` value, or None when absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If present and not a non-negative number.
    """
    _check_optional_nonnegative_number(value, 'star_catalog_scatter_px', source=source)


def _check_expected(value: Any, *, source: str) -> None:
    """Validate the scene-level ``expected`` block (the expected outcome).

    The block declares what the navigator should produce for the scene, read by
    the sim integration suite's assertion machinery (not by the renderer or the
    navigator).  ``status`` is required; ``confidence_tier`` is required but may
    be null (assert the status only); ``status_reason`` is optional.  The
    cross-field rules mirror the image-library sidecar taxonomy: a ``failed`` or
    ``conflicted`` status pins the matching tier, and those tiers require the
    matching status, but only when the tier is asserted (non-null).

    ``known_offset_error_px`` is the honest-pin field: the measured fused
    offset-error magnitude a scene deliberately pins as a documented hazard (a
    confidently wrong result the ensemble cannot currently detect).  It
    requires ``status: success`` (a failed result has no offset to be wrong)
    and a positive ``known_offset_error_tol_px`` tolerance, so the assertion
    is a band -- a silent improvement and a worsening regression both fail.

    Parameters:
        value: The ``expected`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid expected field, or a
            status / confidence_tier inconsistency.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: expected must be a mapping when present')
    unknown = set(value) - _EXPECTED_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: expected: unknown keys: {sorted(unknown)}')
    status = value.get('status')
    if status not in _EXPECTED_STATUSES:
        raise SimSceneValidationError(
            f'{source}: expected.status must be one of {sorted(_EXPECTED_STATUSES)}; got {status!r}'
        )
    tier = value.get('confidence_tier')
    if tier is not None and tier not in _EXPECTED_TIERS:
        raise SimSceneValidationError(
            f'{source}: expected.confidence_tier must be one of {sorted(_EXPECTED_TIERS)} or '
            f'null; got {tier!r}'
        )
    if tier is not None:
        if status == 'failed' and tier != 'failed':
            raise SimSceneValidationError(
                f'{source}: expected.status=failed requires expected.confidence_tier=failed'
            )
        if status != 'failed' and tier == 'failed':
            raise SimSceneValidationError(
                f'{source}: expected.confidence_tier=failed requires expected.status=failed'
            )
        if status == 'conflicted' and tier != 'conflicted':
            raise SimSceneValidationError(
                f'{source}: expected.status=conflicted requires expected.confidence_tier=conflicted'
            )
        if status != 'conflicted' and tier == 'conflicted':
            raise SimSceneValidationError(
                f'{source}: expected.confidence_tier=conflicted requires expected.status=conflicted'
            )
    reason = value.get('status_reason')
    if reason is not None and reason not in _EXPECTED_STATUS_REASONS:
        raise SimSceneValidationError(
            f'{source}: expected.status_reason must be one of '
            f'{sorted(_EXPECTED_STATUS_REASONS)} when present; got {reason!r}'
        )
    known_error = value.get('known_offset_error_px')
    known_tol = value.get('known_offset_error_tol_px')
    _check_optional_positive_number(known_error, 'expected.known_offset_error_px', source=source)
    _check_optional_positive_number(known_tol, 'expected.known_offset_error_tol_px', source=source)
    if known_error is not None and status != 'success':
        raise SimSceneValidationError(
            f'{source}: expected.known_offset_error_px requires expected.status=success'
        )
    if known_error is not None and known_tol is None:
        raise SimSceneValidationError(
            f'{source}: expected.known_offset_error_px requires '
            f'expected.known_offset_error_tol_px (the pin is a band)'
        )
    if known_tol is not None and known_error is None:
        raise SimSceneValidationError(
            f'{source}: expected.known_offset_error_tol_px requires expected.known_offset_error_px'
        )


def _require_ranges_for_spk_error(sim_params: dict[str, Any], *, source: str) -> None:
    """Every body and the ring system need a physical ``range_km`` under spk_error.

    The parallax displacement scales as ``reference_range_km / range_km``, so a
    scene that plants spacecraft-ephemeris error must give the renderer a
    physical range for each object it displaces.

    Parameters:
        sim_params: The scene mapping (already type-checked).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If any body or the ring system lacks ``range_km``.
    """
    for index, obj in enumerate(sim_params.get('bodies') or []):
        if obj.get('range_km') is None:
            raise SimSceneValidationError(
                f'{source}: bodies[{index}] needs range_km when spk_error is present'
            )
    ring_system = sim_params.get('ring_system')
    if isinstance(ring_system, dict) and ring_system.get('range_km') is None:
        raise SimSceneValidationError(
            f'{source}: ring_system needs range_km when spk_error is present'
        )


def _require_str(raw: dict[str, Any], key: str, *, source: str) -> str:
    """Return ``raw[key]`` as a non-empty string, or fail validation."""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SimSceneValidationError(f'{source}: {key} must be a non-empty string')
    return value


def _require_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    """Return ``raw[key]`` as an integer (bools rejected), or fail validation."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimSceneValidationError(f'{source}: {key} must be an integer')
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    """Return ``raw[key]`` as a positive integer, or fail validation."""
    value = _require_int(raw, key, source=source)
    if value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer')
    return value


def _check_optional_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a finite number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimSceneValidationError(f'{source}: {key} must be a number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{source}: {key} must be finite; got {value!r}')


def _check_optional_positive_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a finite positive number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{source}: {key} must be finite; got {value!r}')


def _check_optional_nonnegative_int(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a non-negative integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimSceneValidationError(
            f'{source}: {key} must be a non-negative integer when present'
        )


def _check_optional_positive_int(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a positive integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer when present')


def _check_optional_nonnegative_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a finite non-negative number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SimSceneValidationError(f'{source}: {key} must be a non-negative number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{source}: {key} must be finite; got {value!r}')


def _check_optional_str(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a string."""
    if value is None:
        return
    if not isinstance(value, str):
        raise SimSceneValidationError(f'{source}: {key} must be a string when present')


def _check_optional_bool(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a boolean."""
    if value is None:
        return
    if not isinstance(value, bool):
        raise SimSceneValidationError(f'{source}: {key} must be a boolean when present')


def _check_optional_mapping(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a mapping."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {key} must be a mapping when present')


def _check_optional_mapping_list(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a list of mappings."""
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimSceneValidationError(f'{source}: {key} must be a list of mappings')
