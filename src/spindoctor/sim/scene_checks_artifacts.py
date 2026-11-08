"""Artifacts-block field validators for the sim-scene schema.

The ``artifacts`` block holds two switches (``instrument_defaults`` and
``adversarial``) and one map per artifact mode, keyed by exactly the mode
registry.  Its checkers live here, beside the block's key inventory: each mode the
scene sets must be implemented, offered by the scene's instrument, and rendered by
the scene's detector chain, and each parameter is type-checked against its registry
schema; ``instrument_defaults`` must not turn on a mode that the scene's detector
chain never renders.  The scalar primitives they share with the other block checkers
stay in :mod:`spindoctor.sim.scene_checks`, which also documents the overall
validation contract (unknown keys fail, every violation raises
:class:`spindoctor.sim.scene_schema.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate_sim_params` drives ``_check_artifacts``
once per scene.
"""

from __future__ import annotations

from typing import Any

from spindoctor.sim.forward.artifact_modes import (
    ARTIFACT_MODES,
    MODE_KEYS,
    ArtifactMode,
    ModeParam,
)
from spindoctor.sim.forward.artifacts_catalog import (
    instrument_default_modes,
    mode_available,
    mode_unavailable_message,
    resolve_detector_model,
)
from spindoctor.sim.scene_checks import (
    _check_optional_bool,
    _check_optional_nonnegative_int,
    _check_optional_nonnegative_number,
    _check_optional_number,
    _check_optional_positive_int,
)
from spindoctor.sim.scene_schema import SimSceneValidationError

# The artifacts block: the two switches (the physical-chain opt-in and the
# adversarial-placement flag) plus one map per artifact mode, keyed by exactly
# the mode-key registry.  Unknown keys fail.
_ARTIFACTS_SWITCH_KEYS: frozenset[str] = frozenset({'instrument_defaults', 'adversarial'})
_ARTIFACTS_KEYS: frozenset[str] = _ARTIFACTS_SWITCH_KEYS | MODE_KEYS


def _check_artifacts(value: Any, *, instrument: str, detector: Any, source: str) -> None:
    """Validate the scene-level ``artifacts`` block against the mode registry.

    Beyond the two switch keys, the block is keyed by exactly the artifact-mode
    registry (unknown keys fail).  If a mode has no implementation yet, it fails as
    not-yet-implemented.  If the scene's instrument does not offer a mode, it fails
    with the reason the instrument's config entry states, or with a list of the
    instruments that offer it.  If the scene's detector chain never applies a mode,
    it fails as such.  Each present mode's parameters are then type-checked against
    its schema.  Under ``instrument_defaults``, the block fails if the camera's
    config turns on a mode that the scene's detector model never renders.

    Parameters:
        value: The ``artifacts`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name, used to
            check per-mode availability.
        detector: The scene's (already validated) ``detector`` block, or None,
            whose ``detector_model`` overrides the instrument's.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, unimplemented, unavailable, or
            mistyped artifacts field, or a mode the detector model never renders.
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
    model = resolve_detector_model(instrument, detector if isinstance(detector, dict) else None)
    for mode_name in set(value) & MODE_KEYS:
        _check_artifact_mode(
            mode_name, value[mode_name], instrument=instrument, model=model, source=source
        )
    if value.get('instrument_defaults'):
        _check_instrument_default_modes(instrument, model=model, source=source)


def _chain_names(mode: ArtifactMode) -> str:
    """The detector chains that render ``mode``, joined for a message."""
    return ' and '.join(sorted(mode.detector_models))


def _check_instrument_default_modes(instrument: str, *, model: str, source: str) -> None:
    """Refuse ``instrument_defaults`` when it turns on a mode the chain never renders.

    The camera's config lists the modes that ``instrument_defaults`` turns on, and
    the loader checks them against the camera's own detector model.  If a scene
    overrides ``detector.detector_model``, it can select a chain without that mode's
    stage, where the mode would render nothing.

    Parameters:
        instrument: The scene's (already validated) instrument name.
        model: The detector model the scene renders with.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If ``model`` never renders one of the modes.
    """
    for mode_name in sorted(instrument_default_modes(instrument)):
        mode = ARTIFACT_MODES[mode_name]
        if model not in mode.detector_models:
            raise SimSceneValidationError(
                f'{source}: artifacts.instrument_defaults turns on artifact mode '
                f'{mode_name!r}, which renders only in the {_chain_names(mode)} detector '
                f'chain; this scene renders with detector_model {model}; drop '
                f'instrument_defaults or the detector_model override'
            )


def _check_artifact_mode(
    mode_name: str, config: Any, *, instrument: str, model: str, source: str
) -> None:
    """Validate one artifact-mode map: availability, implementation, chain, and params.

    An unknown key fails, and a map that sets both ``count`` and
    ``expected_events`` to a value (``dead_pixels``, ``dead_columns``) fails as
    ambiguous; a key set to null counts as unset.  The ``probability`` and
    ``pixel_fraction`` keys must lie in [0, 1], and
    ``expected_events`` and ``expected_events_per_sec`` must be non-negative.
    """
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
    if model not in mode.detector_models:
        raise SimSceneValidationError(
            f'{source}: artifact mode {mode_name!r} renders only in the '
            f'{_chain_names(mode)} detector chain; this scene renders with detector_model '
            f'{model}'
        )
    param_map = mode.param_map
    unknown = set(config) - set(param_map)
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    count_key = mode.count_alternative
    if (
        count_key is not None
        and config.get(count_key) is not None
        and config.get(mode.activation) is not None
    ):
        raise SimSceneValidationError(
            f'{source}: {label}: {count_key!r} (a fixed event count) and '
            f'{mode.activation!r} (a Poisson expected event count) are mutually exclusive; '
            'give one'
        )
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
