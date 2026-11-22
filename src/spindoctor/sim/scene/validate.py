"""Validate a sim scene against the schema.

``validate_sim_params`` validates an in-memory ``sim_params`` dict; programmatic
scene authors call it directly, and loading and saving a scene file
(:mod:`spindoctor.sim.scene.io`) call it too.  ``validate_scene`` puts that check
behind a small cache, which loading a file and every render share, so a scene is
not checked twice while its contents and label stay the same.  The validator is
hand-rolled (no pydantic dependency): the key inventory and boundary classification
live in :mod:`spindoctor.sim.scene.boundary` and the per-field type checks in
:mod:`spindoctor.sim.scene.rules_common` (body entries:
:mod:`spindoctor.sim.scene.rules_body`; the ring_system block:
:mod:`spindoctor.sim.scene.rules_ring`; the artifacts block:
:mod:`spindoctor.sim.scene.rules_artifacts`; the instrument_config block:
:mod:`spindoctor.sim.scene.rules_instrument_config`).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from spindoctor.sim.instruments.catalog import load_instrument_catalog
from spindoctor.sim.scene.boundary import _ALLOWED_KEYS, _OBJECT_BLOCKS
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.fields import scene_binning
from spindoctor.sim.scene.rules_artifacts import _check_artifacts
from spindoctor.sim.scene.rules_body import _check_body_names_unique, _check_body_object
from spindoctor.sim.scene.rules_common import (
    _check_detector,
    _check_downlink_table_used,
    _check_expected,
    _check_noise,
    _check_optics,
    _check_optional_bool,
    _check_optional_mapping_list,
    _check_optional_number,
    _check_optional_positive_int,
    _check_optional_positive_number,
    _check_optional_str,
    _check_sky_counts,
    _check_spk_error,
    _check_star_catalog_scatter,
    _check_star_object,
    _require_int,
    _require_positive_int,
    _require_ranges_for_spk_error,
    _require_str,
)
from spindoctor.sim.scene.rules_instrument_config import _check_instrument_config
from spindoctor.sim.scene.rules_ring import _check_ring_system
from spindoctor.sim.scene.truth_echo import check_truth_echo

__all__ = [
    'CURRENT_SCHEMA_VERSION',
    'clear_validation_cache',
    'validate_scene',
    'validate_sim_params',
]

CURRENT_SCHEMA_VERSION: int = 2

# Top-level numbers every consumer reads with ``float(sim_params.get(key,
# default))``, so a key present as null would reach float() as None.  A block or
# a list set to null reads as unset, but these refuse null: the author leaves the
# key out to take its default.
_NON_NULL_TOP_LEVEL_NUMBERS: tuple[str, ...] = (
    'exposure_sec',
    'offset_v',
    'offset_u',
    'offset_rotation_deg',
    'time',
    'ring_epoch',
    'star_catalog_scatter_px',
)

# The largest noise_seed the scene editor's control can show and save.
_NOISE_SEED_MAX = 2**31 - 1


# A handful of entries.  Validation is cheap next to a render, so a scene that has
# dropped out of the cache costs little to check again.
@lru_cache(maxsize=8)
def _validate_scene_json(scene_json: str, source: str) -> None:
    """Validate the scene that ``scene_json`` encodes, once per scene and source.

    The mapping checked is decoded from the scene's JSON, as the renderer's input
    is, so validation sees the values the pipeline reads.  A refused scene raises
    every time, since ``lru_cache`` does not cache exceptions.
    """
    validate_sim_params(json.loads(scene_json), source=source)


def validate_scene(sim_params: dict[str, Any], *, source: str = 'sim_params') -> None:
    """Validate a scene through the shared validation cache.

    Loading a scene file, every render, and the scene editor's check before each
    preview all validate through this function.  A scene checked once is not
    checked again while its contents and ``source`` stay the same, so a scene file
    that is loaded and then rendered is validated once, and a scene that passes
    prints its truth-echo warnings once.  A scene that a caller changes after a
    check is checked again, because its contents differ.

    A mapping that JSON cannot encode, such as one holding a date that a YAML file
    spelled without quotes, is validated directly instead, outside the cache.

    Parameters:
        sim_params: The flat scene parameter mapping.
        source: The label that begins each validation error message, such as the
            path of the scene file the mapping came from.

    Raises:
        SimSceneValidationError: If the scene fails validation.
    """
    try:
        scene_json = json.dumps(sim_params, sort_keys=True)
    except (TypeError, ValueError):
        validate_sim_params(sim_params, source=source)
        return
    _validate_scene_json(scene_json, source)


def clear_validation_cache() -> None:
    """Forget every scene validated so far, so the next check runs in full."""
    _validate_scene_json.cache_clear()


def validate_sim_params(
    sim_params: dict[str, Any], *, source: str = 'sim_params'
) -> dict[str, Any]:
    """Validate a flat ``sim_params`` mapping against the schema inventory.

    This is the validation core shared by
    :func:`~spindoctor.sim.scene.io.load_sim_scene`,
    :func:`~spindoctor.sim.scene.io.save_sim_scene`, and programmatic scene authors
    (the calibration campaign generator, the doc-image galleries), which build dicts
    rather than files.  ``schema_version`` and ``scene_name`` are optional here (a
    dict author has no filename); when present, the version must be current.

    Validation never rewrites the scene: ``optics.psf: {match_navigator: true}``
    stays in its authored form (the renderer resolves it into the navigator's
    concrete Gaussian when it builds the kernel), so an editor's live mapping
    and the persisted file both keep the author's intent.

    A truth value that ``instrument_config`` repeats, such as the planted read
    noise copied into ``instrument_config.noise``, isn't an error, but it warns
    with :class:`~spindoctor.sim.scene.truth_echo.SimSceneTruthEchoWarning`
    unless the scene lists its path in ``published_equals_truth`` (see
    :mod:`spindoctor.sim.scene.truth_echo`).

    Parameters:
        sim_params: The flat scene parameter mapping.
        source: Label used in error messages (a path for file authors).

    Returns:
        ``sim_params``, unchanged, for call-chaining.

    Raises:
        SimSceneValidationError: On any unknown or invalid field.
    """
    unknown = set(sim_params) - _ALLOWED_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: unknown scene keys: {sorted(unknown)}')

    if 'schema_version' in sim_params:
        schema_version = _require_int(sim_params, 'schema_version', source=source)
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise SimSceneValidationError(
                f'{source}: schema_version must be {CURRENT_SCHEMA_VERSION}, got {schema_version}'
            )
    if 'scene_name' in sim_params:
        _require_str(sim_params, 'scene_name', source=source)

    instrument = _require_str(sim_params, 'instrument', source=source)
    # The names a scene may write: every instrument in the catalog, plus the
    # aliases the entries declare.
    allowed = load_instrument_catalog().allowed_names()
    if instrument not in allowed:
        raise SimSceneValidationError(
            f'{source}: instrument {instrument!r} is not one of {sorted(allowed)}'
        )
    _require_positive_int(sim_params, 'size_v', source=source)
    _require_positive_int(sim_params, 'size_u', source=source)
    _require_int(sim_params, 'random_seed', source=source)
    # The scene editor shows -1 as an absent noise_seed, so a negative value could not
    # be told apart from leaving the key out, and its control stops at the 32-bit limit.
    if 'noise_seed' in sim_params and not (
        0 <= _require_int(sim_params, 'noise_seed', source=source) <= _NOISE_SEED_MAX
    ):
        raise SimSceneValidationError(
            f'{source}: noise_seed must be an integer from 0 to {_NOISE_SEED_MAX}'
        )
    null_numbers = [
        key for key in _NON_NULL_TOP_LEVEL_NUMBERS if key in sim_params and sim_params[key] is None
    ]
    if null_numbers:
        raise SimSceneValidationError(
            f'{source}: {null_numbers} must be numbers when present; leave a key out '
            f'to take its default'
        )

    _check_optional_positive_number(sim_params.get('exposure_sec'), 'exposure_sec', source=source)
    _check_optional_number(sim_params.get('offset_v'), 'offset_v', source=source)
    _check_optional_number(sim_params.get('offset_u'), 'offset_u', source=source)
    _check_optional_number(
        sim_params.get('offset_rotation_deg'), 'offset_rotation_deg', source=source
    )
    _check_optional_number(sim_params.get('time'), 'time', source=source)
    _check_optional_number(sim_params.get('ring_epoch'), 'ring_epoch', source=source)
    _check_optional_str(sim_params.get('midtime_utc'), 'midtime_utc', source=source)
    _check_optional_str(sim_params.get('closest_planet'), 'closest_planet', source=source)
    _check_optional_bool(
        sim_params.get('fit_camera_rotation'), 'fit_camera_rotation', source=source
    )
    _check_noise(sim_params.get('noise'), source=source)
    _check_optional_positive_int(sim_params.get('oversample'), 'oversample', source=source)
    _check_ring_system(sim_params.get('ring_system'), source=source)
    _check_optics(sim_params.get('optics'), source=source)
    _check_detector(sim_params.get('detector'), instrument=instrument, source=source)
    _check_instrument_config(
        sim_params.get('instrument_config'),
        instrument=instrument,
        binning=scene_binning(sim_params),
        source=source,
    )
    _check_artifacts(
        sim_params.get('artifacts'),
        instrument=instrument,
        detector=sim_params.get('detector'),
        source=source,
    )
    _check_downlink_table_used(
        sim_params.get('detector'),
        sim_params.get('artifacts'),
        instrument=instrument,
        source=source,
    )
    _check_spk_error(sim_params.get('spk_error'), source=source)
    _check_sky_counts(sim_params.get('sky_counts'), source=source)
    _check_star_catalog_scatter(sim_params.get('star_catalog_scatter_px'), source=source)
    _check_expected(sim_params.get('expected'), source=source)

    for block in ('bodies', 'stars'):
        _check_optional_mapping_list(sim_params.get(block), block, source=source)
        allowed = _OBJECT_BLOCKS[block][0]
        for index, obj in enumerate(sim_params.get(block) or []):
            unknown = set(obj) - allowed
            if unknown:
                raise SimSceneValidationError(
                    f'{source}: {block}[{index}]: unknown keys: {sorted(unknown)}'
                )
            if block == 'bodies':
                _check_body_object(obj, index=index, source=source)
            else:
                _check_star_object(obj, index=index, source=source)
    _check_body_names_unique(sim_params.get('bodies') or [], source=source)

    if sim_params.get('spk_error') is not None:
        _require_ranges_for_spk_error(sim_params, source=source)
    check_truth_echo(sim_params, source=source)

    return sim_params
