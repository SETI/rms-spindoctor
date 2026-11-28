"""Validate a sim scene, and build its typed form.

:func:`parse_scene` is the scene validator: it checks a scene's mapping and returns the
typed :class:`~spindoctor.sim.scene.truth.Scene`.  The typed blocks are checked by
their types (:mod:`spindoctor.sim.scene.truth` and the modules it builds on) and by the
rules that read the scene's camera: :mod:`spindoctor.sim.scene.rules_detector` and
:mod:`spindoctor.sim.scene.rules_instrument_config`.  The rules that compare one body
with another, or read the camera for a body, are in
:mod:`spindoctor.sim.scene.rules_body`.  The blocks not typed yet
(:data:`~spindoctor.sim.scene.truth.UNTYPED_BLOCKS`) are checked by their rules
modules: the ring_system block by :mod:`spindoctor.sim.scene.rules_ring`, the artifacts
block by :mod:`spindoctor.sim.scene.rules_artifacts`, and the rest by
:mod:`spindoctor.sim.scene.rules_common`.  :func:`parse_body` builds the typed form of
one body entry, which the renderer reads.

``validate_sim_params`` validates an in-memory ``sim_params`` dict and returns it;
programmatic scene authors call it directly, and loading and saving a scene file
(:mod:`spindoctor.sim.scene.io`) call it too.  ``validate_scene`` puts that check
behind a small cache, which loading a file and every render share, so a scene is
not checked twice while its contents and label stay the same.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from spindoctor.schema import SchemaError, parse
from spindoctor.schema.errors import describe_value
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.fields import scene_binning
from spindoctor.sim.scene.rules_artifacts import _check_artifacts
from spindoctor.sim.scene.rules_body import check_body_names_unique, check_model_atmosphere_filters
from spindoctor.sim.scene.rules_common import (
    _check_expected,
    _check_sky_counts,
    _refuse_nulls,
    _require_ranges_for_spk_error,
)
from spindoctor.sim.scene.rules_detector import check_scene_camera
from spindoctor.sim.scene.rules_instrument_config import check_instrument_config
from spindoctor.sim.scene.rules_ring import _check_ring_system
from spindoctor.sim.scene.truth import UNTYPED_BLOCKS, Scene
from spindoctor.sim.scene.truth_body import Body
from spindoctor.sim.scene.truth_echo import check_truth_echo

__all__ = [
    'clear_validation_cache',
    'parse_body',
    'parse_scene',
    'validate_scene',
    'validate_sim_params',
]


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

    The typed blocks are checked for values a scene file cannot hold before the cache is
    consulted, so a warm cache never accepts what a cold check refuses.  A mapping that
    JSON cannot encode, such as one holding a date that a YAML file spelled without
    quotes, is validated directly instead, outside the cache.  So is one with an integer
    key, which JSON would write as a string.

    Parameters:
        sim_params: The flat scene parameter mapping.
        source: The label that begins each validation error message, such as the
            path of the scene file the mapping came from.

    Raises:
        SimSceneValidationError: If the scene fails validation.
    """
    integer_keys = _refuse_non_file_values(sim_params, source=source)
    if integer_keys:
        validate_sim_params(sim_params, source=source)
        return
    try:
        scene_json = json.dumps(sim_params, sort_keys=True)
    except (TypeError, ValueError):
        validate_sim_params(sim_params, source=source)
        return
    _validate_scene_json(scene_json, source)


def clear_validation_cache() -> None:
    """Forget every scene validated so far, so the next check runs in full."""
    _validate_scene_json.cache_clear()


def parse_scene(raw: Mapping[str, Any], *, source: str = 'sim_params') -> Scene:
    """Check a scene's mapping and build its typed form.

    This is the scene validator.  A scene either sets a key or leaves it out to take
    its default.  A null is refused at any depth, as a key of the scene, of a block,
    or of a mapping in a list, and as a list entry, and the message gives the full key
    path, such as ``bodies[1].nav_override.mesh_lumpiness``.  The typed blocks take only
    what a scene file holds: mappings, lists, and JSON scalars.  A mapping's keys are
    strings, or integers, which YAML reads from a key written as a number.  A tuple, a
    set, a generator, or a typed instance is refused.  Every
    other problem with a typed block is reported at once, one line each, in the form
    ``source: key.path: message``.

    The typed scene is a read-only product of the mapping, which stays the scene's
    saved form: a key the scene leaves out stays out of the mapping, and is None in
    the typed scene.  So ``optics.psf: {match_navigator: true}`` keeps its authored
    form, and the renderer resolves it into the navigator's Gaussian when it builds
    the kernel.

    A truth value that ``instrument_config`` repeats, such as the planted read
    noise copied into ``instrument_config.noise``, isn't an error, but it warns
    with :class:`~spindoctor.sim.scene.truth_echo.SimSceneTruthEchoWarning`
    unless the scene lists its path in ``published_equals_truth`` (see
    :mod:`spindoctor.sim.scene.truth_echo`).

    Parameters:
        raw: The scene's mapping.
        source: The label that begins each error message, such as the path of the
            scene file.

    Returns:
        The typed scene.

    Raises:
        SimSceneValidationError: On any unknown or invalid key, a null at any depth,
            or a value in a typed block that a scene file cannot hold.
    """
    _refuse_nulls(raw, '', source=source)
    _refuse_non_file_values(raw, source=source)
    typed = {key: value for key, value in raw.items() if key not in UNTYPED_BLOCKS}
    try:
        scene = parse(Scene, typed, source=source)
    except SchemaError as exc:
        raise SimSceneValidationError(str(exc)) from exc
    check_scene_camera(scene, source=source)
    check_instrument_config(
        scene.instrument_config,
        instrument=scene.instrument,
        binning=scene_binning(raw),
        source=source,
    )
    check_body_names_unique(scene.bodies or (), source=source)
    check_model_atmosphere_filters(scene, source=source)
    _check_untyped_blocks(raw, instrument=scene.instrument, source=source)
    if scene.spk_error is not None:
        _require_ranges_for_spk_error(raw, source=source)
    check_truth_echo(raw, source=source)
    return scene


def parse_body(raw: Mapping[str, Any], *, source: str = 'body') -> Body:
    """Check one ``bodies`` entry and build its typed form.

    The entry is checked as :func:`parse_scene` checks it inside a scene: its keys,
    their ranges, and the rules among its own keys.  The rules that compare it with
    the other bodies, or read the scene's camera, need the whole scene and are not
    applied.  The renderer parses each body of a validated scene here, after the
    planted roll has turned it.

    Parameters:
        raw: One entry of a scene's ``bodies`` list.
        source: The label that begins each error message.

    Returns:
        The typed body.

    Raises:
        SimSceneValidationError: On a null at any depth of the entry, or any unknown or
            invalid key.
    """
    _refuse_nulls(raw, '', source=source)
    try:
        return parse(Body, raw, source=source)
    except SchemaError as exc:
        raise SimSceneValidationError(str(exc)) from exc


def _refuse_non_file_values(raw: Mapping[str, Any], *, source: str) -> bool:
    """Refuse, in the typed blocks, any value that a scene file cannot hold.

    A scene file holds mappings, lists, and scalars: JSON scalars and the dates that YAML
    reads.  A mapping's keys are strings, or integers, which YAML reads from a key written
    as a number, such as a binning factor.  Anything
    else can come only from a scene built in Python: a tuple or a set (a set's order is
    arbitrary), a generator, which the schema would consume, or a typed instance such as
    a detector override, which the renderer cannot encode.  The check runs on the raw
    mapping before any JSON encoding, so the validation cache sees the same refusals as
    a cold check.  The blocks that are not typed yet are left to their rules.

    Parameters:
        raw: The scene's mapping.
        source: The label that begins the error message.

    Returns:
        True if a typed block has an integer key, which a JSON round trip would turn
        into a string.

    Raises:
        SimSceneValidationError: If a typed block holds such a value at any depth.
    """
    integer_keys = False
    for key, value in raw.items():
        if key not in UNTYPED_BLOCKS:
            integer_keys |= _refuse_non_file_value(value, str(key), source=source)
    return integer_keys


def _refuse_non_file_value(value: Any, path: str, *, source: str) -> bool:
    """Refuse ``value``, or a value inside it, that a scene file cannot hold.

    Parameters:
        value: The value to search: a mapping, a list, or a scalar.
        path: The value's key path.
        source: The label that begins the error message.

    Returns:
        True if ``value`` or a value inside it is a mapping with an integer key.

    Raises:
        SimSceneValidationError: If a value at any depth is not a mapping with string or
            integer keys, a list, a JSON scalar, or a date, or is a mapping whose keys mix
            integers and strings.
    """
    found = False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str | int) or isinstance(key, bool):
                raise SimSceneValidationError(
                    f'{source}: {path}: expected a string or integer key, got {describe_value(key)}'
                )
            found |= _refuse_non_file_value(item, f'{path}.{key}', source=source)
        int_keys = sum(isinstance(key, int) for key in value)
        if 0 < int_keys < len(value):
            # The renderer encodes the scene with sorted keys, and an integer key does not
            # sort against a string one.
            raise SimSceneValidationError(
                f'{source}: {path}: expected keys all written as numbers or all as strings, '
                f'got {sorted(str(key) for key in value)}'
            )
        found |= int_keys > 0
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found |= _refuse_non_file_value(item, f'{path}[{index}]', source=source)
    elif isinstance(value, tuple | set | frozenset):
        raise SimSceneValidationError(
            f'{source}: {path}: expected a list, got {describe_value(value)}'
        )
    elif value is not None and not isinstance(value, str | int | float | bool | datetime.date):
        raise SimSceneValidationError(
            f'{source}: {path}: expected a mapping, a list, a string, a number, or a '
            f'boolean, got {describe_value(value)}'
        )
    return found


def _check_untyped_blocks(raw: Mapping[str, Any], *, instrument: str, source: str) -> None:
    """Check the blocks that the typed scene does not hold yet with their rules.

    Parameters:
        raw: The scene's mapping, whose typed part has been checked.
        instrument: The scene's camera, which the catalog holds.
        source: The label that begins each error message.

    Raises:
        SimSceneValidationError: On any unknown or invalid key in those blocks.
    """
    _check_ring_system(raw.get('ring_system'), source=source)
    _check_artifacts(
        raw.get('artifacts'),
        instrument=instrument,
        detector=raw.get('detector'),
        source=source,
    )
    _check_sky_counts(raw.get('sky_counts'), source=source)
    _check_expected(raw.get('expected'), source=source)


def validate_sim_params(
    sim_params: dict[str, Any], *, source: str = 'sim_params'
) -> dict[str, Any]:
    """Validate a flat ``sim_params`` mapping with :func:`parse_scene`.

    This is the validation core shared by
    :func:`~spindoctor.sim.scene.io.load_sim_scene`,
    :func:`~spindoctor.sim.scene.io.save_sim_scene`, and programmatic scene authors
    (the calibration campaign generator, the doc-image galleries), which build dicts
    rather than files.  ``schema_version`` and ``scene_name`` are optional here (a
    dict author has no filename); when present, the version must be current.

    Parameters:
        sim_params: The flat scene parameter mapping.
        source: Label used in error messages (a path for file authors).

    Returns:
        ``sim_params``, unchanged, for call-chaining.

    Raises:
        SimSceneValidationError: On any unknown or invalid field, or a null at any
            depth.
    """
    parse_scene(sim_params, source=source)
    return sim_params
