"""Scene-spec schema for the simulator scene catalog.

A sim scene is a YAML file describing a synthetic frame the navigator can be run
against: instrument, geometry, noise, stray light, and the planted offset the
navigator should recover.  The catalog is laid out as
``<scene_class>/<scene_name>.yaml`` (the directory is the registry).

The YAML fields are the flat runtime parameter names that the renderer
(:func:`spindoctor.sim.render.render_combined_model`),
:class:`spindoctor.obs.obs_inst_sim.ObsSim`, and the GUI consume, so a validated scene
file IS the ``sim_params`` mapping with no translation layer.  ``load_sim_scene``
parses and validates a file and returns that dict; ``save_sim_scene`` validates a
``sim_params`` dict and writes it (injecting ``schema_version`` and
``scene_name``); ``validate_sim_params`` validates an in-memory dict for
programmatic scene authors.  The validator is hand-rolled (no pydantic
dependency): the key inventory and boundary classification live in
:mod:`spindoctor.sim.scene_schema` and the per-field type checks in
:mod:`spindoctor.sim.scene_checks` (body entries:
:mod:`spindoctor.sim.scene_checks_body`; the ring_system block:
:mod:`spindoctor.sim.scene_checks_ring`); this module is the public entry point
and re-exports the boundary names.

**The information boundary.**  Every key in the schema is classified as either
idealized (information the production pipeline could know from catalogs,
SPICE, labels, or config: exposed to the navigator through ``obs.nav_params``)
or truth (nature's values, planted errors, variance knobs, and contaminants:
readable only by the image-side renderer).  ``build_nav_params`` constructs
the filtered idealized view; :data:`TRUTH_KEYS` is the machine-readable truth
set the boundary test iterates.  A key added to the schema without a
classification fails the import-time completeness assertion (which runs when
:mod:`spindoctor.sim.scene_schema` is imported here), so every future schema
change must extend the boundary in the same change.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.sim.instruments import GENERIC_INSTRUMENT_ALIASES, SIM_INSTRUMENTS
from spindoctor.sim.scene_checks import (
    _check_artifacts,
    _check_detector,
    _check_expected,
    _check_noise,
    _check_optics,
    _check_optional_bool,
    _check_optional_mapping,
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
from spindoctor.sim.scene_checks_body import (
    ATMOSPHERE_KEYS,
    _check_body_names_unique,
    _check_body_object,
)
from spindoctor.sim.scene_checks_ring import _check_ring_system

# The two private inventories keep their redundant aliases: they are explicit
# re-exports (the schema tests exercise them through this module).
from spindoctor.sim.scene_schema import (
    _ALLOWED_KEYS as _ALLOWED_KEYS,
)
from spindoctor.sim.scene_schema import (
    _OBJECT_BLOCKS as _OBJECT_BLOCKS,
)
from spindoctor.sim.scene_schema import (
    _RING_FEATURE_IDEALIZED_KEYS,
    _RING_SYSTEM_IDEALIZED_KEYS,
    TOP_LEVEL_IDEALIZED_KEYS,
    TOP_LEVEL_TEST_ONLY_KEYS,
    TOP_LEVEL_TRUTH_KEYS,
    TRUTH_KEYS,
    SimSceneValidationError,
)

__all__ = [
    'ALLOWED_INSTRUMENTS',
    'ATMOSPHERE_KEYS',
    'CURRENT_SCHEMA_VERSION',
    'DECLARED_SIM_SCENE_CLASSES',
    'TOP_LEVEL_IDEALIZED_KEYS',
    'TOP_LEVEL_TEST_ONLY_KEYS',
    'TOP_LEVEL_TRUTH_KEYS',
    'TRUTH_KEYS',
    'SimSceneValidationError',
    'build_nav_params',
    'iter_scene_paths',
    'load_sim_scene',
    'save_sim_scene',
    'scene_class_for_path',
    'validate_sim_params',
]

# Scene classes for the sim catalog.  The structural test asserts every
# subdirectory under a catalog root is one of these so typos fail loudly.
DECLARED_SIM_SCENE_CLASSES: frozenset[str] = frozenset(
    {
        'phase_sweep_regular_body',
        'phase_sweep_irregular_body',
        'noise_sweep',
        'smear_sweep',
        'range_sweep',
        'multi_body_geometry',
        'mutual_event',
        'atmosphere',
        'algorithmic_invariants',
        'model_mismatch',
        'regression',
        'artifact_sweep',
        'star_confounder',
        'ring_system',
        'expected_fail',
    }
)

# Instrument names a scene may name (the sim instruments plus the generic alias).
ALLOWED_INSTRUMENTS: frozenset[str] = frozenset(SIM_INSTRUMENTS) | GENERIC_INSTRUMENT_ALIASES

CURRENT_SCHEMA_VERSION: int = 2


def iter_scene_paths(root: Path) -> list[Path]:
    """Return every ``<class>/<name>.yaml`` scene path under ``root``, sorted."""
    return sorted(root.glob('*/*.yaml'))


def scene_class_for_path(path: Path) -> str:
    """The scene class is the immediate parent directory name."""
    return path.parent.name


def load_sim_scene(path: Path) -> dict[str, Any]:
    """Parse and validate a sim scene YAML into a flat ``sim_params`` dict.

    The returned mapping is exactly what
    :func:`spindoctor.sim.render.render_combined_model` and
    :class:`spindoctor.obs.obs_inst_sim.ObsSim` consume; the ``schema_version`` and
    ``scene_name`` keys are metadata the renderer ignores.

    Parameters:
        path: Path to a ``<scene_name>.yaml`` file.  The ``scene_name`` field
            must equal the filename stem.

    Returns:
        The validated flat ``sim_params`` mapping.

    Raises:
        SimSceneValidationError: On any missing/invalid field.
    """
    yaml = YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except Exception as exc:  # ruamel may raise several exception types
        raise SimSceneValidationError(f'{path}: cannot parse YAML: {exc}') from exc
    if not isinstance(raw, dict):
        raise SimSceneValidationError(f'{path}: top-level YAML must be a mapping')
    _require_int(raw, 'schema_version', source=str(path))
    scene_name = _require_str(raw, 'scene_name', source=str(path))
    if scene_name != path.stem:
        raise SimSceneValidationError(
            f'{path}: scene_name {scene_name!r} must match filename stem {path.stem!r}'
        )
    return validate_sim_params(raw, source=str(path))


def save_sim_scene(sim_params: dict[str, Any], path: Path) -> None:
    """Validate ``sim_params`` and write it to ``path`` as a flat YAML scene.

    The ``schema_version`` and ``scene_name`` (= the filename stem) keys are
    injected so the written file validates on reload.  Saving never mutates
    ``sim_params`` (the scene is deep-copied first), and the authored form is
    persisted verbatim -- in particular ``optics.psf: {match_navigator: true}``
    is written as authored, so it survives a save / load round-trip and is
    resolved only when the renderer builds the kernel.

    Parameters:
        sim_params: The flat GUI / render parameter mapping.
        path: Destination ``<scene_name>.yaml`` path; its stem is the scene name.
    """
    scene: dict[str, Any] = {
        **copy.deepcopy(sim_params),
        'schema_version': CURRENT_SCHEMA_VERSION,
        'scene_name': path.stem,
    }
    validate_sim_params(scene, source=str(path))
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with path.open('w') as handle:
        yaml.dump(scene, handle)


def validate_sim_params(
    sim_params: dict[str, Any], *, source: str = 'sim_params'
) -> dict[str, Any]:
    """Validate a flat ``sim_params`` mapping against the schema inventory.

    This is the validation core shared by :func:`load_sim_scene`,
    :func:`save_sim_scene`, and programmatic scene authors (the calibration
    campaign generator, the doc-image galleries), which build dicts rather
    than files.  ``schema_version`` and ``scene_name`` are optional here (a
    dict author has no filename); when present, the version must be current.

    Validation never rewrites the scene: ``optics.psf: {match_navigator: true}``
    stays in its authored form (the renderer resolves it into the navigator's
    concrete Gaussian when it builds the kernel), so an editor's live mapping
    and the persisted file both keep the author's intent.

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
    if instrument not in ALLOWED_INSTRUMENTS:
        raise SimSceneValidationError(
            f'{source}: instrument {instrument!r} is not one of {sorted(ALLOWED_INSTRUMENTS)}'
        )
    _require_positive_int(sim_params, 'size_v', source=source)
    _require_positive_int(sim_params, 'size_u', source=source)
    _require_int(sim_params, 'random_seed', source=source)

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
    _check_optional_mapping(sim_params.get('instrument_config'), 'instrument_config', source=source)
    _check_optional_positive_int(sim_params.get('oversample'), 'oversample', source=source)
    _check_ring_system(sim_params.get('ring_system'), source=source)
    _check_optics(sim_params.get('optics'), source=source)
    _check_detector(sim_params.get('detector'), instrument=instrument, source=source)
    _check_artifacts(sim_params.get('artifacts'), instrument=instrument, source=source)
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

    return sim_params


def build_nav_params(sim_params: dict[str, Any]) -> dict[str, Any]:
    """Build the navigator's filtered idealized view of a scene.

    This is the information boundary (the independence guarantee of the
    simulator-realism program): the returned mapping contains only keys
    classified idealized, with every :data:`TRUTH_KEYS` entry stripped.  For
    bodies, a ``nav_override`` mapping is overlaid first and the key dropped,
    so the navigator sees the geometry it *believes* without learning the
    true values underneath.  An object flagged ``navigable: false`` is dropped
    entirely, so a surviving object's flag is always true and carries no hidden
    truth.  All values are deep copies, so navigator-side code cannot mutate the
    renderer's scene.

    Parameters:
        sim_params: The full scene mapping (the renderer's input).

    Returns:
        The filtered ``nav_params`` mapping exposed as ``obs.nav_params``.
    """
    nav: dict[str, Any] = {}
    for key, value in sim_params.items():
        if key in _OBJECT_BLOCKS or key == 'ring_system':
            continue  # handled below
        if key in TOP_LEVEL_IDEALIZED_KEYS:
            nav[key] = copy.deepcopy(value)
        # Anything else (truth keys, unknown keys) stays behind the boundary:
        # the filter is default-deny.
    ring_system = sim_params.get('ring_system')
    if isinstance(ring_system, dict):
        nav['ring_system'] = _filter_ring_system(ring_system)
    for block, (_allowed, idealized, _truth) in _OBJECT_BLOCKS.items():
        if block not in sim_params:
            continue
        filtered_objects: list[dict[str, Any]] = []
        for obj in sim_params.get(block) or []:
            if not isinstance(obj, dict):
                continue
            if obj.get('navigable') is False:
                continue
            merged = dict(obj)
            if block == 'bodies':
                override = merged.pop('nav_override', None)
                if isinstance(override, dict):
                    merged.update(override)
            filtered_objects.append(
                {k: copy.deepcopy(v) for k, v in merged.items() if k in idealized}
            )
        nav[block] = filtered_objects
    return nav


def _filter_ring_system(ring_system: dict[str, Any]) -> dict[str, Any]:
    """The navigator's view of the ``ring_system`` block.

    Block-level idealized keys (the shared projection geometry, range, pixel
    scale, phase angle) pass through as deep copies.  Features cross only
    when flagged ``navigable: true`` (the default is false), so the rendered
    system is full of structure the navigator was never told about -- the
    distractor regime the ring system exists to produce.  A crossing feature
    exposes only its idealized keys (kind, shape, catalog orbit, tau, and
    the declared orbit uncertainty), never the photometric truth (albedo,
    phase_g) or the planted ``orbit_error`` -- the navigator knows the error
    bars, never the drawn error values.

    Parameters:
        ring_system: The full ``ring_system`` mapping (renderer input).

    Returns:
        The filtered mapping exposed under ``nav_params['ring_system']``.
    """
    filtered: dict[str, Any] = {
        key: copy.deepcopy(value)
        for key, value in ring_system.items()
        if key in _RING_SYSTEM_IDEALIZED_KEYS and key != 'features'
    }
    features: list[dict[str, Any]] = []
    for feature in ring_system.get('features') or []:
        if not isinstance(feature, dict) or feature.get('navigable') is not True:
            continue
        features.append(
            {k: copy.deepcopy(v) for k, v in feature.items() if k in _RING_FEATURE_IDEALIZED_KEYS}
        )
    filtered['features'] = features
    return filtered
