"""Load and save sim scene files.

A sim scene is a YAML file describing a synthetic frame the navigator can be run
against: instrument, geometry, noise, stray light, and the planted offset the
navigator should recover.  The catalog is laid out as
``<scene_class>/<scene_name>.yaml`` (the directory is the registry).

The YAML fields are the flat runtime parameter names that the renderer
(:func:`spindoctor.sim.render.driver.render_scene`),
:class:`spindoctor.obs.obs_inst_sim.ObsSim`, and the GUI consume, so a validated scene
file IS the ``sim_params`` mapping with no translation layer.  ``load_sim_scene``
parses and validates a file and returns that dict; ``save_sim_scene`` validates a
``sim_params`` dict and writes it (injecting ``schema_version`` and
``scene_name``).  Both validate through :mod:`spindoctor.sim.scene.validate`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.rules_common import (
    _refuse_nulls,
    _require_int,
    _require_str,
)
from spindoctor.sim.scene.validate import (
    CURRENT_SCHEMA_VERSION,
    validate_scene,
    validate_sim_params,
)

__all__ = [
    'DECLARED_SIM_SCENE_CLASSES',
    'iter_scene_paths',
    'load_sim_scene',
    'save_sim_scene',
    'scene_class_for_path',
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


def iter_scene_paths(root: Path) -> list[Path]:
    """Return every ``<class>/<name>.yaml`` scene path under ``root``, sorted."""
    return sorted(root.glob('*/*.yaml'))


def scene_class_for_path(path: Path) -> str:
    """The scene class is the immediate parent directory name."""
    return path.parent.name


def load_sim_scene(path: Path) -> dict[str, Any]:
    """Parse and validate a sim scene YAML into a flat ``sim_params`` dict.

    The returned mapping is exactly what
    :func:`spindoctor.sim.render.driver.render_scene` and
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
    _refuse_nulls(raw, '', source=str(path))
    _require_int(raw, 'schema_version', source=str(path))
    scene_name = _require_str(raw, 'scene_name', source=str(path))
    if scene_name != path.stem:
        raise SimSceneValidationError(
            f'{path}: scene_name {scene_name!r} must match filename stem {path.stem!r}'
        )
    validate_scene(raw, source=str(path))
    return raw


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
