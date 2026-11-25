"""Structural invariants for the simulator scene catalog (Phase T1).

These tests only parse and render in-process (no holdings or SPICE), so they run
in the default suite.  They assert every scene validates, every directory is a
declared class, names are consistent and unique, and every scene renders.  They also
check the wiring of each class that a test module of its own navigates: the class is a
declared validation class with scenes, its module builds its tests from the shared
checks, and every module that does so belongs to such a class.
"""

from pathlib import Path

import pytest

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.io import (
    DECLARED_SIM_SCENE_CLASSES,
    iter_scene_paths,
    load_sim_scene,
    scene_class_for_path,
)
from tests.integration.sim_scene_checks import (
    class_for_module,
    class_module_path,
    class_module_paths,
    class_scene_paths,
)
from tests.integration.sim_truth import OWN_MODULE_CLASSES, VALIDATION_CLASSES

# The catalog root is co-located with this test module.
_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_SCENE_PATHS = iter_scene_paths(_SCENES_ROOT)
_SCENE_IDS = [f'{p.parent.name}/{p.stem}' for p in _SCENE_PATHS]


def test_catalog_is_not_empty() -> None:
    """The catalog contains at least one scene."""
    assert _SCENE_PATHS


def test_subdirectories_are_declared_classes() -> None:
    """Every populated scene-class directory is a declared class."""
    classes = {p.parent.name for p in _SCENE_PATHS}
    unknown = classes - DECLARED_SIM_SCENE_CLASSES
    assert not unknown, f'undeclared scene classes: {sorted(unknown)}'


def test_own_module_classes_are_validation_classes() -> None:
    """Every class with a module of its own is a validation class."""
    assert sorted(OWN_MODULE_CLASSES - VALIDATION_CLASSES) == []


def test_own_module_classes_are_declared_classes() -> None:
    """Every class with a module of its own is a declared scene class."""
    assert sorted(OWN_MODULE_CLASSES - DECLARED_SIM_SCENE_CLASSES) == []


def test_own_module_classes_have_scenes() -> None:
    """Every class with a module of its own holds at least one scene."""
    assert [c for c in sorted(OWN_MODULE_CLASSES) if not class_scene_paths(c)] == []


def test_own_module_classes_have_their_modules() -> None:
    """Every class with a module of its own has test_sim_<class>.py building its tests."""
    modules = set(class_module_paths())
    assert [c for c in sorted(OWN_MODULE_CLASSES) if class_module_path(c) not in modules] == []


def test_class_modules_belong_to_own_module_classes() -> None:
    """Every module that builds class tests is named for a class with a module of its own."""
    stray = [p.name for p in class_module_paths() if class_for_module(p) not in OWN_MODULE_CLASSES]
    assert stray == []


def test_scene_names_are_unique() -> None:
    """No two scenes share a scene_name across the catalog."""
    names = [p.stem for p in _SCENE_PATHS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_scene_validates(path: Path) -> None:
    """Each scene YAML parses and validates."""
    scene = load_sim_scene(path)
    assert scene['scene_name'] == path.stem
    assert scene_class_for_path(path) in DECLARED_SIM_SCENE_CLASSES


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_scene_renders(path: Path) -> None:
    """Each scene's sim params render without error and produce signal."""
    scene = load_sim_scene(path)
    img, _ = render_scene(scene)
    assert img.shape == (scene['size_v'], scene['size_u'])
    assert float(img.max()) > 0.0


def test_loaded_scene_is_flat_sim_params() -> None:
    """A loaded scene is the flat sim_params dict the renderer consumes."""
    scene = load_sim_scene(
        next(p for p in _SCENE_PATHS if p.parent.name == 'algorithmic_invariants')
    )
    assert 'offset_v' in scene
    assert 'offset_u' in scene
    assert 'size_v' in scene
    assert 'image_size_vu' not in scene
    assert 'ground_truth' not in scene


def test_validator_rejects_unknown_instrument(tmp_path: Path) -> None:
    """An unknown instrument name fails validation."""
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'schema_version: 2\nscene_name: bad\ninstrument: hubble\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(bad)


def test_validator_rejects_name_mismatch(tmp_path: Path) -> None:
    """A scene_name that does not match the filename fails validation."""
    bad = tmp_path / 'actual_name.yaml'
    bad.write_text(
        'schema_version: 2\nscene_name: other_name\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='must match filename stem'):
        load_sim_scene(bad)


def test_validator_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A non-current schema_version fails validation."""
    bad = tmp_path / 'v1.yaml'
    bad.write_text(
        'schema_version: 1\nscene_name: v1\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='schema_version'):
        load_sim_scene(bad)


def test_validator_rejects_unknown_key(tmp_path: Path) -> None:
    """An unmodeled top-level key fails validation so typos do not pass silently."""
    bad = tmp_path / 'typo.yaml'
    bad.write_text(
        'schema_version: 2\nscene_name: typo\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\nbogus_key: 3\n'
    )
    with pytest.raises(SimSceneValidationError, match='unknown scene keys'):
        load_sim_scene(bad)
