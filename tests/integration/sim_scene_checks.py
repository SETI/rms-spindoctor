"""The checks one navigation of a validation scene must pass, shared by the test modules.

A validation scene is navigated once per test run, and that one result is held to
every check that applies to the scene:

* its fused answer against its truth tolerance in :mod:`tests.integration.sim_truth`;
* its ``expected`` block, when it has one (:mod:`tests.integration.sim_expected`);
* each known wrong answer of one technique against its band, and each accurate answer
  of one technique that has a tolerance against that tolerance;
* its fused error against the sigma the navigator reports;
* its regression baseline (:mod:`tests.integration.sim_baseline`).

Most validation scenes are small, and ``test_sim_truth_checks``, ``test_sim_expected``,
and ``test_sim_baselines`` navigate them.  A class in
:data:`~tests.integration.sim_truth.OWN_MODULE_CLASSES` holds full-frame scenes, which
cost far more to navigate.  Its own module, ``test_sim_<class>.py``, builds its tests
with :func:`scene_class_tests`, which navigates each of the class's scenes once and
applies every check here to that one result, and the three shared modules leave the
class out of their navigation.  Each class in its own module lets ``--dist=loadfile``
give the class a worker of its own.

This module selects the scenes each check applies to and holds the assertions, so a
class module and the shared modules apply the same checks the same way.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import pytest

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.sim.scene.io import iter_scene_paths, load_sim_scene, scene_class_for_path
from tests.integration.sim_baseline import (
    baseline_from_result,
    baseline_mismatches,
    load_sim_baseline,
    sim_baseline_path,
)
from tests.integration.sim_expected import (
    assert_result_matches_expected,
    expected_from_scene,
    navigate_scene,
)
from tests.integration.sim_truth import (
    FUSED,
    KNOWN_ERRORS,
    OWN_MODULE_CLASSES,
    SIGMA_BANDS,
    SIGMA_MULTIPLE,
    TRUTH_TOLERANCES,
    assert_matches_truth,
    error_in_sigmas,
)

SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
"""The scene catalog root."""

BASELINES_DIR = Path(__file__).parent / 'sim_baselines'
"""The directory of the committed regression baselines."""

TESTS_DIR = Path(__file__).parent
"""The directory that holds the class modules, ``test_sim_<class>.py``."""

_CLASS_MODULE_PREFIX = 'test_sim_'
"""The file-name prefix a class module puts before its class name."""


def has_own_module(path: Path) -> bool:
    """True when the scene's class is navigated by a test module of its own."""
    return scene_class_for_path(path) in OWN_MODULE_CLASSES


def class_scene_paths(scene_class: str) -> list[Path]:
    """Return every catalog scene in ``scene_class``, sorted."""
    return [p for p in iter_scene_paths(SCENES_ROOT) if scene_class_for_path(p) == scene_class]


def expects_success(scene: dict[str, Any]) -> bool:
    """True unless the scene's expected block declares a failed or conflicted outcome."""
    expected = scene.get('expected')
    return not isinstance(expected, dict) or expected.get('status') == 'success'


def has_known_error(scene: dict[str, Any]) -> bool:
    """True when the scene's expected block holds a known fused error."""
    expected = scene.get('expected')
    return isinstance(expected, dict) and expected.get('known_offset_error_px') is not None


def success_paths(paths: list[Path]) -> list[Path]:
    """Return the scenes among ``paths`` expected to succeed."""
    return [p for p in paths if expects_success(load_sim_scene(p))]


def fused_tolerance_paths(paths: list[Path]) -> list[Path]:
    """Return the scenes among ``paths`` whose fused answer has a truth tolerance.

    ``algorithmic_invariants`` scenes are left out: ``test_sim_algorithmic_invariants``
    checks them in the default suite.
    """
    return [
        p
        for p in success_paths(paths)
        if scene_class_for_path(p) != 'algorithmic_invariants'
        and (p.stem, FUSED) in TRUTH_TOLERANCES
    ]


def expected_block_paths(paths: list[Path]) -> list[Path]:
    """Return the scenes among ``paths`` that carry an ``expected`` block."""
    return [p for p in paths if load_sim_scene(p).get('expected')]


def technique_band_keys(paths: list[Path]) -> list[tuple[str, str]]:
    """Return the single-technique known-error keys of the scenes among ``paths``."""
    names = {p.stem for p in paths}
    return sorted(key for key in KNOWN_ERRORS if key[1] != FUSED and key[0] in names)


def technique_tolerance_keys(paths: list[Path]) -> list[tuple[str, str]]:
    """Return the single-technique truth-tolerance keys of the scenes among ``paths``.

    ``algorithmic_invariants`` scenes are left out: ``test_sim_algorithmic_invariants``
    navigates the technique alone and checks them in the default suite.
    """
    names = {p.stem for p in paths if scene_class_for_path(p) != 'algorithmic_invariants'}
    return sorted(key for key in TRUTH_TOLERANCES if key[1] != FUSED and key[0] in names)


def assert_fused_within_tolerance(scene: dict[str, Any], result: NavResult) -> None:
    """Assert the fused answer succeeded and lies within the scene's truth tolerance."""
    name = scene['scene_name']
    assert result.status == 'success', f'{name}: expected success, got {result.status}'
    assert_matches_truth(result.offset_px, scene, FUSED)


def assert_expected_outcome(scene: dict[str, Any], result: NavResult) -> None:
    """Assert the result matches the scene's ``expected`` block."""
    expected = expected_from_scene(scene)
    assert expected is not None, f'{scene["scene_name"]}: no expected block'
    planted = (float(scene.get('offset_v', 0.0)), float(scene.get('offset_u', 0.0)))
    assert_result_matches_expected(
        scene_name=scene['scene_name'], expected=expected, result=result, planted_offset_vu=planted
    )


def _assert_technique_matches_truth(
    scene: dict[str, Any], result: NavResult, technique: str
) -> None:
    """Assert one technique's answer stays unflagged and passes its truth check."""
    name = scene['scene_name']
    matches = [t for t in result.per_technique if t.technique_name == technique]
    assert matches, f'{name}: {technique} produced no technique result'
    assert not matches[0].spurious, f'{name}: {technique} is now flagged spurious'
    assert_matches_truth(matches[0].offset_px, scene, technique)


def assert_technique_in_band(scene: dict[str, Any], result: NavResult, technique: str) -> None:
    """Assert one technique's known wrong answer stays unflagged and inside its band."""
    _assert_technique_matches_truth(scene, result, technique)


def assert_technique_within_tolerance(
    scene: dict[str, Any], result: NavResult, technique: str
) -> None:
    """Assert one technique's accurate answer stays unflagged and within its tolerance."""
    _assert_technique_matches_truth(scene, result, technique)


def assert_within_sigma_multiple(scene: dict[str, Any], result: NavResult) -> None:
    """Assert the fused error stays within a multiple of the sigma the navigator reports.

    A scene that sits above the multiple for a known reason is held to its band in
    sigmas instead, so a fix leaves the band.
    """
    name = scene['scene_name']
    assert result.offset_px is not None, f'{name}: navigation gave no offset'
    sigmas = error_in_sigmas(result, scene)
    if name in SIGMA_BANDS:
        band = SIGMA_BANDS[name]
        assert band.contains(sigmas), (
            f'{name}: fused error is {sigmas:.2f} sigma, outside its known band '
            f'[{band.low}, {band.high}] sigma'
        )
        return
    assert sigmas <= SIGMA_MULTIPLE, (
        f'{name}: fused error is {sigmas:.2f} sigma, above the {SIGMA_MULTIPLE} sigma bound'
    )


def assert_matches_baseline(scene: dict[str, Any], result: NavResult) -> None:
    """Assert the result reproduces the scene's committed regression baseline."""
    name = scene['scene_name']
    baseline = load_sim_baseline(sim_baseline_path(BASELINES_DIR, name))
    assert baseline_mismatches(baseline, baseline_from_result(scene, result)) == []


def class_module_path(scene_class: str) -> Path:
    """Return the path of the test module of its own that ``scene_class`` needs."""
    return TESTS_DIR / f'{_CLASS_MODULE_PREFIX}{scene_class}.py'


def class_for_module(module_path: Path) -> str:
    """Return the scene class a class module tests, read from the module's file name."""
    return module_path.stem.removeprefix(_CLASS_MODULE_PREFIX)


def class_module_paths() -> list[Path]:
    """Return every test module that builds its tests with :func:`scene_class_tests`."""
    call = f'{scene_class_tests.__name__}('
    return [
        p for p in sorted(TESTS_DIR.glob(f'{_CLASS_MODULE_PREFIX}*.py')) if call in p.read_text()
    ]


def _ids(paths: list[Path]) -> list[str]:
    """Return the scene names that label the parametrized cases."""
    return [p.stem for p in paths]


def scene_class_tests(module_file: str) -> type[object]:
    """Build the tests of a class module, ``test_sim_<class>.py``.

    The class comes from the module's file name.  Each scene of the class is navigated
    once per test process, and that one result meets every check that applies to the
    scene: its truth tolerance, its expected block, each single-technique band or
    tolerance, the sigma check, and its baseline.  A check that applies to none of the
    class's scenes gets no test, so no empty parameter set is collected as a skip.

    Parameters:
        module_file: The class module's ``__file__``.

    Returns:
        A test class, which the module binds to a name starting with ``Test`` so that
        pytest collects it.

    Raises:
        ValueError: If the class is not in
            :data:`~tests.integration.sim_truth.OWN_MODULE_CLASSES` or holds no scenes.
    """
    module_name = Path(module_file).name
    scene_class = class_for_module(Path(module_file))
    if scene_class not in OWN_MODULE_CLASSES:
        raise ValueError(f'{module_name}: {scene_class!r} is not in OWN_MODULE_CLASSES')
    paths = class_scene_paths(scene_class)
    if not paths:
        raise ValueError(f'{module_name}: scene class {scene_class!r} has no scenes')
    by_name = {p.stem: p for p in paths}
    fused = fused_tolerance_paths(paths)
    expected = expected_block_paths(paths)
    sigma = success_paths(paths)
    bands = technique_band_keys(paths)
    technique_tolerances = technique_tolerance_keys(paths)

    @functools.cache
    def navigated(name: str) -> NavResult:
        """Render and navigate one scene of the class once per test process."""
        return navigate_scene(load_sim_scene(by_name[name]))

    class SceneClassTests:
        """The checks of every scene in one class, each scene navigated once."""

        if fused:

            @pytest.mark.parametrize('path', fused, ids=_ids(fused))
            def test_fused_offset_is_within_its_truth_tolerance(self, path: Path) -> None:
                """Each scene's fused offset lies within its measured tolerance."""
                assert_fused_within_tolerance(load_sim_scene(path), navigated(path.stem))

        if expected:

            @pytest.mark.parametrize('path', expected, ids=_ids(expected))
            def test_scene_matches_expected_outcome(self, path: Path) -> None:
                """Each scene with an expected block navigates to its declared outcome."""
                assert_expected_outcome(load_sim_scene(path), navigated(path.stem))

        if bands:

            @pytest.mark.parametrize(
                ('name', 'technique'), bands, ids=[f'{name}-{tech}' for name, tech in bands]
            )
            def test_known_technique_error_stays_in_its_band(
                self, name: str, technique: str
            ) -> None:
                """Each known wrong answer of one technique stays unflagged and in its band."""
                scene = load_sim_scene(by_name[name])
                assert_technique_in_band(scene, navigated(name), technique)

        if technique_tolerances:

            @pytest.mark.parametrize(
                ('name', 'technique'),
                technique_tolerances,
                ids=[f'{name}-{tech}' for name, tech in technique_tolerances],
            )
            def test_technique_answer_is_within_its_truth_tolerance(
                self, name: str, technique: str
            ) -> None:
                """Each accurate answer of one technique stays unflagged and in its tolerance."""
                scene = load_sim_scene(by_name[name])
                assert_technique_within_tolerance(scene, navigated(name), technique)

        if sigma:

            @pytest.mark.parametrize('path', sigma, ids=_ids(sigma))
            def test_fused_error_is_within_the_sigma_multiple(self, path: Path) -> None:
                """Each scene's fused error stays within a multiple of its reported sigma."""
                assert_within_sigma_multiple(load_sim_scene(path), navigated(path.stem))

        @pytest.mark.parametrize('path', paths, ids=_ids(paths))
        def test_scene_matches_baseline(self, path: Path) -> None:
            """Each scene reproduces its recorded regression baseline within tolerance."""
            assert_matches_baseline(load_sim_scene(path), navigated(path.stem))

    return SceneClassTests
