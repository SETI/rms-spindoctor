"""Every integration-marked simulator module also runs in CI.

CI runs the ``sim_integration`` tests and skips the rest of the integration tier,
which needs archive holdings.  A simulator module (``test_sim_*.py`` or
``test_render_diffs.py``) needs none, so one marked ``integration`` without
``sim_integration`` would drop out of CI unnoticed.  Two modules are the
exceptions: the performance budgets, because they need a quiet host, and the
realism smoke, because it reads real frames.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeGuard

_INTEGRATION_DIR = Path(__file__).parent

_QUIET_HOST_MODULES: frozenset[str] = frozenset({'test_sim_perf.py'})
"""Simulator modules kept out of CI because they time themselves."""

_REAL_DATA_MODULES: frozenset[str] = frozenset({'test_sim_realism_smoke.py'})
"""Simulator modules kept out of CI because they read archive holdings and SPICE."""


_TestDef = ast.FunctionDef | ast.AsyncFunctionDef


def _pytestmark_marks(body: list[ast.stmt]) -> set[str]:
    """Return the marker names the last ``pytestmark`` assignment in ``body`` applies.

    Python binds the last assignment, so that is the one pytest reads.  A body
    without one applies none.
    """
    marks: set[str] = set()
    for node in body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == 'pytestmark' for t in node.targets):
            continue
        marks = {n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)} - {'mark'}
    return marks


def _decorator_marks(node: _TestDef | ast.ClassDef) -> set[str]:
    """Return the attribute names in the decorators of a test function or class."""
    return {
        n.attr for d in node.decorator_list for n in ast.walk(d) if isinstance(n, ast.Attribute)
    }


def _module_marks(path: Path) -> set[str]:
    """Return the marker names a module's ``pytestmark`` assignment applies."""
    return _pytestmark_marks(ast.parse(path.read_text(encoding='utf-8')).body)


def _test_items(path: Path) -> list[tuple[str, set[str]]]:
    """Return each test in a module with the marks pytest applies to it.

    A test function's marks are the module's ``pytestmark`` plus its own
    decorators.  A test method's are the module's, plus its class's decorators
    and class-body ``pytestmark``, plus its own decorators.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    module_marks = _pytestmark_marks(tree.body)
    items = []
    for node in tree.body:
        if not _is_test(node):
            continue
        if isinstance(node, ast.ClassDef):
            class_marks = module_marks | _decorator_marks(node) | _pytestmark_marks(node.body)
            items += [
                (f'{path.name}::{node.name}::{m.name}', class_marks | _decorator_marks(m))
                for m in node.body
                if isinstance(m, _TestDef) and _is_test(m)
            ]
        else:
            items.append((f'{path.name}::{node.name}', module_marks | _decorator_marks(node)))
    return items


def _is_test(node: ast.stmt) -> TypeGuard[_TestDef | ast.ClassDef]:
    """Whether a statement defines a test function or class, by pytest's name prefix."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False
    return node.name.lower().startswith('test')


def _items_missing_the_ci_mark(path: Path) -> list[str]:
    """Return the module's tests marked ``integration`` without ``sim_integration``."""
    return [
        name
        for name, marks in _test_items(path)
        if 'integration' in marks and 'sim_integration' not in marks
    ]


def _simulator_modules() -> list[Path]:
    """Return the simulator test modules under ``tests/integration``."""
    paths = sorted(_INTEGRATION_DIR.glob('test_sim_*.py'))
    return [*paths, _INTEGRATION_DIR / 'test_render_diffs.py']


def test_integration_marked_simulator_tests_also_run_in_ci() -> None:
    """Each integration-marked simulator test carries ``sim_integration`` too."""
    missing = [
        item
        for p in _simulator_modules()
        if p.name not in _QUIET_HOST_MODULES | _REAL_DATA_MODULES
        for item in _items_missing_the_ci_mark(p)
    ]
    assert missing == []


def test_the_performance_budgets_stay_out_of_ci() -> None:
    """The performance budgets are integration-only, so CI's shared runners skip them."""
    path = _INTEGRATION_DIR / 'test_sim_perf.py'
    assert _module_marks(path) == {'integration'}
    assert [name for name, marks in _test_items(path) if 'sim_integration' in marks] == []


def test_the_real_frame_realism_smoke_stays_out_of_ci() -> None:
    """The realism smoke reads real frames, so it is integration-only."""
    path = _INTEGRATION_DIR / 'test_sim_realism_smoke.py'
    assert _module_marks(path) == {'integration'}
    assert [name for name, marks in _test_items(path) if 'sim_integration' in marks] == []


def test_a_test_method_marked_integration_alone_is_missing_the_ci_mark(tmp_path: Path) -> None:
    """The guard reads a class's methods, its decorators, and its class-body ``pytestmark``."""
    module = tmp_path / 'test_sim_probe.py'
    module.write_text(
        'import pytest\n'
        '\n'
        'class TestProbe:\n'
        '    pytestmark = pytest.mark.slow\n'
        '\n'
        '    @pytest.mark.integration\n'
        '    def test_alone(self) -> None: ...\n'
        '\n'
        '    @pytest.mark.integration\n'
        '    @pytest.mark.sim_integration\n'
        '    def test_both(self) -> None: ...\n',
        encoding='utf-8',
    )
    assert _items_missing_the_ci_mark(module) == ['test_sim_probe.py::TestProbe::test_alone']


def test_the_last_pytestmark_assignment_is_the_one_read(tmp_path: Path) -> None:
    """A reassigned ``pytestmark`` applies its last value, as Python binds it."""
    module = tmp_path / 'test_sim_probe.py'
    module.write_text(
        'import pytest\n'
        '\n'
        'pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]\n'
        'pytestmark = pytest.mark.integration\n',
        encoding='utf-8',
    )
    assert _module_marks(module) == {'integration'}
