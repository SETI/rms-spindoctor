"""Every integration-marked simulator module also runs in CI.

CI runs the ``sim_integration`` tests and skips the rest of the integration tier,
which needs archive holdings.  A simulator module (``test_sim_*.py`` or
``test_render_diffs.py``) needs none, so one marked ``integration`` without
``sim_integration`` would drop out of CI unnoticed.  The performance budgets are
the one exception, because they need a quiet host.
"""

from __future__ import annotations

import ast
from pathlib import Path

_INTEGRATION_DIR = Path(__file__).parent

_QUIET_HOST_MODULES: frozenset[str] = frozenset({'test_sim_perf.py'})
"""Simulator modules kept out of CI because they time themselves."""


def _module_marks(path: Path) -> set[str]:
    """Return the marker names a module's ``pytestmark`` assignment applies."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == 'pytestmark' for t in node.targets):
            continue
        return {n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)} - {'mark'}
    return set()


def _simulator_modules() -> list[Path]:
    """Return the simulator test modules under ``tests/integration``."""
    paths = sorted(_INTEGRATION_DIR.glob('test_sim_*.py'))
    return [*paths, _INTEGRATION_DIR / 'test_render_diffs.py']


def test_integration_marked_simulator_modules_also_run_in_ci() -> None:
    """Each integration-marked simulator module carries ``sim_integration`` too."""
    missing = [
        p.name
        for p in _simulator_modules()
        if p.name not in _QUIET_HOST_MODULES
        and 'integration' in (marks := _module_marks(p))
        and 'sim_integration' not in marks
    ]
    assert missing == []


def test_the_performance_budgets_stay_out_of_ci() -> None:
    """The performance budgets are integration-only, so CI's shared runners skip them."""
    marks = _module_marks(_INTEGRATION_DIR / 'test_sim_perf.py')
    assert marks == {'integration'}
