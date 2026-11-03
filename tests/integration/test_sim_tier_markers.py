"""Every integration-marked simulator module also runs in CI.

CI runs the ``sim_integration`` tests and skips the rest of the integration tier,
which needs archive holdings.  A simulator module (``test_sim_*.py`` or
``test_render_diffs.py``) needs none, so one marked ``integration`` without
``sim_integration`` would drop out of CI unnoticed.  Two modules are the
exceptions: the performance budgets, because they need a quiet host, and the
realism smoke, because it reads real frames.

The guard asks pytest itself which tests a marker expression selects, with
``--collect-only``.  Pytest's own collection applies every way a mark can
reach a test: a module or class ``pytestmark``, a decorator, an inherited
method, and a ``pytest.param`` case.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_INTEGRATION_DIR = Path(__file__).parent
_REPO_ROOT = _INTEGRATION_DIR.parents[1]

_CI_MARKER_EXPR = '(not integration or sim_integration) and not postgres'
"""The marker expression CI's test job selects with (``run-tests.yml``)."""

_QUIET_HOST_MODULES: frozenset[str] = frozenset({'test_sim_perf.py'})
"""Simulator modules kept out of CI because they time themselves."""

_REAL_DATA_MODULES: frozenset[str] = frozenset({'test_sim_realism_smoke.py'})
"""Simulator modules kept out of CI because they read archive holdings and SPICE."""

_NO_TESTS_COLLECTED = 5
"""Pytest's exit status when the expression selects no test."""


def _collected(marker_expr: str, paths: list[Path], *, rootdir: Path = _REPO_ROOT) -> list[str]:
    """Return the node ids pytest selects from ``paths`` under ``-m marker_expr``.

    The run uses the repository's ``pyproject.toml``, so its registered markers
    and strict-marker check apply, and the given ``-m`` replaces the ``addopts``
    one.  A collection error fails the calling test with pytest's output.
    """
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '--collect-only',
            '-q',
            '-p',
            'no:cacheprovider',
            '-c',
            str(_REPO_ROOT / 'pyproject.toml'),
            '--rootdir',
            str(rootdir),
            '-m',
            marker_expr,
            *(str(p) for p in paths),
        ],
        capture_output=True,
        text=True,
        cwd=rootdir,
        check=False,
    )
    assert result.returncode in (0, _NO_TESTS_COLLECTED), result.stdout + result.stderr
    return [line for line in result.stdout.splitlines() if '::' in line]


def _simulator_modules() -> list[Path]:
    """Return the simulator test modules under ``tests/integration``."""
    paths = sorted(_INTEGRATION_DIR.glob('test_sim_*.py'))
    return [*paths, _INTEGRATION_DIR / 'test_render_diffs.py']


def _exempt_modules() -> list[Path]:
    """Return the simulator modules deliberately kept out of CI."""
    return [_INTEGRATION_DIR / name for name in sorted(_QUIET_HOST_MODULES | _REAL_DATA_MODULES)]


def test_integration_marked_simulator_tests_also_run_in_ci() -> None:
    """Outside the exempt modules, no simulator test is marked ``integration`` alone."""
    ci_modules = [p for p in _simulator_modules() if p not in _exempt_modules()]
    assert _collected('integration and not sim_integration', ci_modules) == []


def test_the_exempt_modules_stay_out_of_ci() -> None:
    """CI's expression selects none of the performance budgets or the realism smoke."""
    assert _collected(_CI_MARKER_EXPR, _exempt_modules()) == []


def test_the_guard_sees_every_way_a_mark_reaches_a_test(tmp_path: Path) -> None:
    """Inherited methods, ``pytest.param`` cases, and a reassigned ``pytestmark`` all count."""
    module = tmp_path / 'test_sim_probe.py'
    module.write_text(
        'import pytest\n'
        '\n'
        'pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]\n'
        'pytestmark = pytest.mark.integration\n'
        '\n'
        'class _Base:\n'
        '    def test_inherited(self) -> None: ...\n'
        '\n'
        '@pytest.mark.sim_integration\n'
        'class TestBoth(_Base):\n'
        '    pass\n'
        '\n'
        'class TestAlone(_Base):\n'
        '    pass\n'
        '\n'
        '@pytest.mark.parametrize(\n'
        "    'x', [1, pytest.param(2, marks=pytest.mark.sim_integration)]\n"
        ')\n'
        'def test_cases(x: int) -> None: ...\n',
        encoding='utf-8',
    )
    assert _collected('integration and not sim_integration', [module], rootdir=tmp_path) == [
        'test_sim_probe.py::TestAlone::test_inherited',
        'test_sim_probe.py::test_cases[1]',
    ]
