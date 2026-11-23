"""Tests of the Qt application the UI tests share.

Qt runs headless (offscreen platform).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import child_interpreter_environment

try:
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication, QScrollArea
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

# Two tests, as two UI test modules would hold them: the first leaves a window behind,
# and the second looks for what is left.
_TWO_TESTS = """
from PyQt6.QtWidgets import QApplication, QWidget

windows = []

def test_first_test_opens_a_window(qapp):
    windows.append(QWidget())

def test_second_test_finds_only_the_application():
    assert QApplication.instance() is not None
    assert QApplication.topLevelWidgets() == []
"""


def test_the_application_outlives_a_test_and_its_windows_do_not(tmp_path: Path) -> None:
    """A second test finds the first one's application, and none of its windows."""
    module = tmp_path / 'test_two.py'
    module.write_text(_TWO_TESTS)
    env = child_interpreter_environment()
    root = Path(__file__).parents[3]
    env['PYTHONPATH'] = os.pathsep.join([str(root), env['PYTHONPATH']])
    result = subprocess.run(
        [
            *(sys.executable, '-m', 'pytest', '-q', '--rootdir', str(tmp_path)),
            *('-c', os.devnull, '-p', 'tests.spindoctor.ui.conftest'),
            *('-p', 'no:cacheprovider', str(module)),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pyqt_notices_the_end_of_an_object_qt_made(qapp: QApplication) -> None:
    """The wrapper of a viewport, which Qt made, is dead once its scroll area is deleted.

    PyQt stops noticing after any application of the process has been destroyed, and then
    takes a new object at the same address for the dead one.
    """
    area = QScrollArea()
    viewport = area.viewport()
    assert viewport is not None
    sip.delete(area)
    assert sip.isdeleted(viewport)


def _builds_application(path: Path) -> bool:
    """Whether the module calls ``QApplication(...)``; a script held in a string doesn't."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')
            if name == 'QApplication':
                return True
    return False


def test_only_the_shared_fixture_builds_an_application() -> None:
    """No test module calls ``QApplication(...)``; the fixture and child scripts do."""
    tests_dir = Path(__file__).parents[2]
    builders = sorted(
        path.relative_to(tests_dir).as_posix()
        for path in tests_dir.rglob('*.py')
        if _builds_application(path)
    )
    assert builders == ['spindoctor/ui/conftest.py']
