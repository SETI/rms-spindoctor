"""Unit tests for the checkout guard the ``tests.integration`` scripts run first."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import spindoctor
from tests.integration.checkout_guard import foreign_package_message, require_checkout_package

_INTEGRATION_DIR = Path(__file__).parent


def test_a_package_under_the_checkouts_src_belongs_to_it(tmp_path: Path) -> None:
    """The checkout's own ``src/spindoctor`` passes the guard."""
    package_dir = tmp_path / 'src' / 'spindoctor'
    package_dir.mkdir(parents=True)
    assert foreign_package_message(tmp_path, package_dir) is None


def test_a_package_from_another_checkout_is_named_with_both_paths(tmp_path: Path) -> None:
    """Another checkout's package is refused, naming it, this checkout, and the fix."""
    checkout = tmp_path / 'this'
    other = tmp_path / 'other' / 'src' / 'spindoctor'
    other.mkdir(parents=True)
    (checkout / 'src').mkdir(parents=True)
    assert foreign_package_message(checkout, other) == (
        f'spindoctor was imported from {other.resolve()}, which is outside this checkout '
        f'({checkout.resolve()}); run with PYTHONPATH={(checkout / "src").resolve()} or '
        'install this checkout in editable mode (pip install -e)'
    )


def test_a_script_in_another_checkout_stops_before_running(tmp_path: Path) -> None:
    """A script whose checkout didn't provide the imported package exits with the message."""
    script = tmp_path / 'tests' / 'integration' / 'regenerate.py'
    script.parent.mkdir(parents=True)
    with pytest.raises(SystemExit, match=r'^spindoctor was imported from .*, which is outside'):
        require_checkout_package(str(script))


def test_a_script_runs_when_its_checkout_provides_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script two levels below the checkout root runs against that checkout's package.

    ``spindoctor.__file__`` is pointed into a temporary checkout, so the test
    doesn't depend on where the environment installed the package, and a script
    in that checkout's ``tests/integration`` passes the guard.
    """
    package_init = tmp_path / 'src' / 'spindoctor' / '__init__.py'
    package_init.parent.mkdir(parents=True)
    package_init.touch()
    monkeypatch.setattr(spindoctor, '__file__', str(package_init))
    script = tmp_path / 'tests' / 'integration' / 'regenerate.py'
    script.parent.mkdir(parents=True)
    require_checkout_package(str(script))


_MAIN_GUARD = "__name__ == '__main__'"


def _main_block_statements(path: Path) -> list[ast.stmt] | None:
    """The statements of a module's ``if __name__ == '__main__':`` block, or None."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == _MAIN_GUARD:
            return node.body
    return None


_RUNNABLE_SCRIPTS = sorted(
    path for path in _INTEGRATION_DIR.glob('*.py') if _main_block_statements(path) is not None
)


@pytest.mark.parametrize('path', _RUNNABLE_SCRIPTS, ids=lambda path: path.name)
def test_every_runnable_script_runs_the_guard_first(path: Path) -> None:
    """Each module with a ``__main__`` block calls the guard before any other work."""
    statements = _main_block_statements(path)
    assert statements is not None
    assert ast.unparse(statements[0]) == 'require_checkout_package(__file__)'
