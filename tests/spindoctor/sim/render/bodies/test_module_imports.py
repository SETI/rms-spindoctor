"""The body modules import one another only at the top of a file.

An import placed inside a function can hide an import cycle among the body
modules.  This test reads every module of the package and fails on any import of a
body module that is not at module level, whether the import is absolute or relative.
"""

import ast
from pathlib import Path

import pytest

import spindoctor.sim.render.bodies as bodies_package

_BODIES_DIR = Path(bodies_package.__file__).parent
_BODIES_PACKAGE = 'spindoctor.sim.render.bodies'


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The dotted module names one import statement in the body package reaches.

    A relative import is resolved against the body package, since every module the
    test reads belongs to it.

    Parameters:
        node: The import statement.

    Returns:
        For ``import a.b``, the module ``a.b``.  For ``from a import b``, both ``a``
        and ``a.b``, since ``b`` may be a submodule.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module
    if node.level > 0:
        package_parts = _BODIES_PACKAGE.split('.')
        anchor = '.'.join(package_parts[: len(package_parts) - (node.level - 1)])
        module = anchor if module is None else f'{anchor}.{module}'
    if module is None:
        return []
    return [module, *(f'{module}.{alias.name}' for alias in node.names)]


def _function_level_body_imports(source: str) -> list[int]:
    """The line of every import of a body module inside a function of one source.

    Parameters:
        source: A module's source text.

    Returns:
        One line number per offending import statement.
    """
    found: list[int] = []
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if any(name.startswith(f'{_BODIES_PACKAGE}.') for name in _imported_modules(node)):
                found.append(node.lineno)
    return found


def test_package_has_modules_to_check() -> None:
    """The package directory holds the body modules (guards the import test)."""
    assert (_BODIES_DIR / 'render.py').is_file()


@pytest.mark.parametrize(
    'statement',
    [
        'from . import mesh',
        'from .mesh import render_single_mesh_body',
        'from spindoctor.sim.render.bodies import mesh',
    ],
)
def test_function_level_body_import_is_found(statement: str) -> None:
    """A body import inside a function is found, in each form it can take.

    Parameters:
        statement: The import statement placed inside a function.
    """
    source = f'def draw() -> None:\n    {statement}\n'
    assert _function_level_body_imports(source) == [2]


def test_no_body_module_imports_another_inside_a_function() -> None:
    """No module in the body package imports a body module inside a function."""
    found = [
        f'{path.name}:{line}'
        for path in sorted(_BODIES_DIR.glob('*.py'))
        for line in _function_level_body_imports(path.read_text())
    ]
    assert found == []
