"""The body modules import one another only at the top of a file.

An import placed inside a function can hide an import cycle among the body
modules.  This test reads every module of the package and fails on any import of a
body module that is not at module level.
"""

import ast
from pathlib import Path

import spindoctor.sim.render.bodies as bodies_package

_BODIES_DIR = Path(bodies_package.__file__).parent
_BODIES_PACKAGE = 'spindoctor.sim.render.bodies'


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The dotted module names one import statement reaches.

    Parameters:
        node: The import statement.

    Returns:
        For ``import a.b``, the module ``a.b``.  For ``from a import b``, both ``a``
        and ``a.b``, since ``b`` may be a submodule.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.module is None:
        return []
    return [node.module, *(f'{node.module}.{alias.name}' for alias in node.names)]


def _function_level_body_imports(path: Path) -> list[str]:
    """Every import of a body module inside a function of one file.

    Parameters:
        path: The module's source file.

    Returns:
        One ``file:line`` entry per offending import statement.
    """
    found: list[str] = []
    for function in ast.walk(ast.parse(path.read_text())):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if any(name.startswith(f'{_BODIES_PACKAGE}.') for name in _imported_modules(node)):
                found.append(f'{path.name}:{node.lineno}')
    return found


def test_package_has_modules_to_check() -> None:
    """The package directory holds the body modules (guards the import test)."""
    assert (_BODIES_DIR / 'render.py').is_file()


def test_no_body_module_imports_another_inside_a_function() -> None:
    """No module in the body package imports a body module inside a function."""
    found = [
        entry
        for path in sorted(_BODIES_DIR.glob('*.py'))
        for entry in _function_level_body_imports(path)
    ]
    assert found == []
