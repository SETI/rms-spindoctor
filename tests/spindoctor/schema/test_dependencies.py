"""What the schema core may import, and that pydantic stays a parser.

The schema package sits below both the simulator and the configuration code, so it
must import nothing else from ``spindoctor``: an import the other way would let one
of them pull the other in.  Beyond the standard library it may import only pydantic,
``annotated_types``, ``ruamel.yaml``, and ``filecache``.  Schemas are plain
dataclasses, so no class in ``src/`` subclasses pydantic's ``BaseModel``.  Both rules
are checked by reading the source.
"""

import ast
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / 'src' / 'spindoctor'
_SCHEMA = _SRC / 'schema'

# The top-level packages, beyond the standard library, the schema core may import.
_ALLOWED_OUTSIDE = frozenset({'annotated_types', 'filecache', 'pydantic', 'ruamel'})


def _imported_modules(path: Path) -> list[str]:
    """The absolute module name of every import in one source file.

    Parameters:
        path: The source file.

    Returns:
        The imported modules; a relative import counts as ``spindoctor``.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append('spindoctor' if node.level > 0 else str(node.module))
    return names


def _disallowed(name: str) -> bool:
    """Whether the schema core may not import the module ``name``."""
    if name == 'spindoctor.schema' or name.startswith('spindoctor.schema.'):
        return False
    top = name.split('.')[0]
    return top not in sys.stdlib_module_names and top not in _ALLOWED_OUTSIDE


def test_schema_package_has_modules() -> None:
    """The scan below has files to read."""
    assert (_SCHEMA / 'core.py').is_file()


@pytest.mark.parametrize('path', sorted(_SCHEMA.glob('*.py')), ids=lambda path: path.name)
def test_schema_imports_only_its_dependencies(path: Path) -> None:
    """A schema module imports only the standard library, its four dependencies, and itself."""
    assert [name for name in _imported_modules(path) if _disallowed(name)] == []


def _base_model_subclasses(path: Path) -> list[str]:
    """The classes in one source file that list ``BaseModel`` among their bases.

    Parameters:
        path: The source file.

    Returns:
        ``file:class`` for each such class.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, 'id', '')
            if name == 'BaseModel':
                found.append(f'{path.relative_to(_SRC)}:{node.name}')
    return found


def test_no_class_subclasses_base_model() -> None:
    """No class in the package subclasses pydantic's BaseModel."""
    found = [hit for path in sorted(_SRC.rglob('*.py')) for hit in _base_model_subclasses(path)]
    assert found == []
