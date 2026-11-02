"""The guards that keep CPU-dependent arithmetic out of the simulator.

``np.hypot`` calls the C library's ``hypot``, and ``@`` and NumPy's matrix
products call OpenBLAS, which picks its kernel by CPU.  Either one's last bit
differs between CPUs, so the committed renders would only reproduce on some
machines.  The simulator uses
:func:`spindoctor.support.stable_math.stable_hypot` and
:func:`spindoctor.support.stable_math.stable_matmul3` instead.
"""

import ast
from pathlib import Path
from typing import TypeGuard

import pytest

_SIM_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'spindoctor' / 'sim'

_SIM_MODULES = sorted(_SIM_ROOT.rglob('*.py'))

_NUMPY_MATRIX_PRODUCTS = frozenset({'dot', 'matmul', 'einsum', 'inner', 'tensordot', 'vdot'})
"""The NumPy functions that hand a product to BLAS."""


def _is_numpy_attribute(node: ast.AST, names: frozenset[str]) -> TypeGuard[ast.Attribute]:
    """Whether ``node`` is ``np.<name>`` or ``numpy.<name>`` for one of ``names``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in names
        and isinstance(node.value, ast.Name)
        and node.value.id in {'np', 'numpy'}
    )


def _is_numpy_import(node: ast.AST, names: frozenset[str]) -> TypeGuard[ast.ImportFrom]:
    """Whether ``node`` is ``from numpy import <name>`` for one of ``names``, under any alias."""
    return (
        isinstance(node, ast.ImportFrom)
        and node.module == 'numpy'
        and any(alias.name in names for alias in node.names)
    )


def _hypot_calls(path: Path) -> list[int]:
    """Return the lines of the NumPy ``hypot`` references and imports in a module.

    That is ``np.hypot`` or ``numpy.hypot``, and ``from numpy import hypot``
    under any alias.
    """
    names = frozenset({'hypot'})
    tree = ast.parse(path.read_text(), filename=str(path))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if _is_numpy_attribute(node, names) or _is_numpy_import(node, names)
    )


def _matrix_products(path: Path) -> list[int]:
    """Return the lines of the BLAS matrix products in a module.

    That is an ``@`` or ``@=`` operator, a ``.dot`` method call, and a NumPy
    matrix-product function referenced as an attribute or imported by name.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult))
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'dot'
        )
        or _is_numpy_attribute(node, _NUMPY_MATRIX_PRODUCTS)
        or _is_numpy_import(node, _NUMPY_MATRIX_PRODUCTS)
    )


def _module_id(path: Path) -> str:
    """The module's path under the simulator package, for a test id."""
    return str(path.relative_to(_SIM_ROOT))


@pytest.mark.parametrize('module', _SIM_MODULES, ids=_module_id)
def test_the_simulator_uses_stable_hypot_in_place_of_numpy_hypot(module: Path) -> None:
    """No simulator module references ``np.hypot``; it uses ``stable_hypot``."""
    assert _hypot_calls(module) == []


@pytest.mark.parametrize('module', _SIM_MODULES, ids=_module_id)
def test_the_simulator_uses_stable_matmul3_in_place_of_a_blas_product(module: Path) -> None:
    """No simulator module takes a BLAS matrix product; it uses ``stable_matmul3``."""
    assert _matrix_products(module) == []


def test_the_hypot_guard_catches_hypot_imported_from_numpy(tmp_path: Path) -> None:
    """``from numpy import hypot`` is caught too, not only the attribute."""
    module = tmp_path / 'probe.py'
    module.write_text('import math\nfrom numpy import hypot as h\n', encoding='utf-8')
    assert _hypot_calls(module) == [2]


def test_the_matmul_guard_catches_every_form_of_a_matrix_product(tmp_path: Path) -> None:
    """``@``, ``@=``, ``.dot``, ``np.matmul``, and an imported ``einsum`` are all caught."""
    module = tmp_path / 'probe.py'
    module.write_text(
        'import numpy as np\n'
        'from numpy import einsum\n'
        'a = b @ c\n'
        'a @= c\n'
        'a = b.dot(c)\n'
        'a = np.matmul(b, c)\n'
        'a = b * c\n',
        encoding='utf-8',
    )
    assert _matrix_products(module) == [2, 3, 4, 5, 6]
