"""The simulator's CPU-stable arithmetic, and the guard that keeps ``np.hypot`` out.

``np.hypot`` calls the C library's ``hypot``, whose last bit differs between
CPUs, so the committed renders would only reproduce on some machines.  The
simulator uses :func:`spindoctor.sim.stable_math.stable_hypot` instead.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from spindoctor.sim.stable_math import stable_hypot

_SIM_ROOT = Path(__file__).resolve().parents[3] / 'src' / 'spindoctor' / 'sim'


def test_stable_hypot_is_the_square_root_of_the_summed_squares() -> None:
    """Each element is ``sqrt(a*a + b*b)``, rounded once per operation."""
    a = np.random.default_rng(3).normal(size=200) * 100.0
    b = np.random.default_rng(4).normal(size=200) * 100.0
    expected = [math.sqrt(x * x + y * y) for x, y in zip(a.tolist(), b.tolist(), strict=True)]
    assert stable_hypot(a, b).tolist() == expected


def test_stable_hypot_broadcasts_its_legs() -> None:
    """A column and a row give the full grid of hypotenuses."""
    grid = stable_hypot(np.array([[3.0], [6.0]]), np.array([4.0, 8.0]))
    assert grid.tolist() == [[5.0, math.sqrt(73.0)], [math.sqrt(52.0), 10.0]]


def test_stable_hypot_agrees_with_numpy_to_rounding() -> None:
    """On pixel-scale values it is the hypotenuse, up to the last bit."""
    a = np.linspace(-500.0, 500.0, 101)
    b = np.linspace(0.0, 300.0, 101)
    np.testing.assert_allclose(stable_hypot(a, b), np.hypot(a, b), rtol=4e-16, atol=0.0)


def _hypot_calls(path: Path) -> list[int]:
    """Return the line of each ``np.hypot`` or ``numpy.hypot`` reference in a module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == 'hypot'
        and isinstance(node.value, ast.Name)
        and node.value.id in {'np', 'numpy'}
    ]


@pytest.mark.parametrize(
    'module', sorted(_SIM_ROOT.rglob('*.py')), ids=lambda p: str(p.relative_to(_SIM_ROOT))
)
def test_the_simulator_uses_stable_hypot_in_place_of_numpy_hypot(module: Path) -> None:
    """No simulator module references ``np.hypot``; it uses ``stable_hypot``."""
    assert _hypot_calls(module) == []
