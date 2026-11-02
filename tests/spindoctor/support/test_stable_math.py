"""CPU-stable arithmetic: ``stable_hypot`` rounds like plain IEEE arithmetic."""

import math

import numpy as np

from spindoctor.support.stable_math import stable_hypot


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
