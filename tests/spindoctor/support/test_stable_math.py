"""CPU-stable arithmetic: each result is the plain IEEE arithmetic, in a fixed order.

``stable_matmul3`` sums each element as ``a0*b0 + a1*b1 + a2*b2``, left to
right, so these tests pin that sum exactly.
"""

import math

import numpy as np

from spindoctor.support.stable_math import stable_hypot, stable_matmul3
from spindoctor.support.types import NDArrayFloatType


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


def _left_to_right(row: list[float], column: list[float]) -> float:
    """Sum ``row[k] * column[k]`` in index order, in Python floats."""
    return row[0] * column[0] + row[1] * column[1] + row[2] * column[2]


def _rows() -> NDArrayFloatType:
    """A few rows whose products round, so a regrouped sum would differ."""
    rows: NDArrayFloatType = np.random.default_rng(7).normal(size=(50, 3))
    rows *= np.array([1e3, 1.0, 1e-3])
    return rows


def test_a_row_times_a_vector_is_the_left_to_right_sum() -> None:
    """Each element of rows times a 3-vector is the index-order sum."""
    rows = _rows()
    vector = np.array([0.1, -2.7, 1e5])
    expected = [_left_to_right(list(row), list(vector)) for row in rows]
    assert stable_matmul3(rows, vector).tolist() == expected


def test_rows_times_a_matrix_are_the_left_to_right_sums() -> None:
    """Each element of rows times a 3-by-K matrix is the index-order sum."""
    rows = _rows()
    matrix = np.random.default_rng(8).normal(size=(3, 4))
    expected = [[_left_to_right(list(row), list(matrix[:, j])) for j in range(4)] for row in rows]
    assert stable_matmul3(rows, matrix).tolist() == expected


def test_stable_matmul3_agrees_with_numpy_to_rounding() -> None:
    """``stable_matmul3`` is the matrix product, up to rounding."""
    rows = _rows()
    matrix = np.random.default_rng(9).normal(size=(3, 3))
    np.testing.assert_allclose(stable_matmul3(rows, matrix), rows @ matrix, rtol=1e-12, atol=1e-9)
