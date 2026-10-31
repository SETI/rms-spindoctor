"""The mesh renderer's matrix products are summed in a fixed order.

NumPy's ``@`` goes to OpenBLAS, whose kernel is chosen by CPU and rounds
differently from one kernel to the next.  The mesh renderer sums its 3-vector
products itself, so each element is the plain left-to-right sum
``a0*b0 + a1*b1 + a2*b2`` on every CPU.  These tests pin that sum exactly.
"""

import numpy as np

from spindoctor.sim.mesh_geometry import _times_xyz
from spindoctor.support.types import NDArrayFloatType


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
    assert _times_xyz(rows, vector).tolist() == expected


def test_rows_times_a_matrix_are_the_left_to_right_sums() -> None:
    """Each element of rows times a 3-by-K matrix is the index-order sum."""
    rows = _rows()
    matrix = np.random.default_rng(8).normal(size=(3, 4))
    expected = [[_left_to_right(list(row), list(matrix[:, j])) for j in range(4)] for row in rows]
    assert _times_xyz(rows, matrix).tolist() == expected


def test_the_fixed_order_product_agrees_with_numpy_to_rounding() -> None:
    """The fixed-order product is the matrix product, up to rounding."""
    rows = _rows()
    matrix = np.random.default_rng(9).normal(size=(3, 3))
    np.testing.assert_allclose(_times_xyz(rows, matrix), rows @ matrix, rtol=1e-12, atol=1e-9)
