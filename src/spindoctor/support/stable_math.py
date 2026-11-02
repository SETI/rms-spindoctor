"""Arithmetic that rounds the same on every CPU.

Code whose output is compared exactly against committed results, such as the
simulator's render path, has to keep to operations whose results don't depend
on the CPU.  Plain IEEE arithmetic (``+``, ``-``, ``*``, ``/``, and ``sqrt``)
qualifies.  NumPy's ``hypot`` doesn't: it calls the C library's ``hypot``, whose
last bit differs between CPUs.  Nor does ``@``: NumPy hands it to OpenBLAS, which
picks its kernel by CPU, and the kernels round differently.  In a noisy simulated
scene one changed bit can change every pixel's noise.
"""

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['stable_hypot', 'stable_matmul3']


def stable_hypot(a: NDArrayFloatType | float, b: NDArrayFloatType | float) -> NDArrayFloatType:
    """Return ``sqrt(a**2 + b**2)`` elementwise, rounded the same on every CPU.

    Unlike ``np.hypot`` it doesn't guard against overflow, so it's meant for
    pixel-scale and unit-scale values.

    Parameters:
        a: The first leg; an array or a scalar.
        b: The second leg, broadcastable against ``a``.

    Returns:
        The float64 hypotenuse, shaped like ``a`` and ``b`` broadcast together.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    result: NDArrayFloatType = np.sqrt(a_arr * a_arr + b_arr * b_arr)
    return result


def stable_matmul3(rows: NDArrayFloatType, other: NDArrayFloatType) -> NDArrayFloatType:
    """Return ``rows @ other`` for ``(N, 3)`` rows, rounded the same on every CPU.

    Each element is the left-to-right sum ``r0*o0 + r1*o1 + r2*o2`` in plain
    multiplies and adds, where ``@`` would take whatever order the CPU's OpenBLAS
    kernel uses.

    Parameters:
        rows: The ``(N, 3)`` left operand.
        other: The right operand, a 3-vector or a ``(3, K)`` matrix.

    Returns:
        The contiguous float64 product: ``(N,)`` for a vector, ``(N, K)`` for a
        matrix.
    """
    cols = rows.T
    if other.ndim == 1:
        product = cols[0] * other[0] + cols[1] * other[1] + cols[2] * other[2]
    else:
        product = (
            cols[0][:, None] * other[0] + cols[1][:, None] * other[1] + cols[2][:, None] * other[2]
        )
    result: NDArrayFloatType = np.ascontiguousarray(product, dtype=np.float64)
    return result
