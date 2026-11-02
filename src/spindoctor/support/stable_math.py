"""Arithmetic that rounds the same on every CPU.

Code whose output is compared exactly against committed results, such as the
simulator's render path, has to keep to operations whose results don't depend
on the CPU.  Plain IEEE arithmetic (``+``, ``-``, ``*``, ``/``, and ``sqrt``)
qualifies.  NumPy's ``hypot`` doesn't: it calls the C library's ``hypot``, whose
last bit differs between CPUs.  In a noisy simulated scene one changed bit can
change every pixel's noise.
"""

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['stable_hypot']


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
