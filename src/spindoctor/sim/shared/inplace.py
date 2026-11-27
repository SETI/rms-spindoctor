"""Helpers for building an arithmetic expression in place, with its plain form's result.

The simulator evaluates several closed forms over whole oversampled frames, where a
fresh frame-sized temporary costs more than the arithmetic in it.  Those forms are
built step by step in one array with NumPy's ``out=`` arguments and in-place
operators.  Each step is the IEEE operation of the plain expression, so the values
are the same to the last bit, but the plain expression also decides what comes back:
the shape the operands broadcast to, the floating type NumPy promotes them to, and a
NumPy scalar rather than an array when every operand is a scalar.  These helpers give
an in-place form the same result.

:func:`result_array` allocates the array to build in, and :func:`ufunc_result`
returns it the way a chain of NumPy functions returns its result.  An expression
whose last step is ``np.where`` returns an array even for scalar operands, so its
in-place form returns the array as it is.
"""

import numpy as np

from spindoctor.support.types import NDArrayFloatType, NDArrayLike

__all__ = ['result_array', 'ufunc_result']


def result_array(*operands: NDArrayLike) -> NDArrayFloatType:
    """An uninitialized array to build an expression of ``operands`` in.

    It has the shape the operands broadcast to and the type NumPy gives the result of
    an arithmetic expression of them with a Python float in it: float32 operands keep
    float32, and integers become float64.  Each operand may be an array, a NumPy
    scalar, or a Python number.

    Parameters:
        operands: The arrays and numbers the expression combines.

    Returns:
        The array, with every dimension of the broadcast, none for scalar operands.
    """
    shape = np.broadcast_shapes(*(np.shape(operand) for operand in operands))
    return np.empty(shape, dtype=np.result_type(*operands, 1.0))


def ufunc_result(array: NDArrayFloatType) -> NDArrayFloatType:
    """Return ``array`` as a chain of NumPy functions returns its result.

    That is the array itself, or its one value as a NumPy scalar when it has no
    dimensions.

    Parameters:
        array: The array the expression was built in.

    Returns:
        The array, or its value when it has no dimensions.
    """
    if array.ndim:
        return array
    return array[()]
