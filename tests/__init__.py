"""The test suite.

Importing this package pins NumPy's SIMD dispatch and OpenBLAS's kernel to the
x86-64-v2 baseline before NumPy is first imported.  Both pick vectorized code
paths by CPU, and the paths differ in the last bits of their results, so a
simulated render, its rounded navigation outcome, and a sweep check near its
threshold depend on the CPU that ran them.  With every x86 machine on the baseline
paths, the committed simulator baselines, renders, and gallery images reproduce on
any of them.  The package is imported both by pytest and by the
``python -m tests.integration...`` tools that regenerate those files, so both see
the same pins.

The pins apply on x86-64; the NumPy pin needs NumPy 2.4 or later, whose dispatch
groups it names.  Each is a default: a value already in
``NPY_DISABLE_CPU_FEATURES`` or ``OPENBLAS_CORETYPE`` wins.
"""

import os
import platform
from importlib.metadata import version

NUMPY_SIMD_ABOVE_BASELINE = 'X86_V3 X86_V4 AVX512_ICL AVX512_SPR'
"""The NumPy dispatch groups above the x86-64-v2 baseline, which the pin disables."""

OPENBLAS_BASELINE_CORETYPE = 'Nehalem'
"""The OpenBLAS kernel the pin selects: the x86-64-v2 one, which every such CPU runs."""


def _pin_simd() -> None:
    """Default NumPy's SIMD dispatch and OpenBLAS's kernel to the x86-64-v2 baseline."""
    if platform.machine().lower() not in {'x86_64', 'amd64'}:
        return
    os.environ.setdefault('OPENBLAS_CORETYPE', OPENBLAS_BASELINE_CORETYPE)
    major, minor = (int(part) for part in version('numpy').split('.')[:2])
    if (major, minor) < (2, 4):
        return
    os.environ.setdefault('NPY_DISABLE_CPU_FEATURES', NUMPY_SIMD_ABOVE_BASELINE)


_pin_simd()
