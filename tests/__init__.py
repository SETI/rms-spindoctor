"""The test suite.

Importing this package pins NumPy's SIMD dispatch to the x86-64 baseline before
NumPy is first imported.  NumPy picks vectorized code paths by CPU, and they
differ in the last bits of their results, so a simulated render, its rounded
navigation outcome, and a sweep check near its threshold depend on the CPU that
ran them.  With every x86 machine on the baseline path, the committed simulator
baselines, renders, and gallery images reproduce on any of them.  The package is
imported both by pytest and by the ``python -m tests.integration...`` tools that
regenerate those files, so both see the same pin.

The pin applies on x86-64 with NumPy 2.4 or later, whose dispatch groups it
names.  It is a default: a value already in ``NPY_DISABLE_CPU_FEATURES`` wins.
"""

import os
import platform
from importlib.metadata import version

NUMPY_SIMD_ABOVE_BASELINE = 'X86_V3 X86_V4 AVX512_ICL AVX512_SPR'
"""The NumPy dispatch groups above the x86-64-v2 baseline, which the pin disables."""


def _pin_numpy_simd() -> None:
    """Default ``NPY_DISABLE_CPU_FEATURES`` to every group above the baseline."""
    if platform.machine().lower() not in {'x86_64', 'amd64'}:
        return
    major, minor = (int(part) for part in version('numpy').split('.')[:2])
    if (major, minor) < (2, 4):
        return
    os.environ.setdefault('NPY_DISABLE_CPU_FEATURES', NUMPY_SIMD_ABOVE_BASELINE)


_pin_numpy_simd()
