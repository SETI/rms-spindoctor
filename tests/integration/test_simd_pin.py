"""The test package's SIMD pins are in effect.

The committed simulator baselines, renders, and gallery images reproduce only
when NumPy and OpenBLAS run their x86-64-v2 code paths, which ``tests/__init__.py``
selects before NumPy is first imported.  A plugin or conftest that imported NumPy
earlier would leave the pins without effect, and the simulator checks would then
depend on the CPU again; these tests catch that.  An explicit value in the
environment overrides a pin on purpose, so the code-path checks skip then.
"""

from __future__ import annotations

import importlib
import os
import platform
from importlib.metadata import version

import pytest
import scipy.linalg  # noqa: F401  (loads SciPy's own OpenBLAS for the kernel check)
from threadpoolctl import threadpool_info

from tests import NUMPY_SIMD_ABOVE_BASELINE, OPENBLAS_BASELINE_CORETYPE

_X86 = platform.machine().lower() in {'x86_64', 'amd64'}
_NUMPY_PINNED = _X86 and tuple(int(part) for part in version('numpy').split('.')[:2]) >= (2, 4)

# NumPy publishes its dispatch state only on this private module, which has no stubs.
_UMATH = importlib.import_module('numpy._core._multiarray_umath')
_DISPATCH: list[str] = list(_UMATH.__cpu_dispatch__)
_FEATURES: dict[str, bool] = dict(_UMATH.__cpu_features__)


@pytest.mark.skipif(not _NUMPY_PINNED, reason='the NumPy pin applies to x86-64 on NumPy >= 2.4')
def test_the_numpy_pin_is_in_the_environment() -> None:
    """Importing the test package leaves the NumPy pin set."""
    assert 'NPY_DISABLE_CPU_FEATURES' in os.environ


@pytest.mark.skipif(not _NUMPY_PINNED, reason='the NumPy pin applies to x86-64 on NumPy >= 2.4')
def test_the_numpy_pin_names_every_dispatch_group() -> None:
    """The pin names exactly the groups this NumPy dispatches to above the baseline."""
    assert NUMPY_SIMD_ABOVE_BASELINE.split() == _DISPATCH


@pytest.mark.skipif(
    not _NUMPY_PINNED or os.environ.get('NPY_DISABLE_CPU_FEATURES') != NUMPY_SIMD_ABOVE_BASELINE,
    reason='the NumPy pin is off or overridden',
)
def test_numpy_runs_its_baseline_path() -> None:
    """No dispatch group above the baseline is enabled."""
    enabled = [group for group in _DISPATCH if _FEATURES[group]]
    assert enabled == []


@pytest.mark.skipif(not _X86, reason='the OpenBLAS pin applies to x86-64')
def test_the_openblas_pin_is_in_the_environment() -> None:
    """Importing the test package leaves the OpenBLAS pin set."""
    assert 'OPENBLAS_CORETYPE' in os.environ


@pytest.mark.skipif(
    not _X86 or os.environ.get('OPENBLAS_CORETYPE') != OPENBLAS_BASELINE_CORETYPE,
    reason='the OpenBLAS pin is off or overridden',
)
def test_every_openblas_runs_the_baseline_kernel() -> None:
    """NumPy's and SciPy's OpenBLAS libraries both load the pinned kernel."""
    kernels = {
        info['filepath']: info.get('architecture')
        for info in threadpool_info()
        if info['internal_api'] == 'openblas'
    }
    assert len(kernels) >= 1
    assert set(kernels.values()) == {OPENBLAS_BASELINE_CORETYPE}
