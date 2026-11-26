"""The committed passband tables are what ``util/passbands/build_passbands.py`` builds.

The builder reads the Cassini ISS calibration volume in the PDS archive, so the test
needs the holdings: it runs only when ``PDS3_HOLDINGS_DIR`` names a local holdings tree
that carries the volume, and skips otherwise.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from spindoctor.sim.instruments.passbands import PASSBAND_DIR

pytestmark = [pytest.mark.integration]

_BUILDER_PATH = Path(__file__).resolve().parents[2] / 'util' / 'passbands' / 'build_passbands.py'
_EFFICIENCY = Path('volumes/COISS_0xxx/COISS_0011/calib/efficiency')


def _load_builder() -> ModuleType:
    """Load ``util/passbands/build_passbands.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('passbands_build', _BUILDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_passbands_are_the_built_passbands() -> None:
    """Rebuilding every table from the calibration volume reproduces the committed files."""
    holdings = os.environ.get('PDS3_HOLDINGS_DIR', '')
    efficiency = Path(holdings) / _EFFICIENCY
    if not holdings or not efficiency.is_dir():
        pytest.skip('needs a local PDS3_HOLDINGS_DIR carrying COISS_0011')
    built = _load_builder().build(efficiency)
    committed = {name: (PASSBAND_DIR / name).read_text() for name in built}
    assert built == committed
