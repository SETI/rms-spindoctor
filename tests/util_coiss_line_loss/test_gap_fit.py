"""Tests for the Cassini ISS downlink-gap comparison.

The comparison lives in ``util/coiss_line_loss/gap_fit.py``, outside the importable
package, and is loaded under its own module name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_FIT_PATH = Path(__file__).resolve().parents[2] / 'util' / 'coiss_line_loss' / 'gap_fit.py'


def _load_gap_fit() -> ModuleType:
    """Load ``util/coiss_line_loss/gap_fit.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('coiss_gap_fit', _FIT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap_fit = _load_gap_fit()


_FULL_FRAMES = [
    {'name': 'N1500000000_1', 'lines': 64, 'mode': 'FULL', 'groups': [[10, 12, 9, None]]},
    {'name': 'N1500000100_1', 'lines': 64, 'mode': 'FULL', 'groups': [[30, 31, None, 8]]},
]


def test_a_census_of_full_frames_skips_the_summed_comparison(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A census of full frames compares all and full frames and skips the summed set."""
    gap_fit.compare(_FULL_FRAMES, 'coiss_nac', 20, {})
    out = capsys.readouterr().out
    assert '== full frames: 2 real, ' in out
    assert '== summed frames: 0 real, 0 simulated; skipped' in out


def test_a_set_the_simulation_lacks_is_skipped(capsys: pytest.CaptureFixture[str]) -> None:
    """With no simulated frames, even the set of all census frames is skipped."""
    gap_fit.compare(_FULL_FRAMES, 'coiss_nac', 0, {})
    out = capsys.readouterr().out
    assert '== all frames: 2 real, 0 simulated; skipped' in out


def test_an_empty_census_is_skipped_for_every_set(capsys: pytest.CaptureFixture[str]) -> None:
    """A camera with no census frames skips every set instead of cycling an empty list."""
    gap_fit.compare([], 'coiss_nac', 20, {})
    out = capsys.readouterr().out
    assert '== all frames: 0 real, 0 simulated; skipped' in out
