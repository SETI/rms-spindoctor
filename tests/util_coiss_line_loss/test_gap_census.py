"""Tests for the Cassini ISS downlink-gap census's tables.

The census lives in ``util/coiss_line_loss/gap_census.py``, outside the importable
package, and is loaded under its own module name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from spindoctor.sim.render.telemetry.loss import _percentile_draws

_CENSUS_PATH = Path(__file__).resolve().parents[2] / 'util' / 'coiss_line_loss' / 'gap_census.py'


def _load_gap_census() -> ModuleType:
    """Load ``util/coiss_line_loss/gap_census.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('coiss_gap_census', _CENSUS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap_census = _load_gap_census()


def test_a_rare_smallest_size_keeps_its_share_under_logarithmic_draws() -> None:
    """A size measured on 2% of the groups comes out of the simulator's draw on 2% of them.

    The smallest size is rarer than the table's 5th percentile, so the stretch below
    that point also holds the next size, where a logarithmic draw from the half line
    below the smallest size would round to it on about 4.7% of the draws.
    """
    sizes = [1] * 2 + [2] * 30 + [3] * 30 + [5] * 38
    table = gap_census.spread_percentiles(sizes)
    draws = _percentile_draws(np.random.default_rng(3), table, 200_000, log=True)
    share = float(np.mean(np.rint(draws) == 1))
    assert abs(share - 0.02) < 0.002
