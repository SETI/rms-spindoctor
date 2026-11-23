"""The fixed pattern: the response, the stitch comb, and the jail bars."""

import numpy as np
import pytest

from spindoctor.sim.render.detector.fixed_pattern import (
    add_fixed_pattern_dn,
    add_fixed_pattern_response,
)

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


def test_fixed_pattern_response_is_stable_per_seed() -> None:
    """The multiplicative fixed pattern is identical for equal seeds."""
    a = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    b = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    add_fixed_pattern_response(
        a, prnu_rms=0.01, vignetting_frac=0.04, dust_donut_count=2, rng=_rng(9)
    )
    add_fixed_pattern_response(
        b, prnu_rms=0.01, vignetting_frac=0.04, dust_donut_count=2, rng=_rng(9)
    )
    assert np.array_equal(a, b)


def test_fixed_pattern_vignetting_darkens_the_corner() -> None:
    """Corner vignetting lowers the corner response below the center."""
    electrons = np.full((_SIZE, _SIZE), 1000.0, dtype=np.float64)
    add_fixed_pattern_response(
        electrons, prnu_rms=0.0, vignetting_frac=0.1, dust_donut_count=0, rng=_rng(1)
    )
    center = float(electrons[_SIZE // 2, _SIZE // 2])
    corner = float(electrons[0, 0])
    assert corner < center


def test_fixed_pattern_stitch_comb_raises_periodic_columns() -> None:
    """The stitch comb raises columns on the stitch period by the amplitude."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_fixed_pattern_dn(
        dn, stitch_period_px=16, stitch_amplitude_dn=5.0, jail_bar_dn=0.0, rng=_rng(2)
    )
    assert float(dn[10, 0]) == pytest.approx(5.0)
    assert float(dn[10, 16]) == pytest.approx(5.0)
    assert float(dn[10, 1]) == pytest.approx(0.0)


def test_fixed_pattern_jail_bars_offset_even_odd_columns() -> None:
    """Jail bars offset even and odd columns in opposite directions."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_fixed_pattern_dn(
        dn, stitch_period_px=0, stitch_amplitude_dn=0.0, jail_bar_dn=0.5, rng=_rng(2)
    )
    assert float(dn[5, 0]) == pytest.approx(-float(dn[5, 1]))
    assert abs(float(dn[5, 0])) == pytest.approx(0.5)


def test_fixed_pattern_dn_disabled_at_zero() -> None:
    """A zeroed stitch comb and jail bar is a no-op."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_fixed_pattern_dn(
        dn, stitch_period_px=0, stitch_amplitude_dn=0.0, jail_bar_dn=0.0, rng=_rng(2)
    )
    assert record['active'] is False
    assert np.all(dn == 0.0)
