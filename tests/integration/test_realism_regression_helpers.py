"""Hermetic tests for the realism regression check's bands and comparison.

The comparison reads nothing but its arguments, so this module carries no
environment guard and no ``integration`` marker and runs in the plain unit tier.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.integration.realism_regression import (
    REALISM_BANDS,
    RELATIVE_FLOOR,
    RealismBand,
    band_rule,
    compare_with_record,
    held_statistics,
)
from tests.integration.sim_realism_report import code_commit

_EE50 = 'divergences.star_ee50.w1_normalized'
_BANDS = {_EE50: RealismBand(value=2.33, spread=0.0, band=0.0466)}


def _entry(w1: float, *, support: str = 'limited') -> dict[str, Any]:
    """An instrument summary entry holding one EE50 divergence."""
    return {
        'divergences': {
            'star_ee50': {
                'w1_normalized': w1,
                'real_median': None,
                'sim_median': None,
                'support': support,
            }
        }
    }


def test_a_move_beyond_the_band_names_the_statistic_and_both_values() -> None:
    """A better-looking value beyond the band fails: the bands are two-sided."""
    problems = compare_with_record(_entry(1.04), _entry(2.33), _BANDS)
    assert problems == [
        f'{_EE50} moved from 2.33 to 1.04, beyond its band of +-0.0466 (run-to-run spread 0).'
    ]


def test_a_move_within_the_band_passes() -> None:
    """A value within its band of the record raises nothing."""
    assert compare_with_record(_entry(2.36), _entry(2.33), _BANDS) == []


def test_a_held_statistic_without_a_band_fails() -> None:
    """A statistic the table has no band for is reported, never passed."""
    problems = compare_with_record(_entry(2.33), _entry(2.33), {})
    assert problems == [f'{_EE50} has no band; measure its spread and add it to REALISM_BANDS.']


def test_an_unsupported_statistic_is_not_held() -> None:
    """A statistic whose support label is unsupported is left out of the check."""
    assert held_statistics(_entry(2.33, support='unsupported')) == {}


def test_a_band_with_no_measured_spread_is_the_floor() -> None:
    """With no run-to-run spread, the band is the relative floor of the value."""
    assert band_rule(0.0, 2.5) == pytest.approx(RELATIVE_FLOOR * 2.5)


@pytest.mark.parametrize('path', sorted(REALISM_BANDS))
def test_every_band_follows_the_rule(path: str) -> None:
    """Each committed band is the rule applied to its measured spread and value."""
    band = REALISM_BANDS[path]
    assert band.band == band_rule(band.spread, band.value)


def test_the_summary_records_the_code_commit() -> None:
    """In a checkout, the summary's code commit is a full commit hash."""
    assert re.fullmatch(r'[0-9a-f]{40}', code_commit() or '')
