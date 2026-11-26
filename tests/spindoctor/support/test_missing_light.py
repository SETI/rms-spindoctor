"""Tests for ``spindoctor.support.missing_light``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spindoctor.support.missing_light import missing_light_shift_px


def _mask_with_rows(rows: list[int]) -> np.ndarray:
    """Return a 10 by 10 mask with the given rows missing."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[rows, :] = True
    return mask


@pytest.mark.parametrize(
    ('rows', 'expected'),
    [
        ([], 0.0),
        ([6], 0.5),
        ([4, 6], 0.0),
        ([4, 5, 6], math.inf),
    ],
    ids=['nothing_missing', 'one_side_missing', 'both_sides_missing', 'all_missing'],
)
def test_missing_light_shift_px(rows: list[int], expected: float) -> None:
    """A model on rows 4 to 6 shifts by the light its missing rows take away.

    Losing row 6 moves the centroid from row 5 to row 4.5; losing rows 4 and 6 leaves it
    on row 5.
    """
    model = np.zeros((3, 3), dtype=np.float64)
    model[:, 1] = 1.0
    assert missing_light_shift_px(model, (4, 2), _mask_with_rows(rows)) == expected


def test_missing_light_shift_px_counts_pixels_beyond_the_mask_as_present() -> None:
    """A model hanging over the mask's top edge keeps the light the mask does not cover."""
    model = np.ones((3, 1), dtype=np.float64)
    mask = _mask_with_rows([0])
    assert missing_light_shift_px(model, (-2, 0), mask) == pytest.approx(0.5)
