"""Tests for :func:`spindoctor.support.correlate.ncc_peak_quality`."""

import math

import numpy as np
import pytest

from spindoctor.support.correlate import (
    NccPeak,
    int_to_signed,
    navigate_single_scale_kpeaks,
    ncc_peak_quality,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType


def _shifted_scene(
    *, with_data_mask: bool
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayBoolType, NDArrayBoolType | None]:
    """Return a noisy image, the model it shows shifted by (3, -2) px, and the masks.

    Parameters:
        with_data_mask: Whether to blank a margin of the image and return a data mask
            marking the rest.

    Returns:
        The image, the model, the model mask, and the data mask (or None).
    """
    rng = np.random.default_rng(11)
    vs, us = np.meshgrid(np.arange(70), np.arange(80), indexing='ij')
    model = np.where(np.hypot(vs - 34.5, (us - 39.5) * 0.7) < 18.0, 40.0 + 0.5 * us, 0.0)
    image = np.roll(model, (3, -2), axis=(0, 1)) + rng.normal(scale=2.0, size=model.shape)
    data_mask: NDArrayBoolType | None = None
    if with_data_mask:
        data_mask = np.zeros(model.shape, dtype=bool)
        data_mask[6:-6, 6:-6] = True
        image = np.where(data_mask, image, 0.0)
    return image, model, model > 0.0, data_mask


@pytest.mark.parametrize('with_data_mask', [False, True], ids=['no_data_mask', 'data_mask'])
def test_peak_quality_matches_the_single_scale_search(with_data_mask: bool) -> None:
    """The peak and quality are those the single-scale search gives its winner."""
    image, model, mask, data_mask = _shifted_scene(with_data_mask=with_data_mask)
    full = navigate_single_scale_kpeaks(
        image=image,
        model=model,
        mask=mask,
        logger=None,
        max_peaks=1,
        max_offset_vu=(10, 10),
        data_mask=data_mask,
    )
    peak = ncc_peak_quality(image, model, mask, max_offset_vu=(10, 10), data_mask=data_mask)
    expected_offset = (
        int_to_signed(full['rc'][0], image.shape[0] + model.shape[0]),
        int_to_signed(full['rc'][1], image.shape[1] + model.shape[1]),
    )
    assert peak == NccPeak(offset_vu=expected_offset, quality=full['quality'])


def test_peak_quality_finds_the_planted_shift() -> None:
    """The whole-pixel peak is the shift the model was planted at."""
    image, model, mask, _ = _shifted_scene(with_data_mask=False)
    peak = ncc_peak_quality(image, model, mask, max_offset_vu=(10, 10))
    assert peak.offset_vu == (3, -2)


def test_peak_quality_without_a_valid_shift_is_minus_infinity() -> None:
    """A search limit that admits no shift returns zero shift and quality -inf."""
    image, model, mask, _ = _shifted_scene(with_data_mask=False)
    peak = ncc_peak_quality(image, model, mask, max_offset_vu=(0, 0))
    assert peak == NccPeak(offset_vu=(0, 0), quality=-math.inf)
