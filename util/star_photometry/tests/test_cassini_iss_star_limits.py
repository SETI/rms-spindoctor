"""Unit tests for the Cassini ISS star-field measurement's raw-image decoding."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from util.star_photometry.cassini_iss_star_limits import decode_raw

#: A stand-in inverse lookup table: 12-bit DN 10 times the 8-bit DN, plus one.
LUT: NDArray[np.float64] = 10.0 * np.arange(256, dtype=np.float64) + 1.0


def test_a_table_encoded_image_is_decoded_through_the_lookup_table() -> None:
    """Each 8-bit value is replaced by its table entry."""
    decoded = decode_raw(np.array([[3, 255]], dtype=np.uint8), 'TABLE', LUT)
    assert decoded.tolist() == [[31.0, 2551.0]]


def test_a_missing_pixel_of_a_table_encoded_image_stays_zero() -> None:
    """A stored 0 marks a missing pixel, not the table's first entry."""
    decoded = decode_raw(np.array([[0, 1]], dtype=np.uint8), 'TABLE', LUT)
    assert decoded.tolist() == [[0.0, 11.0]]


def test_a_twelve_bit_image_is_read_as_it_is() -> None:
    """A 12BIT image's values are already 12-bit DN."""
    decoded = decode_raw(np.array([[0, 4095]], dtype=np.uint16), '12BIT', LUT)
    assert decoded.tolist() == [[0.0, 4095.0]]


def test_an_eight_bit_truncated_image_is_refused() -> None:
    """An 8LSB image kept only the low 8 bits, so it cannot be decoded."""
    with pytest.raises(ValueError, match="unsupported DATA_CONVERSION_TYPE '8LSB'"):
        decode_raw(np.zeros((1, 1)), '8LSB', LUT)
