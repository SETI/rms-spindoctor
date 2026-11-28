"""Unit tests for cutting Voyager ISS stars out of the raw frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from util.star_photometry.star_field import kernel_star_image
from util.star_photometry.voyager_iss_star_limits import (
    RAW_CENTER,
    fit_frame_twist,
    geomed_stretch,
    raw_star_cutouts,
)

FloatArray = NDArray[np.float64]

# GEOMED pixels per raw pixel of the synthetic frames.
STRETCH = 1.25

# The raw position of the synthetic star, and the planted translation and twist.
STAR_RAW_VU = (400.3, 380.6)
SHIFT_V, SHIFT_U, TWIST_RAD = 0.6, -0.4, math.radians(0.2)


def _to_raw(points: FloatArray) -> FloatArray:
    """Map GEOMED positions to raw ones by a uniform stretch about the origin.

    Parameters:
        points: GEOMED ``(v, u)``, one row per point.

    Returns:
        Raw ``(v, u)``.
    """
    return points / STRETCH


def _frame() -> tuple[FloatArray, FloatArray, list[dict[str, Any]]]:
    """Return a raw frame holding one star on a flat sky, its GEOMED frame, and its star.

    Returns:
        ``(raw, geomed, stars)``.
    """
    kernel = {'sigma_v': 0.7, 'sigma_u': 0.7, 'w': 0.0, 'r0': 2.0, 'n': 3.0}
    raw = np.full((800, 800), 20.0)
    v, u = STAR_RAW_VU
    raw[380:421, 360:401] += np.round(
        800.0 * kernel_star_image(kernel, 41, v - 380, u - 360, truncation_px=8)
    )
    star = {'v': v * STRETCH, 'u': u * STRETCH, 'vmag': 5.0, 'detected': True}
    return raw, np.full((1000, 1000), 20.0), [star]


def _twisted(predicted: FloatArray) -> FloatArray:
    """Return predicted positions moved by the planted translation and twist.

    Parameters:
        predicted: ``(v, u)``, one row per star.

    Returns:
        The moved positions.
    """
    dv, du = predicted[:, 0] - RAW_CENTER, predicted[:, 1] - RAW_CENTER
    return np.column_stack(
        [
            dv + SHIFT_V - TWIST_RAD * du + RAW_CENTER,
            du + SHIFT_U + TWIST_RAD * dv + RAW_CENTER,
        ]
    )


PREDICTED = np.array(
    [[100.0, 120.0], [650.0, 200.0], [400.0, 700.0], [210.0, 580.0], [720.0, 640.0], [380.0, 90.0]]
)


def test_a_frame_twist_is_recovered_from_its_stars() -> None:
    """Six stars moved by a translation and twist give back the twist."""
    twist = fit_frame_twist(PREDICTED, _twisted(PREDICTED), center=RAW_CENTER)
    assert twist.twist_rad == pytest.approx(TWIST_RAD, abs=1e-9)


def test_a_star_off_the_frame_twist_is_left_out() -> None:
    """A star 3 pixels from where the other stars put it is left out of the fit."""
    measured = _twisted(PREDICTED)
    measured[1, 0] += 3.0
    twist = fit_frame_twist(PREDICTED, measured, center=RAW_CENTER)
    assert twist.used.tolist() == [True, False, True, True, True, True]


def test_two_stars_fit_only_a_translation() -> None:
    """With fewer than three stars the frame takes a translation and no twist."""
    twist = fit_frame_twist(PREDICTED[:2], _twisted(PREDICTED[:2]), center=RAW_CENTER)
    assert twist.model == 'translation'


def test_the_stretch_is_geomed_pixels_per_raw_pixel() -> None:
    """A map shrinking GEOMED positions by 1.25 has a stretch of 1.25."""
    assert geomed_stretch(_to_raw, 500.0, 500.0) == pytest.approx(STRETCH)


def _despiked_cutout() -> dict[str, Any]:
    """Return the star's cutout when despike changed one core pixel and one sky pixel.

    Returns:
        The star's record.
    """
    raw, geomed, stars = _frame()
    cleaned = raw.copy()
    cleaned[400, 381] -= 30.0
    cleaned[400, 388] += 9.0
    (cutout,) = raw_star_cutouts(raw, cleaned, geomed, _to_raw, stars, np.zeros((0, 2)))
    return cutout


def test_a_despiked_core_pixel_keeps_its_raw_value() -> None:
    """A core pixel the despike step changed stays usable, since it is the star's own."""
    cutout = _despiked_cutout()
    assert cutout['valid'][400 - cutout['v0'], 381 - cutout['u0']]


def test_a_despiked_sky_pixel_is_left_out() -> None:
    """A pixel the despike step changed outside the core held a spike, so it goes."""
    cutout = _despiked_cutout()
    assert not cutout['valid'][400 - cutout['v0'], 388 - cutout['u0']]


def test_the_despiked_core_pixels_are_counted() -> None:
    """The record counts the core pixels the despike step changed."""
    assert _despiked_cutout()['core_changed'] == 1


def test_a_star_beside_a_reseau_mark_is_not_cut_out() -> None:
    """A star 4 raw pixels from a reseau mark is not cut out."""
    raw, geomed, stars = _frame()
    reseau = np.array([[STAR_RAW_VU[0] + 4.0, STAR_RAW_VU[1]]])
    assert raw_star_cutouts(raw, raw.copy(), geomed, _to_raw, stars, reseau) == []
