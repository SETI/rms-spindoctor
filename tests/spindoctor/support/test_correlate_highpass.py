"""Tests for the high-pass cross-power sub-pixel refinement."""

from typing import Any

import numpy as np
import pytest

import spindoctor.support.correlate as correlate_module
from spindoctor.support.correlate import navigate_single_scale_kpeaks, navigate_with_pyramid_kpeaks
from spindoctor.support.correlate_subpixel import (
    HIGHPASS_SIGMA_PX,
    SubpixelRefinement,
    ncc_vertex_window_steps,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_SHAPE = (64, 64)
"""Shape of the image and the model in the test scenes, in pixels."""

_TRUE_SHIFT_VU = (1.45, -0.45)
"""Offset of the image's blob from the model's, in pixels."""

_RAMP_PER_PX = 0.003
"""Slope of the image's background along v, in blob peak heights per pixel."""


def _blob(center_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return a unit Gaussian blob of sigma 2 px centered at ``center_vu``."""
    rows, cols = np.mgrid[: _SHAPE[0], : _SHAPE[1]].astype(np.float64)
    return np.asarray(
        np.exp(-((rows - center_vu[0]) ** 2 + (cols - center_vu[1]) ** 2) / 8.0), np.float64
    )


def _sloped_scene() -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Return an image, model, and mask of one blob over a background that rises along v.

    The blob lies ``_TRUE_SHIFT_VU`` from the model's.  The sloped background is the
    kind of smooth light the model leaves out, such as stray light.
    """
    center = (_SHAPE[0] / 2.0, _SHAPE[1] / 2.0)
    rows = np.mgrid[: _SHAPE[0], : _SHAPE[1]][0].astype(np.float64)
    image = _blob((center[0] + _TRUE_SHIFT_VU[0], center[1] + _TRUE_SHIFT_VU[1]))
    image = image + _RAMP_PER_PX * rows
    model = _blob(center)
    mask: NDArrayBoolType = model > 1e-3
    return image, model, mask


def _quadratic_ncc(vertex_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return an exact downward quadratic surface maximal ``vertex_vu`` from cell (5, 5)."""
    rows, cols = np.mgrid[:11, :11].astype(np.float64)
    return np.asarray(
        1.0 - 0.1 * (rows - 5.0 - vertex_vu[0]) ** 2 - 0.2 * (cols - 5.0 - vertex_vu[1]) ** 2,
        np.float64,
    )


def _refine_sloped_scene(refinement: SubpixelRefinement) -> tuple[float, float]:
    """Return the single-scale offset of the sloped scene under one refinement."""
    image, model, mask = _sloped_scene()
    result = navigate_single_scale_kpeaks(
        image=image,
        model=model,
        mask=mask,
        upsample_factor=128,
        refinement=refinement,
        logger=None,
    )
    offset_v, offset_u = result['offset']
    return float(offset_v), float(offset_u)


# =========================================================================
# The window center
# =========================================================================


def test_window_center_is_the_ncc_vertex_on_the_upsampled_grid() -> None:
    """A vertex 0.7 px off the candidate in v becomes 90 steps of 1/128 px (89.6 rounded)."""
    steps = ncc_vertex_window_steps(_quadratic_ncc((0.7, -0.2)), (5, 5), 128)
    assert steps == (90, -26)


def test_window_center_stays_on_the_candidate_when_the_fit_fails() -> None:
    """When the quadratic fit cannot place a maximum, the window stays on the candidate."""
    corr = _quadratic_ncc((0.7, 0.0))
    corr[4, 5] = -np.inf
    assert ncc_vertex_window_steps(corr, (5, 5), 128) == (0, 0)


# =========================================================================
# The refinement on a scene the raw cross-power cannot resolve
# =========================================================================


def test_cross_power_refinement_stops_at_its_window_edge_on_a_sloped_background() -> None:
    """The sloped background drags the raw cross-power past its window: v stops at 1.5."""
    assert _refine_sloped_scene('cross_power')[0] == 1.5


def test_highpass_refinement_recovers_the_offset_on_a_sloped_background() -> None:
    """The high-pass refinement ignores the smooth background and recovers the offset."""
    offset = _refine_sloped_scene('highpass_cross_power')
    assert offset == pytest.approx(_TRUE_SHIFT_VU, abs=1.0 / 128.0)


def _ncc_peaking_at(vertex_vu: tuple[float, float]) -> Any:
    """Return a stand-in for ``masked_ncc`` whose surface is a quadratic maximal at a lag.

    The surface has the correlator's circular layout: index ``j`` along an axis of length
    ``n`` holds lag ``j`` for ``j < n // 2`` and lag ``j - n`` otherwise.
    """

    def _fake(
        image: NDArrayFloatType, model: NDArrayFloatType, mask: NDArrayBoolType, **_: Any
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        lags = [
            np.where(np.arange(n) < n // 2, np.arange(n), np.arange(n) - n) for n in image.shape
        ]
        surface = np.asarray(
            1.0
            - 0.001 * (lags[0][:, None] - vertex_vu[0]) ** 2
            - 0.001 * (lags[1][None, :] - vertex_vu[1]) ** 2,
            np.float64,
        )
        return surface, surface

    return _fake


def _refine_with_ncc_vertex(
    monkeypatch: pytest.MonkeyPatch, refinement: SubpixelRefinement
) -> tuple[float, float]:
    """Return the single-scale offset of a blob 0.3 px off when the NCC peaks at 0.75 px.

    The blob lies ``(0.3, -0.3)`` px from the model's on a flat background, and the NCC
    is replaced by a quadratic maximal at ``(0.75, -0.75)``, so its whole-pixel peak is
    ``(1, -1)`` and the window around that pixel, ``0.5`` to ``1.5`` px on v, misses the
    cross-power maximum.
    """
    monkeypatch.setattr(correlate_module, 'masked_ncc', _ncc_peaking_at((0.75, -0.75)))
    center = (_SHAPE[0] / 2.0, _SHAPE[1] / 2.0)
    model = _blob(center)
    mask: NDArrayBoolType = model > 1e-3
    result = navigate_single_scale_kpeaks(
        image=_blob((center[0] + 0.3, center[1] - 0.3)),
        model=model,
        mask=mask,
        upsample_factor=128,
        refinement=refinement,
        logger=None,
    )
    offset_v, offset_u = result['offset']
    return float(offset_v), float(offset_u)


def test_cross_power_window_around_the_whole_pixel_misses_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Centered on the whole pixel (1, -1), the raw window stops at (0.5, -0.5)."""
    offset = _refine_with_ncc_vertex(monkeypatch, 'cross_power')
    assert offset == pytest.approx((0.5, -0.5), abs=1e-12)


def test_highpass_window_centered_on_the_ncc_vertex_reaches_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Centered on the NCC vertex (0.75, -0.75), the window reaches the true (0.3, -0.3)."""
    offset = _refine_with_ncc_vertex(monkeypatch, 'highpass_cross_power')
    assert offset == pytest.approx((0.3, -0.3), abs=1.0 / 128.0)


# =========================================================================
# The pyramid
# =========================================================================


def test_only_the_final_pass_builds_a_highpass_spectrum(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coarser levels' spectra are unfiltered; the final pass's is high-passed."""
    sigmas: list[float] = []
    real_spectrum = correlate_module.refinement_cross_spectrum

    def _spy(*args: Any, **kwargs: Any) -> Any:
        sigmas.append(kwargs.get('highpass_sigma_px', 0.0))
        return real_spectrum(*args, **kwargs)

    monkeypatch.setattr(correlate_module, 'refinement_cross_spectrum', _spy)
    image, model, mask = _sloped_scene()
    navigate_with_pyramid_kpeaks(
        image, model, mask, pyramid_levels=3, final_refinement='highpass_cross_power'
    )
    assert sigmas == [0.0, 0.0, 0.0, HIGHPASS_SIGMA_PX]


def test_a_boundary_peak_refined_more_than_a_pixel_inward_is_at_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-pixel peak one pixel inside the search window's edge is flagged at_edge.

    The NCC is replaced by a quadratic maximal at 9.4 px on v and the search window is
    10 px, so the whole-pixel peak is 9 px, the eligible lag nearest the edge.  The NCC
    fit's maximum is set a full pixel inward of it, the farthest the fit may move, so the
    high-pass window spans 7.5 to 8.5 px and reaches the blob's true 7.6 px: the reported
    offset lies more than 2 px inside the edge, where an offset test alone misses it.
    """
    monkeypatch.setattr(correlate_module, 'masked_ncc', _ncc_peaking_at((9.4, 0.0)))
    monkeypatch.setattr(correlate_module, 'ncc_vertex_window_steps', lambda *_args: (-128, 0))
    center = (_SHAPE[0] / 2.0, _SHAPE[1] / 2.0)
    model = _blob(center)
    mask: NDArrayBoolType = model > 1e-3
    result = navigate_with_pyramid_kpeaks(
        _blob((center[0] + 7.6, center[1])),
        model,
        mask,
        pyramid_levels=3,
        upsample_factor=128,
        quality_thresh=-np.inf,
        consistency_tol=100.0,
        max_offset_vu=(10, 10),
        final_refinement='highpass_cross_power',
    )
    assert result['offset'][0] == pytest.approx(7.6, abs=1.0 / 128.0)
    assert result['at_edge']
