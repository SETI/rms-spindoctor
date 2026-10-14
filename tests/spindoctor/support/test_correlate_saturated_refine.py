"""Tests for the correlator's saturated sub-pixel refinement fallback.

When the upsampled-DFT maximum lands on the edge of its half-pixel window, each
saturated axis falls back to the quadratic vertex of the NCC surface.  These tests
cover the three-point quadratic and the fallback inside ``evaluate_candidate``.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.support.conftest import gaussian_patch

from spindoctor.config import IMAGE_LOGGER
from spindoctor.support.correlate import (
    _ncc_quadratic_axis_offset,
    evaluate_candidate,
    fourier_shift,
)


def test_ncc_quadratic_axis_offset_recovers_parabola_vertex() -> None:
    """The three-point fit returns the exact vertex of a sampled parabola."""
    vertex = 0.31
    samples = [-((t - vertex) ** 2) for t in (-1.0, 0.0, 1.0)]
    assert _ncc_quadratic_axis_offset(*samples) == pytest.approx(vertex)


def test_ncc_quadratic_axis_offset_nonfinite_neighbor_returns_zero() -> None:
    """An invalid-shift sentinel next to the peak disables the fit."""
    assert _ncc_quadratic_axis_offset(float('-inf'), 1.0, 0.5) == 0.0


def test_ncc_quadratic_axis_offset_flat_neighborhood_returns_zero() -> None:
    """A flat (curvature-free) neighborhood carries no sub-pixel information."""
    assert _ncc_quadratic_axis_offset(0.5, 0.5, 0.5) == 0.0


def test_ncc_quadratic_axis_offset_degenerate_curvature_returns_zero() -> None:
    """Rounding-noise curvature keeps the integer peak instead of exploding."""
    assert _ncc_quadratic_axis_offset(0.1, 0.2, 0.3) == 0.0


def test_ncc_quadratic_axis_offset_neighbor_tie_is_half_pixel() -> None:
    """A neighbor equal to the peak puts the vertex exactly halfway between them."""
    assert _ncc_quadratic_axis_offset(0.5, 1.0, 1.0) == pytest.approx(0.5)


def test_ncc_quadratic_axis_offset_neighbor_above_center_returns_zero() -> None:
    """A neighbor above the center is not a local maximum; keep the integer peak.

    A steeply tilted concave triple has negative curvature yet its highest
    sample is a neighbor; fitting it would report a clipped half-pixel shift
    away from the true maximum.
    """
    assert _ncc_quadratic_axis_offset(1.0, 0.9, 0.0) == 0.0
    assert _ncc_quadratic_axis_offset(0.0, 0.9, 1.0) == 0.0


def _evaluate_candidate_for_shift(
    true_shift_vu: tuple[float, float],
    corr: np.ndarray | None = None,
    *,
    upsample_factor: int = 64,
) -> dict[str, Any]:
    """Run evaluate_candidate on a Gaussian scene with integer peak (0, 0).

    The cross-power spectrum's true peak sits at ``true_shift_vu``; passing
    ``rc=(0, 0)`` makes the refinement window span ``[-0.5, +0.5]`` px, so a
    true shift beyond half a pixel saturates the window.  ``corr`` defaults
    to a surface whose quadratic vertex along v is ``+0.25``.

    Parameters:
        true_shift_vu: The ``(dv, du)`` shift in pixels planted between the
            image and the model via a Fourier shift.
        corr: Optional fabricated NCC surface handed to the candidate; the
            default puts the integer peak at ``(0, 0)`` with v-neighbors 0.8
            and 0.4 (quadratic vertex ``+0.25``) and a flat u axis.
        upsample_factor: The refinement upsample factor handed to the candidate.

    Returns:
        The candidate dictionary from :func:`evaluate_candidate`; the tests
        read its ``offset`` entry.
    """
    base = gaussian_patch((64, 64), 3.0)
    shifted = fourier_shift(base, true_shift_vu[0], true_shift_vu[1])
    mask = np.ones_like(base, dtype=bool)
    if corr is None:
        corr = np.zeros_like(base)
        corr[0, 0] = 1.0
        corr[1, 0] = 0.8
        corr[-1, 0] = 0.4
    return evaluate_candidate(
        image_pad=shifted,
        model_pad=base,
        mask_pad=mask,
        corr=corr,
        rc=(0, 0),
        upsample_factor=upsample_factor,
        model_shape=(64, 64),
        image_shape=(64, 64),
        logger=IMAGE_LOGGER,
    )


def test_evaluate_candidate_interior_peak_refines_on_cross_power() -> None:
    """A sub-half-pixel true shift is refined by the upsampled DFT as before."""
    result = _evaluate_candidate_for_shift((0.3, -0.2))
    assert result['offset'][0] == pytest.approx(0.3, abs=0.03)
    assert result['offset'][1] == pytest.approx(-0.2, abs=0.03)


def test_evaluate_candidate_saturated_axis_falls_back_to_ncc_quadratic() -> None:
    """A cross-power peak beyond the half-pixel window uses the NCC vertex.

    The fabricated NCC surface has its quadratic vertex at ``+0.25`` along v,
    so the saturated v axis reports that instead of pinning at exactly
    ``+0.5``; the u axis stays on the (interior) upsampled-DFT estimate.
    """
    result = _evaluate_candidate_for_shift((1.2, 0.0))
    assert result['offset'][0] == pytest.approx(0.25, abs=1e-6)
    assert result['offset'][1] == pytest.approx(0.0, abs=0.03)


def test_evaluate_candidate_saturated_axis_never_reports_half_pixel_pin() -> None:
    """With a flat NCC neighborhood the saturated axis keeps the integer peak."""
    corr = np.zeros((64, 64))
    corr[0, 0] = 1.0
    result = _evaluate_candidate_for_shift((1.2, 0.0), corr=corr)
    assert result['offset'][0] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_candidate_lower_window_edge_falls_back_to_ncc_quadratic() -> None:
    """A cross-power peak below the window's lower edge also uses the NCC vertex.

    The true v shift of -1.2 px puts the upsampled-DFT maximum on the window's first
    sample.  The fabricated NCC surface has its quadratic vertex at ``-0.25`` along v.
    """
    corr = np.zeros((64, 64))
    corr[0, 0] = 1.0
    corr[-1, 0] = 0.8
    corr[1, 0] = 0.4
    result = _evaluate_candidate_for_shift((-1.2, 0.0), corr=corr)
    assert result['offset'][0] == pytest.approx(-0.25, abs=1e-6)


def test_evaluate_candidate_saturated_u_axis_falls_back_to_ncc_quadratic() -> None:
    """A saturated u axis uses the NCC vertex along u; the interior v axis does not.

    The fabricated NCC surface has its quadratic vertex at ``+0.25`` along u and a
    flat v axis, and the true shift saturates only u.
    """
    corr = np.zeros((64, 64))
    corr[0, 0] = 1.0
    corr[0, 1] = 0.8
    corr[0, -1] = 0.4
    result = _evaluate_candidate_for_shift((0.0, 1.2), corr=corr)
    assert result['offset'][1] == pytest.approx(0.25, abs=1e-6)
    assert result['offset'][0] == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize('upsample_factor', [63, 1, 0], ids=['odd', 'one', 'zero'])
def test_evaluate_candidate_rejects_a_factor_that_is_not_even(upsample_factor: int) -> None:
    """An odd or sub-2 upsample factor would make the refinement window asymmetric."""
    with pytest.raises(ValueError, match='upsample_factor must be an even integer >= 2'):
        _evaluate_candidate_for_shift((0.3, -0.2), upsample_factor=upsample_factor)
