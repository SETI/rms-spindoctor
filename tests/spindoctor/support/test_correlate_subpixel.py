"""Tests for the NCC quadratic sub-pixel refinement and how the correlator applies it."""

from typing import Any

import numpy as np
import pytest

import spindoctor.support.correlate as correlate_module
from spindoctor.support.correlate import (
    evaluate_candidate,
    navigate_with_pyramid_kpeaks,
    refinement_cross_spectrum,
)
from spindoctor.support.correlate_subpixel import (
    NCC_QUADRATIC_MAX_SHIFT_PX,
    ncc_quadratic_vertex,
)
from spindoctor.support.image import pad_top_left
from spindoctor.support.types import NDArrayBoolType, NDArrayComplexType, NDArrayFloatType

# =========================================================================
# Helpers
# =========================================================================

_TILE = 32
"""Side of the image and the model in the single-candidate scenes, in pixels."""


def _quadratic_surface(
    shape: tuple[int, int],
    center_rc: tuple[int, int],
    vertex_vu: tuple[float, float],
    *,
    curvature_vu: tuple[float, float] = (0.1, 0.2),
    cross: float = 0.05,
) -> NDArrayFloatType:
    """Return an exact downward quadratic whose maximum lies ``vertex_vu`` from a cell."""
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    dv = rows - center_rc[0] - vertex_vu[0]
    du = cols - center_rc[1] - vertex_vu[1]
    return np.asarray(
        1.0 - (curvature_vu[0] * dv**2 + curvature_vu[1] * du**2 + cross * dv * du), np.float64
    )


def _gaussian(shape: tuple[int, int], center_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return a unit Gaussian blob of sigma 2 px centered at ``center_vu``."""
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    return np.asarray(
        np.exp(-((rows - center_vu[0]) ** 2 + (cols - center_vu[1]) ** 2) / 8.0), np.float64
    )


def _signed_lag_surface(vertex_vu: tuple[float, float]) -> NDArrayFloatType:
    """Return an exact quadratic NCC surface over signed lags, maximal at ``vertex_vu``.

    The surface has the correlator's circular layout at the period image + model: index
    ``j`` holds lag ``j`` for ``j < _TILE`` and lag ``j - 2 * _TILE`` otherwise.
    """
    period = 2 * _TILE
    lags = np.where(np.arange(period) < _TILE, np.arange(period), np.arange(period) - period)
    dv = lags[:, None] - vertex_vu[0]
    du = lags[None, :] - vertex_vu[1]
    return np.asarray(1.0 - 0.1 * dv**2 - 0.2 * du**2, np.float64)


def _single_candidate_inputs(true_shift_vu: tuple[float, float]) -> dict[str, Any]:
    """Return evaluate_candidate inputs for a blob the image holds ``true_shift_vu`` away.

    The NCC surface is an exact quadratic maximal at the true shift, and the candidate is
    the zero lag, so the true peak lies off the candidate by ``true_shift_vu``.
    """
    center = (_TILE / 2.0, _TILE / 2.0)
    image = _gaussian((_TILE, _TILE), (center[0] + true_shift_vu[0], center[1] + true_shift_vu[1]))
    model = _gaussian((_TILE, _TILE), center)
    mask: NDArrayBoolType = np.ones((_TILE, _TILE), dtype=bool)
    period = 2 * _TILE
    image_pad = pad_top_left(image, period, period)
    model_pad = pad_top_left(model, period, period)
    mask_pad = pad_top_left(mask, period, period)
    spectrum: NDArrayComplexType = refinement_cross_spectrum(image_pad, model_pad, mask_pad)
    return {
        'image_pad': image_pad,
        'model_pad': model_pad,
        'corr': _signed_lag_surface(true_shift_vu),
        'refine_spectrum': spectrum,
        'rc': (0, 0),
        'upsample_factor': 128,
        'model_shape': (_TILE, _TILE),
        'image_shape': (_TILE, _TILE),
        'logger': None,
    }


def _star_scene() -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayBoolType]:
    """Return an image, model, and mask of one blob displaced by (1.3, -0.6) px."""
    shape = (64, 64)
    image = _gaussian(shape, (33.3, 31.4))
    model = _gaussian(shape, (32.0, 32.0))
    mask: NDArrayBoolType = model > 1e-3
    return image, model, mask


def _record_single_scale_refinements(
    monkeypatch: pytest.MonkeyPatch, **pyramid_kwargs: Any
) -> list[str]:
    """Run the pyramid on the star scene and return each single-scale pass's refinement.

    The list is in call order; a pass that names no refinement records the default.
    """
    calls: list[str] = []
    real_single = correlate_module.navigate_single_scale_kpeaks

    def _spy(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs.get('refinement', 'cross_power'))
        return real_single(**kwargs)

    monkeypatch.setattr(correlate_module, 'navigate_single_scale_kpeaks', _spy)
    image, model, mask = _star_scene()
    navigate_with_pyramid_kpeaks(image, model, mask, pyramid_levels=3, **pyramid_kwargs)
    return calls


# =========================================================================
# The quadratic fit
# =========================================================================


def test_vertex_of_an_exact_quadratic_is_recovered() -> None:
    """The fit returns the maximum of a quadratic surface, 0.7 px off the cell in v."""
    corr = _quadratic_surface((7, 7), (3, 3), (0.7, -0.3))
    assert ncc_quadratic_vertex(corr, (3, 3)) == pytest.approx((0.7, -0.3), abs=1e-12)


def test_vertex_reads_wrapped_neighbors_at_the_surface_edge() -> None:
    """A cell in the first row and column reads its neighbors from the far edges."""
    corr = np.roll(_quadratic_surface((7, 7), (3, 3), (0.4, 0.2)), (-3, -3), axis=(0, 1))
    assert ncc_quadratic_vertex(corr, (0, 0)) == pytest.approx((0.4, 0.2), abs=1e-12)


def test_vertex_beyond_the_cap_is_limited_on_that_axis() -> None:
    """A fitted maximum 1.6 px away in v is moved back to the cap in v only."""
    corr = _quadratic_surface((7, 7), (3, 3), (1.6, 0.2), cross=0.0)
    assert ncc_quadratic_vertex(corr, (3, 3)) == pytest.approx(
        (NCC_QUADRATIC_MAX_SHIFT_PX, 0.2), abs=1e-12
    )


def test_vertex_of_a_non_quadratic_block_is_the_least_squares_fit() -> None:
    """On nine values no quadratic fits exactly, the vertex is the least-squares surface's."""
    block = np.array([[0.20, 0.45, 0.30], [0.50, 0.90, 0.62], [0.35, 0.60, 0.41]], dtype=np.float64)
    v, u = np.mgrid[-1:2, -1:2].astype(np.float64)
    design = np.stack(
        [np.ones(9), v.ravel(), u.ravel(), v.ravel() ** 2, u.ravel() ** 2, (v * u).ravel()],
        axis=1,
    )
    _, cv, cu, cvv, cuu, cvu = np.linalg.lstsq(design, block.ravel(), rcond=None)[0]
    hessian = np.array([[2.0 * cvv, cvu], [cvu, 2.0 * cuu]])
    expected = np.linalg.solve(hessian, [-cv, -cu])
    assert ncc_quadratic_vertex(block, (1, 1)) == pytest.approx(tuple(expected), abs=1e-12)


def test_a_non_finite_neighbor_gives_no_vertex() -> None:
    """An invalid lag (``-inf``) among the nine values leaves the fit unusable."""
    corr = _quadratic_surface((7, 7), (3, 3), (0.2, 0.1))
    corr[2, 4] = -np.inf
    assert ncc_quadratic_vertex(corr, (3, 3)) is None


def test_a_minimum_gives_no_vertex() -> None:
    """A surface curved upward along both axes has no maximum."""
    corr = -_quadratic_surface((7, 7), (3, 3), (0.2, 0.1))
    assert ncc_quadratic_vertex(corr, (3, 3)) is None


def test_a_saddle_curved_down_in_v_gives_no_vertex() -> None:
    """A surface curved downward in v but upward in u has no maximum."""
    corr = _quadratic_surface((7, 7), (3, 3), (0.2, 0.1), curvature_vu=(0.1, -0.2), cross=0.0)
    assert ncc_quadratic_vertex(corr, (3, 3)) is None


# =========================================================================
# One candidate
# =========================================================================


def test_cross_power_refinement_stops_half_a_pixel_from_the_candidate() -> None:
    """With the true peak 0.7 px off the candidate, the cross-power answer stops at 0.5."""
    result = evaluate_candidate(**_single_candidate_inputs((0.7, 0.0)))
    assert result['offset'] == pytest.approx((0.5, 0.0), abs=1e-12)


def test_ncc_quadratic_refinement_reaches_the_true_peak() -> None:
    """With the true peak 0.7 px off the candidate, the NCC fit reaches it."""
    inputs = _single_candidate_inputs((0.7, 0.0))
    inputs['refine_spectrum'] = None
    result = evaluate_candidate(**inputs, refinement='ncc_quadratic')
    assert result['offset'] == pytest.approx((0.7, 0.0), abs=1e-12)


def test_an_unusable_fit_keeps_the_whole_pixel_candidate() -> None:
    """When a neighbor of the candidate is an invalid lag, the offset is the candidate."""
    inputs = _single_candidate_inputs((0.7, 0.0))
    inputs['corr'][1, 0] = -np.inf
    result = evaluate_candidate(**inputs, refinement='ncc_quadratic')
    assert result['offset'] == (0.0, 0.0)


def test_prior_penalty_is_measured_from_the_refined_offset() -> None:
    """A prior at the fitted maximum costs no quality."""
    inputs = _single_candidate_inputs((0.7, 0.0))
    plain = evaluate_candidate(**inputs, refinement='ncc_quadratic')
    with_prior = evaluate_candidate(
        **inputs, refinement='ncc_quadratic', prior_shift=(0.7, 0.0), prior_weight=1.0
    )
    assert with_prior['quality'] == pytest.approx(plain['quality'], abs=1e-12)


def test_residual_is_measured_at_the_refined_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model is shifted by the fitted offset, not the whole pixel, before the residual."""
    shifts: list[tuple[float, float]] = []
    real_shift = correlate_module.fourier_shift

    def _spy(img: NDArrayFloatType, dy: float, dx: float) -> NDArrayFloatType:
        shifts.append((dy, dx))
        return real_shift(img, dy, dx)

    monkeypatch.setattr(correlate_module, 'fourier_shift', _spy)
    evaluate_candidate(**_single_candidate_inputs((0.7, 0.0)), refinement='ncc_quadratic')
    assert shifts == [pytest.approx((0.7, 0.0), abs=1e-12)]


def test_cross_power_refinement_without_a_spectrum_is_refused() -> None:
    """The cross-power refinement cannot run without its spectrum."""
    inputs = _single_candidate_inputs((0.7, 0.0))
    inputs['refine_spectrum'] = None
    with pytest.raises(ValueError, match='cross_power refinement needs a refine_spectrum'):
        evaluate_candidate(**inputs)


def test_an_unknown_refinement_is_refused() -> None:
    """A refinement name outside the accepted set is refused."""
    unknown: Any = 'spline'
    with pytest.raises(ValueError, match=r"refinement must be one of .*not 'spline'"):
        evaluate_candidate(**_single_candidate_inputs((0.7, 0.0)), refinement=unknown)


# =========================================================================
# The pyramid
# =========================================================================


def test_final_pass_uses_the_requested_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """The final full-resolution pass refines on the NCC when asked to."""
    calls = _record_single_scale_refinements(monkeypatch, final_refinement='ncc_quadratic')
    assert calls[-1] == 'ncc_quadratic'


def test_coarser_levels_keep_the_cross_power_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every pyramid level before the final pass keeps the cross-power refinement."""
    calls = _record_single_scale_refinements(monkeypatch, final_refinement='ncc_quadratic')
    assert calls[:-1] == ['cross_power', 'cross_power', 'cross_power']


def test_default_is_the_cross_power_refinement_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the option every pass, the final one included, uses the cross-power refinement."""
    calls = _record_single_scale_refinements(monkeypatch)
    assert calls == ['cross_power'] * 4


def test_auto_mode_gives_both_final_passes_the_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In auto mode the raw and the gradient pyramids each refine their final pass on the NCC."""
    calls = _record_single_scale_refinements(
        monkeypatch, final_refinement='ncc_quadratic', use_gradient='auto'
    )
    assert calls == ['cross_power'] * 3 + ['ncc_quadratic'] + ['cross_power'] * 3 + [
        'ncc_quadratic'
    ]


def test_an_unknown_final_refinement_is_refused_before_any_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad final refinement name is refused before the first pyramid level runs."""
    unknown: Any = 'spline'
    with pytest.raises(ValueError, match=r"final_refinement must be one of .*not 'spline'"):
        _record_single_scale_refinements(monkeypatch, final_refinement=unknown)


def test_consistency_does_not_depend_on_the_final_refinement() -> None:
    """The level consistency comes from the coarser levels alone."""
    image, model, mask = _star_scene()
    cross_power = navigate_with_pyramid_kpeaks(image, model, mask, pyramid_levels=3)
    ncc_quadratic = navigate_with_pyramid_kpeaks(
        image, model, mask, pyramid_levels=3, final_refinement='ncc_quadratic'
    )
    assert ncc_quadratic['consistency'] == cross_power['consistency']
