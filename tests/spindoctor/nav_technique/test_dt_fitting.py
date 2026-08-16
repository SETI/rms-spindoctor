"""Unit tests for the shared distance-transform fitting helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from spindoctor.nav_orchestrator.image_derivatives import (
    DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
    compute_image_gradient_vu,
)
from spindoctor.nav_technique.dt_fitting import (
    DEFAULT_TUKEY_C,
    LMRefineResult,
    RidgeRefineResult,
    coarse_ncc_search,
    coarse_ncc_search_scored,
    coarse_polarity_search_scored,
    find_secondary_dt_minimum,
    gradient_ridge_refine,
    information_matrix_to_covariance,
    lm_subpixel_refine,
    polarity_filter,
    tukey_biweight_weights,
)


def _build_circle_polyline(
    center_vu: tuple[float, float], radius_px: float, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N, 2)`` vertices and ``(N, 2)`` outward normals on a circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    cv, cu = center_vu
    vs = cv + radius_px * np.sin(angles)
    us = cu + radius_px * np.cos(angles)
    nv = np.sin(angles)
    nu = np.cos(angles)
    vertices = np.stack([vs, us], axis=-1).astype(np.float64)
    normals = np.stack([nv, nu], axis=-1).astype(np.float64)
    return vertices, normals


def _render_circle_mask(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    radius_px: float,
    thickness_px: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask of an annulus around the given circle."""
    vs, us = np.meshgrid(
        np.arange(shape_vu[0]),
        np.arange(shape_vu[1]),
        indexing='ij',
    )
    cv, cu = center_vu
    rr = np.hypot(vs - cv, us - cu)
    out: np.ndarray = np.abs(rr - radius_px) <= thickness_px
    return out


def _render_image_with_circle(
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
    radius_px: float,
    *,
    inside_dn: float = 100.0,
    outside_dn: float = 0.0,
) -> np.ndarray:
    """Return a step-edge disc image (inside_dn inside, outside_dn outside)."""
    vs, us = np.meshgrid(
        np.arange(shape_vu[0]),
        np.arange(shape_vu[1]),
        indexing='ij',
    )
    cv, cu = center_vu
    rr = np.hypot(vs - cv, us - cu)
    image = np.where(rr <= radius_px, inside_dn, outside_dn)
    return image.astype(np.float64)


# ---------------------------------------------------------------------------
# tukey_biweight_weights
# ---------------------------------------------------------------------------


def test_tukey_biweight_weights_returns_one_at_zero_residual() -> None:
    weights = tukey_biweight_weights(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(weights, [1.0, 1.0, 1.0])


def test_tukey_biweight_weights_is_zero_outside_cutoff() -> None:
    c = DEFAULT_TUKEY_C
    weights = tukey_biweight_weights(np.array([c + 0.01, -c - 0.5, 100.0]))
    assert np.array_equal(weights, [0.0, 0.0, 0.0])


def test_tukey_biweight_weights_matches_holland_welsch_at_half_cutoff() -> None:
    c = DEFAULT_TUKEY_C
    r = c / 2.0
    weights = tukey_biweight_weights(np.array([r]))
    expected = (1.0 - 0.25) ** 2
    assert weights[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ('invalid_c', 'message'),
    [
        (-1.0, 'c must be a positive finite'),
        (0.0, 'c must be a positive finite'),
        (float('inf'), 'c must be a positive finite'),
        (float('nan'), 'c must be a positive finite'),
    ],
)
def test_tukey_biweight_weights_rejects_invalid_c(invalid_c: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tukey_biweight_weights(np.array([1.0]), c=invalid_c)


@pytest.mark.parametrize('shape', [(2, 2), (3, 3, 3), (1, 4)])
def test_tukey_biweight_weights_rejects_non_1d_input(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match='residuals must be 1-D'):
        tukey_biweight_weights(np.zeros(shape))


# ---------------------------------------------------------------------------
# information_matrix_to_covariance
# ---------------------------------------------------------------------------


def test_information_matrix_to_covariance_recovers_identity() -> None:
    jacobian = np.eye(2, dtype=np.float64)
    weights = np.array([1.0, 1.0])
    cov = information_matrix_to_covariance(jacobian, weights)
    assert np.allclose(cov, np.eye(2), atol=1e-12)


def test_information_matrix_to_covariance_handles_rank_one_input() -> None:
    # All Jacobian rows along axis 0: only "v" axis is observable.
    jacobian = np.zeros((10, 2), dtype=np.float64)
    jacobian[:, 0] = 1.0
    weights = np.ones(10)
    cov = information_matrix_to_covariance(jacobian, weights)
    eigvals = np.linalg.eigvalsh(cov)
    null_eigval = float(eigvals.min())
    observed_eigval = float(eigvals.max())
    assert abs(null_eigval) < 1.0e-12
    assert observed_eigval == pytest.approx(0.1, rel=1e-9)


@pytest.mark.parametrize(
    ('jacobian', 'weights', 'message'),
    [
        # Negative weight: rejected by the non-negativity guard.
        (np.eye(2), np.array([1.0, -0.5]), 'weights must be non-negative'),
        # Wrong-rank weight vector: caught by the 1-D shape guard.
        (np.eye(3), np.ones(2), 'must be a 1-D vector'),
        # 1-D Jacobian: caught by the 2-D shape guard.
        (np.zeros(3), np.ones(3), 'jacobian must be 2-D'),
    ],
)
def test_information_matrix_to_covariance_rejects_invalid_inputs(
    jacobian: np.ndarray, weights: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        information_matrix_to_covariance(jacobian, weights)


# ---------------------------------------------------------------------------
# coarse_ncc_search
# ---------------------------------------------------------------------------


def test_coarse_ncc_search_recovers_planted_integer_offset() -> None:
    shape = (64, 64)
    polyline_mask = _render_circle_mask(shape, (32.0, 32.0), 12.0)
    edge_mask = np.roll(polyline_mask, shift=(3, -2), axis=(0, 1))
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (10, 10))
    assert (dv, du) == (3, -2)


def test_coarse_ncc_search_returns_zero_for_aligned_inputs() -> None:
    shape = (32, 32)
    polyline_mask = _render_circle_mask(shape, (16.0, 16.0), 8.0)
    dv, du = coarse_ncc_search(polyline_mask, polyline_mask, (4, 4))
    assert (dv, du) == (0, 0)


def test_coarse_ncc_search_returns_zero_with_empty_polyline() -> None:
    edge_mask = np.zeros((16, 16), dtype=bool)
    edge_mask[8, 8] = True
    polyline_mask = np.zeros_like(edge_mask)
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (3, 3))
    assert (dv, du) == (0, 0)


def test_coarse_ncc_search_uses_ncc_not_raw_count_under_clipping() -> None:
    """The seed is the per-vertex (NCC) argmax, not the raw-overlap argmax.

    When shifts clip a different number of polyline vertices off the image,
    the in-bounds vertex count varies, so the raw-overlap-count argmax and the
    per-vertex (NCC) argmax differ.  ``coarse_ncc_search`` must return the
    latter (CODE-NAV-007).
    """
    h, w = 20, 20
    poly_rows = np.arange(1, 11)  # 10-vertex vertical segment near the top edge
    polyline_mask = np.zeros((h, w), dtype=bool)
    polyline_mask[poly_rows, 10] = True
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[0:9, 10] = True  # edges only on rows 0..8 of that column
    window = (10, 0)

    def brute(*, normalize: bool) -> tuple[int, int]:
        best = (0, 0)
        best_score = -1.0
        best_key = (math.inf, math.inf, math.inf, math.inf)
        for dv in range(-window[0], window[0] + 1):
            sv = poly_rows + dv
            valid = (sv >= 0) & (sv < h)
            if not valid.any():
                continue
            n_in = int(valid.sum())
            overlap = int(edge_mask[sv[valid], 10].sum())
            score = overlap / n_in if normalize else float(overlap)
            key = (abs(dv), abs(dv), float(dv), 0.0)
            if score > best_score or (score == best_score and key < best_key):
                best_score, best_key, best = score, key, (dv, 0)
        return best

    raw_argmax = brute(normalize=False)
    ncc_argmax = brute(normalize=True)
    # The fixture genuinely distinguishes the two definitions.
    assert raw_argmax == (-1, 0)
    assert ncc_argmax == (-2, 0)
    # The function returns the NCC argmax (it would return raw_argmax if it
    # still summed the raw overlap count).
    assert coarse_ncc_search(edge_mask, polyline_mask, window) == ncc_argmax


def test_coarse_ncc_search_low_support_shift_loses_to_dense_partial_match() -> None:
    """A near-fully-clipped perfect score cannot beat a dense partial match.

    The per-vertex match fraction over-rewards shifts that clip nearly the
    whole polyline off-frame: a handful of surviving vertices that happen to
    land on edge pixels score a perfect 1.0.  The minimum-support guard
    makes such shifts ineligible, so a dense partial match (18 of 20
    vertices on edges, fraction 0.9) at the planted offset must win.
    """
    h, w = 40, 40
    poly_rows = np.arange(5, 25)  # 20-vertex vertical segment
    polyline_mask = np.zeros((h, w), dtype=bool)
    polyline_mask[poly_rows, 20] = True
    edge_mask = np.zeros((h, w), dtype=bool)
    # Dense partial match at the planted shift dv=+3: edges on rows 8..27
    # except rows 12 and 18, so 18 of the 20 shifted vertices land on edges.
    edge_mask[8:28, 20] = True
    edge_mask[12, 20] = False
    edge_mask[18, 20] = False
    # Adversarial far edges near the bottom border: a dv=+33 shift clips all
    # but two vertices off-frame and both survivors land on these edges,
    # scoring a perfect (but 10 %-supported) match fraction of 1.0.
    edge_mask[38, 20] = True
    edge_mask[39, 20] = True
    window = (35, 0)

    def unguarded_brute() -> tuple[int, int]:
        best = (0, 0)
        best_score = -1.0
        best_key = (math.inf, math.inf, math.inf, math.inf)
        for dv in range(-window[0], window[0] + 1):
            sv = poly_rows + dv
            valid = (sv >= 0) & (sv < h)
            if not valid.any():
                continue
            score = float(edge_mask[sv[valid], 20].sum()) / float(valid.sum())
            key = (abs(dv), abs(dv), float(dv), 0.0)
            if score > best_score or (score == best_score and key < best_key):
                best_score, best_key, best = score, key, (dv, 0)
        return best

    # Without the support guard the clipped perfect score wins the argmax,
    # so the fixture genuinely exercises the guard.
    assert unguarded_brute() == (33, 0)
    assert coarse_ncc_search(edge_mask, polyline_mask, window) == (3, 0)


@pytest.mark.parametrize(
    ('edge_mask', 'polyline_mask', 'window', 'message'),
    [
        # Shape mismatch between the two masks.
        (
            np.zeros((4, 4), bool),
            np.zeros((5, 5), bool),
            (1, 1),
            'shape mismatch',
        ),
        # 1-D edge mask: caught by the 2-D shape guard.
        (np.zeros(4, bool), np.zeros((4, 4), bool), (1, 1), 'must be 2-D'),
        # Negative window margin: caught by the non-negative guard.
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (-1, 1),
            'must be non-negative',
        ),
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1, -1),
            'must be non-negative',
        ),
        # Wrong-length window: caught by the length-2 guard.
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1, 2, 3),
            'length-2 sequence of ints',
        ),
        (
            np.zeros((4, 4), bool),
            np.zeros((4, 4), bool),
            (1,),
            'length-2 sequence of ints',
        ),
    ],
)
def test_coarse_ncc_search_rejects_invalid_inputs(
    edge_mask: np.ndarray,
    polyline_mask: np.ndarray,
    window: tuple[int, ...],
    message: str,
) -> None:
    """Invalid mask shapes or window tuples are rejected with a named message."""
    with pytest.raises(ValueError, match=message):
        coarse_ncc_search(edge_mask, polyline_mask, window)  # type: ignore[arg-type]


def test_coarse_ncc_search_rejects_float_window_entry() -> None:
    """A float window entry is rejected with TypeError instead of being truncated."""
    edge_mask = np.zeros((4, 4), bool)
    polyline_mask = np.zeros((4, 4), bool)
    with pytest.raises(TypeError, match='search_window_vu\\[0\\] must be int'):
        coarse_ncc_search(edge_mask, polyline_mask, (1.5, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize('fraction', [-0.1, 1.5])
def test_coarse_ncc_search_rejects_out_of_range_min_support_fraction(fraction: float) -> None:
    """A min_support_fraction outside [0, 1] is rejected with a named message."""
    edge_mask = np.zeros((4, 4), bool)
    polyline_mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match=r'min_support_fraction must be in \[0, 1\]'):
        coarse_ncc_search(edge_mask, polyline_mask, (1, 1), min_support_fraction=fraction)


def test_coarse_ncc_search_rejects_non_sequence_window() -> None:
    """A non-tuple/list window is rejected by the length-2 sequence guard."""
    edge_mask = np.zeros((4, 4), bool)
    polyline_mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match='length-2 sequence of ints'):
        coarse_ncc_search(edge_mask, polyline_mask, np.array([1, 1]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# coarse_polarity_search_scored
# ---------------------------------------------------------------------------


def _uniform_gradient_field(
    shape_vu: tuple[int, int], edge_mask: np.ndarray, direction_vu: tuple[float, float]
) -> np.ndarray:
    """Return an ``(H, W, 2)`` gradient field pointing ``direction_vu`` on edges."""
    grad = np.zeros((*shape_vu, 2), np.float64)
    grad[edge_mask, 0] = direction_vu[0]
    grad[edge_mask, 1] = direction_vu[1]
    return grad


def test_coarse_polarity_search_recovers_planted_offset_on_a_disc() -> None:
    """A polarity-aligned disc limb recovers the planted integer offset."""
    shape = (64, 64)
    image = _render_image_with_circle(shape, (35.0, 30.0), 12.0)
    gradient = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    edge_mask = _render_circle_mask(shape, (35.0, 30.0), 12.0)
    vertices, outward_normals = _build_circle_polyline((32.0, 32.0), 12.0, 180)
    # The bright-disc gradient points inward, so the polarity normal is the
    # inward (negated outward) normal -- the technique's convention.
    polarity_normals = -outward_normals
    result = coarse_polarity_search_scored(
        edge_mask, gradient, vertices, polarity_normals, (10, 10)
    )
    assert result.offset_vu == (3, -2)


def test_coarse_polarity_search_rejects_wrong_polarity_dense_basin() -> None:
    """A denser but wrong-polarity edge population loses to the true arc.

    This is the #179 failure mode in miniature: a competing edge region has
    more model vertices land on edges (so the polarity-blind mask-overlap
    search picks it), but its image gradient runs opposite the model normal.
    The polarity-weighted search must instead pick the true, sparser arc
    whose gradient direction agrees with the model.
    """
    h, w = 40, 40
    poly_rows = np.arange(10, 30)  # 20-vertex vertical segment at column 20
    vertices = np.stack([poly_rows.astype(np.float64), np.full(20, 20.0)], axis=-1)
    # Model normals all point toward +u; a matching edge has its gradient in
    # the same direction.
    normals = np.tile(np.array([0.0, 1.0]), (20, 1))
    edge_mask = np.zeros((h, w), dtype=bool)
    # True arc: 12 of 20 vertices land on edges at column 23 (shift du=+3),
    # gradient aligned with the model normal (+u).
    edge_mask[10:22, 23] = True
    # Distractor: all 20 vertices land on edges at column 15 (shift du=-5),
    # a denser overlap, but gradient anti-parallel to the model normal.
    edge_mask[10:30, 15] = True
    gradient = np.zeros((h, w, 2), np.float64)
    gradient[10:22, 23, 1] = 1.0  # aligned (+u)
    gradient[10:30, 15, 1] = -1.0  # anti-aligned (-u)
    window = (0, 8)
    # The polarity-blind search is fooled by the denser distractor.
    polyline_mask = np.zeros((h, w), dtype=bool)
    polyline_mask[poly_rows, 20] = True
    assert coarse_ncc_search(edge_mask, polyline_mask, window) == (0, -5)
    # The polarity-weighted search recovers the true arc.
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, window)
    assert result.offset_vu == (0, 3)


def test_coarse_polarity_search_reduces_to_overlap_when_fully_aligned() -> None:
    """With every edge gradient aligned to its normal the score is the overlap fraction."""
    h, w = 40, 40
    poly_rows = np.arange(10, 30)
    vertices = np.stack([poly_rows.astype(np.float64), np.full(20, 20.0)], axis=-1)
    normals = np.tile(np.array([0.0, 1.0]), (20, 1))
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[10:25, 23] = True  # 15 of 20 vertices on edges at du=+3
    gradient = _uniform_gradient_field((h, w), edge_mask, (0.0, 1.0))
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 3)
    # 15 aligned matches over 20 vertices == 0.75, the same fraction the
    # polarity-blind overlap search would report.
    assert result.score == pytest.approx(0.75)


def test_coarse_polarity_search_weights_by_cosine_not_a_hard_threshold() -> None:
    """A partially-misaligned edge contributes cos(theta), not a full unit.

    Pins the continuous ``max(0, cos theta)`` weighting: a hard-threshold
    ``cos > 0 -> 1`` implementation would score both vertices as full matches
    and report 1.0, so this fixture distinguishes the two designs.
    """
    h, w = 20, 20
    # Two vertices, both on edges, both with a +u model normal.
    vertices = np.array([[5.0, 10.0], [6.0, 10.0]])
    normals = np.array([[0.0, 1.0], [0.0, 1.0]])
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[5, 10] = True
    edge_mask[6, 10] = True
    gradient = np.zeros((h, w, 2), np.float64)
    gradient[5, 10] = (0.0, 1.0)  # aligned: cos = 1
    gradient[6, 10] = (np.sqrt(3.0) / 2.0, 0.5)  # 60 deg off the normal: cos = 0.5
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (0, 0))
    assert result.offset_vu == (0, 0)
    # (1.0 + 0.5) / 2 vertices == 0.75, not the 1.0 a hard threshold would give.
    assert result.score == pytest.approx(0.75)


def test_coarse_polarity_search_low_support_shift_is_ineligible() -> None:
    """A near-fully-clipped perfect-polarity shift loses to a dense match.

    The minimum-support guard applies in the polarity variant too: a shift
    that clips all but two vertices off-frame cannot win on those two even
    when their polarity is perfect.
    """
    h, w = 40, 40
    poly_rows = np.arange(5, 25)  # 20-vertex vertical segment at column 20
    vertices = np.stack([poly_rows.astype(np.float64), np.full(20, 20.0)], axis=-1)
    normals = np.tile(np.array([0.0, 1.0]), (20, 1))
    edge_mask = np.zeros((h, w), dtype=bool)
    # Dense partial match at dv=+3: 18 of 20 shifted vertices on edges.
    edge_mask[8:28, 20] = True
    edge_mask[12, 20] = False
    edge_mask[18, 20] = False
    # Adversarial far edges: dv=+33 clips all but two vertices off-frame.
    edge_mask[38, 20] = True
    edge_mask[39, 20] = True
    gradient = _uniform_gradient_field((h, w), edge_mask, (0.0, 1.0))  # all aligned
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (35, 0))
    assert result.offset_vu == (3, 0)


def test_coarse_polarity_search_zero_normal_never_matches() -> None:
    """A degenerate zero-length normal contributes no weight to any shift."""
    h, w = 20, 20
    vertices = np.array([[10.0, 10.0]])
    normals = np.array([[0.0, 0.0]])  # degenerate
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[10, 10] = True
    gradient = np.zeros((h, w, 2), np.float64)
    gradient[10, 10, 1] = 1.0
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (2, 2))
    assert result.score == 0.0


def test_coarse_polarity_search_empty_polyline_returns_origin() -> None:
    """An empty polyline yields the origin offset and a zero score."""
    edge_mask = np.zeros((16, 16), dtype=bool)
    gradient = np.zeros((16, 16, 2), np.float64)
    vertices = np.empty((0, 2), np.float64)
    normals = np.empty((0, 2), np.float64)
    result = coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (3, 3))
    assert result.offset_vu == (0, 0)
    assert result.score == 0.0


@pytest.mark.parametrize(
    ('gradient', 'vertices', 'normals', 'window', 'message'),
    [
        # Gradient field missing its (v, u) last axis.
        (
            np.zeros((8, 8), np.float64),
            np.zeros((1, 2), np.float64),
            np.zeros((1, 2), np.float64),
            (1, 1),
            r'image_gradient_vu must have shape \(H, W, 2\)',
        ),
        # Gradient field shape disagrees with the edge mask.
        (
            np.zeros((6, 6, 2), np.float64),
            np.zeros((1, 2), np.float64),
            np.zeros((1, 2), np.float64),
            (1, 1),
            'shape mismatch',
        ),
        # Vertices not (N, 2).
        (
            np.zeros((8, 8, 2), np.float64),
            np.zeros((1, 3), np.float64),
            np.zeros((1, 2), np.float64),
            (1, 1),
            r'vertices_vu must have shape \(N, 2\)',
        ),
        # Normals do not match vertices.
        (
            np.zeros((8, 8, 2), np.float64),
            np.zeros((2, 2), np.float64),
            np.zeros((1, 2), np.float64),
            (1, 1),
            'normals_vu must match vertices_vu shape',
        ),
    ],
)
def test_coarse_polarity_search_rejects_invalid_inputs(
    gradient: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    window: tuple[int, ...],
    message: str,
) -> None:
    """Malformed array shapes are rejected with a named message."""
    edge_mask = np.zeros((8, 8), bool)
    with pytest.raises(ValueError, match=message):
        coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, window)  # type: ignore[arg-type]


def test_coarse_polarity_search_rejects_float_window_entry() -> None:
    """A float window entry is rejected with TypeError, matching the overlap search."""
    edge_mask = np.zeros((8, 8), bool)
    gradient = np.zeros((8, 8, 2), np.float64)
    vertices = np.zeros((1, 2), np.float64)
    normals = np.zeros((1, 2), np.float64)
    with pytest.raises(TypeError, match=r'search_window_vu\[0\] must be int'):
        coarse_polarity_search_scored(edge_mask, gradient, vertices, normals, (1.5, 1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# polarity_filter
# ---------------------------------------------------------------------------


def test_polarity_filter_accepts_normals_aligned_with_image_gradient() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    # Bright-disc on dark sky: the image gradient at the limb points INTO the
    # body (low DN to high DN).  ``polarity_filter`` tests strict
    # ``model_dir . image_gradient > 0``, so the model must supply the inward
    # direction (the negation of the geometric outward normal) to be accepted.
    inward_normals = -outward_normals
    keep = polarity_filter(vertices, inward_normals, grad)
    assert keep.sum() == 16


def test_polarity_filter_rejects_normals_opposing_image_gradient() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    keep = polarity_filter(vertices, outward_normals, grad)
    assert keep.sum() == 0


def test_polarity_filter_per_vertex_decision() -> None:
    shape = (32, 32)
    image = _render_image_with_circle(shape, (16.0, 16.0), 8.0)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    inward_normals = -outward_normals
    # Half the vertices kept aligned (inward), half flipped to outward.
    mixed_normals = inward_normals.copy()
    mixed_normals[::2] = outward_normals[::2]
    keep = polarity_filter(vertices, mixed_normals, grad)
    assert keep.sum() == 8


def test_polarity_filter_rejects_out_of_bounds_vertices() -> None:
    """CODE-NAV-015: an off-image vertex is rejected even when the clamped
    boundary pixel carries a strong gradient aligned with its normal."""
    grad = np.zeros((8, 8, 2), dtype=np.float64)
    # Strong gradient at the top-edge pixel (0, 4) pointing in +v.
    grad[0, 4] = (10.0, 0.0)
    grad[4, 4] = (10.0, 0.0)
    # Vertex 0 is above the image (v = -5); it clamps to pixel (0, 4) whose
    # gradient aligns with its normal.  Vertex 1 is in-bounds at (4, 4).
    vertices = np.array([[-5.0, 4.0], [4.0, 4.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    keep = polarity_filter(vertices, normals, grad)
    assert not bool(keep[0])
    assert bool(keep[1])


def test_polarity_filter_rejects_misshaped_inputs() -> None:
    grad = np.zeros((4, 4, 2))
    with pytest.raises(ValueError, match='vertices_vu must have shape'):
        polarity_filter(np.zeros(2), np.zeros(2), grad)


def test_polarity_filter_rejects_2d_gradient() -> None:
    with pytest.raises(ValueError, match='image_gradient_vu must have shape'):
        polarity_filter(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# lm_subpixel_refine — translation-only
# ---------------------------------------------------------------------------


def _build_dt_for_circle(shape: tuple[int, int], radius: float) -> np.ndarray:
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    edge_mask = _render_circle_mask(shape, (cv, cu), radius, thickness_px=0.5)
    raw = distance_transform_edt(~edge_mask)
    dt: np.ndarray = np.asarray(raw, dtype=np.float64)
    return dt


def test_lm_subpixel_refine_recovers_subpixel_translation() -> None:
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    inward_normals = -outward_normals
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=inward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
    )
    assert result.offset_vu[0] == pytest.approx(-1.5, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(-2.5, abs=0.05)
    assert result.iterations >= 1
    assert result.inlier_count == 64


def test_lm_subpixel_refine_is_shift_equivariant_for_translated_polyline() -> None:
    """A rigidly translated copy of the polyline converges to the same absolute pose.

    Shift equivariance of the DT fit: against one fixed image DT, fitting
    vertices ``V`` and fitting ``V + delta`` must return offsets that differ
    by ``-delta``, because the translated polyline's cost function is exactly
    the original's evaluated at a translated argument.  The tolerance is the
    measured optimizer floor at this scale (about 0.04 px): the LM stalls at
    slightly different points of the rasterized DT's plateau depending on
    where the seed sits relative to the optimum.  A systematic pull toward
    the seed or toward the pixel grid would exceed it by an order of
    magnitude.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    center = (shape[0] / 2.0 + 0.4, shape[1] / 2.0 - 0.3)
    vertices, outward_normals = _build_circle_polyline(center, radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    delta = (0.37, -0.61)
    shifted_vertices = vertices.copy()
    shifted_vertices[:, 0] += delta[0]
    shifted_vertices[:, 1] += delta[1]
    result_base = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=False,
    )
    result_shifted = lm_subpixel_refine(
        vertices_vu=shifted_vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=False,
    )
    assert result_shifted.offset_vu[0] - result_base.offset_vu[0] == pytest.approx(
        -delta[0], abs=0.06
    )
    assert result_shifted.offset_vu[1] - result_base.offset_vu[1] == pytest.approx(
        -delta[1], abs=0.06
    )


def test_lm_subpixel_refine_rejects_polarity_vertex_with_enormous_sigma() -> None:
    """CODE-NAV-019: a polarity-rejected vertex is excluded regardless of sigma.

    Rejection zeroes the weight via the polarity mask, not by driving the
    penalty residual past the Tukey cutoff.  With an enormous per-vertex
    sigma the old ``penalty / sigma > c`` chain would fail (the penalty
    residual would scale below the cutoff and keep a non-zero weight); the
    mask makes exclusion independent of sigma.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    normals = -outward_normals  # inward normals are accepted on a bright disc
    # Flip vertex 0 to the wrong polarity so the polarity filter rejects it,
    # and give it a sigma far above the old 1e6 / tukey_c ~ 2.1e5 px bound.
    normals[0] = outward_normals[0]
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    sigmas[0] = 1.0e7
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
    )
    # The wrong-polarity, huge-sigma vertex must be excluded (weight 0).
    assert result.inlier_count == 63
    assert result.offset_vu[0] == pytest.approx(-1.5, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(-2.5, abs=0.05)


def test_lm_subpixel_refine_raw_rms_excludes_polarity_rejected_vertices() -> None:
    """Polarity-rejected vertices must not inflate ``raw_rms_px``.

    A polarity-rejected vertex carries the large ``_INFINITY_DT_PENALTY_PX``
    sentinel residual.  If ``raw_rms_px`` pooled that sentinel it would jump
    to ~1e6 / sqrt(N) the moment a single vertex is polarity-rejected,
    degenerating the limb / terminator ``raw_rms_px > floor`` spurious gate
    into "any polarity rejection is spurious".  In a multi-body frame a
    small secondary body contributes wrong-polarity vertices while a
    dominant body's limb fits cleanly, so that pooling wrongly killed the
    whole fit.  ``raw_rms_px`` must average over the polarity-ACCEPTED
    vertices only, so a clean accepted-vertex fit stays small regardless of
    how many vertices are polarity-rejected.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    normals = -outward_normals  # inward normals are accepted on a bright disc
    # Flip a contiguous run of vertices (a stand-in for a small secondary
    # body whose limb faces the wrong way) to the wrong polarity so the
    # polarity filter rejects them and records the sentinel residual.
    rejected = np.arange(0, 12)
    normals[rejected] = outward_normals[rejected]
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
    )
    # The 12 wrong-polarity vertices are excluded from the fit.
    assert result.inlier_count == 52
    # The accepted vertices align cleanly, so the unweighted raw RMS stays
    # sub-pixel; the sentinel-poisoned value would be ~1e6 / sqrt(64) ~ 1e5.
    assert result.raw_rms_px < 1.0


def test_lm_subpixel_refine_raw_rms_retains_tukey_rejected_arc() -> None:
    """``raw_rms_px`` still surfaces a polarity-accepted, Tukey-rejected arc.

    The fix that excludes polarity-rejected vertices from ``raw_rms_px``
    must not weaken the gate's original purpose: a wholly
    mis-aligned but polarity-ACCEPTED arc that the Tukey reweighting drives
    to ~0 weight must still inflate the raw RMS so the spurious gate fires.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, normals = _build_circle_polyline((cv, cu), radius, 100)
    # Displace 20 % of the vertices far from the circle.  They keep the
    # correct (accepted) polarity but sit tens of pixels off the edge, so
    # Tukey rejects them while their real DT residuals remain large.
    bad = np.arange(0, 100, 5)
    vertices[bad, 0] += 25.0
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (cv, cu), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=False,
    )
    # Weighted RMS collapses (Tukey rejected the bad arc) but the raw RMS
    # over the accepted vertices retains the mis-aligned arc.
    assert result.rms_px < 1.0
    assert result.raw_rms_px > 3.0


def test_lm_subpixel_refine_trust_region_caps_offset_displacement() -> None:
    """The trust-region kwarg physically prevents the LM from leaving the seed.

    Plant a circle at (-1.5, -2.5) but seed the LM at (5, 5) — well
    outside the planted basin.  Without a trust region, the LM either
    walks back toward the truth or, on noisy data, walks to an
    unrelated DT minimum.  With a 1.0-px trust region the converged
    offset is constrained to ``hypot(dv-5, du-5) <= 1.0``: the LM can
    refine inside the trust radius but cannot escape it.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    inward_normals = -outward_normals
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    initial = (5.0, 5.0)
    trust_region = 1.0
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=inward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=initial,
        use_polarity=True,
        trust_region_px=trust_region,
    )
    displacement = float(
        np.hypot(result.offset_vu[0] - initial[0], result.offset_vu[1] - initial[1])
    )
    # The displacement is bounded by the trust radius (with a small
    # tolerance for the final commit; the LM may accept a step
    # exactly at the boundary).
    assert displacement <= trust_region + 1.0e-6


def test_lm_subpixel_refine_rejects_outliers_via_tukey() -> None:
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, normals = _build_circle_polyline((cv, cu), radius, 100)
    # Move 10 % of the vertices far away from the actual circle: they should
    # be Tukey-rejected so the recovered offset is dominated by the inliers.
    bad = np.arange(0, 100, 10)
    vertices[bad, 0] += 30.0
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (cv, cu), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=False,
    )
    assert result.offset_vu[0] == pytest.approx(0.0, abs=0.1)
    assert result.offset_vu[1] == pytest.approx(0.0, abs=0.1)
    # Outlier vertices should have zero (or near-zero) Tukey weight.
    assert float(result.weights[bad].max()) < 1.0e-6
    # In-lier vertices should retain weight of 1 / sigma**2 = 4 (with sigma=0.5)
    inlier_idx = np.setdiff1d(np.arange(100), bad)
    assert float(result.weights[inlier_idx].min()) > 1.0


def test_lm_subpixel_refine_degenerate_when_all_vertices_rejected() -> None:
    """A fit with no surviving inliers reports +inf RMS and inf covariance.

    With a zero gradient image every polarity dot product is zero (not
    strictly positive), so the polarity filter rejects every vertex.
    Each rejected vertex gets the infinity penalty, the Tukey biweight
    zeroes its weight, and no evidence remains to constrain the fit.  The
    result must advertise this honestly: ``rms_px`` is ``+inf`` (not the
    misleading ``0.0`` that downstream spurious gates would read as a
    perfect fit), ``degenerate`` is True, and the covariance is all-inf.
    """
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    zero_grad = np.zeros((*shape, 2), dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=zero_grad,
        initial_offset_vu=(0.0, 0.0),
        use_polarity=True,
    )
    assert result.inlier_count == 0
    assert result.degenerate is True
    assert result.rms_px == float('inf')
    assert np.isinf(result.covariance).all()


# ---------------------------------------------------------------------------
# lm_subpixel_refine — translation + rotation
# ---------------------------------------------------------------------------


def test_lm_subpixel_refine_recovers_planted_rotation_and_translation() -> None:
    shape = (200, 200)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    # Use a four-arm cross template: well-constrained for translation in both
    # axes and for in-plane rotation about the cross centre.
    arm_length_px = 60.0
    arm_density = 60
    arm_offsets_along = np.linspace(8.0, arm_length_px, arm_density)
    east_v = np.full_like(arm_offsets_along, cv)
    east_u = cu + arm_offsets_along
    west_v = np.full_like(arm_offsets_along, cv)
    west_u = cu - arm_offsets_along
    north_v = cv - arm_offsets_along
    north_u = np.full_like(arm_offsets_along, cu)
    south_v = cv + arm_offsets_along
    south_u = np.full_like(arm_offsets_along, cu)
    vs = np.concatenate([east_v, west_v, north_v, south_v])
    us = np.concatenate([east_u, west_u, north_u, south_u])
    vertices = np.stack([vs, us], axis=-1)
    # The normals are placeholders here (``use_polarity=False`` below).
    normals = np.zeros_like(vertices)
    edge_mask = np.zeros(shape, dtype=bool)
    iv = np.rint(vs).astype(int)
    iu = np.rint(us).astype(int)
    edge_mask[iv, iu] = True
    dt = distance_transform_edt(~edge_mask).astype(np.float64)
    theta_true = 0.04
    dv_true = -1.2
    du_true = 0.7
    pivot = (cv, cu)
    cos_t = math.cos(theta_true)
    sin_t = math.sin(theta_true)
    rot_v = pivot[0] + cos_t * (vertices[:, 0] - pivot[0]) - sin_t * (vertices[:, 1] - pivot[1])
    rot_u = pivot[1] + sin_t * (vertices[:, 0] - pivot[0]) + cos_t * (vertices[:, 1] - pivot[1])
    misaligned_vertices = np.stack([rot_v + dv_true, rot_u + du_true], axis=-1)
    sigmas = np.full(misaligned_vertices.shape[0], 0.5, dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=misaligned_vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        initial_offset_vu=(0.0, 0.0),
        initial_rotation_rad=0.0,
        fit_rotation=True,
        pivot_vu=pivot,
        pivot_distance_px=arm_length_px,
        use_polarity=False,
    )
    # The undoing parameters: dtheta_lm = -theta_true; the LM translation is
    # ``-R(dtheta_lm) (dv_true, du_true)``.
    expected_dtheta = -theta_true
    expected_cos = math.cos(expected_dtheta)
    expected_sin = math.sin(expected_dtheta)
    expected_dv_lm = -(expected_cos * dv_true - expected_sin * du_true)
    expected_du_lm = -(expected_sin * dv_true + expected_cos * du_true)
    assert result.offset_vu[0] == pytest.approx(expected_dv_lm, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(expected_du_lm, abs=0.05)
    assert result.rotation_rad == pytest.approx(expected_dtheta, abs=2.0e-3)


def test_lm_subpixel_refine_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match='must have shape'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((4,)),
            normals_vu=np.zeros((4, 2)),
            sigma_normal_per_vertex_px=np.ones(4),
            image_edge_dt=np.zeros((8, 8)),
        )


def test_lm_subpixel_refine_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match='must be finite and > 0'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([0.0]),
            image_edge_dt=np.zeros((4, 4)),
        )


def test_lm_subpixel_refine_requires_pivot_distance_for_rotation() -> None:
    with pytest.raises(ValueError, match='pivot_distance_px > 0'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([1.0]),
            image_edge_dt=np.zeros((4, 4)),
            fit_rotation=True,
        )


def test_lm_refine_result_freezes_arrays() -> None:
    shape = (32, 32)
    dt = np.full(shape, 5.0, dtype=np.float64)
    vertices = np.array([[16.0, 16.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0]], dtype=np.float64)
    sigmas = np.array([1.0], dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        use_polarity=False,
    )
    assert isinstance(result, LMRefineResult)
    assert not result.covariance.flags.writeable
    assert not result.weights.flags.writeable
    assert not result.residuals_px.flags.writeable


# ---------------------------------------------------------------------------
# Coverage-completion tests for dt_fitting helpers
# ---------------------------------------------------------------------------


def test_tukey_biweight_weights_zero_at_exact_cutoff() -> None:
    """Holland-Welsch is half-open: ``|r| == c`` evaluates to weight zero.

    The implementation uses ``np.abs(scaled) <= 1.0`` to gate the
    polynomial, so the boundary itself is *kept* but the polynomial
    ``(1 - 1**2) ** 2`` is exactly zero.  Verify both halves of the
    statement: weight is zero, no off-by-one excludes the boundary.
    """
    c = DEFAULT_TUKEY_C
    weights = tukey_biweight_weights(np.array([c, -c]), c=c)
    assert float(weights[0]) == 0.0
    assert float(weights[1]) == 0.0


def test_coarse_ncc_search_zero_window_returns_origin() -> None:
    """A search window of ``(0, 0)`` means only the origin shift is scanned.

    Off-by-one safety: the implementation iterates
    ``range(-margin_v, margin_v + 1)`` which yields a single ``0`` step.
    The function must return ``(0, 0)`` for any inputs without raising.
    """
    edge_mask = np.zeros((8, 8), dtype=bool)
    edge_mask[3, 3] = True
    polyline_mask = np.zeros((8, 8), dtype=bool)
    polyline_mask[3, 3] = True
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (0, 0))
    assert (dv, du) == (0, 0)


def test_coarse_ncc_search_unit_window_visits_each_axis_step() -> None:
    """A window of ``(1, 1)`` must cover -1, 0, +1 in each axis.

    Plant a single edge pixel at the boundary of the (1, 1) window and
    verify the function reports that exact integer offset (which means
    the iteration scanned all nine cells, not just the eight non-origin
    ones).
    """
    edge_mask = np.zeros((8, 8), dtype=bool)
    edge_mask[4, 4] = True
    polyline_mask = np.zeros((8, 8), dtype=bool)
    polyline_mask[3, 3] = True
    dv, du = coarse_ncc_search(edge_mask, polyline_mask, (1, 1))
    assert (dv, du) == (1, 1)


def test_lm_subpixel_refine_requires_image_gradient_when_polarity_enabled() -> None:
    """Polarity-enabled LM must reject a ``None`` gradient input.

    The validation path returns a clear ``ValueError`` so a caller that
    forgot to populate ``NavContext.image_gradient_vu_ext`` learns about
    the omission immediately rather than via a silent NoneType crash
    inside the polarity filter.
    """
    with pytest.raises(ValueError, match='use_polarity=True requires image_gradient_vu'):
        lm_subpixel_refine(
            vertices_vu=np.zeros((1, 2)),
            normals_vu=np.zeros((1, 2)),
            sigma_normal_per_vertex_px=np.array([1.0]),
            image_edge_dt=np.zeros((4, 4)),
            use_polarity=True,
            image_gradient_vu=None,
        )


def test_lm_subpixel_refine_bails_out_when_damping_saturates() -> None:
    """The LM loop has an early exit when ``lambda`` saturates at ``1.0e6``.

    With a flat (constant-valued) DT the cost function has zero gradient
    and zero Hessian; every trial step fails to reduce cost; the damping
    factor ramps up multiplicatively until it crosses ``1.0e6`` and the
    loop terminates without convergence.  This exercise covers the
    ``if lambda_ >= 1.0e6: break`` branch in the LM iteration body.
    """
    shape = (16, 16)
    flat_dt = np.full(shape, 5.0, dtype=np.float64)
    vertices = np.array([[8.0, 8.0]], dtype=np.float64)
    normals = np.array([[1.0, 0.0]], dtype=np.float64)
    sigmas = np.array([1.0], dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=flat_dt,
        use_polarity=False,
        max_iterations=50,
    )
    # The loop terminates without converging on a flat DT.
    assert result.converged is False
    # The offset stays at the initial value (no successful step reduced cost).
    assert result.offset_vu == (0.0, 0.0)


# ---------------------------------------------------------------------------
# gradient_ridge_refine
# ---------------------------------------------------------------------------


def _gradient_magnitude_for_circle(shape: tuple[int, int], radius: float) -> np.ndarray:
    """Return the continuous gradient magnitude of a smoothed step-edge disc.

    A disc step edge is symmetric, so after the Gaussian smooth the gradient
    magnitude ridge coincides with the geometric circle -- the unbiased case
    the ridge refinement is correct for.
    """
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    image = _render_image_with_circle(shape, (cv, cu), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    mag: np.ndarray = np.hypot(grad[..., 0], grad[..., 1])
    return mag


def test_gradient_ridge_refine_recovers_subpixel_translation() -> None:
    """On a symmetric edge the ridge stage recovers the planted sub-pixel shift."""
    shape = (96, 96)
    radius = 18.0
    grad_mag = _gradient_magnitude_for_circle(shape, radius)
    # Model circle sits 1.5 px / 2.5 px off the image circle; the aligning
    # offset is therefore (-1.5, -2.5).
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    inward_normals = -outward_normals
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = gradient_ridge_refine(
        vertices_vu=vertices,
        normals_vu=inward_normals,
        sigma_normal_per_vertex_px=sigmas,
        gradient_magnitude=grad_mag,
        initial_offset_vu=(-1.0, -2.0),
    )
    assert result.applied is True
    assert result.offset_vu[0] == pytest.approx(-1.5, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(-2.5, abs=0.05)
    assert result.iterations >= 1


def test_gradient_ridge_refine_returns_ridge_refine_result_type() -> None:
    """The helper returns the documented dataclass."""
    shape = (96, 96)
    radius = 18.0
    grad_mag = _gradient_magnitude_for_circle(shape, radius)
    vertices, outward_normals = _build_circle_polyline((shape[0] / 2.0, shape[1] / 2.0), radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = gradient_ridge_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        gradient_magnitude=grad_mag,
        initial_offset_vu=(0.0, 0.0),
    )
    assert isinstance(result, RidgeRefineResult)


def test_gradient_ridge_refine_keeps_pose_when_displacement_capped() -> None:
    """An over-tight displacement cap discards the refinement and keeps the seed."""
    shape = (96, 96)
    radius = 18.0
    grad_mag = _gradient_magnitude_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = gradient_ridge_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        gradient_magnitude=grad_mag,
        initial_offset_vu=(-1.0, -2.0),
        max_total_displacement_px=0.05,
    )
    assert result.applied is False
    assert result.offset_vu == (-1.0, -2.0)


def test_gradient_ridge_refine_keeps_pose_when_no_gradient() -> None:
    """A flat (zero) gradient field yields no usable ridge; the seed is kept."""
    shape = (64, 64)
    grad_mag = np.zeros(shape, dtype=np.float64)
    vertices, outward_normals = _build_circle_polyline((32.0, 32.0), 16.0, 48)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = gradient_ridge_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        gradient_magnitude=grad_mag,
        initial_offset_vu=(0.3, -0.4),
    )
    assert result.applied is False
    assert result.offset_vu == (0.3, -0.4)


def test_gradient_ridge_refine_recovers_small_rotation() -> None:
    """With ``fit_rotation`` the stage refines a small planted roll."""
    shape = (128, 128)
    radius = 24.0
    grad_mag = _gradient_magnitude_for_circle(shape, radius)
    cv = shape[0] / 2.0
    cu = shape[1] / 2.0
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 96)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = gradient_ridge_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        gradient_magnitude=grad_mag,
        initial_offset_vu=(0.0, 0.0),
        fit_rotation=True,
        pivot_vu=(cv, cu),
        pivot_distance_px=float(math.hypot(cv, cu)),
    )
    # A centred circle is rotationally symmetric, so the fit stays put and
    # exercises the rotation Jacobian path without diverging.
    assert result.offset_vu[0] == pytest.approx(0.0, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'vertices_vu': np.zeros((0, 2))}, r'vertices_vu must have shape'),
        ({'normals_vu': np.zeros((3, 2))}, r'normals_vu must match'),
        ({'sigma_normal_per_vertex_px': np.zeros(8)}, r'must be finite and > 0'),
        ({'gradient_magnitude': np.zeros((5, 5, 2))}, r'gradient_magnitude must be 2-D'),
    ],
)
def test_gradient_ridge_refine_rejects_invalid_inputs(
    kwargs: dict[str, np.ndarray], message: str
) -> None:
    """Shape / value guards fire with informative messages."""
    shape = (64, 64)
    vertices, outward_normals = _build_circle_polyline((32.0, 32.0), 16.0, 8)
    base = {
        'vertices_vu': vertices,
        'normals_vu': -outward_normals,
        'sigma_normal_per_vertex_px': np.full(vertices.shape[0], 0.5, dtype=np.float64),
        'gradient_magnitude': np.zeros(shape, dtype=np.float64),
        'initial_offset_vu': (0.0, 0.0),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        gradient_ridge_refine(**base)  # type: ignore[arg-type]


def test_gradient_ridge_refine_requires_pivot_distance_for_rotation() -> None:
    """``fit_rotation`` without a positive pivot distance is rejected."""
    vertices, outward_normals = _build_circle_polyline((32.0, 32.0), 16.0, 8)
    with pytest.raises(ValueError, match='requires pivot_distance_px'):
        gradient_ridge_refine(
            vertices_vu=vertices,
            normals_vu=-outward_normals,
            sigma_normal_per_vertex_px=np.full(vertices.shape[0], 0.5, dtype=np.float64),
            gradient_magnitude=np.zeros((64, 64), dtype=np.float64),
            initial_offset_vu=(0.0, 0.0),
            fit_rotation=True,
        )


def test_lm_subpixel_refine_final_gradient_ridge_runs_on_symmetric_edge() -> None:
    """The ``final_gradient_ridge`` path recovers the planted shift on a clean edge."""
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
        final_gradient_ridge=True,
    )
    assert result.offset_vu[0] == pytest.approx(-1.5, abs=0.05)
    assert result.offset_vu[1] == pytest.approx(-2.5, abs=0.05)


def test_lm_subpixel_refine_final_gradient_ridge_requires_gradient() -> None:
    """``final_gradient_ridge`` without an image gradient is rejected up front."""
    shape = (32, 32)
    dt = _build_dt_for_circle(shape, 8.0)
    vertices, outward_normals = _build_circle_polyline((16.0, 16.0), 8.0, 16)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    with pytest.raises(ValueError, match='final_gradient_ridge=True requires image_gradient_vu'):
        lm_subpixel_refine(
            vertices_vu=vertices,
            normals_vu=-outward_normals,
            sigma_normal_per_vertex_px=sigmas,
            image_edge_dt=dt,
            use_polarity=False,
            final_gradient_ridge=True,
        )


# --- find_secondary_dt_minimum (issue #125) ---


def _two_line_dt(shape: tuple[int, int], u1: int, u2: int) -> np.ndarray:
    """DT with zero cost along two vertical lines at columns ``u1`` and ``u2``."""
    us = np.arange(shape[1], dtype=np.float64)
    row = np.minimum(np.abs(us - u1), np.abs(us - u2))
    return np.tile(row, (shape[0], 1))


def _vertical_polyline(u: float, v_lo: int, v_hi: int) -> np.ndarray:
    """Vertical polyline of vertices at column ``u`` spanning rows ``[v_lo, v_hi)``."""
    vs = np.arange(v_lo, v_hi, dtype=np.float64)
    return np.stack([vs, np.full_like(vs, u)], axis=-1)


def test_find_secondary_dt_minimum_reports_competing_line() -> None:
    """A second zero-cost line inside the window is reported as a rival basin."""
    dt = _two_line_dt((100, 100), u1=30, u2=50)
    vertices = _vertical_polyline(30.0, 20, 80)
    basin = find_secondary_dt_minimum(
        dt,
        vertices,
        converged_offset_vu=(0.0, 0.0),
        search_window_vu=(2, 40),
        exclude_radius_px=5.0,
    )
    assert basin is not None
    assert basin.offset_vu == (0, 20)
    assert basin.distance_px == pytest.approx(20.0)
    assert basin.cost_px == pytest.approx(0.0)
    assert basin.converged_cost_px == pytest.approx(0.0)


def test_find_secondary_dt_minimum_exclude_radius_hides_converged_basin() -> None:
    """Shifts inside the exclusion radius never count as rivals."""
    dt = _two_line_dt((100, 100), u1=30, u2=30)  # single line: unimodal surface
    vertices = _vertical_polyline(30.0, 20, 80)
    basin = find_secondary_dt_minimum(
        dt,
        vertices,
        converged_offset_vu=(0.0, 0.0),
        search_window_vu=(2, 40),
        exclude_radius_px=5.0,
    )
    assert basin is not None
    # The best rival sits just outside the exclusion radius with a cost equal
    # to its column distance from the single zero line -- clearly worse than
    # the perfect converged fit. The minimum eligible cost is exactly 5.0
    # (du = +/-5 needs |dv| >= 1 to clear the 5.0 px exclusion radius), and
    # the cost-then-distance tie-break lands on (dv, du) = (+/-1, +/-5) at
    # distance sqrt(26).
    assert basin.cost_px == pytest.approx(5.0)
    assert basin.distance_px == pytest.approx(math.sqrt(26.0))


def test_find_secondary_dt_minimum_empty_polyline_returns_none() -> None:
    """An empty polyline yields no basin verdict."""
    dt = _two_line_dt((50, 50), u1=10, u2=30)
    basin = find_secondary_dt_minimum(
        dt,
        np.zeros((0, 2), dtype=np.float64),
        converged_offset_vu=(0.0, 0.0),
        search_window_vu=(2, 10),
        exclude_radius_px=5.0,
    )
    assert basin is None


# ---------------------------------------------------------------------------
# coarse_ncc_search_scored
# ---------------------------------------------------------------------------


def test_coarse_ncc_search_scored_matches_offset_form() -> None:
    """The scored form returns the same peak offset as the offset-only form."""
    shape = (64, 64)
    polyline_mask = _render_circle_mask(shape, (32.0, 32.0), 12.0)
    edge_mask = np.roll(polyline_mask, shift=(3, -2), axis=(0, 1))
    scored = coarse_ncc_search_scored(edge_mask, polyline_mask, (10, 10))
    assert scored.offset_vu == (3, -2)


def test_coarse_ncc_search_scored_perfect_overlap_scores_one() -> None:
    """A fully self-aligned mask puts every vertex on an edge pixel."""
    shape = (32, 32)
    polyline_mask = _render_circle_mask(shape, (16.0, 16.0), 8.0)
    scored = coarse_ncc_search_scored(polyline_mask, polyline_mask, (4, 4))
    assert scored.score == pytest.approx(1.0)


def test_coarse_ncc_search_scored_partial_overlap_scores_fraction() -> None:
    """The score is the winning shift's edge-pixel match fraction."""
    shape = (40, 40)
    polyline_mask = np.zeros(shape, dtype=bool)
    polyline_mask[20, 10:30] = True  # 20 vertices on one row
    edge_mask = np.zeros(shape, dtype=bool)
    edge_mask[20, 10:20] = True  # only half of them are detected
    scored = coarse_ncc_search_scored(edge_mask, polyline_mask, (2, 2))
    assert scored.offset_vu == (0, 0)
    assert scored.score == pytest.approx(0.5)


def test_coarse_ncc_search_scored_empty_polyline_scores_zero() -> None:
    shape = (24, 24)
    edge_mask = np.zeros(shape, dtype=bool)
    edge_mask[10, 10] = True
    scored = coarse_ncc_search_scored(edge_mask, np.zeros(shape, dtype=bool), (3, 3))
    assert scored.offset_vu == (0, 0)
    assert scored.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# polarity_rejected_count
# ---------------------------------------------------------------------------


def test_lm_subpixel_refine_reports_polarity_rejected_count() -> None:
    """The rejected-vertex count from the seed polarity filter is surfaced."""
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    cv = shape[0] / 2.0 + 1.5
    cu = shape[1] / 2.0 + 2.5
    vertices, outward_normals = _build_circle_polyline((cv, cu), radius, 64)
    normals = -outward_normals  # inward normals are accepted on a bright disc
    rejected = np.arange(0, 12)
    normals[rejected] = outward_normals[rejected]
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(-1.0, -2.0),
        use_polarity=True,
    )
    assert result.polarity_rejected_count == 12


def test_lm_subpixel_refine_polarity_rejected_count_zero_without_polarity() -> None:
    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    vertices, outward_normals = _build_circle_polyline((shape[0] / 2.0, shape[1] / 2.0), radius, 48)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        use_polarity=False,
    )
    assert result.polarity_rejected_count == 0


# ---------------------------------------------------------------------------
# Ridge-verified convergence
# ---------------------------------------------------------------------------


def test_lm_converged_flag_set_by_applied_converged_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged ridge stage verifies a pose the DT-LM left at its cap.

    The DT-LM routinely burns its iteration budget stalled on the
    integer-quantized DT of a dense edge scene (the condition the ridge
    stage exists to polish), so an applied AND converged ridge marks the
    reported pose verified.  The ridge is forged so the test pins the OR
    logic itself, not the ridge numerics.
    """
    from spindoctor.nav_technique.dt_fitting import ridge as _ridge_mod

    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    vertices, outward_normals = _build_circle_polyline((shape[0] / 2.0, shape[1] / 2.0), radius, 48)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)

    def forged_ridge(**kwargs: object) -> RidgeRefineResult:
        offset = kwargs['initial_offset_vu']
        assert isinstance(offset, tuple)
        return RidgeRefineResult(
            offset_vu=(float(offset[0]), float(offset[1])),
            rotation_rad=0.0,
            iterations=2,
            converged=True,
            applied=True,
        )

    # ``lm`` calls through the ridge MODULE (``_ridge.gradient_ridge_refine``),
    # so patching the attribute on that module is what the driver sees.
    monkeypatch.setattr(_ridge_mod, 'gradient_ridge_refine', forged_ridge)
    # A far-off seed with a one-iteration budget cannot meet the step
    # tolerance, so the DT-LM itself reports converged=False.
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(4.0, 4.0),
        use_polarity=False,
        max_iterations=1,
        final_gradient_ridge=True,
    )
    assert result.converged is True


def test_lm_unconverged_when_ridge_not_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unapplied ridge leaves the DT-LM's own convergence verdict standing."""
    from spindoctor.nav_technique.dt_fitting import ridge as _ridge_mod

    shape = (96, 96)
    radius = 18.0
    dt = _build_dt_for_circle(shape, radius)
    vertices, outward_normals = _build_circle_polyline((shape[0] / 2.0, shape[1] / 2.0), radius, 48)
    sigmas = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    image = _render_image_with_circle(shape, (shape[0] / 2.0, shape[1] / 2.0), radius)
    grad = compute_image_gradient_vu(image, sigma_px=DEFAULT_IMAGE_GRADIENT_SIGMA_PX)

    def forged_ridge(**kwargs: object) -> RidgeRefineResult:
        offset = kwargs['initial_offset_vu']
        assert isinstance(offset, tuple)
        return RidgeRefineResult(
            offset_vu=(float(offset[0]), float(offset[1])),
            rotation_rad=0.0,
            iterations=0,
            converged=False,
            applied=False,
        )

    # ``lm`` calls through the ridge MODULE (``_ridge.gradient_ridge_refine``),
    # so patching the attribute on that module is what the driver sees.
    monkeypatch.setattr(_ridge_mod, 'gradient_ridge_refine', forged_ridge)
    result = lm_subpixel_refine(
        vertices_vu=vertices,
        normals_vu=-outward_normals,
        sigma_normal_per_vertex_px=sigmas,
        image_edge_dt=dt,
        image_gradient_vu=grad,
        initial_offset_vu=(4.0, 4.0),
        use_polarity=False,
        max_iterations=1,
        final_gradient_ridge=True,
    )
    assert result.converged is False
