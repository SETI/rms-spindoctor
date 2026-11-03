"""Unit tests for the edge-strength coarse search used to seed ``RingEdgeNav``."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.nav_technique.dt_fitting import (
    coarse_edge_strength_search_scored,
    coarse_ncc_search,
    edge_strength_score,
)


def _vertical_segment(column: float, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and ``+u`` normals of a vertical model segment at ``column``."""
    vertices = np.stack([rows.astype(np.float64), np.full(rows.size, column)], axis=-1)
    normals = np.tile(np.array([0.0, 1.0]), (rows.size, 1))
    return vertices, normals


def test_strong_edge_outweighs_a_denser_population_of_weak_edges() -> None:
    """One strong step wins over weak parallel edges nearer the origin.

    This is the ring-texture failure in miniature.  Every column of the
    distractor carries a detected edge under the whole model, as does the
    true edge, so the binary overlap search ties them and its tie-break picks
    the distractor nearest the origin.  Weighting by the gradient under each
    vertex picks the strong step.
    """
    h, w = 40, 60
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(30.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    # Weak texture edges at du = -1, 0 and +1.
    for column in (29, 30, 31):
        edge_mask[10:30, column] = True
        gradient[10:30, column, 1] = 0.5
    # The strong cataloged step at du = +6.
    edge_mask[10:30, 36] = True
    gradient[10:30, 36, 1] = 20.0
    polyline_mask = np.zeros((h, w), dtype=bool)
    polyline_mask[rows, 30] = True
    assert coarse_ncc_search(edge_mask, polyline_mask, (0, 8)) == (0, 0)
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 6)


def test_edge_strength_search_is_polarity_blind() -> None:
    """A step whose gradient runs against the model normal still matches in full."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    edge_mask[10:30, 23] = True
    gradient[10:30, 23, 1] = -5.0
    edge_mask[10:30, 16] = True
    gradient[10:30, 16, 1] = 4.0
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 3)


def test_edge_strength_search_discounts_an_edge_crossing_at_an_angle() -> None:
    """A strong edge whose gradient is perpendicular to the normal contributes nothing."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    # A strong edge at du = -4 whose gradient runs along the model's tangent.
    edge_mask[10:30, 16] = True
    gradient[10:30, 16, 0] = 50.0
    # A modest edge aligned with the model normal at du = +5.
    edge_mask[10:30, 25] = True
    gradient[10:30, 25, 1] = 1.0
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 5)


def test_edge_strength_search_ignores_edges_outside_the_valid_mask() -> None:
    """The step onto the zero-padded margin cannot capture the seed."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    # The frame border, strongest edge in the image, at du = +6.
    edge_mask[:, 26] = True
    gradient[:, 26, 1] = 100.0
    # The real ring edge at du = -3.
    edge_mask[10:30, 17] = True
    gradient[10:30, 17, 1] = 2.0
    valid_mask = np.zeros((h, w), dtype=bool)
    valid_mask[:, :24] = True
    unmasked = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert unmasked.offset_vu == (0, 6)
    masked = coarse_edge_strength_search_scored(
        edge_mask, gradient, vertices, normals, (0, 8), valid_mask=valid_mask
    )
    assert masked.offset_vu == (0, -3)


def test_edge_strength_match_fraction_excludes_masked_edges() -> None:
    """Vertices on masked-out edge pixels stay in the denominator but never match."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    # A strong edge under the model whose lower half lies on masked-out pixels.
    edge_mask[10:30, 20] = True
    gradient[10:30, 20, 1] = 5.0
    valid_mask = np.ones((h, w), dtype=bool)
    valid_mask[20:, :] = False
    result = coarse_edge_strength_search_scored(
        edge_mask, gradient, vertices, normals, (0, 0), valid_mask=valid_mask
    )
    assert result.score == pytest.approx(0.5)


def test_edge_strength_search_reports_the_plain_match_fraction() -> None:
    """The returned score is the binary per-vertex match fraction at the winner."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    edge_mask[10:25, 23] = True  # 15 of 20 vertices on a detected edge at du = +3
    gradient[10:25, 23, 1] = 7.0
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 3)
    assert result.score == pytest.approx(0.75)


def test_edge_strength_search_low_support_shift_is_ineligible() -> None:
    """A shift that pushes most vertices off the array cannot win, however strong its edge.

    The model is three short vertical lines at different rows.  Only a shift
    that carries the two right-hand lines off the array lays the left line on
    the very strong edge, so the minimum-support guard must rule it out and
    leave the weak match under the middle line.
    """
    h, w = 40, 40
    left, _ = _vertical_segment(2.0, np.arange(0, 10))
    middle, _ = _vertical_segment(20.0, np.arange(10, 20))
    right, _ = _vertical_segment(37.0, np.arange(20, 30))
    vertices = np.concatenate([left, middle, right])
    normals = np.tile(np.array([0.0, 1.0]), (30, 1))
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    edge_mask[0:10, 39] = True
    gradient[0:10, 39, 1] = 1000.0
    edge_mask[10:20, 23] = True
    gradient[10:20, 23, 1] = 1.0
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 37))
    assert result.offset_vu == (0, 3)


def test_edge_strength_search_zero_normal_never_contributes() -> None:
    """A degenerate zero-length normal adds no weight to any shift."""
    h, w = 20, 20
    vertices = np.array([[10.0, 10.0], [11.0, 10.0]])
    normals = np.array([[0.0, 0.0], [0.0, 1.0]])
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    # Only the zero-normal vertex finds a strong edge at du = +2; the other
    # vertex finds a weak one at du = -1.
    edge_mask[10, 12] = True
    gradient[10, 12, 1] = 100.0
    edge_mask[11, 9] = True
    gradient[11, 9, 1] = 1.0
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 3))
    assert result.offset_vu == (0, -1)


def test_edge_strength_search_empty_polyline_returns_origin() -> None:
    """An empty polyline yields the origin offset and a zero score."""
    edge_mask = np.zeros((16, 16), dtype=bool)
    gradient = np.zeros((16, 16, 2), np.float64)
    vertices = np.empty((0, 2), np.float64)
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, vertices, (3, 3))
    assert result.offset_vu == (0, 0)
    assert result.score == 0.0


@pytest.mark.parametrize(
    ('gradient', 'vertices', 'normals', 'valid_mask', 'message'),
    [
        (
            np.zeros((8, 8), np.float64),
            np.zeros((1, 2), np.float64),
            np.zeros((1, 2), np.float64),
            None,
            r'image_gradient_vu must have shape \(H, W, 2\)',
        ),
        (
            np.zeros((6, 6, 2), np.float64),
            np.zeros((1, 2), np.float64),
            np.zeros((1, 2), np.float64),
            None,
            'shape mismatch: edge_mask',
        ),
        (
            np.zeros((8, 8, 2), np.float64),
            np.zeros((1, 2), np.float64),
            np.zeros((1, 2), np.float64),
            np.ones((7, 8), dtype=bool),
            'vs valid_mask',
        ),
        (
            np.zeros((8, 8, 2), np.float64),
            np.zeros((1, 3), np.float64),
            np.zeros((1, 2), np.float64),
            None,
            r'vertices_vu must have shape \(N, 2\)',
        ),
        (
            np.zeros((8, 8, 2), np.float64),
            np.zeros((2, 2), np.float64),
            np.zeros((1, 2), np.float64),
            None,
            'normals_vu must match vertices_vu shape',
        ),
    ],
)
def test_edge_strength_search_rejects_invalid_inputs(
    gradient: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    valid_mask: np.ndarray | None,
    message: str,
) -> None:
    """Malformed array shapes are rejected with a named message."""
    edge_mask = np.zeros((8, 8), bool)
    with pytest.raises(ValueError, match=message):
        coarse_edge_strength_search_scored(
            edge_mask, gradient, vertices, normals, (1, 1), valid_mask=valid_mask
        )


def test_edge_strength_search_rejects_float_window_entry() -> None:
    """A float window entry is rejected with TypeError, matching the other searches."""
    edge_mask = np.zeros((8, 8), bool)
    gradient = np.zeros((8, 8, 2), np.float64)
    vertices = np.zeros((1, 2), np.float64)
    with pytest.raises(TypeError, match=r'search_window_vu\[0\] must be int'):
        coarse_edge_strength_search_scored(
            edge_mask,
            gradient,
            vertices,
            vertices,
            (1.5, 1),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize('fraction', [-0.1, 1.5])
def test_edge_strength_search_rejects_out_of_range_min_support_fraction(fraction: float) -> None:
    """A support fraction outside [0, 1] is rejected."""
    edge_mask = np.zeros((8, 8), bool)
    gradient = np.zeros((8, 8, 2), np.float64)
    vertices = np.zeros((1, 2), np.float64)
    with pytest.raises(ValueError, match='min_support_fraction must be in'):
        coarse_edge_strength_search_scored(
            edge_mask, gradient, vertices, vertices, (1, 1), min_support_fraction=fraction
        )


def test_edge_strength_score_matches_the_search_at_its_winner() -> None:
    """The single-shift score is the quantity the search maximizes."""
    h, w = 40, 40
    rows = np.arange(10, 30)
    vertices, normals = _vertical_segment(20.0, rows)
    edge_mask = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w, 2), np.float64)
    edge_mask[10:25, 23] = True  # 15 of 20 vertices at du = +3
    gradient[10:25, 23, 1] = -4.0
    score = edge_strength_score(edge_mask, gradient, vertices, normals, (0, 3))
    # 15 vertices x |-4| / 20 vertices.
    assert score == pytest.approx(3.0)


def test_edge_strength_score_ignores_vertices_shifted_off_the_array() -> None:
    """Vertices pushed outside the array contribute nothing to a shift's score."""
    edge_mask = np.ones((10, 10), dtype=bool)
    gradient = np.zeros((10, 10, 2), np.float64)
    gradient[:, :, 1] = 1.0
    vertices = np.array([[5.0, 2.0], [5.0, 8.0]])
    normals = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert edge_strength_score(edge_mask, gradient, vertices, normals, (0, 3)) == pytest.approx(0.5)


def test_edge_strength_score_of_an_empty_polyline_is_zero() -> None:
    """An empty polyline scores zero rather than dividing by zero."""
    edge_mask = np.ones((4, 4), dtype=bool)
    gradient = np.ones((4, 4, 2), np.float64)
    vertices = np.empty((0, 2), np.float64)
    assert edge_strength_score(edge_mask, gradient, vertices, vertices, (0, 0)) == 0.0
