"""Unit tests for the tolerant edge-strength scores that check ``RingEdgeNav``'s seed."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.nav_technique.dt_fitting import (
    TolerantEdgeStrength,
    coarse_edge_strength_search_scored,
    edge_strength_score,
)

_ROWS = np.arange(10, 30)


def _vertical_edges(columns: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and ``+u`` normals of vertical model edges at ``columns``."""
    vertices = np.concatenate(
        [np.stack([_ROWS.astype(np.float64), np.full(_ROWS.size, c)], axis=-1) for c in columns]
    )
    normals = np.tile(np.array([0.0, 1.0]), (vertices.shape[0], 1))
    return vertices, normals


def _image_with_steps(
    steps: dict[int, float], shape: tuple[int, int] = (40, 70)
) -> tuple[np.ndarray, np.ndarray]:
    """Return an edge mask and gradient holding a vertical edge of each strength at each column."""
    edge_mask = np.zeros(shape, dtype=bool)
    gradient = np.zeros((*shape, 2), np.float64)
    for column, strength in steps.items():
        edge_mask[10:30, column] = True
        gradient[10:30, column, 1] = strength
    return edge_mask, gradient


def _three_edges_and_one_strong_step() -> tuple[np.ndarray, ...]:
    """Return the radial-alias scene in miniature.

    Three model edges at columns 20, 30 and 40 have image edges 1, 2 and -1 px
    from them, each of strength 6.  A stronger step of 10 lies 6 px beyond the
    third model edge, where shifting the model lays that edge exactly on it and
    leaves the other two on nothing.
    """
    vertices, normals = _vertical_edges((20.0, 30.0, 40.0))
    edge_mask, gradient = _image_with_steps({21: 6.0, 32: 6.0, 39: 6.0, 46: 10.0})
    return edge_mask, gradient, vertices, normals


def test_exact_strength_lays_one_model_edge_on_the_strong_step() -> None:
    """Counting only vertices on edge pixels picks the alias."""
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    result = coarse_edge_strength_search_scored(edge_mask, gradient, vertices, normals, (0, 8))
    assert result.offset_vu == (0, 6)


def test_tolerant_strength_finds_the_alignment_of_all_three_edges() -> None:
    """Crediting the edge within 2 px of each vertex puts every model edge on its own."""
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    result = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=2).search(
        (0, 8)
    )
    assert result.offset_vu == (0, 0)


def test_tolerant_search_reports_the_plain_match_fraction() -> None:
    """The winning shift reports the fraction of vertices exactly on an edge pixel.

    At the three-edge alignment no vertex lies exactly on its edge, so the
    fraction the coarse-acquisition gate reads is zero, not the fraction within
    the tolerance.
    """
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    result = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=2).search(
        (0, 8)
    )
    assert result.score == 0.0


@pytest.mark.parametrize(('tolerance', 'expected'), [(2, 4.0), (1, 0.0)])
def test_tolerant_score_reaches_an_edge_exactly_at_the_tolerance(
    tolerance: int, expected: float
) -> None:
    """An edge 2 px across the model counts at a 2 px tolerance and not at 1 px."""
    vertices, normals = _vertical_edges((20.0,))
    edge_mask, gradient = _image_with_steps({22: 4.0})
    score = TolerantEdgeStrength(
        edge_mask, gradient, vertices, normals, tolerance_px=tolerance
    ).score((0, 0))
    assert score == pytest.approx(expected)


def test_tolerant_score_at_zero_tolerance_is_the_exact_score() -> None:
    """With no tolerance the score is the exact edge-strength score."""
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    exact = edge_strength_score(edge_mask, gradient, vertices, normals, (0, 1))
    tolerant = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).score(
        (0, 1)
    )
    assert tolerant == pytest.approx(exact)


def test_tolerant_score_ignores_an_edge_outside_the_valid_mask() -> None:
    """An edge the valid mask leaves out contributes nothing within the tolerance."""
    vertices, normals = _vertical_edges((20.0,))
    edge_mask, gradient = _image_with_steps({21: 4.0})
    valid = np.ones(edge_mask.shape, dtype=bool)
    valid[:, 21] = False
    score = TolerantEdgeStrength(
        edge_mask, gradient, vertices, normals, tolerance_px=2, valid_mask=valid
    ).score((0, 0))
    assert score == 0.0


def test_best_match_fraction_ignores_an_edge_outside_the_valid_mask() -> None:
    """A masked-out edge within the tolerance cannot lift the basin's match fraction."""
    vertices, normals = _vertical_edges((20.0,))
    edge_mask, gradient = _image_with_steps({21: 4.0})
    valid = np.ones(edge_mask.shape, dtype=bool)
    valid[:, 21] = False
    fraction = TolerantEdgeStrength(
        edge_mask, gradient, vertices, normals, tolerance_px=2, valid_mask=valid
    ).best_match_fraction((0, 0))
    assert fraction == 0.0


@pytest.mark.parametrize(('edge_vu', 'expected'), [((11, 11), 3.0), ((10, 11), 0.0)])
def test_tolerant_score_steps_along_the_diagonal_of_a_diagonal_normal(
    edge_vu: tuple[int, int], expected: float
) -> None:
    """A diagonal normal reaches the next pixel on its diagonal, not the one beside it."""
    vertices = np.array([[10.0, 10.0]])
    normals = np.array([[1.0, 1.0]])
    edge_mask = np.zeros((20, 20), dtype=bool)
    gradient = np.zeros((20, 20, 2), np.float64)
    edge_mask[edge_vu] = True
    gradient[edge_vu] = (3.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0))
    score = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=1).score(
        (0, 0)
    )
    assert score == pytest.approx(expected)


def test_tolerant_score_gives_a_zero_normal_nothing() -> None:
    """A vertex without a normal never contributes, even on an edge whose gradient
    lies along another vertex's normal."""
    vertices = np.array([[5.0, 5.0], [20.0, 21.0]])
    normals = np.array([[1.0, 0.0], [0.0, 0.0]])
    edge_mask = np.zeros((40, 40), dtype=bool)
    gradient = np.zeros((40, 40, 2), np.float64)
    edge_mask[20, 21] = True
    gradient[20, 21] = (4.0, 0.0)
    score = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=2).score(
        (0, 0)
    )
    assert score == 0.0


def test_tolerant_score_divides_by_every_vertex() -> None:
    """A vertex shifted off the array loses its credit rather than leaving the count."""
    vertices = np.array([[10.0, 1.0], [10.0, 30.0]])
    normals = np.array([[0.0, 1.0], [0.0, 1.0]])
    edge_mask, gradient = _image_with_steps({28: 4.0})
    score = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).score(
        (0, -2)
    )
    assert score == pytest.approx(2.0)


def test_tolerant_score_credits_only_the_gradient_across_the_edge() -> None:
    """A strong gradient along the model's tangent earns nothing."""
    vertices = np.array([[20.0, 21.0]])
    normals = np.array([[0.0, 1.0]])
    edge_mask = np.zeros((40, 40), dtype=bool)
    gradient = np.zeros((40, 40, 2), np.float64)
    edge_mask[20, 21] = True
    gradient[20, 21] = (4.0, 0.0)
    score = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).score(
        (0, 0)
    )
    assert score == pytest.approx(0.0, abs=1e-9)


def test_tolerant_score_reads_a_normal_at_its_nearest_bin_center() -> None:
    """A normal 10 degrees off the row axis is read at the 11.25-degree bin."""
    vertices = np.array([[20.0, 21.0]])
    angle = np.radians(10.0)
    normals = np.array([[np.cos(angle), np.sin(angle)]])
    edge_mask = np.zeros((40, 40), dtype=bool)
    gradient = np.zeros((40, 40, 2), np.float64)
    edge_mask[20, 21] = True
    gradient[20, 21] = (4.0, 0.0)
    score = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).score(
        (0, 0)
    )
    assert score == pytest.approx(4.0 * np.cos(np.radians(11.25)), rel=1e-6)


def test_tolerant_search_reports_the_fraction_of_in_bounds_vertices_on_edges() -> None:
    """Of two in-bounds vertices at the winning shift, the one on an edge pixel makes half.

    A third vertex lies off the array and is not counted in the fraction.
    """
    vertices = np.array([[10.0, 5.0], [10.0, 20.0], [10.0, -30.0]])
    normals = np.tile(np.array([0.0, 1.0]), (3, 1))
    edge_mask, gradient = _image_with_steps({6: 4.0})
    result = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).search(
        (0, 3)
    )
    assert result.score == pytest.approx(0.5)


def test_tolerant_search_with_no_eligible_shift_returns_the_origin() -> None:
    """A model lying wholly outside the array has no eligible shift."""
    vertices = np.array([[10.0, -20.0], [11.0, -20.0]])
    normals = np.tile(np.array([0.0, 1.0]), (2, 1))
    edge_mask, gradient = _image_with_steps({6: 4.0})
    result = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).search(
        (0, 3)
    )
    assert result.offset_vu == (0, 0)


def test_best_match_fraction_finds_the_exact_placement_within_the_tolerance() -> None:
    """A shift 2 px from the model's edge reaches the shift that lays it on the edge."""
    vertices, normals = _vertical_edges((20.0,))
    edge_mask, gradient = _image_with_steps({22: 4.0})
    tolerant = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=2)
    assert tolerant.best_match_fraction((0, 0)) == pytest.approx(1.0)


def test_compare_drops_a_vertex_off_the_array_without_a_valid_mask() -> None:
    """With no valid mask a vertex outside the array at one shift is left out of both sums."""
    vertices = np.array([[10.0, 1.0], [10.0, 30.0]])
    normals = np.tile(np.array([0.0, 1.0]), (2, 1))
    edge_mask, gradient = _image_with_steps({1: 7.0, 32: 5.0, 30: 3.0})
    tolerant = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0)
    _first, _second, shared = tolerant.compare((0, 0), (0, -2))
    assert shared == 1


def test_tolerant_search_skips_a_shift_that_clips_most_of_the_model() -> None:
    """A strong edge reachable only by pushing most vertices off the array is ineligible.

    Shifting by 3 px lays the first vertex on the strong edge but pushes the other
    two off the right of the array; the weak edge 1 px away keeps all three in.
    """
    vertices = np.array([[10.0, 1.0], [10.0, 37.0], [10.0, 38.0]])
    normals = np.tile(np.array([0.0, 1.0]), (3, 1))
    edge_mask = np.zeros((20, 40), dtype=bool)
    gradient = np.zeros((20, 40, 2), np.float64)
    for column, strength in ((2, 1.0), (4, 50.0)):
        edge_mask[10, column] = True
        gradient[10, column, 1] = strength
    result = TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=0).search(
        (0, 6)
    )
    assert result.offset_vu == (0, 1)


def test_tolerant_search_rejects_a_negative_tolerance() -> None:
    """A negative tolerance fails with a message naming it."""
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    with pytest.raises(ValueError, match='tolerance_px must be non-negative'):
        TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=-1)


def test_tolerant_search_rejects_a_boolean_tolerance() -> None:
    """A boolean tolerance is a caller mistake, not a one-pixel window."""
    edge_mask, gradient, vertices, normals = _three_edges_and_one_strong_step()
    with pytest.raises(TypeError, match='tolerance_px must be int'):
        TolerantEdgeStrength(edge_mask, gradient, vertices, normals, tolerance_px=True)


def test_tolerant_search_of_an_empty_polyline_returns_the_origin() -> None:
    """With no vertices there is nothing to score."""
    edge_mask, gradient = _image_with_steps({21: 4.0})
    result = TolerantEdgeStrength(
        edge_mask, gradient, np.empty((0, 2)), np.empty((0, 2)), tolerance_px=2
    ).search((0, 8))
    assert result.offset_vu == (0, 0)


def test_compare_counts_only_vertices_on_the_sensor_at_both_shifts() -> None:
    """A vertex that only one shift brings onto the sensor adds to neither credit.

    The second vertex sits on a strong edge at the shift that moves it onto the
    valid region and off it at the other, so it is left out of both sums.
    """
    vertices = np.array([[10.0, 10.0], [10.0, 30.0]])
    normals = np.tile(np.array([0.0, 1.0]), (2, 1))
    edge_mask, gradient = _image_with_steps({10: 3.0, 12: 5.0, 32: 9.0})
    valid = np.ones(edge_mask.shape, dtype=bool)
    valid[:, 30] = False
    tolerant = TolerantEdgeStrength(
        edge_mask, gradient, vertices, normals, tolerance_px=0, valid_mask=valid
    )
    _first, second, _shared = tolerant.compare((0, 0), (0, 2))
    assert second == pytest.approx(5.0)
