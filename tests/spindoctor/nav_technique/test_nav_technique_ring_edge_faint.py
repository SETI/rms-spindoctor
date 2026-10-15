"""``RingEdgeNav``'s floor on the strength of the edges a fit matched."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    FlatPolylineFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique.nav_technique_ring_edge import RingEdgeNav
from spindoctor.nav_technique.ring_edge_stats import _matched_edge_strength_fraction


def _step_image(shape_vu: tuple[int, int], *, faint_step: float, bright_disc: float) -> np.ndarray:
    """Return a faint horizontal step at row 100 beside a bright disc in one corner."""
    rows, cols = np.meshgrid(
        np.arange(shape_vu[0], dtype=np.float64),
        np.arange(shape_vu[1], dtype=np.float64),
        indexing='ij',
    )
    image = np.where(rows < 100.0, faint_step, 0.0)
    image[np.hypot(rows - 30.0, cols - 170.0) <= 12.0] += bright_disc
    return image


def test_ring_edge_nav_marks_a_fit_on_faint_edges_spurious(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A fit whose edge is a tiny fraction of the frame's strongest is rejected.

    The model fits the faint step cleanly, but a disc a thousand times
    brighter elsewhere in the frame shows what a real edge in this image looks
    like, so the match is the kind a thresholded glow contour gives.
    """
    image = _step_image((200, 200), faint_step=10.0, bright_disc=10000.0)
    vertices, outward = flat_polyline(101.0, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.spurious is True


def test_ring_edge_nav_keeps_a_fit_on_an_edge_comparable_to_the_frames_strongest(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The same fit survives when the disc is no brighter than the step."""
    image = _step_image((200, 200), faint_step=10.0, bright_disc=10.0)
    vertices, outward = flat_polyline(101.0, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.spurious is False


def test_matched_edge_strength_fraction_is_zero_without_a_match() -> None:
    """No vertex within reach of a detected edge reads as no strength at all."""
    edge_dt = np.full((20, 20), 9.0)
    gradient = np.ones((20, 20, 2), np.float64)
    vertices = np.array([[10.0, 10.0]])
    normals = np.array([[1.0, 0.0]])
    valid = np.ones((20, 20), dtype=bool)
    fraction = _matched_edge_strength_fraction(
        edge_dt, gradient, vertices, normals, (0.0, 0.0), valid_mask=valid
    )
    assert fraction == 0.0


def test_matched_edge_strength_fraction_reads_the_projected_gradient() -> None:
    """The fraction is the matched projection over the frame's 99.9th percentile."""
    edge_dt = np.full((20, 20), 9.0)
    edge_dt[10, 12] = 0.0
    gradient = np.zeros((20, 20, 2), np.float64)
    gradient[:, :, 0] = 4.0  # every pixel's magnitude is 4
    gradient[10, 12] = (2.0, 3.0)  # projects to 2 on a +v normal, magnitude ~3.6
    vertices = np.array([[10.0, 10.0]])
    normals = np.array([[1.0, 0.0]])
    valid = np.ones((20, 20), dtype=bool)
    fraction = _matched_edge_strength_fraction(
        edge_dt, gradient, vertices, normals, (0.0, 2.0), valid_mask=valid
    )
    assert fraction == pytest.approx(0.5)


@pytest.mark.parametrize('floor', [-0.1, 1.5])
def test_ring_edge_nav_rejects_an_edge_strength_floor_outside_unit_range(
    floor: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edge-strength floor outside [0, 1] fails at construction."""
    bad_tuning = dict(RingEdgeNav.tuning)
    bad_tuning['spurious_min_edge_strength_fraction'] = floor
    monkeypatch.setattr(RingEdgeNav, 'tuning', bad_tuning)
    with pytest.raises(ValueError, match='spurious_min_edge_strength_fraction must lie in'):
        RingEdgeNav()
