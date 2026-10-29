"""``RingEdgeNav``'s floor on the strength of the edges a fit matched."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    FlatPolylineFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique import nav_technique_ring_edge
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


def _rotated_corner_scene(
    tilt_deg: float, faint_half_width_px: float
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return a bright corner imaged turned by ``tilt_deg`` and the unturned model edges.

    The model is a horizontal edge at row 60 and a vertical edge at column 580 meeting
    in a corner; the image is that corner turned about the model's vertex centroid.
    Each edge is a step of 10000 except within ``faint_half_width_px`` of the centroid
    along it, where the step is 10, so most of what a translation-only overlay of the
    model still matches, the stretch of each edge nearest the centroid, is faint.
    """
    h_u = np.linspace(20.0, 560.0, 541)
    h_vertices = np.stack([np.full_like(h_u, 60.0), h_u], axis=-1)
    v_v = np.linspace(80.0, 280.0, 201)
    v_vertices = np.stack([v_v, np.full_like(v_v, 580.0)], axis=-1)
    pivot = np.concatenate([h_vertices, v_vertices]).mean(axis=0)
    rows, cols = np.meshgrid(
        np.arange(300, dtype=np.float64), np.arange(600, dtype=np.float64), indexing='ij'
    )
    theta = -math.radians(tilt_deg)
    model_v = pivot[0] + math.cos(theta) * (rows - pivot[0]) - math.sin(theta) * (cols - pivot[1])
    model_u = pivot[1] + math.sin(theta) * (rows - pivot[0]) + math.cos(theta) * (cols - pivot[1])
    faint = ((np.abs(model_u - pivot[1]) < faint_half_width_px) & (model_v < 100.0)) | (
        (np.abs(model_v - pivot[0]) < faint_half_width_px) & (model_u > 540.0)
    )
    inside = (model_v > 60.0) & (model_u < 580.0)
    image = np.where(inside, np.where(faint, 10.0, 10000.0), 0.0)
    edges = [
        (h_vertices, np.tile([-1.0, 0.0], (h_u.size, 1))),
        (v_vertices, np.tile([0.0, 1.0], (v_v.size, 1))),
    ]
    return image, edges


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


@pytest.mark.parametrize(('at_floor', 'spurious'), [(True, False), (False, True)])
def test_ring_edge_nav_edge_strength_floor_is_inclusive(
    at_floor: bool,
    spurious: bool,
    monkeypatch: pytest.MonkeyPatch,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A fraction exactly at the configured floor passes; just below it is spurious."""
    floor = RingEdgeNav()._spurious_min_edge_strength_fraction
    fraction = floor if at_floor else float(np.nextafter(floor, 0.0))
    monkeypatch.setattr(
        nav_technique_ring_edge,
        '_matched_edge_strength_fraction',
        lambda *_args, **_kwargs: fraction,
    )
    image = _step_image((200, 200), faint_step=10.0, bright_disc=10.0)
    vertices, outward = flat_polyline(101.0, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate([feature], make_nav_context(image))
    assert result.spurious is spurious


def test_matched_edge_strength_takes_the_median_not_the_mean() -> None:
    """One very strong matched vertex does not lift the fraction.

    The top rows hold the frame's strongest gradient, 1000, which sets the
    reference; the three matched vertices read 1, 1 and 1000, whose median is 1.
    """
    edge_dt = np.zeros((20, 20))
    gradient = np.zeros((20, 20, 2))
    gradient[:, :, 0] = 1.0
    gradient[:5, :, 0] = 1000.0
    gradient[10, 12, 0] = 1000.0
    vertices = np.array([[10.0, 10.0], [10.0, 11.0], [10.0, 12.0]])
    normals = np.tile([1.0, 0.0], (3, 1))
    valid = np.ones((20, 20), dtype=bool)
    fraction = _matched_edge_strength_fraction(
        edge_dt, gradient, vertices, normals, (0.0, 0.0), valid_mask=valid
    )
    assert fraction == pytest.approx(0.001)


def test_matched_edge_strength_projects_and_ignores_the_sign() -> None:
    """The strength is ``|g . n|``: a gradient against the normal counts in full.

    Every pixel's gradient is ``(-3, 4)``, magnitude 5, so the reference is 5 and
    the projection onto a ``+v`` normal is -3.
    """
    edge_dt = np.zeros((20, 20))
    gradient = np.zeros((20, 20, 2))
    gradient[:, :, 0] = -3.0
    gradient[:, :, 1] = 4.0
    vertices = np.array([[10.0, 10.0]])
    normals = np.array([[1.0, 0.0]])
    valid = np.ones((20, 20), dtype=bool)
    fraction = _matched_edge_strength_fraction(
        edge_dt, gradient, vertices, normals, (0.0, 0.0), valid_mask=valid
    )
    assert fraction == pytest.approx(0.6)


def test_matched_edge_strength_counts_only_the_valid_region() -> None:
    """Neither a match nor the reference strength comes from outside ``valid_mask``.

    The top rows carry gradients of 1000 and 500 against 1 elsewhere but lie
    outside the valid region, so the reference is the valid region's own 1; the
    two vertices matched on the 500 rows count for nothing, and the one matched
    inside reads 1.
    """
    edge_dt = np.zeros((20, 20))
    gradient = np.zeros((20, 20, 2))
    gradient[:, :, 0] = 1.0
    gradient[:2, :, 0] = 1000.0
    gradient[2:5, :, 0] = 500.0
    vertices = np.array([[3.0, 10.0], [3.0, 11.0], [10.0, 10.0]])
    normals = np.tile([1.0, 0.0], (3, 1))
    valid = np.ones((20, 20), dtype=bool)
    valid[:5] = False
    fraction = _matched_edge_strength_fraction(
        edge_dt, gradient, vertices, normals, (0.0, 0.0), valid_mask=valid
    )
    assert fraction == pytest.approx(1.0)


def test_ring_edge_nav_measures_the_matched_edge_against_the_sensor_interior(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A sensor-border step a thousand times the ring step does not veto the ring fit.

    The frame is 10000 bright up to row 130, where the sensor ends; the ring
    step of 10 sits at row 120.  Measured against the whole frame the ring
    step would be far under the floor; measured against the sensor interior
    it is the strongest edge there.
    """
    shape = (200, 160)
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    image = np.where(rows < 130.0, 10000.0, 0.0) + np.where(rows < 120.0, 10.0, 0.0)
    context = make_nav_context(image)
    vertices, outward = flat_polyline(120.5, 20.0, 140.0, 121)
    feature = make_ring_feature(
        'edge', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    result = RingEdgeNav().navigate(
        [feature], dataclasses.replace(context, sensor_mask_ext=rows < 130.0)
    )
    assert result.spurious is False


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


def test_ring_edge_nav_reads_edge_strength_on_the_rotated_fit(
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A fit that recovers a camera rotation is not vetoed for the edges it left behind.

    The corner is imaged turned by half a degree.  Overlaid without that turn, the model
    matches only the faint stretches near its centroid; turned as the fit found it, it
    lies along the whole edge, which is mostly strong.
    """
    image, edges = _rotated_corner_scene(0.5, 110.0)
    features = [
        make_ring_feature(name, vertices=vertices, outward_normals=normals, is_straight_line=False)
        for name, (vertices, normals) in zip(('horizontal', 'vertical'), edges, strict=True)
    ]
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = RingEdgeNav().navigate(features, context)
    assert result.rotation_rad == pytest.approx(math.radians(0.5), abs=math.radians(0.05))
    assert result.spurious is False
