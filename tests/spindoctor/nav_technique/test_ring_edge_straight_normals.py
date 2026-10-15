"""The rank-1 axis of a straight ring edge comes from the line, not its pixel normals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import NavFeatureFactory

from spindoctor.nav_model.ring_polyline import _polyline_from_edge_mask
from spindoctor.nav_technique.nav_technique_ring_edge import aggregate_edge_normal_angle_deg
from spindoctor.nav_technique.ring_edge_geometry import _straight_edge_normals


def _tilted_line(normal_angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a straight edge whose normal makes ``normal_angle_deg`` with +v.

    The vertices and normals come from the rings model's own polyline builder,
    so the normals carry the same 45-degree quantization the model emits; the
    ring radius increases along the normal, which signs them.
    """
    shape = (200, 200)
    angle = math.radians(normal_angle_deg)
    normal = np.array([math.cos(angle), math.sin(angle)])
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    distance = (vs - 100.0) * normal[0] + (us - 100.0) * normal[1]
    mask = np.abs(distance) <= 0.5
    radius = 1.0e5 + distance
    return _polyline_from_edge_mask(mask, radius)


def test_straight_edge_normals_recover_the_line_normal() -> None:
    """A 21.6-degree line's quantized normals are replaced by its true normal."""
    vertices, normals = _tilted_line(68.4)
    replaced = _straight_edge_normals(vertices, normals)
    angle = math.degrees(math.atan2(float(replaced[0, 1]), float(replaced[0, 0])))
    assert angle == pytest.approx(68.4, abs=0.3)


def test_straight_edge_normals_keep_the_outward_sense() -> None:
    """The replacement normal points the way the model's normals summed."""
    vertices, normals = _tilted_line(68.4)
    replaced = _straight_edge_normals(vertices, normals)
    assert float(replaced[0] @ normals.sum(axis=0)) > 0.0


def test_quantized_input_normals_are_off_the_line_normal() -> None:
    """The input normals themselves only take 45-degree directions (the defect)."""
    _vertices, normals = _tilted_line(68.4)
    angles = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    assert np.all(np.isclose(np.mod(angles, 45.0), 0.0) | np.isclose(np.mod(angles, 45.0), 45.0))


def test_aggregate_edge_normal_angle_reads_the_line_not_the_pixel_grid(
    make_ring_feature: NavFeatureFactory,
) -> None:
    """The reported rank-1 constraint angle is the line's, not a multiple of 45 degrees."""
    vertices, normals = _tilted_line(68.4)
    feature = make_ring_feature(
        'tilted', vertices=vertices, outward_normals=normals, is_straight_line=True
    )
    angle = aggregate_edge_normal_angle_deg([feature])
    assert angle is not None
    assert angle == pytest.approx(68.4, abs=0.3)
