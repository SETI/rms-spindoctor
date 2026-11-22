"""Unit tests for the navigator's predicted silhouette of a simulated body.

The silhouette on the extended frame is held against a render at the stated center
plus the margin, and its bounding box against the silhouette the same body has on
the frame and against the extent of a circle far larger than the frame.
"""

import math
from typing import Any

import numpy as np

from spindoctor.nav_model.simulated.predicted_body import (
    predicted_silhouette,
    predicted_silhouette_bbox,
    render_predicted_body,
)

_SHAPE_VU = (200, 240)
_MARGIN_VU = (30, 20)


def _mesh_body(**overrides: Any) -> dict[str, Any]:
    """A lumpy mesh body off the frame center, at a fractional center."""
    return {
        'name': 'MOON',
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': 0.45,
        'mesh_seed': 2,
        'center_v': 90.7,
        'center_u': 130.3,
        'semi_a_px': 30.0,
        'semi_b_px': 18.0,
        'semi_c_px': 18.0,
        'illumination_angle': 30.0,
        'phase_angle': 45.0,
        **overrides,
    }


def test_extended_silhouette_keeps_the_sub_pixel_center() -> None:
    """The extended-frame silhouette is the body at its stated center plus the margin.

    The center sits half a pixel past a whole number, where moving it to a whole
    pixel changes which edge pixels the body touches.
    """
    body = _mesh_body(center_v=90.5, center_u=130.5)
    expected = (
        render_predicted_body(
            body,
            math.radians(45.0),
            shape_vu=(_SHAPE_VU[0] + 2 * _MARGIN_VU[0], _SHAPE_VU[1] + 2 * _MARGIN_VU[1]),
            center_vu=(90.5 + _MARGIN_VU[0], 130.5 + _MARGIN_VU[1]),
        )
        > 0.0
    )
    got = predicted_silhouette(body, _SHAPE_VU, margin_vu=_MARGIN_VU)
    assert np.array_equal(got, expected)


def test_bbox_bounds_the_silhouette_on_the_frame() -> None:
    """For a body inside the frame the box is exactly its frame silhouette's bounds."""
    body = _mesh_body()
    vs, us = np.nonzero(predicted_silhouette(body, _SHAPE_VU))
    expected = (int(vs.min()), int(vs.max()), int(us.min()), int(us.max()))
    assert predicted_silhouette_bbox(body, _SHAPE_VU, margin_vu=_MARGIN_VU) == expected


def test_bbox_reaches_past_the_frame_edge() -> None:
    """A body over the top-left corner gets a box that starts before row and column 0."""
    body = _mesh_body(center_v=5.0, center_u=5.0)
    bbox = predicted_silhouette_bbox(body, _SHAPE_VU, margin_vu=_MARGIN_VU)
    assert bbox is not None
    assert bbox[0] < 0
    assert bbox[2] < 0


def test_bbox_of_a_planet_beside_the_frame_stops_at_the_extended_frame() -> None:
    """A circle far larger than the frame is boxed where it covers the extended frame.

    The circle of radius 1500 px is centered left of the frame with its right edge
    at column 99.5, so it covers every row of the extended frame and every column
    from the start of the margin to column 99.
    """
    planet = {
        'name': 'PLANET',
        'center_v': 100.0,
        'center_u': -1400.5,
        'semi_a_px': 1500.0,
        'semi_b_px': 1500.0,
        'semi_c_px': 1500.0,
    }
    bbox = predicted_silhouette_bbox(planet, _SHAPE_VU, margin_vu=_MARGIN_VU)
    assert bbox == (-_MARGIN_VU[0], _SHAPE_VU[0] + _MARGIN_VU[0] - 1, -_MARGIN_VU[1], 99)
