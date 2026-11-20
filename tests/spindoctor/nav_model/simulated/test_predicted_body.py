"""Unit tests for the navigator's predicted silhouette of a simulated body.

The bounding box is rendered on a canvas around the body rather than on the frame, so
these tests hold it against the silhouette the same body has on the frame.
"""

from typing import Any

import numpy as np

from spindoctor.nav_model.simulated.predicted_body import (
    predicted_silhouette,
    predicted_silhouette_bbox,
)

_SHAPE_VU = (200, 240)


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


def test_bbox_bounds_the_silhouette_on_the_frame() -> None:
    """The box is exactly the bounds of the body's silhouette on the frame."""
    body = _mesh_body()
    vs, us = np.nonzero(predicted_silhouette(body, _SHAPE_VU))
    expected = (int(vs.min()), int(vs.max()), int(us.min()), int(us.max()))
    assert predicted_silhouette_bbox(body, _SHAPE_VU) == expected


def test_bbox_reaches_past_the_frame_edge() -> None:
    """A body over the top-left corner gets a box that starts before row and column 0."""
    bbox = predicted_silhouette_bbox(_mesh_body(center_v=5.0, center_u=5.0), _SHAPE_VU)
    assert bbox is not None
    assert bbox[0] < 0
    assert bbox[2] < 0
