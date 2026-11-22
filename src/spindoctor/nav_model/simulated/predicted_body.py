"""The navigator's predicted silhouette of a simulated body.

Every navigator-side model that draws a simulated body -- the body model's template
and its occluders, and the haze model's occluders and contaminant boxes -- draws it
here, from the body's entry in the navigator view (``obs.nav_params['bodies']``).  The
entry carries only idealized keys, since the boundary filter strips the truth before
this code sees it.  Three rules apply in this one place, so the models cannot draw the
same body differently:

- a body that states no center sits at the frame center, where the image-side
  renderer puts it;
- a body that states no ``semi_c_px`` takes the shorter of its two image-plane
  semi-axes;
- a ``polyhedral_mesh`` body is drawn as its mesh through
  :func:`~spindoctor.sim.shared.mesh.render_mesh_body_image`, and any other body as an
  ellipsoid through :func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid` and
  :func:`~spindoctor.sim.shared.ellipsoid.render_lambert_ellipsoid`.

The predicted brightness is 0 on the body's night side, as the SPICE-backed body
model's is, so the silhouette comes from the geometry instead: the pixels the outline
touches, lit or not.  A render returns both as a :class:`PredictedBody`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.sim.shared.ellipsoid import (
    ellipsoid_coverage,
    project_ellipsoid,
    render_lambert_ellipsoid,
)
from spindoctor.sim.shared.mesh import mesh_spec_from_params, render_mesh_body_image
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'PredictedBody',
    'predicted_center_vu',
    'predicted_semi_axes_px',
    'predicted_silhouette',
    'predicted_silhouette_bbox',
    'render_predicted_body',
]


@dataclass(frozen=True)
class PredictedBody:
    """A body's predicted render on one canvas.

    Parameters:
        image: The predicted brightness in ``[0, 1]``, 0 outside the body and on its
            night side.
        silhouette: The pixels the body's outline touches, lit or not.
    """

    image: NDArrayFloatType
    silhouette: NDArrayBoolType


def predicted_center_vu(
    nav_body: Mapping[str, Any], data_shape_vu: tuple[int, int]
) -> tuple[float, float]:
    """Return a body's predicted center on the frame, in pixel-corner coordinates.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame.

    Returns:
        ``(center_v, center_u)``: the stated center, or the frame center for an axis
        the entry does not state.
    """
    return (
        float(nav_body.get('center_v', data_shape_vu[0] / 2.0)),
        float(nav_body.get('center_u', data_shape_vu[1] / 2.0)),
    )


def predicted_semi_axes_px(nav_body: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return a body's three semi-axes in pixels.

    Parameters:
        nav_body: The body's entry in the navigator view.

    Returns:
        ``(semi_a_px, semi_b_px, semi_c_px)``; ``semi_c_px`` is the shorter of the
        other two when the entry does not state it.
    """
    semi_a_px = float(nav_body.get('semi_a_px', 0.0))
    semi_b_px = float(nav_body.get('semi_b_px', 0.0))
    semi_c_px = float(nav_body.get('semi_c_px', min(semi_a_px, semi_b_px)))
    return semi_a_px, semi_b_px, semi_c_px


def render_predicted_body(
    nav_body: Mapping[str, Any],
    phase_angle_rad: float,
    *,
    shape_vu: tuple[int, int],
    center_vu: tuple[float, float],
) -> PredictedBody:
    """Render a body's predicted image and silhouette on a canvas.

    The body is drawn at full anti-aliasing, with the shape, pose, and lighting its
    entry states.

    Parameters:
        nav_body: The body's entry in the navigator view.
        phase_angle_rad: Phase angle of the render in radians.
        shape_vu: ``(rows, columns)`` of the canvas.
        center_vu: The body center on the canvas, in pixel-corner coordinates.  A
            render on the frame passes :func:`predicted_center_vu`.

    Returns:
        The :class:`PredictedBody` on the canvas.
    """
    semi_axes_px = predicted_semi_axes_px(nav_body)
    illumination_angle_rad = math.radians(float(nav_body.get('illumination_angle', 0.0)))
    if str(nav_body.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh':
        mesh_render = render_mesh_body_image(
            size=shape_vu,
            center=center_vu,
            semi_axes_px=semi_axes_px,
            spec=mesh_spec_from_params(nav_body),
            illumination_angle=illumination_angle_rad,
            phase_angle=phase_angle_rad,
            anti_aliasing=1.0,
        )
        return PredictedBody(
            image=mesh_render.intensity,
            silhouette=np.asarray(mesh_render.coverage > 0.0, dtype=bool),
        )
    projection = project_ellipsoid(
        shape_vu,
        center_vu,
        semi_axes_px[0],
        semi_b_px=semi_axes_px[1],
        semi_c_px=semi_axes_px[2],
        rotation_z=math.radians(float(nav_body.get('rotation_z', 0.0))),
        sub_observer_lat=math.radians(float(nav_body.get('sub_observer_lat_deg', 0.0))),
        anti_aliasing=1.0,
    )
    return PredictedBody(
        image=render_lambert_ellipsoid(
            projection, illumination_angle=illumination_angle_rad, phase_angle=phase_angle_rad
        ),
        silhouette=ellipsoid_coverage(projection),
    )


def predicted_silhouette(
    nav_body: Mapping[str, Any],
    data_shape_vu: tuple[int, int],
    *,
    margin_vu: tuple[int, int] = (0, 0),
) -> NDArrayBoolType:
    """Return a body's predicted silhouette on the frame, or on the frame extended.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame, which places a body that
            states no center.
        margin_vu: ``(margin_v, margin_u)`` added to the frame on each side.  The
            default draws the frame alone; the extended field of view passes its
            margins.

    Returns:
        The boolean mask, of the frame's shape plus twice each margin, of the pixels
        the body covers, rendered at its own phase.
    """
    center_v, center_u = predicted_center_vu(nav_body, data_shape_vu)
    return render_predicted_body(
        nav_body,
        math.radians(float(nav_body.get('phase_angle', 0.0))),
        shape_vu=(data_shape_vu[0] + 2 * margin_vu[0], data_shape_vu[1] + 2 * margin_vu[1]),
        center_vu=(center_v + margin_vu[0], center_u + margin_vu[1]),
    ).silhouette


def predicted_silhouette_bbox(
    nav_body: Mapping[str, Any],
    data_shape_vu: tuple[int, int],
    *,
    margin_vu: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of a body's predicted silhouette on the extended frame.

    The silhouette is drawn on the frame extended by the margins, so the cost is set
    by the frame and not by the body, and the box bounds exactly the part of the body
    that falls on the extended frame.  Where a body runs past the extended frame,
    the box stops at its edge and bounds only the part inside it.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame, which places a body that
            states no center.
        margin_vu: ``(margin_v, margin_u)`` added to the frame on each side.

    Returns:
        ``(v_min, v_max, u_min, u_max)``, the inclusive pixel-index bounds in frame
        coordinates (from ``-margin`` to ``size + margin - 1``), or None when the body
        covers no pixel of the extended frame.
    """
    vs, us = np.nonzero(predicted_silhouette(nav_body, data_shape_vu, margin_vu=margin_vu))
    if vs.size == 0:
        return None
    return (
        int(vs.min()) - margin_vu[0],
        int(vs.max()) - margin_vu[0],
        int(us.min()) - margin_vu[1],
        int(us.max()) - margin_vu[1],
    )
