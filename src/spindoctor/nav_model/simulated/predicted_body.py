"""The navigator's predicted silhouette of a simulated body.

Every navigator-side model that draws a simulated body -- the body model's template and its
occluders, and the haze model's occluders and contaminant boxes -- draws it here, from the
body's entry in the navigator view (``obs.nav_params['bodies']``).  The entry carries only
idealized keys, since the boundary filter strips the truth before this code sees it.  Three
rules apply in this one place, so the models cannot draw the same body differently:

- a body that states no center sits at the frame center, as the image-side renderer puts it;
- a body that states no ``semi_c_px`` takes the shorter of its two image-plane semi-axes;
- a ``polyhedral_mesh`` body is drawn as its mesh through
  :func:`~spindoctor.sim.shared.mesh.render_mesh_body_image`, and any other body as an
  ellipsoid through :func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid` and
  :func:`~spindoctor.sim.shared.ellipsoid.render_lambert_ellipsoid`.

The rendered brightness is above zero over the whole visible disc, because the shading floors
the unlit side above zero, so ``image > 0`` is the body's full silhouette, lit or not.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.sim.shared.ellipsoid import project_ellipsoid, render_lambert_ellipsoid
from spindoctor.sim.shared.mesh import mesh_spec_from_params, render_mesh_body_image
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'predicted_center_vu',
    'predicted_semi_axes_px',
    'predicted_silhouette',
    'predicted_silhouette_bbox',
    'render_predicted_body',
]

# Reach of the canvas a whole silhouette is rendered on, as a multiple of the longest
# semi-axis: half again the axis, because mesh relief can push past the ellipsoid bound.
_SILHOUETTE_REACH_FACTOR = 1.5
# Pixels added to that reach on every side, so a silhouette never touches the canvas edge.
_SILHOUETTE_CANVAS_PAD_PX = 8


def predicted_center_vu(
    nav_body: Mapping[str, Any], data_shape_vu: tuple[int, int]
) -> tuple[float, float]:
    """Return a body's predicted center on the frame, in pixel-corner coordinates.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame.

    Returns:
        ``(center_v, center_u)``: the stated center, or the frame center for an axis the
        entry does not state.
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
        ``(semi_a_px, semi_b_px, semi_c_px)``; ``semi_c_px`` is the shorter of the other two
        when the entry does not state it.
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
) -> NDArrayFloatType:
    """Render a body's predicted image on a canvas.

    The body is drawn at full anti-aliasing, with the shape, pose, and lighting its entry
    states.

    Parameters:
        nav_body: The body's entry in the navigator view.
        phase_angle_rad: Phase angle of the render in radians.
        shape_vu: ``(rows, columns)`` of the canvas.
        center_vu: The body center on the canvas, in pixel-corner coordinates.  A render on
            the frame passes :func:`predicted_center_vu`.

    Returns:
        The rendered image, brightness in ``[0, 1]``.
    """
    semi_axes_px = predicted_semi_axes_px(nav_body)
    illumination_angle_rad = math.radians(float(nav_body.get('illumination_angle', 0.0)))
    if str(nav_body.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh':
        return render_mesh_body_image(
            size=shape_vu,
            center=center_vu,
            semi_axes_px=semi_axes_px,
            spec=mesh_spec_from_params(nav_body),
            illumination_angle=illumination_angle_rad,
            phase_angle=phase_angle_rad,
            anti_aliasing=1.0,
        )
    projection = project_ellipsoid(
        shape_vu,
        center_vu,
        semi_axes_px[0],
        semi_b_px=semi_axes_px[1],
        semi_c_px=semi_axes_px[2],
        rotation_z=math.radians(float(nav_body.get('rotation_z', 0.0))),
        rotation_tilt=math.radians(float(nav_body.get('rotation_tilt', 0.0))),
        anti_aliasing=1.0,
    )
    return render_lambert_ellipsoid(
        projection, illumination_angle=illumination_angle_rad, phase_angle=phase_angle_rad
    )


def predicted_silhouette(
    nav_body: Mapping[str, Any], data_shape_vu: tuple[int, int]
) -> NDArrayBoolType:
    """Return a body's predicted silhouette on the frame.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame.

    Returns:
        The boolean mask of the frame pixels the body covers, rendered at its own phase.
    """
    image = render_predicted_body(
        nav_body,
        math.radians(float(nav_body.get('phase_angle', 0.0))),
        shape_vu=data_shape_vu,
        center_vu=predicted_center_vu(nav_body, data_shape_vu),
    )
    mask: NDArrayBoolType = np.asarray(image > 0.0, dtype=bool)
    return mask


def predicted_silhouette_bbox(
    nav_body: Mapping[str, Any], data_shape_vu: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of a body's whole predicted silhouette on the frame.

    The silhouette is rendered on a canvas around the body, with the body's center on the
    same sub-pixel position it has on the frame, so the box is that of the frame render and
    reaches past the frame edges where the body does.

    Parameters:
        nav_body: The body's entry in the navigator view.
        data_shape_vu: ``(rows, columns)`` of the frame, which places a body that states no
            center.

    Returns:
        ``(v_min, v_max, u_min, u_max)``, the inclusive pixel-index bounds on the frame, or
        None for a silhouette too small to cover a pixel.
    """
    center_v, center_u = predicted_center_vu(nav_body, data_shape_vu)
    reach = math.ceil(_SILHOUETTE_REACH_FACTOR * max(predicted_semi_axes_px(nav_body)))
    half = reach + _SILHOUETTE_CANVAS_PAD_PX
    origin_v = math.floor(center_v) - half
    origin_u = math.floor(center_u) - half
    side = 2 * half + 1
    image = render_predicted_body(
        nav_body,
        math.radians(float(nav_body.get('phase_angle', 0.0))),
        shape_vu=(side, side),
        center_vu=(center_v - origin_v, center_u - origin_u),
    )
    vs, us = np.nonzero(image > 0.0)
    if vs.size == 0:
        return None
    return (
        origin_v + int(vs.min()),
        origin_v + int(vs.max()),
        origin_u + int(us.min()),
        origin_u + int(us.max()),
    )
