"""The planted camera roll: turning the directions a scene states.

A planted roll is a rotation of the camera frame about the boresight.  Everything in
the field of view turns by the roll angle in one direction about one point: body
positions and orientations, the direction a body is lit from, stars, star smear,
companions, the background sky, and the ring system.  The scene states all of it as
the unrolled camera would see it, the navigator predicts that unrolled scene, and the
renderer draws it rolled.  This module turns a body about its own center, turns any
image-plane vector, and gives the translation that follows the turn; the scene
radiance stage turns body and ring centers about the frame center, and the star
renderer and the optics smear turn the rest.

The roll angle runs from ``+v`` toward ``+u``: a positive roll carries a point at
``(dv, du)`` from the pivot to ``(cos * dv - sin * du, sin * dv + cos * du)``.  This
is the rotation the navigator's star similarity fit reports, so the rotation it
recovers equals the planted roll.

The pivot is the boresight: the frame center, ``(size_v / 2, size_u / 2)`` in pixel
corner coordinates, after the planted offset has moved the camera.  The offset ``d``
is the boresight's pointing error, so a point the scene states at ``x`` from the
frame center lands at ``R (x + d)`` from it, ``R`` being the roll.  The renderer
writes that as ``R x + R d``: it turns the scene about the frame center and then
moves everything by the offset turned by the roll, which :func:`boresight_shift`
gives.  The radiance stage takes that shift once and moves every star, body, and
ring system by it, so no object can apply the offset and the roll in another order.
The planted offset itself stays the truth for the navigator's translation.

Two of a body's angles run the other way.  ``rotation_z`` turns the grid an ellipsoid
is sampled on, so the drawn outline turns by ``-rotation_z``, and
``illumination_angle`` is measured from the top of the image toward ``+u``, so the
light direction ``(-cos, sin)`` turns by ``-illumination_angle``.  A roll therefore
subtracts from both.  A mesh body's pose is ``Rz @ Ry @ Rx`` in camera axes ordered
``(u, v, depth)``, so the line-of-sight angle is the outermost one and a roll
subtracts from it too.

The camera's own optics stay where they are: the PSF, the distortion, the ghosts,
and the radial stray-light bump belong to the detector rather than the field of
view.  Motion smear is the scene drifting across the detector, so it turns, and a
linear stray-light ramp rises toward a bright source beyond the field of view, so
its direction turns too.

Only the image side reads this module.  The roll is a truth key, so the navigator
never learns it.
"""

import copy
import math
from collections.abc import Mapping
from typing import Any

from spindoctor.sim.scene.fields import scene_value

__all__ = ['boresight_shift', 'roll_body_params', 'turn_vector']


def turn_vector(dv: float, du: float, *, roll_deg: float) -> tuple[float, float]:
    """Turn an image-plane vector by the roll.

    Parameters:
        dv: The vector's v component.
        du: The vector's u component.
        roll_deg: The roll in degrees, from ``+v`` toward ``+u``.

    Returns:
        The turned vector ``(dv, du)``.
    """
    roll = math.radians(roll_deg)
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    return cos_r * dv - sin_r * du, sin_r * dv + cos_r * du


def boresight_shift(offset_v: float, offset_u: float, *, roll_deg: float) -> tuple[float, float]:
    """The translation that follows the roll's turn about the frame center.

    The roll turns the frame about the boresight, which the planted offset ``d`` has
    moved, so a point at ``x`` from the frame center lands at ``R (x + d)``.  That is
    the point turned about the frame center, ``R x``, moved by ``R d``, which this
    returns.  With no roll it is the offset itself, and with no offset it is zero.

    Parameters:
        offset_v: The planted offset's v component.
        offset_u: The planted offset's u component.
        roll_deg: The roll in degrees, from ``+v`` toward ``+u``.

    Returns:
        The shift ``(dv, du)``, in the offset's units.
    """
    return turn_vector(offset_v, offset_u, roll_deg=roll_deg)


def _turned_offset_block(block: Any, *, roll_deg: float) -> Any:
    """A transit ``moon`` or ``shadow`` block with its offset from the body turned.

    Parameters:
        block: The block, or None.
        roll_deg: The roll in degrees.

    Returns:
        A copy with ``dv_px`` and ``du_px`` turned, or the input unchanged when it is
        not a mapping.
    """
    if not isinstance(block, Mapping):
        return block
    turned = dict(block)
    dv, du = turn_vector(
        float(scene_value(block, 'dv_px', 0.0)),
        float(scene_value(block, 'du_px', 0.0)),
        roll_deg=roll_deg,
    )
    turned['dv_px'] = dv
    turned['du_px'] = du
    return turned


def _turned_atmosphere(atmosphere: Any, *, roll_deg: float) -> Any:
    """An ``atmosphere`` block with each cloud's offset from the body center turned.

    The haze's other directions follow the body's orientation and light, which the
    caller turns.

    Parameters:
        atmosphere: The block, or None.
        roll_deg: The roll in degrees.

    Returns:
        A copy with each ``cloud_blobs`` entry's ``center_vu`` turned, or the input
        unchanged when it carries no clouds.
    """
    if not isinstance(atmosphere, Mapping) or not atmosphere.get('cloud_blobs'):
        return atmosphere
    turned = copy.deepcopy(dict(atmosphere))
    for blob in turned['cloud_blobs']:
        center = scene_value(blob, 'center_vu', (0.0, 0.0))
        blob['center_vu'] = list(turn_vector(float(center[0]), float(center[1]), roll_deg=roll_deg))
    return turned


def roll_body_params(body_params: Mapping[str, Any], *, roll_deg: float) -> dict[str, Any]:
    """A scene body turned by the roll about its own center.

    The orientation (``rotation_z`` for an ellipsoid, the line-of-sight angle of
    ``pose_euler_deg`` for a mesh) and ``illumination_angle`` take the roll with the
    sign each runs in, and a transit's moon and shadow and an atmosphere's clouds,
    which the scene places by their offset from the body center, have that offset
    turned.  A key the scene leaves out turns from its default.
    The body's relief, albedo texture, and disc texture are fixed to its surface and
    turn with its orientation.  Its craters are drawn on the image grid, so the
    renderer turns their layout itself (see
    :class:`~spindoctor.sim.render.bodies.spec.BodySpec`).  The body's center is
    left alone: it turns about the frame center, which the caller handles.

    Parameters:
        body_params: One validated scene body.
        roll_deg: The roll in degrees, from ``+v`` toward ``+u``.

    Returns:
        A new body mapping; the input is only read.
    """
    rolled = dict(body_params)
    if scene_value(body_params, 'shape_model', 'ellipsoid') == 'polyhedral_mesh':
        pose = scene_value(body_params, 'pose_euler_deg', (0.0, 0.0, 0.0))
        rolled['pose_euler_deg'] = [float(pose[0]), float(pose[1]), float(pose[2]) - roll_deg]
    else:
        rolled['rotation_z'] = float(scene_value(body_params, 'rotation_z', 0.0)) - roll_deg
    rolled['illumination_angle'] = (
        float(scene_value(body_params, 'illumination_angle', 0.0)) - roll_deg
    )
    transits = body_params.get('transits')
    if isinstance(transits, list):
        rolled['transits'] = [
            {key: _turned_offset_block(value, roll_deg=roll_deg) for key, value in entry.items()}
            for entry in transits
        ]
    if 'atmosphere' in body_params:
        rolled['atmosphere'] = _turned_atmosphere(body_params['atmosphere'], roll_deg=roll_deg)
    return rolled
