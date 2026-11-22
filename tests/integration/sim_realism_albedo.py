"""Measure a real body's normal albedo, so its matched simulated body is as bright.

A simulated body is white unless its scene gives it a normal albedo.  A white body in
a matched limb frame, exposed as long as the real frame, can overflow the detector's
full well, and the overflowing charge spreads along the columns and moves the limb's
gradient peak outward.  The real body is not white, so the realism match sets the
matched body's normal albedo from the real disc.

:func:`real_normal_albedo` reads the lighting and viewing angles of each pixel of the
real body from its SPICE geometry through oops, moves them by the frame's verified
offset onto the image, and passes the disc's I/F and the two cosines to
:func:`spindoctor.sim.realism.albedo.disc_normal_albedo`.  Only a calibrated product
is in I/F, so a raw product gives no albedo and its matched body stays white.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.ndimage

from spindoctor.sim.realism.albedo import disc_normal_albedo
from tests.integration.sim_realism_scenes import MATCHED_BODY_LAW

__all__ = ['EDGE_CLEARANCE_PX', 'real_normal_albedo']

EDGE_CLEARANCE_PX = 3
"""How many pixels a disc pixel must lie inside the body's silhouette to count.

The optics blur the last pixel or two of the disc into the sky, and the verified offset
places the silhouette to about a pixel.
"""


def real_normal_albedo(
    obs: Any, body_name: str, *, offset_vu: tuple[float, float], phase_angle_deg: float
) -> float | None:
    """The normal albedo of one body in a calibrated real frame.

    Parameters:
        obs: The real observation, whose data are I/F.
        body_name: The body, as SPICE knows it (``'DIONE'``).
        offset_vu: The frame's verified offset: a body predicted at ``(v, u)`` lies at
            ``(v + dv, u + du)`` in the image.
        phase_angle_deg: The body's phase angle in degrees.

    Returns:
        The median normal albedo of the disc pixels at least 3 px inside the
        silhouette, measured against the matched body's law, or None when the disc
        has too few such pixels (see
        :func:`~spindoctor.sim.realism.albedo.disc_normal_albedo`).
    """
    backplane = obs.bp
    on_body = np.asarray(backplane.where_intercepted(body_name).vals, dtype=bool)
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX)
    incidence = np.ma.filled(backplane.incidence_angle(body_name).mvals, np.pi)
    emission = np.ma.filled(backplane.emission_angle(body_name).mvals, np.pi)
    data = np.asarray(obs.data, dtype=np.float64)
    shift_v = round(offset_vu[0])
    shift_u = round(offset_vu[1])
    rows, cols = np.nonzero(interior)
    image_rows = rows + shift_v
    image_cols = cols + shift_u
    on_image = (
        (image_rows >= 0)
        & (image_rows < data.shape[0])
        & (image_cols >= 0)
        & (image_cols < data.shape[1])
    )
    rows = rows[on_image]
    cols = cols[on_image]
    return disc_normal_albedo(
        data[image_rows[on_image], image_cols[on_image]],
        np.cos(np.asarray(incidence, dtype=np.float64)[rows, cols]),
        np.cos(np.asarray(emission, dtype=np.float64)[rows, cols]),
        law=MATCHED_BODY_LAW,
        phase_angle=math.radians(phase_angle_deg),
    )
