"""What the realism match takes from the largest body of a real frame.

A matched limb frame, and a matched ring frame that holds a body, renders one sphere at
the apparent size and phase of the real frame's largest body.  This module reads three
more facts about that body from the real frame.

**Whether its disc is clipped.**  A disc that reaches the camera's full scale reads one
flat level over its brightest part, so its I/F is the clip level and its limb rises to a
plateau.  :func:`disc_is_clipped` calls a disc clipped when at least 1% of its interior
pixels read the full scale.  On the Cassini NAC cohort the clipped discs have 51% to 93%
of their interior pixels at the full scale and the rest have at most 0.007%, so the
choice of 1% does not decide any frame.  Which pixels read the full scale comes from the
raw product: on a raw-DN cohort it is the frame itself, and on a calibrated cohort it is
the raw product a mission's frame reader opens beside the calibrated one
(:class:`~tests.integration.sim_realism_support.FrameSettings`).  A calibrated cohort
whose mission has no such reader is taken as unclipped.

**Its normal albedo.**  A simulated body is white unless its scene gives it a normal
albedo.  A white body in a matched frame, exposed as long as the real frame, can
overflow the detector's full well, and the overflowing charge spreads along the columns
and moves the limb's gradient peak outward.  :func:`real_normal_albedo` reads the
lighting and viewing angles of each pixel of the real body from its SPICE geometry
through oops, moves them by the frame's verified offset onto the image, and passes the
disc's I/F and the two cosines to
:func:`spindoctor.sim.realism.albedo.disc_normal_albedo`.  Only a calibrated product is
in I/F, and a clipped disc's I/F is the clip level, so a raw product or a clipped disc
gives no albedo and the matched body stays white.

**Whether the navigator places its limb.**  The navigator emits no limb for a resolved
body it treats as highly irregular, such as Prometheus, or for one whose shape lies too
far from its ellipsoid, such as Pan.  The matched body is a sphere, whose limb the
navigator always places, so the matched frame's limb is measured only when the real
frame's is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.sim.instruments.resolvers import (
    resolve_data_units,
    resolve_detector_defaults_native,
)
from spindoctor.sim.realism.albedo import disc_normal_albedo
from spindoctor.support.types import NDArrayBoolType, NDArrayIntType
from tests.integration.sim_realism_scenes import MATCHED_BODY_LAW
from tests.integration.sim_realism_support import largest_body

__all__ = [
    'EDGE_CLEARANCE_PX',
    'MIN_CLIPPED_FRACTION',
    'RealBody',
    'disc_is_clipped',
    'real_body',
    'real_normal_albedo',
    'saturation_level',
]

EDGE_CLEARANCE_PX = 3
"""How many steps of erosion the body's mask takes before its pixels count as disc.

The erosion steps to the four neighbors of a pixel, so a pixel that counts lies at least
3.5 px inside the silhouette along a row or a column and about 2.1 px inside it along a
diagonal.  The optics blur the last pixel or two of the disc into the sky, and the
verified offset places the silhouette to about a pixel.
"""

MIN_CLIPPED_FRACTION = 0.01
"""The fraction of a disc's interior pixels at the full scale that makes it clipped."""


def saturation_level(instrument: str) -> float:
    """The full-scale level in cohort units, from the instrument entry.

    The level feeds the saturation statistics, the FOM 2 saturated-star filter, and,
    on a raw-DN cohort, the test of a clipped disc.

    Parameters:
        instrument: The sim instrument a cohort is matched against.

    Returns:
        The entry's ``detector.saturation_dn`` for a ``raw_dn`` product; NaN for a
        calibrated one, whose I/F units define no fixed level.
    """
    if resolve_data_units(instrument) != 'raw_dn':
        return math.nan
    return resolve_detector_defaults_native(instrument).saturation_dn


@dataclass(frozen=True)
class RealBody:
    """The facts a matched scene takes from a real frame's largest body.

    Parameters:
        name: The body, as SPICE knows it (``'DIONE'``).
        diameter_px: Its predicted apparent diameter in pixels.
        phase_angle_deg: Its phase angle in degrees.
        normal_albedo: Its normal albedo, measured from the disc of a calibrated frame
            whose disc is not clipped, else None.
        disc_clipped: Whether its disc reaches the camera's full scale.
        limb_emitted: Whether the navigator placed its limb in the real frame.
    """

    name: str
    diameter_px: float
    phase_angle_deg: float
    normal_albedo: float | None
    disc_clipped: bool
    limb_emitted: bool

    @property
    def limb_peak_model(self) -> str | None:
        """The body model whose limb gives the real frame's gradient peak, or None.

        A clipped disc's limb rises to a plateau, so it gives none.
        """
        return None if self.disc_clipped else f'body:{self.name}'


def _disc_pixels(
    obs: Any, body_name: str, offset_vu: tuple[float, float]
) -> tuple[NDArrayIntType, NDArrayIntType, NDArrayIntType, NDArrayIntType]:
    """The body's interior pixels that lie on the image, before and after the offset.

    Parameters:
        obs: The real observation.
        body_name: The body, as SPICE knows it.
        offset_vu: The frame's verified offset, rounded to whole pixels here.

    Returns:
        The rows and columns of each pixel in the backplanes, and the rows and columns
        of the same pixels in the image.
    """
    on_body = np.asarray(obs.bp.where_intercepted(body_name).vals, dtype=bool)
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX)
    shape = np.shape(obs.data)
    rows, cols = np.nonzero(interior)
    image_rows = rows + round(offset_vu[0])
    image_cols = cols + round(offset_vu[1])
    on_image = (
        (image_rows >= 0) & (image_rows < shape[0]) & (image_cols >= 0) & (image_cols < shape[1])
    )
    return rows[on_image], cols[on_image], image_rows[on_image], image_cols[on_image]


def disc_is_clipped(
    obs: Any, body_name: str, *, offset_vu: tuple[float, float], clipped: NDArrayBoolType
) -> bool:
    """Whether a body's disc reaches the camera's full scale.

    Parameters:
        obs: The real observation.
        body_name: The body, as SPICE knows it.
        offset_vu: The frame's verified offset: a body predicted at ``(v, u)`` lies at
            ``(v + dv, u + du)`` in the image.
        clipped: Which of the image's pixels read the full scale, in the image's shape.

    Returns:
        True when at least 1% of the disc's interior pixels on the image read the full
        scale; False when fewer do or none of the disc's interior lies on the image.
    """
    _rows, _cols, image_rows, image_cols = _disc_pixels(obs, body_name, offset_vu)
    if image_rows.size == 0:
        return False
    fraction = float(np.mean(np.asarray(clipped, dtype=bool)[image_rows, image_cols]))
    return fraction >= MIN_CLIPPED_FRACTION


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
        The median normal albedo of the disc's interior pixels (see
        :data:`EDGE_CLEARANCE_PX`), measured against the matched body's law, or None
        when the disc has too few such pixels (see
        :func:`~spindoctor.sim.realism.albedo.disc_normal_albedo`).
    """
    rows, cols, image_rows, image_cols = _disc_pixels(obs, body_name, offset_vu)
    backplane = obs.bp
    incidence = np.ma.filled(backplane.incidence_angle(body_name).mvals, np.pi)
    emission = np.ma.filled(backplane.emission_angle(body_name).mvals, np.pi)
    data = np.asarray(obs.data, dtype=np.float64)
    return disc_normal_albedo(
        data[image_rows, image_cols],
        np.cos(np.asarray(incidence, dtype=np.float64)[rows, cols]),
        np.cos(np.asarray(emission, dtype=np.float64)[rows, cols]),
        law=MATCHED_BODY_LAW,
        phase_angle=math.radians(phase_angle_deg),
    )


def real_body(
    obs: Any,
    features: list[NavFeature],
    model_metadata: dict[str, dict[str, Any]],
    *,
    offset_vu: tuple[float, float],
    calibrated: bool,
    clipped: NDArrayBoolType | None,
) -> RealBody | None:
    """The facts a matched scene takes from a real frame's largest body.

    Parameters:
        obs: The real observation.
        features: The features the navigator extracted from the frame.
        model_metadata: The frame's per-model metadata, keyed by model name.
        offset_vu: The frame's verified offset.
        calibrated: Whether the frame's data are I/F.
        clipped: Which of the image's pixels read the camera's full scale, or None when
            that is not known.

    Returns:
        The largest body's facts, or None when no body model states both a diameter and
        a phase.
    """
    name, diameter, phase = largest_body(model_metadata)
    if name is None or diameter is None or phase is None:
        return None
    disc_clipped = clipped is not None and disc_is_clipped(
        obs, name, offset_vu=offset_vu, clipped=clipped
    )
    albedo = None
    if calibrated and not disc_clipped:
        albedo = real_normal_albedo(obs, name, offset_vu=offset_vu, phase_angle_deg=phase)
    model = f'body:{name}'
    limb_emitted = any(
        feature.source_model == model and isinstance(feature.geometry, LimbPolyline)
        for feature in features
    )
    return RealBody(
        name=name,
        diameter_px=diameter,
        phase_angle_deg=phase,
        normal_albedo=albedo,
        disc_clipped=disc_clipped,
        limb_emitted=limb_emitted,
    )
