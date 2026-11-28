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
lighting and viewing angles of the real body from its SPICE geometry through oops, at
each image pixel's center after moving the frame by its verified offset
(:func:`offset_backplane`), and passes the disc's I/F and the two cosines to
:func:`spindoctor.sim.realism.albedo.disc_normal_albedo`.  The disc leaves out the pixels
a nearer body, the planet's main rings, or the planet's shadow covers, as the disc fits
of the photometric-law study do.  Only a calibrated product is
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage
from oops import Meshgrid
from oops.backplane import Backplane
from polymath import Pair

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.sim.instruments.resolvers import (
    resolve_data_units,
    resolve_detector_defaults_native,
)
from spindoctor.sim.realism.albedo import disc_normal_albedo
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType
from tests.integration.sim_realism_scenes import MATCHED_BODY_LAW
from tests.integration.sim_realism_support import largest_body

__all__ = [
    'EDGE_CLEARANCE_PX',
    'MIN_CLIPPED_FRACTION',
    'RealBody',
    'disc_is_clipped',
    'offset_backplane',
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
    def model(self) -> str:
        """The body's model, whose limb gives the real frame's FOM 3 widths."""
        return f'body:{self.name}'

    @property
    def limb_peak_model(self) -> str | None:
        """The body model whose limb gives the real frame's gradient peak, or None.

        A clipped disc's limb rises to a plateau, so it gives none.
        """
        return None if self.disc_clipped else self.model


def offset_backplane(obs: Any, offset_vu: tuple[float, float]) -> Backplane:
    """A backplane of the frame's geometry at each image pixel's center, offset applied.

    A body the geometry predicts at ``(v, u)`` lies in the image at ``(v + dv, u + du)``,
    so image pixel ``(v, u)`` samples the geometry at ``(v - dv, u - du)``.  The offset
    is applied as it is, fraction and all.

    Parameters:
        obs: The real observation.
        offset_vu: The frame's verified offset ``(dv, du)``.

    Returns:
        A backplane in the image's shape.
    """
    rows, cols = np.shape(obs.data)
    image_v, image_u = np.indices((rows, cols), dtype=np.float64)
    uv = np.stack(
        [
            image_u + PIXEL_CENTER_TO_CORNER_PX - offset_vu[1],
            image_v + PIXEL_CENTER_TO_CORNER_PX - offset_vu[0],
        ],
        axis=-1,
    )
    return Backplane(obs, meshgrid=Meshgrid(obs.fov, Pair(uv)))


def _disc_pixels(
    backplane: Any, body_name: str, *, hidden: NDArrayBoolType | None = None
) -> tuple[NDArrayIntType, NDArrayIntType]:
    """The rows and columns of the body's interior pixels in the image.

    Parameters:
        backplane: The frame's offset backplane (:func:`offset_backplane`).
        body_name: The body, as SPICE knows it.
        hidden: Pixels to leave out, in the image's shape, or None.

    Returns:
        The rows and columns of each pixel at least :data:`EDGE_CLEARANCE_PX` erosion
        steps inside the body's silhouette and outside ``hidden``.
    """
    on_body = np.asarray(backplane.where_intercepted(body_name).vals, dtype=bool)
    interior = scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX)
    if hidden is not None:
        interior &= ~np.asarray(hidden, dtype=bool)
    rows, cols = np.nonzero(interior)
    return rows, cols


def disc_is_clipped(backplane: Any, body_name: str, *, clipped: NDArrayBoolType) -> bool:
    """Whether a body's disc reaches the camera's full scale.

    Parameters:
        backplane: The frame's offset backplane (:func:`offset_backplane`).
        body_name: The body, as SPICE knows it.
        clipped: Which of the image's pixels read the full scale, in the image's shape.

    Returns:
        True when at least 1% of the disc's interior pixels read the full scale; False
        when fewer do or none of the disc's interior lies on the image.
    """
    rows, cols = _disc_pixels(backplane, body_name)
    if rows.size == 0:
        return False
    fraction = float(np.mean(np.asarray(clipped, dtype=bool)[rows, cols]))
    return fraction >= MIN_CLIPPED_FRACTION


def real_normal_albedo(
    data: NDArrayFloatType,
    backplane: Any,
    body_name: str,
    *,
    phase_angle_deg: float,
    hidden: NDArrayBoolType | None = None,
) -> float | None:
    """The normal albedo of one body in a calibrated real frame.

    Parameters:
        data: The real frame's image, in I/F.
        backplane: The frame's offset backplane (:func:`offset_backplane`).
        body_name: The body, as SPICE knows it (``'DIONE'``).
        phase_angle_deg: The body's phase angle in degrees.
        hidden: The pixels of the disc that something else hides, which are left out
            (:func:`~tests.integration.sim_realism_laws_discs.hidden_pixels`), or None.

    Returns:
        The median normal albedo of the disc's interior pixels (see
        :data:`EDGE_CLEARANCE_PX`) that nothing hides, measured against the matched
        body's law, or None when the disc has too few such pixels (see
        :func:`~spindoctor.sim.realism.albedo.disc_normal_albedo`).
    """
    rows, cols = _disc_pixels(backplane, body_name, hidden=hidden)
    incidence = np.ma.filled(backplane.incidence_angle(body_name).mvals, np.pi)
    emission = np.ma.filled(backplane.emission_angle(body_name).mvals, np.pi)
    return disc_normal_albedo(
        np.asarray(data, dtype=np.float64)[rows, cols],
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
    backplane: Any,
    calibrated: bool,
    clipped: NDArrayBoolType | None,
    hidden: Callable[[str], NDArrayBoolType | None] | None = None,
) -> RealBody | None:
    """The facts a matched scene takes from a real frame's largest body.

    Parameters:
        obs: The real observation.
        features: The features the navigator extracted from the frame.
        model_metadata: The frame's per-model metadata, keyed by model name.
        backplane: The frame's offset backplane (:func:`offset_backplane`).
        calibrated: Whether the frame's data are I/F.
        clipped: Which of the image's pixels read the camera's full scale, or None when
            that is not known.
        hidden: Gives the pixels of a body's disc that something else hides, which the
            albedo leaves out, or None to leave none out.

    Returns:
        The largest body's facts, or None when no body model states both a diameter and
        a phase.
    """
    name, diameter, phase = largest_body(model_metadata)
    if name is None or diameter is None or phase is None:
        return None
    disc_clipped = clipped is not None and disc_is_clipped(backplane, name, clipped=clipped)
    albedo = None
    if calibrated and not disc_clipped:
        albedo = real_normal_albedo(
            obs.data,
            backplane,
            name,
            phase_angle_deg=phase,
            hidden=None if hidden is None else hidden(name),
        )
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
