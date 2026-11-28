"""The limb gradient-peak statistic of the realism match.

Figure of merit 3 compares how wide a limb's rise is.  Two limbs can rise over the same
width and still peak at different radii, and the navigator's limb technique locates a
limb at its gradient peak.  This module adds the peak's position: for each limb the
navigator's feature extraction places, the distance from the body's true silhouette to
the point where the brightness falls fastest, reduced to one radial offset per limb by
:func:`spindoctor.sim.realism.limb_peak.fit_limb_peak_offset`.

The silhouette comes from each side's own geometry.  On a real frame it is where the
rays along each vertex's track stop hitting the body, from the SPICE geometry through
oops (:func:`real_silhouette_offsets`), with the body radii the mission's frame reader
gives where they differ from the kernels'.  On a matched frame it is the circle the scene
states (:func:`matched_silhouette_offsets`).  The fit solves for a translation of the
whole limb beside the radial offset, so the frame's offset error, the body's ephemeris
error, and the half pixel between two coordinate conventions all move the translation
and leave the radial offset alone.

Every limb statistic reads a vertex by its direction from the body's center
(:class:`LimbOrientation`).  The gradient peak is looked for along that direction, the
silhouette is found along it, the fit takes it as the vertex's normal, and it decides
whether the vertex is lit and which vertices of a matched limb match a real one.  The
navigator's own normals point only in steps of 45 degrees, so they would make the
statistic depend on how the limb is turned in the image.

A real limb vertex is used only when nothing but its own body and the sky lies along
its track.  :func:`tracks_touching` tests each track against a mask of the other bodies,
the planet, the planet's rings, and its shadow over the body, which the real frame's
geometry gives (:func:`~tests.integration.sim_realism_laws_discs.foreign_pixels`).  The
matched frame's sphere is lit from the real body's direction and has nothing beside it,
so its limb is measured over the same stretch of limb instead: a matched vertex is used
only near the directions of the real vertices the real fit kept, each measured from the
direction the light comes from.  A fit whose used vertices span too short an arc is
refused, because over a short arc a translation of the limb mimics a radial offset
(:data:`MAX_OFFSET_TRANSLATION_CORRELATION`).

Each limb's offset is a sample of kind ``limb_peak_<bin>``, binned like the limb widths.
Each frame measures one limb, that of the body its matched scene renders, and
:func:`pair_limb_peaks` keeps a frame pair's two offsets only when both frames measured
it.  :func:`limb_peak_refusals` lists why a frame measured none.
:func:`compare_limb_peaks` then sets each bin's band from the real offsets alone and says
whether the matched frames' median lies within it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from oops import Body, Meshgrid
from oops.backplane import Backplane
from oops.surface.ellipsoid import Ellipsoid
from polymath import Pair

from spindoctor.feature.geometry import LimbPolyline
from spindoctor.sim.realism.limb_peak import (
    MIN_FIT_VERTICES,
    fit_limb_peak_offset,
    gradient_peak_positions,
)
from spindoctor.support.constants import HALFPI, PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'LIMB_PEAK_KIND_PREFIX',
    'MAX_OFFSET_SIGMA_PX',
    'MAX_OFFSET_TRANSLATION_CORRELATION',
    'PEAK_TRACK_HALF_LENGTH_PX',
    'LimbOrientation',
    'LimbPeak',
    'LimbPeakBin',
    'LimbSelection',
    'SilhouetteOffsets',
    'TrackTest',
    'compare_limb_peaks',
    'directions_near',
    'limb_peak_offset',
    'limb_peak_refusals',
    'matched_orientation',
    'matched_silhouette_offsets',
    'pair_limb_peaks',
    'real_silhouette_offsets',
    'record_has_limb_peaks',
    'record_limb_peak_failures',
    'tracks_touching',
]

LIMB_PEAK_KIND_PREFIX = 'limb_peak_'
"""The prefix of the per-bin limb gradient-peak sample kinds."""

# A function that takes a body model's name, the (N, 2) limb vertices in extended-FOV
# pixel-centric coordinates, and their (N, 2) outward normals, and returns the (N,)
# signed distances from each vertex to the silhouette along its normal, positive
# outward and NaN where the silhouette was not found.
SilhouetteOffsets = Callable[[str, NDArrayFloatType, NDArrayFloatType], NDArrayFloatType]

PEAK_TRACK_HALF_LENGTH_PX = 4.0
"""How far to each side of the silhouette the gradient peak's track reaches, in pixels."""

# The search for a real silhouette: from 3 px inside each vertex to 3 px outside it,
# every 1/64 px, which leaves an error far below the peak's own scatter.
_SILHOUETTE_REACH_PX = 3.0
_SILHOUETTE_STEP_PX = 1.0 / 64.0

MAX_OFFSET_SIGMA_PX = 0.2
"""A limb whose offset is known to worse than this, in pixels, is left out."""

MAX_OFFSET_TRANSLATION_CORRELATION = 0.97
"""The highest correlation of the offset with the translation that a kept limb may have.

The correlation is the fit's ``offset_translation_correlation``, with each vertex's
direction from the body's center as its normal, so it depends only on which directions
the kept vertices cover.  Over a short arc a translation moves every peak almost as the
offset does, so a small error that varies along the limb, such as light another body
spills onto part of it, is magnified into the offset: its variance grows by
``1 / (1 - R**2)``.  The lit half of a limb is a half circle, which reads 0.90 when
evenly sampled.  On Dione in N1487595731_1_CALIB, which passes in front of Rhea, the
whole limb reads 0.91, and 0.88 with the 84 vertices backed by Rhea's lit crescent left
out; Rhea's limb in the same frame reads 0.95, and 0.96 with the vertices near Dione left
out.  With every vertex whose track crosses Rhea left out, Dione keeps 43 vertices on a
46 degree arc, which read 0.9997, and the fit gives +2.0 px at a sigma of 0.40 px, which
the sigma limit refuses.  The ceiling sits above every limb of that frame that keeps
most of its lit half, and an evenly sampled arc reaches it at about 135 degrees, three
quarters of a lit half: a limb that loses more than about a quarter of its lit half, to
the frame's edge or to another body, is refused.  The variance grows by 17 at the
ceiling, against 5.3 for an evenly sampled half and 12 for the largest limb of that
frame.
"""

# A matched vertex is used when a real vertex the real fit kept lies within this many
# of the real limb's median vertex spacings of its direction, so a gap the rule left in
# the real limb stays a gap in the matched one.
_DIRECTION_MATCH_SPACINGS = 2.0

# The step along a track at which the mask of other bodies is read.
_TRACK_STEP_PX = 0.25

# The band rule.  Only a bin holding at least MIN_BIN_LIMBS_FOR_SCATTER real limbs gets
# a band and a verdict.  Its half-width, around the real median, is BAND_MULTIPLE
# standard errors of the mean: the bin's own robust scatter of the real offsets over the
# square root of their count, and never less than BAND_FLOOR_PX.  With fewer real limbs
# one limb that the body's shape pulls off sets the median (operator ruling of
# 2026-11-22, #842), so such a bin is reported without a verdict.
BAND_MULTIPLE = 2.0
BAND_FLOOR_PX = 0.1
MIN_BIN_LIMBS_FOR_SCATTER = 3
_MAD_TO_SIGMA = 1.4826


# A function that takes ``(N, 2)`` points in the image, their ``(N, 2)`` outward normals,
# and a half-length in pixels, and returns ``(N,)`` True where the track from that far
# inside each point to that far outside it touches nothing but its own body and the sky.
TrackTest = Callable[[NDArrayFloatType, NDArrayFloatType, float], NDArrayBoolType]


def tracks_touching(
    mask: NDArrayBoolType,
    points_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    half_length_px: float,
) -> NDArrayBoolType:
    """Which tracks along the normals touch a pixel of ``mask``.

    Each track runs from ``half_length_px`` inside its point to as far outside it and is
    read every 0.25 px at the nearest pixel.  Samples that leave the image read nothing.

    Parameters:
        mask: The pixels to look for, in the image's shape.
        points_vu: ``(N, 2)`` points in the image's pixel-centric coordinates.
        normals_vu: ``(N, 2)`` outward normals at those points.
        half_length_px: How far the track reaches to each side of its point.

    Returns:
        ``(N,)`` True where any sample of the track lands on ``mask``.
    """
    points = np.asarray(points_vu, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    normals = np.asarray(normals_vu, dtype=np.float64)
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    steps = np.arange(-half_length_px, half_length_px + _TRACK_STEP_PX / 2.0, _TRACK_STEP_PX)
    rows = np.rint(points[:, 0:1] + unit[:, 0:1] * steps[np.newaxis, :]).astype(np.int64)
    cols = np.rint(points[:, 1:2] + unit[:, 1:2] * steps[np.newaxis, :]).astype(np.int64)
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    hits = np.zeros(rows.shape, dtype=bool)
    hits[inside] = np.asarray(mask, dtype=bool)[rows[inside], cols[inside]]
    return np.asarray(hits.any(axis=1), dtype=bool)


@dataclass(frozen=True)
class LimbOrientation:
    """Where a body lies in the image and where its light comes from.

    Every limb statistic reads a vertex by its direction from the center: the gradient
    peak's track runs along it, the fit takes it as the vertex's normal, and a matched
    vertex is matched to a real one by it.  The navigator's own normals point only in
    steps of 45 degrees, so a statistic read along them would change as the limb turns
    in the image.

    Parameters:
        center_vu: The body's center in the image's pixel-centric coordinates.
        light_deg: The in-image direction the light comes from, as a scene's
            ``illumination_angle`` gives it: 0 from the top of the image, 90 from the
            right.
    """

    center_vu: tuple[float, float]
    light_deg: float

    def directions_deg(self, points_vu: NDArrayFloatType) -> NDArrayFloatType:
        """Each point's direction from the center, in degrees from the light's.

        Parameters:
            points_vu: ``(N, 2)`` points in the image's pixel-centric coordinates.

        Returns:
            ``(N,)`` angles in ``[-180, 180)``, measured the way ``light_deg`` is.
        """
        points = np.asarray(points_vu, dtype=np.float64)
        rel_v = points[:, 0] - self.center_vu[0]
        rel_u = points[:, 1] - self.center_vu[1]
        angle = np.degrees(np.arctan2(rel_u, -rel_v)) - self.light_deg
        return np.asarray((angle + 180.0) % 360.0 - 180.0, dtype=np.float64)

    def outward(self, points_vu: NDArrayFloatType) -> NDArrayFloatType:
        """The unit vector from the center toward each point.

        Parameters:
            points_vu: ``(N, 2)`` points in the image's pixel-centric coordinates.

        Returns:
            ``(N, 2)`` unit vectors; a point on the center gives a zero vector.
        """
        rel = np.asarray(points_vu, dtype=np.float64) - np.asarray(self.center_vu)
        lengths = np.linalg.norm(rel, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        return np.asarray(rel / lengths, dtype=np.float64)


def matched_orientation(scene: Mapping[str, Any]) -> LimbOrientation:
    """The orientation of a matched scene's one body.

    The scene states its center as a pixel-corner position, half a pixel above the
    image's pixel-centric one on each axis.

    Parameters:
        scene: The matched scene.

    Returns:
        The body's orientation.
    """
    body = scene['bodies'][0]
    return LimbOrientation(
        center_vu=(
            float(body['center_v']) - PIXEL_CENTER_TO_CORNER_PX,
            float(body['center_u']) - PIXEL_CENTER_TO_CORNER_PX,
        ),
        light_deg=float(body['illumination_angle']),
    )


def directions_near(
    directions_deg: NDArrayFloatType, used_deg: tuple[float, ...]
) -> NDArrayBoolType:
    """Which directions lie near the directions a real limb used.

    A direction is near when a used direction lies within two of the used directions'
    median spacings of it, so the matched limb leaves out the same stretches the real
    limb did.  The widest gap around the circle is the stretch the real limb left
    out, so the median is taken over the other spacings.

    Parameters:
        directions_deg: ``(N,)`` directions to test, in degrees.
        used_deg: The directions of the real vertices a real measurement used, in
            degrees.

    Returns:
        ``(N,)`` True where a direction lies near a used one; all False when fewer than
        two directions were used.
    """
    query = np.asarray(directions_deg, dtype=np.float64)
    if len(used_deg) < 2:
        return np.zeros(query.shape, dtype=bool)
    used = np.sort(np.asarray(used_deg, dtype=np.float64) % 360.0)
    spacing = np.sort(np.diff(np.concatenate([used, used[:1] + 360.0])))[:-1]
    tolerance = _DIRECTION_MATCH_SPACINGS * float(np.median(spacing))
    gap = np.abs(((query[:, np.newaxis] - used[np.newaxis, :]) + 180.0) % 360.0 - 180.0)
    return np.asarray(gap.min(axis=1) <= tolerance, dtype=bool)


@dataclass(frozen=True)
class LimbSelection:
    """Which vertices of a frame's measured limb the limb statistics may use.

    Parameters:
        orientation: The body's orientation, which gives each vertex its direction.
        clear_track: On a real frame, the test that a vertex's track touches only its
            own body and the sky; a vertex that fails it is left out.  None on a matched
            frame.
        peak_deg: On a matched frame, the directions of the real vertices the real
            gradient-peak fit kept; a vertex far from all of them gives no peak.  None on
            a real frame.
        width_deg: On a matched frame, the directions of the real vertices that gave
            rise widths; a vertex far from all of them gives no width.  None on a real
            frame.
    """

    orientation: LimbOrientation
    clear_track: TrackTest | None = None
    peak_deg: tuple[float, ...] | None = None
    width_deg: tuple[float, ...] | None = None

    def keeps(
        self,
        points_vu: NDArrayFloatType,
        normals_vu: NDArrayFloatType,
        half_length_px: float,
        near_deg: tuple[float, ...] | None,
    ) -> NDArrayBoolType:
        """Which points the selection keeps.

        A point is kept when its track passes :attr:`clear_track` and, when
        ``near_deg`` is given, its direction lies near one of those directions
        (:func:`directions_near`).

        Parameters:
            points_vu: ``(N, 2)`` points in the image's pixel-centric coordinates.
            normals_vu: ``(N, 2)`` directions of their tracks.
            half_length_px: How far each track reaches to each side of its point.
            near_deg: The directions a kept point must lie near, or None to keep any.

        Returns:
            ``(N,)`` True where the point is kept.
        """
        keep = np.ones(np.shape(points_vu)[0], dtype=bool)
        if self.clear_track is not None:
            keep &= self.clear_track(points_vu, normals_vu, half_length_px)
        if near_deg is not None:
            keep &= directions_near(self.orientation.directions_deg(points_vu), near_deg)
        return keep


@dataclass(frozen=True)
class LimbPeak:
    """One limb's gradient-peak measurement.

    Parameters:
        offset_px: The offset from the silhouette, negative inside it, or None when the
            limb was refused.
        refusal: Why the limb was refused, or None when it gave an offset.
        used_deg: The directions of the vertices whose peaks the fit kept, or of every
            vertex that had a peak when the fit failed (see :class:`LimbOrientation`).
    """

    offset_px: float | None
    refusal: str | None
    used_deg: tuple[float, ...] = ()


@contextmanager
def _body_radii(body_name: str, radii_km: tuple[float, float, float] | None) -> Iterator[None]:
    """Give a body an ellipsoid of other radii for as long as the block runs.

    The ellipsoid keeps the body's center and frame.  oops caches each body's surface for
    its backplanes, so the cache is cleared when the surface changes and again when the
    body's own surface comes back.

    Parameters:
        body_name: The body, as oops knows it.
        radii_km: The ellipsoid's three radii in km, longest first, or None to leave the
            body as it is.
    """
    if radii_km is None:
        yield
        return
    # oops types neither the surface's center and frame nor the backplane's surface cache.
    body: Any = Body.lookup(body_name)
    surface_cache: Any = Backplane.get_surface
    ellipsoid: Any = Ellipsoid
    saved = (body.surface, body.radius, body.inner_radius)
    body.apply_surface(
        ellipsoid(body.surface.origin, body.surface.frame, radii_km), radii_km[0], radii_km[2]
    )
    surface_cache.cache_clear()
    try:
        yield
    finally:
        body.apply_surface(*saved)
        surface_cache.cache_clear()


def real_silhouette_offsets(
    obs: Any, *, radii_km: Mapping[str, tuple[float, float, float]] | None = None
) -> SilhouetteOffsets:
    """The silhouette of a real frame's bodies from its SPICE geometry.

    Along each vertex's normal, points from 3 px inside the vertex to 3 px outside it
    are traced through oops every 1/64 px.  The silhouette lies halfway between the last
    point whose ray hits the body and the first that misses it.  A vertex gets NaN when
    the track does not cross the silhouette exactly once, or when the Sun is below the
    horizon at the last point that hits the body: only the lit limb has an edge to
    measure.

    A body listed in ``radii_km`` is traced as an ellipsoid of those radii, with the
    body's own center and frame, in place of the shape the frame's kernels give it.  The
    body's shape is restored afterward, so nothing else the frame computes sees it.

    Parameters:
        obs: The real observation the body models were built from.
        radii_km: Radii in km, longest first, keyed by body name, for the bodies whose
            silhouette takes other radii than the kernels give.

    Returns:
        A :data:`SilhouetteOffsets` for the frame.  The body is the part of the model's
        name after its colon (``body:DIONE`` gives ``DIONE``).
    """
    radii_by_body = dict(radii_km or {})
    margin_v, margin_u = obs.extfov_margin_vu
    steps = np.arange(
        -_SILHOUETTE_REACH_PX, _SILHOUETTE_REACH_PX + _SILHOUETTE_STEP_PX / 2.0, _SILHOUETTE_STEP_PX
    )

    def offsets(
        model_name: str, vertices_vu: NDArrayFloatType, normals_vu: NDArrayFloatType
    ) -> NDArrayFloatType:
        body = model_name.split(':', 1)[1]
        unit = normals_vu / np.linalg.norm(normals_vu, axis=1, keepdims=True)
        track_v = vertices_vu[:, 0:1] + unit[:, 0:1] * steps[np.newaxis, :]
        track_u = vertices_vu[:, 1:2] + unit[:, 1:2] * steps[np.newaxis, :]
        # Extended-FOV pixel centers to the geometry layer's pixel-corner coordinates.
        uv = np.stack(
            [
                (track_u - margin_u + PIXEL_CENTER_TO_CORNER_PX).ravel(),
                (track_v - margin_v + PIXEL_CENTER_TO_CORNER_PX).ravel(),
            ],
            axis=-1,
        )
        with _body_radii(body, radii_by_body.get(body)):
            backplane = Backplane(obs, meshgrid=Meshgrid(obs.fov, Pair(uv)))
            hits = np.asarray(backplane.where_intercepted(body).vals, dtype=bool).reshape(
                track_v.shape
            )
            incidence = backplane.incidence_angle(body).mvals
        incidence_rad = np.ma.filled(incidence, np.pi).reshape(track_v.shape)
        out = np.full(vertices_vu.shape[0], np.nan)
        leaves = hits[:, :-1] & ~hits[:, 1:]
        enters = ~hits[:, :-1] & hits[:, 1:]
        crossing = (leaves.sum(axis=1) == 1) & ~enters.any(axis=1)
        last_hit = np.argmax(leaves, axis=1)
        rows = np.arange(vertices_vu.shape[0])
        lit = incidence_rad[rows, last_hit] < HALFPI
        keep = crossing & lit
        out[keep] = steps[last_hit[keep]] + _SILHOUETTE_STEP_PX / 2.0
        return out

    return offsets


def matched_silhouette_offsets(scene: dict[str, Any], obs: Any) -> SilhouetteOffsets:
    """The silhouette of a matched frame's spherical body, from its scene.

    A matched limb scene holds one sphere.  Its center is a pixel-corner position, so
    its pixel-centric center lies half a pixel lower on each axis, and its silhouette is
    the circle of radius ``semi_a_px`` around that center.  Only the lit half of the
    circle has an edge to measure.  At the limb point in the direction ``r`` from the
    center, the cosine of the solar incidence is ``sin(phase)`` times the component of
    ``r`` along the direction the light comes from in the image (``illumination_angle``
    0 is the top of the image, 90 the right).  A vertex whose direction gives a cosine
    of 0 or less gets NaN, whatever the normal it is traced along.  At zero phase the
    light comes from behind the camera and lights the whole limb, so every vertex keeps
    its silhouette.

    Parameters:
        scene: The matched scene the frame was rendered from.
        obs: The matched observation, for its extended-FOV margins.

    Returns:
        A :data:`SilhouetteOffsets` for the frame; it ignores the model's name.

    Raises:
        ValueError: If the scene does not hold exactly one body, or its body is not a
            sphere.
    """
    bodies = scene.get('bodies', [])
    if len(bodies) != 1:
        raise ValueError(f'a matched limb scene holds one body, not {len(bodies)}')
    body = bodies[0]
    radius = float(body['semi_a_px'])
    if float(body['semi_b_px']) != radius or float(body['semi_c_px']) != radius:
        raise ValueError(f'the matched body {body["name"]!r} is not a sphere')
    margin_v, margin_u = obs.extfov_margin_vu
    center = np.array(
        [
            float(body['center_v']) - PIXEL_CENTER_TO_CORNER_PX + margin_v,
            float(body['center_u']) - PIXEL_CENTER_TO_CORNER_PX + margin_u,
        ]
    )
    light_angle = np.radians(float(body['illumination_angle']))
    toward_light_vu = np.array([-np.cos(light_angle), np.sin(light_angle)])
    phase_deg = float(body['phase_angle'])
    sin_phase = float(np.sin(np.radians(phase_deg)))

    def offsets(
        _model_name: str, vertices_vu: NDArrayFloatType, normals_vu: NDArrayFloatType
    ) -> NDArrayFloatType:
        unit = normals_vu / np.linalg.norm(normals_vu, axis=1, keepdims=True)
        rel = vertices_vu - center
        along = np.sum(rel * unit, axis=1)
        discriminant = along**2 - (np.sum(rel * rel, axis=1) - radius**2)
        lit = (phase_deg == 0.0) | (sin_phase * np.sum(rel * toward_light_vu, axis=1) > 0.0)
        out = np.full(vertices_vu.shape[0], np.nan)
        keep = (discriminant >= 0.0) & lit
        out[keep] = -along[keep] + np.sqrt(discriminant[keep])
        return out

    return offsets


def limb_peak_offset(
    image: NDArrayFloatType,
    geometry: LimbPolyline,
    model_name: str,
    *,
    shift_vu: NDArrayFloatType,
    silhouette_offsets: SilhouetteOffsets,
    selection: LimbSelection,
) -> LimbPeak:
    """One limb's gradient-peak offset from the silhouette, in pixels.

    Each vertex is read along its direction from the body's center
    (:meth:`LimbOrientation.outward`): the silhouette is found along it, the gradient
    peak is looked for along it, and the fit takes it as the vertex's normal.  A vertex
    is used when its silhouette was found and ``selection`` keeps it.  On a real frame
    the track the gradient peak is looked for along, 4 px to each side of the
    silhouette, must touch only the body and the sky.  On a matched frame the vertex
    must lie near a direction whose peak the real fit kept.

    The arc is judged on the real limb alone, and a matched limb is measured over the
    real limb's directions only after the real limb kept its offset.

    Parameters:
        image: The sensor-area image.
        geometry: The limb's polyline at its predicted position.
        model_name: The body model that emitted the limb.
        shift_vu: What moves a predicted extended-FOV position onto ``image`` at the
            frame's known offset: the offset less the extended-FOV margins.
        silhouette_offsets: The frame's silhouette.
        selection: Which vertices may be used, and the body's orientation.

    Returns:
        The measurement.  The limb is refused when fewer than 20 vertices have a peak,
        when the fit fails, or when it knows the offset to worse than 0.2 px; a real limb
        also when the offset's correlation with the translation exceeds the ceiling.
    """
    vertices = np.asarray(geometry.vertices_vu, dtype=np.float64)
    if vertices.shape[0] == 0:
        return LimbPeak(None, 'the limb has no vertices')
    outward = selection.orientation.outward(vertices + shift_vu)
    to_silhouette = silhouette_offsets(model_name, vertices, outward)
    found = np.isfinite(to_silhouette)
    silhouette = vertices + outward * np.where(found, to_silhouette, 0.0)[:, np.newaxis] + shift_vu
    found &= selection.keeps(silhouette, outward, PEAK_TRACK_HALF_LENGTH_PX, selection.peak_deg)
    peaks = np.full(vertices.shape[0], np.nan)
    peaks[found] = gradient_peak_positions(image, silhouette[found], outward[found])
    measured = np.isfinite(peaks)
    directions = selection.orientation.directions_deg(silhouette[measured])
    used = tuple(float(d) for d in directions)
    n_measured = int(np.count_nonzero(measured))
    if n_measured < MIN_FIT_VERTICES:
        return LimbPeak(None, f'only {n_measured} vertices have a gradient peak', used)
    fit = fit_limb_peak_offset(peaks, outward)
    if fit is None:
        return LimbPeak(None, 'the robust fit kept too few vertices to separate the offset', used)
    # A matched limb is measured over the directions whose peaks the fit kept, so a
    # stretch the robust fit dropped on the real limb is left out of it as well.
    used = tuple(float(d) for d in selection.orientation.directions_deg(silhouette[fit.kept]))
    if fit.offset_sigma_px > MAX_OFFSET_SIGMA_PX:
        return LimbPeak(
            None, f'the offset is known to {fit.offset_sigma_px:.2f} px, worse than 0.2 px', used
        )
    judges_arc = selection.peak_deg is None
    if judges_arc and fit.offset_translation_correlation > MAX_OFFSET_TRANSLATION_CORRELATION:
        return LimbPeak(
            None,
            f'the arc is too short: the offset correlates with the translation at '
            f'{fit.offset_translation_correlation:.3f}, above '
            f'{MAX_OFFSET_TRANSLATION_CORRELATION}',
            used,
        )
    return LimbPeak(fit.offset_px, None, used)


def limb_peak_refusals(
    image_id: str, body: str | None, real: LimbPeak | None, sim: LimbPeak | None
) -> list[dict[str, str]]:
    """Why each frame of a pair gave no limb gradient peak, for the realism record.

    Parameters:
        image_id: The real frame's image ID.
        body: The body whose limb the pair measured.
        real: The real frame's measurement, or None when it measured no limb.
        sim: The matched frame's, likewise.

    Returns:
        One entry per refused measurement, giving the image ID, the body, the side
        (``'real'`` or ``'matched'``), and the reason.
    """
    return [
        {'image_id': image_id, 'body': body or '', 'side': side, 'reason': peak.refusal}
        for side, peak in (('real', real), ('matched', sim))
        if peak is not None and peak.refusal is not None
    ]


def pair_limb_peaks(real: dict[str, list[float]], sim: dict[str, list[float]]) -> None:
    """Keep a frame pair's gradient-peak offsets only when both frames measured the limb.

    Each frame measures the limb of one body, the one its matched scene renders.  The
    real frame gives none when the navigator places no limb for that body or its disc is
    clipped, and the matched frame gives none when its limb is not measured.  Unless
    each frame holds exactly one offset and both offsets fall in the same bin, both
    frames' offsets are removed.

    Parameters:
        real: The real frame's samples, keyed by kind; its ``limb_peak_*`` kinds may be
            removed.
        sim: The matched frame's samples, changed the same way.
    """
    real_kinds = [k for k in real if k.startswith(LIMB_PEAK_KIND_PREFIX)]
    sim_kinds = [k for k in sim if k.startswith(LIMB_PEAK_KIND_PREFIX)]
    if (
        len(real_kinds) == 1
        and real_kinds == sim_kinds
        and len(real[real_kinds[0]]) == 1
        and len(sim[sim_kinds[0]]) == 1
    ):
        return
    for kind in real_kinds:
        del real[kind]
    for kind in sim_kinds:
        del sim[kind]


@dataclass(frozen=True)
class LimbPeakBin:
    """One limb bin's gradient-peak comparison.

    Parameters:
        n_real: Real limbs in the bin.
        n_sim: Matched limbs in the bin.
        real_median_px: The real offsets' median, or None without real limbs.
        sim_median_px: The matched offsets' median, or None without matched limbs.
        real_scatter_px: The robust scatter of the bin's real offsets, or None when
            the bin holds fewer than 3 real limbs.
        band_px: The band's half-width around the real median, or None when the bin
            holds fewer than 3 real limbs.
        within_band: Whether the matched median lies within the band, or None when
            the bin has no band or no matched limbs.
    """

    n_real: int
    n_sim: int
    real_median_px: float | None
    sim_median_px: float | None
    real_scatter_px: float | None
    band_px: float | None
    within_band: bool | None


def _robust_scatter(values: NDArrayFloatType) -> float:
    """The scaled median absolute deviation of ``values``."""
    return float(_MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def compare_limb_peaks(
    real: dict[str, list[float]], sim: dict[str, list[float]]
) -> dict[str, LimbPeakBin]:
    """Compare each bin's matched gradient-peak offsets with its real ones.

    The band is set from the real offsets before the matched ones are looked at.  Only
    a bin that holds at least 3 real limbs gets one.  It is centered on the real median,
    and its half-width is 2 standard errors of the mean: the robust scatter of the bin's
    own real offsets over the square root of their count, and never less than 0.1 px.
    The robust scatter stands in for the standard deviation.  A bin with fewer real
    limbs records both medians and no band or verdict.

    Parameters:
        real: The real side's samples, keyed by kind; only ``limb_peak_*`` kinds are
            read.
        sim: The matched side's samples, keyed the same way.

    Returns:
        One comparison per bin that either side populates, keyed by the bin's label.
    """
    kinds = sorted(k for k in set(real) | set(sim) if k.startswith(LIMB_PEAK_KIND_PREFIX))
    out: dict[str, LimbPeakBin] = {}
    for kind in kinds:
        real_values = np.asarray(real.get(kind, []), dtype=np.float64)
        sim_values = np.asarray(sim.get(kind, []), dtype=np.float64)
        real_median = float(np.median(real_values)) if real_values.size else None
        sim_median = float(np.median(sim_values)) if sim_values.size else None
        scatter: float | None = None
        band: float | None = None
        if real_values.size >= MIN_BIN_LIMBS_FOR_SCATTER:
            scatter = _robust_scatter(real_values)
            band = max(BAND_MULTIPLE * scatter / float(np.sqrt(real_values.size)), BAND_FLOOR_PX)
        within: bool | None = None
        if band is not None and real_median is not None and sim_median is not None:
            within = bool(abs(sim_median - real_median) <= band)
        out[kind[len(LIMB_PEAK_KIND_PREFIX) :]] = LimbPeakBin(
            n_real=int(real_values.size),
            n_sim=int(sim_values.size),
            real_median_px=real_median,
            sim_median_px=sim_median,
            real_scatter_px=scatter,
            band_px=band,
            within_band=within,
        )
    return out


def record_limb_peak_failures(record: dict[str, Any]) -> dict[tuple[str, str], str]:
    """The bins of a realism record whose matched median lies outside its band.

    Parameters:
        record: A realism summary, as
            :func:`tests.integration.realism_regression.load_record` reads it.

    Returns:
        For each bin outside its band, keyed by ``(instrument, bin)``, a line that gives
        the instrument, the bin, both medians, the band, and both counts.  Bins without a
        verdict are left out.
    """
    failures: dict[tuple[str, str], str] = {}
    for instrument, entry in sorted(record.get('instruments', {}).items()):
        for label, peak in sorted(entry.get('limb_peaks', {}).items()):
            if peak['within_band'] is not False:
                continue
            failures[(instrument, label)] = (
                f'{instrument} {label}: matched median {peak["sim_median_px"]:+.3f} px '
                f'({_limbs(peak["n_sim"])}) lies more than {peak["band_px"]:.3f} px from the '
                f'real median {peak["real_median_px"]:+.3f} px ({_limbs(peak["n_real"])})'
            )
    return failures


def _limbs(count: int) -> str:
    """``'1 limb'`` or ``'<count> limbs'``."""
    return '1 limb' if count == 1 else f'{count} limbs'


def record_has_limb_peaks(record: dict[str, Any]) -> bool:
    """Whether a realism record holds the limb gradient-peak comparison.

    Parameters:
        record: A realism summary.

    Returns:
        True when any instrument's entry carries a ``limb_peaks`` block.
    """
    return any('limb_peaks' in entry for entry in record.get('instruments', {}).values())
