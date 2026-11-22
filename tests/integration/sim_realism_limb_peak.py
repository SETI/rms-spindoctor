"""The limb gradient-peak statistic of the realism match.

Figure of merit 3 compares how wide a limb's rise is.  Two limbs can rise over the same
width and still peak at different radii, and the navigator's limb technique locates a
limb at its gradient peak.  This module adds the peak's position: for each limb the
navigator's feature extraction places, the distance from the body's true silhouette to
the point where the brightness falls fastest, reduced to one radial offset per limb by
:func:`spindoctor.sim.realism.limb_peak.fit_limb_peak_offset`.

The silhouette comes from each side's own geometry.  On a real frame it is where the
rays along each vertex's normal stop hitting the body, from the SPICE geometry through
oops (:func:`real_silhouette_offsets`).  On a matched frame it is the circle the scene
states (:func:`matched_silhouette_offsets`).  The fit solves for a translation of the
whole limb beside the radial offset, so the frame's offset error, the body's ephemeris
error, and the half pixel between two coordinate conventions all move the translation
and leave the radial offset alone.

Each limb's offset is a sample of kind ``limb_peak_<bin>``, binned like the limb widths.
Each frame measures one limb, that of the body its matched scene renders, and
:func:`pair_limb_peaks` keeps a frame pair's two offsets only when both frames measured
it.  :func:`compare_limb_peaks` then sets each bin's band from the real offsets alone
and says whether the matched frames' median lies within it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from oops import Meshgrid
from oops.backplane import Backplane
from polymath import Pair

from spindoctor.feature.geometry import LimbPolyline
from spindoctor.sim.realism.limb_peak import fit_limb_peak_offset, gradient_peak_positions
from spindoctor.support.constants import HALFPI, PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'LIMB_PEAK_KIND_PREFIX',
    'LimbPeakBin',
    'SilhouetteOffsets',
    'compare_limb_peaks',
    'limb_peak_offset',
    'matched_silhouette_offsets',
    'pair_limb_peaks',
    'real_silhouette_offsets',
    'record_has_limb_peaks',
    'record_limb_peak_problems',
]

LIMB_PEAK_KIND_PREFIX = 'limb_peak_'
"""The prefix of the per-bin limb gradient-peak sample kinds."""

# A function that takes a body model's name, the (N, 2) limb vertices in extended-FOV
# pixel-centric coordinates, and their (N, 2) outward normals, and returns the (N,)
# signed distances from each vertex to the silhouette along its normal, positive
# outward and NaN where the silhouette was not found.
SilhouetteOffsets = Callable[[str, NDArrayFloatType, NDArrayFloatType], NDArrayFloatType]

# The search for a real silhouette: from 3 px inside each vertex to 3 px outside it,
# every 1/64 px, which leaves an error far below the peak's own scatter.
_SILHOUETTE_REACH_PX = 3.0
_SILHOUETTE_STEP_PX = 1.0 / 64.0

# A limb whose offset is known to worse than this is left out: its arc is too short to
# tell a radial offset from a translation.
MAX_OFFSET_SIGMA_PX = 0.2

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


def real_silhouette_offsets(obs: Any) -> SilhouetteOffsets:
    """The silhouette of a real frame's bodies from its SPICE geometry.

    Along each vertex's normal, points from 3 px inside the vertex to 3 px outside it
    are traced through oops every 1/64 px.  The silhouette lies halfway between the last
    point whose ray hits the body and the first that misses it.  A vertex gets NaN when
    the track does not cross the silhouette exactly once, or when the Sun is below the
    horizon at the last point that hits the body: only the lit limb has an edge to
    measure.

    Parameters:
        obs: The real observation the body models were built from.

    Returns:
        A :data:`SilhouetteOffsets` for the frame.  The body is the part of the model's
        name after its colon (``body:DIONE`` gives ``DIONE``).
    """
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
        backplane = Backplane(obs, meshgrid=Meshgrid(obs.fov, Pair(uv)))
        hits = np.asarray(backplane.where_intercepted(body).vals, dtype=bool).reshape(track_v.shape)
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
    circle has an edge to measure.  At a limb point whose outward normal is ``n``, the
    cosine of the solar incidence is ``sin(phase)`` times the component of ``n`` along
    the direction the light comes from in the image (``illumination_angle`` 0 is the
    top of the image, 90 the right).  A vertex whose normal gives a cosine of 0 or less
    gets NaN.

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
    sin_phase = float(np.sin(np.radians(float(body['phase_angle']))))

    def offsets(
        _model_name: str, vertices_vu: NDArrayFloatType, normals_vu: NDArrayFloatType
    ) -> NDArrayFloatType:
        unit = normals_vu / np.linalg.norm(normals_vu, axis=1, keepdims=True)
        rel = vertices_vu - center
        along = np.sum(rel * unit, axis=1)
        discriminant = along**2 - (np.sum(rel * rel, axis=1) - radius**2)
        lit = sin_phase * np.sum(unit * toward_light_vu, axis=1) > 0.0
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
) -> float | None:
    """One limb's gradient-peak offset from the silhouette, in pixels.

    Parameters:
        image: The sensor-area image.
        geometry: The limb's polyline at its predicted position.
        model_name: The body model that emitted the limb.
        shift_vu: What moves a predicted extended-FOV position onto ``image`` at the
            frame's known offset: the offset less the extended-FOV margins.
        silhouette_offsets: The frame's silhouette.

    Returns:
        The offset (negative inside the silhouette), or None when the fit fails or knows
        the offset to worse than 0.2 px.
    """
    vertices = np.asarray(geometry.vertices_vu, dtype=np.float64)
    normals = np.asarray(geometry.normals_vu, dtype=np.float64)
    if vertices.shape[0] == 0:
        return None
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    to_silhouette = silhouette_offsets(model_name, vertices, normals)
    found = np.isfinite(to_silhouette)
    peaks = np.full(vertices.shape[0], np.nan)
    silhouette = vertices[found] + unit[found] * to_silhouette[found, np.newaxis] + shift_vu
    peaks[found] = gradient_peak_positions(image, silhouette, normals[found])
    fit = fit_limb_peak_offset(peaks, normals)
    if fit is None or fit.offset_sigma_px > MAX_OFFSET_SIGMA_PX:
        return None
    return fit.offset_px


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


def record_limb_peak_problems(record: dict[str, Any]) -> list[str]:
    """The bins of a realism record whose matched median lies outside its band.

    Parameters:
        record: A realism summary, as
            :func:`tests.integration.realism_regression.load_record` reads it.

    Returns:
        One line per bin that lies outside its band, giving the instrument, the bin, both
        medians, the band, and both counts.  Bins without a verdict are left out.
    """
    problems: list[str] = []
    for instrument, entry in sorted(record.get('instruments', {}).items()):
        for label, peak in sorted(entry.get('limb_peaks', {}).items()):
            if peak['within_band'] is not False:
                continue
            problems.append(
                f'{instrument} {label}: matched median {peak["sim_median_px"]:+.3f} px '
                f'({_limbs(peak["n_sim"])}) lies more than {peak["band_px"]:.3f} px from the '
                f'real median {peak["real_median_px"]:+.3f} px ({_limbs(peak["n_real"])})'
            )
    return problems


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
