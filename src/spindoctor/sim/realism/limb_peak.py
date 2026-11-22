"""Where a limb's brightness gradient peaks, relative to the body's silhouette.

Across a body's limb the brightness falls from the disc to the sky.  The point along
the limb's outward normal where it falls fastest is the *gradient peak*, and the
navigator's limb technique locates the limb there.  Where a body's brightness rolls off
before its edge, as a Lambert disc does, the peak lies inside the silhouette.  Where the
disc stays bright to its edge, as an airless icy moon does at low phase, the peak lies
on it, blurred only by the optics and the pixels.

:func:`gradient_peak_positions` finds the peak along each vertex's normal, measured from
a point on the silhouette.  :func:`fit_limb_peak_offset` reduces one limb's peaks to a
single radial offset.  The fit also solves for a translation of the whole limb, so a
frame's pointing error, or an error in the body's predicted position, moves the
translation and leaves the radial offset alone.  The same two functions run on a real
frame and on its matched simulated frame, so the realism match compares like with like.

Coordinates are (v, u) in the image array's pixel-centric convention: integer
coordinates are pixel centers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'LimbPeakFit',
    'fit_limb_peak_offset',
    'gradient_peak_positions',
]

# Sampling along each normal: +-4 px from the silhouette, every 0.05 px.
_HALF_LENGTH_PX = 4.0
_SPACING_PX = 0.05
# The peak is looked for within this distance of the silhouette.  The plateaus that
# show which side of the limb is the disc are taken from beyond it.
_SEARCH_PX = 3.0

# The fewest vertices whose peaks the fit keeps.  Fewer leave the radial offset and the
# translation poorly separated.
MIN_FIT_VERTICES = 20
# Tukey biweight tuning constant, in units of the residuals' robust scale.
_TUKEY_C = 4.685
_TUKEY_ITERATIONS = 20
# Converts a median absolute deviation into a Gaussian standard deviation.
_MAD_TO_SIGMA = 1.4826
# Keeps the robust scale above zero for a limb whose peaks all agree exactly.
_MIN_SCALE_PX = 1e-6


@dataclass(frozen=True)
class LimbPeakFit:
    """One limb's gradient peak reduced to a radial offset from the silhouette.

    Parameters:
        offset_px: The gradient peak's distance outside the silhouette, in pixels. A
            negative value means the peak lies inside it.
        offset_sigma_px: The offset's standard error, from the scatter of the kept
            vertices' peaks about the fit.
        translation_vu: The translation of the whole limb the fit solved for, in
            pixels.
        n_measured: How many vertices had a gradient peak.
        n_kept: How many of those the robust fit kept.
        residual_scale_px: The robust scale of the kept peaks about the fit.
    """

    offset_px: float
    offset_sigma_px: float
    translation_vu: tuple[float, float]
    n_measured: int
    n_kept: int
    residual_scale_px: float


def _unit_normals(normals_vu: NDArrayFloatType) -> NDArrayFloatType:
    """Each normal scaled to unit length; a zero normal is left as it is."""
    norms = np.asarray(normals_vu, dtype=np.float64)
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return np.asarray(norms / lengths, dtype=np.float64)


def gradient_peak_positions(
    image: NDArrayFloatType,
    silhouette_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
) -> NDArrayFloatType:
    """The distance from the silhouette to the gradient peak along each outward normal.

    The image is sampled along each normal from 4 px inside the silhouette point to 4 px
    outside it, every 0.05 px, by cubic spline interpolation, so the slope is smooth
    between pixel centers.  The peak is the point within 3 px of the silhouette where
    the brightness falls fastest going outward, refined to a fraction of a sample by a
    parabola through the three samples around it.  A vertex gets no peak (NaN) when its
    samples leave the image, when the median brightness beyond 3 px inside does not
    exceed the median beyond 3 px outside, or when the steepest fall lies at the end of
    the search range.

    A sharp edge sampled by pixels and interpolated back is not quite symmetric, so a
    single vertex's peak moves with where the edge falls within its pixel: by up to
    0.08 px for an edge blurred by a Gaussian of 0.8 px, and 0.04 px at 1.2 px.  The
    error averages to zero over the edge's sub-pixel positions, which a limb's vertices
    cover evenly.

    Parameters:
        image: The 2-D frame.
        silhouette_vu: ``(N, 2)`` points on the body's silhouette, already moved by the
            frame's offset so that they lie on the limb in the image.
        normals_vu: ``(N, 2)`` outward normals at those points (scaled to unit length
            here).

    Returns:
        ``(N,)`` signed distances in pixels, positive outward, NaN where no peak was
        found.
    """
    arr = np.asarray(image, dtype=np.float64)
    points = np.asarray(silhouette_vu, dtype=np.float64)
    out = np.full(points.shape[0], np.nan)
    if points.shape[0] == 0:
        return out
    unit = _unit_normals(normals_vu)
    taps = np.arange(-_HALF_LENGTH_PX, _HALF_LENGTH_PX + _SPACING_PX / 2.0, _SPACING_PX)
    coords_v = points[:, 0:1] + unit[:, 0:1] * taps[np.newaxis, :]
    coords_u = points[:, 1:2] + unit[:, 1:2] * taps[np.newaxis, :]
    # The cubic spline reads two pixels beyond each sample, so a track stays one pixel
    # clear of the border, where the spline's mirrored extension would set the slope.
    in_image = (
        (coords_v.min(axis=1) >= 1.0)
        & (coords_u.min(axis=1) >= 1.0)
        & (coords_v.max(axis=1) <= arr.shape[0] - 2.0)
        & (coords_u.max(axis=1) <= arr.shape[1] - 2.0)
    )
    if not in_image.any():
        return out
    coeffs = scipy.ndimage.spline_filter(arr, order=3)
    profiles = scipy.ndimage.map_coordinates(
        coeffs,
        [coords_v[in_image].ravel(), coords_u[in_image].ravel()],
        order=3,
        prefilter=False,
    ).reshape(-1, taps.size)
    fall = -np.gradient(profiles, _SPACING_PX, axis=1)
    search = np.abs(taps) <= _SEARCH_PX + _SPACING_PX / 2.0
    inside = np.median(profiles[:, taps <= -_SEARCH_PX], axis=1)
    outside = np.median(profiles[:, taps >= _SEARCH_PX], axis=1)
    first = int(np.argmax(search))
    last = int(search.size - 1 - np.argmax(search[::-1]))
    peaks = np.full(profiles.shape[0], np.nan)
    for row in range(profiles.shape[0]):
        if not inside[row] > outside[row]:
            continue
        k = first + int(np.argmax(fall[row, first : last + 1]))
        if k in (first, last):
            continue
        y0, y1, y2 = fall[row, k - 1], fall[row, k], fall[row, k + 1]
        curvature = y0 - 2.0 * y1 + y2
        frac = 0.5 * (y0 - y2) / curvature if curvature < 0.0 else 0.0
        peaks[row] = taps[k] + frac * _SPACING_PX
    out[in_image] = peaks
    return out


def _normal_matrix(design: NDArrayFloatType, weights: NDArrayFloatType) -> NDArrayFloatType:
    """The weighted normal matrix of a least-squares design, summed column by column.

    The sums stand in for a matrix product, whose last bit depends on the CPU's BLAS
    kernel.
    """
    n_cols = design.shape[1]
    out = np.empty((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            out[i, j] = float(np.sum(weights * design[:, i] * design[:, j]))
    return out


def _weighted_sums(
    design: NDArrayFloatType, weights: NDArrayFloatType, values: NDArrayFloatType
) -> NDArrayFloatType:
    """The weighted right-hand side of the normal equations, column by column."""
    return np.asarray(
        [float(np.sum(weights * design[:, i] * values)) for i in range(design.shape[1])]
    )


def fit_limb_peak_offset(
    peaks_px: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
) -> LimbPeakFit | None:
    """Fit one limb's gradient peaks with a radial offset and a translation.

    Each vertex's peak distance ``x`` is modeled as ``d + t . n``: ``d`` is the radial
    offset shared by the whole limb, ``t`` is a translation of the limb, and ``n`` is
    the vertex's unit outward normal.  A translation moves a vertex's peak by its
    component along the normal, so on a limb whose normals turn through a wide angle,
    the two separate.  The fit is iteratively reweighted least squares with a Tukey
    biweight at 4.685 times the residuals' robust scale, which drops vertices on a
    crater rim or a cosmic-ray hit.

    Parameters:
        peaks_px: ``(N,)`` peak distances from :func:`gradient_peak_positions`; NaN
            entries are left out.
        normals_vu: ``(N, 2)`` outward normals at the same vertices.

    Returns:
        The fit, or None when fewer than 20 vertices have a peak or the fit keeps fewer
        than 20 of them.
    """
    x_all = np.asarray(peaks_px, dtype=np.float64)
    measured = np.isfinite(x_all)
    n_measured = int(np.count_nonzero(measured))
    if n_measured < MIN_FIT_VERTICES:
        return None
    x = x_all[measured]
    unit = _unit_normals(normals_vu)[measured]
    design = np.column_stack([np.ones(n_measured), unit])
    weights = np.ones(n_measured)
    coef: NDArrayFloatType = np.zeros(3)
    scale = _MIN_SCALE_PX
    for _ in range(_TUKEY_ITERATIONS):
        # A least-squares solve of the 3 x 3 normal equations stays finite when a short
        # arc leaves them nearly singular.
        solution = np.linalg.lstsq(
            _normal_matrix(design, weights), _weighted_sums(design, weights, x), rcond=None
        )
        coef = np.asarray(solution[0], dtype=np.float64)
        resid = x - np.sum(design * coef, axis=1)
        scale = max(_MAD_TO_SIGMA * float(np.median(np.abs(resid[weights > 0.0]))), _MIN_SCALE_PX)
        z = resid / (_TUKEY_C * scale)
        weights = np.where(np.abs(z) < 1.0, (1.0 - z**2) ** 2, 0.0)
    n_kept = int(np.count_nonzero(weights > 0.0))
    if n_kept < MIN_FIT_VERTICES:
        return None
    covariance = np.linalg.pinv(_normal_matrix(design, weights)) * scale**2
    return LimbPeakFit(
        offset_px=float(coef[0]),
        offset_sigma_px=float(np.sqrt(covariance[0, 0])),
        translation_vu=(float(coef[1]), float(coef[2])),
        n_measured=n_measured,
        n_kept=n_kept,
        residual_scale_px=scale,
    )
