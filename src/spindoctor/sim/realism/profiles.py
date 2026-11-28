"""Star radial profiles and edge-normal profiles for realism FOMs 2-4.

FOM 2 compares star-cutout radial profiles and encircled energy between
matched sim and real frames.  FOMs 3 and 4 compare intensity profiles
sampled perpendicular to a predicted polyline (a body limb or a ring edge)
after shifting the polyline by the frame's known offset; the profile's
10-90% rise width is the scalar each vertex contributes to the compared
distribution -- it is what a DT-based technique's gradient threshold
actually sees.

All sampling is bilinear (``scipy.ndimage.map_coordinates`` order 1) on
the caller's image array; coordinates follow the (v, u) convention.
"""

import numpy as np
import scipy.ndimage

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'edge_normal_profiles',
    'edge_tracks_in_image',
    'ee_radius',
    'encircled_energy',
    'profile_rise_width',
    'radial_profile',
]


def radial_profile(
    image: NDArrayFloatType,
    center_vu: tuple[float, float],
    *,
    r_max: float = 8.0,
    n_bins: int = 16,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Azimuthally averaged radial profile around a point source.

    The local background (median of the annulus just outside ``r_max``) is
    subtracted so profiles from frames with different pedestals compare.

    Parameters:
        image: 2-D frame.
        center_vu: Sub-pixel (v, u) center of the source.
        r_max: Outer radius of the profile in pixels.
        n_bins: Number of radial bins between 0 and ``r_max``.

    Returns:
        ``(radius, intensity)`` arrays of length ``n_bins``; empty bins
        carry NaN.  Returns all-NaN intensity when the cutout leaves the
        image.
    """
    arr = np.asarray(image, dtype=np.float64)
    cv, cu = float(center_vu[0]), float(center_vu[1])
    half = int(np.ceil(r_max)) + 3
    v0, v1 = int(np.floor(cv)) - half, int(np.floor(cv)) + half + 1
    u0, u1 = int(np.floor(cu)) - half, int(np.floor(cu)) + half + 1
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if v0 < 0 or u0 < 0 or v1 > arr.shape[0] or u1 > arr.shape[1]:
        return centers, np.full(n_bins, np.nan)
    cut = arr[v0:v1, u0:u1]
    vv, uu = np.meshgrid(
        np.arange(v0, v1, dtype=np.float64), np.arange(u0, u1, dtype=np.float64), indexing='ij'
    )
    radius = np.sqrt((vv - cv) ** 2 + (uu - cu) ** 2)
    background_ring = (radius > r_max) & (radius <= r_max + 3.0)
    background = float(np.median(cut[background_ring])) if np.any(background_ring) else 0.0
    net = cut - background
    intensity = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (radius >= edges[i]) & (radius < edges[i + 1])
        if np.any(in_bin):
            intensity[i] = float(net[in_bin].mean())
    return centers, intensity


def encircled_energy(
    radius: NDArrayFloatType, intensity: NDArrayFloatType
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Cumulative encircled energy from an azimuthally averaged profile.

    Each radial bin contributes ``intensity * 2 * pi * r * dr``; the
    cumulative sum is normalized to 1 at the outer radius.  Negative bin
    contributions (background over-subtraction) are clipped at zero before
    accumulation.

    Parameters:
        radius: Bin-center radii from :func:`radial_profile`.
        intensity: Matching mean intensities (NaN bins contribute zero).

    Returns:
        ``(radius, ee)`` where ``ee`` climbs from ~0 to 1; all-NaN when the
        profile carries no positive energy.
    """
    r = np.asarray(radius, dtype=np.float64)
    inten = np.nan_to_num(np.asarray(intensity, dtype=np.float64), nan=0.0)
    if r.size < 2:
        return r, np.full(r.size, np.nan)
    dr = np.diff(r, prepend=max(0.0, 2.0 * r[0] - r[1]))
    contributions = np.clip(inten, 0.0, None) * 2.0 * np.pi * r * dr
    total = float(contributions.sum())
    if total <= 0.0:
        return r, np.full(r.size, np.nan)
    return r, np.cumsum(contributions) / total


def ee_radius(radius: NDArrayFloatType, ee: NDArrayFloatType, fraction: float) -> float:
    """Radius at which the encircled energy first reaches ``fraction``.

    Parameters:
        radius: Radii from :func:`encircled_energy`.
        ee: Monotone encircled-energy curve.
        fraction: Energy fraction in (0, 1), e.g. 0.5 for EE50.

    Returns:
        Linear interpolation of the crossing radius; NaN when the curve
        never reaches ``fraction`` or is all-NaN.
    """
    r = np.asarray(radius, dtype=np.float64)
    e = np.asarray(ee, dtype=np.float64)
    if r.size == 0 or not np.all(np.isfinite(e)):
        return float('nan')
    above = np.nonzero(e >= fraction)[0]
    if above.size == 0:
        return float('nan')
    i = int(above[0])
    if i == 0:
        return float(r[0])
    frac = (fraction - e[i - 1]) / (e[i] - e[i - 1])
    return float(r[i - 1] + frac * (r[i] - r[i - 1]))


def edge_tracks_in_image(
    shape_vu: tuple[int, ...],
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    *,
    half_length_px: float = 8.0,
) -> NDArrayBoolType:
    """Which vertices' sampling tracks stay inside the image.

    Parameters:
        shape_vu: The image's shape.
        vertices_vu: ``(N, 2)`` vertex positions (v, u).
        normals_vu: ``(N, 2)`` outward normal per vertex (normalized here).
        half_length_px: Half-length of the sampling track in pixels.

    Returns:
        ``(N,)`` True where the track from ``half_length_px`` inside the vertex to as far
        outside it lies within the image's pixel centers.
    """
    verts = np.asarray(vertices_vu, dtype=np.float64)
    if verts.size == 0:
        return np.zeros(0, dtype=bool)
    norms = np.asarray(normals_vu, dtype=np.float64)
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    unit = norms / lengths
    ends = np.array([-half_length_px, half_length_px])
    coords_v = verts[:, 0:1] + unit[:, 0:1] * ends[np.newaxis, :]
    coords_u = verts[:, 1:2] + unit[:, 1:2] * ends[np.newaxis, :]
    return np.asarray(
        (coords_v.min(axis=1) >= 0.0)
        & (coords_v.max(axis=1) <= shape_vu[0] - 1.0)
        & (coords_u.min(axis=1) >= 0.0)
        & (coords_u.max(axis=1) <= shape_vu[1] - 1.0),
        dtype=bool,
    )


def edge_normal_profiles(
    image: NDArrayFloatType,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    *,
    half_length_px: float = 8.0,
    n_samples: int = 33,
) -> NDArrayFloatType:
    """Sample intensity profiles along each vertex's outward normal.

    Each row of the result is the image sampled at ``n_samples`` points
    from ``-half_length_px`` (inside the edge) to ``+half_length_px``
    (outside, along the outward normal) centered on the vertex.  Vertices
    whose sample track leaves the image (:func:`edge_tracks_in_image`) are
    dropped.

    Parameters:
        image: 2-D frame.
        vertices_vu: ``(N, 2)`` vertex positions (v, u) -- already shifted
            by the frame's known offset so they lie on the actual edge.
        normals_vu: ``(N, 2)`` outward normal per vertex (normalized here).
        half_length_px: Half-length of the sampling track in pixels.
        n_samples: Sample count along the track (odd keeps a center tap).

    Returns:
        ``(M, n_samples)`` array of sampled profiles, ``M <= N``.
    """
    arr = np.asarray(image, dtype=np.float64)
    verts = np.asarray(vertices_vu, dtype=np.float64)
    norms = np.asarray(normals_vu, dtype=np.float64)
    if verts.size == 0:
        return np.empty((0, n_samples))
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    unit = norms / lengths
    taps = np.linspace(-half_length_px, half_length_px, n_samples)
    # (N, n_samples) coordinates per axis.
    coords_v = verts[:, 0:1] + unit[:, 0:1] * taps[np.newaxis, :]
    coords_u = verts[:, 1:2] + unit[:, 1:2] * taps[np.newaxis, :]
    in_bounds = edge_tracks_in_image(arr.shape, verts, norms, half_length_px=half_length_px)
    coords_v = coords_v[in_bounds]
    coords_u = coords_u[in_bounds]
    if coords_v.size == 0:
        return np.empty((0, n_samples))
    sampled = scipy.ndimage.map_coordinates(
        arr, np.stack([coords_v.ravel(), coords_u.ravel()]), order=1, mode='nearest'
    )
    return np.asarray(sampled.reshape(coords_v.shape), dtype=np.float64)


def profile_rise_width(
    profile: NDArrayFloatType, *, spacing_px: float, lo: float = 0.1, hi: float = 0.9
) -> float:
    """10-90% rise width of one edge profile, in pixels.

    The profile is normalized between its inside plateau (median of the
    first quarter of samples) and outside plateau (median of the last
    quarter); the width is the distance between the outermost ``hi``
    crossing and the innermost ``lo`` crossing of the normalized descent.
    Works for either polarity (bright-inside limbs and bright-outside
    ring gaps) by orienting on the plateau difference.

    Parameters:
        profile: 1-D sampled profile (inside first, outside last).
        spacing_px: Pixel distance between adjacent samples.
        lo: Lower normalized level of the rise measurement.
        hi: Upper normalized level of the rise measurement.

    Returns:
        The rise width in pixels; NaN when the plateaus do not separate
        (no measurable edge) or the crossings cannot be bracketed.
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = prof.size
    if n < 8 or not np.all(np.isfinite(prof)):
        return float('nan')
    quarter = max(2, n // 4)
    inside = float(np.median(prof[:quarter]))
    outside = float(np.median(prof[-quarter:]))
    span = inside - outside
    scale = max(abs(inside), abs(outside), 1e-300)
    if abs(span) < 1e-6 * scale:
        return float('nan')
    # Normalize so the curve descends from ~1 (inside) to ~0 (outside).
    norm = (prof - outside) / span
    # Last index still at or above hi before the first dip at or below lo,
    # then the first at-or-below-lo index after it.
    above = np.nonzero(norm >= hi)[0]
    below = np.nonzero(norm <= lo)[0]
    if above.size == 0 or below.size == 0:
        return float('nan')
    first_below = int(below[0])
    candidates = above[above < first_below]
    if candidates.size == 0:
        return float('nan')
    i_hi = int(candidates[-1])
    later_below = below[below > i_hi]
    if later_below.size == 0:
        return float('nan')
    i_lo = int(later_below[0])
    if i_lo <= i_hi:
        return float('nan')
    # Sub-sample the two crossings linearly to beat the tap quantization.
    x_hi = float(i_hi)
    if norm[i_hi + 1] < hi:
        x_hi += (norm[i_hi] - hi) / (norm[i_hi] - norm[i_hi + 1])
    x_lo = float(i_lo)
    if norm[i_lo - 1] > lo:
        x_lo -= (lo - norm[i_lo]) / (norm[i_lo - 1] - norm[i_lo])
    if x_lo <= x_hi:
        return float('nan')
    return float((x_lo - x_hi) * spacing_px)
