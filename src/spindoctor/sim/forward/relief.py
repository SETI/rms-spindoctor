"""Truth-side body relief: the limb/terminator topography field.

The relief is a 2-D field on the body surface, not a 1-D field on the limb,
so the limb perturbation and the terminator shadowing are slices of one
consistent surface.  ``h(lat, lon)`` is fractional relief (height / local
radius): an isotropic Gaussian random field on the sphere, synthesized as a
sum of real spherical harmonics ``Y_lm`` whose coefficients are independent
Gaussians with variance ``C_l proportional to exp(-(l * corr_rad / 2)**2)``,
where ``l`` is the harmonic degree and ``corr_rad`` is the correlation length
in radians of surface arc; the band limit is ``lmax = ceil(8 / corr_rad)``,
where C has fallen to ~1e-7 of its peak.  Because the field is built on the
sphere, it is single-valued everywhere, its poles included, and it has the
same statistics at every point: the disc center is as smooth as the limb.

Degrees below 3 are zeroed: the degree-1 content of a radius perturbation
is, to first order, a *translation* of the body -- a planted, untruthed
center offset no limb fit could distinguish from the pointing error -- and
degree-2 content aliases ellipsoid shape error, which is its own scene knob.
After zeroing, the field is rescaled so the *limb slice's* standard
deviation equals the commanded RMS per-realization.  The RMS is a fraction
of the body radius (``h = H / R``), so a body of radius R pixels gets a limb
displacement of ``rms * R`` pixels RMS.

**Coordinate convention.**  The ellipsoid renderer puts the field's poles
on the observer axis, so the sub-observer horizon circle (the limb) is the
field's equator: latitude is elevation out of the limb plane toward the
observer, longitude is azimuth around the limb.  The field is stored on a
``(lat, lon)`` grid covering the torus ``[-pi, pi) x [0, 2*pi)``, the sphere
twice over (``h(lat, lon) = h(pi - lat, lon + pi)``), so bilinear sampling
wraps through the poles and the date line with no seam.

The terminator shadow march (:func:`march_shadows`) works in *absolute*
heights ``H = h * R`` in pixels (R the local body radius in pixels): the
fractional field (~0.01) and surface distances (pixels) are never compared
directly.  A point is shadowed iff some upstream sample at surface distance
``d`` toward the sun satisfies ``H_up - H_pt > d / tan(i_pt) + d**2 / (2 R)``,
with ``i_pt`` the point's incidence angle: the terrain must rise above both
the sun ray and the body's curvature drop.  The march is capped at an upper
bound on the distance at which that can happen (:func:`shadow_reach`),
``min((H_max - H_pt) * tan(i_pt), sqrt(2 * R * (H_max - H_pt)))``: the
longest shadow the terrain could cast on a flat body, and the horizon
distance of the highest peak, so cost is bounded and the geometry stays
physical.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from spindoctor.sim.spherical_harmonics import draw_sh_coefficients, legendre_order_sums
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'MAX_LIMB_RELIEF_RMS',
    'MIN_LIMB_RELIEF_CORR_DEG',
    'MarchStats',
    'ReliefField',
    'march_shadows',
    'shadow_reach',
    'synthesize_albedo_field',
    'synthesize_relief_field',
]

# The largest limb-relief RMS (a fraction of the body radius) a scene may
# command.  The renderer displaces only the limb: it draws the silhouette at
# the field's equator slice, R (1 + h(0, phi)), and reads the terrain under
# each disc pixel at the smooth sphere's direction.  The relief surface itself
# projects to the envelope R * max over lat of (1 + h(lat, phi)) * cos(lat),
# and its point under a pixel at elevation lat lies h * cot(lat) radians from
# the one the renderer reads.  Both errors grow with the RMS and shrink with
# the correlation length.  At the default 15-degree correlation length (8
# seeds each), the drawn limb's RMS distance from the envelope is 0.65 times
# the RMS at an RMS of 0.02, 1.03 times at 0.05, and 1.28 times at 0.1.  The
# terrain under 2.5%, 12%, and 30% of the disc is read more than half a
# correlation length from the point the pixel shows.  At 0.05 the limb's
# error reaches the size of the relief itself; above it, the limb and the
# terminator shadows stop approximating one surface.
MAX_LIMB_RELIEF_RMS = 0.05
# The shortest relief correlation length (degrees of arc) a scene may command.
# Synthesis costs about (1 / corr)**3: 0.5 s and 0.2 GB at 1 degree, 3.1 s and
# 0.5 GB at 0.5 degrees, and 26 s and 1.7 GB at 0.25 degrees.  The renderer
# synthesizes one field per body, and the scene editor re-renders on every
# change.
MIN_LIMB_RELIEF_CORR_DEG = 1.0
# Relief harmonic degrees below this are zeroed (degree 1 = body
# translation, degree 2 = ellipsoid shape aliasing).
_LOW_DEGREE_CUTOFF = 3.0
# The albedo field only zeroes the mean (k = 0): albedo texture multiplies
# the shading and never moves the silhouette, so its low-degree content
# cannot alias a body translation or an ellipsoid shape error the way
# relief's can, and large-scale albedo asymmetry is exactly the realistic
# hemispheric contrast the texture exists to plant.
_ALBEDO_LOW_DEGREE_CUTOFF = 1.0
# Band limit factor: kmax = ceil(8 / corr_rad); S(kmax) is ~1e-7 of peak.
_BAND_LIMIT_FACTOR = 8.0
# Synthesis grid bounds: at least 64 for a usable limb slice, at most 8192
# so a degenerate correlation length cannot demand an unbounded FFT.
_MIN_GRID = 64
_MAX_GRID = 8192


@dataclass(frozen=True)
class ReliefField:
    """One realization of the fractional relief field ``h(lat, lon)``.

    Parameters:
        grid: ``(n, n)`` periodic field over the full (lat, lon) torus; row
            ``i`` is latitude ``-pi + 2*pi*i/n`` (the limb equator is row
            ``n // 2``), column ``j`` is longitude ``2*pi*j/n``.
        rms: The commanded limb-slice standard deviation (h/R, unitless).
        corr_deg: The correlation length in degrees of surface arc.
        h_min: Global field minimum (fractional).
        h_max: Global field maximum (fractional).
    """

    grid: NDArrayFloatType
    rms: float
    corr_deg: float
    h_min: float
    h_max: float

    def sample(self, lat: NDArrayFloatType, lon: NDArrayFloatType) -> NDArrayFloatType:
        """Bilinearly sample the periodic field at (lat, lon) in radians.

        Parameters:
            lat: Latitudes above the limb plane (any shape; wrapped).
            lon: Longitudes around the limb (same shape as ``lat``; wrapped).

        Returns:
            Fractional relief values, same shape as the inputs.
        """
        n = self.grid.shape[0]
        rows = (np.asarray(lat, dtype=np.float64) + np.pi) * (n / (2.0 * np.pi))
        cols = np.asarray(lon, dtype=np.float64) * (n / (2.0 * np.pi))
        sampled = ndimage.map_coordinates(
            self.grid, [rows, cols], order=1, mode='grid-wrap', prefilter=False
        )
        return np.asarray(sampled, dtype=np.float64)

    def limb_slice(self) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """The limb slice ``h(0, phi)`` as interpolation nodes.

        Returns:
            Tuple ``(phi_nodes, values)``: longitudes ``0 .. 2*pi`` inclusive
            (the final node repeats the first, closing the circle) and the
            field values on the equator row, ready for ``np.interp``.
        """
        n = self.grid.shape[0]
        values = np.concatenate([self.grid[n // 2], self.grid[n // 2, :1]])
        phi_nodes = np.linspace(0.0, 2.0 * np.pi, n + 1)
        return phi_nodes, values

    def limb_delta(self, phi: NDArrayFloatType) -> NDArrayFloatType:
        """The limb perturbation ``delta(phi)`` at azimuths ``phi`` (radians).

        Parameters:
            phi: Azimuths around the limb (any real values; wrapped to
                ``[0, 2*pi)``).

        Returns:
            Fractional limb displacement values, same shape as ``phi``.
        """
        phi_nodes, values = self.limb_slice()
        wrapped = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * np.pi)
        return np.interp(wrapped, phi_nodes, values)


def _shaped_spectrum_field(
    corr_deg: float, seed: int, *, low_degree_cutoff: float
) -> NDArrayFloatType:
    """Synthesize the albedo texture's raw band-limited field (no normalization).

    The field is periodic on the ``(lat, lon)`` torus and synthesized by 2-D FFT
    with spectral variance ``exp(-(|k| * corr_rad / 2)**2)``.  It is not
    single-valued at the torus's poles; the albedo texture puts those poles on
    the body's axis-1 direction.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).
        low_degree_cutoff: Spectral modes with total wavenumber strictly below
            this are zeroed.

    Returns:
        The ``(n, n)`` un-normalized field grid.
    """
    corr_rad = math.radians(corr_deg)
    kmax = math.ceil(_BAND_LIMIT_FACTOR / corr_rad)
    n = int(min(_MAX_GRID, max(_MIN_GRID, 2 ** math.ceil(math.log2(max(4 * kmax, 1))))))

    # White Gaussian noise shaped in the spectral domain: the FFT of white
    # noise is a field of independent complex Gaussians (with the Hermitian
    # symmetry a real field requires), so multiplying by sqrt(S(k)) gives
    # coefficients with variance proportional to S(k).
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    spectrum = np.fft.rfft2(white)

    k_lat = np.fft.fftfreq(n, d=1.0 / n)
    k_lon = np.fft.rfftfreq(n, d=1.0 / n)
    k_total = np.hypot(k_lat[:, np.newaxis], k_lon[np.newaxis, :])
    amplitude = np.exp(-((k_total * corr_rad / 2.0) ** 2) / 2.0)
    amplitude[k_total < low_degree_cutoff] = 0.0
    amplitude[k_total > kmax] = 0.0

    return np.asarray(np.fft.irfft2(spectrum * amplitude, s=(n, n)), dtype=np.float64)


def _spherical_harmonic_field(corr_deg: float, seed: int) -> NDArrayFloatType:
    """Synthesize the raw relief field on the sphere (no amplitude normalization).

    The field is a sum of real spherical harmonics ``Y_lm`` with independent
    Gaussian coefficients of variance ``exp(-(l * corr_rad / 2)**2)``, so it is
    isotropic on the sphere and single-valued at the poles.  Degrees below
    :data:`_LOW_DEGREE_CUTOFF` and above ``lmax = ceil(8 / corr_rad)`` are zero.

    It is returned on the same ``(n, n)`` torus grid the albedo field uses (row
    ``i`` at latitude ``-pi + 2*pi*i/n``, column ``j`` at longitude
    ``2*pi*j/n``).  The rows in ``[-pi/2, pi/2]`` are synthesized directly; the
    others are filled from ``h(lat, lon) = h(pi - lat, lon + pi)``, the identity
    that names one surface point twice, so bilinear sampling across a pole
    reads the same surface from either side.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).

    Returns:
        The ``(n, n)`` un-normalized field grid.
    """
    corr_rad = math.radians(corr_deg)
    kmax = math.ceil(_BAND_LIMIT_FACTOR / corr_rad)
    n = int(min(_MAX_GRID, max(_MIN_GRID, 2 ** math.ceil(math.log2(max(4 * kmax, 1))))))
    # A degenerate correlation length can ask for more degrees than the capped
    # grid resolves in longitude; the band limit then follows the grid.
    lmax = min(kmax, n // 2 - 1)

    rng = np.random.default_rng(seed)
    degrees = np.arange(lmax + 1, dtype=np.float64)
    sigma = np.exp(-((degrees * corr_rad / 2.0) ** 2) / 2.0)
    sigma[degrees < _LOW_DEGREE_CUTOFF] = 0.0
    coeffs = draw_sh_coefficients(rng, sigma)

    # Synthesized rows: latitudes -pi/2 .. pi/2 inclusive (n is a multiple of 4).
    rows = np.arange(n // 4, 3 * n // 4 + 1)
    lat = -np.pi + 2.0 * np.pi * rows / n
    by_order = legendre_order_sums(coeffs, np.sin(lat), np.cos(lat))

    # Each row is sum_m Re(Z_m exp(i m lon)); irfft's 1/n and its doubling of
    # the positive orders are undone in the scaling.
    spectrum = np.zeros((rows.size, n // 2 + 1), dtype=np.complex128)
    spectrum[:, 0] = n * by_order[0]
    spectrum[:, 1 : lmax + 1] = (n / 2.0) * by_order[1:].T
    half = np.fft.irfft(spectrum, n=n, axis=1)

    grid = np.empty((n, n), dtype=np.float64)
    grid[rows] = half
    other = np.setdiff1d(np.arange(n), rows)
    # h(lat_i, lon) = h(pi - lat_i, lon + pi): row 3n/2 - i, shifted half a turn.
    grid[other] = np.roll(grid[np.mod(3 * n // 2 - other, n)], -(n // 2), axis=1)
    return grid


@lru_cache(maxsize=8)
def _unit_relief_grid(corr_deg: float, seed: int) -> NDArrayFloatType:
    """A field realization normalized to unit limb-slice standard deviation.

    The RMS knob scales the field linearly, so realizations are cached per
    (correlation length, seed) at unit limb RMS and scaled on request.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).

    Returns:
        The ``(n, n)`` unit-normalized field grid.  All-zero when the degree
        band ``[3, lmax]`` is empty (a correlation length so long that no
        degree survives the low-degree zeroing).
    """
    field = _spherical_harmonic_field(corr_deg, seed)
    n = field.shape[0]
    limb_std = float(np.std(field[n // 2]))
    if limb_std <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    return np.asarray(field / limb_std, dtype=np.float64)


@lru_cache(maxsize=8)
def _unit_albedo_grid(corr_deg: float, seed: int) -> NDArrayFloatType:
    """An albedo-field realization normalized to unit global standard deviation.

    Unlike the relief grid, the normalization is over the whole surface (the
    commanded RMS is the disc-wide texture contrast, not a limb statistic) and
    only the mean is zeroed (see :data:`_ALBEDO_LOW_DEGREE_CUTOFF`).

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).

    Returns:
        The ``(n, n)`` unit-normalized field grid; all-zero when no mode
        survives the band limit.
    """
    field = _shaped_spectrum_field(corr_deg, seed, low_degree_cutoff=_ALBEDO_LOW_DEGREE_CUTOFF)
    n = field.shape[0]
    std = float(np.std(field))
    if std <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    return np.asarray(field / std, dtype=np.float64)


def synthesize_relief_field(rms: float, corr_deg: float, seed: int) -> ReliefField:
    """Synthesize one relief-field realization.

    Parameters:
        rms: Commanded limb-slice standard deviation (h/R, unitless);
            0 yields the all-zero field.
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed, derived from the body's seed chain.

    Returns:
        The scaled :class:`ReliefField` realization.
    """
    grid = _unit_relief_grid(float(corr_deg), int(seed)) * float(rms)
    return ReliefField(
        grid=grid,
        rms=float(rms),
        corr_deg=float(corr_deg),
        h_min=float(grid.min()),
        h_max=float(grid.max()),
    )


def synthesize_albedo_field(rms: float, corr_deg: float, seed: int) -> ReliefField:
    """Synthesize one multiplicative-albedo field realization.

    The albedo texture reuses the relief field's :class:`ReliefField` sampling
    container and spectrum shape, synthesized on the flat ``(lat, lon)`` torus
    rather than on the sphere.  The commanded RMS is the field's *global*
    standard deviation (disc-wide texture contrast rather than a limb
    statistic), and only the spectral mean is zeroed -- albedo multiplies the
    shading and never moves the silhouette, so low-degree content cannot alias
    pointing or shape error.

    Parameters:
        rms: Commanded global standard deviation of the fractional albedo
            perturbation; 0 yields the all-zero field.
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed, derived from the body's 'albedo' stream.

    Returns:
        The scaled :class:`ReliefField` realization (holding the albedo
        perturbation; the multiplicative factor is ``1 + field``).
    """
    grid = _unit_albedo_grid(float(corr_deg), int(seed)) * float(rms)
    return ReliefField(
        grid=grid,
        rms=float(rms),
        corr_deg=float(corr_deg),
        h_min=float(grid.min()),
        h_max=float(grid.max()),
    )


def clear_relief_caches() -> None:
    """Drop the cached field realizations (test/tooling helper)."""
    _unit_relief_grid.cache_clear()
    _unit_albedo_grid.cache_clear()


@dataclass(frozen=True)
class MarchStats:
    """Diagnostics of one terminator shadow march.

    Parameters:
        candidate_count: Number of points the march started from.
        steps_executed: Surface-arc steps actually executed (the loop count).
        max_steps: The global step bound implied by the march cap; the loop
            can never exceed it regardless of the domain's extent.
    """

    candidate_count: int
    steps_executed: int
    max_steps: int


def shadow_reach(
    h_point: NDArrayFloatType,
    tan_incidence: NDArrayFloatType,
    radius_px: NDArrayFloatType,
    *,
    h_max: float,
) -> NDArrayFloatType:
    """An upper bound on the surface distance from which terrain can shadow each point.

    Terrain at surface distance ``d`` toward the sun shadows a point only if it
    rises ``d / tan(i) + d**2 / (2 R)`` above it, and no terrain rises more than
    ``(h_max - h_point) * R``.  Each term alone must fit under that rise, so
    ``min(rise * tan(i), sqrt(2 * R * rise))`` bounds the distance: the longest
    shadow the terrain could cast on a flat body, and the horizon distance of
    the highest peak.  The bound is at most 1.62 times the exact distance, the
    root of ``rise = d / tan(i) + d**2 / (2 R)``; the worst case is where the
    two terms are equal.  Marching to the bound changes no shadow decision.

    Parameters:
        h_point: Fractional relief at each point.
        tan_incidence: Tangent of each point's incidence angle (> 0).
        radius_px: Local body radius at each point, in pixels.
        h_max: Global field maximum (fractional).

    Returns:
        The bound in pixels of surface arc, same shape as the inputs; 0 where
        the point is the highest terrain.
    """
    radius = np.asarray(radius_px, dtype=np.float64)
    rise = np.maximum(h_max - np.asarray(h_point, dtype=np.float64), 0.0) * radius
    tan_i = np.asarray(tan_incidence, dtype=np.float64)
    return np.asarray(np.minimum(rise * tan_i, np.sqrt(2.0 * radius * rise)), dtype=np.float64)


def march_shadows(
    start_v: NDArrayFloatType,
    start_u: NDArrayFloatType,
    *,
    h_point: NDArrayFloatType,
    tan_incidence: NDArrayFloatType,
    radius_px: NDArrayFloatType,
    height_map: NDArrayFloatType,
    arc_per_px_map: NDArrayFloatType,
    domain_mask: NDArrayBoolType,
    h_max: float,
    h_min: float,
    sun_v: float,
    sun_u: float,
    step_arc_px: float = 1.0,
) -> tuple[NDArrayBoolType, MarchStats]:
    """March candidate points toward the sun and flag the shadowed ones.

    Each candidate steps toward the sun in units of ``step_arc_px`` pixels
    of *surface* arc (the image-plane step is ``step / arc_per_px`` at the
    current sample, so foreshortening near the limb is corrected: an image
    pixel there spans a large surface step).  A candidate is shadowed iff
    an upstream sample at surface distance ``d`` satisfies
    ``H_up - H_pt > d / tan(i_pt) + d**2 / (2 R)``: the terrain must rise
    above the sun ray and above the body's curvature drop (``R`` the
    point's local radius, standing in for the radius of curvature).  The
    per-point cap ``d_max`` of :func:`shadow_reach`,
    ``min((H_max - H_pt) * tan(i_pt), sqrt(2 * R * (H_max - H_pt)))``, is an
    upper bound on the distance at which that inequality can hold, so it
    bounds the work without changing any decision.

    Parameters:
        start_v: Candidate image-row positions (work-grid pixels, float).
        start_u: Candidate image-column positions, same shape.
        h_point: Fractional relief at each candidate's surface point.
        tan_incidence: Tangent of each candidate's incidence angle (> 0).
        radius_px: Local body radius at each candidate, in work pixels.
        height_map: Absolute heights ``h * R_local`` (work px) sampled at
            each domain pixel's surface point; zero outside the domain.
        arc_per_px_map: Surface arc (work px) spanned by one image pixel
            along the sun direction, per domain pixel.
        domain_mask: Pixels where the maps are valid; a ray leaving the
            domain stops marching (the caller sizes the domain to cover the
            longest capped march, so no caster can sit beyond it).
        h_max: Global field maximum (fractional).
        h_min: Global field minimum (fractional).
        sun_v: Image-plane unit direction toward the sun, row component.
        sun_u: Image-plane unit direction toward the sun, column component.
        step_arc_px: Surface arc per step, in work pixels.  1 (the default)
            samples the terrain at the render grid itself; a caller whose
            terrain is band-limited far above the render grid's scale may
            step at the terrain's own resolution instead (see the
            topographic renderer), which changes no decision the sampled
            terrain can express.

    Returns:
        Tuple of (shadow flags per candidate, :class:`MarchStats`).
    """
    n = int(start_v.size)
    shadow = np.zeros(n, dtype=np.bool_)
    if n == 0 or h_max <= h_min:
        return shadow, MarchStats(candidate_count=n, steps_executed=0, max_steps=0)

    size_v, size_u = height_map.shape
    step_arc = max(float(step_arc_px), 1e-6)
    tan_i = np.maximum(np.asarray(tan_incidence, dtype=np.float64), 1e-12)
    radius = np.asarray(radius_px, dtype=np.float64)
    height_pt = np.asarray(h_point, dtype=np.float64) * radius
    d_max = shadow_reach(h_point, tan_i, radius, h_max=h_max)
    max_steps = math.ceil(float(d_max.max()) / step_arc) if d_max.size else 0

    # Live-set arrays, compressed as candidates resolve.
    index = np.arange(n)
    pos_v = np.asarray(start_v, dtype=np.float64).copy()
    pos_u = np.asarray(start_u, dtype=np.float64).copy()
    cur_v = np.clip(np.floor(pos_v).astype(np.intp), 0, size_v - 1)
    cur_u = np.clip(np.floor(pos_u).astype(np.intp), 0, size_u - 1)
    distance = np.zeros(n, dtype=np.float64)

    steps = 0
    while index.size and steps < max_steps:
        steps += 1
        # One step of surface arc; the image step shrinks where an image
        # pixel spans more surface (foreshortening near the limb).
        arc = np.maximum(arc_per_px_map[cur_v, cur_u], 0.1)
        step_px = step_arc / arc
        pos_v = pos_v + sun_v * step_px
        pos_u = pos_u + sun_u * step_px
        distance = distance + step_arc

        cur_v = np.floor(pos_v).astype(np.intp)
        cur_u = np.floor(pos_u).astype(np.intp)
        inside = (cur_v >= 0) & (cur_v < size_v) & (cur_u >= 0) & (cur_u < size_u)
        cur_v = np.clip(cur_v, 0, size_v - 1)
        cur_u = np.clip(cur_u, 0, size_u - 1)
        inside &= domain_mask[cur_v, cur_u]

        # Samples beyond the point's cap cannot shadow it: the terrain term
        # makes such hits impossible algebraically, and the horizon term
        # excludes terrain the body's curvature hides.
        # The body's curvature drops the upstream surface d**2 / (2 R) below
        # the point's tangent plane, so the terrain must clear that drop as
        # well as the sun ray's rise d / tan(i).
        clearance = distance / tan_i + distance * distance / (2.0 * radius)
        hit = inside & (distance <= d_max) & (height_map[cur_v, cur_u] - height_pt > clearance)
        shadow[index[hit]] = True

        keep = inside & ~hit & (distance < d_max)
        if not keep.all():
            index = index[keep]
            pos_v = pos_v[keep]
            pos_u = pos_u[keep]
            cur_v = cur_v[keep]
            cur_u = cur_u[keep]
            distance = distance[keep]
            tan_i = tan_i[keep]
            height_pt = height_pt[keep]
            radius = radius[keep]
            d_max = d_max[keep]

    return shadow, MarchStats(candidate_count=n, steps_executed=steps, max_steps=max_steps)
