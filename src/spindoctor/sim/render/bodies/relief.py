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
After zeroing, the field is rescaled so the standard deviation of the *limb
slice*, its equator, equals the commanded RMS per-realization.  The RMS is a fraction
of the body radius (``h = H / R``), so a body of radius R pixels gets a limb
displacement of ``rms * R`` pixels RMS.

**Coordinate convention.**  The field's poles lie on the body's semi-axis c,
which points at the observer when the body is seen equator-on.  The limb is then
the field's equator: latitude is elevation out of the limb plane toward the
observer, longitude is azimuth around the limb.  The field turns with the body, so
from any other sub-observer latitude the limb is another great circle of the field
(:func:`directions_about_view`).  The field is stored on a
``(lat, lon)`` grid covering the torus ``[-pi, pi) x [0, 2*pi)``, the sphere
twice over (``h(lat, lon) = h(pi - lat, lon + pi)``), so bilinear sampling
wraps through the poles and the date line with no seam.

The cast-shadow test (:func:`cast_shadows`) works in the body's *parameter*
space, where the smooth ellipsoid is the unit sphere and the relief surface is
``r = 1 + h(r_hat)``.  The map from the body frame to that space is affine, and
an affine map carries a ray to a ray and a surface to a surface, so a ray from a
surface point toward the sun meets the terrain in parameter space exactly when
it does on the body.  A point is shadowed iff its sun ray, once it has risen above
the terrain, dips below it again before it leaves the relief shell
``r <= 1 + h_max``: only terrain between the point and the sun shadows it
(:func:`cast_shadows`).  The terrain right at the point can also turn away from the
sun, which :func:`turned_from_sun` tells from the bilinear field itself.  Which
points can be shadowed at all follows from the field's steepest slope ``s_max`` (per
radian of arc, :attr:`ReliefField.slope_max`): along the ray the terrain rises
at most ``s_max / (1 + h_min)`` per unit of the ray's own rise, so a point whose
sun elevation has ``tan(elevation) >= s_max / (1 + h_min)`` is never shadowed,
and one with ``tan(elevation) <= -s_max / (1 + h_min)`` always is.

The relief also tilts the surface normal: the relief surface's normal in
parameter space is ``r_hat - grad h / (1 + h)``, with ``grad h`` the field's
surface gradient (:meth:`ReliefField.surface_gradient`).  A pixel shows the
point where its line of sight first meets the relief surface, which
:func:`~spindoctor.sim.render.bodies.relief_sight.visible_surface_directions`
finds.  Near the limb that point can lie well away from where the line of sight
meets the smooth sphere, and behind a ridge the terrain is hidden altogether.

The limb is the relief surface's own outline.  With latitude and longitude taken
about the direction toward the observer, a surface point at ``(lat, lon)`` projects
to azimuth ``lon`` at radius ``(1 + h) cos(lat)``.  So the outline at azimuth ``phi``
sits at the envelope ``max over lat of (1 + h(lat, phi)) cos(lat)``
(:meth:`ReliefField.outline_delta`).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from spindoctor.sim.shared.spherical_harmonics import draw_sh_coefficients, legendre_order_sums
from spindoctor.support.stable_math import stable_matmul3
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'MAX_LIMB_RELIEF_RMS',
    'MIN_LIMB_RELIEF_CORR_DEG',
    'ReliefField',
    'cast_shadows',
    'clear_relief_caches',
    'directions_about_view',
    'synthesize_albedo_field',
    'synthesize_relief_field',
    'turned_from_sun',
]

# The largest limb-relief RMS (a fraction of the body radius) a scene may
# command, a limit of the scene format.  At 0.1 the surface stays about half the
# radius out from the center: over 16 seeds at each of 1, 2, 5, 15, and 30 degrees
# of correlation length, the deepest point lies 0.5004 of the radius below the
# smooth surface.  At 0.2 it reaches the center (h_min -1.0 over 8 seeds).  The
# ellipsoid renderer reads the terrain where each line of sight meets the relief
# surface; at 0.1 and 2 to 30 degrees, a brute-force search puts its point more
# than half a correlation length off on at most 0.025% of the disc.  The validator
# holds mesh bodies to the same bound.
MAX_LIMB_RELIEF_RMS = 0.1
# The shortest relief correlation length (degrees of arc) a scene may command.
# It bounds cost.  Synthesis costs about (1 / corr)**3: 0.5 s and 0.2 GB at
# 1 degree, 3.1 s and 0.5 GB at 0.5 degrees, and 26 s and 1.7 GB at
# 0.25 degrees.  A realization is cached by (corr, seed) and rescaled to the
# RMS, so a render, or the scene editor, pays that cost again only when the
# correlation length or the relief seed changes.  The outline envelope costs
# about 2 s more at 1 degree (0.5 s at 2 degrees) and is cached by (corr, seed,
# RMS).
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
# The cast-shadow march samples its first step this many times more finely, so a
# terrain wall close to the point is still seen apart from the terrain the point
# lies on (see cast_shadows).
_FIRST_STEP_SPLIT = 8
# Before those, it takes this many samples whose distances double up to the finer
# spacing, from 1/64 of it, so the ray is seen to rise off the ground before a wall
# even nearer the point.  Against a brute-force march, a ray starting on the surface
# that took only the evenly spaced samples missed walls within 1/64 of a cell.
_LEAD_SAMPLES = 6
# The march skips the samples a slope bound proves above the terrain (see
# _clear_distances).  The bound's rates are inflated by this fraction and each
# sample's clearance above the terrain is reduced by this much, both far beyond
# their rounding, so a skipped sample is never one the march would have found
# below the terrain, and the march's answer is unchanged.
_CLEAR_RATE_MARGIN = 1e-9
_CLEARANCE_MARGIN = 1e-12
# The length, in field grid cells, over which turned_from_sun differences the depth
# below the terrain at a point seen: short enough to read the bilinear terrain the
# point lies on, not the next cell's.
_FACET_DIFFERENCE_CELLS = 1.0 / 4096.0
# The least rate, per unit of distance, at which that depth must change for
# turned_from_sun to count the terrain as turning away from the sun.  It lies far above
# the depth's rounding over that length, so at phase 0, where the two differences
# agree to rounding, one cannot exceed it while the other falls below its negative.
_FACET_TOLERANCE = 1e-6


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
        slope_max: An upper bound on the field's surface slope, ``|grad h|``
            per radian of arc, over the whole sphere; 0 for the zero field.
        outline: The outline envelope ``max over lat of (1 + h) cos(lat) - 1``
            at evenly spaced longitudes ``0 .. 2*pi`` inclusive (the last
            repeats the first), or None for a field that never draws a limb.
    """

    grid: NDArrayFloatType
    rms: float
    corr_deg: float
    h_min: float
    h_max: float
    slope_max: float = 0.0
    outline: NDArrayFloatType | None = None

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

    def outline_delta(
        self, phi: NDArrayFloatType, *, view: tuple[float, float] = (0.0, 1.0)
    ) -> NDArrayFloatType:
        """The relief surface's outline displacement at azimuths ``phi`` (radians).

        The outline at azimuth ``phi`` sits at ``1 + delta(phi)`` times the
        smooth limb radius, where ``delta`` is the envelope ``max over lat of
        (1 + h(lat, phi)) cos(lat) - 1``, with latitude measured from the plane
        perpendicular to the line of sight: the farthest any surface point
        projects along that azimuth.  It is at least the limb slice there.

        Seen along the field's own pole, ``z``, the envelope is the stored
        :attr:`outline`.  Seen along a direction turned toward ``x``, the limb is
        the great circle perpendicular to it, and the envelope is evaluated for
        that circle: azimuth 0 points along the turned ``x`` axis, and azimuth
        ``pi / 2`` along ``y``.

        Parameters:
            phi: Azimuths around the limb (any real values; wrapped to
                ``[0, 2*pi)``).
            view: The unit direction toward the observer in the field's frame,
                as its ``x`` and ``z`` components (its ``y`` component is 0).

        Returns:
            Fractional outline displacement values, same shape as ``phi``.

        Raises:
            ValueError: If the field carries no outline (an albedo field).
        """
        if self.outline is None:
            raise ValueError('this field carries no outline envelope')
        wrapped = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * np.pi)
        if view[0] == 0.0:
            phi_nodes = np.linspace(0.0, 2.0 * np.pi, self.outline.size)
            return np.interp(wrapped, phi_nodes, self.outline)
        lons, envelope = _turned_outline_envelope(self, view)
        return np.interp(wrapped, lons, envelope)

    def sample_points(self, points: NDArrayFloatType) -> NDArrayFloatType:
        """Sample the field at unit-sphere directions in the field's own frame.

        The field's poles are on the ``z`` axis, so a direction ``(x, y, z)``
        reads latitude ``arcsin(z)`` and longitude ``atan2(y, x)``.

        Parameters:
            points: ``(..., 3)`` directions; each is normalized first.

        Returns:
            Fractional relief values, shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=np.float64)
        return self.sample_points_at(pts, np.linalg.norm(pts, axis=-1))

    def sample_points_at(
        self, points: NDArrayFloatType, norm: NDArrayFloatType
    ) -> NDArrayFloatType:
        """Sample the field in the directions of points whose norms are known.

        The same as :meth:`sample_points`, for a caller that already holds the
        points' norms.

        Parameters:
            points: ``(..., 3)`` positions in the field's frame.
            norm: Their norms, ``points.shape[:-1]``.

        Returns:
            Fractional relief values, shape ``points.shape[:-1]``.
        """
        safe_norm = np.maximum(norm, 1e-300)
        lat = np.arcsin(np.clip(points[..., 2] / safe_norm, -1.0, 1.0))
        lon = np.mod(np.arctan2(points[..., 1], points[..., 0]), 2.0 * np.pi)
        return self.sample(lat, lon)

    def depth_below(self, points: NDArrayFloatType) -> NDArrayFloatType:
        """How far parameter-space positions lie below the relief surface.

        Parameters:
            points: ``(..., 3)`` positions in the field's frame.

        Returns:
            ``1 + h(r_hat) - |r|`` at each position: positive inside the terrain,
            negative above it, shape ``points.shape[:-1]``.
        """
        pts = np.asarray(points, dtype=np.float64)
        norm = np.linalg.norm(pts, axis=-1)
        return np.asarray(1.0 + self.sample_points_at(pts, norm) - norm)

    def surface_gradient(self, points: NDArrayFloatType) -> NDArrayFloatType:
        """The field's surface gradient at unit-sphere directions, per radian of arc.

        Central differences along two tangent directions at each point,
        half a field grid cell apart, so the result is the slope the
        bilinear field carries.  The tangent basis avoids every coordinate
        singularity, so the poles are as well defined as the equator.

        Parameters:
            points: ``(..., 3)`` unit directions in the field's frame.

        Returns:
            ``(..., 3)`` tangent gradient vectors (orthogonal to ``points``).
        """
        p = np.asarray(points, dtype=np.float64)
        p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-300)
        # A helper axis never parallel to p: x where p leans toward z, else z.
        helper = np.zeros_like(p)
        use_x = np.abs(p[..., 2]) > 0.9
        helper[..., 0] = use_x
        helper[..., 2] = ~use_x
        e1 = np.cross(p, helper)
        e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = np.cross(p, e1)
        step = np.pi / self.grid.shape[0]
        gradient = np.zeros_like(p)
        for tangent in (e1, e2):
            forward = self.sample_points(p * math.cos(step) + tangent * math.sin(step))
            back = self.sample_points(p * math.cos(step) - tangent * math.sin(step))
            gradient += ((forward - back) / (2.0 * step))[..., np.newaxis] * tangent
        return gradient


def _spherical_harmonic_field(
    corr_deg: float, seed: int, *, low_degree_cutoff: float = _LOW_DEGREE_CUTOFF
) -> NDArrayFloatType:
    """Synthesize a raw field on the sphere (no amplitude normalization).

    The field is a sum of real spherical harmonics ``Y_lm`` with independent
    Gaussian coefficients of variance ``exp(-(l * corr_rad / 2)**2)``, so it is
    isotropic on the sphere and single-valued at the poles.  Degrees below
    ``low_degree_cutoff`` and above ``lmax = ceil(8 / corr_rad)`` are zero.

    It is returned on an ``(n, n)`` torus grid (row
    ``i`` at latitude ``-pi + 2*pi*i/n``, column ``j`` at longitude
    ``2*pi*j/n``).  The rows in ``[-pi/2, pi/2]`` are synthesized directly; the
    others are filled from ``h(lat, lon) = h(pi - lat, lon + pi)``, the identity
    that names one surface point twice, so bilinear sampling across a pole
    reads the same surface from either side.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).
        low_degree_cutoff: Degrees strictly below this are zeroed.

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
    sigma[degrees < low_degree_cutoff] = 0.0
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

    The field is synthesized on the sphere like the relief, so it is
    single-valued at its poles.  Unlike the relief grid, the normalization is
    the area-weighted standard deviation over the whole sphere (the commanded
    RMS is the disc-wide texture contrast, not a limb statistic) and only the
    mean is zeroed (see :data:`_ALBEDO_LOW_DEGREE_CUTOFF`).

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).

    Returns:
        The ``(n, n)`` unit-normalized field grid; all-zero when no mode
        survives the band limit.
    """
    field = _spherical_harmonic_field(corr_deg, seed, low_degree_cutoff=_ALBEDO_LOW_DEGREE_CUTOFF)
    n = field.shape[0]
    # The rows from the south pole to the north pole cover the sphere once;
    # cos(lat) weights each by the area it stands for.
    rows = np.arange(n // 4, 3 * n // 4 + 1)
    weights = np.cos(-np.pi + 2.0 * np.pi * rows / n)[:, np.newaxis]
    std = float(np.sqrt(np.sum(weights * field[rows] ** 2) / (np.sum(weights) * n)))
    if std <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    return np.asarray(field / std, dtype=np.float64)


@lru_cache(maxsize=8)
def _unit_relief_slope(corr_deg: float, seed: int) -> float:
    """An upper bound on the unit relief field's surface slope, per radian of arc.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed.

    Returns:
        The bound (:func:`_grid_slope_bound`) for the unit-RMS realization; the slope
        scales with the RMS.
    """
    return _grid_slope_bound(_unit_relief_grid(corr_deg, seed))


def _grid_slope_bound(grid: NDArrayFloatType) -> float:
    """An upper bound on a field grid's surface slope, per radian of arc.

    The bilinear field's slope along latitude is the row difference over the
    row spacing, and along longitude the column difference over the column
    arc ``cos(lat)`` times the spacing; each is maximized over the rows
    strictly between the poles (the pole rows are single points, with no
    longitude extent), and the two maxima combine in quadrature.

    Parameters:
        grid: The ``(n, n)`` torus grid of a :class:`ReliefField`.

    Returns:
        The bound.
    """
    n = grid.shape[0]
    spacing = 2.0 * np.pi / n
    half = grid[n // 4 : 3 * n // 4 + 1]
    lat_slope = float(np.max(np.abs(np.diff(half, axis=0)))) / spacing
    rows = np.arange(n // 4 + 1, 3 * n // 4)
    cos_lat = np.cos(-np.pi + 2.0 * np.pi * rows / n)[:, np.newaxis]
    lon_step = np.abs(np.roll(grid[rows], -1, axis=1) - grid[rows])
    lon_slope = float(np.max(lon_step / (spacing * cos_lat)))
    return math.hypot(lat_slope, lon_slope)


# Outline envelope samples per field cell: in latitude the envelope's maximum
# is located to an eighth of a row, and in longitude the envelope is stored at
# eight nodes per column, because between two columns the maximizing latitude
# can jump and the envelope bends where a straight line between the columns'
# values would not.  Together they hold the outline within 7e-4 of the bilinear
# field's exact envelope (0.06 px on an 80 px radius) on 99.9% of the azimuths at 5%
# relief and 1.4e-3 at 10%, and within 2e-3 and 4e-3 everywhere, at a 15-degree
# correlation length over 8 seeds and sub-observer latitudes of 0, 30, -60, and 89
# degrees.
# The longitudes are evaluated in chunks so the sample grid stays bounded.
_OUTLINE_SAMPLES_PER_ROW = 8.0
_OUTLINE_NODES_PER_COLUMN = 8
_OUTLINE_CHUNK_SAMPLES = 1 << 22


@lru_cache(maxsize=8)
def _outline_envelope(corr_deg: float, seed: int, rms: float) -> NDArrayFloatType:
    """The relief surface's outline envelope at the field's longitude nodes.

    Only latitudes within ``acos((1 + h_min) / (1 + h_max))`` of the equator
    can reach the envelope: beyond them ``(1 + h) cos(lat)`` stays below
    ``1 + h_min``, which the equator itself always reaches.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed.
        rms: The commanded RMS the unit realization is scaled by.

    Returns:
        ``max over lat of (1 + h) cos(lat) - 1`` at the ``4 n + 1`` longitudes
        ``0 .. 2*pi`` inclusive.
    """
    grid = _unit_relief_grid(corr_deg, seed) * rms
    n = grid.shape[0]
    field = ReliefField(
        grid=grid, rms=rms, corr_deg=corr_deg, h_min=float(grid.min()), h_max=float(grid.max())
    )
    lat_cap = math.acos(min(1.0, (1.0 + field.h_min) / (1.0 + field.h_max)))
    half_count = math.ceil(lat_cap / (2.0 * np.pi / n) * _OUTLINE_SAMPLES_PER_ROW)
    lats = np.linspace(-lat_cap, lat_cap, 2 * half_count + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, _OUTLINE_NODES_PER_COLUMN * n + 1)
    cos_lat = np.cos(lats)[:, np.newaxis]
    envelope = np.empty(lons.size, dtype=np.float64)
    chunk = max(1, _OUTLINE_CHUNK_SAMPLES // lats.size)
    for start in range(0, lons.size, chunk):
        lat_grid, lon_grid = np.meshgrid(lats, lons[start : start + chunk], indexing='ij')
        radius = (1.0 + field.sample(lat_grid, lon_grid)) * cos_lat
        envelope[start : start + chunk] = radius.max(axis=0)
    return envelope - 1.0


def directions_about_view(
    lat: NDArrayFloatType, lon: NDArrayFloatType, view: tuple[float, float]
) -> NDArrayFloatType:
    """Unit directions in the field's frame, at latitudes and longitudes about a view.

    Latitude is measured from the plane perpendicular to the line of sight, toward
    the observer.  Longitude runs around that plane, 0 along the ``x`` axis turned
    with the view and ``pi / 2`` along ``y``.

    Parameters:
        lat: Latitudes in radians.
        lon: Longitudes in radians, broadcastable with ``lat``.
        view: The unit direction toward the observer, its ``x`` and ``z``
            components in the field's frame (its ``y`` component is 0).

    Returns:
        The directions, the broadcast shape plus ``(3,)``.
    """
    view_x, view_z = view
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    turned_x = cos_lat * np.cos(lon)
    turned_y = cos_lat * np.sin(lon)
    return np.stack(
        [
            turned_x * view_z + sin_lat * view_x,
            turned_y,
            sin_lat * view_z - turned_x * view_x,
        ],
        axis=-1,
    )


def _turned_outline_envelope(
    field: ReliefField, view: tuple[float, float]
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """The outline envelope for a line of sight turned away from the field's pole.

    The same envelope :func:`_outline_envelope` stores, at the same sample spacing,
    evaluated around the great circle perpendicular to ``view`` instead of the
    field's equator.

    Parameters:
        field: The relief realization.
        view: The unit direction toward the observer, its ``x`` and ``z``
            components in the field's frame.

    Returns:
        The longitudes ``0 .. 2*pi`` inclusive around the turned limb, and the
        envelope less 1 at each.
    """
    n = field.grid.shape[0]
    lat_cap = math.acos(min(1.0, (1.0 + field.h_min) / (1.0 + field.h_max)))
    half_count = math.ceil(lat_cap / (2.0 * np.pi / n) * _OUTLINE_SAMPLES_PER_ROW)
    lats = np.linspace(-lat_cap, lat_cap, 2 * half_count + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, _OUTLINE_NODES_PER_COLUMN * n + 1)
    cos_lat = np.cos(lats)[:, np.newaxis]
    envelope = np.empty(lons.size, dtype=np.float64)
    chunk = max(1, _OUTLINE_CHUNK_SAMPLES // lats.size)
    for start in range(0, lons.size, chunk):
        lon = lons[start : start + chunk][np.newaxis, :]
        points = directions_about_view(lats[:, np.newaxis], lon, view)
        radius = (1.0 + field.sample_points(points)) * cos_lat
        envelope[start : start + chunk] = radius.max(axis=0)
    return lons, envelope - 1.0


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
    corr = float(corr_deg)
    seed_int = int(seed)
    rms_f = float(rms)
    grid = _unit_relief_grid(corr, seed_int) * rms_f
    return ReliefField(
        grid=grid,
        rms=rms_f,
        corr_deg=corr,
        h_min=float(grid.min()),
        h_max=float(grid.max()),
        slope_max=_unit_relief_slope(corr, seed_int) * abs(rms_f),
        outline=_outline_envelope(corr, seed_int, rms_f),
    )


def synthesize_albedo_field(rms: float, corr_deg: float, seed: int) -> ReliefField:
    """Synthesize one multiplicative-albedo field realization.

    The albedo texture reuses the relief field's :class:`ReliefField` sampling
    container and spectrum shape, synthesized on the sphere the same way, so it
    is single-valued and statistically the same everywhere.  The commanded RMS
    is the field's *global* standard deviation (disc-wide texture contrast
    rather than a limb statistic), and only the spectral mean is zeroed --
    albedo multiplies the shading and never moves the silhouette, so
    low-degree content cannot alias pointing or shape error.

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
    _unit_relief_slope.cache_clear()
    _outline_envelope.cache_clear()


def cast_shadows(
    points: NDArrayFloatType,
    sun: NDArrayFloatType,
    field: ReliefField,
    *,
    step: float,
) -> NDArrayBoolType:
    """Whether terrain between each surface point and the sun shadows it, by ray march.

    Everything is in the body's parameter space (the smooth body is the unit
    sphere, the field's poles on the ``z`` axis).  Each point's ray
    ``P + t * sun`` is marched until it leaves the relief shell
    ``|r| <= 1 + h_max``; a sample lies below the terrain where
    ``|r| < 1 + h(r_hat)``.  The point is shadowed iff the ray, once it has risen
    above the terrain, comes down below it again.  Terrain the ray is in before it
    first rises is the terrain the point lies on, which :func:`turned_from_sun`
    judges instead: there the ray starts into the terrain because the ground turns
    away from the sun, or because the point lies just inside it.  The first step is
    sampled :data:`_FIRST_STEP_SPLIT` times more finely than the rest, after
    :data:`_LEAD_SAMPLES` samples nearer still, so the ray still shows open sky
    between the point and a wall less than a step away.  Past the first step, a
    sample above the terrain lets the ray skip the samples it provably stays above
    the terrain for (:func:`_clear_distances`), and a ray that can never come down
    again stops; none of those samples could shadow the point, so the answer is the
    one every sample would give.  The test is exact up to the step and the bilinear
    field, and it reads terrain on the far hemisphere as readily as on the near one.

    Parameters:
        points: ``(N, 3)`` parameter-space positions of the points, on the relief
            surface or within a search's precision of it.
        sun: The unit sun direction in the same space, ``(3,)``.
        field: The relief realization.
        step: The ray step in parameter-space units (a fraction of the
            radius); the caller keeps it near the field's own resolution.

    Returns:
        ``(N,)`` booleans, True where the point is in a cast shadow.

    Raises:
        ValueError: If ``step`` is not a finite positive number, since the
            march would then never leave the shell.
    """
    if not (math.isfinite(step) and step > 0.0):
        raise ValueError(f'cast_shadows step must be finite and positive, got {step!r}')
    start = np.asarray(points, dtype=np.float64)
    count = start.shape[0]
    shadowed = np.zeros(count, dtype=np.bool_)
    if count == 0 or field.h_max <= field.h_min:
        return shadowed
    sun_hat = np.asarray(sun, dtype=np.float64) / float(np.linalg.norm(sun))
    # The ray leaves the shell at the positive root of |P + t s| = 1 + h_max.
    along = stable_matmul3(start, sun_hat)
    reach = -along + np.sqrt(
        np.maximum(along**2 - (np.sum(start**2, axis=1) - (1.0 + field.h_max) ** 2), 0.0)
    )
    risen = np.zeros(count, dtype=np.bool_)
    # The samples within the first step, the same distance along every ray.
    sample = 1
    active = np.nonzero(reach > _march_distance(sample, step))[0]
    while active.size and sample <= _LEAD_SAMPLES + _FIRST_STEP_SPLIT:
        ray = start[active] + _march_distance(sample, step) * sun_hat
        radius = np.linalg.norm(ray, axis=1)
        below = radius < 1.0 + field.sample_points_at(ray, radius)
        hit = below & risen[active]
        shadowed[active[hit]] = True
        risen[active[~below]] = True
        sample += 1
        active = active[~hit & (reach[active] > _march_distance(sample, step))]
    # The later samples, a whole step apart.  A ray above the terrain skips the
    # samples a slope bound proves it stays above until (_clear_distances): none of
    # them could shadow the point.  A ray that can never come down again is done.
    # Each ray's data is carried in arrays that shrink with the rays still marching.
    origin = start[active]
    impact = np.linalg.norm(np.cross(origin, sun_hat), axis=1)
    reach = reach[active]
    risen = risen[active]
    steps = np.full(active.size, sample - _LEAD_SAMPLES - _FIRST_STEP_SPLIT + 1, dtype=np.intp)
    while active.size:
        ray = origin + (steps * step)[:, np.newaxis] * sun_hat
        radius = np.linalg.norm(ray, axis=1)
        surface = 1.0 + field.sample_points_at(ray, radius)
        below = radius < surface
        hit = below & risen
        shadowed[active[hit]] = True
        risen |= ~below
        above = np.nonzero(~below)[0]
        clear = np.zeros(active.size)
        clear[above] = _clear_distances(
            ray[above],
            radius[above],
            surface[above],
            impact=impact[above],
            sun=sun_hat,
            field=field,
        )
        done = np.isinf(clear)
        # Every sample short of the clear distance is skipped.  A sample within
        # rounding of it lies well inside the margins the distance carries.
        skipped = np.ceil(np.minimum(clear, reach) / step)
        steps += np.where(done, 0, np.maximum(skipped, 1.0)).astype(np.intp)
        keep = ~hit & ~done & (reach > steps * step)
        active, origin, impact, reach = active[keep], origin[keep], impact[keep], reach[keep]
        risen, steps = risen[keep], steps[keep]
    return shadowed


def turned_from_sun(
    points: NDArrayFloatType,
    sun: NDArrayFloatType,
    view: NDArrayFloatType,
    field: ReliefField,
) -> NDArrayBoolType:
    """Whether the terrain at each point the camera sees turns away from the sun.

    Everything is in the body's parameter space.  Each point lies on the relief
    surface, where a line of sight crosses it.  The depth below the terrain is
    differenced over :data:`_FACET_DIFFERENCE_CELLS` of a field grid cell toward the
    sun and toward the observer, so it reads the exact bilinear terrain there rather
    than the smoothed normal the shading uses.  The terrain turns away from the sun
    where the depth grows toward the sun and shrinks toward the observer, each faster
    than :data:`_FACET_TOLERANCE` per unit of distance.  The second condition holds
    wherever the line of sight crosses into the terrain, so it keeps out a point the
    search has placed just past a crease, on terrain the camera does not see.  At
    phase 0, where the sun lies toward the observer, the two conditions cannot hold
    together, so the sun lights all the terrain the camera sees.

    Parameters:
        points: ``(N, 3)`` parameter-space positions on the relief surface.
        sun: The unit sun direction in the same space, ``(3,)``.
        view: The unit direction toward the observer in the same space, ``(3,)``.
        field: The relief realization.

    Returns:
        ``(N,)`` booleans, True where the terrain at the point faces away from the
        sun.
    """
    start = np.asarray(points, dtype=np.float64)
    if start.shape[0] == 0 or field.h_max <= field.h_min:
        return np.zeros(start.shape[0], dtype=np.bool_)
    length = 2.0 * np.pi / field.grid.shape[0] * _FACET_DIFFERENCE_CELLS
    here = field.depth_below(start)
    sun_hat = np.asarray(sun, dtype=np.float64) / float(np.linalg.norm(sun))
    view_hat = np.asarray(view, dtype=np.float64) / float(np.linalg.norm(view))
    toward_sun = (field.depth_below(start + length * sun_hat) - here) / length
    toward_view = (field.depth_below(start + length * view_hat) - here) / length
    return np.asarray((toward_sun > _FACET_TOLERANCE) & (toward_view < -_FACET_TOLERANCE))


def _march_distance(sample: int, step: float) -> float:
    """How far along its ray the cast-shadow march takes a sample.

    Parameters:
        sample: The sample's number, counted from 1.
        step: The march's step.

    Returns:
        The distance: the first :data:`_LEAD_SAMPLES` samples double in distance up
        to the first step's finer spacing, the next :data:`_FIRST_STEP_SPLIT` divide
        the first step evenly, and each later one lies a whole step beyond the one
        before.
    """
    fine = step / _FIRST_STEP_SPLIT
    if sample <= _LEAD_SAMPLES:
        return fine * 2.0 ** (sample - _LEAD_SAMPLES - 1)
    sample -= _LEAD_SAMPLES
    if sample <= _FIRST_STEP_SPLIT:
        return sample * fine
    return (sample - _FIRST_STEP_SPLIT + 1) * step


def _clear_distances(
    ray: NDArrayFloatType,
    radius: NDArrayFloatType,
    surface: NDArrayFloatType,
    *,
    impact: NDArrayFloatType,
    sun: NDArrayFloatType,
    field: ReliefField,
) -> NDArrayFloatType:
    """How far beyond samples above the terrain their rays stay above it, at least.

    A ray ``r = P + t * sun`` lies above the terrain by its clearance ``|r| - 1 - h``.
    Along the ray ``|r|`` changes by at most 1 per unit of ``t``, and the direction
    ``r_hat`` turns by ``sin(theta) / |r|``, with ``theta`` the angle between ``r``
    and the sun.  The height changes by at most :attr:`ReliefField.slope_max` per
    radian of turn, a bound that holds everywhere for the bilinear field.  So while
    the ray stays above the terrain, where ``|r| > 1 + h_min``, the clearance falls by
    at most ``1 + slope_max / (1 + h_min)`` per unit of ``t``.  Once the ray climbs,
    ``|r|`` grows by ``cos(theta)`` per unit, ``cos(theta)`` only grows, and
    ``sin(theta) / |r|`` only shrinks, so the clearance falls by at most
    ``slope_max * sin(theta) / |r| - cos(theta)`` per unit from then on, and never
    falls where that is not positive.  The bounds are inflated, and the clearance
    reduced, by far more than their rounding.

    Parameters:
        ray: ``(N, 3)`` positions of the samples, in parameter space.
        radius: Their distances from the body center, ``|r|``.
        surface: The relief surface's radius in their directions, ``1 + h``, each
            below ``radius``.
        impact: Each ray's distance from the body center at its closest approach,
            ``|P x sun|``.
        sun: The unit sun direction, ``(3,)``.
        field: The relief realization.

    Returns:
        For each sample, a distance along the ray over which it stays above the
        terrain: 0 where none is proved, and infinite where the clearance never
        falls.
    """
    along = stable_matmul3(ray, sun)
    climbing = along >= 0.0
    falling_rate = np.where(
        climbing,
        field.slope_max * impact / radius**2 * (1.0 + _CLEAR_RATE_MARGIN)
        - along / radius * (1.0 - _CLEAR_RATE_MARGIN),
        (1.0 + field.slope_max / max(1.0 + field.h_min, 1e-300)) * (1.0 + _CLEAR_RATE_MARGIN),
    )
    margin = radius - surface - _CLEARANCE_MARGIN
    clear = np.where(
        falling_rate > 0.0, np.maximum(margin, 0.0) / np.maximum(falling_rate, 1e-300), np.inf
    )
    # A sample whose margin is gone proves nothing, even on a ray that climbs clear.
    return np.where(margin > 0.0, clear, 0.0)
