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

The cast-shadow test (:func:`cast_shadows`) works in the body's *parameter*
space, where the smooth ellipsoid is the unit sphere and the relief surface is
``r = 1 + h(r_hat)``.  The map from the body frame to that space is affine, and
an affine map carries a ray to a ray and a surface to a surface, so a ray from a
surface point toward the sun meets the terrain in parameter space exactly when
it does on the body.  A point is shadowed iff its sun ray dips below the terrain
somewhere before it leaves the relief shell ``r <= 1 + h_max``.  Which points
can be shadowed at all follows from the field's steepest slope ``s_max`` (per
radian of arc, :attr:`ReliefField.slope_max`): along the ray the terrain rises
at most ``s_max / (1 + h_min)`` per unit of the ray's own rise, so a point whose
sun elevation has ``tan(elevation) >= s_max / (1 + h_min)`` is never shadowed,
and one with ``tan(elevation) <= -s_max / (1 + h_min)`` always is.

The relief also tilts the surface normal: the relief surface's normal in
parameter space is ``r_hat - grad h / (1 + h)``, with ``grad h`` the field's
surface gradient (:meth:`ReliefField.surface_gradient`).  A pixel shows the
point where its line of sight first meets the relief surface, which
:func:`visible_surface_directions` finds.  Near the limb that point can lie well
away from where the line of sight meets the smooth sphere, and behind a ridge the
terrain is hidden altogether.

The limb is the relief surface's own outline.  With the field's poles on the
observer axis, a surface point at ``(lat, lon)`` projects to azimuth ``lon`` at
radius ``(1 + h) cos(lat)``, so the outline at azimuth ``phi`` sits at the
envelope ``max over lat of (1 + h(lat, phi)) cos(lat)``
(:meth:`ReliefField.outline_delta`).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from spindoctor.sim.shared.spherical_harmonics import draw_sh_coefficients, legendre_order_sums
from spindoctor.support.stable_math import stable_hypot, stable_matmul3
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = [
    'MAX_LIMB_RELIEF_RMS',
    'MIN_LIMB_RELIEF_CORR_DEG',
    'ReliefField',
    'cast_shadows',
    'clear_relief_caches',
    'synthesize_albedo_field',
    'synthesize_relief_field',
    'visible_surface_directions',
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
# The line-of-sight search (visible_surface_directions) steps down a meridian by
# at least this fraction of a field grid cell, and then narrows the crossing step
# this many times by linear interpolation before a last one.  Against a brute-force
# search, one narrowing puts 99% of the points within 1e-4 rad (none narrowing,
# 8e-4 rad), against a field cell of 0.05 rad at 15 degrees.
_SIGHT_STEPS_PER_CELL = 4.0
_SIGHT_REFINEMENTS = 1
# A step of that minimum length can pass over a sliver of terrain whose top pokes
# into the line of sight between two samples outside it.  Wherever the slope bound
# leaves room for one, the step is halved until its pieces are this fraction of a
# field grid cell, so a sliver the search misses is narrower than that and rises at
# most slope_max / 512 of a cell into the line of sight.
_SIGHT_SLIVER_CELLS = 1.0 / 256.0
# Steps whose slivers are searched together, which bounds the search's memory.
_SIGHT_SLIVER_CHUNK = 1 << 12


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
        norm = np.maximum(np.linalg.norm(pts, axis=-1), 1e-300)
        lat = np.arcsin(np.clip(pts[..., 2] / norm, -1.0, 1.0))
        lon = np.mod(np.arctan2(pts[..., 1], pts[..., 0]), 2.0 * np.pi)
        return self.sample(lat, lon)

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

    The bilinear field's slope along latitude is the row difference over the
    row spacing, and along longitude the column difference over the column
    arc ``cos(lat)`` times the spacing; each is maximized over the rows
    strictly between the poles (the pole rows are single points, with no
    longitude extent), and the two maxima combine in quadrature.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed.

    Returns:
        The bound for the unit-RMS realization; the slope scales with the RMS.
    """
    grid = _unit_relief_grid(corr_deg, seed)
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
# values would not.  Together they hold the stored outline within 1e-3 of the
# bilinear field's exact envelope (0.08 px on an 80 px radius) at 5% relief and
# 1.4e-3 at 10%, at a 15-degree correlation length over 8 seeds.
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
    view_x, view_z = view
    n = field.grid.shape[0]
    lat_cap = math.acos(min(1.0, (1.0 + field.h_min) / (1.0 + field.h_max)))
    half_count = math.ceil(lat_cap / (2.0 * np.pi / n) * _OUTLINE_SAMPLES_PER_ROW)
    lats = np.linspace(-lat_cap, lat_cap, 2 * half_count + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, _OUTLINE_NODES_PER_COLUMN * n + 1)
    # The limb circle's axes: the turned x axis (perpendicular to the view in the
    # x-z plane) and y, with the view direction as its pole.
    first = np.array([view_z, 0.0, -view_x])
    second = np.array([0.0, 1.0, 0.0])
    pole = np.array([view_x, 0.0, view_z])
    cos_lat = np.cos(lats)[:, np.newaxis]
    sin_lat = np.sin(lats)[:, np.newaxis]
    envelope = np.empty(lons.size, dtype=np.float64)
    chunk = max(1, _OUTLINE_CHUNK_SAMPLES // lats.size)
    for start in range(0, lons.size, chunk):
        lon = lons[start : start + chunk][np.newaxis, :]
        points = (
            (cos_lat * np.cos(lon))[..., np.newaxis] * first
            + (cos_lat * np.sin(lon))[..., np.newaxis] * second
            + np.broadcast_to(sin_lat, (lats.size, lon.size))[..., np.newaxis] * pole
        )
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
    """Whether the relief shadows each surface point from the sun, by exact ray march.

    Everything is in the body's parameter space (the smooth body is the unit
    sphere, the field's poles on the ``z`` axis).  Each point sits on the
    relief surface at ``(1 + h) * p_hat``; its ray ``P + t * sun`` is marched
    from ``t = step`` until it leaves the relief shell ``|r| <= 1 + h_max``,
    and the point is shadowed iff some sample lies below the terrain,
    ``|r| < 1 + h(r_hat)``.  The test is exact up to the step and the bilinear
    field, and it reads terrain on the far hemisphere as readily as on the
    near one.

    Parameters:
        points: ``(N, 3)`` smooth-surface directions (normalized here).
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
    p_hat = np.asarray(points, dtype=np.float64)
    p_hat = p_hat / np.maximum(np.linalg.norm(p_hat, axis=1, keepdims=True), 1e-300)
    count = p_hat.shape[0]
    shadowed = np.zeros(count, dtype=np.bool_)
    if count == 0 or field.h_max <= field.h_min:
        return shadowed
    sun_hat = np.asarray(sun, dtype=np.float64) / float(np.linalg.norm(sun))
    start = p_hat * (1.0 + field.sample_points(p_hat))[:, np.newaxis]
    # The ray leaves the shell at the positive root of |P + t s| = 1 + h_max.
    along = stable_matmul3(start, sun_hat)
    reach = -along + np.sqrt(
        np.maximum(along**2 - (np.sum(start**2, axis=1) - (1.0 + field.h_max) ** 2), 0.0)
    )
    active = np.nonzero(reach > step)[0]
    t = step
    while active.size:
        ray = start[active] + t * sun_hat
        radius = np.linalg.norm(ray, axis=1)
        below = radius < 1.0 + field.sample_points(ray)
        shadowed[active[below]] = True
        t += step
        active = active[~below & (reach[active] > t)]
    return shadowed


def _sight_gap(
    field: ReliefField, lat: NDArrayFloatType, phi: NDArrayFloatType, rho: NDArrayFloatType
) -> NDArrayFloatType:
    """How far inside the relief surface a line of sight lies at a latitude.

    Parameters:
        field: The relief realization.
        lat: Latitudes along each line of sight's meridian.
        phi: The meridians' longitudes, the same shape as ``lat``.
        rho: The lines of sight's distances from the observer axis, the same shape.

    Returns:
        ``(1 + h(lat, phi)) cos(lat) - rho``, positive inside the surface.
    """
    return np.asarray((1.0 + field.sample(lat, phi)) * np.cos(lat) - rho, dtype=np.float64)


@dataclass(frozen=True)
class _Crossing:
    """The highest crossing found in each of a set of latitude intervals.

    Parameters:
        found: Whether the interval holds a crossing the search found.
        upper: The upper end of the crossing's bracket, outside the surface.
        lower: The lower end of the bracket, inside the surface.
        g_upper: The line of sight's gap (:func:`_sight_gap`) at ``upper``.
        g_lower: The gap at ``lower``.
    """

    found: NDArrayBoolType
    upper: NDArrayFloatType
    lower: NDArrayFloatType
    g_upper: NDArrayFloatType
    g_lower: NDArrayFloatType


def _latitude_slope_bound(
    row_slope: NDArrayFloatType,
    ends: tuple[NDArrayFloatType, NDArrayFloatType],
    phi: NDArrayFloatType,
) -> NDArrayFloatType:
    """A bound on the field's latitude slope along short stretches of meridians.

    Along a meridian the bilinear field is linear between grid rows, with a slope
    between those of the two grid columns beside it.  A stretch no longer than a
    grid cell touches at most two row gaps, so the largest of the four slopes there
    bounds the field's slope along it.

    Parameters:
        row_slope: ``|h(row + 1) - h(row)|`` over the row spacing, at every grid node.
        ends: The stretches' upper and lower latitudes, at most a grid cell apart.
        phi: The meridians' longitudes.

    Returns:
        The bound for each stretch, per radian of latitude.
    """
    n = row_slope.shape[0]
    scale = n / (2.0 * np.pi)
    col = np.floor(np.mod(phi, 2.0 * np.pi) * scale).astype(np.intp) % n
    col_next = (col + 1) % n
    rows = [np.floor((lat + np.pi) * scale).astype(np.intp) % n for lat in ends]
    return np.asarray(
        np.max([row_slope[row, c] for row in rows for c in (col, col_next)], axis=0),
        dtype=np.float64,
    )


def _first_crossing_between(
    field: ReliefField,
    rho: NDArrayFloatType,
    phi: NDArrayFloatType,
    ends: tuple[NDArrayFloatType, NDArrayFloatType],
    gaps: tuple[NDArrayFloatType, NDArrayFloatType],
    *,
    slope: NDArrayFloatType,
    min_width: float,
) -> _Crossing:
    """Find the highest crossing in latitude intervals whose ends lie outside the surface.

    Over an interval the gap ``g`` changes by at most ``s + (1 + h_max) |sin(lat)|``
    per radian, where ``s`` bounds the field's latitude slope along the interval and
    ``|sin(lat)|`` is the larger of its two ends'.  So it can reach 0 inside the
    interval only if its two end values plus that bound times the width add up to at
    least 0.  An interval that passes this test is halved at a
    sample.  A midpoint inside the surface brackets a crossing with the upper end;
    otherwise both halves are tested again, down to pieces narrower than
    ``min_width``, and the highest bracket found is kept.  The intervals are searched
    in chunks of :data:`_SIGHT_SLIVER_CHUNK`, which bounds the memory the search
    takes.

    Parameters:
        field: The relief realization.
        rho: Each line of sight's distance from the observer axis.
        phi: Each line of sight's longitude.
        ends: The intervals' upper and lower latitudes.
        gaps: The gap at the upper and lower ends, both negative.
        slope: For each interval, a bound on the field's latitude slope along it
            (:func:`_latitude_slope_bound`), which also holds for its pieces.
        min_width: The narrowest piece searched, in radians.

    Returns:
        For each interval, the bracket of the highest crossing found.
    """
    count = rho.size
    found = np.zeros(count, dtype=np.bool_)
    upper = np.zeros(count)
    lower = np.zeros(count)
    g_upper = np.zeros(count)
    g_lower = np.zeros(count)
    top = 1.0 + field.h_max
    for start in range(0, count, _SIGHT_SLIVER_CHUNK):
        which = np.arange(start, min(start + _SIGHT_SLIVER_CHUNK, count))
        a, b = ends[0][which], ends[1][which]
        g_a, g_b = gaps[0][which], gaps[1][which]
        while which.size:
            width = a - b
            bound = slope[which] + top * np.maximum(np.abs(np.sin(a)), np.abs(np.sin(b)))
            live = (width >= min_width) & (g_a + g_b + bound * width >= 0.0)
            # A piece below a crossing already found can hold only a lower one.
            live &= ~(found[which] & (a <= lower[which]))
            which, a, b, g_a, g_b = which[live], a[live], b[live], g_a[live], g_b[live]
            mid = 0.5 * (a + b)
            g_mid = _sight_gap(field, mid, phi[which], rho[which])
            hit = g_mid >= 0.0
            # Keep the highest bracket per interval: write the hits from low to high,
            # so the last write to each interval is its highest.
            order = np.nonzero(hit)[0]
            order = order[np.argsort(a[order], kind='stable')]
            order = order[~found[which[order]] | (a[order] > upper[which[order]])]
            target = which[order]
            found[target] = True
            upper[target] = a[order]
            lower[target] = mid[order]
            g_upper[target] = g_a[order]
            g_lower[target] = g_mid[order]
            rest = ~hit
            which = np.concatenate([which[rest], which[rest]])
            a, b = np.concatenate([a[rest], mid[rest]]), np.concatenate([mid[rest], b[rest]])
            g_a = np.concatenate([g_a[rest], g_mid[rest]])
            g_b = np.concatenate([g_mid[rest], g_b[rest]])
    return _Crossing(found, upper, lower, g_upper, g_lower)


def visible_surface_directions(
    x: NDArrayFloatType, y: NDArrayFloatType, field: ReliefField
) -> NDArrayFloatType:
    """Where each line of sight first meets the relief surface, as unit directions.

    Everything is in the body's parameter space (the smooth body is the unit sphere,
    the field's poles on the ``z`` axis), with the observer far out on the ``+z``
    axis.  The line of sight through ``(x, y)`` runs parallel to ``z`` and stays in
    the meridian plane at longitude ``phi = atan2(y, x)``.  At latitude ``lat`` it
    lies at radius ``rho / cos(lat)``, with ``rho = hypot(x, y)``, so it is inside the
    relief surface ``r = 1 + h`` where ``g(lat) = (1 + h(lat, phi)) cos(lat) - rho``
    is positive.  The first point it meets is the highest latitude where ``g``
    reaches 0.

    The search walks down the meridian from the highest latitude the terrain can
    reach, ``acos(rho / (1 + h_max))``.  Each step is as long as the field's slope
    bound allows without passing a crossing, and at least a quarter of a field grid
    cell, so the walk resolves the terrain at its own resolution whatever the body's
    size in pixels.  A step of that minimum length can pass over a sliver of terrain
    between two samples outside it, so wherever the slope bound leaves room for one
    the step is searched by halving (:func:`_first_crossing_between`).  The step
    that crosses is then narrowed by linear interpolation of ``g``.

    A line of sight that misses the surface, outside its outline, reads the latitude
    of the walk's sample that came closest to the surface, the one with the largest
    ``g``.  For a pixel just outside the outline that is the outline point beside
    it, to the walk's quarter-cell spacing.  A line of sight with ``rho`` at or
    beyond ``1 + h_max`` reads latitude 0, the limb direction.

    Parameters:
        x: Parameter-space coordinates of the lines of sight along the body's first
            axis (any shape).
        y: The coordinates along its second axis, the same shape as ``x``.
        field: The relief realization.

    Returns:
        ``(..., 3)`` unit directions of the points seen, ``x.shape + (3,)``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    shape = x_arr.shape
    x_flat = x_arr.ravel()
    y_flat = np.asarray(y, dtype=np.float64).ravel()
    rho = stable_hypot(x_flat, y_flat)
    phi = np.mod(np.arctan2(y_flat, x_flat), 2.0 * np.pi)
    top = 1.0 + field.h_max
    bottom = 1.0 + field.h_min
    # No point of the surface is inside the line of sight above lat_top, and every
    # line of sight with rho below 1 + h_min is inside it by lat_low.  A line of
    # sight farther out can meet only terrain no lower than -lat_top.
    lat_top = np.arccos(np.minimum(rho / top, 1.0))
    lat_low = np.where(rho < bottom, np.arccos(np.minimum(rho / bottom, 1.0)), -lat_top)
    min_step = 2.0 * np.pi / field.grid.shape[0] / _SIGHT_STEPS_PER_CELL

    def inside(lat: NDArrayFloatType, index: NDArrayIntType) -> NDArrayFloatType:
        return _sight_gap(field, lat, phi[index], rho[index])

    # hi is the last latitude outside the surface and lo the first inside it.
    hi = lat_top.copy()
    lo = lat_top.copy()
    g_hi = np.full(rho.size, -np.inf)
    g_lo = np.zeros(rho.size)
    found = np.zeros(rho.size, dtype=np.bool_)
    best_lat = np.zeros(rho.size)
    best_g = np.full(rho.size, -np.inf)
    lat = lat_top.copy()
    # Whether the step that led to each line of sight's current sample was longer
    # than the slope bound allows, and so could have passed over a sliver.
    forced = np.zeros(rho.size, dtype=np.bool_)
    sliver_width = 2.0 * np.pi / field.grid.shape[0] * _SIGHT_SLIVER_CELLS
    row_slope = np.abs(np.roll(field.grid, -1, axis=0) - field.grid) * (
        field.grid.shape[0] / (2.0 * np.pi)
    )
    active = np.arange(rho.size)
    while active.size:
        lat_a = lat[active]
        g = inside(lat_a, active)
        closer = g > best_g[active]
        best_g[active[closer]] = g[closer]
        best_lat[active[closer]] = lat_a[closer]
        crossed = g >= 0.0
        found[active[crossed]] = True
        lo[active[crossed]] = lat_a[crossed]
        g_lo[active[crossed]] = g[crossed]
        # A forced step that ends outside the surface may have passed over a sliver.
        suspect = np.nonzero(~crossed & forced[active])[0]
        if suspect.size:
            pix = active[suspect]
            ends = (hi[pix], lat_a[suspect])
            sliver = _first_crossing_between(
                field,
                rho[pix],
                phi[pix],
                ends,
                (g_hi[pix], g[suspect]),
                slope=_latitude_slope_bound(row_slope, ends, phi[pix]),
                min_width=sliver_width,
            )
            inner = pix[sliver.found]
            crossed[suspect[sliver.found]] = True
            found[inner] = True
            hi[inner] = sliver.upper[sliver.found]
            g_hi[inner] = sliver.g_upper[sliver.found]
            lo[inner] = sliver.lower[sliver.found]
            g_lo[inner] = sliver.g_lower[sliver.found]
        walking = ~crossed & (lat_a > lat_low[active])
        moving = active[walking]
        hi[moving] = lat_a[walking]
        g_hi[moving] = g[walking]
        # Going down the meridian g rises by at most slope_max + (1 + h_max) sin(lat)
        # per radian, so a step of -g over that bound cannot pass a crossing.
        bound = field.slope_max + top * np.maximum(np.sin(lat_a[walking]), 0.0)
        safe = -g[walking] / np.maximum(bound, 1e-300)
        lat[moving] = np.maximum(lat_a[walking] - np.maximum(safe, min_step), lat_low[moving])
        forced[moving] = lat_a[walking] - lat[moving] > safe
        active = moving

    hit = np.where(found, lo, best_lat)
    # A crossing found at lat_top itself needs no narrowing; elsewhere g goes from
    # negative at hi to non-negative at lo.
    narrow = np.nonzero(found & np.isfinite(g_hi))[0]
    for _ in range(_SIGHT_REFINEMENTS):
        if narrow.size == 0:
            break
        a, b = hi[narrow], lo[narrow]
        ga, gb = g_hi[narrow], g_lo[narrow]
        mid = a + (b - a) * (-ga / np.maximum(gb - ga, 1e-300))
        g_mid = inside(mid, narrow)
        crossed = g_mid >= 0.0
        lo[narrow[crossed]] = mid[crossed]
        g_lo[narrow[crossed]] = g_mid[crossed]
        hi[narrow[~crossed]] = mid[~crossed]
        g_hi[narrow[~crossed]] = g_mid[~crossed]
    if narrow.size:
        a, b = hi[narrow], lo[narrow]
        ga, gb = g_hi[narrow], g_lo[narrow]
        hit[narrow] = a + (b - a) * (-ga / np.maximum(gb - ga, 1e-300))

    cos_lat = np.cos(hit)
    directions = np.stack([cos_lat * np.cos(phi), cos_lat * np.sin(phi), np.sin(hit)], axis=-1)
    return directions.reshape(*shape, 3)
