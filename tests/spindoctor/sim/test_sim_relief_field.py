"""Statistics of the body relief field and the terminator shadow march.

The relief-field contract pins the spectrum and the limb-slice statistics:
the field is an isotropic random field on the sphere (single-valued at its
poles, with the same statistics near the disc center as at the limb), the
limb slice's standard deviation equals the commanded RMS per-realization,
it has no content in spherical-harmonic degrees below 3 (degree 1 is a body
translation, degree 2 aliases ellipsoid shape error) or above the band
limit, and distinct seeds are distinct terrains.  The outline envelope, the
slope bound, and the surface gradient are pinned against brute-force
evaluations of the same field, and the cast-shadow ray march against the
exact geometry of a ray grazing a ridge.
"""

import functools
import math

import numpy as np
import pytest
from scipy import special

from spindoctor.sim.forward.relief import ReliefField, cast_shadows, synthesize_relief_field
from spindoctor.support.types import NDArrayComplexType

_RMS = 0.01
_CORR_DEG = 15.0


def test_limb_slice_rms_matches_commanded_over_100_seeds() -> None:
    """Every realization's limb-slice standard deviation equals the commanded RMS.

    The field is rescaled per-realization after the low-degree zeroing, so
    each seed's limb slice carries the commanded RMS; 100 seeds bound the
    worst case well inside the 10% acceptance band.
    """
    worst = 0.0
    for seed in range(100):
        field = synthesize_relief_field(_RMS, _CORR_DEG, seed)
        n = field.grid.shape[0]
        limb_std = float(np.std(field.grid[n // 2]))
        worst = max(worst, abs(limb_std - _RMS) / _RMS)
    assert worst < 0.10
    # The per-realization normalization makes the match essentially exact.
    assert worst < 1e-9


_PROJECTION_LAT = np.arcsin(np.polynomial.legendre.leggauss(96)[0])
_PROJECTION_LON = np.arange(192) * (2.0 * np.pi / 192)
_PROJECTION_WEIGHT = np.polynomial.legendre.leggauss(96)[1][:, np.newaxis] * (2.0 * np.pi / 192)


@functools.cache
def _projection_harmonic(ell: int, m: int) -> NDArrayComplexType:
    """``Y_lm`` on the projection grid (Gauss-Legendre latitudes, uniform longitudes)."""
    lat2, lon2 = np.meshgrid(_PROJECTION_LAT, _PROJECTION_LON, indexing='ij')
    return np.asarray(special.sph_harm_y(ell, m, np.pi / 2.0 - lat2, lon2), dtype=np.complex128)


def _harmonic_coefficient_power(rms: float, seed: int, degrees: list[int]) -> dict[int, float]:
    """Mean squared real spherical-harmonic coefficient per degree of one realization.

    The field is sampled at Gauss-Legendre latitudes and uniform longitudes, so the
    projection is exact up to the field's own bilinear sampling error.  Orders
    ``m >= 1`` are projected on ``sqrt(2)`` times the real and imaginary parts of
    ``Y_lm``, the orthonormal real basis, so each coefficient's expected square is
    the degree's variance.
    """
    field = synthesize_relief_field(rms, _CORR_DEG, seed)
    lat2, lon2 = np.meshgrid(_PROJECTION_LAT, _PROJECTION_LON, indexing='ij')
    weighted = field.sample(lat2, lon2) * _PROJECTION_WEIGHT
    result: dict[int, float] = {}
    for ell in degrees:
        coeffs: list[float] = []
        for m in range(ell + 1):
            projection = complex(np.sum(weighted * _projection_harmonic(ell, m)))
            if m == 0:
                coeffs.append(projection.real)
            else:
                coeffs.extend([math.sqrt(2.0) * projection.real, math.sqrt(2.0) * projection.imag])
        result[ell] = float(np.mean(np.square(coeffs)))
    return result


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_low_degree_content_is_zero(seed: int) -> None:
    """Degrees 0, 1, and 2 carry no relief: no mean, no translation, no shape error.

    Each low-degree coefficient is compared with the typical coefficient of degrees
    3-6, which the field does carry; the residual is the bilinear sampling error.
    """
    power = _harmonic_coefficient_power(1.0, seed, list(range(7)))
    band = float(np.sqrt(np.mean([power[ell] for ell in range(3, 7)])))
    assert band > 0.0
    for ell in range(3):
        assert math.sqrt(power[ell]) < 0.02 * band, f'degree {ell}'


def test_degree_power_follows_the_gaussian_spectrum() -> None:
    """Each degree's power falls as ``exp(-(l * corr_rad / 2)**2)`` across the band.

    The mean squared real-harmonic coefficient of degrees 3 to 12, averaged over 16
    realizations, is fit by ``log(power) = a + b * l**2``; the slope ``b`` must be
    ``-(corr_rad / 2)**2`` within 10%.  A spectrum twice as steep, or flat, misses by
    100%.
    """
    degrees = list(range(3, 13))
    power = np.mean(
        [list(_harmonic_coefficient_power(1.0, seed, degrees).values()) for seed in range(16)],
        axis=0,
    )
    slope = float(np.polyfit(np.square(degrees), np.log(power), 1)[0])
    assert slope == pytest.approx(-((math.radians(_CORR_DEG) / 2.0) ** 2), rel=0.1)


def test_limb_slice_has_no_power_beyond_band_limit() -> None:
    """The limb slice carries no longitude wavenumber above lmax = ceil(8 / corr_rad).

    On the equator every harmonic of degree l is a sinusoid of order m <= l, so the
    band limit in degree is exactly a band limit of the limb slice.
    """
    field = synthesize_relief_field(_RMS, _CORR_DEG, 7)
    n = field.grid.shape[0]
    lmax = math.ceil(8.0 / math.radians(_CORR_DEG))
    spectrum = np.abs(np.fft.rfft(field.grid[n // 2]))
    assert lmax + 1 < spectrum.size
    assert float(spectrum[lmax + 1 :].max()) == pytest.approx(0.0, abs=1e-9 * float(spectrum.max()))


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_field_is_single_valued_at_its_poles(seed: int) -> None:
    """Every longitude reaches the same height at a pole.

    The ellipsoid renderer puts a pole at the disc center: a field that depended on
    longitude there would cut radial cliffs through the middle of the disc.
    """
    field = synthesize_relief_field(_RMS, _CORR_DEG, seed)
    lon = np.arange(64) * (2.0 * np.pi / 64)
    for pole in (np.pi / 2.0, -np.pi / 2.0):
        ring = field.sample(np.full(64, pole - math.copysign(1e-4, pole)), lon)
        assert float(np.ptp(ring)) < 0.01 * _RMS


def _structure_function(field_seed: int, lat_band: tuple[float, float], lag: float) -> float:
    """Mean squared height difference of point pairs ``lag`` radians apart.

    The first point of each pair is uniform over the latitude band (uniform in area);
    the second lies ``lag`` away along a random direction on the sphere.
    """
    field = synthesize_relief_field(_RMS, _CORR_DEG, field_seed)
    rng = np.random.default_rng(1000 + field_seed)
    count = 4000
    z = rng.uniform(math.sin(lat_band[0]), math.sin(lat_band[1]), count)
    lon = rng.uniform(0.0, 2.0 * np.pi, count)
    ring = np.sqrt(1.0 - z * z)
    start = np.stack([ring * np.cos(lon), ring * np.sin(lon), z], axis=1)
    tangent = rng.standard_normal((count, 3))
    tangent -= np.sum(tangent * start, axis=1)[:, np.newaxis] * start
    tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
    end = start * math.cos(lag) + tangent * math.sin(lag)

    def height(points: np.ndarray) -> np.ndarray:
        lat = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
        return field.sample(lat, np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi))

    return float(np.mean((height(start) - height(end)) ** 2))


def test_field_is_as_rough_at_the_disc_center_as_at_the_limb() -> None:
    """Height differences at a fixed arc match near the poles and near the equator.

    The renderer's disc center is the field's pole and its limb the equator.  A
    field built on a (lat, lon) map stretches its longitude wavelengths by
    ``1 / cos(lat)`` toward the pole, so the disc center would be far rougher than
    the limb (about 4x).  A 20-degree cap holds only a few correlation lengths, so
    one realization's cap estimate is noisy: the check reads both caps and averages
    48 realizations, over which the caps and the limb band agree within 25%.
    """
    lag = math.radians(_CORR_DEG) / 2.0
    caps = [
        _structure_function(seed, band, lag)
        for seed in range(48)
        for band in (
            (math.radians(70.0), math.radians(89.99)),
            (math.radians(-89.99), math.radians(-70.0)),
        )
    ]
    limb = [
        _structure_function(seed, (math.radians(-20.0), math.radians(20.0)), lag)
        for seed in range(48)
    ]
    ratio = float(np.mean(caps)) / float(np.mean(limb))
    assert 0.8 < ratio < 1.25


def test_band_power_is_present() -> None:
    """The surviving degrees [3, lmax] actually carry the field's power."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 7)
    assert float(np.std(field.grid)) > 0.0
    assert field.h_max > 0.0
    assert field.h_min < 0.0


def test_limb_correlation_length_tracks_the_knob() -> None:
    """A longer correlation length yields a limb slice with fewer zero crossings."""

    def crossings(corr_deg: float) -> int:
        field = synthesize_relief_field(_RMS, corr_deg, 11)
        _phi, values = field.limb_slice()
        signs = np.sign(values[:-1])
        return int(np.sum(signs[1:] != signs[:-1]))

    assert crossings(30.0) < crossings(5.0)


def test_distinct_seeds_are_distinct_terrains() -> None:
    """Fresh coefficient draws per seed: two seeds never share a realization."""
    first = synthesize_relief_field(_RMS, _CORR_DEG, 1)
    second = synthesize_relief_field(_RMS, _CORR_DEG, 2)
    assert not np.array_equal(first.grid, second.grid)


def test_same_seed_is_deterministic() -> None:
    """The same (rms, corr, seed) triple reproduces the same field exactly."""
    first = synthesize_relief_field(_RMS, _CORR_DEG, 3)
    second = synthesize_relief_field(_RMS, _CORR_DEG, 3)
    assert np.array_equal(first.grid, second.grid)


def test_rms_zero_is_the_zero_field() -> None:
    """RMS 0 (the default) is exactly the flat, relief-free surface."""
    field = synthesize_relief_field(0.0, _CORR_DEG, 5)
    assert float(np.abs(field.grid).max()) == 0.0


@pytest.mark.parametrize('rms', [0.02, 0.05])
def test_the_outline_reaches_at_least_the_equator_slice(rms: float) -> None:
    """At every node longitude the outline is at or beyond the equator slice."""
    field = synthesize_relief_field(rms, _CORR_DEG, 9)
    n = field.grid.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    assert bool(np.all(field.outline_delta(phi) >= field.grid[n // 2] - 1e-12))


def test_the_outline_is_the_envelope_over_latitude() -> None:
    """The outline matches a brute-force maximum of (1 + h) cos(lat) over latitude."""
    field = synthesize_relief_field(0.05, _CORR_DEG, 9)
    phi = np.linspace(0.0, 2.0 * np.pi, 97)[:-1]
    lats = np.linspace(-0.8, 0.8, 4001)
    lat_grid, phi_grid = np.meshgrid(lats, phi, indexing='ij')
    brute = ((1.0 + field.sample(lat_grid, phi_grid)) * np.cos(lat_grid)).max(axis=0) - 1.0
    np.testing.assert_allclose(field.outline_delta(phi), brute, atol=5e-4)


def test_the_outline_departs_from_the_equator_slice_at_high_rms() -> None:
    """At 5% relief the outline differs from the equator slice by a sizable share of the RMS."""
    field = synthesize_relief_field(0.05, _CORR_DEG, 9)
    n = field.grid.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    gap = field.outline_delta(phi) - field.grid[n // 2]
    assert float(np.sqrt(np.mean(gap**2))) > 0.3 * field.rms


def test_an_albedo_field_draws_no_outline() -> None:
    """Asking a field without an outline for one fails loudly."""
    field = ReliefField(grid=np.zeros((64, 64)), rms=0.0, corr_deg=15.0, h_min=0.0, h_max=0.0)
    with pytest.raises(ValueError, match='no outline envelope'):
        field.outline_delta(np.zeros(3))


def _random_directions(count: int, seed: int) -> np.ndarray:
    """Uniform random unit vectors."""
    points = np.random.default_rng(seed).normal(size=(count, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def test_the_slope_bound_covers_every_measured_gradient() -> None:
    """No surface gradient anywhere on the sphere exceeds the field's slope bound."""
    field = synthesize_relief_field(0.05, _CORR_DEG, 9)
    gradient = field.surface_gradient(_random_directions(20000, 1))
    assert float(np.linalg.norm(gradient, axis=1).max()) <= field.slope_max


def test_the_slope_bound_scales_with_the_rms() -> None:
    """Doubling the RMS doubles the slope bound."""
    single = synthesize_relief_field(0.02, _CORR_DEG, 9).slope_max
    double = synthesize_relief_field(0.04, _CORR_DEG, 9).slope_max
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_the_surface_gradient_matches_finite_differences_in_latitude_and_longitude() -> None:
    """Away from the poles the gradient's components are the field's lat and lon slopes."""
    field = synthesize_relief_field(0.05, _CORR_DEG, 9)
    rng = np.random.default_rng(2)
    lat = rng.uniform(-1.0, 1.0, 200)
    lon = rng.uniform(0.0, 2.0 * np.pi, 200)
    points = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=1)
    gradient = field.surface_gradient(points)
    delta = 1e-4
    d_lat = (field.sample(lat + delta, lon) - field.sample(lat - delta, lon)) / (2.0 * delta)
    d_lon = (field.sample(lat, lon + delta) - field.sample(lat, lon - delta)) / (
        2.0 * delta * np.cos(lat)
    )
    scale = field.slope_max
    np.testing.assert_allclose(np.sum(gradient * north, axis=1), d_lat, atol=0.15 * scale)
    np.testing.assert_allclose(np.sum(gradient * east, axis=1), d_lon, atol=0.15 * scale)


def test_sample_wraps_periodically() -> None:
    """Sampling wraps in both coordinates (the field lives on a torus)."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 9)
    lat = np.array([0.3])
    lon = np.array([1.1])
    np.testing.assert_allclose(
        field.sample(lat, lon), field.sample(lat + 2.0 * np.pi, lon + 2.0 * np.pi), atol=1e-12
    )


def test_sample_reads_one_surface_point_by_either_name() -> None:
    """(lat, lon) and (pi - lat, lon + pi) name one point and sample one height.

    The torus covers the sphere twice; past a pole the grid must continue onto the
    far side of the same surface.
    """
    field = synthesize_relief_field(_RMS, _CORR_DEG, 9)
    rng = np.random.default_rng(4)
    lat = rng.uniform(-np.pi / 2.0, np.pi / 2.0, 200)
    lon = rng.uniform(0.0, 2.0 * np.pi, 200)
    np.testing.assert_allclose(
        field.sample(np.pi - lat, lon + np.pi), field.sample(lat, lon), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Cast shadows: the exact ray march.
# ---------------------------------------------------------------------------

_RIDGE_GRID = 512


def _ridge_field(height: float) -> ReliefField:
    """Flat terrain with a ``height`` ridge from longitude 9 to 11 degrees."""
    grid = np.zeros((_RIDGE_GRID, _RIDGE_GRID), dtype=np.float64)
    lon = np.arange(_RIDGE_GRID) * (360.0 / _RIDGE_GRID)
    grid[:, (lon >= 9.0) & (lon <= 11.0)] = height
    return ReliefField(
        grid=grid, rms=height, corr_deg=1.0, h_min=0.0, h_max=height, slope_max=100.0
    )


def _equator_point_shadowed(height: float, elevation_deg: float) -> bool:
    """Whether the ridge shadows the equator point at longitude 0, sun toward +lon.

    The point sits on flat ground, so its ray toward the sun is the line through
    ``(1, 0)`` at ``elevation_deg`` above the tangent; in polar form it lies at
    radius ``cos(e) / cos(lon + e)``, 1.0267 at the ridge's near edge for
    ``e`` = 5 degrees.
    """
    e = math.radians(elevation_deg)
    sun = np.array([math.sin(e), math.cos(e), 0.0])
    shadowed = cast_shadows(
        np.array([[1.0, 0.0, 0.0]]), sun, _ridge_field(height), step=2.0 * np.pi / 8192
    )
    return bool(shadowed[0])


def test_a_ridge_above_the_ray_casts_a_shadow() -> None:
    """A 4% ridge rises above the 5 degree ray's 2.67% at its near edge."""
    assert _equator_point_shadowed(0.04, 5.0)


def test_a_ridge_below_the_ray_casts_none() -> None:
    """A 2% ridge stays under the same ray."""
    assert not _equator_point_shadowed(0.02, 5.0)


def test_a_higher_sun_clears_the_ridge() -> None:
    """At 15 degrees the ray stands at 6.3% over the ridge, clearing the 4% one."""
    assert not _equator_point_shadowed(0.04, 15.0)


def test_flat_terrain_casts_no_shadow() -> None:
    """A field with no height range shadows nothing."""
    flat = ReliefField(grid=np.full((64, 64), 0.01), rms=0.0, corr_deg=15.0, h_min=0.01, h_max=0.01)
    sun = np.array([0.1, 0.995, 0.0])
    assert not cast_shadows(_random_directions(50, 3), sun, flat, step=0.01).any()


def test_a_finer_step_changes_few_shadow_decisions() -> None:
    """Halving an eighth-cell step flips under 3% of a realization's cast shadows."""
    field = synthesize_relief_field(0.035, 12.0, 2)
    points = _random_directions(20000, 4)
    points[:, 2] = np.abs(points[:, 2])
    sun = np.array([0.5, 0.2, 0.84])
    sun /= np.linalg.norm(sun)
    cell = 2.0 * np.pi / field.grid.shape[0]
    coarse = cast_shadows(points, sun, field, step=cell / 8.0)
    fine = cast_shadows(points, sun, field, step=cell / 16.0)
    assert int(fine.sum()) > 200
    assert int((coarse != fine).sum()) < 0.03 * int(fine.sum())
