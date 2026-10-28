"""Statistics of the body relief field and the terminator shadow march.

The relief-field contract pins the spectrum and the limb-slice statistics:
the field is an isotropic random field on the sphere (single-valued at its
poles, with the same statistics near the disc center as at the limb), the
limb slice's standard deviation equals the commanded RMS per-realization,
it has no content in spherical-harmonic degrees below 3 (degree 1 is a body
translation, degree 2 aliases ellipsoid shape error) or above the band
limit, and distinct seeds are distinct terrains.  The shadow march is pinned
by its inequality on a curved body
(``H_up - H_pt > d / tan(i) + d**2 / (2 R)``) and its work cap
``d_max = min((H_max - H_pt) * tan(i), sqrt(2 * R * (H_max - H_pt)))``.
"""

import functools
import math
from typing import Any

import numpy as np
import pytest
from scipy import special

from spindoctor.sim.forward.relief import (
    march_shadows,
    synthesize_relief_field,
)
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


def test_limb_delta_interpolates_the_equator_row() -> None:
    """limb_delta at node longitudes returns the equator row values exactly."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 9)
    n = field.grid.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    np.testing.assert_allclose(field.limb_delta(phi), field.grid[n // 2], atol=1e-12)


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
# Shadow march: inequality and cap.
# ---------------------------------------------------------------------------


def _flat_maps(
    size: int, *, ridge_col: int | None, ridge_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat unit-arc terrain maps with an optional tall ridge column."""
    height = np.zeros((size, size), dtype=np.float64)
    if ridge_col is not None:
        height[:, ridge_col] = ridge_h
    arc = np.ones((size, size), dtype=np.float64)
    domain = np.ones((size, size), dtype=np.bool_)
    return height, arc, domain


def test_march_shadow_inequality_both_sides() -> None:
    """A ridge shadows a downstream point iff it clears the sun ray and the curvature.

    The candidate sits at height 0 with the sun along +u; a ridge of height
    H sits ``d`` pixels upstream on a body of radius R.  The point is shadowed
    exactly when ``H > d / tan(i) + d**2 / (2 R)``.
    """
    ridge_h = 5.0
    tan_i = 4.0
    # R = 100.  The ridge 15 px upstream needs 3.75 + 1.125 = 4.875 px of rise
    # (shadowed); 30 px upstream it needs 7.5 + 4.5 = 12 px (lit).
    height, arc, domain = _flat_maps(96, ridge_col=45, ridge_h=ridge_h)
    common: dict[str, Any] = {
        'h_point': np.array([0.0, 0.0]),
        'tan_incidence': np.array([tan_i, tan_i]),
        'radius_px': np.array([100.0, 100.0]),
        'height_map': height,
        'arc_per_px_map': arc,
        'domain_mask': domain,
        'h_max': ridge_h / 100.0,
        'h_min': 0.0,
        'sun_v': 0.0,
        'sun_u': 1.0,
    }
    shadow, stats = march_shadows(np.array([48.0, 48.0]), np.array([30.5, 15.5]), **common)
    assert stats.candidate_count == 2
    assert bool(shadow[0])
    assert not bool(shadow[1])


def test_march_curvature_drop_hides_a_ridge_the_flat_test_would_count() -> None:
    """A ridge that clears the sun ray but not the body's curvature drop casts no shadow.

    The ridge 15 px upstream rises 4.5 px: above the sun ray's 15 / 4 = 3.75 px, so a
    flat-terrain test shadows the point, but below 3.75 + 15**2 / (2 * 100) = 4.875 px,
    because the curved surface falls 1.125 px below the point's horizon there.
    """
    height, arc, domain = _flat_maps(96, ridge_col=45, ridge_h=4.5)
    shadow, _stats = march_shadows(
        np.array([48.0]),
        np.array([30.5]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([4.0]),
        radius_px=np.array([100.0]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=0.045,
        h_min=0.0,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert not bool(shadow[0])


def test_march_reaches_the_highest_peak_from_a_valley() -> None:
    """A valley point sees the highest peak out to sqrt(2 R (H_max - H_pt)).

    The point sits at H_min = -4 px and a peak of H_max = 4 px stands 48 px upstream
    on a body of radius 200 with a grazing sun: the peak clears the curvature drop
    (48**2 / 400 = 5.76 < 8 px), so the point is shadowed although the peak lies past
    the horizon distance sqrt(2 R H_max) = 40 px of a point at height 0.
    """
    radius = 200.0
    h_max = 0.02
    height, arc, domain = _flat_maps(128, ridge_col=68, ridge_h=h_max * radius)
    shadow, stats = march_shadows(
        np.array([64.0]),
        np.array([20.5]),
        h_point=np.array([-h_max]),
        tan_incidence=np.array([1e6]),
        radius_px=np.array([radius]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=h_max,
        h_min=-h_max,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert bool(shadow[0])
    assert stats.max_steps == math.ceil(radius * math.sqrt(2.0 * 2.0 * h_max))


def test_march_cap_bounds_the_work() -> None:
    """The march never exceeds the cap even when the domain invites it.

    A near-terminator candidate (tan(i) enormous) on a huge flat domain
    would march thousands of steps naively; the horizon term
    ``sqrt(2 * R * H_max)`` caps the step count at its predicted bound.
    """
    size = 4096
    height = np.zeros((8, size), dtype=np.float64)
    arc = np.ones((8, size), dtype=np.float64)
    domain = np.ones((8, size), dtype=np.bool_)
    radius = 200.0
    h_max = 0.02
    # Horizon cap: R * sqrt(2 * h_max) = 40 px; the terrain term
    # (h_max - h_min) * R * tan(i) = 4 * 1e6 px would run off the domain.
    shadow, stats = march_shadows(
        np.array([4.0]),
        np.array([2.0]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([1e6]),
        radius_px=np.array([radius]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=h_max,
        h_min=-h_max,
        sun_v=0.0,
        sun_u=1.0,
    )
    predicted_cap = radius * math.sqrt(2.0 * h_max)
    assert stats.max_steps == math.ceil(predicted_cap)
    assert stats.steps_executed <= stats.max_steps
    assert not bool(shadow[0])


def test_march_ignores_terrain_beyond_the_horizon_cap() -> None:
    """A ridge past the horizon cap cannot shadow, however tall the tangent says.

    With tan(i) huge the terrain term alone would let a 60-px-distant ridge
    shadow the point; the horizon term (40 px here) excludes it.
    """
    radius = 200.0
    h_max = 0.02
    height, arc, domain = _flat_maps(128, ridge_col=80, ridge_h=h_max * radius)
    shadow, _stats = march_shadows(
        np.array([64.0]),
        np.array([20.5]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([1e6]),
        radius_px=np.array([radius]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=h_max,
        h_min=0.0,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert not bool(shadow[0])


def test_march_foreshortening_scales_surface_distance() -> None:
    """Where one image px spans two surface px, the shadow reach halves in image px.

    With ``arc_per_px = 2`` everywhere, a ridge 15 image px upstream lies 30
    surface px away, beyond the 20-surface-px shadow reach that shadowed the
    same layout at unit arc.
    """
    ridge_h = 5.0
    tan_i = 4.0
    height, arc, domain = _flat_maps(96, ridge_col=45, ridge_h=ridge_h)
    arc *= 2.0
    shadow, _stats = march_shadows(
        np.array([48.0]),
        np.array([30.5]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([tan_i]),
        radius_px=np.array([100.0]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=ridge_h / 100.0,
        h_min=0.0,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert not bool(shadow[0])


def test_march_empty_candidates_is_a_noop() -> None:
    """No candidates: no work, no shadow flags."""
    height, arc, domain = _flat_maps(8, ridge_col=None, ridge_h=0.0)
    shadow, stats = march_shadows(
        np.zeros(0),
        np.zeros(0),
        h_point=np.zeros(0),
        tan_incidence=np.zeros(0),
        radius_px=np.zeros(0),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=0.01,
        h_min=-0.01,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert shadow.size == 0
    assert stats.steps_executed == 0
