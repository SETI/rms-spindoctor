"""Acceptance checks for the multiplicative body albedo texture.

The contract: the noise field carries the commanded global RMS and
correlation scale, spots land where commanded (in the observer-centered
surface frame) with their commanded factor, the texture is multiplicative
on the shading (never additive), and the silhouette is untouched.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.body_texture import AlbedoTextureSpec
from spindoctor.sim.forward.body_topo import TopoBodySpec, create_topographic_body
from spindoctor.sim.forward.relief import synthesize_albedo_field
from spindoctor.sim.render import render_combined_model
from spindoctor.support.types import NDArrayFloatType

_SIZE = 200
_CENTER = _SIZE / 2.0
_RADIUS = 80.0
# Phase 1 rad keeps the smooth shading well below the signal ceiling, so a
# multiplied texture can never clip at 1 and the ratio probe stays exact.
_PHASE = 1.0


def _spec(albedo: AlbedoTextureSpec | None) -> TopoBodySpec:
    """A sphere at moderate phase with an optional albedo texture."""
    return TopoBodySpec(
        axis1=2.0 * _RADIUS,
        axis2=2.0 * _RADIUS,
        axis3=2.0 * _RADIUS,
        phase_angle=_PHASE,
        albedo_texture=albedo,
    )


def _render(albedo: AlbedoTextureSpec | None) -> NDArrayFloatType:
    return create_topographic_body((_SIZE, _SIZE), (_CENTER, _CENTER), _spec(albedo))


def _sphere_moments(grid: NDArrayFloatType) -> tuple[float, float]:
    """Area-weighted mean and RMS of a torus-stored field over the sphere it covers once."""
    n = grid.shape[0]
    rows = np.arange(n // 4, 3 * n // 4 + 1)
    weights = np.cos(-np.pi + 2.0 * np.pi * rows / n)[:, np.newaxis]
    total = float(np.sum(weights)) * n
    mean = float(np.sum(weights * grid[rows])) / total
    return mean, float(np.sqrt(np.sum(weights * grid[rows] ** 2) / total))


def test_albedo_field_rms_matches_commanded() -> None:
    """The field's area-weighted RMS over the sphere equals the commanded RMS."""
    field = synthesize_albedo_field(0.15, 10.0, 42)
    assert _sphere_moments(field.grid)[1] == pytest.approx(0.15, abs=1e-12)


def test_albedo_field_mean_is_zero() -> None:
    """The degree-0 term is zeroed, so the field averages 0 over the sphere."""
    field = synthesize_albedo_field(0.15, 10.0, 42)
    assert abs(_sphere_moments(field.grid)[0]) < 1e-5


@pytest.mark.parametrize('seed', [3, 11])
def test_albedo_field_is_single_valued_at_its_poles(seed: int) -> None:
    """Around each pole the field closes on one value, so no pinwheel shows on the limb."""
    field = synthesize_albedo_field(0.15, 10.0, seed)
    lons = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    for pole in (np.pi / 2.0, -np.pi / 2.0):
        ring = field.sample(np.full_like(lons, pole - np.sign(pole) * 1e-6), lons)
        assert float(np.ptp(ring)) < 1e-4 * field.rms


def test_albedo_field_correlation_scales_with_corr() -> None:
    """A longer correlation length yields a smoother field (less gradient energy)."""
    fine = synthesize_albedo_field(0.1, 3.0, 7)
    coarse = synthesize_albedo_field(0.1, 30.0, 7)

    def grad_energy(grid: NDArrayFloatType) -> float:
        # Per radian of surface arc, so grids of different resolution compare.
        gv, gu = np.gradient(grid)
        cells_per_radian = grid.shape[0] / (2.0 * np.pi)
        return float(np.mean(gv * gv + gu * gu)) * float(cells_per_radian**2)

    assert grad_energy(fine.grid) > 10.0 * grad_energy(coarse.grid)


def test_noise_texture_is_multiplicative_on_the_shading() -> None:
    """The textured/smooth ratio is 1 + field: same statistics at any shading level."""
    albedo = AlbedoTextureSpec(rms=0.2, corr_px=6.0, seed=99)
    smooth = _render(None)
    textured = _render(albedo)
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE].astype(float)
    r = np.hypot(vv + 0.5 - _CENTER, uu + 0.5 - _CENTER)
    ratio = np.divide(textured, smooth, out=np.ones_like(smooth), where=smooth > 0)
    disc = smooth > 0
    # Global multiplicative statistics: mean 1, spread = the commanded RMS.
    assert float(np.mean(ratio[disc])) == pytest.approx(1.0, abs=0.03)
    assert float(np.std(ratio[disc])) == pytest.approx(0.2, abs=0.04)
    # Multiplicativity probe: the ratio spread is the same in a bright inner
    # region and a dim outer annulus; an additive texture would inflate the
    # dim band's relative spread by the shading contrast.
    bright = ratio[disc & (r < 0.4 * _RADIUS)]
    dim = ratio[disc & (r > 0.75 * _RADIUS) & (r < 0.9 * _RADIUS)]
    assert float(np.std(bright)) == pytest.approx(0.2, abs=0.06)
    assert float(np.std(dim)) == pytest.approx(0.2, abs=0.06)


def test_spot_lands_at_commanded_surface_position() -> None:
    """A dark spot at (lat, lon) darkens exactly the predicted disc location."""
    lat_deg = 40.0
    albedo = AlbedoTextureSpec(rms=0.0, spots=((lat_deg, 90.0, 10.0, 0.5),), seed=0)
    smooth = _render(None)
    textured = _render(albedo)
    # Body-polar frame: lat = arcsin(x) with x = v_rot / a, lon 90 = the
    # sub-observer meridian, so the spot sits at v = center + R * sin(lat).
    spot_v = int(_CENTER + _RADIUS * math.sin(math.radians(lat_deg)))
    spot_u = int(_CENTER)
    assert textured[spot_v, spot_u] / smooth[spot_v, spot_u] == pytest.approx(0.5, abs=1e-9)
    # The mirror latitude (lat -40) is untouched, so the spot is a localized
    # mark, not a global rescale.
    anti_v = int(_CENTER - _RADIUS * math.sin(math.radians(lat_deg)))
    assert textured[anti_v, spot_u] == smooth[anti_v, spot_u]


def test_spot_factor_is_multiplicative_at_the_spot() -> None:
    """The spot multiplies the local shading by its commanded factor exactly."""
    albedo = AlbedoTextureSpec(rms=0.0, spots=((0.0, 90.0, 12.0, 0.6),), seed=0)
    spec = dataclasses.replace(_spec(albedo), illumination_angle=0.3)
    smooth_spec = dataclasses.replace(spec, albedo_texture=None)
    textured = create_topographic_body((_SIZE, _SIZE), (_CENTER, _CENTER), spec)
    smooth = create_topographic_body((_SIZE, _SIZE), (_CENTER, _CENTER), smooth_spec)
    center = int(_CENTER)
    assert textured[center, center] / smooth[center, center] == pytest.approx(0.6, abs=1e-9)


def test_silhouette_is_untouched_by_albedo_texture() -> None:
    """The albedo texture never moves the silhouette."""
    albedo = AlbedoTextureSpec(rms=0.3, corr_px=8.0, spots=((30.0, 45.0, 15.0, 0.4),), seed=5)
    smooth = _render(None)
    textured = _render(albedo)
    assert np.array_equal(textured > 0.0, smooth > 0.0)


def _scene(seed: int) -> dict[str, Any]:
    """A minimal one-body scene with an albedo texture."""
    return {
        'instrument': 'coiss_nac',
        'size_v': 128,
        'size_u': 128,
        'random_seed': seed,
        'bodies': [
            {
                'name': 'MOON',
                'center_v': 64.0,
                'center_u': 64.0,
                'axis1': 80.0,
                'axis2': 80.0,
                'axis3': 80.0,
                'phase_angle': 45.0,
                'albedo_texture': {'rms': 0.2, 'corr_px': 10.0},
            }
        ],
    }


def test_scene_albedo_texture_draws_its_own_seeded_stream() -> None:
    """The scene seed selects the albedo realization deterministically."""
    img_a, _ = render_combined_model(_scene(1))
    img_b, _ = render_combined_model(_scene(2))
    assert not np.array_equal(img_a, img_b)
    img_a2, _ = render_combined_model(_scene(1))
    assert np.array_equal(img_a, img_a2)
