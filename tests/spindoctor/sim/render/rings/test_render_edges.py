"""Where a rendered ring edge lands, and how a pixel that a boundary cuts is mixed.

Each test draws one feature on a ring so large (radius 1e5 px) that its edge is a
straight line across the frame, measures the edge on the detector grid along the
radial direction, and compares it with the authored radius.  The measure is the
gradient centroid of the normalized brightness profile: the brightness-weighted mean
position of the step, which for a pixel-integrated edge equals the edge's position
exactly.  The pixel phase is where the edge falls inside its pixel, 0 on the pixel's
lower boundary and 0.5 at its center.

The remaining tests compare one rendered row with a dense reference: the authored
profile, written out here independently of the renderer, sampled 4000 times per pixel
along the radial direction and lit by the same closed forms.  For a straight face-on
or ansa edge that is the exact integral the renderer's footprint model aims at.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene, ring_system_block

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.rings.clutter import moonlet_disc_coverage, propeller_tau_factor
from spindoctor.sim.render.rings.render import (
    RING_ALBEDO_DEFAULT,
    RING_PHASE_G_DEFAULT,
    render_ring_system,
    ring_reflection_factor,
)
from spindoctor.sim.shared.lighting import henyey_greenstein_phase
from spindoctor.sim.shared.ring import ring_plane_from_sky, ring_sky_from_plane
from spindoctor.support.types import NDArrayFloatType

_RADIUS = 1.0e5
_SIZE = 32
_TOLERANCE_PX = 0.005


def _orbit(a_px: float) -> dict[str, float]:
    """A circular orbit of radius ``a_px``."""
    return {'a_px': a_px, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0}


def _edge(tau: float, side: str = 'in') -> dict[str, Any]:
    """A one-sided sheet whose edge is at the test radius."""
    return {'kind': 'edge', 'tau': tau, 'side': side, 'orbit': _orbit(_RADIUS)}


def _band(kind: str, tau: float, which: str) -> dict[str, Any]:
    """A 30 px ringlet or gap with its inner or outer edge at the test radius."""
    a_px = _RADIUS if which == 'inner' else _RADIUS - 30.0
    return {'kind': kind, 'tau': tau, 'width_px': 30.0, 'orbit': _orbit(a_px)}


def _sheet() -> dict[str, Any]:
    """A broad ringlet for a gap to be cut into."""
    return {'kind': 'ringlet', 'tau': 1.0, 'width_px': 200.0, 'orbit': _orbit(_RADIUS - 100.0)}


def _placed_system(
    features: list[dict[str, Any]],
    *,
    phase: float,
    b_obs: float = 90.0,
    b_sun: float = 90.0,
    node_deg: float = 0.0,
    lam_deg: float = 0.0,
    axis: str = 'u',
) -> tuple[dict[str, Any], float]:
    """Center the ring so the test radius at ``lam_deg`` sits mid-frame plus ``phase``.

    Along the other axis the point sits on the center of the middle row or column, so
    the edge crosses that row or column exactly at the authored position; the edge's
    curve moves it by about 1e-6 px a few rows away.

    Returns:
        The ``ring_system`` block and the edge's authored position along ``axis`` in
        detector pixel corner coordinates.
    """
    dv, du = ring_sky_from_plane(
        np.array([_RADIUS]),
        np.array([math.radians(lam_deg)]),
        opening_deg_obs=b_obs,
        node_deg=node_deg,
    )
    if axis == 'u':
        target_v, target_u = _SIZE / 2.0 + 0.5, _SIZE / 2.0 + phase
    else:
        target_v, target_u = _SIZE / 2.0 + phase, _SIZE / 2.0 + 0.5
    system = ring_system_block(
        target_v - float(dv[0]),
        target_u - float(du[0]),
        features,
        opening_deg_obs=b_obs,
        opening_deg_sun=b_sun,
        node_deg=node_deg,
    )
    return system, (target_u if axis == 'u' else target_v)


def _detector_intensity(system: dict[str, Any], *, os: int = 1) -> NDArrayFloatType:
    """Render the system at ``os`` and box-average it onto the detector grid."""
    geometry = system['geometry']
    maps = render_ring_system(
        (_SIZE * os, _SIZE * os),
        system,
        center_v=float(geometry['center_v']) * os,
        center_u=float(geometry['center_u']) * os,
        node_deg=float(geometry['node_deg']),
        oversample=os,
    )
    reduced: NDArrayFloatType = maps.intensity.reshape(_SIZE, os, _SIZE, os).mean(axis=(1, 3))
    return reduced


def _edge_position(image: NDArrayFloatType, axis: str, near: float) -> float:
    """The gradient centroid of the step near ``near``, in pixel corner coordinates."""
    middle = slice(_SIZE // 2 - 6, _SIZE // 2 + 6)
    profile = image[middle, :].mean(axis=0) if axis == 'u' else image[:, middle].mean(axis=1)
    gradient = np.diff(profile)
    # gradient[i] is the step from pixel i to pixel i + 1, at corner coordinate i + 1.
    position = np.arange(1, profile.size, dtype=np.float64)
    window = slice(round(near) - 3, round(near) + 2)
    return float(np.sum(position[window] * gradient[window]) / np.sum(gradient[window]))


def _landing_error(features: list[dict[str, Any]], *, os: int = 1, **placement: Any) -> float:
    """How far the rendered edge lands from its authored position, in pixels."""
    axis = str(placement.get('axis', 'u'))
    system, authored = _placed_system(features, **placement)
    return _edge_position(_detector_intensity(system, os=os), axis, authored) - authored


@pytest.mark.parametrize('tau', [0.2, 2.0, 5.0])
def test_lone_edge_lands_on_its_radius_at_each_optical_depth(tau: float) -> None:
    """A sharp edge lands within 0.005 px of its radius, thin or opaque."""
    error = _landing_error([_edge(tau)], phase=0.25)
    assert abs(error) < _TOLERANCE_PX


@pytest.mark.parametrize('phase', [0.25, 0.5, 0.75])
def test_lone_edge_lands_on_its_radius_at_each_pixel_phase(phase: float) -> None:
    """A tau 2 edge lands within 0.005 px of its radius wherever it falls in its pixel."""
    error = _landing_error([_edge(2.0)], phase=phase)
    assert abs(error) < _TOLERANCE_PX


def test_lone_edge_lands_on_its_radius_when_oversampled() -> None:
    """At oversample 4 an edge that cuts a render-grid pixel lands on its radius."""
    error = _landing_error([_edge(2.0)], phase=0.3, os=4)
    assert abs(error) < _TOLERANCE_PX


def test_outward_facing_edge_lands_on_its_radius() -> None:
    """An edge with its sheet outside it (side 'out') lands on its radius."""
    error = _landing_error([_edge(2.0, side='out')], phase=0.25)
    assert abs(error) < _TOLERANCE_PX


@pytest.mark.parametrize(
    ('kind', 'which'),
    [('ringlet', 'inner'), ('ringlet', 'outer'), ('gap', 'inner'), ('gap', 'outer')],
)
def test_band_edge_lands_on_its_radius(kind: str, which: str) -> None:
    """Each edge of a ringlet, and of a gap cut into a sheet, lands on its radius."""
    features = (
        [_band(kind, 2.0, which)] if kind == 'ringlet' else [_sheet(), _band(kind, 2.0, which)]
    )
    error = _landing_error(features, phase=0.25)
    assert abs(error) < _TOLERANCE_PX


@pytest.mark.parametrize(
    'placement',
    [
        {'b_obs': 30.0, 'b_sun': 30.0, 'lam_deg': 0.0, 'axis': 'u'},
        {'b_obs': 30.0, 'b_sun': 30.0, 'lam_deg': 90.0, 'axis': 'v'},
        {'b_obs': -25.0, 'b_sun': -40.0, 'node_deg': 90.0, 'lam_deg': 0.0, 'axis': 'v'},
        {'b_obs': 12.0, 'b_sun': -20.0, 'lam_deg': 180.0, 'axis': 'u'},
    ],
    ids=['inclined_ansa', 'foreshortened', 'rotated_node', 'unlit_face'],
)
def test_edge_lands_on_its_radius_in_inclined_views(placement: dict[str, Any]) -> None:
    """An edge lands on its radius on an inclined, rotated, or unlit ring."""
    error = _landing_error([_edge(2.0)], phase=0.25, **placement)
    assert abs(error) < _TOLERANCE_PX


def test_edge_on_a_pixel_boundary_is_not_blurred_at_oversample_4() -> None:
    """At oversample 4 an edge on a detector-pixel boundary leaves a clean step.

    The pixel inside the edge reads the full sheet and the pixel outside reads
    nothing: the box downsample is the only averaging across the edge.  The
    tolerance allows for the edge's curve across the pixel's four render-grid
    rows, which moves it by under 1e-5 px.
    """
    system, authored = _placed_system([_edge(0.2)], phase=0.0)
    image = _detector_intensity(system, os=4)
    full = float(image[_SIZE // 2, 2])
    column = round(authored)
    assert image[_SIZE // 2, column - 1] == pytest.approx(full, abs=1e-5 * full)
    assert image[_SIZE // 2, column] == pytest.approx(0.0, abs=1e-5 * full)


def _half_covered_maps() -> tuple[float, float, float, float]:
    """The intensity and transmission of a half-covered and a fully covered pixel."""
    system, authored = _placed_system([_edge(2.0)], phase=0.5)
    geometry = system['geometry']
    maps = render_ring_system(
        (_SIZE, _SIZE),
        system,
        center_v=float(geometry['center_v']),
        center_u=float(geometry['center_u']),
        node_deg=0.0,
    )
    row = _SIZE // 2
    column = math.floor(authored)
    return (
        float(maps.intensity[row, column]),
        float(maps.transmission[row, column]),
        float(maps.intensity[row, 2]),
        float(maps.transmission[row, 2]),
    )


def test_half_covered_pixel_emits_half_the_full_brightness() -> None:
    """A pixel half covered by a tau 2 sheet emits half of the sheet's brightness."""
    intensity, _transmission, full_intensity, _full_transmission = _half_covered_maps()
    expected_full = (
        RING_ALBEDO_DEFAULT
        / 4.0
        * henyey_greenstein_phase(RING_PHASE_G_DEFAULT, 0.0)
        * float(ring_reflection_factor(np.array([2.0]), 1.0, 1.0, lit=True)[0])
    )
    assert full_intensity == pytest.approx(expected_full, rel=1e-12)
    assert intensity == pytest.approx(0.5 * expected_full, rel=1e-12)


def test_half_covered_pixel_passes_half_the_background() -> None:
    """A pixel half covered by a tau 2 sheet passes half the background freely.

    The other half passes ``exp(-2)``, the sheet's own transmission face-on.
    """
    _intensity, transmission, _full_intensity, full_transmission = _half_covered_maps()
    assert full_transmission == pytest.approx(math.exp(-2.0), rel=1e-12)
    assert transmission == pytest.approx(0.5 + 0.5 * math.exp(-2.0), rel=1e-12)


def test_rendered_scene_edge_lands_on_its_radius_at_oversample_4() -> None:
    """Through the whole render at oversample 4, a lone edge lands on its radius."""
    system, authored = _placed_system([_edge(2.0)], phase=0.3)
    scene = frame_scene(size=_SIZE, seed=3, exposure_sec=1.0, ring_system=system, oversample=4)
    image, _meta = render_scene(scene)
    error = _edge_position(np.asarray(image, dtype=np.float64), 'u', authored) - authored
    assert abs(error) < _TOLERANCE_PX


def test_moonlet_rim_is_blurred_only_once_at_oversample_4() -> None:
    """A render-grid pixel 1.5 grid pixels inside a moonlet's rim is fully covered."""
    moonlet = {'a_px': 20.0, 'lam_deg': 0.0, 'radius_px': 3.0, 'amplitude': 1.0}
    os = 4
    disc = moonlet_disc_coverage(
        moonlet,
        (64 * os, 64 * os),
        center_v=32.0 * os,
        center_u=32.0 * os,
        b_obs=90.0,
        node_deg=0.0,
        os=os,
    )
    assert disc is not None
    (v_slice, u_slice), coverage = disc
    # The moonlet sits at (32, 52) detector px = (128, 208) render-grid px, radius 12
    # render-grid px.  The grid pixel centered at (128.5, 218.5) is 10.51 grid px
    # from it, about 1.5 grid px inside the rim.
    row = 128 - v_slice.start
    column = 218 - u_slice.start
    assert coverage[row, column] == 1.0


_REFERENCE_SAMPLES = 4000
_UNLIT: dict[str, Any] = {'b_obs': 12.0, 'b_sun': -20.0, 'lam_deg': 180.0}
_ROW = _SIZE // 2


def _authored_tau(feature: dict[str, Any], r: NDArrayFloatType) -> NDArrayFloatType:
    """An edge, ringlet, gap, or ramp's optical depth at radii ``r``, gaps negative."""
    tau = float(feature['tau'])
    inner = float(feature['orbit']['a_px'])
    if feature['kind'] == 'edge':
        return (
            np.where(r <= inner, tau, 0.0)
            if feature['side'] == 'in'
            else np.where(r >= inner, tau, 0.0)
        )
    width = float(feature['width_px'])
    if feature['kind'] == 'ramp':
        x = np.clip((r - inner) / width, 0.0, 1.0)
        if feature['side'] == 'out':
            return np.where(r <= inner + width, tau * x, 0.0)
        return np.where(r >= inner, tau * (1.0 - x), 0.0)
    inside = np.where((r >= inner) & (r <= inner + width), tau, 0.0)
    return -inside if feature['kind'] == 'gap' else inside


def _reference_row(system: dict[str, Any]) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """The middle row's intensity and transmission, integrated densely along u."""
    geometry = system['geometry']
    b_obs = float(geometry['opening_deg_obs'])
    b_sun = float(geometry['opening_deg_sun'])
    u = (np.arange(_SIZE * _REFERENCE_SAMPLES) + 0.5) / _REFERENCE_SAMPLES
    v = np.full(u.shape, _ROW + 0.5)
    r, lam, _x, _y = ring_plane_from_sky(
        v - float(geometry['center_v']),
        u - float(geometry['center_u']),
        opening_deg_obs=b_obs,
        node_deg=float(geometry['node_deg']),
    )
    tau = np.zeros(r.shape, dtype=np.float64)
    for feature in system['features']:
        tau += _authored_tau(feature, r)
    np.clip(tau, 0.0, None, out=tau)
    for moonlet in system.get('moonlets', []):
        tau = tau * propeller_tau_factor(moonlet, moonlet['propeller'], r=r, lam=lam, os=1)
    mu = abs(math.sin(math.radians(b_obs)))
    mu0 = abs(math.sin(math.radians(b_sun)))
    ap = RING_ALBEDO_DEFAULT / 4.0 * henyey_greenstein_phase(RING_PHASE_G_DEFAULT, 0.0)
    emission = ap * ring_reflection_factor(tau, mu, mu0, lit=(b_obs > 0.0) == (b_sun > 0.0))
    shape = (_SIZE, _REFERENCE_SAMPLES)
    return (
        emission.reshape(shape).mean(axis=1),
        np.exp(-tau / mu).reshape(shape).mean(axis=1),
    )


def _rendered_maps(system: dict[str, Any], *, os: int = 1) -> tuple[Any, Any]:
    """The middle row's intensity and transmission, and the full render's maps."""
    geometry = system['geometry']
    maps = render_ring_system(
        (_SIZE * os, _SIZE * os),
        system,
        center_v=float(geometry['center_v']) * os,
        center_u=float(geometry['center_u']) * os,
        node_deg=float(geometry['node_deg']),
        oversample=os,
    )
    return maps.intensity[_ROW * os : (_ROW + 1) * os].reshape(os, _SIZE, os).mean(
        axis=(0, 2)
    ), maps


def _ramp(width: float, side: str = 'in') -> dict[str, Any]:
    """A tau 2 ramp whose end at the test radius carries its full optical depth."""
    inner = _RADIUS if side == 'in' else _RADIUS - width
    return {'kind': 'ramp', 'tau': 2.0, 'width_px': width, 'side': side, 'orbit': _orbit(inner)}


def _ringlet(inner: float, width: float, tau: float) -> dict[str, Any]:
    """A ringlet from ``inner`` outward."""
    return {'kind': 'ringlet', 'tau': tau, 'width_px': width, 'orbit': _orbit(inner)}


def test_narrow_ramp_keeps_its_flux() -> None:
    """A ramp 0.3 px wide, narrower than a pixel, emits the light the reference does."""
    system, _authored = _placed_system([_ramp(0.3)], phase=0.3)
    reference, _transmission = _reference_row(system)
    row, _maps = _rendered_maps(system)
    assert float(row.sum()) == pytest.approx(float(reference.sum()), rel=1e-3)


def test_wide_unlit_ramp_lands_where_the_reference_does() -> None:
    """A 6 px ramp on the unlit face has the reference's brightness centroid.

    Brightness on the unlit face curves strongly with optical depth, so sampling the
    ramp at pixel centers moves its centroid by about 0.08 px.
    """
    system, _authored = _placed_system([_ramp(6.0)], phase=0.3, **_UNLIT)
    reference, _transmission = _reference_row(system)
    row, _maps = _rendered_maps(system)
    columns = np.arange(_SIZE) + 0.5
    centroid = float(np.sum(columns * row) / np.sum(row))
    expected = float(np.sum(columns * reference) / np.sum(reference))
    assert centroid == pytest.approx(expected, abs=_TOLERANCE_PX)


def test_sub_pixel_ringlet_matches_the_reference() -> None:
    """A ringlet 0.3 px wide, both of whose edges fall in one pixel, matches the reference."""
    system, authored = _placed_system([_ringlet(_RADIUS, 0.3, 2.0)], phase=0.3)
    reference, _transmission = _reference_row(system)
    row, _maps = _rendered_maps(system)
    column = math.floor(authored)
    assert row[column] == pytest.approx(reference[column], rel=1e-3)


def test_two_features_ending_in_one_pixel_match_the_reference() -> None:
    """Two overlapping ringlets with an edge each in one pixel match the reference.

    The pixel holds the outer edge of a tau 0.5 ringlet and the inner edge of a tau 3
    ringlet that overlaps it, so it has three pieces of different optical depth.
    """
    features = [_ringlet(_RADIUS - 3.0, 3.0, 0.5), _ringlet(_RADIUS - 0.4, 2.0, 3.0)]
    system, authored = _placed_system(features, phase=0.6)
    reference, _transmission = _reference_row(system)
    row, _maps = _rendered_maps(system)
    column = math.floor(authored)
    assert row[column] == pytest.approx(reference[column], rel=1e-3)


def test_pixel_a_ringlet_covers_off_center_is_masked() -> None:
    """A pixel that a 0.3 px ringlet covers away from its center joins the mask."""
    system, authored = _placed_system([_ringlet(_RADIUS, 0.3, 2.0)], phase=0.1)
    _row, maps = _rendered_maps(system)
    assert bool(maps.mask[_ROW, math.floor(authored)])


def test_propeller_shapes_the_pixels_an_edge_cuts() -> None:
    """A propeller lobe over an edge darkens the cut pixel as the reference does.

    The renderer evaluates the smooth lobe once in each piece of the pixel, which
    leaves it about 0.1% off the dense reference.
    """
    length_deg = 0.004
    moonlet = {
        'a_px': _RADIUS + 2.0,
        'lam_deg': length_deg / 2.0,
        'radius_px': 0.5,
        'amplitude': 0.0,
        'propeller': {'length_deg': length_deg, 'width_px': 2.0, 'contrast': -0.7},
    }
    system, authored = _placed_system([_edge(1.5)], phase=0.3)
    system['moonlets'] = [moonlet]
    reference, _transmission = _reference_row(system)
    row, _maps = _rendered_maps(system)
    column = math.floor(authored)
    assert row[column] == pytest.approx(reference[column], rel=5e-3)
