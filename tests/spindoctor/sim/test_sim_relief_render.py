"""Rendered-body acceptance checks for the limb-relief field.

The relief contract at the rendered-image level: the silhouette radius at
azimuth theta lands on the relief surface's outline,
``r_ellipse * (1 + delta(theta))``, the relief normals shade the disc, the
terminator grows ragged with every shadow inside the field's slope band and
concentrated against the terminator, relief OFF reproduces the relief-free
render bit-identically, and each body's terrain realization is seeded
independently.
"""

import math
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.sim import render
from spindoctor.sim.forward import body_topo
from spindoctor.sim.forward.body_topo import TopoBodySpec, create_topographic_body
from spindoctor.sim.forward.photometry import DARK_SIDE_FLOOR
from spindoctor.sim.forward.relief import ReliefField, cast_shadows, synthesize_relief_field
from spindoctor.support.types import NDArrayFloatType

_RMS = 0.02
_CORR_DEG = 15.0


def _limb_spec(*, rms: float, seed: int) -> TopoBodySpec:
    """A fully lit sphere shaded flat (Lommel-Seeliger at phase 0).

    The flat disc makes the rendered image a pure silhouette probe: every
    interior pixel is bright, so the mask boundary is the limb.
    """
    return TopoBodySpec(
        axis1=160.0,
        axis2=160.0,
        axis3=160.0,
        photometric_law='lommel_seeliger',
        oversample=4,
        limb_relief_rms=rms,
        limb_relief_corr_deg=_CORR_DEG,
        relief_seed=seed,
    )


def _ray_edge_radius(mask: np.ndarray, center: float, angle: float) -> float:
    """The mask boundary radius along one ray from the body center."""
    dv = math.cos(angle)
    du = math.sin(angle)
    radius = 60.0
    while radius < 100.0:
        pv = int(center + radius * dv)
        pu = int(center + radius * du)
        if not mask[pv, pu]:
            return radius
        radius += 0.05
    return radius


@pytest.mark.parametrize('rms', [_RMS, 0.05])
def test_limb_lands_on_the_relief_outline(rms: float) -> None:
    """Silhouette radius matches r_ellipse * (1 + delta(theta)) within 0.5 os px.

    ``delta`` is the relief surface's outline envelope, which at 5% relief
    departs from the equator slice by a sizable share of the RMS.

    Per-ray radii carry ~half-pixel staircase noise from the rasterized
    mask, so the acceptance is asserted on azimuth-window means (16 windows
    of 16 rays); the pooled displacement must also track the commanded
    field realization tightly.
    """
    seed = 42
    size = 256
    img = create_topographic_body(
        (size, size), (size / 2.0, size / 2.0), _limb_spec(rms=rms, seed=seed)
    )
    mask = img > 0.0
    field = synthesize_relief_field(rms, _CORR_DEG, seed)

    n_windows, rays_per_window = 16, 16
    angles = np.arange(n_windows * rays_per_window) * (2.0 * np.pi / (n_windows * rays_per_window))
    measured = np.array([_ray_edge_radius(mask, size / 2.0, a) for a in angles])
    # phi convention: atan2(u_rot / b, v_rot / a); for an unrotated sphere
    # this is the image azimuth of the ray itself.
    predicted = 80.0 * (1.0 + field.outline_delta(angles))

    errors = measured - predicted
    window_means = errors.reshape(n_windows, rays_per_window).mean(axis=1)
    assert float(np.abs(window_means).max()) < 0.5
    # The rendered displacement tracks the field realization, not just its scale.
    correlation = np.corrcoef(measured - 80.0, predicted - 80.0)[0, 1]
    assert correlation > 0.95


def test_rendered_limb_rms_tracks_the_commanded_value() -> None:
    """The pooled silhouette displacement RMS lands near rms * r_ellipse."""
    seed = 7
    size = 256
    img = create_topographic_body(
        (size, size), (size / 2.0, size / 2.0), _limb_spec(rms=_RMS, seed=seed)
    )
    mask = img > 0.0
    angles = np.arange(256) * (2.0 * np.pi / 256)
    measured = np.array([_ray_edge_radius(mask, size / 2.0, a) for a in angles])
    commanded_px = _RMS * 80.0
    assert abs(float(np.std(measured - 80.0)) - commanded_px) < 0.35 * commanded_px


def _terminator_spec(*, rms: float) -> TopoBodySpec:
    """A high-phase crescent with the sun from the right."""
    return TopoBodySpec(
        axis1=240.0,
        axis2=240.0,
        axis3=240.0,
        illumination_angle=math.radians(90.0),
        phase_angle=math.radians(100.0),
        oversample=4,
        limb_relief_rms=rms,
        limb_relief_corr_deg=_CORR_DEG,
        relief_seed=5,
    )


def _band_elevation_deg(field_rms: float, seed: int) -> float:
    """The sun elevation, in degrees, above which a realization casts no shadow."""
    field = synthesize_relief_field(field_rms, _CORR_DEG, seed)
    band = body_topo._SLOPE_BAND_MARGIN * field.slope_max / (1.0 + field.h_min)
    return math.degrees(math.atan(band))


def test_terminator_is_ragged_with_excursions_at_the_terminator() -> None:
    """Relief shadows displace the terminator raggedly, inside the slope band.

    Comparing the relief render against the smooth render of the same
    geometry: newly dark pixels exist, every one lies where the smooth sun
    elevation is inside the field's slope band, and they concentrate against
    the smooth terminator (the shadowed fraction falls with elevation).
    """
    size = 320
    smooth = create_topographic_body(
        (size, size), (size / 2.0, size / 2.0), _terminator_spec(rms=0.0)
    )
    ragged = create_topographic_body(
        (size, size), (size / 2.0, size / 2.0), _terminator_spec(rms=_RMS)
    )
    both = (smooth > 0.0) & (ragged > 0.0)
    lit_smooth = both & (smooth > DARK_SIDE_FLOOR * 1.2)
    dark_ragged = both & (ragged <= DARK_SIDE_FLOOR * 1.05)
    newly_dark = lit_smooth & dark_ragged
    assert int(newly_dark.sum()) > 50

    spec = _terminator_spec(rms=_RMS)
    elevation = _sun_elevation_deg(size, semi_axes=(120.0, 120.0), spec=spec)
    band_deg = _band_elevation_deg(_RMS, 5)
    # Half a pixel of smooth-elevation slack covers the shading upsample.
    assert float(elevation[newly_dark].max()) <= band_deg + 0.5

    # Excursions grow toward the terminator: the shadowed fraction falls
    # with the sun's elevation.
    def shadow_fraction(lo: float, hi: float) -> float:
        band = lit_smooth & (elevation > lo) & (elevation <= hi)
        assert int(band.sum()) > 0
        return float(newly_dark[band].sum()) / float(band.sum())

    near = shadow_fraction(0.0, 3.0)
    mid = shadow_fraction(3.0, 6.0)
    far = shadow_fraction(9.0, 15.0)
    assert near > mid
    assert near > 4.0 * far


def _sun_elevation_deg(
    size: int, *, semi_axes: tuple[float, float], spec: TopoBodySpec
) -> NDArrayFloatType:
    """Sun elevation above the smooth surface at each pixel center, degrees.

    The body is centered in a ``size`` frame and unrotated; the depth semi-axis is
    taken equal to the mean of the two in-plane ones (a near-sphere).
    """
    v, u = np.mgrid[0:size, 0:size].astype(float) + 0.5 - size / 2.0
    x = v / semi_axes[0]
    y = u / semi_axes[1]
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    sin_phase = math.sin(spec.phase_angle)
    sun = (
        -math.cos(spec.illumination_angle) * sin_phase,
        math.sin(spec.illumination_angle) * sin_phase,
        math.cos(spec.phase_angle),
    )
    mu0 = x * sun[0] + y * sun[1] + z * sun[2]
    return np.asarray(np.degrees(np.arcsin(np.clip(mu0, -1.0, 1.0))), dtype=np.float64)


def _moon_spec(seed: int, *, rms: float = _RMS, **extra: Any) -> TopoBodySpec:
    """An 85 px moon at phase 58 degrees, the geometry that once cut wedges through its disc."""
    return TopoBodySpec(
        axis1=170.0,
        axis2=168.0,
        axis3=164.0,
        illumination_angle=math.radians(28.0),
        phase_angle=math.radians(58.0),
        photometric_law='lommel_seeliger',
        limb_relief_rms=rms,
        limb_relief_corr_deg=_CORR_DEG,
        relief_seed=seed,
        **extra,
    )


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_relief_darkens_no_pixel_above_the_slope_band(seed: int) -> None:
    """Every pixel the relief renders as night sits where the sun is inside the band.

    The band is the field's own slope bound, ``atan(1.5 s_max / (1 + h_min))``; above
    it no terrain can rise into the sun ray.  The relief must still shadow something.
    """
    size = 256
    spec = _moon_spec(seed)
    smooth_spec = _moon_spec(seed, rms=0.0)
    center = (size / 2.0, size / 2.0)
    smooth = create_topographic_body((size, size), center, smooth_spec)
    ragged = create_topographic_body((size, size), center, spec)
    elevation = _sun_elevation_deg(size, semi_axes=(85.0, 84.0), spec=spec)
    v, u = np.mgrid[0:size, 0:size].astype(float) + 0.5 - size / 2.0
    inner = np.hypot(v / 85.0, u / 84.0) < 0.85
    newly_dark = inner & (smooth > DARK_SIDE_FLOOR * 1.2) & (ragged <= DARK_SIDE_FLOOR * 1.05)
    assert int(newly_dark.sum()) > 0
    assert float(elevation[newly_dark].max()) <= _band_elevation_deg(_RMS, seed) + 0.5


def _surface_normals(field: ReliefField, points: NDArrayFloatType) -> NDArrayFloatType:
    """Outward normals of the relief surface ``(1 + h) p``, from its own height samples.

    The cross product of the surface's central differences along east and north,
    half a field grid cell apart, so it shares nothing with the renderer's
    gradient formula but the field's samples.
    """
    step = math.pi / field.grid.shape[0]
    east = np.stack([-points[:, 1], points[:, 0], np.zeros(len(points))], axis=-1)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(points, east)

    def chord(tangent: NDArrayFloatType) -> NDArrayFloatType:
        ahead = points * math.cos(step) + tangent * math.sin(step)
        behind = points * math.cos(step) - tangent * math.sin(step)
        return cast(
            NDArrayFloatType,
            ahead * (1.0 + field.sample_points(ahead))[:, np.newaxis]
            - behind * (1.0 + field.sample_points(behind))[:, np.newaxis],
        )

    normals = np.cross(chord(east), chord(north))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return cast(NDArrayFloatType, normals * np.sign(np.sum(normals * points, axis=-1))[:, None])


def _lambert_relief_disc(*, rms: float, corr_deg: float) -> tuple[TopoBodySpec, Any, ReliefField]:
    """A 75 px Lambert sphere with relief, shaded at the render grid, and its grid and field."""
    spec = TopoBodySpec(
        axis1=150.0,
        axis2=150.0,
        axis3=150.0,
        illumination_angle=math.radians(40.0),
        phase_angle=math.radians(35.0),
        anti_aliasing=0.0,
        limb_relief_rms=rms,
        limb_relief_corr_deg=corr_deg,
        relief_seed=3,
    )
    grid = body_topo._body_grid(
        (200, 200), (100.0, 100.0), (75.0, 75.0, 75.0), cos_rz=1.0, sin_rz=0.0, cos_rt=1.0
    )
    return spec, grid, synthesize_relief_field(rms, corr_deg, 3)


def test_relief_normals_shade_a_lambert_disc() -> None:
    """On a Lambert disc the relief brightness is the relief surface's cosine of incidence.

    At oversample 1 without supersampling the shading grid is the render grid, so a
    lit, unshadowed pixel renders at ``n . sun`` for the surface's own normal, to the
    0.02 the two difference schemes differ by (a sign-flipped relief misses by about
    0.5).  Where the relief moves a pixel by more than 0.03, it moves it the same way.
    """
    spec, grid, field = _lambert_relief_disc(rms=_RMS, corr_deg=_CORR_DEG)
    ragged = create_topographic_body((200, 200), (100.0, 100.0), spec)
    smooth = create_topographic_body(
        (200, 200), (100.0, 100.0), TopoBodySpec(**{**spec.__dict__, 'limb_relief_rms': 0.0})
    )
    near = grid.e2 < 0.8
    points = np.stack(
        [grid.v_rot[near] / 75.0, grid.u_rot[near] / 75.0, np.sqrt(1.0 - grid.e2[near])], axis=-1
    )
    sin_phase = math.sin(spec.phase_angle)
    sun = np.array(
        [
            -math.cos(spec.illumination_angle) * sin_phase,
            math.sin(spec.illumination_angle) * sin_phase,
            math.cos(spec.phase_angle),
        ]
    )
    cos_i = np.sum(_surface_normals(field, points) * sun, axis=-1)
    shadowed = body_topo._relief_shading(grid, spec, field, near, cos_rz=1.0, sin_rz=0.0)[3]
    lit = (cos_i > 0.05) & ~shadowed[near]
    np.testing.assert_allclose(ragged[near][lit], cos_i[lit], rtol=0.0, atol=0.02)
    expected_change = cos_i[lit] - smooth[near][lit]
    moved = np.abs(expected_change) > 0.03
    assert int(moved.sum()) > 1000
    rendered_change = ragged[near][lit][moved] - smooth[near][lit][moved]
    np.testing.assert_array_equal(np.sign(rendered_change), np.sign(expected_change[moved]))


def test_relief_normals_near_the_limb_face_the_observer() -> None:
    """A slope tilted away from the observer near the limb is shaded as its mirror.

    Rough relief tilts some of the surface's own normals away from the observer
    inside the outline; every normal the renderer shades has ``normal_z >= 0``.
    """
    spec, grid, field = _lambert_relief_disc(rms=0.035, corr_deg=5.0)
    reach = 1.0 + field.h_max + 2.0 / 75.0
    near = grid.e2 < reach * reach
    inside = body_topo._silhouette_mask(grid, field)[near] > 0.0
    radial = np.sqrt(np.maximum(grid.e2[near], 1.0))
    points = np.stack(
        [
            grid.v_rot[near] / 75.0 / radial,
            grid.u_rot[near] / 75.0 / radial,
            np.sqrt(np.maximum(0.0, 1.0 - grid.e2[near])),
        ],
        axis=-1,
    )
    away = inside & (_surface_normals(field, points)[:, 2] < -0.05)
    assert int(away.sum()) > 100
    normal_z = body_topo._relief_shading(grid, spec, field, near, cos_rz=1.0, sin_rz=0.0)[2]
    assert float(normal_z.min()) >= 0.0


def test_shadow_march_resolves_fine_relief_on_a_small_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The march steps at the terrain's resolution, not the body's pixel size.

    On a 30 px body with 1 degree relief one shading pixel spans several field
    cells, so a pixel-sized step skips ridge tips.  The renderer's shadow
    decisions match a march at a quarter of its step on all but 1% of the
    shadowed pixels.
    """
    spec = TopoBodySpec(
        axis1=60.0,
        axis2=60.0,
        axis3=60.0,
        illumination_angle=math.radians(80.0),
        phase_angle=math.radians(70.0),
        anti_aliasing=0.0,
        limb_relief_rms=0.03,
        limb_relief_corr_deg=1.0,
        relief_seed=3,
    )
    grid = body_topo._body_grid(
        (80, 80), (40.0, 40.0), (30.0, 30.0, 30.0), cos_rz=1.0, sin_rz=0.0, cos_rt=1.0
    )
    field = synthesize_relief_field(0.03, 1.0, 3)
    near = grid.e2 < (1.0 + field.h_max + 2.0 / 30.0) ** 2
    shadowed = body_topo._relief_shading(grid, spec, field, near, cos_rz=1.0, sin_rz=0.0)[3]

    cell = 2.0 * np.pi / field.grid.shape[0]

    def fine_march(
        points: NDArrayFloatType, sun: NDArrayFloatType, relief: ReliefField, *, step: float
    ) -> Any:
        return cast_shadows(points, sun, relief, step=min(step, cell / 32.0))

    monkeypatch.setattr(body_topo, 'cast_shadows', fine_march)
    reference = body_topo._relief_shading(grid, spec, field, near, cos_rz=1.0, sin_rz=0.0)[3]
    assert int(reference.sum()) > 100
    assert int((shadowed != reference).sum()) <= 0.01 * int(reference.sum())


def _body_scene(*, oversample: int | None, body_extra: dict[str, Any]) -> dict[str, Any]:
    """A minimal single-body scene for whole-render comparisons."""
    body: dict[str, Any] = {
        'name': 'B',
        'center_v': 40.0,
        'center_u': 40.0,
        'axis1': 44.0,
        'axis2': 40.0,
        'axis3': 40.0,
        'illumination_angle': 30.0,
        'phase_angle': 70.0,
        'crater_fill': 0.3,
        'crater_max_radius': 0.2,
    }
    body.update(body_extra)
    scene: dict[str, Any] = {
        'size_v': 80,
        'size_u': 80,
        'random_seed': 11,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [body],
    }
    if oversample is not None:
        scene['oversample'] = oversample
    return scene


def test_relief_rms_zero_reproduces_the_relief_free_render_bit_identically() -> None:
    """limb_relief_rms 0.0 and an absent key render byte-identically (os 1 and 4)."""
    for oversample in (None, 4):
        absent, _ = render.render_combined_model(_body_scene(oversample=oversample, body_extra={}))
        zero, _ = render.render_combined_model(
            _body_scene(
                oversample=oversample,
                body_extra={'limb_relief_rms': 0.0, 'limb_relief_corr_deg': 15.0},
            )
        )
        assert np.array_equal(absent, zero)


def test_explicit_lambert_law_reproduces_the_default_bit_identically() -> None:
    """photometric_law 'lambert' and an absent key take the same render path."""
    absent, _ = render.render_combined_model(_body_scene(oversample=None, body_extra={}))
    explicit, _ = render.render_combined_model(
        _body_scene(oversample=None, body_extra={'photometric_law': 'lambert'})
    )
    assert np.array_equal(absent, explicit)


def test_non_lambert_law_changes_the_disc() -> None:
    """A non-Lambert law renders a different disc profile than Lambert."""
    lambert, _ = render.render_combined_model(_body_scene(oversample=None, body_extra={}))
    minnaert, _ = render.render_combined_model(
        _body_scene(oversample=None, body_extra={'photometric_law': 'minnaert'})
    )
    assert not np.array_equal(lambert, minnaert)


def test_opposition_surge_scales_the_disc_by_the_documented_factor() -> None:
    """The surge dims a high-phase disc by (1 + A exp(-a/w)) / (1 + A)."""
    spec = TopoBodySpec(
        axis1=120.0,
        axis2=120.0,
        axis3=120.0,
        illumination_angle=math.radians(30.0),
        phase_angle=math.radians(70.0),
        oversample=2,
    )
    plain = create_topographic_body((160, 160), (80.0, 80.0), spec)
    surged_spec = TopoBodySpec(
        axis1=120.0,
        axis2=120.0,
        axis3=120.0,
        illumination_angle=math.radians(30.0),
        phase_angle=math.radians(70.0),
        oversample=2,
        surge_amplitude=1.0,
        surge_width_deg=6.0,
    )
    surged = create_topographic_body((160, 160), (80.0, 80.0), surged_spec)
    factor = (1.0 + math.exp(-70.0 / 6.0)) / 2.0
    interior = (plain > 0.1) & (plain < 0.99)
    assert interior.any()
    np.testing.assert_allclose(surged[interior], plain[interior] * factor, rtol=1e-9)


def test_relief_terrain_is_stable_when_craters_toggle() -> None:
    """The relief stream is named independently of the crater draws.

    Toggling craters must not reseed the limb: the silhouette (where only
    relief acts) stays identical while the disc shading changes.
    """
    with_craters, _ = render.render_combined_model(
        _body_scene(oversample=None, body_extra={'limb_relief_rms': 0.02})
    )
    body_extra: dict[str, Any] = {'limb_relief_rms': 0.02, 'crater_fill': 0.0}
    without_craters, _ = render.render_combined_model(
        _body_scene(oversample=None, body_extra=body_extra)
    )
    assert np.array_equal(with_craters > 0.0, without_craters > 0.0)
    assert not np.array_equal(with_craters, without_craters)


def test_two_identical_bodies_get_independent_terrains() -> None:
    """Per-body identity seeds two same-geometry bodies' relief independently."""
    scene: dict[str, Any] = {
        'size_v': 120,
        'size_u': 120,
        'random_seed': 11,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [
            {
                'name': 'FIRST',
                'center_v': 30.0,
                'center_u': 30.0,
                'axis1': 40.0,
                'axis2': 40.0,
                'axis3': 40.0,
                'limb_relief_rms': 0.03,
                'range_km': 1.0,
            },
            {
                'name': 'SECOND',
                'center_v': 88.0,
                'center_u': 88.0,
                'axis1': 40.0,
                'axis2': 40.0,
                'axis3': 40.0,
                'limb_relief_rms': 0.03,
                'range_km': 2.0,
            },
        ],
    }
    img, _meta = render.render_combined_model(scene)
    first = img[0:60, 0:60] > 0.0
    second = img[58:118, 58:118] > 0.0
    assert not np.array_equal(first, second)
