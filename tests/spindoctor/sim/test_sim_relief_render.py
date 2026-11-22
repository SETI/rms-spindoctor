"""Rendered-body acceptance checks for the limb-relief field.

The relief contract at the rendered-image level: the silhouette radius at
azimuth theta lands on the relief surface's outline,
``r_ellipse * (1 + delta(theta))``, each pixel reads the terrain where its line
of sight meets the surface, the relief normals shade the disc, the
terminator grows ragged with every shadow inside the field's slope band and
concentrated against the terminator, relief OFF reproduces the relief-free
render bit-identically, and each body's terrain realization is seeded
independently.
"""

import math
from typing import Any, cast

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise

from spindoctor.sim.render.bodies import ellipsoid_relief
from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, create_ellipsoid_body
from spindoctor.sim.render.bodies.ellipsoid_grid import silhouette_mask
from spindoctor.sim.render.bodies.photometry import DARK_SIDE_FLOOR
from spindoctor.sim.render.bodies.relief import ReliefField, cast_shadows, synthesize_relief_field
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.shared.ellipsoid import EllipsoidProjection, project_ellipsoid
from spindoctor.support.types import NDArrayFloatType

_RMS = 0.02
_CORR_DEG = 15.0


def _limb_spec(*, rms: float, seed: int) -> EllipsoidBodySpec:
    """A fully lit sphere shaded flat (Lommel-Seeliger at phase 0).

    The flat disc makes the rendered image a pure silhouette probe: every
    interior pixel is bright, so the mask boundary is the limb.
    """
    return EllipsoidBodySpec(
        semi_a_px=80.0,
        semi_b_px=80.0,
        semi_c_px=80.0,
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
    img = create_ellipsoid_body(
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
    img = create_ellipsoid_body(
        (size, size), (size / 2.0, size / 2.0), _limb_spec(rms=_RMS, seed=seed)
    )
    mask = img > 0.0
    angles = np.arange(256) * (2.0 * np.pi / 256)
    measured = np.array([_ray_edge_radius(mask, size / 2.0, a) for a in angles])
    commanded_px = _RMS * 80.0
    assert abs(float(np.std(measured - 80.0)) - commanded_px) < 0.35 * commanded_px


def _terminator_spec(*, rms: float) -> EllipsoidBodySpec:
    """A high-phase crescent with the sun from the right."""
    return EllipsoidBodySpec(
        semi_a_px=120.0,
        semi_b_px=120.0,
        semi_c_px=120.0,
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
    band = ellipsoid_relief._SLOPE_BAND_MARGIN * field.slope_max / (1.0 + field.h_min)
    return math.degrees(math.atan(band))


def test_terminator_is_ragged_with_excursions_at_the_terminator() -> None:
    """Relief shadows displace the terminator raggedly, inside the slope band.

    Comparing the relief render against the smooth render of the same
    geometry: newly dark pixels exist, every one lies where the smooth sun
    elevation is inside the field's slope band, and they concentrate against
    the smooth terminator (the shadowed fraction falls with elevation).
    """
    size = 320
    smooth = create_ellipsoid_body(
        (size, size), (size / 2.0, size / 2.0), _terminator_spec(rms=0.0)
    )
    ragged = create_ellipsoid_body(
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
    # Half a pixel of smooth-elevation slack covers the interpolated shadows.
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
    # Shaded on the working grid this realization gives near / far = 3.8.
    assert near > 3.0 * far


def _sun_elevation_deg(
    size: int, *, semi_axes: tuple[float, float], spec: EllipsoidBodySpec
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


def _moon_spec(seed: int, *, rms: float = _RMS, **extra: Any) -> EllipsoidBodySpec:
    """An 85 px moon at phase 58 degrees, the geometry that once cut wedges through its disc."""
    return EllipsoidBodySpec(
        semi_a_px=85.0,
        semi_b_px=84.0,
        semi_c_px=82.0,
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
    smooth = create_ellipsoid_body((size, size), center, smooth_spec)
    ragged = create_ellipsoid_body((size, size), center, spec)
    elevation = _sun_elevation_deg(size, semi_axes=(85.0, 84.0), spec=spec)
    v, u = np.mgrid[0:size, 0:size].astype(float) + 0.5 - size / 2.0
    inner = np.hypot(v / 85.0, u / 84.0) < 0.85
    newly_dark = inner & (smooth > DARK_SIDE_FLOOR * 1.2) & (ragged <= DARK_SIDE_FLOOR * 1.05)
    assert int(newly_dark.sum()) > 0
    assert float(elevation[newly_dark].max()) <= _band_elevation_deg(_RMS, seed) + 0.5


def _relief_shading(
    grid: EllipsoidProjection, spec: EllipsoidBodySpec, field: ReliefField, near: Any
) -> ellipsoid_relief.ReliefShading:
    """The renderer's relief shading of ``near`` on ``grid``, lit as ``spec`` says."""
    return ellipsoid_relief.relief_shading(
        grid,
        field,
        near,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
    )


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


def _first_surface_points(
    field: ReliefField, x: NDArrayFloatType, y: NDArrayFloatType
) -> NDArrayFloatType:
    """Where each line of sight first meets the relief surface, found by brute force.

    In parameter space each line of sight runs parallel to ``z``.  It is sampled every
    1/32 of a field cell in latitude, from the highest the terrain reaches down to
    where it must be inside the surface; the first sample inside and the one before
    it are then bisected.  It shares only the field's samples with the renderer's
    search.  A line of sight that misses the surface gets NaN.
    """
    rho = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    lat_top = np.arccos(np.minimum(rho / (1.0 + field.h_max), 1.0))
    bottom = 1.0 + field.h_min
    lat_low = np.where(rho < bottom, np.arccos(np.minimum(rho / bottom, 1.0)), -lat_top)
    step = 2.0 * np.pi / field.grid.shape[0] / 32.0
    lat = np.full(rho.size, np.nan)
    for start in range(0, rho.size, 1000):
        part = slice(start, start + 1000)
        count = int(np.ceil(float((lat_top[part] - lat_low[part]).max()) / step)) + 2
        lats = np.maximum(lat_top[part, None] - step * np.arange(count), lat_low[part, None])
        phis = np.broadcast_to(phi[part, None], lats.shape)
        inside = (1.0 + field.sample(lats, phis)) * np.cos(lats) >= rho[part, None]
        first = np.argmax(inside, axis=1)
        rows = np.arange(lats.shape[0])
        low = lats[rows, first]
        high = lats[rows, np.maximum(first - 1, 0)]
        for _ in range(30):
            mid = 0.5 * (low + high)
            mid_inside = (1.0 + field.sample(mid, phi[part])) * np.cos(mid) >= rho[part]
            low = np.where(mid_inside, mid, low)
            high = np.where(mid_inside, high, mid)
        lat[part] = np.where(inside.any(axis=1), 0.5 * (low + high), np.nan)
    return np.stack([np.cos(lat) * np.cos(phi), np.cos(lat) * np.sin(phi), np.sin(lat)], axis=-1)


def _lambert_relief_disc(
    *, rms: float, corr_deg: float
) -> tuple[EllipsoidBodySpec, EllipsoidProjection, ReliefField]:
    """A 75 px Lambert sphere with relief, shaded at the render grid, and its grid and field."""
    spec = EllipsoidBodySpec(
        semi_a_px=75.0,
        semi_b_px=75.0,
        semi_c_px=75.0,
        illumination_angle=math.radians(40.0),
        phase_angle=math.radians(35.0),
        anti_aliasing=0.0,
        limb_relief_rms=rms,
        limb_relief_corr_deg=corr_deg,
        relief_seed=3,
    )
    grid = project_ellipsoid((200, 200), (100.0, 100.0), 75.0, semi_b_px=75.0, semi_c_px=75.0)
    return spec, grid, synthesize_relief_field(rms, corr_deg, 3)


def test_relief_normals_shade_a_lambert_disc() -> None:
    """On a Lambert disc the relief brightness is the relief surface's cosine of incidence.

    At oversample 1 without supersampling the shading grid is the render grid, so a
    lit, unshadowed pixel renders at ``n . sun`` for the surface's own normal where its
    line of sight meets the surface, to the 0.02 the two difference schemes differ by
    (a sign-flipped relief misses by about 0.5).  Where the relief moves a pixel by
    more than 0.03, it moves it the same way.
    """
    spec, grid, field = _lambert_relief_disc(rms=_RMS, corr_deg=_CORR_DEG)
    ragged = create_ellipsoid_body((200, 200), (100.0, 100.0), spec)
    smooth = create_ellipsoid_body(
        (200, 200), (100.0, 100.0), EllipsoidBodySpec(**{**spec.__dict__, 'limb_relief_rms': 0.0})
    )
    near = grid.ellipse_dist_sq < 0.8
    points = _first_surface_points(field, grid.v_rot[near] / 75.0, grid.u_rot[near] / 75.0)
    sin_phase = math.sin(spec.phase_angle)
    sun = np.array(
        [
            -math.cos(spec.illumination_angle) * sin_phase,
            math.sin(spec.illumination_angle) * sin_phase,
            math.cos(spec.phase_angle),
        ]
    )
    cos_i = np.sum(_surface_normals(field, points) * sun, axis=-1)
    shadowed = _relief_shading(grid, spec, field, near).shadowed
    lit = (cos_i > 0.05) & ~shadowed[near]
    np.testing.assert_allclose(ragged[near][lit], cos_i[lit], rtol=0.0, atol=0.02)
    expected_change = cos_i[lit] - smooth[near][lit]
    moved = np.abs(expected_change) > 0.03
    assert int(moved.sum()) > 1000
    rendered_change = ragged[near][lit][moved] - smooth[near][lit][moved]
    np.testing.assert_array_equal(np.sign(rendered_change), np.sign(expected_change[moved]))


def test_disc_terrain_is_read_where_the_line_of_sight_meets_it() -> None:
    """Each pixel reads the terrain where its line of sight first meets the surface.

    At 5% relief and a 15-degree correlation length, the smooth sphere's direction
    under a pixel lies more than half a correlation length from the point the pixel
    shows on about 12% of the disc.  The renderer's points lie that far from a brute
    force search's on under 1% of it.
    """
    spec, grid, field = _lambert_relief_disc(rms=0.05, corr_deg=_CORR_DEG)
    disc = grid.ellipse_dist_sq < 1.0
    truth = _first_surface_points(field, grid.v_rot[disc] / 75.0, grid.u_rot[disc] / 75.0)
    hit = np.isfinite(truth[:, 0])
    seen = _relief_shading(grid, spec, field, disc).seen
    smooth = np.stack(
        [
            grid.v_rot[disc] / 75.0,
            grid.u_rot[disc] / 75.0,
            np.sqrt(1.0 - grid.ellipse_dist_sq[disc]),
        ],
        axis=-1,
    )
    half_corr = math.radians(_CORR_DEG) / 2.0

    def share_misread(points: NDArrayFloatType) -> float:
        cosines = np.clip(np.sum(points[hit] * truth[hit], axis=-1), -1.0, 1.0)
        return float(np.mean(np.arccos(cosines) > half_corr))

    assert np.isfinite(seen[hit]).all()
    assert share_misread(smooth) > 0.05
    assert share_misread(seen) < 0.01


def test_relief_normals_near_the_limb_face_the_observer() -> None:
    """The surface each pixel sees faces the observer, so no normal needs mirroring.

    Rough relief tilts some of the surface's own normals away from the observer
    inside the outline: at 2,470 of the 19,156 pixels there, the smooth sphere's
    direction reads such a normal.  The surface the renderer reads, where each line of
    sight first meets it, faces the observer at all but 2 of them, where the line of
    sight grazes a ridge and the test's differences, half a field cell apart, tilt the
    normal away.  The test allows 0.1% of the pixels inside the outline, 19 here.
    """
    spec, grid, field = _lambert_relief_disc(rms=0.035, corr_deg=5.0)
    reach = 1.0 + field.h_max + 2.0 / 75.0
    near = grid.ellipse_dist_sq < reach * reach
    inside = silhouette_mask(grid, field)[near] > 0.0
    radial = np.sqrt(np.maximum(grid.ellipse_dist_sq[near], 1.0))
    points = np.stack(
        [
            grid.v_rot[near] / 75.0 / radial,
            grid.u_rot[near] / 75.0 / radial,
            np.sqrt(np.maximum(0.0, 1.0 - grid.ellipse_dist_sq[near])),
        ],
        axis=-1,
    )
    away = inside & (_surface_normals(field, points)[:, 2] < -0.05)
    assert int(away.sum()) > 100
    seen = _relief_shading(grid, spec, field, near).seen
    assert np.isfinite(seen[inside]).all()
    facing_away = _surface_normals(field, seen[inside])[:, 2] < 0.0
    assert int(facing_away.sum()) <= 0.001 * int(inside.sum())


def test_a_normal_leaning_away_is_laid_into_the_sky_plane() -> None:
    """A relief normal leaning away from the observer is laid into the sky plane.

    Where a line of sight grazes a ridge, the normal the field's differences give the
    point it meets can lean away from the observer.  At 5% relief and a 2-degree
    correlation length, some pixels on the disc read such a normal.  The renderer
    gives each of them a toward-observer component of exactly 0, neither mirrored
    toward the observer nor left leaning away.
    """
    spec, grid, field = _lambert_relief_disc(rms=0.05, corr_deg=2.0)
    near = grid.ellipse_dist_sq < 1.0
    shading = _relief_shading(grid, spec, field, near)
    seen = shading.seen
    normals = seen - field.surface_gradient(seen) / (1.0 + field.sample_points(seen))[:, None]
    away = normals[:, 2] < 0.0
    assert int(away.sum()) > 0
    assert float(np.abs(shading.normal_z[away]).max()) == 0.0


def test_shadow_march_resolves_fine_relief_on_a_small_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The march steps at the terrain's resolution, not the body's pixel size.

    On a 30 px body with 1 degree relief one shading pixel spans several field
    cells, so a pixel-sized step skips ridge tips.  The renderer's shadow
    decisions match a march at a quarter of its step on all but 1% of the
    shadowed pixels.
    """
    spec = EllipsoidBodySpec(
        semi_a_px=30.0,
        semi_b_px=30.0,
        semi_c_px=30.0,
        illumination_angle=math.radians(80.0),
        phase_angle=math.radians(70.0),
        anti_aliasing=0.0,
        limb_relief_rms=0.03,
        limb_relief_corr_deg=1.0,
        relief_seed=3,
    )
    grid = project_ellipsoid((80, 80), (40.0, 40.0), 30.0, semi_b_px=30.0, semi_c_px=30.0)
    field = synthesize_relief_field(0.03, 1.0, 3)
    near = grid.ellipse_dist_sq < (1.0 + field.h_max + 2.0 / 30.0) ** 2
    shadowed = _relief_shading(grid, spec, field, near).shadowed

    cell = 2.0 * np.pi / field.grid.shape[0]

    def fine_march(
        points: NDArrayFloatType, sun: NDArrayFloatType, relief: ReliefField, *, step: float
    ) -> Any:
        return cast_shadows(points, sun, relief, step=min(step, cell / 32.0))

    monkeypatch.setattr(ellipsoid_relief, 'cast_shadows', fine_march)
    reference = _relief_shading(grid, spec, field, near).shadowed
    assert int(reference.sum()) > 100
    assert int((shadowed != reference).sum()) <= 0.01 * int(reference.sum())


def _body_scene(*, oversample: int | None, body_extra: dict[str, Any]) -> dict[str, Any]:
    """A minimal single-body scene for whole-render comparisons."""
    body: dict[str, Any] = {
        'name': 'B',
        'center_v': 40.0,
        'center_u': 40.0,
        'semi_a_px': 22.0,
        'semi_b_px': 20.0,
        'semi_c_px': 20.0,
        'illumination_angle': 30.0,
        'phase_angle': 70.0,
        'crater_fill': 0.3,
        'crater_max_radius': 0.2,
    }
    body.update(body_extra)
    scene = frame_scene(size=80, seed=11, noise=quiet_noise(), bodies=[body])
    if oversample is not None:
        scene['oversample'] = oversample
    return scene


def test_relief_rms_zero_reproduces_the_relief_free_render_bit_identically() -> None:
    """limb_relief_rms 0.0 and an absent key render byte-identically (os 1 and 4)."""
    for oversample in (None, 4):
        absent, _ = render_scene(_body_scene(oversample=oversample, body_extra={}))
        zero, _ = render_scene(
            _body_scene(
                oversample=oversample,
                body_extra={'limb_relief_rms': 0.0, 'limb_relief_corr_deg': 15.0},
            )
        )
        assert np.array_equal(absent, zero)


def test_explicit_lambert_law_reproduces_the_default_bit_identically() -> None:
    """photometric_law 'lambert' and an absent key take the same render path."""
    absent, _ = render_scene(_body_scene(oversample=None, body_extra={}))
    explicit, _ = render_scene(
        _body_scene(oversample=None, body_extra={'photometric_law': 'lambert'})
    )
    assert np.array_equal(absent, explicit)


def test_non_lambert_law_changes_the_disc() -> None:
    """A non-Lambert law renders a different disc profile than Lambert."""
    lambert, _ = render_scene(_body_scene(oversample=None, body_extra={}))
    minnaert, _ = render_scene(
        _body_scene(oversample=None, body_extra={'photometric_law': 'minnaert'})
    )
    assert not np.array_equal(lambert, minnaert)


def test_opposition_surge_scales_the_disc_by_the_documented_factor() -> None:
    """The surge dims a high-phase disc by (1 + A exp(-a/w)) / (1 + A)."""
    spec = EllipsoidBodySpec(
        semi_a_px=60.0,
        semi_b_px=60.0,
        semi_c_px=60.0,
        illumination_angle=math.radians(30.0),
        phase_angle=math.radians(70.0),
        oversample=2,
    )
    plain = create_ellipsoid_body((160, 160), (80.0, 80.0), spec)
    surged_spec = EllipsoidBodySpec(
        semi_a_px=60.0,
        semi_b_px=60.0,
        semi_c_px=60.0,
        illumination_angle=math.radians(30.0),
        phase_angle=math.radians(70.0),
        oversample=2,
        surge_amplitude=1.0,
        surge_width_deg=6.0,
    )
    surged = create_ellipsoid_body((160, 160), (80.0, 80.0), surged_spec)
    factor = (1.0 + math.exp(-70.0 / 6.0)) / 2.0
    interior = (plain > 0.1) & (plain < 0.99)
    assert interior.any()
    np.testing.assert_allclose(surged[interior], plain[interior] * factor, rtol=1e-9)


def test_relief_terrain_is_stable_when_craters_toggle() -> None:
    """The relief stream is named independently of the crater draws.

    Toggling craters must not reseed the limb: the silhouette (where only
    relief acts) stays identical while the disc shading changes.
    """
    with_craters, _ = render_scene(
        _body_scene(oversample=None, body_extra={'limb_relief_rms': 0.02})
    )
    body_extra: dict[str, Any] = {'limb_relief_rms': 0.02, 'crater_fill': 0.0}
    without_craters, _ = render_scene(_body_scene(oversample=None, body_extra=body_extra))
    assert np.array_equal(with_craters > 0.0, without_craters > 0.0)
    assert not np.array_equal(with_craters, without_craters)


def test_two_identical_bodies_get_independent_terrains() -> None:
    """Per-body identity seeds two same-geometry bodies' relief independently."""
    scene = frame_scene(
        size=120,
        seed=11,
        noise=quiet_noise(),
        bodies=[
            disc_body(30.0, 30.0, 20.0, name='FIRST', limb_relief_rms=0.03, range_km=1.0),
            disc_body(88.0, 88.0, 20.0, name='SECOND', limb_relief_rms=0.03, range_km=2.0),
        ],
    )
    img, _meta = render_scene(scene)
    first = img[0:60, 0:60] > 0.0
    second = img[58:118, 58:118] > 0.0
    assert not np.array_equal(first, second)
