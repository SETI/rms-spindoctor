"""Rendered-body acceptance checks for the limb-relief field.

The relief contract at the rendered-image level: the silhouette radius at
azimuth theta lands at ``r_ellipse * (1 + delta(theta))``, the terminator
grows ragged with excursions concentrated at (and bounded near) the
terminator, relief OFF reproduces the relief-free render bit-identically,
and each body's terrain realization is seeded independently.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim import render
from spindoctor.sim.forward import body_topo
from spindoctor.sim.forward.body_topo import TopoBodySpec, create_topographic_body
from spindoctor.sim.forward.photometry import DARK_SIDE_FLOOR
from spindoctor.sim.forward.relief import march_shadows, shadow_reach, synthesize_relief_field
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


def test_limb_lands_at_r_ellipse_times_one_plus_delta() -> None:
    """Silhouette radius matches r_ellipse * (1 + delta(theta)) within 0.5 os px.

    Per-ray radii carry ~half-pixel staircase noise from the rasterized
    mask, so the acceptance is asserted on azimuth-window means (16 windows
    of 16 rays); the pooled displacement must also track the commanded
    field realization tightly.
    """
    seed = 42
    size = 256
    img = create_topographic_body(
        (size, size), (size / 2.0, size / 2.0), _limb_spec(rms=_RMS, seed=seed)
    )
    mask = img > 0.0
    field = synthesize_relief_field(_RMS, _CORR_DEG, seed)

    n_windows, rays_per_window = 16, 16
    angles = np.arange(n_windows * rays_per_window) * (2.0 * np.pi / (n_windows * rays_per_window))
    measured = np.array([_ray_edge_radius(mask, size / 2.0, a) for a in angles])
    # phi convention: atan2(u_rot / b, v_rot / a); for an unrotated sphere
    # this is the image azimuth of the ray itself.
    predicted = 80.0 * (1.0 + field.limb_delta(angles))

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


def test_terminator_is_ragged_with_excursions_at_the_terminator() -> None:
    """Relief shadows displace the terminator raggedly, bounded near it.

    Comparing the relief render against the smooth render of the same
    geometry: newly dark pixels exist, they concentrate against the smooth
    terminator (the shadowed fraction falls with distance into the lit
    side), and none lies beyond the march cap's horizon bound.
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

    # Distance of each lit pixel into the lit side, from the smooth
    # terminator (the boundary of the smooth lit region).
    distance_into_lit = ndimage.distance_transform_edt(lit_smooth)
    shadow_distances = distance_into_lit[newly_dark]

    # Bounded by the horizon cap: no shadow reaches beyond
    # sqrt(2 * R * (H_max - H_min)) surface px of the terminator, the horizon
    # of the highest peak seen from the deepest valley (image distance is
    # never longer than surface distance here).
    field = synthesize_relief_field(_RMS, _CORR_DEG, 5)
    cap_px = 120.0 * math.sqrt(2.0 * (field.h_max - field.h_min))
    assert float(shadow_distances.max()) <= cap_px + 2.0

    # Excursions grow toward the terminator: the shadowed fraction falls
    # with distance from it.
    def shadow_fraction(lo: float, hi: float) -> float:
        band = lit_smooth & (distance_into_lit > lo) & (distance_into_lit <= hi)
        assert int(band.sum()) > 0
        return float(newly_dark[band].sum()) / float(band.sum())

    near = shadow_fraction(0.0, cap_px / 3.0)
    mid = shadow_fraction(cap_px / 3.0, 2.0 * cap_px / 3.0)
    far = shadow_fraction(2.0 * cap_px / 3.0, cap_px)
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


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_relief_leaves_the_high_sun_disc_untouched(seed: int) -> None:
    """At 2% relief, every pixel the sun lights from above 24 degrees renders unchanged.

    The geometry is the one that exposed radial wedges cut through the middle of the
    disc (an 85 px moon at phase 58 degrees).  Relief of 2% of the radius with a
    15-degree correlation length has slopes of ~9 degrees RMS, so its shadows stay in
    the low-sun band along the terminator; the silhouette moves only near the limb,
    which the inner-disc mask excludes.  The relief must still shadow something there.
    """
    size = 256
    spec = TopoBodySpec(
        axis1=170.0,
        axis2=168.0,
        axis3=164.0,
        illumination_angle=math.radians(28.0),
        phase_angle=math.radians(58.0),
        photometric_law='lommel_seeliger',
        limb_relief_rms=_RMS,
        limb_relief_corr_deg=_CORR_DEG,
        relief_seed=seed,
    )
    smooth_spec = TopoBodySpec(
        axis1=spec.axis1,
        axis2=spec.axis2,
        axis3=spec.axis3,
        illumination_angle=spec.illumination_angle,
        phase_angle=spec.phase_angle,
        photometric_law=spec.photometric_law,
    )
    center = (size / 2.0, size / 2.0)
    smooth = create_topographic_body((size, size), center, smooth_spec)
    ragged = create_topographic_body((size, size), center, spec)
    elevation = _sun_elevation_deg(size, semi_axes=(85.0, 84.0), spec=spec)
    v, u = np.mgrid[0:size, 0:size].astype(float) + 0.5 - size / 2.0
    inner = np.hypot(v / 85.0, u / 84.0) < 0.85
    changed = inner & (np.abs(ragged - smooth) > 1e-9)
    assert int(changed.sum()) > 0
    assert int(np.sum(changed & (elevation > 24.0))) == 0


def test_shadow_maps_cover_every_candidates_reach(monkeypatch: pytest.MonkeyPatch) -> None:
    """The march domain contains each candidate's whole sunward reach on the disc.

    The march stops where it leaves the domain, so a domain shorter than a
    candidate's reach (``shadow_reach``: up to ``sqrt(2 R (H_max - H_min))`` for a
    valley point) would silently drop shadows its distant peaks cast.  Image distance
    never exceeds surface distance for an untilted body, so the image segment of
    length ``d_max`` toward the sun bounds where the march can sample.
    """
    calls: list[dict[str, Any]] = []
    real_march = march_shadows

    def recording_march(start_v: Any, start_u: Any, **kwargs: Any) -> Any:
        calls.append({'start_v': start_v, 'start_u': start_u, **kwargs})
        return real_march(start_v, start_u, **kwargs)

    monkeypatch.setattr(body_topo, 'march_shadows', recording_march)
    size = 256
    spec = TopoBodySpec(
        axis1=170.0,
        axis2=170.0,
        axis3=170.0,
        illumination_angle=math.radians(28.0),
        phase_angle=math.radians(58.0),
        limb_relief_rms=0.05,
        limb_relief_corr_deg=_CORR_DEG,
        relief_seed=3,
    )
    create_topographic_body((size, size), (size / 2.0, size / 2.0), spec)
    assert len(calls) == 1
    call = calls[0]
    d_max = shadow_reach(
        call['h_point'], call['tan_incidence'], call['radius_px'], h_max=call['h_max']
    )
    v, u = np.mgrid[0:size, 0:size].astype(float) + 0.5 - size / 2.0
    disc = np.hypot(v, u) < 85.0
    steps = np.arange(0.0, float(d_max.max()) + 1.0, 0.5)
    pos_v = call['start_v'][:, np.newaxis] + call['sun_v'] * steps[np.newaxis, :]
    pos_u = call['start_u'][:, np.newaxis] + call['sun_u'] * steps[np.newaxis, :]
    within = steps[np.newaxis, :] <= d_max[:, np.newaxis]
    rows = np.clip(np.floor(pos_v).astype(int), 0, size - 1)
    cols = np.clip(np.floor(pos_u).astype(int), 0, size - 1)
    reachable = within & disc[rows, cols]
    assert int(reachable.sum()) > 0
    assert bool(np.all(call['domain_mask'][rows[reachable], cols[reachable]]))


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
