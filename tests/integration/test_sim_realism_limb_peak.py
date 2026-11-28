"""Tests of the realism match's limb gradient-peak statistic, and its regression check.

The regression check holds the committed realism record (or the summary that
``SPINDOCTOR_TEST_REALISM_RECORD`` points to): in every limb bin that has a verdict, the
matched frames' median gradient-peak offset must lie within the band set from the real
frames, apart from the bins :data:`KNOWN_LIMB_PEAK_FINDINGS` lists, which must still lie
outside it.  The other tests build their frames and scenes in memory.  Nothing here reads
holdings, so the file runs in the plain unit tier.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import erfc

from spindoctor.feature.geometry import LimbPolyline
from tests.integration import sim_realism_limb_peak
from tests.integration.realism_regression import REGENERATE_HINT, load_record, made_at
from tests.integration.sim_realism_limb_peak import (
    LimbPeak,
    LimbSelection,
    compare_limb_peaks,
    limb_peak_offset,
    limb_peak_refusals,
    matched_orientation,
    matched_silhouette_offsets,
    pair_limb_peaks,
    real_silhouette_offsets,
    record_has_limb_peaks,
    record_limb_peak_failures,
    tracks_touching,
)

_RECORD = load_record()

KNOWN_LIMB_PEAK_FINDINGS: dict[tuple[str, str], str] = {
    ('coiss_calib_nac', 'p0_r2'): (
        'below 60 deg phase and over 400 px across, two of the three real limbs, both '
        "Enceladus, peak 0.83 and 1.17 px outside their shape model's silhouette, where "
        'the matched spheres peak within 0.05 px of theirs'
    ),
}
"""Limb-peak bins of the record known to lie outside their band, with the finding."""

_MARGIN_VU = (5, 7)
_OBS = SimpleNamespace(extfov_margin_vu=_MARGIN_VU)
# The sphere's center at (60.5, 60.5) as a pixel corner, in extended-FOV pixel centers.
_CENTER_EXTFOV = (60.0 + _MARGIN_VU[0], 60.0 + _MARGIN_VU[1])


def _sphere_scene(center_vu: tuple[float, float], radius: float) -> dict[str, Any]:
    """A matched-style scene holding one sphere at a pixel-corner center, lit from +u."""
    return {
        'bodies': [
            {
                'name': 'REALISM_BODY',
                'center_v': center_vu[0],
                'center_u': center_vu[1],
                'semi_a_px': radius,
                'semi_b_px': radius,
                'semi_c_px': radius,
                'illumination_angle': 90.0,
                'phase_angle': 30.0,
            }
        ]
    }


def _disc_image(center_vu: tuple[float, float], radius: float, *, size: int = 120) -> np.ndarray:
    """A uniformly bright disc whose edge is blurred by a Gaussian of 0.8 px.

    The center is pixel centric, the convention of the image array.
    """
    vv, uu = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.hypot(vv - center_vu[0], uu - center_vu[1])
    return np.asarray(0.5 * erfc((r - radius) / (np.sqrt(2.0) * 0.8)))


def _limb(
    center_vu: tuple[float, float],
    radius: float,
    n: int = 90,
    *,
    facing_deg: float = 0.0,
    half_deg: float = 85.0,
    quantized: bool = False,
) -> LimbPolyline:
    """An arc of vertices 0.4 px inside a circle, in extended-FOV coordinates.

    Its middle normal points ``facing_deg`` from +u toward +v: 0 faces +u, the lit half
    under ``_sphere_scene``.  The arc reaches ``half_deg`` to each side of it.  With
    ``quantized`` each vertex's normal is rounded to a multiple of 45 degrees, as the
    navigator's are.
    """
    angles = np.radians(np.linspace(-half_deg, half_deg, n) + facing_deg)
    directions = np.column_stack([np.sin(angles), np.cos(angles)])
    vertices = np.asarray(center_vu) + directions * (radius - 0.4)
    normals = directions
    if quantized:
        steps = np.radians(np.round(np.degrees(angles) / 45.0) * 45.0)
        normals = np.column_stack([np.sin(steps), np.cos(steps)])
    return LimbPolyline(
        vertices_vu=vertices,
        normals_vu=normals,
        sigma_normal_per_vertex_px=np.full(n, 0.5),
        sigma_tangent_per_vertex_px=np.full(n, 0.5),
        bbox_extfov_vu=(0, 0, 1, 1),
    )


def test_the_matched_silhouette_is_the_scenes_circle() -> None:
    """A vertex 0.4 px inside the scene's circle lies 0.4 px from its silhouette."""
    offsets = matched_silhouette_offsets(_sphere_scene((60.5, 60.5), 30.0), _OBS)
    limb = _limb(_CENTER_EXTFOV, 30.0)
    distances = offsets('body_sim:REALISM_BODY', limb.vertices_vu, limb.normals_vu)
    assert np.max(np.abs(distances - 0.4)) < 1e-9


@pytest.mark.parametrize(
    ('illumination_deg', 'facing_deg'),
    [(90.0, 180.0), (0.0, 90.0)],
    ids=['lit-from-right', 'lit-from-top'],
)
def test_the_matched_silhouette_leaves_out_the_unlit_limb(
    illumination_deg: float, facing_deg: float
) -> None:
    """Vertices on the half of the limb facing away from the light get no silhouette.

    ``illumination_angle`` 0 lights the top of the image (-v) and 90 the right (+u), so
    the unlit half faces -u in the first case and +v in the second.
    """
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['illumination_angle'] = illumination_deg
    offsets = matched_silhouette_offsets(scene, _OBS)
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=facing_deg)
    distances = offsets('body_sim:REALISM_BODY', limb.vertices_vu, limb.normals_vu)
    assert np.all(np.isnan(distances))


def test_the_matched_silhouette_lights_a_vertex_by_its_direction_from_the_center() -> None:
    """A vertex 30 degrees from the light keeps its silhouette, whatever its normal.

    Lit from the top of the image, its normal rounded to point to the right lies at right
    angles to the light, yet the limb point it stands for is lit.
    """
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['illumination_angle'] = 0.0
    offsets = matched_silhouette_offsets(scene, _OBS)
    angle = np.radians(-60.0)
    vertex = np.asarray(_CENTER_EXTFOV) + 29.6 * np.array([[np.sin(angle), np.cos(angle)]])
    distances = offsets('body_sim:REALISM_BODY', vertex, np.array([[0.0, 1.0]]))
    assert np.isfinite(distances[0])


def test_the_matched_silhouette_leaves_out_the_limb_at_right_angles_to_the_light() -> None:
    """The two limb points at right angles to the light lie on the terminator, unlit."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['illumination_angle'] = 0.0
    offsets = matched_silhouette_offsets(scene, _OBS)
    directions = np.array([[0.0, 1.0], [0.0, -1.0]])
    vertices = np.asarray(_CENTER_EXTFOV) + 29.6 * directions
    distances = offsets('body_sim:REALISM_BODY', vertices, directions)
    assert np.all(np.isnan(distances))


def test_a_matched_scene_with_two_bodies_is_refused() -> None:
    """The matched silhouette reads a scene that holds one body."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'].append(dict(scene['bodies'][0]))
    with pytest.raises(ValueError, match='holds one body, not 2'):
        matched_silhouette_offsets(scene, _OBS)


def test_a_matched_body_that_is_not_a_sphere_is_refused() -> None:
    """The matched silhouette is a circle, so a flattened body is refused."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['semi_c_px'] = 28.0
    with pytest.raises(ValueError, match="'REALISM_BODY' is not a sphere"):
        matched_silhouette_offsets(scene, _OBS)


def _measure(
    limb: LimbPolyline,
    image: np.ndarray,
    *,
    scene: dict[str, Any] | None = None,
    clear_track: Any = None,
    peak_deg: tuple[float, ...] | None = None,
) -> LimbPeak:
    """The gradient peak of ``limb`` on ``image`` under a sphere scene.

    The scene is :func:`_sphere_scene`'s, lit from +u, unless another is given; the
    selection takes its orientation from the same scene.
    """
    scene = _sphere_scene((60.5, 60.5), 30.0) if scene is None else scene
    return limb_peak_offset(
        image,
        limb,
        'body_sim:REALISM_BODY',
        shift_vu=-np.asarray(_MARGIN_VU, dtype=np.float64),
        silhouette_offsets=matched_silhouette_offsets(scene, _OBS),
        selection=LimbSelection(
            matched_orientation(scene), clear_track=clear_track, peak_deg=peak_deg
        ),
    )


@pytest.mark.parametrize(
    ('edge_radius', 'expected'),
    [(30.0, 0.0), (29.5, -0.5)],
    ids=['edge-on-silhouette', 'edge-inside'],
)
def test_a_limbs_peak_offset_is_its_edge_less_its_silhouette(
    edge_radius: float, expected: float
) -> None:
    """A disc stated at radius 30 with its edge at ``edge_radius`` peaks that far off."""
    peak = _measure(_limb(_CENTER_EXTFOV, 30.0), _disc_image((60.0, 60.0), edge_radius))
    assert peak.offset_px == pytest.approx(expected, abs=0.02)


def test_a_misplaced_limb_moves_the_translation_and_keeps_the_offset() -> None:
    """A frame offset 1.3 px wrong toward the lit limb leaves the peak offset alone."""
    peak = _measure(_limb(_CENTER_EXTFOV, 30.0), _disc_image((60.0, 61.3), 29.5))
    assert peak.offset_px == pytest.approx(-0.5, abs=0.03)


def test_a_limb_at_zero_phase_is_measured() -> None:
    """At zero phase the whole limb is lit, so even the half facing -u is measured."""
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['phase_angle'] = 0.0
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=180.0)
    peak = _measure(limb, _disc_image((60.0, 60.0), 30.0), scene=scene)
    assert peak.offset_px == pytest.approx(0.0, abs=0.02)


def test_a_limb_is_read_along_its_directions_from_the_center() -> None:
    """A limb whose normals are rounded to 45 degree steps peaks where its edge lies.

    Read along the rounded normals, the edge would lie up to 1 / cos(22.5 deg) times
    farther from the silhouette, 0.54 px for an edge 0.5 px inside it.
    """
    limb = _limb(_CENTER_EXTFOV, 30.0, quantized=True)
    peak = _measure(limb, _disc_image((60.0, 60.0), 29.5))
    assert peak.offset_px == pytest.approx(-0.5, abs=0.02)


def test_a_noisy_limb_on_a_short_arc_is_refused_for_its_sigma() -> None:
    """A noisy 30 degree arc fits, but knows its offset to worse than 0.2 px.

    Over so short an arc a radial offset and a translation along the arc's middle normal
    move the peaks almost alike, so the noise is magnified into the offset.
    """
    rng = np.random.default_rng(1)
    image = _disc_image((60.0, 60.0), 30.0) + rng.normal(0.0, 0.05, (120, 120))
    peak = _measure(_limb(_CENTER_EXTFOV, 30.0, 40, half_deg=15.0), image)
    assert peak.refusal is not None
    assert 'worse than 0.2 px' in peak.refusal


@pytest.mark.parametrize('facing_deg', [0.0, 11.25, 22.5, 33.75])
def test_a_short_arc_is_refused_however_it_is_turned(facing_deg: float) -> None:
    """A clean 120 degree arc is refused for its arc at every turn of the limb.

    Read along its directions from the center, the arc's offset correlates with the
    translation at 0.983, above the 0.97 ceiling, however it is turned.  Read along the
    normals rounded to 45 degrees, the correlation would range from 0.959 to 0.985 as the
    arc turns, and the arc would be kept at some turns.  The whole limb is lit at zero
    phase, so every turn is measured.
    """
    scene = _sphere_scene((60.5, 60.5), 30.0)
    scene['bodies'][0]['phase_angle'] = 0.0
    limb = _limb(_CENTER_EXTFOV, 30.0, 120, facing_deg=facing_deg, half_deg=60.0, quantized=True)
    peak = _measure(limb, _disc_image((60.0, 60.0), 30.0), scene=scene)
    assert peak.refusal is not None
    assert 'the arc is too short' in peak.refusal


def test_a_vertex_whose_track_touches_another_body_is_left_out() -> None:
    """The vertices whose 4 px track reaches another body's disc give no peak.

    The other body covers the sky just outside the limb from 20 to 50 degrees past the
    direction the light comes from, so the vertices between 25 and 45 degrees are left
    out.
    """
    vv, uu = np.mgrid[0:120, 0:120].astype(np.float64)
    angle = np.degrees(np.arctan2(vv - 60.0, uu - 60.0))
    radius = np.hypot(vv - 60.0, uu - 60.0)
    other = (angle >= 20.0) & (angle <= 50.0) & (radius > 31.0) & (radius < 36.0)
    peak = _measure(
        _limb(_CENTER_EXTFOV, 30.0),
        _disc_image((60.0, 60.0), 30.0),
        clear_track=lambda points, normals, half: ~tracks_touching(other, points, normals, half),
    )
    assert len(peak.used_deg) > 60
    assert not any(25.0 <= d <= 45.0 for d in peak.used_deg)


def test_the_matched_limb_is_measured_over_the_real_limbs_directions() -> None:
    """A real limb that used only the half before the light's direction limits its twin.

    The real fit read directions from -85 to -1 degrees, so the twin's vertices beyond
    two vertex spacings past -1 degree are left out.
    """
    used = tuple(float(d) for d in np.linspace(-85.0, -1.0, 45))
    limb = _limb(_CENTER_EXTFOV, 30.0)
    peak = _measure(limb, _disc_image((60.0, 60.0), 30.0), peak_deg=used)
    assert max(peak.used_deg) < 3.0


def test_a_matched_limb_keeps_the_arc_the_real_limb_kept() -> None:
    """A matched limb measured over a real limb's 90 degree arc is not refused for it.

    The real limb's own fit judged the arc; the matched limb only follows it.
    """
    used = tuple(float(d) for d in np.linspace(-45.0, 45.0, 45))
    limb = _limb(_CENTER_EXTFOV, 30.0)
    peak = _measure(limb, _disc_image((60.0, 60.0), 30.0), peak_deg=used)
    assert peak.refusal is None


def test_a_refused_limb_is_listed_with_its_reason() -> None:
    """A real limb refused beside a measured twin gives one entry, for the real side."""
    refusals = limb_peak_refusals(
        'N1', 'DIONE', LimbPeak(None, 'the arc is too short'), LimbPeak(0.1, None)
    )
    assert refusals == [
        {'image_id': 'N1', 'body': 'DIONE', 'side': 'real', 'reason': 'the arc is too short'}
    ]


class _CircleBackplane:
    """A backplane of a round body 30 px in radius at (60.5, 60.5), lit from +u.

    It reads its meshgrid as an ``(N, 2)`` array of (u, v) pixel-corner coordinates.
    """

    def __init__(self, _obs: Any, *, meshgrid: np.ndarray) -> None:
        du = meshgrid[:, 0] - 60.5
        dv = meshgrid[:, 1] - 60.5
        self.hits = np.hypot(du, dv) < 30.0
        self.incidence = np.where(du > 0.0, np.pi / 4.0, 3.0 * np.pi / 4.0)

    def where_intercepted(self, _body: str) -> Any:
        return SimpleNamespace(vals=self.hits)

    def incidence_angle(self, _body: str) -> Any:
        return SimpleNamespace(mvals=np.ma.array(self.incidence, mask=~self.hits))


@pytest.fixture
def _circle_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real silhouette's oops calls, replaced by :class:`_CircleBackplane`."""
    monkeypatch.setattr(sim_realism_limb_peak, 'Pair', lambda uv: uv)
    monkeypatch.setattr(sim_realism_limb_peak, 'Meshgrid', lambda _fov, uv: uv)
    monkeypatch.setattr(sim_realism_limb_peak, 'Backplane', _CircleBackplane)


@pytest.mark.usefixtures('_circle_geometry')
def test_the_real_silhouette_is_where_the_rays_stop_hitting_the_body() -> None:
    """A vertex 0.4 px inside the body's edge lies 0.4 px from its silhouette.

    The rays are traced every 1/64 px, so the silhouette is found to within that.
    """
    offsets = real_silhouette_offsets(SimpleNamespace(extfov_margin_vu=_MARGIN_VU, fov=None))
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=0.0)
    distances = offsets('body:MOON', limb.vertices_vu, limb.normals_vu)
    assert np.max(np.abs(distances - 0.4)) <= 1.0 / 64.0


@pytest.mark.usefixtures('_circle_geometry')
def test_the_real_silhouette_leaves_out_the_unlit_limb() -> None:
    """Vertices where the Sun is below the horizon just inside the edge get NaN."""
    offsets = real_silhouette_offsets(SimpleNamespace(extfov_margin_vu=_MARGIN_VU, fov=None))
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=180.0)
    distances = offsets('body:MOON', limb.vertices_vu, limb.normals_vu)
    assert np.all(np.isnan(distances))


class _MoonBody:
    """A stand-in for oops' MOON body: a sphere whose radius, in km, is 1 px."""

    def __init__(self) -> None:
        self.surface = SimpleNamespace(origin='MOON', frame='IAU_MOON', radii=(30.0, 30.0, 30.0))
        self.radius = 30.0
        self.inner_radius = 30.0

    def apply_surface(self, surface: Any, radius: float, inner_radius: float) -> None:
        self.surface = surface
        self.radius = radius
        self.inner_radius = inner_radius


class _MoonBackplane(_CircleBackplane):
    """A circle backplane whose radius is the MOON body's surface radius at the time."""

    moon = _MoonBody()
    get_surface = SimpleNamespace(cache_clear=lambda: None)

    def __init__(self, _obs: Any, *, meshgrid: np.ndarray) -> None:
        super().__init__(_obs, meshgrid=meshgrid)
        radius = self.moon.surface.radii[0]
        self.hits = np.hypot(meshgrid[:, 0] - 60.5, meshgrid[:, 1] - 60.5) < radius


@pytest.fixture
def moon_body(monkeypatch: pytest.MonkeyPatch) -> _MoonBody:
    """The real silhouette's oops calls, replaced by :class:`_MoonBackplane`."""
    moon = _MoonBody()
    monkeypatch.setattr(_MoonBackplane, 'moon', moon)
    monkeypatch.setattr(sim_realism_limb_peak, 'Pair', lambda uv: uv)
    monkeypatch.setattr(sim_realism_limb_peak, 'Meshgrid', lambda _fov, uv: uv)
    monkeypatch.setattr(sim_realism_limb_peak, 'Backplane', _MoonBackplane)
    monkeypatch.setattr(sim_realism_limb_peak, 'Body', SimpleNamespace(lookup=lambda _n: moon))
    monkeypatch.setattr(
        sim_realism_limb_peak,
        'Ellipsoid',
        lambda origin, frame, radii: SimpleNamespace(origin=origin, frame=frame, radii=radii),
    )
    return moon


def test_a_body_given_radii_is_traced_at_them(moon_body: _MoonBody) -> None:
    """A body 0.5 px larger puts the silhouette 0.5 px further out."""
    offsets = real_silhouette_offsets(
        SimpleNamespace(extfov_margin_vu=_MARGIN_VU, fov=None),
        radii_km={'MOON': (30.5, 30.5, 30.5)},
    )
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=0.0)
    distances = offsets('body:MOON', limb.vertices_vu, limb.normals_vu)
    assert np.max(np.abs(distances - 0.9)) <= 1.0 / 64.0


def test_a_body_given_radii_gets_its_own_shape_back(moon_body: _MoonBody) -> None:
    """After the trace the body has the surface it had before."""
    surface = moon_body.surface
    offsets = real_silhouette_offsets(
        SimpleNamespace(extfov_margin_vu=_MARGIN_VU, fov=None),
        radii_km={'MOON': (30.5, 30.5, 30.5)},
    )
    limb = _limb(_CENTER_EXTFOV, 30.0, facing_deg=0.0)
    offsets('body:MOON', limb.vertices_vu, limb.normals_vu)
    assert moon_body.surface is surface


def test_a_frame_pair_that_both_measured_the_limb_keeps_both_offsets() -> None:
    """One offset on each side, in the same bin, enters the pools."""
    real = {'limb_peak_p0_r0': [-0.02], 'limb_width_p0_r0': [1.1]}
    sim = {'limb_peak_p0_r0': [0.03]}
    pair_limb_peaks(real, sim)
    assert real == {'limb_peak_p0_r0': [-0.02], 'limb_width_p0_r0': [1.1]}
    assert sim == {'limb_peak_p0_r0': [0.03]}


def test_a_matched_limb_without_its_real_twin_is_dropped() -> None:
    """A matched offset whose real frame measured no limb is removed, and nothing else."""
    real = {'limb_width_p0_r0': [1.1]}
    sim = {'limb_peak_p0_r0': [0.42], 'limb_width_p0_r0': [1.0]}
    pair_limb_peaks(real, sim)
    assert sim == {'limb_width_p0_r0': [1.0]}


def test_a_frame_pair_whose_offsets_fall_in_different_bins_is_dropped() -> None:
    """Offsets of a real and a matched limb in different bins are both removed."""
    real = {'limb_peak_p0_r1': [0.1]}
    sim = {'limb_peak_p0_r0': [0.2]}
    pair_limb_peaks(real, sim)
    assert (real, sim) == ({}, {})


def test_a_bin_with_three_real_limbs_sets_its_band_from_its_own_scatter() -> None:
    """Three real limbs at -0.5, -0.3, and -0.1 px give a band of 0.34 px."""
    bins = compare_limb_peaks({'limb_peak_p0_r1': [-0.5, -0.3, -0.1]}, {'limb_peak_p0_r1': [-0.05]})
    peak = bins['p0_r1']
    assert peak.band_px == pytest.approx(2.0 * 1.4826 * 0.2 / np.sqrt(3.0))
    assert peak.within_band is True


def test_a_matched_median_beyond_the_band_is_outside_it() -> None:
    """A matched median 0.36 px from the real one lies outside a 0.34 px band."""
    bins = compare_limb_peaks({'limb_peak_p0_r1': [-0.5, -0.3, -0.1]}, {'limb_peak_p0_r1': [0.06]})
    assert bins['p0_r1'].within_band is False


def test_a_bin_with_two_real_limbs_is_not_judged() -> None:
    """Two real limbs give no band and no verdict, however many the instrument has."""
    real = {'limb_peak_p0_r1': [-0.5, -0.3, -0.1], 'limb_peak_p1_r2': [0.4, 1.08]}
    bins = compare_limb_peaks(real, {'limb_peak_p1_r2': [0.0]})
    assert bins['p1_r2'].band_px is None
    assert bins['p1_r2'].within_band is None


def test_a_tight_bin_is_held_to_the_band_floor() -> None:
    """Real limbs that agree exactly still get a band of 0.1 px."""
    bins = compare_limb_peaks({'limb_peak_p0_r0': [0.2, 0.2, 0.2]}, {'limb_peak_p0_r0': [0.0]})
    assert bins['p0_r0'].band_px == pytest.approx(0.1)


def test_a_matched_median_on_the_band_edge_is_within_it() -> None:
    """A matched median exactly one band half-width from the real median is within."""
    bins = compare_limb_peaks({'limb_peak_p0_r0': [0.0, 0.0, 0.0]}, {'limb_peak_p0_r0': [0.1]})
    assert bins['p0_r0'].within_band is True


def test_a_bin_without_matched_limbs_is_not_judged() -> None:
    """A bin populated only by real limbs records its band and no verdict."""
    bins = compare_limb_peaks({'limb_peak_p0_r0': [0.2, 0.0, -0.2]}, {})
    assert bins['p0_r0'].within_band is None


def _record_bin(within: bool | None) -> dict[str, Any]:
    """One limb bin's entry as the realism summary writes it."""
    return {
        'n_real': 4,
        'n_sim': 4,
        'real_median_px': -0.1,
        'sim_median_px': 0.3,
        'real_scatter_px': 0.2,
        'band_px': 0.2,
        'within_band': within,
    }


def test_a_record_bin_outside_its_band_is_a_failure() -> None:
    """Only the bin whose verdict is False is reported."""
    record = {
        'instruments': {
            'coiss_calib_nac': {
                'limb_peaks': {
                    'p0_r0': _record_bin(False),
                    'p0_r1': _record_bin(True),
                    'p1_r0': _record_bin(None),
                }
            }
        }
    }
    failures = record_limb_peak_failures(record)
    assert failures == {
        ('coiss_calib_nac', 'p0_r0'): 'coiss_calib_nac p0_r0: matched median +0.300 px '
        '(4 limbs) lies more than 0.200 px from the real median -0.100 px (4 limbs)'
    }


@pytest.mark.skipif(
    not record_has_limb_peaks(_RECORD),
    reason='the realism record predates the limb gradient-peak comparison',
)
def test_the_recorded_limb_peaks_lie_within_their_bands() -> None:
    """In every bin with a verdict, the matched median lies within the real band.

    A bin :data:`KNOWN_LIMB_PEAK_FINDINGS` lists is a recorded finding: it must still lie
    outside its band, and while one does, the test is an expected failure that gives the
    finding.  A listed bin back within its band fails, so the list can only shrink.
    """
    failures = record_limb_peak_failures(_RECORD)
    new = [line for key, line in failures.items() if key not in KNOWN_LIMB_PEAK_FINDINGS]
    assert not new, (
        f'The limb gradient peaks of the record made at {made_at(_RECORD)} leave their '
        'bands:\n' + '\n'.join(new) + f'\n{REGENERATE_HINT}'
    )
    back = sorted(set(KNOWN_LIMB_PEAK_FINDINGS) - set(failures))
    assert not back, f'These known findings now lie within their bands; drop them: {back}'
    if failures:
        pytest.xfail(
            'known limb-peak findings: '
            + '; '.join(KNOWN_LIMB_PEAK_FINDINGS[key] for key in sorted(failures))
        )
