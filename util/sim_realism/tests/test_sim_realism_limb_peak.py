"""Tests of the realism match's limb gradient-peak statistic.

The tests build their frames and scenes in memory.  Nothing here reads holdings or the
committed record, and CI does not run the file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.special import erfc
from util.sim_realism import sim_realism_limb_peak
from util.sim_realism.sim_realism_limb_peak import (
    LimbPeak,
    LimbSelection,
    compare_limb_peaks,
    limb_peak_offset,
    matched_orientation,
    matched_silhouette_offsets,
    pair_limb_peaks,
    real_silhouette_offsets,
    tracks_touching,
)

from spindoctor.feature.geometry import LimbPolyline

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
