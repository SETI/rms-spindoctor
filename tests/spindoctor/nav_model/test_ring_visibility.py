"""Tests for the ring model's visibility pre-check.

The pre-check decides, from the extended frame's boundary and the planet center alone,
whether the ring plane out to the catalog's outermost radius can appear anywhere in the
frame. These tests build the geometry from first principles: a pinhole camera over an
unbounded ring plane ``z = 0`` centered on the planet, with no SPICE kernels. The camera
answers ``ring_radius`` for whatever pixel coordinates the code under test asks about, so a
check that samples the frame too sparsely sees exactly what it would see on a real frame.

The frame is the 100 x 100 detector of :func:`_obs` plus a 5 pixel margin, so its pixel
centers run from -4.5 to 104.5 in the geometry layer's coordinates. A 16 x 16 grid over it
would sample every sixth pixel, at -4.5, 1.5, ..., 49.5, 55.5, ..., and the scenes that
must be seen place the rings between those samples.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polymath
import pytest
from tests.shims import FakeObs

import spindoctor.nav_model.ring_visibility as ring_visibility_module
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.ring_visibility import (
    boundary_closest_approach_km,
    frame_edge_sides,
    frame_edge_uv,
    ring_disc_view,
)

_TARGET = 'saturn:ring'
_PLANET = 'SATURN'
_OUTER_RADIUS_KM = 140_000.0


@dataclass(frozen=True)
class _PinholeRingScene:
    """A pinhole camera looking at the unbounded ring plane ``z = 0``.

    The planet sits at the origin of the ring frame, the ring plane is ``z = 0``, and
    pixel ``(u, v)`` looks along ``boresight + du * ifov * right + dv * ifov * down``,
    where ``(du, dv)`` is ``(u - u0, v - v0)`` turned by the camera's roll about the
    boresight and ``(u0, v0)`` is where the boresight lands.

    Parameters:
        observer_km: Observer position in the ring frame, in km.
        target_km: Point the boresight passes through, in km.
        ifov_rad: Angular size of one pixel, in radians.
        boresight_uv: ``(u, v)`` where the boresight lands, in geometry coordinates.
        roll_rad: Camera roll about the boresight, in radians.
        nan_misses: True to report a ray that misses the plane as an unmasked NaN
            instead of a masked value.
    """

    observer_km: tuple[float, float, float]
    target_km: tuple[float, float, float]
    ifov_rad: float
    boresight_uv: tuple[float, float] = (50.0, 50.0)
    roll_rad: float = 0.0
    nan_misses: bool = False

    def _axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the unit boresight, right, and down vectors."""
        boresight = np.subtract(self.target_km, self.observer_km)
        boresight = boresight / np.linalg.norm(boresight)
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(boresight @ up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(boresight, up)
        right = right / np.linalg.norm(right)
        down = np.cross(boresight, right)
        return boresight, right, down

    def _intercepts(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the ring-plane ``(x, y)`` of each ray and where the rays miss the plane.

        Parameters:
            uv: Array of shape ``(..., 2)`` of pixel coordinates.
        """
        boresight, right, down = self._axes()
        du0 = uv[..., 0] - self.boresight_uv[0]
        dv0 = uv[..., 1] - self.boresight_uv[1]
        cos_roll, sin_roll = np.cos(self.roll_rad), np.sin(self.roll_rad)
        du = (cos_roll * du0 - sin_roll * dv0) * self.ifov_rad
        dv = (sin_roll * du0 + cos_roll * dv0) * self.ifov_rad
        rays = boresight + du[..., None] * right + dv[..., None] * down
        observer = np.asarray(self.observer_km)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -observer[2] / rays[..., 2]
        misses = ~(t > 0.0)
        hit = observer + np.where(misses, 0.0, t)[..., None] * rays
        return hit[..., :2], misses

    def _scalar(self, vals: np.ndarray, misses: np.ndarray) -> polymath.Scalar:
        """Wrap per-ray values, marking misses as masked or as NaN."""
        if self.nan_misses:
            return polymath.Scalar(np.where(misses, np.nan, vals))
        return polymath.Scalar(vals, misses)

    def ring_radius(self, uv: np.ndarray) -> polymath.Scalar:
        """Ring-plane radius at each ``(u, v)``, marked where the ray misses the plane.

        Parameters:
            uv: Array of shape ``(..., 2)`` of pixel coordinates.
        """
        xy, misses = self._intercepts(uv)
        return self._scalar(np.hypot(xy[..., 0], xy[..., 1]), misses)

    def ring_longitude(self, uv: np.ndarray) -> polymath.Scalar:
        """Ring-plane longitude at each ``(u, v)``, marked where the ray misses the plane.

        Parameters:
            uv: Array of shape ``(..., 2)`` of pixel coordinates.
        """
        xy, misses = self._intercepts(uv)
        return self._scalar(np.arctan2(xy[..., 1], xy[..., 0]), misses)

    def planet_uv(self) -> tuple[float, float]:
        """Return where the planet center projects, in geometry coordinates."""
        boresight, right, down = self._axes()
        to_planet = -np.asarray(self.observer_km)
        depth = float(to_planet @ boresight)
        du = float(to_planet @ right) / depth / self.ifov_rad
        dv = float(to_planet @ down) / depth / self.ifov_rad
        cos_roll, sin_roll = np.cos(self.roll_rad), np.sin(self.roll_rad)
        return (
            self.boresight_uv[0] + float(cos_roll * du + sin_roll * dv),
            self.boresight_uv[1] + float(-sin_roll * du + cos_roll * dv),
        )


class _SceneBackplane:
    """Backplane stand-in answering ``ring_radius`` and ``ring_longitude`` from a scene.

    Parameters:
        scene: The scene to evaluate.
        uv: The meshgrid's pixel coordinates.
    """

    def __init__(self, scene: _PinholeRingScene, uv: np.ndarray) -> None:
        self._scene = scene
        self._uv = uv

    def ring_radius(self, ring_target: str) -> polymath.Scalar:
        """Return the scene's ring radius at the meshgrid's pixel coordinates."""
        del ring_target
        return self._scene.ring_radius(self._uv)

    def ring_longitude(self, ring_target: str) -> polymath.Scalar:
        """Return the scene's ring longitude at the meshgrid's pixel coordinates."""
        del ring_target
        return self._scene.ring_longitude(self._uv)


def _obs(
    monkeypatch: pytest.MonkeyPatch,
    scene: _PinholeRingScene,
    *,
    planet_uv: tuple[float, float] | None = None,
    project_planet: bool = True,
) -> FakeObs:
    """Wire the pre-check's backplane to a scene and plant the projected planet center.

    Parameters:
        monkeypatch: Pytest monkeypatch fixture.
        scene: Scene the boundary backplane evaluates.
        planet_uv: Planted planet center; None plants the scene's own projection.
        project_planet: False plants a planet center that cannot be projected.
    """
    center = planet_uv if planet_uv is not None else scene.planet_uv()
    obs = FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(5, 5),
        closest_planet=_PLANET,
        path_uv={_PLANET: center if project_planet else None},
    )
    monkeypatch.setattr(ring_visibility_module, 'Meshgrid', lambda _fov, uv: np.asarray(uv))
    monkeypatch.setattr(
        ring_visibility_module,
        'Backplane',
        lambda _obs, *, meshgrid: _SceneBackplane(scene, meshgrid),
    )
    return obs


# A system 7e8 km away spans about 4 pixels and is centered between grid samples.
_DISTANT_SYSTEM = _PinholeRingScene(
    observer_km=(-7.0e8 * np.cos(0.5), 0.0, 7.0e8 * np.sin(0.5)),
    target_km=(0.0, 0.0, 0.0),
    ifov_rad=1.0e-4,
    boresight_uv=(52.5, 52.5),
)

# From 1000 km above the ring plane and 1e6 km away, the rings at the ansa are a band
# about 3 pixels tall crossing the whole frame between two grid rows; the planet is far
# off the right of the frame.
_EDGE_ON_BAND = _PinholeRingScene(
    observer_km=(-1.0e6, 0.0, 1000.0),
    target_km=(0.0, 100_000.0, 0.0),
    ifov_rad=5.0e-5,
    boresight_uv=(50.0, 52.5),
)

# From 150 km above the ring plane, rolled 0.2 rad, the rings at the ansa are a tilted
# band thinner than a pixel. It crosses the frame's boundary between two neighboring
# boundary pixel centers, so no boundary pixel center lies on it, though dozens of
# interior pixel centers do.
_SUBPIXEL_TILTED_BAND = _PinholeRingScene(
    observer_km=(-1.0e6, 0.0, 150.0),
    target_km=(0.0, 100_000.0, 0.0),
    ifov_rad=5.0e-5,
    boresight_uv=(50.0, 52.0),
    roll_rad=0.2,
)

# The frame looks at the ring plane 600,000 km from the planet.
_BEYOND_THE_RINGS = _PinholeRingScene(
    observer_km=(-1.0e6, 0.0, 300_000.0),
    target_km=(0.0, 600_000.0, 0.0),
    ifov_rad=1.0e-4,
)

# The frame looks above the ring plane's horizon, so no ray meets the plane.
_ABOVE_THE_HORIZON = _PinholeRingScene(
    observer_km=(-1.0e6, 0.0, 300_000.0),
    target_km=(0.0, 0.0, 900_000.0),
    ifov_rad=1.0e-4,
)

# The observer hangs 20,000 km over the B ring looking straight down, so the whole frame
# is ring plane inside the catalog and the planet is far outside the frame.
_INSIDE_THE_DISC = _PinholeRingScene(
    observer_km=(100_000.0, 0.0, 20_000.0),
    target_km=(100_000.0, 0.0, 0.0),
    ifov_rad=1.0e-3,
)


def test_frame_edge_uv_is_every_boundary_pixel_center() -> None:
    """The boundary samples are the outermost pixel centers on all four sides, once each."""
    obs = FakeObs(data=np.zeros((4, 3), dtype=np.float64), extfov_margin_vu=(1, 2))
    uv = frame_edge_uv(cast(Any, obs))
    us = np.arange(-2, 5) + 0.5
    vs = np.arange(-1, 5) + 0.5
    expected = {(u, v) for u in us for v in vs if u in (us[0], us[-1]) or v in (vs[0], vs[-1])}
    assert sorted(map(tuple, uv.tolist())) == sorted(expected)
    steps = np.abs(np.diff(uv, axis=0, append=uv[:1])).sum(axis=-1)
    assert steps.tolist() == [1.0] * len(uv)


def test_frame_edge_sides_run_corner_to_corner() -> None:
    """Each side lists its samples in order from one corner to the next."""
    obs = FakeObs(data=np.zeros((4, 3), dtype=np.float64), extfov_margin_vu=(1, 2))
    uv = frame_edge_uv(cast(Any, obs))
    sides = frame_edge_sides(7, 6)
    corners = [(uv[side[0]].tolist(), uv[side[-1]].tolist()) for side in sides]
    assert corners == [
        ([-1.5, -0.5], [4.5, -0.5]),
        ([4.5, -0.5], [4.5, 4.5]),
        ([4.5, 4.5], [-1.5, 4.5]),
        ([-1.5, 4.5], [-1.5, -0.5]),
    ]


def test_distant_system_between_grid_samples_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ring system a few pixels across, inside the frame, is not skipped."""
    obs = _obs(monkeypatch, _DISTANT_SYSTEM)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_edge_on_band_between_grid_rows_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A thin near edge-on band crossing the frame is not skipped."""
    obs = _obs(monkeypatch, _EDGE_ON_BAND)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_subpixel_band_between_boundary_pixels_is_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band crossing the boundary between two boundary pixel centers is not skipped."""
    obs = _obs(monkeypatch, _SUBPIXEL_TILTED_BAND)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_frame_inside_the_disc_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame wholly filled by ring plane inside the catalog is not skipped."""
    obs = _obs(monkeypatch, _INSIDE_THE_DISC)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_frame_beyond_the_rings_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame seeing only ring plane past the catalog's outermost radius is skipped."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is True


def test_frame_above_the_horizon_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame none of whose boundary rays meets the ring plane is skipped."""
    obs = _obs(monkeypatch, _ABOVE_THE_HORIZON)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is True


def test_planet_center_in_frame_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planet center projecting just inside the margin keeps the dense path."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS, planet_uv=(-5.4, 50.0))
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_planet_center_past_the_margin_is_outside(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planet center projecting past the margin does not count as in the frame."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS, planet_uv=(-5.6, 50.0))
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.center_in_frame is False


@pytest.mark.parametrize('planet_uv', [(105.4, 50.0), (50.0, 105.4)])
def test_planet_center_inside_the_far_margin_is_in_frame(
    monkeypatch: pytest.MonkeyPatch, planet_uv: tuple[float, float]
) -> None:
    """A planet center just inside the margin past the last column or row is in the frame."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS, planet_uv=planet_uv)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.center_in_frame is True


def test_unmasked_nan_boundary_samples_miss_the_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary samples that come back as unmasked NaN count as missing the plane."""
    scene = dataclasses.replace(_ABOVE_THE_HORIZON, nan_misses=True)
    obs = _obs(monkeypatch, scene)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.edge_meets_plane is False


def test_unprojectable_planet_center_keeps_the_dense_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planet center that cannot be projected counts as in the frame."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS, project_planet=False)
    view = ring_disc_view(cast(Any, obs), _TARGET, _PLANET, _OUTER_RADIUS_KM)
    assert view.disc_outside_frame is False


def test_model_logs_the_disc_outside_the_frame(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skip on radii states the catalog extent, the boundary radii and the center."""
    obs = _obs(monkeypatch, _BEYOND_THE_RINGS)
    model = NavModelRings('rings:SATURN', cast(Any, obs))
    assert model._ring_disc_outside_frame(_TARGET, _PLANET, _OUTER_RADIUS_KM) is True
    logged = capsys.readouterr().out
    assert 'ring plane out to the catalog max extent of 140000 km lies wholly outside' in logged


def test_model_logs_a_frame_off_the_ring_plane(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skip with no boundary ray on the ring plane says the frame misses the plane."""
    obs = _obs(monkeypatch, _ABOVE_THE_HORIZON)
    model = NavModelRings('rings:SATURN', cast(Any, obs))
    assert model._ring_disc_outside_frame(_TARGET, _PLANET, _OUTER_RADIUS_KM) is True
    logged = capsys.readouterr().out
    assert 'so the frame does not see the ring plane' in logged


# The trace tests below follow one side of the frame, five samples long, whose
# intercepts lie on the line y = 5 km, so a ray running along the line passes 5 km from
# the ring center while every sample is more than 11 km from it.
_NAN = math.nan
_ONE_SIDE = [np.arange(5)]


def _trace(*points: tuple[float, float]) -> np.ndarray:
    """Return ring-plane intercepts as an ``(N, 2)`` array."""
    return np.array(points, dtype=np.float64)


def test_trace_follows_the_horizon_past_the_last_hit() -> None:
    """Hits followed by misses run out along the line through the last two hits."""
    xy = _trace((-30.0, 5.0), (-20.0, 5.0), (-11.0, 5.0), (_NAN, _NAN), (_NAN, _NAN))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == pytest.approx(5.0)


def test_trace_follows_the_horizon_before_the_first_hit() -> None:
    """Misses followed by hits run out along the line through the first two hits."""
    xy = _trace((_NAN, _NAN), (_NAN, _NAN), (-11.0, 5.0), (-20.0, 5.0), (-30.0, 5.0))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == pytest.approx(5.0)


def test_trace_receding_from_the_center_is_nearest_at_its_first_hit() -> None:
    """A trace heading away from the ring center comes closest at its nearest sample."""
    xy = _trace((11.0, 5.0), (20.0, 5.0), (30.0, 5.0), (_NAN, _NAN), (_NAN, _NAN))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == pytest.approx(math.hypot(11.0, 5.0))


def test_trace_bridges_masked_samples_between_hits() -> None:
    """Masked samples between two hits on a side are bridged by the segment joining them."""
    xy = _trace((-20.0, 5.0), (_NAN, _NAN), (_NAN, _NAN), (20.0, 5.0), (30.0, 5.0))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == pytest.approx(5.0)


def test_trace_with_a_lone_hit_at_an_opening_corner_is_unbounded() -> None:
    """A lone hit at the corner that starts its side leaves the trace's direction unknown."""
    xy = _trace((-20.0, 5.0), (_NAN, _NAN), (_NAN, _NAN), (_NAN, _NAN), (_NAN, _NAN))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == 0.0


def test_trace_with_a_lone_hit_at_a_closing_corner_is_unbounded() -> None:
    """A lone hit at the corner that ends its side leaves the trace's direction unknown."""
    xy = _trace((_NAN, _NAN), (_NAN, _NAN), (_NAN, _NAN), (_NAN, _NAN), (-20.0, 5.0))
    assert boundary_closest_approach_km(xy, _ONE_SIDE) == 0.0
