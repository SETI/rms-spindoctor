"""Tests for the visible-limb altitude term in ``BodyLimbNav``'s covariance."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from tests.spindoctor.nav_technique.conftest import (
    ArcPolylineFactory,
    DiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import LimbPolyline
from spindoctor.nav_technique.nav_technique_body_limb import (
    BodyLimbNav,
    limb_altitude_covariance,
)


def _arc_normals(start_rad: float, stop_rad: float, n: int) -> np.ndarray:
    """Return ``n`` unit normals evenly spread over ``[start, stop]``.

    Parameters:
        start_rad: First position angle.
        stop_rad: Last position angle.
        n: Number of normals.
    """
    ang = np.linspace(start_rad, stop_rad, n)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1)


def test_full_circle_cancels_the_altitude() -> None:
    """A common radial displacement of a whole circle does not move the translation."""
    normals = _arc_normals(0.0, 2.0 * math.pi * (1.0 - 1.0 / 400.0), 400)
    cov = limb_altitude_covariance(normals, np.ones(400), [np.full(400, 1.0)])
    assert float(np.abs(cov).max()) < 1e-12


def test_half_circle_turns_the_altitude_into_translation_along_its_normal() -> None:
    """On a half circle a 1 px radial displacement moves the translation 4/pi px.

    With uniform weights the information matrix of a half circle is ``N/2`` times
    the identity and the summed normal is ``2N/pi`` along the arc's mean normal, so
    the translation moves ``(2N/pi) / (N/2) = 4/pi`` pixels along that normal.
    """
    normals = _arc_normals(-math.pi / 2.0, math.pi / 2.0, 2001)
    cov = limb_altitude_covariance(normals, np.ones(2001), [np.full(2001, 1.0)])
    assert cov[0, 0] == pytest.approx((4.0 / math.pi) ** 2, rel=1e-3)
    assert cov[1, 1] == pytest.approx(0.0, abs=1e-9)


def test_independent_bodies_add() -> None:
    """Two bodies' altitude errors are independent, so their terms add."""
    normals = _arc_normals(-math.pi / 2.0, math.pi / 2.0, 200)
    weights = np.ones(400)
    both = limb_altitude_covariance(
        np.concatenate([normals, normals]), weights, [np.full(200, 1.0), np.full(200, 1.0)]
    )
    one = limb_altitude_covariance(
        np.concatenate([normals, normals]), weights, [np.full(200, 1.0), np.zeros(200)]
    )
    assert both == pytest.approx(2.0 * one)


def test_no_uncertainty_adds_nothing() -> None:
    """Zero displacements give a zero covariance."""
    normals = _arc_normals(0.0, 1.0, 50)
    cov = limb_altitude_covariance(normals, np.ones(50), [np.zeros(50)])
    assert float(np.abs(cov).max()) == 0.0


def test_unconstrained_direction_carries_no_term() -> None:
    """With every normal parallel the term lies along the normal and not across it."""
    normals = np.tile([[1.0, 0.0]], (50, 1))
    cov = limb_altitude_covariance(normals, np.ones(50), [np.full(50, 1.0)])
    assert cov == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_zero_weights_add_nothing() -> None:
    """A fit that kept no vertex adds no altitude term."""
    normals = _arc_normals(0.0, 3.0, 50)
    cov = limb_altitude_covariance(normals, np.zeros(50), [np.full(50, 1.0)])
    assert float(np.abs(cov).max()) == 0.0


def test_navigate_adds_the_altitude_term_along_the_arc_normal(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A partial arc carrying an altitude uncertainty reports it along its normal.

    The arc faces +u, so the term lands on the u variance: 0.1 px/km times 10 km is
    1 px of radial displacement, (4/pi)^2 px^2 of u variance for a half circle.
    """
    image = disc_image((200, 200), (100.0, 100.0), 30.0)
    vertices, outward = arc_polyline((100.0, 100.0), 30.0, 120, -np.pi / 2, np.pi / 2)
    plain = make_limb_feature('moonC', vertices=vertices, outward_normals=outward)
    assert isinstance(plain.geometry, LimbPolyline)
    with_altitude = dataclasses.replace(
        plain,
        geometry=dataclasses.replace(
            plain.geometry,
            altitude_px_per_km=np.full(120, 0.1),
            altitude_sigma_km=10.0,
        ),
    )
    context = make_nav_context(image)
    base = BodyLimbNav().navigate([plain], context).covariance_px2
    added = BodyLimbNav().navigate([with_altitude], context).covariance_px2
    assert added[1, 1] - base[1, 1] == pytest.approx((4.0 / math.pi) ** 2, rel=0.1)
    assert added[0, 0] - base[0, 0] == pytest.approx(0.0, abs=0.05)


def test_weights_enter_the_altitude_term() -> None:
    """With non-uniform weights the term matches the hand-computed ``g g^T``.

    Two normals, (1, 0) with weight 3 and (0, 1) with weight 1, and displacements
    2 and 1 px: ``A = diag(3, 1)``, ``sum w n d = (6, 1)``, so ``g = (2, 1)``.
    """
    normals = np.array([[1.0, 0.0], [0.0, 1.0]])
    cov = limb_altitude_covariance(normals, np.array([3.0, 1.0]), [np.array([2.0, 1.0])])
    assert cov == pytest.approx(np.array([[4.0, 2.0], [2.0, 1.0]]))


def test_each_body_uses_its_own_vertices() -> None:
    """Bodies keep their place in the concatenation: swapping them changes the term.

    The first body faces +v and carries the altitude uncertainty; the second faces +u
    and carries none, so all of the term lands on v.
    """
    first = _arc_normals(-math.pi / 2.0, math.pi / 2.0, 101)
    second = _arc_normals(0.0, math.pi, 101)
    normals = np.concatenate([first, second])
    cov = limb_altitude_covariance(normals, np.ones(202), [np.full(101, 1.0), np.zeros(101)])
    assert abs(cov[0, 0]) > 10.0 * abs(cov[1, 1])


def test_rotation_fit_gets_a_three_by_three_term() -> None:
    """With a rotation lever the term covers (dv, du, theta)."""
    normals = _arc_normals(-math.pi / 2.0, math.pi / 2.0, 200)
    lever = np.linspace(-30.0, 30.0, 200)
    cov = limb_altitude_covariance(
        normals, np.ones(200), [np.full(200, 1.0)], rotation_lever_px=lever
    )
    assert cov.shape == (3, 3)


def test_rotation_term_matches_a_least_squares_solve() -> None:
    """The 3 x 3 term is ``g g^T`` with ``g`` the weighted least-squares pose shift.

    ``g`` solves ``sqrt(w) J g = sqrt(w) d`` for the rows ``(n_v, n_u, lever)``,
    computed here independently of the normal equations.
    """
    normals = _arc_normals(-math.pi / 2.0, math.pi / 3.0, 150)
    lever = np.linspace(-20.0, 35.0, 150) ** 2 / 30.0
    weights = np.linspace(0.5, 2.0, 150)
    disp = np.linspace(0.8, 1.2, 150)
    jac = np.column_stack([normals, lever])
    root = np.sqrt(weights)
    g = np.linalg.lstsq(jac * root[:, None], disp * root, rcond=None)[0]
    cov = limb_altitude_covariance(normals, weights, [disp], rotation_lever_px=lever)
    assert cov == pytest.approx(np.outer(g, g), rel=1e-6, abs=1e-9)


def _with_altitude(feature: NavFeature, per_km: np.ndarray, sigma_km: float) -> NavFeature:
    """Return ``feature`` carrying an altitude sensitivity and uncertainty.

    Parameters:
        feature: A LIMB_ARC feature.
        per_km: Per-vertex pixels per km of altitude.
        sigma_km: The altitude uncertainty.
    """
    assert isinstance(feature.geometry, LimbPolyline)
    return dataclasses.replace(
        feature,
        geometry=dataclasses.replace(
            feature.geometry, altitude_px_per_km=per_km, altitude_sigma_km=sigma_km
        ),
    )


def _ellipse_arc(
    center_vu: tuple[float, float], axes_vu: tuple[float, float], start: float, stop: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and outward unit normals of an elliptical arc.

    Parameters:
        center_vu: The ellipse's center.
        axes_vu: Its semi-axes along v and u.
        start: First parameter angle (radians, 0 along +u).
        stop: Last parameter angle.
        n: Number of vertices.
    """
    t = np.linspace(start, stop, n)
    vertices = np.stack(
        [center_vu[0] + axes_vu[0] * np.sin(t), center_vu[1] + axes_vu[1] * np.cos(t)], axis=-1
    )
    normals = np.stack([np.sin(t) / axes_vu[0], np.cos(t) / axes_vu[1]], axis=-1)
    return vertices, normals / np.linalg.norm(normals, axis=1)[:, None]


def test_rotation_fit_term_matches_a_planted_altitude_shift(
    make_limb_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """With rotation fitted, the 3 x 3 term is the outer product of the pose shift.

    On an elliptical arc, where rotation and translation are separable, moving every
    vertex 0.3 px out along its normal (a one-sigma error of 0.03 px/km times 10 km)
    and refitting moves the pose by about ``g``.  The arc faces +u, so the u and
    rotation entries of ``g g^T`` dominate; the reported term matches them to the
    accuracy of the normal-projection linearization, with the cross-term sign the
    lever sets.
    """
    vv, uu = np.mgrid[0:200, 0:200]
    inside = ((vv - 100.0) / 25.0) ** 2 + ((uu - 100.0) / 45.0) ** 2 <= 1.0
    image = gaussian_filter(np.where(inside, 100.0, 0.0), 0.7)
    vertices, outward = _ellipse_arc((100.0, 100.0), (25.0, 45.0), -np.pi / 2, np.pi / 4, 150)
    per_km = np.full(150, 0.03)
    plain = make_limb_feature('moonG', vertices=vertices, outward_normals=outward)
    moved = make_limb_feature('moonG', vertices=vertices + 0.3 * outward, outward_normals=outward)
    context = make_nav_context(image, fit_camera_rotation=True)
    r0 = BodyLimbNav().navigate([plain], context)
    r1 = BodyLimbNav().navigate([moved], context)
    rotation = float(r1.rotation_rad or 0.0) - float(r0.rotation_rad or 0.0)
    shift = np.array([*np.subtract(r1.offset_px, r0.offset_px), rotation])
    priced = BodyLimbNav().navigate([_with_altitude(plain, per_km, 10.0)], context)
    added = np.asarray(priced.covariance_px2) - np.asarray(r0.covariance_px2)
    expected = np.outer(shift, shift)
    assert np.sign(added[1, 2]) == np.sign(expected[1, 2])
    assert np.sqrt(added[1, 1]) == pytest.approx(abs(shift[1]), rel=0.5)
    assert np.sqrt(added[2, 2]) == pytest.approx(abs(shift[2]), rel=0.5)
