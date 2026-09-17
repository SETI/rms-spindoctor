"""Direct unit tests for the ring-edge covariance geometry.

The orbit-uncertainty channel's absorbed-translation sensitivity is newly
derived math that demotes operator-verified library frames, so its limits are
pinned here directly rather than only through composite ``navigate()`` tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.nav_technique.ring_edge_geometry import (
    _absorbed_orbit_sensitivity,
    _aggregate_normal_orientation,
    _orbit_inflated_covariance,
)


def _arc_normals(coverage_deg: float, n: int = 400) -> np.ndarray:
    """Unit outward-radial normals spanning ``coverage_deg`` of a circle."""
    half = np.deg2rad(coverage_deg) / 2.0
    angles = np.linspace(-half, half, n)
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1)


# ---------------------------------------------------------------------------
# The four geometric limits
# ---------------------------------------------------------------------------


def test_absorbed_sensitivity_short_arc_absorbs_along_its_normal() -> None:
    """A short arc absorbs a radial displacement one-for-one along its axis."""
    normals = _arc_normals(10.0)
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) == pytest.approx(1.0, abs=0.01)
    # The arc is centered on +u, so the sensitivity points along u.
    assert abs(float(g[1])) == pytest.approx(1.0, abs=0.01)
    assert abs(float(g[0])) < 0.01


def test_absorbed_sensitivity_closed_annulus_absorbs_almost_nothing() -> None:
    """A closed ring is dilated, not translated, by a uniform radial error."""
    normals = _arc_normals(360.0)
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) < 0.02


def test_absorbed_sensitivity_opposite_sides_cancel() -> None:
    """Features on opposite radial sides cancel instead of adding."""
    arc = _arc_normals(30.0, 200)
    normals = np.vstack([arc, -arc])
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) < 1.0e-9


def _opposed_ansae(tilt_deg: float, n: int = 200) -> np.ndarray:
    """Two edge groups with nearly antiparallel outward normals.

    The geometry of two ansae of one ring in a wide field, or the near and far
    side of the ring plane: the normals differ from exactly antiparallel by
    ``tilt_deg``.
    """
    tilt = np.deg2rad(tilt_deg)
    near = np.tile([[np.cos(tilt), np.sin(tilt)]], (n, 1))
    far = np.tile([[-np.cos(tilt), np.sin(tilt)]], (n, 1))
    return np.vstack([near, far])


@pytest.mark.parametrize('tilt_deg', [10.0, 5.0, 1.0, 0.1])
def test_absorbed_sensitivity_does_not_diverge_on_opposed_ansae(tilt_deg: float) -> None:
    """Nearly-opposed normals must not amplify the reported uncertainty.

    An unguarded least-squares solve diverges as ``1 / sin(tilt)`` here --
    measured 5.8 at 10 degrees, 57 at 1 degree and 573 at 0.1 -- because
    ``b`` survives only along the direction the geometry least constrains.
    That amplification is the fit's own ill-conditioning, which the LM
    covariance already prices, so it must not be fed back in as a coherent
    displacement sensitivity.  The conditioning cutoff drops the direction and
    the caller reports the isotropic bound instead.
    """
    normals = _opposed_ansae(tilt_deg)
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) == pytest.approx(0.0, abs=1.0e-9)


def test_absorbed_sensitivity_bounded_on_partially_opposed_geometry() -> None:
    """Partially opposed normals stay inside the same-sense maximum.

    At 30 degrees of tilt the two groups still share a radial component and
    the solve stays inside the conditioning cutoff, but its unbounded value
    (2.0) exceeds anything the same-sense family can reach, so it is bounded
    at ``4 / pi``.
    """
    normals = _opposed_ansae(30.0)
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) == pytest.approx(4.0 / np.pi, rel=1.0e-6)


def test_absorbed_sensitivity_never_exceeds_the_same_sense_maximum() -> None:
    """No geometry reports a sensitivity above the analytic bound."""
    cases = [_arc_normals(deg) for deg in (5.0, 45.0, 120.0, 180.0, 270.0, 360.0)]
    cases += [_opposed_ansae(tilt) for tilt in (0.1, 1.0, 5.0, 10.0, 30.0, 60.0)]
    for normals in cases:
        g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
        assert float(np.linalg.norm(g)) <= 4.0 / np.pi + 1.0e-9


def test_absorbed_sensitivity_overshoots_on_a_half_ring() -> None:
    """A half ring absorbs MORE than the displacement (the 4/pi overshoot).

    To explain "every vertex moved outward by d" with one translation the fit
    overshoots the middle of the arc to reduce the error at its ends.  This is
    the behavior an earlier revision clamped away at 1.0.
    """
    normals = _arc_normals(180.0)
    g = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(g)) == pytest.approx(4.0 / np.pi, rel=0.02)
    assert float(np.linalg.norm(g)) > 1.0


# ---------------------------------------------------------------------------
# Exactness on the rank-1 path
# ---------------------------------------------------------------------------


def test_absorbed_sensitivity_is_exactly_the_normal_for_parallel_normals() -> None:
    """A straight edge gives ``g = n`` exactly, matching the rank-1 axis.

    The rank-1 projection rebuilds the covariance around
    ``_aggregate_normal_orientation``; the sensitivity must agree with that
    axis exactly or the projected covariance would stop being singular along
    the tangent once the orbit term is added.
    """
    n_hat = np.array([0.6, 0.8])
    normals = np.tile(n_hat, (50, 1))
    g = _absorbed_orbit_sensitivity(normals, np.ones(50))
    axis = _aggregate_normal_orientation(normals)
    assert float(np.linalg.norm(g)) == pytest.approx(1.0, abs=1.0e-9)
    assert np.allclose(np.outer(g, g), np.outer(axis, axis))


def test_absorbed_sensitivity_weights_select_the_carrying_arc() -> None:
    """Zero-weight vertices do not steer the sensitivity."""
    normals = np.vstack([_arc_normals(20.0, 100), -_arc_normals(20.0, 100)])
    weights = np.concatenate([np.ones(100), np.zeros(100)])
    g = _absorbed_orbit_sensitivity(normals, weights)
    # Only the +u arc carries weight, so the cancellation must not occur.
    assert float(np.linalg.norm(g)) == pytest.approx(1.0, abs=0.01)
    assert float(g[1]) > 0.99


def test_absorbed_sensitivity_all_zero_weights_falls_back_to_uniform() -> None:
    """A degenerate all-zero weight vector still yields a usable direction."""
    normals = _arc_normals(10.0)
    g = _absorbed_orbit_sensitivity(normals, np.zeros(len(normals)))
    assert float(np.linalg.norm(g)) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# The inflation blend
# ---------------------------------------------------------------------------


def test_inflation_is_purely_directional_at_unit_sensitivity() -> None:
    cov = np.zeros((2, 2))
    g = np.array([0.0, 1.0])
    out = _orbit_inflated_covariance(cov, g, 2.0)
    assert out[1, 1] == pytest.approx(4.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_inflation_is_isotropic_at_zero_sensitivity() -> None:
    cov = np.zeros((2, 2))
    out = _orbit_inflated_covariance(cov, np.zeros(2), 2.0)
    assert out[0, 0] == pytest.approx(4.0)
    assert out[1, 1] == pytest.approx(4.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_inflation_major_eigenvalue_is_sigma_squared_below_unit_sensitivity() -> None:
    """For ``|g| <= 1`` the major axis is exactly sigma**2 whatever ``g`` is.

    This is why the derived direction alone cannot move a tier outcome in that
    regime -- the tier gate reads ``max(sigma_dv, sigma_du)``.  The geometry
    sets the MINOR axis, i.e. the isotropic floor.
    """
    for magnitude in (0.0, 0.25, 0.5, 0.9, 1.0):
        g = np.array([0.0, magnitude])
        out = _orbit_inflated_covariance(np.zeros((2, 2)), g, 3.0)
        eigvals = np.linalg.eigvalsh(out)
        assert float(eigvals.max()) == pytest.approx(9.0)
        assert float(eigvals.min()) == pytest.approx(9.0 * (1.0 - magnitude**2))


def test_inflation_above_unit_sensitivity_raises_the_major_axis() -> None:
    """Above 1 the magnitude does move the major axis, with no isotropic term."""
    g = np.array([0.0, 4.0 / np.pi])
    out = _orbit_inflated_covariance(np.zeros((2, 2)), g, 2.0)
    eigvals = np.linalg.eigvalsh(out)
    assert float(eigvals.max()) == pytest.approx(4.0 * (4.0 / np.pi) ** 2)
    assert float(eigvals.min()) == pytest.approx(0.0, abs=1.0e-12)


def test_inflation_stays_positive_semidefinite() -> None:
    for magnitude in (0.0, 0.3, 1.0, 1.27):
        g = np.array([0.3, 0.4]) / 0.5 * magnitude
        out = _orbit_inflated_covariance(np.eye(2) * 0.01, g, 1.5)
        assert float(np.linalg.eigvalsh(out).min()) >= -1.0e-12


def test_inflation_leaves_rotation_block_untouched() -> None:
    cov = np.diag([0.1, 0.2, 0.3])
    out = _orbit_inflated_covariance(cov, np.array([0.0, 1.0]), 2.0)
    assert out[2, 2] == pytest.approx(0.3)
    assert out[0, 2] == pytest.approx(0.0)
