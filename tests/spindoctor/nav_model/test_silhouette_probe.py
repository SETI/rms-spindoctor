"""Tests for the sub-pixel silhouette-probe vertex refinement helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spindoctor.nav_model.silhouette_probe import (
    PROBE_OFFSETS_PX,
    boundary_crossing_offsets,
    probe_positions_uv,
    refined_vertex_positions,
)


def test_probe_offsets_bracket_the_ridge_pixel() -> None:
    """The probe ladder spans inward and outward of the ridge pixel center."""
    assert PROBE_OFFSETS_PX[0] == pytest.approx(-1.0)
    assert PROBE_OFFSETS_PX[-1] == pytest.approx(1.5)


def test_probe_offsets_include_the_vertex_itself() -> None:
    """One probe sits exactly on the vertex so insideness at t=0 is measured."""
    assert 0.0 in PROBE_OFFSETS_PX.tolist()


def test_crossing_center_inside_boundary_outward() -> None:
    """A vertex inside the region crosses at the midpoint of the bracketing probes."""
    ts = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    inside = np.array([[True, True, True, True, False, False]])
    offsets = boundary_crossing_offsets(inside, ts)
    assert offsets[0] == pytest.approx(0.375)


def test_crossing_center_outside_boundary_inward() -> None:
    """A vertex outside the region finds the boundary inward of its center."""
    ts = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    inside = np.array([[True, False, False, False, False]])
    offsets = boundary_crossing_offsets(inside, ts)
    assert offsets[0] == pytest.approx(-0.375)


def test_crossing_all_inside_is_nan() -> None:
    """No outward exit within the ladder leaves the vertex undetermined."""
    ts = np.array([-0.5, 0.0, 0.5])
    inside = np.array([[True, True, True]])
    assert math.isnan(boundary_crossing_offsets(inside, ts)[0])


def test_crossing_all_outside_is_nan() -> None:
    """No region found anywhere along the ladder leaves the vertex undetermined."""
    ts = np.array([-0.5, 0.0, 0.5])
    inside = np.array([[False, False, False]])
    assert math.isnan(boundary_crossing_offsets(inside, ts)[0])


def test_crossing_uses_first_exit_not_far_holes() -> None:
    """The nearest outward exit wins even when the region reappears farther out."""
    ts = np.array([-0.25, 0.0, 0.25, 0.5, 0.75])
    inside = np.array([[True, True, False, True, False]])
    offsets = boundary_crossing_offsets(inside, ts)
    assert offsets[0] == pytest.approx(0.125)


def test_crossing_rows_are_independent() -> None:
    """Each vertex row resolves its own crossing."""
    ts = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    inside = np.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    offsets = boundary_crossing_offsets(inside, ts)
    assert offsets[0] == pytest.approx(0.125)
    assert math.isnan(offsets[1])


def test_probe_positions_walk_along_the_normal() -> None:
    """Probe (u, v) positions step from the vertex center along its outward normal."""
    vertices_vu = np.array([[10.0, 20.0]])
    normals_vu = np.array([[0.0, 1.0]])
    ts = np.array([-0.5, 0.0, 0.5])
    uv = probe_positions_uv(vertices_vu, normals_vu, ts, margin_vu=(2, 3))
    # extfov (v=10, u=20) with margins (2, 3) is FOV pixel (v=8, u=17); its
    # center is (u, v) = (17.5, 8.5), and the normal steps move u only.
    assert uv[0, 0, 0] == pytest.approx(17.0)
    assert uv[0, 1, 0] == pytest.approx(17.5)
    assert uv[0, 2, 0] == pytest.approx(18.0)
    assert uv[0, 1, 1] == pytest.approx(8.5)


def test_refined_positions_apply_crossing_along_normal() -> None:
    """A finite crossing moves the vertex by that distance along its normal."""
    vertices_vu = np.array([[5.0, 6.0], [7.0, 8.0]])
    normals_vu = np.array([[1.0, 0.0], [0.0, -1.0]])
    offsets = np.array([0.25, float('nan')])
    refined = refined_vertex_positions(vertices_vu, normals_vu, offsets)
    assert refined[0, 0] == pytest.approx(5.25)
    assert refined[0, 1] == pytest.approx(6.0)
    # An undetermined crossing leaves the vertex where it was.
    assert refined[1, 0] == pytest.approx(7.0)
    assert refined[1, 1] == pytest.approx(8.0)


def test_crossing_rejects_nonfinite_ladder() -> None:
    """A ladder with a non-finite offset is a contract violation, not a guess."""
    inside = np.array([[True, True, False]])
    ts = np.array([-0.5, 0.0, float('nan')])
    with pytest.raises(ValueError, match='finite and strictly increasing'):
        boundary_crossing_offsets(inside, ts)


def test_crossing_rejects_non_increasing_ladder() -> None:
    """A non-monotone ladder would misorder the bracketing; reject it."""
    inside = np.array([[True, True, False]])
    ts = np.array([-0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match='finite and strictly increasing'):
        boundary_crossing_offsets(inside, ts)


def test_crossing_rejects_ladder_without_exact_zero() -> None:
    """The vertex's own probe must exist; a nearest-to-zero stand-in is wrong."""
    inside = np.array([[True, True, False]])
    ts = np.array([-0.5, 0.1, 0.6])
    with pytest.raises(ValueError, match=r'exactly one 0\.0 entry'):
        boundary_crossing_offsets(inside, ts)


def test_refined_positions_mixed_finite_normal_leaves_vertex_whole() -> None:
    """A normal with one finite component still keeps the whole vertex in place.

    Moving only the finite coordinate would bend the vertex off its normal;
    the contract is all-or-nothing per vertex.
    """
    vertices_vu = np.array([[5.0, 6.0]])
    normals_vu = np.array([[float('nan'), 1.0]])
    refined = refined_vertex_positions(vertices_vu, normals_vu, np.array([0.25]))
    assert refined[0, 0] == pytest.approx(5.0)
    assert refined[0, 1] == pytest.approx(6.0)


def test_refined_positions_nonfinite_normal_leaves_vertex_unchanged() -> None:
    """A non-finite normal must not corrupt the vertex it belongs to.

    A finite crossing offset multiplied by a NaN normal is NaN; the
    refinement's contract is to leave any vertex it cannot place exactly
    where it was, so the NaN move is dropped rather than propagated.
    """
    vertices_vu = np.array([[5.0, 6.0], [7.0, 8.0]])
    normals_vu = np.array([[float('nan'), float('nan')], [0.0, 1.0]])
    offsets = np.array([0.25, 0.25])
    refined = refined_vertex_positions(vertices_vu, normals_vu, offsets)
    assert refined[0, 0] == pytest.approx(5.0)
    assert refined[0, 1] == pytest.approx(6.0)
    # The well-formed row still refines normally.
    assert refined[1, 0] == pytest.approx(7.0)
    assert refined[1, 1] == pytest.approx(8.25)


def test_refined_positions_do_not_mutate_input() -> None:
    """The input vertex array is returned as a refined copy, never mutated."""
    vertices_vu = np.array([[5.0, 6.0]])
    normals_vu = np.array([[1.0, 0.0]])
    refined = refined_vertex_positions(vertices_vu, normals_vu, np.array([0.5]))
    assert vertices_vu[0, 0] == pytest.approx(5.0)
    assert refined[0, 0] == pytest.approx(5.5)
