"""Pure-math tests for ``spindoctor.ui.mosaic_viewer.graticule``.

These tests exercise the projection geometry of :func:`graticule_polylines`,
:func:`graticule_label_anchors`, and :func:`_split_polyline` without
requiring PyQt6 or any display server.
"""

import math

import numpy as np
import pytest

from spindoctor.ui.mosaic_viewer.graticule import (
    _split_polyline,
    graticule_label_anchors,
    graticule_polylines,
)
from spindoctor.ui.mosaic_viewer.projections import ProjectionKind, ProjectionParams

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CX = 300.0
CY = 200.0
SCALE = 150.0

POLAR_N_PARAMS = ProjectionParams(
    kind=ProjectionKind.POLAR_N,
    cx=CX,
    cy=CY,
    scale=SCALE,
)

MOLLWEIDE_PARAMS = ProjectionParams(
    kind=ProjectionKind.MOLLWEIDE,
    cx=CX,
    cy=CY,
    scale=100.0,
)

SPHERE_PARAMS = ProjectionParams(
    kind=ProjectionKind.SPHERE_3D,
    cx=CX,
    cy=CY,
    scale=SCALE,
    yaw_deg=0.0,
    pitch_deg=0.0,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _find_parallel_at_lat(
    segs: list[list[tuple[float, float]]],
    lat_deg: float,
    params: ProjectionParams,
    tol: float = 0.5,
) -> list[tuple[float, float]]:
    """Return the first segment whose points lie on the expected parallel circle.

    For POLAR_N the expected radius in viewport pixels is
    ``scale * tan(0.5 * (pi/2 - lat_rad))``.  The segment is identified by
    checking that every point in it satisfies the radial distance criterion.

    Parameters:
        segs: All polyline segments returned by ``graticule_polylines``.
        lat_deg: Target latitude in degrees.
        params: Projection parameters (must be POLAR_N).
        tol: Acceptable pixel deviation from expected radius.

    Returns:
        The matching segment (at least 2 points).
    """
    lat_r = math.radians(lat_deg)
    half_angle = 0.5 * (0.5 * math.pi - lat_r)
    expected_r = params.scale * math.tan(half_angle)
    for seg in segs:
        if len(seg) < 2:
            continue
        all_match = all(
            abs(math.hypot(x - params.cx, y - params.cy) - expected_r) < tol for x, y in seg
        )
        if all_match:
            return seg
    return []


# ===========================================================================
# Test 1 -- Polar-N parallel at lat=60 lies on correct circle
# ===========================================================================


class TestPolarNParallelCircle:
    """Verify that the lat=60 parallel projects onto the correct circle."""

    def test_parallel_segment_exists(self) -> None:
        """graticule_polylines produces at least one segment for lat=60."""
        parallel_segs, _ = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=0,
            show_parallels=True,
            show_meridians=False,
        )
        seg = _find_parallel_at_lat(parallel_segs, 60.0, POLAR_N_PARAMS)
        assert len(seg) >= 2

    def test_all_points_on_expected_radius(self) -> None:
        """Every point of the lat=60 parallel is within 0.5 px of the expected radius."""
        parallel_segs, _ = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=0,
            show_parallels=True,
            show_meridians=False,
        )
        seg = _find_parallel_at_lat(parallel_segs, 60.0, POLAR_N_PARAMS)
        lat_r = math.radians(60.0)
        half_angle = 0.5 * (0.5 * math.pi - lat_r)
        expected_r = SCALE * math.tan(half_angle)
        for x, y in seg:
            r = math.hypot(x - CX, y - CY)
            assert abs(r - expected_r) < 0.5

    def test_expected_radius_value(self) -> None:
        """The expected radius equals scale * tan(15 deg) numerically."""
        lat_r = math.radians(60.0)
        half_angle = 0.5 * (0.5 * math.pi - lat_r)
        expected_r = SCALE * math.tan(half_angle)
        # Reference: SCALE * tan(15°) (same geometry as the parallel at lat=60°).
        ref_r = 40.192378864668406
        assert expected_r == pytest.approx(ref_r, rel=1e-12)

    def test_no_meridian_segments_returned(self) -> None:
        """show_meridians=False yields an empty meridian list."""
        _, meridian_segs = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=0,
            show_parallels=True,
            show_meridians=False,
        )
        assert meridian_segs == []


# ===========================================================================
# Test 2 -- Polar-N meridian at lon=0 is a vertical line at vx=cx
# ===========================================================================


class TestPolarNMeridianLon0:
    """Verify that the lon=0 meridian is a vertical radial at vx=cx."""

    def test_meridian_segment_exists(self) -> None:
        """graticule_polylines produces at least one segment for lon=0."""
        _, meridian_segs = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        assert len(meridian_segs) >= 1

    def test_lon0_segment_has_vx_near_cx(self) -> None:
        """Every point of the lon=0 meridian has vx within 0.5 px of cx.

        For POLAR_N: xn = r * sin(0) = 0, so vx = cx for all latitudes.
        """
        _, meridian_segs = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        # The lon=0 meridian is the first one generated (lon starts at 0)
        lon0_seg = meridian_segs[0]
        for x, _y in lon0_seg:
            assert abs(x - CX) < 0.5

    def test_lon0_segment_spans_upward_from_pole(self) -> None:
        """The lon=0 meridian extends above cy (i.e. min vy < cy).

        In POLAR_N the pole is at cy; lon=0 runs toward negative-y (up on screen).
        For cos(0)=1, yn=-r so vy = cy - r*scale < cy for all non-pole lats.
        """
        _, meridian_segs = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        lon0_seg = meridian_segs[0]
        min_vy = min(y for _x, y in lon0_seg)
        assert min_vy < CY

    def test_no_parallel_segments_returned(self) -> None:
        """show_parallels=False yields an empty parallel list."""
        parallel_segs, _ = graticule_polylines(
            POLAR_N_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        assert parallel_segs == []


# ===========================================================================
# Test 3 -- Mollweide equator is horizontal (vy = cy)
# ===========================================================================


class TestMollweideEquatorHorizontal:
    """Verify that the Mollweide equator (lat=0) maps to vy=cy everywhere."""

    def _equator_segments(self) -> list[list[tuple[float, float]]]:
        """Return segments for the equator parallel only."""
        parallel_segs, _ = graticule_polylines(
            MOLLWEIDE_PARAMS,
            lat_step_deg=30,
            lon_step_deg=0,
            show_parallels=True,
            show_meridians=False,
        )
        cy = MOLLWEIDE_PARAMS.cy
        tol = 0.5
        return [seg for seg in parallel_segs if all(abs(y - cy) < tol for _x, y in seg)]

    def test_equator_segment_exists(self) -> None:
        """At least one segment lies entirely on vy=cy."""
        equator_segs = self._equator_segments()
        assert len(equator_segs) >= 1

    def test_equator_vy_equals_cy(self) -> None:
        """Every point of the equator has vy within 0.5 px of cy.

        Mollweide: lat=0 => theta=0 => yn = sqrt(2)*sin(0) = 0 => vy = cy.
        """
        equator_segs = self._equator_segments()
        cy = MOLLWEIDE_PARAMS.cy
        for seg in equator_segs:
            for _x, y in seg:
                assert abs(y - cy) < 0.5

    def test_equator_spans_horizontal_range(self) -> None:
        """The equator extends over a meaningful horizontal range (> 100 px)."""
        equator_segs = self._equator_segments()
        all_x = [x for seg in equator_segs for x, _y in seg]
        assert (max(all_x) - min(all_x)) > 100.0


# ===========================================================================
# Test 4 -- Sphere3D meridian at lon=0 has vx=cx and is partial (front only)
# ===========================================================================


class TestSphere3DMeridianLon0:
    """Verify Sphere3D projection of the lon=0 meridian with yaw=pitch=0."""

    def test_meridian_segment_exists(self) -> None:
        """At least one segment is produced for the lon=0 meridian."""
        _, meridian_segs = graticule_polylines(
            SPHERE_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        assert len(meridian_segs) >= 1

    def test_lon0_vx_near_cx(self) -> None:
        """Every point of the lon=0 meridian has vx within 0.5 px of cx.

        With yaw=pitch=0, world point at lon=0 has camera-frame xn=q[1]=0,
        so vx = 0*scale + cx = cx.
        """
        _, meridian_segs = graticule_polylines(
            SPHERE_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        lon0_seg = meridian_segs[0]
        for x, _y in lon0_seg:
            assert abs(x - CX) < 0.5

    def test_lon0_vy_stays_within_sphere_disk(self) -> None:
        """All lon=0 meridian points lie within the sphere disk (vy in [cy-scale, cy+scale]).

        The Sphere3D disk has radius ``scale`` in viewport pixels; no visible
        point can have ``|vy - cy| > scale``.
        """
        _, meridian_segs = graticule_polylines(
            SPHERE_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        lon0_seg = meridian_segs[0]
        for _x, y in lon0_seg:
            assert abs(y - CY) <= SCALE + 0.5

    def test_lon0_segment_centered_on_cy(self) -> None:
        """The lon=0 meridian segment is vertically symmetric about cy.

        With yaw=pitch=0 the equator lat=0 maps to yn=0, so the midpoint of
        the vy range should be close to cy.
        """
        _, meridian_segs = graticule_polylines(
            SPHERE_PARAMS,
            lat_step_deg=0,
            lon_step_deg=90,
            show_parallels=False,
            show_meridians=True,
        )
        lon0_seg = meridian_segs[0]
        vy_vals = [y for _x, y in lon0_seg]
        midpoint = 0.5 * (max(vy_vals) + min(vy_vals))
        assert abs(midpoint - CY) < 1.0


# ===========================================================================
# Test 5 -- Label anchors are non-empty for visible graticule
# ===========================================================================


class TestLabelAnchors:
    """Verify that label anchors are populated when graticule lines exist."""

    def test_parallel_anchors_non_empty_polar_n(self) -> None:
        """POLAR_N with lat_step=30 produces at least one parallel label anchor."""
        par_anchors, _ = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=30,
        )
        assert len(par_anchors) > 0

    def test_meridian_anchors_non_empty_polar_n(self) -> None:
        """POLAR_N with lon_step=30 produces at least one meridian label anchor."""
        _, mer_anchors = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=30,
        )
        assert len(mer_anchors) > 0

    def test_anchor_label_format_parallel(self) -> None:
        """Parallel label strings end with a degree sign and are numeric."""
        par_anchors, _ = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=30,
        )
        for _vx, _vy, label in par_anchors:
            assert label.endswith('°')
            assert label[:-1].lstrip('-').isdigit()

    def test_anchor_label_format_meridian(self) -> None:
        """Meridian label strings end with a degree sign and are numeric."""
        _, mer_anchors = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=30,
        )
        for _vx, _vy, label in mer_anchors:
            assert label.endswith('°')
            assert label[:-1].lstrip('-').isdigit()

    def test_anchors_empty_when_step_is_zero(self) -> None:
        """Passing lat_step_deg=0 and lon_step_deg=0 yields empty anchor lists."""
        par_anchors, mer_anchors = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=0,
            lon_step_deg=0,
        )
        assert par_anchors == []
        assert mer_anchors == []

    def test_anchor_coordinates_are_finite(self) -> None:
        """All anchor (vx, vy) values are finite floats."""
        par_anchors, mer_anchors = graticule_label_anchors(
            POLAR_N_PARAMS,
            lat_step_deg=30,
            lon_step_deg=30,
        )
        for vx, vy, _ in par_anchors + mer_anchors:
            assert math.isfinite(vx)
            assert math.isfinite(vy)


# ===========================================================================
# Test 6 -- _split_polyline splits on invisible (vis=False) points
# ===========================================================================


class TestSplitPolylineInvisible:
    """Verify that invisible samples break the polyline into separate segments."""

    def test_invisible_in_middle_produces_two_segments(self) -> None:
        """A False in the middle of vis causes two segments to be emitted."""
        vx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        vy = np.zeros(5)
        vis = np.array([True, True, False, True, True])
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 2

    def test_first_segment_correct_points(self) -> None:
        """The first segment contains only the points before the invisible sample."""
        vx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        vy = np.zeros(5)
        vis = np.array([True, True, False, True, True])
        segs = _split_polyline(vx, vy, vis)
        assert segs[0] == [(0.0, 0.0), (1.0, 0.0)]

    def test_second_segment_correct_points(self) -> None:
        """The second segment contains only the points after the invisible sample."""
        vx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        vy = np.zeros(5)
        vis = np.array([True, True, False, True, True])
        segs = _split_polyline(vx, vy, vis)
        assert segs[1] == [(3.0, 0.0), (4.0, 0.0)]

    def test_all_invisible_yields_empty(self) -> None:
        """All-invisible input produces no segments."""
        vx = np.array([0.0, 1.0, 2.0])
        vy = np.zeros(3)
        vis = np.array([False, False, False])
        segs = _split_polyline(vx, vy, vis)
        assert segs == []

    def test_single_invisible_at_start_drops_it(self) -> None:
        """An invisible first point does not appear in any segment."""
        vx = np.array([0.0, 1.0, 2.0])
        vy = np.zeros(3)
        vis = np.array([False, True, True])
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 1
        assert segs[0] == [(1.0, 0.0), (2.0, 0.0)]

    def test_single_point_runs_not_emitted(self) -> None:
        """A run of exactly one visible point between invisible ones is discarded.

        Segments must have >= 2 points to be useful for drawing.
        """
        vx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        vy = np.zeros(5)
        vis = np.array([True, False, True, False, True])
        segs = _split_polyline(vx, vy, vis)
        # Runs of length 1 are dropped; no segment should survive
        assert segs == []


# ===========================================================================
# Test 7 -- _split_polyline splits on large position gaps (discontinuous seam)
# ===========================================================================


class TestSplitPolylineLargeGap:
    """Verify that a >100 px jump triggers a segment break."""

    def test_large_gap_produces_two_segments(self) -> None:
        """A jump larger than 100 px between two visible points creates two segments."""
        # Points 0-2 form one group; point 3 jumps >100 px; points 3-4 form another.
        vx = np.array([0.0, 1.0, 2.0, 200.0, 201.0])
        vy = np.zeros(5)
        vis = np.ones(5, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 2

    def test_first_group_correct_before_gap(self) -> None:
        """The first segment ends just before the jump."""
        vx = np.array([0.0, 1.0, 2.0, 200.0, 201.0])
        vy = np.zeros(5)
        vis = np.ones(5, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert segs[0] == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    def test_second_group_correct_after_gap(self) -> None:
        """The second segment starts with the point after the jump."""
        vx = np.array([0.0, 1.0, 2.0, 200.0, 201.0])
        vy = np.zeros(5)
        vis = np.ones(5, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert segs[1] == [(200.0, 0.0), (201.0, 0.0)]

    def test_gap_exactly_100px_is_not_split(self) -> None:
        """A gap of exactly 100 px is not split (_MAX_STEP_SQ uses strict >)."""
        vx = np.array([0.0, 100.0, 101.0])
        vy = np.zeros(3)
        vis = np.ones(3, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        # 100^2 == _MAX_STEP_SQ, but the check is strictly >, so no break
        assert len(segs) == 1

    def test_gap_of_101px_is_split(self) -> None:
        """A gap of 101 px triggers a split (101^2 > 100^2).

        Two points precede the gap so the first segment (>= 2 pts) is retained.
        """
        vx = np.array([0.0, 1.0, 102.0, 103.0])
        vy = np.zeros(4)
        vis = np.ones(4, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 2

    def test_no_gap_all_points_in_one_segment(self) -> None:
        """Closely spaced points (no gap, all visible) form a single segment."""
        vx = np.linspace(0.0, 10.0, 20)
        vy = np.zeros(20)
        vis = np.ones(20, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 1

    def test_diagonal_gap_uses_euclidean_distance(self) -> None:
        """A diagonal jump of sqrt(2)*80 px (> 100 px) also causes a split.

        dx=dy=80 gives dist=113.1 px > 100 px.  Two points precede the gap
        so the first segment (>= 2 pts) survives.
        """
        # Points 0-1 form the first group (unit step apart).
        # The jump from (1, 0) to (1+80, 80) = (81, 80) is 113.1 px > 100 px.
        vx = np.array([0.0, 1.0, 81.0, 82.0])
        vy = np.array([0.0, 0.0, 80.0, 80.0])
        vis = np.ones(4, dtype=bool)
        segs = _split_polyline(vx, vy, vis)
        assert len(segs) == 2
