"""Relief on a body seen from a sub-observer latitude.

A triaxial body with relief is turned 60 degrees south.  The renderer's points seen,
normals, and cast shadows are compared with a brute-force search that follows each
line of sight and each sun ray through the body in its own frame, in pixels.  The
brute force shares only the relief field's samples with the renderer.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
import pytest

from spindoctor.sim.render.bodies import ellipsoid_relief
from spindoctor.sim.render.bodies.ellipsoid_grid import silhouette_mask, stretched_sight_cells
from spindoctor.sim.render.bodies.ellipsoid_relief import (
    ReliefShading,
    relief_normals,
    relief_resolved_on_detector_grid,
    relief_shading,
)
from spindoctor.sim.render.bodies.photometry import incidence_cosines
from spindoctor.sim.render.bodies.relief import ReliefField, synthesize_relief_field
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    ellipsoid_image_normals,
    project_ellipsoid,
)
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_SEMI = np.array([60.0, 50.0, 30.0])
_LAT = math.radians(-60.0)
_ROTATION_Z = 0.6
_SUN = math.radians(40.0)
_PHASE = math.radians(75.0)
_CORR_DEG = 15.0
# From the body frame (the pole, semi-axis b, semi-axis c) to the rotated frame (along
# the pole's image direction, across it, and toward the observer).
_BODY_TO_ROTATED = np.array(
    [
        [math.cos(_LAT), 0.0, -math.sin(_LAT)],
        [0.0, 1.0, 0.0],
        [math.sin(_LAT), 0.0, math.cos(_LAT)],
    ]
)


@dataclass(frozen=True)
class _TurnedBody:
    """The turned body at a sample of its pixels.

    Parameters:
        grid: The body's projection.
        field: The relief realization.
        near: The sampled pixels' mask on the grid.
        hit: Which sampled pixels' lines of sight meet the body.
        truth: ``(N, 3)`` unit parameter-space directions of the points those lines of
            sight meet first, NaN where they miss.
        shading: The renderer's shading of the sampled pixels.
    """

    grid: EllipsoidProjection
    field: ReliefField
    near: NDArrayBoolType
    hit: NDArrayBoolType
    truth: NDArrayFloatType
    shading: ReliefShading


def _inside(body: NDArrayFloatType, field: ReliefField) -> NDArrayBoolType:
    """Whether body-frame points in pixels lie inside the relief surface."""
    scaled = body / _SEMI
    radius = np.linalg.norm(scaled, axis=-1)
    inside = np.asarray(radius < 1.0 + field.h_min, dtype=np.bool_)
    shell = ~inside & (radius < 1.0 + field.h_max)
    inside[shell] = radius[shell] < 1.0 + field.sample_points(scaled[shell])
    return inside


def _first_points_met(
    v_rot: NDArrayFloatType, u_rot: NDArrayFloatType, field: ReliefField
) -> NDArrayFloatType:
    """The unit parameter-space directions lines of sight meet first, NaN where they miss.

    Each line of sight is sampled in depth every 1/32 of a field cell, and the first
    sample inside the surface and the one before it are bisected.
    """
    top = float(_SEMI.max()) * (1.0 + field.h_max)
    step = 2.0 * np.pi / field.grid.shape[0] / 32.0 * float(_SEMI.min())
    depths = np.arange(top, -top, -step)

    def body_points(depth: NDArrayFloatType, pixels: slice) -> NDArrayFloatType:
        v = v_rot[pixels].reshape(-1, *(1,) * (depth.ndim - 1))
        u = u_rot[pixels].reshape(-1, *(1,) * (depth.ndim - 1))
        rotated = np.stack(np.broadcast_arrays(v, u, depth), axis=-1)
        return np.asarray(rotated @ _BODY_TO_ROTATED, dtype=np.float64)

    points = np.full((v_rot.size, 3), np.nan)
    for start in range(0, v_rot.size, 200):
        pixels = slice(start, min(start + 200, v_rot.size))
        inside = _inside(body_points(depths[np.newaxis, :], pixels), field)
        first = np.argmax(inside, axis=1)
        low, high = depths[first], depths[np.maximum(first - 1, 0)]
        for _ in range(30):
            mid = 0.5 * (low + high)
            mid_inside = _inside(body_points(mid, pixels), field)
            low, high = np.where(mid_inside, mid, low), np.where(mid_inside, high, mid)
        scaled = body_points(low, pixels) / _SEMI
        scaled /= np.linalg.norm(scaled, axis=-1, keepdims=True)
        points[pixels] = np.where(inside.any(axis=1)[:, np.newaxis], scaled, np.nan)
    return points


@functools.lru_cache(maxsize=1)
def _turned_body() -> _TurnedBody:
    """The turned body at 2,000 pixels sampled from everywhere its relief can reach."""
    field = synthesize_relief_field(0.05, _CORR_DEG, 3)
    grid = project_ellipsoid(
        (160, 160),
        (80.0, 80.0),
        _SEMI[0],
        semi_b_px=_SEMI[1],
        semi_c_px=_SEMI[2],
        rotation_z=_ROTATION_Z,
        sub_observer_lat=_LAT,
    )
    reach = np.flatnonzero(grid.ellipse_dist_sq < (1.0 + field.h_max) ** 2)
    near = np.zeros(grid.ellipse_dist_sq.shape, dtype=np.bool_)
    near.ravel()[np.random.default_rng(4).choice(reach, 2000, replace=False)] = True
    truth = _first_points_met(grid.v_rot[near], grid.u_rot[near], field)
    shading = relief_shading(grid, field, near, illumination_angle=_SUN, phase_angle=_PHASE)
    return _TurnedBody(grid, field, near, np.isfinite(truth[:, 0]), truth, shading)


def _angles(points: NDArrayFloatType, truth: NDArrayFloatType) -> NDArrayFloatType:
    """The angles between unit directions, in radians."""
    return np.asarray(np.arccos(np.clip(np.sum(points * truth, axis=-1), -1.0, 1.0)))


def test_terrain_is_read_where_the_line_of_sight_meets_a_turned_body() -> None:
    """Each pixel of the turned body reads the terrain its line of sight meets first.

    The smooth body's points lie more than half a correlation length from there on
    over 5% of the pixels.  The renderer's do on none of them.
    """
    body = _turned_body()
    smooth = np.stack(body.grid.surface_unit_coords(), axis=-1)[body.near]
    half_corr = math.radians(_CORR_DEG) / 2.0
    smooth_error = _angles(smooth[body.hit], body.truth[body.hit])
    assert float(np.mean(smooth_error > half_corr)) > 0.05
    assert np.isfinite(body.shading.seen).all()
    seen_error = _angles(body.shading.seen[body.hit], body.truth[body.hit])
    assert float(np.mean(seen_error > half_corr)) == 0.0


def test_a_turned_bodys_terrain_meets_its_outline() -> None:
    """A line of sight meets the terrain exactly where it passes inside the silhouette.

    Pixels within 2e-3 of the outline are left out, where the sampling of the outline
    and of the line of sight decides.
    """
    body = _turned_body()
    drawn = silhouette_mask(body.grid, body.field)[body.near] > 0.0
    distance = np.sqrt(body.grid.ellipse_dist_sq[body.near])
    phi = np.arctan2(
        body.grid.u_rot[body.near] / body.grid.semi_b,
        body.grid.v_rot[body.near] / body.grid.semi_a_outline,
    )
    # The direction toward the observer in the field's frame: the line of sight in the
    # body frame, scaled by the inverse semi-axes.
    toward = np.array([0.0, 0.0, 1.0]) @ _BODY_TO_ROTATED / _SEMI
    toward /= np.linalg.norm(toward)
    outline = 1.0 + body.field.outline_delta(phi, view=(float(toward[0]), float(toward[2])))
    decided = np.abs(distance - outline) > 2e-3
    np.testing.assert_array_equal(drawn[decided], body.hit[decided])


def test_a_turned_bodys_relief_normals_are_the_relief_surfaces() -> None:
    """The shaded normals are the gradients of the relief surface, turned into the image.

    The gradient is taken in the body frame by differences half a field cell long, while
    the renderer's follow the surface.  Where it faces the observer, the 99th percentile
    of the distance between the two is 0.061.
    """
    body = _turned_body()
    seen = body.shading.seen[body.hit]
    points = seen * (1.0 + body.field.sample_points(seen))[:, np.newaxis] * _SEMI
    step = np.pi / body.field.grid.shape[0] * float(_SEMI.min())

    def height(at: NDArrayFloatType) -> NDArrayFloatType:
        scaled = at / _SEMI
        return np.asarray(np.linalg.norm(scaled, axis=-1) - 1.0 - body.field.sample_points(scaled))

    gradient = np.stack(
        [height(points + step * axis) - height(points - step * axis) for axis in np.eye(3)],
        axis=-1,
    )
    gradient /= np.linalg.norm(gradient, axis=-1, keepdims=True)
    rotated = gradient @ _BODY_TO_ROTATED.T
    expected = np.stack(
        [
            rotated[:, 0] * math.cos(_ROTATION_Z) + rotated[:, 1] * math.sin(_ROTATION_Z),
            rotated[:, 1] * math.cos(_ROTATION_Z) - rotated[:, 0] * math.sin(_ROTATION_Z),
            rotated[:, 2],
        ],
        axis=-1,
    )
    shaded = np.stack(
        [body.shading.normal_v, body.shading.normal_u, body.shading.normal_z], axis=-1
    )[body.hit]
    facing = expected[:, 2] > 0.05
    apart = np.linalg.norm(shaded - expected, axis=-1)[facing]
    assert float(np.percentile(apart, 99.0)) < 0.1


def test_a_turned_bodys_cast_shadows_follow_the_sun() -> None:
    """The cast shadows are those of the sun rays through the body in its own frame.

    Each sun ray starts on the pixel's line of sight, level with the point it meets,
    and the pixel is shadowed when the ray falls below the terrain after it has risen
    above it.  The comparison covers the pixels whose relief normal faces the sun,
    the only ones a cast shadow darkens.
    """
    body = _turned_body()
    illum_v, illum_u, illum_z = illumination_vector(illumination_angle=_SUN, phase_angle=_PHASE)
    sun = (
        np.array(
            [
                illum_v * math.cos(_ROTATION_Z) - illum_u * math.sin(_ROTATION_Z),
                illum_v * math.sin(_ROTATION_Z) + illum_u * math.cos(_ROTATION_Z),
                illum_z,
            ]
        )
        @ _BODY_TO_ROTATED
    )
    seen = body.truth[body.hit]
    surface = seen * (1.0 + body.field.sample_points(seen))[:, np.newaxis] * _SEMI
    # In the rotated frame, the line of sight is the pixel's (v, u) at every depth.
    rotated = surface @ _BODY_TO_ROTATED.T
    rotated[:, 0] = body.grid.v_rot[body.near][body.hit]
    rotated[:, 1] = body.grid.u_rot[body.near][body.hit]
    points = rotated @ _BODY_TO_ROTATED
    step = 2.0 * np.pi / body.field.grid.shape[0] / 32.0 * float(_SEMI.min())
    distances = np.arange(step, 2.0 * float(_SEMI.max()) * (1.0 + body.field.h_max), step)
    expected = np.zeros(seen.shape[0], dtype=np.bool_)
    for start in range(0, seen.shape[0], 200):
        part = slice(start, start + 200)
        rays = points[part, np.newaxis] + distances[:, np.newaxis] * sun
        inside = _inside(rays, body.field)
        risen = np.logical_or.accumulate(~inside, axis=1)
        expected[part] = (inside[:, 1:] & risen[:, :-1]).any(axis=1)
    shading = body.shading
    lit = (
        incidence_cosines(
            shading.normal_v,
            shading.normal_u,
            shading.normal_z,
            illumination_angle=_SUN,
            phase_angle=_PHASE,
        )[body.hit]
        > 0.0
    )
    shadowed = shading.shadowed[body.near][body.hit]
    assert int(expected[lit].sum()) > 50
    assert int((shadowed != expected)[lit].sum()) <= 0.01 * int(expected[lit].sum())


# A body much shorter along its c axis than the others, seen almost along its pole, so
# that its outline is far shorter along the pole's direction than semi-axis a is.
_EDGE_ON_SEMI = (60.0, 60.0, 12.0)
_EDGE_ON_LAT = math.radians(85.0)


def _edge_on_grid() -> EllipsoidProjection:
    """The flattened body's projection, seen almost along its pole."""
    semi_a, semi_b, semi_c = _EDGE_ON_SEMI
    return project_ellipsoid(
        (160, 160),
        (80.0, 80.0),
        semi_a,
        semi_b_px=semi_b,
        semi_c_px=semi_c,
        sub_observer_lat=_EDGE_ON_LAT,
    )


def test_the_relief_cutover_measures_the_turned_outline() -> None:
    """Relief is too fine for the detector grid where the turned outline shortens it.

    A 10-degree correlation length spans 10.5 px of semi-axis a and 2.3 px of the
    outline's semi-axis along the pole, below the 5 px cutover.
    """
    field = synthesize_relief_field(0.02, 10.0, 3)
    assert not relief_resolved_on_detector_grid(field, _edge_on_grid())


def test_faint_relief_stretches_no_cell_of_a_turned_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On relief a billionth of the radius high, no cell's view pulls away from a neighbor's.

    The points seen are then the smooth body's, so they lie as far apart as the points
    the renderer compares them with.

    Parameters:
        monkeypatch: Records the cells the renderer marks.
    """
    field = synthesize_relief_field(1e-9, 15.0, 3)
    grid = _edge_on_grid()
    marked: list[NDArrayBoolType] = []

    def recording(seen: NDArrayFloatType, smooth: NDArrayFloatType) -> NDArrayBoolType:
        marked.append(stretched_sight_cells(seen, smooth))
        return marked[-1]

    monkeypatch.setattr(ellipsoid_relief, 'stretched_sight_cells', recording)
    ellipsoid_relief.add_detector_relief(
        grid,
        grid,
        field,
        ellipsoid_image_normals(grid),
        up_factor=1,
        illumination_angle=_SUN,
        phase_angle=_PHASE,
    )
    assert int(marked[0][grid.ellipse_dist_sq < 0.8].sum()) == 0


def test_relief_normals_reach_two_pixels_past_the_turned_outline() -> None:
    """Relief is shaded to two pixels past its reach along the outline's short axis."""
    field = synthesize_relief_field(0.02, 15.0, 3)
    grid = _edge_on_grid()
    shading = relief_normals(
        grid,
        field,
        ellipsoid_image_normals(grid),
        illumination_angle=_SUN,
        phase_angle=_PHASE,
    )
    reach = 1.0 + field.h_max + 2.0 / grid.semi_a_outline
    np.testing.assert_array_equal(shading.near, grid.ellipse_dist_sq < reach * reach)
