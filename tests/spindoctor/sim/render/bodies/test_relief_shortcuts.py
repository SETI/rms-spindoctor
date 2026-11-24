"""The relief renderer's shortcuts change none of its answers.

The cast-shadow march skips the samples a slope bound proves above the terrain, the
line-of-sight search off the field's pole bounds the slope by the cells near each
step, and the working grid computes the relief only inside the silhouette.  Each is
checked against the work it saves: a march that takes every sample, a search that
uses the field's steepest slope anywhere, and a body that computes the relief at
every pixel it reaches.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies import ellipsoid, ellipsoid_relief, relief, relief_sight
from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, create_ellipsoid_body
from spindoctor.sim.render.bodies.relief import (
    ReliefField,
    cast_shadows,
    synthesize_relief_field,
)
from spindoctor.sim.render.bodies.relief_sight import visible_surface_directions
from spindoctor.sim.shared.ellipsoid import EllipsoidProjection
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType


def _surface_points(field: ReliefField, count: int, seed: int) -> NDArrayFloatType:
    """Points on the relief surface in uniform random directions."""
    directions = np.random.default_rng(seed).normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return np.asarray(directions * (1.0 + field.sample_points(directions))[:, np.newaxis])


def _unit(vector: tuple[float, float, float]) -> NDArrayFloatType:
    """The unit vector along ``vector``."""
    array = np.array(vector, dtype=np.float64)
    return np.asarray(array / np.linalg.norm(array))


def _tilted_plane_field(amplitude: float) -> ReliefField:
    """The field ``amplitude * y``, which rises toward ``+y`` at most ``amplitude`` per radian."""
    n = 256
    lat = -np.pi + 2.0 * np.pi * np.arange(n) / n
    lon = 2.0 * np.pi * np.arange(n) / n
    grid = amplitude * np.cos(lat)[:, np.newaxis] * np.sin(lon)[np.newaxis, :]
    return ReliefField(
        grid=grid,
        rms=amplitude,
        corr_deg=60.0,
        h_min=float(grid.min()),
        h_max=float(grid.max()),
        slope_max=relief._grid_slope_bound(grid),
    )


@pytest.mark.parametrize(
    'sun',
    [(0.0, 1.0, 0.0), (-0.8, 0.6, 0.0)],
    ids=['skimming-up-the-slope', 'falling-toward-the-body'],
)
def test_a_ray_stays_above_the_terrain_over_its_clear_distance(
    sun: tuple[float, float, float],
) -> None:
    """Sampled every 1/256 of its clear distance, a ray never dips into the terrain.

    The terrain rises as steeply as the field allows, 5% per radian, under rays that
    start 1e-4 above it.  One runs level up the slope, and the slope catches it
    after about 1e-4 / 0.05 of travel.  The other falls toward the body, and meets
    the terrain after about 1e-4 / 0.83.  The clear distance must fall short of both.

    Parameters:
        sun: The rays' direction.
    """
    field = _tilted_plane_field(0.05)
    start = np.array([[1.0001, 0.0, 0.0]])
    direction = _unit(sun)
    radius = np.linalg.norm(start, axis=1)
    surface = 1.0 + field.sample_points(start)
    impact = np.linalg.norm(np.cross(start, direction), axis=1)
    clear = relief._clear_distances(
        start, radius, surface, impact=impact, sun=direction, field=field
    )
    along = start + (clear[0] * np.linspace(0.0, 1.0, 257))[:, np.newaxis] * direction
    clearance = np.linalg.norm(along, axis=1) - 1.0 - field.sample_points(along)
    assert float(clear[0]) > 0.0
    assert float(clearance.min()) > 0.0


def test_the_march_answers_as_if_it_took_every_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skipping the samples proved above the terrain leaves every shadow as it was."""
    field = synthesize_relief_field(0.05, 5.0, 3)
    points = _surface_points(field, 20000, 4)
    sun = _unit((0.5, 0.2, 0.84))
    step = 2.0 * np.pi / field.grid.shape[0] / 8.0
    skipping = cast_shadows(points, sun, field, step=step)
    monkeypatch.setattr(
        relief, '_clear_distances', lambda ray, radius, surface, **_: np.zeros(ray.shape[0])
    )
    every_sample = cast_shadows(points, sun, field, step=step)
    assert int(every_sample.sum()) > 1000
    np.testing.assert_array_equal(skipping, every_sample)


def test_the_cells_near_a_point_bound_the_slope_within_a_cell_of_it() -> None:
    """Between two points a cell of arc apart, the height changes no faster than the bound.

    Half the first points lie within ten cells of a pole, where a cell's neighbors
    span many columns and the nearest points can lie across the pole.
    """
    field = synthesize_relief_field(0.05, 5.0, 3)
    n = field.grid.shape[0]
    cell = 2.0 * np.pi / n
    nearby = relief_sight._nearby_cell_slopes(field.grid)
    rng = np.random.default_rng(6)
    count = 200000
    lat = np.where(
        np.arange(count) % 2 == 0,
        rng.uniform(-np.pi / 2.0, np.pi / 2.0, count),
        np.sign(rng.uniform(-1.0, 1.0, count))
        * rng.uniform(np.pi / 2.0 - 10.0 * cell, np.pi / 2.0, count),
    )
    lon = rng.uniform(0.0, 2.0 * np.pi, count)
    first = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    # A second point at most a cell of arc away, in a random direction.
    toward = rng.normal(size=(count, 3))
    toward -= np.sum(toward * first, axis=1, keepdims=True) * first
    toward /= np.linalg.norm(toward, axis=1, keepdims=True)
    arc = rng.uniform(0.0, cell, count)
    second = first * np.cos(arc)[:, np.newaxis] + toward * np.sin(arc)[:, np.newaxis]
    rise = np.abs(field.sample_points(second) - field.sample_points(first))
    row = np.clip(np.floor((lat + np.pi) / cell).astype(np.intp) - n // 4, 0, nearby.shape[0] - 1)
    col = np.floor(lon / cell).astype(np.intp) % n
    excess = rise - nearby[row, col] * arc
    assert float(excess.max()) <= 1e-12


def test_the_turned_search_finds_what_the_steepest_slope_finds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off the pole, the search reads the same points with either slope bound.

    At 10% relief and a 1-degree correlation length, the search's quarter-cell steps
    pass over ridge tops.  A search that took no slope into account would miss one on
    17 of these lines of sight.
    """
    field = synthesize_relief_field(0.1, 1.0, 1)
    view = (math.sin(math.radians(50.0)), math.cos(math.radians(50.0)))
    rng = np.random.default_rng(1)
    radius = np.sqrt(rng.random(20000))
    angle = rng.uniform(0.0, 2.0 * np.pi, radius.size)
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    nearby = visible_surface_directions(x, y, field, view=view)

    def steepest(self: Any, upper: NDArrayFloatType, index: NDArrayFloatType) -> NDArrayFloatType:
        return np.full(index.size, self.field.slope_max)

    monkeypatch.setattr(relief_sight._SightLines, '_turned_slope_bound', steepest)
    np.testing.assert_array_equal(nearby, visible_surface_directions(x, y, field, view=view))


def _relief_spec(**extra: Any) -> EllipsoidBodySpec:
    """A moon of radius 60 detector pixels with rough relief, at a high phase.

    Most of its disc lies within the field's slope band, so the march visits it.
    """
    values: dict[str, Any] = {
        'semi_a_px': 240.0,
        'semi_b_px': 240.0,
        'semi_c_px': 240.0,
        'illumination_angle': math.radians(30.0),
        'phase_angle': math.radians(80.0),
        'limb_relief_rms': 0.05,
        'limb_relief_corr_deg': 8.0,
        'relief_seed': 3,
        'oversample': 4,
    }
    values.update(extra)
    return EllipsoidBodySpec(**values)


def _render(spec: EllipsoidBodySpec) -> NDArrayFloatType:
    """The body on a frame 2.6 radii across, its center off the pixel grid."""
    size = int(2.6 * spec.semi_a_px / spec.oversample) * spec.oversample
    return create_ellipsoid_body((size, size), (size / 2.0 + 0.3, size / 2.0 - 0.7), spec)


@pytest.mark.parametrize(
    'extra',
    [
        {},
        {'limb_relief_corr_deg': 3.0},
        {'oversample': 1, 'semi_a_px': 60.0, 'semi_b_px': 60.0, 'semi_c_px': 60.0},
    ],
    ids=['detector-grid-tilt', 'whole-working-grid', 'oversample-1'],
)
def test_a_relief_body_renders_as_if_shaded_at_every_pixel(
    extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Computing the relief only inside the silhouette changes no pixel of the render.

    Parameters:
        extra: The spec's departure from the base moon: relief whose tilt is
            interpolated from the detector grid, relief too fine for that and shaded
            wholly on the working grid, and a body drawn at oversample 1.
        monkeypatch: Widens the silhouette the relief is computed in.
    """
    spec = _relief_spec(**extra)
    masked = _render(spec)
    for name in ('relief_normals', 'add_detector_relief'):
        original = getattr(ellipsoid, name)

        def everywhere(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            kwargs['inside'] = np.ones_like(kwargs['inside'])
            return _original(*args, **kwargs)

        monkeypatch.setattr(ellipsoid, name, everywhere)
    np.testing.assert_array_equal(masked, _render(spec))


def test_the_working_grid_shades_no_pixel_outside_the_silhouette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every working-grid pixel the relief is computed at lies inside the silhouette."""
    calls: list[tuple[EllipsoidProjection, NDArrayBoolType]] = []
    insides: list[tuple[EllipsoidProjection, NDArrayBoolType]] = []
    shading = ellipsoid_relief.relief_shading
    detector_relief = ellipsoid_relief.add_detector_relief

    def recording_shading(
        grid: EllipsoidProjection, field: ReliefField, near: Any, **kwargs: Any
    ) -> Any:
        calls.append((grid, near.copy()))
        return shading(grid, field, near, **kwargs)

    def recording_detector(work_grid: EllipsoidProjection, *args: Any, **kwargs: Any) -> Any:
        insides.append((work_grid, kwargs['inside']))
        return detector_relief(work_grid, *args, **kwargs)

    monkeypatch.setattr(ellipsoid_relief, 'relief_shading', recording_shading)
    monkeypatch.setattr(ellipsoid, 'add_detector_relief', recording_detector)
    _render(_relief_spec())
    work_grid, inside = insides[0]
    work_near = [near for grid, near in calls if grid is work_grid]
    assert int(np.count_nonzero(~inside)) > 0
    assert int(np.count_nonzero(work_near[0] & ~inside)) == 0
