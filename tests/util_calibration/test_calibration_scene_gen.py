"""Tests for the calibration campaign scene generator.

The generator lives in ``util/calibration/scene_gen.py``, outside the importable package,
and is loaded under its own module name so it cannot collide with the agreement
campaign's ``scene_gen`` module.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from spindoctor.sim.scene.fields import MESH_UNSUPPORTED_KEYS

_SCENE_GEN_PATH = Path(__file__).resolve().parents[2] / 'util' / 'calibration' / 'scene_gen.py'
_SEED = 20260718
_PHOTOMETRIC_KEYS = frozenset(
    ('photometric_law', 'minnaert_k', 'opposition_surge', 'albedo_texture')
)
# Enough draws that a fraction of 0.15 or more is present on any seeded run.
_MANY = 150


def _load_scene_gen() -> ModuleType:
    """Load ``util/calibration/scene_gen.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('calibration_scene_gen', _SCENE_GEN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene_gen = _load_scene_gen()


def _bodies(
    *,
    irregular_fraction: float,
    count: int = 60,
    radius: float = 40.0,
    pose_error_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Build ``count`` catalog bodies from one seeded generator.

    Parameters:
        irregular_fraction: Probability of drawing an irregular-catalog body.
        count: Number of bodies to build.
        radius: Apparent body radius (px).
        pose_error_deg: Pose error handed to every body.

    Returns:
        The body entries.
    """
    rng = random.Random(_SEED)
    return [
        scene_gen._catalog_body(
            rng,
            center_v=100.0,
            center_u=100.0,
            radius=radius,
            phase=30.0,
            illumination=10.0,
            irregular_fraction=irregular_fraction,
            pose_error_deg=pose_error_deg,
        )
        for _ in range(count)
    ]


def _disc_bodies(count: int) -> list[dict[str, Any]]:
    """Return the body of each of ``count`` disc scenes drawn from one generator."""
    rng = random.Random(_SEED)
    return [scene_gen.gen_disc(rng)['bodies'][0] for _ in range(count)]


@pytest.mark.parametrize('family', scene_gen.FAMILIES)
def test_family_generates_and_validates(family: str) -> None:
    """Every scene of every family builds and passes the scene validator."""
    scenes = scene_gen.generate_scenes(family, 8, campaign_seed=_SEED)
    assert len(scenes) == 8


def test_catalog_body_width_is_twice_its_radius() -> None:
    """A catalog body's semi-axis a is its radius, so it is twice the radius across."""
    body = _bodies(irregular_fraction=0.0, count=1)[0]
    assert 2.0 * body['semi_a_px'] == 80.0


@pytest.mark.parametrize(
    ('irregular_fraction', 'b_range', 'c_range'),
    [(0.0, (0.97, 1.0), (0.95, 1.0)), (1.0, (0.75, 0.95), (0.6, 0.85))],
    ids=['regular', 'irregular'],
)
def test_semi_axes_scale_from_the_radius(
    irregular_fraction: float,
    b_range: tuple[float, float],
    c_range: tuple[float, float],
) -> None:
    """Semi-axes b and c are drawn fractions of the full radius, not of half of it."""
    for body in _bodies(irregular_fraction=irregular_fraction):
        assert b_range[0] * 40.0 <= body['semi_b_px'] <= b_range[1] * 40.0
        assert c_range[0] * 40.0 <= body['semi_c_px'] <= c_range[1] * 40.0


@pytest.mark.parametrize('irregular_fraction', [0.0, 1.0], ids=['regular', 'irregular'])
def test_drawn_mean_radius_in_km_is_the_catalog_mean_radius(
    irregular_fraction: float,
) -> None:
    """``km_per_pixel`` times the mean of the drawn semi-axes is the catalog mean radius."""
    for body in _bodies(irregular_fraction=irregular_fraction):
        mean_radius_px = (body['semi_a_px'] + body['semi_b_px'] + body['semi_c_px']) / 3.0
        catalog_mean_radius_km = scene_gen._catalog()[body['name']][0]
        assert body['km_per_pixel'] * mean_radius_px == pytest.approx(catalog_mean_radius_km)


def test_regular_moons_are_ellipsoids_with_photometric_keys() -> None:
    """Regular moons render as ellipsoids, and the photometric keys are drawn on them."""
    bodies = _bodies(irregular_fraction=0.0)
    assert {body['shape_model'] for body in bodies} == {'ellipsoid'}
    drawn = set().union(*(body.keys() for body in bodies))
    assert drawn >= _PHOTOMETRIC_KEYS


def test_regular_moons_are_oriented() -> None:
    """Regular moons are turned by a drawn rotation and sub-observer latitude."""
    bodies = _bodies(irregular_fraction=0.0)
    assert max(body['sub_observer_lat_deg'] for body in bodies) > 10.0
    assert max(body['rotation_z'] for body in bodies) > 10.0


def test_irregular_moons_are_meshes_without_ellipsoid_keys() -> None:
    """Irregular moons render as meshes and carry no key the mesh renderer ignores."""
    bodies = _bodies(irregular_fraction=1.0)
    assert {body['shape_model'] for body in bodies} == {'polyhedral_mesh'}
    drawn = set().union(*(body.keys() for body in bodies))
    assert drawn.isdisjoint(MESH_UNSUPPORTED_KEYS)


def test_irregular_moons_plant_lumpiness_from_the_catalog_residual() -> None:
    """A mesh's lumpiness is 3x its moon's fractional RMS residual."""
    for body in _bodies(irregular_fraction=1.0, count=10):
        mean_radius_km, rms_residual_km = scene_gen._catalog()[body['name']]
        assert body['mesh_lumpiness'] == pytest.approx(3.0 * rms_residual_km / mean_radius_km)


@pytest.mark.parametrize('irregular_fraction', [0.0, 1.0], ids=['regular', 'irregular'])
def test_both_kinds_of_moon_draw_limb_relief(irregular_fraction: float) -> None:
    """Regular and irregular moons both draw the limb-relief field on some bodies."""
    bodies = _bodies(irregular_fraction=irregular_fraction, count=_MANY)
    relieved = [body for body in bodies if 'limb_relief_rms' in body]
    assert relieved
    assert len(relieved) < len(bodies)


@pytest.mark.parametrize('key', ['pose_scatter', 'shading'])
def test_meshes_draw_pose_scatter_and_shading(key: str) -> None:
    """Meshes draw pose scatter and gouraud shading on some bodies but not all."""
    bodies = _bodies(irregular_fraction=1.0, count=_MANY)
    drawn = [body for body in bodies if key in body]
    assert drawn
    assert len(drawn) < len(bodies)


def test_ellipsoid_pose_error_is_added_to_the_navigator_rotation() -> None:
    """A pose error shifts the navigator's predicted ellipsoid rotation by that amount."""
    for body in _bodies(irregular_fraction=0.0, count=10, pose_error_deg=12.0):
        assert body['nav_override']['rotation_z'] == pytest.approx(body['rotation_z'] + 12.0)


def test_mesh_pose_error_is_added_to_the_navigator_pose() -> None:
    """A pose error shifts the navigator's predicted mesh pose by that amount."""
    for body in _bodies(irregular_fraction=1.0, count=10, pose_error_deg=12.0):
        drawn = body['pose_euler_deg']
        predicted = body['nav_override']['pose_euler_deg']
        assert predicted[1] == pytest.approx(drawn[1] + 12.0)


def test_disc_family_draws_the_banded_disc_on_some_ellipsoids() -> None:
    """The disc family draws the banded disc texture, and only on ellipsoids."""
    bodies = _disc_bodies(_MANY)
    banded = [body for body in bodies if 'disc_texture' in body]
    assert banded
    assert {body['shape_model'] for body in banded} == {'ellipsoid'}


def test_disc_family_draws_transits_inside_the_banded_disc() -> None:
    """The disc family draws transits on some banded discs and on no plain body."""
    bodies = _disc_bodies(4 * _MANY)
    transiting = [body for body in bodies if 'transits' in body]
    assert transiting
    assert all('disc_texture' in body for body in transiting)


def test_transiting_moon_is_a_fraction_of_the_body_radius() -> None:
    """A transiting moon's radius is 0.10 to 0.25 of the body's radius."""
    bodies = _disc_bodies(4 * _MANY)
    moon_fractions = [
        transit['moon']['radius_px'] / body['semi_a_px']
        for body in bodies
        for transit in body.get('transits', [])
    ]
    assert moon_fractions
    assert min(moon_fractions) >= 0.10
    assert max(moon_fractions) <= 0.25


def test_blob_family_spans_below_the_emission_floor() -> None:
    """Some blob bodies are narrower than the 5 px BodyBlobNav emission floor."""
    rng = random.Random(_SEED)
    widths = [2.0 * scene_gen.gen_blob(rng)['bodies'][0]['semi_a_px'] for _ in range(_MANY)]
    assert min(widths) < 5.0
    assert max(widths) > 5.0
