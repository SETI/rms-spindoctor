"""Tests for the calibration campaign scene generator's catalog bodies.

Run from the repository checkout, where ``pyproject.toml`` puts ``src`` on the path:
``pytest util/calibration/tests``.  The generator is loaded under its own module name
so it cannot collide with the agreement campaign's ``scene_gen`` module.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from spindoctor.sim.scene.fields import MESH_UNSUPPORTED_KEYS

_SCENE_GEN_PATH = Path(__file__).resolve().parent.parent / 'scene_gen.py'
_SEED = 20260718
_BODY_FAMILIES = ('disc', 'limb', 'terminator', 'blob')
_PHOTOMETRIC_KEYS = frozenset(
    ('photometric_law', 'minnaert_k', 'opposition_surge', 'albedo_texture')
)


def _load_scene_gen() -> ModuleType:
    """Load ``util/calibration/scene_gen.py`` as a uniquely named module."""
    spec = importlib.util.spec_from_file_location('calibration_scene_gen', _SCENE_GEN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene_gen = _load_scene_gen()


def _bodies(*, irregular_fraction: float, count: int = 60) -> list[dict[str, Any]]:
    """Build ``count`` catalog bodies from one seeded generator.

    Parameters:
        irregular_fraction: Probability of drawing an irregular-catalog body.
        count: Number of bodies to build.

    Returns:
        The body entries.
    """
    rng = random.Random(_SEED)
    return [
        scene_gen._catalog_body(
            rng,
            center_v=100.0,
            center_u=100.0,
            radius=40.0,
            phase=30.0,
            illumination=10.0,
            irregular_fraction=irregular_fraction,
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize('family', _BODY_FAMILIES)
def test_body_family_generates_and_validates(family: str) -> None:
    """Every scene of a body family builds and passes the scene validator."""
    scenes = scene_gen.generate_scenes(family, 30, campaign_seed=_SEED)
    assert len(scenes) == 30


def test_catalog_body_width_is_twice_its_radius() -> None:
    """A catalog body's semi-axis a is its radius, so it is twice the radius across."""
    body = _bodies(irregular_fraction=0.0, count=1)[0]
    assert 2.0 * body['semi_a_px'] == 80.0


def test_regular_moons_are_ellipsoids_with_photometric_keys() -> None:
    """Regular moons render as ellipsoids, and the photometric keys are drawn on them."""
    bodies = _bodies(irregular_fraction=0.0)
    assert {body['shape_model'] for body in bodies} == {'ellipsoid'}
    drawn = set().union(*(body.keys() for body in bodies))
    assert drawn >= _PHOTOMETRIC_KEYS


def test_irregular_moons_are_meshes_without_ellipsoid_keys() -> None:
    """Irregular moons render as meshes and carry no key the mesh renderer ignores."""
    bodies = _bodies(irregular_fraction=1.0)
    assert {body['shape_model'] for body in bodies} == {'polyhedral_mesh'}
    drawn = set().union(*(body.keys() for body in bodies))
    assert drawn.isdisjoint(MESH_UNSUPPORTED_KEYS)
