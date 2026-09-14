"""Simulated-scene star NavModel.

``NavModelStarsSimulated`` builds its star list from the scene's idealized
catalog entries (``obs.nav_params['stars']``) instead of a catalog reduction,
then reuses the catalog-driven ``NavModelStars`` feature-emission machinery.
These tests cover the obs-gated instance construction, that the model builds
its catalog from the filtered view, and that it emits one STAR feature per
star at the *unshifted* predicted position (so a technique recovers the
planted offset).
"""

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model.stars.nav_model_stars_simulated import NavModelStarsSimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim

_SIZE = 128


def _obs(stars: list[dict[str, Any]] | None) -> ObsSim:
    """A coiss_nac sim obs rendering the given star list (or none)."""
    sim_params: dict[str, Any] = {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 7,
        'bodies': [],
    }
    if stars is not None:
        sim_params['stars'] = stars
    return ObsSim.from_file('/tmp/stars_sim.json', sim_params=sim_params)


@dataclass
class _FakeContext:
    """NavContext stand-in carrying the masks the star model consults."""

    image_noise_sigma: float = 4.0
    saturation_mask_ext: np.ndarray = field(
        default_factory=lambda: np.zeros((_SIZE + 100, _SIZE + 100), dtype=bool)
    )
    cosmic_ray_mask_ext: np.ndarray = field(
        default_factory=lambda: np.zeros((_SIZE + 100, _SIZE + 100), dtype=bool)
    )


def test_instances_built_for_simulated_obs_with_stars() -> None:
    """A simulated obs that rendered stars yields exactly one star model."""
    obs = _obs([{'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0}])
    instances = NavModelStarsSimulated.instances_for_obs(obs)
    assert len(instances) == 1


def test_instance_is_the_simulated_subclass() -> None:
    """The built instance is the simulated star model, not the base class."""
    obs = _obs([{'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0}])
    instances = NavModelStarsSimulated.instances_for_obs(obs)
    assert isinstance(instances[0], NavModelStarsSimulated)


def test_no_instance_for_simulated_obs_without_stars() -> None:
    """A simulated obs with no rendered stars yields no star model."""
    obs = _obs(None)
    assert NavModelStarsSimulated.instances_for_obs(obs) == []


def test_create_model_adopts_scene_star_catalog() -> None:
    """``create_model`` populates the star list from ``obs.nav_params``."""
    obs = _obs(
        [
            {'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0},
            {'name': 'S2', 'v': 80.5, 'u': 35.5, 'vmag': 4.0},
        ]
    )
    model = NavModelStarsSimulated('stars', obs)
    model.create_model()
    assert len(model.stars) == 2


def test_metadata_records_star_count() -> None:
    """The model metadata reports the rendered star count."""
    obs = _obs([{'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0}])
    model = NavModelStarsSimulated('stars', obs)
    model.create_model()
    assert model.metadata['star_count'] == 1


def test_to_features_emits_one_star_feature_per_star() -> None:
    """One STAR feature is emitted for each rendered star."""
    obs = _obs(
        [
            {'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0},
            {'name': 'S2', 'v': 80.5, 'u': 35.5, 'vmag': 4.0},
        ]
    )
    model = NavModelStarsSimulated('stars', obs)
    model.create_model()
    features = model.to_features(cast(NavContext, _FakeContext()))
    assert len(features) == 2


def test_to_features_predicts_unshifted_position_in_extfov() -> None:
    """The emitted STAR feature predicts the unshifted ``(v, u)`` in extfov coords.

    The renderer applies the planted offset only to the image, not to the star
    record, so the prediction is the scene's own position in extfov coordinates
    -- what a technique differences against the shifted detection.  A scene
    states a position as a pixel corner, so the star at ``(40.5, 50.5)`` sits on
    pixel ``(40, 50)`` and the prediction is that index plus the margin.
    """
    obs = _obs([{'name': 'S1', 'v': 40.5, 'u': 50.5, 'vmag': 3.0}])
    model = NavModelStarsSimulated('stars', obs)
    model.create_model()
    feature = model.to_features(cast(NavContext, _FakeContext()))[0]
    assert feature.feature_type is NavFeatureType.STAR
    geometry = feature.geometry
    assert isinstance(geometry, StarGeometry)
    expected = (40.0 + obs.extfov_margin_v, 50.0 + obs.extfov_margin_u)
    assert geometry.predicted_vu == expected
