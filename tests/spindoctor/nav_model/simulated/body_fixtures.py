"""Shared fixtures for the simulated body model tests."""

from typing import Any

from tests.shims import bare_nav_context

from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim

__all__ = [
    'LIMB_SIZE',
    'SIZE',
    'body_feature_types',
    'large_body',
    'limb_feature',
    'limb_obs',
    'small_obs',
]

SIZE = 96


def small_obs() -> ObsSim:
    """A coiss_nac sim obs sized for the body scenes below."""
    return ObsSim.from_file(
        '/tmp/body_sim.json',
        sim_params={
            'size_v': SIZE,
            'size_u': SIZE,
            'instrument': 'coiss_nac',
            'random_seed': 1,
        },
    )


LIMB_SIZE = 220


def limb_obs() -> ObsSim:
    """A coiss_nac sim obs sized for a well-resolved limb body."""
    return ObsSim.from_file(
        '/tmp/limb_sim.json',
        sim_params={
            'size_v': LIMB_SIZE,
            'size_u': LIMB_SIZE,
            'instrument': 'coiss_nac',
            'random_seed': 1,
        },
    )


def body_feature_types(obs: ObsSim, body_params: dict[str, Any]) -> list[str]:
    """Build the model and return the feature-type names it emits."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    features = model.to_features(bare_nav_context(obs))
    return [f.feature_type.name for f in features]


def large_body(**overrides: Any) -> dict[str, Any]:
    """A well-resolved sphere (diameter 130 px) at low phase."""
    params = {
        'name': 'RHEA',
        'center_v': LIMB_SIZE / 2.0,
        'center_u': LIMB_SIZE / 2.0,
        'semi_a_px': 65.0,
        'semi_b_px': 65.0,
        'semi_c_px': 65.0,
        'illumination_angle': 25.0,
        'phase_angle': 30.0,
    }
    params.update(overrides)
    return params


def limb_feature(
    obs: ObsSim,
    body_params: dict[str, Any],
    *,
    sibling_bodies: list[dict[str, Any]] | None = None,
) -> Any:
    """Build the model and return its emitted LIMB_ARC feature, or None."""
    model = NavModelBodySimulated(
        'body', obs, body_params['name'], body_params, sibling_bodies=sibling_bodies
    )
    model.create_model()
    for feature in model.to_features(bare_nav_context(obs)):
        if feature.feature_type.name == 'LIMB_ARC':
            return feature
    return None
