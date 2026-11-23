"""The body identity seed: the scene's seed mixed with the body's index and name.

Every random stream of a simulated body derives from one identity seed, and
:func:`~spindoctor.sim.render.bodies.spec.body_identity_seed` is the one place that
chooses it.
"""

from spindoctor.sim.render.bodies.spec import body_identity_seed, body_spec_from_params
from spindoctor.sim.seeds import derive_effect_seed

_SCENE_SEED = 2_718_281_828


def test_scene_seed_mixes_in_the_body_index_and_upper_cased_name() -> None:
    """The seed is the scene seed's stream for the body's index and upper-cased name."""
    body = body_spec_from_params({'name': 'mimas', 'semi_a_px': 20.0}, index=1)
    seed = body_identity_seed(_SCENE_SEED, body, 1)
    assert seed == derive_effect_seed(_SCENE_SEED, 'body:1:MIMAS')
