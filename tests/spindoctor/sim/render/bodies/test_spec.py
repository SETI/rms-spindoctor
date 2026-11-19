"""The body identity seed: the scene's seed, else the body's own seed, else a digest.

Every random stream of a simulated body derives from one identity seed, and
:func:`~spindoctor.sim.render.bodies.spec.body_identity_seed` is the one place that
chooses it.
"""

from typing import Any

from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed, body_spec_from_params
from spindoctor.sim.seeds import derive_effect_seed, stable_param_seed

# A fallback key whose digest has its top bit set, so keeping the low 31 bits changes it.
_FALLBACK_KEY = (40.0, 36.0, 34.0, 1, 'MIMAS')
_SCENE_SEED = 2_718_281_828
# A body seed above 2**31, so keeping its low 31 bits changes it.
_BODY_SEED = 3_000_000_000


def _body(**keys: Any) -> BodySpec:
    """A body named in lower case, so the seed's upper-casing of the name shows."""
    return body_spec_from_params({'name': 'mimas', 'semi_a_px': 20.0, **keys}, index=1)


def test_fallback_key_digest_has_its_top_bit_set() -> None:
    """The fallback key's digest is at least 2**31 (guards the masking tests)."""
    assert stable_param_seed(*_FALLBACK_KEY) >= 2**31


def test_scene_seed_mixes_in_the_body_index_and_upper_cased_name() -> None:
    """With a scene seed, the seed is the scene seed's stream for the body's identity."""
    seed = body_identity_seed(_SCENE_SEED, _body(), 1, fallback_key=_FALLBACK_KEY)
    assert seed == derive_effect_seed(_SCENE_SEED, 'body:1:MIMAS')


def test_scene_seed_wins_over_the_body_seed() -> None:
    """A body's own seed does not replace the scene's seed."""
    seed = body_identity_seed(_SCENE_SEED, _body(seed=_BODY_SEED), 1, fallback_key=_FALLBACK_KEY)
    assert seed == derive_effect_seed(_SCENE_SEED, 'body:1:MIMAS')


def test_body_seed_keeps_its_low_31_bits() -> None:
    """With no scene seed, the body's own seed is used, keeping its low 31 bits."""
    seed = body_identity_seed(None, _body(seed=_BODY_SEED), 1, fallback_key=_FALLBACK_KEY)
    assert seed == _BODY_SEED & 0x7FFFFFFF


def test_fallback_is_the_full_digest_of_the_key() -> None:
    """With neither seed, the seed is the digest of the fallback key, all 32 bits."""
    seed = body_identity_seed(None, _body(), 1, fallback_key=_FALLBACK_KEY)
    assert seed == stable_param_seed(*_FALLBACK_KEY)


def test_masked_fallback_keeps_the_digest_low_31_bits() -> None:
    """With ``mask_fallback``, the fallback digest keeps its low 31 bits."""
    seed = body_identity_seed(None, _body(), 1, fallback_key=_FALLBACK_KEY, mask_fallback=True)
    assert seed == stable_param_seed(*_FALLBACK_KEY) & 0x7FFFFFFF
