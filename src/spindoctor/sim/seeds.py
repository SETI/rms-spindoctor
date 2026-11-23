"""Deterministic seed derivation for the simulator's randomized effects.

Every randomized effect in the simulator (background noise, background stars,
crater placement, and any effect added later) must draw from a stream that is
byte-identical across processes and runs for identical inputs.  Python's
built-in ``hash`` is unsuitable: it is salted per process for ``str`` and
``bytes`` (controlled by ``PYTHONHASHSEED``), so ``hash((seed, 'noise'))``
varies between interpreter runs.  These helpers derive sub-seeds from a stable
cryptographic digest instead, so the same scene always renders the same pixels.

Sub-seeds are derived per effect *by name* rather than by draw order.  Adding a
new randomized effect therefore does not perturb the seeds of pre-existing
effects, which keeps earlier rendered scenes and their regression baselines
stable when a new effect lands.
"""

import hashlib

# A 32-bit seed keeps derived values in [0, 2**32 - 1]; four digest bytes
# land exactly in that range.
_SEED_BYTES = 4


def _digest_to_seed(key: bytes) -> int:
    """Map an arbitrary byte key to a 32-bit seed via a stable digest.

    Parameters:
        key: The byte string to hash.

    Returns:
        An integer in [0, 2**32 - 1] suitable for seeding numpy.random.Generator.
    """
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:_SEED_BYTES], 'big')


def derive_effect_seed(scene_seed: int, effect: str) -> int:
    """Derive a per-effect sub-seed from the scene's single random seed.

    The derivation is process-stable (independent of PYTHONHASHSEED) and keyed
    on the effect name, so distinct effects draw independent streams and adding
    a new effect leaves existing effects' seeds unchanged.

    Parameters:
        scene_seed: The scene's top-level random seed.
        effect: A stable name identifying the effect (e.g. 'noise').

    Returns:
        An integer in [0, 2**32 - 1] suitable for seeding numpy.random.Generator.
    """
    return _digest_to_seed(f'{int(scene_seed)}:{effect}'.encode())
