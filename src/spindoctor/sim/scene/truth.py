"""The typed scene: :class:`Scene`, the whole scene as the renderer reads it.

:class:`Scene` extends :class:`~spindoctor.sim.scene.idealized.IdealizedScene`, the
keys a real pipeline could know, with the keys only the test knows: the planted
pointing error, the seeds, the render settings, and the blocks that plant the optics,
the detector, and its noise.  It also holds ``published_equals_truth``, a key that
neither the renderer nor the navigator reads.
:func:`~spindoctor.sim.scene.validate.parse_scene` builds a :class:`Scene` from a
scene's mapping, which stays the scene's saved form.

The scene's blocks land in the typed scene one at a time.  A block not typed yet is
listed in :data:`UNTYPED_BLOCKS` with its boundary class, left out of :class:`Scene`,
and checked by its rules module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import AfterValidator

from spindoctor.schema import SCHEMA_CONFIG, Float, Int, NonNeg, Positive, PositiveInt, Str
from spindoctor.sim.scene.declare import Boundary, EditorHint, IntWithin, optional, required
from spindoctor.sim.scene.idealized import IdealizedScene
from spindoctor.sim.scene.truth_detector import DetectorOverride, Noise
from spindoctor.sim.scene.truth_optics import Optics

__all__ = ['NOISE_SEED_MAX', 'UNTYPED_BLOCKS', 'NoiseSeed', 'Scene', 'SpkError']

#: The largest ``noise_seed``: the scene editor's control stops at the 32-bit limit.
NOISE_SEED_MAX = 2**31 - 1

#: A seed for the detector's per-frame noise.  The scene editor shows -1 for a scene
#: that leaves the key out, so a seed is never negative.
NoiseSeed = Annotated[Int, AfterValidator(IntWithin(0, NOISE_SEED_MAX))]

#: The top-level blocks :class:`Scene` does not type yet, each with its boundary class.
#: Each is checked by its module of rules until its type lands.
UNTYPED_BLOCKS: dict[str, Boundary] = {
    'bodies': 'idealized',
    'stars': 'idealized',
    'ring_system': 'idealized',
    'sky_counts': 'truth',
    'artifacts': 'truth',
    'expected': 'test',
}


@dataclass(frozen=True, kw_only=True)
class SpkError:
    """A planted spacecraft-ephemeris error, which moves each object by parallax.

    The displacement at an object scales as ``reference_range_km / range_km``, so a
    scene that sets this block gives every body and the ring system a ``range_km``.

    Parameters:
        dv_px: The displacement along v at the reference range, in pixels.
        du_px: The displacement along u at the reference range, in pixels.
        reference_range_km: The range the displacement is stated at, in km.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    dv_px: Float | None = optional(
        'truth',
        'The displacement along v at the reference range, in pixels.',
        default=0.0,
        editor=EditorHint(minimum=-500.0, maximum=500.0, decimals=3, step=0.5),
    )
    du_px: Float | None = optional(
        'truth',
        'The displacement along u at the reference range, in pixels.',
        default=0.0,
        editor=EditorHint(minimum=-500.0, maximum=500.0, decimals=3, step=0.5),
    )
    reference_range_km: Positive | None = optional(
        'truth',
        'The range the displacement is stated at, in km.',
        when_absent='no displacement',
        editor=EditorHint(minimum=0.0, maximum=1.0e9, decimals=2, step=100.0),
    )


@dataclass(frozen=True, kw_only=True)
class Scene(IdealizedScene):
    """A whole scene: the idealized keys and the keys only the test knows.

    Parameters:
        random_seed: The scene's seed.
        noise_seed: The seed of the detector's per-frame noise.
        offset_v: The planted pointing offset along v, in pixels.
        offset_u: The planted pointing offset along u, in pixels.
        offset_rotation_deg: The planted roll about the boresight, in degrees.
        oversample: The render's oversampling factor.
        star_catalog_scatter_px: The sigma of each star's scatter off its catalog
            position, in pixels.
        optics: The optical path.
        detector: The detector values the scene replaces.
        noise: The detector noise.
        spk_error: The planted spacecraft-ephemeris error.
        published_equals_truth: The truth paths the navigator block repeats on purpose.
    """

    random_seed: Int = required(
        'truth',
        "The scene's seed. Every stage's and effect's seed derives from it, except the "
        "detector noise's when the scene sets ``noise_seed``.",
        editor=EditorHint(minimum=0, maximum=2147483647),
    )
    noise_seed: NoiseSeed | None = optional(
        'truth',
        "The seed of the detector's per-frame noise alone. The scene, the telemetry "
        'losses, the fixed pattern, and the artifact activations stay on random_seed.',
        when_absent='random_seed',
        editor=EditorHint(minimum=-1, maximum=2147483647),
    )
    offset_v: Float | None = optional(
        'truth',
        'The planted pointing offset along v, in pixels, which moves every body, ring, and star.',
        default=0.0,
        editor=EditorHint(minimum=-10000.0, maximum=10000.0, decimals=3),
    )
    offset_u: Float | None = optional(
        'truth',
        'The planted pointing offset along u, in pixels, which moves every body, ring, and star.',
        default=0.0,
        editor=EditorHint(minimum=-10000.0, maximum=10000.0, decimals=3),
    )
    offset_rotation_deg: Float | None = optional(
        'truth',
        'The planted roll, in degrees from +v toward +u, which turns everything in the '
        "field of view about the boresight. The camera's own PSF, distortion, ghosts, "
        'and radial stray light do not turn.',
        default=0.0,
        editor=EditorHint(minimum=-180.0, maximum=180.0, decimals=3),
    )
    oversample: PositiveInt | None = optional(
        'truth',
        "The render's oversampling factor.",
        when_absent='4 with an active PSF, else 1',
        editor=EditorHint(minimum=1, maximum=8),
    )
    star_catalog_scatter_px: NonNeg | None = optional(
        'truth',
        'The sigma, in pixels, of a seeded Gaussian scatter of every star off its '
        "catalog position, added to the star's own catalog error.",
        default=0.0,
    )
    optics: Optics | None = optional(
        'truth',
        'The optical path: the PSF, smear, distortion, ghosts, and stray light.',
        when_absent="the camera's optics under instrument_defaults, else none",
    )
    detector: DetectorOverride | None = optional(
        'truth',
        "The camera's detector values the scene replaces.",
        when_absent="the camera's values",
    )
    noise: Noise | None = optional(
        'truth',
        'The detector noise the scene plants.',
        when_absent="the camera's noise under instrument_defaults, else none",
    )
    spk_error: SpkError | None = optional(
        'truth',
        'A planted spacecraft-ephemeris error, which moves each object by parallax.',
        when_absent='no parallax error',
    )
    published_equals_truth: tuple[Str, ...] | None = optional(
        'test',
        "The truth paths, such as ``noise.read_noise_dn``, whose value the scene's "
        '``instrument_config`` repeats on purpose. The validator checks it, and neither '
        'the renderer nor the navigator reads it.',
        when_absent='none',
    )
