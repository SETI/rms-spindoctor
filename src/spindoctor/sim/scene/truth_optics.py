"""The scene's ``optics`` block: the optical path the optics stage applies.

Every sub-block is truth: the navigator knows only its own configured PSF and the
camera's published distortion, and never the scene's.  Each sub-block is optional,
and a scene that leaves one out gets no such effect unless
``artifacts.instrument_defaults`` turns on the camera's own (the camera's PSF and
distortion residual).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spindoctor.schema import SCHEMA_CONFIG, Bool, Float, NonNeg, Positive, RuleError, Unit
from spindoctor.sim.scene.declare import EditorHint, optional

__all__ = [
    'DEFAULT_PSF_N',
    'DEFAULT_PSF_R0_PX',
    'DEFAULT_PSF_W',
    'DEFAULT_SMEAR_OBJECT_CLASS',
    'DEFAULT_STRAY_LIGHT_MODEL',
    'Optics',
    'OpticsDistortion',
    'OpticsGhost',
    'OpticsPsf',
    'OpticsSmear',
    'OpticsStrayLight',
    'SmearObjectClass',
    'StrayLightModel',
]

#: The objects one motion-smear entry smears.
SmearObjectClass = Literal['all', 'stars', 'bodies', 'rings']
#: The shape of the stray-light field: a ramp across the frame, or a bump.
StrayLightModel = Literal['linear', 'radial']

#: The Moffat wing's energy fraction when an explicit PSF kernel leaves ``w`` out: the
#: kernel then has no wing.
DEFAULT_PSF_W = 0.0
#: The Moffat core radius, in pixels, when an explicit PSF kernel leaves ``r0`` out.
DEFAULT_PSF_R0_PX = 2.0
#: The Moffat index when an explicit PSF kernel leaves ``n`` out.
DEFAULT_PSF_N = 3.0
#: The objects a smear entry smears when it leaves ``object_class`` out: all of them.
DEFAULT_SMEAR_OBJECT_CLASS: SmearObjectClass = 'all'
#: The stray-light field's shape when the block leaves ``model`` out.
DEFAULT_STRAY_LIGHT_MODEL: StrayLightModel = 'linear'

# The keys of an explicit PSF kernel, which the navigator-matched form excludes.
_KERNEL_KEYS: tuple[str, ...] = ('sigma_v', 'sigma_u', 'w', 'r0', 'n')


@dataclass(frozen=True, kw_only=True)
class OpticsPsf:
    """The whole-scene point-spread function.

    A scene writes either the explicit kernel, a Gaussian core and a Moffat wing, or
    the navigator-matched form ``{match_navigator: true}`` alone.

    Parameters:
        match_navigator: The navigator-matched form.
        sigma_v: The Gaussian core's sigma along v, in pixels.
        sigma_u: The Gaussian core's sigma along u, in pixels.
        w: The fraction of the energy in the Moffat wing.
        r0: The Moffat core radius, in pixels.
        n: The Moffat index.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    match_navigator: Bool | None = optional(
        'truth',
        "Resolve the PSF to the navigator's own Gaussian at the instrument's "
        '``star_psf_sigma``. It is written true or left out, and no other PSF key may '
        'appear beside it.',
        when_absent='the explicit kernel the other keys give',
    )
    sigma_v: Positive | None = optional(
        'truth',
        "The Gaussian core's sigma along v, in pixels. An explicit kernel needs it.",
        when_absent='required unless match_navigator is set',
        editor=EditorHint(minimum=0.01, maximum=20.0, decimals=3, step=0.05),
    )
    sigma_u: Positive | None = optional(
        'truth',
        "The Gaussian core's sigma along u, in pixels.",
        when_absent='sigma_v',
        editor=EditorHint(minimum=0.01, maximum=20.0, decimals=3, step=0.05),
    )
    w: Unit | None = optional(
        'truth',
        'The fraction of the energy in the Moffat wing.',
        default=DEFAULT_PSF_W,
        editor=EditorHint(minimum=0.0, maximum=1.0, decimals=4, step=0.005),
    )
    r0: Positive | None = optional(
        'truth',
        'The Moffat core radius, in pixels.',
        default=DEFAULT_PSF_R0_PX,
        editor=EditorHint(minimum=0.01, maximum=50.0, decimals=3, step=0.1),
    )
    n: Float | None = optional(
        'truth',
        'The Moffat index.',
        default=DEFAULT_PSF_N,
        editor=EditorHint(minimum=0.1, maximum=20.0, decimals=3, step=0.1),
    )

    def __post_init__(self) -> None:
        """Refuse a false ``match_navigator``, a mixed form, or a kernel with no core."""
        if self.match_navigator is not None:
            if not self.match_navigator:
                raise RuleError(
                    'a false value describes a PSF block with no kernel; drop the optics.psf '
                    'block instead',
                    at='match_navigator',
                )
            extra = [key for key in _KERNEL_KEYS if getattr(self, key) is not None]
            if extra:
                raise RuleError(
                    f'match_navigator is exclusive; drop {extra}',
                    at='match_navigator',
                    reads=('match_navigator', *extra),
                )
            return
        if self.sigma_v is None:
            raise RuleError(
                'an explicit kernel needs sigma_v, its core width (or write the '
                'match_navigator form)',
                at='sigma_v',
            )


@dataclass(frozen=True, kw_only=True)
class OpticsSmear:
    """One motion-smear entry: a drift during the exposure, for one class of object.

    Parameters:
        dv_px: The drift along v, in pixels.
        du_px: The drift along u, in pixels.
        object_class: The objects the entry smears.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    dv_px: Float | None = optional(
        'truth',
        'The drift along v over the exposure, in pixels.',
        default=0.0,
        editor=EditorHint(minimum=-200.0, maximum=200.0, decimals=3, step=0.5),
    )
    du_px: Float | None = optional(
        'truth',
        'The drift along u over the exposure, in pixels.',
        default=0.0,
        editor=EditorHint(minimum=-200.0, maximum=200.0, decimals=3, step=0.5),
    )
    object_class: SmearObjectClass | None = optional(
        'truth',
        'The objects the entry smears. One ``all`` entry smears the whole scene, and '
        'several entries give differential smear.',
        default=DEFAULT_SMEAR_OBJECT_CLASS,
    )


@dataclass(frozen=True, kw_only=True)
class OpticsDistortion:
    """The residual geometric distortion: a radial polynomial and a non-radial wander.

    Parameters:
        k1: The radial polynomial's coefficient of rho squared.
        k2: The radial polynomial's coefficient of rho to the fourth.
        center_v: The optical center's v, in pixels.
        center_u: The optical center's u, in pixels.
        nonradial_rms_px: The RMS of the non-radial wander, in pixels.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    k1: Float | None = optional(
        'truth',
        'The coefficient of rho**2 in the radial polynomial, where rho is the distance from '
        'the optical center over half the frame diagonal.',
        default=0.0,
        editor=EditorHint(minimum=-1.0, maximum=1.0, decimals=5, step=0.001),
    )
    k2: Float | None = optional(
        'truth',
        'The coefficient of rho**4 in the radial polynomial.',
        default=0.0,
        editor=EditorHint(minimum=-1.0, maximum=1.0, decimals=5, step=0.001),
    )
    center_v: Float | None = optional(
        'truth',
        "The optical center's v, in pixels.",
        when_absent='the frame center',
        editor=EditorHint(minimum=0.0, maximum=20000.0, decimals=2, step=1.0),
    )
    center_u: Float | None = optional(
        'truth',
        "The optical center's u, in pixels.",
        when_absent='the frame center',
        editor=EditorHint(minimum=0.0, maximum=20000.0, decimals=2, step=1.0),
    )
    nonradial_rms_px: NonNeg | None = optional(
        'truth',
        'The RMS of the seeded, smooth non-radial wander, in pixels.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=50.0, decimals=3, step=0.05),
    )


@dataclass(frozen=True, kw_only=True)
class OpticsGhost:
    """One ghost reflection: a displaced, defocused, dimmed copy of the scene.

    Parameters:
        dv_px: The copy's offset along v, in pixels.
        du_px: The copy's offset along u, in pixels.
        amplitude: The copy's brightness, as a fraction of the scene's.
        defocus_sigma: The copy's blur sigma, in pixels.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    dv_px: Float | None = optional(
        'truth',
        "The copy's offset along v, in pixels.",
        default=0.0,
        editor=EditorHint(minimum=-2000.0, maximum=2000.0, decimals=2, step=1.0),
    )
    du_px: Float | None = optional(
        'truth',
        "The copy's offset along u, in pixels.",
        default=0.0,
        editor=EditorHint(minimum=-2000.0, maximum=2000.0, decimals=2, step=1.0),
    )
    amplitude: Float | None = optional(
        'truth',
        "The copy's brightness, as a fraction of the scene's. Zero turns the entry off.",
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=1.0, decimals=4, step=0.005),
    )
    defocus_sigma: NonNeg | None = optional(
        'truth',
        "The copy's blur sigma, in pixels.",
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=100.0, decimals=2, step=0.5),
    )


@dataclass(frozen=True, kw_only=True)
class OpticsStrayLight:
    """The smooth scattered-light field, applied last before the detector stage.

    Parameters:
        amplitude: The field's peak, as a fraction of full scale.
        direction_deg: The direction the linear ramp rises toward, in degrees.
        model: The field's shape.
        center_v: The radial bump's center v, in pixels.
        center_u: The radial bump's center u, in pixels.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    amplitude: Float | None = optional(
        'truth',
        "The field's peak, in the units of the signal, where 1.0 is the brightness of a "
        'surface at I/F 1. Zero turns it off.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=1.0, decimals=3, step=0.01),
    )
    direction_deg: Float | None = optional(
        'truth',
        'The direction the linear ramp rises toward, in degrees from +v toward +u.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1),
    )
    model: StrayLightModel | None = optional(
        'truth',
        "The field's shape: a ``linear`` ramp across the frame or a ``radial`` bump.",
        default=DEFAULT_STRAY_LIGHT_MODEL,
    )
    center_v: Float | None = optional(
        'truth',
        "The radial bump's center v, in pixels.",
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )
    center_u: Float | None = optional(
        'truth',
        "The radial bump's center u, in pixels.",
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )


@dataclass(frozen=True, kw_only=True)
class Optics:
    """The scene's ``optics`` block.

    Parameters:
        psf: The whole-scene point-spread function.
        smear: The motion-smear entries.
        distortion: The residual geometric distortion.
        ghosts: The ghost reflections.
        stray_light: The scattered-light field.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    psf: OpticsPsf | None = optional(
        'truth',
        'The whole-scene point-spread function.',
        when_absent="the camera's PSF under instrument_defaults, else none",
    )
    smear: tuple[OpticsSmear, ...] | None = optional(
        'truth',
        'The motion-smear entries, one per class of object.',
        when_absent='no smear',
    )
    distortion: OpticsDistortion | None = optional(
        'truth',
        'The residual geometric distortion.',
        when_absent="the camera's residual under instrument_defaults, else none",
    )
    ghosts: tuple[OpticsGhost, ...] | None = optional(
        'truth',
        'The ghost reflections.',
        when_absent='no ghosts',
    )
    stray_light: OpticsStrayLight | None = optional(
        'truth',
        "The scattered-light field. It exercises the navigator's background filter.",
        when_absent='no stray light',
    )
