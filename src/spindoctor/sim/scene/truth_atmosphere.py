"""A body's ``atmosphere`` block: the haze layer the renderer draws above its surface.

The whole block is truth: the navigator's predicted body keeps a hard limb at the
reference radius, so the soft rendered limb is a designed mismatch.  The block gives
either an explicit exponential column (``scale_height_px`` and ``tau_ref``, with the
optional phase function, reference altitude, detached shell, and albedo) or a named
haze (``haze_model: titan``), which the renderer turns into the column's keys for the
scene's filter.  The structure keys break the column's mirror symmetry about the
sunward axis, and either form may carry them.

:class:`BodyHazeColumn` holds the column's keys and :class:`BodyAtmosphere` adds the
structure keys, so :data:`HAZE_STRUCTURE_KEYS`, the keys whose absence lets the
renderer skip the structure, is the difference of the two.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator

from spindoctor.schema import SCHEMA_CONFIG, Float, NonNeg, Positive, RuleError, Unit
from spindoctor.sim.scene.declare import FloatWithin, optional, required

__all__ = [
    'ATMOSPHERE_KEYS',
    'DEFAULT_HAZE_ALBEDO',
    'DEFAULT_NS_FALLOFF_RATIO',
    'HAZE_MODEL_KEYS',
    'HAZE_STRUCTURE_KEYS',
    'MIN_SECTOR_SHARPNESS_GRADIENT',
    'SECTOR_REFERENCE_HALF_ANGLE_DEG',
    'TITAN_HAZE_MODEL',
    'BodyAtmosphere',
    'BodyCloudBlob',
    'BodyHazeColumn',
    'HazeModel',
]

#: The named hazes a block can take in place of the column's keys.
HazeModel = Literal['titan']
#: The ``haze_model`` value that selects Titan's published haze.
TITAN_HAZE_MODEL: HazeModel = 'titan'

DEFAULT_HAZE_ALBEDO: float = 0.6
"""The haze ``albedo`` a scene's ``atmosphere`` block takes when it leaves the key out.

The albedo is the haze particles' single-scattering albedo and lies in [0, 1].  A
fully lit, optically thick grazing haze column reflects ``albedo * P / 4`` in I/F,
where ``P`` is the phase factor.  The haze's emergent intensity is not capped, so a
strong forward-scattering peak can carry the signal above 1, and the detector
saturates there as a real camera does.
"""

#: The haze falloff-length factor over the scaled hemisphere when the block leaves
#: ``ns_falloff_ratio`` out: the two hemispheres alike.
DEFAULT_NS_FALLOFF_RATIO = 1.0

SECTOR_REFERENCE_HALF_ANGLE_DEG: float = 60.0
"""Azimuthal half-width the sharpness gradient is expressed against.

``sector_sharpness_gradient`` states the fractional change in falloff length
between the sunward point of the limb and the edge of a sector this wide, so
a gradient of 1.0 doubles the falloff length there.  A fixed sim-side
reference, deliberately not read from the navigator's sector configuration:
the rendered scene is nature, and it must not move when a navigation
threshold is retuned.
"""

MIN_SECTOR_SHARPNESS_GRADIENT: float = -SECTOR_REFERENCE_HALF_ANGLE_DEG / 180.0
"""Exclusive lower bound on ``sector_sharpness_gradient``.

The gradient scales the haze falloff length by ``1 + g * azimuth /
reference``, and the azimuth runs to ``pi`` at the anti-sunward point, so a
``g`` at or below ``-reference / pi`` asks for a zero or negative length
there.  Bounding it here means the renderer never has to decide what a
negative atmosphere means.
"""

#: A Henyey-Greenstein asymmetry parameter, in (-1, 1).
Asymmetry = Annotated[Float, AfterValidator(FloatWithin(-1.0, 1.0, low_open=True, high_open=True))]
#: A sharpness gradient, above :data:`MIN_SECTOR_SHARPNESS_GRADIENT`.
SharpnessGradient = Annotated[
    Float, AfterValidator(FloatWithin(MIN_SECTOR_SHARPNESS_GRADIENT, low_open=True))
]

#: The column keys a named haze sets, which its block may not also set.
HAZE_MODEL_KEYS: tuple[str, ...] = (
    'scale_height_px',
    'tau_ref',
    'ref_altitude_px',
    'g',
    'g2',
    'g2_fraction',
    'albedo',
)


@dataclass(frozen=True, kw_only=True)
class BodyCloudBlob:
    """One Gaussian cloud painted on the haze disc.

    Parameters:
        center_vu: The cloud center's offset from the body center, in pixels.
        sigma_px: The Gaussian's standard deviation, in pixels.
        amplitude: The brightness added at the cloud center.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    center_vu: tuple[Float, Float] | None = optional(
        'truth',
        "The cloud center's offset from the body center along v and u, in detector "
        'pixels. A planted roll turns it with the body.',
        when_absent='``[0, 0]``, the body center',
    )
    sigma_px: Positive = required(
        'truth',
        "The cloud's Gaussian standard deviation, in detector pixels.",
    )
    amplitude: Float = required(
        'truth',
        "The brightness added at the cloud center, in the units of the body's shading, "
        'where 1 is a white disc lit face-on. A negative amplitude darkens.',
    )


@dataclass(frozen=True, kw_only=True)
class BodyHazeColumn:
    """The keys of an ``atmosphere`` block that describe its haze column.

    Parameters:
        haze_model: A named haze in place of the column's keys.
        scale_height_px: The haze's e-folding scale height, in pixels.
        tau_ref: The tangent optical depth at ``ref_altitude_px``.
        ref_altitude_px: The altitude ``tau_ref`` is stated at, in pixels.
        g: The asymmetry of the phase function's first term.
        g2: The asymmetry of the phase function's second term.
        g2_fraction: The share of the scattered light in the second term.
        detached_px: The altitude of a detached haze shell, in pixels.
        albedo: The haze's single-scattering albedo.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    haze_model: HazeModel | None = optional(
        'truth',
        "A published haze in place of the column's keys: ``titan`` takes Titan's haze, "
        "averaged over the scene filter's passband. A block that sets it sets none of "
        '``scale_height_px``, ``tau_ref``, ``ref_altitude_px``, ``g``, ``g2``, '
        '``g2_fraction``, ``albedo``, and ``detached_px``. The camera needs a passband '
        'table, and the filter may not see the methane bands.',
        when_absent='the column the other keys give',
    )
    scale_height_px: Positive | None = optional(
        'truth',
        "The haze's e-folding scale height, in detector pixels.",
        when_absent='required unless haze_model is set',
    )
    tau_ref: NonNeg | None = optional(
        'truth',
        'The tangent optical depth at ``ref_altitude_px``. Zero is a clear atmosphere, '
        'which leaves the limb hard.',
        when_absent='required unless haze_model is set',
    )
    ref_altitude_px: NonNeg | None = optional(
        'truth',
        'The altitude above the surface, in detector pixels, at which the tangent optical '
        'depth is ``tau_ref``.',
        default=0.0,
    )
    g: Asymmetry | None = optional(
        'truth',
        "The Henyey-Greenstein asymmetry of the phase function's first term, in (-1, 1). "
        'A positive value scatters forward, which brightens the limb at high phase.',
        default=0.0,
    )
    g2: Asymmetry | None = optional(
        'truth',
        "The asymmetry of the phase function's second term, in (-1, 1). It comes with "
        '``g2_fraction``.',
        when_absent='no second term',
    )
    g2_fraction: Unit | None = optional(
        'truth',
        'The share of the scattered light in the second term. It comes with ``g2``.',
        when_absent='no second term',
    )
    detached_px: Positive | None = optional(
        'truth',
        'The altitude above the surface, in detector pixels, of a detached haze shell: a '
        'Gaussian bump one scale height wide in the tangent optical depth, adding '
        '``tau_ref`` at its peak. It shows only above the limb.',
        when_absent='no shell',
    )
    albedo: Unit | None = optional(
        'truth',
        "The haze's single-scattering albedo. Zero is a purely absorbing haze, which "
        'dims what lies behind the halo but adds nothing of its own.',
        default=DEFAULT_HAZE_ALBEDO,
    )


@dataclass(frozen=True, kw_only=True)
class BodyAtmosphere(BodyHazeColumn):
    """A body's ``atmosphere`` block: its haze column and the column's structure.

    The structure keys break the column's mirror symmetry about the sunward axis.  The
    hemispheric keys act on the hemisphere on the +v side of the body's equator before
    ``rotation_z`` turns it, fully poleward of a band 10 degrees either side of the
    equator and blending smoothly across it.

    Parameters:
        interior_ramp_amplitude: The peak of a linear brightness ramp inside the disc.
        ns_asymmetry_amplitude: The fraction one hemisphere's brightness is scaled by.
        ns_falloff_ratio: The factor on one hemisphere's falloff length.
        axis_tilt_deg: The tilt of the haze's illumination axis, in degrees.
        sector_sharpness_gradient: The change in falloff length with azimuth.
        cloud_blobs: The clouds on the disc.
    """

    interior_ramp_amplitude: Float | None = optional(
        'truth',
        'The peak of a linear brightness ramp along the haze illumination axis inside '
        'the disc, added at the sunward limb and subtracted at the anti-sunward one, in '
        "the units of the body's shading.",
        default=0.0,
    )
    ns_asymmetry_amplitude: Float | None = optional(
        'truth',
        "The fraction by which the +v hemisphere's brightness is scaled. Poleward of the "
        'blend band it is a pure scaling wherever the scaled half stays below saturation.',
        default=0.0,
    )
    ns_falloff_ratio: Positive | None = optional(
        'truth',
        "The factor on the haze's falloff length over the +v hemisphere, a difference "
        'between the two halves that no scaling of brightness reproduces.',
        default=DEFAULT_NS_FALLOFF_RATIO,
    )
    axis_tilt_deg: Float | None = optional(
        'truth',
        "The angle, in degrees, by which the haze's own illumination axis is turned away "
        "from the body's sun direction, which the navigator predicts.",
        default=0.0,
    )
    sector_sharpness_gradient: SharpnessGradient | None = optional(
        'truth',
        'The fractional change in falloff length at 60 degrees of azimuth from the '
        'sunward axis, growing linearly with azimuth. It exceeds -1/3, below which the '
        'length turns negative at the anti-sunward point.',
        default=0.0,
    )
    cloud_blobs: tuple[BodyCloudBlob, ...] | None = optional(
        'truth',
        'Gaussian clouds painted on the disc.',
        when_absent='no clouds',
    )

    def __post_init__(self) -> None:
        """Refuse a named haze beside the column's keys, or a column that is incomplete."""
        if self.haze_model is not None:
            stated = [key for key in HAZE_MODEL_KEYS if getattr(self, key) is not None]
            if stated:
                raise RuleError(
                    f'haze_model sets the haze, so the block may not also set {stated}',
                    at='haze_model',
                    reads=('haze_model', *stated),
                )
            if self.detached_px is not None:
                raise RuleError(
                    "detached_px cannot join a haze_model: the shell's depth follows the "
                    "column's reference depth, which would make it as opaque as Titan's "
                    'main haze',
                    at='detached_px',
                    reads=('haze_model', 'detached_px'),
                )
            return
        for key in ('scale_height_px', 'tau_ref'):
            if getattr(self, key) is None:
                raise RuleError('required unless haze_model is set', at=key)
        if (self.g2 is None) != (self.g2_fraction is None):
            missing = 'g2' if self.g2 is None else 'g2_fraction'
            raise RuleError(
                'g2 and g2_fraction are given together',
                at=missing,
                reads=('g2', 'g2_fraction'),
            )


#: The keys of an ``atmosphere`` block that break the column's mirror symmetry.  A
#: block that leaves all of them out renders the plain column.
HAZE_STRUCTURE_KEYS: frozenset[str] = frozenset(
    item.name for item in dataclasses.fields(BodyAtmosphere)
) - frozenset(item.name for item in dataclasses.fields(BodyHazeColumn))

#: Every key an ``atmosphere`` block may carry.  The whole block is one truth key of a
#: body, so the boundary tests read the keys inside it from here.
ATMOSPHERE_KEYS: frozenset[str] = frozenset(
    item.name for item in dataclasses.fields(BodyAtmosphere)
)
