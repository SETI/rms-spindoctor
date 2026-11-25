"""The idealized scene types: what a real pipeline could know about a frame.

The typed scene is split along the information boundary by its structure.  Each
idealized type here holds the keys a real pipeline could know from catalogs, SPICE,
labels, or configuration.  The truth-bearing type of the same thing extends it with
the keys only the test knows: :class:`~spindoctor.sim.scene.truth.Scene` extends
:class:`IdealizedScene`.  A truth block that has no idealized part, such as the
``noise`` block, has no idealized type.

The scene's types land block by block.  Every type, its module, and the block it
types:

=========================  =========================  ==================================
Idealized type             Truth type                 Block
=========================  =========================  ==================================
``IdealizedScene``         ``Scene``                  the top level
(none)                     ``Optics`` and its parts   ``optics``
(none)                     ``DetectorOverride``       ``detector``
(none)                     ``Noise``                  ``noise``
(none)                     ``SpkError``               ``spk_error``
``IdealizedStar``          ``Star``                   ``stars``
(none)                     ``SkyCounts``              ``sky_counts``
``IdealizedBody``          ``Body``                   ``bodies``
(none)                     ``BodyAtmosphere``         ``bodies[].atmosphere``
``IdealizedRingSystem``    ``RingSystem``             ``ring_system``
``IdealizedRingFeature``   ``RingFeature``            ``ring_system.features``
(none)                     ``Artifacts``              ``artifacts``
``InstrumentConfig``       (none)                     ``instrument_config``
(none)                     ``Expected``               ``expected``
=========================  =========================  ==================================

The idealized types live in this module, and the truth types in the ``truth``
modules (:mod:`spindoctor.sim.scene.truth`, :mod:`spindoctor.sim.scene.truth_optics`,
and :mod:`spindoctor.sim.scene.truth_detector`).  The top level, the optics, detector,
and noise blocks, and the ``spk_error`` block are typed so far.  The blocks not yet
typed are listed in :data:`spindoctor.sim.scene.truth.UNTYPED_BLOCKS` and checked by
the rules modules until their types land.
"""

from __future__ import annotations

from dataclasses import dataclass

from spindoctor.schema import SCHEMA_CONFIG, Bool, Float, Int, Positive, PositiveInt, RuleError, Str
from spindoctor.sim.scene.declare import EditorHint, NonEmptyStr, optional, required

__all__ = ['CURRENT_SCHEMA_VERSION', 'IdealizedScene']

#: The scene format version every scene is written in.
CURRENT_SCHEMA_VERSION: int = 2


@dataclass(frozen=True, kw_only=True)
class IdealizedScene:
    """The top-level keys of a scene that a real pipeline could know.

    Parameters:
        schema_version: The scene format version.
        scene_name: The scene's name.
        instrument: The simulated camera.
        size_v: The frame's height, in pixels.
        size_u: The frame's width, in pixels.
        exposure_sec: The exposure time, in seconds.
        midtime_utc: The observation midtime the frame reports.
        closest_planet: The planet the frame reports as closest.
        time: The observation time, in days.
        ring_epoch: The epoch of the ring orbits, in days.
        fit_camera_rotation: Whether the navigator fits the camera's rotation.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    schema_version: Int | None = optional(
        'idealized',
        'The scene format version. Only the current version, 2, is accepted.',
        when_absent='the current version',
    )
    scene_name: NonEmptyStr | None = optional(
        'idealized',
        "The scene's name. A scene file's name must equal its file name without the extension.",
        when_absent='no name',
    )
    instrument: NonEmptyStr = required(
        'idealized',
        'The simulated camera, by its name or one of its aliases in the instrument '
        'catalog. It sets the detector chain, the PSF, and the navigator block.',
    )
    size_v: PositiveInt = required(
        'idealized',
        "The frame's height, in pixels.",
        editor=EditorHint(minimum=10, maximum=9999),
    )
    size_u: PositiveInt = required(
        'idealized',
        "The frame's width, in pixels.",
        editor=EditorHint(minimum=10, maximum=9999),
    )
    exposure_sec: Positive | None = optional(
        'idealized',
        'The exposure time, in seconds. It scales the charge the detector collects and '
        'the counts of cosmic rays, radiation transients, and bright-dark pairs.',
        default=1.0,
        editor=EditorHint(minimum=0.001, maximum=100000.0, decimals=3),
    )
    midtime_utc: Str | None = optional(
        'idealized',
        'The observation midtime the frame reports, as its label would.',
        when_absent='no midtime',
    )
    closest_planet: Str | None = optional(
        'idealized',
        'The planet the frame reports as closest.',
        when_absent='no planet',
    )
    time: Float | None = optional(
        'idealized',
        'The observation time, in days. A ring edge precesses over the time since ``ring_epoch``.',
        default=0.0,
        editor=EditorHint(minimum=-1e10, maximum=1e10, decimals=1),
    )
    ring_epoch: Float | None = optional(
        'idealized',
        'The epoch of the ring orbits, in days: catalog knowledge both sides read.',
        default=0.0,
        editor=EditorHint(minimum=-1e10, maximum=1e10, decimals=1),
    )
    fit_camera_rotation: Bool | None = optional(
        'idealized',
        "Whether the navigator fits the camera's rotation, overriding the setting of the "
        "instrument's navigator block.",
        when_absent="the navigator block's setting",
    )

    def __post_init__(self) -> None:
        """Refuse a scene written in another format version."""
        version = self.schema_version
        if version is not None and version != CURRENT_SCHEMA_VERSION:
            raise RuleError(
                f'expected {CURRENT_SCHEMA_VERSION}, the current scene format, got {version}',
                at='schema_version',
            )
