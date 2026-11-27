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

The idealized types live in this module, except the ``instrument_config`` block's,
which live in :mod:`spindoctor.sim.scene.idealized_camera`.  The truth types live in
the ``truth`` modules (:mod:`spindoctor.sim.scene.truth`,
:mod:`spindoctor.sim.scene.truth_optics`, and
:mod:`spindoctor.sim.scene.truth_detector`).  The top level, the optics, detector,
noise, and ``instrument_config`` blocks, and the ``spk_error`` block are typed so far.
The blocks not yet typed are listed in :data:`spindoctor.sim.scene.truth.UNTYPED_BLOCKS`
and checked by the rules modules until their types land.

One value crosses from a truth block into the navigator's view: the readout binning.
A real frame's label states its readout mode, so a real pipeline knows it, but a
scene sets it in the truth ``detector`` block, beside the detector values it selects.
The navigator's view carries it as a top-level ``binning`` key.  So that the view can
be built from the idealized types alone, :class:`IdealizedScene` gets a ``binning``
field that a scene never writes: the parse fills it from ``detector.binning``, and
from that key's default when the scene leaves it out.  The field lands with the
navigator's view, along with a rule that refuses a top-level ``binning`` key in a
scene.
"""

from __future__ import annotations

from dataclasses import dataclass

from spindoctor.schema import SCHEMA_CONFIG, Bool, Float, Int, Positive, PositiveInt, RuleError, Str
from spindoctor.sim.scene.declare import EditorHint, NonEmptyStr, optional, required
from spindoctor.sim.scene.idealized_camera import InstrumentConfig

__all__ = ['CURRENT_SCHEMA_VERSION', 'DEFAULT_EXPOSURE_SEC', 'IdealizedScene']

#: The scene format version every scene is written in.
CURRENT_SCHEMA_VERSION: int = 2

#: The exposure time, in seconds, of a scene that leaves ``exposure_sec`` out.
DEFAULT_EXPOSURE_SEC = 1.0


@dataclass(frozen=True, kw_only=True)
class IdealizedScene:
    """The top-level keys of a scene that a real pipeline could know.

    Parameters:
        schema_version: The scene format version.
        scene_name: The scene's name.
        instrument: The simulated camera.
        size_v: The frame's height, in pixels.
        size_u: The frame's width, in pixels.
        exposure_sec: The commanded exposure time, in seconds.
        midtime_utc: The observation midtime the frame reports.
        closest_planet: The planet the frame reports as closest.
        time: The observation time, in TDB seconds.
        ring_epoch: The epoch of the ring orbits, in TDB seconds.
        fit_camera_rotation: Whether the navigator fits the camera's rotation.
        instrument_config: The keys the scene changes in its camera's navigator block.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    schema_version: Int | None = optional(
        'idealized',
        'The scene format version. Only the current version, 2, is accepted, and a scene '
        'file must set it.',
        when_absent='the current version',
    )
    scene_name: NonEmptyStr | None = optional(
        'idealized',
        "The scene's name. A scene file must set it to the file's name without the extension.",
        when_absent='no name',
    )
    instrument: NonEmptyStr = required(
        'idealized',
        'The simulated camera, by its name or one of its aliases in the instrument '
        "catalog. The camera's catalog entry supplies the detector values, the navigator "
        'block, and the optics that ``artifacts.instrument_defaults`` turns on.',
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
        'The commanded exposure time, in seconds, which a calibrated frame divides by. '
        "On the CCD chain the frame collects light for this time plus the camera's "
        'extra integration time, and on the vidicon chain for this time alone. That '
        "integration time scales each star's flux and the counts of cosmic rays, "
        'radiation transients, and bright-dark pairs. On the CCD chain it also scales '
        'the charge from extended sources, as a ratio to the reference exposure, and '
        'the charge from the dark current.',
        default=DEFAULT_EXPOSURE_SEC,
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
        'The observation time, in TDB seconds. A ring edge precesses over the time since '
        '``ring_epoch``.',
        default=0.0,
        editor=EditorHint(minimum=-1e10, maximum=1e10, decimals=1),
    )
    ring_epoch: Float | None = optional(
        'idealized',
        'The epoch of the ring orbits, in TDB seconds: catalog knowledge both sides read.',
        default=0.0,
        editor=EditorHint(minimum=-1e10, maximum=1e10, decimals=1),
    )
    fit_camera_rotation: Bool | None = optional(
        'idealized',
        "Whether the navigator fits the camera's rotation, overriding the setting of the "
        "instrument's navigator block.",
        when_absent="the navigator block's setting",
    )
    instrument_config: InstrumentConfig | None = optional(
        'idealized',
        "The keys the scene changes in its camera's navigator block, which is what the "
        'navigator knows about the camera. Each key a scene leaves out keeps the '
        "block's value. A scene can't set ``data_units``: the frame's units come from the "
        'instrument entry.',
        when_absent="the navigator block's values",
    )

    def __post_init__(self) -> None:
        """Refuse a scene written in another format version."""
        version = self.schema_version
        if version is not None and version != CURRENT_SCHEMA_VERSION:
            raise RuleError(
                f'expected {CURRENT_SCHEMA_VERSION}, the current scene format, got {version}',
                at='schema_version',
            )
