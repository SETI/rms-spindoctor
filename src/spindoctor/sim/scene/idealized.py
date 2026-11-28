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
``IdealizedBody``          (none)                     ``bodies[].nav_override``
``IdealizedRingSystem``    ``RingSystem``             ``ring_system``
``IdealizedRingFeature``   ``RingFeature``            ``ring_system.features``
(none)                     ``Artifacts``              ``artifacts``
``InstrumentConfig``       (none)                     ``instrument_config``
(none)                     ``Expected``               ``expected``
=========================  =========================  ==================================

The idealized types live in this module, except the ``instrument_config`` block's,
which live in :mod:`spindoctor.sim.scene.idealized_camera`.  The truth types live in
the ``truth`` modules (:mod:`spindoctor.sim.scene.truth`,
:mod:`spindoctor.sim.scene.truth_optics`, :mod:`spindoctor.sim.scene.truth_detector`,
:mod:`spindoctor.sim.scene.truth_body`, and
:mod:`spindoctor.sim.scene.truth_atmosphere`).  The top level, the optics, detector,
noise, ``instrument_config``, and ``spk_error`` blocks, and the ``stars`` and
``bodies`` entries are typed so far.  The blocks not yet typed are listed in
:data:`spindoctor.sim.scene.truth.UNTYPED_BLOCKS` and checked by the rules modules
until their types land.

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
from typing import Annotated, Literal

from pydantic import AfterValidator

from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Float,
    Int,
    NonNeg,
    Positive,
    RuleError,
    Str,
)
from spindoctor.sim.scene.declare import (
    EditorHint,
    FloatWithin,
    IntWithin,
    NonEmptyStr,
    optional,
    required,
)
from spindoctor.sim.scene.fields import default_body_name
from spindoctor.sim.scene.idealized_camera import InstrumentConfig
from spindoctor.sim.shared.mesh import (
    DEFAULT_MESH_LUMPINESS,
    DEFAULT_MESH_N_LAT,
    DEFAULT_MESH_N_LON,
)
from spindoctor.sim.shared.stars import DEFAULT_CATALOG_NAME, DEFAULT_SPECTRAL_CLASS, DEFAULT_VMAG

__all__ = [
    'CURRENT_SCHEMA_VERSION',
    'DEFAULT_EXPOSURE_SEC',
    'DEFAULT_SHAPE_MODEL',
    'MIN_FRAME_SIZE_PX',
    'MIN_MESH_N_LAT',
    'MIN_MESH_N_LON',
    'IdealizedBody',
    'IdealizedScene',
    'IdealizedStar',
    'ShapeModel',
]

#: The scene format version every scene is written in.
CURRENT_SCHEMA_VERSION: int = 2

#: The exposure time, in seconds, of a scene that leaves ``exposure_sec`` out.
DEFAULT_EXPOSURE_SEC = 1.0

#: The smallest frame, in pixels on a side.  The margin the lens distortion needs around
#: the frame grows steeply as the frame shrinks, so a smaller frame cannot be rendered.
#: The scene editor's size controls stop at the same number.
MIN_FRAME_SIZE_PX = 10

#: A frame side: an integer of at least ``MIN_FRAME_SIZE_PX``.
FrameSide = Annotated[Int, AfterValidator(IntWithin(MIN_FRAME_SIZE_PX))]

#: The two body shapes: a triaxial ellipsoid, or an irregular polyhedral mesh.
ShapeModel = Literal['ellipsoid', 'polyhedral_mesh']
#: The shape of a body that leaves ``shape_model`` out.
DEFAULT_SHAPE_MODEL: ShapeModel = 'ellipsoid'
#: The fewest latitude bands a mesh can have.
MIN_MESH_N_LAT = 2
#: The most latitude bands a mesh can have, the editor's limit.
MAX_MESH_N_LAT = 256
#: The fewest longitude divisions a mesh can have.
MIN_MESH_N_LON = 3
#: The most longitude divisions a mesh can have, the editor's limit.
MAX_MESH_N_LON = 512
#: The largest mesh seed, the editor's limit.
MAX_MESH_SEED = 2147483647
#: The most detail octaves a mesh can have. The relief's highest harmonic degree is 5
#: times 2 to the power of the octaves, and its coefficient array grows as the square of
#: the degree, so 8 octaves keep the largest array near 30 MB.
MAX_MESH_DETAIL_OCTAVES = 8
#: The largest side of a star's PSF fitting window, in pixels, the editor's limit.
MAX_PSF_SIZE = 23

#: A sub-observer latitude, in degrees.
Latitude = Annotated[Float, AfterValidator(FloatWithin(-90.0, 90.0))]
#: A mesh's number of latitude bands.
MeshLatitudeBands = Annotated[Int, AfterValidator(IntWithin(MIN_MESH_N_LAT, MAX_MESH_N_LAT))]
#: A mesh's number of longitude divisions.
MeshLongitudeDivisions = Annotated[Int, AfterValidator(IntWithin(MIN_MESH_N_LON, MAX_MESH_N_LON))]
#: A mesh's seed.
MeshSeed = Annotated[Int, AfterValidator(IntWithin(0, MAX_MESH_SEED))]
#: A mesh's number of detail octaves.
MeshDetailOctaves = Annotated[Int, AfterValidator(IntWithin(0, MAX_MESH_DETAIL_OCTAVES))]
#: One side of a star's PSF fitting window, in pixels.
PsfWindowSide = Annotated[Int, AfterValidator(IntWithin(1, MAX_PSF_SIZE))]


@dataclass(frozen=True, kw_only=True)
class IdealizedStar:
    """The keys of a ``stars`` entry that a real pipeline could know from its catalog.

    Parameters:
        name: The star's name.
        catalog_name: The catalog the star's record gives.
        v: The star's catalog position along v, in pixels.
        u: The star's catalog position along u, in pixels.
        vmag: The star's cataloged visual magnitude.
        spectral_class: The star's spectral class.
        move_v: The star's smear along v over the exposure, in pixels.
        move_u: The star's smear along u over the exposure, in pixels.
        psf_size: The window the navigator fits the star's PSF in, in pixels.
        navigable: Whether the navigator is told about the star.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    name: Str | None = optional(
        'idealized',
        "The star's name in the render's star records and in the navigator's catalog.",
        when_absent="``SIM-<n>``, where n is the entry's position counting from 1",
    )
    catalog_name: Str | None = optional(
        'idealized',
        "The catalog the star's record gives.",
        default=DEFAULT_CATALOG_NAME,
    )
    v: Float | None = optional(
        'idealized',
        "The star's catalog position along v, in pixels, as a pixel corner. A planted "
        'catalog error moves the drawn star off it.',
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )
    u: Float | None = optional(
        'idealized',
        "The star's catalog position along u, in pixels, as a pixel corner.",
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )
    vmag: Float | None = optional(
        'idealized',
        "The star's cataloged visual magnitude; a lower magnitude is brighter.",
        default=DEFAULT_VMAG,
        editor=EditorHint(minimum=-10.0, maximum=30.0, decimals=2),
    )
    spectral_class: Str | None = optional(
        'idealized',
        "The star's spectral class, such as ``G2``, which sets the color and temperature "
        'its catalog record gives.',
        default=DEFAULT_SPECTRAL_CLASS,
    )
    move_v: Float | None = optional(
        'idealized',
        "The star's smear along v over the exposure, in pixels: the streak a real "
        "pipeline predicts from the spacecraft's attitude.",
        default=0.0,
        editor=EditorHint(minimum=-200.0, maximum=200.0, decimals=2),
    )
    move_u: Float | None = optional(
        'idealized',
        "The star's smear along u over the exposure, in pixels.",
        default=0.0,
        editor=EditorHint(minimum=-200.0, maximum=200.0, decimals=2),
    )
    psf_size: tuple[PsfWindowSide, PsfWindowSide] | None = optional(
        'idealized',
        "The window the navigator fits the star's PSF in, in pixels along v and u.",
        when_absent='``[11, 11]``',
    )
    navigable: Bool | None = optional(
        'idealized',
        'Whether the navigator is told about the star. A star set false is drawn but '
        "left out of the navigator's view, so every star the navigator sees has it true.",
        default=True,
    )


@dataclass(frozen=True, kw_only=True)
class IdealizedBody:
    """The keys of a ``bodies`` entry that a real pipeline could know.

    These are the body's geometry, pose, lighting, and scale, which a real pipeline knows
    from SPICE and shape catalogs.  The mesh keys are idealized because the published
    shape model of an irregular body is catalog knowledge.  A body's ``nav_override``
    is this type too: it holds the values the navigator is given in place of the body's,
    and each key it leaves out keeps the body's value, so no key is required there.

    Parameters:
        name: The body's name.
        shape_model: The body's shape.
        center_v: The body center's v, in pixels.
        center_u: The body center's u, in pixels.
        semi_a_px: Semi-axis a, in pixels.
        semi_b_px: Semi-axis b, in pixels.
        semi_c_px: Semi-axis c, in pixels.
        rotation_z: An ellipsoid body's in-plane rotation, in degrees.
        sub_observer_lat_deg: The sub-observer latitude, in degrees.
        illumination_angle: The image-plane direction of the light, in degrees.
        phase_angle: The phase angle, in degrees.
        range_km: The body's distance from the observer, in km.
        km_per_pixel: The physical scale at the limb, in km per pixel.
        mesh_lumpiness: A mesh body's relief amplitude.
        mesh_n_lat: A mesh body's latitude bands.
        mesh_n_lon: A mesh body's longitude divisions.
        mesh_seed: The seed of a mesh body's shape.
        mesh_detail_octaves: The higher-degree relief bands of a mesh body.
        pose_euler_deg: A mesh body's orientation, in degrees.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    name: Str | None = optional(
        'idealized',
        "The body's name, which keys its render records and its navigator model. Two "
        'bodies may not share a name, compared without regard to case.',
        when_absent="``SIM-BODY-<n>``, where n is the entry's position counting from 1",
    )
    shape_model: ShapeModel | None = optional(
        'idealized',
        "The body's shape: ``ellipsoid`` draws a triaxial ellipsoid, and "
        '``polyhedral_mesh`` draws an irregular mesh built from the ``mesh_*`` keys and '
        'posed by ``pose_euler_deg``.',
        default=DEFAULT_SHAPE_MODEL,
    )
    center_v: Float | None = optional(
        'idealized',
        "The body center's v, in pixels, as a pixel corner.",
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )
    center_u: Float | None = optional(
        'idealized',
        "The body center's u, in pixels, as a pixel corner.",
        when_absent='the frame center',
        editor=EditorHint(minimum=-10000.0, maximum=20000.0, decimals=1),
    )
    semi_a_px: Positive | None = optional(
        'idealized',
        "Semi-axis a, in pixels: half the body's extent along v before ``rotation_z`` "
        "turns it. The body's pole lies along it.",
        when_absent="the body's value",
        editor=EditorHint(minimum=0.5, maximum=2500.0, decimals=2, step=0.5),
    )
    semi_b_px: Positive | None = optional(
        'idealized',
        "Semi-axis b, in pixels: half the body's extent along u before ``rotation_z`` turns it.",
        when_absent="the body's value",
        editor=EditorHint(minimum=0.5, maximum=2500.0, decimals=2, step=0.5),
    )
    semi_c_px: Positive | None = optional(
        'idealized',
        "Semi-axis c, in pixels: half the body's depth along the line of sight.",
        when_absent='the smaller of semi_a_px and semi_b_px',
        editor=EditorHint(minimum=0.5, maximum=2500.0, decimals=2, step=0.5),
    )
    rotation_z: Float | None = optional(
        'idealized',
        "An ellipsoid body's in-plane rotation, in degrees. A mesh body is turned by "
        '``pose_euler_deg`` instead.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1),
    )
    sub_observer_lat_deg: Latitude | None = optional(
        'idealized',
        'The latitude of the point on an ellipsoid body directly below the observer, in '
        'degrees. A positive latitude tips the north pole, the end of semi-axis a on +v '
        'before ``rotation_z`` turns it, toward the observer.',
        default=0.0,
        editor=EditorHint(minimum=-90.0, maximum=90.0, decimals=1),
    )
    illumination_angle: Float | None = optional(
        'idealized',
        'The direction the light comes from in the image plane, in degrees; 0 is from the '
        'top of the frame.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1),
    )
    phase_angle: Float | None = optional(
        'idealized',
        'The phase angle, in degrees: 0 is fully lit, and 180 is lit from behind.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=180.0, decimals=1),
    )
    range_km: Positive | None = optional(
        'idealized',
        "The body's distance from the observer, in km. A nearer body is drawn in front of "
        "a farther one, and a scene's ``spk_error`` moves each body in inverse proportion "
        'to it.',
        when_absent="the entry's position counting from 1, which orders the bodies",
        editor=EditorHint(minimum=0.001, maximum=1e9, decimals=3),
    )
    km_per_pixel: Positive | None = optional(
        'idealized',
        'The physical scale at the limb, in km per pixel. The navigator reads it, and a '
        'Titan haze takes its altitudes from it.',
        when_absent="unknown; a Titan haze takes Titan's radius over the mean semi-axis",
        editor=EditorHint(minimum=0.0001, maximum=1.0e9, decimals=4),
    )
    mesh_lumpiness: NonNeg | None = optional(
        'idealized',
        "A mesh body's relief amplitude: the relief that departs from an ellipsoid has an "
        'RMS of one eighth of it, as a fraction of the radius.',
        default=DEFAULT_MESH_LUMPINESS,
        editor=EditorHint(minimum=0.0, maximum=1.0, decimals=3, step=0.01),
    )
    mesh_n_lat: MeshLatitudeBands | None = optional(
        'idealized',
        "A mesh body's latitude bands, at least 2.",
        default=DEFAULT_MESH_N_LAT,
        editor=EditorHint(minimum=MIN_MESH_N_LAT, maximum=MAX_MESH_N_LAT),
    )
    mesh_n_lon: MeshLongitudeDivisions | None = optional(
        'idealized',
        "A mesh body's longitude divisions, at least 3.",
        default=DEFAULT_MESH_N_LON,
        editor=EditorHint(minimum=MIN_MESH_N_LON, maximum=MAX_MESH_N_LON),
    )
    mesh_seed: MeshSeed | None = optional(
        'idealized',
        "The seed that chooses a mesh body's irregular shape.",
        default=0,
        editor=EditorHint(minimum=0, maximum=MAX_MESH_SEED),
    )
    mesh_detail_octaves: MeshDetailOctaves | None = optional(
        'idealized',
        "The bands of higher harmonic degrees added to a mesh body's relief. The mesh "
        'needs more longitude divisions to show each band it adds.',
        default=0,
    )
    pose_euler_deg: tuple[Float, Float, Float] | None = optional(
        'idealized',
        "A mesh body's orientation, as intrinsic X, Y, and Z Euler angles in degrees.",
        when_absent='``[0, 0, 0]``',
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1),
    )

    def name_at(self, index: int) -> str:
        """The body's name, or the name an unnamed body takes from its position.

        Parameters:
            index: The entry's zero-based position in the scene's ``bodies`` list.

        Returns:
            ``name``, or ``SIM-BODY-<index + 1>`` when the entry leaves it out.
        """
        return default_body_name(index) if self.name is None else self.name


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
        bodies: The bodies in the frame.
        stars: The catalog stars in the frame.
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
    size_v: FrameSide = required(
        'idealized',
        "The frame's height, in pixels, at least 10.  A smaller frame is refused because the "
        'margin the lens distortion needs grows steeply as the frame shrinks.',
        editor=EditorHint(minimum=MIN_FRAME_SIZE_PX, maximum=9999),
    )
    size_u: FrameSide = required(
        'idealized',
        "The frame's width, in pixels, at least 10.  A smaller frame is refused because the "
        'margin the lens distortion needs grows steeply as the frame shrinks.',
        editor=EditorHint(minimum=MIN_FRAME_SIZE_PX, maximum=9999),
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
    bodies: tuple[IdealizedBody, ...] | None = optional(
        'idealized',
        'The bodies in the frame, each a mapping of its geometry and appearance.',
        when_absent='no bodies',
    )
    stars: tuple[IdealizedStar, ...] | None = optional(
        'idealized',
        'The catalog stars in the frame, each a mapping of its catalog entry and of the '
        'errors the render plants on it.',
        when_absent='no catalog stars',
    )

    def __post_init__(self) -> None:
        """Refuse a scene written in another format version."""
        version = self.schema_version
        if version is not None and version != CURRENT_SCHEMA_VERSION:
            raise RuleError(
                f'expected {CURRENT_SCHEMA_VERSION}, the current scene format, got {version}',
                at='schema_version',
            )
