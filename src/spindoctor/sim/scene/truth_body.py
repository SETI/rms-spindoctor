"""A scene's ``bodies`` entry as the renderer reads it: :class:`Body` and its blocks.

:class:`Body` extends :class:`~spindoctor.sim.scene.idealized.IdealizedBody`, the keys
a real pipeline could know, with the keys only the test knows: the body's craters,
relief, scattering law, textures, transits, haze
(:mod:`spindoctor.sim.scene.truth_atmosphere`), the rendered mesh's shading and pose
scatter, the limb supersampling, and the ``nav_override`` that gives the navigator other
values for the idealized keys.  The navigator's predicted body is the smooth template
the idealized keys describe, so every truth key is a planted difference between the
frame and the prediction.

The ellipsoid renderer reads the crater, scattering, texture, and transit keys, and a
mesh body refuses them (:data:`~spindoctor.sim.scene.fields.MESH_UNSUPPORTED_KEYS`).
The mesh renderer reads ``shading`` and ``pose_scatter``, and an ellipsoid body refuses
them (:data:`~spindoctor.sim.scene.fields.ELLIPSOID_UNSUPPORTED_KEYS`).  Both read the
relief keys.

The constants here are the defaults and limits the renderer and the scene editor read
too, so each is written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, get_args

from pydantic import AfterValidator

from spindoctor.schema import SCHEMA_CONFIG, Float, NonNeg, Positive, RuleError, Unit
from spindoctor.sim.scene.declare import EditorHint, FloatWithin, optional, required
from spindoctor.sim.scene.fields import (
    ELLIPSOID_UNSUPPORTED_KEYS,
    MESH_UNSUPPORTED_KEYS,
    ellipsoid_unsupported_message,
    mesh_unsupported_message,
)
from spindoctor.sim.scene.idealized import IdealizedBody
from spindoctor.sim.scene.truth_atmosphere import BodyAtmosphere

__all__ = [
    'DEFAULT_ALBEDO_CORR_PX',
    'DEFAULT_ANTI_ALIASING',
    'DEFAULT_BAND_WAVENUMBER',
    'DEFAULT_CRATER_MAX_RADIUS',
    'DEFAULT_CRATER_MIN_RADIUS',
    'DEFAULT_CRATER_POWER_LAW_EXPONENT',
    'DEFAULT_CRATER_RELIEF_SCALE',
    'DEFAULT_LIMB_RELIEF_CORR_DEG',
    'DEFAULT_MINNAERT_K',
    'DEFAULT_MOON_ALBEDO_FACTOR',
    'DEFAULT_PHOTOMETRIC_LAW',
    'DEFAULT_SHADING',
    'DEFAULT_SHADOW_DARKNESS',
    'DEFAULT_SPOT_ALBEDO_FACTOR',
    'DEFAULT_SPOT_LON_DEG',
    'DEFAULT_SPOT_RADIUS_DEG',
    'DEFAULT_SURGE_WIDTH_DEG',
    'DEFAULT_TRANSIT_RADIUS_PX',
    'MAX_LIMB_RELIEF_RMS',
    'MIN_LIMB_RELIEF_CORR_DEG',
    'PHOTOMETRIC_LAWS',
    'Body',
    'BodyAlbedoTexture',
    'BodyDiscTexture',
    'BodyOppositionSurge',
    'BodyPoseScatter',
    'BodySurfaceSpot',
    'BodyTransit',
    'BodyTransitMoon',
    'BodyTransitShadow',
    'MeshShading',
    'PhotometricLaw',
]

#: The surface scattering laws an ellipsoid body can take.
PhotometricLaw = Literal['lambert', 'lommel_seeliger', 'minnaert', 'lunar_lambert']
#: Every surface scattering law.
PHOTOMETRIC_LAWS: frozenset[str] = frozenset(get_args(PhotometricLaw))
#: How a mesh body is shaded: per face, or smoothly between vertices.
MeshShading = Literal['flat', 'gouraud']

#: The smallest crater radius a body draws when it leaves the key out, as a fraction of
#: semi-axis a.
DEFAULT_CRATER_MIN_RADIUS = 0.05
#: The largest crater radius a body draws when it leaves the key out, as a fraction of
#: semi-axis a.
DEFAULT_CRATER_MAX_RADIUS = 0.25
#: The exponent of the crater radii's power law when a body leaves it out.
DEFAULT_CRATER_POWER_LAW_EXPONENT = 3.0
#: The crater depth scale when a body leaves it out.
DEFAULT_CRATER_RELIEF_SCALE = 0.6
#: The limb supersampling of a body that leaves ``anti_aliasing`` out: the finest.
DEFAULT_ANTI_ALIASING = 1.0
#: The relief correlation length, in degrees of arc, when a body leaves it out.
DEFAULT_LIMB_RELIEF_CORR_DEG = 15.0
#: The scattering law of a body that leaves ``photometric_law`` out.
DEFAULT_PHOTOMETRIC_LAW: PhotometricLaw = 'lambert'
#: The Minnaert exponent when a body leaves ``minnaert_k`` out.
DEFAULT_MINNAERT_K = 0.5
#: The opposition surge's width, in degrees of phase, when the block leaves it out.
DEFAULT_SURGE_WIDTH_DEG = 6.0
#: The shading of a mesh body that leaves ``shading`` out.
DEFAULT_SHADING: MeshShading = 'flat'
#: The albedo noise field's correlation length, in detector pixels on the disc, when
#: the block leaves ``corr_px`` out.
DEFAULT_ALBEDO_CORR_PX = 20.0
#: The longitude, in degrees, of a spot or storm that leaves ``lon_deg`` out: the
#: sub-observer meridian of a body seen equator-on.
DEFAULT_SPOT_LON_DEG = 90.0
#: The angular radius, in degrees, of a spot or storm that leaves ``radius_deg`` out.
DEFAULT_SPOT_RADIUS_DEG = 10.0
#: The brightness factor of a spot or storm that leaves ``albedo_factor`` out.
DEFAULT_SPOT_ALBEDO_FACTOR = 0.7
#: The band pattern's cosine cycles per radian of latitude when the block leaves
#: ``band_wavenumber`` out.
DEFAULT_BAND_WAVENUMBER = 8.0
#: The radius, in detector pixels, of a transiting moon or shadow that leaves
#: ``radius_px`` out.
DEFAULT_TRANSIT_RADIUS_PX = 5.0
#: The brightness factor of a transiting moon that leaves ``albedo_factor`` out.
DEFAULT_MOON_ALBEDO_FACTOR = 1.0
#: The darkness of a transit shadow that leaves ``darkness`` out.
DEFAULT_SHADOW_DARKNESS = 0.8

# The largest limb-relief RMS (a fraction of the body radius) a scene may
# command, a limit of the scene format.  At 0.1 the surface stays about half the
# radius out from the center: over 16 seeds at each of 1, 2, 5, 15, and 30 degrees
# of correlation length, the deepest point lies 0.5004 of the radius below the
# smooth surface.  At 0.2 it reaches the center (h_min -1.0 over 8 seeds).  The
# ellipsoid renderer reads the terrain where each line of sight meets the relief
# surface; at 0.1 and 2 to 30 degrees, a brute-force search puts its point more
# than half a correlation length off on at most 0.025% of the disc.  Mesh bodies
# are held to the same bound.
MAX_LIMB_RELIEF_RMS = 0.1
# The shortest relief correlation length (degrees of arc) a scene may command.
# It bounds cost.  Synthesis costs about (1 / corr)**3: 0.5 s and 0.2 GB at
# 1 degree, 3.1 s and 0.5 GB at 0.5 degrees, and 26 s and 1.7 GB at
# 0.25 degrees.  A realization is cached by (corr, seed) and rescaled to the
# RMS, so a render, or the scene editor, pays that cost again only when the
# correlation length or the relief seed changes.  The outline envelope costs
# about 2 s more at 1 degree (0.5 s at 2 degrees) and is cached by (corr, seed,
# RMS).
MIN_LIMB_RELIEF_CORR_DEG = 1.0

#: A limb-relief RMS, a fraction of the body radius.
LimbReliefRms = Annotated[Float, AfterValidator(FloatWithin(0.0, MAX_LIMB_RELIEF_RMS))]
#: A limb-relief correlation length, in degrees of arc.
LimbReliefCorrDeg = Annotated[Float, AfterValidator(FloatWithin(MIN_LIMB_RELIEF_CORR_DEG))]
#: The exponent of the crater radii's power law, which normalizes only above 1.
CraterExponent = Annotated[Float, AfterValidator(FloatWithin(1.0, low_open=True))]


@dataclass(frozen=True, kw_only=True)
class BodyOppositionSurge:
    """The brightening of an ellipsoid body near zero phase.

    The surface brightness is multiplied by ``(1 + amplitude * exp(-phase / width))
    / (1 + amplitude)``: 1 at zero phase, and ``1 / (1 + amplitude)`` far from it.

    Parameters:
        amplitude: The surge's amplitude.
        width_deg: The surge's angular width, in degrees of phase.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    amplitude: NonNeg | None = optional(
        'truth',
        'The surge amplitude: the surface far from zero phase is dimmed by '
        '``1 / (1 + amplitude)`` relative to zero phase. Zero turns the surge off.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=10.0, decimals=3, step=0.05),
    )
    width_deg: Positive | None = optional(
        'truth',
        "The surge's angular e-folding width, in degrees of phase.",
        default=DEFAULT_SURGE_WIDTH_DEG,
        editor=EditorHint(minimum=0.1, maximum=90.0, decimals=2, step=0.5),
    )


@dataclass(frozen=True, kw_only=True)
class BodyPoseScatter:
    """A per-frame error in a mesh body's pose, which the navigator cannot model.

    Parameters:
        sigma_deg: The sigma of the scatter on each Euler angle, in degrees.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    sigma_deg: NonNeg | None = optional(
        'truth',
        "The sigma, in degrees, of a seeded Gaussian added to each of the rendered mesh's "
        'three Euler angles. The navigator predicts the catalog pose. Zero turns the '
        'scatter off.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=45.0, decimals=3, step=0.5),
    )


@dataclass(frozen=True, kw_only=True)
class BodySurfaceSpot:
    """One circular albedo spot or storm on an ellipsoid body's surface.

    The spot lies in the body-polar frame: the pole is the direction of semi-axis a, and
    longitude runs from the +b direction toward +c.

    Parameters:
        lat_deg: The spot center's latitude, in degrees.
        lon_deg: The spot center's longitude, in degrees.
        radius_deg: The spot's angular radius, in degrees.
        albedo_factor: The factor the spot multiplies the surface brightness by.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    lat_deg: Float | None = optional(
        'truth',
        "The spot center's latitude, in degrees, with the pole along semi-axis a.",
        default=0.0,
        editor=EditorHint(minimum=-90.0, maximum=90.0, decimals=1, step=1.0),
    )
    lon_deg: Float | None = optional(
        'truth',
        "The spot center's longitude, in degrees from the +b direction toward +c. For a "
        'body seen equator-on, 90 is the meridian facing the observer.',
        default=DEFAULT_SPOT_LON_DEG,
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1, step=1.0),
    )
    radius_deg: Positive | None = optional(
        'truth',
        "The spot's angular radius, in degrees.",
        default=DEFAULT_SPOT_RADIUS_DEG,
        editor=EditorHint(minimum=0.1, maximum=180.0, decimals=1, step=1.0),
    )
    albedo_factor: NonNeg | None = optional(
        'truth',
        'The factor the spot multiplies the surface brightness by.',
        default=DEFAULT_SPOT_ALBEDO_FACTOR,
        editor=EditorHint(minimum=0.0, maximum=5.0, decimals=3, step=0.05),
    )


@dataclass(frozen=True, kw_only=True)
class BodyAlbedoTexture:
    """An ellipsoid body's albedo texture: a noise field and circular spots.

    Both multiply the body's shading and leave its outline alone.

    Parameters:
        rms: The standard deviation of the multiplicative noise field.
        corr_px: The noise field's correlation length, in detector pixels.
        spots: The circular albedo spots.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    rms: NonNeg | None = optional(
        'truth',
        'The standard deviation of the band-limited multiplicative noise field. Zero '
        'turns the field off, and the spots still apply.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=2.0, decimals=4, step=0.01),
    )
    corr_px: Positive | None = optional(
        'truth',
        "The noise field's correlation length, in detector pixels on the disc, exact "
        'where the surface faces the observer and foreshortened toward the limb.',
        default=DEFAULT_ALBEDO_CORR_PX,
        editor=EditorHint(minimum=0.01, maximum=1000.0, decimals=2, step=1.0),
    )
    spots: tuple[BodySurfaceSpot, ...] | None = optional(
        'truth',
        'Circular albedo spots on the surface.',
        when_absent='no spots',
    )


@dataclass(frozen=True, kw_only=True)
class BodyDiscTexture:
    """A giant planet's disc texture: latitude bands and storm ovals.

    The bands multiply the shading by ``1 + band_amplitude * cos(band_wavenumber *
    latitude + band_phase_deg)``, with the body-polar latitude in radians.

    Parameters:
        band_amplitude: The bands' contrast.
        band_wavenumber: The bands' cosine cycles per radian of latitude.
        band_phase_deg: The bands' phase, in degrees.
        storms: The storm ovals.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    band_amplitude: NonNeg | None = optional(
        'truth',
        'The contrast of the latitude bands. Zero turns the bands off, and the storms still apply.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=2.0, decimals=4, step=0.01),
    )
    band_wavenumber: NonNeg | None = optional(
        'truth',
        "The bands' cosine cycles per radian of body-polar latitude; a Saturn-like disc "
        'takes about 6 to 10.',
        default=DEFAULT_BAND_WAVENUMBER,
        editor=EditorHint(minimum=0.0, maximum=64.0, decimals=2, step=0.5),
    )
    band_phase_deg: Float | None = optional(
        'truth',
        "The bands' phase, in degrees, which slides the pattern in latitude.",
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=360.0, decimals=1, step=5.0),
    )
    storms: tuple[BodySurfaceSpot, ...] | None = optional(
        'truth',
        'Storm ovals on the disc.',
        when_absent='no storms',
    )


@dataclass(frozen=True, kw_only=True)
class BodyTransitMoon:
    """A moon's disc drawn in front of the body.

    Parameters:
        dv_px: The disc center's offset from the body center along v, in pixels.
        du_px: The disc center's offset from the body center along u, in pixels.
        radius_px: The disc's radius, in pixels.
        albedo_factor: The disc's brightness relative to the body's smooth shading.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    dv_px: Float | None = optional(
        'truth',
        "The disc center's offset from the body center along v, in detector pixels. A "
        'planted roll turns it with the body.',
        default=0.0,
        editor=EditorHint(minimum=-5000.0, maximum=5000.0, decimals=2, step=1.0),
    )
    du_px: Float | None = optional(
        'truth',
        "The disc center's offset from the body center along u, in detector pixels.",
        default=0.0,
        editor=EditorHint(minimum=-5000.0, maximum=5000.0, decimals=2, step=1.0),
    )
    radius_px: Positive | None = optional(
        'truth',
        "The disc's radius, in detector pixels. It shows only where it overlaps the body.",
        default=DEFAULT_TRANSIT_RADIUS_PX,
        editor=EditorHint(minimum=0.01, maximum=5000.0, decimals=2, step=1.0),
    )
    albedo_factor: NonNeg | None = optional(
        'truth',
        "The disc's brightness as a factor on the body's untextured shading beneath it.",
        default=DEFAULT_MOON_ALBEDO_FACTOR,
        editor=EditorHint(minimum=0.0, maximum=5.0, decimals=3, step=0.05),
    )


@dataclass(frozen=True, kw_only=True)
class BodyTransitShadow:
    """A moon's shadow cast on the body.

    Parameters:
        dv_px: The shadow center's offset from the body center along v, in pixels.
        du_px: The shadow center's offset from the body center along u, in pixels.
        radius_px: The shadow's radius, in pixels.
        darkness: The fraction of the surface brightness the shadow takes away.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    dv_px: Float | None = optional(
        'truth',
        "The shadow center's offset from the body center along v, in detector pixels. A "
        'planted roll turns it with the body.',
        default=0.0,
        editor=EditorHint(minimum=-5000.0, maximum=5000.0, decimals=2, step=1.0),
    )
    du_px: Float | None = optional(
        'truth',
        "The shadow center's offset from the body center along u, in detector pixels.",
        default=0.0,
        editor=EditorHint(minimum=-5000.0, maximum=5000.0, decimals=2, step=1.0),
    )
    radius_px: Positive | None = optional(
        'truth',
        "The shadow's radius, in detector pixels.",
        default=DEFAULT_TRANSIT_RADIUS_PX,
        editor=EditorHint(minimum=0.01, maximum=5000.0, decimals=2, step=1.0),
    )
    darkness: Unit | None = optional(
        'truth',
        'The fraction of the surface brightness the shadow takes away; 1 is a black umbra.',
        default=DEFAULT_SHADOW_DARKNESS,
    )


@dataclass(frozen=True, kw_only=True)
class BodyTransit:
    """One transit on an ellipsoid body: a moon's disc, its shadow, or both.

    Both are drawn on the body's disc.  The shadow is the sharp circular mark that disc
    correlation and blob techniques can lock onto.

    Parameters:
        moon: The moon's disc.
        shadow: The moon's shadow.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    moon: BodyTransitMoon | None = optional(
        'truth',
        "A moon's disc drawn in front of the body.",
        when_absent='no moon disc',
    )
    shadow: BodyTransitShadow | None = optional(
        'truth',
        "A moon's shadow cast on the body.",
        when_absent='no shadow',
    )

    def __post_init__(self) -> None:
        """Refuse a transit that has neither a moon nor a shadow."""
        if self.moon is None and self.shadow is None:
            raise RuleError('a transit needs a moon, a shadow, or both', at=())


@dataclass(frozen=True, kw_only=True)
class Body(IdealizedBody):
    """A whole ``bodies`` entry: the idealized keys and the keys only the test knows.

    Parameters:
        semi_a_px: Semi-axis a, in pixels.
        semi_b_px: Semi-axis b, in pixels.
        crater_fill: The fraction of the surface the craters cover.
        crater_min_radius: The smallest crater radius, a fraction of semi-axis a.
        crater_max_radius: The largest crater radius, a fraction of semi-axis a.
        crater_power_law_exponent: The exponent of the crater radii's power law.
        crater_relief_scale: The crater depth scale.
        anti_aliasing: The limb supersampling.
        limb_relief_rms: The relief's RMS, a fraction of the radius.
        limb_relief_corr_deg: The relief's correlation length, in degrees.
        photometric_law: The surface scattering law.
        minnaert_k: The Minnaert exponent.
        opposition_surge: The brightening near zero phase.
        shading: A mesh body's shading.
        pose_scatter: A mesh body's per-frame pose error.
        albedo_texture: The albedo noise field and spots.
        disc_texture: The latitude bands and storms.
        transits: The moons and shadows drawn on the disc.
        atmosphere: The haze layer above the surface.
        nav_override: The values the navigator is given in place of the body's.
    """

    semi_a_px: Positive = required(
        'idealized',
        "Semi-axis a, in pixels: half the body's extent along v before ``rotation_z`` "
        "turns it. The body's pole lies along it.",
        editor=EditorHint(minimum=0.5, maximum=2500.0, decimals=2, step=0.5),
    )
    semi_b_px: Positive = required(
        'idealized',
        "Semi-axis b, in pixels: half the body's extent along u before ``rotation_z`` turns it.",
        editor=EditorHint(minimum=0.5, maximum=2500.0, decimals=2, step=0.5),
    )
    crater_fill: NonNeg | None = optional(
        'truth',
        "Roughly the fraction of an ellipsoid body's surface the craters cover; 0 draws none.",
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=10.0, decimals=3, step=0.01),
    )
    crater_min_radius: Positive | None = optional(
        'truth',
        'The smallest crater radius, as a fraction of semi-axis a. It is less than '
        '``crater_max_radius``, and an omitted bound takes its default in that '
        'comparison.',
        default=DEFAULT_CRATER_MIN_RADIUS,
        editor=EditorHint(minimum=0.01, maximum=0.25, decimals=3, step=0.005),
    )
    crater_max_radius: Positive | None = optional(
        'truth',
        'The largest crater radius, as a fraction of semi-axis a.',
        default=DEFAULT_CRATER_MAX_RADIUS,
        editor=EditorHint(minimum=0.01, maximum=0.25, decimals=3, step=0.005),
    )
    crater_power_law_exponent: CraterExponent | None = optional(
        'truth',
        'The exponent alpha of the crater radii distribution, ``p(R) ~ R**-alpha``, '
        'above 1, where the distribution can be normalized.',
        default=DEFAULT_CRATER_POWER_LAW_EXPONENT,
        editor=EditorHint(minimum=1.1, maximum=5.0, decimals=2, step=0.05),
    )
    crater_relief_scale: Float | None = optional(
        'truth',
        "The craters' depth scale.",
        default=DEFAULT_CRATER_RELIEF_SCALE,
        editor=EditorHint(minimum=0.0, maximum=3.0, decimals=3, step=0.01),
    )
    anti_aliasing: Unit | None = optional(
        'truth',
        "How finely the body's limb is supersampled when the render does not "
        'oversample: up to 4 times along each axis at 1, and not at all at 0. The '
        "navigator's template is always drawn at full anti-aliasing.",
        default=DEFAULT_ANTI_ALIASING,
        editor=EditorHint(minimum=0.0, maximum=1.0, decimals=3, step=0.01),
    )
    limb_relief_rms: LimbReliefRms | None = optional(
        'truth',
        "The RMS of the body's relief along the limb, as a fraction of the radius, at "
        'most 0.1, where the deepest point lies about half the radius below the smooth '
        'surface. An ellipsoid shows the relief in its outline, its slope shading, and '
        'its cast shadows, and a mesh in its vertices. Zero turns the relief off.',
        default=0.0,
        editor=EditorHint(minimum=0.0, maximum=MAX_LIMB_RELIEF_RMS, decimals=4, step=0.005),
    )
    limb_relief_corr_deg: LimbReliefCorrDeg | None = optional(
        'truth',
        "The relief's correlation length, in degrees of arc on the surface, at least 1. "
        'The cost of drawing the relief grows as the inverse cube of it.',
        default=DEFAULT_LIMB_RELIEF_CORR_DEG,
        editor=EditorHint(minimum=MIN_LIMB_RELIEF_CORR_DEG, maximum=180.0, decimals=1, step=1.0),
    )
    photometric_law: PhotometricLaw | None = optional(
        'truth',
        "An ellipsoid body's surface scattering law. The navigator's template is always "
        'shaded with the Lambert law.',
        default=DEFAULT_PHOTOMETRIC_LAW,
    )
    minnaert_k: Positive | None = optional(
        'truth',
        'The Minnaert exponent, which only the ``minnaert`` law reads.',
        default=DEFAULT_MINNAERT_K,
        editor=EditorHint(minimum=0.001, maximum=2.0, decimals=3, step=0.05),
    )
    opposition_surge: BodyOppositionSurge | None = optional(
        'truth',
        "The brightening of an ellipsoid body's surface near zero phase.",
        when_absent='no surge',
    )
    shading: MeshShading | None = optional(
        'truth',
        'How a mesh body is shaded: ``flat`` per face, or ``gouraud``, smoothly between '
        "vertices. The navigator's predicted mesh is always shaded flat.",
        default=DEFAULT_SHADING,
    )
    pose_scatter: BodyPoseScatter | None = optional(
        'truth',
        "A per-frame error added to a mesh body's rendered pose.",
        when_absent='no scatter',
    )
    albedo_texture: BodyAlbedoTexture | None = optional(
        'truth',
        "An ellipsoid body's albedo texture: a multiplicative noise field and circular spots.",
        when_absent='no albedo texture',
    )
    disc_texture: BodyDiscTexture | None = optional(
        'truth',
        "An ellipsoid body's latitude bands and storm ovals, as on a giant planet.",
        when_absent='no disc texture',
    )
    transits: tuple[BodyTransit, ...] | None = optional(
        'truth',
        "Moons and their shadows drawn on an ellipsoid body's disc.",
        when_absent='no transits',
    )
    atmosphere: BodyAtmosphere | None = optional(
        'truth',
        "An ellipsoid body's haze layer, which softens its limb and adds a halo above it.",
        when_absent='a hard limb',
    )
    nav_override: IdealizedBody | None = optional(
        'truth',
        "The values the navigator is given in place of the body's own, for idealized "
        'keys only. The renderer ignores it and draws the true geometry, and each key it '
        "leaves out keeps the body's value.",
        when_absent="the body's own values",
    )

    def __post_init__(self) -> None:
        """Refuse keys the body's shape ignores and an empty crater band.

        Raises:
            RuleError: If the body sets a key its shape model does not draw, or the
                crater radius band is empty.
        """
        if self.shape_model == 'polyhedral_mesh':
            refused = [key for key in MESH_UNSUPPORTED_KEYS if getattr(self, key) is not None]
            if refused:
                raise RuleError(mesh_unsupported_message(refused), at=(), reads=refused)
        else:
            refused = [key for key in ELLIPSOID_UNSUPPORTED_KEYS if getattr(self, key) is not None]
            if refused:
                raise RuleError(ellipsoid_unsupported_message(refused), at=(), reads=refused)
        if self.crater_min_radius is None and self.crater_max_radius is None:
            return
        low = (
            DEFAULT_CRATER_MIN_RADIUS if self.crater_min_radius is None else self.crater_min_radius
        )
        high = (
            DEFAULT_CRATER_MAX_RADIUS if self.crater_max_radius is None else self.crater_max_radius
        )
        if low >= high:
            raise RuleError(
                f'crater_min_radius must be less than crater_max_radius; got {low!r} >= '
                f'{high!r} (an omitted bound takes its default, {DEFAULT_CRATER_MIN_RADIUS} '
                f'or {DEFAULT_CRATER_MAX_RADIUS})',
                at=(),
                reads=('crater_min_radius', 'crater_max_radius'),
            )

    def semi_axes_px(self) -> tuple[float, float, float]:
        """The body's three semi-axes, in pixels, with semi-axis c at its default.

        Returns:
            Semi-axes a, b, and c.  Semi-axis c defaults to the smaller of a and b.
        """
        semi_c_px = self.semi_c_px
        if semi_c_px is None:
            semi_c_px = min(self.semi_a_px, self.semi_b_px)
        return self.semi_a_px, self.semi_b_px, semi_c_px
