"""The typed configuration sections the navigation models read.

Five top-level sections of the navigator configuration are parsed here:

- ``bodies`` (:class:`BodiesSection`): how a body is rendered, when it emits a blob,
  and how its label is placed.
- ``stars`` (:class:`StarsSection`): which catalogs the star model reads, how it
  reduces them, and which stars it marks as hidden by a body or a ring.
- ``titan`` (:class:`TitanSection`): the haze envelope of a body whose atmosphere hides
  its surface, the reliability its feature reports, and the haze fit's tunables.
- ``satellites``: each planet's satellites, keyed by the planet's SPICE name.
- ``feature_emission`` (:class:`FeatureEmissionSection`): when the ring model emits one
  annulus in place of the edges it would trace.

Every field without a default is a key whose value lives in the packaged YAML only.
"""

from dataclasses import dataclass
from typing import Annotated, Literal

from annotated_types import Ge, Le

from spindoctor.nav_technique.titan_fitting import ArcFitParams, SymmetryFitParams
from spindoctor.schema import (
    SCHEMA_CONFIG,
    Bool,
    Float,
    Int,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    RuleError,
    Str,
    Unit,
)

__all__ = [
    'BodiesSection',
    'FeatureEmissionSection',
    'RingAnnulusEmission',
    'RingAnnulusEmissionOverride',
    'RingAnnulusEmissionTable',
    'StarsSection',
    'TitanAnnotation',
    'TitanNavigation',
    'TitanSection',
]

#: One channel of an RGB color, from 0 through 255.
Channel = Annotated[Int, Ge(0), Le(255)]
#: An RGB color, written as ``[red, green, blue]``.
Rgb = tuple[Channel, Channel, Channel]
#: An annulus of the ring plane, written as ``[inner_km, outer_km]``.
AnnulusKm = tuple[NonNeg, NonNeg]
#: A star catalog the star model can read.
StarCatalog = Literal['ucac4', 'tycho2', 'ybsc']


def _check_annulus(annulus: AnnulusKm, *, at: tuple[str | int, ...]) -> None:
    """Refuse an annulus whose inner radius is not below its outer radius.

    Parameters:
        annulus: The annulus, ``(inner_km, outer_km)``.
        at: The annulus's key path, relative to the type that checks it.

    Raises:
        RuleError: If the inner radius is at or above the outer radius.
    """
    inner_km, outer_km = annulus
    if inner_km >= outer_km:
        raise RuleError(
            f'the inner radius ({inner_km} km) must be below the outer radius ({outer_km} km)',
            at=at,
        )


@dataclass(frozen=True)
class BodiesSection:
    """The ``bodies`` section: how a body is rendered, gated, and labeled.

    Parameters:
        min_bounding_box_area: The least area, in pixels, of a body's bounding box for
            the body to be resolved; its square root is the least resolved diameter.
        oversample_edge_limit: The rendered size, in pixels, that the oversampled body
            model may reach along each axis.
        oversample_maximum: The largest oversampling factor along each axis.
        blob_max_overflow_fraction: The largest share of a body's silhouette that may
            fall outside the sensor for the body to emit a blob.
        geometric_albedo: Each body's geometric albedo, keyed by SPICE name; the model
            of a body left out is not scaled.
        use_lambert: Whether the body model is shaded by Lambert's law.
        use_albedo: Whether the body model is scaled by ``geometric_albedo``.
        label_mask_enlarge: The distance, in pixels, around a body that a label of
            another feature avoids.
        label_limb_color: The color of the body outline in the overlay.
        label_font: The font file of a body's label.
        label_font_size: The size of a body's label, in points.
        label_font_color: The color of a body's label.
        label_horiz_gap: The gap, in pixels, between the limb and a label to its left
            or right.
        label_vert_gap: The gap, in pixels, between the limb and a label above or
            below it.
        label_scan_v: The least number of rows the label placement scans from the
            body's center.
        label_grid_v: The row spacing, in pixels, of the grid a label falls back to.
        label_grid_u: The column spacing, in pixels, of the same grid.
        outline_thicken: The half-width, in pixels, the body outline is thickened by.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    min_bounding_box_area: NonNegInt
    oversample_edge_limit: PositiveInt
    oversample_maximum: PositiveInt
    blob_max_overflow_fraction: Unit
    geometric_albedo: dict[Str, NonNeg]
    use_lambert: Bool
    use_albedo: Bool
    label_mask_enlarge: PositiveInt
    label_limb_color: Rgb
    label_font: Str
    label_font_size: PositiveInt
    label_font_color: Rgb
    label_horiz_gap: NonNegInt
    label_vert_gap: NonNegInt
    label_scan_v: PositiveInt
    label_grid_v: PositiveInt
    label_grid_u: PositiveInt
    outline_thicken: NonNegInt


@dataclass(frozen=True)
class StarsSection:
    """The ``stars`` section: the catalogs, their reduction, and the conflict rules.

    Parameters:
        catalogs: The catalogs read, most precise first.
        body_conflict_margin: The slop, in pixels, around a star when testing it
            against a body.
        default_star_class: The spectral class given to a star whose catalog entry has
            none.
        stellar_aberration: Whether star positions are corrected for stellar
            aberration.
        proper_motion: Whether star positions are moved by their proper motion.
        max_stars: The most stars the reduction keeps.
        label_font: The font file of a star's label.
        label_font_size: The size of a star's label, in points.
        label_font_color: The color of a star's label.
        label_star_color: The color of the box drawn around a star.
        duplicate_ra_dec_threshold_arcsec: The separation, in arcseconds, below which
            two catalog entries are one star.
        duplicate_vmag_threshold: The magnitude difference below which two entries
            that close are one star.
        overlapping_vmag_threshold: The magnitude difference below which two stars
            that overlap in the image are both dropped.
        max_smear: The smear length, in pixels, above which a star is dropped.
        min_predicted_snr: The effective SNR at which the ``predicted_snr`` entry of a
            star's reliability breakdown reaches 1.0; 0 uses the built-in scale.  The
            entry is reported only and changes no reliability.
        ring_occlusion_enabled: Whether a star behind an opaque ring is marked hidden.
        ring_occlusion_min_opaque_fraction: The least share of a star's conflict
            window over opaque ring for the star to be hidden.
        ring_occlusion_radii_km: Each planet's opaque annuli, keyed by SPICE name.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    catalogs: tuple[StarCatalog, ...]
    body_conflict_margin: NonNeg
    default_star_class: Str
    stellar_aberration: Bool
    proper_motion: Bool
    max_stars: PositiveInt
    label_font: Str
    label_font_size: PositiveInt
    label_font_color: Rgb
    label_star_color: Rgb
    duplicate_ra_dec_threshold_arcsec: NonNeg
    duplicate_vmag_threshold: NonNeg
    overlapping_vmag_threshold: NonNeg
    max_smear: Positive
    min_predicted_snr: NonNeg
    ring_occlusion_enabled: Bool
    ring_occlusion_min_opaque_fraction: Unit
    ring_occlusion_radii_km: dict[Str, tuple[AnnulusKm, ...]]

    def __post_init__(self) -> None:
        """Refuse an opaque annulus whose inner radius is not below its outer one."""
        for planet, annuli in self.ring_occlusion_radii_km.items():
            for index, annulus in enumerate(annuli):
                _check_annulus(annulus, at=('ring_occlusion_radii_km', planet, index))


@dataclass(frozen=True)
class TitanNavigation:
    """The ``titan.navigation`` block: the haze feature's gates and the fit's tunables.

    Parameters:
        min_envelope_diameter_px: The envelope diameter, in pixels, below which the
            feature's reliability is 0.
        max_occluded_fraction: The share of the envelope disc hidden by other bodies or
            rings above which the feature's reliability is 0.
        ring_occlusion_radii_km: The annulus of opaque ring that hides the haze,
            ``[inner_km, outer_km]``.
        axis_min_offset_px: The least distance, in pixels, between the disc center and
            the sub-solar point for the symmetry axis to be defined.
        backplane_max_samples: The most samples the symmetry axis's backplane is
            evaluated over; a larger box is strided.  Zero or below sets no bound.
        recenter_threshold_px: The offset, in pixels, above which the fit is run again
            about the moved center.
        star_mask_vmag_limit: The magnitude below which a star is masked out of the
            fit.
        star_mask_radius_px: The radius, in pixels, of a masked star.
        reliability_diameter_midpoint_px: The envelope diameter, in pixels, at which
            the reliability sigmoid is half way.
        reliability_diameter_scale_px: The sigmoid's width, in pixels.
        surface_window_filters: The filters that see through the haze to the surface.
        high_phase_deg: The phase angle, in degrees, at or above which the feature is
            marked high phase.
        symmetry: The cross-track symmetry fit's tunables.
        arc: The along-track limb arc fit's tunables.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    min_envelope_diameter_px: NonNeg
    max_occluded_fraction: Unit
    ring_occlusion_radii_km: AnnulusKm
    axis_min_offset_px: NonNeg
    backplane_max_samples: Int
    recenter_threshold_px: NonNeg
    star_mask_vmag_limit: Float
    star_mask_radius_px: NonNeg
    reliability_diameter_midpoint_px: NonNeg
    reliability_diameter_scale_px: Positive
    surface_window_filters: tuple[Str, ...]
    high_phase_deg: Annotated[NonNeg, Le(180)]
    symmetry: SymmetryFitParams
    arc: ArcFitParams

    def __post_init__(self) -> None:
        """Refuse an opaque annulus whose inner radius is not below its outer one."""
        _check_annulus(self.ring_occlusion_radii_km, at=('ring_occlusion_radii_km',))


@dataclass(frozen=True)
class TitanAnnotation:
    """The ``titan.annotation`` block: how the haze feature is drawn in the overlay.

    Parameters:
        gated_dot_spacing: The sample spacing that dots the curves of a feature below
            its reliability gate.
        center_marker_half_px: The half-length, in pixels, of the cross at the disc
            center.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    gated_dot_spacing: PositiveInt
    center_marker_half_px: Positive


@dataclass(frozen=True)
class TitanSection:
    """The ``titan`` section: the haze envelope and how it is navigated and drawn.

    Parameters:
        atmosphere_height: The height, in km, of the haze envelope above the solid
            radius.
        navigation: The feature's gates and the fit's tunables.
        annotation: How the feature is drawn.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    atmosphere_height: Positive
    navigation: TitanNavigation
    annotation: TitanAnnotation


@dataclass(frozen=True)
class RingAnnulusEmission:
    """When the ring model emits one annulus in place of the edges it would trace.

    Parameters:
        max_radial_px: The radial extent, in pixels, below which an edge is too narrow
            to trace and the annulus is emitted.
        kmpp_threshold: The radial resolution, in km per pixel, at or above which the
            whole ring system is emitted as the annulus.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    max_radial_px: Positive
    kmpp_threshold: Positive


@dataclass(frozen=True)
class RingAnnulusEmissionOverride:
    """One planet's settings in place of the default; a key left out keeps the default.

    Parameters:
        max_radial_px: The planet's ``max_radial_px``, or None for the default's.
        kmpp_threshold: The planet's ``kmpp_threshold``, or None for the default's.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    max_radial_px: Positive | None = None
    kmpp_threshold: Positive | None = None


@dataclass(frozen=True)
class RingAnnulusEmissionTable:
    """The ``feature_emission.ring_annulus`` block: the default and each planet's.

    Parameters:
        default: The settings of a planet with no entry, and of each key an entry
            leaves out.
        planets: Each planet's settings, keyed by SPICE name.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    default: RingAnnulusEmission
    planets: dict[Str, RingAnnulusEmissionOverride]

    def for_planet(self, planet: str) -> RingAnnulusEmission:
        """Return the settings that apply to one planet.

        Parameters:
            planet: The planet's SPICE name, such as ``'SATURN'``.

        Returns:
            The planet's entry, with each key it leaves out taken from the default.
        """
        entry = self.planets.get(planet)
        if entry is None:
            return self.default
        return RingAnnulusEmission(
            max_radial_px=(
                self.default.max_radial_px if entry.max_radial_px is None else entry.max_radial_px
            ),
            kmpp_threshold=(
                self.default.kmpp_threshold
                if entry.kmpp_threshold is None
                else entry.kmpp_threshold
            ),
        )


@dataclass(frozen=True)
class FeatureEmissionSection:
    """The ``feature_emission`` section: which features the ring model emits.

    Parameters:
        ring_annulus: When the ring model emits one annulus in place of edges.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    ring_annulus: RingAnnulusEmissionTable
