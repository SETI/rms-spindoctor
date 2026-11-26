"""Sum-type flag dataclasses carried on NavFeature.flags.

Each feature type has a small dataclass listing the technique-specific
boolean / scalar flags relevant to that type.  Carrying them as a typed sum
type instead of a free-form ``dict[str, Any]`` lets static type checkers see
which fields exist for each type and lets curator code copy known fields by
attribute.
"""

from dataclasses import dataclass

__all__ = [
    'BodyBlobFlags',
    'BodyDiscFlags',
    'CartographicModelFlags',
    'LimbArcFlags',
    'NavFeatureFlags',
    'RingAnnulusFlags',
    'RingEdgeFlags',
    'StarFlags',
    'TerminatorArcFlags',
    'TitanHazeFlags',
]


@dataclass(frozen=True)
class StarFlags:
    """Flags carried on a STAR feature.

    Parameters:
        saturated: True if the detected star peak hit the camera's full-well
            DN; centroiding switches from peak-Gaussian-fit to annular
            brightness-weighted moment.
        smear_length_px: Expected smear length in pixels at this image's
            spacecraft attitude rate.  Must be ``>= 0``.
        in_body_silhouette: True if the predicted star position falls inside
            a predicted body silhouette in extfov.
        in_saturation_or_cosmic_mask: True if the predicted star position
            falls inside a saturation or cosmic-ray mask pixel.
        predicted_snr: Magnitude-margin-derived effective SNR for the
            catalog star (``SNR_REF * 2.512 ** (mag_limit - vmag)``), not a
            DN-based photometric SNR.  Monotone in catalog brightness, so
            ``StarUniqueMatchNav`` can still rank stars by it when picking
            the unique-bright pair.  ``0.0`` for fixtures or features whose
            model did not populate it.  Must be ``>= 0``.
        vmag: Catalog V-band magnitude of the star, or ``None`` when the
            catalog entry has no magnitude.  Used by ``StarUniqueMatchNav``
            to compute the magnitude margin to the next-brightest star.
        photometry_corrected: True when this star's catalog magnitude was
            replaced by a trusted YBSC or Tycho-2 reference value because the
            source catalog's bright-end aperture photometry was saturated, so
            its ``vmag`` (and therefore ``predicted_snr``) is trustworthy.
        photometry_saturated: True when the catalog magnitude is a saturated
            bright-end reading with no YBSC or Tycho-2 reference to correct
            it, so ``vmag`` is an untrusted lower bound on brightness (the
            star is at least this bright, likely brighter).  The detectability
            model keeps such a star rather than rejecting it on its too-faint
            reading, and a wide-offset brightness anchor is trusted only when
            this is False.
    """

    saturated: bool = False
    smear_length_px: float = 0.0
    in_body_silhouette: bool = False
    in_saturation_or_cosmic_mask: bool = False
    predicted_snr: float = 0.0
    vmag: float | None = None
    photometry_corrected: bool = False
    photometry_saturated: bool = False

    def __post_init__(self) -> None:
        """Validate ``smear_length_px`` and ``predicted_snr`` are non-negative."""
        if self.smear_length_px < 0.0:
            raise ValueError(f'smear_length_px must be >= 0; got {self.smear_length_px!r}')
        if self.predicted_snr < 0.0:
            raise ValueError(f'predicted_snr must be >= 0; got {self.predicted_snr!r}')


@dataclass(frozen=True)
class LimbArcFlags:
    """Flags carried on a LIMB_ARC feature.

    Parameters:
        body_name: SPICE body name whose limb this arc traces.
        visible_arc_fraction: Fraction of total limb length inside extfov
            and not occluded ``[0, 1]``.
    """

    body_name: str = ''
    visible_arc_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate ``visible_arc_fraction`` is in ``[0, 1]``."""
        if not 0.0 <= self.visible_arc_fraction <= 1.0:
            raise ValueError(
                f'visible_arc_fraction must lie in [0, 1]; got {self.visible_arc_fraction!r}'
            )


@dataclass(frozen=True)
class TerminatorArcFlags:
    """Flags carried on a TERMINATOR_ARC feature.

    Parameters:
        body_name: SPICE body name whose terminator this arc traces.
        visible_arc_fraction: Fraction of total terminator length inside
            extfov and lit ``[0, 1]``.
        phase_angle_factor: ``sin(phase_angle)`` factor used in reliability;
            peaks at 90-degree crescent.  Must lie in ``[0, 1]``.
    """

    body_name: str = ''
    visible_arc_fraction: float = 0.0
    phase_angle_factor: float = 0.0

    def __post_init__(self) -> None:
        """Validate fractions and the phase-angle factor."""
        if not 0.0 <= self.visible_arc_fraction <= 1.0:
            raise ValueError(
                f'visible_arc_fraction must lie in [0, 1]; got {self.visible_arc_fraction!r}'
            )
        if not 0.0 <= self.phase_angle_factor <= 1.0:
            raise ValueError(
                f'phase_angle_factor must lie in [0, 1]; got {self.phase_angle_factor!r}'
            )


@dataclass(frozen=True)
class RingEdgeFlags:
    """Flags carried on a RING_EDGE feature.

    Parameters:
        is_straight_line: True when the polyline's deviation from a
            straight-line fit is below threshold.  Triggers rank-1
            covariance handling at the technique level.
        polarity_predictable: True only when the per-edge static-catalog
            entry guarantees the gradient direction across the edge in
            this scene.  Default is False.
        edge_name: Name of the ring edge in the static catalog.
        planet_name: Planet whose rings this edge belongs to.
    """

    is_straight_line: bool = False
    polarity_predictable: bool = False
    edge_name: str = ''
    planet_name: str = ''


@dataclass(frozen=True)
class BodyDiscFlags:
    """Flags carried on a BODY_DISC feature.

    Parameters:
        body_name: SPICE body name whose disc this feature renders.
        overflow_fov_fraction: Fraction of the disc area outside the sensor
            ``[0, 1]``.  Same value as ``BodyDiscGeometry.overflow_fraction``;
            duplicated here for type-specific access.
    """

    body_name: str = ''
    overflow_fov_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate ``overflow_fov_fraction`` is in ``[0, 1]``."""
        if not 0.0 <= self.overflow_fov_fraction <= 1.0:
            raise ValueError(
                f'overflow_fov_fraction must lie in [0, 1]; got {self.overflow_fov_fraction!r}'
            )


@dataclass(frozen=True)
class BodyBlobFlags:
    """Flags carried on a BODY_BLOB feature.

    Parameters:
        body_name: SPICE body name whose blob this feature represents.
        predicted_diameter_px: Predicted disc diameter in pixels (longer
            axis of the predicted ellipse silhouette).  Must be ``>= 0``.
        phase_angle_deg: Phase angle (Sun -> body -> observer) at the
            body's center, in degrees.  Recorded for diagnostic
            inspection; the BLOB confidence formula consumes
            ``phase_irregularity_factor`` instead, since raw phase alone
            understates the centroid uncertainty for an irregular body.
            Must be in ``[0, 180]``.
        phase_irregularity_factor: Dimensionless coupling of phase angle
            and shape irregularity, computed by the body NavModel as
            ``(ellipsoid_rms_residual_km / body_radius_km) *
            (1 + 2 * sin^2(phase / 2))``.  Captures the centroid-bias
            risk that the lit-weighted predicted centroid cannot fully
            correct for.  The fractional ``residual / radius`` term is
            ~ 0.005 for regular moons and ~ 0.05-0.10 for irregular
            satellites; the phase factor goes from 1 at full-phase
            (rotational orientation always unknown) to 3 at full
            crescent (most of the body unlit, hiding most of the
            irregularity).  Must be ``>= 0``.
        sub_solar_dir_vu: Unit ``(v, u)`` image-plane direction from the
            body's geometric center toward the bright limb (the
            projection of the body-to-Sun vector).  ``(0.0, 0.0)`` when
            the direction is unknown or undefined (a near-full-phase body
            whose lit centroid coincides with its geometric center).
            ``BodyBlobNav`` orients its phase-aware coarse-acquisition
            template along this direction so a high-phase crescent
            displaced beyond its predicted bounding box is still found; a
            filled-disc template cannot match a thin crescent.  Both
            components must lie in ``[-1, 1]``.
    """

    body_name: str = ''
    predicted_diameter_px: float = 0.0
    phase_angle_deg: float = 0.0
    phase_irregularity_factor: float = 0.0
    sub_solar_dir_vu: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate per-field constraints."""
        if self.predicted_diameter_px < 0.0:
            raise ValueError(
                f'predicted_diameter_px must be >= 0; got {self.predicted_diameter_px!r}'
            )
        if not (0.0 <= self.phase_angle_deg <= 180.0):
            raise ValueError(f'phase_angle_deg must be in [0, 180]; got {self.phase_angle_deg!r}')
        if self.phase_irregularity_factor < 0.0:
            raise ValueError(
                f'phase_irregularity_factor must be >= 0; got {self.phase_irregularity_factor!r}'
            )
        dir_v, dir_u = self.sub_solar_dir_vu
        if not (-1.0 <= dir_v <= 1.0 and -1.0 <= dir_u <= 1.0):
            raise ValueError(
                f'sub_solar_dir_vu components must lie in [-1, 1]; got {self.sub_solar_dir_vu!r}'
            )


@dataclass(frozen=True)
class RingAnnulusFlags:
    """Flags carried on a RING_ANNULUS feature.

    Parameters:
        planet_name: Planet whose ring system this annulus represents.
        constituent_edge_count: Number of catalog edges fused into this
            annulus template.  Must be a non-negative integer.
    """

    planet_name: str = ''
    constituent_edge_count: int = 0

    def __post_init__(self) -> None:
        """Validate ``constituent_edge_count`` is a non-negative integer."""
        if not isinstance(self.constituent_edge_count, int) or isinstance(
            self.constituent_edge_count, bool
        ):
            raise TypeError(
                f'constituent_edge_count must be int; got '
                f'{type(self.constituent_edge_count).__name__}'
            )
        if self.constituent_edge_count < 0:
            raise ValueError(
                f'constituent_edge_count must be >= 0; got {self.constituent_edge_count!r}'
            )


@dataclass(frozen=True)
class CartographicModelFlags:
    """Flags carried on a CARTOGRAPHIC_MODEL feature.

    Parameters:
        body_name: SPICE body name the cartographic mosaic represents.
        mosaic_source: Identifier of the mosaic file (e.g. file basename or
            URL).
    """

    body_name: str = ''
    mosaic_source: str = ''


@dataclass(frozen=True)
class TitanHazeFlags:
    """Flags carried on a TITAN_LIMB feature.

    Parameters:
        body_name: SPICE body name whose haze envelope this feature
            traces.  Read by ``NavFeature.body_name`` for feature-level
            body attribution, exactly as on the other body-feature flag
            dataclasses.
        surface_window_filter: True when the image was taken through a
            filter that sees past the haze to the surface.  Recorded for
            diagnosis and future refinement; the haze fit itself does not
            branch on it.
        high_phase: True when the phase angle is large enough that the
            sunward limb arc has shrunk toward a crescent, which is the
            regime where the sector fit has the least support.
    """

    body_name: str = ''
    surface_window_filter: bool = False
    high_phase: bool = False


NavFeatureFlags = (
    StarFlags
    | LimbArcFlags
    | TerminatorArcFlags
    | RingEdgeFlags
    | BodyDiscFlags
    | BodyBlobFlags
    | RingAnnulusFlags
    | CartographicModelFlags
    | TitanHazeFlags
)
"""Sum type spanning every NavFeatureType's flag dataclass."""
