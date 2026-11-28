"""The typed ``body_shape`` section of the navigator configuration.

The ``body_shape`` section is the body-shape table: one entry per body, keyed by the
body's SPICE name in upper case, each a :class:`BodyShapeEntry`.  Its values are
static data taken from published shape fits, and each one carries a citation in the
entry's ``_sources`` mapping, which the loader drops before the section is parsed.

Every key of an entry is optional.  A key an entry leaves out is read as None, and
:func:`~spindoctor.nav_model.body_shape.load_body_shape` takes that value from the next
tier: the body's built-in profile, or the generic profile for a body that has none.

A body key must be in upper case.  The same rule covers ``bodies.geometric_albedo``, so
:class:`~spindoctor.config.sections.NavConfig` checks it.
"""

from dataclasses import dataclass
from typing import Literal

from spindoctor.schema import SCHEMA_CONFIG, NonNeg, Positive, Unit

__all__ = [
    'BodyShapeEntry',
    'ShapeClass',
]

#: A body's coarse shape class, from a regular ellipsoid to a highly irregular body.
ShapeClass = Literal['regular', 'irregular', 'highly_irregular', 'unknown']


@dataclass(frozen=True)
class BodyShapeEntry:
    """One body's entry in the body-shape table; a key left out uses the next tier.

    Parameters:
        radii_km: The body's triaxial ellipsoid, ``[a, b, c]`` in km.  The navigator
            takes its radii from SPICE; the calibration tooling reads these.
        ellipsoid_rms_residual_km: The RMS deviation, in km, of the body's limb from the
            best-fit ellipsoid.
        crater_scale_km: The characteristic limb relief, in km, from craters and
            topography beyond the ellipsoid.
        albedo_variation: The fractional brightness variation across the disc.
        spice_orbital_residual_km: The SPK ephemeris uncertainty, in km, projected to the
            limb plane.
        min_blob_diameter_px: The predicted disc diameter, in pixels, below which a blob
            is preferred over an unresolved limb.
        shape_class_hint: The body's coarse shape class.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    radii_km: tuple[Positive, Positive, Positive] | None = None
    ellipsoid_rms_residual_km: NonNeg | None = None
    crater_scale_km: NonNeg | None = None
    albedo_variation: Unit | None = None
    spice_orbital_residual_km: NonNeg | None = None
    min_blob_diameter_px: Positive | None = None
    shape_class_hint: ShapeClass | None = None
