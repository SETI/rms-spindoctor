"""Per-body shape parameters used by the body NavModel.

The body extractor's covariance and emission gates consult the per-body
shape, albedo, and SPICE-residual quantities returned by
:func:`load_body_shape`.  The lookup takes each value from three tiers in
turn: the body's entry in the ``body_shape`` section of the configuration
(``config_220_body_shape.yaml``), then the hard-coded ``BODY_SHAPE_TABLE``
profile for a value the entry leaves out, and finally ``DEFAULT_BODY_SHAPE``
for a body without a profile.

Each ``BodyShape`` instance carries the values the navigation pipeline
actually consumes:

- ``ellipsoid_rms_residual_km`` — RMS deviation of the body silhouette
  from the best-fit ellipsoid.  Drives the LIMB_ARC normal-sigma.
- ``crater_scale_km`` — characteristic per-image limb roughness from
  craters and topography.
- ``albedo_variation`` — fractional brightness variation across the
  disc; drives the terminator photometric error budget.
- ``spice_orbital_residual_km`` — SPK ephemeris uncertainty projected
  to the limb plane.
- ``min_blob_diameter_px`` — minimum predicted disc diameter at which
  ``BODY_BLOB`` is preferred over an unresolved limb.
- ``shape_class_hint`` — ``regular`` / ``irregular`` /
  ``highly_irregular`` / ``unknown``; used in human-readable logs and
  by reviewers reading sidecar diagnostics.

An entry's ``radii_km`` is not surfaced on the runtime dataclass, since it would compete
with the ``oops`` body radii; the calibration tooling reads it from the
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from spindoctor.config import DEFAULT_CONFIG, Config

if TYPE_CHECKING:  # pragma: no cover - typing-only import, which would be circular at run time
    from spindoctor.config.sections_models import ShapeClass

__all__ = [
    'BODY_SHAPE_TABLE',
    'DEFAULT_BODY_SHAPE',
    'BodyShape',
    'load_body_shape',
]


@dataclass(frozen=True)
class BodyShape:
    """Per-body shape and SPICE-residual quantities consumed at run time.

    Parameters:
        ellipsoid_rms_residual_km: RMS shape residual from the best-fit
            ellipsoid (km).  Primary contribution in the limb-arc
            normal-sigma quadrature sum.
        crater_scale_km: Characteristic crater / topographic scale (km),
            independent of ``ellipsoid_rms_residual_km``.  Adds to the
            limb-arc normal-sigma in quadrature.
        albedo_variation: Fractional disc brightness variation in
            ``[0, 1]``; drives terminator-arc reliability.
        spice_orbital_residual_km: SPK ephemeris uncertainty in km
            (~0.5 for major moons; up to 5 for irregular satellites).
        min_blob_diameter_px: Predicted disc diameter (px) at which the
            extractor stops emitting LIMB_ARC and switches to BODY_BLOB.
        shape_class_hint: Coarse classification used by the log /
            reviewer.  One of ``regular``, ``irregular``,
            ``highly_irregular``, ``unknown``.
    """

    ellipsoid_rms_residual_km: float
    crater_scale_km: float
    albedo_variation: float
    spice_orbital_residual_km: float
    min_blob_diameter_px: float = 5.0
    shape_class_hint: ShapeClass = 'unknown'


DEFAULT_BODY_SHAPE: BodyShape = BodyShape(
    ellipsoid_rms_residual_km=2.0,
    crater_scale_km=5.0,
    albedo_variation=0.15,
    spice_orbital_residual_km=2.0,
    min_blob_diameter_px=5.0,
    shape_class_hint='unknown',
)
"""Fallback shape used when a body has no specific entry.

The numbers reflect a generic small icy moon: ~2 km bulk-shape residual,
~5 km crater scale, modest albedo variation, generous 2 km SPK residual.
"""


_SATURN_MOON_SHAPE: BodyShape = BodyShape(
    ellipsoid_rms_residual_km=1.0,
    crater_scale_km=2.0,
    albedo_variation=0.10,
    spice_orbital_residual_km=0.5,
    min_blob_diameter_px=5.0,
    shape_class_hint='regular',
)
"""Profile for the major Saturn moons whose shape is well-measured."""


_IRREGULAR_MOON_SHAPE: BodyShape = BodyShape(
    ellipsoid_rms_residual_km=10.0,
    crater_scale_km=5.0,
    albedo_variation=0.20,
    spice_orbital_residual_km=2.0,
    min_blob_diameter_px=5.0,
    shape_class_hint='highly_irregular',
)
"""Profile for very irregular bodies (Hyperion, Phoebe, etc.)."""


_GAS_GIANT_SHAPE: BodyShape = BodyShape(
    ellipsoid_rms_residual_km=50.0,
    crater_scale_km=0.0,
    albedo_variation=0.30,
    spice_orbital_residual_km=10.0,
    min_blob_diameter_px=20.0,
    shape_class_hint='regular',
)
"""Profile for gas / ice giants whose limbs are atmospheric scattering."""


BODY_SHAPE_TABLE: dict[str, BodyShape] = {
    'SATURN': _GAS_GIANT_SHAPE,
    'JUPITER': _GAS_GIANT_SHAPE,
    'URANUS': _GAS_GIANT_SHAPE,
    'NEPTUNE': _GAS_GIANT_SHAPE,
    'MIMAS': _SATURN_MOON_SHAPE,
    'ENCELADUS': _SATURN_MOON_SHAPE,
    'TETHYS': _SATURN_MOON_SHAPE,
    'DIONE': _SATURN_MOON_SHAPE,
    'RHEA': _SATURN_MOON_SHAPE,
    'IAPETUS': _SATURN_MOON_SHAPE,
    'TITAN': _SATURN_MOON_SHAPE,
    'HYPERION': _IRREGULAR_MOON_SHAPE,
    'PHOEBE': _IRREGULAR_MOON_SHAPE,
    'EUROPA': _SATURN_MOON_SHAPE,
    'IO': _SATURN_MOON_SHAPE,
    'GANYMEDE': _SATURN_MOON_SHAPE,
    'CALLISTO': _SATURN_MOON_SHAPE,
}
"""Hard-coded fallback profiles, keyed by upper-case SPICE body name.

Used when the corresponding body is missing from
``config_220_body_shape.yaml`` or when its entry leaves out a
key (the configured value wins when it is set).
Bodies absent from this table fall through to ``DEFAULT_BODY_SHAPE``.
"""


_T = TypeVar('_T')


def load_body_shape(body_name: str, config: Config | None = None) -> BodyShape:
    """Build the runtime ``BodyShape`` for ``body_name`` (case-insensitive).

    Merges three sources, in priority order:

    1. ``config.body_shape[<BODY>]`` -- operator-curated entry from
       ``config_220_body_shape.yaml``.  Each key it sets overrides
       the hard-coded baseline.
    2. ``BODY_SHAPE_TABLE[<BODY>]`` -- hard-coded per-class profile, for
       a key the entry leaves out and for a body without an entry.
    3. ``DEFAULT_BODY_SHAPE`` -- final fallback for bodies unknown to
       both sources.

    Parameters:
        body_name: Body name in any case (``'mimas'`` / ``'MIMAS'``).
        config: The configuration to read; defaults to ``DEFAULT_CONFIG``.

    Returns:
        A ``BodyShape`` populated from the best available source for
        each field.
    """
    upper = body_name.upper()
    baseline = BODY_SHAPE_TABLE.get(upper, DEFAULT_BODY_SHAPE)
    entry = (DEFAULT_CONFIG if config is None else config).body_shape.get(upper)
    if entry is None:
        return baseline
    return BodyShape(
        ellipsoid_rms_residual_km=_first_set(
            entry.ellipsoid_rms_residual_km, baseline.ellipsoid_rms_residual_km
        ),
        crater_scale_km=_first_set(entry.crater_scale_km, baseline.crater_scale_km),
        albedo_variation=_first_set(entry.albedo_variation, baseline.albedo_variation),
        spice_orbital_residual_km=_first_set(
            entry.spice_orbital_residual_km, baseline.spice_orbital_residual_km
        ),
        min_blob_diameter_px=_first_set(entry.min_blob_diameter_px, baseline.min_blob_diameter_px),
        shape_class_hint=_first_set(entry.shape_class_hint, baseline.shape_class_hint),
    )


def _first_set(value: _T | None, fallback: _T) -> _T:
    """Return ``value``, or ``fallback`` when ``value`` is None.

    Parameters:
        value: The configuration's value, None when its entry leaves the key out.
        fallback: The value of the next tier.

    Returns:
        The value that applies.
    """
    return fallback if value is None else value
