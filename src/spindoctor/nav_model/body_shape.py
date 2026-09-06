"""Per-body shape parameters used by the body NavModel.

The body extractor's covariance and emission gates consult the per-body
shape, albedo, and SPICE-residual quantities returned by
:func:`load_body_shape`.  The lookup pulls operator-curated values from
``config_220_body_shape.yaml`` first, falling back to the hard-coded
``BODY_SHAPE_TABLE`` profiles for bodies the YAML has not populated yet,
and finally to ``DEFAULT_BODY_SHAPE`` for entirely unknown bodies.

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

Documentation-only fields in the YAML (``radii_km``, ``albedo_mean``,
``_sources``) are intentionally not surfaced on the runtime dataclass:
``radii_km`` would compete with ``oops`` body radii and ``albedo_mean``
has no consumer yet.  Adding them at runtime is a one-line dataclass
extension whenever a consumer materializes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

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
    shape_class_hint: str = 'unknown'


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
``config_220_body_shape.yaml`` or when individual numeric fields in
the YAML entry are ``null`` (the YAML value wins when it is set).
Bodies absent from this table fall through to ``DEFAULT_BODY_SHAPE``.
"""


# YAML field names that map directly onto ``BodyShape`` fields.  The
# loader walks this list and overwrites the hard-coded baseline only
# when the YAML carries a non-null value.  Keep ordering aligned with
# the ``BodyShape`` field declaration above for grep-ability.
_YAML_FIELDS: tuple[str, ...] = (
    'ellipsoid_rms_residual_km',
    'crater_scale_km',
    'albedo_variation',
    'spice_orbital_residual_km',
    'min_blob_diameter_px',
    'shape_class_hint',
)


def load_body_shape(body_name: str, config: Any = None) -> BodyShape:
    """Build the runtime ``BodyShape`` for ``body_name`` (case-insensitive).

    Merges three sources, in priority order:

    1. ``config.body_shape[<BODY>]`` — operator-curated YAML entry from
       ``config_220_body_shape.yaml``.  Each non-null field overrides
       the hard-coded baseline.
    2. ``BODY_SHAPE_TABLE[<BODY>]`` — hard-coded per-class profile for
       bodies the YAML has not populated yet (e.g. minor moons that
       Phase 10 §B has not reached).
    3. ``DEFAULT_BODY_SHAPE`` — final fallback for bodies unknown to
       both sources.

    Parameters:
        body_name: Body name in any case (``'mimas'`` / ``'MIMAS'``).
        config: Optional ``Config`` override; defaults to
            ``DEFAULT_CONFIG``.  Tests may pass a stub that exposes a
            ``body_shape`` attribute (mapping or ``AttrDict``).

    Returns:
        A ``BodyShape`` populated from the best available source for
        each field.
    """
    upper = body_name.upper()
    baseline = BODY_SHAPE_TABLE.get(upper, DEFAULT_BODY_SHAPE)
    yaml_block = _yaml_entry_for(upper, config)
    if yaml_block is None:
        return baseline
    overrides: dict[str, Any] = {}
    for field in _YAML_FIELDS:
        if field not in yaml_block:
            continue
        value = yaml_block[field]
        if value is None:
            continue
        overrides[field] = value
    if not overrides:
        return baseline
    return replace(baseline, **overrides)


def _yaml_entry_for(upper_body_name: str, config: Any) -> dict[str, Any] | None:
    """Return the YAML mapping for ``upper_body_name`` if present.

    Resolves ``config.body_shape`` against either an explicit
    ``Config`` instance or the global ``DEFAULT_CONFIG``.  Returns
    ``None`` when the body is absent, the YAML block is empty / not a
    mapping, or the loader has not been run yet (``config.body_shape``
    raises during early bootstrapping).
    """
    cfg = config
    if cfg is None:
        # Local import keeps this module import-cycle-free; the global
        # default is the one nav_model_body.py uses when no per-instance
        # override is supplied.
        from spindoctor.config import DEFAULT_CONFIG

        cfg = DEFAULT_CONFIG
    # Read straight through. A configuration with no body_shape section is a
    # configuration this code cannot work from, and answering None for one
    # would drop every per-body override without a word -- silently navigating
    # against the wrong shape. A test stub that does not expose the section is
    # an incomplete stub, not a case for production to carry.
    body_shape_section = cfg.body_shape
    if not isinstance(body_shape_section, dict):
        return None
    entry = body_shape_section.get(upper_body_name)
    if entry is None or not isinstance(entry, dict):
        return None
    return dict(entry)
