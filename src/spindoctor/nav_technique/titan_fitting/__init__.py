"""Pure fitting library for haze-symmetry navigation of Titan.

The method (Hanson, French, Waugh, Barth and Anderson, 2025, GRL,
doi:10.1029/2024GL113415) exploits two properties of a cloud-free hazy
atmosphere.  Absent surface detail the disc is mirror-symmetric about the
image-plane line through the body center and the sub-solar point, so the
image displacement perpendicular to that line ("cross-track") is the shift
that maximizes mirror symmetry.  The limb arc facing the sub-solar point is
close to circular, so a circle fit with FREE radius to that arc pins the
displacement along the line ("along-track") without assuming a haze
altitude.  Together the two constraints give a full ``(dv, du)`` offset.

Everything here is a pure function over numpy arrays plus the parameter and
result dataclasses; there is no observation, context, config or logger
access, so the whole algorithm is exercisable on synthetic images.

Conventions used throughout:

* ``(v, u)`` is ``(row, column)``; an offset ``(dv, du)`` means the actual
  position of a feature predicted at ``(v, u)`` is ``(v + dv, u + du)``.
* ``theta`` is the symmetry-axis angle.  ``a_hat = (sin theta, cos theta)``
  is the unit vector along the axis pointing toward the sub-solar side and
  ``c_hat = (cos theta, -sin theta)`` is the perpendicular; a positive
  cross-track value is a displacement along ``c_hat``.
* The rotated resampling grid has axes ``(s, t)``: ``s`` runs along
  ``c_hat``, ``t`` runs along ``a_hat``, both at 1 px spacing.
* A contaminant mask marks pixels the fits must ignore (occluders, sibling
  bodies, bright stars).  It is supplied UNDILATED at predicted geometry.
  Because a pointing error translates the whole scene identically, the mask
  is always applied SHIFTED by the current center hypothesis and dilated
  along the axis by the pass pad, never statically.

The implementation is split by pipeline stage; this module re-exports the
whole surface so consumers import from
``spindoctor.nav_technique.titan_fitting`` regardless of which stage module
a helper lives in:

* :mod:`~spindoctor.nav_technique.titan_fitting.grid` -- axis unit vectors
  and the rotated-grid resampler.
* :mod:`~spindoctor.nav_technique.titan_fitting.symmetry` -- the
  mirror-correlation cross-track scan and its parameter and result types.
* :mod:`~spindoctor.nav_technique.titan_fitting.arc` -- radial profiles,
  limb extraction, and the axis-constrained robust circle fit.
* :mod:`~spindoctor.nav_technique.titan_fitting.driver` -- the two-pass
  sequence that assembles the offset.
"""

from spindoctor.nav_technique.titan_fitting.arc import (
    ARC_RADIUS_MAX_FRACTION,
    ARC_RADIUS_MIN_FRACTION,
    ArcFitParams,
    ArcFitResult,
    constrained_circle_fit,
    limb_radii_from_profiles,
    radial_profiles,
)
from spindoctor.nav_technique.titan_fitting.driver import fit_titan_center
from spindoctor.nav_technique.titan_fitting.grid import axis_vectors, resample_rotated_grid
from spindoctor.nav_technique.titan_fitting.symmetry import (
    SymmetryFitParams,
    SymmetryFitResult,
    symmetry_scan,
)

__all__ = [
    'ARC_RADIUS_MAX_FRACTION',
    'ARC_RADIUS_MIN_FRACTION',
    'ArcFitParams',
    'ArcFitResult',
    'SymmetryFitParams',
    'SymmetryFitResult',
    'axis_vectors',
    'constrained_circle_fit',
    'fit_titan_center',
    'limb_radii_from_profiles',
    'radial_profiles',
    'resample_rotated_grid',
    'symmetry_scan',
]
