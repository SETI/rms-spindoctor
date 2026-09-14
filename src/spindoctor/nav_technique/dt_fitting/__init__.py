"""Shared distance-transform fitting machinery for polyline-based techniques.

The body-limb, body-terminator, and ring-edge techniques all follow the
same algorithm: render the model polyline as a binary mask, take a coarse
2-D NCC against the image edge mask to get an integer offset, then refine
to sub-pixel precision by Levenberg-Marquardt minimization against the
image distance transform with Tukey-biweight outlier rejection.  After
convergence the M-estimator information matrix is inverted to produce a
covariance estimate.

Each helper here is a pure function over numpy arrays; the per-technique
classes simply assemble vertices / normals / weights and call into them.
The interface is:

* :func:`coarse_ncc_search` -- integer-pixel offset from binary masks.
* :func:`coarse_ncc_search_scored` -- the same search, also reporting the
  winning shift's match fraction (the acquisition-quality signal).
* :func:`polarity_filter` -- per-vertex acceptance from gradient direction.
* :func:`tukey_biweight_weights` -- Holland-Welsch redescender weights.
* :func:`lm_subpixel_refine` -- translation (or translation + rotation) LM
  refinement with Tukey reweighting against a precomputed DT.
* :func:`information_matrix_to_covariance` -- Hessian to covariance via
  ``pinvh`` so rank-deficient inputs are handled.

The implementation is split by pipeline stage; this module re-exports the
whole surface so consumers import from ``spindoctor.nav_technique.dt_fitting``
regardless of which stage module a helper lives in:

* :mod:`~spindoctor.nav_technique.dt_fitting.constants` -- tunable defaults.
* :mod:`~spindoctor.nav_technique.dt_fitting.coarse` -- rasterization, the
  integer search, and the polarity classifier.
* :mod:`~spindoctor.nav_technique.dt_fitting.weights` -- Tukey weighting and
  the information-matrix inversion.
* :mod:`~spindoctor.nav_technique.dt_fitting.transforms` -- pose transforms,
  the DT Jacobian, and the weighted normal equations.
* :mod:`~spindoctor.nav_technique.dt_fitting.ridge` -- the continuous
  gradient-ridge polish.
* :mod:`~spindoctor.nav_technique.dt_fitting.lm` -- the Levenberg-Marquardt
  driver and its result type.
* :mod:`~spindoctor.nav_technique.dt_fitting.basin` -- the competing-basin
  second opinion.
"""

from spindoctor.nav_technique.dt_fitting.basin import (
    SecondaryBasin,
    find_secondary_dt_minimum,
)
from spindoctor.nav_technique.dt_fitting.coarse import (
    CoarseSearchResult,
    build_polyline_mask,
    coarse_ncc_search,
    coarse_ncc_search_scored,
    coarse_polarity_search_scored,
    polarity_filter,
)
from spindoctor.nav_technique.dt_fitting.constants import (
    _INFINITY_DT_PENALTY_PX as _INFINITY_DT_PENALTY_PX,
)
from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
    DEFAULT_LM_DAMPING,
    DEFAULT_LM_MAX_ITERATIONS,
    DEFAULT_LM_STEP_TOLERANCE,
    DEFAULT_PINVH_RCOND,
    DEFAULT_RIDGE_HALF_WIDTH_PX,
    DEFAULT_RIDGE_MAX_ITERATIONS,
    DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX,
    DEFAULT_RIDGE_SAMPLE_STEP_PX,
    DEFAULT_TUKEY_C,
)
from spindoctor.nav_technique.dt_fitting.lm import (
    LMRefineResult,
    lm_subpixel_refine,
)
from spindoctor.nav_technique.dt_fitting.ridge import (
    RidgeRefineResult,
    gradient_ridge_refine,
)
from spindoctor.nav_technique.dt_fitting.transforms import (
    _rotate_directions as _rotate_directions,
)
from spindoctor.nav_technique.dt_fitting.transforms import (
    _rotate_vertices as _rotate_vertices,
)
from spindoctor.nav_technique.dt_fitting.weights import (
    information_matrix_to_covariance,
    tukey_biweight_weights,
)

__all__ = [
    'DEFAULT_COARSE_MIN_SUPPORT_FRACTION',
    'DEFAULT_LM_DAMPING',
    'DEFAULT_LM_MAX_ITERATIONS',
    'DEFAULT_LM_STEP_TOLERANCE',
    'DEFAULT_PINVH_RCOND',
    'DEFAULT_RIDGE_HALF_WIDTH_PX',
    'DEFAULT_RIDGE_MAX_ITERATIONS',
    'DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX',
    'DEFAULT_RIDGE_SAMPLE_STEP_PX',
    'DEFAULT_TUKEY_C',
    'CoarseSearchResult',
    'LMRefineResult',
    'RidgeRefineResult',
    'SecondaryBasin',
    'build_polyline_mask',
    'coarse_ncc_search',
    'coarse_ncc_search_scored',
    'coarse_polarity_search_scored',
    'find_secondary_dt_minimum',
    'gradient_ridge_refine',
    'information_matrix_to_covariance',
    'lm_subpixel_refine',
    'polarity_filter',
    'tukey_biweight_weights',
]
