"""Statistics machinery for the sim-vs-real realism match.

This package holds the figure-of-merit (FOM) statistics the realism-match
runner (``tests/integration/sim_realism.py``) computes over the curated
image-library cohort and matched simulated frames:

- :mod:`spindoctor.sim.realism.divergence`: the Wasserstein-1 divergence on
  quantile-clipped samples, normalized by the real distribution's IQR, plus
  cohort support labeling.
- :mod:`spindoctor.sim.realism.noise`: sky-region statistics -- near-uniform
  patch selection, the paired-difference noise estimator, and the sky spatial
  power spectrum (FOM 1).
- :mod:`spindoctor.sim.realism.profiles`: star-cutout radial profiles and
  encircled energy (FOM 2), and edge-normal profile sampling shared by the
  limb (FOM 3) and ring-edge (FOM 4) comparisons.
- :mod:`spindoctor.sim.realism.dynamic_range`: exposure-stratified dynamic
  range statistics (FOM 5) and the stratification logic itself.
- :mod:`spindoctor.sim.realism.artifact_rates`: measured artifact rates
  (missing lines, hot pixels, transient spikes) for FOM 6.

Everything here is a pure function of numpy arrays and small dataclasses:
no holdings access, no SPICE, no rendering.  The runner supplies the pixels
and the metadata; this package supplies the statistics, so the statistics
are unit-testable on synthetic distributions with known answers.

FOM 7 (technique-diagnostic distributions) deliberately has no module here:
it reuses :func:`divergence.w1_divergence` on diagnostics the runner collects
from navigation runs, and is a read-only report -- never a tuning target.
"""

from spindoctor.sim.realism.artifact_rates import (
    ArtifactRates,
    measure_artifact_rates,
)
from spindoctor.sim.realism.divergence import (
    CohortSupport,
    W1Result,
    cohort_support,
    w1_between_densities,
    w1_divergence,
)
from spindoctor.sim.realism.dynamic_range import (
    DynamicRangeStats,
    frame_dynamic_range,
    stratify_by_exposure,
)
from spindoctor.sim.realism.noise import (
    SkyPatch,
    find_uniform_patches,
    paired_difference_sigma,
    radial_power_spectrum,
)
from spindoctor.sim.realism.profiles import (
    edge_normal_profiles,
    encircled_energy,
    profile_rise_width,
    radial_profile,
)

__all__ = [
    'ArtifactRates',
    'CohortSupport',
    'DynamicRangeStats',
    'SkyPatch',
    'W1Result',
    'cohort_support',
    'edge_normal_profiles',
    'encircled_energy',
    'find_uniform_patches',
    'frame_dynamic_range',
    'measure_artifact_rates',
    'paired_difference_sigma',
    'profile_rise_width',
    'radial_power_spectrum',
    'radial_profile',
    'stratify_by_exposure',
    'w1_between_densities',
    'w1_divergence',
]
