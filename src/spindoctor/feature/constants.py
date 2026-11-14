"""Module-level constants used across feature extraction and reliability scoring.

These constants encode physically-motivated thresholds and caps referenced
across feature-extraction and confidence-scoring formulas.  Keeping them in
one file makes the formulas self-documenting at the point of use and gives
a single edit-site when calibration changes their values.

Each constant carries a one-line docstring with units and intent.  Constants
that have a YAML-config equivalent (per-instrument tunables) belong in the
config files; only physically-motivated values live here.
"""

__all__ = [
    'AGREEMENT_FACTOR_CAP',
    'COMBINED_CONFIDENCE_CAP',
    'INCIDENCE_FACTOR_ANGLE_CAP_DEG',
    'INCIDENCE_FACTOR_CLIP_DEG',
    'JSON_INF_SENTINEL',
    'MAX_INCIDENCE_FACTOR_CAP',
    'MIN_ANISOTROPIC_SMEAR_PX',
]


MAX_INCIDENCE_FACTOR_CAP: float = 4.76
"""Cap on incidence-angle softening factor (dimensionless).

Equal to ``1 / cos(80 deg) - 1``.  Beyond an incidence angle of 80 degrees,
the cosine projection becomes so steep that the limb pixel contributes
essentially zero useful information about limb position; treating any pixel
beyond 80 deg as having the same softness factor is the principled cap.
"""


INCIDENCE_FACTOR_ANGLE_CAP_DEG: float = 80.0
"""Incidence angle (degrees) at which the softening factor saturates.

Beyond this angle, ``incidence_factor`` is clamped to
``MAX_INCIDENCE_FACTOR_CAP`` rather than continuing to grow.
"""


INCIDENCE_FACTOR_CLIP_DEG: float = 85.0
"""Maximum incidence angle (degrees) used in cosine projection.

Used inside ``cos(min(i, INCIDENCE_FACTOR_CLIP_DEG))`` to keep the cosine
positive and bounded.  Slightly larger than the saturation angle so the
clamp behavior is smooth across the boundary.
"""


AGREEMENT_FACTOR_CAP: float = 1.5
"""Maximum boost (multiplicative) the ensemble agreement factor may apply.

Bounds the ``1 + 0.5 * max(0, log2(n_significant))`` formula; even when many
techniques agree, the boost never exceeds this multiplier.
"""


COMBINED_CONFIDENCE_CAP: float = 0.99
"""Maximum value the precision-weighted combined confidence may take.

Two correlated estimators that agree strongly reach at most 0.99,
never 1.0 — confidence is a calibrated proxy, not a probability.
"""


JSON_INF_SENTINEL: float = 1e9
"""Finite sentinel substituted for ``inf`` in JSON output.

In-memory ``NavResult`` keeps real ``inf`` for unbounded uncertainty axes
(e.g. flat-ring-only scenes); the JSON curator clamps to this finite value
for cross-language compatibility (strict JSON disallows ``Infinity``).
Downstream consumers should treat any value ``>= 1e8`` as
"axis unconstrained".
"""


MIN_ANISOTROPIC_SMEAR_PX: float = 0.5
"""Smear length below which star centroid covariance becomes isotropic.

When ``L < MIN_ANISOTROPIC_SMEAR_PX``, the smear axis is shorter than the
PSF and anisotropy is sub-pixel-meaningless; the extractor uses
``(sigma_PSF / sqrt(SNR))^2 * I_2`` instead of the anisotropic formula.
"""
