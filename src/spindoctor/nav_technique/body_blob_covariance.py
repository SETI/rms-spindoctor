"""Covariance of ``BodyBlobNav``'s joint brightness-centroid fit.

The joint fit's statistical scatter, the size-scaled centroid-bias floor, and the
per-blob phase- and shape-dependent centroid bias.  These live apart from
``nav_technique_body_blob`` so that module stays under the 1000-line limit.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__: list[str] = []


_PHOTOCENTER_PHASE_GUARD_RAD: float = 1.0e-3
"""Distance short of 180 degrees (radians) at which the photocenter fraction is held.

The closed form is 0 / 0 at 180 degrees; a thousandth of a radian short of it the
fraction is within 1e-6 of its limit and the formula is still well conditioned.
"""


def _lambert_photocenter_fraction(phase_deg: NDArrayFloatType) -> NDArrayFloatType:
    """Return a Lambertian sphere's photocenter offset as a fraction of its radius.

    The brightness-weighted centroid of a sphere lit at phase angle ``alpha`` sits
    toward the Sun, away from the geometric center, by

    .. math::
        \\frac{3\\pi}{16}\\,
        \\frac{\\sin\\alpha\\,(1 + \\cos\\alpha)}{\\sin\\alpha + (\\pi - \\alpha)\\cos\\alpha}

    times its radius: 0 at zero phase, 0.20 at 30 degrees, 0.59 at 90 degrees, and
    ``9 * pi / 32`` (0.88) in the limit of a vanishing crescent.  Phases are clamped to
    ``[0, 180)`` degrees, the last thousandth of a radian short of 180 taking the value
    at that bound.

    Parameters:
        phase_deg: Phase angles in degrees.

    Returns:
        The offset fractions, the same shape as ``phase_deg``.
    """
    alpha = np.radians(np.clip(np.asarray(phase_deg, np.float64), 0.0, 180.0))
    alpha = np.minimum(alpha, math.pi - _PHOTOCENTER_PHASE_GUARD_RAD)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    numerator = sin_a * (1.0 + cos_a)
    denominator = sin_a + (math.pi - alpha) * cos_a
    fraction = (3.0 * math.pi / 16.0) * numerator / denominator
    return cast(NDArrayFloatType, np.where(alpha > 0.0, fraction, 0.0))


def _centroid_bias_fractions(
    phase_angles_deg: NDArrayFloatType,
    irregularity_factors: NDArrayFloatType,
    *,
    phase_frac: float,
) -> NDArrayFloatType:
    """Return each blob's phase- and shape-dependent centroid bias, per unit radius.

    The predicted center a blob is measured against is the brightness-weighted centroid
    of the rendered body, so the lit-hemisphere shift is modeled, but not exactly: the
    rendered photometric law, albedo, and shape differ from the real body's, and the
    measured centroid counts only pixels above the noise threshold, which drops the faint
    terminator side of a crescent.  Two terms, added in quadrature, size what is left:

    - ``phase_frac`` times the Lambertian photocenter offset
      (:func:`_lambert_photocenter_fraction`), the part of the modeled phase shift that is
      uncertain;
    - the body model's ``phase_irregularity_factor``, the centroid bias of a shape
      irregular beyond its ellipsoid, which grows as less of the body is lit.

    Parameters:
        phase_angles_deg: ``(N,)`` per-blob phase angles in degrees.
        irregularity_factors: ``(N,)`` per-blob phase-irregularity factors (``>= 0``).
        phase_frac: Uncertain fraction of the modeled photocenter shift; ``0.0``
            disables the phase term.

    Returns:
        ``(N,)`` per-blob bias sigmas as fractions of each blob's radius.
    """
    phase_term = phase_frac * _lambert_photocenter_fraction(phase_angles_deg)
    irregular = np.clip(np.asarray(irregularity_factors, np.float64), 0.0, None)
    return cast(NDArrayFloatType, np.hypot(phase_term, irregular))


def _centroid_model_error_var(
    extents_px: NDArrayFloatType,
    frac: float,
    *,
    bias_fracs: NDArrayFloatType | None = None,
) -> float:
    """Return the size-scaled centroid-model-error variance for the joint fit.

    A brightness-weighted centroid is a biased estimate of the geometric
    center: the lit hemisphere and any shape irregularity shift it by a
    roughly fixed *fraction* of the body radius, an error the photon-only
    per-blob CRLB is blind to.  Each blob therefore carries a floor
    variance ``(f_i * R_i)**2`` with ``R_i`` its predicted radius and
    ``f_i**2 = frac**2 + bias_fracs[i]**2``: ``frac`` a bias every blob
    carries, and ``bias_fracs`` the blob's own phase- and shape-dependent
    bias (see :func:`_centroid_bias_fractions`).  These combine by
    inverse-variance across blobs (the joint centroid is a precision-weighted
    mean), so the joint size-error variance is ``1 / sum_i 1 / (f_i * R_i)**2``
    -- for a single blob this is exactly ``(f * R)**2``, and it shrinks as more
    blobs constrain the fit.

    Parameters:
        extents_px: Per-blob predicted diameters in pixels.
        frac: Fraction of the body radius taken as the centroid bias sigma of
            every blob; ``0.0`` disables that part.
        bias_fracs: Optional per-blob additional bias fractions, the same length
            as ``extents_px``; ``None`` adds none.

    Returns:
        The joint centroid-model-error variance in pixels squared (``0.0``
        when both parts are zero or no blob has a positive extent).
    """
    radii = 0.5 * np.asarray(extents_px, np.float64)
    frac_sq = np.full(radii.shape, frac * frac, np.float64)
    if bias_fracs is not None:
        frac_sq = frac_sq + np.asarray(bias_fracs, np.float64) ** 2
    per_blob_var = frac_sq * radii * radii
    positive = per_blob_var[per_blob_var > 0.0]
    if positive.size == 0:
        return 0.0
    return float(1.0 / np.sum(1.0 / positive))


def _joint_covariance(
    *,
    offsets_v: NDArrayFloatType,
    offsets_u: NDArrayFloatType,
    weights: NDArrayFloatType,
    dv: float,
    du: float,
    extents_px: NDArrayFloatType,
    model_error_floor_px: float = 0.0,
    centroid_model_error_frac: float = 0.0,
    bias_fracs: NDArrayFloatType | None = None,
) -> NDArrayFloatType:
    """Return the per-axis reduced-chi-square covariance of the joint fit.

    The covariance is diagonal.  With ``N`` blobs and ``p = 2`` fitted
    translation parameters, the per-axis reduced chi-square is

    ::

        chi2_nu_axis = sum_i w_i * r_axis_i**2 / max(N - p, 1)

    and the weighted-mean variance is ``chi2_nu_axis / sum(w_i)``.

    NAV-005: a single blob (``N = 1``) cannot constrain a 2-D
    translation -- ``N - p = -1`` so ``max(N - p, 1) = 1`` and there is
    no residual scatter to estimate (the lone residual is zero by
    construction).  The result therefore collapses to the
    inverse-precision floor ``1 / sum(w_i)``, which for a single blob is
    the (large) per-blob centroid CRLB variance -- correctly reflecting
    that one point is near-unobservable for two parameters rather than
    over-confident.  The previous ``sum(w r^2) / (sum w)^2`` form was the
    wrong power of ``sum w`` and carried no degrees-of-freedom factor.

    The positive-definite floor is ``1 / sum(w_i)`` (pure
    inverse-precision), NOT ``1 / (sum w_i)^2``.  Two model-error terms
    are then added in quadrature: the size-scaled centroid-bias variance
    (see :func:`_centroid_model_error_var`, which takes ``bias_fracs``, the
    per-blob phase- and shape-dependent part), which makes the reported
    sigma track body size and lighting the way the photon-only weights
    cannot, and the absolute ``model_error_floor_px**2`` floor.

    The cross-term ``cov(v, u)`` is intentionally zero -- per-axis
    residuals are independent under the BODY_BLOB CRLB derivation,
    and the precision-weighted ensemble combine downstream consumes
    diagonals correctly.  Future readers tempted to add
    ``cov_vu = sum(w * res_v * res_u) / sum(w)``: that term has no
    physical interpretation here because the per-axis errors come
    from independent moment integrals along orthogonal axes.
    """
    total_weight = float(weights.sum())
    floor = 1.0 / max(total_weight, 1e-12)
    model_error = model_error_floor_px * model_error_floor_px + _centroid_model_error_var(
        extents_px, centroid_model_error_frac, bias_fracs=bias_fracs
    )
    n = int(offsets_v.size)
    if n <= 1:
        # Single blob: under-determined for 2 translation params.  Report
        # the inverse-precision floor (the per-blob centroid CRLB variance),
        # not a tiny over-confident value.
        return float(floor) * np.eye(2, dtype=np.float64) + model_error * np.eye(
            2, dtype=np.float64
        )
    residuals_v = offsets_v - dv
    residuals_u = offsets_u - du
    dof = max(n - 2, 1)
    chi2_nu_v = float(np.sum(weights * residuals_v * residuals_v)) / dof
    chi2_nu_u = float(np.sum(weights * residuals_u * residuals_u)) / dof
    var_v = max(chi2_nu_v / total_weight, floor) + model_error
    var_u = max(chi2_nu_u / total_weight, floor) + model_error
    return np.diag([var_v, var_u]).astype(np.float64)
