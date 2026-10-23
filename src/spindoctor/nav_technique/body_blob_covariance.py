"""Covariance of ``BodyBlobNav``'s joint brightness-centroid fit.

Each blob's centroid carries photon noise and a model bias: the measured
brightness-weighted centroid is compared with the rendered lit centroid, and the
rendering's photometry, shape, and albedo are not the real body's.  The bias has an
isotropic part (a fixed fraction of the radius and the shape irregularity) and a phase
part along the image direction toward the Sun, where an error in the modeled
lit-hemisphere shift lands, with a smaller share across it.  The along-Sun share is
larger for a regular body than for an irregular one, whose irregularity term already
carries most of its bias.  The blobs combine by generalized least squares with weights
that are the inverses of these covariances, so the reported covariance is that of the
combination actually formed.  These helpers live apart from ``nav_technique_body_blob``
so that module stays under the 1000-line limit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__: list[str] = []


_PHOTOCENTER_PHASE_GUARD_RAD: float = 1.0e-3
"""Distance short of 180 degrees (radians) at which the photocenter fraction is held.

The closed form is 0 / 0 at 180 degrees; a thousandth of a radian short of it the
fraction is within 1e-6 of its limit and the formula is still well conditioned.
"""


_MIN_SUN_DIRECTION_NORM: float = 0.5
"""Length below which a blob's direction toward the Sun counts as unknown.

The body model reports a unit vector or, when the lit and geometric centers nearly
coincide, the zero vector; any length between the two halves marks the boundary.
"""


@dataclass(frozen=True)
class BlobCentroidBiasModel:
    """Per-blob centroid bias terms of the ``BodyBlobNav`` covariance.

    Parameters:
        size_frac: Isotropic bias sigma every blob carries, as a fraction of its
            radius.
        along_sun_frac_regular: Bias sigma along the image direction toward the Sun,
            as a fraction of the Lambertian photocenter offset (see
            :func:`_lambert_photocenter_fraction`), for a regular body: one whose
            phase-irregularity factor is at most ``regular_irregularity_max``.
        along_sun_frac_irregular: The same fraction for an irregular body, whose
            factor is at least ``irregular_irregularity_min``; between the two
            factors the fraction is interpolated linearly.
        across_sun_frac: Bias sigma across the direction toward the Sun, as a
            fraction of the photocenter offset, for every blob.
        regular_irregularity_max: Phase-irregularity factor at and below which a body
            takes ``along_sun_frac_regular``.
        irregular_irregularity_min: Factor at and above which a body takes
            ``along_sun_frac_irregular``; greater than ``regular_irregularity_max``.
        bias_radius_cap_px: Radius, in pixels, beyond which the phase and
            irregularity terms stop growing with the body; ``size_frac`` is not
            capped.
        bias_correlation: Fraction, in ``[0, 1)``, of the smallest bias variance among
            the blobs of one frame that all of them share (see
            :func:`_generalized_least_squares`).
    """

    size_frac: float
    along_sun_frac_regular: float
    along_sun_frac_irregular: float
    across_sun_frac: float
    regular_irregularity_max: float
    irregular_irregularity_min: float
    bias_radius_cap_px: float
    bias_correlation: float

    def __post_init__(self) -> None:
        """Validate the terms.

        Raises:
            ValueError: If a term is negative or non-finite, the irregularity bounds
                are not increasing, the radius cap is not positive, or the correlation
                is not below 1.
        """
        for name in (
            'size_frac',
            'along_sun_frac_regular',
            'along_sun_frac_irregular',
            'across_sun_frac',
            'regular_irregularity_max',
            'irregular_irregularity_min',
            'bias_radius_cap_px',
            'bias_correlation',
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'BodyBlobNav.{name} must be finite and >= 0; got {value!r}')
        if self.irregular_irregularity_min <= self.regular_irregularity_max:
            raise ValueError(
                'BodyBlobNav.irregular_irregularity_min must exceed regular_irregularity_max; '
                f'got {self.irregular_irregularity_min!r} <= {self.regular_irregularity_max!r}'
            )
        if self.bias_radius_cap_px <= 0.0:
            raise ValueError(
                f'BodyBlobNav.bias_radius_cap_px must be > 0; got {self.bias_radius_cap_px!r}'
            )
        if self.bias_correlation >= 1.0:
            raise ValueError(
                f'BodyBlobNav.bias_correlation must be < 1; got {self.bias_correlation!r}'
            )


def load_blob_bias_model(tuning: Mapping[str, Any]) -> BlobCentroidBiasModel:
    """Read the centroid bias terms from ``BodyBlobNav``'s tuning mapping.

    Parameters:
        tuning: The technique's YAML ``tuning`` mapping.

    Returns:
        The validated model.

    Raises:
        KeyError: If a term is missing.
        ValueError: If a term is invalid (see :class:`BlobCentroidBiasModel`).
    """
    return BlobCentroidBiasModel(
        size_frac=float(tuning['model_error_size_frac']),
        along_sun_frac_regular=float(tuning['along_sun_frac_regular']),
        along_sun_frac_irregular=float(tuning['along_sun_frac_irregular']),
        across_sun_frac=float(tuning['across_sun_frac']),
        regular_irregularity_max=float(tuning['regular_irregularity_max']),
        irregular_irregularity_min=float(tuning['irregular_irregularity_min']),
        bias_radius_cap_px=float(tuning['bias_radius_cap_px']),
        bias_correlation=float(tuning['bias_correlation']),
    )


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


def _along_sun_fractions(
    irregularity_factors: NDArrayFloatType, model: BlobCentroidBiasModel
) -> NDArrayFloatType:
    """Return each blob's along-Sun bias fraction, interpolated in its irregularity.

    Parameters:
        irregularity_factors: ``(N,)`` phase-irregularity factors.
        model: The bias model.

    Returns:
        ``(N,)`` fractions: ``along_sun_frac_regular`` at and below
        ``regular_irregularity_max``, ``along_sun_frac_irregular`` at and above
        ``irregular_irregularity_min``, linear between.
    """
    weight_irregular = np.clip(
        (np.asarray(irregularity_factors, np.float64) - model.regular_irregularity_max)
        / (model.irregular_irregularity_min - model.regular_irregularity_max),
        0.0,
        1.0,
    )
    return cast(
        NDArrayFloatType,
        model.along_sun_frac_regular
        + weight_irregular * (model.along_sun_frac_irregular - model.along_sun_frac_regular),
    )


def _blob_bias_covariances(
    *,
    diameters_px: NDArrayFloatType,
    phase_angles_deg: NDArrayFloatType,
    irregularity_factors: NDArrayFloatType,
    sun_dirs_vu: NDArrayFloatType,
    ephemeris_sigmas_px: NDArrayFloatType,
    model: BlobCentroidBiasModel,
) -> NDArrayFloatType:
    """Return each blob's 2x2 centroid bias covariance (photon noise excluded).

    With ``R`` the blob's radius, ``Rc = min(R, bias_radius_cap_px)``, ``g`` its
    Lambertian photocenter fraction, ``s`` the unit image direction toward the Sun and
    ``n`` the direction across it::

        iso = size_frac**2 * R**2 + (irregularity * Rc)**2 + ephemeris_sigma**2
        B   = (iso + (a * g * Rc)**2) s s^T + (iso + (across_sun_frac * g * Rc)**2) n n^T

    where ``a`` is the blob's along-Sun fraction (:func:`_along_sun_fractions`).  A blob
    whose direction toward the Sun is unknown (a vector shorter than
    :data:`_MIN_SUN_DIRECTION_NORM`, which the body model reports at low phase) splits
    the two phase terms evenly over both axes.

    Parameters:
        diameters_px: ``(N,)`` predicted diameters.
        phase_angles_deg: ``(N,)`` phase angles.
        irregularity_factors: ``(N,)`` phase-irregularity factors (negative values are
            taken as zero).
        sun_dirs_vu: ``(N, 2)`` unit ``(v, u)`` directions toward the Sun, or zero rows.
        ephemeris_sigmas_px: ``(N,)`` ephemeris position sigmas of the bodies, pixels.
        model: The bias model.

    Returns:
        ``(N, 2, 2)`` covariances in pixels squared.
    """
    radii = 0.5 * np.asarray(diameters_px, np.float64)
    capped = np.minimum(radii, model.bias_radius_cap_px)
    irregular = np.clip(np.asarray(irregularity_factors, np.float64), 0.0, None)
    photocenter_px = _lambert_photocenter_fraction(np.asarray(phase_angles_deg)) * capped
    iso = (
        (model.size_frac * radii) ** 2
        + (irregular * capped) ** 2
        + np.asarray(ephemeris_sigmas_px, np.float64) ** 2
    )
    along = iso + (_along_sun_fractions(irregular, model) * photocenter_px) ** 2
    across = iso + (model.across_sun_frac * photocenter_px) ** 2
    sun = np.asarray(sun_dirs_vu, np.float64).reshape(-1, 2)
    norms = np.hypot(sun[:, 0], sun[:, 1])
    known = norms >= _MIN_SUN_DIRECTION_NORM
    unit = np.where(known[:, None], sun / np.where(known, norms, 1.0)[:, None], 0.0)
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])
    covariances = along[:, None, None] * np.einsum('ni,nj->nij', unit, unit) + across[
        :, None, None
    ] * np.einsum('ni,nj->nij', normal, normal)
    mean_var = 0.5 * (along + across)
    covariances[~known] = mean_var[~known, None, None] * np.eye(2)
    return cast(NDArrayFloatType, covariances)


@dataclass(frozen=True)
class _GlsFit:
    """Generalized-least-squares combination of the per-blob offsets."""

    offset_vu: tuple[float, float]
    covariance: NDArrayFloatType
    chi_square: float


def _generalized_least_squares(
    offsets_vu: NDArrayFloatType,
    photon_vars_px2: NDArrayFloatType,
    bias_covariances: NDArrayFloatType,
    *,
    bias_correlation: float,
) -> _GlsFit:
    """Combine per-blob offsets by generalized least squares over all blobs at once.

    Each blob's error is its photon noise plus its centroid bias, and part of the bias
    is common to every blob in the frame: the bodies share the Sun direction, the
    rendering's photometric law, and an ephemeris solution, so their biases do not
    average away.  The common part is isotropic, ``K = rho * m * I``, with ``rho`` the
    ``bias_correlation`` and ``m`` the smallest eigenvalue of any blob's bias
    covariance, so every blob's own part ``photon_i I + B_i - K`` stays
    positive-semidefinite.  The ``2N x 2N`` covariance has blocks ``photon_i I + B_i``
    on the diagonal and ``K`` off it.  With ``H`` the stack of ``N`` 2x2 identities, the
    joint offset is ``(H^T C^-1 H)^-1 H^T C^-1 o`` and its covariance
    ``(H^T C^-1 H)^-1``, which is the own parts combined by their inverses plus ``K``:
    a common error moves every blob alike, so it neither enters the weights nor
    averages away, and the joint offset always lies within the blobs' offsets.  With
    two or more blobs the residual chi-square over ``2 N - 2`` degrees of freedom
    measures whether the blobs agree as well as the covariance says; when they agree
    worse, the covariance is scaled up by that ratio (never down).

    Parameters:
        offsets_vu: ``(N, 2)`` per-blob ``observed - predicted`` offsets, ``N >= 1``.
        photon_vars_px2: ``(N,)`` per-blob photon-noise variances.
        bias_covariances: ``(N, 2, 2)`` per-blob bias covariances.
        bias_correlation: Fraction ``rho`` in ``[0, 1)`` of the smallest bias variance
            that every blob shares.

    Returns:
        The joint offset, its covariance, and the residual chi-square (``0.0`` for one
        blob).
    """
    offsets = np.asarray(offsets_vu, np.float64).reshape(-1, 2)
    n_blobs = offsets.shape[0]
    bias = np.asarray(bias_covariances, np.float64).reshape(n_blobs, 2, 2)
    photon = np.asarray(photon_vars_px2, np.float64).reshape(n_blobs)
    shared = (
        bias_correlation * float(min(np.linalg.eigvalsh(b)[0] for b in bias))
        if n_blobs >= 2
        else 0.0
    )
    full = np.kron(np.ones((n_blobs, n_blobs)), shared * np.eye(2))
    for i in range(n_blobs):
        full[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = photon[i] * np.eye(2) + bias[i]
    design = np.tile(np.eye(2), (n_blobs, 1))
    weights = np.linalg.inv(full)
    joint_cov = np.linalg.inv(design.T @ weights @ design)
    stacked = offsets.reshape(-1)
    offset = joint_cov @ (design.T @ weights @ stacked)
    residuals = stacked - design @ offset
    chi_square = float(residuals @ weights @ residuals)
    if n_blobs >= 2:
        joint_cov = joint_cov * max(1.0, chi_square / (2.0 * n_blobs - 2.0))
    return _GlsFit(
        offset_vu=(float(offset[0]), float(offset[1])),
        covariance=cast(NDArrayFloatType, joint_cov),
        chi_square=chi_square,
    )
