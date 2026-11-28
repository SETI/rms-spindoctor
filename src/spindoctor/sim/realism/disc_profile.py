"""How a body's brightness falls across its disc, measured the same way on any frame.

A photometric law sets how a body's brightness changes from the middle of its disc out
to the limb and in to the terminator.  This module measures that profile from a frame's
pixels and the lighting and viewing angles at each, so a real frame and a simulated twin
of it can be compared without either side's brightness scale entering.

**The pixels.**  :func:`select_disc_pixels` keeps the lit pixels of the body that lie at
least :data:`DISC_CLEARANCE_PX` from the edge of the lit region, which is the limb on one
side and the terminator on the other.  The optics blur the last pixel or two of the disc
into the sky, so pixels nearer the edge measure the camera rather than the surface.  Both
cosines must also be at least :data:`MIN_DISC_COSINE`.

**Minnaert's exponent.**  :func:`fit_minnaert` fits ``I = B0 * mu0**k * mu**(k - 1)`` to
the pixels, where ``mu0`` and ``mu`` are the cosines of the incidence and emission
angles.  Taking logarithms, ``log(I * mu) = log(B0) + k * log(mu0 * mu)`` is a straight
line, so ``k`` is its slope.  The slope does not depend on the overall brightness, and an
albedo marking moves a few pixels off the line, where the fit's outlier rejection drops
them.  ``k = 1`` is Lambert's law and ``k = 0.5`` the classic limb-brightened lunar value.
:func:`fit_minnaert_together` fits several images of one disc, such as a real disc and
its simulated twins, over the pixels that every one of their fits keeps, so a marking
that one image's fit drops is left out of all of them.

**The lunar-Lambert blend.**  :func:`fit_lunar_lambert` fits
``I = B0 * (2 * L * mu0 / (mu0 + mu) + (1 - L) * mu0)``, the blend of Lommel-Seeliger
(``L = 1``) and Lambert (``L = 0``) that McEwen's law gives as a function of phase.

**The profile along the photometric equator.**  :func:`equator_profile` bins the pixels
within :data:`EQUATOR_HALF_WIDTH_DEG` of the great circle through the sub-solar and
sub-observer points by their luminance longitude, from the terminator to the sunlit limb.
The luminance coordinates come from the two cosines divided by the sine of the phase, so
any error in the cosines grows as the phase falls, and at zero phase the circle is not
defined.  Below :data:`LOW_PHASE_DEG` the pixels are binned by ``mu`` instead.
:func:`profile_residual` compares two such profiles after scaling each to the same
median, so only their shapes enter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DISC_CLEARANCE_PX',
    'EQUATOR_HALF_WIDTH_DEG',
    'LOW_PHASE_DEG',
    'MIN_DISC_COSINE',
    'MIN_FIT_PIXELS',
    'LunarLambertFit',
    'MinnaertFit',
    'equator_profile',
    'fit_lunar_lambert',
    'fit_minnaert',
    'fit_minnaert_together',
    'photometric_coordinates',
    'profile_residual',
    'select_disc_pixels',
]

DISC_CLEARANCE_PX = 3.0
"""How far inside the edge of the lit region a pixel must lie to count, in pixels.

About two widths of the narrow-angle camera's point-spread function at half maximum, so
the blur of the limb and the terminator leaves the counted pixels alone.
"""

MIN_DISC_COSINE = 0.05
"""The smallest cosine of incidence or emission a counted pixel may have."""

MIN_FIT_PIXELS = 100
"""The fewest counted pixels a fit accepts."""

EQUATOR_HALF_WIDTH_DEG = 8.0
"""How far from the photometric equator a pixel may lie and enter its profile."""

LOW_PHASE_DEG = 5.0
"""Below this phase the profile is binned by the emission cosine.

The luminance coordinates divide by the sine of the phase, so near zero phase a small
error in either cosine moves a pixel far along the photometric equator.
"""

_LONGITUDE_EDGES_DEG = np.arange(-90.0, 90.0 + 1e-9, 5.0)
_MU_EDGES = np.linspace(0.0, 1.0, 21)
_MIN_BIN_PIXELS = 8
_MIN_PROFILE_BINS = 3

# The robust fits: reject pixels further than _CLIP_SIGMAS robust standard deviations
# from the line and refit, at most _CLIP_ROUNDS times.
_CLIP_SIGMAS = 3.0
_CLIP_ROUNDS = 6
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class MinnaertFit:
    """Minnaert's law fitted to one disc.

    Parameters:
        k: The exponent.
        b0: The brightness the law gives at ``mu0 = mu = 1``.
        k_sigma: The exponent's standard error from the fit's residuals.
        n_pixels: The pixels the fit kept.
    """

    k: float
    b0: float
    k_sigma: float
    n_pixels: int


@dataclass(frozen=True)
class LunarLambertFit:
    """The lunar-Lambert blend fitted to one disc.

    Parameters:
        blend: The Lommel-Seeliger share ``L``; it is not limited to [0, 1].
        b0: The brightness the law gives at ``mu0 = mu = 1``.
        blend_sigma: The blend's standard error from the fit's residuals.
        n_pixels: The pixels the fit kept.
    """

    blend: float
    b0: float
    blend_sigma: float
    n_pixels: int


def select_disc_pixels(
    on_body: NDArrayBoolType,
    mu0: NDArrayFloatType,
    mu: NDArrayFloatType,
    *,
    usable: NDArrayBoolType,
    clearance_px: float = DISC_CLEARANCE_PX,
) -> NDArrayBoolType:
    """The pixels of a body's lit disc that a profile measurement counts.

    Parameters:
        on_body: Which pixels the body covers.
        mu0: The cosine of the incidence angle at each pixel.
        mu: The cosine of the emission angle at each pixel.
        usable: Which pixels hold a value the measurement may use: finite, positive,
            and below the camera's full scale.
        clearance_px: How far inside the edge of the lit region a pixel must lie.

    Returns:
        A mask, in the arrays' shape, of the lit body pixels at least ``clearance_px``
        from the edge of the lit region whose two cosines are both at least
        :data:`MIN_DISC_COSINE` and that ``usable`` allows.
    """
    lit = np.asarray(on_body, dtype=bool) & (mu0 > 0.0) & (mu > 0.0)
    depth = scipy.ndimage.distance_transform_edt(lit)
    keep = (
        lit
        & (depth >= clearance_px)
        & (mu0 >= MIN_DISC_COSINE)
        & (mu >= MIN_DISC_COSINE)
        & np.asarray(usable, dtype=bool)
    )
    return np.asarray(keep, dtype=bool)


def _robust_line(
    design: NDArrayFloatType, target: NDArrayFloatType, *, clip: bool = True
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayBoolType] | None:
    """Least squares with iterative outlier rejection.

    Parameters:
        design: The (N, P) design matrix.
        target: The (N,) values to fit.
        clip: Whether to reject outliers; without it every point is kept.

    Returns:
        The coefficients, their covariance from the kept points' residuals, and which
        points were kept.  None when fewer than P points are given or kept, or when the
        kept points cannot fix every coefficient: the final solve's rank is below P.
    """
    n_coef = design.shape[1]
    if target.size < n_coef:
        return None
    keep = np.ones(target.size, dtype=bool)
    for _ in range(_CLIP_ROUNDS if clip else 0):
        coef, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        residual = target - np.sum(design * coef, axis=1)
        center = float(np.median(residual[keep]))
        spread = _MAD_TO_SIGMA * float(np.median(np.abs(residual[keep] - center)))
        new_keep = np.abs(residual - center) < _CLIP_SIGMAS * max(spread, 1e-9)
        if int(np.count_nonzero(new_keep)) < n_coef:
            return None
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coef, _, rank, _ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
    if int(rank) < n_coef:
        return None
    residual = target[keep] - np.sum(design[keep] * coef, axis=1)
    dof = max(int(keep.sum()) - n_coef, 1)
    variance = float(np.sum(residual**2)) / dof
    # The covariance is variance * (A^T A)^-1 for the kept design A, taken from A's own
    # singular values, s_i^-2 v_i v_i^T summed, so a narrow spread in the design keeps
    # its condition number instead of squaring it as the normal matrix would.  The rank
    # check above leaves every singular value above the solve's own cutoff.  The sum is
    # of elementwise products because the simulator package admits no NumPy matrix
    # product (its guard test enforces it).  The solves and the decomposition still go
    # through LAPACK, so the fits' last bits may differ between CPUs; they feed only the
    # realism record, and never a render.
    _, singular, vt = np.linalg.svd(design[keep], full_matrices=False)
    outer = vt[:, :, np.newaxis] * vt[:, np.newaxis, :]
    covariance = variance * np.sum(outer / (singular**2)[:, np.newaxis, np.newaxis], axis=0)
    return coef, covariance, keep


def _minnaert(
    intensity: NDArrayFloatType, mu0: NDArrayFloatType, mu: NDArrayFloatType, *, clip: bool
) -> tuple[MinnaertFit, NDArrayBoolType] | None:
    """Minnaert's law fitted to a disc's pixels, and which pixels the fit kept.

    Parameters:
        intensity: Each pixel's brightness, in any units, positive.
        mu0: The cosine of the incidence angle at each pixel, positive.
        mu: The cosine of the emission angle at each pixel, positive.
        clip: Whether the fit rejects outliers.

    Returns:
        The fit and its kept pixels, or None as :func:`fit_minnaert` gives it.
    """
    x = np.log(mu0 * mu)
    design = np.column_stack([np.ones_like(x), x])
    solved = _robust_line(design, np.log(intensity * mu), clip=clip)
    if solved is None:
        return None
    coef, covariance, keep = solved
    if int(keep.sum()) < MIN_FIT_PIXELS:
        return None
    fit = MinnaertFit(
        k=float(coef[1]),
        b0=float(np.exp(coef[0])),
        k_sigma=float(math.sqrt(covariance[1, 1])),
        n_pixels=int(keep.sum()),
    )
    return fit, keep


def fit_minnaert(
    intensity: NDArrayFloatType, mu0: NDArrayFloatType, mu: NDArrayFloatType
) -> MinnaertFit | None:
    """Fit Minnaert's law to a disc's counted pixels.

    Parameters:
        intensity: Each pixel's brightness, in any units, positive.
        mu0: The cosine of the incidence angle at each pixel, positive.
        mu: The cosine of the emission angle at each pixel, positive.

    Returns:
        The fit, or None when the fit keeps fewer than :data:`MIN_FIT_PIXELS` pixels or
        the kept pixels cannot fix both coefficients.
    """
    solved = _minnaert(intensity, mu0, mu, clip=True)
    return None if solved is None else solved[0]


def fit_minnaert_together(
    discs: Sequence[tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]],
) -> list[MinnaertFit | None]:
    """Fit Minnaert's law to several images of one disc over the same pixels.

    Each disc is fitted on its own first, as :func:`fit_minnaert` fits it.  Then every
    disc is fitted again, without rejecting any pixel, over the pixels that all of those
    fits kept.  So each exponent describes the same part of the disc.  When every fit
    keeps every pixel, each disc's exponent is the one its own fit gave.

    Parameters:
        discs: Each disc's ``(intensity, mu0, mu)``, as :func:`fit_minnaert` takes them,
            over the same pixels in the same order.

    Returns:
        One fit per disc, in order.  A disc whose own fit fails gives None and does
        not narrow the others' pixels.  Every disc gives None when fewer than
        :data:`MIN_FIT_PIXELS` pixels are kept by every fit that succeeded.

    Raises:
        ValueError: If the discs do not all hold the same number of pixels.
    """
    sizes = {np.size(array) for disc in discs for array in disc}
    if len(sizes) > 1:
        raise ValueError(f'the discs hold different numbers of pixels: {sorted(sizes)}')
    solved = [_minnaert(*disc, clip=True) for disc in discs]
    shared = np.ones(sizes.pop() if sizes else 0, dtype=bool)
    for one in solved:
        if one is not None:
            shared &= one[1]
    fits: list[MinnaertFit | None] = []
    for (intensity, mu0, mu), one in zip(discs, solved, strict=True):
        refit = (
            None
            if one is None
            else _minnaert(intensity[shared], mu0[shared], mu[shared], clip=False)
        )
        fits.append(None if refit is None else refit[0])
    return fits


def fit_lunar_lambert(
    intensity: NDArrayFloatType, mu0: NDArrayFloatType, mu: NDArrayFloatType
) -> LunarLambertFit | None:
    """Fit the lunar-Lambert blend to a disc's counted pixels.

    The law is ``B0 * mu0 + B0 * L * mu0 * (2 / (mu0 + mu) - 1)``, linear in ``B0`` and
    ``B0 * L``, and each pixel is weighted by the inverse of its brightness so that the
    fit compares relative residuals.

    Parameters:
        intensity: Each pixel's brightness, in any units, positive.
        mu0: The cosine of the incidence angle at each pixel, positive.
        mu: The cosine of the emission angle at each pixel, positive.

    Returns:
        The fit, or None when the fit keeps fewer than :data:`MIN_FIT_PIXELS` pixels,
        the kept pixels cannot fix both coefficients, or the Lambert term comes out
        zero.
    """
    weight = 1.0 / intensity
    design = np.column_stack([mu0, mu0 * (2.0 / (mu0 + mu) - 1.0)]) * weight[:, np.newaxis]
    solved = _robust_line(design, np.ones_like(intensity))
    if solved is None:
        return None
    coef, covariance, keep = solved
    if int(keep.sum()) < MIN_FIT_PIXELS or coef[0] == 0.0:
        return None
    a, b = float(coef[0]), float(coef[1])
    blend = b / a
    # The ratio's standard error, propagated through its gradient (-b / a**2, 1 / a),
    # which holds for a zero blend as well.
    variance = (
        covariance[0, 0] * b**2 / a**4 + covariance[1, 1] / a**2 - 2.0 * covariance[0, 1] * b / a**3
    )
    return LunarLambertFit(
        blend=blend,
        b0=a,
        blend_sigma=float(math.sqrt(max(variance, 0.0))),
        n_pixels=int(keep.sum()),
    )


def photometric_coordinates(
    mu0: NDArrayFloatType, mu: NDArrayFloatType, phase_rad: float
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Luminance longitude and latitude from the two cosines and the phase angle.

    The longitude runs from ``phase - 90`` degrees at the terminator through 0 at the
    sub-observer point to +90 at the sunlit limb.  The latitude's sign is not
    recoverable from the two cosines, so it is returned as its absolute value.

    Parameters:
        mu0: The cosine of the incidence angle at each pixel.
        mu: The cosine of the emission angle at each pixel.
        phase_rad: The phase angle in radians, above 0.

    Returns:
        The longitude and the absolute latitude, in degrees.
    """
    sin_phase = max(math.sin(phase_rad), 1e-9)
    toward_sun = (mu0 - mu * math.cos(phase_rad)) / sin_phase
    longitude = np.degrees(np.arctan2(toward_sun, mu))
    cos_latitude = np.clip(np.sqrt(mu**2 + toward_sun**2), 0.0, 1.0)
    return longitude, np.degrees(np.arccos(cos_latitude))


def equator_profile(
    intensity: NDArrayFloatType, mu0: NDArrayFloatType, mu: NDArrayFloatType, phase_rad: float
) -> NDArrayFloatType:
    """The median brightness in bins across the disc.

    Above :data:`LOW_PHASE_DEG` the bins are 5 degrees of luminance longitude, from -90
    to +90, over the pixels within :data:`EQUATOR_HALF_WIDTH_DEG` of the photometric
    equator.  Below it they are 20 equal bins of ``mu`` from 0 to 1 over every pixel.

    Parameters:
        intensity: Each counted pixel's brightness.
        mu0: The cosine of the incidence angle at each pixel.
        mu: The cosine of the emission angle at each pixel.
        phase_rad: The phase angle in radians.

    Returns:
        The median in each bin, NaN where a bin holds fewer than 8 pixels.
    """
    if math.degrees(phase_rad) < LOW_PHASE_DEG:
        coordinate, edges, member = mu, _MU_EDGES, np.ones(mu.shape, dtype=bool)
    else:
        longitude, latitude = photometric_coordinates(mu0, mu, phase_rad)
        coordinate, edges, member = (
            longitude,
            _LONGITUDE_EDGES_DEG,
            latitude < EQUATOR_HALF_WIDTH_DEG,
        )
    selected = coordinate[member]
    index = np.digitize(selected, edges) - 1
    # The top edge closes the last bin: a head-on pixel (mu = 1) and a pixel on the
    # sunlit limb (+90 deg) belong in it.
    index[selected == edges[-1]] = edges.size - 2
    values = intensity[member]
    out = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        in_bin = values[index == b]
        if in_bin.size >= _MIN_BIN_PIXELS:
            out[b] = float(np.median(in_bin))
    return out


def profile_residual(real: NDArrayFloatType, twin: NDArrayFloatType) -> float | None:
    """The rms difference between two profiles' shapes.

    Over the bins where both profiles are finite and positive, the ratio of the real
    profile to the twin's is divided by its median, and the residual is the rms of that
    ratio less 1.

    Parameters:
        real: One profile from :func:`equator_profile`.
        twin: Another, binned the same way.

    Returns:
        The residual, or None when fewer than 3 bins are shared.
    """
    shared = np.isfinite(real) & np.isfinite(twin) & (real > 0.0) & (twin > 0.0)
    if int(shared.sum()) < _MIN_PROFILE_BINS:
        return None
    ratio = real[shared] / twin[shared]
    ratio = ratio / float(np.median(ratio))
    return float(np.sqrt(np.mean((ratio - 1.0) ** 2)))
