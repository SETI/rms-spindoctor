"""Summarize the star PSF fits of measured star fields per readout mode and filter.

The per-star fits come from :mod:`util.star_photometry.frame_measure`; the fitting program
:mod:`util.star_photometry.star_limit_fit` reports this summary beside the photometric limit.
The encircled-energy radii of a Gaussian and of a simulator kernel, measured the same way as
a real star's, are here too, for comparing a model with the stars: a kernel's noise-free
radii, and its radii on copies of the real stars drawn with their noise and rounding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from util.star_photometry.star_field import (
    KERNEL_OVERSAMPLE,
    QUANTIZATION_VARIANCE_DN2,
    KernelStarFit,
    StarCutout,
    encircled_energy_radii,
    integrated_gaussian,
    kernel_star_image,
)

__all__ = [
    'PSF_MAX_SMEAR_PX',
    'gaussian_ee',
    'kernel_encircled_energy',
    'kernel_encircled_energy_like_stars',
    'psf_summary',
]

#: Smallest smear (pixels) of a star excluded from the PSF summary.
PSF_MAX_SMEAR_PX = 0.5


def gaussian_ee(sigma: float, r_max: float, n_bins: int) -> tuple[float, float]:
    """Return EE50 and EE80 of a pixel-integrated Gaussian through the same estimator.

    Parameters:
        sigma: The Gaussian sigma (pixels).
        r_max: The profile's outer radius.
        n_bins: The profile's bins.

    Returns:
        ``(ee50, ee80)``.
    """
    half = math.ceil(r_max) + 8
    gv, gu = np.mgrid[-half : half + 1, -half : half + 1].astype(float)
    ee = [
        encircled_energy_radii(
            integrated_gaussian(gv, gu, cv, cu, sigma, sigma),
            half + cv,
            half + cu,
            r_max=r_max,
            n_bins=n_bins,
        )
        for cv, cu in ((0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.5, 0.5))
    ]
    image_ee = np.array(ee)
    return float(np.mean(image_ee[:, 0])), float(np.mean(image_ee[:, 1]))


def kernel_encircled_energy(
    kernel: dict[str, float],
    centers: Sequence[tuple[float, float]],
    *,
    size: int,
    truncation_px: int,
    r_max: float,
    n_bins: int,
    oversample: int = KERNEL_OVERSAMPLE,
) -> tuple[float, float]:
    """Return the median EE50 and EE80 of a kernel's stars at the given centers.

    Each star is drawn noise-free by :func:`kernel_star_image` and measured by
    :func:`encircled_energy_radii`.  Both radii depend on where the star falls within its
    pixel, so a kernel is compared with real stars at the real stars' own centers.

    Parameters:
        kernel: The kernel, with the keys of the simulator's catalog kernel.
        centers: Each star's ``(v, u)`` in a ``size``-square grid's array-index
            coordinates.
        size: The side of the grid each star is drawn on.
        truncation_px: The kernel's half-width in detector pixels.
        r_max: Outer radius of the profile (pixels).
        n_bins: Radial bins between 0 and ``r_max``.
        oversample: The render grid's oversampling.

    Returns:
        ``(ee50, ee80)`` in pixels, the medians over the stars.
    """
    radii = np.array(
        [
            encircled_energy_radii(
                kernel_star_image(
                    kernel, size, v, u, truncation_px=truncation_px, oversample=oversample
                ),
                v,
                u,
                r_max=r_max,
                n_bins=n_bins,
            )
            for v, u in centers
        ]
    )
    return float(np.median(radii[:, 0])), float(np.median(radii[:, 1]))


def kernel_encircled_energy_like_stars(
    kernel: dict[str, float],
    cutouts: Sequence[StarCutout],
    fits: Sequence[KernelStarFit],
    *,
    truncation_px: int,
    dn_max: float,
    r_max: float,
    n_bins: int,
    seed: int,
    draws: int,
    oversample: int = KERNEL_OVERSAMPLE,
) -> tuple[float, float]:
    """Return a kernel's EE50 and EE80 on stars rendered and measured like real ones.

    Each real star is drawn again with the kernel, at its fitted center, flux, and
    background and on its cutout's grid.  Its electrons take shot noise at the cutout's
    gain, the sky takes the noise the cutout's sky shows less the share of rounding, and
    the result is rounded to whole DN and clipped to the converter's range, as the
    camera's image is.  Each copy is then measured at its
    center by :func:`encircled_energy_radii`, as the real star is.  Rounding takes away
    the faint part of a narrow star's wing, so the radii measured this way can fall
    well below the noise-free ones of :func:`kernel_encircled_energy`.

    Parameters:
        kernel: The kernel, with the keys of the simulator's catalog kernel.
        cutouts: The real stars.
        fits: Each star's fit (its center, flux, and background), in the order of the
            cutouts.
        truncation_px: The kernel's half-width in detector pixels.
        dn_max: The largest DN the camera records.
        r_max: Outer radius of the profile (pixels).
        n_bins: Radial bins between 0 and ``r_max``.
        seed: The seed of the noise.
        draws: How many noise draws of every star to measure.
        oversample: The render grid's oversampling.

    Returns:
        ``(ee50, ee80)`` in pixels: for each draw the median over the stars, averaged
        over the draws.
    """
    rng = np.random.default_rng(seed)
    medians = []
    sky_sigmas = [
        math.sqrt(max(cutout.sky_var - QUANTIZATION_VARIANCE_DN2, 0.0)) for cutout in cutouts
    ]
    for _ in range(draws):
        radii = []
        for cutout, fit, sky_sigma in zip(cutouts, fits, sky_sigmas, strict=True):
            star = fit.flux * kernel_star_image(
                kernel,
                cutout.image.shape[0],
                fit.center_v,
                fit.center_u,
                truncation_px=truncation_px,
                oversample=oversample,
            )
            electrons = rng.poisson(np.clip(star, 0.0, None) * cutout.gain_e_per_dn)
            sky = fit.background + rng.normal(0.0, sky_sigma, star.shape)
            image = np.clip(np.round(electrons / cutout.gain_e_per_dn + sky), 0.0, dn_max)
            radii.append(
                encircled_energy_radii(
                    image, fit.center_v, fit.center_u, r_max=r_max, n_bins=n_bins
                )
            )
        medians.append(np.median(np.array(radii), axis=0))
    ee50, ee80 = np.mean(np.array(medians), axis=0)
    return float(ee50), float(ee80)


def psf_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the PSF fits per mode, and per mode and filter.

    Parameters:
        rows: Star rows.

    Returns:
        Group name to its summary, which carries the group's readout mode as ``mode``.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        psf = r.get('psf')
        if psf is None or r['smear_px'] > PSF_MAX_SMEAR_PX or not math.isfinite(psf['ee50']):
            continue
        groups[f'mode {r["mode"]}'].append(r)
        groups[f'mode {r["mode"]} filter {r["filter"]}'].append(r)
    out: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        if len(members) < 3:
            continue
        sig = np.array([m['psf']['sigma'] for m in members])
        major = np.array([m['psf']['sigma_major'] for m in members])
        minor = np.array([m['psf']['sigma_minor'] for m in members])
        ee50 = np.array([m['psf']['ee50'] for m in members])
        ee80 = np.array([m['psf']['ee80'] for m in members])
        resid = np.array([m['psf']['rms_residual'] for m in members])
        out[name] = {
            'mode': members[0]['mode'],
            'n_stars': len(members),
            'n_frames': len({m['image_id'] for m in members}),
            'sigma_median': float(np.median(sig)),
            'sigma_p16_p84': [float(np.percentile(sig, 16)), float(np.percentile(sig, 84))],
            'sigma_min': float(sig.min()),
            'sigma_p2': float(np.percentile(sig, 2)),
            'major_median': float(np.median(major)),
            'minor_median': float(np.median(minor)),
            'axis_ratio_median': float(np.median(major / minor)),
            'ee50_median': float(np.median(ee50)),
            'ee80_median': float(np.median(ee80)),
            'ee80_over_ee50': float(np.median(ee80) / np.median(ee50)),
            'fit_rms_residual_median': float(np.median(resid)),
        }
    return out
