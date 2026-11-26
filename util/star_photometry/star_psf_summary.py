"""Summarize the star PSF fits of measured star fields per readout mode and filter.

The per-star fits come from :mod:`util.star_photometry.frame_measure`; the fitting program
:mod:`util.star_photometry.star_limit_fit` reports this summary beside the photometric limit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from util.star_photometry.star_field import encircled_energy_radii, integrated_gaussian

__all__ = ['PSF_MAX_SMEAR_PX', 'gaussian_ee', 'psf_summary']

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
