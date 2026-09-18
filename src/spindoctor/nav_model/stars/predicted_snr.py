"""Raw-DN photometry helpers for stars (predicted-SNR diagnostic).

The star NavModel no longer gates on this DN-based SNR: STAR features are
selected purely by magnitude against ``obs.star_max_usable_vmag()`` (see
``spindoctor.nav_model.stars.nav_model_stars``), which carries no dependence on
any DN-to-image-unit scale.  This module is retained for its reusable
photometry helpers — ``psf_sigma_px`` (imported by ``detection`` and
``nav_model_body``), ``psf_aperture_pixels``, ``integrated_signal_dn``,
and the ``SCLASS_TO_B_MINUS_V`` re-export — and the ``predicted_snr``
formula is kept as a raw-DN diagnostic, not used by the navigator's star
gate.

The (diagnostic) ``predicted_snr`` estimate of how detectable a star is
at its predicted pixel position uses three inputs:

- ``obs.star_psf()`` — the per-camera-per-filter PSF.
- ``NavContext.image_noise_sigma`` — robust MAD-based noise estimate over
  the sensor area in the image's native units.
- ``star.dn`` — the integrated DN expected for the star in the
  instrument's bandpass, computed from its catalog V magnitude via the
  ``2.512 ** -(vmag - 4)`` flux-to-DN scaling.  ``star.dn`` is the
  total signal across the PSF support; the per-pixel signal is that
  total spread over a circular Gaussian-PSF aperture.

The integrated SNR follows the form in Part 1's "Position covariance
per feature type" section:

::

    SNR = total_signal / sqrt(total_signal + read_noise**2 * N_aperture)

with ``total_signal`` in DN and ``read_noise**2 * N_aperture`` standing
in for the variance contribution from background and read noise.
``image_noise_sigma`` is treated as a Gaussian read-noise proxy because
the MAD estimator is dominated by background pixels and combines shot,
read, and dark contributions into a single per-pixel sigma.

For raw-DN instruments the catalog signal and ``image_noise_sigma`` are
already in the same units (DN) and the formula applies directly.  For
calibrated-IF instruments (Cassini ISS ``_CALIB.IMG`` and similar) the
image's noise sigma is in I/F while the catalog signal is still in DN;
``signal_dn_to_image_unit_scale`` (the per-camera DN-to-image-unit
factor) converts ``image_noise_sigma`` back to a DN-equivalent before
the SNR is formed.  Without this conversion the SNR for every catalog
star collapses to ``sqrt(signal_dn)`` on calibrated images and the
reliability gate drops them all.

``SCLASS_TO_B_MINUS_V`` is re-exported here so callers that need the
spectral-class color mapping can pull it from the same module that
owns the predicted-SNR formula.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from starcat import SCLASS_TO_B_MINUS_V

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

    from spindoctor.support.types import MutableStar

__all__ = [
    'SCLASS_TO_B_MINUS_V',
    'integrated_signal_dn',
    'predicted_snr',
    'psf_aperture_pixels',
    'psf_sigma_px',
]


def psf_sigma_px(psf: PSF) -> float:
    """Return the Gaussian-equivalent sigma of ``psf`` in pixels.

    Treats every PSF as a 2-D Gaussian for SNR / CRLB purposes.  The
    pipeline ships ``psfmodel.GaussianPSF`` instances populated from
    ``star_psf_sigma`` in ``config_4N0_inst_*.yaml``; that class exposes
    per-axis ``sigma_x`` / ``sigma_y`` attributes (typically equal).
    When neither per-axis sigma is available we fall back to a single
    ``sigma`` attribute (legacy interface) or ``fwhm() / 2.3548`` (a
    third-party PSF subclass).

    Parameters:
        psf: PSF instance from ``obs.star_psf()``.

    Returns:
        Gaussian sigma in pixels.  When the PSF is anisotropic, the
        per-axis values are averaged.
    """
    sigma_x = getattr(psf, 'sigma_x', None)
    sigma_y = getattr(psf, 'sigma_y', None)
    if sigma_x is not None and sigma_y is not None:
        return float((float(sigma_x) + float(sigma_y)) / 2.0)
    if hasattr(psf, 'sigma'):
        return float(cast(float, psf.sigma))
    fwhm_method = getattr(psf, 'fwhm', None)
    if callable(fwhm_method):
        return float(fwhm_method()) / 2.3548200450309493
    raise AttributeError(
        f'PSF {type(psf).__name__} exposes neither sigma_x/sigma_y, sigma, nor fwhm()'
    )


def psf_aperture_pixels(sigma_px_value: float) -> float:
    """Return the effective number of pixels in the PSF support.

    Uses the standard "noise-equivalent area" of a 2-D Gaussian,
    ``4 * pi * sigma**2``, which is the right scale for converting an
    integrated DN signal into a per-pixel matched-filter SNR.

    Parameters:
        sigma_px_value: Per-pixel PSF sigma in pixels.

    Returns:
        Effective aperture area in pixels (always > 0 for sigma > 0).
    """
    if sigma_px_value <= 0.0:
        raise ValueError(f'sigma_px_value must be > 0; got {sigma_px_value!r}')
    return 4.0 * math.pi * sigma_px_value * sigma_px_value


def integrated_signal_dn(star: MutableStar, mag_offset: float) -> float:
    """Return the predicted in-band DN for a catalog star.

    Applies a per-camera-per-filter ``mag_offset`` to convert the
    catalog V-band magnitude into the instrument's bandpass, then uses
    the standard ``2.512 ** -(vmag - 4)`` flux-to-DN scaling.  Stars
    without a catalog magnitude (vmag is None) are not detectable and
    return 0.0.

    Parameters:
        star: Star record carrying ``vmag``.
        mag_offset: Per-instrument-per-filter magnitude offset
            (mag_in_band - mag_v).  Positive values mean the instrument
            sees the star fainter than the catalog magnitude.

    Returns:
        Predicted integrated DN in the instrument's bandpass.
    """
    if star.vmag is None:
        return 0.0
    in_band_mag = float(star.vmag) + mag_offset
    return float(2.512 ** -(in_band_mag - 4.0))


def predicted_snr(
    star: MutableStar,
    *,
    psf: PSF,
    image_noise_sigma: float,
    mag_offset: float = 0.0,
    signal_dn_to_image_unit_scale: float = 1.0,
) -> float:
    """Predicted integrated SNR for a star at its predicted pixel.

    Treats ``image_noise_sigma`` as the per-pixel Gaussian noise from
    background + read noise + dark current; ``star.dn`` is the
    integrated signal across the PSF support.  Implements the formula
    from the design's STAR section:

    ::

        SNR = total_signal / sqrt(total_signal + sigma_dn**2 * N_aperture)

    with ``N_aperture = 4 * pi * sigma_PSF**2``.  ``sigma_dn`` is the
    DN-equivalent of ``image_noise_sigma`` obtained by dividing through
    ``signal_dn_to_image_unit_scale``; for raw-DN instruments the scale
    is 1.0 and ``sigma_dn == image_noise_sigma`` so the formula reduces
    to the classic form.  For calibrated-IF instruments the scale is
    typically of order ``1e-7`` (DN-to-I/F).

    Parameters:
        star: Star record.
        psf: PSF (typically from ``obs.star_psf()``).
        image_noise_sigma: Robust per-pixel noise sigma in the image's
            native units (DN for raw, I/F for calibrated).
        mag_offset: Catalog-to-instrument magnitude offset (default 0).
        signal_dn_to_image_unit_scale: Scale that converts a DN signal
            into the same units as ``image_noise_sigma``.  ``1.0`` for
            raw-DN instruments; per-camera value loaded from
            ``noise.signal_dn_to_image_unit_scale`` for calibrated-IF
            instruments.

    Returns:
        Predicted SNR (dimensionless, >= 0).

    Raises:
        ValueError: If ``image_noise_sigma`` or
            ``signal_dn_to_image_unit_scale`` is non-positive.
    """
    if image_noise_sigma <= 0.0:
        raise ValueError(f'image_noise_sigma must be > 0; got {image_noise_sigma!r}')
    if signal_dn_to_image_unit_scale <= 0.0:
        raise ValueError(
            f'signal_dn_to_image_unit_scale must be > 0; got {signal_dn_to_image_unit_scale!r}'
        )
    sig = integrated_signal_dn(star, mag_offset)
    if sig <= 0.0:
        return 0.0
    sigma_px_value = psf_sigma_px(psf)
    aperture = psf_aperture_pixels(sigma_px_value)
    image_noise_sigma_dn = image_noise_sigma / signal_dn_to_image_unit_scale
    variance = sig + image_noise_sigma_dn * image_noise_sigma_dn * aperture
    return sig / math.sqrt(variance)
