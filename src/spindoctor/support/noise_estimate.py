"""Robust per-image noise estimate over the sensor area.

The autonomous-navigation orchestrator computes ``image_noise_sigma`` once per
image and stores it on ``NavContext`` so every extractor and technique uses
the same value.  This module owns the implementation.

The estimate is global -- it does not require knowledge of where any feature
lives in the image, and is therefore not biased by a wrong SPICE pointing
prediction.  It is computed from a 3x3 second-difference (Laplacian) response
so that smooth scene structure (ring brightness ramps, limb shading, a bright
extended disc) cancels and only pixel-to-pixel noise survives; the MAD over
that response further rejects the minority of pixels sitting on sharp edges or
cosmic rays.  A plain MAD of the raw intensities is *not* used: when the scene
is dominated by structure (for example rings filling the frame) it measures the
bright-to-dark spread rather than the noise and overestimates sigma by orders
of magnitude, which in turn pushes the edge-detection threshold above every
real gradient and empties the distance transform.
"""

import numpy as np

from spindoctor.support.misc import mad_std
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'estimate_background_and_sky_sigma',
    'estimate_image_noise_sigma',
]

# The 3x3 second-difference (Laplacian) kernel [[1, -2, 1], [-2, 4, -2],
# [1, -2, 1]] has coefficients whose sum of squares is 36, so for white noise
# of standard deviation ``s`` the response variance is ``36 * s**2``; dividing
# the robust response standard deviation by 6 recovers ``s``.  The double
# difference cancels any locally linear brightness ramp, so smooth scene
# content does not inflate the estimate.
_LAPLACIAN_NORM = 6.0


def estimate_image_noise_sigma(
    image: NDArrayFloatType, sensor_mask: NDArrayBoolType | None = None
) -> float:
    """Return a robust noise sigma over the sensor pixels.

    The estimate is the MAD-based standard deviation of a 3x3 second-difference
    (Laplacian) response, divided by 6.  The second difference cancels smooth
    brightness structure so the value reflects pixel-to-pixel noise rather than
    scene content, and the MAD rejects the minority of pixels on sharp edges or
    cosmic rays.

    Parameters:
        image: 2-D float input array.
        sensor_mask: Optional boolean mask with ``True`` for sensor pixels and
            ``False`` for extfov padding.  If ``None``, every pixel of
            ``image`` is treated as sensor data.  Only response pixels whose
            full 3x3 neighbourhood lies inside the mask contribute.

    Returns:
        Robust noise sigma in the same DN units as ``image``.

    Raises:
        TypeError: if ``image`` is not 2-D.
        ValueError: if no sensor pixels are available.
    """
    if image.ndim != 2:
        raise TypeError(f'estimate_image_noise_sigma requires a 2-D image; got ndim={image.ndim}')
    if sensor_mask is not None:
        if sensor_mask.shape != image.shape:
            raise ValueError(
                f'sensor_mask shape {sensor_mask.shape} differs from image shape {image.shape}'
            )
        if not bool(sensor_mask.any()):
            raise ValueError('sensor_mask selects no pixels')

    img = np.asarray(image, np.float64)

    # Images smaller than the 3x3 kernel cannot form a second difference; fall
    # back to a masked global MAD so a value is still returned.
    if img.shape[0] < 3 or img.shape[1] < 3:
        sensor = img if sensor_mask is None else img[sensor_mask]
        return float(mad_std(sensor.ravel()))

    # Laplacian response over every interior pixel (corners +1, edges -2,
    # center +4), aligned to image pixels [1:-1, 1:-1].
    response = (
        img[:-2, :-2]
        + img[:-2, 2:]
        + img[2:, :-2]
        + img[2:, 2:]
        - 2.0 * (img[:-2, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:] + img[2:, 1:-1])
        + 4.0 * img[1:-1, 1:-1]
    )

    if sensor_mask is None:
        samples = response.ravel()
        fallback = img.ravel()
    else:
        m = sensor_mask
        interior_valid = (
            m[:-2, :-2]
            & m[:-2, 1:-1]
            & m[:-2, 2:]
            & m[1:-1, :-2]
            & m[1:-1, 1:-1]
            & m[1:-1, 2:]
            & m[2:, :-2]
            & m[2:, 1:-1]
            & m[2:, 2:]
        )
        samples = response[interior_valid]
        fallback = img[sensor_mask].ravel()

    # A second difference touching a NaN missing-data marker is NaN, so drop
    # non-finite responses.  If none survive (no fully-finite 3x3 neighbourhood,
    # e.g. an image that is almost entirely markers), fall back to a global MAD
    # over the finite sensor pixels rather than failing.
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float(mad_std(fallback))

    return float(mad_std(finite) / _LAPLACIAN_NORM)


NOISE_CLIP_SIGMA: float = 3.0
"""Sigmas above background at which a pixel stops counting as sky.

Shared between the estimator's high-tail rejection (``clip_sigma``) and the
blob technique's lit-pixel threshold
(``nav_technique_body_blob._BLOB_NOISE_THRESHOLD_SIGMA``) so the model's
detection-SNR estimate and the technique's actual threshold cannot drift
apart.
"""


def estimate_background_and_sky_sigma(
    image: NDArrayFloatType,
    valid_mask: NDArrayBoolType,
    *,
    noise_sigma: float,
    clip_sigma: float = NOISE_CLIP_SIGMA,
) -> tuple[float, float]:
    """Estimate the frame's background pedestal and sky-noise sigma.

    Returns two frame-global quantities body-brightness photometry needs
    (the BODY_BLOB centroid moment in ``BodyBlobNav`` and the BODY_BLOB
    detection SNR in ``NavModelBodyBase`` share this estimate):

    * **Background (bias + dark) pedestal.** Real raw frames carry a
      non-zero bias/dark pedestal, and the sim adds a ``bias_dn`` pedestal
      so dark sky is distinguishable from the missing-data marker.  That
      pedestal sits on every pixel, so without subtracting it a
      brightness-weighted moment is pulled toward the bbox geometric
      center -- a bias that grows with phase as the lit signal
      concentrates away from center.
    * **Sky-noise sigma.** Lit-pixel thresholds must reject *sky* noise,
      but the global ``image_noise_sigma`` is a MAD over the whole
      sensor, which a bright body inflates (its brightness gradient widens
      the global spread).  An inflated sigma raises the threshold and cuts
      the dim crescent.  The sky noise estimated here, over the
      body-excluded sky population, is the correct floor; for a small body
      it equals the global sigma, so this only matters when a large body
      would otherwise inflate it.

    Both are estimated over the valid sky region (sensor data, excluding
    saturation / cosmic rays) by seeding at the overall median and the
    global sigma, then rejecting the bright body (the high tail) and
    re-estimating the median and MAD of the surviving sky population.  On
    a zero-background fixture this returns ``(~0, ~0)``.

    Parameters:
        image: The extended-FOV image (native units).
        valid_mask: ``True`` where the pixel is real sensor data and not
            saturated / a cosmic ray (the sky-candidate population).
        noise_sigma: Global per-pixel image noise sigma (native units);
            seeds the sky-population selection.
        clip_sigma: High-tail rejection threshold in sigmas; pixels above
            ``background + clip_sigma * sky_sigma`` are excluded from the
            sky population on each iteration.

    Returns:
        ``(background, sky_sigma)`` in the image's native units;
        ``(0.0, max(noise_sigma, 1e-9))`` when there are no valid pixels.
    """
    sigma0 = max(noise_sigma, 1e-9)
    vals = image[valid_mask]
    if vals.size == 0:
        return 0.0, sigma0
    background = float(np.median(vals))
    sky_sigma = sigma0
    for _ in range(3):
        sky = vals[vals <= background + clip_sigma * sky_sigma]
        if sky.size == 0:
            break
        new_bg = float(np.median(sky))
        new_sigma = max(1.4826 * float(np.median(np.abs(sky - new_bg))), 1e-9)
        converged = (
            abs(new_bg - background) <= 1e-3 * sigma0
            and abs(new_sigma - sky_sigma) <= 1e-3 * sigma0
        )
        background, sky_sigma = new_bg, new_sigma
        if converged:
            break
    return background, sky_sigma
