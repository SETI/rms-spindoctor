"""Whole-scene point-spread function for the optics stage.

The camera's PSF blurs the entire composed radiance image, so the limb
gradient, the ring-edge gradient, and every star shape inherit one profile.
The kernel is a core Gaussian plus a Moffat wing::

    K(r) = (1 - w) * G_norm(r; sigma_v, sigma_u) + w * M_norm(r; r0, n)

Each term is separately normalized to unit sum over the truncation window, so
``w`` is exactly the fraction of the kernel's energy in the wing and the whole
kernel conserves flux.  The Gaussian core is elliptical (``sigma_v`` may differ
from ``sigma_u``); the Moffat wing is isotropic.  All radii are in detector
pixels; the kernel is sampled on the oversampled render grid, and the whole
convolution runs there before the box downsample.

A plane is usually zero away from the bodies, rings, and stars.  The blur
therefore convolves each cluster of lit pixels on its own crop (see the
lit_clusters module) and adds the results.  Convolution is linear, so the sum is
the blur of the whole plane.  A crop's blur reaches the kernel's half-width past
the crop and is exactly zero beyond it.  The FFTs are smaller than the frame's,
so they round differently from one whole-frame convolution: by up to about 1e-15
of the peak on lit pixels.  In the dark, where the whole-frame FFT leaves residue
of that size, the crops leave exact zeros.  Each crop's FFT covers the crop
widened by the kernel.  When those areas would add up to at least the frame's
area widened the same way, the plane is blurred in one piece.
"""

from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve

from spindoctor.sim.forward.lit_clusters import cluster_crop, find_lit_clusters
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'apply_psf',
    'binned_psf_kernel',
    'psf_kernel',
]


@lru_cache(maxsize=8)
def psf_kernel(
    sigma_v: float,
    sigma_u: float,
    w: float,
    r0: float,
    n: float,
    *,
    truncation_px: int,
    oversample: int,
) -> NDArrayFloatType:
    """Build the core-plus-wing PSF kernel on the oversampled grid.

    Parameters:
        sigma_v: Gaussian core sigma along v, in detector pixels.
        sigma_u: Gaussian core sigma along u, in detector pixels.
        w: Wing energy fraction in [0, 1] (the Moffat term's total weight).
        r0: Moffat core radius in detector pixels.
        n: Moffat index.
        truncation_px: Kernel half-width in detector pixels.
        oversample: Oversampling factor of the render grid.

    Returns:
        A ``(2*truncation_px*os + 1)`` square kernel summing to 1.0.
    """
    radius_os = int(truncation_px) * int(oversample)
    offsets = np.arange(-radius_os, radius_os + 1, dtype=np.float64)
    dv_os, du_os = np.meshgrid(offsets, offsets, indexing='ij')
    # The kernel is sampled on the oversampled grid but parameterized in
    # detector pixels, so convert each subsample offset back to detector units.
    dv_det = dv_os / oversample
    du_det = du_os / oversample

    gaussian = np.exp(-0.5 * ((dv_det / sigma_v) ** 2 + (du_det / sigma_u) ** 2))
    gaussian /= gaussian.sum()

    r_det = np.sqrt(dv_det**2 + du_det**2)
    moffat = (1.0 + (r_det / r0) ** 2) ** (-n / 2.0)
    moffat /= moffat.sum()

    kernel: NDArrayFloatType = (1.0 - w) * gaussian + w * moffat
    return kernel


def apply_psf(
    signal: NDArrayFloatType,
    point_e: NDArrayFloatType,
    *,
    sigma_v: float,
    sigma_u: float,
    w: float,
    r0: float,
    n: float,
    truncation_px: int,
    oversample: int,
) -> None:
    """Convolve the signal and point-source planes with the PSF kernel in place.

    Both planes share every optical transform, so the same kernel blurs the
    intensive signal (bodies, rings, sky) and the point-source electrons.  Each
    plane is blurred by FFT convolution, cluster by cluster as the module
    docstring describes.

    Parameters:
        signal: The oversampled intensive-signal plane, modified in place.
        point_e: The oversampled point-source plane, modified in place.
        sigma_v: Gaussian core sigma along v, in detector pixels.
        sigma_u: Gaussian core sigma along u, in detector pixels.
        w: Wing energy fraction in [0, 1].
        r0: Moffat core radius in detector pixels.
        n: Moffat index.
        truncation_px: Kernel half-width in detector pixels.
        oversample: Oversampling factor of the render grid.
    """
    kernel = psf_kernel(
        float(sigma_v),
        float(sigma_u),
        float(w),
        float(r0),
        float(n),
        truncation_px=int(truncation_px),
        oversample=int(oversample),
    )
    _blur_plane(signal, kernel)
    _blur_plane(point_e, kernel)


def _blur_plane(plane: NDArrayFloatType, kernel: NDArrayFloatType) -> None:
    """Convolve one plane with an odd-sized kernel in place, cluster by cluster.

    Parameters:
        plane: The render-grid plane, modified in place.
        kernel: The square kernel, of odd size ``2 * half + 1``.
    """
    clusters = find_lit_clusters(plane)
    if not clusters.boxes:
        return
    size_v, size_u = plane.shape
    half = kernel.shape[0] // 2
    # An FFT convolution costs about the area it transforms: each crop grown by
    # the kernel's width, against the whole frame grown by the same.
    cluster_area = sum(
        (v.stop - v.start + 2 * half) * (u.stop - u.start + 2 * half) for v, u in clusters.boxes
    )
    if cluster_area >= (size_v + 2 * half) * (size_u + 2 * half):
        plane[:] = fftconvolve(plane, kernel, mode='same')
        return
    source = plane.copy()
    plane[:] = 0.0
    for index, (rows, cols) in enumerate(clusters.boxes):
        blurred = fftconvolve(cluster_crop(clusters, source, index, rows, cols), kernel)
        # The full convolution's first sample sits ``half`` pixels before the
        # crop; clip it to the frame.
        out_v = slice(max(rows.start - half, 0), min(rows.stop + half, size_v))
        out_u = slice(max(cols.start - half, 0), min(cols.stop + half, size_u))
        first_v = out_v.start - (rows.start - half)
        first_u = out_u.start - (cols.start - half)
        plane[out_v, out_u] += blurred[
            first_v : first_v + out_v.stop - out_v.start,
            first_u : first_u + out_u.stop - out_u.start,
        ]


def binned_psf_kernel(kernel: Mapping[str, float], binning: int) -> dict[str, float]:
    """A native PSF kernel restated in pixels binned by ``binning``.

    The radii (``sigma_v``, ``sigma_u``, ``r0``) shrink by the factor; the wing
    energy fraction ``w`` and the wing index ``n`` are dimensionless.  The
    binned pixel's own b-by-b box is the render grid's pixel integration, so
    it needs no term here.

    Parameters:
        kernel: The native kernel, radii in native pixels.
        binning: The binning factor.

    Returns:
        A new kernel mapping in binned pixels.
    """
    scaled = dict(kernel)
    for radius in ('sigma_v', 'sigma_u', 'r0'):
        if radius in scaled:
            scaled[radius] = float(scaled[radius]) / binning
    return scaled
