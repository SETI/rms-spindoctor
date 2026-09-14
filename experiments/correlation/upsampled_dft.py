import numpy as np
from numpy.fft import fft2, ifft2

from spindoctor.support.correlate import int_to_signed, upsampled_dft


def gaussian_patch(
    shape: tuple[int, int], sigma: float, offset: tuple[float, float]
) -> np.ndarray:
    """Return a Gaussian patch whose peak sits at ``offset`` from the center.

    Parameters:
        shape: The ``(v, u)`` patch size in pixels.
        sigma: The Gaussian standard deviation in pixels.
        offset: The ``(v, u)`` peak displacement from the patch center, in
            pixel corner coordinates -- the convention psfmodel's ``eval_rect``
            uses, measured from the upper left corner of the center pixel, so
            ``(0.5, 0.5)`` puts the peak on that pixel's center and a whole
            number puts it on a pixel boundary.  All three scripts in this
            directory name their shifts this way.

    Returns:
        The patch, peak-normalized to 1.0.
    """
    v_size, u_size = shape
    ov, ou = offset
    cv = (v_size - 1) / 2.0 - 0.5
    cu = (u_size - 1) / 2.0 - 0.5
    vv, uu = np.meshgrid(np.arange(v_size), np.arange(u_size), indexing='ij')
    dv = vv - (cv + ov)
    du = uu - (cu + ou)
    g = np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))
    return g


def estimate_subpixel_shift(usfac: int, frac: float) -> float:
    shape = (64, 64)
    true_shift = (frac, 0.0)
    A = gaussian_patch(shape, sigma=2.0, offset=true_shift)
    B = gaussian_patch(shape, sigma=2.0, offset=(0.0, 0.0))
    X = fft2(A) * np.conj(fft2(B))

    # For X = fft2(A) * conj(fft2(B)) with A displaced by +frac, the correlation
    # peak is at lag +frac, so the nearest integer lag is +1 once frac passes a
    # half, not -1.  Take it from the real argmax through int_to_signed, which is
    # what support.correlate does, rather than hardcoding a lag here: with the
    # window centered on the wrong lag the argmax pins to the window edge and the
    # table reads as a ~1 px error in the refinement itself.
    # Use region that scales with upsample factor
    region = usfac + 1
    corr = np.real(ifft2(X))
    peak_v, _peak_u = np.unravel_index(np.argmax(corr), corr.shape)
    dy_i = int_to_signed(int(peak_v), X.shape[0])
    oy = region // 2
    Up = upsampled_dft(X, usfac, (region, region), (oy - dy_i * usfac, oy))
    upy, _ = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (upy - oy) / usfac
    return float(dy)


if __name__ == '__main__':
    usfacs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    # Include grid-aligned fractions and additional non-aligned fractions
    fracs = [
        0.00,
        0.125,
        0.25,
        0.375,
        0.50,
        0.625,
        0.75,
        0.875,
        0.07,
        0.11,
        0.19,
        0.23,
        0.31,
        0.44,
        0.58,
        0.67,
        0.73,
        0.86,
        0.99,
    ]
    # The true displacement is +frac for every frac in [0, 1): a shift of +0.75
    # is a lag of +0.75, not -0.25.  Wrapping only folds at half the transform
    # length (32 px here), which no entry in this table reaches.
    print('usfac, frac, est_dy, gt_signed, abs_err')
    for usfac in usfacs:
        for frac in fracs:
            gt_signed = frac
            dy = estimate_subpixel_shift(usfac, frac)
            err = abs(dy - gt_signed)
            print(f'{usfac:>5d}, {frac:5.3f}, {dy:8.4f}, {gt_signed:8.4f}, {err:8.5f}')
