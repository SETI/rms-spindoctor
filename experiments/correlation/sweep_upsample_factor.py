import numpy as np

from spindoctor.support.correlate import navigate_with_pyramid_kpeaks


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
    g /= g.max() if g.max() > 0 else 1.0
    return g


def make_synthetic(
    image_size=(100, 100),
    model_size=(100, 100),
    psf_size=(7, 7),
    image_offset=(0.5, 0.0),
    model_offset=(0.0, 0.0),
    sigma=2.0,
):
    image = np.zeros(image_size, dtype=np.float64)
    model = np.zeros(model_size, dtype=np.float64)

    # Image PSF
    img_psf = gaussian_patch(psf_size, sigma=sigma, offset=image_offset)
    icu = image_size[1] // 2
    icv = image_size[0] // 2
    hsu = psf_size[1] // 2
    hsv = psf_size[0] // 2
    image[icv - hsv : icv + hsv + 1, icu - hsu : icu + hsu + 1] = img_psf

    # Model PSF
    mdl_psf = gaussian_patch(psf_size, sigma=sigma, offset=model_offset)
    mcu = model_size[1] // 2
    mcv = model_size[0] // 2
    model[mcv - hsv : mcv + hsv + 1, mcu - hsu : mcu + hsu + 1] = mdl_psf

    # Mask equals model support
    mask = np.zeros(model_size, dtype=bool)
    mask[mcv - hsv : mcv + hsv + 1, mcu - hsu : mcu + hsu + 1] = True

    return image, model, mask


def run_sweep(upsample_factors: list[int]) -> None:
    gt = (0.5, 0.0)
    image, model, mask = make_synthetic(image_offset=gt)
    print('usfac, est_dy, est_dx, err')
    for usfac in upsample_factors:
        res = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=1,
            max_peaks=1,
            upsample_factor=usfac,
            metric='psr',
            quality_thresh=-1e9,
            consistency_tol=1e9,
            nms_radius=3,
        )
        dy, dx = res['offset']
        err = float(np.hypot(dy - gt[0], dx - gt[1]))
        print(f'{usfac:>5d}, {dy:8.4f}, {dx:8.4f}, {err:8.5f}')


if __name__ == '__main__':
    run_sweep([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
