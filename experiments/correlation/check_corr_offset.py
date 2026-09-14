import numpy as np
from psfmodel.gaussian import GaussianPSF

from spindoctor.support.correlate import navigate_with_pyramid_kpeaks


def try_corr_offset(image, model, mask, **kwargs):
    result = navigate_with_pyramid_kpeaks(image, model, mask, **kwargs)
    return result['offset']


def main():
    gauss_psf = GaussianPSF(sigma=2.0)

    # ``eval_rect``'s offset is measured from the upper left corner of the centre
    # pixel -- its own default (0.5, 0.5) sits on that pixel's centre -- so these
    # are pixel corner offsets, the same convention gaussian_patch uses in the
    # other two scripts here.  Only the image-minus-model difference is measured,
    # so the shift under test is (0.3, 0.0) px.
    image_size = (100, 100)
    image_psf_size = (7, 7)
    image_offset = (0.3, 0.0)

    model_size = (100, 100)
    model_psf_size = (31, 31)
    model_offset = (0.0, 0.0)

    image_psf = gauss_psf.eval_rect(image_psf_size, offset=image_offset, scale=1.0)
    image = np.zeros(image_size)
    image_center_u = int(image_size[1] // 2)
    image_center_v = int(image_size[0] // 2)
    image_psf_half_size_u = int(image_psf_size[1] // 2)
    image_psf_half_size_v = int(image_psf_size[0] // 2)
    image[
        image_center_v - image_psf_half_size_v : image_center_v + image_psf_half_size_v + 1,
        image_center_u - image_psf_half_size_u : image_center_u + image_psf_half_size_u + 1,
    ] = image_psf

    model_psf = gauss_psf.eval_rect(model_psf_size, offset=model_offset, scale=1.0)
    model = np.zeros(model_size)
    model_center_u = int(model_size[1] // 2)
    model_center_v = int(model_size[0] // 2)
    model_psf_half_size_u = int(model_psf_size[1] // 2)
    model_psf_half_size_v = int(model_psf_size[0] // 2)
    model[
        model_center_v - model_psf_half_size_v : model_center_v + model_psf_half_size_v + 1,
        model_center_u - model_psf_half_size_u : model_center_u + model_psf_half_size_u + 1,
    ] = model_psf
    mask = np.zeros(model_size)
    mask[
        model_center_v - model_psf_half_size_v : model_center_v + model_psf_half_size_v + 1,
        model_center_u - model_psf_half_size_u : model_center_u + model_psf_half_size_u + 1,
    ] = 1.0

    # plt.imshow(image)
    # plt.figure()
    # plt.imshow(model)
    # plt.show()
    result = try_corr_offset(image, model, mask, upsample_factor=64)
    print(result)


if __name__ == '__main__':
    main()
