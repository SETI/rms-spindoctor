"""Tests for ``spindoctor.nav_technique._star_sharpness``."""

import numpy as np
import pytest
from psfmodel import GaussianPSF

from spindoctor.nav_technique._star_sharpness import (
    PsfConcentration,
    psf_concentration,
    sharper_than_psf,
)

_BACKGROUND = 100.0
_NOISE_SIGMA = 1.0


def _bright_floor() -> np.ndarray:
    """Return a 41x41 frame on a uniform floor far above the noise, with mild texture."""
    rng = np.random.default_rng(7)
    return _BACKGROUND + rng.normal(0.0, _NOISE_SIGMA, size=(41, 41))


# The Cassini NAC's star point-spread function: a Gaussian of sigma 0.54 px.
_NAC_PSF_SIGMA = 0.54


def _nac_limits() -> PsfConcentration:
    """Return the concentration limits of the NAC star PSF."""
    return psf_concentration(GaussianPSF(sigma=_NAC_PSF_SIGMA))


def _is_sharp(image: np.ndarray, v: int, u: int) -> bool:
    """Apply the NAC sharpness test at ``(v, u)`` with unit noise and a 4-sigma margin."""
    return sharper_than_psf(
        image, v, u, limits=_nac_limits(), noise_sigma=_NOISE_SIGMA, detection_sigma=4.0
    )


def _stamp_psf(image: np.ndarray, v: int, u: int, phase_vu: tuple[float, float]) -> None:
    """Add a pixel-integrated 2000 DN NAC PSF at ``(v, u)`` and the given phase."""
    stamp = GaussianPSF(sigma=_NAC_PSF_SIGMA).eval_rect((9, 9), offset=phase_vu)
    image[v - 4 : v + 5, u - 4 : u + 5] += 2000.0 * stamp / float(stamp.sum())


def test_psf_concentration_matches_the_pixel_integrated_gaussian() -> None:
    """The shares are the integrated Gaussian's at its most concentrated phases."""
    limits = _nac_limits()
    # Centered on a pixel: erf(0.5 / (0.54 sqrt 2)) ** 2 in one pixel.
    assert limits.one_px == pytest.approx(0.4167, abs=1e-3)
    # Centered on an edge: two pixels of erf(1 / (0.54 sqrt 2)) / 2 * erf(0.5 / ...).
    assert limits.two_px == pytest.approx(0.6042, abs=1e-3)
    # Centered on a corner: erf(1 / (0.54 sqrt 2)) ** 2 in the 2x2 block.
    assert limits.two_by_two_px == pytest.approx(0.8762, abs=1e-3)


def test_a_two_pixel_hit_is_sharper_than_the_psf() -> None:
    """Two bright pixels with nothing around them hold too much light in the brightest."""
    image = _bright_floor()
    image[20, 20] += 300.0
    image[20, 21] += 280.0
    assert _is_sharp(image, 20, 20) is True


def test_a_spike_on_a_faint_halo_is_sharper_than_the_psf_in_its_brightest_pixel() -> None:
    """Half the light in one pixel and the rest spread evenly around it is no star."""
    image = _bright_floor()
    image[19:22, 19:22] += 50.0
    image[20, 20] += 350.0
    assert _is_sharp(image, 20, 20) is True


def test_a_three_pixel_streak_is_sharper_than_the_psf_in_its_two_brightest() -> None:
    """A short streak with no pixel too bright alone is caught by its brightest pair."""
    image = _bright_floor()
    image[19, 20] += 400.0
    image[20, 20] += 400.0
    image[21, 20] += 200.0
    assert _is_sharp(image, 20, 20) is True


def test_a_two_by_two_hit_is_sharper_than_the_psf_in_its_block() -> None:
    """A square hit spread evenly over four pixels is caught by its 2x2 block."""
    image = _bright_floor()
    image[20:22, 20:22] += 300.0
    assert _is_sharp(image, 20, 20) is True


@pytest.mark.parametrize(
    'phase_vu', [(0.5, 0.5), (0.0, 0.5), (0.0, 0.0)], ids=['center', 'edge', 'corner']
)
def test_a_star_at_the_psf_is_not_sharper_than_the_psf(phase_vu: tuple[float, float]) -> None:
    """A bright star exactly as concentrated as the PSF allows is kept at every phase."""
    image = _bright_floor()
    _stamp_psf(image, 20, 20, phase_vu)
    assert _is_sharp(image, 20, 20) is False


def test_a_dark_column_through_a_star_box_does_not_make_it_sharp() -> None:
    """Pixels below the background count as no light, not as negative light."""
    image = _bright_floor()
    _stamp_psf(image, 20, 20, (0.5, 0.5))
    image[:, 18] -= 50.0
    assert _is_sharp(image, 20, 20) is False


def test_a_hit_at_the_noise_floor_is_not_judged() -> None:
    """A faint two-pixel hit does not exceed the PSF's share by the noise margin."""
    image = _bright_floor()
    image[20, 20] += 6.0
    image[20, 21] += 6.0
    assert _is_sharp(image, 20, 20) is False


def test_a_hit_whose_ring_leaves_the_image_is_not_judged() -> None:
    """Too near the edge for the background ring, a candidate is never set aside."""
    image = _bright_floor()
    image[2, 20] += 300.0
    image[2, 21] += 280.0
    assert _is_sharp(image, 2, 20) is False


@pytest.mark.parametrize(
    ('amplitude', 'expected'), [(10.0, False), (16.0, True)], ids=['within', 'beyond']
)
def test_a_lone_hit_is_judged_against_the_noise_of_its_excess(
    amplitude: float, expected: bool
) -> None:
    """The margin is the noise of the pixel less its share of the box, not of the pixel.

    On a noiseless floor a one-pixel hit's excess over the NAC share is 0.583 of its
    amplitude, and the noise of that difference is 2.12 per-pixel sigmas: 10 DN stays
    inside the 4-sigma margin (8.5 DN), 16 DN does not.
    """
    image = np.full((41, 41), _BACKGROUND)
    image[20, 20] += amplitude
    assert _is_sharp(image, 20, 20) is expected


def test_a_hit_beside_a_star_box_does_not_make_the_star_sharp() -> None:
    """A brighter hit in the star's box, outside its search, is no part of the star."""
    image = _bright_floor()
    _stamp_psf(image, 20, 21, (0.5, 0.5))
    image[20, 23] += 1500.0
    assert _is_sharp(image, 20, 20) is False


def test_a_hit_across_a_pixel_corner_is_sharper_than_the_psf_in_its_pair() -> None:
    """Two pixels touching at a corner count as the brightest pair.

    On a faint plateau over the rest of the box, neither pixel alone nor any 2x2 block
    holds more than the PSF allows; only the corner pair does.
    """
    image = _bright_floor()
    image[18:23, 18:23] += 10.0
    image[20:22, 20:22] -= 10.0
    image[20, 20] += 300.0
    image[21, 21] += 290.0
    assert _is_sharp(image, 20, 20) is True


def test_a_brighter_hit_two_pixels_from_a_star_does_not_make_the_star_sharp() -> None:
    """The box stays on the star's own brightest pixel, not on a brighter hit beside it.

    A star centered on a pixel corner has four brightest pixels of 438 DN; a 1000 DN hit
    two pixels from the one detected would otherwise take the box and its light.
    """
    image = _bright_floor()
    _stamp_psf(image, 20, 20, (0.0, 0.0))
    image[20, 22] += 1000.0
    assert _is_sharp(image, 20, 20) is False


def test_a_fainter_hit_two_pixels_from_a_peak_does_not_pair_with_it() -> None:
    """Only a pixel touching the brightest one joins it in the brightest pair.

    A broad source whose own pixels stay within every share, with a fainter hit two pixels
    from its peak in the box: paired with the peak, the hit would exceed the pair's share.
    """
    image = _bright_floor()
    image[18:23, 18:23] += 10.0
    image[19:22, 19:22] += 20.0
    image[20, 20] += 370.0
    image[20, 22] += 340.0
    assert _is_sharp(image, 20, 20) is False
