"""Flux-normalized point-source star deposition and the background sky field.

Stars deposit their total flux (``zero_point * 10**(-0.4 * vmag) * exposure``)
as sub-pixel point masses in the detector-native point-source plane; the
whole-scene optics PSF is their only convolution, and the box-mean downsample
conserves the per-star detector-grid sum.  A floor scene's rendered star sigma
equals the navigator's configured sigma, an ``instrument_defaults`` star
reproduces the catalog kernel, and the flux scales as expected with exposure and
magnitude.  The background sky field draws its counts from a star-count law and
renders them through the same path.
"""

from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from spindoctor.sim.forward.artifacts_catalog import PSF_KERNELS
from spindoctor.sim.forward.psf import psf_kernel, psf_truncation_for_instrument
from spindoctor.sim.forward.star import faint_sky_cutoff_mag, render_sky_counts
from spindoctor.sim.render import clear_render_caches, render_combined_model

# Explicit zeroing of every stochastic detector knob, so the star-profile
# measurements below see only the deterministic signal chain.
_QUIET_NOISE: dict[str, Any] = {
    'poisson': False,
    'read_noise_dn': 0.0,
    'bias_dn': 0.0,
    'dark_current_e_per_sec': 0.0,
    'hot_pixel_fraction': 0.0,
    'banding_amplitude_e': 0.0,
    'bias_pedestal_sigma_dn': 0.0,
    'bias_row_gradient_dn': 0.0,
    'bias_col_gradient_dn': 0.0,
}


def _star_scene(instrument: str, *, vmag: float = 4.0, **extra: Any) -> dict[str, Any]:
    """A single centered star on a quiet detector, plus extra top-level keys.

    The default magnitude is bright enough to sample the PSF well but below
    saturation at the coiss_wac zero point the profile tests use.
    """
    scene: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'noise': dict(_QUIET_NOISE),
        'stars': [{'name': 'S', 'v': 32.5, 'u': 32.5, 'vmag': vmag}],
    }
    scene.update(extra)
    return scene


def _moment_stats(img: np.ndarray, center: int, radius: int) -> tuple[float, float, float, float]:
    """Centroid and box-corrected moment sigmas of a star cutout.

    The detector image samples the pixel-integrated profile, which inflates
    the raw second moment by the pixel box variance (1/12); subtracting it
    recovers the underlying profile sigma.
    """
    win = img[center - radius : center + radius + 1, center - radius : center + radius + 1]
    vv, uu = np.mgrid[0 : win.shape[0], 0 : win.shape[1]].astype(np.float64)
    total = float(win.sum())
    mv = float((win * vv).sum()) / total
    mu = float((win * uu).sum()) / total
    var_v = float((win * (vv - mv) ** 2).sum()) / total
    var_u = float((win * (uu - mu) ** 2).sum()) / total
    sigma_v = float(np.sqrt(max(var_v - 1.0 / 12.0, 0.0)))
    sigma_u = float(np.sqrt(max(var_u - 1.0 / 12.0, 0.0)))
    return mv + center - radius, mu + center - radius, sigma_v, sigma_u


def _star_window_sum(instrument: str, *, vmag: float, exposure_sec: float) -> float:
    """Total DN in a wide window around a single centered star (background 0)."""
    scene = _star_scene(
        instrument,
        vmag=vmag,
        exposure_sec=exposure_sec,
        optics={'psf': {'match_navigator': True}},
    )
    img, _ = render_combined_model(scene)
    return float(img[20:45, 20:45].sum())


def test_flux_doubles_with_exposure() -> None:
    """Doubling the exposure doubles a star's integrated counts."""
    one = _star_window_sum('generic', vmag=6.0, exposure_sec=1.0)
    two = _star_window_sum('generic', vmag=6.0, exposure_sec=2.0)
    assert abs(two - 2.0 * one) < 0.02 * two


def test_flux_drops_ten_fold_per_two_and_a_half_mag() -> None:
    """A star 2.5 magnitudes fainter carries one tenth the flux."""
    bright = _star_window_sum('generic', vmag=4.0, exposure_sec=1.0)
    faint = _star_window_sum('generic', vmag=6.5, exposure_sec=1.0)
    assert abs(faint - bright / 10.0) < 0.02 * bright


def _point_mass_conservation(oversample: int) -> tuple[float, float]:
    """Return (integrated DN, expected DN) for one faint generic star at os."""
    scene = _star_scene(
        'generic',
        vmag=6.0,
        optics={'psf': {'match_navigator': True}},
        oversample=oversample,
    )
    img, meta = render_combined_model(scene)
    total_flux = float(meta['star_info'][0]['total_flux'])
    # Generic detector: gain 1, bias 0 in _QUIET_NOISE, so DN == electrons.
    return float(img[20:45, 20:45].sum()), total_flux


def test_point_mass_conserves_flux_at_os1() -> None:
    """At oversample 1 the integrated DN matches the deposited total flux."""
    got, expected = _point_mass_conservation(1)
    assert abs(got - expected) < 0.01 * expected


def test_point_mass_conserves_flux_at_os4() -> None:
    """At oversample 4 the box-mean downsample still conserves the total flux."""
    got, expected = _point_mass_conservation(4)
    assert abs(got - expected) < 0.01 * expected


def test_floor_star_sigma_equals_the_navigator_sigma() -> None:
    """A match_navigator floor star renders at the configured 0.77, not 0.77*sqrt(2)."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    img, _ = render_combined_model(scene)
    _, _, sigma_v, sigma_u = _moment_stats(img, 32, 8)
    assert abs(sigma_v - 0.77) < 0.02 * 0.77
    assert abs(sigma_u - 0.77) < 0.02 * 0.77


def test_floor_star_centroid_lands_at_the_catalog_position() -> None:
    """The point-mass deposit puts the star centroid exactly at its position."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    img, _ = render_combined_model(scene)
    cv, cu, _, _ = _moment_stats(img, 32, 8)
    assert abs(cv - 32.0) < 0.05
    assert abs(cu - 32.0) < 0.05


def test_floor_star_truth_records_the_rendered_sigma() -> None:
    """The star hit-test metadata carries the scene-PSF sigma, not a pre-spread one."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    _, meta = render_combined_model(scene)
    assert meta['star_info'][0]['sigma'] == 0.77


def test_no_psf_star_truth_records_zero_sigma() -> None:
    """With no PSF block the truth records the spike's actual extent, 0.0.

    A stars scene without any PSF renders 1-pixel spikes; recording the
    navigator's configured sigma there would falsely mark the render as
    PSF-matched and hide the mismatch from floor-integrity checks.
    """
    scene = _star_scene('coiss_wac')
    _, meta = render_combined_model(scene)
    assert meta['star_info'][0]['sigma'] == 0.0


def test_no_psf_star_pixels_unchanged_by_sigma_record() -> None:
    """The recorded sigma is metadata only: the no-PSF deposit stays a spike."""
    scene = _star_scene('coiss_wac')
    img, _ = render_combined_model(scene)
    cutout = img[28:37, 28:37]
    # A 1-px spike concentrates essentially all cutout flux in one pixel.
    assert float(cutout.max()) / float(cutout.sum()) > 0.9


def test_instrument_defaults_star_reproduces_the_catalog_kernel() -> None:
    """An instrument_defaults star's profile is the catalog kernel to quantization.

    The reference is built the way the renderer builds it: the star's total flux
    deposited bilinearly on the oversampled grid, convolved with the instrument's
    catalog kernel, and box-downsampled.  A least-squares amplitude fit removes
    the flux scale; the per-pixel residual is then bounded by the detector's
    integer-DN rounding.  A faint magnitude keeps the star below saturation.
    """
    scene = _star_scene(
        'coiss_nac',
        vmag=8.0,
        artifacts={'instrument_defaults': True},
        # A no-op distortion block disables the catalog residual so the
        # profile comparison sees only the PSF.
        optics={'distortion': {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 0.0}},
    )
    img, _ = render_combined_model(scene)
    radius = 10
    cut = img[32 - radius : 32 + radius + 1, 32 - radius : 32 + radius + 1].astype(np.float64)

    os = 4
    k = PSF_KERNELS['coiss_nac']
    kernel = psf_kernel(
        k['sigma_v'],
        k['sigma_u'],
        k['w'],
        k['r0'],
        k['n'],
        truncation_px=psf_truncation_for_instrument('coiss_nac'),
        oversample=os,
    )
    grid = np.zeros((64 * os, 64 * os), dtype=np.float64)
    pos = 32.0 * os + (os - 1) / 2.0
    low = int(np.floor(pos))
    frac = pos - low
    for dv, wv in ((0, 1.0 - frac), (1, frac)):
        for du, wu in ((0, 1.0 - frac), (1, frac)):
            grid[low + dv, low + du] += wv * wu
    det = fftconvolve(grid, kernel, mode='same').reshape(64, os, 64, os).mean(axis=(1, 3))
    ref = det[32 - radius : 32 + radius + 1, 32 - radius : 32 + radius + 1]

    amplitude = float((cut * ref).sum() / (ref * ref).sum())
    residual = np.abs(cut - amplitude * ref)
    assert float(cut.max()) > 20.0
    assert float(residual.max()) <= 0.6


def test_vidicon_star_renders_in_the_dn_domain() -> None:
    """A Voyager (vidicon) star deposits DN with no electron conversion.

    The vidicon zero point is in DN, so a faint star's integrated DN tracks
    ``zero_point * 10**(-0.4 * vmag) * exposure`` directly (no gain divide).
    """
    scene = _star_scene(
        'vgiss',
        # Bright enough that the PSF wings clear the 1-DN quantization step (so
        # little flux rounds away) yet below the 8-bit ceiling.
        vmag=1.2,
        optics={'psf': {'match_navigator': True}},
        # Voyager GEOMED products are calibrated; override to the raw DN path so
        # the deposited DN reads out directly (no calibration inverse).
        instrument_config={'data_units': 'raw_dn'},
    )
    img, meta = render_combined_model(scene)
    total_dn = float(meta['star_info'][0]['total_flux'])
    got = float(img[15:50, 15:50].sum())
    assert abs(got - total_dn) < 0.05 * total_dn


def test_ghost_of_a_star_exists() -> None:
    """A star casts a ghost: a displaced faint copy of its point source."""
    dv, du = 12, -8
    scene = _star_scene(
        'coiss_nac',
        vmag=5.0,
        optics={
            'psf': {'match_navigator': True},
            'ghosts': [{'dv_px': dv, 'du_px': du, 'amplitude': 0.2, 'defocus_sigma': 1.0}],
        },
    )
    img, _ = render_combined_model(scene)
    # The main star sits at (32, 32); its ghost sits at (32 + dv, 32 + du).
    ghost = img[32 + dv - 3 : 32 + dv + 4, 32 + du - 3 : 32 + du + 4]
    assert float(ghost.max()) > 1.0


def test_sky_counts_scale_with_density_factor() -> None:
    """Doubling the density factor roughly doubles the deposited sky-star count."""

    def deposited(density: float) -> int:
        plane = np.zeros((512, 512), dtype=np.float64)
        cutoff = faint_sky_cutoff_mag(
            zero_point=1.0e7, exposure_sec=1.0, read_noise=12.0, psf_sigma=0.54
        )
        render_sky_counts(
            plane,
            seed=5,
            a=-3.1,
            b=0.34,
            density_factor=density,
            pixel_scale_arcsec=1.237,
            faint_cutoff_mag=cutoff,
            zero_point=1.0e7,
            exposure_sec=1.0,
            diffuse_flux_per_px=0.0,
            oversample=1,
        )
        return int((plane > 0).sum())

    low = deposited(100.0)
    high = deposited(200.0)
    assert low > 100
    assert abs(high - 2.0 * low) < 0.2 * high


def test_sky_counts_are_deterministic() -> None:
    """The same seed reproduces the same sky field bit-for-bit."""

    def render() -> np.ndarray:
        plane = np.zeros((128, 128), dtype=np.float64)
        render_sky_counts(
            plane,
            seed=9,
            a=-3.1,
            b=0.34,
            density_factor=2000.0,
            pixel_scale_arcsec=1.237,
            faint_cutoff_mag=15.0,
            zero_point=1.0e7,
            exposure_sec=1.0,
            diffuse_flux_per_px=0.0,
            oversample=1,
        )
        return plane

    assert np.array_equal(render(), render())


def test_faint_cutoff_deepens_with_a_brighter_zero_point() -> None:
    """A higher zero point (more sensitive camera) pushes the cutoff fainter."""
    dim = faint_sky_cutoff_mag(zero_point=1.0e6, exposure_sec=1.0, read_noise=12.0, psf_sigma=0.54)
    bright = faint_sky_cutoff_mag(
        zero_point=1.0e7, exposure_sec=1.0, read_noise=12.0, psf_sigma=0.54
    )
    # A 10x brighter zero point buys 2.5 magnitudes of depth.
    assert abs((bright - dim) - 2.5) < 1e-6


def test_sky_diffuse_floor_adds_a_flat_pedestal() -> None:
    """The diffuse-sky floor deposits a flat per-pixel level."""
    plane = np.zeros((16, 16), dtype=np.float64)
    render_sky_counts(
        plane,
        seed=1,
        a=-3.1,
        b=0.34,
        density_factor=0.0,
        pixel_scale_arcsec=1.0,
        faint_cutoff_mag=10.0,
        zero_point=1.0e4,
        exposure_sec=1.0,
        diffuse_flux_per_px=7.0,
        oversample=1,
    )
    assert np.allclose(plane, 7.0)


def test_render_caches_cleared_between_sky_renders() -> None:
    """clear_render_caches drops the sky cache so a re-render re-runs."""
    clear_render_caches()
    scene = _star_scene(
        'generic',
        optics={'psf': {'match_navigator': True}},
        sky_counts={'density_factor': 500.0},
    )
    first, _ = render_combined_model(scene)
    clear_render_caches()
    second, _ = render_combined_model(scene)
    assert np.array_equal(first, second)
