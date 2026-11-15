"""Unit tests for the FOM 1 sky-noise machinery (noise estimators, patches, PSD)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import scipy.ndimage
import scipy.optimize

import spindoctor.sim.realism.noise as noise_module
from spindoctor.sim.realism.noise import (
    SkyPatch,
    dark_sky_patches,
    find_uniform_patches,
    lattice_noise_sigma,
    noisy_patches,
    paired_difference_sigma,
    patch_noise_sigma,
    quantization_step,
    radial_power_spectrum,
)
from spindoctor.support.types import NDArrayFloatType


def test_paired_difference_recovers_known_sigma() -> None:
    """On a pure-noise frame the estimator recovers the planted sigma."""
    rng = np.random.default_rng(11)
    patch = rng.normal(100.0, 3.0, (64, 64))
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(3.0, rel=0.1)


def test_paired_difference_ignores_smooth_gradient() -> None:
    """A strong smooth gradient does not inflate the noise estimate.

    This is the reason for local differencing: naive std() would report
    the gradient as noise.
    """
    rng = np.random.default_rng(12)
    vv, _uu = np.mgrid[0:64, 0:64].astype(np.float64)
    patch = 50.0 * vv / 64.0 + rng.normal(0.0, 2.0, (64, 64))
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(2.0, rel=0.1)
    assert float(patch.std()) > 5.0  # naive std really is inflated


def test_paired_difference_robust_to_spikes() -> None:
    """A handful of huge spikes barely moves the MAD-based estimate."""
    rng = np.random.default_rng(13)
    patch = rng.normal(0.0, 1.5, (64, 64))
    patch[10, 10] = 1e5
    patch[40, 20] = -1e5
    sigma = paired_difference_sigma(patch)
    assert sigma == pytest.approx(1.5, rel=0.1)


def test_paired_difference_degenerate_patch_is_nan() -> None:
    """A 1-D or tiny patch yields NaN, not a crash."""
    assert np.isnan(paired_difference_sigma(np.zeros((1, 5))))


def test_quantized_noise_below_one_step_is_recovered() -> None:
    """Gaussian noise of sigma 0.3 DN rounded to whole DN gives back 0.3 DN.

    Paired differences report 0 here: most neighbors share a value.
    """
    rng = np.random.default_rng(21)
    patch = np.round(rng.normal(100.2, 0.3, (64, 64)))
    assert patch_noise_sigma(patch) == pytest.approx(0.3, rel=0.05)


def test_quantized_sky_with_gradient_and_spikes_is_recovered() -> None:
    """A real-like whole-DN sky recovers its sigma.

    The sky carries a gradient of 3 DN across the patch, sigma 1 DN, two hot pixels, and
    a cosmic ray.  The four spike pixels are cut off from the sky by empty levels and
    stay well within the pixels the quantized fit lets lie beyond the noise.
    """
    rng = np.random.default_rng(22)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    sky = 540.0 + 3.0 * (vv + uu) / 64.0 + rng.normal(0.0, 1.0, (32, 32))
    sky[4, 7] += 300.0
    sky[20, 11] += 80.0
    sky[12:14, 25] += 2000.0
    assert patch_noise_sigma(np.round(sky)) == pytest.approx(1.0, rel=0.08)


@pytest.mark.parametrize(
    ('level', 'rows', 'columns'),
    [(9.0, slice(10, 12), slice(None)), (100.0, slice(None), slice(0, 8))],
    ids=['two-dropout-rows', 'eight-edge-columns'],
)
def test_zeros_cut_off_from_the_sky_are_not_measurable(
    level: float, rows: slice, columns: slice
) -> None:
    """A block of zero-filled pixels under a sky of sigma 1 DN makes the patch unmeasurable.

    Empty levels cut the zeros off from the sky, and they are far more than Gaussian noise
    puts beyond three sigmas.  A fit that takes them for a floor clip reads two dropout
    rows under a sky at 9 DN about 55% high, and eight zero-filled columns beside a sky at
    100 DN about 22 times too high, and the inflated sigma lets the patch pass the
    structure test.
    """
    rng = np.random.default_rng(36)
    patch = np.round(level + rng.normal(0.0, 1.0, (32, 32)))
    patch[rows, columns] = 0.0
    assert np.isnan(patch_noise_sigma(patch))


def test_zeros_joined_to_the_sky_are_not_measurable() -> None:
    """Two dropout rows under a sky at 9 DN of sigma 3 DN, floored at 0, give NaN.

    The zeros join the sky's own floored pixels without an empty level between, so the
    fit takes them for a floor clip, and the zero level holds far more pixels than the
    fitted model predicts for it.  A fit that accepts the clip reads about 24% high, and
    the patch passes the structure test.
    """
    rng = np.random.default_rng(40)
    patch = np.maximum(np.round(9.0 + rng.normal(0.0, 3.0, (32, 32))), 0.0)
    patch[10:12, :] = 0.0
    assert np.isnan(patch_noise_sigma(patch))


def test_noise_digitized_with_one_wide_level_stays_measurable() -> None:
    """Whole-DN noise of 1 DN whose central level is wider than a step keeps its sigma.

    The central level spans 1.4 DN, so it holds about 1.35 times the pixels rounded
    Gaussian noise puts there, as the fullest level of real Galileo sky does.  Only the
    bulk's end levels are held to the fitted model's prediction; a check of every level
    leaves this patch unmeasurable.
    """
    rng = np.random.default_rng(41)
    values = 100.0 + rng.normal(0.0, 1.0, (32, 32))
    patch = np.round(values)
    patch[np.abs(values - 100.0) < 0.7] = 100.0
    assert np.isfinite(patch_noise_sigma(patch))


def test_noise_the_two_estimators_disagree_on_is_not_measurable() -> None:
    """Whole-DN noise of 3 DN smoothed across neighbors, as resampling does, gives NaN.

    Smoothing correlates neighboring pixels, so paired differences read about 1.7 DN,
    below two steps, while the quantized fit reads the pixels' own spread of about 3 DN.
    """
    rng = np.random.default_rng(40)
    kernel = np.array([0.25, 0.5, 0.25])
    white = rng.normal(0.0, 1.0, (36, 36))
    smooth = scipy.ndimage.convolve1d(
        scipy.ndimage.convolve1d(white, kernel, axis=0), kernel, axis=1
    )
    patch = np.round(50.0 + 8.0 * smooth[2:34, 2:34])
    assert np.isnan(patch_noise_sigma(patch))


def test_texture_is_not_read_as_noise_once_the_noise_spans_two_steps() -> None:
    """A whole-DN sky of sigma 3 DN under a texture a quadratic cannot follow keeps its sigma.

    The texture is a ripple of 6 DN amplitude along both axes; paired differences cancel
    most of it, while the quantized fit reads it as noise, about 40% high.
    """
    rng = np.random.default_rng(37)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    ripple = 6.0 * np.sin(2.0 * np.pi * vv / 16.0) * np.sin(2.0 * np.pi * uu / 16.0)
    patch = np.round(200.0 + ripple + rng.normal(0.0, 3.0, (32, 32)))
    assert patch_noise_sigma(patch) == pytest.approx(3.0, rel=0.08)


def test_quantized_noise_on_a_fractional_lattice_is_recovered() -> None:
    """A lattice whose step is not 1, such as 8-bit I/F, is fitted in its own units."""
    rng = np.random.default_rng(23)
    step = 1.0 / 255.0
    patch = np.round(rng.normal(3.3, 0.6, (64, 64))) * step
    assert patch_noise_sigma(patch) == pytest.approx(0.6 * step, rel=0.05)


def test_quantization_step_is_none_for_continuous_values() -> None:
    """Values that take any real value are not quantized."""
    rng = np.random.default_rng(24)
    assert quantization_step(rng.normal(0.0, 1.0, (32, 32))) is None


def test_quantization_step_is_none_for_float32_values() -> None:
    """Float32 values sit on the float spacing, which is not a quantization."""
    rng = np.random.default_rng(27)
    patch = rng.normal(3e-3, 2e-4, (32, 32)).astype(np.float32)
    assert quantization_step(patch) is None


def test_two_level_patch_sigma_is_not_measurable() -> None:
    """A patch whose pixels take only two values gives NaN, since any sigma fits it."""
    rng = np.random.default_rng(25)
    patch = np.round(rng.normal(100.5, 0.15, (32, 32)))
    assert np.isnan(lattice_noise_sigma(patch, 1.0))


def test_two_levels_and_one_stray_pixel_are_not_measurable() -> None:
    """One stray pixel on a third level doesn't make a two-level patch measurable."""
    rng = np.random.default_rng(28)
    patch = np.round(rng.normal(100.5, 0.15, (32, 32)))
    patch[5, 5] = 102.0
    assert np.isnan(patch_noise_sigma(patch))


def test_a_patch_of_one_value_is_not_measurable() -> None:
    """Like a two-level patch, a tile of one value gives an unmeasurable sigma."""
    assert np.isnan(patch_noise_sigma(np.full((32, 32), 7.0)))


def test_quantized_noise_on_a_curved_sky_is_recovered() -> None:
    """Curvature across the patch is fitted, never read as noise.

    The sky is a bowl 4 DN deep with sigma 1 DN; a fitted plane leaves the bowl in the
    residuals and reads about 1.31 DN.
    """
    rng = np.random.default_rng(29)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    bowl = 2.0 * (((vv - 15.5) / 16.0) ** 2 + ((uu - 15.5) / 16.0) ** 2)
    patch = np.round(100.3 + bowl + rng.normal(0.0, 1.0, (32, 32)))
    assert patch_noise_sigma(patch) == pytest.approx(1.0, rel=0.05)


def test_quantized_noise_well_below_one_step_on_a_gradient_is_recovered() -> None:
    """Sigma 0.2 DN on a gradient of 1 DN across the patch gives back 0.2 DN.

    The gradient is fitted together with the noise; a gradient fixed beforehand by least
    squares reads about 11% high here.  The median runs over 20 patches at random levels,
    since one patch this far below a step carries only a few pixels off its level.
    """
    rng = np.random.default_rng(30)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    gradient = (vv + uu) / 64.0
    sigmas = [
        patch_noise_sigma(
            np.round(100.0 + rng.uniform() + gradient + rng.normal(0.0, 0.2, (32, 32)))
        )
        for _ in range(20)
    ]
    assert float(np.nanmedian(sigmas)) == pytest.approx(0.2, rel=0.03)


def test_quantized_noise_clipped_at_the_floor_is_recovered() -> None:
    """Sky at the zero floor, where half the values are clipped to 0, gives back its sigma."""
    rng = np.random.default_rng(31)
    patch = np.clip(np.round(rng.normal(0.0, 1.0, (32, 32))), 0.0, None)
    assert patch_noise_sigma(patch) == pytest.approx(1.0, rel=0.08)


def test_a_sky_sloping_below_the_floor_keeps_its_sigma() -> None:
    """Sky tilted 12 DN across the patch, clipped at 0 over about half of it, gives back 0.5 DN.

    Where the surface lies several sigmas below the floor, the clipped pixels sit above
    their windows, on the side toward the clip, which is where a clipped pixel belongs;
    counting them as lying beyond the noise would leave the patch unmeasurable.
    """
    rng = np.random.default_rng(39)
    vv, uu = np.mgrid[0:32, 0:32].astype(np.float64)
    tilt = 12.0 * (vv + uu) / 62.0 - 6.0
    patch = np.maximum(np.round(tilt + rng.normal(0.0, 0.5, (32, 32))), 0.0)
    assert patch_noise_sigma(patch) == pytest.approx(0.5, rel=0.1)


def _fit_over_drawn_sigma(
    rng: np.random.Generator,
    sigma: float,
    clip: str,
    *,
    estimator: Callable[[NDArrayFloatType], float] = patch_noise_sigma,
) -> float:
    """Draw one 32 x 32 sky of noise ``sigma``, round it, and divide its estimate by its sigma.

    The sky is unclipped at about 1000, or clipped at the zero floor or at a ceiling of
    4095, each with about half its values on the clipped level.  Dividing by the standard
    deviation of the noise drawn, before rounding and clipping, removes the sampling
    scatter of the sigma itself.  ``estimator`` measures the rounded sky.
    """
    noise = rng.normal(0.0, sigma, (32, 32))
    sky = rng.uniform() + noise
    if clip == 'floor':
        patch = np.maximum(np.round(sky), 0.0)
    elif clip == 'ceiling':
        patch = np.minimum(np.round(4094.0 + sky), 4095.0)
    else:
        patch = np.round(1000.0 + sky)
    return estimator(patch) / float(noise.std())


def _fitted(patch: NDArrayFloatType) -> float:
    """The quantized fit of a whole-DN patch."""
    return lattice_noise_sigma(patch, 1.0)


@pytest.mark.parametrize('sigma', [10.0, 20.0])
def test_quantized_fit_of_noise_of_many_steps_counts_every_pixel(sigma: float) -> None:
    """Noise of many steps leaves empty levels in its tails, and the fit still counts them.

    A fit that drops the pixels beyond the first empty level truncates the distribution
    and reads about 3% low at 10 steps and 8% low at 20.
    """
    rng = np.random.default_rng(33)
    ratios = [_fit_over_drawn_sigma(rng, sigma, 'none', estimator=_fitted) for _ in range(6)]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.01)


def _grouped(patch: NDArrayFloatType) -> float:
    """The paired-difference estimate of a whole-DN patch, read from its histogram."""
    return noise_module._grouped_difference_sigma(patch - patch.min())


def test_paired_differences_of_quantized_noise_read_between_whole_steps() -> None:
    """Whole-DN noise of sigma 2.5 DN read from its paired differences gives back 2.5 DN.

    The plain MAD of whole-step differences is a whole or half step here and reads about
    16% low; the differences read from their histogram without removing the variance the
    rounding adds read about 1% high.  The median runs over 1000 patches to resolve that.
    """
    rng = np.random.default_rng(38)
    ratios = [_fit_over_drawn_sigma(rng, 2.5, 'none', estimator=_grouped) for _ in range(1000)]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.005)


def _dense_grouped_sigma(q: NDArrayFloatType) -> float:
    """The grouped paired-difference sigma with one histogram bin per step of the span.

    This is the reference the estimator must reproduce: every step between the lowest and
    highest difference gets a bin, occupied or not.  Where the cumulative fraction holds
    at one half across a run of knots, the median is the run's midpoint, and where the
    mass about the median holds within 1e-9 of one half across a run of radii, the
    median absolute deviation is that run's midpoint.  The mass is 0 at radius 0, which
    is a knot as well.
    """
    diffs = (q[:, 1:] - q[:, :-1]).ravel()
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    lowest = int(diffs.min())
    counts = np.bincount((diffs - lowest).astype(np.int64))
    edges = lowest - 0.5 + np.arange(counts.size + 1, dtype=np.float64)
    below = np.concatenate([[0], np.cumsum(counts)])
    cdf = below / diffs.size
    halves = np.flatnonzero(2 * below == diffs.size)
    if halves.size:
        center = 0.5 * float(edges[halves[0]] + edges[halves[-1]])
    else:
        center = float(np.interp(0.5, cdf, edges))
    radii = np.unique(np.concatenate([[0.0], np.abs(edges - center)]))
    mass = np.interp(center + radii, edges, cdf) - np.interp(center - radii, edges, cdf)
    at_half = np.flatnonzero(np.abs(mass - 0.5) <= 1e-9)
    if at_half.size > 1:
        mad = 0.5 * float(radii[at_half[0]] + radii[at_half[-1]])
    else:
        mad = float(np.interp(0.5, mass, radii))
    variance = (mad * 1.4826) ** 2 / 2.0 - 1.0 / 8.0
    return float(np.sqrt(max(variance, 0.0)))


def _grouped_test_patch(kind: str, rng: np.random.Generator) -> NDArrayFloatType:
    """A whole-step patch of the named kind for the dense-reference comparison."""
    if kind.startswith('gaussian-'):
        return np.round(rng.uniform() + rng.normal(50.0, float(kind[9:]), (32, 32)))
    if kind == 'few-separated-levels':
        levels = np.array([0.0, 7.0, 50.0, 51.0])
        return levels[rng.choice(4, size=(32, 32), p=[0.5, 0.3, 0.15, 0.05])]
    if kind == 'outliers':
        patch = np.round(rng.normal(100.0, 2.0, (32, 32)))
        patch[rng.integers(0, 32, 6), rng.integers(0, 32, 6)] = 100_000.0
        return patch
    if kind == 'clipped':
        return np.maximum(np.round(rng.normal(0.5, 3.0, (32, 32))), 0.0)
    if kind == 'non-finite':
        patch = np.round(rng.normal(100.0, 3.0, (32, 32)))
        patch[rng.integers(0, 32, 20), rng.integers(0, 32, 20)] = np.nan
        return patch
    if kind == 'half-on-each-side-of-a-gap':
        # 16 differences of +10 and 16 of -10 in every row: the cumulative fraction is
        # exactly one half across the whole gap between them.
        return np.tile(np.array([0.0, 10.0] * 16 + [0.0]), (32, 1))
    if kind == 'constant':
        return np.full((32, 32), 7.0)
    raise ValueError(f'unknown patch kind {kind!r}')


@pytest.mark.parametrize(
    'kind',
    [
        'gaussian-0.3',
        'gaussian-0.8',
        'gaussian-2.5',
        'gaussian-10',
        'gaussian-40',
        'few-separated-levels',
        'outliers',
        'clipped',
        'non-finite',
        'half-on-each-side-of-a-gap',
        'constant',
    ],
)
def test_grouped_differences_match_a_bin_for_every_step(kind: str) -> None:
    """Reading only the occupied steps gives what a bin for every step of the span gives.

    Noise of 10 and 40 steps, separated levels, and far outliers leave empty steps among
    the differences, which a bin per step fills with zeros.
    """
    rng = np.random.default_rng(39)
    for _ in range(5):
        q = _grouped_test_patch(kind, rng)
        expected = _dense_grouped_sigma(q)
        assert noise_module._grouped_difference_sigma(q) == pytest.approx(
            expected, rel=1e-12, abs=1e-12
        )


def _sigma_from_mad(mad: float) -> float:
    """The grouped estimator's sigma for a median absolute deviation of ``mad`` steps."""
    return float(np.sqrt(max((mad * 1.4826) ** 2 / 2.0 - 1.0 / 8.0, 0.0)))


@pytest.mark.parametrize(
    ('q', 'mad'),
    [
        (np.array([[0.0, -1.0, -2.0, 4.0, 11.0]]), 11.0 / 3.0),
        (np.array([[47.0, 57.0, 47.0, 57.0], [1.0, 47.0, 57.0, 1.0]]), 10.0),
        (
            np.concatenate(
                [[0.0], np.cumsum(np.repeat([-20.0, -12, -5, 4, 8, 9], [2, 3, 3, 1, 2, 1]))]
            )[None, :],
            8.0,
        ),
    ],
    ids=['median-between-two-halves', 'deviation-range-6-differences', 'deviation-range-12'],
)
def test_grouped_differences_take_the_midpoint_of_an_exact_half(
    q: NDArrayFloatType, mad: float
) -> None:
    """Where exactly half the differences hold over a range, its midpoint is taken.

    The first patch's differences are -1, -1, 6, and 7: every point from -0.5 to 5.5
    splits them in half, the median is 2.5, and the median absolute deviation is 11/3.
    The second's are -56, -10, 10, 10, 10, and 46, with median 9 5/6: every deviation
    from 2/3 to 19 1/3 takes in exactly three, so the deviation is 10.  The third's
    twelve differences have median -5 1/6, and every deviation from 7 1/3 to 8 2/3 takes
    in six, so the deviation is 8.  Points of those ranges picked by the float rounding
    of the mass gave sigmas of 0.60 and 7.68 steps for the last two.
    """
    assert noise_module._grouped_difference_sigma(q) == pytest.approx(
        _sigma_from_mad(mad), rel=1e-12
    )


@pytest.mark.parametrize(
    ('q', 'mad'),
    [
        (np.array([[-10.0, -10.0]]), 0.25),
        (np.concatenate([[0.0], np.cumsum([0.0] * 6 + [3.0, -3.0, 5.0, -5.0])])[None, :], 5 / 12),
    ],
    ids=['one-difference', 'six-of-ten-differences-zero'],
)
def test_grouped_differences_read_a_deviation_inside_the_central_step(
    q: NDArrayFloatType, mad: float
) -> None:
    """A median absolute deviation smaller than the nearest step edge is read in full.

    The first patch's single difference, spread across its step, puts half its mass
    within a quarter step of the center, which reads as a sigma of 0: the spread is no
    larger than the rounding's.  Six of the second patch's ten differences are 0 and the
    rest lie 3 and 5 steps out, so half the mass lies within 5/12 of a step.  Both
    deviations lie inside the half step out to the first edge, where interpolating from
    the first edge alone would read half a step.
    """
    assert noise_module._grouped_difference_sigma(q) == pytest.approx(
        _sigma_from_mad(mad), rel=1e-12, abs=1e-12
    )


def test_an_outliers_distance_does_not_change_the_grouped_differences() -> None:
    """One pixel 1e15 steps above the sky reads as one 1e5 steps above it does.

    A histogram with a bin for every step of the wider span would need petabytes; the
    pixel 1e5 steps above is checked against that dense histogram.
    """
    rng = np.random.default_rng(40)
    near = np.round(rng.normal(100.0, 2.0, (32, 32)))
    near[16, 16] = 1e5
    far = near.copy()
    far[16, 16] = 1e15
    assert noise_module._grouped_difference_sigma(far) == pytest.approx(
        _dense_grouped_sigma(near), rel=1e-12
    )


@pytest.mark.parametrize(
    ('clip', 'sigma'), [('floor', 3.0), ('ceiling', 5.0)], ids=['floor-3', 'ceiling-5']
)
def test_clipped_quantized_noise_of_a_few_steps_is_recovered(clip: str, sigma: float) -> None:
    """Sky clipped at the floor or at saturation, half its values on one level, keeps its sigma.

    Half the residuals share a value there, so a robust first guess of the sigma comes out
    far too small; a fit held to the window that guess builds reads about 15% low at 3
    steps and 40% low at 5.  Half of each patch is clipped, so the median runs over 20.
    """
    rng = np.random.default_rng(34)
    ratios = [_fit_over_drawn_sigma(rng, sigma, clip) for _ in range(20)]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.025)


def test_a_quantized_fit_that_fails_to_converge_is_not_measurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit the optimizer reports as failed gives NaN rather than where it stopped."""
    real_minimize = scipy.optimize.minimize

    def failing_minimize(*args: Any, **kwargs: Any) -> scipy.optimize.OptimizeResult:
        result = real_minimize(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(scipy.optimize, 'minimize', failing_minimize)
    rng = np.random.default_rng(35)
    assert np.isnan(lattice_noise_sigma(np.round(rng.normal(100.0, 1.0, (32, 32))), 1.0))


def test_a_window_that_repeats_stops_the_quantized_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window equal to one fitted before ends the iteration instead of alternating.

    On this patch of sigma 1 DN the window alternates between two states, and a fit that
    stops only on an unchanged window runs all ten fits.  How many fits it takes to reach
    the repeat depends on the optimizer's last bits, so only the cap is held.
    """
    real_fit = noise_module._fit_censored
    fits: list[int] = []

    def counting_fit(*args: Any, **kwargs: Any) -> Any:
        fits.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(noise_module, '_fit_censored', counting_fit)
    rng = np.random.default_rng(26)
    patch = np.round(1000.0 + rng.uniform() + rng.normal(0.0, 1.0, (32, 32)))
    lattice_noise_sigma(patch, 1.0)
    assert len(fits) < noise_module._MAX_WINDOW_FITS


@pytest.mark.parametrize(
    ('scale', 'offset', 'level'),
    [(1.234e-3, 0.0, 3000.0), (1e-4, 0.5, 30.0), (1e-4, 8.0, 30.0)],
    ids=['scaled', 'scaled-and-offset', 'offset-beyond-the-step-tolerance'],
)
def test_quantization_step_finds_whole_dn_scaled_into_float32(
    scale: float, offset: float, level: float
) -> None:
    """Whole DN scaled into float32 sit on a lattice although each value carries rounding.

    Near 8.0, a float32 value's rounding exceeds a thousandth of a 1e-4 step, so the
    tolerance there comes from the float precision at the values' magnitude.
    """
    rng = np.random.default_rng(32)
    patch = (np.round(rng.normal(level, 3.0, (32, 32))) * scale + offset).astype(np.float32)
    assert quantization_step(patch) == pytest.approx(scale, rel=1e-4)


def test_one_bright_pixel_does_not_hide_a_fine_lattice() -> None:
    """A sky on a 1e-8 lattice near 1e-3 keeps its lattice beside one pixel at 1.0.

    The step is 1e-8 of the bright pixel's value, the spacing of float32 numbers there,
    but a hundred times the spacing at the sky's level.  In float32 the bright pixel
    itself sits up to three steps off the lattice.
    """
    rng = np.random.default_rng(41)
    patch = np.round(rng.normal(1e5, 3.0, (32, 32))) * 1e-8
    patch[16, 16] = 1.0
    assert quantization_step(patch.astype(np.float32)) == pytest.approx(1e-8, rel=1e-3)


def test_one_bright_pixel_does_not_widen_the_sky_levels_tolerance() -> None:
    """Sky levels a tenth of a step off a lattice are off it, beside one pixel at 1.0.

    The bright pixel's own float32 rounding is 48 steps of the 1e-8 lattice, so a
    tolerance taken from the largest value would pass every sky level.
    """
    rng = np.random.default_rng(42)
    index = np.round(rng.normal(1e5, 3.0, (32, 32)))
    patch = (index + 0.1 * (-1.0) ** index) * 1e-8
    patch[16, 16] = 1.0
    assert quantization_step(patch.astype(np.float32)) is None


@pytest.mark.parametrize('dtype', [np.float64, np.float32], ids=['float64', 'float32'])
def test_uneven_levels_beside_one_bright_pixel_are_no_lattice(dtype: type) -> None:
    """Levels spaced 1 and 1.3 apart are no lattice beside a pixel 800,000 steps up.

    The bright pixel's float32 allowance, four rounding errors of its value, is 0.38 of
    a step, under half a step, so the pixel is checked like the levels.  That allowance
    taken as every level's tolerance would pass levels three tenths of a step off.
    """
    rng = np.random.default_rng(43)
    levels = np.array([0.0, 1.0, 2.3, 3.3, 4.6, 5.6, 6.9, 7.9, 9.2]) + 100.0
    patch = levels[rng.integers(0, levels.size, (32, 32))]
    patch[3, 3] = 8e5
    assert quantization_step(patch.astype(dtype)) is None


def test_a_bright_pixel_the_fit_cannot_place_keeps_the_lattice() -> None:
    """Whole DN times 1.234e-3 keep their lattice beside a pixel 1.01 million steps up.

    The pixel sits three tenths of a step off the lattice.  Its float32 allowance, four
    rounding errors of its value, is 0.48 step, so the allowance alone nearly passes it,
    and the fit to the sky can't place it either: carried out a million steps, the
    fit's standard error is 14 steps.  So the pixel passes, and the step comes from the
    sky.
    """
    rng = np.random.default_rng(44)
    step = 1.234e-3
    patch = np.round(rng.normal(4000.0, 3.0, (32, 32))) * step
    patch[16, 16] = (np.round(1250.0 / step) + 0.3) * step
    assert quantization_step(patch.astype(np.float32)) == pytest.approx(step, rel=1e-4)


def test_a_bright_pixel_on_the_lattice_keeps_a_scaled_float32_lattice() -> None:
    """Whole DN times 1.234e-3 keep their lattice and its step beside one pixel on it at 5x.

    Carried out 16,000 steps beyond the sky, the sky's fit misses the pixel's level by
    about a tenth of a step, ten times the pixel's own float32 rounding.  That fit can't
    pin down the pixel's lattice index, so the step comes from the sky's levels alone.
    """
    rng = np.random.default_rng(45)
    step = 1.234e-3
    dn = np.round(rng.normal(4000.0, 3.0, (32, 32)))
    dn[16, 16] = 5.0 * dn.max()
    assert quantization_step((dn * step).astype(np.float32)) == pytest.approx(step, rel=1e-4)


def test_a_level_off_the_lattice_beyond_the_fits_reach_passes() -> None:
    """Whole DN keep their lattice beside a pixel 20,000 steps up and half a step off.

    The sky's levels fit exactly, so the fit's standard error comes from one float32
    rounding at the sky's level and is about 0.32 step carried out to the pixel.  Four
    standard errors reach past half a step, so the fit can't place the pixel and it
    passes; one would place it and refuse the lattice.
    """
    rng = np.random.default_rng(49)
    patch = np.round(rng.normal(4000.0, 3.0, (32, 32)))
    patch[16, 16] = 24_000.5
    assert quantization_step(patch) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    ('neighbor', 'far', 'expected'),
    [(999.0, 1500.0, 1.0), (999.0, 1500.4, None), (1001.0, 499.6, None)],
    ids=['bright-on-the-lattice', 'bright-off-the-lattice', 'dark-off-the-lattice'],
)
def test_one_sky_level_fits_its_near_neighbor_alone(
    neighbor: float, far: float, expected: float | None
) -> None:
    """A sky of one level fits its step to the level beside it, leaving a far pixel out.

    Every pixel but three is 1000, two sit on the neighboring level, and one is the far
    pixel, about 500 steps away on the other side.  The fit takes 1000 and its neighbor,
    so the step is 1 with the far pixel on the lattice.  Four tenths of a step off, the
    far pixel is refused, above the sky or below it: fitted along with the sky, it would
    tilt the step by 0.0008 and pass.
    """
    patch = np.full((32, 32), 1000.0)
    patch[0, :2] = neighbor
    patch[16, 16] = far
    expected_step = None if expected is None else pytest.approx(expected, rel=1e-12)
    assert quantization_step(patch) == expected_step


def test_the_step_is_refitted_across_the_levels_the_fit_places() -> None:
    """A placed pixel 4000 steps up sharpens a float32 step; an unplaced one doesn't.

    The sky's levels of whole DN times 1.234e-3 give the step to about 5e-6.  One pixel
    on the lattice at 8000 DN, which the fit places, carries the refitted step to about
    5e-8.  A pixel 100,000 steps up and four tenths of a step off passes because the fit
    can't place it, and taking it into the refit would move the step by about 6e-6.
    """
    rng = np.random.default_rng(46)
    step = 1.234e-3
    dn = np.round(rng.normal(4000.0, 3.0, (32, 32)))
    dn[16, 16] = 8000.0
    dn[8, 8] = 104_000.4
    assert quantization_step((dn * step).astype(np.float32)) == pytest.approx(step, rel=1e-6)


@pytest.mark.parametrize(
    'patch',
    [
        np.where(
            np.arange(1024).reshape(32, 32) == 100,
            -1e17,
            np.round(np.random.default_rng(47).normal(1000.0, 3.0, (32, 32))),
        ),
        np.where(
            np.arange(1024).reshape(32, 32) == 100,
            -3.4e38,
            np.round(np.random.default_rng(48).normal(1e5, 3.0, (32, 32))) * 1e-8,
        ).astype(np.float32),
        np.where(
            np.arange(1024).reshape(32, 32) == 100,
            1000.0 - 3e9,
            np.round(np.random.default_rng(50).normal(1000.0, 3.0, (32, 32))),
        ),
    ],
    ids=['whole-dn-beside-1e17-steps', 'float32-beside-3e46-steps', 'whole-dn-beside-3e9-steps'],
)
def test_a_pixel_beyond_two_to_the_31_steps_puts_the_sample_on_no_lattice(
    patch: NDArrayFloatType,
) -> None:
    """A sentinel value far below the sky leaves the sky to the paired differences.

    Counted in steps from that pixel, the sky's indices lose their whole-step precision
    in float64, and the quantized estimators read NaN.  So the sample does not count as
    quantized, and the patch's sigma is the paired-difference reading.
    """
    assert quantization_step(patch) is None
    assert patch_noise_sigma(patch) == paired_difference_sigma(patch)


def test_a_fitted_step_that_places_no_level_puts_the_sample_on_no_lattice() -> None:
    """A sky on two levels near 1e6 beside a bright pair leaves the step unknown.

    The sky sits on 1e6 and 1e6 + 1.75, and the bright pair sits 100 and 101.125 above
    it.  The step starts at the pair's gap, 1.125, and the sky's two levels refit it to
    0.875.  Each sky level's float32 allowance near 1e6 is 0.48, above half that step,
    and the bright pair lies too far out for the fit to place.  No level pins the step
    down, so the sample does not count as quantized, and the patch's sigma is the
    paired-difference reading.
    """
    patch = np.full((32, 32), 1e6)
    patch[:, ::2] = 1e6 + 1.75
    patch[0, 0] = 1e6 + 100.0
    patch[0, 1] = 1e6 + 101.125
    assert quantization_step(patch) is None
    assert patch_noise_sigma(patch) == paired_difference_sigma(patch)


def test_find_uniform_patches_keeps_quantized_sky_below_one_step() -> None:
    """Whole-DN sky whose noise is below one DN still yields every patch."""
    rng = np.random.default_rng(26)
    image = np.round(rng.normal(20.0, 0.4, (64, 64)))
    assert len(find_uniform_patches(image, patch_size=32, max_mean_quantile=None)) == 4


def test_find_uniform_patches_rejects_structured_tiles() -> None:
    """Tiles crossed by a sharp edge are rejected; pure-noise tiles kept."""
    rng = np.random.default_rng(14)
    image = rng.normal(10.0, 1.0, (128, 128))
    image[:, 64:] += 500.0  # hard step through the right half tiles' columns
    patches = find_uniform_patches(image, patch_size=32, max_mean_quantile=None)
    # The step lands between tile columns, so tiles remain uniform except
    # none straddle it; shift the step to cut through tiles instead.
    image2 = rng.normal(10.0, 1.0, (128, 128))
    image2[:, 48:] += 500.0  # cuts through the second tile column
    patches2 = find_uniform_patches(image2, patch_size=32, max_mean_quantile=None)
    coords2 = {(p.v0, p.u0) for p in patches2}
    assert all((v0, 32) not in coords2 for v0 in range(0, 128, 32))
    assert len(patches) == 16


def test_find_uniform_patches_sky_cut_keeps_low_means() -> None:
    """With the sky cut on, only the lowest-mean patches survive."""
    rng = np.random.default_rng(15)
    image = rng.normal(10.0, 1.0, (128, 128))
    image[64:, :] += 100.0  # bottom half is bright
    patches = find_uniform_patches(image, patch_size=32, max_mean_quantile=0.25)
    assert patches
    assert all(p.mean < 50.0 for p in patches)


def test_radial_power_spectrum_flat_for_white_noise() -> None:
    """White noise has an approximately flat radially averaged spectrum."""
    rng = np.random.default_rng(16)
    freq, power = radial_power_spectrum(rng.normal(0.0, 1.0, (64, 64)), n_bins=8)
    assert freq.shape == (8,)
    finite = power[np.isfinite(power)]
    assert finite.size >= 6
    assert float(finite.max() / finite.min()) < 5.0


def test_radial_power_spectrum_peaks_at_banding_frequency() -> None:
    """Coherent banding concentrates power in its frequency bin."""
    vv = np.arange(64)[:, np.newaxis].astype(np.float64)
    period_px = 8.0
    banding = 5.0 * np.sin(2.0 * np.pi * vv / period_px) * np.ones((64, 64))
    rng = np.random.default_rng(17)
    freq, power = radial_power_spectrum(banding + rng.normal(0.0, 0.1, (64, 64)), n_bins=16)
    peak_bin = int(np.nanargmax(power))
    expected_freq = 1.0 / period_px
    assert freq[peak_bin] == pytest.approx(expected_freq, abs=0.5 / 16)


def test_dark_sky_is_selected_by_level_alone() -> None:
    """A patch above the ceiling is dropped however noisy it is, and one below is kept.

    The ring patch lies within three of its own sigmas of the ceiling, so a selection
    scaled by any sigma could keep it.
    """
    sky = SkyPatch(v0=0, u0=0, size=32, mean=2e-4, sigma=1e-8)
    ring = SkyPatch(v0=0, u0=32, size=32, mean=0.07, sigma=0.05)
    assert dark_sky_patches([sky, ring], 2.54e-3) == [sky]


def test_dark_sky_keeps_a_patch_far_below_the_ceiling() -> None:
    """A patch far below the ceiling is dark sky."""
    dark = SkyPatch(v0=0, u0=0, size=32, mean=-1e-3, sigma=1e-4)
    assert dark_sky_patches([dark], 2.54e-3) == [dark]


def test_a_constant_patch_has_no_noise_to_compare() -> None:
    """A patch whose sigma is 0 is dropped."""
    constant = SkyPatch(v0=0, u0=0, size=32, mean=0.0, sigma=0.0)
    sky = SkyPatch(v0=0, u0=32, size=32, mean=4e-5, sigma=1e-4)
    assert noisy_patches([constant, sky]) == [sky]
