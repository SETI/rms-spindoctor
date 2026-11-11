"""Per-pixel Poisson sampling and the locality of the detector's noise.

The detector draws one uniform per pixel and turns it into a count through that
pixel's own Poisson distribution, so changing one pixel's mean can change only that
pixel's noise.  These tests pin the locality at the sampler and through the detector
chain, the exactness of the inverse distribution function below the exact limit, the
accuracy of the approximation above it, and the sampler's mean and variance.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy import special, stats
from scipy.signal import fftconvolve

from spindoctor.sim.forward.detector import apply_detector
from spindoctor.sim.forward.detector.shot_noise import (
    EXACT_MEAN_LIMIT,
    poisson_inverse_cdf,
    sample_poisson,
)
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed

# The coiss_nac floor converts a signal of 1.0 to 0.5 * 110k electrons.
_NAC_ELECTRONS_PER_SIGNAL = 55.0e3

# Samples per moment test.  With 5-sigma bands the chance that any one band fails
# for a correct sampler is below 1e-6, whatever the seed.
_MOMENT_SAMPLES = 1_000_000
_MOMENT_SIGMAS = 5.0

_SEED = 911
_SHAPE = (32, 32)
_CHANGED = (13, 17)


def _base_means() -> np.ndarray:
    """A frame of means spanning dark, faint, and bright pixels."""
    means = np.linspace(0.0, 400.0, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE)
    means[:, :4] = 0.0
    return means


@pytest.mark.parametrize(
    ('before', 'after'),
    [
        (0.0, 1e-300),
        (np.nextafter(10.0, 0.0), np.nextafter(10.0, 20.0)),
        (np.nextafter(EXACT_MEAN_LIMIT, 0.0), EXACT_MEAN_LIMIT),
    ],
    ids=['zero_to_tiny', 'across_ten', 'across_exact_limit'],
)
def test_one_changed_mean_leaves_every_other_pixel_unchanged(before: float, after: float) -> None:
    """Changing one pixel's mean leaves the counts of all other pixels as they were."""
    means_a = _base_means()
    means_a[_CHANGED] = before
    means_b = means_a.copy()
    means_b[_CHANGED] = after
    counts_a = sample_poisson(np.random.default_rng(_SEED), means_a)
    counts_b = sample_poisson(np.random.default_rng(_SEED), means_b)
    others = np.ones(_SHAPE, dtype=bool)
    others[_CHANGED] = False
    assert np.array_equal(counts_a[others], counts_b[others])


def test_the_sampler_draws_one_uniform_per_pixel_whatever_the_means() -> None:
    """The generator is left where one uniform per pixel leaves it, for any means."""
    rng_dark = np.random.default_rng(_SEED)
    rng_lit = np.random.default_rng(_SEED)
    rng_reference = np.random.default_rng(_SEED)
    sample_poisson(rng_dark, np.zeros(_SHAPE))
    sample_poisson(rng_lit, _base_means())
    rng_reference.random(_SHAPE)
    expected = rng_reference.random()
    assert rng_dark.random() == expected
    assert rng_lit.random() == expected


@pytest.mark.parametrize('mean', [0.0, -3.0], ids=['zero', 'negative'])
def test_a_mean_that_is_not_positive_gives_zero(mean: float) -> None:
    """A zero or negative mean gives a count of 0 at any uniform."""
    counts = poisson_inverse_cdf(np.array([0.0, 0.5, 1.0 - 2.0**-53]), np.full(3, mean))
    assert np.array_equal(counts, np.zeros(3))


def test_a_zero_uniform_gives_zero() -> None:
    """A uniform of exactly 0 gives a count of 0, the smallest count there is."""
    counts = poisson_inverse_cdf(np.zeros(4), np.array([0.5, 10.0, 1.0e5, 4.0 * EXACT_MEAN_LIMIT]))
    assert np.array_equal(counts, np.zeros(4))


def test_a_mean_that_is_not_finite_is_refused() -> None:
    """A NaN or infinite mean raises and counts the bad pixels."""
    with pytest.raises(ValueError, match='finite, but 2 of 3 are NaN or infinite'):
        poisson_inverse_cdf(np.full(3, 0.5), np.array([1.0, np.nan, np.inf]))


def test_a_uniform_outside_the_unit_interval_is_refused() -> None:
    """A uniform of 1 or a NaN uniform raises and counts the bad pixels."""
    with pytest.raises(ValueError, match=r'\[0, 1\), but 2 of 3 are NaN or outside it'):
        poisson_inverse_cdf(np.array([0.5, 1.0, np.nan]), np.full(3, 100.0))


def test_uniforms_and_means_of_different_dimensions_are_refused() -> None:
    """Uniforms and means of the same size but different dimensions raise."""
    with pytest.raises(ValueError, match=r'the uniforms are \(6,\) and the means \(2, 3\)'):
        poisson_inverse_cdf(np.full(6, 0.5), np.full((2, 3), 3.0))


@pytest.mark.parametrize('mean', [1e-300, 1e-12], ids=['1e-300', '1e-12'])
def test_a_tiny_mean_gives_the_exact_count_at_every_uniform(mean: float) -> None:
    """A tiny mean gives 0, or 1 where the uniform is within the mean of 1, promptly.

    Below a mean of about 1e-8 the chance of two counts is smaller than the gap between
    the uniforms nearest 1, so the count is 1 exactly when ``1 - u`` falls below
    ``P(X >= 1)`` and 0 otherwise.  The uniforms span the generator's whole range,
    including its smallest and largest values.
    """
    uniform = np.concatenate(
        [np.random.default_rng(_SEED).random(10_000), [2.0**-53, 0.5, 1.0 - 2.0**-53]]
    )
    means = np.full(uniform.size, mean)
    expected = (1.0 - uniform < -np.expm1(-means)).astype(np.float64)
    assert np.array_equal(poisson_inverse_cdf(uniform, means), expected)


def test_the_inverse_matches_scipy_below_the_exact_limit() -> None:
    """Below the exact limit each count is SciPy's Poisson quantile at its uniform."""
    rng = np.random.default_rng(_SEED)
    uniform = rng.random(10_000)
    means = 10.0 ** rng.uniform(-3.0, math.log10(EXACT_MEAN_LIMIT), 10_000)
    counts = poisson_inverse_cdf(uniform, means)
    assert np.array_equal(counts, stats.poisson.ppf(uniform, means))


def _exact_by_definition(uniform: np.ndarray, means: np.ndarray) -> np.ndarray:
    """The smallest count whose distribution function reaches each uniform, by brute force.

    Below one half the lower tail is compared with the uniform, and above it the upper
    tail with one minus the uniform, the comparison the sampler's search makes.
    SciPy's ``poisson.ppf`` is not used here because it is not exact for a uniform that
    lies on a step of the distribution.
    """
    counts = np.full(uniform.shape, np.nan)
    upper = uniform > 0.5
    for k in range(200):
        reaches = np.where(
            upper, special.pdtrc(k, means) <= 1.0 - uniform, special.pdtr(k, means) >= uniform
        )
        counts[np.isnan(counts) & reaches] = k
    return counts


@pytest.mark.parametrize('where', ['on', 'below', 'above'])
def test_a_uniform_at_a_distribution_step_gets_the_exact_count(where: str) -> None:
    """A uniform on, just below, or just above a step of the distribution gets its count.

    These are the uniforms where rounding in the running sum could pick the
    neighboring count, so the sampler settles them again exactly.
    """
    rng = np.random.default_rng(_SEED)
    steps = np.floor(rng.uniform(0.0, 60.0, 5_000))
    means = rng.uniform(0.01, np.nextafter(EXACT_MEAN_LIMIT, 0.0), steps.size)
    uniform = special.pdtr(steps, means)
    keep = (uniform > 0.0) & (uniform < 1.0)
    uniform, means = uniform[keep], means[keep]
    if where == 'below':
        uniform = np.nextafter(uniform, 0.0)
    elif where == 'above':
        uniform = np.minimum(np.nextafter(uniform, 1.0), 1.0 - 2.0**-53)
    assert np.array_equal(poisson_inverse_cdf(uniform, means), _exact_by_definition(uniform, means))


def test_the_largest_uniform_gets_the_exact_count_below_the_exact_limit() -> None:
    """The generator's largest uniform gets its exact count at means below the exact limit.

    The running sum's total rounds to 1 before it reaches this uniform, so the sampler
    has to settle these pixels with the search instead.
    """
    means = np.linspace(0.01, np.nextafter(EXACT_MEAN_LIMIT, 0.0), 500)
    uniform = np.full(means.size, 1.0 - 2.0**-53)
    assert np.array_equal(poisson_inverse_cdf(uniform, means), _exact_by_definition(uniform, means))


def test_above_the_exact_limit_a_count_is_off_by_at_most_one_on_few_pixels() -> None:
    """Above the exact limit under 1% of counts differ from the exact quantile, by one.

    At random uniforms SciPy's quantile is exact, so it serves as the reference here.
    """
    rng = np.random.default_rng(_SEED)
    uniform = rng.random(100_000)
    means = 10.0 ** rng.uniform(math.log10(EXACT_MEAN_LIMIT), 5.0, uniform.size)
    difference = np.abs(poisson_inverse_cdf(uniform, means) - stats.poisson.ppf(uniform, means))
    assert difference.max() <= 1.0
    assert np.count_nonzero(difference) < 0.01 * uniform.size


@pytest.mark.parametrize(
    'mean',
    [0.5, 10.0, 50.0, 1000.0],
    ids=['small', 'ten', 'approximated_fifty', 'approximated_thousand'],
)
def test_sample_mean_matches_the_poisson_mean(mean: float) -> None:
    """The sample mean is within 5 standard errors, sqrt(mean / n), of the mean."""
    counts = sample_poisson(np.random.default_rng(_SEED), np.full(_MOMENT_SAMPLES, mean))
    standard_error = math.sqrt(mean / _MOMENT_SAMPLES)
    assert abs(float(counts.mean()) - mean) < _MOMENT_SIGMAS * standard_error


@pytest.mark.parametrize(
    'mean',
    [0.5, 10.0, 50.0, 1000.0],
    ids=['small', 'ten', 'approximated_fifty', 'approximated_thousand'],
)
def test_sample_variance_matches_the_poisson_mean(mean: float) -> None:
    """The sample variance is within 5 standard errors of the mean.

    A Poisson variable's fourth central moment is ``mean + 3 mean**2``, so the
    sample variance has standard error ``sqrt((mean + 2 mean**2) / n)``.
    """
    counts = sample_poisson(np.random.default_rng(_SEED), np.full(_MOMENT_SAMPLES, mean))
    standard_error = math.sqrt((mean + 2.0 * mean**2) / _MOMENT_SAMPLES)
    assert abs(float(counts.var()) - mean) < _MOMENT_SIGMAS * standard_error


# --- through the detector chain ----------------------------------------------

_FRAME_SHAPE = (48, 48)
_REGION = (slice(8, 16), slice(30, 40))


def _detector_dn(signal: np.ndarray, *, noise: dict[str, float] | None = None) -> np.ndarray:
    """Run the coiss_nac detector with shot and read noise over ``signal``.

    Parameters:
        signal: The signal plane.
        noise: Extra scene noise keys, added to the shot and read noise.
    """
    frame = SimFrame(signal=signal.copy(), point_e=np.zeros_like(signal))
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'random_seed': _SEED,
        'exposure_sec': 1.0,
        'noise': {'poisson': True, 'read_noise_dn': 5.0, **(noise or {})},
    }
    rng = np.random.default_rng(derive_effect_seed(_SEED, 'detector'))
    apply_detector(frame, params=params, rng=rng)
    return frame.signal


def _region_signal(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """A frame and a copy whose ``_REGION`` holds the change ``kind`` describes."""
    signal = np.zeros(_FRAME_SHAPE, dtype=np.float64)
    signal[:, 24:] = np.linspace(0.0, 0.2, 24)
    changed = signal.copy()
    if kind == 'zero_to_tiny':
        signal[_REGION] = 0.0
        changed[_REGION] = 1e-300
    elif kind == 'across_ten':
        signal[_REGION] = 9.9 / _NAC_ELECTRONS_PER_SIGNAL
        changed[_REGION] = 10.1 / _NAC_ELECTRONS_PER_SIGNAL
    else:
        changed[_REGION] *= 2.0
    return signal, changed


@pytest.mark.parametrize('kind', ['zero_to_tiny', 'across_ten', 'doubled'])
def test_a_changed_region_leaves_the_rest_of_the_frame_unchanged(kind: str) -> None:
    """Changing the signal in one region leaves every pixel outside it unchanged."""
    signal, changed = _region_signal(kind)
    outside = np.ones(_FRAME_SHAPE, dtype=bool)
    outside[_REGION] = False
    dn = _detector_dn(signal)
    dn_changed = _detector_dn(changed)
    assert np.array_equal(dn[outside], dn_changed[outside])


def test_a_psf_tail_on_an_empty_frame_adds_no_counts() -> None:
    """A blurred star's convolution tail leaves the dark sky as an exactly zero one does.

    With no dark charge, the sky outside the PSF's footprint holds only the FFT's
    rounding: signals that give means from about 1e-17 to 1e-13 electrons, and about
    as many slightly below zero.  The conversion to electrons clips the signal at 0,
    so the negative ones reach the sampler as means of exactly 0.  Each positive mean
    samples to no counts, and promptly, so the sky's DN equals that of the same frame
    with the sky set to exactly 0.
    """
    star = np.zeros(_FRAME_SHAPE, dtype=np.float64)
    star[24, 24] = 0.5
    v, u = np.mgrid[-7:8, -7:8]
    psf = np.exp(-(v * v + u * u) / (2.0 * 1.5**2))
    blurred = fftconvolve(star, psf / psf.sum(), mode='same')
    sky = np.ones(_FRAME_SHAPE, dtype=bool)
    sky[17:32, 17:32] = False
    trimmed = np.where(sky, 0.0, blurred)
    noise = {'dark_current_e_per_sec': 0.0, 'readout_dark_e': 0.0}
    assert np.count_nonzero(sky & (blurred > 0.0)) > 500
    assert np.array_equal(
        _detector_dn(blurred, noise=noise)[sky], _detector_dn(trimmed, noise=noise)[sky]
    )
