"""Per-pixel Poisson sampling and the locality of the detector's noise.

The detector draws one uniform per pixel and turns it into a count through that
pixel's own Poisson distribution, so changing one pixel's mean can change only that
pixel's noise.  These tests pin the locality at the sampler and through the detector
chain, the exactness of the inverse distribution function, and the sampler's mean and
variance.
"""

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

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
        (50.0, 50.0 * (1.0 + 2.0**-52)),
    ],
    ids=['zero_to_tiny', 'across_ten', 'last_bit'],
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
    counts = poisson_inverse_cdf(np.zeros(3), np.array([0.5, 10.0, 1.0e5]))
    assert np.array_equal(counts, np.zeros(3))


def test_the_inverse_matches_scipy_up_to_the_exact_limit() -> None:
    """Up to the exact limit each count is SciPy's Poisson quantile at its uniform."""
    rng = np.random.default_rng(_SEED)
    uniform = rng.random(10_000)
    means = 10.0 ** rng.uniform(-3.0, math.log10(EXACT_MEAN_LIMIT), 10_000)
    counts = poisson_inverse_cdf(uniform, means)
    assert np.array_equal(counts, stats.poisson.ppf(uniform, means))


@pytest.mark.parametrize(
    'mean',
    [0.5, 10.0, 1000.0, 4.0 * EXACT_MEAN_LIMIT],
    ids=['small', 'ten', 'large', 'above_exact_limit'],
)
def test_sample_mean_matches_the_poisson_mean(mean: float) -> None:
    """The sample mean is within 5 standard errors, sqrt(mean / n), of the mean."""
    counts = sample_poisson(np.random.default_rng(_SEED), np.full(_MOMENT_SAMPLES, mean))
    standard_error = math.sqrt(mean / _MOMENT_SAMPLES)
    assert abs(float(counts.mean()) - mean) < _MOMENT_SIGMAS * standard_error


@pytest.mark.parametrize(
    'mean',
    [0.5, 10.0, 1000.0, 4.0 * EXACT_MEAN_LIMIT],
    ids=['small', 'ten', 'large', 'above_exact_limit'],
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


def _detector_dn(signal: np.ndarray) -> np.ndarray:
    """Run the coiss_nac detector with shot and read noise over ``signal``."""
    frame = SimFrame(signal=signal.copy(), point_e=np.zeros_like(signal))
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'random_seed': _SEED,
        'exposure_sec': 1.0,
        'noise': {'poisson': True, 'read_noise_dn': 5.0},
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
