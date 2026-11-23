"""The hits of cosmic rays and of a radiation environment.

The unit of the cosmic-ray rate, and its reach through every detector chain, are
tested in ``tests/spindoctor/sim/test_cosmic_ray_rate.py``.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.render.detector.chain import _apply_radiation_transients
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.detector.transients import add_cosmic_rays, deposit_morphological_events

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


def test_cosmic_rays_deposit_events() -> None:
    """A nonzero cosmic-ray rate deposits charge somewhere in the frame."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=0.01,
        exposure_sec=1.0,
        amplitude_e=1.0e5,
        rng=np.random.default_rng(9),
    )
    assert int((electrons > 0.0).sum()) >= 1


def test_cosmic_rays_disabled_at_zero_rate() -> None:
    """A zero cosmic-ray rate is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_cosmic_rays(
        electrons,
        events_per_pixel_per_sec=0.0,
        exposure_sec=1.0,
        amplitude_e=1.0e5,
        rng=np.random.default_rng(9),
    )
    assert np.all(electrons == 0.0)


def _radiation_expected(
    exposure_sec: float, rate: float, environment_factor: float, readout_dwell_sec: float
) -> float:
    """The radiation_transients stage's expected hit count for one frame."""
    dp = resolve_detector_params({'instrument': 'coiss_nac', 'random_seed': 5})
    dp = dataclasses.replace(dp, exposure_sec=exposure_sec)
    cfg: dict[str, Any] = {
        'expected_events_per_sec': rate,
        'environment_factor': environment_factor,
        'readout_dwell_sec': readout_dwell_sec,
        'amplitude_e': 1000.0,
    }
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    return float(_apply_radiation_transients(electrons, cfg, dp)['expected'])


@pytest.mark.parametrize(
    ('exposure_sec', 'rate', 'environment_factor', 'readout_dwell_sec', 'expected'),
    [
        (2.0, 5.0, 1.0, 0.0, 10.0),
        (0.5, 5.0, 1.0, 0.0, 2.5),
        (2.0, 5.0, 3.0, 0.0, 30.0),
        (2.0, 5.0, 3.0, 4.0, 90.0),
        (0.0, 5.0, 1.0, 4.0, 20.0),
        (-1.0, 5.0, 1.0, 4.0, 20.0),
        (0.0, 5.0, 1.0, 0.0, 0.0),
    ],
    ids=[
        'rate_times_exposure',
        'short_exposure_expects_fewer',
        'environment_scales',
        'dwell_adds_its_seconds',
        'zero_exposure_counts_only_the_dwell',
        'negative_exposure_counts_only_the_dwell',
        'zero_exposure_and_dwell_expect_none',
    ],
)
def test_radiation_transients_expect_the_rate_times_the_exposed_seconds(
    exposure_sec: float,
    rate: float,
    environment_factor: float,
    readout_dwell_sec: float,
    expected: float,
) -> None:
    """The expected count is rate * environment_factor * (exposure_sec + readout_dwell_sec).

    A non-positive exposure contributes zero seconds, leaving only the dwell.
    """
    assert _radiation_expected(
        exposure_sec, rate, environment_factor, readout_dwell_sec
    ) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_radiation_amplitudes_fall_steeply() -> None:
    """The radiation regime's exponential amplitudes fall steeply below the scale."""
    electrons = np.zeros((256, 256), dtype=np.float64)
    deposit_morphological_events(
        electrons, n_events=300, amplitude_e=1000.0, rng=_rng(7), amplitude_dist='exponential'
    )
    deposits = electrons[electrons > 0.0]
    assert deposits.size > 0
    assert float(np.median(deposits)) < 500.0


def test_cosmic_amplitudes_sit_above_the_scale() -> None:
    """The cosmic-ray lognormal amplitudes sit at or above the deposit scale."""
    electrons = np.zeros((256, 256), dtype=np.float64)
    deposit_morphological_events(
        electrons, n_events=300, amplitude_e=1000.0, rng=_rng(7), amplitude_dist='lognormal'
    )
    deposits = electrons[electrons > 0.0]
    assert deposits.size > 0
    assert float(np.median(deposits)) > 1000.0


def test_radiation_morphology_mix_present() -> None:
    """The event mix carries single-pixel hits plus multi-pixel streaks / splatters."""
    electrons = np.zeros((256, 256), dtype=np.float64)
    deposit_morphological_events(
        electrons, n_events=300, amplitude_e=1000.0, rng=_rng(7), amplitude_dist='exponential'
    )
    labels, count = ndimage.label(electrons > 0.0)
    sizes = ndimage.sum_labels(np.ones_like(electrons), labels, index=range(1, count + 1))
    assert int((sizes == 1).sum()) > 0
    assert int((sizes >= 3).sum()) > 0
