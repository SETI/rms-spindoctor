"""Hot pixels and the anti-blooming bright and dark pairs.

The hot pixels are driven both directly and through a rendered scene, whose truth
records the population it planted.
"""

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene, frame_scene, quiet_noise

from spindoctor.sim.render.detector.detector_maps import AmplitudeLaw, detector_hot_pixel_map
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.detector.pixel_defects import (
    HotPixelPopulation,
    add_bright_dark_pairs,
    add_hot_pixels,
)
from spindoctor.sim.render.driver import render_scene

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


def _one_hot_pixel(
    *, exposure_sec: float = 1.0, binning: int = 1, **charges: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """A 64 by 64 plane holding one adversarial hot pixel at (30, 20), and its record.

    Parameters:
        exposure_sec: The frame's exposure.
        binning: The readout binning.
        **charges: The population's rate, readout, and streak terms; the rate
            defaults to 100 e-/s.
    """
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    population = dataclasses.replace(
        HotPixelPopulation(fraction=1.0 / (_SIZE * _SIZE), rate_e_per_sec=100.0), **charges
    )
    record = add_hot_pixels(
        electrons,
        population,
        exposure_sec=exposure_sec,
        detector_id='coiss_nac',
        rng=_rng(7),
        candidate_pool=(np.array([30]), np.array([20])),
        binning=binning,
    )
    return electrons, record


def _relative_amplitude() -> float:
    """The relative amplitude :func:`_one_hot_pixel` draws for its pixel."""
    electrons, _ = _one_hot_pixel(exposure_sec=1.0, rate_e_per_sec=1.0)
    return float(electrons[30, 20])


def test_hot_pixels_elevate_a_sparse_population() -> None:
    """Hot pixels raise a fraction of pixels well above zero."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons,
        HotPixelPopulation(fraction=0.01, rate_e_per_sec=1.0e4),
        exposure_sec=1.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(3),
    )
    assert int((electrons > 0.0).sum()) > 0


def test_hot_pixel_charge_scales_with_exposure() -> None:
    """Doubling the exposure doubles a hot pixel's dark charge."""
    short_e, _ = _one_hot_pixel(exposure_sec=1.0)
    long_e, _ = _one_hot_pixel(exposure_sec=2.0)
    assert float(long_e[30, 20]) == pytest.approx(2.0 * float(short_e[30, 20]))


def test_hot_pixel_holds_the_fixed_readout_charge_at_zero_exposure() -> None:
    """A frame of zero exposure holds readout_e times the relative amplitude."""
    electrons, _ = _one_hot_pixel(exposure_sec=0.0, readout_e=15.0)
    assert float(electrons[30, 20]) == pytest.approx(15.0 * _relative_amplitude())


def test_hot_pixel_collects_its_rate_for_the_readout_time_at_zero_exposure() -> None:
    """A frame of zero exposure holds the pixel's rate times readout_sec."""
    electrons, _ = _one_hot_pixel(exposure_sec=0.0, rate_e_per_sec=200.0, readout_sec=0.09)
    assert float(electrons[30, 20]) == pytest.approx(200.0 * 0.09 * _relative_amplitude())


def test_hot_pixel_streak_is_flat_on_the_rows_read_after_it() -> None:
    """Every row below a hot pixel gains the streak charge, and no row above it."""
    electrons, _ = _one_hot_pixel(rate_e_per_sec=0.0, readout_e=10.0, streak_e=2.0)
    relative = _relative_amplitude()
    assert np.all(electrons[:30, 20] == 0.0)
    assert electrons[31:, 20] == pytest.approx(np.full(_SIZE - 31, 2.0 * relative))


def test_hot_pixel_streak_follows_the_pixels_rate() -> None:
    """Each row below a hot pixel gains its rate times the dwell, streak_sec."""
    electrons, _ = _one_hot_pixel(rate_e_per_sec=3.0e4, streak_sec=0.0146)
    assert float(electrons[40, 20]) == pytest.approx(3.0e4 * 0.0146 * _relative_amplitude())


def test_binned_hot_pixel_streak_holds_each_native_rows_pickup() -> None:
    """A binned row below a hot pixel gains the streak of its binning native rows."""
    electrons, record = _one_hot_pixel(rate_e_per_sec=0.0, readout_e=10.0, streak_e=2.0, binning=2)
    relative = record['charges_e'][0] / 10.0
    assert float(electrons[40, 20]) == pytest.approx(2.0 * 2.0 * relative)


def test_binned_hot_pixel_row_holds_the_average_in_block_streak() -> None:
    """The hot pixel's own binned row gains (binning - 1) / 2 native rows' streak."""
    electrons, record = _one_hot_pixel(rate_e_per_sec=0.0, readout_e=10.0, streak_e=2.0, binning=4)
    charge = record['charges_e'][0]
    relative = charge / 10.0
    assert float(electrons[30, 20]) == pytest.approx(charge + 1.5 * 2.0 * relative)


def test_power_law_amplitudes_start_at_one() -> None:
    """The power law's smallest amplitude is 1."""
    law = AmplitudeLaw(kind='power_law', index=1.0, maximum=100.0)
    assert float(law.from_uniform(np.array([0.0]))[0]) == pytest.approx(1.0)


def test_power_law_amplitudes_end_at_the_maximum() -> None:
    """The power law's largest amplitude is its maximum."""
    law = AmplitudeLaw(kind='power_law', index=1.0, maximum=100.0)
    assert float(law.from_uniform(np.array([1.0 - 1.0e-15]))[0]) == pytest.approx(100.0)


def test_power_law_count_above_an_amplitude_follows_the_truncated_law() -> None:
    """Half the amplitudes lie above the amplitude where the law's survival is 1/2."""
    law = AmplitudeLaw(kind='power_law', index=1.0, maximum=100.0)
    # (a^-1 - 0.01) / (1 - 0.01) = 1/2 at a = 1 / 0.505.
    assert float(law.from_uniform(np.array([0.5]))[0]) == pytest.approx(1.0 / 0.505)


def test_an_unknown_amplitude_law_is_refused() -> None:
    """A law other than the two is refused by name."""
    with pytest.raises(ValueError, match="unknown hot-pixel amplitude law 'gaussian'"):
        AmplitudeLaw(kind='gaussian').from_uniform(np.array([0.5]))


def test_adversarial_hot_pixels_are_distinct_and_capped_at_the_pool() -> None:
    """A request beyond the pool plants each pool pixel once, at its recorded charge."""
    electrons = np.zeros((64, 64), dtype=np.float64)
    pool_v, pool_u = np.divmod(np.arange(10, dtype=np.intp) * 37, 64)
    record = add_hot_pixels(
        electrons,
        HotPixelPopulation(fraction=25.0 / (64 * 64), rate_e_per_sec=1.0e4),
        exposure_sec=1.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(11),
        candidate_pool=(pool_v, pool_u),
    )
    assert sorted(map(tuple, record['pixels'])) == sorted(
        zip(pool_v.tolist(), pool_u.tolist(), strict=True)
    )
    planted = [float(electrons[v, u]) for v, u in record['pixels']]
    assert planted == pytest.approx(record['charges_e'])


def test_hot_pixels_disabled_at_zero_fraction() -> None:
    """A zero hot-pixel fraction is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    population = HotPixelPopulation(
        fraction=0.0, rate_e_per_sec=1.0e4, readout_e=10.0, streak_e=2.0
    )
    add_hot_pixels(
        electrons,
        population,
        exposure_sec=1.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(3),
    )
    assert np.all(electrons == 0.0)


def test_hot_pixels_record_matches_planted_pixels() -> None:
    """The record's coordinates carry the record's charge on the electron plane."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_hot_pixels(
        electrons,
        HotPixelPopulation(fraction=0.01, rate_e_per_sec=1.0e4),
        exposure_sec=1.0,
        detector_id='coiss_nac',
        rng=_rng(4),
    )
    assert record['pixels']
    for (v, u), amp in zip(record['pixels'], record['charges_e'], strict=True):
        assert float(electrons[v, u]) == pytest.approx(amp)


def test_hot_pixels_mode_records_truth_end_to_end() -> None:
    """The hot_pixels mode writes its planted population into the frame truth."""
    scene = disc_scene(
        'coiss_nac', artifacts={'hot_pixels': {'pixel_fraction': 0.002, 'rate_e_per_sec': 40.0}}
    )
    scene['bodies'] = []
    img, truth = render_scene(scene)
    record = truth['artifacts']['hot_pixels']
    assert record['pixels']
    assert len(record['charges_e']) == len(record['pixels'])
    # The strongest planted pixel stands above the frame's background level.
    idx = int(np.argmax(record['charges_e']))
    v, u = record['pixels'][idx]
    assert float(img[v, u]) > float(np.median(img))


def _hot_pixels_of(instrument: str, seed: int) -> list[list[int]]:
    """The hot pixels an empty ``instrument`` frame plants at scene seed ``seed``."""
    scene = disc_scene(instrument, artifacts={'hot_pixels': {'pixel_fraction': 0.002}})
    scene['bodies'] = []
    scene['random_seed'] = seed
    _img, truth = render_scene(scene)
    return cast(list[list[int]], truth['artifacts']['hot_pixels']['pixels'])


def test_hot_pixels_recur_across_scene_seeds() -> None:
    """Two scenes on one instrument with different seeds plant the same hot pixels."""
    assert _hot_pixels_of('coiss_nac', 1) == _hot_pixels_of('coiss_nac', 2)


def test_a_calibrated_product_places_hot_pixels_on_its_cameras_map() -> None:
    """At one fraction, the calibrated NAC frame plants the raw NAC detector's pixels."""
    assert _hot_pixels_of('coiss_calib_nac', 1) == _hot_pixels_of('coiss_nac', 5)


def _nac_map_pixels() -> list[list[int]]:
    """The NAC detector's hot pixels in a 96 by 96 frame at the camera's fraction."""
    vs, us, _amps = detector_hot_pixel_map((96, 96), fraction=1.0e-2, detector_id='coiss_nac')
    return [[int(v), int(u)] for v, u in zip(vs, us, strict=True)]


def test_the_instrument_defaults_record_their_hot_pixels() -> None:
    """The hot pixels a camera's instrument defaults plant are in the truth."""
    _img, truth = render_scene(disc_scene('coiss_nac', artifacts={'instrument_defaults': True}))
    assert _nac_map_pixels()
    assert truth['artifacts']['hot_pixels']['pixels'] == _nac_map_pixels()


def test_the_defaults_hot_pixels_stay_on_the_map_under_the_adversarial_switch() -> None:
    """The camera's own hot pixels keep the detector's map in an adversarial scene."""
    artifacts = {'instrument_defaults': True, 'adversarial': True}
    _img, truth = render_scene(disc_scene('coiss_nac', artifacts=artifacts))
    assert truth['artifacts']['hot_pixels']['pixels'] == _nac_map_pixels()


def test_a_scene_fraction_places_hot_pixels_adversarially_under_the_defaults() -> None:
    """A scene that sets its own pixel_fraction draws it onto the features."""
    artifacts = {
        'instrument_defaults': True,
        'adversarial': True,
        'hot_pixels': {'pixel_fraction': 2.0e-3},
    }
    _img, truth = render_scene(disc_scene('coiss_nac', artifacts=artifacts))
    assert truth['artifacts']['hot_pixels']['pixels'] != _nac_map_pixels()


def test_bright_dark_pairs_are_vertical_and_charge_conserving() -> None:
    """Each pair raises one pixel and lowers the pixel below it by the same charge."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_bright_dark_pairs(electrons, count=5, amplitude_e=1000.0, rng=_rng(3))
    assert record['pairs']
    for v, u in record['pairs']:
        assert float(electrons[v, u]) == pytest.approx(1000.0)
        assert float(electrons[v + 1, u]) == pytest.approx(-1000.0)


def test_bright_dark_pairs_disabled_at_zero_count() -> None:
    """A zero pair count is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_bright_dark_pairs(electrons, count=0, amplitude_e=1000.0, rng=_rng(3))
    assert record['pairs'] == []
    assert np.all(electrons == 0.0)


def test_bright_dark_pairs_adversarial_land_on_the_pool() -> None:
    """An adversarial pool confines the pairs to the feature region."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    pool_v = np.arange(28, 36, dtype=np.int64)
    pool_u = np.full(pool_v.size, 40, dtype=np.int64)
    record = add_bright_dark_pairs(
        electrons, count=6, amplitude_e=1000.0, rng=_rng(3), candidate_pool=(pool_v, pool_u)
    )
    for v, _u in record['pairs']:
        assert 28 <= v <= 35


def test_a_binned_frame_plants_the_binned_map() -> None:
    """A SUM2 frame's hot pixels are the detector's map read through the binning."""
    scene = frame_scene(
        detector={'binning': 2},
        noise=quiet_noise(),
        artifacts={'hot_pixels': {'pixel_fraction': 0.01}},
    )
    _img, truth = render_scene(scene)
    vs, us, _amps = detector_hot_pixel_map(
        (64, 64), fraction=0.01, detector_id='coiss_nac', binning=2
    )
    expected = [[int(v), int(u)] for v, u in zip(vs, us, strict=True)]
    assert truth['artifacts']['hot_pixels']['pixels'] == expected


def _rendered_hot_pixels(
    exposure_sec: float, hot: dict[str, Any], **blocks: Any
) -> tuple[Any, Any]:
    """A 64 px NAC frame with an exponential hot-pixel population, rendered.

    Parameters:
        exposure_sec: The frame's exposure.
        hot: The hot_pixels keys besides the law; the fraction defaults to 0.002.
        **blocks: Further scene blocks; the noise block defaults to quiet.
    """
    hot_pixels = {'pixel_fraction': 0.002, 'amplitude_law': 'exponential'} | hot
    scene = frame_scene(
        exposure_sec=exposure_sec,
        detector={'quantization': 'exact', 'bias_dn': 0.0},
        artifacts={'hot_pixels': hot_pixels},
        **({'noise': quiet_noise()} | blocks),
    )
    img, truth = render_scene(scene)
    return img, truth['artifacts']['hot_pixels']


def test_rendered_hot_pixel_charge_follows_the_scene_exposure() -> None:
    """A scene of twice the exposure plants twice the charge in each hot pixel."""
    hot = {'rate_e_per_sec': 1.0e3, 'readout_sec': 0.0, 'streak_sec': 0.0}
    _img, short_record = _rendered_hot_pixels(0.1, hot)
    _img, long_record = _rendered_hot_pixels(0.2, hot)
    assert long_record['charges_e'] == pytest.approx([2.0 * c for c in short_record['charges_e']])


def test_a_rendered_hot_pixel_streaks_the_rows_read_after_it() -> None:
    """The column below a lone hot pixel carries the streak, and the column above none."""
    hot = {'rate_e_per_sec': 0.0, 'readout_e': 1.0e4, 'streak_e': 300.0}
    img, record = _rendered_hot_pixels(0.1, hot)
    columns = [u for _v, u in record['pixels']]
    charge, (v, u) = max(
        (charge, pixel)
        for pixel, charge in zip(record['pixels'], record['charges_e'], strict=True)
        if columns.count(pixel[1]) == 1 and 0 < pixel[0] < 63
    )
    gain = resolve_detector_params(frame_scene()).gain_e_per_dn
    # Each pixel rounds to a whole DN.
    per_row_dn = float(np.rint(300.0 * charge / 1.0e4 / gain))
    assert per_row_dn > 0.0
    assert np.all(img[:v, u] == 0.0)
    assert np.all(img[v + 1 :, u] == per_row_dn)


def test_a_rendered_hot_pixels_shot_noise_follows_its_charge() -> None:
    """Across noise seeds the hot pixels' variance in electrons is their charge.

    Hot-pixel charge is dark charge, so it joins the collected charge before the shot
    noise is drawn; added after it, a hot pixel's variance would be the background's.
    """
    hot = {'pixel_fraction': 0.05, 'rate_e_per_sec': 0.0, 'readout_e': 4.0e3}
    noise = {'poisson': True, 'read_noise_dn': 0.0}
    frames = []
    record: dict[str, Any] = {}
    for noise_seed in range(101, 141):
        img, record = _rendered_hot_pixels(0.01, hot, noise=noise, noise_seed=noise_seed)
        frames.append(np.asarray(img, dtype=np.float64))
    gain = resolve_detector_params(frame_scene()).gain_e_per_dn
    vs, us = np.array(record['pixels']).T
    variance_e2 = np.var(np.array(frames)[:, vs, us] * gain, axis=0, ddof=1)
    assert float(variance_e2.sum()) == pytest.approx(float(np.sum(record['charges_e'])), rel=0.15)
