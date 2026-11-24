"""The maps fixed to the emulated detector: its hot pixels and its column pattern."""

import numpy as np
import pytest

from spindoctor.sim.render.detector.detector_maps import detector_column_map, detector_hot_pixel_map
from spindoctor.sim.render.detector.pixel_defects import add_hot_pixels


def _hot_set(shape: tuple[int, int], fraction: float, detector_id: str) -> set[tuple[int, int]]:
    """The detector map's hot pixels over ``shape`` as a coordinate set."""
    hot_v, hot_u, _amps = detector_hot_pixel_map(shape, fraction=fraction, detector_id=detector_id)
    return set(zip(hot_v.tolist(), hot_u.tolist(), strict=True))


def test_the_detector_map_ignores_the_scene_seed() -> None:
    """Two populations on one detector share every pixel and amplitude, whatever the rng."""
    planted = []
    for seed in (1, 99):
        electrons = np.zeros((256, 256), dtype=np.float64)
        add_hot_pixels(
            electrons,
            fraction=0.002,
            rate_e_per_sec=1.0e4,
            readout_e=0.0,
            streak_e=0.0,
            exposure_sec=1.0,
            detector_id='gossi',
            rng=np.random.default_rng(seed),
        )
        planted.append(electrons)
    assert np.array_equal(planted[0], planted[1])


def test_a_smaller_frame_sees_the_corner_of_the_detector_map() -> None:
    """The 128 px map is the 512 px map's top-left corner."""
    big = _hot_set((512, 512), 0.002, 'coiss_nac')
    corner = {(v, u) for v, u in big if v < 128 and u < 128}
    assert _hot_set((128, 128), 0.002, 'coiss_nac') == corner


def test_a_higher_fraction_only_adds_hot_pixels() -> None:
    """The map at a higher fraction contains the map at a lower one."""
    low = _hot_set((256, 256), 0.001, 'coiss_nac')
    high = _hot_set((256, 256), 0.004, 'coiss_nac')
    assert low < high


def test_two_detectors_have_different_maps() -> None:
    """The NAC and WAC maps differ."""
    assert _hot_set((256, 256), 0.002, 'coiss_nac') != _hot_set((256, 256), 0.002, 'coiss_wac')


def test_the_detector_map_count_follows_the_fraction() -> None:
    """A 1024 px map holds the expected count within five binomial sigmas."""
    fraction = 1.0e-3
    n = 1024 * 1024
    count = len(_hot_set((1024, 1024), fraction, 'coiss_nac'))
    sigma = (n * fraction * (1.0 - fraction)) ** 0.5
    assert abs(count - n * fraction) < 5.0 * sigma


def test_the_detector_map_amplitudes_average_one() -> None:
    """The relative amplitudes are exponential with mean 1 (within 10 percent on ~1000)."""
    _v, _u, amps = detector_hot_pixel_map((1024, 1024), fraction=1.0e-3, detector_id='gossi')
    assert float(amps.mean()) == pytest.approx(1.0, rel=0.1)


def test_the_column_map_is_standard_normal() -> None:
    """Over many columns the map has mean 0 and sigma 1."""
    values = detector_column_map(200_000, effect='calibration_columns', detector_id='coiss_nac')
    assert float(values.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(values.std()) == pytest.approx(1.0, abs=0.01)


def test_the_column_map_keys_each_column_on_the_detector_alone() -> None:
    """A narrower frame sees the first columns of the same map."""
    wide = detector_column_map(512, effect='calibration_columns', detector_id='coiss_nac')
    narrow = detector_column_map(128, effect='calibration_columns', detector_id='coiss_nac')
    assert np.array_equal(wide[:128], narrow)


@pytest.mark.parametrize(
    ('effect', 'detector_id'),
    [('calibration_columns', 'coiss_wac'), ('other_effect', 'coiss_nac')],
    ids=['other-detector', 'other-effect'],
)
def test_another_detector_or_effect_draws_another_map(effect: str, detector_id: str) -> None:
    """The map is keyed on both the effect and the detector."""
    base = detector_column_map(4096, effect='calibration_columns', detector_id='coiss_nac')
    other = detector_column_map(4096, effect=effect, detector_id=detector_id)
    assert abs(float(np.corrcoef(base, other)[0, 1])) < 0.05
