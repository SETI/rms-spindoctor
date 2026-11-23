"""Hot pixels and the anti-blooming bright and dark pairs.

The hot pixels are driven both directly and through a rendered scene, whose truth
records the population it planted.
"""

from typing import cast

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene

from spindoctor.sim.render.detector.detector_maps import detector_hot_pixel_map
from spindoctor.sim.render.detector.pixel_defects import add_bright_dark_pairs, add_hot_pixels
from spindoctor.sim.render.driver import render_scene

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


def test_hot_pixels_elevate_a_sparse_population() -> None:
    """Hot pixels raise a fraction of pixels well above zero."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons,
        fraction=0.01,
        amplitude_e=1.0e4,
        column_factor=0.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(3),
    )
    assert int((electrons > 0.0).sum()) > 0


def test_hot_pixel_warm_column_is_a_total_charge_fraction() -> None:
    """The warm-column integral is bounded by column_factor times the charge.

    The column contamination is a fraction of the hot pixel's TOTAL charge
    spread along the column, not a per-pixel amplitude, so the streak stays
    conservative and frame-size-invariant.
    """
    electrons = np.zeros((512, 512), dtype=np.float64)
    column_factor = 0.3
    add_hot_pixels(
        electrons,
        fraction=1.0 / (512 * 512),  # exactly one hot pixel, drawn from a one-pixel pool
        amplitude_e=4.0e4,
        column_factor=column_factor,
        detector_id='coiss_nac',
        rng=np.random.default_rng(7),
        candidate_pool=(np.array([300]), np.array([200])),
    )
    hot_v, hot_u = np.unravel_index(int(np.argmax(electrons)), electrons.shape)
    hot_charge = float(electrons[hot_v, hot_u])
    column_integral = float(electrons[:hot_v, hot_u].sum())
    assert hot_charge > 0.0
    assert column_integral > 0.0
    assert column_integral <= column_factor * hot_charge + 1e-6


def test_hot_pixel_warm_column_is_frame_size_invariant() -> None:
    """The same column_factor bleeds the same total charge at any frame height."""
    integrals = []
    for size in (128, 512):
        electrons = np.zeros((size, size), dtype=np.float64)
        add_hot_pixels(
            electrons,
            fraction=1.0 / (size * size),
            amplitude_e=4.0e4,
            column_factor=0.3,
            detector_id='coiss_nac',
            rng=np.random.default_rng(7),
            candidate_pool=(np.array([100]), np.array([50])),
        )
        hot_v, hot_u = np.unravel_index(int(np.argmax(electrons)), electrons.shape)
        hot_charge = float(electrons[hot_v, hot_u])
        integrals.append(float(electrons[:hot_v, hot_u].sum()) / hot_charge)
    assert abs(integrals[0] - integrals[1]) < 1e-9


def test_adversarial_hot_pixels_are_distinct_and_capped_at_the_pool() -> None:
    """A request beyond the pool plants each pool pixel once, at its recorded charge."""
    electrons = np.zeros((64, 64), dtype=np.float64)
    pool_v, pool_u = np.divmod(np.arange(10, dtype=np.intp) * 37, 64)
    record = add_hot_pixels(
        electrons,
        fraction=25.0 / (64 * 64),
        amplitude_e=1.0e4,
        column_factor=0.0,
        detector_id='coiss_nac',
        rng=np.random.default_rng(11),
        candidate_pool=(pool_v, pool_u),
    )
    assert sorted(map(tuple, record['pixels'])) == sorted(
        zip(pool_v.tolist(), pool_u.tolist(), strict=True)
    )
    planted = [float(electrons[v, u]) for v, u in record['pixels']]
    assert planted == pytest.approx(record['amplitudes_e'])


def test_hot_pixels_disabled_at_zero_fraction() -> None:
    """A zero hot-pixel fraction is a no-op."""
    electrons = np.zeros((100, 100), dtype=np.float64)
    add_hot_pixels(
        electrons,
        fraction=0.0,
        amplitude_e=1.0e4,
        column_factor=0.5,
        detector_id='coiss_nac',
        rng=np.random.default_rng(3),
    )
    assert np.all(electrons == 0.0)


def test_hot_pixels_record_matches_planted_pixels() -> None:
    """The record's coordinates carry the record's charge on the electron plane."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_hot_pixels(
        electrons,
        fraction=0.01,
        amplitude_e=1.0e4,
        column_factor=0.0,
        detector_id='coiss_nac',
        rng=_rng(4),
    )
    assert record['pixels']
    for (v, u), amp in zip(record['pixels'], record['amplitudes_e'], strict=True):
        assert float(electrons[v, u]) == pytest.approx(amp)


def test_hot_pixels_mode_records_truth_end_to_end() -> None:
    """The hot_pixels mode writes its planted population into the frame truth."""
    scene = disc_scene(
        'coiss_nac', artifacts={'hot_pixels': {'pixel_fraction': 0.002, 'amplitude_e': 4.0e4}}
    )
    scene['bodies'] = []
    img, truth = render_scene(scene)
    record = truth['artifacts']['hot_pixels']
    assert record['pixels']
    assert len(record['amplitudes_e']) == len(record['pixels'])
    # The strongest planted pixel stands above the frame's background level.
    idx = int(np.argmax(record['amplitudes_e']))
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


def test_a_calibrated_product_shares_its_cameras_hot_pixels() -> None:
    """The calibrated NAC frame carries the raw NAC detector's hot pixels."""
    assert _hot_pixels_of('coiss_calib_nac', 1) == _hot_pixels_of('coiss_nac', 5)


def _nac_map_pixels() -> list[list[int]]:
    """The NAC detector's hot pixels in a 96 by 96 frame at the camera's fraction."""
    vs, us, _amps = detector_hot_pixel_map((96, 96), fraction=2.0e-3, detector_id='coiss_nac')
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
