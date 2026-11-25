"""Compression-truncated lines and downlink gaps: the measured line-loss forms.

``compression_truncated_lines`` codes each pair of lines with a block-adaptive Rice
coder and cuts the second line where the pair exceeds its budget, so the frame's
content decides what is lost.  ``downlink_gaps`` loses groups of whole lines: one plus
a negative-binomial number of them, sized from a table, clustered by a gap table, with
partial lines above and below.  Each test here covers one property.
"""

import itertools
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene

from spindoctor.sim.instruments.resolvers import instrument_default_modes, mode_available
from spindoctor.sim.render.detector.params import calibration_map, resolve_detector_params
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.loss import apply_compression_truncated_lines
from spindoctor.sim.render.telemetry.stage import apply_telemetry
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params

_FILL = 500.0
_RICE = {'block_samples': 16, 'word_bits': 16, 'id_bits': 4, 'max_split': 13}
_NAC_SIZES = [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.2, 83.2, 373.0]
_NAC_GAPS = [1.0, 1.0, 1.0, 2.0, 6.0, 19.0, 54.8, 114.3, 331.5, 822.0]
_NAC_HEAD_KEPT = [0.0156, 0.0312, 0.0781, 0.2031, 0.3438, 0.5625, 0.75, 0.875, 0.9688, 0.9688]


def _compress(signal: np.ndarray, probability: float = 1.0, **params: Any) -> dict[str, Any]:
    """Run the compressor on a raw-DN frame in place; return its record."""
    cfg = {'probability': probability, **_RICE, **params}
    return apply_compression_truncated_lines(
        signal, cfg, marker_dn=0.0, rng=np.random.default_rng(0), calibration=None
    )


def _noise(size_v: int, size_u: int, sigma: float, seed: int = 1) -> np.ndarray:
    """A frame of integer DN about 2,000 with Gaussian noise of the given spread."""
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(2000.0 + sigma * rng.standard_normal((size_v, size_u))), 0, 4095)


def _gaps(
    params: dict[str, Any], *, size_v: int = 1024, size_u: int = 256, seed: int = 7
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Render downlink gaps on a flat generic frame; return the signal and the groups."""
    frame = SimFrame(
        signal=np.full((size_v, size_u), _FILL, dtype=np.float64),
        point_e=np.zeros((size_v, size_u), dtype=np.float64),
        oversample=1,
        truth={},
    )
    artifacts = {'downlink_gaps': {'probability': 1.0, **params}}
    apply_telemetry(
        frame, params={'instrument': 'generic', 'random_seed': seed, 'artifacts': artifacts}
    )
    groups: list[dict[str, int]] = frame.truth['artifacts']['downlink_gaps']['groups']
    return frame.signal, groups


def _many_groups(**params: Any) -> list[dict[str, int]]:
    """Draw about 400 separate 2-line groups on a tall frame."""
    base = {'mean_groups': 400.0, 'size_percentiles': [2.0] * 10, 'step_samples': 16}
    return _gaps({**base, **params}, size_v=8192)[1]


# --- compression_truncated_lines --------------------------------------------------


def test_a_flat_frame_loses_nothing() -> None:
    """A flat frame codes far below its budget, so no line is cut."""
    record = _compress(np.full((64, 1024), _FILL))
    assert record['cuts'] == []


def test_the_cut_falls_at_the_block_the_rice_cost_predicts() -> None:
    """A flat line paired with an alternating line keeps the blocks that fit the budget.

    The flat line codes each of its 64 blocks at split 0: 16 bits plus 4 option bits,
    1,280 bits in all.  The second line alternates 0 and 2048, so its differences map
    to 4096 and 4095.  Its first block, whose first difference is 0, codes at split 11
    in 192 + 23 + 4 = 219 bits, and every later block at split 11 or 12 in 220 bits.
    With 8 bits a sample the pair may use 8,192 bits, which leaves 6,912 for the
    second line: 219 + 220 * 30 = 6,819 fits and one more block does not, so the line
    keeps 31 blocks, 496 samples.
    """
    signal = np.zeros((2, 1024))
    signal[1, 1::2] = 2048.0
    record = _compress(signal, word_bits=8)
    assert [(c['row'], c['lost_from']) for c in record['cuts']] == [(1, 496)]


def test_a_cut_line_keeps_whole_blocks_from_its_start() -> None:
    """Every cut keeps a whole number of 16-sample blocks and loses the rest."""
    signal = _noise(64, 1024, 300.0)
    before = signal.copy()
    cuts = _compress(signal)['cuts']
    assert len(cuts) == 32
    for cut in cuts:
        row, lost_from = cut['row'], cut['lost_from']
        assert lost_from % 16 == 0
        assert np.array_equal(signal[row, :lost_from], before[row, :lost_from])
        assert not signal[row, lost_from:].any()


def test_only_the_second_line_of_a_pair_is_cut() -> None:
    """Where the first line of each pair fits, only the odd rows lose samples."""
    record = _compress(_noise(64, 1024, 300.0))
    assert [c['row'] for c in record['cuts']] == list(range(1, 64, 2))


def test_a_first_line_over_the_budget_loses_the_second_line_whole() -> None:
    """A pair whose first line alone exceeds the budget loses all of its second line."""
    record = _compress(_noise(2, 1024, 300.0), word_bits=4)
    assert [(c['row'], c['lost_from'] % 16) for c in record['cuts']] == [(0, 0), (1, 0)]
    assert record['cuts'][1]['lost_from'] == 0


def test_only_the_pairs_whose_content_exceeds_the_budget_are_cut() -> None:
    """Noise across rows 100 to 199 of a flat frame cuts lines there and nowhere else."""
    signal = np.full((256, 1024), _FILL)
    signal[100:200] = _noise(100, 1024, 300.0)
    rows = [c['row'] for c in _compress(signal)['cuts']]
    assert (min(rows), max(rows)) == (101, 199)


def test_the_probability_decides_whether_the_frame_was_compressed() -> None:
    """A frame whose compression draw fails keeps every line."""
    record = _compress(_noise(64, 1024, 300.0), probability=1e-12)
    assert record['cuts'] == []


def test_a_calibrated_frame_is_cut_in_its_transmitted_dn() -> None:
    """The I/F frame of a calibrated camera loses the lines its DN frame loses."""
    params = frame_scene(instrument='coiss_calib_nac', size=96, seed=7, exposure_sec=2.5)
    calibration = calibration_map(resolve_detector_params(params))
    assert calibration is not None
    dn = _noise(16, 1024, 300.0)
    cfg = {'probability': 1.0, **_RICE}
    raw = apply_compression_truncated_lines(
        dn.copy(), cfg, marker_dn=0.0, rng=np.random.default_rng(0), calibration=None
    )
    calibrated = apply_compression_truncated_lines(
        calibration.to_if(dn),
        cfg,
        marker_dn=0.0,
        rng=np.random.default_rng(0),
        calibration=calibration,
    )
    assert len(raw['cuts']) == 8
    assert calibrated['cuts'] == raw['cuts']


def test_an_eight_bit_readout_refuses_the_compressor() -> None:
    """A scene that quantizes to 8 bits cannot set the 12-bit compressor."""
    scene = frame_scene(
        instrument='coiss_nac',
        scene_name='probe',
        detector={'quantization': 'ls8b'},
        artifacts={'compression_truncated_lines': {'probability': 1.0}},
    )
    with pytest.raises(SimSceneValidationError, match="quantizes with 'ls8b'"):
        validate_sim_params(scene)


def test_a_moon_on_dark_sky_fits_the_compressor() -> None:
    """A Cassini frame of a moon on dark sky, noise and all, loses no line to it."""
    scene = frame_scene(
        instrument='coiss_nac',
        size=256,
        seed=7,
        artifacts={
            'instrument_defaults': True,
            'compression_truncated_lines': {'probability': 1.0},
        },
        bodies=[disc_body(128.0, 128.0, 60.0, name='RHEA', illumination_angle=18.0)],
    )
    _img, truth = render_scene(scene)
    assert truth['artifacts']['compression_truncated_lines']['cuts'] == []


# --- downlink_gaps -----------------------------------------------------------------


def test_gap_groups_are_whole_lines() -> None:
    """Every line of a recorded group is lost across its whole width."""
    signal, groups = _gaps({'mean_groups': 6.0, 'size_percentiles': _NAC_SIZES})
    rows = [v for g in groups for v in range(g['row_start'], g['row_end'])]
    assert np.all(signal[rows] == 0.0)


def test_every_frame_that_loses_packets_loses_a_group() -> None:
    """With a mean of one group, every frame loses exactly one."""
    counts = {len(_gaps({}, size_v=64, seed=seed)[1]) for seed in range(40)}
    assert counts == {1}


def test_the_group_count_widens_with_the_dispersion() -> None:
    """The groups beyond the first have the stated mean and negative-binomial variance.

    With a mean of 3 groups and a dispersion of 2, the extra groups have the mean 2 and
    the variance 2 + 2 * 2 ** 2 = 10.
    """
    params = {'mean_groups': 3.0, 'dispersion': 2.0, 'size_percentiles': [1.0] * 10}
    counts = np.array(
        [len(_gaps(params, size_v=4096, size_u=4, seed=seed)[1]) for seed in range(3000)]
    )
    assert abs(float(counts.mean()) - 3.0) < 0.15
    assert abs(float(counts.var()) - 10.0) < 1.5


def test_gap_sizes_follow_the_size_table() -> None:
    """Group sizes reproduce the table, interpolated in the logarithm between its points.

    Halfway in probability between the 95th and 99th percentiles, 30.2 and 83.2 lines,
    the logarithmic interpolation gives 50.1 lines, where a linear one would give 56.7.
    """
    params = {'mean_groups': 5000.0, 'size_percentiles': _NAC_SIZES}
    groups = _gaps(params, size_v=2**20, size_u=4)[1]
    sizes = np.array([g['row_end'] - g['row_start'] for g in groups], dtype=np.float64)
    assert np.percentile(sizes, [25, 50, 75]).tolist() == [3.0, 5.0, 8.0]
    assert abs(float(np.percentile(sizes, 97)) - 50.1) < 3.0


def test_gap_groups_start_on_any_line() -> None:
    """Starts of equal 7-line groups fall on every residue of 7 and of 8."""
    groups = _gaps({'mean_groups': 200.0, 'size_percentiles': [7.0] * 10}, size_v=8192)[1]
    starts = [g['row_start'] for g in groups]
    assert sorted({s % 7 for s in starts}) == list(range(7))
    assert sorted({s % 8 for s in starts}) == list(range(8))


def test_clustered_groups_are_a_drawn_gap_apart() -> None:
    """Within a cluster the kept lines between groups follow the gap table."""
    params = {
        'mean_groups': 40.0,
        'size_percentiles': [3.0] * 10,
        'gap_percentiles': _NAC_GAPS,
    }
    gaps: list[int] = []
    for seed in range(20):
        groups = _gaps(params, size_v=16384, seed=seed)[1]
        gaps += [b['row_start'] - a['row_end'] for a, b in itertools.pairwise(groups)]
    within = [g for g in gaps if g > 0]
    assert np.percentile(within, [50, 75]).tolist() == pytest.approx([6.0, 19.0], rel=0.25)


def test_a_cluster_that_reaches_the_bottom_starts_again_inside_the_frame() -> None:
    """Groups that would run past the bottom start a new cluster, so all stay inside."""
    params = {
        'mean_groups': 60.0,
        'size_percentiles': [9.0] * 10,
        'gap_percentiles': [30.0] * 10,
    }
    groups = _gaps(params, size_v=512)[1]
    assert len(groups) > 15
    assert max(g['row_end'] for g in groups) <= 512


def test_the_head_probability_sets_the_share_of_groups_with_a_head_line() -> None:
    """With a head probability of 0.5, about half the groups cut the line above."""
    groups = _many_groups(head_probability=0.5)
    rate = np.mean(['head_lost_from' in g for g in groups])
    assert abs(float(rate) - 0.5) < 0.07


def test_a_head_line_keeps_at_least_one_step_and_loses_at_least_one() -> None:
    """The line above a group keeps a whole number of steps, never none and never all."""
    params = {
        'mean_groups': 6.0,
        'size_percentiles': [3.0] * 10,
        'head_probability': 1.0,
        'step_samples': 16,
    }
    signal, groups = _gaps(params)
    lost = {v for g in groups for v in range(g['row_start'], g['row_end'])}
    heads = [g for g in groups if g['row_start'] > 0 and g['row_start'] - 1 not in lost]
    kept = [int(np.count_nonzero(signal[g['row_start'] - 1])) for g in heads]
    assert kept == [g['head_lost_from'] for g in heads]
    assert all(k % 16 == 0 for k in kept)
    assert min(kept) >= 16
    assert max(kept) <= 240


def test_head_kept_shares_follow_their_table() -> None:
    """The kept share of the head lines reproduces the table's median within a step."""
    groups = _many_groups(head_probability=1.0, head_kept_percentiles=_NAC_HEAD_KEPT)
    shares = [g['head_lost_from'] / 256 for g in groups if 'head_lost_from' in g]
    assert abs(float(np.median(shares)) - 0.3438) <= 16 / 256


def test_the_tail_probability_sets_the_share_of_groups_with_a_tail_line() -> None:
    """With a tail probability of 0.5, about half the groups cut the line below."""
    groups = _many_groups(tail_probability=0.5)
    rate = np.mean(['tail_lost_from' in g for g in groups])
    assert abs(float(rate) - 0.5) < 0.07


def test_a_tail_line_keeps_its_start_and_loses_its_end() -> None:
    """The line below a group keeps whole steps from its start and loses the rest."""
    params = {
        'mean_groups': 6.0,
        'size_percentiles': [3.0] * 10,
        'tail_probability': 1.0,
        'step_samples': 16,
    }
    signal, groups = _gaps(params)
    lost = {v for g in groups for v in range(g['row_start'], g['row_end'])}
    tails = [g for g in groups if g['row_end'] < 1024 and g['row_end'] not in lost]
    kept = [int(np.count_nonzero(signal[g['row_end']])) for g in tails]
    assert kept == [g['tail_lost_from'] for g in tails]
    assert all(np.all(signal[g['row_end'], : g['tail_lost_from']] == _FILL) for g in tails)


@pytest.mark.parametrize(
    ('key', 'value', 'message'),
    [
        ('mean_groups', 0.5, 'expected a number of at least 1'),
        ('size_percentiles', [0.0, *_NAC_SIZES[1:]], 'expected a positive number'),
        ('size_percentiles', [1.0, 3.0, 2.0, *_NAC_SIZES[3:]], 'decreases'),
    ],
    ids=['mean-below-one', 'zero-size', 'falling-table'],
)
def test_an_invalid_gap_parameter_is_refused(key: str, value: Any, message: str) -> None:
    """A mean below one group, a size of 0, and a falling table are each refused."""
    scene = frame_scene(
        instrument='generic',
        scene_name='probe',
        artifacts={'downlink_gaps': {'probability': 1.0, key: value}},
    )
    with pytest.raises(SimSceneValidationError, match=message):
        validate_sim_params(scene)


# --- the Cassini cameras -------------------------------------------------------------


def test_cassini_cameras_offer_both_forms() -> None:
    """The raw Cassini cameras offer both measured forms."""
    modes = ('compression_truncated_lines', 'downlink_gaps')
    offered = [mode_available(m, cam) for m in modes for cam in ('coiss_nac', 'coiss_wac')]
    assert offered == [True] * 4


def test_instrument_defaults_turn_neither_form_on() -> None:
    """A scene must turn the compressor and the downlink gaps on itself."""
    defaults = {*instrument_default_modes('coiss_nac'), *instrument_default_modes('coiss_wac')}
    assert defaults.isdisjoint({'compression_truncated_lines', 'downlink_gaps'})
