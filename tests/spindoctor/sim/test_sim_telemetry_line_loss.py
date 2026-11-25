"""Compression-truncated lines and downlink gaps: the measured line-loss forms.

``compression_truncated_lines`` loses the end of the second line of every line pair
in whole compressor steps, with each line's lost share drawn from a percentile table.
``downlink_gaps`` loses groups of whole lines whose sizes are drawn log-normal, that
start on any line, and that may also lose the end of the line just above.  Each test
here covers one of those properties.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.sim.instruments.resolvers import mode_available
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.stage import apply_telemetry
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params

_FILL = 50.0
_TABLE = [0.015625, 0.0625, 0.109375, 0.1875, 0.34375, 0.375, 0.40625]


def _render(
    artifacts: dict[str, Any], *, size_v: int = 256, size_u: int = 256, seed: int = 7
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the telemetry stage to a flat generic frame; return its signal and records."""
    frame = SimFrame(
        signal=np.full((size_v, size_u), _FILL, dtype=np.float64),
        point_e=np.zeros((size_v, size_u), dtype=np.float64),
        oversample=1,
        truth={},
    )
    apply_telemetry(
        frame, params={'instrument': 'generic', 'random_seed': seed, 'artifacts': artifacts}
    )
    records: dict[str, Any] = frame.truth['artifacts']
    return frame.signal, records


def _truncated(**params: Any) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Render compression-truncated lines; return the signal and the cuts."""
    signal, records = _render({'compression_truncated_lines': {'probability': 1.0, **params}})
    cuts: list[dict[str, int]] = records['compression_truncated_lines']['cuts']
    return signal, cuts


def _gaps(**params: Any) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Render downlink gaps on a 1024-line frame; return the signal and the groups."""
    signal, records = _render({'downlink_gaps': params}, size_v=1024)
    groups: list[dict[str, int]] = records['downlink_gaps']['groups']
    return signal, groups


# --- compression_truncated_lines --------------------------------------------------


def test_truncation_cuts_every_odd_line_and_no_even_line() -> None:
    """The second line of every pair, the odd rows, loses its end, and no other row does."""
    signal, cuts = _truncated()
    touched = sorted(int(v) for v in np.nonzero((signal != _FILL).any(axis=1))[0])
    assert touched == list(range(1, 256, 2))
    assert [cut['row'] for cut in cuts] == touched


def test_truncation_keeps_a_whole_number_of_steps() -> None:
    """Each cut line keeps a whole number of 16-sample steps from its start."""
    signal, cuts = _truncated(step_samples=16)
    kept = [int(np.count_nonzero(signal[cut['row']] != 0.0)) for cut in cuts]
    assert sorted({k % 16 for k in kept}) == [0]


def test_truncation_loses_only_the_end_of_the_line() -> None:
    """A cut line keeps its start and loses everything from the cut to its end."""
    signal, cuts = _truncated(step_samples=16)
    row, lost_from = cuts[0]['row'], cuts[0]['lost_from']
    assert np.all(signal[row, :lost_from] == _FILL)
    assert np.all(signal[row, lost_from:] == 0.0)


def test_truncation_lost_shares_follow_the_percentile_table() -> None:
    """Over 2,048 lines the lost share sits within one step of the table's percentiles."""
    _signal, records = _render(
        {
            'compression_truncated_lines': {
                'probability': 1.0,
                'step_samples': 16,
                'lost_share_percentiles': _TABLE,
            }
        },
        size_v=4096,
        size_u=1024,
    )
    cuts = records['compression_truncated_lines']['cuts']
    shares = [1.0 - cut['lost_from'] / 1024 for cut in cuts]
    measured = np.percentile(shares, [10, 25, 50, 75, 90])
    assert np.max(np.abs(measured - np.array(_TABLE[1:6]))) <= 16 / 1024


def test_truncation_starts_at_the_start_fraction() -> None:
    """A lower section starts on the first odd row at or below its fraction of the frame."""
    _signal, cuts = _truncated(start_fraction=0.75)
    assert cuts[0]['row'] == 193


def test_a_decreasing_percentile_table_is_refused() -> None:
    """A percentile table whose values fall is refused when the scene is validated."""
    table = [0.1, 0.3, 0.2, 0.4, 0.5, 0.6, 0.7]
    scene = frame_scene(
        instrument='generic',
        scene_name='probe',
        artifacts={
            'compression_truncated_lines': {'probability': 1.0, 'lost_share_percentiles': table}
        },
    )
    with pytest.raises(SimSceneValidationError, match='decreases; each value must be at least'):
        validate_sim_params(scene)


# --- downlink_gaps -----------------------------------------------------------------


def test_gap_groups_are_whole_lines() -> None:
    """Every line of a recorded group is lost across its whole width."""
    signal, groups = _gaps(expected_events=6.0, median_lines=5.0, log_sigma=0.87)
    rows = [v for g in groups for v in range(g['row_start'], g['row_end'])]
    assert np.all(signal[rows] == 0.0)


def test_gap_sizes_follow_the_log_normal() -> None:
    """Group sizes have the stated median and spread of their natural log."""
    _signal, groups = _gaps(expected_events=400.0, median_lines=5.0, log_sigma=0.87)
    sizes = np.array([g['row_end'] - g['row_start'] for g in groups], dtype=np.float64)
    assert np.median(sizes) == 5.0
    assert abs(float(np.std(np.log(sizes))) - 0.87) < 0.1


def test_gap_groups_start_on_any_line() -> None:
    """Group starts fall on every residue of an 8-line grid."""
    _signal, groups = _gaps(expected_events=200.0, median_lines=5.0, log_sigma=0.87)
    assert sorted({g['row_start'] % 8 for g in groups}) == list(range(8))


def test_gap_head_line_loses_its_end_in_whole_steps() -> None:
    """With a head probability of 1, the line above each group loses its end in steps.

    A head line that another group loses whole is left out.
    """
    signal, groups = _gaps(
        expected_events=6.0, median_lines=5.0, head_probability=1.0, step_samples=16
    )
    lost = {v for g in groups for v in range(g['row_start'], g['row_end'])}
    heads = [g for g in groups if g['row_start'] > 0 and g['row_start'] - 1 not in lost]
    kept = [int(np.count_nonzero(signal[g['row_start'] - 1] != 0.0)) for g in heads]
    assert kept == [g['head_lost_from'] for g in heads]
    assert sorted({k % 16 for k in kept}) == [0]


def test_cassini_cameras_offer_both_forms() -> None:
    """The raw Cassini cameras offer both measured forms."""
    modes = ('compression_truncated_lines', 'downlink_gaps')
    offered = [mode_available(m, cam) for m in modes for cam in ('coiss_nac', 'coiss_wac')]
    assert offered == [True] * 4
