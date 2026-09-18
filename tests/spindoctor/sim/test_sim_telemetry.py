"""Telemetry structured-loss geometry: a missing line is a full line.

Every implemented artifact loss mode gets a geometry test here: the shape it
plants is exact (whole lines, grid-aligned blocks, spikes that flip rather than
zero), it is a no-op at incidence 0, its realized geometry is recorded in the
frame truth, and its adversarial placement lands on the navigation features.
The registry / validation surface is exercised alongside, since the two are one
contract: a mode is available on an instrument, or it is not.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifact_modes import (
    ARTIFACT_MODES,
    MODE_KEYS,
    STRUCTURED_LOSS_ORDER,
    mode_available,
    resolve_mode_config,
)
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.telemetry import apply_telemetry
from spindoctor.sim.scene import SimSceneValidationError, validate_sim_params

_SIZE = 64


def _frame(
    *, fill: float = 1.0, truth: dict[str, Any] | None = None, size: int = _SIZE
) -> SimFrame:
    """A detector-grid frame filled with a constant DN, plus optional truth."""
    return SimFrame(
        signal=np.full((size, size), fill, dtype=np.float64),
        point_e=np.zeros((size, size), dtype=np.float64),
        oversample=1,
        truth=truth or {},
    )


def _render(
    frame: SimFrame, artifacts: dict[str, Any], *, seed: int = 7, instrument: str = 'generic'
) -> dict[str, Any]:
    """Apply the telemetry stage with an artifacts block; return its truth record."""
    apply_telemetry(
        frame,
        params={'instrument': instrument, 'random_seed': seed, 'artifacts': artifacts},
        rng=np.random.default_rng(0),
    )
    record: dict[str, Any] = frame.truth.get('artifacts', {})
    return record


def _body_truth(v0: int, v1: int, u0: int, u1: int) -> dict[str, Any]:
    """A truth dict carrying one rectangular body mask spanning the given rows."""
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[v0:v1, u0:u1] = True
    return {'body_masks': [mask]}


# --- registry surface -------------------------------------------------------


def test_registry_covers_every_5_1_mode_key() -> None:
    """The registry holds exactly 31 mode keys (the 5.1 registry)."""
    assert len(MODE_KEYS) == 31


def test_structured_loss_order_covers_the_loss_modes() -> None:
    """The loss order holds exactly the telemetry loss modes.

    The truth window is a protective carve-out, and the compression and GEOMED
    archive-scar modes flank the loss loop in their own sub-stages, so none of
    those belong in the structured-loss order.
    """
    from spindoctor.sim.forward.artifact_modes import (
        TELEMETRY_POST_LOSS_ORDER,
        TELEMETRY_PRE_LOSS_ORDER,
    )

    telemetry_modes = {
        name
        for name, mode in ARTIFACT_MODES.items()
        if mode.implemented and mode.stage == 'telemetry' and name != 'truth_window'
    }
    flanking = set(TELEMETRY_PRE_LOSS_ORDER) | set(TELEMETRY_POST_LOSS_ORDER)
    assert set(STRUCTURED_LOSS_ORDER) == telemetry_modes - flanking


def test_generic_instrument_accepts_every_mode() -> None:
    """The generic block accepts every implemented mode (test-friendly)."""
    for name, mode in ARTIFACT_MODES.items():
        if mode.implemented:
            assert mode_available(name, 'generic')


# --- geometry: line and frame modes -----------------------------------------


def test_missing_lines_zeroes_whole_lines() -> None:
    """A missing line is a full line: the whole row reads the marker."""
    frame = _frame()
    record = _render(frame, {'missing_lines': {'incidence': 5.0}})
    lines = record['missing_lines']['lines']
    assert lines
    for row in lines:
        assert np.all(frame.signal[row] == 0.0)


def test_missing_lines_leaves_other_rows_untouched() -> None:
    """Rows not selected keep their original value."""
    frame = _frame()
    record = _render(frame, {'missing_lines': {'incidence': 5.0}})
    lines = set(record['missing_lines']['lines'])
    for row in range(_SIZE):
        if row not in lines:
            assert np.all(frame.signal[row] == 1.0)


def test_missing_lines_contiguous_run_is_adjacent() -> None:
    """A contiguous run loses one band of adjacent lines."""
    frame = _frame()
    record = _render(frame, {'missing_lines': {'incidence': 6.0, 'contiguous_run': True}})
    lines = record['missing_lines']['lines']
    assert lines == list(range(lines[0], lines[0] + len(lines)))


def test_missing_lines_disabled_at_zero_incidence() -> None:
    """A zero incidence plants nothing."""
    frame = _frame()
    record = _render(frame, {'missing_lines': {'incidence': 0.0}})
    assert record['missing_lines']['lines'] == []
    assert np.all(frame.signal == 1.0)


def test_partial_lines_truncate_from_a_column_to_the_end() -> None:
    """A partial line is zero-filled from a column to the line end."""
    frame = _frame()
    record = _render(frame, {'partial_lines': {'incidence': 5.0}}, instrument='coiss_nac')
    cuts = record['partial_lines']['cuts']
    assert cuts
    for cut in cuts:
        row, k, end = cut['row'], cut['lost_from'], cut['lost_to']
        assert np.all(frame.signal[row, k:end] == 0.0)
        assert np.all(frame.signal[row, :k] == 1.0)


def test_partial_lines_two_segments_keeps_a_trailing_segment() -> None:
    """With two surviving segments a middle segment can be lost, leaving a tail."""
    frame = _frame()
    record = _render(
        frame,
        {'partial_lines': {'incidence': 30.0, 'max_surviving_segments': 2}},
        instrument='coiss_nac',
    )
    cuts = record['partial_lines']['cuts']
    middles = [c for c in cuts if c['lost_to'] < _SIZE]
    assert middles
    cut = middles[0]
    assert np.all(frame.signal[cut['row'], cut['lost_to'] :] == 1.0)


def test_partial_lines_two_column_frame_truncates_without_error() -> None:
    """A 2-column frame cannot host a middle segment, so every cut truncates.

    A middle-segment loss needs a surviving pixel on each side of the cut;
    on a 2-column line the segment draw used to request an empty integer
    range and crash, so the degenerate width must fall back to truncation.
    """
    from spindoctor.sim.forward.feature_loci import FeatureLoci
    from spindoctor.sim.forward.telemetry_loss import apply_partial_lines

    signal = np.ones((16, 2), dtype=np.float64)
    empty = np.empty(0, dtype=np.int64)
    loci = FeatureLoci(rows=empty, pixel_v=empty, pixel_u=empty)
    record = apply_partial_lines(
        signal,
        {'incidence': 8.0, 'max_surviving_segments': 2},
        marker_dn=0.0,
        rng=np.random.default_rng(3),
        loci=loci,
        adversarial=False,
    )
    assert record['cuts']
    for cut in record['cuts']:
        assert cut['lost_to'] == 2
        assert cut['lost_from'] == 1


def test_alternating_lines_blanks_every_period_line() -> None:
    """Every Nth line from the phase is blanked when the mode fires."""
    frame = _frame()
    record = _render(
        frame,
        {'alternating_lines': {'incidence': 1.0, 'period': 2, 'phase': 0}},
        instrument='coiss_nac',
    )
    assert record['alternating_lines']['active'] is True
    assert record['alternating_lines']['lines'] == list(range(0, _SIZE, 2))
    assert np.all(frame.signal[0] == 0.0)
    assert np.all(frame.signal[1] == 1.0)


def test_alternating_lines_period_four_phase_one() -> None:
    """Period 4 with a phase offset blanks lines 1, 5, 9, ..."""
    frame = _frame()
    record = _render(
        frame,
        {'alternating_lines': {'incidence': 1.0, 'period': 4, 'phase': 1}},
        instrument='coiss_nac',
    )
    assert record['alternating_lines']['lines'] == list(range(1, _SIZE, 4))


def test_alternating_lines_drop_is_the_default_semantics() -> None:
    """The default 'drop' mode loses one line per period (a quarter at period 4)."""
    frame = _frame()
    record = _render(
        frame,
        {'alternating_lines': {'incidence': 1.0, 'period': 4}},
        instrument='coiss_nac',
    )
    assert record['alternating_lines']['mode'] == 'drop'
    assert len(record['alternating_lines']['lines']) == _SIZE // 4


def test_alternating_lines_keep_blanks_all_but_every_nth() -> None:
    """The 'keep' mode (HMA/HCA decimation) blanks 3/4 of the lines at period 4."""
    frame = _frame()
    record = _render(
        frame,
        {'alternating_lines': {'incidence': 1.0, 'period': 4, 'phase': 0, 'mode': 'keep'}},
        instrument='coiss_nac',
    )
    assert record['alternating_lines']['mode'] == 'keep'
    assert len(record['alternating_lines']['lines']) == _SIZE * 3 // 4
    assert np.all(frame.signal[0] == 1.0)
    assert np.all(frame.signal[1] == 0.0)
    assert np.all(frame.signal[4] == 1.0)


def test_gossi_catalog_defaults_alternating_lines_to_keep() -> None:
    """The Galileo catalog's HMA/HCA 'keep' default is honored by the loss loop."""
    frame = _frame()
    record = _render(
        frame, {'alternating_lines': {'incidence': 1.0, 'period': 4}}, instrument='gossi'
    )
    assert record['alternating_lines']['mode'] == 'keep'
    assert len(record['alternating_lines']['lines']) == _SIZE * 3 // 4


def test_edited_frame_keeps_only_a_centered_band() -> None:
    """An edited frame keeps a centered vertical band and blanks the rest."""
    frame = _frame()
    record = _render(
        frame, {'edited_frame': {'incidence': 1.0, 'band_width_px': 20}}, instrument='vgiss'
    )
    u0, u1 = record['edited_frame']['kept_band']
    assert np.all(frame.signal[:, u0:u1] == 1.0)
    assert np.all(frame.signal[:, :u0] == 0.0)
    assert np.all(frame.signal[:, u1:] == 0.0)


def test_edited_frame_bare_incidence_keeps_the_default_band() -> None:
    """A bare incidence renders the default 440-px Voyager IM centered band."""
    frame = _frame(size=512)
    record = _render(frame, {'edited_frame': {'incidence': 1.0}}, instrument='vgiss')
    u0, u1 = record['edited_frame']['kept_band']
    assert u1 - u0 == 440
    assert np.all(frame.signal[:, :u0] == 0.0)
    assert np.all(frame.signal[:, u1:] == 0.0)
    assert np.all(frame.signal[:, u0:u1] == 1.0)


def test_edited_frame_explicit_half_frame_wins_over_the_band_default() -> None:
    """An explicit half_frame renders the half shape, not the default band."""
    frame = _frame()
    record = _render(
        frame,
        {'edited_frame': {'incidence': 1.0, 'half_frame': True, 'half': 'bottom'}},
        instrument='vgiss',
    )
    assert record['edited_frame']['kept_rows'] == [_SIZE // 2, _SIZE]
    assert np.all(frame.signal[: _SIZE // 2] == 0.0)
    assert np.all(frame.signal[_SIZE // 2 :] == 1.0)


def test_edited_frame_half_frame_keeps_one_half() -> None:
    """A half-frame edit keeps the top half and blanks the bottom."""
    frame = _frame()
    record = _render(
        frame,
        {'edited_frame': {'incidence': 1.0, 'half_frame': True, 'half': 'top'}},
        instrument='vgiss',
    )
    assert record['edited_frame']['kept_rows'] == [0, _SIZE // 2]
    assert np.all(frame.signal[: _SIZE // 2] == 1.0)
    assert np.all(frame.signal[_SIZE // 2 :] == 0.0)


def test_truncated_frame_cuts_the_bottom() -> None:
    """A truncated frame removes a clean band of lines from the bottom."""
    frame = _frame()
    record = _render(
        frame,
        {'truncated_frame': {'incidence': 1.0, 'lines': 10, 'from': 'bottom'}},
        instrument='gossi',
    )
    assert record['truncated_frame']['lines'] == 10
    assert np.all(frame.signal[-10:] == 0.0)
    assert np.all(frame.signal[:-10] == 1.0)


def test_truncated_frame_bare_incidence_cuts_a_quarter() -> None:
    """A bare incidence renders the default quarter-frame bottom truncation."""
    frame = _frame()
    record = _render(frame, {'truncated_frame': {'incidence': 1.0}}, instrument='gossi')
    assert record['truncated_frame']['lines'] == _SIZE // 4
    assert np.all(frame.signal[-_SIZE // 4 :] == 0.0)
    assert np.all(frame.signal[: -_SIZE // 4] == 1.0)


def test_truncated_frame_fraction_from_top() -> None:
    """A fractional truncation from the top removes that fraction of lines."""
    frame = _frame()
    _render(
        frame,
        {'truncated_frame': {'incidence': 1.0, 'fraction': 0.25, 'from': 'top'}},
        instrument='gossi',
    )
    assert np.all(frame.signal[: _SIZE // 4] == 0.0)
    assert np.all(frame.signal[_SIZE // 4 :] == 1.0)


# --- geometry: block, garble, and pixel modes -------------------------------


def test_missing_blocks_align_to_the_block_grid() -> None:
    """Missing blocks start on the compression-block row grid."""
    frame = _frame()
    record = _render(
        frame, {'missing_blocks': {'incidence': 4.0, 'block_lines': 8}}, instrument='gossi'
    )
    blocks = record['missing_blocks']['blocks']
    assert blocks
    for block in blocks:
        assert block['row_start'] % 8 == 0
        assert np.all(frame.signal[block['row_start'] : block['row_end']] == 0.0)


def test_missing_blocks_start_mid_line_runs_right() -> None:
    """A mid-line block loses its first row from a column to the right, then whole rows."""
    frame = _frame()
    record = _render(
        frame,
        {'missing_blocks': {'incidence': 40.0, 'block_lines': 8, 'start_mid_line': True}},
        instrument='gossi',
    )
    mid = next(b for b in record['missing_blocks']['blocks'] if b['start_col'] > 0)
    r0, k = mid['row_start'], mid['start_col']
    assert np.all(frame.signal[r0, :k] == 1.0)
    assert np.all(frame.signal[r0, k:] == 0.0)
    assert np.all(frame.signal[r0 + 1 : mid['row_end']] == 0.0)


def test_truth_window_is_untouched_by_missing_blocks() -> None:
    """A commanded truth window stays clean even under heavy block loss."""
    frame = _frame()
    record = _render(
        frame,
        {
            'missing_blocks': {'incidence': 60.0, 'block_lines': 8},
            'truth_window': {'incidence': 1.0, 'size': 16, 'position': [24, 24]},
        },
        instrument='gossi',
    )
    assert record['truth_window']['rect'] == [24, 40, 24, 40]
    assert np.all(frame.signal[24:40, 24:40] == 1.0)


def test_line_garble_replaces_with_nonmarker_values() -> None:
    """A garbled line carries garbage from a column, not the zero marker."""
    frame = _frame(fill=100.0)
    record = _render(frame, {'line_garble': {'incidence': 6.0}}, instrument='vgiss')
    garbled = record['line_garble']['lines']
    assert garbled
    entry = garbled[0]
    tail = frame.signal[entry['row'], entry['garble_from'] :]
    # Garbage spans a range of values rather than a single marker fill.
    assert float(tail.std()) > 0.0


def test_pixel_spikes_flip_pixels_without_zeroing() -> None:
    """A pixel spike changes a pixel to a wrong value, never zeroing it."""
    frame = _frame(fill=100.0)
    record = _render(frame, {'pixel_spikes': {'incidence': 10.0}}, instrument='vgiss')
    pixels = record['pixel_spikes']['pixels']
    assert pixels
    for v, u in pixels:
        assert frame.signal[v, u] != 100.0


def test_pixel_spikes_bitflip_shifts_by_a_power_of_two() -> None:
    """The bitflip model shifts a pixel by a power of two."""
    frame = _frame(fill=100.0)
    record = _render(
        frame, {'pixel_spikes': {'incidence': 20.0, 'amplitude': 'bitflip'}}, instrument='vgiss'
    )
    v, u = record['pixel_spikes']['pixels'][0]
    delta = abs(round(float(frame.signal[v, u])) - 100)
    assert delta > 0
    assert (delta & (delta - 1)) == 0  # a power of two


def test_vgiss_pixel_spikes_stay_inside_the_8bit_word() -> None:
    """Voyager bit-flip spikes never exceed the vidicon's 255 DN ceiling."""
    frame = _frame(fill=200.0)
    record = _render(frame, {'pixel_spikes': {'incidence': 50.0}}, instrument='vgiss')
    assert record['pixel_spikes']['pixels']
    assert float(frame.signal.max()) <= 255.0


def test_vgiss_uniform_spikes_stay_inside_the_8bit_word() -> None:
    """Voyager uniform spikes draw inside the 255 DN word, not the 12-bit one."""
    frame = _frame(fill=200.0)
    record = _render(
        frame, {'pixel_spikes': {'incidence': 50.0, 'amplitude': 'uniform'}}, instrument='vgiss'
    )
    assert record['pixel_spikes']['pixels']
    assert float(frame.signal.max()) <= 255.0


def test_vgiss_line_garble_stays_inside_the_8bit_word() -> None:
    """Voyager garbage fills stay inside the vidicon's 255 DN ceiling."""
    frame = _frame(fill=200.0)
    record = _render(frame, {'line_garble': {'incidence': 20.0}}, instrument='vgiss')
    assert record['line_garble']['lines']
    assert float(frame.signal.max()) <= 255.0


def test_12bit_line_garble_keeps_the_deep_word() -> None:
    """A 12-bit detector's garbage still spans DN above 255 up to 4095."""
    frame = _frame(fill=200.0)
    record = _render(frame, {'line_garble': {'incidence': 20.0}})
    assert record['line_garble']['lines']
    assert float(frame.signal.max()) > 255.0
    assert float(frame.signal.max()) <= 4095.0


def test_unavailable_loss_mode_via_dict_path_is_skipped() -> None:
    """A mode unavailable on the instrument is skipped even without validation.

    Scenes that reach the telemetry stage as raw dicts bypass the scene
    validator, so the loss loop itself must honor the registry availability.
    """
    frame = _frame(fill=200.0)
    record = _render(frame, {'pixel_spikes': {'incidence': 50.0}}, instrument='coiss_nac')
    assert 'pixel_spikes' not in record
    assert np.all(frame.signal == 200.0)


def test_dead_pixels_fixed_count_sets_low_response() -> None:
    """A fixed dead-pixel count sets exactly that many pixels low."""
    frame = _frame(fill=50.0)
    record = _render(frame, {'dead_pixels': {'count': 12, 'low_dn': 0.0}}, instrument='coiss_nac')
    pixels = record['dead_pixels']['pixels']
    assert len(pixels) == 12
    for v, u in pixels:
        assert frame.signal[v, u] == 0.0


def test_dead_columns_blank_whole_columns() -> None:
    """A dead column reads low over the whole column height."""
    frame = _frame(fill=50.0)
    record = _render(frame, {'dead_columns': {'count': 3, 'low_dn': 0.0}}, instrument='coiss_nac')
    columns = record['dead_columns']['columns']
    assert len(columns) >= 1
    for col in columns:
        assert np.all(frame.signal[:, col] == 0.0)


def test_embedded_header_overwrites_row_zero_prefix() -> None:
    """The embedded header overwrites the first header_px pixels of row 0."""
    frame = _frame(fill=200.0)
    record = _render(
        frame, {'embedded_header': {'incidence': 1.0, 'header_px': 34}}, instrument='nhlorri'
    )
    assert record['embedded_header']['header_px'] == 34
    assert np.all(frame.signal[0, 34:] == 200.0)
    assert np.any(frame.signal[0, :34] != 200.0)


def test_cutout_window_hard_zeroes_the_border() -> None:
    """A cut-out window keeps the rect and hard-zeroes everything else."""
    frame = _frame(fill=50.0)
    record = _render(
        frame,
        {'cutout_window': {'incidence': 1.0, 'rect': [10, 40, 12, 44]}},
        instrument='gossi',
    )
    v0, v1, u0, u1 = record['cutout_window']['rect']
    assert np.all(frame.signal[v0:v1, u0:u1] == 50.0)
    border = frame.signal.copy()
    border[v0:v1, u0:u1] = 0.0
    assert np.all(border == 0.0)


# --- adversarial placement --------------------------------------------------


def test_adversarial_missing_line_intersects_the_limb_rows() -> None:
    """An adversarial missing line lands within the body's limb row range."""
    frame = _frame(truth=_body_truth(30, 51, 20, 44))
    record = _render(frame, {'adversarial': True, 'missing_lines': {'incidence': 3.0}}, seed=9)
    for line in record['missing_lines']['lines']:
        assert 30 <= line <= 50


def test_adversarial_placement_is_deterministic_per_seed() -> None:
    """Adversarial placement is byte-identical for equal seeds."""
    artifacts = {'adversarial': True, 'missing_lines': {'incidence': 3.0}}
    frame_a = _frame(truth=_body_truth(30, 51, 20, 44))
    frame_b = _frame(truth=_body_truth(30, 51, 20, 44))
    record_a = _render(frame_a, artifacts, seed=15)
    record_b = _render(frame_b, artifacts, seed=15)
    assert record_a['missing_lines']['lines'] == record_b['missing_lines']['lines']


def test_uniform_placement_ignores_features() -> None:
    """Without adversarial, line placement is not confined to the feature rows."""
    frame = _frame(truth=_body_truth(30, 34, 20, 44))
    record = _render(frame, {'missing_lines': {'incidence': 20.0}}, seed=3)
    lines = record['missing_lines']['lines']
    assert any(not (30 <= line <= 33) for line in lines)


# --- adversarial hot-pixel routing (detector stage) -------------------------


def test_adversarial_hot_pixels_concentrate_on_features() -> None:
    """Adversarial hot pixels land on and beside the navigation features."""
    from spindoctor.sim.forward.detector.noise_stages import add_hot_pixels
    from spindoctor.sim.forward.feature_loci import dilated_pixels, extract_feature_loci

    truth = _body_truth(28, 36, 28, 36)
    loci = extract_feature_loci(truth, (_SIZE, _SIZE))
    pool = dilated_pixels(loci, radius=3, shape=(_SIZE, _SIZE))
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_hot_pixels(
        electrons,
        fraction=0.02,
        amplitude_e=1.0e4,
        column_factor=0.0,
        rng=np.random.default_rng(4),
        candidate_pool=pool,
    )
    hot = np.argwhere(electrons > 0.0)
    # Every hot pixel sits within the dilated feature region (rows 25..38).
    assert hot.size > 0
    assert np.all((hot[:, 0] >= 25) & (hot[:, 0] <= 38))


def test_empty_loci_pool_falls_back_to_uniform() -> None:
    """With no features the dilation pool is empty and placement is uniform."""
    from spindoctor.sim.forward.feature_loci import dilated_pixels, extract_feature_loci

    loci = extract_feature_loci({}, (_SIZE, _SIZE))
    pool_v, pool_u = dilated_pixels(loci, radius=3, shape=(_SIZE, _SIZE))
    assert pool_v.size == 0
    assert pool_u.size == 0


# --- validation -------------------------------------------------------------


def _base_scene(instrument: str = 'generic') -> dict[str, Any]:
    """A minimal validatable scene mapping for the given instrument."""
    return {
        'schema_version': 2,
        'scene_name': 'probe',
        'instrument': instrument,
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
    }


def test_unknown_artifact_key_fails() -> None:
    """An unknown key in the artifacts block fails validation."""
    scene = {**_base_scene(), 'artifacts': {'not_a_mode': {}}}
    with pytest.raises(SimSceneValidationError, match='unknown keys'):
        validate_sim_params(scene)


def test_detector_mode_validates_when_available() -> None:
    """An implemented detector mode validates on an instrument it is available on."""
    scene = {**_base_scene('coiss_nac'), 'artifacts': {'bloom': {'incidence': 0.5}}}
    validate_sim_params(scene)


def test_bloom_on_lorri_fails_with_bespoke_message() -> None:
    """bloom on LORRI fails: an antiblooming CCD has no column bloom."""
    scene = {**_base_scene('nhlorri'), 'artifacts': {'bloom': {'incidence': 0.5}}}
    with pytest.raises(SimSceneValidationError, match='antiblooming'):
        validate_sim_params(scene)


def test_hot_pixels_on_lorri_fails_with_bespoke_message() -> None:
    """hot_pixels on LORRI fails with the explicit LORRI message."""
    scene = {**_base_scene('nhlorri'), 'artifacts': {'hot_pixels': {'incidence': 0.01}}}
    with pytest.raises(SimSceneValidationError, match='disabled for LORRI'):
        validate_sim_params(scene)


def test_partial_lines_unavailable_on_galileo_fails() -> None:
    """A mode with no Galileo catalog entry fails on a Galileo scene."""
    scene = {**_base_scene('gossi'), 'artifacts': {'partial_lines': {'incidence': 1.0}}}
    with pytest.raises(SimSceneValidationError, match='not available'):
        validate_sim_params(scene)


def test_unknown_mode_param_fails() -> None:
    """An unknown parameter inside a mode map fails validation."""
    scene = {**_base_scene(), 'artifacts': {'missing_lines': {'incidence': 1.0, 'bogus': 1}}}
    with pytest.raises(SimSceneValidationError, match='unknown keys'):
        validate_sim_params(scene)


def test_enum_param_rejects_bad_value() -> None:
    """An out-of-set enum value fails validation."""
    scene = {
        **_base_scene('coiss_nac'),
        'artifacts': {'alternating_lines': {'incidence': 1.0, 'period': 3}},
    }
    with pytest.raises(SimSceneValidationError, match='must be one of'):
        validate_sim_params(scene)


def test_adversarial_flag_is_a_boolean() -> None:
    """The adversarial switch must be a boolean."""
    scene = {**_base_scene(), 'artifacts': {'adversarial': 'yes'}}
    with pytest.raises(SimSceneValidationError, match='adversarial'):
        validate_sim_params(scene)


def test_resolve_mode_config_fills_defaults() -> None:
    """Resolving a mode config fills unset parameters with their defaults."""
    resolved = resolve_mode_config('missing_lines', {'incidence': 2.0})
    assert resolved == {'incidence': 2.0, 'contiguous_run': False}


# --- telemetry-artifact modes: compression and GEOMED scars -----------------


def test_compression_dct_changes_the_frame_and_is_blockwise() -> None:
    """DCT compression quantizes 8x8 coefficients, altering a textured frame."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_compression_dct

    rng = np.random.default_rng(0)
    signal = rng.uniform(0.0, 100.0, size=(_SIZE, _SIZE))
    before = signal.copy()
    record = apply_compression_dct(
        signal,
        {'incidence': 1.0, 'scale_factor': 40.0, 'block': 8},
        rng=np.random.default_rng(1),
        protect=None,
    )
    assert record['active'] is True
    assert not np.array_equal(signal, before)


def test_compression_dct_leaves_the_truth_window_clean() -> None:
    """A protected truth window survives compression untouched."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_compression_dct

    rng = np.random.default_rng(2)
    signal = rng.uniform(0.0, 100.0, size=(_SIZE, _SIZE))
    protect = (16, 32, 16, 32)
    saved = signal[16:32, 16:32].copy()
    apply_compression_dct(
        signal,
        {'incidence': 1.0, 'scale_factor': 40.0, 'block': 8},
        rng=np.random.default_rng(1),
        protect=protect,
    )
    assert np.array_equal(signal[16:32, 16:32], saved)


def test_compression_dct_disabled_at_zero_incidence() -> None:
    """A zero incidence leaves the frame untouched."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_compression_dct

    rng = np.random.default_rng(3)
    signal = rng.uniform(0.0, 100.0, size=(_SIZE, _SIZE))
    before = signal.copy()
    record = apply_compression_dct(
        signal,
        {'incidence': 0.0, 'scale_factor': 40.0, 'block': 8},
        rng=np.random.default_rng(1),
        protect=None,
    )
    assert record['active'] is False
    assert np.array_equal(signal, before)


def test_reseau_scars_smooth_lattice_patches() -> None:
    """Reseau scars smooth patches on the lattice, lowering local variation."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_reseau_scars

    rng = np.random.default_rng(4)
    signal = rng.uniform(0.0, 100.0, size=(128, 128))
    record = apply_reseau_scars(
        signal,
        {'incidence': 1.0, 'spacing_px': 46, 'patch_radius_px': 4},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is True
    assert record['marks'] > 0


def test_reseau_lattice_spans_the_archive_frame() -> None:
    """At the 800x800 archive size the 202 marks span the full frame height."""
    from spindoctor.sim.forward.telemetry_artifacts import _reseau_lattice

    points = _reseau_lattice(800, 800, 46)
    assert len(points) == 202
    rows = [v for v, _u in points]
    assert max(rows) - min(rows) >= 0.9 * 800


def test_reseau_scars_disabled_at_zero_incidence() -> None:
    """A zero incidence plants no scars."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_reseau_scars

    rng = np.random.default_rng(5)
    signal = rng.uniform(0.0, 100.0, size=(128, 128))
    before = signal.copy()
    record = apply_reseau_scars(
        signal,
        {'incidence': 0.0, 'spacing_px': 46, 'patch_radius_px': 4},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is False
    assert np.array_equal(signal, before)


def test_resample_texture_blank_border_zeroes_the_edge() -> None:
    """The resample blank border zeroes a frame of the requested width."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_resample_texture

    signal = np.full((_SIZE, _SIZE), 50.0, dtype=np.float64)
    record = apply_resample_texture(
        signal,
        {'incidence': 1.0, 'warp_amp_px': 0.0, 'blank_border_px': 3, 'missing_line_interp': False},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is True
    assert np.all(signal[:3, :] == 0.0)
    assert np.all(signal[-3:, :] == 0.0)
    assert float(signal[_SIZE // 2, _SIZE // 2]) == 50.0


def test_resample_texture_disabled_at_zero_incidence() -> None:
    """A zero incidence leaves the frame untouched."""
    from spindoctor.sim.forward.telemetry_artifacts import apply_resample_texture

    signal = np.full((_SIZE, _SIZE), 50.0, dtype=np.float64)
    before = signal.copy()
    record = apply_resample_texture(
        signal,
        {'incidence': 0.0, 'warp_amp_px': 0.3, 'blank_border_px': 3, 'missing_line_interp': True},
        rng=np.random.default_rng(1),
    )
    assert record['active'] is False
    assert np.array_equal(signal, before)


def test_telemetry_artifact_modes_record_truth_end_to_end() -> None:
    """Compression and reseau scars record their truth through the telemetry stage."""
    frame = _frame(fill=80.0, size=96)
    record = _render(
        frame,
        {'compression_dct': {'incidence': 1.0, 'scale_factor': 30.0}},
        instrument='gossi',
    )
    assert record['compression_dct']['active'] is True
