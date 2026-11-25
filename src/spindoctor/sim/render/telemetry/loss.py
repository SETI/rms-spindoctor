"""The structured telemetry loss modes: what a downlink erases or mangles.

Each function here renders one loss mode from the artifact-mode registry onto a
detector-grid DN image in place, and returns the realized geometry (which lines,
blocks, or pixels it touched) for the frame's truth record so a later
measurement can compare planted against measured.  Every mode is a no-op when
its activation key is 0, or its ``count`` where given is 0 (the stage-activation
rule): a count mode draws a Poisson number of events with mean
``expected_events`` (or plants a fixed ``count``, on the dead-pixel and
dead-column modes), and a commanded, periodic, or frame-level mode fires with
per-frame ``probability``.  Two frame-level modes draw what they lose from the
frame itself or from measured tables: ``compression_truncated_lines`` cuts the
line pairs a lossless compressor cannot fit, and ``downlink_gaps`` loses groups of
lines whose count, sizes, and spacing it draws.  Each mode draws from its own seeded
generator (so toggling one never perturbs another), and honors adversarial
placement through the shared :mod:`~spindoctor.sim.render.feature_loci` helpers:
its stochastic placement biases onto the navigation features when adversarial is
on and is uniform otherwise.

The geometry is exact: a missing line is a whole line, missing blocks align to
the compression-block row grid, a spike flips a pixel to a wrong value rather
than a marker, and a garbled line carries noise rather than the zero marker.
The appliers share one keyword signature so the telemetry stage can dispatch
them from the registry's order list; each uses the arguments its shape needs.
The stage passes each applier the emulated instrument's ADC ceiling
(``dn_ceiling``), so the wrong values a garble or spike writes stay inside the
camera's word depth (255 on an 8-bit camera, 4095 on a 12-bit one).  On a
calibrated frame, which holds I/F, it also passes the frame's
:class:`~spindoctor.sim.render.detector.params.CalibrationMap`: garble and
spikes mangle the DN the spacecraft transmitted and write their I/F, so a
garbled Voyager pixel lands inside the I/F range an 8-bit word can calibrate to.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spindoctor.sim.artifact_modes import PERCENTILE_POINTS
from spindoctor.sim.render.detector.chain import keep_off_marker
from spindoctor.sim.render.detector.params import CalibrationMap
from spindoctor.sim.render.feature_loci import FeatureLoci, choose_pixels, choose_rows
from spindoctor.sim.render.telemetry.lossless import block_bits, pair_cuts
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'LOSS_APPLIERS',
    'apply_alternating_lines',
    'apply_compression_truncated_lines',
    'apply_cutout_window',
    'apply_dead_columns',
    'apply_dead_pixels',
    'apply_downlink_gaps',
    'apply_edited_frame',
    'apply_embedded_header',
    'apply_line_garble',
    'apply_missing_blocks',
    'apply_missing_lines',
    'apply_partial_lines',
    'apply_pixel_spikes',
    'apply_truncated_frame',
    'mode_activated',
]


def _poisson_count(rng: np.random.Generator, expected_events: float, cap: int) -> int:
    """Draw a Poisson event count with mean ``expected_events``, capped at ``cap``."""
    if expected_events <= 0.0:
        return 0
    return int(min(cap, rng.poisson(expected_events)))


def mode_activated(rng: np.random.Generator, probability: float) -> bool:
    """Whether a commanded or periodic telemetry mode fires this frame.

    Parameters:
        rng: The mode's seeded generator; one draw is taken when ``probability`` is
            positive.
        probability: The mode's per-frame ``probability``.

    Returns:
        False when ``probability`` is 0 or less, otherwise whether the draw falls
        below it.
    """
    if probability <= 0.0:
        return False
    return bool(rng.random() < probability)


def _as_dn(values: NDArrayFloatType, calibration: CalibrationMap | None) -> NDArrayFloatType:
    """The transmitted DN of frame values: themselves on a raw frame, mapped back from I/F."""
    return values if calibration is None else calibration.to_dn(values)


def _as_frame(dn: NDArrayFloatType, calibration: CalibrationMap | None) -> NDArrayFloatType:
    """Frame values for transmitted DN: themselves on a raw frame, their I/F otherwise."""
    return dn if calibration is None else calibration.to_if(dn)


def _as_valid_frame(
    dn: NDArrayFloatType, calibration: CalibrationMap | None, marker_dn: float
) -> NDArrayFloatType:
    """Frame values for transmitted DN that a calibrated frame keeps off its marker.

    Whole DN calibrate to a lattice of I/F values that can cross the numeric
    marker exactly, which a real calibrated product never does on a valid
    pixel, so the detector's calibration inverse moves such values off it.  A
    garbled or spiked value is a valid pixel too and gets the same step.  A raw
    frame's value is the transmitted DN itself, so a flip to the marker DN stays
    there, as it would in a real raw product.
    """
    values = _as_frame(dn, calibration)
    if calibration is not None:
        keep_off_marker(values, marker_dn)
    return values


def _word_bits(dn_ceiling: float) -> int:
    """The ADC word width in bits for a given DN ceiling (255 -> 8, 4095 -> 12)."""
    return max(1, int(math.log2(max(dn_ceiling, 1.0))) + 1)


def _percentile_draws(
    rng: np.random.Generator, percentiles: Any, count: int, *, log: bool = False
) -> NDArrayFloatType:
    """Draw ``count`` values from the distribution a percentile table gives.

    Parameters:
        rng: The mode's seeded generator; ``count`` uniform draws are taken.
        percentiles: The distribution's values at
            :data:`~spindoctor.sim.artifact_modes.PERCENTILE_POINTS`.
        count: The number of values to draw.
        log: Interpolate between the table's points in the logarithm of the value,
            which suits a size whose table spans decades; the table's values must then
            be positive.

    Returns:
        The values, linear in probability between the table's points.
    """
    table = np.asarray(percentiles, dtype=np.float64)
    draws = rng.random(count)
    if log:
        logs: NDArrayFloatType = np.exp(np.interp(draws, PERCENTILE_POINTS, np.log(table)))
        return logs
    values: NDArrayFloatType = np.interp(draws, PERCENTILE_POINTS, table)
    return values


def _kept_samples(rng: np.random.Generator, percentiles: Any, size_u: int, step: int) -> int:
    """Draw how many samples a line that loses its end keeps, in whole steps.

    The kept share of the line is drawn from ``percentiles`` and rounded to the nearest
    whole step, at least one step and at most all but one.
    """
    max_steps = max(1, (size_u - 1) // step)
    share = float(_percentile_draws(rng, percentiles, 1)[0])
    steps = int(np.clip(np.rint(share * size_u / step), 1, max_steps))
    return steps * step


def apply_missing_lines(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Zero-fill whole image lines: a missing line is a full line.

    A contiguous run loses a single band of adjacent lines; otherwise the lost
    lines are scattered.  Adversarial placement chooses lines that cross a
    feature (a run starts on a feature row; scattered lines are drawn from the
    feature rows).
    """
    expected_events = float(cfg['expected_events'])
    contiguous = bool(cfg['contiguous_run'])
    size_v = signal.shape[0]
    n = _poisson_count(rng, expected_events, size_v)
    if n == 0:
        return {'lines': []}
    if contiguous:
        if adversarial and loci.has_rows:
            start = int(rng.choice(loci.rows))
            start = min(start, size_v - n)
        else:
            start = int(rng.integers(0, size_v - n + 1))
        lines = list(range(start, start + n))
    else:
        lines = sorted(int(r) for r in choose_rows(loci, n, size_v, rng, adversarial=adversarial))
    signal[lines, :] = marker_dn
    return {'lines': lines}


def apply_partial_lines(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Truncate lines from a random column to the end (up to two surviving segments).

    With ``max_surviving_segments`` >= 2 a line may instead lose a middle
    segment, leaving two good segments -- the Cassini partial-line-segment
    shape, where one image line spans several telemetry packets.
    """
    expected_events = float(cfg['expected_events'])
    max_segments = int(cfg['max_surviving_segments'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, expected_events, size_v)
    if n == 0 or size_u < 2:
        return {'cuts': []}
    rows = choose_rows(loci, n, size_v, rng, adversarial=adversarial)
    cuts: list[dict[str, int]] = []
    for row in rows.tolist():
        # A middle-segment loss needs at least one surviving pixel on each
        # side of a nonempty cut, so it requires 3 or more columns; a 2-column
        # line always takes the truncation branch (the size guard precedes the
        # rng draw so wider frames consume the stream unchanged).
        if max_segments >= 2 and size_u >= 3 and bool(rng.random() < 0.5):
            a = int(rng.integers(1, size_u - 1))
            b = int(rng.integers(a + 1, size_u))
            signal[row, a:b] = marker_dn
            cuts.append({'row': int(row), 'lost_from': a, 'lost_to': b})
        else:
            k = int(rng.integers(1, size_u))
            signal[row, k:] = marker_dn
            cuts.append({'row': int(row), 'lost_from': k, 'lost_to': size_u})
    return {'cuts': cuts}


def apply_alternating_lines(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Blank lines on a periodic grid (the jail-bar alternating-line shape).

    Two semantics share the periodic grid, selected by ``mode``: ``drop``
    blanks every Nth line from the phase (the severe-Huffman entropy dropout,
    which loses one line per period), while ``keep`` blanks every line EXCEPT
    the Nth (the Galileo HMA / HCA vertical decimation, where only every Nth
    line carries valid data).  The pattern is periodic and deterministic once
    active, so adversarial placement does not steer it; the probability only
    decides whether the pattern fires this frame.
    """
    del loci, adversarial
    probability = float(cfg['probability'])
    period = int(cfg['period'])
    phase = int(cfg['phase']) % period
    line_mode = str(cfg['mode'])
    size_v = signal.shape[0]
    if not mode_activated(rng, probability):
        return {'active': False, 'lines': []}
    if line_mode == 'keep':
        lines = [row for row in range(size_v) if (row - phase) % period != 0]
    else:
        lines = list(range(phase, size_v, period))
    signal[lines, :] = marker_dn
    return {'active': True, 'mode': line_mode, 'period': period, 'phase': phase, 'lines': lines}


def apply_compression_truncated_lines(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    calibration: CalibrationMap | None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Cut the line pairs a lossless compressor with a fixed pair budget cannot fit.

    The compressor codes the transmitted DN (on a calibrated frame, the DN its
    calibration maps back to) as a block-adaptive Rice coder does (see
    :mod:`~spindoctor.sim.render.telemetry.lossless`): blocks of ``block_samples``
    first differences, each at the cheapest of the options that send 0 to
    ``max_split`` low bits as they are, plus ``id_bits`` naming the option.  Each pair
    of lines, rows 0 and 1, 2 and 3, and so on, may use ``word_bits`` bits per sample
    of one line.  The pair's second line keeps the blocks that still fit and loses the
    rest; if the first line alone does not fit, it keeps the blocks that do and the
    second line is lost whole.  How much is cut, and where, follows from the content:
    a smooth frame loses nothing, and noise or fine detail cuts the pairs it crosses.
    The probability decides whether the frame was compressed this way at all, and
    adversarial placement does not steer it.

    Returns:
        The record: whether the mode fired, the pair budget in bits, and one cut per
        line that lost samples, with the first sample it lost and the coded size of
        the samples its pair kept.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False, 'cuts': []}
    size_v, size_u = signal.shape
    block = int(cfg['block_samples'])
    budget = int(cfg['word_bits']) * size_u
    dn = np.rint(_as_dn(signal, calibration)).astype(np.int64)
    bits = block_bits(
        dn, block_samples=block, id_bits=int(cfg['id_bits']), max_split=int(cfg['max_split'])
    )
    kept = pair_cuts(bits, budget_bits=budget)
    cuts: list[dict[str, int]] = []
    for pair in range(size_v // 2):
        for row, blocks in (
            (2 * pair, int(kept.first_kept[pair])),
            (2 * pair + 1, int(kept.second_kept[pair])),
        ):
            lost_from = min(blocks * block, size_u)
            if lost_from == size_u:
                continue
            signal[row, lost_from:] = marker_dn
            cuts.append(
                {
                    'row': row,
                    'lost_from': lost_from,
                    'lost_to': size_u,
                    'pair_bits': int(kept.pair_bits[pair]),
                }
            )
    return {'active': True, 'budget_bits': budget, 'cuts': cuts}


def apply_edited_frame(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    **_ignored: Any,
) -> dict[str, Any]:
    """Keep only a centered vertical band of each line, or a half-height frame.

    An explicit ``half_frame`` keeps one half of the frame height (the 2:1
    scan modes); otherwise a band width keeps a centered column band and blanks
    the rest of every line (the edited downlink modes; the band width is the
    camera's shape default or the registry's fallback, so a bare probability
    renders the band shape).  A commanded mode: activation is by ``probability``.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False}
    size_v, size_u = signal.shape
    if bool(cfg.get('half_frame')):
        half = str(cfg['half'])
        mid = size_v // 2
        if half == 'top':
            signal[mid:, :] = marker_dn
            return {'active': True, 'kept_rows': [0, mid]}
        signal[:mid, :] = marker_dn
        return {'active': True, 'kept_rows': [mid, size_v]}
    band_width = cfg.get('band_width_px')
    if band_width is not None:
        width = int(band_width)
        u0 = max(0, (size_u - width) // 2)
        u1 = min(size_u, u0 + width)
        signal[:, :u0] = marker_dn
        signal[:, u1:] = marker_dn
        return {'active': True, 'kept_band': [u0, u1]}
    return {'active': False}


def apply_truncated_frame(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    **_ignored: Any,
) -> dict[str, Any]:
    """Cut a clean full-width band of lines from the bottom or top of the frame.

    An explicit ``lines`` count wins over ``fraction``; the registry's fraction
    default of 0.25 gives a bare probability the common quarter-frame truncation.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False}
    size_v = signal.shape[0]
    lines_param = cfg.get('lines')
    fraction = cfg.get('fraction')
    if lines_param is not None:
        m = int(lines_param)
    elif fraction is not None:
        m = round(float(fraction) * size_v)
    else:
        return {'active': False}
    m = min(max(m, 0), size_v)
    if m == 0:
        return {'active': False}
    from_edge = str(cfg['from'])
    if from_edge == 'bottom':
        rows = list(range(size_v - m, size_v))
    else:
        rows = list(range(0, m))
    signal[rows, :] = marker_dn
    return {'active': True, 'from': from_edge, 'lines': m}


def apply_missing_blocks(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    protect: tuple[int, int, int, int] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Zero-fill bands quantized to the compression-block row grid.

    Blocks align to the ``block_lines`` row grid (the height of a compression
    slice, the camera's shape default or the registry's fallback).  With
    ``start_mid_line`` the first row of a block is lost only from a random column
    to the right, then the remaining rows of the block are whole (the
    bit-error-mid-line shape).  A ``protect`` rectangle (the commanded
    truth window) is left untouched.  Adversarial placement chooses blocks whose
    rows cross a feature.
    """
    expected_events = float(cfg['expected_events'])
    block_lines = int(cfg['block_lines'])
    start_mid_line = bool(cfg['start_mid_line'])
    size_v, size_u = signal.shape
    n_grid = math.ceil(size_v / block_lines)
    n = _poisson_count(rng, expected_events, n_grid)
    if n == 0:
        return {'blocks': [], 'block_lines': block_lines}
    if adversarial and loci.has_rows:
        pool = np.unique(loci.rows // block_lines)
        replace = n > pool.size
        indices = rng.choice(pool, size=n, replace=replace)
    else:
        replace = n > n_grid
        indices = rng.choice(n_grid, size=n, replace=replace)

    saved = None
    if protect is not None:
        pv0, pv1, pu0, pu1 = protect
        saved = signal[pv0:pv1, pu0:pu1].copy()

    blocks: list[dict[str, int]] = []
    for idx in indices.tolist():
        r0 = int(idx) * block_lines
        r1 = min(size_v, r0 + block_lines)
        if start_mid_line and size_u > 0:
            k = int(rng.integers(0, size_u))
            signal[r0, k:] = marker_dn
            signal[r0 + 1 : r1, :] = marker_dn
            blocks.append({'row_start': r0, 'row_end': r1, 'start_col': k})
        else:
            signal[r0:r1, :] = marker_dn
            blocks.append({'row_start': r0, 'row_end': r1, 'start_col': 0})

    if protect is not None and saved is not None:
        pv0, pv1, pu0, pu1 = protect
        signal[pv0:pv1, pu0:pu1] = saved

    return {'blocks': blocks, 'block_lines': block_lines}


def _group_count(rng: np.random.Generator, mean_groups: float, dispersion: float) -> int:
    """Draw how many groups a frame that lost packets loses: one plus a negative binomial.

    The groups beyond the first have the mean ``mean_groups - 1``, and their variance is
    that mean plus ``dispersion`` times its square; a dispersion of 0 draws them Poisson.
    """
    extra = mean_groups - 1.0
    if dispersion > 0.0 and extra > 0.0:
        extra = float(rng.gamma(1.0 / dispersion, dispersion * extra))
    return 1 + int(rng.poisson(extra))


# How many uniform lines a group that starts a new cluster tries before it settles for
# one that overlaps a group already placed.
_FRESH_START_TRIES = 20


def _group_starts(
    rng: np.random.Generator,
    sizes: list[int],
    gap_percentiles: Any,
    size_v: int,
    first_rows: NDArrayIntType | None,
) -> list[int]:
    """Place groups of the given sizes down a frame; return each group's first line.

    Without ``gap_percentiles`` every group starts a cluster of its own.  With them the
    groups cluster: the first starts a cluster, and each later group starts a gap after
    the end of the one before, the gap drawn from the table in kept lines, interpolated
    in the logarithm, at least one.  A group that would run past the bottom of the
    frame starts a new cluster.  A group that starts a cluster starts on a uniform line
    that keeps it inside the frame, chosen where it neither overlaps nor touches a
    group already placed if such a line turns up in a few tries; with ``first_rows``,
    it starts on one of those rows instead.
    """
    lost = np.zeros(size_v + 2, dtype=bool)

    def place(start: int, size: int) -> int:
        """Record a group's lines, with a margin of one, and return its start."""
        lost[start : start + size + 2] = True
        return start

    def fresh(size: int) -> int:
        """A cluster's first line: a feature row, or a uniform line keeping it inside."""
        last = size_v - size
        if first_rows is not None:
            return min(int(rng.choice(first_rows)), last)
        start = int(rng.integers(0, last + 1))
        for _ in range(_FRESH_START_TRIES):
            if not lost[start : start + size + 2].any():
                break
            start = int(rng.integers(0, last + 1))
        return start

    starts: list[int] = []
    cursor: int | None = None
    for size in sizes:
        if gap_percentiles is None or cursor is None:
            start = fresh(size)
        else:
            gap = max(1, int(np.rint(_percentile_draws(rng, gap_percentiles, 1, log=True)[0])))
            start = cursor + gap
            if start + size > size_v:
                start = fresh(size)
        starts.append(place(start, size))
        cursor = start + size
    return starts


def apply_downlink_gaps(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Zero-fill groups of whole lines that packets lost on the way to the ground.

    A frame that lost packets, with the mode's probability, loses one group plus a
    negative-binomial number more, with the mean ``mean_groups`` in all and the
    ``dispersion`` that widens the count beyond Poisson.  Each group's size is drawn
    from ``size_percentiles``, in lines, interpolated in the logarithm and rounded to a
    whole line, at least one and at most the frame's height.  The groups start where
    the placement puts them: on uniform lines, or, with ``gap_percentiles``, in
    clusters with drawn gaps between them; with adversarial placement each cluster
    starts on a feature row.  With ``head_probability`` the line just above a group
    loses its end, and with ``tail_probability`` the line just below does; such a line
    keeps a whole number of ``step_samples``, its kept share drawn from
    ``head_kept_percentiles`` or ``tail_kept_percentiles``.  Groups that overlap merge
    in the frame and stay separate in the record.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False, 'groups': []}
    size_v, size_u = signal.shape
    n = min(_group_count(rng, float(cfg['mean_groups']), float(cfg['dispersion'])), size_v)
    sizes = np.clip(
        np.rint(_percentile_draws(rng, cfg['size_percentiles'], n, log=True)), 1, size_v
    ).astype(np.int64)
    first_rows = loci.rows if adversarial and loci.has_rows else None
    starts = _group_starts(rng, sizes.tolist(), cfg['gap_percentiles'], size_v, first_rows)
    step = int(cfg['step_samples'])
    groups: list[dict[str, int]] = []
    for start, size in zip(starts, sizes.tolist(), strict=True):
        end = start + int(size)
        signal[start:end, :] = marker_dn
        groups.append({'row_start': start, 'row_end': end})
    for group in groups:
        start, end = group['row_start'], group['row_end']
        if start > 0 and mode_activated(rng, float(cfg['head_probability'])):
            kept = _kept_samples(rng, cfg['head_kept_percentiles'], size_u, step)
            signal[start - 1, kept:] = marker_dn
            group['head_lost_from'] = kept
        if end < size_v and mode_activated(rng, float(cfg['tail_probability'])):
            kept = _kept_samples(rng, cfg['tail_kept_percentiles'], size_u, step)
            signal[end, kept:] = marker_dn
            group['tail_lost_from'] = kept
    return {'active': True, 'groups': groups}


def apply_line_garble(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    dn_ceiling: float,
    calibration: CalibrationMap | None = None,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Replace lines from a bit-error column onward with garbage (not markers).

    A variable-length code cannot resync mid-line, so the remainder of the line
    carries garbage values rather than the zero marker (Voyager IDC / Galileo
    Reed-Solomon overflow).  The garbage stays inside the instrument's ADC word
    (``dn_ceiling``), so an 8-bit camera never carries a 12-bit garbage value.
    The garbage is drawn in transmitted DN; on a calibrated frame
    (``calibration`` given) it is written as its I/F, kept off ``marker_dn``.
    """
    expected_events = float(cfg['expected_events'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, expected_events, size_v)
    if n == 0 or size_u < 1:
        return {'lines': []}
    rows = choose_rows(loci, n, size_v, rng, adversarial=adversarial)
    garbled: list[dict[str, int]] = []
    for row in rows.tolist():
        k = int(rng.integers(0, size_u))
        garbage = rng.uniform(0.0, dn_ceiling, size=size_u - k)
        signal[row, k:] = _as_valid_frame(garbage, calibration, marker_dn)
        garbled.append({'row': int(row), 'garble_from': k})
    return {'lines': garbled}


def apply_pixel_spikes(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    dn_ceiling: float,
    calibration: CalibrationMap | None = None,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Flip isolated pixels to wrong values (salt-and-pepper), never zeroing them.

    The ``bitflip`` amplitude model XORs a power-of-two bit into the integer DN
    (the Voyager uncompressed-era bit error, which shifts a pixel by a power of
    2); the ``uniform`` model replaces it with a random DN.  Both stay inside
    the instrument's ADC word: the flipped bit positions and the uniform draw
    are bounded by ``dn_ceiling``, so an 8-bit camera never spikes above 255.
    Both act on the transmitted DN; on a calibrated frame (``calibration``
    given) the pixel's I/F is mapped back to DN first and the result written as
    its I/F, kept off ``marker_dn``.
    """
    expected_events = float(cfg['expected_events'])
    model = str(cfg['amplitude'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, expected_events, size_v * size_u)
    if n == 0:
        return {'pixels': []}
    vs, us = choose_pixels(loci, n, (size_v, size_u), rng, adversarial=adversarial)
    if model == 'bitflip':
        bits = (1 << rng.integers(0, _word_bits(dn_ceiling), size=vs.size)).astype(np.int64)
        current = np.rint(_as_dn(signal[vs, us], calibration)).astype(np.int64)
        flipped = np.minimum((current ^ bits).astype(np.float64), dn_ceiling)
        signal[vs, us] = _as_valid_frame(flipped, calibration, marker_dn)
    else:
        draws = rng.uniform(0.0, dn_ceiling, size=vs.size)
        signal[vs, us] = _as_valid_frame(draws, calibration, marker_dn)
    pixels = [[int(v), int(u)] for v, u in zip(vs.tolist(), us.tolist(), strict=True)]
    return {'pixels': pixels}


def apply_dead_columns(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Set a fixed per-seed set of whole columns to a low response.

    A ``count`` fixes the number of dead columns; otherwise ``expected_events``
    is the Poisson mean (the validator refuses a map that sets both).  Either is
    a no-op at 0.  Adversarial placement prefers columns that cross a feature.
    """
    _size_v, size_u = signal.shape
    count = cfg.get('count')
    low_dn = float(cfg['low_dn'])
    n = (
        int(count)
        if count is not None
        else _poisson_count(rng, float(cfg['expected_events']), size_u)
    )
    n = min(n, size_u)
    if n == 0:
        return {'columns': []}
    if adversarial and loci.has_pixels:
        pool = np.unique(loci.pixel_u)
        replace = n > pool.size
        cols = np.unique(rng.choice(pool, size=n, replace=replace))
    else:
        cols = np.unique(rng.choice(size_u, size=n, replace=n > size_u))
    signal[:, cols] = low_dn
    return {'columns': sorted(int(c) for c in cols)}


def apply_dead_pixels(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
    loci: FeatureLoci,
    adversarial: bool,
    **_ignored: Any,
) -> dict[str, Any]:
    """Set a fixed per-seed set of singleton pixels to a low response.

    A ``count`` fixes the number of dead pixels; otherwise ``expected_events``
    is the Poisson mean (the validator refuses a map that sets both).  Either is
    a no-op at 0.  Adversarial placement prefers pixels on or beside a feature.
    """
    size_v, size_u = signal.shape
    count = cfg.get('count')
    low_dn = float(cfg['low_dn'])
    n = (
        int(count)
        if count is not None
        else _poisson_count(rng, float(cfg['expected_events']), size_v * size_u)
    )
    if n == 0:
        return {'pixels': []}
    vs, us = choose_pixels(loci, n, (size_v, size_u), rng, adversarial=adversarial)
    signal[vs, us] = low_dn
    pixels = [[int(v), int(u)] for v, u in zip(vs.tolist(), us.tolist(), strict=True)]
    return {'pixels': pixels}


def apply_embedded_header(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    rng: np.random.Generator,
    **_ignored: Any,
) -> dict[str, Any]:
    """Overwrite the first ``header_px`` pixels of row 0 with housekeeping values.

    An embedded header is binary housekeeping in row 0 of every image, never
    scene data; its length is the camera's shape default or the registry's
    fallback.  A commanded mode: activation is by ``probability``.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False}
    size_u = signal.shape[1]
    header_px = min(int(cfg['header_px']), size_u)
    if header_px <= 0:
        return {'active': False}
    signal[0, :header_px] = rng.integers(0, 256, size=header_px).astype(np.float64)
    return {'active': True, 'header_px': header_px}


def apply_cutout_window(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    marker_dn: float,
    rng: np.random.Generator,
    **_ignored: Any,
) -> dict[str, Any]:
    """Confine the scene to a commanded rectangle, hard-zeroing the border.

    The Galileo / LORRI windowed-downlink shape: only pixels inside the rect
    survive, everything else is blanked.  A commanded mode: activation is by
    ``probability``.
    """
    probability = float(cfg['probability'])
    if not mode_activated(rng, probability):
        return {'active': False}
    size_v, size_u = signal.shape
    rect = cfg.get('rect')
    if rect is not None:
        v0, v1, u0, u1 = (int(x) for x in rect)
    else:
        v0, v1 = size_v // 4, 3 * size_v // 4
        u0, u1 = size_u // 4, 3 * size_u // 4
    v0 = max(0, min(v0, size_v))
    v1 = max(v0, min(v1, size_v))
    u0 = max(0, min(u0, size_u))
    u1 = max(u0, min(u1, size_u))
    keep = signal[v0:v1, u0:u1].copy()
    signal[:, :] = marker_dn
    signal[v0:v1, u0:u1] = keep
    return {'active': True, 'rect': [v0, v1, u0, u1]}


# The dispatch table the telemetry stage reads, keyed by registry mode name.
# ``truth_window`` is absent: it is a protective carve-out resolved before the
# loop, not a signal-mutating loss.
LOSS_APPLIERS: dict[str, Any] = {
    'missing_lines': apply_missing_lines,
    'partial_lines': apply_partial_lines,
    'alternating_lines': apply_alternating_lines,
    'compression_truncated_lines': apply_compression_truncated_lines,
    'edited_frame': apply_edited_frame,
    'truncated_frame': apply_truncated_frame,
    'missing_blocks': apply_missing_blocks,
    'downlink_gaps': apply_downlink_gaps,
    'line_garble': apply_line_garble,
    'pixel_spikes': apply_pixel_spikes,
    'dead_columns': apply_dead_columns,
    'dead_pixels': apply_dead_pixels,
    'embedded_header': apply_embedded_header,
    'cutout_window': apply_cutout_window,
}
