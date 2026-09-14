"""The structured telemetry loss modes: what a downlink erases or mangles.

Each function here renders one loss mode from the artifact-mode registry onto a
detector-grid DN image in place, and returns the realized geometry (which lines,
blocks, or pixels it touched) for the frame's truth record so a later
incidence measurement can compare planted against measured.  Every mode is a
no-op at incidence 0 (the stage-activation rule), draws from its own seeded
generator (so toggling one never perturbs another), and honors adversarial
placement through the shared :mod:`~spindoctor.sim.forward.feature_loci` helpers:
its stochastic placement biases onto the navigation features when adversarial is
on and is uniform otherwise.

The geometry is exact: a missing line is a whole line, missing blocks align to
the compression-block row grid, a spike flips a pixel to a wrong value rather
than a marker, and a garbled line carries noise rather than the zero marker.
The appliers share one keyword signature so the telemetry stage can dispatch
them from the registry's order list; each uses the arguments its shape needs.
The stage passes each applier the emulated instrument's ADC ceiling
(``dn_ceiling``), so the wrong values a garble or spike writes stay inside the
camera's word depth (255 on an 8-bit camera, 4095 on a 12-bit one).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spindoctor.sim.forward.feature_loci import FeatureLoci, choose_pixels, choose_rows
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'LOSS_APPLIERS',
    'apply_alternating_lines',
    'apply_cutout_window',
    'apply_dead_columns',
    'apply_dead_pixels',
    'apply_edited_frame',
    'apply_embedded_header',
    'apply_line_garble',
    'apply_missing_blocks',
    'apply_missing_lines',
    'apply_partial_lines',
    'apply_pixel_spikes',
    'apply_truncated_frame',
]


def _poisson_count(rng: np.random.Generator, incidence: float, cap: int) -> int:
    """Draw a Poisson event count from an incidence, capped at ``cap``."""
    if incidence <= 0.0:
        return 0
    return int(min(cap, rng.poisson(incidence)))


def _activated(rng: np.random.Generator, incidence: float) -> bool:
    """Whether a commanded / periodic mode activates this frame (prob = incidence)."""
    if incidence <= 0.0:
        return False
    return bool(rng.random() < incidence)


def _garbage_ceiling(signal: NDArrayFloatType, dn_ceiling: float) -> float:
    """A plausible DN ceiling for garbage fills: the frame's peak, or the ADC word."""
    peak = float(signal.max()) if signal.size else 0.0
    return max(peak, dn_ceiling)


def _word_bits(dn_ceiling: float) -> int:
    """The ADC word width in bits for a given DN ceiling (255 -> 8, 4095 -> 12)."""
    return max(1, int(math.log2(max(dn_ceiling, 1.0))) + 1)


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
    incidence = float(cfg['incidence'])
    contiguous = bool(cfg['contiguous_run'])
    size_v = signal.shape[0]
    n = _poisson_count(rng, incidence, size_v)
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
    incidence = float(cfg['incidence'])
    max_segments = int(cfg['max_surviving_segments'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, incidence, size_v)
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
    active, so adversarial placement does not steer it; the incidence only
    decides whether the pattern fires this frame.
    """
    del loci, adversarial
    incidence = float(cfg['incidence'])
    period = int(cfg['period'])
    phase = int(cfg['phase']) % period
    line_mode = str(cfg['mode'])
    size_v = signal.shape[0]
    if not _activated(rng, incidence):
        return {'active': False, 'lines': []}
    if line_mode == 'keep':
        lines = [row for row in range(size_v) if (row - phase) % period != 0]
    else:
        lines = list(range(phase, size_v, period))
    signal[lines, :] = marker_dn
    return {'active': True, 'mode': line_mode, 'period': period, 'phase': phase, 'lines': lines}


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
    the rest of every line (the Voyager edited modes, whose registry default
    of 440 px matches the Voyager IM band widths, so a bare incidence renders
    the band shape).  A commanded mode: activation is by incidence.
    """
    incidence = float(cfg['incidence'])
    if not _activated(rng, incidence):
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
    default of 0.25 gives a bare incidence the common quarter-frame truncation.
    """
    incidence = float(cfg['incidence'])
    if not _activated(rng, incidence):
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

    Blocks align to the ``block_lines`` row grid (8 lines for Galileo ICT
    slices).  With ``start_mid_line`` the first row of a block is lost only from
    a random column to the right, then the remaining rows of the block are whole
    (the bit-error-mid-line ICT shape).  A ``protect`` rectangle (the commanded
    truth window) is left untouched.  Adversarial placement chooses blocks whose
    rows cross a feature.
    """
    incidence = float(cfg['incidence'])
    block_lines = int(cfg['block_lines'])
    start_mid_line = bool(cfg['start_mid_line'])
    size_v, size_u = signal.shape
    n_grid = math.ceil(size_v / block_lines)
    n = _poisson_count(rng, incidence, n_grid)
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


def apply_line_garble(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    dn_ceiling: float,
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
    """
    incidence = float(cfg['incidence'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, incidence, size_v)
    if n == 0 or size_u < 1:
        return {'lines': []}
    ceiling = _garbage_ceiling(signal, dn_ceiling)
    rows = choose_rows(loci, n, size_v, rng, adversarial=adversarial)
    garbled: list[dict[str, int]] = []
    for row in rows.tolist():
        k = int(rng.integers(0, size_u))
        signal[row, k:] = rng.uniform(0.0, ceiling, size=size_u - k)
        garbled.append({'row': int(row), 'garble_from': k})
    return {'lines': garbled}


def apply_pixel_spikes(
    signal: NDArrayFloatType,
    cfg: dict[str, Any],
    *,
    dn_ceiling: float,
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
    """
    incidence = float(cfg['incidence'])
    model = str(cfg['amplitude'])
    size_v, size_u = signal.shape
    n = _poisson_count(rng, incidence, size_v * size_u)
    if n == 0:
        return {'pixels': []}
    vs, us = choose_pixels(loci, n, (size_v, size_u), rng, adversarial=adversarial)
    if model == 'bitflip':
        bits = (1 << rng.integers(0, _word_bits(dn_ceiling), size=vs.size)).astype(np.int64)
        current = np.rint(signal[vs, us]).astype(np.int64)
        flipped = np.minimum((current ^ bits).astype(np.float64), dn_ceiling)
        signal[vs, us] = flipped
    else:
        signal[vs, us] = rng.uniform(0.0, _garbage_ceiling(signal, dn_ceiling), size=vs.size)
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

    A ``count`` fixes the number of dead columns; otherwise incidence is the
    Poisson mean.  Adversarial placement prefers columns that cross a feature.
    """
    _size_v, size_u = signal.shape
    count = cfg.get('count')
    low_dn = float(cfg['low_dn'])
    n = int(count) if count is not None else _poisson_count(rng, float(cfg['incidence']), size_u)
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

    A ``count`` fixes the number of dead pixels; otherwise incidence is the
    Poisson mean.  Adversarial placement prefers pixels on or beside a feature.
    """
    size_v, size_u = signal.shape
    count = cfg.get('count')
    low_dn = float(cfg['low_dn'])
    n = (
        int(count)
        if count is not None
        else _poisson_count(rng, float(cfg['incidence']), size_v * size_u)
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

    The LORRI embedded header is binary housekeeping in row 0 of every image,
    never scene data.  A commanded mode: activation is by incidence.
    """
    incidence = float(cfg['incidence'])
    if not _activated(rng, incidence):
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
    incidence.
    """
    incidence = float(cfg['incidence'])
    if not _activated(rng, incidence):
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
    'edited_frame': apply_edited_frame,
    'truncated_frame': apply_truncated_frame,
    'missing_blocks': apply_missing_blocks,
    'line_garble': apply_line_garble,
    'pixel_spikes': apply_pixel_spikes,
    'dead_columns': apply_dead_columns,
    'dead_pixels': apply_dead_pixels,
    'embedded_header': apply_embedded_header,
    'cutout_window': apply_cutout_window,
}
