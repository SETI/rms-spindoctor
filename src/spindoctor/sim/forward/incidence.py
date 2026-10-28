"""Artifact-incidence measurement: planted truth vs image-measured counts.

The realism match compares how many defects a scene *planted* against how many a
detector could *measure* on the rendered frame.  This module carries both sides:

* :func:`planted_incidence` reads the realized geometry the telemetry stage
  wrote into ``frame.truth['artifacts']`` and returns a realized event count per
  mode (lost lines, truncated lines, lost blocks, dropped or spiked pixels, dead
  columns, or a commanded on/off).  This is ground truth: it is exact by
  construction, since each applier records exactly what it touched.

* The ``measured_*`` estimators recover the same counts from the DN image alone,
  the way a detector run against a real archive frame would, with no access to
  the truth record.  They key on the missing-data marker (the instrument's
  ``noise.marker_value``: 0 for every shipped instrument, raw and calibrated), so
  they are exact for the marker-based structural modes and only those.

**Scope of the image estimators.**  A lost line, a truncated line, a lost block,
and a dead pixel each overwrite pixels with the marker value, so a marker-run
analysis recovers them exactly on a frame whose scene content is not itself at
the marker level (a body or star frame with a non-zero bias floor).  The
dark-sky floor of an unbiased frame sits at the marker and is indistinguishable
from a dropout, and so does the part of a ``vgiss`` frame's calibrated dark sky
that renders at exactly 0 I/F: all of it on a noise-free frame, and much of it
under ``instrument_defaults``, where quantization leaves many sky pixels at 0.
Modes that plant *wrong* values rather than the marker -- garble (random DN),
pixel spikes (bit-flips), and every detector-electronics mode -- are not
recoverable pixel-by-pixel from a single frame: distinguishing them from
scene structure needs multi-frame or noise statistics, which is out of scope
here.  For those modes, use :func:`planted_incidence` (the truth side) only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'marker_mask',
    'measured_missing_blocks',
    'measured_missing_lines',
    'measured_partial_lines',
    'measured_pixel_dropouts',
    'planted_incidence',
]

# The per-mode truth-record fields whose list length is the realized event
# count, tried in order.  A record that carries none of them but does carry an
# ``active`` flag is a commanded on/off mode (count 1 when active, else 0).
_COUNT_FIELDS: tuple[str, ...] = ('lines', 'cuts', 'blocks', 'pixels', 'columns')


def _realized_count(record: Mapping[str, Any]) -> int:
    """The realized event count for one mode's truth record.

    Parameters:
        record: A ``frame.truth['artifacts'][mode]`` map.

    Returns:
        The number of geometric events the mode planted: the length of its
        geometry list (lost lines, cuts, blocks, pixels, or columns), or 1/0 for
        a commanded on/off mode.
    """
    for field in _COUNT_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            return len(value)
    if 'active' in record:
        return 1 if bool(record['active']) else 0
    return 0


def planted_incidence(truth: Mapping[str, Any]) -> dict[str, int]:
    """Realized per-mode event counts from a rendered frame's truth record.

    Parameters:
        truth: A rendered frame's ``truth`` mapping (``frame.truth``); its
            ``artifacts`` sub-map holds one record per applied mode.  A frame
            with no artifacts returns an empty mapping.

    Returns:
        A mapping from mode name to its realized event count, for every mode the
        telemetry stage recorded.  A mode that rendered as a no-op (activation
        key 0) leaves no record and does not appear.
    """
    artifacts = truth.get('artifacts')
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        mode: _realized_count(record)
        for mode, record in artifacts.items()
        if isinstance(record, Mapping)
    }


def marker_mask(image: NDArrayFloatType, marker_value: float) -> NDArrayBoolType:
    """Boolean mask of pixels carrying the missing-data marker.

    Parameters:
        image: The DN image.
        marker_value: The instrument's ``noise.marker_value`` (0 for every
            shipped instrument; a scene may override it to NaN, which is matched
            with :func:`numpy.isnan`).

    Returns:
        A boolean array, True where the pixel equals the marker.
    """
    mask: NDArrayBoolType = np.isnan(image) if math.isnan(marker_value) else (image == marker_value)
    return mask


def measured_missing_lines(image: NDArrayFloatType, *, marker_value: float = 0.0) -> int:
    """Count whole image lines lost to the marker (every pixel is the marker).

    A missing line zeros the full row, so a fully-marker row is a lost line.  A
    scene row that merely crosses the dark sky is not fully marker as long as the
    frame carries a non-zero floor (a bias pedestal or exposed scene), so the
    count is exact for the structural loss mode on such a frame.

    Parameters:
        image: The DN image.
        marker_value: The missing-data marker DN.

    Returns:
        The number of fully-marker rows.
    """
    mask = marker_mask(image, marker_value)
    return int(mask.all(axis=1).sum())


def measured_partial_lines(
    image: NDArrayFloatType, *, marker_value: float = 0.0, min_run: int = 2
) -> int:
    """Count truncated lines: rows with a long marker run but not a whole one.

    A partial line loses a contiguous segment (from a column to the row end, or a
    middle segment), leaving a survivor, so an affected row carries a contiguous
    marker run yet is not fully marker.  The ``min_run`` floor separates a
    truncated line from an incidental single dropped pixel or a lone dead column
    crossing the row; a fully-marker row is a whole missing line and is excluded.

    Parameters:
        image: The DN image.
        marker_value: The missing-data marker DN.
        min_run: The shortest contiguous marker run counted as a truncation.

    Returns:
        The number of rows whose longest contiguous marker run is at least
        ``min_run`` and that are not entirely marker.
    """
    mask = marker_mask(image, marker_value)
    count = 0
    for row in mask:
        if row.all() or not row.any():
            continue
        if _longest_run(row) >= min_run:
            count += 1
    return count


def measured_missing_blocks(
    image: NDArrayFloatType, *, block_lines: int, marker_value: float = 0.0
) -> int:
    """Count compression blocks lost to the marker on the block-row grid.

    Blocks align to the ``block_lines`` row grid; a lost block zeros every row of
    its block, so a grid-aligned band of ``block_lines`` fully-marker rows is one
    lost block.

    Parameters:
        image: The DN image.
        block_lines: The compression block height in rows.
        marker_value: The missing-data marker DN.

    Returns:
        The number of grid-aligned blocks that are entirely marker.
    """
    if block_lines <= 0:
        raise ValueError(f'block_lines must be positive; got {block_lines}')
    mask = marker_mask(image, marker_value)
    full_rows = mask.all(axis=1)
    size_v = full_rows.shape[0]
    count = 0
    for start in range(0, size_v, block_lines):
        if bool(full_rows[start : start + block_lines].all()):
            count += 1
    return count


def measured_pixel_dropouts(image: NDArrayFloatType, *, marker_value: float = 0.0) -> int:
    """Count isolated marker pixels (dead pixels), excluding lines and columns.

    A dead pixel is a singleton at the marker.  Marker pixels that belong to a
    fully-marker row (a missing line) or a fully-marker column (a dead column)
    are excluded, so the count is the isolated-dropout population on a frame
    whose scene floor is above the marker.

    Parameters:
        image: The DN image.
        marker_value: The missing-data marker DN.

    Returns:
        The number of marker pixels not lying in a fully-marker row or column.
    """
    mask = marker_mask(image, marker_value)
    if not mask.any():
        return 0
    full_rows = mask.all(axis=1)
    full_cols = mask.all(axis=0)
    isolated = mask & ~full_rows[:, None] & ~full_cols[None, :]
    return int(isolated.sum())


def _longest_run(row: np.ndarray) -> int:
    """The length of the longest contiguous True run in a boolean row."""
    best = 0
    current = 0
    for flag in row.tolist():
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
