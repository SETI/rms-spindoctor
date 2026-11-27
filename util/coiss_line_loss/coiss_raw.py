"""Read a Cassini ISS raw EDR image and its binary line prefixes from the PDS3 holdings.

Each line of a COISS_2xxx raw image carries a 24-byte binary prefix ahead of its
samples.  Its 16-bit words give, among other things, the first and last valid sample of
up to two valid segments of the line (one-based, zero where a segment is absent).  A
line whose first segment runs from sample 1 to the line length is whole; one whose
first segment ends early and that has no second segment kept its start and lost its
end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from filecache import FCPath

__all__ = ['RawFrame', 'read_raw_frame']

# The label keywords this module reads.
_KEYWORDS = (
    'RECORD_BYTES',
    'LINES',
    'LINE_SAMPLES',
    'SAMPLE_BITS',
    'LINE_PREFIX_BYTES',
    'INSTRUMENT_MODE_ID',
)


@dataclass(frozen=True)
class RawFrame:
    """One raw frame: its samples and, per line, the valid segments its prefix gives.

    Parameters:
        name: The image file's name.
        mode: The label's INSTRUMENT_MODE_ID (FULL, SUM2, or SUM4).
        image: The samples, one row per line.
        first_start: The first valid sample of each line's first segment (one-based).
        first_end: The last valid sample of each line's first segment.
        second_start: The first valid sample of each line's second segment, 0 if none.
    """

    name: str
    mode: str
    image: np.ndarray
    first_start: np.ndarray
    first_end: np.ndarray
    second_start: np.ndarray

    @property
    def whole(self) -> np.ndarray:
        """Whether each line kept every sample."""
        samples = self.image.shape[1]
        return (self.first_start == 1) & (self.first_end == samples) & (self.second_start == 0)

    @property
    def end_lost(self) -> np.ndarray:
        """Whether each line kept its start and lost its end."""
        samples = self.image.shape[1]
        return (
            (self.first_start == 1)
            & (self.first_end > 0)
            & (self.first_end < samples)
            & (self.second_start == 0)
        )


def _label_values(text: str) -> dict[str, str]:
    """Return the label keywords this module reads, and the image's record."""
    values: dict[str, str] = {}
    for key in _KEYWORDS:
        match = re.search(rf'^\s*{key}\s*=\s*(.+)$', text, re.MULTILINE)
        if match:
            values[key] = match.group(1).strip().strip('"')
    match = re.search(r'\^IMAGE\s*=\s*\("[^"]+",\s*(\d+)\)', text)
    values['IMAGE_RECORD'] = match.group(1) if match else '1'
    return values


def read_raw_frame(path: FCPath) -> RawFrame:
    """Read one raw EDR image file and the label beside it.

    Parameters:
        path: The ``.IMG`` file, local or remote; its ``.LBL`` lies beside it.

    Returns:
        The frame.
    """
    label = _label_values(path.with_suffix('.LBL').read_text())
    record_bytes = int(label['RECORD_BYTES'])
    lines, samples = int(label['LINES']), int(label['LINE_SAMPLES'])
    prefix_bytes = int(label.get('LINE_PREFIX_BYTES', '0'))
    sample_bytes = int(label['SAMPLE_BITS']) // 8
    start = (int(label['IMAGE_RECORD']) - 1) * record_bytes
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    records = data[start : start + lines * record_bytes].reshape(lines, record_bytes)
    body = records[:, prefix_bytes : prefix_bytes + samples * sample_bytes].copy()
    dtype = '>i2' if sample_bytes == 2 else 'u1'
    image = body.view(dtype).reshape(lines, samples).astype(np.int64)
    words = records[:, :24].copy().view('>u2').astype(np.int64)
    return RawFrame(
        name=path.name,
        mode=label.get('INSTRUMENT_MODE_ID', ''),
        image=image,
        first_start=words[:, 2],
        first_end=words[:, 3],
        second_start=words[:, 4],
    )
