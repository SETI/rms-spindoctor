"""The browse image a navigation run leaves beside each document it writes.

A real PNG, because the browse label states the file's size in bytes and its
checksum, and a blob of bytes with a PNG signature on the front has neither.
It is encoded by ``PIL``, which is the encoder the pipeline's own summary
writer ends on; everything that writer does before that -- the stretch, the
annotation overlay and the metadata block -- reads a loaded image and a
navigation result's annotations, neither of which a cohort built without an
image has.  So what is shared with the product is its encoding, and what is
synthesized is its content.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from filecache import FCPath
from PIL import Image

from .backplanes import COHORT_SHAPE_VU

_SCALE = 4
"""How many browse pixels one frame pixel becomes.

The summary image a run writes is larger than the frame it annotates, since it
carries a metadata block beside the stretched image.  A whole multiple of the
frame keeps the browse image the frame's own shape at a size worth looking at.
"""


def write_summary_png(png_file_path: FCPath) -> None:
    """Write one image's summary PNG.

    The content is a deterministic ramp, so a run of the cohort produces the
    same bytes as the last, and a label that states this file's size and
    checksum states the same ones.

    Parameters:
        png_file_path: Where the PNG goes.
    """
    size_v, size_u = COHORT_SHAPE_VU
    rows = np.linspace(0, 255, size_v * _SCALE, dtype=np.uint8)[:, np.newaxis]
    columns = np.linspace(0, 255, size_u * _SCALE, dtype=np.uint8)[np.newaxis, :]
    red = np.broadcast_to(rows, (size_v * _SCALE, size_u * _SCALE))
    green = np.broadcast_to(columns, (size_v * _SCALE, size_u * _SCALE))
    blue = np.full((size_v * _SCALE, size_u * _SCALE), 64, dtype=np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    png_file_path.write_bytes(buffer.getvalue())
