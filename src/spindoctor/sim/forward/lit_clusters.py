"""Group the lit pixels of a render-grid plane into clusters.

Until the PSF blurs it, a plane is usually zero away from the bodies, rings, and
stars.  The distortion warp and the PSF blur are both linear, so each can process
every cluster of lit pixels on its own crop and add the results.  The sum is the
same as processing the whole plane.

Lit pixels are grouped in square blocks.  Lit blocks that touch, diagonally
included, form one cluster.  A crop around one cluster can take in pixels of
another, so ``cluster_crop`` keeps only the cluster's own blocks.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['CLUSTER_BLOCK_PX', 'LitClusters', 'cluster_crop', 'find_lit_clusters', 'grow']

# Side of the square blocks, in render-grid pixels, in which lit pixels are grouped.
CLUSTER_BLOCK_PX = 32


@dataclass(frozen=True)
class LitClusters:
    """The clusters of lit pixels in a plane.

    Parameters:
        labels: The cluster label of every block: 0 for an unlit block, and
            ``n + 1`` for cluster ``n``.
        boxes: Each cluster's bounding box in pixels, block-aligned at its start
            and clipped to the plane.
    """

    labels: NDArrayIntType
    boxes: list[tuple[slice, slice]]


def grow(box: slice, by: int, size: int) -> slice:
    """A slice widened by ``by`` on each side and clipped to ``[0, size)``."""
    return slice(max(box.start - by, 0), min(box.stop + by, size))


def find_lit_clusters(plane: NDArrayFloatType) -> LitClusters:
    """Group a plane's nonzero pixels into clusters of touching lit blocks.

    Parameters:
        plane: The render-grid plane.

    Returns:
        The clusters; no boxes when every pixel is zero.
    """
    size_v, size_u = plane.shape
    block = CLUSTER_BLOCK_PX
    blocks_v = -(-size_v // block)
    blocks_u = -(-size_u // block)
    lit = np.zeros((blocks_v * block, blocks_u * block), dtype=bool)
    lit[:size_v, :size_u] = plane != 0.0
    lit_blocks = lit.reshape(blocks_v, block, blocks_u, block).any(axis=(1, 3))
    labels, _ = ndimage.label(lit_blocks, structure=np.ones((3, 3), dtype=bool))
    boxes = [
        (
            slice(rows.start * block, min(rows.stop * block, size_v)),
            slice(cols.start * block, min(cols.stop * block, size_u)),
        )
        for rows, cols in ndimage.find_objects(labels)
    ]
    return LitClusters(labels=labels, boxes=boxes)


def cluster_crop(
    clusters: LitClusters, plane: NDArrayFloatType, index: int, rows: slice, cols: slice
) -> NDArrayFloatType:
    """A crop of the plane holding only one cluster's pixels.

    Parameters:
        clusters: The plane's clusters.
        plane: The plane.
        index: The cluster's index in ``clusters.boxes``.
        rows: The crop's rows; its start must be a whole number of blocks.
        cols: The crop's columns; its start must be a whole number of blocks.

    Returns:
        ``plane[rows, cols]`` with every pixel of another cluster set to zero.
    """
    block = CLUSTER_BLOCK_PX
    block_labels = clusters.labels[
        rows.start // block : -(-rows.stop // block),
        cols.start // block : -(-cols.stop // block),
    ]
    own = np.repeat(np.repeat(block_labels == index + 1, block, axis=0), block, axis=1)
    own = own[: rows.stop - rows.start, : cols.stop - cols.start]
    crop: NDArrayFloatType = np.where(own, plane[rows, cols], 0.0)
    return crop
