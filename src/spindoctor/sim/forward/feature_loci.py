"""Navigation-feature loci for adversarial artifact placement.

Adversarial placement (the worst-case artifact stress mode) seeds stochastic
artifacts preferentially on the navigation features rather than uniformly.  The
renderer already knows where those features are: the radiance stage records the
body and ring masks and the rendered star positions in ``frame.truth``, all on
the detector grid by the time the telemetry and detector stages run.  This
module extracts the loci from that truth -- the rows a body or ring edge crosses
and the pixels on a limb / ring arc or at a star -- and offers the biased
sampling helpers each adversarial mode's placement hook calls.  With adversarial
off the samplers fall back to uniform draws, so the same code path serves both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayIntType

__all__ = [
    'FeatureLoci',
    'choose_pixels',
    'choose_rows',
    'dilated_pixels',
    'extract_feature_loci',
]


@dataclass(frozen=True)
class FeatureLoci:
    """The navigation-feature loci extracted from a rendered frame's truth.

    Parameters:
        rows: Sorted unique detector rows a body / ring feature or a star
            crosses (the candidate lines for adversarial line-loss placement).
        pixel_v: Row coordinates of feature pixels (limb / ring-edge arcs and
            star centers) for adversarial per-pixel placement.
        pixel_u: Column coordinates of those feature pixels.
    """

    rows: NDArrayIntType
    pixel_v: NDArrayIntType
    pixel_u: NDArrayIntType

    @property
    def has_rows(self) -> bool:
        """Whether any feature rows were found."""
        return bool(self.rows.size)

    @property
    def has_pixels(self) -> bool:
        """Whether any feature pixels were found."""
        return bool(self.pixel_v.size)


def _boundary_pixels(mask: Any) -> tuple[NDArrayIntType, NDArrayIntType]:
    """Return the (v, u) coordinates of a mask's one-pixel boundary."""
    boolean = np.asarray(mask, dtype=bool)
    if not boolean.any():
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    edge = boolean & ~ndimage.binary_erosion(boolean)
    vs, us = np.nonzero(edge)
    return vs.astype(np.int64), us.astype(np.int64)


def extract_feature_loci(truth: Mapping[str, Any], shape: tuple[int, int]) -> FeatureLoci:
    """Extract the navigation-feature loci from a frame's truth metadata.

    Bodies and rings contribute the rows their disc / annulus spans and the
    pixels on their limb / edge arc; stars contribute their center row and
    pixel.  A frame with no features (an empty truth dict) yields empty loci,
    and the samplers then fall back to uniform placement.

    Parameters:
        truth: The frame's ``truth`` mapping (``body_masks``, ``ring_masks``,
            ``star_info``), as recorded by the radiance stage on the detector
            grid.
        shape: The detector-grid ``(size_v, size_u)`` shape, for bounds checks.

    Returns:
        The extracted :class:`FeatureLoci`.
    """
    size_v, size_u = shape
    rows: set[int] = set()
    pixel_v: list[int] = []
    pixel_u: list[int] = []

    for key in ('body_masks', 'ring_masks'):
        for mask in truth.get(key) or []:
            boolean = np.asarray(mask, dtype=bool)
            if boolean.shape != (size_v, size_u) or not boolean.any():
                continue
            rows.update(int(r) for r in np.nonzero(boolean.any(axis=1))[0])
            vs, us = _boundary_pixels(boolean)
            pixel_v.extend(int(v) for v in vs)
            pixel_u.extend(int(u) for u in us)

    for info in truth.get('star_info') or []:
        cv = info.get('center_v')
        cu = info.get('center_u')
        if cv is None or cu is None:
            continue
        v = round(float(cv))
        u = round(float(cu))
        if 0 <= v < size_v and 0 <= u < size_u:
            rows.add(v)
            pixel_v.append(v)
            pixel_u.append(u)

    return FeatureLoci(
        rows=np.array(sorted(rows), dtype=np.int64),
        pixel_v=np.array(pixel_v, dtype=np.int64),
        pixel_u=np.array(pixel_u, dtype=np.int64),
    )


def choose_rows(
    loci: FeatureLoci,
    n: int,
    size_v: int,
    rng: np.random.Generator,
    *,
    adversarial: bool,
) -> NDArrayIntType:
    """Choose ``n`` rows: biased onto feature rows when adversarial, else uniform.

    Parameters:
        loci: The feature loci (feature rows are the adversarial pool).
        n: The number of rows to choose.
        size_v: The detector-grid height.
        rng: The mode's seeded generator.
        adversarial: Whether to bias placement onto the feature rows.

    Returns:
        The chosen row indices (distinct where the pool allows).
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if adversarial and loci.has_rows:
        pool = loci.rows
        replace = n > pool.size
        return rng.choice(pool, size=n, replace=replace).astype(np.int64)
    replace = n > size_v
    return rng.choice(size_v, size=n, replace=replace).astype(np.int64)


def choose_pixels(
    loci: FeatureLoci,
    n: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    adversarial: bool,
    radius: int = 3,
) -> tuple[NDArrayIntType, NDArrayIntType]:
    """Choose ``n`` pixels: near features when adversarial, else uniform.

    An adversarial draw picks a feature pixel and jitters it within ``radius``
    pixels, so the events land on or beside a limb arc or a star.

    Parameters:
        loci: The feature loci (feature pixels are the adversarial pool).
        n: The number of pixels to choose.
        shape: The detector-grid ``(size_v, size_u)`` shape.
        rng: The mode's seeded generator.
        adversarial: Whether to bias placement onto the feature pixels.
        radius: The jitter radius, in pixels, around a chosen feature pixel.

    Returns:
        The ``(v, u)`` coordinate arrays of the chosen pixels.
    """
    size_v, size_u = shape
    if n <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if adversarial and loci.has_pixels:
        pick = rng.integers(0, loci.pixel_v.size, size=n)
        jitter_v = rng.integers(-radius, radius + 1, size=n)
        jitter_u = rng.integers(-radius, radius + 1, size=n)
        vs = np.clip(loci.pixel_v[pick] + jitter_v, 0, size_v - 1)
        us = np.clip(loci.pixel_u[pick] + jitter_u, 0, size_u - 1)
        return vs.astype(np.int64), us.astype(np.int64)
    vs = rng.integers(0, size_v, size=n)
    us = rng.integers(0, size_u, size=n)
    return vs.astype(np.int64), us.astype(np.int64)


def dilated_pixels(
    loci: FeatureLoci,
    *,
    radius: int,
    shape: tuple[int, int],
) -> tuple[NDArrayIntType, NDArrayIntType]:
    """Return every pixel within ``radius`` of a feature pixel (a placement pool).

    The detector hot-pixel routing draws its adversarial population from this
    pool so hot pixels concentrate on and beside the features.  Empty loci yield
    an empty pool, and the caller then falls back to uniform placement.

    Parameters:
        loci: The feature loci.
        radius: The dilation radius in pixels.
        shape: The detector-grid ``(size_v, size_u)`` shape.

    Returns:
        The ``(v, u)`` coordinate arrays of the dilated feature region.
    """
    size_v, size_u = shape
    if not loci.has_pixels:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    seed = np.zeros((size_v, size_u), dtype=bool)
    seed[loci.pixel_v, loci.pixel_u] = True
    structure = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(seed, structure=structure, iterations=max(1, radius))
    vs, us = np.nonzero(dilated)
    return vs.astype(np.int64), us.astype(np.int64)
