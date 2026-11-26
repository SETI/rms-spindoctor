"""Immutable rendering context for ring feature backplane rendering.

This module defines ``RingsRenderContext``, a frozen dataclass that bundles
all dependencies needed to render one ring feature. Passing a single context
object instead of many individual parameters achieves two goals:

1. **Clean method signatures**: ``RingFeature.render(context)`` takes one
   argument instead of six. Adding a new rendering parameter only requires
   updating ``RingsRenderContext``, not every call site.

2. **Immutability contract**: Because the context is frozen, each call to
   ``render()`` receives the same data. Features cannot accidentally modify
   shared rendering state.

``RingsRenderContext`` carries ``all_edge_radii`` -- the sorted sequence of
(radius, label) pairs for all features that survived filtering. This is needed
at render time by ``compute_edge_fade`` to reduce the fade width when a
neighboring edge falls within the fade zone (halving the fade at the conflict
boundary). The filter has already handled *exclusion* (edges whose adjusted
fade would be too narrow); ``compute_edge_fade`` handles *reduction* (edges
whose adjusted fade is still acceptable but narrower than the requested
``fade_width_pix``).
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType


@dataclass(frozen=True)
class RingsRenderContext:
    """Immutable context for backplane-based ring feature rendering.

    Constructed by the orchestrator (``NavModelRings``) once per observation
    and passed unchanged to every ``RingFeature.render()`` call. Contains
    observation data, computed backplane arrays, fade configuration, and the
    sorted list of all surviving edge radii for conflict-based fade reduction.

    The ``all_edge_radii`` tuple is built from features that survived all four
    filter passes. It is used by ``compute_edge_fade`` to reduce fade width
    when a neighboring feature's edge is within the fade zone, preserving the
    current behavior of halving the fade extent at a conflict boundary rather
    than rendering with full width. This is a *width reduction*, not exclusion
    -- exclusion is handled by ``RingFeatureFilter`` before rendering.

    Parameters:
        obs: The observation object (``oops.Observation``). Provides access to
            all backplane computation methods.
        ring_target: Ring target string used for backplane calls, e.g.
            ``'saturn:ring'``.
        epoch: TDB epoch time in seconds used as the reference time for
            multi-mode orbital perturbation calculations.
        resolutions: 2-D array of per-pixel radial resolution in km/pixel.
            Shape matches the extended FOV. Used to compute per-pixel fade
            widths: ``fade_width_km = fade_width_pix * resolutions``.
        fade_width_pix: Fade extent in pixels, the ring catalog's positive
            ``fade_width_pix``; per-pixel km extent is computed at render time from this
            value and ``resolutions``.
        all_edge_radii: Sorted tuple of ``(radius_km, edge_label)`` pairs for
            all edges of all features that survived filtering. Used by
            ``compute_edge_fade`` for conflict detection and width reduction.
        logger: ``PdsLogger`` from the ring ``NavModel`` (same instance as
            ``NavModelRings._logger``).
    """

    obs: Any  # oops.Observation; typed as Any to avoid oops import at module level
    ring_target: str
    epoch: float
    resolutions: NDArrayFloatType
    fade_width_pix: float
    all_edge_radii: tuple[tuple[float, str], ...]
    logger: Any

    def __post_init__(self) -> None:
        """Validate fields at construction (frozen dataclass: no mutation)."""
        if self.obs is None:
            raise ValueError('RingsRenderContext.obs must not be None')
        if not isinstance(self.ring_target, str) or self.ring_target.strip() == '':
            raise ValueError('RingsRenderContext.ring_target must be a non-empty string')
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, (int, float)):
            raise TypeError('RingsRenderContext.epoch must be int or float')

        if not isinstance(self.resolutions, np.ndarray):
            raise TypeError('RingsRenderContext.resolutions must be a numpy ndarray')
        if self.resolutions.ndim != 2:
            raise ValueError('RingsRenderContext.resolutions must be a 2-D array')
        if not np.issubdtype(self.resolutions.dtype, np.number):
            raise TypeError('RingsRenderContext.resolutions must have a numeric dtype')
        if self.resolutions.size == 0:
            raise ValueError('RingsRenderContext.resolutions must not be empty')
        res64 = np.asarray(self.resolutions, dtype=np.float64)
        if not np.all(np.isfinite(res64)):
            raise ValueError('RingsRenderContext.resolutions must contain only finite values')
        if np.any(res64 <= 0.0):
            raise ValueError('RingsRenderContext.resolutions must be positive everywhere')

        if self.logger is None:
            raise ValueError('RingsRenderContext.logger must not be None')

        if not isinstance(self.all_edge_radii, tuple):
            raise TypeError('RingsRenderContext.all_edge_radii must be a tuple')

        prev_rad: float | None = None
        for j, pair in enumerate(self.all_edge_radii):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f'RingsRenderContext.all_edge_radii[{j}] must be (radius_km, label) pair'
                )
            rad, label = pair
            if isinstance(rad, bool) or not isinstance(rad, (int, float)):
                raise TypeError(f'RingsRenderContext.all_edge_radii[{j}][0] must be numeric')
            if not math.isfinite(float(rad)) or float(rad) <= 0.0:
                raise ValueError(
                    f'RingsRenderContext.all_edge_radii[{j}][0] must be finite and positive, '
                    f'got {rad!r}'
                )
            if not isinstance(label, str) or label.strip() == '':
                raise ValueError(
                    f'RingsRenderContext.all_edge_radii[{j}][1] must be a non-empty string'
                )
            rad_f = float(rad)
            if prev_rad is not None and rad_f < prev_rad:
                raise ValueError(
                    'RingsRenderContext.all_edge_radii radii must be sorted in non-decreasing '
                    f'order; index {j} has radius {rad_f} km < previous {prev_rad} km'
                )
            prev_rad = rad_f
