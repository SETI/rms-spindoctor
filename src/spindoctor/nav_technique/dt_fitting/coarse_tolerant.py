"""Coarse acquisition scored by edge strength within a tolerance across the edge.

The edge-strength scores of :mod:`~spindoctor.nav_technique.dt_fitting.coarse`
credit a model vertex only when its rounded pixel is itself a detected edge
pixel.  The edge mask is one pixel thin, and a cataloged edge's image position
differs from its rounded model vertex by a pixel or two: sub-pixel phase, and
catalog residuals that differ from one edge to the next.  At the true shift a
model of several edges therefore collects only part of each edge's strength,
while a shift that lays a single model edge exactly on one strong image step
collects all of it.  The scores here credit each vertex with the strongest
counted edge within a few pixels across its edge, so every edge of the true
alignment contributes in full.
"""

import math

import numpy as np

from spindoctor.nav_technique.dt_fitting.coarse import (
    CoarseSearchResult,
    _edge_strength_terms,
    _validate_search_window,
)
from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['TolerantEdgeStrength']

_DIRECTION_BINS = 16
"""Number of normal-direction bins over 180 degrees.

Each vertex's normal is read as the center of its bin, at most 5.6 degrees
away, which changes ``|g . n|`` by at most a factor of ``cos(5.6 deg)``, or half
a percent.  The eight pixel-neighbor directions lie on bin centers.
"""

_PIXEL_STEPS = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.int64)
"""The four pixel-grid steps a tolerance window can run along."""


def _read_shifted(values: NDArrayFloatType, step_v: int, step_u: int) -> NDArrayFloatType:
    """Return ``out[v, u] = values[v + step_v, u + step_u]``, zero outside the array.

    Parameters:
        values: ``(H, W)`` array.
        step_v: Row step.
        step_u: Column step.

    Returns:
        The shifted copy.
    """
    height, width = values.shape
    out = np.zeros_like(values)
    v0, v1 = max(0, -step_v), min(height, height - step_v)
    u0, u1 = max(0, -step_u), min(width, width - step_u)
    if v0 < v1 and u0 < u1:
        out[v0:v1, u0:u1] = values[v0 + step_v : v1 + step_v, u0 + step_u : u1 + step_u]
    return out


class TolerantEdgeStrength:
    """Edge-strength scores that credit the strongest edge within a tolerance.

    A vertex is credited with the largest ``|g . n|`` over the counted edge
    pixels (detected edge pixels inside ``valid_mask``) within ``tolerance_px``
    pixel steps of it, along the pixel-grid step nearest its normal (a diagonal
    step spans 1.4 px).  The score of an integer shift is the sum of those
    credits divided by the total vertex count, so a vertex pushed off the array
    simply loses its credit.  With ``tolerance_px`` zero the score is the exact
    score of :func:`~spindoctor.nav_technique.dt_fitting.edge_strength_score`, up
    to the direction binning.

    The per-direction strength maps are built once, at construction, and every
    score and search reuses them.

    Parameters:
        edge_mask: ``(H, W)`` boolean image edge map.
        image_gradient_vu: ``(H, W, 2)`` per-pixel gradient vector aligned to
            ``edge_mask``.
        vertices_vu: ``(N, 2)`` model polyline vertex positions in ``(v, u)``.
        normals_vu: ``(N, 2)`` model normal at each vertex; either sign, need
            not be unit length.  A zero normal never contributes.
        tolerance_px: Number of pixel steps searched on each side of a vertex.
        valid_mask: Optional ``(H, W)`` boolean mask of pixels whose edges may
            contribute; ``None`` admits every pixel.

    Raises:
        TypeError: if ``tolerance_px`` is not an int.
        ValueError: if ``tolerance_px`` is negative, or under the conditions of
            :func:`~spindoctor.nav_technique.dt_fitting.edge_strength_score`.
    """

    def __init__(
        self,
        edge_mask: NDArrayBoolType,
        image_gradient_vu: NDArrayFloatType,
        vertices_vu: NDArrayFloatType,
        normals_vu: NDArrayFloatType,
        *,
        tolerance_px: int,
        valid_mask: NDArrayBoolType | None = None,
    ) -> None:
        """Validate the inputs and build one tolerant strength map per used direction.

        Parameters:
            edge_mask: See the class docstring.
            image_gradient_vu: See the class docstring.
            vertices_vu: See the class docstring.
            normals_vu: See the class docstring.
            tolerance_px: See the class docstring.
            valid_mask: See the class docstring.
        """
        if not isinstance(tolerance_px, int) or isinstance(tolerance_px, bool):
            raise TypeError(f'tolerance_px must be int; got {type(tolerance_px).__name__}')
        if tolerance_px < 0:
            raise ValueError(f'tolerance_px must be non-negative; got {tolerance_px!r}')
        terms = _edge_strength_terms(
            edge_mask, image_gradient_vu, vertices_vu, normals_vu, valid_mask=valid_mask
        )
        self._edge_mask = edge_mask
        self._valid_mask = valid_mask
        # The edge pixels a plain match fraction may count: a vertex on an excluded
        # pixel stays in the denominator but never matches.
        self._counted = edge_mask if valid_mask is None else edge_mask & valid_mask
        self._tolerance_px = tolerance_px
        self._base_v = terms.base_v
        self._base_u = terms.base_u
        # A normal and its negative score alike, so fold the direction into [0, pi).
        angle = np.mod(np.arctan2(terms.unit_n_u, terms.unit_n_v), math.pi)
        bins = np.rint(angle / (math.pi / _DIRECTION_BINS)).astype(np.int64) % _DIRECTION_BINS
        nonzero = np.hypot(terms.unit_n_v, terms.unit_n_u) > 0.0
        used = np.unique(bins[nonzero])
        steps_unit = _PIXEL_STEPS / np.hypot(_PIXEL_STEPS[:, 0], _PIXEL_STEPS[:, 1])[:, None]
        # One map per used direction plus an all-zero map a zero normal reads.
        self._maps = np.zeros((used.size + 1, *edge_mask.shape), dtype=np.float32)
        for index, b in enumerate(used):
            theta = float(b) * math.pi / _DIRECTION_BINS
            dir_v, dir_u = math.cos(theta), math.sin(theta)
            strength = np.abs(dir_v * terms.strength_v + dir_u * terms.strength_u)
            step = _PIXEL_STEPS[
                int(np.argmax(np.abs(steps_unit[:, 0] * dir_v + steps_unit[:, 1] * dir_u)))
            ]
            widened = strength.copy()
            for k in range(1, tolerance_px + 1):
                for sign in (1, -1):
                    np.maximum(
                        widened,
                        _read_shifted(strength, sign * k * int(step[0]), sign * k * int(step[1])),
                        out=widened,
                    )
            self._maps[index] = widened
        lookup = np.full(_DIRECTION_BINS, used.size, dtype=np.int64)
        lookup[used] = np.arange(used.size)
        self._map_index = np.where(nonzero, lookup[bins], used.size)

    def score(self, offset_vu: tuple[int, int]) -> float:
        """Return the tolerant edge-strength score of one integer shift.

        Parameters:
            offset_vu: ``(dv, du)`` integer shift to score.

        Returns:
            The score, ``0.0`` for an empty polyline.
        """
        n_vertices = int(self._base_v.size)
        if n_vertices == 0:
            return 0.0
        height, width = self._edge_mask.shape
        shifted_v = self._base_v + int(offset_vu[0])
        shifted_u = self._base_u + int(offset_vu[1])
        valid = (shifted_v >= 0) & (shifted_v < height) & (shifted_u >= 0) & (shifted_u < width)
        picked = self._maps[self._map_index[valid], shifted_v[valid], shifted_u[valid]]
        return float(picked.sum(dtype=np.float64)) / float(n_vertices)

    def best_match_fraction(self, offset_vu: tuple[int, int]) -> float:
        """Return the best plain match fraction within the tolerance of a shift.

        The tolerant maximum is by construction a shift where vertices may sit a
        pixel or two off their edge pixels, so its own plain match fraction (the
        fraction of in-bounds vertices exactly on a detected edge pixel that
        ``valid_mask`` admits) can be
        near zero for a right alignment.  This is the largest plain fraction over
        the integer shifts within ``tolerance_px`` of ``offset_vu`` on each axis:
        what the best exact placement in the same basin achieves.

        Parameters:
            offset_vu: ``(dv, du)`` integer shift.

        Returns:
            The fraction, ``0.0`` for an empty polyline.
        """
        height, width = self._edge_mask.shape
        best = 0.0
        for dv in range(-self._tolerance_px, self._tolerance_px + 1):
            shifted_v = self._base_v + int(offset_vu[0]) + dv
            for du in range(-self._tolerance_px, self._tolerance_px + 1):
                shifted_u = self._base_u + int(offset_vu[1]) + du
                inside = (
                    (shifted_v >= 0) & (shifted_v < height) & (shifted_u >= 0) & (shifted_u < width)
                )
                if not inside.any():
                    continue
                best = max(best, float(self._counted[shifted_v[inside], shifted_u[inside]].mean()))
        return best

    def compare(
        self, offset_a: tuple[int, int], offset_b: tuple[int, int]
    ) -> tuple[float, float, int]:
        """Return the summed credits of two shifts over the vertices both keep in view.

        A vertex counts only when its shifted pixel lies inside the array and on
        ``valid_mask`` at both shifts, so a shift that drags more of the model onto
        the sensor gains nothing from the vertices only it puts there: the two
        alignments are compared on the same model vertices, with evidence under
        both.

        Parameters:
            offset_a: ``(dv, du)`` integer shift.
            offset_b: ``(dv, du)`` integer shift.

        Returns:
            The summed credit of ``offset_a`` and of ``offset_b`` over those
            vertices, and the number of those vertices.
        """
        height, width = self._edge_mask.shape
        picked = []
        seen = np.ones(self._base_v.size, dtype=bool)
        for offset in (offset_a, offset_b):
            shifted_v = self._base_v + int(offset[0])
            shifted_u = self._base_u + int(offset[1])
            inside = (
                (shifted_v >= 0) & (shifted_v < height) & (shifted_u >= 0) & (shifted_u < width)
            )
            sv = np.where(inside, shifted_v, 0)
            su = np.where(inside, shifted_u, 0)
            if self._valid_mask is not None:
                inside &= self._valid_mask[sv, su]
            seen &= inside
            picked.append(self._maps[self._map_index, sv, su])
        return (
            float(picked[0][seen].sum(dtype=np.float64)),
            float(picked[1][seen].sum(dtype=np.float64)),
            int(np.count_nonzero(seen)),
        )

    def search(
        self,
        search_window_vu: tuple[int, int],
        *,
        min_support_fraction: float = DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
    ) -> CoarseSearchResult:
        """Return the integer shift with the largest tolerant score in the window.

        Eligibility and tie-breaking follow
        :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search`: a shift
        that keeps fewer than ``min_support_fraction`` of the vertices inside the
        array is skipped, and ties go to the smaller ``(|dv| + |du|, |dv|, dv,
        du)``.  Support counts every vertex inside the array, as the exact
        searches do, and a vertex off ``valid_mask`` still earns credit from a
        counted edge within the tolerance; only :meth:`compare` restricts the
        vertices to ``valid_mask``.  The reported
        :attr:`CoarseSearchResult.score` is the winning shift's plain
        per-vertex match fraction, the fraction of its in-bounds vertices on a
        detected edge pixel that ``valid_mask`` admits, as
        :func:`~spindoctor.nav_technique.dt_fitting.coarse_edge_strength_search_scored`
        reports it.

        Parameters:
            search_window_vu: ``(margin_v, margin_u)`` non-negative integers
                bounding the search range in v and u.
            min_support_fraction: Minimum fraction of the vertices that must stay
                inside the array for a shift to be eligible.

        Returns:
            :class:`CoarseSearchResult` with the peak integer offset and its plain
            match fraction.

        Raises:
            TypeError: if either entry of ``search_window_vu`` is not an int.
            ValueError: if ``search_window_vu`` is not a length-2 sequence of
                non-negative ints, or ``min_support_fraction`` is outside
                ``[0, 1]``.
        """
        margin_v, margin_u = _validate_search_window(search_window_vu)
        if not 0.0 <= min_support_fraction <= 1.0:
            raise ValueError(
                f'min_support_fraction must be in [0, 1]; got {min_support_fraction!r}'
            )
        height, width = self._edge_mask.shape
        n_vertices = int(self._base_v.size)
        if n_vertices == 0:
            return CoarseSearchResult(offset_vu=(0, 0), score=0.0)
        base_v, base_u, map_index, maps = self._base_v, self._base_u, self._map_index, self._maps
        min_support = min_support_fraction * float(n_vertices)
        best_dv = 0
        best_du = 0
        best_score = -1.0
        best_key = (math.inf, math.inf, math.inf, math.inf)
        for dv in range(-margin_v, margin_v + 1):
            shifted_v = base_v + dv
            valid_v = (shifted_v >= 0) & (shifted_v < height)
            if not valid_v.any():
                continue
            for du in range(-margin_u, margin_u + 1):
                shifted_u = base_u + du
                valid = valid_v & (shifted_u >= 0) & (shifted_u < width)
                if float(np.count_nonzero(valid)) < max(min_support, 1.0):
                    continue
                picked = maps[map_index[valid], shifted_v[valid], shifted_u[valid]]
                score = float(picked.sum(dtype=np.float64)) / float(n_vertices)
                key = (abs(dv) + abs(du), abs(dv), dv, du)
                if score > best_score or (score == best_score and key < best_key):
                    best_score = score
                    best_key = key
                    best_dv = dv
                    best_du = du
        if best_score < 0.0:
            return CoarseSearchResult(offset_vu=(0, 0), score=0.0)
        win_v = base_v + best_dv
        win_u = base_u + best_du
        in_bounds = (win_v >= 0) & (win_v < height) & (win_u >= 0) & (win_u < width)
        match_fraction = float(self._counted[win_v[in_bounds], win_u[in_bounds]].mean())
        return CoarseSearchResult(offset_vu=(best_dv, best_du), score=match_fraction)
