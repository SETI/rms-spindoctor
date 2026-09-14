"""Coarse acquisition: polyline rasterization, integer search, polarity.

The first stage of the shared DT pipeline -- render the model polyline to a
binary mask, find the best integer shift against the image edge mask, and
classify each vertex by gradient polarity.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'CoarseSearchResult',
    'build_polyline_mask',
    'coarse_ncc_search',
    'coarse_ncc_search_scored',
    'coarse_polarity_search_scored',
    'polarity_filter',
]


def _validate_search_window(search_window_vu: tuple[int, int]) -> tuple[int, int]:
    """Validate a ``(margin_v, margin_u)`` search window and return it as ints.

    Shared by the coarse search entry points so the mask-overlap and the
    polarity-weighted variants reject malformed windows with identical
    messages.  Booleans are rejected because ``bool`` is an ``int`` subclass
    and a ``True`` margin is far more likely a caller mistake than an
    intended one-pixel window.

    Parameters:
        search_window_vu: ``(margin_v, margin_u)`` non-negative integers
            bounding the search range in v and u.

    Returns:
        ``(margin_v, margin_u)`` as plain ints.

    Raises:
        TypeError: if either entry is not an int.
        ValueError: if the argument is not a length-2 sequence or an entry
            is negative.
    """
    if not isinstance(search_window_vu, tuple | list) or len(search_window_vu) != 2:
        raise ValueError(
            f'search_window_vu must be a length-2 sequence of ints; got {search_window_vu!r}'
        )
    margin_v_raw, margin_u_raw = search_window_vu[0], search_window_vu[1]
    if not isinstance(margin_v_raw, int) or isinstance(margin_v_raw, bool):
        raise TypeError(f'search_window_vu[0] must be int; got {type(margin_v_raw).__name__}')
    if not isinstance(margin_u_raw, int) or isinstance(margin_u_raw, bool):
        raise TypeError(f'search_window_vu[1] must be int; got {type(margin_u_raw).__name__}')
    if margin_v_raw < 0 or margin_u_raw < 0:
        raise ValueError(f'search_window_vu must be non-negative; got {search_window_vu!r}')
    return margin_v_raw, margin_u_raw


@dataclass(frozen=True)
class CoarseSearchResult:
    """Structured output of :func:`coarse_ncc_search_scored`.

    Parameters:
        offset_vu: ``(dv, du)`` integer offset pair at the peak.
        score: The winning shift's match fraction -- the fraction of its
            in-bounds RASTERIZED POLYLINE PIXELS landing on edge pixels, in
            ``[0, 1]``.  Note the denominator counts mask pixels, not model
            vertices: :func:`build_polyline_mask` collapses vertices that
            round to the same pixel, so this is not directly comparable to
            a per-vertex count such as a technique's Tukey inlier count.
            ``0.0`` when the polyline is empty or no candidate shift was
            eligible.  A low score means the coarse acquisition never had a
            lock: even at its best shift, most of the model found no
            detected edge underneath it.
    """

    offset_vu: tuple[int, int]
    score: float


def coarse_ncc_search(
    edge_mask: NDArrayBoolType,
    polyline_mask: NDArrayBoolType,
    search_window_vu: tuple[int, int],
    *,
    min_support_fraction: float = DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
) -> tuple[int, int]:
    """Return the integer offset that maximizes the normalized overlap of two masks.

    For each integer shift ``(dv, du)`` in
    ``[-margin_v, +margin_v] x [-margin_u, +margin_u]`` the score is the
    fraction of in-bounds polyline vertices that land on an edge pixel,
    ``f(dv, du) = (sum_{v, u} polyline_mask[v, u] * edge_mask[v + dv, u + du])
    / N_in_bounds(dv, du)``, where ``N_in_bounds`` is the number of polyline
    vertices that remain inside the image after the shift.  Dividing the raw
    overlap count by the in-bounds vertex count removes the bias of a raw
    overlap count toward shifts that simply keep more vertices in bounds.
    Note this per-vertex match fraction is NOT the binary normalized
    cross-correlation (a binary NCC normalizes by ``sqrt(N_in_bounds)``, not
    ``N_in_bounds``, so its argmax can differ); the plain fraction is used
    because it fully cancels the in-bounds-count advantage, at the cost of
    over-rewarding shifts with very few surviving vertices.  That failure
    mode is closed by the minimum-support guard: a shift whose in-bounds
    vertex fraction falls below ``min_support_fraction`` is ineligible, so a
    perfect score on a handful of edge-pixel stragglers cannot beat a dense
    well-supported match (see :data:`DEFAULT_COARSE_MIN_SUPPORT_FRACTION`).

    Ties are broken by the smaller ``(|dv| + |du|, |dv|, dv, du)`` tuple,
    so the nearest-to-origin shift (by Manhattan distance) wins on
    perfectly flat inputs.

    Parameters:
        edge_mask: ``(H, W)`` boolean image edge map.
        polyline_mask: ``(H, W)`` boolean model polyline mask aligned to
            ``edge_mask``.  Must have the same shape as ``edge_mask``.
        search_window_vu: ``(margin_v, margin_u)`` non-negative integers
            bounding the search range in v and u.
        min_support_fraction: Minimum fraction of the polyline's vertices
            that must remain in bounds after a candidate shift for that
            shift to be eligible.  The zero shift keeps every mask vertex
            in bounds, so with any value at most 1.0 at least one candidate
            is always eligible when the window includes the origin.

    Returns:
        ``(dv, du)`` integer offset pair at the peak.

    Raises:
        TypeError: if either entry of ``search_window_vu`` is not an int.
        ValueError: if shapes disagree, masks are not 2-D,
            ``search_window_vu`` is not a length-2 sequence of
            non-negative ints, or ``min_support_fraction`` is outside
            ``[0, 1]``.
    """
    return coarse_ncc_search_scored(
        edge_mask,
        polyline_mask,
        search_window_vu,
        min_support_fraction=min_support_fraction,
    ).offset_vu


def coarse_ncc_search_scored(
    edge_mask: NDArrayBoolType,
    polyline_mask: NDArrayBoolType,
    search_window_vu: tuple[int, int],
    *,
    min_support_fraction: float = DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
) -> CoarseSearchResult:
    """Run :func:`coarse_ncc_search` and also report the winning peak's score.

    Identical search to :func:`coarse_ncc_search` (same arguments, same
    validation, same tie-breaking); additionally returns the winning shift's
    per-vertex match fraction so callers can gate on coarse-acquisition
    quality.  The DT techniques call this form; the offset-only form is the
    convenience wrapper.

    Parameters:
        edge_mask: See :func:`coarse_ncc_search`.
        polyline_mask: See :func:`coarse_ncc_search`.
        search_window_vu: See :func:`coarse_ncc_search`.
        min_support_fraction: See :func:`coarse_ncc_search`.

    Returns:
        :class:`CoarseSearchResult` with the peak offset and its match
        fraction.

    Raises:
        TypeError: if either entry of ``search_window_vu`` is not an int.
        ValueError: same conditions as :func:`coarse_ncc_search`.
    """
    if edge_mask.ndim != 2 or polyline_mask.ndim != 2:
        raise ValueError(
            'edge_mask and polyline_mask must be 2-D; got '
            f'ndims {edge_mask.ndim}, {polyline_mask.ndim}'
        )
    if edge_mask.shape != polyline_mask.shape:
        raise ValueError(
            f'shape mismatch: edge_mask {edge_mask.shape} vs polyline_mask {polyline_mask.shape}'
        )
    # Validate explicitly rather than relying on int() coercion, which would
    # silently truncate floats and raise an unhelpful IndexError on
    # wrong-length sequences.
    margin_v, margin_u = _validate_search_window(search_window_vu)
    if not 0.0 <= min_support_fraction <= 1.0:
        raise ValueError(f'min_support_fraction must be in [0, 1]; got {min_support_fraction!r}')
    height, width = edge_mask.shape
    edge_f = edge_mask.astype(np.float64, copy=False)
    # Scan the bounded window directly: brute-force over O(margin_v * margin_u)
    # offsets is faster than FFT for the typical (50, 50) margins on a
    # 1024 x 1024 image because the cross-correlation involves only the
    # sparse polyline support.  Pre-fetching the polyline indices avoids
    # repeat numpy re-broadcasts.
    poly_vs, poly_us = np.where(polyline_mask)
    if poly_vs.size == 0:
        return CoarseSearchResult(offset_vu=(0, 0), score=0.0)
    # Minimum in-bounds vertex count for a candidate shift to be eligible.
    # The per-vertex match fraction below is a ratio over the surviving
    # vertices, so a shift that clips nearly the whole polyline off-frame
    # can score a perfect 1.0 from a few edge-pixel stragglers; requiring
    # at least this many survivors keeps such shifts out of the running.
    min_support = min_support_fraction * float(poly_vs.size)
    best_dv = 0
    best_du = 0
    best_score = -1.0
    best_key = (math.inf, math.inf, math.inf, math.inf)
    for dv in range(-margin_v, margin_v + 1):
        shifted_v = poly_vs + dv
        valid_v = (shifted_v >= 0) & (shifted_v < height)
        if not valid_v.any():
            continue
        for du in range(-margin_u, margin_u + 1):
            shifted_u = poly_us + du
            valid = valid_v & (shifted_u >= 0) & (shifted_u < width)
            if not valid.any():
                continue
            sv = shifted_v[valid]
            su = shifted_u[valid]
            if float(sv.size) < min_support:
                # Too few vertices survive this shift for the match
                # fraction to be trustworthy; the candidate is ineligible.
                continue
            # Score is the fraction of in-bounds polyline points (after the
            # shift) that fall on edge pixels: the raw overlap count divided
            # by the in-bounds vertex count.  Normalizing by ``sv.size``
            # (== valid.sum(), guaranteed >= 1 by the ``valid.any()`` check
            # above) removes the raw count's bias toward shifts that keep
            # more vertices in bounds.  This is not the binary NCC (which
            # would divide by ``sqrt(sv.size)``); see the docstring.
            score = float(edge_f[sv, su].sum()) / float(sv.size)
            key = (abs(dv) + abs(du), abs(dv), dv, du)
            if score > best_score or (score == best_score and key < best_key):
                best_score = score
                best_key = key
                best_dv = dv
                best_du = du
    return CoarseSearchResult(offset_vu=(best_dv, best_du), score=max(best_score, 0.0))


def coarse_polarity_search_scored(
    edge_mask: NDArrayBoolType,
    image_gradient_vu: NDArrayFloatType,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    search_window_vu: tuple[int, int],
    *,
    min_support_fraction: float = DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
) -> CoarseSearchResult:
    """Integer coarse search scored by polarity-weighted edge overlap.

    A polarity-blind mask-overlap search (:func:`coarse_ncc_search_scored`)
    counts a model vertex as a match whenever it lands on any detected edge
    pixel, regardless of that edge's orientation.  In a cluttered scene --
    a ring ansa behind a limb, the terminator on a lit disc, a second moon
    in the field -- a dense population of unrelated edges can outscore the
    short true-limb arc, seeding the Levenberg-Marquardt stage in the wrong
    basin where it converges to a confident-but-wrong offset.

    This search adds the discriminator the mask-overlap count discards: the
    *direction* of the image gradient under each vertex.  A vertex still has
    to land on a detected edge pixel (the same density-neutral binary signal
    the overlap search uses), but its contribution is then weighted by
    ``max(0, cos theta)``, where ``theta`` is the angle between the model's
    per-vertex outward normal and the image gradient vector sampled at the
    shifted vertex.  A vertex whose local edge runs the wrong way (gradient
    anti-parallel to the model normal -- a terminator, a far-side ring edge,
    a crater rim) contributes nothing; an unrelated clutter edge at a random
    orientation contributes on average only ``1 / pi ~ 0.32`` of its overlap.
    The true limb, whose edges run along the predicted normals by
    construction, keeps nearly full weight.  Only the gradient *direction*
    enters (both vectors are unit-normalised before the dot product), so a
    faint-but-correctly-oriented true edge is never out-weighted by a bright
    high-contrast clutter edge -- decoupling the score from edge magnitude is
    what keeps a high-contrast distractor from dominating.

    Where the model normals are perfectly aligned with the image gradient
    (``cos theta = 1`` at every vertex) each match reverts to a hard 0/1 and
    the score becomes the plain per-vertex edge-overlap fraction, so the seed
    is unchanged on clean, uncluttered frames; the search only diverges from
    the overlap argmax where polarity actually discriminates.  (The absolute
    score is not identical to :func:`coarse_ncc_search_scored`'s even at
    ``cos theta = 1``: this search divides the match sum by the raw in-bounds
    vertex count, whereas the mask-overlap search divides by the count of
    distinct rasterized pixels, which collapses vertices that round together.
    Only the argmax matters for seeding, and it coincides in the aligned case.)

    The score is defined and validated identically to
    :func:`coarse_ncc_search_scored` -- the per-vertex weighted match summed
    over the in-bounds vertices and divided by their count, with the same
    minimum-support guard against near-fully-clipped shifts and the same
    ``(|dv| + |du|, |dv|, dv, du)`` tie-break -- except that each match is
    the polarity weight in ``[0, 1]`` rather than a hard 0/1.  The reported
    :attr:`CoarseSearchResult.score` is therefore the winning shift's mean
    polarity-weighted match fraction, a stricter acquisition-quality signal
    than the polarity-blind fraction for the shared coarse-peak spurious
    gate to consume.

    Parameters:
        edge_mask: ``(H, W)`` boolean image edge map.
        image_gradient_vu: ``(H, W, 2)`` per-pixel gradient vector as
            produced by
            :func:`spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu`,
            aligned to ``edge_mask``.
        vertices_vu: ``(N, 2)`` model polyline vertex positions in ``(v, u)``.
        normals_vu: ``(N, 2)`` model outward normal at each vertex, in the
            same polarity convention as :func:`polarity_filter` (the
            direction the image gradient is expected to point at a matching
            edge).  Need not be unit length; each normal is normalized
            internally.
        search_window_vu: ``(margin_v, margin_u)`` non-negative integers
            bounding the search range in v and u.
        min_support_fraction: Minimum fraction of the polyline's vertices
            that must remain in bounds after a candidate shift for that
            shift to be eligible.  See :func:`coarse_ncc_search`.

    Returns:
        :class:`CoarseSearchResult` with the peak integer offset and its
        mean polarity-weighted match fraction.

    Raises:
        TypeError: if either entry of ``search_window_vu`` is not an int.
        ValueError: if array shapes disagree, ``search_window_vu`` is not a
            length-2 sequence of non-negative ints, or ``min_support_fraction``
            is outside ``[0, 1]``.
    """
    if edge_mask.ndim != 2:
        raise ValueError(f'edge_mask must be 2-D; got ndim={edge_mask.ndim}')
    if image_gradient_vu.ndim != 3 or image_gradient_vu.shape[2] != 2:
        raise ValueError(
            f'image_gradient_vu must have shape (H, W, 2); got {image_gradient_vu.shape}'
        )
    if image_gradient_vu.shape[:2] != edge_mask.shape:
        raise ValueError(
            f'shape mismatch: edge_mask {edge_mask.shape} vs '
            f'image_gradient_vu {image_gradient_vu.shape[:2]}'
        )
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f'vertices_vu must have shape (N, 2); got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    margin_v, margin_u = _validate_search_window(search_window_vu)
    if not 0.0 <= min_support_fraction <= 1.0:
        raise ValueError(f'min_support_fraction must be in [0, 1]; got {min_support_fraction!r}')
    height, width = edge_mask.shape
    n_vertices = int(verts.shape[0])
    if n_vertices == 0:
        return CoarseSearchResult(offset_vu=(0, 0), score=0.0)
    # Round the vertices once; an integer shift commutes with rounding, so the
    # sampled pixel for shift (dv, du) is (round(v) + dv, round(u) + du).
    base_v = np.rint(verts[:, 0]).astype(np.int64)
    base_u = np.rint(verts[:, 1]).astype(np.int64)
    # Unit-normalise the model normals up front; a degenerate zero normal
    # gets a zero unit vector so it can never contribute a match.
    norm_len = np.hypot(norms[:, 0], norms[:, 1])
    safe_len = np.where(norm_len > 0.0, norm_len, 1.0)
    unit_n_v = norms[:, 0] / safe_len
    unit_n_u = norms[:, 1] / safe_len
    grad_v_img = image_gradient_vu[:, :, 0]
    grad_u_img = image_gradient_vu[:, :, 1]
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
            if not valid.any():
                continue
            sv = shifted_v[valid]
            su = shifted_u[valid]
            if float(sv.size) < min_support:
                continue
            on_edge = edge_mask[sv, su]
            gv = grad_v_img[sv, su]
            gu = grad_u_img[sv, su]
            grad_len = np.hypot(gv, gu)
            safe_grad = np.where(grad_len > 0.0, grad_len, 1.0)
            # Cosine of the angle between the (unit) model normal and the
            # (unit) image gradient; ``max(0, .)`` drops wrong-polarity edges.
            cos_theta = (unit_n_v[valid] * gv + unit_n_u[valid] * gu) / safe_grad
            weight = np.where(on_edge, np.maximum(cos_theta, 0.0), 0.0)
            score = float(weight.sum()) / float(sv.size)
            key = (abs(dv) + abs(du), abs(dv), dv, du)
            if score > best_score or (score == best_score and key < best_key):
                best_score = score
                best_key = key
                best_dv = dv
                best_du = du
    return CoarseSearchResult(offset_vu=(best_dv, best_du), score=max(best_score, 0.0))


def build_polyline_mask(
    vertices_vu: NDArrayFloatType, shape_vu: tuple[int, int]
) -> NDArrayBoolType:
    """Render polyline vertices into a boolean image mask aligned to ``shape_vu``.

    Each vertex is rounded to the nearest integer pixel; vertices that fall
    outside ``shape_vu`` are silently dropped.  The integer rasterization is
    deliberate -- the mask feeds :func:`coarse_ncc_search`, which scores
    integer-pixel shifts of the binary mask against the edge DT.

    Shared by the limb / terminator / ring-edge techniques (previously a
    verbatim per-module copy).

    Parameters:
        vertices_vu: ``(N, 2)`` polyline vertex positions in ``(v, u)``.
        shape_vu: ``(H, W)`` target mask shape.

    Returns:
        ``(H, W)`` boolean mask, True at each in-bounds rounded vertex.
    """
    vs = np.rint(vertices_vu[:, 0]).astype(np.int64)
    us = np.rint(vertices_vu[:, 1]).astype(np.int64)
    valid = (vs >= 0) & (vs < shape_vu[0]) & (us >= 0) & (us < shape_vu[1])
    mask = np.zeros(shape_vu, dtype=bool)
    if valid.any():
        mask[vs[valid], us[valid]] = True
    return mask


def polarity_filter(
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    image_gradient_vu: NDArrayFloatType,
    *,
    offset_vu: tuple[float, float] = (0.0, 0.0),
) -> NDArrayBoolType:
    """Return per-vertex polarity acceptance against an image gradient.

    For each vertex the helper samples the image gradient vector at the
    vertex's current shifted position and compares the gradient to the
    model's outward normal.  The polarity test is *strictly* greater than
    zero: orthogonal hits (dot product exactly zero, vanishingly rare in
    floating-point arithmetic) are rejected, never silently kept.

    Out-of-bounds vertices are rejected explicitly: a vertex whose
    rounded shifted position falls outside the image is never accepted,
    regardless of the gradient at the clamped boundary pixel.  (Sampling
    the clamped pixel and "letting it reject on its own merits" is unsafe
    because a strong frame-edge gradient -- common after zero-padding into
    the extended FOV -- can align with an outward normal and spuriously
    accept an off-image vertex.)  The clamp below only keeps the gather
    indices valid; the gathered value is discarded for such vertices.

    Parameters:
        vertices_vu: ``(N, 2)`` model vertex positions.
        normals_vu: ``(N, 2)`` model outward normal at each vertex.
        image_gradient_vu: ``(H, W, 2)`` per-pixel gradient image as
            produced by
            :func:`spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu`.
        offset_vu: ``(dv, du)`` shift applied to each vertex before
            sampling.  Defaults to no shift.

    Returns:
        ``(N,)`` boolean mask True where the vertex's polarity agrees
        with the image's local edge direction.

    Raises:
        ValueError: if shape requirements are violated.
    """
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f'vertices_vu must have shape (N, 2); got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    if image_gradient_vu.ndim != 3 or image_gradient_vu.shape[2] != 2:
        raise ValueError(
            f'image_gradient_vu must have shape (H, W, 2); got {image_gradient_vu.shape}'
        )
    height, width, _ = image_gradient_vu.shape
    dv, du = float(offset_vu[0]), float(offset_vu[1])
    rint_v = np.rint(verts[:, 0] + dv).astype(np.int64)
    rint_u = np.rint(verts[:, 1] + du).astype(np.int64)
    in_bounds = (rint_v >= 0) & (rint_v < height) & (rint_u >= 0) & (rint_u < width)
    sample_v = np.clip(rint_v, 0, height - 1)
    sample_u = np.clip(rint_u, 0, width - 1)
    gv = image_gradient_vu[sample_v, sample_u, 0]
    gu = image_gradient_vu[sample_v, sample_u, 1]
    dot = norms[:, 0] * gv + norms[:, 1] * gu
    return cast(NDArrayBoolType, in_bounds & (dot > 0.0))
