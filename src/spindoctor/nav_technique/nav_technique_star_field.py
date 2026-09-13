"""``StarFieldFromCatalogNav`` — multi-star RANSAC pattern matcher.

The technique is the long-pole prior-free star fit: it makes no assumption
about which detection corresponds to which catalog star, and recovers a
2-D translation purely from the *geometry* of bright sources in the image
matched against the predicted catalog field.

Algorithm — four sub-pieces:

1. **Source detection.**  The image is matched-filtered against a
   small Gaussian PSF kernel; pixels that are local maxima above
   ``detection_sigma * image_noise_sigma`` are accepted; a brightness-
   weighted centroid pins each peak's sub-pixel position.  The
   brightest ``max_sources`` survivors (default 30) feed the matcher.
2. **Triplet hashing.**  For every unordered triplet of detected
   sources ``{A, B, C}`` the hash ``(d_AB / d_AC, d_BC / d_AC, ∠BAC)``
   is computed.  The vertex order is canonicalised by triangle geometry
   -- ``A`` sits opposite the longest side, ``B`` opposite the next,
   ``C`` opposite the shortest -- which is the same physical labelling
   for a triangle and any translated, rotated, or uniformly scaled copy.
   The same hash is computed for catalog triplets.  The hash is
   similarity-invariant -- translation, rotation, and uniform scale all
   leave it unchanged -- so the matcher recovers correspondences without
   already knowing the offset.  A purely geometric canonical order is
   used rather than a brightness order because brightness ties on
   equal-magnitude fields, where a brightness-keyed order would flip the
   labelling between the detection and catalog sides run to run and under
   image rotation.
3. **RANSAC.**  Each (detection-triplet, catalog-triplet) candidate is
   scored by counting detection-to-catalog inliers under the
   translation it implies.  Candidates are iterated in deterministic
   order — sorted first by hash distance ascending, then by sorted
   ``(idx_A, idx_B, idx_C)`` ascending — so the matcher's choice of
   winner is fully reproducible (Cardinal Principle 3, Part 3
   §"Determinism in RANSAC").
4. **Verification.**  With the best transform's inlier set, the
   technique refits the translation by Tukey-biweight-reweighted
   least squares; the per-axis variance of the surviving residuals
   becomes the reported covariance.

Phase 8 ships translation-only fitting; the
``fit_camera_rotation`` rotation upgrade arrives in Phase 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.nav_model.stars.detection import matched_filter_image
from spindoctor.nav_technique._star_helpers import (
    SimilarityFit,
    brightness_margin_mag,
    predicted_snr,
    predicted_vu,
    similarity_transform_fit,
    usable_stars,
)
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import StarFieldDiagnostics
from spindoctor.nav_technique.dt_fitting import DEFAULT_TUKEY_C, tukey_biweight_weights
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    embed_rotation_unobservable,
    load_model_error_floor,
    log_confidence_breakdown,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['StarFieldFromCatalogNav']


@runtime_checkable
class _StarPSFProvider(Protocol):
    """Structural type for an observation that can supply a star PSF model.

    The PSF-refinement step only needs ``star_psf()``; typing the obs against
    this Protocol (rather than the concrete ``ObsSnapshotInst``) keeps the
    technique decoupled from the obs class hierarchy and lets a lightweight
    test double opt into refinement just by exposing the one method.
    """

    def star_psf(self) -> PSF: ...


_COLLINEAR_REL_EPS: float = 1.0e-6
"""Relative tolerance for the collinear-triplet rejection in ``_triplet_hash``.

A triplet is treated as collinear (and rejected) when the magnitude of
its 2-D cross product falls below ``_COLLINEAR_REL_EPS * d_ab * d_ac``.
The dimensionless ratio survives uniform scaling of the input, so the
floor expresses "the smaller-angle deviation from a perfect line" in
the same way a relative-error tolerance does for floating-point
comparisons.  ``1e-6`` admits numerical jitter on near-degenerate
real-detection triplets while still catching the geometric
degeneracy.
"""


# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.StarFieldFromCatalogNav.tuning``.  Missing-key access in
# ``__init__`` is a KeyError so a config typo fails fast at process
# startup.


@dataclass(frozen=True)
class _DetectedSource:
    """Internal detection record produced by ``_detect_image_sources``.

    The technique uses only ``(v, u, peak_dn)``; richer DAOPHOT
    statistics (sharpness / roundness, saturation tag) are not needed
    for triplet matching and are therefore not surfaced.
    """

    v: float
    u: float
    peak_dn: float


@dataclass(frozen=True)
class _WideOffsetMatch:
    """Accepted wide-offset asterism lock (see ``_wide_offset_asterism_match``).

    Parameters:
        n_inliers: Number of detection-to-catalog inliers at the winning
            bright-anchor-seeded offset.
        correspondences: ``(det_idx, cat_idx)`` inlier pairs, detection-index
            ascending.
        offset: Winning ``(dv, du)`` translation.
        expectation: Expected number of chance locks of this inlier count,
            the false-lock significance the acceptance budget bounds.
        n_seeds: Number of distinct bright-anchor seed offsets evaluated (the
            multiple-testing trial count folded into ``expectation``).
        is_one_star: True for a single-bright-star wide-offset lock, False for
            an asterism (multi-inlier) lock; the ensemble caps a one-star lock's
            confidence.
    """

    n_inliers: int
    correspondences: list[tuple[int, int]]
    offset: tuple[float, float]
    expectation: float
    n_seeds: int
    is_one_star: bool


def _star_photometry_saturated(feature: NavFeature) -> bool:
    """Return the ``photometry_saturated`` flag off a STAR feature.

    True marks a bright star whose catalog magnitude is an untrusted lower
    bound (no reference-catalog match); such a star's brightness ordering is
    uncertain, so it is not used to anchor a wide-offset seed.  A feature
    whose flags are not STAR flags (so lack the field) yields ``False``.
    """
    flags = feature.flags
    return isinstance(flags, StarFlags) and flags.photometry_saturated


def _binomial_upper_tail(n_trials: int, p_success: float, k_min: int) -> float:
    """Return ``P(X >= k_min)`` for ``X ~ Binomial(n_trials, p_success)``.

    Summed exactly over the upper tail; the counts here (a dozen-odd
    catalog stars) keep the sum tiny, so no normal / Poisson approximation
    is needed.

    Parameters:
        n_trials: Number of Bernoulli trials (``>= 0``).
        p_success: Per-trial success probability, clamped to ``[0, 1]``.
        k_min: Lower bound of the tail.

    Returns:
        The upper-tail probability in ``[0, 1]``.
    """
    if k_min <= 0:
        return 1.0
    if k_min > n_trials:
        return 0.0
    p = min(max(p_success, 0.0), 1.0)
    q = 1.0 - p
    tail = 0.0
    for k in range(k_min, n_trials + 1):
        tail += math.comb(n_trials, k) * (p**k) * (q ** (n_trials - k))
    return min(max(tail, 0.0), 1.0)


def _wide_offset_false_lock_expectation(
    *,
    n_inliers: int,
    n_seeds: int,
    n_catalog: int,
    n_detected: int,
    tolerance_px: float,
    image_area_px2: float,
) -> float:
    """Expected number of chance wide-offset locks reaching ``n_inliers``.

    Under the null that the ``n_detected`` detections are scattered
    uniformly over the image, a single catalog star coincides with some
    detection within ``tolerance_px`` of a fixed offset with probability
    ``p = n_detected * pi * tolerance_px**2 / image_area_px2``.  Each
    bright-anchor seed contributes one guaranteed inlier (the anchor pair)
    plus ``Binomial(n_catalog - 1, p)`` chance inliers among the remaining
    catalog stars, so a seed reaches ``n_inliers`` total with probability
    ``P(Binom(n_catalog - 1, p) >= n_inliers - 1)``.  Multiplying by the
    ``n_seeds`` distinct seed offsets gives the expected count of chance
    locks -- the quantity the acceptance budget bounds.

    This is what lets a sparse field lock on three inliers where the
    triplet RANSAC needs six: the RANSAC evaluates thousands of triplet
    seeds, so three inliers (a seed triangle plus zero corroboration) are
    routinely reachable by chance; restricting to a handful of
    brightness-anchored seeds collapses the trial count so the same three
    inliers become statistically decisive.

    Parameters:
        n_inliers: Inlier count of the candidate lock.
        n_seeds: Number of distinct bright-anchor seed offsets evaluated.
        n_catalog: Number of predicted catalog stars.
        n_detected: Number of detected image sources.
        tolerance_px: Inlier match radius in pixels.
        image_area_px2: Image area in square pixels (detection density
            denominator).

    Returns:
        Expected number of chance locks of at least ``n_inliers`` inliers.
    """
    if n_inliers <= 0 or n_seeds <= 0 or n_catalog <= 1 or image_area_px2 <= 0.0:
        return 0.0
    p_star = n_detected * math.pi * tolerance_px * tolerance_px / image_area_px2
    tail = _binomial_upper_tail(n_catalog - 1, p_star, n_inliers - 1)
    return float(n_seeds) * tail


def _gaussian_kernel(sigma_px: float, *, size: int) -> NDArrayFloatType:
    """Return a normalised 2-D isotropic Gaussian matched-filter kernel.

    Parameters:
        sigma_px: PSF sigma in pixels.  Must be strictly positive.
        size: Side length of the (odd) square kernel.  Coerced up to
            the next odd integer when even.

    Returns:
        ``(size, size)`` float64 array, sum-to-unity.
    """
    if sigma_px <= 0.0:
        raise ValueError(f'sigma_px must be > 0; got {sigma_px!r}')
    if size % 2 == 0:
        size += 1
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    vv, uu = np.meshgrid(coords, coords, indexing='ij')
    kernel = np.exp(-(vv * vv + uu * uu) / (2.0 * sigma_px * sigma_px))
    kernel /= float(kernel.sum())
    return kernel


def _detect_image_sources(
    image: NDArrayFloatType,
    *,
    image_noise_sigma: float,
    sigma_px: float,
    detection_sigma: float,
    centroid_box_half_px: int,
    max_sources: int,
) -> list[_DetectedSource]:
    """Detect bright local-max sources in ``image`` and return the brightest set.

    Three-stage pipeline:

    1. Matched-filter the image against a Gaussian kernel sized by
       ``sigma_px``; the response peak amplitude at each pixel is the
       maximum-likelihood signal estimate at that location.
    2. Find pixels that are local maxima within a
       ``(2 * centroid_box_half_px + 1)`` window AND clear
       ``detection_sigma * image_noise_sigma``.
    3. For each surviving peak, fit a brightness-weighted moment in a
       small box around the peak to pull a sub-pixel centroid.

    Detections are ranked by ``(peak_dn descending, v ascending,
    u ascending)`` and the first ``max_sources`` are returned — the
    triplet matcher is M^3 in the input count, so 30 is the typical
    cap (Part 3 §"Determinism in RANSAC").

    Parameters:
        image: 2-D float input.
        image_noise_sigma: Robust per-pixel noise sigma in DN.
        sigma_px: PSF sigma in pixels for the matched-filter kernel.
        detection_sigma: Threshold multiplier on
            ``image_noise_sigma``.
        centroid_box_half_px: Half-width of the centroid box (and of
            the local-max window).
        max_sources: Maximum number of detections to return.

    Returns:
        List of ``_DetectedSource`` records, ordered by brightness
        descending.  May be empty when no peak clears the threshold.
    """
    kernel_size = 2 * centroid_box_half_px + 1
    kernel = _gaussian_kernel(sigma_px, size=kernel_size)
    response = matched_filter_image(image, kernel=kernel)
    threshold = detection_sigma * image_noise_sigma
    local_max = maximum_filter(response, size=kernel_size, mode='reflect')
    candidate_mask = (response == local_max) & (response > threshold)
    h, w = image.shape
    out: list[_DetectedSource] = []
    for v_idx, u_idx in np.argwhere(candidate_mask):
        v_i = int(v_idx)
        u_i = int(u_idx)
        v_lo = max(0, v_i - centroid_box_half_px)
        u_lo = max(0, u_i - centroid_box_half_px)
        v_hi = min(h, v_i + centroid_box_half_px + 1)
        u_hi = min(w, u_i + centroid_box_half_px + 1)
        box = image[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
        bg = float(np.median(box))
        weights = np.clip(box - bg, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        vs = np.arange(v_lo, v_hi, dtype=np.float64)
        us = np.arange(u_lo, u_hi, dtype=np.float64)
        cv = float(np.sum(vs[:, None] * weights) / total)
        cu = float(np.sum(us[None, :] * weights) / total)
        out.append(_DetectedSource(v=cv, u=cu, peak_dn=float(response[v_i, u_i])))
    out.sort(key=lambda s: (-s.peak_dn, s.v, s.u))
    return out[:max_sources]


def _triplet_hash(
    pa: tuple[float, float],
    pb: tuple[float, float],
    pc: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Return the similarity-invariant ``(d_AB / d_AC, d_BC / d_AC, ∠BAC)`` triple.

    The hash is computed with ``A`` as the caller's canonical vertex
    (the vertex opposite the longest side; the caller is responsible for
    that canonicalisation); both ratios survive arbitrary rotation,
    translation, and uniform scale, which is what lets the matcher
    recover correspondences without knowing the transform.

    Parameters:
        pa: Canonical apex vertex of the triplet (opposite the longest
            side).
        pb, pc: The other two vertices, in caller-canonical order
            (opposite the next-longest and shortest side respectively)
            so each unordered triplet hashes once.

    Returns:
        ``(r1, r2, theta)`` tuple, or ``None`` for degenerate
        configurations (collinear vertices, coincident points) which
        are silently dropped.
    """
    av, au = pa
    bv, bu = pb
    cv, cu = pc
    ab_v = bv - av
    ab_u = bu - au
    ac_v = cv - av
    ac_u = cu - au
    bc_v = cv - bv
    bc_u = cu - bu
    d_ab = math.hypot(ab_v, ab_u)
    d_ac = math.hypot(ac_v, ac_u)
    d_bc = math.hypot(bc_v, bc_u)
    if d_ab <= 0.0 or d_ac <= 0.0:
        return None
    # Collinear-but-distinct triplets produce ``theta = 0`` or ``pi`` —
    # a hash that can match any other 0-or-pi hash regardless of scale,
    # which is a false-match trap for the RANSAC matcher.  Reject when
    # the 2-D cross product magnitude (``d_ab * d_ac * |sin(theta)|``)
    # falls below a small fraction of the side-length product, which
    # is dimensionless and survives uniform scaling of the input.
    cross = ab_v * ac_u - ab_u * ac_v
    if abs(cross) <= _COLLINEAR_REL_EPS * d_ab * d_ac:
        return None
    cos_theta = (ab_v * ac_v + ab_u * ac_u) / (d_ab * d_ac)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return (d_ab / d_ac, d_bc / d_ac, math.acos(cos_theta))


@dataclass(frozen=True)
class _Triplet:
    """One enumerated triplet, hashed in canonical form.

    ``idx_a / idx_b / idx_c`` reference back into the source point
    list; ``a`` is the vertex opposite the longest side, ``b`` opposite
    the next-longest, and ``c`` opposite the shortest, so each unordered
    triplet appears once in a rotation-stable order.
    """

    idx_a: int
    idx_b: int
    idx_c: int
    hash_v: tuple[float, float, float]


def _canonical_triplet_order(
    points: list[tuple[float, float]],
    triple: tuple[int, int, int],
    brightness_rank: list[int],
) -> tuple[int, int, int]:
    """Return the vertex indices of ``triple`` in geometric canonical order.

    Vertices are ordered by the length of the side opposite each one,
    longest first: ``A`` sits opposite the longest side, ``B`` opposite
    the next, ``C`` opposite the shortest.  Side lengths scale uniformly
    under a similarity transform, so this order is identical for a
    triangle and any translated, rotated, or uniformly scaled copy of it;
    the detection triplet and its catalog counterpart therefore
    canonicalise to the same physical vertex assignment without appealing
    to brightness -- which ties on the equal-magnitude fields that made a
    brightness-keyed order a run-to-run and by-rotation seed lottery.  A
    near-isosceles triangle (two opposite sides of equal length) breaks
    the tie on brightness rank ascending, then original index ascending,
    so the order stays total and deterministic.

    Parameters:
        points: The full point list the triple indexes into.
        triple: The three point indices, in ascending order.
        brightness_rank: Per-point brightness rank (``0`` = brightest),
            the secondary tie-break for a near-isosceles triangle.

    Returns:
        ``(idx_a, idx_b, idx_c)`` in canonical apex-first order.
    """
    i, j, k = triple
    pi, pj, pk = points[i], points[j], points[k]
    opp_i = math.hypot(pj[0] - pk[0], pj[1] - pk[1])
    opp_j = math.hypot(pi[0] - pk[0], pi[1] - pk[1])
    opp_k = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
    keyed = sorted(
        (
            (-opp_i, brightness_rank[i], i),
            (-opp_j, brightness_rank[j], j),
            (-opp_k, brightness_rank[k], k),
        )
    )
    return keyed[0][2], keyed[1][2], keyed[2][2]


def _enumerate_triplets(
    points: list[tuple[float, float]],
    brightness_rank: list[int],
) -> list[_Triplet]:
    """Return every canonical ``_Triplet`` over ``points``.

    Each unordered ``(i, j, k)`` triple is canonicalised by triangle
    geometry via :func:`_canonical_triplet_order` (``brightness_rank``
    only breaks a near-isosceles tie).  Degenerate triplets (collinear or
    coincident) are dropped silently.
    """
    n = len(points)
    out: list[_Triplet] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_a, idx_b, idx_c = _canonical_triplet_order(points, (i, j, k), brightness_rank)
                hashed = _triplet_hash(points[idx_a], points[idx_b], points[idx_c])
                if hashed is None:
                    continue
                out.append(_Triplet(idx_a=idx_a, idx_b=idx_b, idx_c=idx_c, hash_v=hashed))
    return out


def _hash_distance_sq(
    h1: tuple[float, float, float],
    h2: tuple[float, float, float],
    *,
    ratio_weight: float,
    angle_weight: float,
) -> float:
    """Return the weighted squared Euclidean distance between two triplet hashes.

    ``ratio_weight`` scales the two ratio components and
    ``angle_weight`` scales the angle component.  Both default to 1.0
    in the YAML — radians and dimensionless ratios end up roughly
    comparable in magnitude over the typical triplet shapes.
    """
    dr1 = h1[0] - h2[0]
    dr2 = h1[1] - h2[1]
    da = h1[2] - h2[2]
    return ratio_weight * (dr1 * dr1 + dr2 * dr2) + angle_weight * da * da


def _optimal_inlier_assignment(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    offset_vu: tuple[float, float],
    *,
    tolerance_px: float,
) -> tuple[int, list[tuple[int, int]]]:
    """Optimal one-to-one detection-catalog matching under the proposed translation.

    Detections are paired one-to-one with catalog stars (after applying
    the offset to catalog positions) so that the number of pairs within
    ``tolerance_px`` is maximised and, among maximum-cardinality
    assignments, the total squared residual distance is minimised.  The
    detection x catalog squared-distance matrix is solved as a linear
    sum assignment (Hungarian algorithm); entries beyond the tolerance
    are masked with a cost large enough that the solver never trades an
    in-tolerance pair for masked ones, and masked pairs the solver is
    still forced to emit are dropped afterwards.  Unlike a greedy
    nearest-neighbour sweep, the result is independent of detection
    ordering: when two detections compete for the same catalog star the
    globally best one-to-one pairing wins.

    Parameters:
        detection_pts: ``(N_det, 2)`` array of detection (v, u).
        catalog_pts: ``(N_cat, 2)`` array of catalog (v, u).
        offset_vu: Translation applied to catalog positions before
            matching.
        tolerance_px: Maximum residual distance for a correspondence.

    Returns:
        ``(n_inliers, correspondences)`` where ``correspondences`` is a
        list of ``(det_idx, cat_idx)`` tuples in detection-index
        ascending order.  ``n_inliers`` equals
        ``len(correspondences)``.
    """
    if detection_pts.size == 0 or catalog_pts.size == 0:
        return 0, []
    shifted_catalog = catalog_pts + np.asarray(offset_vu, np.float64)[None, :]
    diffs = detection_pts[:, None, :] - shifted_catalog[None, :, :]
    dist_sq = np.sum(diffs * diffs, axis=2)
    tol_sq = tolerance_px * tolerance_px
    # Any masked pair must cost more than every in-tolerance pair an
    # assignment can contain combined, so maximum cardinality always
    # beats any residual-distance trade.
    n_assignable = min(detection_pts.shape[0], shifted_catalog.shape[0])
    masked_cost = tol_sq * (n_assignable + 1.0) + 1.0
    cost = np.where(dist_sq <= tol_sq, dist_sq, masked_cost)
    det_indices, cat_indices = linear_sum_assignment(cost)
    pairs = [
        (int(d_idx), int(c_idx))
        for d_idx, c_idx in zip(det_indices, cat_indices, strict=True)
        if dist_sq[d_idx, c_idx] <= tol_sq
    ]
    pairs.sort()
    return len(pairs), pairs


def _triplet_centroid_offset(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    *,
    det_indices: tuple[int, int, int],
    cat_indices: tuple[int, int, int],
) -> tuple[float, float]:
    """Return the translation aligning two triplet centroids.

    Centroids are the unweighted mean of three vertex positions.
    Numerically equivalent to ``mean(detection - catalog)`` over the
    triplet but expressed centroid-style for clarity.
    """
    det_v = (
        detection_pts[det_indices[0], 0]
        + detection_pts[det_indices[1], 0]
        + detection_pts[det_indices[2], 0]
    ) / 3.0
    det_u = (
        detection_pts[det_indices[0], 1]
        + detection_pts[det_indices[1], 1]
        + detection_pts[det_indices[2], 1]
    ) / 3.0
    cat_v = (
        catalog_pts[cat_indices[0], 0]
        + catalog_pts[cat_indices[1], 0]
        + catalog_pts[cat_indices[2], 0]
    ) / 3.0
    cat_u = (
        catalog_pts[cat_indices[0], 1]
        + catalog_pts[cat_indices[1], 1]
        + catalog_pts[cat_indices[2], 1]
    ) / 3.0
    return float(det_v - cat_v), float(det_u - cat_u)


def _solve_translation(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> tuple[float, float]:
    """Return the weighted-mean translation ``mean(d - c)``.

    Returns ``(0.0, 0.0)`` when the total weight is non-positive.
    """
    diffs = detection_pts - catalog_pts
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0, 0.0
    dv = float(np.sum(weights * diffs[:, 0]) / total)
    du = float(np.sum(weights * diffs[:, 1]) / total)
    return dv, du


class _StarFieldConfidenceContext:
    """Adapter binding ``StarFieldDiagnostics`` plus side flags."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: StarFieldDiagnostics,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.n_inliers = float(diagnostics.n_inliers)
        self.median_residual_px = diagnostics.median_residual_px
        self.n_detected_sources = float(diagnostics.n_detected_sources)
        self.n_catalog_predicted = float(diagnostics.n_catalog_predicted)


class StarFieldFromCatalogNav(NavTechnique):
    """Multi-star pattern-match translation fit (≥ 3 catalog + image stars).

    Class attributes:
        accepts_feature_types: ``frozenset({STAR})``.
        requires_prior: ``False`` — runs in pass 1.
    """

    name = 'StarFieldFromCatalogNav'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'n_inliers',
            'median_residual_px',
            'n_detected_sources',
            'n_catalog_predicted',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._max_sources = int(self.tuning['max_sources'])
        self._detection_sigma = float(self.tuning['detection_sigma'])
        self._psf_sigma_px = float(self.tuning['psf_sigma_px'])
        self._centroid_box_half_px = int(self.tuning['centroid_box_half_px'])
        self._hash_match_tolerance = float(self.tuning['hash_match_tolerance'])
        self._hash_ratio_weight = float(self.tuning['hash_ratio_weight'])
        self._hash_angle_weight = float(self.tuning['hash_angle_weight'])
        self._inlier_tolerance_px = float(self.tuning['inlier_tolerance_px'])
        self._min_inliers = int(self.tuning['pattern_match_min_inliers'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        # Roll / translation separability floor (deg).  Below this a fitted
        # rotation is not separable from a pure translation (see the
        # camera-roll separability analysis in docs/simulator_report) and is
        # reported as unobservable; see config_510_techniques.yaml.
        self._rotation_separability_floor_deg = float(
            self.tuning['rotation_separability_floor_deg']
        )
        # Uncalibrated model-error variance floor (px); added in quadrature to
        # the reported covariance diagonal.  Default 0.0 -> no-op.  See
        # ORCH-001 / config_510_techniques.yaml.
        self._model_error_floor_px = load_model_error_floor(self.tuning, self.name)
        # Wide-offset asterism matching for sparse few-bright-star fields
        # (see config_510_techniques.yaml and ``_wide_offset_asterism_match``).
        self._wide_offset_enabled = bool(int(self.tuning['wide_offset_enabled']))
        self._wide_offset_min_inliers = int(self.tuning['wide_offset_min_inliers'])
        self._wide_offset_anchor_catalog = int(self.tuning['wide_offset_anchor_catalog'])
        self._wide_offset_anchor_detections = int(self.tuning['wide_offset_anchor_detections'])
        self._wide_offset_false_lock_budget = float(self.tuning['wide_offset_false_lock_budget'])
        self._wide_offset_max_residual_rms_px = float(
            self.tuning['wide_offset_max_residual_rms_px']
        )
        self._wide_offset_confidence_cap = float(self.tuning['wide_offset_confidence_cap'])
        # Single-bright-star wide-offset acquisition (see
        # config_510_techniques.yaml and ``_wide_offset_one_star_match``).
        self._wide_offset_one_star_enabled = bool(int(self.tuning['wide_offset_one_star_enabled']))
        self._wide_offset_one_star_catalog_margin_mag = float(
            self.tuning['wide_offset_one_star_catalog_margin_mag']
        )
        self._wide_offset_one_star_detection_margin_ratio = float(
            self.tuning['wide_offset_one_star_detection_margin_ratio']
        )
        self._wide_offset_one_star_confidence_cap = float(
            self.tuning['wide_offset_one_star_confidence_cap']
        )
        # PSF-fit re-centroiding of matched inliers (see config_510_techniques.yaml).
        self._psf_refine_enabled = bool(int(self.tuning['psf_refine_enabled']))
        self._psf_refine_box_px = int(self.tuning['psf_refine_box_px'])
        self._psf_refine_search_limit_px = float(self.tuning['psf_refine_search_limit_px'])
        self._psf_refine_snr_max = float(self.tuning['psf_refine_snr_max'])
        if self._psf_refine_box_px % 2 == 0:
            raise ValueError(f'psf_refine_box_px must be odd; got {self._psf_refine_box_px}')
        if self._min_inliers < 3:
            raise ValueError(
                f'pattern_match_min_inliers must be >= 3 (the matcher needs at '
                f'least one triplet per side); got {self._min_inliers}'
            )
        if self._max_sources < 3:
            raise ValueError(f'max_sources must be >= 3; got {self._max_sources}')
        if self._wide_offset_min_inliers < 3:
            raise ValueError(
                f'wide_offset_min_inliers must be >= 3 (two stars cannot cross-'
                f'check a rigid translation); got {self._wide_offset_min_inliers}'
            )
        if self._wide_offset_anchor_catalog < 1:
            raise ValueError(
                f'wide_offset_anchor_catalog must be >= 1; got {self._wide_offset_anchor_catalog}'
            )
        if self._wide_offset_anchor_detections < 1:
            raise ValueError(
                f'wide_offset_anchor_detections must be >= 1; '
                f'got {self._wide_offset_anchor_detections}'
            )
        if self._wide_offset_false_lock_budget <= 0.0:
            raise ValueError(
                f'wide_offset_false_lock_budget must be > 0; '
                f'got {self._wide_offset_false_lock_budget}'
            )
        if not 0.0 <= self._wide_offset_confidence_cap <= 1.0:
            raise ValueError(
                f'wide_offset_confidence_cap must lie in [0, 1]; '
                f'got {self._wide_offset_confidence_cap}'
            )
        if self._wide_offset_one_star_catalog_margin_mag <= 0.0:
            raise ValueError(
                f'wide_offset_one_star_catalog_margin_mag must be > 0; '
                f'got {self._wide_offset_one_star_catalog_margin_mag}'
            )
        if self._wide_offset_one_star_detection_margin_ratio < 1.0:
            raise ValueError(
                f'wide_offset_one_star_detection_margin_ratio must be >= 1; '
                f'got {self._wide_offset_one_star_detection_margin_ratio}'
            )
        if not 0.0 <= self._wide_offset_one_star_confidence_cap <= 1.0:
            raise ValueError(
                f'wide_offset_one_star_confidence_cap must lie in [0, 1]; '
                f'got {self._wide_offset_one_star_confidence_cap}'
            )

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Report feasibility based on the predictable-star cohort.

        Reads only feature metadata; never any pixels.  Feasible when
        at least three usable STAR features are in the input set —
        below that the matcher cannot form a single triplet.
        """
        usable = usable_stars(features)
        if len(usable) < 3:
            return NavFeasibilityReport(
                feasible=False,
                reason=f'fewer_than_3_predicted_stars (got {len(usable)})',
            )
        return NavFeasibilityReport(feasible=True, reason='ok', consumed_feature_count=len(usable))

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Recover translation from triplet pattern matching against the catalog.

        Parameters:
            features: STAR features (the orchestrator pre-filters by
                type).  At least three must survive
                ``usable_stars``.
            context: Per-image NavContext.

        Returns:
            ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`StarFieldDiagnostics`.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            usable = usable_stars(features)
            self.logger.info(
                'Consuming %d usable STAR feature(s) (out of %d offered)',
                len(usable),
                len(features),
            )
            ranked_catalog = sorted(usable, key=predicted_snr, reverse=True)[: self._max_sources]
            n_catalog_predicted = len(ranked_catalog)
            if n_catalog_predicted < 3:
                return self._fail(
                    features=features,
                    reason=f'fewer_than_3_predicted_stars (got {n_catalog_predicted})',
                    diagnostics=StarFieldDiagnostics(
                        n_inliers=0,
                        median_residual_px=0.0,
                        n_detected_sources=0,
                        n_catalog_predicted=n_catalog_predicted,
                        n_triplets_evaluated=0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            detected = _detect_image_sources(
                image_ext,
                image_noise_sigma=noise_sigma,
                sigma_px=self._psf_sigma_px,
                detection_sigma=self._detection_sigma,
                centroid_box_half_px=self._centroid_box_half_px,
                max_sources=self._max_sources,
            )
            n_detected_sources = len(detected)
            self.logger.info(
                'Detected %d source(s); using up to %d catalog stars',
                n_detected_sources,
                n_catalog_predicted,
            )
            if n_detected_sources < 3:
                return self._fail(
                    features=features,
                    reason=f'fewer_than_3_detected_sources (got {n_detected_sources})',
                    diagnostics=StarFieldDiagnostics(
                        n_inliers=0,
                        median_residual_px=0.0,
                        n_detected_sources=n_detected_sources,
                        n_catalog_predicted=n_catalog_predicted,
                        n_triplets_evaluated=0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            return self._match_and_fit(
                features=features,
                detected=detected,
                ranked_catalog=ranked_catalog,
                context=context,
                n_detected_sources=n_detected_sources,
                n_catalog_predicted=n_catalog_predicted,
            )

    def _psf_refine_positions(
        self, det_inliers: NDArrayFloatType, context: NavContext
    ) -> NDArrayFloatType:
        """Re-centroid each matched inlier with a true PSF fit where it helps.

        The moment centroid is unbiased but noise-limited; a maximum-likelihood
        PSF fit against the instrument's modelled point-spread function reaches
        the minimum variance and so sharply reduces the per-star error of faint
        detections.  An undersampled PSF fit, however, carries a fixed
        sub-pixel-phase bias floor, so a detection bright enough that its moment
        noise already sits below that floor keeps the moment.  The brightness
        test is a per-detection integrated SNR over the fit box.

        Detections whose fit fails (off the edge, too few good pixels, no
        convergence) silently fall back to the moment centroid.

        Parameters:
            det_inliers: ``(N, 2)`` moment centroids ``(v, u)`` of the matched
                inliers, in the extended-FOV frame.
            context: Per-image NavContext (supplies the image, the noise sigma,
                and the observation's PSF model).

        Returns:
            ``(N, 2)`` refined positions; rows that could not be improved are
            returned unchanged.
        """
        obs = context.obs
        if not isinstance(obs, _StarPSFProvider):
            return det_inliers
        psf = obs.star_psf()
        image = np.asarray(context.image_ext, np.float64)
        noise_sigma = float(max(context.image_noise_sigma, 1e-9))
        box = self._psf_refine_box_px
        half = box // 2
        search_limit = (self._psf_refine_search_limit_px, self._psf_refine_search_limit_px)
        h, w = image.shape
        refined = det_inliers.copy()
        n_refined = 0
        for i, (v, u) in enumerate(det_inliers):
            v_pix = round(float(v))
            u_pix = round(float(u))
            if not (half <= v_pix < h - half and half <= u_pix < w - half):
                continue
            if self._box_snr(image, v_pix, u_pix, half, noise_sigma) > self._psf_refine_snr_max:
                continue  # bright: moment beats the PSF fit's bias floor
            try:
                result = psf.find_position(
                    image, (box, box), (float(v_pix), float(u_pix)), search_limit=search_limit
                )
            except (ValueError, RuntimeError):
                continue
            if result is None:
                continue
            # ``find_position`` reports the position in ``eval_rect`` convention
            # (offset measured from the pixel's lower edge, which is oops uv);
            # this technique works in pixel indices, so subtract the half-pixel
            # datum to match the moment centroids it replaces.
            refined[i, 0] = result[0] - 0.5
            refined[i, 1] = result[1] - 0.5
            n_refined += 1
        self.logger.debug(
            'PSF-refined %d of %d matched inlier(s); the rest kept their moment centroid',
            n_refined,
            len(det_inliers),
        )
        return refined

    @staticmethod
    def _box_snr(
        image: NDArrayFloatType, v_pix: int, u_pix: int, half: int, noise_sigma: float
    ) -> float:
        """Return the integrated SNR of a source in a square box around a pixel.

        ``signal = sum(clip(box - median, 0))`` is the total net counts; the
        variance is ``signal + n_pix * noise_sigma**2`` (source-shot plus
        background/read), so the ratio is the photon-noise-limited detection
        SNR used to pick the moment-vs-PSF crossover.
        """
        box = image[v_pix - half : v_pix + half + 1, u_pix - half : u_pix + half + 1]
        net = np.clip(box - float(np.median(box)), 0.0, None)
        signal = float(net.sum())
        variance = signal + net.size * noise_sigma * noise_sigma
        if variance <= 0.0:
            return 0.0
        return signal / math.sqrt(variance)

    def _match_and_fit(
        self,
        *,
        features: list[NavFeature],
        detected: list[_DetectedSource],
        ranked_catalog: list[NavFeature],
        context: NavContext,
        n_detected_sources: int,
        n_catalog_predicted: int,
    ) -> NavTechniqueResult:
        """RANSAC + Tukey-biweight verification on the prepared cohorts."""
        det_points = [(s.v, s.u) for s in detected]
        det_points_arr = np.asarray(det_points, np.float64)
        det_brightness_rank = list(range(n_detected_sources))  # already sorted
        cat_points = [predicted_vu(f) for f in ranked_catalog]
        cat_points_arr = np.asarray(cat_points, np.float64)
        cat_brightness_rank = list(range(n_catalog_predicted))  # already sorted by SNR
        det_triplets = _enumerate_triplets(det_points, det_brightness_rank)
        cat_triplets = _enumerate_triplets(cat_points, cat_brightness_rank)
        if not det_triplets or not cat_triplets:
            return self._fail(
                features=features,
                reason='no_valid_triplets_after_canonicalisation',
                diagnostics=StarFieldDiagnostics(
                    n_inliers=0,
                    median_residual_px=0.0,
                    n_detected_sources=n_detected_sources,
                    n_catalog_predicted=n_catalog_predicted,
                    n_triplets_evaluated=0,
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        candidates = self._enumerate_candidates(det_triplets, cat_triplets)
        n_triplets_evaluated = len(candidates)
        self.logger.debug(
            '%d detection triplet(s) and %d catalog triplet(s) -> %d hash-distance candidates',
            len(det_triplets),
            len(cat_triplets),
            n_triplets_evaluated,
        )
        best = self._score_candidates(
            candidates=candidates,
            cat_triplets=cat_triplets,
            det_points_arr=det_points_arr,
            cat_points_arr=cat_points_arr,
        )
        wide_offset_lock = False
        one_star_lock = False
        wide_offset_expectation = 0.0
        if best is None or best[0] < self._min_inliers:
            # The triplet RANSAC could not reach its strong-evidence floor.
            # Sparse few-bright-star fields with a large unknown offset land
            # here routinely (the faint remainder is undetectable, so no
            # triangle carries six inliers); try the wide-offset asterism
            # matcher, which seeds a translation from a handful of trusted
            # bright-anchor pairings and accepts a lower inlier count only
            # when the chance-alignment significance clears the false-lock
            # budget.
            det_dn = np.asarray([s.peak_dn for s in detected], np.float64)
            image_shape = np.asarray(context.image_ext).shape
            image_area = float(image_shape[0] * image_shape[1])
            wide = self._wide_offset_asterism_match(
                det_points_arr=det_points_arr,
                det_dn=det_dn,
                cat_points_arr=cat_points_arr,
                ranked_catalog=ranked_catalog,
                context=context,
                image_area=image_area,
            )
            if wide is None:
                # A field can still carry exactly one confidently-detectable
                # star at a large offset (the fainter members below the
                # glare-elevated detection limit).  A one-star lock has no
                # corroborating inlier, so the anchor pairing itself must be
                # unambiguous: both the catalog anchor and the detection
                # anchor uniquely bright by a large margin.
                wide = self._wide_offset_one_star_match(
                    det_points_arr=det_points_arr,
                    det_dn=det_dn,
                    cat_points_arr=cat_points_arr,
                    ranked_catalog=ranked_catalog,
                    context=context,
                )
            if wide is None:
                n_inliers = 0 if best is None else best[0]
                return self._fail(
                    features=features,
                    reason=(
                        f'too_few_inliers ({n_inliers} < min {self._min_inliers}) and no '
                        f'wide-offset asterism lock cleared the false-lock budget'
                    ),
                    diagnostics=StarFieldDiagnostics(
                        n_inliers=n_inliers,
                        median_residual_px=0.0,
                        n_detected_sources=n_detected_sources,
                        n_catalog_predicted=n_catalog_predicted,
                        n_triplets_evaluated=n_triplets_evaluated,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            best = (wide.n_inliers, wide.correspondences, wide.offset)
            wide_offset_lock = True
            one_star_lock = wide.is_one_star
            wide_offset_expectation = wide.expectation
        n_inliers, correspondences, _coarse_offset = best
        det_inliers: NDArrayFloatType = det_points_arr[[d for d, _ in correspondences]]
        cat_inliers = cat_points_arr[[c for _, c in correspondences]]
        if self._psf_refine_enabled:
            det_inliers = self._psf_refine_positions(det_inliers, context)
        fit_rotation = bool(context.fit_camera_rotation)
        if fit_rotation:
            sim_fit, weights = self._similarity_refit(det_inliers, cat_inliers)
            offset_vu = sim_fit.translation_vu
            residuals = sim_fit.residuals_vu
            rotation_rad: float | None = float(sim_fit.rotation_rad)
        else:
            offset_vu, weights, residuals = self._tukey_refit(det_inliers, cat_inliers)
            rotation_rad = None
        residual_distances = np.hypot(residuals[:, 0], residuals[:, 1])
        median_residual_px = float(np.median(residual_distances))
        if fit_rotation:
            cov = self._build_covariance_3dof(
                weights=weights,
                residuals=residuals,
                cat_inliers=cat_inliers,
            )
        else:
            cov = self._build_covariance(weights=weights, residuals=residuals)
        # Below the roll/translation separability floor the fitted rotation
        # is not distinguishable from a pure translation (see the camera-roll
        # separability analysis in docs/simulator_report): the matcher
        # collapses a sub-floor planted roll toward a spurious zero.  Report
        # the rotation as unobservable (sentinel variance + diagnostics
        # flag) so no downstream consumer reads a confident near-zero roll.
        rotation_below_floor = fit_rotation and (
            rotation_rad is not None
            and abs(rotation_rad) < math.radians(self._rotation_separability_floor_deg)
        )
        if rotation_below_floor:
            cov[2, 2] = ROTATION_UNOBSERVABLE_VARIANCE
            self.logger.info(
                'Fitted rotation %.4f deg is below the %.2f deg separability floor; '
                'reporting rotation as unobservable',
                math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                self._rotation_separability_floor_deg,
            )
        diagnostics = StarFieldDiagnostics(
            n_inliers=n_inliers,
            median_residual_px=median_residual_px,
            n_detected_sources=n_detected_sources,
            n_catalog_predicted=n_catalog_predicted,
            n_triplets_evaluated=n_triplets_evaluated,
            rotation_below_separability_floor=rotation_below_floor,
            wide_offset_lock=wide_offset_lock,
            wide_offset_false_lock_expectation=wide_offset_expectation,
        )
        margin_v, margin_u = search_window_for_obs(context)
        max_rotation_rad = math.radians(context.max_rotation_deg)
        rotation_at_edge = fit_rotation and (
            rotation_rad is not None
            and abs(rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
        )
        at_edge = (
            abs(offset_vu[0]) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_vu[1]) >= margin_u - self._at_edge_tolerance_px
            or rotation_at_edge
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics, at_edge=at_edge, spurious=False
        )
        if wide_offset_lock:
            # A wide-offset lock rests on fewer corroborating stars than a
            # strong-tier pattern match, so its confidence is capped below
            # the strong-tier ceiling regardless of how the formula scores
            # the inlier count; it must still clear the ensemble gate to
            # navigate, but it cannot claim a high-tier result on its own.
            # A one-star lock has no corroborating inlier at all and is
            # capped lower still.
            cap = (
                self._wide_offset_one_star_confidence_cap
                if one_star_lock
                else self._wide_offset_confidence_cap
            )
            capped = min(confidence, cap)
            if capped < confidence:
                self.logger.info(
                    'Wide-offset lock: confidence %.4f capped to %.4f',
                    confidence,
                    capped,
                )
            confidence = capped
        consumed_ids = tuple(ranked_catalog[c].feature_id for _, c in correspondences)
        self.logger.info(
            'Pattern-match offset (%.4f, %.4f) px; %d inlier(s); median residual %.4f px; '
            'confidence %.4f%s',
            offset_vu[0],
            offset_vu[1],
            n_inliers,
            median_residual_px,
            confidence,
            ' [wide-offset asterism lock]' if wide_offset_lock else '',
        )
        sigma_rotation_rad: float | None
        if fit_rotation:
            sigma_rotation_rad = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
            self.logger.info(
                'Rotation = %+.4f deg (sigma %.4f deg)%s',
                math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                math.degrees(sigma_rotation_rad),
                ', AT_EDGE' if (rotation_at_edge or at_edge) else '',
            )
        else:
            sigma_rotation_rad = None
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=consumed_ids,
            offset_px=(offset_vu[0], offset_vu[1]),
            covariance_px2=cov,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=rotation_rad,
            sigma_rotation_rad=sigma_rotation_rad,
        )

    def _enumerate_candidates(
        self,
        det_triplets: list[_Triplet],
        cat_triplets: list[_Triplet],
    ) -> list[tuple[float, int, int, int, int, int, int, int]]:
        """Return surviving ``(dist_sq, sorted_det_idx..., a, b, c, cat_idx)`` candidates.

        Each detection triplet is matched against every catalog triplet
        whose hash-distance falls within ``hash_match_tolerance``.  The
        candidate list is sorted by ``(hash_dist_sq, sorted
        detection-source indices ascending, catalog-triplet index)`` —
        per plans/archive/AUTONAV_PLAN_2026-06-19.md §33 the sorted-ascending detection tuple
        ``(min, mid, max)`` is the canonical tie-breaker, not the
        geometric apex-first ``(a, b, c)`` order, because the
        sorted-ascending key is invariant under the vertex
        canonicalisation and yields bit-identical iteration across two
        back-to-back invocations on the same obs (Cardinal Principle 3).
        """
        tol_sq = self._hash_match_tolerance * self._hash_match_tolerance
        out: list[tuple[float, int, int, int, int, int, int, int]] = []
        for det in det_triplets:
            sorted_indices = sorted([det.idx_a, det.idx_b, det.idx_c])
            sort_lo, sort_mid, sort_hi = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            for ci, cat in enumerate(cat_triplets):
                dist_sq = _hash_distance_sq(
                    det.hash_v,
                    cat.hash_v,
                    ratio_weight=self._hash_ratio_weight,
                    angle_weight=self._hash_angle_weight,
                )
                if dist_sq <= tol_sq:
                    out.append(
                        (
                            dist_sq,
                            sort_lo,
                            sort_mid,
                            sort_hi,
                            det.idx_a,
                            det.idx_b,
                            det.idx_c,
                            ci,
                        )
                    )
        out.sort()
        return out

    def _score_candidates(
        self,
        *,
        candidates: list[tuple[float, int, int, int, int, int, int, int]],
        cat_triplets: list[_Triplet],
        det_points_arr: NDArrayFloatType,
        cat_points_arr: NDArrayFloatType,
    ) -> tuple[int, list[tuple[int, int]], tuple[float, float]] | None:
        """Score every candidate; return the (n_inliers, correspondences, offset) winner.

        Each (det_triplet, cat_triplet) candidate proposes the
        translation that maps the catalog-triplet centroid onto the
        detection-triplet centroid.  Inliers are counted by optimal
        one-to-one assignment under that translation.  The first
        candidate (in sorted order) that ties the best inlier count
        wins, which matches the deterministic-iteration contract.
        """
        best: tuple[int, list[tuple[int, int]], tuple[float, float]] | None = None
        for _dist_sq, _lo, _mid, _hi, det_a, det_b, det_c, cat_pos in candidates:
            cat = cat_triplets[cat_pos]
            offset = _triplet_centroid_offset(
                det_points_arr,
                cat_points_arr,
                det_indices=(det_a, det_b, det_c),
                cat_indices=(cat.idx_a, cat.idx_b, cat.idx_c),
            )
            n_inliers, pairs = _optimal_inlier_assignment(
                det_points_arr,
                cat_points_arr,
                offset_vu=offset,
                tolerance_px=self._inlier_tolerance_px,
            )
            if best is None or n_inliers > best[0]:
                best = (n_inliers, pairs, offset)
        return best

    def _wide_offset_asterism_match(
        self,
        *,
        det_points_arr: NDArrayFloatType,
        det_dn: NDArrayFloatType,
        cat_points_arr: NDArrayFloatType,
        ranked_catalog: list[NavFeature],
        context: NavContext,
        image_area: float,
    ) -> _WideOffsetMatch | None:
        """Attempt a wide-offset lock from a few trusted bright-anchor seeds.

        The triplet RANSAC needs six inliers because it evaluates thousands
        of triplet seeds, so a three-inlier coincidence is reachable by
        chance.  This fallback instead seeds a translation only from
        ``brightest catalog star x brightest detection`` pairings: a handful
        of trials, each pinned to a genuinely bright, ``#284``-trusted anchor
        (a star whose corrected magnitude is not a saturated lower bound).
        With the trial count collapsed, an inlier count as low as
        ``wide_offset_min_inliers`` can be statistically decisive.  A
        candidate is accepted only when it clears every guard: the inlier
        floor, a tight residual RMS, at least one trusted bright anchor among
        the inliers, and -- the primary guard -- a chance-alignment
        expectation below ``wide_offset_false_lock_budget``
        (see :func:`_wide_offset_false_lock_expectation`).

        Parameters:
            det_points_arr: ``(N_det, 2)`` detection ``(v, u)`` positions.
            det_dn: ``(N_det,)`` detection matched-filter peak amplitudes
                (brightness ranking for the detection anchors).
            cat_points_arr: ``(N_cat, 2)`` predicted catalog ``(v, u)``
                positions, brightest first.
            ranked_catalog: The catalog features parallel to
                ``cat_points_arr`` (supplies the ``#284`` photometry flags).
            context: Per-image NavContext (search-window bounds).
            image_area: Image area in square pixels (detection-density
                denominator for the false-lock model).

        Returns:
            A :class:`_WideOffsetMatch` when a lock cleared every guard, else
            ``None``.
        """
        if not self._wide_offset_enabled:
            return None
        n_det = int(det_points_arr.shape[0])
        n_cat = int(cat_points_arr.shape[0])
        if n_det < self._wide_offset_min_inliers or n_cat < self._wide_offset_min_inliers:
            return None
        # Trusted bright catalog anchors: the brightest stars whose corrected
        # magnitude is trustworthy (#284 photometry_saturated == False), so a
        # bright-pair seed rests on a real brightness ordering rather than a
        # saturated lower bound.  ``ranked_catalog`` is sorted by predicted
        # SNR descending, so the first survivors are the brightest.
        anchor_cat = [j for j in range(n_cat) if not _star_photometry_saturated(ranked_catalog[j])][
            : self._wide_offset_anchor_catalog
        ]
        if not anchor_cat:
            self.logger.debug('Wide-offset: no photometry-trusted bright catalog anchor; skipping')
            return None
        anchor_cat_set = set(anchor_cat)
        # ``detected`` reaches this method already sorted by
        # ``(-peak_dn, v, u)``, so the brightest detections are the leading
        # rows; a stable argsort preserves that ``(v, u)`` tie-break and is
        # reproducible across numpy versions (Cardinal Principle 3).
        anchor_det = [
            int(i)
            for i in np.argsort(-det_dn, kind='stable')[: self._wide_offset_anchor_detections]
        ]
        margin_v, margin_u = search_window_for_obs(context)
        seeds: list[tuple[float, float]] = []
        for j in anchor_cat:
            for i in anchor_det:
                off = (
                    float(det_points_arr[i, 0] - cat_points_arr[j, 0]),
                    float(det_points_arr[i, 1] - cat_points_arr[j, 1]),
                )
                if abs(off[0]) <= margin_v and abs(off[1]) <= margin_u:
                    seeds.append(off)
        if not seeds:
            return None
        best: tuple[int, list[tuple[int, int]], tuple[float, float], float] | None = None
        for off in seeds:
            n_inliers, pairs = _optimal_inlier_assignment(
                det_points_arr,
                cat_points_arr,
                offset_vu=off,
                tolerance_px=self._inlier_tolerance_px,
            )
            if n_inliers < self._wide_offset_min_inliers:
                continue
            resid = det_points_arr[[d for d, _ in pairs]] - (
                cat_points_arr[[c for _, c in pairs]] + np.asarray(off, np.float64)[None, :]
            )
            rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
            if best is None or n_inliers > best[0] or (n_inliers == best[0] and rms < best[3]):
                best = (n_inliers, pairs, off, rms)
        if best is None:
            return None
        n_best, pairs, off, rms = best
        if rms > self._wide_offset_max_residual_rms_px:
            self.logger.info(
                'Wide-offset: best %d-inlier lock residual RMS %.3f px exceeds '
                'max %.3f px; rejecting',
                n_best,
                rms,
                self._wide_offset_max_residual_rms_px,
            )
            return None
        if not anchor_cat_set & {c for _, c in pairs}:
            self.logger.info(
                'Wide-offset: best %d-inlier lock has no trusted bright anchor among '
                'its inliers; rejecting',
                n_best,
            )
            return None
        n_unique_seeds = len({(round(o[0], 1), round(o[1], 1)) for o in seeds})
        expectation = _wide_offset_false_lock_expectation(
            n_inliers=n_best,
            n_seeds=n_unique_seeds,
            n_catalog=n_cat,
            n_detected=n_det,
            tolerance_px=self._inlier_tolerance_px,
            image_area_px2=image_area,
        )
        if expectation > self._wide_offset_false_lock_budget:
            self.logger.info(
                'Wide-offset: %d-inlier lock false-lock expectation %.3e exceeds '
                'budget %.3e; rejecting',
                n_best,
                expectation,
                self._wide_offset_false_lock_budget,
            )
            return None
        self.logger.info(
            'Wide-offset asterism lock: %d inlier(s), residual RMS %.3f px, false-lock '
            'expectation %.3e over %d bright-anchor seed(s)',
            n_best,
            rms,
            expectation,
            n_unique_seeds,
        )
        return _WideOffsetMatch(
            n_inliers=n_best,
            correspondences=pairs,
            offset=off,
            expectation=expectation,
            n_seeds=n_unique_seeds,
            is_one_star=False,
        )

    def _wide_offset_one_star_match(
        self,
        *,
        det_points_arr: NDArrayFloatType,
        det_dn: NDArrayFloatType,
        cat_points_arr: NDArrayFloatType,
        ranked_catalog: list[NavFeature],
        context: NavContext,
    ) -> _WideOffsetMatch | None:
        """Attempt a wide-offset lock from a single uniquely-bright anchor pair.

        The last resort when neither the triplet RANSAC nor the
        ``>= wide_offset_min_inliers`` asterism fallback locks and a field
        carries exactly one confidently-detectable star at a large offset.
        A one-star lock has no corroborating inlier, so the chance-alignment
        budget cannot vouch for it; the anchor pairing itself must instead be
        unambiguous.  That holds only when the single trusted bright catalog
        anchor outshines the next predictable star by
        ``wide_offset_one_star_catalog_margin_mag`` AND the brightest
        detection outshines the next by a peak-DN factor of
        ``wide_offset_one_star_detection_margin_ratio``.  A glare field
        routinely raises peaks brighter than a real faint star, so a frame
        whose brightest detection is not uniquely dominant fails the
        detection-margin gate and stays unlocked -- the correct verdict on a
        field with two comparably bright peaks, where pairing the anchor with
        the brightest peak would be a guess.

        Parameters:
            det_points_arr: ``(N_det, 2)`` detection ``(v, u)`` positions.
            det_dn: ``(N_det,)`` detection matched-filter peak amplitudes.
            cat_points_arr: ``(N_cat, 2)`` predicted catalog ``(v, u)``
                positions, brightest first.
            ranked_catalog: The catalog features parallel to
                ``cat_points_arr`` (supplies the photometry flags).
            context: Per-image NavContext (search-window bounds).

        Returns:
            A single-inlier :class:`_WideOffsetMatch` when the anchor pairing
            is unambiguous and inside the search window, else ``None``.
        """
        if not self._wide_offset_one_star_enabled:
            return None
        n_det = int(det_points_arr.shape[0])
        n_cat = int(cat_points_arr.shape[0])
        if n_det < 1 or n_cat < 1:
            return None
        # The catalog anchor is the brightest predictable star.  It must be
        # photometry-trusted (its brightness ordering is real, not a saturated
        # lower bound) and uniquely bright over the next predictable star.
        if _star_photometry_saturated(ranked_catalog[0]):
            self.logger.debug('One-star: brightest catalog anchor is photometry-saturated')
            return None
        next_cat_snr = predicted_snr(ranked_catalog[1]) if n_cat > 1 else 0.0
        cat_margin_mag = brightness_margin_mag(predicted_snr(ranked_catalog[0]), next_cat_snr)
        if cat_margin_mag < self._wide_offset_one_star_catalog_margin_mag:
            self.logger.debug(
                'One-star: catalog anchor margin %.3f mag below floor %.3f; skipping',
                cat_margin_mag,
                self._wide_offset_one_star_catalog_margin_mag,
            )
            return None
        # The detection anchor is the brightest peak; it must be uniquely
        # bright over the next peak so the single pairing is not a guess.
        order = np.argsort(-det_dn, kind='stable')
        det_i = int(order[0])
        dn_top = float(det_dn[det_i])
        dn_next = float(det_dn[int(order[1])]) if n_det > 1 else 0.0
        det_ratio = math.inf if dn_next <= 0.0 else dn_top / dn_next
        if det_ratio < self._wide_offset_one_star_detection_margin_ratio:
            self.logger.info(
                'One-star: brightest detection peak ratio %.3f below floor %.3f; '
                'rejecting (not uniquely dominant)',
                det_ratio,
                self._wide_offset_one_star_detection_margin_ratio,
            )
            return None
        off = (
            float(det_points_arr[det_i, 0] - cat_points_arr[0, 0]),
            float(det_points_arr[det_i, 1] - cat_points_arr[0, 1]),
        )
        margin_v, margin_u = search_window_for_obs(context)
        if abs(off[0]) > margin_v or abs(off[1]) > margin_u:
            self.logger.debug('One-star: anchor offset outside search window; skipping')
            return None
        self.logger.info(
            'One-star wide-offset lock: offset (%.4f, %.4f) px; catalog margin '
            '%.3f mag; detection peak ratio %.3f',
            off[0],
            off[1],
            cat_margin_mag,
            det_ratio,
        )
        return _WideOffsetMatch(
            n_inliers=1,
            correspondences=[(det_i, 0)],
            offset=off,
            expectation=0.0,
            n_seeds=1,
            is_one_star=True,
        )

    def _tukey_refit(
        self,
        det_inliers: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> tuple[tuple[float, float], NDArrayFloatType, NDArrayFloatType]:
        """Refit translation by Tukey-biweight-reweighted least squares.

        Two-pass: (i) unweighted mean to seed residual scale; (ii) one
        biweight-reweighted mean at the conventional 4.685 breakdown
        constant.  Returns the final ``(offset, weights, residuals)``
        triple where ``residuals`` is the per-correspondence
        ``(d - (c + offset))`` array.
        """
        n = det_inliers.shape[0]
        weights0 = np.ones(n, dtype=np.float64)
        offset0 = _solve_translation(det_inliers, cat_inliers, weights0)
        residuals0 = det_inliers - cat_inliers - np.asarray(offset0, np.float64)[None, :]
        distances0 = np.hypot(residuals0[:, 0], residuals0[:, 1])
        scale = float(np.median(distances0)) if distances0.size > 0 else 0.0
        if scale <= 0.0:
            # Perfect fit — no scatter — biweight weights are unity by
            # construction (every residual is zero).
            return offset0, weights0, residuals0
        scaled = distances0 / scale
        weights = tukey_biweight_weights(scaled, c=DEFAULT_TUKEY_C)
        if float(weights.sum()) <= 0.0:
            return offset0, weights0, residuals0
        offset = _solve_translation(det_inliers, cat_inliers, weights)
        residuals = det_inliers - cat_inliers - np.asarray(offset, np.float64)[None, :]
        return offset, weights, residuals

    def _similarity_refit(
        self,
        det_inliers: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> tuple[SimilarityFit, NDArrayFloatType]:
        """Run a Tukey-biweight-reweighted Procrustes / similarity fit.

        Two-pass: an unweighted Kabsch fit seeds the residual scale; a
        biweight-reweighted Kabsch fit at the conventional 4.685
        breakdown constant produces the final ``(rotation, translation)``
        pair.  Falls back to the unweighted fit when every residual
        survives at full weight (a perfect fit) or when the biweight
        weights total zero.

        Returns:
            ``(SimilarityFit, weights)`` — the converged fit plus the
            per-correspondence weights used in the final iteration.
        """
        n = det_inliers.shape[0]
        weights0 = np.ones(n, dtype=np.float64)
        seed = similarity_transform_fit(det_inliers, cat_inliers, weights0)
        distances0 = np.hypot(seed.residuals_vu[:, 0], seed.residuals_vu[:, 1])
        scale = float(np.median(distances0)) if distances0.size > 0 else 0.0
        if scale <= 0.0:
            return seed, weights0
        scaled = distances0 / scale
        weights = tukey_biweight_weights(scaled, c=DEFAULT_TUKEY_C)
        if float(weights.sum()) <= 0.0:
            return seed, weights0
        refined = similarity_transform_fit(det_inliers, cat_inliers, weights)
        return refined, weights

    def _build_covariance(
        self, *, weights: NDArrayFloatType, residuals: NDArrayFloatType, n_params: int = 2
    ) -> NDArrayFloatType:
        """Return the per-axis covariance of the translation estimate.

        Reduced-chi-square weighted-mean form.  With ``N`` weighted
        residuals and ``p = n_params`` fitted parameters (2 for the
        translation-only fit, 3 when an outer rotation is co-fitted),
        the per-axis reduced chi-square is

        ::

            chi2_nu_axis = sum_i w_i * r_axis_i**2 / max(N - p, 1)

        and the weighted-mean variance is ``chi2_nu_axis / sum(w_i)``.
        The ``max(N - p, 1)`` degrees-of-freedom factor inflates the
        variance when fewer points than parameters constrain the fit
        (an under-determined geometry cannot be reported as confident).
        A positive-definite floor of ``1 / sum(w_i)`` (pure
        inverse-precision) keeps the covariance non-degenerate in the
        noise-free case where every residual is zero.  Finally the
        uncalibrated ``model_error_floor_px**2`` is added to the
        diagonal (a no-op at the default 0.0).
        """
        total = float(weights.sum())
        if total <= 0.0:
            return np.eye(2, dtype=np.float64)
        n = residuals.shape[0]
        dof = max(n - n_params, 1)
        chi2_nu_v = float(np.sum(weights * residuals[:, 0] ** 2)) / dof
        chi2_nu_u = float(np.sum(weights * residuals[:, 1] ** 2)) / dof
        floor = 1.0 / total
        model_error = self._model_error_floor_px * self._model_error_floor_px
        cov_v = max(chi2_nu_v / total, floor) + model_error
        cov_u = max(chi2_nu_u / total, floor) + model_error
        return np.diag([cov_v, cov_u]).astype(np.float64)

    def _build_covariance_3dof(
        self,
        *,
        weights: NDArrayFloatType,
        residuals: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Return the 3x3 covariance for the similarity-transform fit.

        Translation block follows :meth:`_build_covariance` with ``p =
        3`` (three parameters are co-fitted: dv, du, theta); the
        rotation diagonal is the inverse of the rotation Fisher
        information built from the per-vertex Jacobian of the residual
        with respect to theta.  For an inlier with lever arm
        ``(dv_i, du_i) = cat_i - cc`` about the (weighted) catalog
        centroid ``cc``, that Jacobian is the tangential lever-arm
        component ``(du_i, -dv_i)``, so with per-axis residual
        variances the information is

        ::

            I_theta = sum_i w_i * (du_i**2 / var_v + dv_i**2 / var_u)
            sigma_theta**2 = 1 / I_theta

        where each per-axis variance is the reduced chi-square of the
        fit residuals on that axis,

        ::

            var_axis = max(sum_i w_i r_axis_i**2 / max(N - 3, 1), 1.0)

        floored at 1 px**2 — the same positive-definite floor the
        translation block applies — so a noise-free fit cannot report
        a degenerate rotation variance.  This is exact for anisotropic
        residuals (``var_v != var_u``); in the isotropic limit it
        reduces to the classic lever-arm form
        ``chi2_nu_residual / sum_i w_i * |cat_i - cc|**2``.
        Cross-terms are zero — with the rotation taken about the
        weighted catalog centroid the translation and rotation
        parameters of a Procrustes fit are uncorrelated.  Whenever the
        catalog spread is too small to constrain rotation (numerically
        zero information, e.g. coincident inliers) the rotation
        diagonal collapses to the rotation-unobservable sentinel so
        ``pinvh`` cleanly drops the rotation contribution.
        """
        cov_2x2 = self._build_covariance(weights=weights, residuals=residuals, n_params=3)
        total = float(weights.sum())
        if total <= 0.0:
            return embed_rotation_unobservable(cov_2x2)
        cat_c_v = float(np.sum(weights * cat_inliers[:, 0]) / total)
        cat_c_u = float(np.sum(weights * cat_inliers[:, 1]) / total)
        dv = cat_inliers[:, 0] - cat_c_v
        du = cat_inliers[:, 1] - cat_c_u
        n = residuals.shape[0]
        dof = max(n - 3, 1)
        var_v = max(float(np.sum(weights * residuals[:, 0] ** 2)) / dof, 1.0)
        var_u = max(float(np.sum(weights * residuals[:, 1] ** 2)) / dof, 1.0)
        fisher_theta = float(np.sum(weights * (du * du / var_v + dv * dv / var_u)))
        if fisher_theta <= 0.0:
            sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
        else:
            sigma_theta_sq = 1.0 / fisher_theta
        cov = np.zeros((3, 3), dtype=np.float64)
        cov[:2, :2] = cov_2x2
        cov[2, 2] = sigma_theta_sq
        return cov

    def _evaluate_confidence(
        self,
        *,
        diagnostics: StarFieldDiagnostics,
        at_edge: bool,
        spurious: bool,
    ) -> float:
        """Run the YAML confidence formula and emit a per-term breakdown."""
        assert self.confidence_spec is not None
        confidence, breakdown = evaluate_sigmoid_combination(
            self.confidence_spec,
            _StarFieldConfidenceContext(
                at_edge=at_edge, spurious=spurious, diagnostics=diagnostics
            ),
            technique_name=self.name,
            return_breakdown=True,
        )
        log_confidence_breakdown(self.logger, breakdown)
        return float(confidence)

    def _fail(
        self,
        *,
        features: list[NavFeature],
        reason: str,
        diagnostics: StarFieldDiagnostics,
        fit_rotation: bool = False,
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result with the supplied reason."""
        self.logger.info('Reporting spurious result: %s', reason)
        return self._spurious_result(
            feature_ids=tuple(f.feature_id for f in features),
            diagnostics=diagnostics,
            fit_rotation=fit_rotation,
        )
