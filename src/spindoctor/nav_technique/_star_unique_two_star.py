"""Pair geometry and covariances for ``StarUniqueMatchNav``'s two-star path.

The two-star path of :class:`~spindoctor.nav_technique.nav_technique_star_unique_match.\
StarUniqueMatchNav` detects two catalog stars and then solves for the offset that
carries the predictions onto the detections.  The pure arithmetic of that solve lives
here: choosing the detection-to-prediction assignment with the smaller residual (with
or without a rotation), and building the translation covariance a match reports and
the three-parameter covariance of a pair fitted with a rotation.  So does the photometry
that says how bright another star found at the pair's offset should be.  The detection,
the gates, the confidence, and the logging stay with the technique.

The submodule is private (leading underscore on the file name); its functions carry no
underscore prefix because the technique module imports them, but they are not part of
the package's public API.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.flags import StarFlags
from spindoctor.nav_technique._star_helpers import (
    SAME_SOURCE_PX,
    SimilarityFit,
    predicted_snr,
    predicted_vu,
    similarity_transform_fit,
)
from spindoctor.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'PairChoice',
    'PairHypothesis',
    'TwoStarAssignment',
    'absent_stars',
    'build_star_covariance',
    'choose_pair',
    'pair_brightness',
    'pair_hypotheses',
    'peak_per_snr',
    'photometry_saturated',
    'refute_pair',
    'similarity_fit_assignment',
    'solve_assigned_pair',
    'source_peak',
    'star_like_source',
    'third_star_votes',
    'two_star_covariance',
    'two_star_covariance_3dof',
]


@dataclass(frozen=True)
class TwoStarAssignment:
    """Two detections solved against the two stars they are assigned to.

    Parameters:
        offset_vu: ``(dv, du)`` offset carrying the predictions onto the detections.
        residual_px: Residual of the assignment (the centroid-relative misfit of the
            first detection), the quantity the pair's cross-check bounds.
        rotation_rad: Fitted rotation in radians when a rotation was fitted, else
            ``None``.
    """

    offset_vu: tuple[float, float]
    residual_px: float
    rotation_rad: float | None

    def place(self, predicted_vu_pos: tuple[float, float]) -> tuple[float, float]:
        """Return where the solution puts a star predicted at ``predicted_vu_pos``.

        Without a rotation that is the prediction moved by the offset.  With one it is the
        Procrustes fit's rotation of the prediction plus its translation, so a star far
        from the pair is placed where the rotation carries it.

        Parameters:
            predicted_vu_pos: The star's predicted ``(v, u)``.

        Returns:
            The star's ``(v, u)`` under the solution.
        """
        pv, pu = predicted_vu_pos
        if self.rotation_rad is None:
            return pv + self.offset_vu[0], pu + self.offset_vu[1]
        cos_t, sin_t = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        return (
            cos_t * pv - sin_t * pu + self.offset_vu[0],
            sin_t * pv + cos_t * pu + self.offset_vu[1],
        )


@dataclass(frozen=True)
class PairHypothesis:
    """Two sources, each in the window of a different star, that cross-check each other.

    Parameters:
        star_indices: Indices of the two stars among the searched stars.
        ranks: Rank of each source among its star's sources, strongest first.
        detections: The two sources, in the order of ``star_indices``.
        solved: The pair solved with each source assigned to its own star.
        support: Number of searched stars, the pair's own two among them, with a source
            within the agreement tolerance of where the pair's solution puts them.
        spread_px: Sum, over those stars, of the distance from where the pair's solution
            puts the star to its nearest source.
    """

    star_indices: tuple[int, int]
    ranks: tuple[int, int]
    detections: tuple[tuple[float, float], tuple[float, float]]
    solved: TwoStarAssignment
    support: int
    spread_px: float


def pair_hypotheses(
    predictions: list[tuple[float, float]],
    sources: list[list[tuple[float, float]]],
    *,
    max_residual_px: float,
    fit_rotation: bool,
) -> list[PairHypothesis]:
    """Return every cross-checked pair of sources, the best supported first.

    For every pair of stars, every source in the first star's window is tried against
    every source in the second star's, each assigned to its own star; two sources less
    than ``SAME_SOURCE_PX`` apart are one source and are not paired.  A pair passes when
    its residual is within ``max_residual_px``.  Its support counts the stars, among all
    those searched, with a source within twice ``max_residual_px`` of where the pair's
    solution puts them (:meth:`TwoStarAssignment.place`), so the true offset gathers every
    star whose source was found while a coincidence between two unrelated sources gathers
    few.  Pairs are ordered by support, most first; then by spread, so that of the pairs a
    set of agreeing stars forms, the one nearest all of them comes first; then by the
    sources' ranks and the stars' order.

    Parameters:
        predictions: Each searched star's predicted ``(v, u)``, brightest first.
        sources: Each searched star's sources, strongest first.
        max_residual_px: Largest residual a pair may have.
        fit_rotation: Whether to fit a rotation as well as a translation.

    Returns:
        The passing pairs, best first; empty when none passes.
    """
    tolerance_px = 2.0 * max_residual_px
    found: list[PairHypothesis] = []
    for i, j in itertools.combinations(range(len(predictions)), 2):
        for rank_a, det_a in enumerate(sources[i]):
            for rank_b, det_b in enumerate(sources[j]):
                if math.hypot(det_a[0] - det_b[0], det_a[1] - det_b[1]) < SAME_SOURCE_PX:
                    continue
                solved = solve_assigned_pair(
                    det_a, det_b, predictions[i], predictions[j], fit_rotation=fit_rotation
                )
                if solved.residual_px > max_residual_px:
                    continue
                support = 0
                spread = 0.0
                for pred, star_sources in zip(predictions, sources, strict=True):
                    tv, tu = solved.place(pred)
                    nearest = min(
                        (math.hypot(d[0] - tv, d[1] - tu) for d in star_sources),
                        default=math.inf,
                    )
                    if nearest <= tolerance_px:
                        support += 1
                        spread += nearest
                found.append(
                    PairHypothesis(
                        star_indices=(i, j),
                        ranks=(rank_a, rank_b),
                        detections=(det_a, det_b),
                        solved=solved,
                        support=support,
                        spread_px=spread,
                    )
                )
    found.sort(key=lambda h: (-h.support, h.spread_px, sum(h.ranks), h.star_indices))
    return found


def build_star_covariance(feature: NavFeature, *, residual_px: float) -> NDArrayFloatType:
    """Return a 2x2 covariance for a star match.

    Uses the per-feature anisotropic CRLB covariance carried on the
    ``NavFeature.position_cov_px`` (computed from predicted SNR + smear
    by ``NavModelStars``) as the floor and inflates it by
    ``residual_px**2`` so a noisy match reports its scatter
    rather than the noise-free CRLB lower bound.

    The same inflation applies in both 1-star and 2-star modes: the
    1-star path's ``residual_px`` is the prediction-to-detection
    distance (no second observation to subtract), and the 2-star path's
    ``residual_px`` is already a per-axis RMS, so squaring either
    yields the right per-axis variance addend.

    Parameters:
        feature: The star feature whose CRLB cov is the floor.
        residual_px: Final per-axis residual magnitude.

    Returns:
        2x2 float64 covariance matrix; the identity stands in for a missing floor.
    """
    floor = feature.position_cov_px
    base = (
        np.eye(2, dtype=np.float64) if floor is None else np.asarray(floor, dtype=np.float64).copy()
    )
    inflation = residual_px * residual_px
    return base + inflation * np.eye(2, dtype=np.float64)


def solve_assigned_pair(
    det_a: tuple[float, float],
    det_b: tuple[float, float],
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
    *,
    fit_rotation: bool,
) -> TwoStarAssignment:
    """Solve the assignment of ``det_a`` to the first star and ``det_b`` to the second.

    The offset is the mean of the two detection-minus-prediction differences, or the
    translation of a Procrustes fit when a rotation is fitted; the residual is half the
    distance between the two differences either way.

    Parameters:
        det_a: Detection of the first star ``(v, u)``.
        det_b: Detection of the second star ``(v, u)``.
        pred_a: Prediction of the first star ``(v, u)``.
        pred_b: Prediction of the second star ``(v, u)``.
        fit_rotation: Whether to fit a rotation as well as a translation.

    Returns:
        The pair's offset, residual, and fitted rotation.
    """
    if fit_rotation:
        det_arr = np.asarray([det_a, det_b], dtype=np.float64)
        fit, residual_px = similarity_fit_assignment(det_arr, pred_a, pred_b)
        return TwoStarAssignment(
            offset_vu=(float(fit.translation_vu[0]), float(fit.translation_vu[1])),
            residual_px=residual_px,
            rotation_rad=float(fit.rotation_rad),
        )
    offset_v = 0.5 * ((det_a[0] - pred_a[0]) + (det_b[0] - pred_b[0]))
    offset_u = 0.5 * ((det_a[1] - pred_a[1]) + (det_b[1] - pred_b[1]))
    residual_px = math.hypot((det_a[0] - pred_a[0]) - offset_v, (det_a[1] - pred_a[1]) - offset_u)
    return TwoStarAssignment(
        offset_vu=(offset_v, offset_u),
        residual_px=residual_px,
        rotation_rad=None,
    )


def similarity_fit_assignment(
    det_arr: NDArrayFloatType,
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
) -> tuple[SimilarityFit, float]:
    """Run a Procrustes fit of two detections to two predictions and report its residual.

    Parameters:
        det_arr: Detection pair ``[det_a, det_b]`` shaped ``(2, 2)``, ``det_a`` matched
            to ``pred_a`` and ``det_b`` to ``pred_b``.
        pred_a: Catalog prediction for the first star.
        pred_b: Catalog prediction for the second star.

    Returns:
        ``(SimilarityFit, residual_px)``.  The residuals of a two-point rigid fit are zero
        by construction, so the residual reported is the one the translation-only solve
        uses, half the distance between the two detection-minus-prediction differences,
        and a pair is judged the same way with and without a rotation.
    """
    cat_arr = np.asarray([pred_a, pred_b], dtype=np.float64)
    fit = similarity_transform_fit(det_arr, cat_arr, np.ones(2, dtype=np.float64))
    offset = np.mean(det_arr - cat_arr, axis=0)
    residual = math.hypot(
        (float(det_arr[0][0]) - pred_a[0]) - float(offset[0]),
        (float(det_arr[0][1]) - pred_a[1]) - float(offset[1]),
    )
    return fit, residual


def two_star_covariance(
    star_a: NavFeature,
    star_b: NavFeature,
    *,
    residual_px: float,
    fit_rotation: bool,
) -> NDArrayFloatType:
    """Return a two-star match's covariance.

    The translation block averages the two stars' covariances from
    :func:`build_star_covariance` as a coarse two-star floor (symmetrized).  When a
    rotation was fitted the result is the 3x3 of :func:`two_star_covariance_3dof`.

    Parameters:
        star_a: The first matched star.
        star_b: The second matched star.
        residual_px: The pair's residual, inflating each star's floor.
        fit_rotation: Whether a rotation was fitted.

    Returns:
        A 2x2 covariance, or a 3x3 one when ``fit_rotation`` is True.
    """
    cov_a = build_star_covariance(star_a, residual_px=residual_px)
    cov_b = build_star_covariance(star_b, residual_px=residual_px)
    cov_2x2 = 0.5 * (cov_a + cov_b)
    cov_2x2 = 0.5 * (cov_2x2 + cov_2x2.T)
    if not fit_rotation:
        return cov_2x2
    return two_star_covariance_3dof(
        cov_2x2=cov_2x2,
        pred_a=predicted_vu(star_a),
        pred_b=predicted_vu(star_b),
    )


def two_star_covariance_3dof(
    *,
    cov_2x2: NDArrayFloatType,
    pred_a: tuple[float, float],
    pred_b: tuple[float, float],
) -> NDArrayFloatType:
    """Build the 3x3 covariance for a 2-star Procrustes fit.

    Translation block uses the per-feature CRLB-derived 2x2; the
    rotation diagonal scales the sum of per-axis position variances
    by ``1 / L**2`` where ``L`` is the catalog separation between
    the two stars.  This is the analytic small-angle uncertainty:
    a per-axis position error ``sigma_pos`` perturbs the rotation
    by approximately ``sigma_pos / L`` per star, and the two stars
    contribute independently.

    Parameters:
        cov_2x2: Per-feature CRLB-derived 2x2 translation covariance
            (already symmetrized).
        pred_a: Catalog prediction for the first chosen star.
        pred_b: Catalog prediction for the second chosen star.

    Returns:
        3x3 covariance with the input ``cov_2x2`` in the
        translation block and either the analytic
        ``2 * (sigma_v**2 + sigma_u**2) / L**2`` rotation variance
        on the diagonal, or :data:`ROTATION_UNOBSERVABLE_VARIANCE`
        when the two predictions coincide (a degenerate input the
        upstream guard already rejects).
    """
    separation_sq = (pred_a[0] - pred_b[0]) ** 2 + (pred_a[1] - pred_b[1]) ** 2
    if separation_sq <= 0.0:
        sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
    else:
        # Sum of per-axis position variances aggregates the (v, u)
        # pixel uncertainty assumed isotropic by the centroid CRLB;
        # 2 / L^2 reflects the two-point lever-arm.
        sigma_theta_sq = 2.0 * float(cov_2x2[0, 0] + cov_2x2[1, 1]) / separation_sq
    cov = np.zeros((3, 3), dtype=np.float64)
    cov[:2, :2] = cov_2x2
    cov[2, 2] = sigma_theta_sq
    return cov


def source_peak(
    image_ext: NDArrayFloatType, center_vu: tuple[float, float], *, box_half_px: int
) -> float:
    """Return a source's brightest pixel above the local background.

    The brightest pixel is taken over the ``(2 * box_half_px + 1)`` box centered on the
    pixel holding ``center_vu``, and the background is the median of the ring of pixels 1
    to 3 px outside the box.  The ring median ignores another source covering less than
    half the ring.  Stars in one frame share a point-spread function and smear, so their
    peaks scale with their catalog fluxes; a sum over the box would follow the flux more
    directly but takes in every gradient of a ring's flank.

    Parameters:
        image_ext: 2-D extfov image array.
        center_vu: ``(v, u)`` center of the source, such as a detection centroid.
        box_half_px: Half-width of the box searched for the brightest pixel.

    Returns:
        The background-subtracted peak, at least ``0.0``; ``0.0`` when the box and its
        ring do not fit inside the image.
    """
    outer = box_half_px + 3
    v_c, u_c = round(center_vu[0]), round(center_vu[1])
    rows, cols = image_ext.shape
    if not (outer <= v_c < rows - outer and outer <= u_c < cols - outer):
        return 0.0
    patch = np.asarray(
        image_ext[v_c - outer : v_c + outer + 1, u_c - outer : u_c + outer + 1], np.float64
    )
    steps = np.abs(np.arange(-outer, outer + 1))
    dist = np.maximum(steps[:, None], steps[None, :])
    background = float(np.median(patch[dist > box_half_px + 1]))
    return max(float(patch[dist <= box_half_px].max()) - background, 0.0)


def photometry_saturated(feature: NavFeature) -> bool:
    """Return whether a STAR feature's catalog magnitude is a saturated lower bound."""
    flags = feature.flags
    return isinstance(flags, StarFlags) and flags.photometry_saturated


def peak_per_snr(
    image_ext: NDArrayFloatType,
    stars: list[NavFeature],
    place: Callable[[tuple[float, float]], tuple[float, float]],
    *,
    detect_at: Callable[[NavFeature, tuple[float, float]], tuple[float, float] | None],
    box_half_px: int,
) -> float | None:
    """Return the peak per unit of predicted SNR that stars found at a solution show.

    Each star is sought with ``detect_at`` where ``place`` puts its prediction, and its
    peak above the local background (:func:`source_peak`) is divided by its predicted SNR,
    which is proportional to its catalog flux.  The result is the geometric mean of the
    quotients, the scale that converts a star's predicted SNR into the peak it should show
    in this image.

    Parameters:
        image_ext: 2-D extfov image array.
        stars: The star features whose light sets the scale.
        place: Maps a star's predicted ``(v, u)`` to where the solution puts it
            (:meth:`TwoStarAssignment.place`).
        detect_at: Returns the detection centroid of a star sought around the given
            ``(v, u)`` position, or ``None``.
        box_half_px: Half-width of the box :func:`source_peak` searches.

    Returns:
        The scale, or ``None`` when ``stars`` is empty or any star is not found, shows no
        peak, has no positive predicted SNR, or has a catalog magnitude that is only a
        saturated lower bound.
    """
    log_sum = 0.0
    for star in stars:
        snr = predicted_snr(star)
        if snr <= 0.0 or photometry_saturated(star):
            return None
        det = detect_at(star, place(predicted_vu(star)))
        if det is None:
            return None
        peak = source_peak(image_ext, det, box_half_px=box_half_px)
        if peak <= 0.0:
            return None
        log_sum += math.log(peak / snr)
    if not stars:
        return None
    return math.exp(log_sum / len(stars))


def third_star_votes(
    others: list[NavFeature],
    place: Callable[[tuple[float, float]], tuple[float, float]],
    *,
    detect: Callable[[NavFeature], tuple[float, float] | None],
    detect_near: Callable[[NavFeature, tuple[float, float]], tuple[float, float] | None],
    tolerance_px: float,
) -> tuple[int, int]:
    """Return how many other stars agree and disagree with a solution.

    Each other star is first sought with ``detect_near`` where ``place`` puts its
    prediction; a detection there within ``tolerance_px`` of that position agrees,
    whatever else the star's own search window holds.  Otherwise the star is searched for
    around its own prediction with ``detect``: a detection within ``tolerance_px`` of where
    the solution puts the star agrees, any other detection disagrees, and a star that
    yields no detection casts no vote.  A brighter source elsewhere in a star's window,
    such as a cosmic-ray hit, therefore outvotes the star only when the star is not where
    the solution puts it.  Overlapping windows can return
    one source for several stars; a disagreeing detection within ``SAME_SOURCE_PX`` of
    one already counted is that source again and is not counted twice.

    Parameters:
        others: Star features other than the two the solution rests on.
        place: Maps a star's predicted ``(v, u)`` to where the solution puts it.
        detect: Returns the detection centroid for a star searched around its own
            prediction, or ``None``.
        detect_near: Returns the detection centroid for a star searched around the given
            ``(v, u)`` position, or ``None``.
        tolerance_px: Largest offset difference that agrees.

    Returns:
        ``(agree, disagree)`` counts.
    """
    agree = 0
    disagreeing: list[tuple[float, float]] = []
    for feature in others:
        pred = predicted_vu(feature)
        target = place(pred)
        near = detect_near(feature, target)
        if near is not None and math.hypot(near[0] - target[0], near[1] - target[1]) <= (
            tolerance_px
        ):
            agree += 1
            continue
        det = detect(feature)
        if det is None:
            continue
        if math.hypot(det[0] - target[0], det[1] - target[1]) <= tolerance_px:
            agree += 1
        elif all(
            math.hypot(det[0] - seen[0], det[1] - seen[1]) >= SAME_SOURCE_PX for seen in disagreeing
        ):
            disagreeing.append(det)
    return agree, len(disagreeing)


def star_like_source(
    image_ext: NDArrayFloatType,
    star: NavFeature,
    sources: list[tuple[float, float]],
    *,
    peak_per_snr: float | None,
    peak_factor: float,
    box_half_px: int,
    exclude: tuple[tuple[float, float], ...] = (),
) -> tuple[float, float] | None:
    """Return a star's one source that is as bright as the star should be, if just one is.

    A source qualifies when its peak above the local background (:func:`source_peak`)
    lies within a factor of ``peak_factor`` of ``peak_per_snr`` times the star's predicted
    SNR; when that peak cannot be predicted (no scale, no predicted SNR, or a catalog
    magnitude that is only a saturated lower bound) every source qualifies.  A source
    within ``SAME_SOURCE_PX`` of one of ``exclude`` (the sources a pair's own stars were
    found at) is already accounted for and does not qualify.  The star is located only
    when exactly one source qualifies; a window holding several that could each be the
    star (a field of saturated cosmic-ray hits) does not locate it.

    Parameters:
        image_ext: 2-D extfov image array.
        star: The star.
        sources: The star's sources, strongest first.
        peak_per_snr: The scale from :func:`peak_per_snr`, or ``None``.
        peak_factor: Largest factor between a source's peak and the star's expected peak.
        box_half_px: Half-width of the box :func:`source_peak` searches.
        exclude: Sources that are not this star's.

    Returns:
        The source, or ``None`` when none or several qualify.
    """
    candidates = [
        det
        for det in sources
        if all(math.hypot(det[0] - e[0], det[1] - e[1]) >= SAME_SOURCE_PX for e in exclude)
    ]
    snr = predicted_snr(star)
    if peak_per_snr is None or snr <= 0.0 or photometry_saturated(star):
        qualifying = candidates
    else:
        expected = peak_per_snr * snr
        qualifying = [
            det
            for det in candidates
            if 1.0 / peak_factor
            <= source_peak(image_ext, det, box_half_px=box_half_px) / expected
            <= peak_factor
        ]
    return qualifying[0] if len(qualifying) == 1 else None


def absent_stars(
    image_ext: NDArrayFloatType,
    stars: list[NavFeature],
    place: Callable[[tuple[float, float]], tuple[float, float]],
    *,
    faint_snr: float,
    peak_per_snr: float | None,
    peak_factor: float,
    margin_vu: tuple[int, int],
    noise_sigma: float,
    box_half_px: int,
) -> int:
    """Count the stars a solution puts on clear sky in the image where nothing is seen.

    A star counts when it is at least as bright (predicted SNR) as ``faint_snr``, the
    fainter of two stars already found at the solution, so it should be seen at least as
    well; its catalog magnitude is not a saturated lower bound; where ``place`` puts its
    prediction lies inside the image, not the extended field of view's padding; the pixels
    around that position are clear sky, their robust spread at most twice the image noise,
    so a ring, a limb, or a bright neighbor that could hide the star rules it out; and the
    brightest pixel of the box there stands above its local background
    (:func:`source_peak`) by less than its expected peak, ``peak_per_snr`` times its
    predicted SNR, divided by ``peak_factor``.

    Parameters:
        image_ext: 2-D extfov image array.
        stars: The stars to judge.
        place: Maps a star's predicted ``(v, u)`` to where the solution puts it.
        faint_snr: Least predicted SNR of a star that is judged.
        peak_per_snr: The scale from :func:`peak_per_snr`, or ``None`` (no star is then
            judged absent).
        peak_factor: Factor below the expected peak at which a star is absent.
        margin_vu: ``(v, u)`` padding of the extended field of view, in pixels.
        noise_sigma: Image noise sigma.
        box_half_px: Half-width of the box :func:`source_peak` searches.

    Returns:
        The number of absent stars.
    """
    if peak_per_snr is None:
        return 0
    rows, cols = image_ext.shape
    reach = box_half_px + 3
    count = 0
    for star in stars:
        snr = predicted_snr(star)
        if snr < faint_snr or photometry_saturated(star):
            continue
        target = place(predicted_vu(star))
        v, u = round(target[0]), round(target[1])
        if not (
            margin_vu[0] + reach <= v < rows - margin_vu[0] - reach
            and margin_vu[1] + reach <= u < cols - margin_vu[1] - reach
        ):
            continue
        patch = np.asarray(image_ext[v - reach : v + reach + 1, u - reach : u + reach + 1])
        spread = 1.4826 * float(np.median(np.abs(patch - np.median(patch))))
        if spread > 2.0 * noise_sigma:
            continue
        seen = source_peak(image_ext, (v, u), box_half_px=box_half_px)
        if seen * peak_factor < peak_per_snr * snr:
            count += 1
    return count


def pair_brightness(
    image_ext: NDArrayFloatType,
    hyp: PairHypothesis,
    searched: list[NavFeature],
    *,
    box_half_px: int,
) -> float | None:
    """Return a pair's sources' peak per unit of predicted SNR (geometric mean).

    Parameters:
        image_ext: 2-D extfov image array.
        hyp: The pair.
        searched: The searched stars, indexed by ``hyp.star_indices``.
        box_half_px: Half-width of the box :func:`source_peak` searches.

    Returns:
        The brightness, or ``None`` when a star's predicted SNR or catalog magnitude cannot
        say how bright it should be, or a source shows no peak.
    """
    log_sum = 0.0
    for k, det in zip(hyp.star_indices, hyp.detections, strict=True):
        star = searched[k]
        snr = predicted_snr(star)
        peak = source_peak(image_ext, det, box_half_px=box_half_px)
        if snr <= 0.0 or peak <= 0.0 or photometry_saturated(star):
            return None
        log_sum += math.log(peak / snr)
    return math.exp(log_sum / 2.0)


def _outside_image(
    predicted_vu_pos: tuple[float, float], margin_vu: tuple[int, int], shape: tuple[int, ...]
) -> bool:
    """Return whether a prediction lies in the extended field of view's padding."""
    v, u = predicted_vu_pos
    return not (
        margin_vu[0] <= v < shape[0] - margin_vu[0] and margin_vu[1] <= u < shape[1] - margin_vu[1]
    )


def refute_pair(
    image_ext: NDArrayFloatType,
    hyp: PairHypothesis,
    searched: list[NavFeature],
    sources: list[list[tuple[float, float]]],
    *,
    detect_at: Callable[
        [NavFeature, tuple[float, float], tuple[float, float]], tuple[float, float] | None
    ],
    detect_star_like_at: Callable[
        [NavFeature, tuple[float, float], tuple[float, float], float | None],
        tuple[float, float] | None,
    ],
    margin_vu: tuple[int, int],
    noise_sigma: float,
    max_residual_px: float,
    peak_factor: float,
    contradiction_stars: int,
    absent_stars_needed: int,
    box_half_px: int,
) -> str | None:
    """Return why the other searched stars refute a pair's offset, or None when they do not.

    The pair's two stars set the brightness scale (:func:`peak_per_snr`), the peak per unit
    of predicted SNR, from the sources found where the pair's solution puts them, so each
    other star's expected peak is known.  The offset is refuted when:

    - no other star is found where the pair puts it as bright as it should be, and at
      least ``contradiction_stars`` distinct sources lie elsewhere, each the one source in
      its star's window as bright as the star should be (:func:`third_star_votes`,
      :func:`star_like_source`; the pair's own two sources are not counted); two stars can
      agree by coincidence, and three found elsewhere say they did; or
    - more other stars are absent where the pair puts them than are found there, and at
      least ``absent_stars_needed`` of them (:func:`absent_stars`), or one when both of the
      pair's stars are predicted in the padding, in the image only if the pointing error
      carries both in.  Each absent star is at least as bright as the pair's fainter star
      and lies on clear sky in the image, so it should be seen.

    Parameters:
        image_ext: 2-D extfov image array.
        hyp: The pair.
        searched: The searched stars, brightest first.
        sources: Each searched star's sources, strongest first.
        detect_at: Returns the isolated point source in a window of the given ``(v, u)``
            half-widths around a position, or ``None``.
        detect_star_like_at: As ``detect_at``, but only a source as bright as the star
            should be at the given scale.
        margin_vu: ``(v, u)`` padding of the extended field of view, in pixels.
        noise_sigma: Image noise sigma.
        max_residual_px: The pair search's residual tolerance; a star agrees within twice it.
        peak_factor: Factor within which a source is as bright as a star should be.
        contradiction_stars: Distinct sources elsewhere that refute an unsupported offset.
        absent_stars_needed: Absent stars that refute an offset.
        box_half_px: Half-width of the box :func:`source_peak` searches.

    Returns:
        The reason, or ``None``.
    """
    tolerance_px = 2.0 * max_residual_px
    tolerance_vu = (tolerance_px, tolerance_px)
    i, j = hyp.star_indices
    pair = [searched[i], searched[j]]
    others = [f for k, f in enumerate(searched) if k not in (i, j)]
    source_lists = {id(f): found for f, found in zip(searched, sources, strict=True)}
    place = hyp.solved.place
    scale = peak_per_snr(
        image_ext,
        pair,
        place,
        detect_at=lambda star, target: detect_at(star, target, tolerance_vu),
        box_half_px=box_half_px,
    )
    agree, disagree = third_star_votes(
        others,
        place,
        detect=lambda star: star_like_source(
            image_ext,
            star,
            source_lists[id(star)],
            peak_per_snr=scale,
            peak_factor=peak_factor,
            box_half_px=box_half_px,
            exclude=hyp.detections,
        ),
        detect_near=lambda star, target: detect_star_like_at(star, target, tolerance_vu, scale),
        tolerance_px=tolerance_px,
    )
    if agree == 0 and disagree >= contradiction_stars:
        return (
            f'{disagree} distinct source(s) as bright as other stars should be lie away '
            f"from the pair's offset and none at it"
        )
    absent = absent_stars(
        image_ext,
        others,
        place,
        faint_snr=min(predicted_snr(f) for f in pair),
        peak_per_snr=scale,
        peak_factor=peak_factor,
        margin_vu=margin_vu,
        noise_sigma=noise_sigma,
        box_half_px=box_half_px,
    )
    needed = (
        1
        if all(_outside_image(predicted_vu(f), margin_vu, image_ext.shape) for f in pair)
        else absent_stars_needed
    )
    if absent >= needed and absent > agree:
        return (
            f"{absent} star(s) at least as bright as the pair's fainter star show nothing "
            f'where the pair puts them in the image, {agree} found there'
        )
    return None


@dataclass(frozen=True)
class PairChoice:
    """The outcome of judging a frame's cross-checked pairs.

    Parameters:
        best: The chosen pair, or ``None`` when every pair is ruled out.
        rivals: Surviving pairs as well supported as ``best``, elsewhere, and nearly as
            bright, which make the choice ambiguous.
        reasons: Why each refuted offset was refuted, in the order judged.
    """

    best: PairHypothesis | None
    rivals: int
    reasons: list[str]


def choose_pair(
    hypotheses: list[PairHypothesis],
    *,
    refute: Callable[[PairHypothesis], str | None],
    brightness: Callable[[PairHypothesis], float | None],
    tolerance_px: float,
    rival_factor: float,
) -> PairChoice:
    """Judge cross-checked pairs, best supported first, and choose one.

    Pairs are taken in the order :func:`pair_hypotheses` returns them.  A pair within
    ``tolerance_px`` of an offset already judged (a surviving pair's or a refuted one's) is
    that offset again and is skipped.  Otherwise ``refute`` judges it; a refuted offset
    sets aside every later pair near it, since the other stars speak to the offset rather
    than to the pair.  Judging stops at the first pair less supported than the first
    survivor.  Of the survivors, all equally supported, the brightest per unit of predicted
    SNR is chosen: a ghost image of each star or a coincidence among faint sources shows
    far less light.  Every other survivor within ``rival_factor`` of that brightness, or
    whose brightness is unknown (as is the chosen one's, making every other a rival), is a
    rival.

    Parameters:
        hypotheses: The pairs, best supported first.
        refute: Returns why a pair's offset is refuted, or ``None``.
        brightness: Returns a pair's peak per unit of predicted SNR, or ``None``.
        tolerance_px: Distance within which two offsets are one.
        rival_factor: Brightness factor within which a survivor is a rival.

    Returns:
        The choice.
    """
    reasons: list[str] = []
    survivors: list[PairHypothesis] = []
    refuted: list[tuple[float, float]] = []

    def near(hyp: PairHypothesis, offsets: list[tuple[float, float]]) -> bool:
        return any(
            math.hypot(hyp.solved.offset_vu[0] - v, hyp.solved.offset_vu[1] - u) <= tolerance_px
            for v, u in offsets
        )

    for hyp in hypotheses:
        if survivors and hyp.support < survivors[0].support:
            break
        if near(hyp, [c.solved.offset_vu for c in survivors]) or near(hyp, refuted):
            continue
        reason = refute(hyp)
        if reason is not None:
            reasons.append(reason)
            refuted.append(hyp.solved.offset_vu)
            continue
        survivors.append(hyp)
    if not survivors:
        return PairChoice(best=None, rivals=0, reasons=reasons)
    lights = [brightness(h) for h in survivors]
    known = [b for b in lights if b is not None]
    best_index = (
        max(range(len(survivors)), key=lambda k: lights[k] or 0.0)
        if len(known) == len(survivors)
        else 0
    )
    best_light = lights[best_index]
    rivals = sum(
        1
        for k, b in enumerate(lights)
        if k != best_index and (b is None or best_light is None or b * rival_factor >= best_light)
    )
    return PairChoice(best=survivors[best_index], rivals=rivals, reasons=reasons)
