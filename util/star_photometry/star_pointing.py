"""A pointing fix from catalog stars and image peaks alone.

The vote matches predicted star positions to detected peaks and refines the agreeing pairs to
a translation and, with enough pairs, a rotation.  It takes plain arrays and knows nothing
about any instrument or the navigator; the peaks come from
:func:`util.star_photometry.star_field.detect_peaks`.

Coordinates are array-index coordinates ``(v, u)``, as in
:mod:`util.star_photometry.star_field`.  The fix follows the navigator's offset convention: a
star predicted at ``p`` is found at ``R(rotation) (p - center) + center + offset``, a
translation plus a rotation about the field center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from util.star_photometry.star_field import PeakList

__all__ = ['PointingFix', 'apply_fix', 'find_pointing']

FloatArray = NDArray[np.float64]

#: The smallest number of star-peak pairs that can define a rotation as well as a translation.
MIN_PAIRS_FOR_ROTATION = 5


@dataclass(frozen=True)
class PointingFix:
    """The pointing a set of catalog stars agrees on.

    Attributes:
        accepted: Whether the fix meets the agreement and significance criteria.
        offset_v: Line offset (pixels) of the fix: actual = predicted + offset, after rotation.
        offset_u: Sample offset (pixels).
        rotation_deg: Rotation about the field center (degrees, counterclockwise from +u
            toward +v), zero when too few stars agree to fit it.
        n_agree: Number of catalog stars whose peak lies within the agreement radius of the fix,
            each peak paired with at most one star.
        n_candidates: Number of catalog stars searched.
        n_peaks: Number of image peaks searched.
        spread_px: RMS distance of the agreeing stars from the fix (pixels).
        runner_up: Stars voting for the best offset at least ``exclusion_px`` from the fix, in
            the kept tier.
        chance_probability: Probability of the kept tier's cluster arising by chance anywhere
            in the search window, from the density of its star-peak pairs.
        agreeing: Indexes (into the candidate list) of the agreeing stars.
        reason: Why the fix was refused, or ``''`` when it was accepted.
        tier: ``[n_stars, n_peaks]`` of the search tier the fix came from.
        tier_agree: Agreeing stars within that tier.
    """

    accepted: bool
    offset_v: float
    offset_u: float
    rotation_deg: float
    n_agree: int
    n_candidates: int
    n_peaks: int
    spread_px: float
    runner_up: int
    chance_probability: float
    agreeing: list[int] = field(default_factory=list)
    reason: str = ''
    tier: list[int] = field(default_factory=list)
    tier_agree: int = 0


def apply_fix(
    fix: PointingFix, v: FloatArray, u: FloatArray, center_vu: tuple[float, float]
) -> tuple[FloatArray, FloatArray]:
    """Return where predicted positions land under a pointing fix.

    Parameters:
        fix: The fix.
        v: Predicted lines.
        u: Predicted samples.
        center_vu: The rotation center.

    Returns:
        The actual ``(v, u)``.
    """
    return _transform(v, u, fix.offset_v, fix.offset_u, math.radians(fix.rotation_deg), center_vu)


def _transform(
    v: FloatArray,
    u: FloatArray,
    dv: float,
    du: float,
    theta: float,
    center_vu: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Rotate positions by ``theta`` radians about ``center_vu``, then translate.

    Parameters:
        v: Lines.
        u: Samples.
        dv: Line translation.
        du: Sample translation.
        theta: Rotation in radians.
        center_vu: The rotation center.

    Returns:
        The transformed ``(v, u)``.
    """
    cv, cu = center_vu
    c, s = math.cos(theta), math.sin(theta)
    rv = cv + (v - cv) * c + (u - cu) * s
    ru = cu - (v - cv) * s + (u - cu) * c
    return rv + dv, ru + du


def find_pointing(
    pred_v: FloatArray,
    pred_u: FloatArray,
    peaks: PeakList,
    *,
    center_vu: tuple[float, float],
    search_px: float,
    vote_bin_px: float,
    agree_px: float,
    min_agree: int,
    max_chance_probability: float,
    tiers: tuple[tuple[int, int], ...] = ((15, 30), (30, 80), (60, 300)),
    exclusion_px: float = 5.0,
    n_seeds: int = 20,
) -> PointingFix:
    """Find the pointing most catalog stars agree on.

    The candidate stars must be ordered brightest first, as the peaks are.  The search runs
    once per tier ``(n_stars, n_peaks)`` on that many of the brightest stars and peaks: a
    true pointing matches bright stars to bright peaks, and the fewer the pairs, the less
    likely a chance cluster, so a few agreeing bright stars can be significant where the same
    stars among hundreds of faint pairs would not be.

    Within a tier, every pair of a star and a peak within ``search_px`` (per axis) votes for
    the offset between them, counting each star once per vote bin.  The ``n_seeds`` bins
    holding the most stars (summed over a 2 x 2 neighborhood, so a cluster split by a bin
    edge still counts whole) are each refined: stars and peaks within ``2 * agree_px`` of
    each other under the seed are paired one to one, closest pairs first (a peak is one
    source and confirms at most one star), and a least-squares translation (plus a rotation
    about ``center_vu`` when at least five stars pair) is fitted, then paired and refitted
    within ``agree_px`` until the agreeing star-peak pairs stop changing.  The refinement with
    the most agreeing stars (then the smallest spread) is the tier's fix.

    A tier's chance probability is the chance of a cluster of its agreeing-star count from
    randomly placed pairs anywhere in the search window: a Poisson tail with the tier's mean
    number of pairs per disk of radius ``agree_px``, times the number of such disks in the
    window, times the number of tiers.  The tier with the smallest chance is kept, and its fix
    is refined once more on every star and peak, which gives the final agreeing set.

    The fix is accepted when the final agreeing set holds at least ``min_agree`` stars and the
    kept tier's chance probability is at most ``max_chance_probability``.

    Parameters:
        pred_v: Predicted line of each candidate star (array-index coordinates), brightest
            first.
        pred_u: Predicted sample of each candidate star.
        peaks: The image peaks, brightest first.
        center_vu: The rotation center (the field center).
        search_px: Largest offset searched on each axis (pixels).
        vote_bin_px: The vote-bin size (pixels).
        agree_px: A star agrees when its peak lies within this distance of the fit (pixels).
        min_agree: The fewest agreeing stars an accepted fix needs.
        max_chance_probability: The largest chance probability an accepted fix may have.
        tiers: ``(n_stars, n_peaks)`` of each search, brightest first.
        exclusion_px: The runner-up cluster is the best one at least this far from the fix.
        n_seeds: The number of best vote bins refined per tier.

    Returns:
        The fix; ``accepted`` is False, with a ``reason``, when it fails a criterion.
    """
    n_star, n_peak = pred_v.size, peaks.v.size
    if n_star == 0 or n_peak == 0:
        return _refused(n_star, n_peak, 'no candidate stars or no peaks')
    best: tuple[float, _TierFit] | None = None
    for n_tier_stars, n_tier_peaks in tiers:
        sub = PeakList(
            v=peaks.v[:n_tier_peaks], u=peaks.u[:n_tier_peaks], snr=peaks.snr[:n_tier_peaks]
        )
        tier = _vote_tier(
            pred_v[:n_tier_stars],
            pred_u[:n_tier_stars],
            sub,
            center_vu=center_vu,
            search_px=search_px,
            vote_bin_px=vote_bin_px,
            agree_px=agree_px,
            exclusion_px=exclusion_px,
            n_seeds=n_seeds,
        )
        if tier is None:
            continue
        chance = min(1.0, tier.chance * len(tiers))
        if best is None or (chance, -len(tier.agreeing)) < (best[0], -len(best[1].agreeing)):
            best = (chance, tier)
    if best is None:
        return _refused(n_star, n_peak, 'no star-peak pair inside the search window')
    chance, tier = best
    dv, du, theta, pairs, rms = _refine(
        pred_v, pred_u, peaks, tier.offset_v, tier.offset_u, center_vu, agree_px, theta=tier.theta
    )
    agreeing = [star for star, _ in pairs]
    n_agree = len(agreeing)
    reason = ''
    if n_agree < min_agree:
        reason = f'only {n_agree} stars agree'
    elif chance > max_chance_probability:
        reason = f'chance probability {chance:.2g} too high'
    return PointingFix(
        accepted=reason == '',
        offset_v=dv,
        offset_u=du,
        rotation_deg=math.degrees(theta),
        n_agree=n_agree,
        n_candidates=n_star,
        n_peaks=n_peak,
        spread_px=rms,
        runner_up=tier.runner_up,
        chance_probability=chance,
        agreeing=agreeing,
        reason=reason,
        tier=[tier.n_stars, tier.n_peaks],
        tier_agree=len(tier.agreeing),
    )


@dataclass(frozen=True)
class _TierFit:
    """One tier's vote and refinement.

    Attributes:
        offset_v: Line offset.
        offset_u: Sample offset.
        theta: Rotation in radians.
        agreeing: Indexes of the agreeing stars.
        rms: RMS residual of the agreeing stars.
        runner_up: Best vote sum at least the exclusion distance from the fix.
        chance: The chance probability for this tier alone.
        n_stars: Stars searched.
        n_peaks: Peaks searched.
    """

    offset_v: float
    offset_u: float
    theta: float
    agreeing: list[int]
    rms: float
    runner_up: int
    chance: float
    n_stars: int
    n_peaks: int


def _vote_tier(
    pred_v: FloatArray,
    pred_u: FloatArray,
    peaks: PeakList,
    *,
    center_vu: tuple[float, float],
    search_px: float,
    vote_bin_px: float,
    agree_px: float,
    exclusion_px: float,
    n_seeds: int,
) -> _TierFit | None:
    """Vote for the offset with one tier of stars and peaks, and refine the best bins.

    Parameters:
        pred_v: Predicted lines.
        pred_u: Predicted samples.
        peaks: Peaks.
        center_vu: Rotation center.
        search_px: Search half-width.
        vote_bin_px: Vote-bin size.
        agree_px: Agreement radius.
        exclusion_px: Runner-up exclusion distance.
        n_seeds: Bins refined.

    Returns:
        The tier's fit, or None when no pair lies inside the window.
    """
    dv = peaks.v[None, :] - pred_v[:, None]
    du = peaks.u[None, :] - pred_u[:, None]
    in_window = (np.abs(dv) <= search_px) & (np.abs(du) <= search_px)
    star_index, peak_index = np.nonzero(in_window)
    if star_index.size == 0:
        return None
    n_bins = math.ceil(2 * search_px / vote_bin_px) + 1
    bv = np.floor((dv[star_index, peak_index] + search_px) / vote_bin_px).astype(int)
    bu = np.floor((du[star_index, peak_index] + search_px) / vote_bin_px).astype(int)
    # One vote per star per bin, so a star near a clump of peaks does not count several times.
    keys = np.unique(np.stack([bv, bu, star_index]), axis=1)
    votes = np.zeros((n_bins + 1, n_bins + 1))
    np.add.at(votes, (keys[0], keys[1]), 1.0)
    summed = votes[:-1, :-1] + votes[1:, :-1] + votes[:-1, 1:] + votes[1:, 1:]
    # Refine the best few bins and keep the fit most stars agree with, since a defect that
    # makes a line of peaks can tie the true bin on raw votes.
    order = np.argsort(-summed, axis=None)[:n_seeds]
    fit: tuple[float, float, float, list[tuple[int, int]], float] | None = None
    for flat in order:
        if summed.flat[flat] < 1:
            break
        best_v, best_u = np.unravel_index(int(flat), summed.shape)
        seed_v = (best_v + 1) * vote_bin_px - search_px
        seed_u = (best_u + 1) * vote_bin_px - search_px
        trial = _refine(pred_v, pred_u, peaks, seed_v, seed_u, center_vu, agree_px)
        if fit is None or (len(trial[3]), -trial[4]) > (len(fit[3]), -fit[4]):
            fit = trial
    if fit is None:
        return None
    n_agree = len(fit[3])
    centers = (np.arange(n_bins) + 1) * vote_bin_px - search_px
    far = (np.abs(centers[:, None] - fit[0]) >= exclusion_px) | (
        np.abs(centers[None, :] - fit[1]) >= exclusion_px
    )
    runner_up = int(np.max(np.where(far, summed, 0.0))) if np.any(far) else 0
    window_area = (2 * search_px) ** 2
    disk_area = math.pi * agree_px**2
    mean_pairs = star_index.size * disk_area / window_area
    tail = float(scipy.stats.poisson.sf(n_agree - 1, mean_pairs)) if n_agree > 0 else 1.0
    return _TierFit(
        offset_v=fit[0],
        offset_u=fit[1],
        theta=fit[2],
        agreeing=[star for star, _ in fit[3]],
        rms=fit[4],
        runner_up=runner_up,
        chance=min(1.0, window_area / disk_area * tail),
        n_stars=int(pred_v.size),
        n_peaks=int(peaks.v.size),
    )


def _refused(n_star: int, n_peak: int, reason: str) -> PointingFix:
    """Return a refused fix with nothing measured.

    Parameters:
        n_star: Candidate stars searched.
        n_peak: Peaks searched.
        reason: Why.

    Returns:
        The refused fix.
    """
    return PointingFix(
        accepted=False,
        offset_v=float('nan'),
        offset_u=float('nan'),
        rotation_deg=0.0,
        n_agree=0,
        n_candidates=n_star,
        n_peaks=n_peak,
        spread_px=float('nan'),
        runner_up=0,
        chance_probability=1.0,
        reason=reason,
    )


def _refine(
    pred_v: FloatArray,
    pred_u: FloatArray,
    peaks: PeakList,
    seed_v: float,
    seed_u: float,
    center_vu: tuple[float, float],
    agree_px: float,
    *,
    theta: float = 0.0,
) -> tuple[float, float, float, list[tuple[int, int]], float]:
    """Refine a seed offset into a translation-plus-rotation fit.

    Stars and peaks are paired one to one and the fit is solved on the pairs, repeatedly,
    until the agreeing ``(star, peak)`` pairs stop changing (a star can keep agreeing while
    its peak changes, which moves the fit) or ten rounds have run.

    Parameters:
        pred_v: Predicted lines.
        pred_u: Predicted samples.
        peaks: Image peaks.
        seed_v: Seed line offset.
        seed_u: Seed sample offset.
        center_vu: Rotation center.
        agree_px: Agreement radius.
        theta: Seed rotation in radians.

    Returns:
        ``(offset_v, offset_u, theta_rad, agreeing_pairs, rms_px)``, the pairs as
        ``(star, peak)`` indexes in increasing star index.
    """
    dv, du = seed_v, seed_u
    radius = 2.0 * agree_px
    agreeing: list[tuple[int, int]] = []
    rms = float('nan')
    for _ in range(10):
        tv, tu = _transform(pred_v, pred_u, dv, du, theta, center_vu)
        chosen, matched = _associate(tv, tu, peaks, radius)
        if chosen.size == 0:
            return dv, du, theta, [], float('nan')
        sv, su = pred_v[chosen], pred_u[chosen]
        ov, ou = peaks.v[matched], peaks.u[matched]
        dv, du, theta = _solve(
            sv, su, ov, ou, center_vu, fit_rotation=chosen.size >= MIN_PAIRS_FOR_ROTATION
        )
        tv, tu = _transform(sv, su, dv, du, theta, center_vu)
        resid = np.hypot(ov - tv, ou - tu)
        close = resid <= agree_px
        rms = float(np.sqrt(np.mean(resid[close] ** 2))) if np.any(close) else rms
        new = [(int(k), int(m)) for k, m in zip(chosen[close], matched[close], strict=True)]
        if new == agreeing and radius == agree_px:
            break
        agreeing = new
        radius = agree_px
    return dv, du, theta, agreeing, rms


def _associate(
    tv: FloatArray, tu: FloatArray, peaks: PeakList, radius: float
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Pair stars with peaks one to one, closest pairs first, within ``radius``.

    Each peak is one source, so it can confirm at most one star: when two predictions lie
    near the same peak, the closer one takes it and the other takes its next-nearest free
    peak within ``radius``, or none.

    Parameters:
        tv: Lines where the stars are predicted under the trial fit.
        tu: Samples where the stars are predicted under the trial fit.
        peaks: Image peaks.
        radius: The largest star-peak distance paired (pixels).

    Returns:
        ``(star_indexes, peak_indexes)`` of the pairs, in increasing star index.
    """
    dist = np.hypot(peaks.v[None, :] - tv[:, None], peaks.u[None, :] - tu[:, None])
    star_index, peak_index = np.nonzero(dist <= radius)
    order = np.argsort(dist[star_index, peak_index], kind='stable')
    star_used: set[int] = set()
    peak_used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for k in order:
        star, peak = int(star_index[k]), int(peak_index[k])
        if star in star_used or peak in peak_used:
            continue
        star_used.add(star)
        peak_used.add(peak)
        pairs.append((star, peak))
    pairs.sort()
    stars = np.array([p[0] for p in pairs], dtype=np.intp)
    matched = np.array([p[1] for p in pairs], dtype=np.intp)
    return stars, matched


def _solve(
    sv: FloatArray,
    su: FloatArray,
    ov: FloatArray,
    ou: FloatArray,
    center_vu: tuple[float, float],
    *,
    fit_rotation: bool,
) -> tuple[float, float, float]:
    """Least-squares translation (and small rotation) mapping predicted onto observed.

    Parameters:
        sv: Predicted lines.
        su: Predicted samples.
        ov: Observed lines.
        ou: Observed samples.
        center_vu: Rotation center.
        fit_rotation: Whether to fit the rotation.

    Returns:
        ``(offset_v, offset_u, theta_rad)``.
    """
    if not fit_rotation:
        return float(np.mean(ov - sv)), float(np.mean(ou - su)), 0.0
    cv, cu = center_vu
    # Linearized rotation: v' = v + theta * (u - cu), u' = u - theta * (v - cv).
    n = sv.size
    a = np.zeros((2 * n, 3))
    a[:n, 0] = 1.0
    a[n:, 1] = 1.0
    a[:n, 2] = su - cu
    a[n:, 2] = -(sv - cv)
    b = np.concatenate([ov - sv, ou - su])
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(sol[0]), float(sol[1]), float(sol[2])
