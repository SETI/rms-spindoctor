"""Lone-star search for ``StarUniqueMatchNav``'s one-star path.

When no two stars agree on an offset, the one-star path of
:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`
looks for a single star over the instrument's whole offset window.  A window that wide
often holds isolated point sources that are not the star: cosmic-ray hits, faint field
stars below the catalog's limit, small moons.  So every searched star keeps the same
candidates the pair search gave it, each (star, source) pairing is one hypothesis about
the offset, and the other catalog stars judge every hypothesis:

- a source sharper than a star outright (:func:`too_sharp`, allowing nothing for noise),
  or brighter per unit of predicted SNR than any star the instrument shows, is not taken
  for a lone star, since nothing cross-checks it;
- a hypothesis is refuted when other stars that the offset puts on clear sky in the
  image, and that should show at least a ``1 / peak_factor`` share of their expected
  light, show nothing there (:func:`absent_stars`);
- of the hypotheses left, the one brightest for its star's predicted SNR is chosen, as
  the pair search chooses among pairs;
- the choice is ambiguous when another source of the same star's window is nearly as
  bright, or another star's surviving hypothesis elsewhere is nearly as bright for its
  predicted SNR.

The technique adds its own checks (another star of comparable brightness that could be
the source, the lone-star ellipse, the no-rival residual bound) and builds the result.

The submodule is private (leading underscore on the file name); its functions carry no
underscore prefix because the technique module imports them, but they are not part of
the package's public API.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique._star_helpers import predicted_snr, predicted_vu
from spindoctor.nav_technique._star_unique_two_star import (
    absent_stars,
    photometry_saturated,
    source_peak,
)
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'LoneStarChoice',
    'LoneStarHypothesis',
    'choose_lone_star',
    'lone_star_hypotheses',
]


@dataclass(frozen=True)
class LoneStarHypothesis:
    """One searched star taken to be one of the sources in its window.

    Parameters:
        star_index: Index of the star among the searched stars.
        source_vu: The source's centroid, ``(v, u)``.
        offset_vu: The offset that carries the star's prediction onto the source.
        peak: The source's peak above its local background (:func:`source_peak`).
        peak_per_snr: ``peak`` divided by the star's predicted SNR, or ``None`` when the
            star's predicted SNR or catalog magnitude cannot say how bright it should be.
        star_like: Whether the source could be the star: no sharper than a star and, where
            the instrument bounds it, no brighter for the star's predicted SNR.
        refutation: Why the other stars refute the offset, or ``None``.
    """

    star_index: int
    source_vu: tuple[float, float]
    offset_vu: tuple[float, float]
    peak: float
    peak_per_snr: float | None
    star_like: bool
    refutation: str | None


def _translation(
    offset_vu: tuple[float, float],
) -> Callable[[tuple[float, float]], tuple[float, float]]:
    """Return the map that carries a prediction ``(v, u)`` by ``offset_vu``."""

    def place(pos: tuple[float, float]) -> tuple[float, float]:
        return (pos[0] + offset_vu[0], pos[1] + offset_vu[1])

    return place


def lone_star_hypotheses(
    image_ext: NDArrayFloatType,
    searched: list[NavFeature],
    sources: list[list[tuple[float, float]]],
    ranked: list[NavFeature],
    *,
    is_star_like: Callable[[tuple[float, float]], bool],
    margin_vu: tuple[int, int],
    noise_sigma: float,
    peak_factor: float,
    absent_stars_needed: int,
    box_half_px: int,
    max_peak_per_snr: float | None = None,
) -> list[LoneStarHypothesis]:
    """Return every (searched star, source) pairing with the other stars' verdict on it.

    A pairing's offset is refuted when at least ``absent_stars_needed`` other usable stars
    are absent where it puts them (:func:`absent_stars`): inside the image, on clear sky,
    and below ``1 / peak_factor`` of the peak the source's brightness per unit of
    predicted SNR says they should show.  Every other usable star is judged, however
    faint; a star too faint to stand above the noise at that scale cannot fall short of
    it.  A pairing whose star cannot say how bright it should be sets no scale, so
    nothing refutes it.

    A source is star-like when ``is_star_like`` says so and, given ``max_peak_per_snr``,
    its peak per unit of the star's predicted SNR does not exceed it.  The bound is the
    camera's measured one, from the brightest stars found on a sample of its frames.

    Parameters:
        image_ext: 2-D extfov image array.
        searched: The searched stars, brightest first.
        sources: Each searched star's sources, strongest first.
        ranked: Every distinct usable star, brightest first; the judges.
        is_star_like: Returns whether a source is no sharper than a star.
        margin_vu: ``(v, u)`` padding of the extended field of view, in pixels.
        noise_sigma: Image noise sigma.
        peak_factor: Factor below the expected peak at which a star is absent.
        absent_stars_needed: Absent stars that refute an offset.
        box_half_px: Half-width of the box :func:`source_peak` searches.
        max_peak_per_snr: Largest peak per unit of predicted SNR, in the image's units,
            of a source that could be the star, or ``None`` for no bound.

    Returns:
        The pairings, searched star by searched star, each star's sources strongest first.
    """
    hypotheses: list[LoneStarHypothesis] = []
    for index, (star, found) in enumerate(zip(searched, sources, strict=True)):
        pred_v, pred_u = predicted_vu(star)
        snr = predicted_snr(star)
        others = [f for f in ranked if f is not star]
        for source_vu in found:
            offset_vu = (source_vu[0] - pred_v, source_vu[1] - pred_u)
            peak = source_peak(image_ext, source_vu, box_half_px=box_half_px)
            scale = None if snr <= 0.0 or photometry_saturated(star) else peak / snr
            absent = absent_stars(
                image_ext,
                others,
                _translation(offset_vu),
                faint_snr=0.0,
                peak_per_snr=scale,
                peak_factor=peak_factor,
                margin_vu=margin_vu,
                noise_sigma=noise_sigma,
                box_half_px=box_half_px,
            )
            refutation = (
                f'{absent} other star(s) show nothing where the offset puts them on clear sky'
                if absent >= absent_stars_needed
                else None
            )
            hypotheses.append(
                LoneStarHypothesis(
                    star_index=index,
                    source_vu=source_vu,
                    offset_vu=offset_vu,
                    peak=peak,
                    peak_per_snr=scale,
                    star_like=is_star_like(source_vu)
                    and (max_peak_per_snr is None or scale is None or scale <= max_peak_per_snr),
                    refutation=refutation,
                )
            )
    return hypotheses


@dataclass(frozen=True)
class LoneStarChoice:
    """The outcome of judging a frame's lone-star hypotheses.

    Parameters:
        best: The chosen hypothesis, or ``None`` when no hypothesis survives or the
            choice is ambiguous.
        reason: Why ``best`` is ``None``; ``None`` when a hypothesis was chosen.
    """

    best: LoneStarHypothesis | None
    reason: str | None


def choose_lone_star(
    hypotheses: list[LoneStarHypothesis],
    *,
    tolerance_px: float,
    runner_up_ratio: float,
    rival_factor: float,
) -> LoneStarChoice:
    """Choose the lone star among hypotheses that are star-like and not refuted.

    The survivor brightest per unit of predicted SNR is chosen (one whose star cannot say
    how bright it should be ranks last).  The choice is ambiguous when, more than
    ``tolerance_px`` from its offset:

    - another source of the chosen star's window, star-like or not and refuted or not,
      has a peak above the chosen source's divided by ``runner_up_ratio``: the star need
      not be the brightest thing in a window as wide as the offset window, but then
      nothing says which of the two it is; or
    - another star's survivor is within ``rival_factor`` of the chosen brightness per unit
      of predicted SNR, or either brightness is unknown: two stars each explain one
      source, and neither explanation is contradicted.

    Parameters:
        hypotheses: The hypotheses (:func:`lone_star_hypotheses`).
        tolerance_px: Distance within which two offsets are one.
        runner_up_ratio: Least ratio of the chosen source's peak to another source's in
            the same window.
        rival_factor: Brightness factor within which another star's survivor is a rival.

    Returns:
        The choice.
    """
    survivors = [h for h in hypotheses if h.star_like and h.refutation is None]
    if not survivors:
        not_star_like = sum(1 for h in hypotheses if not h.star_like)
        refuted = [h.refutation for h in hypotheses if h.star_like and h.refutation]
        detail = f'; {refuted[0]}' if refuted else ''
        return LoneStarChoice(
            best=None,
            reason=(
                f'no_lone_star_candidate: {len(hypotheses)} source(s), {not_star_like} '
                f'sharper or brighter than a star, {len(refuted)} refuted{detail}'
            ),
        )
    best = max(
        survivors,
        key=lambda h: -math.inf if h.peak_per_snr is None else h.peak_per_snr,
    )

    def elsewhere(h: LoneStarHypothesis) -> bool:
        return (
            math.hypot(h.offset_vu[0] - best.offset_vu[0], h.offset_vu[1] - best.offset_vu[1])
            > tolerance_px
        )

    for h in hypotheses:
        if h is best or h.star_index != best.star_index or not elsewhere(h):
            continue
        if h.peak * runner_up_ratio > best.peak:
            return LoneStarChoice(
                best=None,
                reason=(
                    f'ambiguous_detection: another source of the window peaks at '
                    f"{h.peak:.4g} against the chosen source's {best.peak:.4g}, "
                    f'within a factor of {runner_up_ratio:.2f}'
                ),
            )
    for h in survivors:
        if h.star_index == best.star_index or not elsewhere(h):
            continue
        if (
            h.peak_per_snr is None
            or best.peak_per_snr is None
            or h.peak_per_snr * rival_factor >= best.peak_per_snr
        ):
            return LoneStarChoice(
                best=None,
                reason=(
                    'ambiguous_lone_stars: another star explains another source nearly as '
                    'well for its predicted brightness, and nothing contradicts either'
                ),
            )
    return LoneStarChoice(best=best, reason=None)
