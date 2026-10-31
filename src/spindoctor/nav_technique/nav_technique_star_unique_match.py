"""``StarUniqueMatchNav`` — translation fit from one star or a cross-checked pair.

Two paths share one technique:

- **One-star path.**  When the catalog reduction produces one star whose
  predicted brightness is at least ``brightness_margin_to_next_catalog_star_mag``
  brighter than the next-brightest predictable star, the strongest
  detection inside its search window is unambiguously its match.  The
  offset is the centroid minus the prediction; confidence is capped at
  the configured one-star limit (default 0.7) because a single match
  cannot cross-check itself.

  No-rival sentinels: the ``one_star_min_peak_ratio`` ambiguity
  gate and the brightness-margin gate both report ``inf`` when
  no rival exists (no runner-up detection above the window background /
  no other predictable catalog star).  The sentinels deliberately PASS
  the ratio checks -- a genuinely unique bright star has no rival and
  must remain matchable -- but an infinite peak ratio also means the
  ambiguity gate measured nothing (a flat or quantized window, exactly
  where a lone hot pixel or artifact would otherwise auto-pass every
  check).  In that vacuous case acceptance is additionally gated on the
  prediction-to-detection distance staying within
  ``one_star_max_residual_px``: with no rival statistics to lean on, a
  lone detection is only promoted to an identification when it sits
  inside the pointing-prior core.  A finite peak ratio leaves
  acceptance to the measured ambiguity gate; genuine one-star matches with
  offsets up to ~24 px exist in the operator-verified library, so the
  10 px bound cannot hold every match.  Every one-star match is held
  instead to the lone-star ellipse described below, above those offsets.

- **Two-star path.**  Tried first whenever two or more stars are usable.
  Each of the ``pair_search_stars`` brightest usable stars is searched for
  over the instrument's whole offset window, and its
  ``pair_search_sources`` strongest isolated point sources are its
  candidates: the star need not be the strongest thing in a wide window.
  Every two candidates of two different stars that agree within
  ``max_residual_px`` form a cross-checked pair, backed by every searched
  star with a candidate where the pair's offset puts it.  The pair backed
  by the most stars is the answer; confidence is capped at the configured
  two-star limit (default 0.8).  It is spurious when another pair backed by
  as many stars disagrees with it, when three other stars are found, as
  bright as they should be, away from its offset and none at it, or when
  two stars at least as bright as its fainter star lie on clear sky in the
  image at its offset and show nothing there.  Searching beyond the
  brightest two is what lets a frame navigate when its brightest predicted
  star is not in the image -- off the detector, or lost in a ring's glare.

Local-window detection is intentional: the search window is the
instrument's offset window, sized to bracket the SPICE pointing error, so
the star lies inside it.  A detection is a matched-filter peak of the
image, so the flank of a brighter source just outside the window is never
taken, and it must be an isolated point source: a ridge of a ring or a
cosmic-ray track is a peak but not a star.  The one-star path takes its
window's strongest peak, so a brighter source anywhere in the window
vetoes a fainter one, and accepts it only inside the ellipse whose
semi-axes are the window's half-widths scaled by
``one_star_ellipse_fraction``.  No global star-detection pass is needed
(and this technique is feasible even on images where multi-star detection
would fail because the rest of the field is too faint or too crowded).
Every star is predicted inside the extended field of view, whose padding
is the instrument's offset window, so every star's window reaches the
image.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique._star_helpers import (
    brightness_margin_mag,
    detection_peak_ratio,
    is_isolated_point_source,
    isolation_lookup,
    isolation_radii,
    local_centroid,
    predicted_snr,
    predicted_vu,
    search_window_bounds,
    smear_length_px,
    usable_stars,
    window_sources,
)
from spindoctor.nav_technique._star_unique_two_star import (
    PairHypothesis,
    TwoStarAssignment,
    absent_stars,
    build_star_covariance,
    pair_hypotheses,
    peak_per_snr,
    photometry_saturated,
    source_peak,
    star_like_source,
    third_star_votes,
    two_star_covariance,
)
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import StarUniqueMatchDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    embed_rotation_unobservable,
    log_confidence_breakdown,
    reported_position_vu,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['StarUniqueMatchNav']


# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.StarUniqueMatchNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


class _UniqueMatchConfidenceContext:
    """Adapter binding ``StarUniqueMatchDiagnostics`` plus side flags."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: StarUniqueMatchDiagnostics,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.predicted_snr = diagnostics.predicted_snr
        self.brightness_margin_mag = diagnostics.brightness_margin_mag
        self.residual_px = diagnostics.residual_px
        # Note: ``mode`` is a string, so it is intentionally not surfaced
        # to the confidence formula (no numeric meaning).  Mode-driven
        # caps are applied inside ``StarUniqueMatchNav.navigate``.


class StarUniqueMatchNav(NavTechnique):
    """Star unique-match translation fit (1- or 2-star path).

    Class attributes:
        accepts_feature_types: ``frozenset({STAR})``.
        requires_prior: ``False`` — runs in pass 1.
    """

    name = 'StarUniqueMatchNav'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'predicted_snr',
            'brightness_margin_mag',
            'residual_px',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._brightness_margin_floor_mag = float(
            self.tuning['brightness_margin_to_next_catalog_star_mag']
        )
        self._centroid_box_half_px = int(self.tuning['centroid_box_half_px'])
        self._psf_sigma_px = float(self.tuning['psf_sigma_px'])
        self._max_peak_sharpness = float(self.tuning['max_peak_sharpness'])
        self._max_residual_px = float(self.tuning['max_residual_px'])
        self._pair_search_stars = int(self.tuning['pair_search_stars'])
        self._pair_search_sources = int(self.tuning['pair_search_sources'])
        self._pair_contradiction_stars = int(self.tuning['pair_contradiction_stars'])
        self._pair_absent_stars = int(self.tuning['pair_absent_stars'])
        self._pair_vote_peak_factor = float(self.tuning['pair_vote_peak_factor'])
        self._one_star_ellipse_fraction = float(self.tuning['one_star_ellipse_fraction'])
        self._isolation_inner_px = int(self.tuning['source_isolation_inner_px'])
        self._isolation_outer_px = int(self.tuning['source_isolation_outer_px'])
        self._isolation_max_ratio = float(self.tuning['source_isolation_max_ratio'])
        self._detection_sigma = float(self.tuning['detection_sigma'])
        self._one_star_confidence_cap = float(self.tuning['one_star_confidence_cap'])
        self._two_star_confidence_cap = float(self.tuning['two_star_confidence_cap'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        self._one_star_min_peak_ratio = float(self.tuning['one_star_min_peak_ratio'])
        self._one_star_max_residual_px = float(self.tuning['one_star_max_residual_px'])
        for key in ('pair_search_sources', 'pair_absent_stars'):
            if int(self.tuning[key]) < 1:
                raise ValueError(f'{key} must be >= 1; got {self.tuning[key]!r}')
        if not 1.0 < self._pair_vote_peak_factor < math.inf:
            raise ValueError(
                f'pair_vote_peak_factor must be finite and > 1; got {self._pair_vote_peak_factor!r}'
            )
        if not 0.0 < self._one_star_ellipse_fraction <= 1.0:
            raise ValueError(
                'one_star_ellipse_fraction must lie in (0, 1]; '
                f'got {self._one_star_ellipse_fraction!r}'
            )
        if self._one_star_min_peak_ratio < 1.0:
            raise ValueError(
                f'one_star_min_peak_ratio must be >= 1; got {self._one_star_min_peak_ratio!r}'
            )
        if self._one_star_max_residual_px <= 0.0:
            raise ValueError(
                f'one_star_max_residual_px must be > 0; got {self._one_star_max_residual_px!r}'
            )
        if not 0.0 <= self._one_star_confidence_cap <= 1.0:
            raise ValueError(
                f'one_star_confidence_cap must lie in [0, 1]; got {self._one_star_confidence_cap!r}'
            )
        if not 0.0 <= self._two_star_confidence_cap <= 1.0:
            raise ValueError(
                f'two_star_confidence_cap must lie in [0, 1]; got {self._two_star_confidence_cap!r}'
            )

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Report feasibility based on the predictable-star cohort.

        Reads only feature metadata; never any pixels.  Feasible when at
        least one usable STAR feature is in the input set; the navigate
        path then decides whether the 1-star or 2-star branch wins.
        """
        usable = usable_stars(features)
        if not usable:
            return NavFeasibilityReport(feasible=False, reason='no_usable_star_features')
        consumed = min(len(usable), 2)
        return NavFeasibilityReport(feasible=True, reason='ok', consumed_feature_count=consumed)

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Recover translation from a cross-checked star pair or a uniquely bright star.

        The pair search runs first (see :meth:`_try_pairs`); the 1-star path runs
        only when no pair passes.  Passing pairs that disagree make the result
        spurious without trying the 1-star path.

        Parameters:
            features: STAR features (the orchestrator pre-filters by type).
            context: Per-image NavContext.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence (capped per-mode by
            ``one_star_confidence_cap`` / ``two_star_confidence_cap``),
            and a populated :class:`StarUniqueMatchDiagnostics`.  The
            covariance shape and rotation fields depend on the chosen
            path and on ``context.fit_camera_rotation``:

            - **1-star path, fit_camera_rotation=False**: ``(2, 2)``
              covariance; ``rotation_rad`` and ``sigma_rotation_rad``
              are ``None``.
            - **1-star path, fit_camera_rotation=True**: rank-deficient
              ``(3, 3)`` covariance via
              :func:`~spindoctor.nav_technique.nav_technique.embed_rotation_unobservable`
              (a single match cannot constrain rotation);
              ``rotation_rad = 0.0`` and ``sigma_rotation_rad`` is the
              rotation-unobservable sentinel.
            - **2-star path, fit_camera_rotation=False**: ``(2, 2)``
              covariance from the per-feature CRLB floor; rotation
              fields ``None``.
            - **2-star path, fit_camera_rotation=True**: full ``(3, 3)``
              covariance with the analytic
              ``2 * (sigma_v**2 + sigma_u**2) / L**2`` rotation
              diagonal (``L`` is the catalog-pair separation);
              ``rotation_rad`` is the Procrustes-fit angle and
              ``sigma_rotation_rad`` is the square root of the rotation
              diagonal.
        """
        with self.log_section(f'TECHNIQUE: {self.name}'):
            usable = usable_stars(features)
            # Every star is predicted inside the extended field of view, whose padding
            # is the instrument's offset window, so every star's window reaches the
            # image and none is set aside for being out of reach.
            self.logger.info(
                'Consuming %d usable STAR feature(s) (out of %d offered)',
                len(usable),
                len(features),
            )
            if not usable:
                return self._fail(
                    features=features,
                    diagnostics=StarUniqueMatchDiagnostics(
                        mode='',
                        predicted_snr=0.0,
                        brightness_margin_mag=0.0,
                        residual_px=0.0,
                    ),
                    reason='no_usable_star_features',
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            ranked = sorted(usable, key=predicted_snr, reverse=True)
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            if len(ranked) >= 2:
                pair_result = self._try_pairs(ranked, features, image_ext, noise_sigma, context)
                if pair_result is not None:
                    return pair_result
            return self._try_one_star(ranked[0], ranked[1:], image_ext, noise_sigma, context)

    def _try_pairs(
        self,
        ranked: list[NavFeature],
        features: list[NavFeature],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        context: NavContext,
    ) -> NavTechniqueResult | None:
        """Find the offset the most of the brightest usable stars agree on.

        Each of the ``pair_search_stars`` brightest usable stars is searched for over the
        instrument's whole offset window, and its ``pair_search_sources`` strongest
        isolated point sources are its candidates (:meth:`_star_sources`).  Every pair of
        candidates, one in each of two stars' windows, whose residual is within
        ``max_residual_px`` is a cross-checked pair (:func:`pair_hypotheses`), supported by
        every searched star with a candidate within ``2 * max_residual_px`` of where the
        pair's solution puts that star.  Pairs are judged best supported first, and a pair
        is set aside when the other stars rule it out (:meth:`_ruled_out`).  The best
        supported pair not set aside is the answer; of several equally supported pairs
        more than ``2 * max_residual_px`` apart, the one whose sources are brightest for
        the stars' predicted SNR (:meth:`_pair_brightness`).  It is spurious when another
        of them is within ``pair_vote_peak_factor`` of that brightness: the stars back two
        answers equally, one of them a coincidence that cannot be told from the other.

        Parameters:
            ranked: Usable star features, brightest predicted SNR first.
            features: Every feature offered, for the spurious result's IDs.
            image_ext: The extended-FOV image.
            noise_sigma: Image noise sigma used by the detection threshold.
            context: Per-image NavContext.

        Returns:
            The best supported pair's result, a spurious result when it is ambiguous or
            every pair is ruled out, or ``None`` when no pair passes the residual
            cross-check (the caller then tries the 1-star path).
        """
        searched = ranked[: self._pair_search_stars]
        window_vu = self._search_window(context)
        sources = [
            self._star_sources(image_ext, star, noise_sigma, search_half_vu=window_vu)
            for star in searched
        ]
        hypotheses = pair_hypotheses(
            [predicted_vu(star) for star in searched],
            sources,
            max_residual_px=self._max_residual_px,
            fit_rotation=bool(context.fit_camera_rotation),
        )
        self.logger.info(
            'Two-star path: %d cross-checked pair(s) among the sources of %d star(s) '
            '(%s source(s) each)',
            len(hypotheses),
            len(searched),
            '/'.join(str(len(s)) for s in sources),
        )
        if not hypotheses:
            return None
        tolerance_px = 2.0 * self._max_residual_px
        reasons: list[str] = []
        clusters: list[PairHypothesis] = []
        refuted: list[tuple[float, float]] = []

        def near(hyp: PairHypothesis, offsets: list[tuple[float, float]]) -> bool:
            return any(
                math.hypot(hyp.solved.offset_vu[0] - v, hyp.solved.offset_vu[1] - u) <= tolerance_px
                for v, u in offsets
            )

        for hyp in hypotheses:
            if clusters and hyp.support < clusters[0].support:
                break
            if near(hyp, [c.solved.offset_vu for c in clusters]) or near(hyp, refuted):
                continue
            ruling = self._ruled_out(hyp, image_ext, searched, sources, noise_sigma, context)
            if ruling is not None:
                reason, offset_refuted = ruling
                reasons.append(reason)
                if offset_refuted:
                    refuted.append(hyp.solved.offset_vu)
                continue
            clusters.append(hyp)
        if not clusters:
            return self._fail(
                features=features,
                diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
                reason=f'pair_contradicted: every cross-checked pair is ruled out ({reasons[0]})',
                fit_rotation=bool(context.fit_camera_rotation),
            )
        # Of equally supported answers, the stars are the brightest one: a ghost image or
        # a coincidence among faint sources shows far less light per unit of predicted
        # brightness.  An answer within pair_vote_peak_factor of the brightest is a rival.
        brightness = [self._pair_brightness(image_ext, h, searched) for h in clusters]
        known = [b for b in brightness if b is not None]
        best_index = (
            max(range(len(clusters)), key=lambda k: brightness[k] or 0.0)
            if len(known) == len(clusters)
            else 0
        )
        best = clusters[best_index]
        best_brightness = brightness[best_index]
        rivals = sum(
            1
            for k, b in enumerate(brightness)
            if k != best_index
            and (
                b is None
                or best_brightness is None
                or b * self._pair_vote_peak_factor >= best_brightness
            )
        )
        i, j = best.star_indices
        self.logger.info(
            'Two-star path: the pair is backed by %d of %d searched star(s); source ranks %s; '
            '%d pair(s) ruled out',
            best.support,
            len(searched),
            best.ranks,
            len(reasons),
        )
        if rivals:
            return self._fail(
                features=features,
                diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
                reason=(
                    f'ambiguous_pairs: {rivals} pair(s) backed by as many stars put the '
                    f'offset more than {tolerance_px:.1f} px from the chosen pair'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        return self._pair_result(
            [searched[i], searched[j]],
            [f for f in ranked if f is not searched[i] and f is not searched[j]],
            best.solved,
            context,
        )

    def _pair_brightness(
        self, image_ext: NDArrayFloatType, hyp: PairHypothesis, searched: list[NavFeature]
    ) -> float | None:
        """Return a pair's sources' peak per unit of predicted SNR (geometric mean).

        Parameters:
            image_ext: The extended-FOV image.
            hyp: The pair.
            searched: The searched stars, brightest first.

        Returns:
            The brightness, or ``None`` when a star's predicted SNR or catalog magnitude
            cannot say how bright it should be, or a source shows no peak.
        """
        log_sum = 0.0
        for k, det in zip(hyp.star_indices, hyp.detections, strict=True):
            star = searched[k]
            snr = predicted_snr(star)
            peak = source_peak(image_ext, det, box_half_px=self._centroid_box_half_px)
            if snr <= 0.0 or peak <= 0.0 or photometry_saturated(star):
                return None
            log_sum += math.log(peak / snr)
        return math.exp(log_sum / 2.0)

    def _ruled_out(
        self,
        hyp: PairHypothesis,
        image_ext: NDArrayFloatType,
        searched: list[NavFeature],
        sources: list[list[tuple[float, float]]],
        noise_sigma: float,
        context: NavContext,
    ) -> tuple[str, bool] | None:
        """Return why the stars rule out a cross-checked pair, or None when they do not.

        The pair's two stars set the brightness scale (:func:`peak_per_snr`), the peak per
        unit of predicted SNR, from the sources found where the pair's solution puts them.
        The pair is ruled out when:

        - its two stars' peaks per unit of predicted SNR differ by more than the square of
          ``pair_vote_peak_factor``, the most two stars can differ when each lies within
          that factor of the scale they share: a star paired with a faint field source
          or a cosmic-ray hit is not a pair of stars;
        - with no other searched star found as bright as it should be where the pair puts
          it, at least ``pair_contradiction_stars`` distinct sources, each the one source
          as bright as its star should be in that star's window, lie elsewhere
          (:func:`third_star_votes`); two stars can agree by coincidence, and three found
          elsewhere say they did; or
        - more other searched stars are absent where the pair puts them than are found
          there, and at least ``pair_absent_stars`` of them (:func:`absent_stars`): each is
          at least as bright as the pair's fainter star and lies on clear sky in the image,
          so it should be seen.

        Parameters:
            hyp: The pair.
            image_ext: The extended-FOV image.
            searched: The searched stars, brightest first.
            sources: Each searched star's sources, strongest first.
            noise_sigma: Image noise sigma used by the detection threshold.
            context: Per-image NavContext.

        Returns:
            ``None``, or the reason and whether it refutes the pair's offset, not just the
            pair: the other stars' votes and absences speak to the offset, so every pair
            within ``2 * max_residual_px`` of a refuted offset is set aside with it, while
            a pair refused for its own photometry leaves other pairs at its offset to be
            judged.
        """
        tolerance_px = 2.0 * self._max_residual_px
        tolerance_vu = (tolerance_px, tolerance_px)
        i, j = hyp.star_indices
        pair = [searched[i], searched[j]]
        quotients = [
            source_peak(image_ext, det, box_half_px=self._centroid_box_half_px)
            / max(predicted_snr(star), 1e-30)
            for star, det in zip(pair, hyp.detections, strict=True)
        ]
        limit = self._pair_vote_peak_factor**2
        if (
            not any(photometry_saturated(star) for star in pair)
            and min(quotients) > 0.0
            and max(quotients) > limit * min(quotients)
        ):
            return (
                f'its stars show {quotients[0]:.3g} and {quotients[1]:.3g} per unit of '
                f'predicted SNR, more than {limit:.0f} times apart',
                False,
            )
        others = [f for k, f in enumerate(searched) if k not in (i, j)]
        source_lists = {id(f): found for f, found in zip(searched, sources, strict=True)}
        place = hyp.solved.place
        scale = peak_per_snr(
            image_ext,
            pair,
            place,
            detect_at=lambda star, target: self._isolated_detection_at(
                image_ext, star, target, noise_sigma, search_half_vu=tolerance_vu
            ),
            box_half_px=self._centroid_box_half_px,
        )
        agree, disagree = third_star_votes(
            others,
            place,
            detect=lambda star: star_like_source(
                image_ext,
                star,
                source_lists[id(star)],
                peak_per_snr=scale,
                peak_factor=self._pair_vote_peak_factor,
                box_half_px=self._centroid_box_half_px,
            ),
            detect_near=lambda star, target: self._star_like_detection_at(
                image_ext,
                star,
                target,
                noise_sigma,
                search_half_vu=tolerance_vu,
                peak_per_snr=scale,
            ),
            tolerance_px=tolerance_px,
        )
        if agree == 0 and disagree >= self._pair_contradiction_stars:
            return (
                f'{disagree} distinct source(s) as bright as other stars should be lie away '
                f"from the pair's offset and none at it",
                True,
            )
        absent = absent_stars(
            image_ext,
            others,
            place,
            faint_snr=min(predicted_snr(f) for f in pair),
            peak_per_snr=scale,
            peak_factor=self._pair_vote_peak_factor,
            margin_vu=search_window_for_obs(context),
            noise_sigma=noise_sigma,
            box_half_px=self._centroid_box_half_px,
        )
        if absent >= self._pair_absent_stars and absent > agree:
            return (
                f"{absent} star(s) at least as bright as the pair's fainter star show "
                f'nothing where the pair puts them in the image, {agree} found there',
                True,
            )
        return None

    def _search_window(self, context: NavContext) -> tuple[float, float]:
        """Return the ``(v, u)`` half-widths every star is searched for over.

        Pointing error is a property of the camera, so a star is searched for over the
        instrument's offset window (:func:`search_window_for_obs`), axis by axis, as every
        other technique searches.

        Parameters:
            context: Per-image NavContext.

        Returns:
            The offset window's ``(v, u)`` half-widths in pixels.
        """
        margin_v, margin_u = search_window_for_obs(context)
        return float(margin_v), float(margin_u)

    def _star_sources(
        self,
        image_ext: NDArrayFloatType,
        star: NavFeature,
        noise_sigma: float,
        *,
        search_half_vu: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Return the centroids of the strongest isolated point sources in a star's window.

        Parameters:
            image_ext: The extended-FOV image.
            star: The star; its expected smear widens the isolation annulus.
            noise_sigma: Image noise sigma used by the detection threshold.
            search_half_vu: ``(v, u)`` half-widths of the window around its prediction.

        Returns:
            Up to ``pair_search_sources`` centroids, strongest first.
        """
        bounds = search_window_bounds(predicted_vu(star), search_half_vu, image_ext.shape)
        if bounds is None:
            return []
        inner_px, outer_px = isolation_radii(
            self._isolation_inner_px, self._isolation_outer_px, smear_length_px(star)
        )
        found = window_sources(
            image_ext,
            predicted_vu(star),
            search_half_vu=search_half_vu,
            centroid_box_half_px=self._centroid_box_half_px,
            psf_sigma_px=self._psf_sigma_px,
            max_peak_sharpness=self._max_peak_sharpness,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
            accept=isolation_lookup(
                image_ext,
                bounds,
                sigma_px=self._psf_sigma_px,
                kernel_half_px=self._centroid_box_half_px,
                inner_px=inner_px,
                outer_px=outer_px,
                max_ratio=self._isolation_max_ratio,
            ),
            max_sources=self._pair_search_sources,
        )
        return [s.centroid_vu for s in found]

    def _isolated_detection_at(
        self,
        image_ext: NDArrayFloatType,
        star: NavFeature,
        center_vu: tuple[float, float],
        noise_sigma: float,
        *,
        search_half_vu: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return the isolated point source in a window around ``center_vu``, else None.

        The window's strongest peak is measured by :func:`local_centroid`, and it must be an
        isolated point source; ``star`` supplies the expected smear the isolation test allows for.
        """
        det, _peak = local_centroid(
            image_ext,
            center_vu,
            search_half_vu=search_half_vu,
            centroid_box_half_px=self._centroid_box_half_px,
            psf_sigma_px=self._psf_sigma_px,
            max_peak_sharpness=self._max_peak_sharpness,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
            refuse_outshone=False,
        )
        if det is None or not self._is_point_source(image_ext, det, smear_px=smear_length_px(star)):
            return None
        return det

    def _star_like_detection_at(
        self,
        image_ext: NDArrayFloatType,
        star: NavFeature,
        center_vu: tuple[float, float],
        noise_sigma: float,
        *,
        search_half_vu: tuple[float, float],
        peak_per_snr: float | None,
    ) -> tuple[float, float] | None:
        """Return the point source near ``center_vu`` when it is as bright as ``star``.

        The source must be the isolated point source :meth:`_isolated_detection_at` finds,
        and its peak above the local background must lie within a factor of
        ``pair_vote_peak_factor`` of ``peak_per_snr`` times the star's predicted SNR, the
        peak the pair's own stars show for a star of that catalog brightness.  A faint
        field source or a cosmic-ray hit that happens to lie where the star should be is
        then not taken for it.

        Parameters:
            image_ext: The extended-FOV image.
            star: The star sought.
            center_vu: Where the star should be, ``(v, u)``.
            noise_sigma: Image noise sigma used by the detection threshold.
            search_half_vu: ``(v, u)`` half-widths of the window searched around
                ``center_vu``.
            peak_per_snr: The pair's peak per unit of predicted SNR
                (:func:`peak_per_snr`), or ``None`` when the pair gave none.

        Returns:
            The source's centroid, or ``None`` when there is no such source, its peak is
            outside the factor, ``peak_per_snr`` is ``None``, or the star's predicted SNR
            or catalog magnitude cannot say how bright it should be.
        """
        snr = predicted_snr(star)
        if peak_per_snr is None or snr <= 0.0 or photometry_saturated(star):
            return None
        det = self._isolated_detection_at(
            image_ext, star, center_vu, noise_sigma, search_half_vu=search_half_vu
        )
        if det is None:
            return None
        ratio = source_peak(image_ext, det, box_half_px=self._centroid_box_half_px) / (
            peak_per_snr * snr
        )
        if not 1.0 / self._pair_vote_peak_factor <= ratio <= self._pair_vote_peak_factor:
            return None
        return det

    def _pair_result(
        self,
        chosen: list[NavFeature],
        rest: list[NavFeature],
        solved: TwoStarAssignment,
        context: NavContext,
    ) -> NavTechniqueResult:
        """Return the result of a pair whose sources passed the cross-check.

        Parameters:
            chosen: The two star features matched, brighter first.
            rest: The other usable stars, brightest first, for the brightness-margin
                diagnostic.
            solved: The pair solved with each source assigned to its own star.
            context: Per-image NavContext.

        Returns:
            The pair's result.
        """
        fit_rotation = bool(context.fit_camera_rotation)
        offset_v, offset_u = solved.offset_vu
        residual_px = solved.residual_px
        rotation_rad = solved.rotation_rad
        # Brightness-margin diagnostic: magnitude difference to the
        # next-brightest *unmatched* predictable star.  When only two
        # predictable stars exist (no third star in ``rest``) the
        # diagnostic reports ``+inf`` — there is no unmatched star to
        # compare against, which is the strongest possible margin.
        brightest_snr = predicted_snr(chosen[0])
        next_snr = predicted_snr(rest[0]) if rest else 0.0
        margin_mag = brightness_margin_mag(brightest_snr, next_snr)
        diagnostics = StarUniqueMatchDiagnostics(
            mode='two_star',
            predicted_snr=brightest_snr,
            brightness_margin_mag=margin_mag,
            residual_px=residual_px,
        )
        margin_v, margin_u = search_window_for_obs(context)
        max_rotation_rad = math.radians(context.max_rotation_deg)
        rotation_at_edge = fit_rotation and (
            rotation_rad is not None
            and abs(rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
        )
        at_edge = (
            abs(offset_v) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_u) >= margin_u - self._at_edge_tolerance_px
            or rotation_at_edge
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics,
            at_edge=at_edge,
            spurious=False,
            cap=self._two_star_confidence_cap,
        )
        self.logger.info(
            'Two-star path: stars %s and %s; offset (%.4f, %.4f) px; residual '
            '%.4f px; brightness margin %.3f mag; confidence %.4f',
            chosen[0].feature_id,
            chosen[1].feature_id,
            offset_v,
            offset_u,
            residual_px,
            margin_mag,
            confidence,
        )
        feature_ids = (chosen[0].feature_id, chosen[1].feature_id)
        cov = two_star_covariance(
            chosen[0], chosen[1], residual_px=residual_px, fit_rotation=fit_rotation
        )
        sigma_rotation_rad: float | None = None
        if fit_rotation:
            sigma_rotation_rad = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
            self.logger.info(
                'Rotation = %+.4f deg (sigma %.4f deg)%s',
                math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                math.degrees(sigma_rotation_rad),
                ', AT_EDGE' if rotation_at_edge else '',
            )
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=feature_ids,
            offset_px=(offset_v, offset_u),
            covariance_px2=cov,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=rotation_rad,
            sigma_rotation_rad=sigma_rotation_rad,
        )

    def _try_one_star(
        self,
        brightest: NavFeature,
        rest: list[NavFeature],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        context: NavContext,
    ) -> NavTechniqueResult:
        """Attempt the 1-star path with its acceptance gates.

        Acceptance requires, in order: the brightness-margin uniqueness
        gate, a detection above the noise threshold, the
        peak-to-runner-up ambiguity gate, and -- only when that
        ratio is the infinite no-rival sentinel -- the residual gate
        bounding the prediction-to-detection distance by
        ``one_star_max_residual_px``.  The sentinel means the ambiguity
        gate measured nothing (no runner-up ever cleared the window
        background), so the residual gate carries the acceptance burden
        there; a finite ratio was measured against real background
        statistics and acceptance stays with the ambiguity gate.
        """
        search_half_vu = self._search_window(context)
        ellipse_v = self._one_star_ellipse_fraction * search_half_vu[0]
        ellipse_u = self._one_star_ellipse_fraction * search_half_vu[1]
        brightest_snr = predicted_snr(brightest)
        next_snr = predicted_snr(rest[0]) if rest else 0.0
        margin_mag = brightness_margin_mag(brightest_snr, next_snr)
        if margin_mag < self._brightness_margin_floor_mag:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
            )
            return self._fail(
                features=[brightest, *rest],
                diagnostics=diagnostics,
                reason=(
                    f'brightness_margin {margin_mag:.3f} mag below floor '
                    f'{self._brightness_margin_floor_mag:.3f} mag'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        det, peak = local_centroid(
            image_ext,
            predicted_vu(brightest),
            search_half_vu=search_half_vu,
            centroid_box_half_px=self._centroid_box_half_px,
            psf_sigma_px=self._psf_sigma_px,
            max_peak_sharpness=self._max_peak_sharpness,
            image_noise_sigma=noise_sigma,
            detection_sigma=self._detection_sigma,
            refuse_outshone=False,
        )
        if det is None:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=f'no_detection_above_threshold (peak {peak:.3f} DN)',
                fit_rotation=bool(context.fit_camera_rotation),
            )
        pred_v, pred_u = predicted_vu(brightest)
        refusal = None
        if not self._is_point_source(image_ext, det, smear_px=smear_length_px(brightest)):
            refusal = (
                'window_peak_not_a_point_source (part of an extended source, or a cosmic-ray track)'
            )
        elif ((det[0] - pred_v) * ellipse_u) ** 2 + ((det[1] - pred_u) * ellipse_v) ** 2 > (
            ellipse_v * ellipse_u
        ) ** 2:
            # The whole window is searched so that a brighter source in it vetoes
            # a fainter one near the prediction; a lone detection is accepted
            # only inside the lone-star ellipse, since nothing cross-checks it and the
            # wider the region it may come from, the likelier it is an artifact.
            # The instrument window is per axis, so the ellipse's semi-axes are its
            # half-widths scaled by one_star_ellipse_fraction: no single radius is
            # taken from the two, the ellipse is a circle when they are equal, and
            # it never reaches past the window on either axis.  The test is written
            # without division, so a zero margin on an axis admits only a zero
            # offset on it.
            refusal = (
                f'one_star_detection_outside_ellipse (offset '
                f'({det[0] - pred_v:.3f}, {det[1] - pred_u:.3f}) px, outside the '
                f'({ellipse_v:.1f}, {ellipse_u:.1f}) px lone-star ellipse)'
            )
        if refusal is not None:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=refusal,
                fit_rotation=bool(context.fit_camera_rotation),
            )
        # A detection that barely beats the window's noise order
        # statistics carries no positional information even though it
        # cleared the absolute detection threshold: with a marginal
        # star, the brightest pixel of a multi-thousand-pixel window is
        # routinely a noise spike within a few percent of its
        # runner-up.  The single-detection premise of this path demands
        # the peak be unambiguous, so an ambiguous one is spurious.
        peak_ratio = detection_peak_ratio(
            image_ext,
            (pred_v, pred_u),
            det,
            search_half_vu=search_half_vu,
            exclude_half_px=self._centroid_box_half_px + 2,
        )
        if peak_ratio < self._one_star_min_peak_ratio:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=0.0,
                detection_peak_ratio=peak_ratio,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=(
                    f'ambiguous_detection (peak/runner-up {peak_ratio:.3f} below '
                    f'{self._one_star_min_peak_ratio:.3f})'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        offset_v = det[0] - pred_v
        offset_u = det[1] - pred_u
        residual_px = math.hypot(offset_v, offset_u)
        # Residual gate (#259), applied only when the #211 ambiguity
        # gate is vacuous: an infinite peak ratio means no runner-up
        # ever cleared the window background, so the gate measured
        # nothing about the detection's uniqueness -- the signature of
        # a lone hot pixel / artifact on a flat or quantized frame, not
        # of a star on a real noisy background (whose runner-up is
        # always finite).  With no rival statistics to lean on, and the
        # residual being the claimed offset itself, acceptance demands
        # the detection sit inside the pointing-prior core bounded by
        # ``one_star_max_residual_px``.  A finite ratio keeps the
        # measured #211 gate authoritative: genuine library matches
        # carry one-star offsets up to ~24 px, so this bound cannot hold
        # every match; every one is held to the lone-star ellipse above.
        if not math.isfinite(peak_ratio) and residual_px > self._one_star_max_residual_px:
            diagnostics = StarUniqueMatchDiagnostics(
                mode='one_star',
                predicted_snr=brightest_snr,
                brightness_margin_mag=margin_mag,
                residual_px=residual_px,
                detection_peak_ratio=peak_ratio,
            )
            return self._fail(
                features=[brightest],
                diagnostics=diagnostics,
                reason=(
                    f'no_rival_detection: residual {residual_px:.3f} px exceeds '
                    f'one_star_max_residual_px {self._one_star_max_residual_px:.3f} '
                    f'and the peak-ratio gate is vacuous (no runner-up above background)'
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        # Even though "residual_px" is really the offset magnitude here
        # (the 1-star path has no second observation to subtract), it is
        # the only honest measure of how far we travelled to find the
        # detection inside the search window.  The technique reports it
        # so the confidence formula can penalize large excursions.
        diagnostics = StarUniqueMatchDiagnostics(
            mode='one_star',
            predicted_snr=brightest_snr,
            brightness_margin_mag=margin_mag,
            residual_px=residual_px,
            detection_peak_ratio=peak_ratio,
        )
        margin_v, margin_u = search_window_for_obs(context)
        # The detection lies inside the lone-star ellipse, so this fires only when
        # one_star_ellipse_fraction brings the ellipse within at_edge_tolerance_px of
        # the window's edge.
        at_edge = (
            abs(offset_v) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_u) >= margin_u - self._at_edge_tolerance_px
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics,
            at_edge=at_edge,
            spurious=False,
            cap=self._one_star_confidence_cap,
        )
        reported_det = reported_position_vu(det, (margin_v, margin_u))
        self.logger.info(
            'One-star path: matched %s at (%.4f, %.4f); offset (%.4f, %.4f) '
            'px; brightness margin %.3f mag; confidence %.4f',
            brightest.feature_id,
            reported_det[0],
            reported_det[1],
            offset_v,
            offset_u,
            margin_mag,
            confidence,
        )
        cov = build_star_covariance(brightest, residual_px=residual_px)
        fit_rotation = bool(context.fit_camera_rotation)
        cov_out = embed_rotation_unobservable(cov) if fit_rotation else cov
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(brightest.feature_id,),
            offset_px=(offset_v, offset_u),
            covariance_px2=cov_out,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
        )

    def _is_point_source(
        self, image_ext: NDArrayFloatType, detection_vu: tuple[float, float], *, smear_px: float
    ) -> bool:
        """Return whether a window's detection is an isolated point source of the image.

        The detection is already a matched-filter peak of the image (:func:`local_centroid`
        takes only local maxima of the response), so the remaining test is isolation
        (:func:`is_isolated_point_source`, the test the star field's source detection
        applies): a ridge of a ring or a limb, or a cosmic-ray track, is a peak but not a
        star.  The annulus moves out by half the star's expected smear
        (:func:`isolation_radii`), so a smeared star's own streak does not fail it.

        Parameters:
            image_ext: The extended-FOV image.
            detection_vu: ``(v, u)`` centroid :func:`local_centroid` returned.
            smear_px: The star's expected smear length in pixels.

        Returns:
            True when the detection is isolated.
        """
        inner_px, outer_px = isolation_radii(
            self._isolation_inner_px, self._isolation_outer_px, smear_px
        )
        return is_isolated_point_source(
            image_ext,
            detection_vu,
            sigma_px=self._psf_sigma_px,
            kernel_half_px=self._centroid_box_half_px,
            inner_px=inner_px,
            outer_px=outer_px,
            max_ratio=self._isolation_max_ratio,
        )

    def _evaluate_confidence(
        self,
        *,
        diagnostics: StarUniqueMatchDiagnostics,
        at_edge: bool,
        spurious: bool,
        cap: float,
    ) -> float:
        """Run the YAML confidence formula and apply the per-mode cap."""
        assert self.confidence_spec is not None
        confidence, breakdown = evaluate_sigmoid_combination(
            self.confidence_spec,
            _UniqueMatchConfidenceContext(
                at_edge=at_edge, spurious=spurious, diagnostics=diagnostics
            ),
            technique_name=self.name,
            return_breakdown=True,
        )
        log_confidence_breakdown(self.logger, breakdown)
        return min(float(confidence), cap)

    def _fail(
        self,
        *,
        features: list[NavFeature],
        diagnostics: StarUniqueMatchDiagnostics,
        reason: str,
        fit_rotation: bool = False,
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result with the supplied reason."""
        self.logger.info('Reporting spurious result: %s', reason)
        return self._spurious_result(
            feature_ids=tuple(f.feature_id for f in features),
            diagnostics=diagnostics,
            fit_rotation=fit_rotation,
        )
