"""``StarUniqueMatchNav`` -- translation fit from one star or a cross-checked pair.

Two paths share one technique:

- **One-star path.**  When no two stars agree on an offset, each searched star paired
  with each of its candidate sources is a hypothesis about the offset (see
  ``_star_unique_one_star``).  A source sharper than a star outright
  (``one_star_max_peak_sharpness``, allowing nothing for noise) is no lone star, and an
  offset is refuted when ``one_star_absent_stars`` other stars it puts on clear sky in the
  image show nothing there.  Of the rest, the hypothesis brightest for its star's
  predicted SNR is the answer, unless another source of that star's window is within
  ``one_star_min_peak_ratio`` of its peak, another star's surviving hypothesis elsewhere is
  within ``pair_vote_peak_factor`` of its brightness, or another star within
  ``brightness_margin_to_next_catalog_star_mag`` of it has the source in its window.  The
  answer must also lie inside the lone-star ellipse (the camera's
  ``one_star_ellipse_fraction`` of the window), and, when no pixel of the window outside
  the source clears the background, within ``one_star_max_residual_px`` of the
  prediction: an infinite peak ratio is the signature of a lone hot pixel on a flat or
  quantized frame.  Confidence is capped at the configured one-star limit (default 0.7)
  because a single match cannot cross-check itself.

- **Two-star path.**  Tried first whenever two or more stars are usable.
  Each of the ``pair_search_stars`` brightest usable stars is searched for
  over the instrument's whole offset window, and its
  ``pair_search_sources`` strongest isolated point sources are its
  candidates: the star need not be the strongest thing in a wide window.
  Every two candidates of two different stars that agree within
  ``max_residual_px`` form a cross-checked pair, backed by every searched
  star with a candidate where the pair's offset puts it.  An offset the
  other stars refute (three found elsewhere, as bright as they should be,
  and none at it; or stars at least as bright as the pair's fainter star,
  on clear sky there, showing nothing) is set aside.  The best backed
  surviving pair is the answer, and of equally backed ones the brightest
  for its stars' catalog magnitudes; another within a factor of
  ``pair_vote_peak_factor`` of that makes the result spurious.  Confidence
  is capped at the configured two-star limit (default 0.8).  Searching beyond the
  brightest two is what lets a frame navigate when its brightest predicted
  star is not in the image -- off the detector, or lost in a ring's glare.

Local-window detection is intentional: the search window is the
instrument's offset window, sized to bracket the SPICE pointing error, so
the star lies inside it.  A detection is a matched-filter peak of the
image, so the flank of a brighter source just outside the window is never
taken, and it must be an isolated point source: a ridge of a ring or a
cosmic-ray track is a peak but not a star.  No global star-detection pass
is needed (and this technique is feasible even on images where multi-star
detection would fail because the rest of the field is too faint or too
crowded).  Every star is predicted inside the extended field of view, whose
padding is the instrument's offset window, so every star's window reaches
the image.
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
    distinct_stars,
    local_centroid,
    predicted_snr,
    predicted_vu,
    search_window_bounds,
    smear_length_px,
    too_sharp,
    usable_stars,
    window_sources,
)
from spindoctor.nav_technique._star_isolation import (
    is_isolated_point_source,
    isolation_lookup,
    isolation_radii,
)
from spindoctor.nav_technique._star_unique_one_star import (
    choose_lone_star,
    lone_star_hypotheses,
)
from spindoctor.nav_technique._star_unique_two_star import (
    PairHypothesis,
    TwoStarAssignment,
    build_star_covariance,
    choose_pair,
    pair_brightness,
    pair_hypotheses,
    photometry_saturated,
    refute_pair,
    source_peak,
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
        requires_prior: ``False`` -- runs in pass 1.
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
        self._one_star_max_peak_sharpness = float(self.tuning['one_star_max_peak_sharpness'])
        self._one_star_absent_stars = int(self.tuning['one_star_absent_stars'])
        for key in ('pair_search_sources', 'pair_absent_stars', 'one_star_absent_stars'):
            if int(self.tuning[key]) < 1:
                raise ValueError(f'{key} must be >= 1; got {self.tuning[key]!r}')
        if not 1.0 < self._pair_vote_peak_factor < math.inf:
            raise ValueError(
                f'pair_vote_peak_factor must be finite and > 1; got {self._pair_vote_peak_factor!r}'
            )
        if not 1.0 <= self._one_star_max_peak_sharpness < math.inf:
            raise ValueError(
                'one_star_max_peak_sharpness must be finite and >= 1; '
                f'got {self._one_star_max_peak_sharpness!r}'
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
        only when no pair passes the residual cross-check.  An ambiguous or refuted pair
        search makes the result spurious without trying the 1-star path.

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
            # One star listed twice (#836) or in two catalogs (#365) is searched once.
            ranked = distinct_stars(sorted(usable, key=predicted_snr, reverse=True))
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            searched = ranked[: self._pair_search_stars]
            window_vu = self._search_window(context)
            sources = [
                self._star_sources(image_ext, star, noise_sigma, search_half_vu=window_vu)
                for star in searched
            ]
            if len(ranked) >= 2:
                pair_result = self._try_pairs(
                    ranked, searched, sources, features, image_ext, noise_sigma, context
                )
                if pair_result is not None:
                    return pair_result
            return self._try_one_star(ranked, searched, sources, image_ext, noise_sigma, context)

    def _try_pairs(
        self,
        ranked: list[NavFeature],
        searched: list[NavFeature],
        sources: list[list[tuple[float, float]]],
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
        the stars' predicted SNR (:func:`pair_brightness`).  It is spurious when another
        of them is within ``pair_vote_peak_factor`` of that brightness: the stars back two
        answers equally, one of them a coincidence that cannot be told from the other.

        Parameters:
            ranked: Usable star features, brightest predicted SNR first.
            searched: The ``pair_search_stars`` brightest of ``ranked``.
            sources: Each searched star's sources, strongest first (:meth:`_star_sources`).
            features: Every feature offered, for the spurious result's IDs.
            image_ext: The extended-FOV image.
            noise_sigma: Image noise sigma used by the detection threshold.
            context: Per-image NavContext.

        Returns:
            The best supported pair's result, a spurious result when it is ambiguous or
            every pair is ruled out, or ``None`` when no pair passes the residual
            cross-check (the caller then tries the 1-star path).
        """
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
        choice = choose_pair(
            hypotheses,
            refute=lambda hyp: self._ruled_out(
                hyp, image_ext, searched, sources, noise_sigma, context
            ),
            brightness=lambda hyp: pair_brightness(
                image_ext, hyp, searched, box_half_px=self._centroid_box_half_px
            ),
            tolerance_px=tolerance_px,
            rival_factor=self._pair_vote_peak_factor,
        )
        best, rivals, reasons = choice.best, choice.rivals, choice.reasons
        if best is None:
            return self._fail(
                features=features,
                diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
                reason=f'pair_contradicted: every cross-checked pair is ruled out ({reasons[0]})',
                fit_rotation=bool(context.fit_camera_rotation),
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

    def _ruled_out(
        self,
        hyp: PairHypothesis,
        image_ext: NDArrayFloatType,
        searched: list[NavFeature],
        sources: list[list[tuple[float, float]]],
        noise_sigma: float,
        context: NavContext,
    ) -> str | None:
        """Return why the other stars refute a pair's offset (:func:`refute_pair`), or None.

        Parameters:
            hyp: The pair.
            image_ext: The extended-FOV image.
            searched: The searched stars, brightest first.
            sources: Each searched star's sources, strongest first.
            noise_sigma: Image noise sigma used by the detection threshold.
            context: Per-image NavContext.

        Returns:
            The reason, or ``None``.
        """
        return refute_pair(
            image_ext,
            hyp,
            searched,
            sources,
            detect_at=lambda star, target, half_vu: self._isolated_detection_at(
                image_ext, star, target, noise_sigma, search_half_vu=half_vu
            ),
            detect_star_like_at=lambda star, target, half_vu, scale: self._star_like_detection_at(
                image_ext, star, target, noise_sigma, search_half_vu=half_vu, peak_per_snr=scale
            ),
            margin_vu=search_window_for_obs(context),
            noise_sigma=noise_sigma,
            max_residual_px=self._max_residual_px,
            peak_factor=self._pair_vote_peak_factor,
            contradiction_stars=self._pair_contradiction_stars,
            absent_stars_needed=self._pair_absent_stars,
            box_half_px=self._centroid_box_half_px,
        )

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
        # diagnostic reports ``+inf`` -- there is no unmatched star to
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
        ranked: list[NavFeature],
        searched: list[NavFeature],
        sources: list[list[tuple[float, float]]],
        image_ext: NDArrayFloatType,
        noise_sigma: float,
        context: NavContext,
    ) -> NavTechniqueResult:
        """Attempt the 1-star path over every searched star's sources.

        Each searched star paired with each of its sources is a hypothesis about the
        offset, judged by the other stars (:func:`lone_star_hypotheses`), and the
        survivors are chosen among (:func:`choose_lone_star`).  The chosen source must
        then not be within reach of another star of comparable brightness
        (``brightness_margin_to_next_catalog_star_mag``), which could equally be the
        source; must lie inside the lone-star ellipse; and, when no pixel of the window
        outside the source clears the background (the infinite no-rival sentinel of the
        peak-to-runner-up ratio), must lie within ``one_star_max_residual_px`` of the
        prediction.

        Parameters:
            ranked: Usable star features, brightest predicted SNR first.
            searched: The ``pair_search_stars`` brightest of ``ranked``.
            sources: Each searched star's sources, strongest first.
            image_ext: The extended-FOV image.
            noise_sigma: Image noise sigma used by the detection threshold.
            context: Per-image NavContext.

        Returns:
            The lone star's result, or a spurious result naming why there is none.
        """
        fit_rotation = bool(context.fit_camera_rotation)
        search_half_vu = self._search_window(context)
        margin_v, margin_u = search_window_for_obs(context)
        half = self._centroid_box_half_px
        hypotheses = lone_star_hypotheses(
            image_ext,
            searched,
            sources,
            ranked,
            is_star_like=lambda vu: (
                not too_sharp(
                    image_ext,
                    round(vu[0]),
                    round(vu[1]),
                    half=half,
                    max_sharpness=self._one_star_max_peak_sharpness,
                    noise_sigma=noise_sigma,
                    detection_sigma=0.0,
                )
            ),
            margin_vu=(margin_v, margin_u),
            noise_sigma=noise_sigma,
            peak_factor=self._pair_vote_peak_factor,
            absent_stars_needed=self._one_star_absent_stars,
            box_half_px=half,
            max_peak_per_snr=self._max_peak_per_snr(context, noise_sigma),
        )
        choice = choose_lone_star(
            hypotheses,
            tolerance_px=2.0 * self._max_residual_px,
            runner_up_ratio=self._one_star_min_peak_ratio,
            rival_factor=self._pair_vote_peak_factor,
        )
        self.logger.info(
            'One-star path: %d hypothesis(es) from %d searched star(s)',
            len(hypotheses),
            len(searched),
        )
        best = choice.best
        if best is None:
            return self._fail(
                features=searched,
                diagnostics=StarUniqueMatchDiagnostics(
                    mode='one_star', predicted_snr=predicted_snr(ranked[0])
                ),
                reason=str(choice.reason),
                fit_rotation=fit_rotation,
            )
        star = searched[best.star_index]
        rest = [f for f in ranked if f is not star]
        star_snr = predicted_snr(star)
        margin_mag = brightness_margin_mag(star_snr, predicted_snr(rest[0]) if rest else 0.0)
        det = best.source_vu
        offset_v, offset_u = best.offset_vu
        ellipse_fraction = self._ellipse_fraction(context)
        ellipse_v = ellipse_fraction * search_half_vu[0]
        ellipse_u = ellipse_fraction * search_half_vu[1]
        peak_ratio = detection_peak_ratio(
            image_ext,
            predicted_vu(star),
            det,
            search_half_vu=search_half_vu,
            exclude_half_px=half + 2,
        )
        # The 1-star path has no second observation to subtract, so its residual is the
        # offset magnitude: the only measure of how far the match travelled, which the
        # confidence formula penalizes.
        residual_px = math.hypot(offset_v, offset_u)
        refusal = None
        rival = next(
            (
                f
                for f in rest
                if abs(brightness_margin_mag(star_snr, predicted_snr(f)))
                < self._brightness_margin_floor_mag
                and abs(det[0] - predicted_vu(f)[0]) <= search_half_vu[0]
                and abs(det[1] - predicted_vu(f)[1]) <= search_half_vu[1]
            ),
            None,
        )
        if rival is not None:
            # Another star of comparable brightness whose window holds the source could
            # be it as well; nothing in the image tells the two apart.
            refusal = (
                f'ambiguous_identity: {rival.feature_id}, within '
                f'{self._brightness_margin_floor_mag:.1f} mag of {star.feature_id}, could be '
                f'the source as well'
            )
        elif (offset_v * ellipse_u) ** 2 + (offset_u * ellipse_v) ** 2 > (
            ellipse_v * ellipse_u
        ) ** 2:
            # Nothing cross-checks a lone detection, and the wider the region it may come
            # from, the likelier it is an artifact, so it is believed only inside the
            # lone-star ellipse.  The instrument window is per axis, so the ellipse's
            # semi-axes are its half-widths scaled by the camera's
            # one_star_ellipse_fraction: no single radius is taken from the two, the
            # ellipse is a circle when they are equal, and it never reaches past the
            # window on either axis.  The test is written without division, so a zero
            # margin on an axis admits only a zero offset on it.
            refusal = (
                f'one_star_detection_outside_ellipse (offset '
                f'({offset_v:.3f}, {offset_u:.3f}) px, outside the '
                f'({ellipse_v:.1f}, {ellipse_u:.1f}) px lone-star ellipse)'
            )
        elif not math.isfinite(peak_ratio) and residual_px > self._one_star_max_residual_px:
            # Residual gate (#259): an infinite peak ratio means no pixel of the window
            # outside the source ever cleared the background -- the signature of a lone
            # hot pixel or artifact on a flat or quantized frame, not of a star on a real
            # noisy background, whose runner-up is always finite.  With no rival
            # statistics to lean on, and the residual being the claimed offset itself,
            # acceptance demands the detection sit within one_star_max_residual_px.
            refusal = (
                f'no_rival_detection: residual {residual_px:.3f} px exceeds '
                f'one_star_max_residual_px {self._one_star_max_residual_px:.3f} '
                f'and the peak-ratio gate is vacuous (no runner-up above background)'
            )
        diagnostics = StarUniqueMatchDiagnostics(
            mode='one_star',
            predicted_snr=star_snr,
            brightness_margin_mag=margin_mag,
            residual_px=residual_px,
            detection_peak_ratio=peak_ratio,
        )
        if refusal is not None:
            return self._fail(
                features=[star],
                diagnostics=diagnostics,
                reason=refusal,
                fit_rotation=fit_rotation,
            )
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
            star.feature_id,
            reported_det[0],
            reported_det[1],
            offset_v,
            offset_u,
            margin_mag,
            confidence,
        )
        cov = build_star_covariance(star, residual_px=residual_px)
        cov_out = embed_rotation_unobservable(cov) if fit_rotation else cov
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=(star.feature_id,),
            offset_px=(offset_v, offset_u),
            covariance_px2=cov_out,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
        )

    @staticmethod
    def _ellipse_fraction(context: NavContext) -> float:
        """Return the camera's lone-star ellipse fraction.

        The fraction is the camera's ``one_star_ellipse_fraction``.  Every camera sets
        one, and loading the configuration holds it to ``(0, 1]``.

        Parameters:
            context: Per-image NavContext.

        Returns:
            The fraction of the offset window, per axis, inside which a lone star is
            believed.

        Raises:
            ValueError: If the camera's configuration sets no fraction.
        """
        inst_config = context.obs.inst_config or {}  # type: ignore[attr-defined]
        fraction = inst_config.get('one_star_ellipse_fraction')
        if fraction is None:
            raise ValueError('the instrument configuration sets no one_star_ellipse_fraction')
        return float(fraction)

    @staticmethod
    def _max_peak_per_snr(context: NavContext, noise_sigma: float) -> float | None:
        """Return the instrument's bound on a lone source's peak per predicted SNR, or None.

        The bound is the camera's ``star_max_peak_per_predicted_snr`` in noise sigmas,
        converted to the image's units.  A camera that sets none has no bound.

        Parameters:
            context: Per-image NavContext.
            noise_sigma: Image noise sigma.

        Returns:
            The bound in the image's units, or ``None``.
        """
        inst_config = context.obs.inst_config or {}  # type: ignore[attr-defined]
        bound = inst_config.get('star_max_peak_per_predicted_snr')
        return None if bound is None else float(bound) * noise_sigma

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
