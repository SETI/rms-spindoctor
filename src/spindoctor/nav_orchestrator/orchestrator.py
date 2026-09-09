"""NavOrchestrator — top-level driver for autonomous navigation.

The orchestrator turns one observation into one ``NavResult``:

1. Build a ``NavContext`` (image, masks, classifier verdict, provenance).
2. Iterate registered ``NavModel`` instances and gather features and
   annotations from each.
3. Apply the ``FeatureReliabilityGate`` to drop bad-data features.
4. Run every feasible prior-free technique on the surviving features.
5. Combine pass-1 results via the ``ensemble`` function to derive a prior.
6. Run prior-required techniques against the derived prior.
7. Combine the union of pass-1 and pass-2 results via ``ensemble``.

Glob-pattern filters at construction time let an operator restrict which
models or techniques run for debugging (``only_models='body:MIMAS'``,
``only_techniques='!StarFieldFromCatalogNav'``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.config import MAIN_LOGGER, Config, logged_section
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.reliability import FeatureReliabilityGate, GatedFeatureRecord
from spindoctor.nav_model import NavModel
from spindoctor.nav_orchestrator.ensemble import EnsembleConfig, ensemble
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier import (
    ImageQualityThresholds,
    NavImageClassifier,
)
from spindoctor.nav_orchestrator.image_classifier_result import (
    ImageClass,
    NavImageClassifierResult,
)
from spindoctor.nav_orchestrator.image_derivatives import (
    ImageDerivativesConfig,
    compute_all_image_derivatives,
)
from spindoctor.nav_orchestrator.instrument_config import (
    InstrumentSettings,
    instrument_settings_from_obs,
)
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.nav_result import NavInternalErrorRecord, NavResult
from spindoctor.nav_orchestrator.provenance import (
    Provenance,
    collect_provenance_metadata,
)
from spindoctor.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    filter_technique_names,
    technique_tier,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.obs import obs_class_to_inst_name
from spindoctor.support.cmatrix import compute_pointing
from spindoctor.support.exceptions import (
    NavContractError,
    NavInternalError,
    NavPointingError,
)
from spindoctor.support.filters import NavFilterKind, NavFilterSpec, apply_filter
from spindoctor.support.image_quality import cosmic_ray_mask, saturation_mask
from spindoctor.support.nav_base import NavBase
from spindoctor.support.noise_estimate import estimate_image_noise_sigma
from spindoctor.support.status_reason import NavStatusReason

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.obs import ObsSnapshotInst

__all__ = [
    'NavOrchestrator',
    'OrchestratorPrep',
]


@dataclass(frozen=True)
class OrchestratorPrep:
    """Per-image artifacts produced by :meth:`NavOrchestrator.prepare`.

    Bundles every prep-phase output a downstream caller might want.  The
    autonomous :meth:`NavOrchestrator.navigate` does not use this struct
    (it inlines the prep so it can short-circuit on hard-failure
    images), but the manual-navigation driver does: it wraps the
    operator's chosen offset in a full :class:`NavResult` populated
    from ``provenance`` / ``image_classifier`` / ``feature_inventory``
    / ``model_metadata`` / ``annotations`` so the manual pipeline
    writes the same ``_metadata.json`` and ``_summary.png`` files the
    autonomous pipeline does.

    Parameters:
        context: NavContext built from the observation.
        provenance: Reproducibility envelope shared with the autonomous
            pipeline.
        image_classifier: Image-quality classifier verdict.
        features: Either the gated-kept feature list (when
            ``prepare(..., apply_gate=True)``) or every emitted feature
            (when ``apply_gate=False``).  The manual driver always
            requests the un-gated list because the operator visually
            overrides gate decisions.
        feature_inventory: Per-feature summary entries.  When the gate
            ran, both kept and gated features are included with their
            ``gated`` flag set accordingly; when the gate was skipped,
            every entry has ``gated=False``.
        model_metadata: Per-NavModel diagnostic dicts keyed by model name.
        annotations: Merged annotation collection assembled from every
            built NavModel's ``to_annotations``.
    """

    context: NavContext
    provenance: Provenance
    image_classifier: NavImageClassifierResult
    features: list[NavFeature]
    feature_inventory: list[NavFeatureSummary]
    model_metadata: dict[str, dict[str, Any]]
    annotations: Annotations


_HARD_FAILURE_TO_REASON: dict[ImageClass, NavStatusReason] = {
    'blank': NavStatusReason.NO_SIGNAL_IN_IMAGE,
    'fully_overexposed': NavStatusReason.IMAGE_OVEREXPOSED,
    'mostly_missing_data': NavStatusReason.MISSING_DATA_DOMINANT,
    'corrupt': NavStatusReason.IMAGE_CORRUPT,
}
"""Image-classifier classes that short-circuit before any technique runs.

Maps each hard-failure ``ImageClass`` to the matching ``NavStatusReason``
returned by the orchestrator's preflight.  Reading from this dict is the
sole admission test for the hard-failure short-circuit.
"""

_INSTRUMENTS_WITHOUT_SPICE_FRAMES = frozenset({'sim'})
"""Instruments that are expected to record no corrected attitude.

A simulated image has no spacecraft and no furnished camera frame, so it is
the one case where recording no attitude is the right outcome.  Everything
else reaching the same path is missing its frame mapping, which is a build
defect and is reported as such rather than passing silently -- including an
observation class the registry cannot identify at all, which carries the
least information of any case and is the last one that should be silenced.
"""


@dataclass
class _ModelRegistry:
    """Registry of NavModel instances built per-image by the orchestrator.

    Concrete NavModel subclasses do not auto-register at import time because
    they require an observation; the orchestrator instantiates them per
    image and registers them here.

    Parameters:
        models: List of constructed NavModel instances for the current image.
    """

    models: list[NavModel] = field(default_factory=list)

    def filter_by_glob(self, patterns: str | list[str]) -> list[NavModel]:
        """Return models whose ``name`` matches the glob pattern list.

        Model names follow the ``prefix:VALUE`` convention
        (``rings:SATURN``, ``body:DIONE``, plain ``stars`` for the
        catalog-driven star model).  This helper normalizes user
        patterns to that convention so the common shorthand forms
        match without requiring explicit globs:

        - ``"rings"`` is expanded to ``"rings:*"`` so a pattern
          missing a colon matches every namespaced model under that
          prefix.  ``"stars"`` (which has no colon in its model name)
          continues to match itself directly.
        - The value part of ``"prefix:VALUE"`` is normalized to
          uppercase so ``"body:saturn"`` matches ``"body:SATURN"``.

        The leading-``!`` exclusion convention is preserved.
        """
        names = [m.name for m in self.models]
        normalized = _normalize_model_patterns(patterns, names)
        kept = set(filter_technique_names(names, normalized))
        return [m for m in self.models if m.name in kept]


def _normalize_model_patterns(patterns: str | list[str], names: list[str]) -> list[str]:
    """Normalize model glob patterns to the ``prefix:VALUE`` convention.

    See :meth:`_ModelRegistry.filter_by_glob` for the rules.

    Parameters:
        patterns: Single pattern or list of patterns from the user.
        names: Available model names; used to detect whether the
            ``prefix:*`` expansion makes sense (any name contains
            ``:``).

    Returns:
        A list of normalized patterns suitable for
        :func:`filter_technique_names`.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    has_namespaced_models = any(':' in n for n in names)
    out: list[str] = []
    for raw in patterns:
        if raw.startswith('!'):
            prefix, body = '!', raw[1:]
        else:
            prefix, body = '', raw
        if ':' in body:
            head, _, tail = body.partition(':')
            normalized = f'{head}:{tail.upper()}'
        elif has_namespaced_models and body and not any(c in body for c in '*?['):
            # Plain prefix-only token (e.g. "rings"): expand to "rings:*"
            # so it matches every namespaced model under that prefix.
            # Names that have no namespace (like "stars") still match
            # via the unmodified pattern below.
            normalized = f'{body}:*'
            out.append(prefix + body)  # also keep the literal for "stars"
        else:
            normalized = body
        out.append(prefix + normalized)
    return out


def _feature_source_bodies(feature: NavFeature) -> frozenset[str]:
    """Return the set of body names a feature was emitted for.

    Reads the structured :attr:`NavFeature.body_name`; ring and star
    features have no body and yield an empty set.
    """
    return frozenset({feature.body_name}) if feature.body_name else frozenset()


def _bodies_with_non_spurious_primary(
    results: list[NavTechniqueResult],
) -> frozenset[str]:
    """Return the set of body names that have a non-spurious primary result.

    Reads each result's structured ``source_bodies`` (populated by the body
    techniques from the consumed features) rather than parsing ``feature_ids``.
    """
    covered: set[str] = set()
    for r in results:
        if r.spurious:
            continue
        if technique_tier(r.technique_name) != 'primary':
            continue
        covered.update(r.source_bodies)
    return frozenset(covered)


class NavOrchestrator(NavBase):
    """Top-level driver for autonomous navigation.

    Parameters:
        models: List of constructed NavModel instances for one observation.
            Caller builds these per-image (since each NavModel binds to an
            ``obs``).
        config: Optional ``Config`` override.
        only_models: Glob-pattern string or list selecting which models run.
            Default ``'*'`` runs every supplied model.
        only_techniques: Glob-pattern string or list selecting which
            techniques run.  Default ``'*'`` runs every registered
            technique.
        ensemble_config: Optional ``EnsembleConfig`` override; when omitted
            the ensemble parameters come from the ``orchestrator.ensemble``
            config section (config_540_orchestrator.yaml defaults).
        image_quality_thresholds: Optional thresholds for the image-quality
            classifier.
        spindoctor_version: Version string written into provenance.
    """

    def __init__(
        self,
        models: list[NavModel],
        *,
        config: Config | None = None,
        only_models: str | list[str] = '*',
        only_techniques: str | list[str] = '*',
        ensemble_config: EnsembleConfig | None = None,
        image_quality_thresholds: ImageQualityThresholds | None = None,
        image_derivatives_config: ImageDerivativesConfig | None = None,
        spindoctor_version: str = '0.0.0',
    ) -> None:
        super().__init__(config=config)
        self._registry = _ModelRegistry(
            models=_ModelRegistry(models=models).filter_by_glob(only_models)
        )
        self._only_techniques = only_techniques
        orch_cfg = self.config.orchestrator
        self._ensemble_config = ensemble_config or EnsembleConfig.from_mapping(
            orch_cfg.get('ensemble')
        )
        # ``image_quality_thresholds`` overrides the per-instrument config
        # block when supplied; otherwise the orchestrator builds the
        # thresholds from ``obs.inst_config`` per image.
        self._explicit_thresholds = image_quality_thresholds
        self._image_derivatives_config = image_derivatives_config or ImageDerivativesConfig()
        self._gate = FeatureReliabilityGate.from_mapping(orch_cfg.get('reliability_gate'))
        self._spindoctor_version = spindoctor_version

    def prepare(
        self,
        obs: ObsSnapshotInst,
        *,
        apply_gate: bool = True,
    ) -> OrchestratorPrep:
        """Build every per-image artifact a downstream caller might use.

        Runs the same pre-technique pipeline as :meth:`navigate`: builds
        the provenance envelope and the :class:`NavContext`, calls every
        registered NavModel's ``create_model``, extracts features,
        builds the feature inventory, snapshots per-model metadata, and
        merges annotations.  Hard-failure image classes are logged but
        **not** short-circuited — the caller (manual-nav dialog,
        debugger) decides what to do on a blank or saturated frame.

        A NavModel that raises propagates here rather than being converted:
        this method returns artifacts, not a ``NavResult``, so it has nothing
        to record a failure on.  Its callers are interactive, where an
        exception surfacing is the right outcome.

        Parameters:
            obs: The observation snapshot to prepare.
            apply_gate: When ``True`` (the default) the reliability gate
                runs; ``OrchestratorPrep.features`` carries only the
                kept features and ``feature_inventory`` records the
                gate verdict for both kept and gated entries.  When
                ``False`` every emitted feature is returned (with
                ``feature_inventory`` marking each as un-gated) — used
                by the manual-nav dialog, where the operator overrides
                the autonomous reliability decision.
        """
        provenance = self._make_provenance(obs)
        context, image_classifier = self._make_context(obs, provenance)
        self._log_classifier_verdict(image_classifier)
        built_models = self._build_models()
        all_features = self._extract_features(context, built_models)
        if apply_gate:
            kept, gated = self._extract_and_gate(all_features)
            features = kept
            feature_inventory = self._build_inventory(kept, gated)
        else:
            self._logger.info(
                'Reliability gate skipped (apply_gate=False); returning %d feature(s)',
                len(all_features),
            )
            features = all_features
            feature_inventory = self._build_inventory(all_features, [])
        return OrchestratorPrep(
            context=context,
            provenance=provenance,
            image_classifier=image_classifier,
            features=features,
            feature_inventory=feature_inventory,
            model_metadata=self._collect_model_metadata(built_models),
            annotations=self._collect_annotations(context, built_models),
        )

    @logged_section('orchestrator', 'ORCHESTRATION')
    def navigate(self, obs: ObsSnapshotInst) -> NavResult:
        """Run the full pipeline on one observation.

        No failure of a NavModel or a NavTechnique raises through to the
        caller.  Every exception out of one arrives here as one of two kinds
        and becomes a failed ``NavResult``: a ``NavContractError`` (an
        internal invariant violated by upstream code) as
        ``NavStatusReason.CONTRACT_VIOLATION``, and anything else, wrapped as
        a ``NavInternalError`` naming the component, as
        ``NavStatusReason.INTERNAL_ERROR`` with that name and the exception's
        class recorded on the result.

        Failing the image is the point.  The alternative -- continuing with
        whatever evidence survived -- produces an offset that is plausible,
        that reports ``success``, and that nothing downstream can tell apart
        from one computed with every model and technique intact.  A batch
        driver sees an ordinary failed frame either way and continues.

        The attitude computation is outside that guarantee: an unexpected
        error inside :meth:`with_pointing` propagates, by design, so a
        defect there fails the run on its first image rather than degrading
        every image quietly.  See that method.

        Parameters:
            obs: The observation snapshot to navigate.

        Returns:
            A single ``NavResult`` summarizing the navigation outcome.

        Raises:
            Exception: Propagated from :meth:`with_pointing` when the
                corrected-attitude computation hits anything other than the
                ``NavPointingError`` it expects and absorbs.  Also whatever
                ``_make_provenance`` and ``_make_context`` raise, since both
                run before the pipeline whose failures become a result:
                provenance reads the environment and the configuration, so a
                fault there would fail every image alike; context construction
                reads the image and its masks, so a fault there is that
                image's own, and it stops the run rather than failing the
                image.
        """
        provenance = self._make_provenance(obs)
        context, image_classifier = self._make_context(obs, provenance)
        self._log_classifier_verdict(image_classifier)
        if image_classifier.image_class in _HARD_FAILURE_TO_REASON:
            return self.with_pointing(
                self._fail(
                    status_reason=_HARD_FAILURE_TO_REASON[image_classifier.image_class],
                    image_classifier=image_classifier,
                    provenance=provenance,
                ),
                obs,
            )
        try:
            result = self._navigate_pipeline(
                context=context,
                image_classifier=image_classifier,
                provenance=provenance,
            )
        except NavContractError as exc:
            self._logger.exception(
                'CONTRACT VIOLATION: %s; an internal navigation invariant '
                'was violated (programming error, not bad image data); '
                'failing this image with status_reason=contract_violation',
                str(exc),
            )
            result = self._fail(
                status_reason=NavStatusReason.CONTRACT_VIOLATION,
                image_classifier=image_classifier,
                provenance=provenance,
            )
        except NavInternalError as exc:
            # error, not exception: the wrapper that raised this already
            # logged the plugin's own traceback, which is the one worth
            # reading.  Logging a second one here would print the re-raise
            # chain under every internal error and bury the first.
            self._logger.error(
                'INTERNAL ERROR: %s; this image was not navigated as designed, '
                'so it is failed rather than answered from the evidence that '
                'survived; failing with status_reason=internal_error',
                str(exc),
            )
            result = self._fail(
                status_reason=NavStatusReason.INTERNAL_ERROR,
                image_classifier=image_classifier,
                provenance=provenance,
                internal_error=NavInternalErrorRecord(
                    component=exc.component,
                    exception_type=exc.exception_type,
                ),
            )
        return self.with_pointing(result, obs)

    def with_pointing(self, result: NavResult, obs: ObsSnapshotInst) -> NavResult:
        """Stamp a result with the corrected-attitude solution for its image.

        :meth:`navigate` applies this to every result it returns.  The
        manual-navigation driver, which builds its own ``NavResult`` outside
        the autonomous pipeline, calls it directly so an operator-picked
        offset carries the same recorded attitude.

        The uncorrected attitude, frame identities and exposure times are
        recorded for every result; a corrected C-matrix additionally requires
        an offset and no fitted camera rotation.

        A host with no SPICE camera frame this build knows records nothing:
        that is expected for a simulated image, and reported as a warning for
        any other instrument, which means the instrument was added without
        its frame mapping.

        A pointing solution is recorded metadata rather than the navigation
        itself, so an attitude the environment cannot supply is reported to
        both logs and simply not recorded, and no wrong C-matrix ever
        reaches a kernel.  The absorbed set is exactly
        ``NavPointingError``, which the computation raises for every failure
        it expects: its own guards, and the kernel and frame lookups SPICE
        cannot answer.  Every other exception propagates by design, so a
        defect in the computation fails on the first image rather than
        silently dropping pointing from a whole batch while every image
        still reports success.

        Parameters:
            result: The result to stamp.
            obs: The observation it was produced from.

        Returns:
            ``result`` with its ``pointing`` field populated, or ``result``
            unchanged when no attitude could be computed.

        Raises:
            Exception: Anything other than ``NavPointingError`` raised by the
                attitude computation propagates unchanged, so that a defect
                there surfaces instead of being absorbed.
        """
        instrument = obs_class_to_inst_name(type(obs))
        try:
            pointing = compute_pointing(
                obs,
                offset_px=result.offset_px,
                rotation_fitted=result.rotation_rad is not None,
            )
        except NavPointingError as exc:
            # Only the expected class; the docstring above says why every
            # other exception must propagate.
            self._logger.exception('Could not compute the corrected pointing: %s', exc)
            # The traceback goes to the per-image log and one line to the run
            # log, so an operator watching a batch sees one line per affected
            # image rather than a traceback per image.
            MAIN_LOGGER.error(
                'Corrected pointing not recorded for this %s image: %s', instrument, exc
            )
            return result
        if pointing is None:
            if instrument in _INSTRUMENTS_WITHOUT_SPICE_FRAMES:
                self._logger.debug(
                    'Instrument %s has no SPICE camera frame; pointing not recorded', instrument
                )
            else:
                self._logger.warning(
                    'No SPICE camera frame is mapped for instrument %s; pointing not recorded',
                    instrument,
                )
                MAIN_LOGGER.warning(
                    'No SPICE camera frame is mapped for instrument %s; no image of it can '
                    'carry a corrected attitude',
                    instrument,
                )
            return result
        self._logger.info(
            'Recorded pointing: camera frame %s (%d), CK object %d, corrected cmatrix %s',
            pointing.baseline.camera_frame,
            pointing.baseline.camera_frame_id,
            pointing.baseline.ck_frame_id,
            'present' if pointing.cmatrix is not None else 'withheld',
        )
        return dataclasses.replace(result, pointing=pointing)

    def _navigate_pipeline(
        self,
        *,
        context: NavContext,
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
    ) -> NavResult:
        """Run the model / feature / technique / ensemble pipeline.

        Called by :meth:`navigate` after the hard-failure short-circuit.
        A ``NavContractError`` or ``NavInternalError`` raised anywhere in
        here -- including one raised by a plugin wrapper -- propagates to
        :meth:`navigate`, which converts it into a failed ``NavResult``.

        Parameters:
            context: Per-image NavContext.
            image_classifier: Image-quality classifier verdict.
            provenance: Reproducibility envelope.

        Returns:
            A single ``NavResult`` summarizing the navigation outcome.
        """
        built_models = self._build_models()
        all_features = self._extract_features(context, built_models)
        kept, gated = self._extract_and_gate(all_features)
        if gated:
            gated_kinds: dict[str, int] = {}
            for record in gated:
                key = record.feature.feature_type.name
                gated_kinds[key] = gated_kinds.get(key, 0) + 1
            self._logger.debug('Gated breakdown: %s', gated_kinds)
        feature_inventory = self._build_inventory(kept, gated)
        model_metadata = self._collect_model_metadata(built_models)
        annotations = self._collect_annotations(context, built_models)
        if not all_features:
            return self._fail(
                status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        if not kept:
            return self._fail(
                status_reason=NavStatusReason.ALL_FEATURES_GATED,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                features=all_features,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        # Pass 1 — prior-free techniques.  Primaries run first so the
        # fallback pass can skip techniques whose source body already
        # has a non-spurious primary result; that matches the
        # "fallback runs only when the primary fails" semantic
        # operators expect, instead of running every technique and
        # filtering downstream.
        self._logger.info('Pass 1: running prior-free primary techniques')
        pass1_primary = self._run_pass(kept, context, requires_prior=False, tier_filter='primary')
        covered_bodies = _bodies_with_non_spurious_primary(pass1_primary)
        if covered_bodies:
            self._logger.debug('Pass 1 primary covered bodies: %s', sorted(covered_bodies))
        self._logger.info('Pass 1: running prior-free fallback techniques')
        pass1_fallback = self._run_pass(
            kept,
            context,
            requires_prior=False,
            tier_filter='fallback',
            excluded_bodies=covered_bodies,
        )
        pass1_results = pass1_primary + pass1_fallback
        self._logger.info('Pass 1: %d technique result(s) produced', len(pass1_results))
        if not pass1_results:
            return self._fail(
                status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
                image_classifier=image_classifier,
                provenance=provenance,
                feature_inventory=feature_inventory,
                model_metadata=model_metadata,
                annotations=annotations,
            )
        pass1_ensemble = ensemble(
            pass1_results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            config=self._ensemble_config,
            model_metadata=model_metadata,
            annotations=annotations,
        )
        if pass1_ensemble.status == 'failed':
            self._log_status_reason(pass1_ensemble.status_reason)
            return pass1_ensemble
        if pass1_ensemble.offset_px is not None:
            self._logger.info(
                'Pass 1 prior: offset (dv, du) = (%.4f, %.4f) px, confidence %.3f',
                pass1_ensemble.offset_px[0],
                pass1_ensemble.offset_px[1],
                pass1_ensemble.confidence,
            )
        # Pass 2 — prior-required techniques refine on the pass-1 prior.  Only
        # a 'success' pass-1 ensemble seeds the prior: a 'conflicted' result is
        # an explicitly untrustworthy "best group" offset, and installing it
        # would lock pass-2 toward one arbitrary mode instead of letting pass-2
        # evidence break the tie.  ('failed' already returned above.)
        prior_sources: frozenset[str] = frozenset()
        if (
            pass1_ensemble.status == 'success'
            and pass1_ensemble.offset_px is not None
            and pass1_ensemble.covariance_px2 is not None
        ):
            pass2_context = context.with_prior(
                offset_px=pass1_ensemble.offset_px,
                covariance_px2=pass1_ensemble.covariance_px2,
            )
            prior_sources = frozenset(pass1_ensemble.consensus_techniques)
        else:
            pass2_context = context
        self._logger.info('Pass 2: running prior-required techniques')
        pass2_results = self._run_pass(kept, pass2_context, requires_prior=True)
        self._logger.info('Pass 2: %d technique result(s) produced', len(pass2_results))
        if prior_sources and pass2_results:
            # Stamp each pass-2 result with the techniques whose pass-1
            # consensus seeded its prior; the final ensemble lets such a
            # result refine those techniques' offset but not corroborate
            # it (issue #222).
            pass2_results = [
                dataclasses.replace(r, prior_source_techniques=prior_sources) for r in pass2_results
            ]
            self._logger.debug(
                'Stamped %d pass-2 result(s) with prior sources: %s',
                len(pass2_results),
                ', '.join(sorted(prior_sources)),
            )
        # Final ensemble over the union of both passes' results.
        final = ensemble(
            pass1_results + pass2_results,
            feature_inventory=feature_inventory,
            image_classifier=image_classifier,
            provenance=provenance,
            config=self._ensemble_config,
            model_metadata=model_metadata,
            annotations=annotations,
        )
        self._log_final_result(final)
        self._log_status_reason(final.status_reason)
        return final

    def _log_classifier_verdict(self, classifier: NavImageClassifierResult) -> None:
        """Emit the per-image image-quality classifier verdict at INFO."""
        self._logger.info(
            'Image classifier: class=%s, max_dn=%.4g, noise_sigma=%.3g, '
            'saturation_frac=%.3f, missing_frac=%.3f',
            classifier.image_class,
            classifier.max_dn,
            classifier.noise_sigma,
            classifier.saturation_frac,
            classifier.missing_frac,
        )
        if classifier.flags:
            self._logger.info('Image classifier flags: %s', list(classifier.flags))

    def _log_final_result(self, result: NavResult) -> None:
        """Emit the final offset / confidence / per-technique summary."""
        if result.offset_px is None:
            self._logger.info('Final result: no offset (status=%s)', result.status)
            return
        sigma_dv = result.sigma_px[0] if result.sigma_px is not None else float('nan')
        sigma_du = result.sigma_px[1] if result.sigma_px is not None else float('nan')
        self._logger.info(
            'Final offset (dv, du) = (%.4f, %.4f) px; sigma (dv, du) = (%.4f, %.4f) px',
            result.offset_px[0],
            result.offset_px[1],
            sigma_dv,
            sigma_du,
        )
        self._logger.info(
            'Final status = %s, confidence = %.3f (rank=%s); %d technique result(s) fused',
            result.status,
            result.confidence,
            result.confidence_rank,
            len(result.per_technique),
        )
        if result.per_technique:
            for tr in result.per_technique:
                self._logger.debug(
                    'Per-technique: %s -> offset (%.4f, %.4f) px, confidence %.3f, '
                    'spurious=%s, at_edge=%s',
                    tr.technique_name,
                    tr.offset_px[0],
                    tr.offset_px[1],
                    tr.confidence,
                    tr.spurious,
                    tr.at_edge,
                )

    def _build_models(self) -> list[NavModel]:
        """Call ``create_model`` on every registered NavModel.

        Wrapped so :meth:`prepare` and :meth:`navigate` share the same log
        line and the same treatment of a model that raises: the exception is
        logged with its traceback and re-raised as a ``NavInternalError``
        naming the model, which fails the image.  A model that cannot build
        is not a model that found nothing -- the returned list would be short
        by a whole source of evidence, and every number computed from what
        remained would be indistinguishable from one computed from all of it.

        Returns:
            Every registered model, built.  A partial list is never returned:
            either all of them built or this raises.

        Raises:
            NavContractError: Propagated from a model that violated an
                internal invariant, for :meth:`navigate` to convert.
            NavInternalError: Raised for any other exception out of
                ``create_model``, for :meth:`navigate` to convert.
        """
        self._logger.info(
            'Building %d NavModel(s): %s',
            len(self._registry.models),
            ', '.join(m.name for m in self._registry.models) or '(none)',
        )
        built_models: list[NavModel] = []
        for model in self._registry.models:
            try:
                model.create_model()
            except NavContractError:
                self._logger.exception(
                    'CONTRACT VIOLATION in NavModel %s.create_model; re-raising',
                    model.name,
                )
                raise
            except Exception as exc:
                component = f'{model.name}.create_model'
                self._logger.exception(
                    'INTERNAL ERROR: %s raised; failing this image with '
                    'status_reason=internal_error',
                    component,
                )
                raise NavInternalError(component, exc) from exc
            built_models.append(model)
        return built_models

    def _extract_and_gate(
        self, all_features: list[NavFeature]
    ) -> tuple[list[NavFeature], list[GatedFeatureRecord]]:
        """Apply the reliability gate and emit the standard summary log line."""
        kept, gated = self._gate.apply(all_features)
        self._logger.info(
            'Reliability gate: %d feature(s) kept, %d gated (out of %d emitted)',
            len(kept),
            len(gated),
            len(all_features),
        )
        # Per-feature DEBUG log so an operator can see exactly which
        # reliability components dragged a gated feature below the
        # threshold (visible_arc_fraction, incidence_factor,
        # albedo_penalty, ...) without having to crack the metadata
        # JSON.  Threshold lookup mirrors ``FeatureReliabilityGate.apply``.
        gated_ids = {record.feature.feature_id for record in gated}
        for feature in all_features:
            verdict = 'gated' if feature.feature_id in gated_ids else 'kept'
            threshold = self._gate.thresholds.get(feature.feature_type, 0.0)
            self._logger.debug(
                '%s: %s %s reliability=%.3f threshold=%.3f reasons={%s}',
                verdict,
                feature.feature_id,
                feature.feature_type.name,
                feature.reliability,
                threshold,
                _format_reliability_breakdown(feature.reliability_reasons),
            )
        return kept, gated

    @staticmethod
    def _hidden_star(feature: NavFeature) -> bool:
        """Whether a feature is a star the star model flagged as behind a body or ring.

        Parameters:
            feature: Any emitted feature.

        Returns:
            True for a STAR feature carrying ``in_body_silhouette``.
        """
        return (
            feature.feature_type is NavFeatureType.STAR
            and isinstance(feature.flags, StarFlags)
            and feature.flags.in_body_silhouette
        )

    def _reason_for_a_covering_body(
        self,
        status_reason: NavStatusReason,
        model_metadata: dict[str, dict[str, Any]] | None,
        features: list[NavFeature],
    ) -> NavStatusReason:
        """Substitute ``BODY_FILLS_FOV`` when nothing navigable was in view.

        The reason exists so that a statistics report can set aside the images
        that could not have been navigated, and an image cannot be when no
        navigable feature is visible: no star, no known ring feature, no body
        showing a limb or a terminator.  A body that covers the extended frame
        with neither edge inside it hides every star behind it, and the star
        model still emits those stars, flagged and at zero reliability, so
        without this substitution the image would be filed under
        ``ALL_FEATURES_GATED`` beside frames that had something to fit.

        Anything navigable in front of the body emits a feature of its own --
        a moon its limb or disc, the near side of the rings its edges -- so the
        substitution applies only when every feature emitted is a star the body
        hides, or none was emitted at all.  A reason reached after a technique
        ran is left alone for the same reason: something was there to run on.

        Parameters:
            status_reason: The reason the failure would otherwise carry.
            model_metadata: Per-model metadata, read for a body model that
                declined its body as covering the frame.
            features: Every feature the models emitted, gated or not.

        Returns:
            ``BODY_FILLS_FOV`` where it applies, and the reason unchanged
            otherwise.
        """
        if status_reason not in (
            NavStatusReason.NO_FEATURES_EXTRACTED,
            NavStatusReason.ALL_FEATURES_GATED,
        ):
            return status_reason
        covering = sorted(
            name
            for name, meta in (model_metadata or {}).items()
            if isinstance(meta, dict) and meta.get('fills_extfov') and not meta.get('edge_in_frame')
        )
        if not covering:
            return status_reason
        in_view = sum(1 for feature in features if not self._hidden_star(feature))
        if in_view:
            self._logger.info(
                'Keeping %s although %s covers the extended frame: %d feature(s) in '
                'front of it were in view',
                status_reason.value,
                ', '.join(covering),
                in_view,
            )
            return status_reason
        self._logger.info(
            'Reporting body_fills_fov rather than %s: %s covers the extended frame and '
            'nothing navigable was in view',
            status_reason.value,
            ', '.join(covering),
        )
        return NavStatusReason.BODY_FILLS_FOV

    def _fail(
        self,
        *,
        status_reason: NavStatusReason,
        image_classifier: NavImageClassifierResult,
        provenance: Provenance,
        feature_inventory: list[NavFeatureSummary] | None = None,
        model_metadata: dict[str, dict[str, Any]] | None = None,
        annotations: Annotations | None = None,
        internal_error: NavInternalErrorRecord | None = None,
        features: list[NavFeature] | None = None,
    ) -> NavResult:
        """Emit the operator-readable INFO lines and return a failed NavResult.

        Wraps ``NavResult.failed`` so every short-circuit and gate emits
        the per-status-reason INFO log lines defined in
        :data:`STATUS_REASON_INFO_TEMPLATE`.

        The feature-stage gates pass ``features``, everything the models
        emitted, so that an image a body covers with nothing navigable in view
        is filed under ``BODY_FILLS_FOV`` rather than under the gate it fell
        through; see ``_reason_for_a_covering_body``.
        """
        status_reason = self._reason_for_a_covering_body(
            status_reason, model_metadata, features or []
        )
        self._log_status_reason(status_reason)
        return NavResult.failed(
            status_reason=status_reason,
            image_classifier=image_classifier,
            provenance=provenance,
            feature_inventory=feature_inventory or [],
            model_metadata=model_metadata or {},
            annotations=annotations or Annotations(),
            internal_error=internal_error,
        )

    def _log_status_reason(self, status_reason: NavStatusReason) -> None:
        """Emit the per-status-reason INFO log lines."""
        for line in STATUS_REASON_INFO_TEMPLATE.get(status_reason, ()):
            self._logger.info(line)

    def _collect_model_metadata(self, models: list[NavModel]) -> dict[str, dict[str, Any]]:
        """Snapshot ``model.metadata`` from every successfully-built NavModel."""
        out: dict[str, dict[str, Any]] = {}
        for model in models:
            out[model.name] = dict(model.metadata)
        return out

    def _collect_annotations(self, context: NavContext, models: list[NavModel]) -> Annotations:
        """Merge per-NavModel annotation collections into one.

        Each model's ``to_annotations`` is invoked; one that raises is logged
        with its traceback and re-raised as a ``NavInternalError`` naming the
        model, which fails the image.  Annotations are only the summary PNG's
        input, so failing the whole image for them may look severe -- but a
        model that cannot describe what it found is a model in a state nobody
        planned for, and this pipeline's rule is that no exception is
        invisible and no image reports an outcome after one was raised.

        Returns:
            One collection merging every model's annotations.

        Raises:
            NavContractError: Propagated from a model that violated an
                internal invariant, for :meth:`navigate` to convert.
            NavInternalError: Raised for any other exception out of
                ``to_annotations``, for :meth:`navigate` to convert.
        """
        merged = Annotations()
        for model in models:
            try:
                model_annotations = model.to_annotations(context)
            except NavContractError:
                self._logger.exception(
                    'CONTRACT VIOLATION in NavModel %s.to_annotations; re-raising',
                    model.name,
                )
                raise
            except Exception as exc:
                component = f'{model.name}.to_annotations'
                self._logger.exception(
                    'INTERNAL ERROR: %s raised; failing this image with '
                    'status_reason=internal_error',
                    component,
                )
                raise NavInternalError(component, exc) from exc
            merged.add_annotations(model_annotations)
        return merged

    def _extract_features(self, context: NavContext, models: list[NavModel]) -> list[NavFeature]:
        """Iterate built models and gather their features.

        A model that raises is logged with a full traceback and re-raised as
        a ``NavInternalError`` naming it, which :meth:`navigate` converts
        into a failed ``NavResult``.  Catching every exception is still
        intentional and for the original reason -- the orchestrator must
        never raise through to its caller, and specific exceptions cannot be
        enumerated because every plugin has its own failure modes -- but
        catching is no longer the same as continuing.  Failing the image is
        not raising through to the caller: the result carries the failure and
        the batch driver moves to the next frame as it does for any other
        failed image.

        A model that emitted zero features because there was nothing to find
        is a different thing entirely, and reaches this method as an empty
        list rather than as an exception; the feature and gating path already
        models that outcome properly.  Conflating the two is what let an
        image report ``success`` on evidence an exception had quietly
        removed.

        ``NavContractError`` keeps its own clause: it is a violated invariant
        rather than an unanticipated failure, and stays classified as
        ``CONTRACT_VIOLATION`` rather than ``INTERNAL_ERROR``.

        Returns:
            Every feature every model emitted, in model order.

        Raises:
            NavContractError: Propagated from a model that violated an
                internal invariant, for :meth:`navigate` to convert.
            NavInternalError: Raised for any other exception out of
                ``to_features``, for :meth:`navigate` to convert.
        """
        all_features: list[NavFeature] = []
        for model in models:
            try:
                emitted = model.to_features(context)
            except NavContractError:
                self._logger.exception(
                    'CONTRACT VIOLATION in NavModel %s.to_features; re-raising',
                    model.name,
                )
                raise
            except Exception as exc:
                component = f'{model.name}.to_features'
                self._logger.exception(
                    'INTERNAL ERROR: %s raised; failing this image with '
                    'status_reason=internal_error',
                    component,
                )
                raise NavInternalError(component, exc) from exc
            all_features.extend(emitted)
        return all_features

    def _run_pass(
        self,
        features: list[NavFeature],
        context: NavContext,
        *,
        requires_prior: bool,
        tier_filter: Literal['primary', 'fallback'] | None = None,
        excluded_bodies: frozenset[str] = frozenset(),
    ) -> list[NavTechniqueResult]:
        """Run every feasible technique whose ``requires_prior`` matches.

        Parameters:
            features: Gated-kept features available to every technique.
            context: Per-image NavContext.
            requires_prior: Restricts the run to pass-1 (``False``) or
                pass-2 (``True``) techniques.
            tier_filter: If set, restrict to techniques in the given
                tier (``'primary'`` or ``'fallback'``).  ``None`` runs
                every tier.
            excluded_bodies: Set of body names whose features should be
                filtered out before each technique sees them.  The
                orchestrator passes the set of body names that have a
                non-spurious primary result so the fallback pass skips
                techniques whose only candidate features belong to
                already-covered bodies (the "fallback runs only when
                the primary fails" semantic; the ensemble's
                ``_drop_superseded_fallbacks`` is the redundant
                downstream gate that catches anything that slips
                through).

        Every candidate technique (right ``requires_prior`` and tier)
        gets a per-technique DEBUG line reporting why it was skipped
        (name filter, no accepted feature type available, infeasible
        with the technique's own rejection reason) or, when feasible,
        how many features it would consume — the per-technique
        feasibility breakdown an engineer needs to answer "why didn't
        technique X run on image Y" from the log alone.

        A NavTechnique that raises is logged with a full traceback and
        re-raised as a ``NavInternalError`` naming it, which fails the image.
        A technique failing among several is closer to a normal outcome than
        a model failing to build, and treating it as one was considered; the
        reason it is not exempted is that the pipeline has no way to record
        the exemption honestly.  A technique with nothing to contribute
        already has two ways to say so -- ``is_feasible`` returning false,
        and a result marked spurious -- and both leave a trace in the
        per-technique inventory.  An exception leaves none, so an exempted
        technique failure would be a silently smaller ensemble, and the
        ensemble's confidence is a function of how many witnesses agreed.

        ``NavContractError`` keeps its own clause and stays classified as
        ``CONTRACT_VIOLATION`` (see ``_extract_features``).

        Returns:
            One entry per technique that ran and produced a result, in
            registry order.  A skipped or infeasible technique contributes
            no entry, so the list may be empty.

        Raises:
            NavContractError: Propagated from a technique that violated an
                internal invariant, for :meth:`navigate` to convert.
            NavInternalError: Raised for any other exception out of a
                technique, for :meth:`navigate` to convert.
        """
        results: list[NavTechniqueResult] = []
        names = [
            cls.name
            for cls in NavTechnique._registry
            if cls.requires_prior == requires_prior
            and (tier_filter is None or cls.tier == tier_filter)
        ]
        kept_names = set(filter_technique_names(names, self._only_techniques))
        # ``features`` is loop-invariant, so the available feature-type set is
        # computed once rather than rebuilt for every registry entry.
        available_types: set[NavFeatureType] = {f.feature_type for f in features}
        for cls in NavTechnique._registry:
            if cls.requires_prior != requires_prior:
                continue
            if tier_filter is not None and cls.tier != tier_filter:
                continue
            if cls.name not in kept_names:
                self._logger.debug(
                    'Technique %s skipped: excluded by the only_techniques filter',
                    cls.name,
                )
                continue
            if not (cls.accepts_feature_types & available_types):
                self._logger.debug(
                    'Technique %s skipped: no feature of accepted type(s) %s available',
                    cls.name,
                    ', '.join(sorted(t.value for t in cls.accepts_feature_types)),
                )
                continue
            technique = cls(config=self.config)
            available_features = features
            if excluded_bodies and cls.tier == 'fallback' and not cls.runs_as_witness:
                pre_filter_count = sum(
                    1 for f in features if f.feature_type in cls.accepts_feature_types
                )
                available_features = [
                    f for f in features if not (_feature_source_bodies(f) & excluded_bodies)
                ]
                post_filter_count = sum(
                    1 for f in available_features if f.feature_type in cls.accepts_feature_types
                )
                dropped = pre_filter_count - post_filter_count
                if dropped > 0:
                    self._logger.info(
                        'Skipping %d %s feature(s) for fallback %s: source body already '
                        'covered by a non-spurious primary',
                        dropped,
                        ', '.join(sorted(t.value for t in cls.accepts_feature_types)),
                        cls.name,
                    )
                if post_filter_count == 0:
                    continue
            feasibility = technique.is_feasible(available_features)
            if not feasibility.feasible:
                self._logger.debug(
                    'Technique %s infeasible: %s',
                    cls.name,
                    feasibility.reason,
                )
                continue
            self._logger.debug(
                'Technique %s feasible: would consume %d feature(s)',
                cls.name,
                feasibility.consumed_feature_count,
            )
            subset = [f for f in available_features if f.feature_type in cls.accepts_feature_types]
            try:
                results.append(technique.navigate(subset, context))
            except NavContractError:
                self._logger.exception(
                    'CONTRACT VIOLATION in NavTechnique %s.navigate; re-raising',
                    cls.name,
                )
                raise
            except Exception as exc:
                component = f'{cls.name}.navigate'
                self._logger.exception(
                    'INTERNAL ERROR: %s raised; failing this image with '
                    'status_reason=internal_error',
                    component,
                )
                raise NavInternalError(component, exc) from exc
        return results

    def _make_context(
        self, obs: ObsSnapshotInst, provenance: Provenance
    ) -> tuple[NavContext, NavImageClassifierResult]:
        """Build a NavContext from an observation.

        The image, sensor mask, saturation mask, and cosmic-ray mask all
        live on the *extended FOV* (zero-padded around the original sensor
        rectangle).  ``obs.extdata`` is the canonical source for the
        extfov-shaped image and matches the extfov sensor mask shape
        regardless of the per-instrument ``extfov_margin_vu``.
        """
        settings = instrument_settings_from_obs(obs)
        raw_image = obs.extdata.astype('float64')
        sensor_mask = obs.extfov_data_sensor_mask()
        # Sanitise the missing-data sentinel before any finite-only
        # computation.  For calibrated-IF instruments the sentinel is
        # literally NaN (CISS CALIB ``marker_value: NaN``); leaving those
        # NaN in place would make ``_smooth_and_compute_gradients`` raise
        # (its finite-only guard), and that ValueError would propagate out
        # of ``navigate``, violating the orchestrator's never-raise
        # contract.  ``missing_mask`` is computed for both the ``==
        # marker`` case and the ``isnan(marker)`` case, then the matching
        # pixels are replaced with a finite fill (0.0) so the gradient /
        # noise path always sees finite data.  The *true* missing fraction
        # is threaded into the classifier so missing/dropout detection
        # still fires for calibrated images.
        marker = settings.marker_value
        if np.isnan(marker):
            missing_mask = np.isnan(raw_image)
        else:
            missing_mask = raw_image == marker
        if missing_mask.any():
            raw_image = np.where(missing_mask, 0.0, raw_image)
        sensor_missing = missing_mask & sensor_mask
        n_sensor = int(sensor_mask.sum())
        missing_frac = float(sensor_missing.sum()) / float(max(n_sensor, 1))
        image, pre_filter = self._apply_source_image_filter(raw_image, obs)
        # The classifier and the saturation mask read the *raw* image, not the
        # source-image-filtered ``image``: their absolute-DN gates (blank /
        # saturation / overexposed, and the full-well saturation mask) are
        # physical properties of the sensor data and become meaningless after a
        # DC-removing filter such as BANDPASS_DOG (a near-full-well region comes
        # out near zero post-DoG).  ``raw_image`` is already NaN-free (markers
        # were filled above), and the true missing fraction is supplied
        # explicitly so the NaN sentinel still drives ``mostly_missing_data`` /
        # ``partial_dropout``.  (With every shipped source_image_filter disabled
        # today, ``image is raw_image`` and this is a no-op.)
        classifier_thresholds = (
            self._explicit_thresholds
            if self._explicit_thresholds is not None
            else settings.thresholds
        )
        classifier = NavImageClassifier(thresholds=classifier_thresholds)
        classifier_result = classifier.classify(raw_image, sensor_mask, missing_frac=missing_frac)
        sat_mask = self._build_saturation_mask(raw_image, sensor_mask, settings)
        # Cosmic-ray detection and the derivative DT threshold operate on the
        # *filtered* working ``image``, so their noise sigma is estimated from
        # that image (not the raw classifier sigma) to stay self-consistent with
        # the gradients the DT techniques actually fit.  A near-zero estimate is
        # clamped to a tiny value so the CR mask is well-defined on near-blank
        # inputs.
        noise_sigma = estimate_image_noise_sigma(image, sensor_mask)
        cr_noise_sigma = max(noise_sigma, 1e-6)
        cr_mask = cosmic_ray_mask(image, image_noise_sigma=cr_noise_sigma)
        # Single-pass derivative computation: one gaussian + sobel pair
        # produces gradient magnitude, edge DT, and the signed gradient
        # vector image together rather than doing the heavy smoothing
        # twice for separate calls.
        gradient_ext, edge_dt_ext, gradient_vu_ext = compute_all_image_derivatives(
            image,
            noise_sigma,
            config=self._image_derivatives_config,
        )
        context = NavContext(
            obs=obs,
            image_ext=image,
            sensor_mask_ext=sensor_mask,
            image_noise_sigma=noise_sigma,
            saturation_mask_ext=sat_mask,
            cosmic_ray_mask_ext=cr_mask,
            image_classifier=classifier_result,
            provenance=provenance,
            image_gradient_ext=gradient_ext,
            image_gradient_vu_ext=gradient_vu_ext,
            image_edge_dt_ext=edge_dt_ext,
            pre_filter_applied=pre_filter,
            fit_camera_rotation=settings.fit_camera_rotation,
            max_rotation_deg=settings.max_rotation_deg,
        )
        return context, classifier_result

    @staticmethod
    def _build_saturation_mask(
        image: np.ndarray,
        sensor_mask: np.ndarray,
        settings: InstrumentSettings,
    ) -> np.ndarray:
        """Construct the per-image saturation mask.

        For raw-DN instruments the mask is ``image >= saturation_dn``.
        For calibrated-IF instruments the saturation gate is intentionally
        off — the calibrated I/F values that survive the CALIB pipeline
        depend on exposure, filter, and gain, so a single threshold
        cannot identify which raw pixels were saturated before
        calibration.  The orchestrator therefore returns an empty mask
        without any logging; operators who need saturation flags on a
        Cassini scene must navigate the corresponding raw frame.
        """
        if settings.saturation_dn is not None:
            return saturation_mask(image, full_well_dn=settings.saturation_dn)
        return np.zeros(image.shape, dtype=bool)

    def _apply_source_image_filter(
        self, image: np.ndarray, obs: ObsSnapshotInst
    ) -> tuple[np.ndarray, NavFilterSpec | None]:
        """Apply the per-instrument source-image filter (when enabled).

        Returns the filtered image and the ``NavFilterSpec`` that produced
        it (so :class:`NavContext.pre_filter_applied` records what ran).
        When the per-instrument ``source_image_filter`` block is missing,
        disabled, or set to ``NONE`` the image is returned unchanged and
        the spec is ``None``.
        """
        inst_config = getattr(obs, 'inst_config', None)
        if inst_config is None:
            return image, None
        block = inst_config.get('source_image_filter')
        if not block or not bool(block.get('enabled', False)):
            return image, None
        kind_str = str(block.get('kind', 'NONE')).upper()
        try:
            kind = NavFilterKind[kind_str]
        except KeyError:
            self._logger.warning(
                'unknown source_image_filter.kind %r; skipping pre-filter', kind_str
            )
            return image, None
        if kind is NavFilterKind.NONE:
            return image, None
        if kind is NavFilterKind.BANDPASS_DOG:
            lo = float(block.get('lo_sigma_px', 0.0))
            hi = float(block.get('hi_sigma_px', 0.0))
            spec = NavFilterSpec(kind=kind, bandpass_cutoffs_px=(lo, hi))
        else:
            sigma = float(block.get('sigma_px', 1.0))
            spec = NavFilterSpec(kind=kind, sigma_xy=(sigma, sigma))
        filtered = apply_filter(image, spec)
        return filtered, spec

    def _make_provenance(self, obs: ObsSnapshotInst) -> Provenance:
        """Build the per-image Provenance envelope.

        Reads the runtime-derived fields (git SHA, loaded SPICE kernels,
        static-data hashes, resolved-config hash plus applied overrides,
        and star-catalog identifiers) once per ``navigate`` call via
        :func:`collect_provenance_metadata`.
        """
        timestamp = datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')
        meta = collect_provenance_metadata(self.config)
        return Provenance(
            spindoctor_version=self._spindoctor_version,
            image_et=float(obs.midtime),
            pipeline_run_iso8601=timestamp,
            spindoctor_git_sha=meta.git_sha,
            spice_kernels=meta.spice_kernels,
            static_data_hashes=meta.static_data_hashes,
            technique_names=tuple(sorted(cls.name for cls in NavTechnique._registry)),
            extractor_names=tuple(sorted(m.name for m in self._registry.models)),
            config_hash=meta.config_hash,
            config_overrides=meta.config_overrides,
            star_catalogs=meta.star_catalogs,
        )

    @staticmethod
    def _build_inventory(
        kept: list[NavFeature], gated: list[GatedFeatureRecord]
    ) -> list[NavFeatureSummary]:
        """Build the feature inventory consumed by the curator."""
        out: list[NavFeatureSummary] = []
        for f in kept:
            out.append(_summary_from_feature(f, gated=False, gate_reason=None))
        for record in gated:
            out.append(_summary_from_feature(record.feature, gated=True, gate_reason=record.reason))
        return out


def _summary_from_feature(
    feature: NavFeature, *, gated: bool, gate_reason: str | None
) -> NavFeatureSummary:
    """Project a NavFeature down to a NavFeatureSummary."""
    bbox = _bbox_from_geometry(feature)
    return NavFeatureSummary(
        feature_id=feature.feature_id,
        feature_type=feature.feature_type,
        source_model=feature.source_model,
        reliability=feature.reliability,
        gated=gated,
        gate_reason=gate_reason,
        bbox_extfov_vu=bbox,
        reliability_reasons=feature.reliability_reasons,
    )


def _bbox_from_geometry(feature: NavFeature) -> tuple[int, int, int, int]:
    """Return the ``bbox_extfov_vu`` tuple for a feature's geometry payload.

    Every ``NavFeatureGeometry`` variant carries ``bbox_extfov_vu``;
    direct attribute access is safe because the union excludes any
    payload that lacks it.
    """
    bbox = feature.geometry.bbox_extfov_vu
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _format_reliability_breakdown(breakdown: Any) -> str:
    """Format a ``NavReliabilityBreakdown`` as ``key=value`` for the per-feature DEBUG log.

    Only fields with non-``None`` values are rendered so the line stays
    readable; floats round to three decimals, bools render as
    ``True``/``False`` directly.
    """
    parts: list[str] = []
    for f in dataclasses.fields(breakdown):
        value = getattr(breakdown, f.name)
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f'{f.name}={value:.3f}')
        else:
            parts.append(f'{f.name}={value}')
    return ', '.join(parts)
