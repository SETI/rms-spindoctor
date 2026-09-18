"""Manual navigation technique — interactive PyQt6 dialog.

Renders the composite predicted scene from every NavModel's template
features and lets the operator pick a (dv, du) offset by hand.  An
``Auto`` button inside the dialog runs the same masked-NCC pyramid that
correlation-based techniques use, so the operator can either accept the
auto-pick or override it manually.

This technique is not part of the autonomous pipeline; it does not appear
in the ``NavTechnique._registry`` and is invoked directly by an
interactive driver.  The plan calls for replacing the dialog's single
``Auto`` button with a per-technique side-by-side panel showing each
registered technique's proposal; that redesign is deferred and tracked
separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.config import IMAGE_LOGGER, Config
from spindoctor.feature.composition import compose_dialog_overlay
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_technique.diagnostics import ManualNavDiagnostics
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.nav_technique_ring_edge import aggregate_edge_normal_angle_deg
from spindoctor.nav_technique.technique_result import NavTechniqueResult

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext
    from spindoctor.nav_orchestrator.nav_result import NavResult
    from spindoctor.obs import ObsSnapshotInst

__all__ = ['NavTechniqueManual', 'run_manual_nav']


# Ext-FOV pixel covariance assigned to a manual-pick offset.  Operator
# precision in this dialog is limited by zoom, eye, and screen pixels;
# 1 px sigma per axis is a reasonable conservative value.  The ensemble
# never combines a manual result with auto results, so this number only
# determines what shows up in JSON metadata.
_MANUAL_OFFSET_SIGMA_PX = 1.0


class NavTechniqueManual(NavTechnique):
    """Interactive manual navigation.

    Composes every renderable feature into a single ext-FOV image plus
    mask via :func:`~spindoctor.feature.composition.compose_dialog_overlay`,
    hands the result plus the observation to the ``ManualNavDialog``,
    and packages the operator's choice into a ``NavTechniqueResult``.
    Renderable feature kinds are template-bearing (``BODY_DISC``,
    ``RING_ANNULUS``, ``CARTOGRAPHIC_MODEL``), polyline-bearing
    (``LIMB_ARC``, ``TERMINATOR_ARC``, ``RING_EDGE``), ``BODY_BLOB``
    (predicted-diameter circle outline), ``TITAN_LIMB``
    (haze-envelope circle outline), and ``STAR`` (rectangle
    outline at the predicted-vu position sized by the per-feature PSF
    bbox).

    Class attributes:
        _abstract: ``True`` — kept out of the auto-discovery registry so
            the orchestrator does not invoke the dialog during background
            navigation runs.
        accepts_feature_types: every feature type — manual navigation
            looks at whatever the scene has rendered.
    """

    _abstract = True

    name = 'NavTechniqueManual'
    accepts_feature_types = frozenset(NavFeatureType)
    requires_prior = False

    def __init__(
        self,
        *,
        config: Config | None = None,
        annotations: Annotations | None = None,
    ) -> None:
        super().__init__(config=config)
        # ``annotations`` is the merged-per-NavModel ``Annotations`` the
        # dialog uses when it writes a labeled summary PNG next to a
        # saved sidecar.  It is optional only because tests that exercise
        # the offset-pick path alone do not need it; ``run_manual_nav``
        # always populates it in normal use.
        self._annotations: Annotations | None = annotations

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Manual navigation runs whenever there is anything to render.

        Five feature kinds paint into the dialog's composite overlay
        (see :func:`~spindoctor.feature.composition.compose_dialog_overlay`):

        - template-bearing (``BODY_DISC``, ``RING_ANNULUS``,
          ``CARTOGRAPHIC_MODEL``) — full bitmap silhouette;
        - polyline-bearing (``LIMB_ARC``, ``TERMINATOR_ARC``,
          ``RING_EDGE``) — single-pixel marks at every vertex;
        - ``BODY_BLOB`` — 1-pixel circle outline at the predicted
          centroid with the predicted-diameter radius;
        - ``TITAN_LIMB`` -- 1-pixel circle outline at the predicted disc
          center with the haze-envelope radius;
        - ``STAR`` — rectangle outline at the predicted-vu position
          sized by the per-feature PSF bbox so the operator can see
          where the catalog says the star sits.

        Without any of them the dialog has nothing to display.
        """
        from spindoctor.feature.geometry import (
            BodyBlobGeometry,
            StarGeometry,
            TitanHazeGeometry,
        )

        renderable = 0
        for f in features:
            if f.template_img is not None and f.template_mask is not None:
                renderable += 1
                continue
            verts = getattr(f.geometry, 'vertices_vu', None)
            if verts is not None and np.asarray(verts).size > 0:
                renderable += 1
                continue
            if isinstance(f.geometry, (BodyBlobGeometry, TitanHazeGeometry, StarGeometry)):
                renderable += 1
        if renderable == 0:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_renderable_features_for_manual_nav',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=renderable,
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Run the dialog and convert the operator's choice to a result.

        Canceling the dialog yields a spurious result with zero
        confidence so the ensemble drops it; accepting the dialog yields
        a result with ``_MANUAL_OFFSET_SIGMA_PX`` per-axis covariance.
        """
        # Local imports keep PyQt6 out of import-time graphs; the dialog
        # is only ever loaded when manual navigation is invoked.
        from PyQt6.QtWidgets import QApplication

        from spindoctor.ui.manual_nav_dialog import ManualNavDialog

        with self.log_section('NAVIGATION PASS: MANUAL'):
            obs = context.obs
            shape = context.image_ext.shape
            model_img, model_mask = compose_dialog_overlay(
                features,
                (int(shape[0]), int(shape[1])),
            )
            app_created = False
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
                app_created = True
            dialog = ManualNavDialog(
                obs=obs,  # type: ignore[arg-type]
                model_img_ext=model_img,
                model_mask_ext=model_mask,
                annotations=self._annotations,
                config=self.config,
                constraint_normal_seed_deg=aggregate_edge_normal_angle_deg(features),
                parent=None,
            )
            accepted, chosen_offset, _last_corr = dialog.run_modal()
            if app_created:
                app.quit()
        if not accepted or chosen_offset is None:
            self.logger.info('Manual navigation canceled by user')
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in features),
                offset_px=(0.0, 0.0),
                covariance_px2=np.eye(2, dtype=np.float64),
                confidence=0.0,
                spurious=True,
                at_edge=False,
                diagnostics=ManualNavDiagnostics(operator_accepted=False),
            )
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=chosen_offset,
            covariance_px2=np.eye(2, dtype=np.float64) * (_MANUAL_OFFSET_SIGMA_PX**2),
            confidence=1.0,
            spurious=False,
            at_edge=False,
            diagnostics=ManualNavDiagnostics(operator_accepted=True),
        )


def run_manual_nav(
    obs: ObsSnapshotInst,
    *,
    config: Config | None = None,
) -> NavResult | None:
    """Open the manual-navigation dialog on a single observation.

    Builds the same NavContext + features the autonomous orchestrator
    would see, then opens :class:`~spindoctor.ui.manual_nav_dialog.ManualNavDialog`
    so the operator can pick an offset by hand.  When the operator
    accepts a pick, the dialog's offset is wrapped in a full
    :class:`~spindoctor.nav_orchestrator.NavResult` (provenance + image
    classifier + feature inventory + annotations populated identically
    to the autonomous pipeline), so callers can write the same
    ``_metadata.json`` and ``_summary.png`` outputs ``navigate_image_files``
    produces.

    Parameters:
        obs: Loaded observation snapshot to navigate.
        config: Optional :class:`~spindoctor.config.Config` override; defaults
            to :data:`~spindoctor.config.DEFAULT_CONFIG`.

    Returns:
        A :class:`NavResult` with ``status='success'`` on accept, carrying
        the same recorded corrected-pointing solution an autonomous result
        does, or ``None`` when the operator cancels or no supported overlay
        feature paints any pixels into the ext-FOV composite.
        Supported overlay types match
        :meth:`NavTechniqueManual.is_feasible`: template-bearing
        features (``BODY_DISC`` / ``RING_ANNULUS`` /
        ``CARTOGRAPHIC_MODEL``), polyline-bearing features (``LIMB_ARC``
        / ``TERMINATOR_ARC`` / ``RING_EDGE``), ``BODY_BLOB`` (1-pixel
        circle outline at the predicted centroid), ``TITAN_LIMB``
        (1-pixel circle outline at the predicted disc center with the
        haze-envelope radius), and ``STAR`` (rectangle outline at the
        predicted-vu position sized by the per-feature PSF bbox).  The
        dialog is opened only when the composed mask is non-empty; an
        off-frame blob, an off-image star, or a polyline whose vertices
        all clip out-of-bounds is treated as if no renderable feature
        were present.
    """
    # Local imports keep heavyweight dependencies (NavOrchestrator, NavModel
    # registry) out of import-time graphs for callers that only need the
    # autonomous pipeline.  ``NavResult`` is imported here too because the
    # ``nav_orchestrator`` package re-exports techniques through its
    # ``curator`` module, making a top-level import of ``NavResult``
    # circular for this module.
    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator.nav_result import NavResult
    from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
    from spindoctor.support.status_reason import NavStatusReason

    models = build_models_for_obs(obs)
    orchestrator = NavOrchestrator(models, config=config)
    # Manual nav bypasses the reliability gate: the operator visually
    # overrides the autonomous reliability decision, so even features the
    # gate would drop (low SNR, large limb uncertainty, etc.) belong on
    # the dialog overlay.
    prep = orchestrator.prepare(obs, apply_gate=False)
    technique = NavTechniqueManual(config=config, annotations=prep.annotations)
    feasibility = technique.is_feasible(prep.features)
    if not feasibility.feasible:
        IMAGE_LOGGER.warning('Manual navigation skipped: %s', feasibility.reason)
        return None
    # The metadata-only feasibility check counted features by geometry
    # presence; verify the actual composed overlay against the same
    # ext-FOV shape the dialog will display so off-frame blobs and
    # all-clipped polylines fail loudly here instead of opening an
    # empty dialog.
    shape = prep.context.image_ext.shape
    _overlay_img, overlay_mask = compose_dialog_overlay(
        prep.features, (int(shape[0]), int(shape[1]))
    )
    if not overlay_mask.any():
        IMAGE_LOGGER.warning(
            'Manual navigation skipped: composed overlay is empty (every renderable '
            'feature clipped out of the ext-FOV)'
        )
        return None
    technique_result = technique.navigate(prep.features, prep.context)
    if technique_result.spurious:
        # Cancel: no NavResult is built so the caller skips the metadata /
        # PNG write step.  The cancel reason has already been logged by
        # ``NavTechniqueManual.navigate``.
        return None
    result = NavResult.success(
        offset_px=technique_result.offset_px,
        covariance_px2=technique_result.covariance_px2,
        confidence=technique_result.confidence,
        # Operator confirmation overrides the autonomous confidence
        # rank; a manually-picked offset is treated as high-confidence
        # so downstream consumers (backplane / bundle) accept it
        # without explicit opt-in.
        confidence_rank='high',
        status_reason=NavStatusReason.OK,
        per_technique=[technique_result],
        feature_inventory=prep.feature_inventory,
        image_classifier=prep.image_classifier,
        provenance=prep.provenance,
        model_metadata=prep.model_metadata,
        annotations=prep.annotations,
    )
    # Operator-picked offsets are the highest-quality pointing in the corpus,
    # so they carry the same recorded attitude an autonomous result does; the
    # orchestrator's own navigate() is bypassed here, so the stamping is
    # applied explicitly.
    return orchestrator.with_pointing(result, obs)
