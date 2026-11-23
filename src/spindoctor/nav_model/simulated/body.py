"""Simulated-body NavModel.

Renders a body from operator-supplied geometric parameters (center, axes,
rotation, lighting) rather than from SPICE.  Used by the simulated-image
GUI to compose synthetic test scenes; the rendered body becomes a
``BODY_DISC`` ``NavFeature`` that the standard pipeline can navigate
against.  A well-resolved low-phase body also emits a ``LIMB_ARC``, and a
body at appreciable phase emits a ``TERMINATOR_ARC``; both arcs are built by
:mod:`spindoctor.nav_model.simulated.body_arcs`, which describes how they
match the SPICE-backed ``NavModelBody``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags
from spindoctor.feature.geometry import BodyDiscGeometry
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import BODY_DISC_MAX_OVERFLOW_FRACTION, TITAN_BODY_NAME
from spindoctor.nav_model.nav_model_body_base import BODY_BLOB_MIN_DIAMETER_PX, NavModelBodyBase
from spindoctor.nav_model.simulated.body_arcs import (
    BodyArcInputs,
    build_limb_arc_feature,
    build_terminator_arc_feature,
)
from spindoctor.nav_model.simulated.predicted_body import (
    PredictedBody,
    predicted_center_vu,
    predicted_semi_axes_px,
    predicted_silhouette,
    render_predicted_body,
)
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX, containing_pixel
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelBodySimulated']

# The largest canvas, in pixels on a side, the silhouette-overflow count
# renders at full resolution.  A larger body is counted on a canvas
# downsampled by a whole factor and scaled back by its square, which keeps
# memory bounded at about 32 MB of float64 however large the body is.
_OVERFLOW_CANVAS_MAX_PX: int = 2048


def _silhouette_diameter_px(body_mask: NDArrayBoolType) -> float:
    """Return the longer pixel extent of a rendered body silhouette.

    Measures the bounding-box span of the lit silhouette in each axis and
    returns the larger of the two.  Returns ``0.0`` for an empty mask.
    """
    if not body_mask.any():
        return 0.0
    rows = np.any(body_mask, axis=1)
    cols = np.any(body_mask, axis=0)
    v_indices = np.where(rows)[0]
    u_indices = np.where(cols)[0]
    v_extent = float(v_indices[-1] - v_indices[0] + 1)
    u_extent = float(u_indices[-1] - u_indices[0] + 1)
    return max(v_extent, u_extent)


def _tight_bbox_extfov(
    body_mask: NDArrayBoolType,
    *,
    ext_margin_vu: tuple[int, int],
    extfov_shape: tuple[int, int],
    diameter_px: float,
) -> tuple[int, int, int, int]:
    """Return a tight extfov-coord bbox around a data-coord body silhouette.

    The bbox is the silhouette's bounding box inflated by a slop margin
    (so the body stays inside it under a modest pointing error) and
    clamped to the extfov shape.  Returns the full extfov frame when the
    mask is empty (a degenerate render the caller will not emit features
    for).

    Parameters:
        body_mask: Data-shape boolean silhouette mask.
        ext_margin_vu: Extfov margins ``(v, u)`` to shift data coords into
            extfov coords.
        extfov_shape: Extfov array shape ``(h, w)`` to clamp against.
        diameter_px: Predicted silhouette diameter, used to size the slop.

    Returns:
        ``(v_min, u_min, v_max, u_max)`` half-open bbox in extfov coords.
    """
    ext_margin_v, ext_margin_u = ext_margin_vu
    h, w = extfov_shape
    if not body_mask.any():
        return (0, 0, int(h), int(w))
    rows = np.where(np.any(body_mask, axis=1))[0]
    cols = np.where(np.any(body_mask, axis=0))[0]
    slop = max(round(0.1 * diameter_px), 4)
    v_min = int(rows[0]) + ext_margin_v - slop
    v_max = int(rows[-1]) + ext_margin_v + slop + 1
    u_min = int(cols[0]) + ext_margin_u - slop
    u_max = int(cols[-1]) + ext_margin_u + slop + 1
    return (
        max(0, v_min),
        max(0, u_min),
        min(int(h), v_max),
        min(int(w), u_max),
    )


class NavModelBodySimulated(NavModelBodyBase):
    """Body NavModel rendered from operator-supplied simulation parameters.

    Attributes:
        apply_limb_emission_gates: When ``True`` (the default) the LIMB_ARC
            is emitted only for a well-resolved, low-phase body (the
            navigation-policy gates of
            :func:`~spindoctor.nav_model.simulated.body_arcs.build_limb_arc_feature`).
            Measurement callers (the realism match) set this to ``False``:
            the silhouette geometry exists regardless of whether a limb fit
            would be reliable, so the gates-off path skips those gates and
            emits the silhouette boundary of a phase-0 render, kept where the
            body is lit at its own phase.

    Parameters:
        name: Name of this model instance.
        obs: Observation containing image geometry (used for output shapes
            and extfov margins).
        body_name: Logical body name used in metadata and labels.
        nav_body: One body's entry in the navigator view, from the
            ``obs.nav_params['bodies']`` list, which carries no
            truth keys.  Expected keys:

            - ``name``
            - ``center_v``, ``center_u`` (pixel coordinates of the center)
            - ``range_km`` (km; subject distance, defaults to inf)
            - ``semi_a_px``, ``semi_b_px``, ``semi_c_px`` (pixels; the
              ellipsoid's semi-axes)
            - ``rotation_z`` (deg; rotation about the line of sight)
            - ``sub_observer_lat_deg`` (deg; latitude of the sub-observer point,
              positive when the north pole, along semi-axis a, tips toward the
              observer)
            - ``illumination_angle`` (deg)
            - ``phase_angle`` (deg)

            Built by ``instances_for_obs``, the entry has no crater or
            anti-aliasing keys, since they are truth.  A direct caller may
            pass them; they are ignored, and anti-aliasing is always maximal
            here.
        config: Optional ``Config`` override.
        sibling_bodies: Idealized parameter dicts of the OTHER bodies in the
            same scene (from the same filtered ``nav_params['bodies']`` list
            this body came from).  A sibling with an explicitly nearer
            ``range_km`` occludes this body's predicted limb / terminator
            arcs: occluded vertices are dropped from the emitted polylines
            and the visible-arc fractions report the loss, mirroring how the
            SPICE-backed model's per-vertex drops feed its arc fraction.
    """

    inputs = 'scene_view'

    def __init__(
        self,
        name: str,
        obs: ObsSnapshotInst,
        body_name: str,
        nav_body: dict[str, Any],
        *,
        config: Config | None = None,
        sibling_bodies: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self.apply_limb_emission_gates: bool = True
        self._body_name = body_name.upper()
        self._nav_body: dict[str, Any] = dict(nav_body)
        self._sibling_bodies: list[dict[str, Any]] = [dict(s) for s in sibling_bodies or []]
        self._model_img: NDArrayFloatType | None = None
        self._body_mask: NDArrayBoolType | None = None
        self._occluder_mask: NDArrayBoolType | None = None
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._predicted_diameter_px: float = 0.0
        self._km_per_pixel_at_limb: float = 0.0
        self._subject_range_km: float = float('inf')
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._overflow_fraction: float = 1.0
        self._arc_inputs: BodyArcInputs | None = None

    @classmethod
    def instances_for_obs(
        cls, obs: ObsSnapshotInst, *, config: Config | None = None
    ) -> list[NavModel]:
        """Build one simulated body model per body of a simulated obs.

        Reads the per-body entries of the navigator view
        (``obs.nav_params['bodies']``) -- never the full scene, whose truth
        keys stay behind the information boundary.

        A scene body may carry a ``nav_override`` mapping: the renderer draws
        the true shape, while the boundary filter overlays the override onto
        the idealized view this model consumes -- the channel that makes the
        navigation geometry diverge from the render geometry.  A scene renders
        an irregular mesh at the true pose yet predicts the same mesh with zero
        relief (the shape-mismatch fixture), or the same mesh at a different
        pose (the pose-disagreement fixture for a chaotic rotator), without
        touching the rendered image.  An override may set any idealized body
        key, including ``center_v`` and ``center_u``, which moves the predicted
        center; the shipped fixtures override only shape or pose, so their
        predicted body stays at the unshifted position the planted offset is
        measured from.

        Titan is excluded, mirroring the catalog-driven model's own
        exclusion: its opaque haze hides the surface an ellipsoid limb /
        terminator / disc prediction assumes, so the shape-based techniques
        are systematically wrong on it rather than merely noisy.  A
        simulated Titan is navigated by the haze model instead
        (``NavModelTitanSimulated``); without this exclusion both models
        would claim the same body and the frame would be navigated twice,
        once by a technique family that cannot see the limb it is fitting.

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instances.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelBodySimulated`` per non-Titan body in the sim
            scene.
        """
        nav_params = getattr(obs, 'nav_params', None)
        if not isinstance(nav_params, dict):
            return []
        bodies = [bp for bp in nav_params.get('bodies', []) if isinstance(bp, dict)]
        out: list[NavModel] = []
        for index, body_params in enumerate(bodies):
            body_name = str(body_params.get('name', 'SIM-BODY'))
            if body_name.upper() == TITAN_BODY_NAME:
                continue
            siblings = [dict(bp) for j, bp in enumerate(bodies) if j != index]
            out.append(
                cls(
                    f'body_sim:{body_name}',
                    obs,
                    body_name,
                    dict(body_params),
                    config=config,
                    sibling_bodies=siblings,
                )
            )
        return out

    def create_model(self) -> None:
        """Render the simulated body and populate masks, annotations, metadata."""
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None
        metadata['body_name'] = self._body_name
        self._metadata.clear()
        self._metadata.update(metadata)
        with self.log_section(f'CREATE SIMULATED BODY MODEL FOR: {self._body_name}'):
            self._render()
        end_time = now_dt()
        self._metadata['end_time'] = end_time.isoformat()
        self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _render_prediction(
        self,
        phase_angle_rad: float,
        *,
        size: tuple[int, int] | None = None,
        center: tuple[float, float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> PredictedBody:
        """Render a body's data-coordinate image and silhouette at the given phase angle.

        The predicted shape is read from this model's own params, which need
        not match what was rendered into the image: an irregular body can be
        predicted as a mesh (matching pose), as an ellipsoid (shape mismatch),
        or at a deliberately different pose (chaotic-rotator fixture).

        Parameters:
            phase_angle_rad: Phase angle of the render in radians.  The
                model's own phase for the navigation prediction; 0 for the
                full-silhouette render of the measurement limb path.
            size: Optional canvas shape ``(v, u)``; defaults to the obs data
                shape.  The unclipped whole-body render behind the
                visible-arc fraction passes a body-sized canvas here.
            center: Optional body center ``(v, u)`` on that canvas; defaults
                to the rendered params' predicted center on the frame, which
                puts a body that states no center at the center of the
                observation's data shape whatever canvas is given.  Every
                caller that passes a canvas passes its center too.
            params: Optional body-parameter dict to render instead of this
                model's own ``nav_body``.  The overflow fraction renders a
                downscaled copy of the body through this.

        Returns:
            The predicted image and silhouette on the requested canvas.
        """
        p = self._nav_body if params is None else params
        data_shape_vu = (int(self.obs.data_shape_v), int(self.obs.data_shape_u))
        return render_predicted_body(
            p,
            phase_angle_rad,
            shape_vu=data_shape_vu if size is None else size,
            center_vu=predicted_center_vu(p, data_shape_vu) if center is None else center,
        )

    def _render(self) -> None:
        """Generate the simulated image and the matching masks."""
        p = self._nav_body
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        phase_angle_rad = float(np.radians(p.get('phase_angle', 0.0)))
        center_v, center_u = predicted_center_vu(p, (data_size_v, data_size_u))
        prediction = self._render_prediction(phase_angle_rad)
        sim_img = prediction.image
        body_mask = prediction.silhouette
        limb_mask = self._compute_limb_mask_from_body_mask(body_mask)
        model_img_full = self.obs.make_extfov_zeros()
        limb_mask_full = self.obs.make_extfov_false()
        body_mask_full = self.obs.make_extfov_false()
        lit_mask_full = self.obs.make_extfov_false()
        slice_v = slice(ext_margin_v, ext_margin_v + data_size_v)
        slice_u = slice(ext_margin_u, ext_margin_u + data_size_u)
        model_img_full[slice_v, slice_u] = sim_img
        limb_mask_full[slice_v, slice_u] = limb_mask
        body_mask_full[slice_v, slice_u] = body_mask
        lit_mask_full[slice_v, slice_u] = prediction.lit
        self._model_img = model_img_full
        self._body_mask = body_mask_full
        # The scene states the body's center in pixel corner coordinates,
        # which is what the silhouette renderer draws against (it puts the
        # center of pixel i at i + 0.5).  The payload is pixel centric, and so
        # is the lit-weighted centroid this value is differenced against, so
        # the half pixel comes off next to the margin that goes on.
        self._predicted_center_vu = (
            center_v - PIXEL_CENTER_TO_CORNER_PX + ext_margin_v,
            center_u - PIXEL_CENTER_TO_CORNER_PX + ext_margin_u,
        )
        self._subject_range_km = float(p.get('range_km', float('inf')))
        self._occluder_mask = self._compute_occluder_mask()
        # Predicted disc diameter: the longer pixel extent of the rendered
        # silhouette.  Drives the BODY_BLOB emission gate and covariance.
        self._predicted_diameter_px = _silhouette_diameter_px(body_mask)
        self._overflow_fraction = self._silhouette_overflow_fraction(
            int(np.count_nonzero(body_mask)), phase_angle_rad, center_v, center_u
        )
        self._metadata['overflow_fraction'] = self._overflow_fraction
        # Tight extfov-coord bounding box around the body silhouette (plus
        # slop), matching the convention of the SPICE-backed ``NavModelBody``.
        # A whole-frame bbox would make ``BodyBlobNav`` integrate its observed
        # centroid over the entire frame, so scattered above-noise sky pixels
        # would swamp a small or high-phase lit region; a tight bbox keeps the
        # moment local to the body.  The slop lets the body stay inside the
        # bbox under a modest pointing error.
        self._bbox_extfov_vu = _tight_bbox_extfov(
            body_mask,
            ext_margin_vu=(ext_margin_v, ext_margin_u),
            extfov_shape=(model_img_full.shape[0], model_img_full.shape[1]),
            diameter_px=self._predicted_diameter_px,
        )
        # Physical scale at the limb.  The sim FOV is a dummy flat FOV with no
        # real angular scale, so km/pixel is only known when the scene states
        # it explicitly; absent that it stays 0, which makes the shared
        # phase-irregularity factor collapse to 0 (the regular-body case).
        self._km_per_pixel_at_limb = float(p.get('km_per_pixel', 0.0))
        phase_angle_deg = float(p.get('phase_angle', 0.0))
        self._metadata['phase_angle_deg'] = phase_angle_deg
        self._metadata['predicted_diameter_px'] = float(self._predicted_diameter_px)
        self._arc_inputs = BodyArcInputs(
            obs=self.obs,
            nav_body=self._nav_body,
            config=self._config,
            body_name=self._body_name,
            source_model=self.name,
            model_img=model_img_full,
            body_mask=body_mask_full,
            lit_mask=lit_mask_full,
            limb_mask=limb_mask_full,
            occluder_mask=self._occluder_mask,
            bbox_extfov_vu=self._bbox_extfov_vu,
            subject_range_km=self._subject_range_km,
            predicted_diameter_px=self._predicted_diameter_px,
            phase_angle_deg=phase_angle_deg,
            render=self._render_prediction,
            silhouette_boundary=self._compute_limb_mask_from_body_mask,
        )

    def _silhouette_overflow_fraction(
        self, in_sensor_pixels: int, phase_angle_rad: float, center_v: float, center_u: float
    ) -> float:
        """The fraction of the whole predicted silhouette outside the sensor.

        A body whose bounding square (half again its longest axis, as mesh
        relief can push past the ellipsoid bound) lies inside the frame
        overflows by exactly 0, with no render.  Otherwise the silhouette is
        rendered unclipped on a canvas comfortably larger than the body, with
        the body's center on the same sub-pixel phase it has in the frame.  A
        canvas wider than ``_OVERFLOW_CANVAS_MAX_PX`` is rendered downsampled
        by a whole factor and its pixel count scaled back, so memory stays
        bounded for a frame-filling body.  The quantity is the SPICE-backed
        model's: ``1 - in-sensor pixels / whole-silhouette pixels``, from this
        model's own predicted geometry.

        Parameters:
            in_sensor_pixels: Silhouette pixels the frame-sized render holds.
            phase_angle_rad: The prediction's phase angle in radians.
            center_v: The predicted center v on the frame, pixel corner.
            center_u: The predicted center u on the frame, pixel corner.

        Returns:
            The overflow fraction in [0, 1]; 1 for a silhouette too small to
            cover a pixel.
        """
        p = self._nav_body
        max_semi = max(predicted_semi_axes_px(p))
        # Half again the longest semi-axis (mesh relief can push past the
        # ellipsoid bound), and a canvas of half again the longest diameter plus
        # a fixed margin, as the arc fractions use.
        reach = max_semi * 1.5
        if (
            center_v - reach >= 0.0
            and center_u - reach >= 0.0
            and center_v + reach <= float(self.obs.data_shape_v)
            and center_u + reach <= float(self.obs.data_shape_u)
        ):
            return 0.0
        canvas = math.ceil(max_semi * 3.0) + 16
        factor = max(1, math.ceil(canvas / _OVERFLOW_CANVAS_MAX_PX))
        if factor == 1:
            half = canvas // 2
            full = self._render_prediction(
                phase_angle_rad,
                size=(canvas, canvas),
                center=(
                    half + center_v - math.floor(center_v),
                    half + center_u - math.floor(center_u),
                ),
            )
            total = float(np.count_nonzero(full.silhouette))
        else:
            small = math.ceil(canvas / factor)
            scaled = dict(p)
            for axis in ('semi_a_px', 'semi_b_px', 'semi_c_px'):
                if axis in scaled:
                    scaled[axis] = float(scaled[axis]) / factor
            small_render = self._render_prediction(
                phase_angle_rad,
                size=(small, small),
                center=(small / 2.0, small / 2.0),
                params=scaled,
            )
            total = float(np.count_nonzero(small_render.silhouette)) * factor * factor
        return max(0.0, 1.0 - in_sensor_pixels / max(total, 1.0))

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the body's NavFeatures.

        Emits a ``BODY_DISC`` carrying the rendered template (for the
        correlation technique) unless more than
        :data:`~spindoctor.nav_model.nav_model_body.BODY_DISC_MAX_OVERFLOW_FRACTION`
        of the predicted silhouette falls outside the sensor.  Also emits a
        ``BODY_BLOB`` whenever the predicted silhouette is large enough and its
        overflow is at most the ``bodies`` configuration value
        ``blob_max_overflow_fraction`` -- the lit-weighted centroid is
        orientation-independent, so it is the technique that navigates small,
        high-phase, or irregular bodies that the disc correlation cannot.  A
        well-resolved low-phase body adds a ``LIMB_ARC``; a body at appreciable
        phase adds a ``TERMINATOR_ARC`` (the lit/unlit boundary interior to the
        disc), each matching the SPICE-backed body model's gate rules.  Both
        overflow gates are the SPICE-backed model's, read from this model's
        own predicted silhouette.
        """
        arc_inputs = self._arc_inputs
        if arc_inputs is None:
            return []
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        template_img = arc_inputs.model_img[v_min:v_max, u_min:u_max].copy()
        template_mask = arc_inputs.body_mask[v_min:v_max, u_min:u_max].copy()
        # A nearer sibling body hides part of this disc; the image shows the
        # occluder there, not this body.  Drop the occluded pixels from the
        # correlation template so it scores only against disc brightness the
        # image actually carries, and report a visible fraction that falls
        # with occlusion depth (the sim mirror of the SPICE-backed model, so
        # the two do not desync on disc occlusion).
        disc_visible_fraction = 1.0
        if self._occluder_mask is not None:
            occluded_bbox = self._occluder_mask[v_min:v_max, u_min:u_max]
            total_disc = int(np.count_nonzero(template_mask))
            template_img[occluded_bbox] = 0.0
            template_mask[occluded_bbox] = False
            visible_disc = int(np.count_nonzero(template_mask))
            disc_visible_fraction = visible_disc / max(total_disc, 1)
        features: list[NavFeature] = []
        if self._overflow_fraction <= BODY_DISC_MAX_OVERFLOW_FRACTION:
            features.append(
                NavFeature(
                    feature_id=f'body_disc:{self._body_name}',
                    feature_type=NavFeatureType.BODY_DISC,
                    source_model=self.name,
                    geometry=BodyDiscGeometry(
                        bbox_extfov_vu=self._bbox_extfov_vu,
                        predicted_center_vu=self._predicted_center_vu,
                        overflow_fraction=self._overflow_fraction,
                    ),
                    subject_range_km=self._subject_range_km,
                    position_cov_px=None,
                    intensity_sigma_rel=0.0,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=disc_visible_fraction,
                    reliability_reasons=NavReliabilityBreakdown(
                        visible_lit_fraction=disc_visible_fraction,
                        overflow_fraction=self._overflow_fraction,
                    ),
                    usable_types=frozenset({NavFeatureType.BODY_DISC}),
                    flags=BodyDiscFlags(
                        body_name=self._body_name, overflow_fov_fraction=self._overflow_fraction
                    ),
                    template_img=template_img,
                    template_mask=template_mask,
                )
            )
        else:
            self._logger.info(
                'BODY_DISC withheld: silhouette overflow %.3f exceeds %.3f',
                self._overflow_fraction,
                BODY_DISC_MAX_OVERFLOW_FRACTION,
            )
        shape = load_body_shape(self._body_name, config=self._config)
        blob_min_px = max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)
        if self._predicted_diameter_px >= blob_min_px:
            if self._blob_overflow_ok():
                features.append(self._build_blob_feature(shape, context=context))
            else:
                # A clipped body's predicted centroid spans the whole
                # silhouette while the image holds only the in-sensor part, so
                # its centroid is displaced before any pointing error (#746).
                self._logger.info(
                    'BODY_BLOB withheld: silhouette overflow %.3f exceeds %.3f',
                    self._overflow_fraction,
                    self._blob_max_overflow_fraction(),
                )
        limb_feature = build_limb_arc_feature(
            arc_inputs, apply_emission_gates=self.apply_limb_emission_gates
        )
        if limb_feature is not None:
            features.append(limb_feature)
        terminator_feature = build_terminator_arc_feature(arc_inputs)
        if terminator_feature is not None:
            features.append(terminator_feature)
        return features

    def _compute_occluder_mask(self) -> NDArrayBoolType | None:
        """Union silhouette (extfov coords) of explicitly nearer sibling bodies.

        A sibling occludes this body only when BOTH ranges are explicit and
        the sibling's is strictly smaller -- the navigator-side mirror of the
        renderer's rule that overlapping bodies must all carry an explicit
        ``range_km`` before their stacking means anything.  Each occluder's
        silhouette is its own predicted silhouette, the pixels its outline
        touches: a solid body occludes with its full silhouette, lit or not.

        Returns:
            The extfov-shape boolean mask, or ``None`` when no sibling
            occludes (the common single-body case costs nothing).
        """
        # Occlusion requires a known depth order: without the subject's own
        # explicit range there is nothing to compare against, so no sibling is
        # treated as nearer and the template is left whole rather than erased.
        if 'range_km' not in self._nav_body:
            return None
        own_range = float(self._nav_body['range_km'])
        occluders = [
            s for s in self._sibling_bodies if 'range_km' in s and float(s['range_km']) < own_range
        ]
        if not occluders:
            return None
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        mask_full = self.obs.make_extfov_false()
        for sibling in occluders:
            mask_full |= predicted_silhouette(
                sibling,
                (data_size_v, data_size_u),
                margin_vu=(ext_margin_v, ext_margin_u),
            )
        if not mask_full.any():
            return None
        return mask_full

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit body silhouette + label annotations for the summary PNG."""
        arc_inputs = self._arc_inputs
        if arc_inputs is None:
            return Annotations()
        # The label anchor is a pixel of the nominal frame, so it comes off
        # the payload's extended-frame pixel-centric center rather than off
        # the scene's pixel-corner one.
        v_center, u_center = self._predicted_center_vu
        return self._create_annotations(
            containing_pixel(u_center - self.obs.extfov_margin_u),
            containing_pixel(v_center - self.obs.extfov_margin_v),
            arc_inputs.model_img,
            arc_inputs.limb_mask,
            arc_inputs.body_mask,
        )
