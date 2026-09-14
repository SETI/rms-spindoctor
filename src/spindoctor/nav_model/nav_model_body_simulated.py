"""Simulated-body NavModel.

Renders a body from operator-supplied geometric parameters (center, axes,
rotation, lighting) rather than from SPICE.  Used by the simulated-image
GUI to compose synthetic test scenes; the rendered body becomes a
``BODY_DISC`` ``NavFeature`` that the standard pipeline can navigate
against.  A well-resolved low-phase body also emits a ``LIMB_ARC``, and a
body at appreciable phase emits a ``TERMINATOR_ARC`` -- both matching the
SPICE-backed ``NavModelBody``'s geometry and gating semantics (polyline
conventions, emission gates, reliability inputs), so a sim scene exercises
the same techniques a real frame would.  The per-vertex sigma model is the
deliberate exception: the real model derives per-vertex sigmas from the
PSF, limb softness, and albedo terms, while the noise-free sim render
carries fixed per-vertex values (see ``_TERMINATOR_SIGMA_NORMAL_PX``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from oops import Observation

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags, LimbArcFlags, TerminatorArcFlags
from spindoctor.feature.geometry import BodyDiscGeometry, LimbPolyline, TerminatorPolyline
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import (
    LIMB_ARC_MIN_VERTICES,
    TERMINATOR_MIN_PHASE_FACTOR,
    TERMINATOR_MIN_VERTICES,
    TITAN_BODY_NAME,
    limb_reliability,
    shape_features_suppressed,
    terminator_reliability,
)
from spindoctor.nav_model.nav_model_body_base import BODY_BLOB_MIN_DIAMETER_PX, NavModelBodyBase
from spindoctor.nav_model.sim_body import create_simulated_body
from spindoctor.sim.ellipsoid_geometry import DARK_SIDE_ILLUM_STRENGTH
from spindoctor.sim.mesh_geometry import mesh_spec_from_params, render_mesh_body_image
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX, containing_pixel
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.image import shift_array
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelBodySimulated']

# Minimum limb-polyline vertices to emit a LIMB_ARC.  Matches BodyLimbNav's
# ``min_arc_vertices`` feasibility floor: a shorter arc cannot constrain the
# fit.  Imported from the catalog body model so the two emission gates cannot
# desync.
_MIN_LIMB_ARC_VERTICES: int = LIMB_ARC_MIN_VERTICES
# Minimum silhouette diameter to emit a LIMB_ARC.  The catalog body model gates
# the limb on its ellipsoid-fit uncertainty (``<= 3 px``) plus the shared
# vertex-count floor; the sim has no km scale, so it gates on resolution
# directly.  Below this the limb fit is imprecise and -- because it is an
# LM-refined distance-transform fit -- it injects cross-process jitter into the
# fused offset of any body scene, so only a well-resolved body emits a limb.
_MIN_LIMB_DIAMETER_PX: float = 100.0
# Per-vertex limb uncertainty for a simulated body.  The rendered silhouette edge
# is sharp and noise-free, so the predicted limb sits within ~1 px of the image
# edge; the tangent sigma reflects the one-pixel polyline sampling resolution.
_LIMB_SIGMA_NORMAL_PX: float = 1.0
_LIMB_SIGMA_TANGENT_PX: float = 0.5
# Above this phase the lit limb is a thin crescent and most of the silhouette
# boundary is terminator (a soft brightness gradient, not a sharp image edge), so
# a limb fit is unreliable -- the body is then navigated by the blob centroid.
# Gating emission here keeps LIMB_ARC off the high-phase scenes and matches the
# catalog body model's limb/blob handoff.
_LIMB_MAX_PHASE_DEG: float = 60.0
# Per-vertex terminator uncertainty for a simulated body.  The terminator is a
# soft photometric boundary (a shading ramp crossing zero) rather than a sharp
# silhouette edge, so its normal sigma is one pixel wider than the limb's; the
# tangent sigma reflects the one-pixel polyline sampling resolution.  The
# catalog body model derives these from the ellipsoid residual and albedo; the
# noise-free sim render has neither, so fixed values stand in.
_TERMINATOR_SIGMA_NORMAL_PX: float = 2.0
_TERMINATOR_SIGMA_TANGENT_PX: float = 0.5


def _limb_polyline_from_mask(
    limb_mask: NDArrayBoolType,
    body_mask: NDArrayBoolType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Extract limb vertices and outward normals from a 1-pixel limb mask.

    Each ``True`` pixel of ``limb_mask`` (a body pixel adjacent to sky) becomes
    one vertex.  The outward normal points away from the body interior: it is the
    normalized sum of the unit directions toward each non-body 4-neighbor, so a
    vertex on the sunward limb gets a normal pointing into the sky.

    Parameters:
        limb_mask: Extfov-shape boolean limb mask (the silhouette boundary).
        body_mask: Extfov-shape boolean body silhouette mask.

    Returns:
        ``(vertices_vu, normals_vu)`` each shaped ``(N, 2)``; empty when the mask
        has no pixels.
    """
    if not limb_mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    vs, us = np.where(limb_mask)
    vertices_vu = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    rows, cols = body_mask.shape
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        dv = 0.0
        du = 0.0
        if v > 0 and not body_mask[v - 1, u]:
            dv -= 1.0
        if v < rows - 1 and not body_mask[v + 1, u]:
            dv += 1.0
        if u > 0 and not body_mask[v, u - 1]:
            du -= 1.0
        if u < cols - 1 and not body_mask[v, u + 1]:
            du += 1.0
        norm = float(np.hypot(dv, du)) or 1.0
        normals_vu[i, 0] = dv / norm
        normals_vu[i, 1] = du / norm
    return vertices_vu, normals_vu


def _terminator_ridge_mask(
    model_img: NDArrayFloatType,
) -> tuple[NDArrayBoolType, NDArrayBoolType]:
    """Interior lit/unlit boundary ridge of a rendered body image.

    The shading floors the visible-but-unlit hemisphere at
    ``DARK_SIDE_ILLUM_STRENGTH``, so brightness ``> 0`` is the whole visible
    disc while brightness above the floor is the lit region; the unlit disc is
    their difference.  A ridge pixel is a lit pixel with an unlit *interior*
    disc pixel as a 4-neighbor -- the interior restriction drops the
    anti-aliased limb ring (unlit only because its edge brightness has ramped
    below the floor), which keeps the ridge off the silhouette everywhere the
    lit and unlit regions are separated by more than a pixel (the cusps of a
    very thin crescent are the exception; see ``_terminator_polyline``).

    Parameters:
        model_img: A rendered body image (any canvas).

    Returns:
        ``(ridge_mask, lit_mask)`` boolean arrays of the same shape.
    """
    disc = model_img > 0.0
    lit = model_img > DARK_SIDE_ILLUM_STRENGTH
    dark_disc = disc & ~lit
    touches_sky = (
        shift_array(~disc, (-1, 0))
        | shift_array(~disc, (1, 0))
        | shift_array(~disc, (0, -1))
        | shift_array(~disc, (0, 1))
    )
    dark_disc_interior = dark_disc & ~touches_sky
    neighbor_dark = (
        shift_array(dark_disc_interior, (-1, 0))
        | shift_array(dark_disc_interior, (1, 0))
        | shift_array(dark_disc_interior, (0, -1))
        | shift_array(dark_disc_interior, (0, 1))
    )
    ridge = lit & neighbor_dark
    return ridge, lit


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
            navigation-policy gates below).  Measurement callers (the
            realism match) set this to ``False``: the silhouette geometry
            exists regardless of whether a limb fit would be reliable, and
            the gates-off path emits the *lit geometric limb* -- the
            phase-0 silhouette boundary restricted to lit vertices --
            matching the real body model's LIMB_ARC definition instead of
            the lit-region boundary (which mixes limb and terminator at
            nonzero phase).

    Parameters:
        name: Name of this model instance.
        obs: Observation containing image geometry (used for output shapes
            and extfov margins).
        body_name: Logical body name used in metadata and labels.
        sim_params: Dictionary of simulation parameters.  Expected keys:

            - ``name``
            - ``center_v``, ``center_u`` (pixel coordinates of the center)
            - ``range_km`` (km; subject distance, defaults to inf)
            - ``axis1``, ``axis2``, ``axis3`` (pixels; full widths of the
              ellipsoid axes)
            - ``rotation_z`` (deg; rotation about the line of sight)
            - ``rotation_tilt`` (deg; tilt of the body)
            - ``illumination_angle`` (deg)
            - ``phase_angle`` (deg)

            Crater and anti-aliasing keys are accepted but ignored;
            anti-aliasing is always maximal here.
        config: Optional ``Config`` override.
        sibling_bodies: Idealized parameter dicts of the OTHER bodies in the
            same scene (from the same filtered ``nav_params['bodies']`` list
            this body came from).  A sibling with an explicitly nearer
            ``range_km`` occludes this body's predicted limb / terminator
            arcs: occluded vertices are dropped from the emitted polylines
            and the visible-arc fractions report the loss, mirroring how the
            SPICE-backed model's per-vertex drops feed its arc fraction.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        body_name: str,
        sim_params: dict[str, Any],
        *,
        config: Config | None = None,
        sibling_bodies: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self.apply_limb_emission_gates: bool = True
        self._body_name = body_name.upper()
        self._sim_params: dict[str, Any] = dict(sim_params)
        self._sibling_bodies: list[dict[str, Any]] = [dict(s) for s in sibling_bodies or []]
        self._model_img: NDArrayFloatType | None = None
        self._body_mask: NDArrayBoolType | None = None
        self._limb_mask: NDArrayBoolType | None = None
        self._occluder_mask: NDArrayBoolType | None = None
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._predicted_diameter_px: float = 0.0
        self._km_per_pixel_at_limb: float = 0.0
        self._subject_range_km: float = float('inf')
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Build one simulated body model per body of a simulated obs.

        Reads the per-body entries of the filtered idealized view
        (``obs.nav_params['bodies']``) -- never the full scene, whose truth
        keys stay behind the information boundary.  Returns an empty list for
        a real obs, so the SPICE-backed ``NavModelBody`` handles those instead.

        A scene body may carry a ``nav_override`` mapping: the renderer draws
        the true shape, while the boundary filter overlays the override onto
        the idealized view this model consumes -- the channel that makes the
        navigation geometry diverge from the render geometry.  A scene renders
        an irregular mesh at the true pose yet predicts an ellipsoid (shape
        mismatch, B7 scenario 2), or the same mesh at a different pose
        (chaotic-rotator pose disagreement, B7 scenario 3), without touching
        the rendered image.  The override never changes the center, so the
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
        if not getattr(obs, 'is_simulated', False):
            return []
        nav_params = getattr(obs, 'nav_params', None)
        if not isinstance(nav_params, dict):
            return []
        bodies = [bp for bp in nav_params.get('bodies', []) or [] if isinstance(bp, dict)]
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

    def _render_body_image(
        self,
        phase_angle_rad: float,
        *,
        size: tuple[int, int] | None = None,
        center: tuple[float, float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> NDArrayFloatType:
        """Render a body's data-coordinate image at the given phase angle.

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
                to the rendered params' own predicted center.
            params: Optional body-parameter dict to render instead of this
                model's own ``sim_params``.  The occlusion path renders each
                nearer sibling's predicted silhouette through this.

        Returns:
            The rendered image on the requested canvas.
        """
        p = self._sim_params if params is None else params
        data_size_v, data_size_u = (
            size
            if size is not None
            else (
                int(self.obs.data_shape_v),
                int(self.obs.data_shape_u),
            )
        )
        rotation_z_rad = float(np.radians(p.get('rotation_z', 0.0)))
        rotation_tilt_rad = float(np.radians(p.get('rotation_tilt', 0.0)))
        illumination_angle_rad = float(np.radians(p.get('illumination_angle', 0.0)))
        if center is not None:
            center_v, center_u = center
        else:
            center_v = float(p.get('center_v', data_size_v / 2.0))
            center_u = float(p.get('center_u', data_size_u / 2.0))
        axis1 = float(p.get('axis1', 0.0))
        axis2 = float(p.get('axis2', 0.0))
        axis3 = float(p.get('axis3', min(axis1, axis2)))
        if str(p.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh':
            return render_mesh_body_image(
                size=(data_size_v, data_size_u),
                center=(center_v, center_u),
                semi_axes_px=(axis1 / 2.0, axis2 / 2.0, axis3 / 2.0),
                spec=mesh_spec_from_params(p),
                illumination_angle=illumination_angle_rad,
                phase_angle=phase_angle_rad,
                anti_aliasing=1.0,
            )
        return create_simulated_body(
            size=(data_size_v, data_size_u),
            center=(center_v, center_u),
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=rotation_z_rad,
            rotation_tilt=rotation_tilt_rad,
            illumination_angle=illumination_angle_rad,
            phase_angle=phase_angle_rad,
            anti_aliasing=1.0,
        )

    def _render(self) -> None:
        """Generate the simulated image and the matching masks."""
        p = self._sim_params
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        phase_angle_rad = float(np.radians(p.get('phase_angle', 0.0)))
        center_v = float(p.get('center_v', data_size_v / 2.0))
        center_u = float(p.get('center_u', data_size_u / 2.0))
        sim_img = self._render_body_image(phase_angle_rad)
        body_mask = sim_img > 0.0
        limb_mask = self._compute_limb_mask_from_body_mask(body_mask)
        model_img_full = self.obs.make_extfov_zeros()
        limb_mask_full = self.obs.make_extfov_false()
        body_mask_full = self.obs.make_extfov_false()
        slice_v = slice(ext_margin_v, ext_margin_v + data_size_v)
        slice_u = slice(ext_margin_u, ext_margin_u + data_size_u)
        model_img_full[slice_v, slice_u] = sim_img
        limb_mask_full[slice_v, slice_u] = limb_mask
        body_mask_full[slice_v, slice_u] = body_mask
        self._model_img = model_img_full
        self._body_mask = body_mask_full
        self._limb_mask = limb_mask_full
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
        self._metadata['phase_angle_deg'] = float(p.get('phase_angle', 0.0))
        self._metadata['predicted_diameter_px'] = float(self._predicted_diameter_px)

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the body's NavFeatures.

        Always emits a ``BODY_DISC`` carrying the rendered template (for the
        correlation technique).  Also emits a ``BODY_BLOB`` whenever the
        predicted silhouette is large enough -- the lit-weighted centroid is
        orientation-independent, so it is the technique that navigates small,
        high-phase, or irregular bodies that the disc correlation cannot.  A
        well-resolved low-phase body adds a ``LIMB_ARC``; a body at appreciable
        phase adds a ``TERMINATOR_ARC`` (the lit/unlit boundary interior to the
        disc), each matching the SPICE-backed body model's gate rules.
        """
        if self._model_img is None or self._body_mask is None:
            return []
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        template_img = self._model_img[v_min:v_max, u_min:u_max].copy()
        template_mask = self._body_mask[v_min:v_max, u_min:u_max].copy()
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
        features: list[NavFeature] = [
            NavFeature(
                feature_id=f'body_disc:{self._body_name}',
                feature_type=NavFeatureType.BODY_DISC,
                source_model=self.name,
                geometry=BodyDiscGeometry(
                    bbox_extfov_vu=self._bbox_extfov_vu,
                    predicted_center_vu=self._predicted_center_vu,
                    overflow_fraction=0.0,
                ),
                subject_range_km=self._subject_range_km,
                position_cov_px=None,
                intensity_sigma_rel=0.0,
                preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                reliability=disc_visible_fraction,
                reliability_reasons=NavReliabilityBreakdown(
                    visible_lit_fraction=disc_visible_fraction, overflow_fraction=0.0
                ),
                usable_types=frozenset({NavFeatureType.BODY_DISC}),
                flags=BodyDiscFlags(body_name=self._body_name, overflow_fov_fraction=0.0),
                template_img=template_img,
                template_mask=template_mask,
            )
        ]
        shape = load_body_shape(self._body_name, config=self._config)
        blob_min_px = max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)
        if self._predicted_diameter_px >= blob_min_px:
            features.append(self._build_blob_feature(shape, context=context))
        limb_feature = self._build_limb_arc_feature()
        if limb_feature is not None:
            features.append(limb_feature)
        terminator_feature = self._build_terminator_arc_feature()
        if terminator_feature is not None:
            features.append(terminator_feature)
        return features

    def _lit_geometric_limb_polyline(self) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Lit geometric-limb polyline for the gates-off measurement path.

        The navigation polyline is the boundary of the rendered *lit* region,
        which at nonzero phase mixes the sharp geometric limb with the soft
        terminator (the mask edge sits where the shading ramp crosses zero).
        The realism measurement compares against the real body model's
        LIMB_ARC, which carries only the lit part of the geometric silhouette
        boundary and emits the terminator as a separate feature -- so this
        path reproduces that definition: the silhouette boundary of a phase-0
        render of the same body, restricted to vertices lit in the true-phase
        image, with outward normals taken from the full silhouette.

        Returns:
            ``(vertices_vu, normals_vu)`` in extfov coordinates; empty when
            the body renders no lit silhouette-boundary pixels.
        """
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        if self._body_mask is None:
            return empty, empty
        full_img = self._render_body_image(0.0)
        full_mask = full_img > 0.0
        boundary = self._compute_limb_mask_from_body_mask(full_mask)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        slice_v = slice(ext_margin_v, ext_margin_v + int(self.obs.data_shape_v))
        slice_u = slice(ext_margin_u, ext_margin_u + int(self.obs.data_shape_u))
        lit = np.asarray(self._body_mask[slice_v, slice_u], dtype=bool)
        full_mask_ext = self.obs.make_extfov_false()
        lit_boundary_ext = self.obs.make_extfov_false()
        full_mask_ext[slice_v, slice_u] = full_mask
        lit_boundary_ext[slice_v, slice_u] = boundary & lit
        return _limb_polyline_from_mask(lit_boundary_ext, full_mask_ext)

    def _compute_occluder_mask(self) -> NDArrayBoolType | None:
        """Union silhouette (extfov coords) of explicitly nearer sibling bodies.

        A sibling occludes this body only when BOTH ranges are explicit and
        the sibling's is strictly smaller -- the navigator-side mirror of the
        renderer's rule that overlapping bodies must all carry an explicit
        ``range_km`` before their stacking means anything.  Each occluder's
        silhouette is its own predicted render (``> 0`` covers the whole
        visible disc: the shading floors the unlit hemisphere above zero, and
        a solid body occludes with its full silhouette, lit or not).

        Returns:
            The extfov-shape boolean mask, or ``None`` when no sibling
            occludes (the common single-body case costs nothing).
        """
        # Occlusion requires a known depth order: without the subject's own
        # explicit range there is nothing to compare against, so no sibling is
        # treated as nearer and the template is left whole rather than erased.
        if 'range_km' not in self._sim_params:
            return None
        own_range = float(self._sim_params['range_km'])
        occluders = [
            s for s in self._sibling_bodies if 'range_km' in s and float(s['range_km']) < own_range
        ]
        if not occluders:
            return None
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        mask = np.zeros((data_size_v, data_size_u), dtype=np.bool_)
        for sibling in occluders:
            phase_rad = float(np.radians(sibling.get('phase_angle', 0.0)))
            mask |= self._render_body_image(phase_rad, params=sibling) > 0.0
        if not mask.any():
            return None
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        mask_full = self.obs.make_extfov_false()
        mask_full[
            ext_margin_v : ext_margin_v + data_size_v,
            ext_margin_u : ext_margin_u + data_size_u,
        ] = mask
        return mask_full

    def _drop_occluded_vertices(
        self,
        vertices_vu: NDArrayFloatType,
        normals_vu: NDArrayFloatType,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Drop polyline vertices hidden behind a nearer sibling body.

        A hidden arc has no counterpart edge in the image (the occluder's
        disc covers it), so fitting it could only anchor on the occluder's
        own limb; dropping it keeps the DT fit on arcs that exist and lets
        the visible-arc fraction report the loss honestly.

        Parameters:
            vertices_vu: Polyline vertices, extfov coords, shape ``(N, 2)``.
            normals_vu: Matching outward normals, shape ``(N, 2)``.

        Returns:
            The surviving ``(vertices_vu, normals_vu)``; unchanged when no
            sibling occludes.
        """
        if self._occluder_mask is None or vertices_vu.shape[0] == 0:
            return vertices_vu, normals_vu
        vs = vertices_vu[:, 0].astype(np.intp)
        us = vertices_vu[:, 1].astype(np.intp)
        keep = ~self._occluder_mask[vs, us]
        return vertices_vu[keep], normals_vu[keep]

    def _limb_visible_arc_fraction(self, visible_vertex_count: int) -> float:
        """Fraction of the full predicted silhouette boundary the fit sees.

        The sim analog of the SPICE-backed model's ``_visible_arc_fraction``
        (survivors over the predicted ridge): the denominator is the
        silhouette-boundary length of an *unclipped* render of the same body
        at the same phase -- a canvas sized to the whole silhouette,
        independent of the frame -- and the numerator is the vertex count
        that survived frame clipping and sibling-body occlusion.  A fully
        framed, unoccluded body scores ~1.0; a body sliding off the frame or
        hiding behind a nearer body scores the surviving fraction.  Both
        renders read only the model's own idealized params.

        Parameters:
            visible_vertex_count: Vertex count of the surviving limb
                polyline.

        Returns:
            The [0, 1] visible-arc fraction.  1.0 when the unclipped render
            has no boundary to compare against (a degenerate geometry).
        """
        p = self._sim_params
        max_axis = max(
            float(p.get('axis1', 0.0)),
            float(p.get('axis2', 0.0)),
            float(p.get('axis3', 0.0)),
        )
        # Canvas comfortably larger than the silhouette: half again the
        # longest axis (mesh relief can push past the ellipsoid bound)
        # plus a fixed margin.
        canvas = math.ceil(max_axis * 1.5) + 16
        full_img = self._render_body_image(
            float(np.radians(p.get('phase_angle', 0.0))),
            size=(canvas, canvas),
            center=(canvas / 2.0, canvas / 2.0),
        )
        full_mask = full_img > 0.0
        full_boundary = self._compute_limb_mask_from_body_mask(full_mask)
        total = int(np.count_nonzero(full_boundary))
        if total <= 0:
            return 1.0
        return min(1.0, float(visible_vertex_count) / float(total))

    def _build_limb_arc_feature(self) -> NavFeature | None:
        """Emit a LIMB_ARC from the rendered silhouette boundary, or ``None``.

        The predicted limb is the silhouette boundary of the rendered body; on a
        low-phase body it is essentially all lit, so BodyLimbNav's distance-
        transform fit aligns it to the image edge and recovers the offset.  The
        feature is emitted only when the boundary has enough vertices to
        constrain the fit (a tiny or barely-resolved body yields too short an
        arc).  The shadow-side vertices of a higher-phase body carry no sharp
        image edge and are down-weighted by the technique's robust fit rather
        than excluded here.

        With ``apply_limb_emission_gates`` off (the measurement path) the
        resolution, phase, and arc-length gates are skipped and the polyline
        is the lit geometric limb instead of the lit-region boundary; see
        :meth:`_lit_geometric_limb_polyline`.
        """
        if self._limb_mask is None or self._body_mask is None:
            return None
        if self.apply_limb_emission_gates:
            if self._predicted_diameter_px < _MIN_LIMB_DIAMETER_PX:
                return None
            if float(self._metadata.get('phase_angle_deg', 0.0)) > _LIMB_MAX_PHASE_DEG:
                return None
            vertices_vu, normals_vu = _limb_polyline_from_mask(self._limb_mask, self._body_mask)
            vertices_vu, normals_vu = self._drop_occluded_vertices(vertices_vu, normals_vu)
            if vertices_vu.shape[0] < _MIN_LIMB_ARC_VERTICES:
                return None
            # Honest reliability inputs, mirroring the SPICE-backed model:
            # the visible-arc fraction compares the surviving polyline (net
            # of frame clipping and sibling-body occlusion) against the
            # unclipped whole-body silhouette boundary, and the shared
            # reliability formula scores it.  These feed BodyLimbNav's
            # visible_limb_arc_fraction confidence term, so a clipped or
            # occluded sim limb scores like a real one instead of pinning
            # every input at 1.0.
            visible_arc_fraction = self._limb_visible_arc_fraction(vertices_vu.shape[0])
            reliability = limb_reliability(
                visible_arc_fraction=visible_arc_fraction,
                visible_arc_px=float(vertices_vu.shape[0]),
            )
        else:
            vertices_vu, normals_vu = self._lit_geometric_limb_polyline()
            if vertices_vu.shape[0] == 0:
                return None
            # Measurement path (realism match): the geometry is the product
            # and no technique consumes the score, so the reliability inputs
            # stay at their vacuous 1.0.
            visible_arc_fraction = 1.0
            reliability = 1.0
        n = vertices_vu.shape[0]
        sigma_normal = np.full(n, _LIMB_SIGMA_NORMAL_PX, dtype=np.float64)
        sigma_tangent = np.full(n, _LIMB_SIGMA_TANGENT_PX, dtype=np.float64)
        return NavFeature(
            feature_id=f'limb_arc:{self._body_name}',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model=self.name,
            geometry=LimbPolyline(
                vertices_vu=vertices_vu,
                normals_vu=normals_vu,
                sigma_normal_per_vertex_px=sigma_normal,
                sigma_tangent_per_vertex_px=sigma_tangent,
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=reliability,
            reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=visible_arc_fraction),
            usable_types=frozenset({NavFeatureType.LIMB_ARC}),
            flags=LimbArcFlags(
                body_name=self._body_name,
                visible_arc_fraction=visible_arc_fraction,
            ),
        )

    def _terminator_polyline(self) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Interior lit/unlit boundary polyline for the terminator arc.

        The terminator is the boundary *inside* the disc where the lit
        hemisphere meets the unlit one -- distinct from the geometric limb,
        which is the disc's silhouette edge against sky.  The shading floors the
        visible-but-unlit hemisphere at ``DARK_SIDE_ILLUM_STRENGTH``, so
        ``self._body_mask`` (brightness ``> 0``) is the whole visible disc while
        ``model_img > DARK_SIDE_ILLUM_STRENGTH`` is the lit region; the unlit
        disc is their difference.  A terminator vertex is a lit pixel with an
        unlit *interior* disc pixel as a 4-neighbor -- the interior restriction
        drops the anti-aliased limb ring (unlit only because its edge brightness
        has ramped below the floor), keeping the polyline interior to the disc
        everywhere except the cusp-adjacent vertices of a very thin crescent,
        where terminator and limb meet within a pixel and a few vertices land
        on the silhouette (at phase 150 roughly 9 of 155 vertices; the
        SPICE-backed model's sampler shares the behavior).  The outward
        normal is the gradient of the lit mask, so it points from the lit side
        toward the unlit side, the convention the SPICE-backed body model's
        terminator sampler uses.

        Returns:
            ``(vertices_vu, normals_vu)`` in extfov coordinates; empty arrays
            when the body renders no interior lit/unlit boundary (a fully lit
            or fully unlit visible disc).
        """
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        if self._body_mask is None or self._model_img is None:
            return empty, empty
        terminator_ridge, lit = _terminator_ridge_mask(
            np.asarray(self._model_img, dtype=np.float64)
        )
        return _limb_polyline_from_mask(terminator_ridge, lit)

    def _terminator_visible_arc_fraction(self, visible_vertex_count: int) -> float:
        """Fraction of the full predicted terminator ridge the frame sees.

        The sim analog of the SPICE-backed model's ``_visible_arc_fraction``
        (survivors over the predicted ridge): the denominator is the
        terminator ridge length of an *unclipped* render of the same body at
        the same phase and lighting -- a canvas sized to the whole silhouette,
        independent of the frame -- and the numerator is the ridge that
        survived frame clipping and sibling-body occlusion.  A fully framed,
        unoccluded body scores ~1.0; a body whose terminator runs off the
        frame edge or behind a nearer body scores the surviving fraction.
        Both renders read only the model's own idealized params.

        Parameters:
            visible_vertex_count: Vertex count of the surviving terminator
                polyline.

        Returns:
            The [0, 1] visible-arc fraction.  1.0 when the unclipped render
            has no ridge to compare against (a degenerate geometry).
        """
        p = self._sim_params
        max_axis = max(
            float(p.get('axis1', 0.0)),
            float(p.get('axis2', 0.0)),
            float(p.get('axis3', 0.0)),
        )
        # Canvas comfortably larger than the silhouette: half again the
        # longest axis (mesh relief can push past the ellipsoid bound)
        # plus a fixed margin.
        canvas = math.ceil(max_axis * 1.5) + 16
        full_img = self._render_body_image(
            float(np.radians(p.get('phase_angle', 0.0))),
            size=(canvas, canvas),
            center=(canvas / 2.0, canvas / 2.0),
        )
        full_ridge, _ = _terminator_ridge_mask(full_img)
        total = int(np.count_nonzero(full_ridge))
        if total <= 0:
            return 1.0
        return min(1.0, float(visible_vertex_count) / float(total))

    def _build_terminator_arc_feature(self) -> NavFeature | None:
        """Emit a TERMINATOR_ARC from the interior lit/unlit boundary, or ``None``.

        Mirrors the SPICE-backed body model's terminator gates: a resolved
        ``highly_irregular`` body suppresses the feature (the shared
        :func:`~spindoctor.nav_model.nav_model_body.shape_features_suppressed`
        policy -- a chaotic rotator's rendered terminator does not match the
        real body's), the phase must be far enough from zero for a terminator
        to separate from the limb (``sin(phase) >=
        TERMINATOR_MIN_PHASE_FACTOR``), and the boundary polyline must be long
        enough to constrain a fit (``>= TERMINATOR_MIN_VERTICES`` vertices).
        Emission is ungated by the limb-path resolution / phase policy gates --
        the terminator is exactly the high-phase feature those gates hand the
        body off to.  The recovered offset it produces stays
        ``confidence_provisional`` like every sim-anchored technique result,
        because ``BodyTerminatorNav`` is not yet calibrated against the
        simulated renderer.
        """
        if self._body_mask is None:
            return None
        shape = load_body_shape(self._body_name, config=self._config)
        if shape_features_suppressed(shape, self._predicted_diameter_px, config=self._config):
            return None
        phase_angle_deg = float(self._metadata.get('phase_angle_deg', 0.0))
        phase_angle_factor = abs(math.sin(math.radians(phase_angle_deg)))
        if phase_angle_factor < TERMINATOR_MIN_PHASE_FACTOR:
            return None
        vertices_vu, normals_vu = self._terminator_polyline()
        vertices_vu, normals_vu = self._drop_occluded_vertices(vertices_vu, normals_vu)
        if vertices_vu.shape[0] < TERMINATOR_MIN_VERTICES:
            return None
        n = vertices_vu.shape[0]
        sigma_normal = np.full(n, _TERMINATOR_SIGMA_NORMAL_PX, dtype=np.float64)
        sigma_tangent = np.full(n, _TERMINATOR_SIGMA_TANGENT_PX, dtype=np.float64)
        # Honest reliability inputs, mirroring the SPICE-backed model: the
        # visible-arc fraction compares the surviving ridge (net of frame
        # clipping and sibling-body occlusion) against the unclipped
        # whole-body ridge, and the shared reliability formula applies the
        # catalog albedo-variation and sin(phase) penalties.  These feed
        # BodyTerminatorNav's confidence terms, so a sim terminator scores
        # like a real one instead of pinning every input at 1.0.
        visible_arc_fraction = self._terminator_visible_arc_fraction(n)
        albedo_penalty = min(1.0, shape.albedo_variation)
        reliability = terminator_reliability(
            visible_arc_fraction=visible_arc_fraction,
            albedo_variation=shape.albedo_variation,
            phase_factor=phase_angle_factor,
        )
        return NavFeature(
            feature_id=f'terminator_arc:{self._body_name}',
            feature_type=NavFeatureType.TERMINATOR_ARC,
            source_model=self.name,
            geometry=TerminatorPolyline(
                vertices_vu=vertices_vu,
                normals_vu=normals_vu,
                sigma_normal_per_vertex_px=sigma_normal,
                sigma_tangent_per_vertex_px=sigma_tangent,
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=reliability,
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=visible_arc_fraction,
                albedo_penalty=albedo_penalty,
            ),
            usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
            flags=TerminatorArcFlags(
                body_name=self._body_name,
                visible_arc_fraction=visible_arc_fraction,
                phase_angle_factor=min(1.0, phase_angle_factor),
            ),
        )

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit body silhouette + label annotations for the summary PNG."""
        if self._model_img is None or self._body_mask is None or self._limb_mask is None:
            return Annotations()
        # The label anchor is a pixel of the nominal frame, so it comes off
        # the payload's extended-frame pixel-centric center rather than off
        # the scene's pixel-corner one.
        v_center, u_center = self._predicted_center_vu
        return self._create_annotations(
            containing_pixel(u_center - self.obs.extfov_margin_u),
            containing_pixel(v_center - self.obs.extfov_margin_v),
            self._model_img,
            self._limb_mask,
            self._body_mask,
        )
