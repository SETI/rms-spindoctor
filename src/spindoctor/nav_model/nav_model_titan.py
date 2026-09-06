"""Titan NavModel -- the haze envelope of a body whose atmosphere hides its surface.

Titan's thick haze hides the ground, so its visible edge is the haze top:
wavelength-dependent, hundreds of kilometers above the solid body, and not
even circular at high phase.  Ellipsoid limb / terminator / disc navigation
is therefore systematically wrong on Titan rather than merely noisy, and the
shape-based body model skips it.  This model renders what a haze navigator
needs instead: the geometric disc center, the image-plane direction toward
the sub-solar point (the axis the haze is mirror-symmetric about), the solid
and envelope radii, and a mask of the pixels the fit must ignore.

Whenever Titan is inside the extended field of view the model emits exactly
one ``TITAN_LIMB`` feature.  Frame quality lives in that feature's
reliability rather than in a decline: an envelope that cannot clear the
detector wherever the true pointing puts it, one too heavily occluded, or one
too small to measure scores exactly zero, and the standard per-type
reliability gate then removes it, so a marginal Titan resolves through the
same statuses as any other marginal scene.

The same geometry is drawn as the frame's overlay -- envelope circle,
symmetry axis, sunward arc sector, center cross -- so an operator reading
the summary PNG sees what the fit worked from and, because the PNG draws
annotations at the navigated offset, where the fit put the body.

Titan's atmosphere is unique among the currently navigated bodies
(transparent at some wavelengths), so this handling is a deliberate special
case and does not generalize to other thick-atmosphere bodies such as Venus.

The observation-side half -- every ``oops`` and star-catalog query -- lives
in :mod:`spindoctor.nav_model.titan_geometry`; everything here is a pure
function of the :class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs`
it returns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from oops import Observation

from spindoctor.annotation import (
    TEXTINFO_BOTTOM_ARROW,
    TEXTINFO_LEFT_ARROW,
    TEXTINFO_RIGHT_ARROW,
    TEXTINFO_TOP_ARROW,
    Annotation,
    Annotations,
    AnnotationTextInfo,
    TextLocInfo,
)
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import TitanHazeFlags
from spindoctor.feature.geometry import TitanHazeGeometry
from spindoctor.feature.reliability import FeatureReliabilityGate
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import TITAN_BODY_NAME, bodies_in_extfov
from spindoctor.nav_model.titan_geometry import TitanGeometryInputs, geometry_from_obs
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = [
    'NavModelTitan',
    'build_titan_feature',
    'haze_overlay',
    'titan_haze_reliability',
]


_CURVE_SAMPLE_STEP_PX: float = 0.5
"""Spacing of the samples every overlay curve is rasterized from.

Deliberately a constant rather than a configuration key: half a pixel is
the largest step that leaves no gaps in a solid curve at any orientation,
so this is the rasterizer's correctness invariant, not a preference.  A
larger configured value would silently break the solid-versus-dotted
distinction the ``titan.annotation`` keys rely on.
"""


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _envelope_fits_in_frame(geometry: TitanGeometryInputs) -> bool:
    """Whether the envelope disc dilated by the search window is fully framed.

    Full visibility is a property of Titan's TRUE position, which may sit
    anywhere inside the pointing search window, so the disc is dilated by
    that window before the containment test.  A frame that only looks
    fully-framed at the predicted pointing would let the fit sample sky.

    The dilation is per image axis, by that axis's own extfov margin, not by
    the scalar search half-window: the extended frame is the detector plus
    those two margins, so an axis-matched dilation makes this test say
    exactly "the envelope clears the detector", which is the physical
    statement intended.  Dilating both axes by the LARGER margin instead
    would shrink the admissible region on the tighter axis by the difference
    between them -- 90 px per side on a Cassini NAC, whose margins are 50
    rows against 140 columns -- for no physical reason.
    """
    v0, u0 = geometry.predicted_center_vu
    rows, cols = geometry.extfov_shape_vu
    margin_v, margin_u = geometry.extfov_margin_vu
    if rows <= 0 or cols <= 0:
        return False
    reach_v = geometry.r_env_px + margin_v
    reach_u = geometry.r_env_px + margin_u
    if v0 - reach_v < 0.0 or v0 + reach_v > rows - 1:
        return False
    return not (u0 - reach_u < 0.0 or u0 + reach_u > cols - 1)


def titan_haze_reliability(
    geometry: TitanGeometryInputs, *, config: Config
) -> tuple[float, NavReliabilityBreakdown]:
    """Score a haze feature's reliability and report the components.

    The score is ``sigmoid((D - midpoint) / scale) * (1 - occluded)`` with
    ``D`` the apparent envelope diameter in pixels, so it rises with
    resolution and falls with occlusion.  It is forced to exactly ``0.0``
    under three hard conditions, each of which makes the fit unusable rather
    than merely imprecise:

    - the envelope disc, dilated per image axis by that axis's extfov
      margin, does not fit inside the extended frame;
    - the occluded fraction exceeds ``max_occluded_fraction``;
    - the envelope diameter is below ``min_envelope_diameter_px``.

    A zero can never clear the per-type reliability gate, so a hard
    condition is exactly as strong as a refusal to emit -- but it travels
    through the standard gate machinery, leaving an attributable record
    instead of a silent absence.

    Parameters:
        geometry: The observation-derived haze geometry.
        config: Configuration supplying the ``titan.navigation`` thresholds
            and the reliability sigmoid's midpoint and scale.

    Returns:
        ``(reliability, breakdown)`` where ``reliability`` lies in
        ``[0, 1]`` and ``breakdown`` carries the envelope diameter and the
        occluded fraction that produced it.
    """
    nav_config = config.titan['navigation']
    diameter_px = 2.0 * geometry.r_env_px
    occluded = geometry.occluded_fraction
    breakdown = NavReliabilityBreakdown(
        titan_envelope_diameter_px=diameter_px,
        titan_occluded_fraction=occluded,
    )
    if not _envelope_fits_in_frame(geometry):
        return 0.0, breakdown
    if occluded > float(nav_config['max_occluded_fraction']):
        return 0.0, breakdown
    if diameter_px < float(nav_config['min_envelope_diameter_px']):
        return 0.0, breakdown
    midpoint = float(nav_config['reliability_diameter_midpoint_px'])
    scale = float(nav_config['reliability_diameter_scale_px'])
    size_term = _sigmoid((diameter_px - midpoint) / scale)
    return float(min(max(size_term * (1.0 - occluded), 0.0), 1.0)), breakdown


def build_titan_feature(
    geometry: TitanGeometryInputs, *, source_model: str, config: Config
) -> NavFeature:
    """Build the single ``TITAN_LIMB`` feature from an evaluated geometry.

    Pure: every observation-dependent quantity already lives on
    ``geometry``, so the emission rules and the reliability formula are
    exercisable without an observation.

    Parameters:
        geometry: The observation-derived haze geometry.
        source_model: Name of the emitting NavModel.
        config: Configuration supplying the ``titan.navigation`` thresholds
            and the surface-window filter list.

    Returns:
        One ``TITAN_LIMB`` :class:`~spindoctor.feature.feature.NavFeature`
        carrying the haze geometry, its flags, and its reliability
        breakdown.
    """
    reliability, breakdown = titan_haze_reliability(geometry, config=config)
    nav_config = config.titan['navigation']
    surface_window_filters = {str(name).upper() for name in nav_config['surface_window_filters']}
    return NavFeature(
        feature_id=f'titan_limb:{TITAN_BODY_NAME}',
        feature_type=NavFeatureType.TITAN_LIMB,
        source_model=source_model,
        geometry=TitanHazeGeometry(
            predicted_center_vu=geometry.predicted_center_vu,
            sun_angle_rad=geometry.theta_rad,
            axis_degenerate=geometry.axis_degenerate,
            phase_deg=geometry.phase_deg,
            r_solid_px=geometry.r_solid_px,
            r_env_px=geometry.r_env_px,
            km_per_px=geometry.km_per_px,
            contaminant_mask=geometry.contaminant_mask,
            filters=geometry.filters,
            bbox_extfov_vu=geometry.bbox_extfov_vu,
        ),
        subject_range_km=geometry.subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=breakdown,
        usable_types=frozenset({NavFeatureType.TITAN_LIMB}),
        flags=TitanHazeFlags(
            body_name=TITAN_BODY_NAME,
            surface_window_filter=any(
                f.upper() in surface_window_filters for f in geometry.filters
            ),
            high_phase=geometry.phase_deg >= float(nav_config['high_phase_deg']),
        ),
    )


def _paint_samples(
    overlay: NDArrayBoolType,
    v_samples: NDArrayFloatType,
    u_samples: NDArrayFloatType,
    *,
    dot_spacing: int,
) -> None:
    """Mark the in-bounds pixels a sampled curve passes through.

    Parameters:
        overlay: Extended-frame boolean overlay, modified in place.
        v_samples: Row coordinates of the curve samples.
        u_samples: Column coordinates of the curve samples.
        dot_spacing: Paint every ``dot_spacing``-th sample, so ``1`` draws a
            solid curve and larger values a dotted one.
    """
    step = max(1, dot_spacing)
    v_idx = np.rint(v_samples[::step]).astype(np.int64)
    u_idx = np.rint(u_samples[::step]).astype(np.int64)
    rows, cols = overlay.shape
    inside = (v_idx >= 0) & (v_idx < rows) & (u_idx >= 0) & (u_idx < cols)
    overlay[v_idx[inside], u_idx[inside]] = True


def _paint_arc(
    overlay: NDArrayBoolType,
    center_vu: tuple[float, float],
    radius_px: float,
    *,
    phi_start_rad: float,
    phi_end_rad: float,
    dot_spacing: int,
) -> None:
    """Draw a circular arc, or a full circle when the span is a full turn.

    Angles follow the fitting library's convention: the point at angle
    ``phi`` and radius ``rho`` sits at ``(v + rho sin phi, u + rho cos phi)``.

    Parameters:
        overlay: Extended-frame boolean overlay, modified in place.
        center_vu: ``(v, u)`` center the arc is drawn about.
        radius_px: Arc radius in pixels; a radius below one pixel, or a
            non-finite one, draws nothing.
        phi_start_rad: First angle of the arc.
        phi_end_rad: Last angle of the arc.
        dot_spacing: Solid (``1``) or dotted (larger) rasterization.
    """
    if not radius_px >= 1.0:
        return
    span = abs(phi_end_rad - phi_start_rad)
    n_samples = max(2, math.ceil(span * radius_px / _CURVE_SAMPLE_STEP_PX) + 1)
    phis = np.linspace(phi_start_rad, phi_end_rad, n_samples)
    _paint_samples(
        overlay,
        center_vu[0] + radius_px * np.sin(phis),
        center_vu[1] + radius_px * np.cos(phis),
        dot_spacing=dot_spacing,
    )


def _paint_segment(
    overlay: NDArrayBoolType,
    start_vu: tuple[float, float],
    end_vu: tuple[float, float],
    *,
    dot_spacing: int,
) -> None:
    """Draw a straight segment between two ``(v, u)`` points.

    Parameters:
        overlay: Extended-frame boolean overlay, modified in place.
        start_vu: ``(v, u)`` start point.
        end_vu: ``(v, u)`` end point.
        dot_spacing: Solid (``1``) or dotted (larger) rasterization.
    """
    length = math.hypot(end_vu[0] - start_vu[0], end_vu[1] - start_vu[1])
    n_samples = max(2, math.ceil(length / _CURVE_SAMPLE_STEP_PX) + 1)
    fractions = np.linspace(0.0, 1.0, n_samples)
    _paint_samples(
        overlay,
        start_vu[0] + fractions * (end_vu[0] - start_vu[0]),
        start_vu[1] + fractions * (end_vu[1] - start_vu[1]),
        dot_spacing=dot_spacing,
    )


def haze_overlay(
    geometry: TitanGeometryInputs,
    *,
    sector_half_angle_deg: float,
    dot_spacing: int = 1,
    center_marker_half_px: float = 4.0,
) -> NDArrayBoolType:
    """Rasterize the haze fit's geometry into an extended-frame overlay.

    Four elements, each drawn from predicted geometry alone:

    - the haze envelope circle, the outer bound of everything the fit
      samples;
    - the symmetry axis, the line the cross-track scan mirrors about, drawn
      as a full chord so the operator can see the mirror plane;
    - the sunward sector the limb-arc fit rays sweep, outlined by an arc at
      the solid radius and the two radial edges joining it to the envelope;
    - a cross at the disc center, which the summary PNG's offset shift moves
      onto the fitted center.

    Parameters:
        geometry: The observation-derived haze geometry.
        sector_half_angle_deg: Half-width of the arc-fit sector, in degrees.
        dot_spacing: ``1`` draws solid curves; larger values dot them.
        center_marker_half_px: Half-length of the center cross, in pixels.

    Returns:
        A boolean array of the extended-frame shape, True on every painted
        pixel.  Nothing is painted for a geometry that did not evaluate --
        a zero-radius envelope, a non-finite center or axis -- because its
        defaults put a zero-radius envelope at the frame origin, and drawing
        a center mark there would place a mark the frame gives no evidence
        for.
    """
    overlay: NDArrayBoolType = np.zeros(geometry.extfov_shape_vu, dtype=bool)
    center = geometry.predicted_center_vu
    theta = geometry.theta_rad
    if not geometry.r_env_px >= 1.0:
        return overlay
    if not all(math.isfinite(x) for x in (center[0], center[1], theta, geometry.r_solid_px)):
        return overlay
    _paint_arc(
        overlay,
        center,
        geometry.r_env_px,
        phi_start_rad=0.0,
        phi_end_rad=2.0 * math.pi,
        dot_spacing=dot_spacing,
    )
    axis_vu = (math.sin(theta), math.cos(theta))
    reach = geometry.r_env_px
    _paint_segment(
        overlay,
        (center[0] - reach * axis_vu[0], center[1] - reach * axis_vu[1]),
        (center[0] + reach * axis_vu[0], center[1] + reach * axis_vu[1]),
        dot_spacing=dot_spacing,
    )
    half_angle = math.radians(sector_half_angle_deg)
    _paint_arc(
        overlay,
        center,
        geometry.r_solid_px,
        phi_start_rad=theta - half_angle,
        phi_end_rad=theta + half_angle,
        dot_spacing=dot_spacing,
    )
    for edge_phi in (theta - half_angle, theta + half_angle):
        edge_vu = (math.sin(edge_phi), math.cos(edge_phi))
        _paint_segment(
            overlay,
            (
                center[0] + geometry.r_solid_px * edge_vu[0],
                center[1] + geometry.r_solid_px * edge_vu[1],
            ),
            (
                center[0] + geometry.r_env_px * edge_vu[0],
                center[1] + geometry.r_env_px * edge_vu[1],
            ),
            dot_spacing=dot_spacing,
        )
    marker = float(center_marker_half_px)
    _paint_segment(
        overlay,
        (center[0] - marker, center[1]),
        (center[0] + marker, center[1]),
        dot_spacing=dot_spacing,
    )
    _paint_segment(
        overlay,
        (center[0], center[1] - marker),
        (center[0], center[1] + marker),
        dot_spacing=dot_spacing,
    )
    return overlay


def _disc_mask(
    shape_vu: tuple[int, int], center_vu: tuple[float, float], radius_px: float
) -> NDArrayBoolType:
    """Return a filled disc of the given radius as a boolean array.

    Parameters:
        shape_vu: ``(rows, cols)`` shape of the array to build.
        center_vu: ``(v, u)`` disc center.
        radius_px: Disc radius in pixels.

    Returns:
        A boolean array True inside the disc.  Built over the disc's
        bounding box alone, so a small body in a large frame costs no
        full-frame arithmetic.
    """
    mask: NDArrayBoolType = np.zeros(shape_vu, dtype=bool)
    rows, cols = shape_vu
    reach = max(radius_px, 0.0)
    v_lo = max(0, math.floor(center_vu[0] - reach))
    v_hi = min(rows, math.ceil(center_vu[0] + reach) + 1)
    u_lo = max(0, math.floor(center_vu[1] - reach))
    u_hi = min(cols, math.ceil(center_vu[1] + reach) + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return mask
    vs = np.arange(v_lo, v_hi, dtype=np.float64)[:, np.newaxis] - center_vu[0]
    us = np.arange(u_lo, u_hi, dtype=np.float64)[np.newaxis, :] - center_vu[1]
    mask[v_lo:v_hi, u_lo:u_hi] = (vs * vs + us * us) <= reach * reach
    return mask


class NavModelTitan(NavModel):
    """Haze-envelope NavModel for Titan.

    Renders the geometry a haze navigator needs -- disc center, sub-solar
    symmetry axis, solid and envelope radii, contaminant mask -- and emits
    exactly one ``TITAN_LIMB`` feature whenever Titan is inside the extended
    field of view.  Marginal frames are expressed as low or zero
    reliability, not as a refusal to emit, so every outcome carries an
    attributable record.

    Marginal is not the same as unevaluable.  The always-emit invariant covers
    the frames this model can read and finds wanting -- a haze too small to
    fit, a disc mostly occluded, an inventory box that is not finite.  A frame
    whose geometry raises is not one of them: nothing is absorbed, the
    exception reaches the orchestrator, and the image fails with
    ``status_reason=internal_error`` naming what raised.  See "Nothing here
    absorbs an exception" in :mod:`spindoctor.nav_model.titan_geometry` for
    why a zero-reliability feature is the worse record of the two.

    Parameters:
        name: Model instance name (``'titan:TITAN'``).
        obs: Observation snapshot.
        inventory: Optional pre-computed Titan inventory entry; pulled from
            ``obs.inventory`` on demand otherwise.
        siblings: ``(body_name, inventory_entry)`` for the other bodies in
            the FOV, used for occlusion and for the contaminant mask;
            enumerated from ``obs`` on demand otherwise.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        *,
        inventory: dict[str, Any] | None = None,
        siblings: list[tuple[str, dict[str, Any]]] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._inventory = inventory
        self._siblings = siblings
        self._geometry: TitanGeometryInputs | None = None

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one instance when Titan is inside the extfov, else none.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose satellite catalog decides whether
                Titan is in the mission set.  ``None`` uses ``DEFAULT_CONFIG``.

        Returns:
            A single ``NavModelTitan`` when Titan is present in the extfov,
            otherwise an empty list.
        """
        # Simulated obs drive model selection from operator parameters, not the
        # SPICE inventory; mirror NavModelBody and build nothing here.
        if getattr(obs, 'is_simulated', False):
            return []
        if config is None:
            config = DEFAULT_CONFIG
        in_extfov = bodies_in_extfov(obs, config=config)
        out: list[NavModel] = []
        for body_name, entry in in_extfov:
            if body_name.upper() != TITAN_BODY_NAME:
                continue
            siblings = [
                (other, other_entry)
                for other, other_entry in in_extfov
                if other.upper() != TITAN_BODY_NAME
            ]
            out.append(cls('titan:TITAN', obs, inventory=entry, siblings=siblings, config=config))
        return out

    @property
    def geometry_inputs(self) -> TitanGeometryInputs:
        """The evaluated haze geometry, computed on first access."""
        if self._geometry is None:
            self._geometry = geometry_from_obs(
                self.obs, self._config, inventory=self._inventory, siblings=self._siblings
            )
        return self._geometry

    def create_model(self) -> None:
        """Evaluate the haze geometry and record it in ``metadata``."""
        self._metadata.clear()
        self._metadata['body'] = TITAN_BODY_NAME
        with self.log_section('TITAN MODEL'):
            geometry = self.geometry_inputs
            self._metadata['predicted_center_vu'] = list(geometry.predicted_center_vu)
            self._metadata['km_per_pixel'] = geometry.km_per_px
            self._metadata['envelope_diameter_px'] = 2.0 * geometry.r_env_px
            self._metadata['solid_diameter_px'] = 2.0 * geometry.r_solid_px
            self._metadata['phase_angle_deg'] = geometry.phase_deg
            self._metadata['sun_angle_deg'] = math.degrees(geometry.theta_rad)
            self._metadata['axis_degenerate'] = geometry.axis_degenerate
            self._metadata['occluded_fraction'] = geometry.occluded_fraction
            self._metadata['filters'] = list(geometry.filters)
            self._logger.info(
                'Predicted center (v, u) = (%.2f, %.2f); envelope diameter = %.2f px; '
                'km/px = %.4f; phase = %.2f deg',
                geometry.predicted_center_vu[0],
                geometry.predicted_center_vu[1],
                2.0 * geometry.r_env_px,
                geometry.km_per_px,
                geometry.phase_deg,
            )
            self._logger.info(
                'Symmetry axis = %.2f deg (degenerate = %s); occluded fraction = %.3f',
                math.degrees(geometry.theta_rad),
                geometry.axis_degenerate,
                geometry.occluded_fraction,
            )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the single ``TITAN_LIMB`` feature for this frame.

        Parameters:
            context: Per-image navigation context; unused because every
                quantity the feature carries is predicted geometry, not
                measured from pixels.

        Returns:
            A one-element list, always.  Frame quality is carried by the
            feature's reliability -- exactly zero when the envelope cannot
            be fully framed, is too heavily occluded, or is too small --
            rather than by an empty list.
        """
        del context
        feature = build_titan_feature(
            self.geometry_inputs, source_model=self.name, config=self._config
        )
        self._logger.info('Emitted TITAN_LIMB feature with reliability %.3f', feature.reliability)
        return [feature]

    def to_annotations(self, context: NavContext) -> Annotations:
        """Draw the haze geometry the fit works from over the frame.

        The overlay carries the envelope circle, the symmetry axis, the
        sunward arc sector, and a center cross (see :func:`haze_overlay`).
        The summary PNG draws every annotation shifted by the navigated
        offset, so the cross and the circle land on the fitted center when
        an offset was committed and stay at the prediction when none was.

        Curves are solid when the emitted feature's reliability clears the
        per-type gate threshold and dotted, with the label saying so, when
        it does not.  The technique's own accept-or-spurious verdict cannot
        be shown here: the orchestrator merges model annotations before any
        technique runs, so the latest state available at this point is the
        feature's own reliability, which is what decides whether the fit is
        attempted at all in an autonomous run.  Manual navigation
        deliberately bypasses the gate, which is why the dotted style
        reports low reliability rather than a gate verdict.

        Parameters:
            context: Per-image navigation context; unused because every
                drawn quantity is predicted geometry.

        Returns:
            An ``Annotations`` collection holding one annotation, or an
            empty collection when no part of the overlay lands inside the
            extended frame.
        """
        del context
        geometry = self.geometry_inputs
        feature = build_titan_feature(geometry, source_model=self.name, config=self._config)
        gate = FeatureReliabilityGate.from_mapping(
            self._config.orchestrator.get('reliability_gate')
        )
        kept, _gated = gate.apply([feature])
        usable = len(kept) > 0
        arc_config = self._config.titan['navigation']['arc']
        annotation_config = self._config.titan['annotation']
        overlay = haze_overlay(
            geometry,
            sector_half_angle_deg=float(arc_config['sector_half_angle_deg']),
            dot_spacing=1 if usable else int(annotation_config['gated_dot_spacing']),
            center_marker_half_px=float(annotation_config['center_marker_half_px']),
        )
        if not overlay.any():
            self._logger.info('Haze overlay skipped: nothing to draw inside the extended frame')
            return Annotations()
        self._logger.info(
            'Drew %s haze overlay (feature reliability %.3f)',
            'solid' if usable else 'dotted',
            feature.reliability,
        )
        annotations = Annotations(config=self._config)
        annotations.add_annotations(
            Annotation(
                self.obs,
                overlay,
                self._config.bodies.label_limb_color,
                thicken_overlay=self._config.bodies.outline_thicken,
                avoid_mask=_disc_mask(
                    geometry.extfov_shape_vu,
                    geometry.predicted_center_vu,
                    geometry.r_env_px + float(self._config.bodies.label_mask_enlarge),
                ),
                text_info=self._label(geometry, usable=usable),
                config=self._config,
            )
        )
        return annotations

    def _label(self, geometry: TitanGeometryInputs, *, usable: bool) -> AnnotationTextInfo:
        """Build the body label, offering one position per side of the disc.

        Parameters:
            geometry: The observation-derived haze geometry.
            usable: Whether the emitted feature's reliability clears the
                per-type threshold; a feature below it says so in the
                label, because the dotted overlay alone does not explain
                why the frame is unlikely to navigate.

        Returns:
            The label and its candidate placements, in extfov coordinates.
        """
        body_config = self._config.bodies
        v_center, u_center = geometry.predicted_center_vu
        gap = geometry.r_env_px
        text_loc = [
            TextLocInfo(TEXTINFO_TOP_ARROW, round(v_center - gap), round(u_center)),
            TextLocInfo(TEXTINFO_BOTTOM_ARROW, round(v_center + gap), round(u_center)),
            TextLocInfo(TEXTINFO_LEFT_ARROW, round(v_center), round(u_center - gap)),
            TextLocInfo(TEXTINFO_RIGHT_ARROW, round(v_center), round(u_center + gap)),
        ]
        return AnnotationTextInfo(
            TITAN_BODY_NAME if usable else f'{TITAN_BODY_NAME} (low reliability)',
            text_loc=text_loc,
            ref_vu=(round(v_center), round(u_center)),
            font=body_config.label_font,
            font_size=body_config.label_font_size,
            color=body_config.label_font_color,
        )
