"""Shared base class for body navigation models.

``NavModelBodyBase`` carries the limb-mask helper and the body-label
annotation pipeline shared between every concrete body NavModel.  It is
abstract — registered subclasses (``NavModelBodySimulated`` today, plus
the real-scene body model when it lands) inherit the helpers and supply
the per-image rendering.

Anti-aliasing math lives separately in
``spindoctor.nav_model.rings.ring_math.compute_antialiasing``; helpers here are
strictly observation-aware (image shape, font config, label-placement
heuristics).
"""

import math
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage as ndimage

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
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyBlobFlags
from spindoctor.feature.geometry import BodyBlobGeometry
from spindoctor.nav_model.body_shape import BodyShape
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.image import shift_array
from spindoctor.support.noise_estimate import NOISE_CLIP_SIGMA, estimate_background_and_sky_sigma
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model import NavModel

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['BODY_BLOB_MIN_DIAMETER_PX', 'NavModelBodyBase']


BODY_BLOB_MIN_DIAMETER_PX: float = 5.0
"""Minimum predicted disc diameter (px) at which BODY_BLOB is emitted.

Below this the silhouette covers so few pixels (< ~20) that the
brightness-weighted centroid is dominated by the PSF and per-pixel
noise rather than the body's position.  The floor admits the
operator-curated ``below_resolution_body`` exemplars (a 6 px Enceladus
whose sidecar requires ``BodyBlobNav`` to run); on the sim calibration
campaign the 5-8 px band recovers planted offsets to under 0.15 px
with honest 2-sigma coverage, so precision above the floor is the
covariance's and the confidence formula's job, not the emission
gate's.  The per-body shape table can override this floor upward for
known difficult bodies (gas giants use 20).
"""


_SUB_SOLAR_MIN_OFFSET_PX: float = 0.5
"""Minimum lit-centroid offset (px) for a meaningful sub-solar direction.

Below this the body is near full phase, the lit and geometric centroids
coincide to within pixel noise, and the direction toward the bright limb
is undefined; the BODY_BLOB feature then carries ``(0.0, 0.0)`` and
BodyBlobNav falls back to the filled-disc coarse template.
"""


_BLOB_DETECTION_SNR_MIDPOINT: float = 3.0
"""Detection SNR at which the blob reliability's SNR term crosses 0.5.

Matches ``BodyBlobNav``'s 3-sigma lit-pixel threshold: pixels below
``background + 3 * sky_sigma`` never contribute to the centroid moment,
so a body whose brightest expected pixels sit at that level is at the
technique's own usability boundary.  A detection SNR of 0 (nothing above
the noise order-statistics in the search window) lands the term at
``sigmoid(-3) ~ 0.05``, decisively below the 0.20 reliability gate for
any body size.
"""


_BLOB_EXTENT_MIDPOINT_PX: float = 4.0
"""Predicted diameter at which the blob reliability's extent term crosses 0.5.

Below the :data:`BODY_BLOB_MIN_DIAMETER_PX` emission floor.  The floor
already guarantees no undersized BODY_BLOB exists, so the extent term
must not re-gate at-floor bodies (with the midpoint at the floor itself
an at-floor body could never clear the 0.20 reliability gate: ``0.4 *
0.5 * snr_term < 0.2`` for every finite SNR).  Sitting the midpoint
just under the floor makes the term a mild near-floor discount that
asks a smaller body for a stronger detection (SNR ~4.4 at 5 px, ~3.3
at 8 px, ~3.1 at 20 px) rather than acting as a second size gate.
"""


_BLOB_EXTENT_SCALE_PX: float = 2.0
"""Sigmoid scale (px) of the blob reliability's extent term."""


class NavModelBodyBase(NavModel):
    """Base class for body navigation models.

    Provides shared helpers to compute a limb mask, build the BODY_BLOB
    feature, and create annotations consistent across every concrete body
    model.

    The BODY_BLOB construction (``_build_blob_feature`` and its
    ``_phase_irregularity_factor`` / ``_lit_weighted_centroid_vu``
    helpers) lives here so the SPICE-backed ``NavModelBody`` and the
    simulated ``NavModelBodySimulated`` share one implementation rather
    than two copies of the same phase-and-irregularity calibration math.
    Subclasses populate the attributes the helpers read: ``_model_img``,
    ``_body_mask``, ``_predicted_center_vu``, ``_predicted_diameter_px``,
    ``_km_per_pixel_at_limb``, ``_subject_range_km``, ``_bbox_extfov_vu``,
    and ``_metadata['phase_angle_deg']``.
    """

    _body_name: str
    """Upper-case name of the body this model is for.

    Declared here and set by every subclass. Reaching it through a getattr
    default would give a subclass that forgot a feature id reading
    ``body_blob:BODY``, which is a label no consumer can trace back to a body
    and nothing would report as wrong.
    """

    _abstract = True

    _model_img: NDArrayFloatType | None
    _body_mask: NDArrayBoolType | None
    _predicted_center_vu: tuple[float, float]
    _predicted_diameter_px: float
    _km_per_pixel_at_limb: float
    _subject_range_km: float
    _bbox_extfov_vu: tuple[int, int, int, int]

    def _compute_limb_mask_from_body_mask(self, body_mask: NDArrayBoolType) -> NDArrayBoolType:
        """Compute limb mask as body pixels adjacent to at least one non-body neighbor."""
        neighbor = (
            shift_array(~body_mask, (-1, 0))
            | shift_array(~body_mask, (1, 0))
            | shift_array(~body_mask, (0, -1))
            | shift_array(~body_mask, (0, 1))
        )
        return body_mask & neighbor

    def _blob_detection_snr(self, context: 'NavContext') -> float:
        """Measured per-pixel detection SNR for the body in its search window.

        Answers the question the reliability gate needs answered before
        any technique runs: *does the image contain body-scale signal
        above the noise floor anywhere the body could be?*  The body's
        true position differs from the prediction by the (unknown)
        pointing error, which is bounded by the extfov margins, so the
        search window is the predicted bbox expanded by those margins --
        the same capture range ``BodyBlobNav``'s coarse acquisition
        scans.

        The estimate is order-statistics based so it needs no photometric
        prediction (no albedo or flux calibration is available for
        bodies):

        1. Compute the model's *effective* lit pixel count ``N`` -- the
           Kish effective sample size ``(sum w)^2 / sum(w^2)`` of the
           rendered lit-pixel brightnesses -- how many image pixels
           carry the body's predicted flux.  For a uniform silhouette
           this equals the plain lit count; for a high-phase Lambert
           crescent it concentrates on the bright core, so the
           near-terminator tail of barely-lit model pixels (whose real
           counterparts are routinely darker than the render predicts)
           does not drag the estimate into the sky population.
        2. Take the median of the ``N`` brightest valid pixels in the
           search window, as sigmas above the frame background (the
           shared iterative sky estimate,
           :func:`~spindoctor.support.noise_estimate.estimate_background_and_sky_sigma`).
        3. Subtract the level pure noise would produce -- the expected
           median of the top ``N`` of ``M`` Gaussian draws, i.e. the
           normal quantile at ``1 - N / (2 M)`` -- so an empty window
           scores ~0 instead of inheriting the selection bias of taking
           a maximum over many pixels.

        If the body is present and detectable its lit pixels dominate the
        top-``N`` population and the excess is the body's per-pixel SNR;
        if nothing is there the top-``N`` median sits at the noise
        quantile and the excess is ~0.  Other bright content in the
        window (a ring, another body) can only raise the score -- a
        false *admit* is cheap because the technique's own spurious
        checks run downstream, while the false *veto* of a small bright
        body is exactly the failure this estimator replaces.  A body
        large relative to the frame pushes the background estimate up
        toward its own brightness and scores low, consistent with the
        technique's centroid, whose lit-pixel threshold fails the same
        way; frame-filling bodies are limb / disc territory.

        Parameters:
            context: Per-image ``NavContext`` carrying the extended-FOV
                image, validity masks, and the global noise sigma.

        Returns:
            Detection SNR in sigmas, >= 0.  0.0 when the model has no
            lit pixels or the search window has no valid pixels.
        """
        assert self._model_img is not None
        assert self._body_mask is not None
        image_ext = np.asarray(context.image_ext, np.float64)
        valid_mask = (
            np.asarray(context.sensor_mask_ext, bool)
            & ~np.asarray(context.saturation_mask_ext, bool)
            & ~np.asarray(context.cosmic_ray_mask_ext, bool)
        )
        # Count only *observable* lit pixels: a mostly-offscreen body's
        # silhouette extends past the sensor, and predicted flux that can
        # never land on a valid pixel must not inflate N (it would push
        # the top-N median into the sky population and falsely veto the
        # visible part of the body).
        lit_mask = self._body_mask & (self._model_img > 0.0) & valid_mask
        lit_weights = np.asarray(self._model_img, np.float64)[lit_mask]
        weight_sq_sum = float((lit_weights * lit_weights).sum())
        if weight_sq_sum <= 0.0:
            return 0.0
        n_lit = max(1, int(float(lit_weights.sum()) ** 2 / weight_sq_sum))
        background, sky_sigma = estimate_background_and_sky_sigma(
            image_ext,
            valid_mask,
            noise_sigma=float(context.image_noise_sigma),
            clip_sigma=NOISE_CLIP_SIGMA,
        )
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        margin_v = int(self._obs.extfov_margin_v)
        margin_u = int(self._obs.extfov_margin_u)
        v_lo = max(0, v_min - margin_v)
        u_lo = max(0, u_min - margin_u)
        v_hi = min(image_ext.shape[0], v_max + margin_v)
        u_hi = min(image_ext.shape[1], u_max + margin_u)
        if v_hi <= v_lo or u_hi <= u_lo:
            return 0.0
        window = image_ext[v_lo:v_hi, u_lo:u_hi]
        window_vals = window[valid_mask[v_lo:v_hi, u_lo:u_hi]]
        n_window = int(window_vals.size)
        if n_window == 0:
            return 0.0
        n_top = min(n_lit, n_window)
        top = np.partition(window_vals, n_window - n_top)[n_window - n_top :]
        observed_sigma = (float(np.median(top)) - background) / sky_sigma
        # Median of the top n_top of n_window standard-normal draws is
        # (approximately) the quantile at 1 - n_top / (2 * n_window);
        # n_top == n_window gives quantile 0.5, i.e. no correction.
        null_sigma = NormalDist().inv_cdf(1.0 - n_top / (2.0 * n_window))
        return max(0.0, observed_sigma - null_sigma)

    def _build_blob_feature(self, shape: BodyShape, *, context: 'NavContext') -> NavFeature:
        """Construct the BODY_BLOB feature.

        The reliability and centroid covariance are driven by the
        *measured* detection SNR (:meth:`_blob_detection_snr`), so a
        small body that is genuinely bright against the image noise is
        admitted by the orchestrator's reliability gate while a
        predicted body with no image signal anywhere in its search
        window is culled.

        Three phase-aware decisions live here:

        * **Lit-weighted predicted centroid.** At high phase the
          measured brightness centroid sits at the centroid of the lit
          hemisphere, not at the geometric body center.  Predicting the
          lit-weighted centroid up front means the navigation offset
          the technique recovers is just the spacecraft pointing error,
          not pointing error plus the systematic phase bias.  The
          weighted centroid is computed over ``_model_img *
          _body_mask`` (the rendered model already encodes the local
          incidence-driven brightness) and collapses to the geometric
          center at phase 0 where the model is uniform.
        * **Inflated centroid covariance.** The lit-weighted centroid
          assumes a body of *known* rotational orientation; the same
          scene on an irregular body whose pose we do not know carries a
          residual centroid bias the ellipsoidal model cannot remove.
          The bias scales with the shape's RMS departure from an
          ellipsoid (in km) and with how much of the body the lit
          hemisphere fails to sample (``1 + 2 * sin^2(phase / 2)`` runs
          from 1 at full-phase to 3 at full crescent).  This
          irregularity sigma is added to the photon-noise-limited
          centroid sigma in quadrature so the joint-fit covariance
          reflects both terms.
        * **``phase_irregularity_factor`` on the flags.** Surfaces the
          dimensionless ``sin(phase/2) * residual / radius`` so the
          BLOB confidence formula can down-weight irregular high-phase
          blobs without the technique having to recompute it.
        """
        assert self._model_img is not None
        assert self._body_mask is not None
        snr = self._blob_detection_snr(context)
        sigma_centroid = self._predicted_diameter_px / (
            2.0 * math.sqrt(max(int(self._body_mask.sum()), 1)) * max(snr, 1e-6)
        )
        phase_angle_deg = float(self._metadata.get('phase_angle_deg', 0.0))
        if not math.isfinite(phase_angle_deg):
            phase_angle_deg = 0.0
        # Clamp to the BodyBlobFlags valid range; phase outside [0, 180]
        # is a corner-case artefact, never a physical value.
        phase_angle_deg = max(0.0, min(180.0, phase_angle_deg))
        phase_irregularity_factor = self._phase_irregularity_factor(shape, phase_angle_deg)
        sigma_irregular_px = phase_irregularity_factor * (self._predicted_diameter_px / 2.0)
        sigma_total_px = math.sqrt(sigma_centroid * sigma_centroid + sigma_irregular_px**2)
        cov = (sigma_total_px * sigma_total_px) * np.eye(2, dtype=np.float64)
        predicted_center_vu = self._lit_weighted_centroid_vu()
        sub_solar_dir_vu = self._sub_solar_dir_vu(predicted_center_vu)
        body_name = self._body_name
        return NavFeature(
            feature_id=f'body_blob:{body_name}',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model=self.name,
            geometry=BodyBlobGeometry(
                predicted_center_vu=predicted_center_vu,
                bbox_extfov_vu=self._bbox_extfov_vu,
                predicted_diameter_px=self._predicted_diameter_px,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=cov,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=_blob_reliability(snr=snr, diameter_px=self._predicted_diameter_px),
            reliability_reasons=NavReliabilityBreakdown(
                blob_snr=min(1.0, snr / 10.0),
                blob_extent_px=min(1.0, self._predicted_diameter_px / 30.0),
            ),
            usable_types=frozenset({NavFeatureType.BODY_BLOB}),
            flags=BodyBlobFlags(
                body_name=body_name,
                predicted_diameter_px=self._predicted_diameter_px,
                phase_angle_deg=phase_angle_deg,
                phase_irregularity_factor=phase_irregularity_factor,
                sub_solar_dir_vu=sub_solar_dir_vu,
            ),
        )

    def _phase_irregularity_factor(self, shape: BodyShape, phase_angle_deg: float) -> float:
        """Return the dimensionless phase-and-irregularity coupling.

        Computed as ``(ellipsoid_rms_residual_km / body_radius_km) *
        (1 + 2 * sin^2(phase / 2))``.  Two physical effects compose:

        * The ``residual / radius`` ratio is the *fractional* shape
          uncertainty of the body relative to its ellipsoidal model.
          For a regular Mimas (residual ~ 1 km, radius ~ 200 km)
          this is ~ 0.005; for irregular Hyperion / Phoebe (residual
          ~ 10 km, radius ~ 100-135 km) it is ~ 0.07-0.10.
        * The ``1 + 2 * sin^2(phase / 2)`` factor captures the
          orientation-uncertainty bound.  At phase 0 the entire
          hemisphere is lit and the factor is ``1`` -- we still do
          not know the body's rotational orientation so a full
          ``residual_km``-scale centroid bias is in play even at
          full phase.  At phase 90 the factor is ``2`` because only
          half the body is lit and the dark side could be hiding an
          equal amount of shape irregularity.  At phase 180 the
          factor is ``3`` since only the crescent is lit, leaving
          most of the body's irregularity hidden.

        ``body_radius_km`` is derived from the predicted disc geometry
        rather than from a separate static-data lookup so a body
        absent from the YAML or hard-coded shape table still gets a
        meaningful factor (the residual itself falls back to
        ``DEFAULT_BODY_SHAPE`` upstream).
        """
        if self._predicted_diameter_px <= 0.0 or self._km_per_pixel_at_limb <= 0.0:
            return 0.0
        body_radius_km = self._km_per_pixel_at_limb * (self._predicted_diameter_px / 2.0)
        if body_radius_km <= 0.0:
            return 0.0
        sin_half_phase = math.sin(math.radians(phase_angle_deg) / 2.0)
        phase_factor = 1.0 + 2.0 * sin_half_phase * sin_half_phase
        residual_fraction = shape.ellipsoid_rms_residual_km / body_radius_km
        return float(max(0.0, residual_fraction * phase_factor))

    def _lit_weighted_centroid_vu(self) -> tuple[float, float]:
        """Return the brightness-weighted centroid of the rendered body.

        Falls back to the geometric center when the model is empty or
        the body mask is all-False (degenerate render).  The centroid
        is in the same extfov coordinate frame ``_predicted_center_vu``
        uses, so the BLOB feature's geometry stays self-consistent.
        """
        assert self._model_img is not None
        assert self._body_mask is not None
        weights = np.where(self._body_mask, self._model_img, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return self._predicted_center_vu
        v_indices, u_indices = np.indices(weights.shape, dtype=np.float64)
        lit_v = float((weights * v_indices).sum() / total)
        lit_u = float((weights * u_indices).sum() / total)
        return (lit_v, lit_u)

    def _sub_solar_dir_vu(self, lit_centroid_vu: tuple[float, float]) -> tuple[float, float]:
        """Return the unit image-plane direction toward the bright limb.

        The brightness-weighted centroid of a partially-lit body sits between
        the geometric center and the bright limb, so the vector from the
        geometric center (``_predicted_center_vu``) to the lit centroid points
        along the projected body-to-Sun direction.  ``BodyBlobNav`` orients its
        phase-aware coarse template along this direction (a filled disc cannot
        match a high-phase crescent).  At low phase the two centroids nearly
        coincide and the direction is meaningless, so it collapses to
        ``(0.0, 0.0)``; the technique uses the disc template there and never
        consults the direction.

        Parameters:
            lit_centroid_vu: The brightness-weighted centroid (the value
                stored as the feature's predicted center), in the same extfov
                frame as ``_predicted_center_vu``.

        Returns:
            Unit ``(v, u)`` direction toward the bright limb, or
            ``(0.0, 0.0)`` when the lit centroid is within
            :data:`_SUB_SOLAR_MIN_OFFSET_PX` of the geometric center.
        """
        dv = lit_centroid_vu[0] - self._predicted_center_vu[0]
        du = lit_centroid_vu[1] - self._predicted_center_vu[1]
        norm = math.hypot(dv, du)
        if norm < _SUB_SOLAR_MIN_OFFSET_PX:
            return (0.0, 0.0)
        return (dv / norm, du / norm)

    def _create_annotations(
        self,
        u_center: int,
        v_center: int,
        model: NDArrayFloatType,
        limb_mask: NDArrayBoolType,
        body_mask: NDArrayBoolType,
    ) -> Annotations:
        """Creates annotation objects for labeling the body in visualizations.

        This is functionally equivalent to the implementation used by the
        normal body model, so annotation behavior is consistent across models.
        """
        obs = self._obs
        body_name = self._body_name
        body_config = self._config.bodies

        text_loc: list[TextLocInfo] = []
        v_center_extfov = v_center + obs.extfov_margin_v
        u_center_extfov = u_center + obs.extfov_margin_u

        v_center_extfov_clipped = np.clip(v_center_extfov, 0, body_mask.shape[0] - 1)
        u_center_extfov_clipped = np.clip(u_center_extfov, 0, body_mask.shape[1] - 1)
        if not body_mask[v_center_extfov_clipped].any():
            body_mask_u_min = 0
            body_mask_u_max = body_mask.shape[1] - 1
        else:
            body_mask_u_min = int(np.argmax(body_mask[v_center_extfov_clipped]))
            body_mask_u_max = int(
                body_mask.shape[1] - np.argmax(body_mask[v_center_extfov_clipped, ::-1]) - 1
            )
        body_mask_v_min = int(np.argmax(body_mask[:, u_center_extfov_clipped]))
        body_mask_v_max = int(
            body_mask.shape[0] - np.argmax(body_mask[::-1, u_center_extfov_clipped]) - 1
        )
        body_mask_u_ctr = (body_mask_u_min + body_mask_u_max) // 2
        body_mask_v_ctr = (body_mask_v_min + body_mask_v_max) // 2

        # Scan around center to place labels on limb
        for orig_dist in range(0, max(body_mask_v_ctr - body_mask_v_min, body_config.label_scan_v)):
            for neg in [-1, 1]:
                dist = orig_dist * neg
                v = body_mask_v_ctr + dist
                if not 0 <= v < body_mask.shape[0]:
                    continue

                # Left side
                u = int(np.argmax(body_mask[v]))
                if u > 0:
                    angle = np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov)) % 360
                    if 135 < angle < 225:  # Left side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_LEFT_ARROW, v, u - body_config.label_horiz_gap)
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u)
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u)
                        )

                # Right side
                u = body_mask.shape[1] - int(np.argmax(body_mask[v, ::-1])) - 1
                if u < body_mask.shape[1] - 1:
                    angle = np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov)) % 360
                    if angle > 315 or angle < 45:  # Right side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u + body_config.label_horiz_gap)
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u)
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u)
                        )

                if orig_dist == 0:
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_TOP_ARROW,
                            body_mask_v_min - body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_BOTTOM_ARROW,
                            body_mask_v_max + body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    break

        # Coarse scan for additional candidates
        for v_orig_dist in range(0, body_mask_v_ctr - body_mask_v_min, body_config.label_grid_v):
            for v_neg in [-1, 1]:
                v_dist = v_orig_dist * v_neg
                v = body_mask_v_ctr + v_dist
                if not 0 <= v < body_mask.shape[0]:
                    continue
                for u_orig_dist in range(
                    0, body_mask_u_ctr - body_mask_u_min, body_config.label_grid_u
                ):
                    for u_neg in [-1, 1]:
                        u_dist = u_orig_dist * u_neg
                        u = body_mask_u_ctr + u_dist
                        if not 0 <= u < body_mask.shape[1]:
                            continue
                        if not body_mask[v, u]:
                            continue
                        if u < model.shape[1] // 2:
                            text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW, v, u))
                        else:
                            text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u))
                if v_orig_dist == 0:
                    break

        text_info = AnnotationTextInfo(
            body_name,
            text_loc=text_loc,
            ref_vu=None,
            font=body_config.label_font,
            font_size=body_config.label_font_size,
            color=body_config.label_font_color,
        )

        text_avoid_mask = ndimage.maximum_filter(body_mask, body_config.label_mask_enlarge)

        annotation = Annotation(
            obs,
            limb_mask,
            body_config.label_limb_color,
            thicken_overlay=body_config.outline_thicken,
            avoid_mask=text_avoid_mask,
            text_info=text_info,
            config=self._config,
        )

        annotations = Annotations()
        annotations.add_annotations(annotation)
        return annotations


def _blob_reliability(*, snr: float, diameter_px: float) -> float:
    """Reliability of BODY_BLOB per the design (cap at 0.4).

    ``snr`` is the measured detection SNR from
    ``NavModelBodyBase._blob_detection_snr``.  Its sigmoid is centered
    at :data:`_BLOB_DETECTION_SNR_MIDPOINT` (the technique's own
    lit-pixel threshold) so a window with no signal above the noise
    order-statistics scores ~0.02 (gated for any body size) while a
    clearly detected body saturates the term.  The extent sigmoid
    (midpoint :data:`_BLOB_EXTENT_MIDPOINT_PX`, below the
    :data:`BODY_BLOB_MIN_DIAMETER_PX` emission floor) applies a mild
    near-floor discount; detection is the primary discriminator against
    the reliability gate's 0.20 threshold, which the combination crosses
    at SNR ~3.0 for a 20 px body up to ~4.4 at the 5 px floor.
    """
    snr_term = _sigmoid(snr - _BLOB_DETECTION_SNR_MIDPOINT)
    extent_term = _sigmoid((diameter_px - _BLOB_EXTENT_MIDPOINT_PX) / _BLOB_EXTENT_SCALE_PX)
    return float(snr_term * extent_term * 0.4)


def _sigmoid(x: float) -> float:
    """Logistic sigmoid (numerically safe)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
