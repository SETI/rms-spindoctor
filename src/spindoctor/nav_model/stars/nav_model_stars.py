"""Real-scene star NavModel.

Orchestrates the catalog-reduction, conflict-marking, predicted-SNR, and
smear-aware PSF helpers so the NavModel ABC can emit one ``STAR``
``NavFeature`` per detectable catalog star.  Heavy lifting lives in
sibling modules; this file is the entry point the orchestrator's
registry sees.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from oops import Observation

from spindoctor.annotation import (
    TEXTINFO_BOTTOM,
    TEXTINFO_BOTTOM_LEFT,
    TEXTINFO_BOTTOM_RIGHT,
    TEXTINFO_LEFT,
    TEXTINFO_RIGHT,
    TEXTINFO_TOP,
    TEXTINFO_TOP_LEFT,
    TEXTINFO_TOP_RIGHT,
    Annotation,
    Annotations,
    AnnotationTextInfo,
    TextLocInfo,
)
from spindoctor.config import Config
from spindoctor.feature.constants import MIN_ANISOTROPIC_SMEAR_PX
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.stars.catalog import reduce_catalogs
from spindoctor.nav_model.stars.conflicts import mark_body_and_ring_conflicts
from spindoctor.nav_model.stars.predicted_snr import (
    SCLASS_TO_B_MINUS_V,
    psf_sigma_px,
)
from spindoctor.nav_model.stars.smeared_psf import compute_smear_vector_px, smear_length_px
from spindoctor.support.flux import clean_sclass
from spindoctor.support.image import draw_rect
from spindoctor.support.time import now_dt

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext
    from spindoctor.support.types import MutableStar

from spindoctor.support.filters import NavFilterKind, NavFilterSpec

__all__ = [
    'SCLASS_TO_B_MINUS_V',
    'NavModelStars',
]

# Effective-SNR constants for the magnitude-margin reliability shape.
#
# The star gate is purely magnitude based: the catalog reduction and this
# model drop any star fainter than ``obs.star_max_usable_vmag()``.  The
# CRLB-covariance and reliability helpers still want an SNR-like quantity,
# so we synthesise one from how far below the limiting magnitude the star
# sits.  A star exactly at the limit gets ``snr_eff == SNR_REF``; every
# extra magnitude of headroom multiplies the effective SNR by one Pogson
# ratio (``2.512``), matching the flux ratio per magnitude.
#
# ``SNR_REF`` is set to 8.0 — just below the reliability sigmoid centre
# (snr=10) so a star at the very edge of usability lands near the steep
# part of the curve (reliability ~0.34) rather than saturating it, while a
# star a few magnitudes brighter saturates the curve toward 1.0.
# ``SNR_FLOOR`` keeps the synthesised SNR strictly positive so the
# covariance never degenerates to the zero-SNR huge-variance branch.
SNR_REF: float = 8.0
SNR_FLOOR: float = 0.1


class NavModelStars(NavModel):
    """Catalog-driven star NavModel.

    Reduces the configured catalogs into a deduplicated star list,
    flags body/ring conflicts, and emits one STAR ``NavFeature`` per
    star whose predicted SNR clears the configured floor and whose
    conflict flags allow it.

    Parameters:
        name: Model name (typically ``'stars'``).
        obs: Observation snapshot.
        config: Optional ``Config`` override.
    """

    _abstract = False

    def __init__(
        self,
        name: str,
        obs: Observation,
        *,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._stars_config = self._config.stars
        self._stars: list[MutableStar] = []
        self._smear_vu: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one star NavModel per real observation.

        Simulated obs have no real star-catalog pointing, so no star model is
        built for them (the sim renders its own stars).

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instance.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            ``[NavModelStars('stars', obs)]`` for a real obs, else ``[]``.
        """
        if getattr(obs, 'is_simulated', False):
            return []
        return [cls('stars', obs, config=config)]

    @property
    def stars(self) -> list[MutableStar]:
        """Reduced star list populated by ``create_model``."""
        return self._stars

    def create_model(self) -> None:
        """Build the reduced star list and populate metadata.

        Steps:

        1. Compute the per-image smear vector via ``compute_smear_vector_px``.
        2. Reduce all configured catalogs through ``reduce_catalogs`` —
           pulls per-bin chunks, dedupes against precedence, marks
           visual overlaps.
        3. Mark body and ring conflicts via ``mark_body_and_ring_conflicts``.
        4. Populate ``self._metadata`` with summary fields used by the
           curator.
        """
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        with self.log_section('CREATE STARS MODEL'):
            self._stars = reduce_catalogs(self.obs, self._config)
            self._smear_vu = compute_smear_vector_px(self.obs)
            mark_body_and_ring_conflicts(self.obs, self._config, self._stars)
            self._metadata['star_count'] = len(self._stars)
            self._metadata['stars'] = [_star_summary(star) for star in self._stars]
            body_conflicts = sum(1 for s in self._stars if (s.conflicts or '').startswith('BODY'))
            ring_conflicts = sum(1 for s in self._stars if (s.conflicts or '').startswith('RING'))
            smear_len = math.hypot(self._smear_vu[0], self._smear_vu[1])
            self._logger.info(
                '%d catalog stars after dedup; smear vector (v, u) = '
                '(%.3f, %.3f) px (|smear| = %.3f)',
                len(self._stars),
                self._smear_vu[0],
                self._smear_vu[1],
                smear_len,
            )
            self._logger.info(
                'Conflicts: body-occluded = %d, ring-occluded = %d',
                body_conflicts,
                ring_conflicts,
            )
            self._logger.info('Final star list (total %d):', len(self._stars))
            for star in self._stars:
                self._logger.info('  %s', _star_short_info(star))
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
            self._logger.info('Model created in %.3f s', self._metadata['elapsed_time_sec'])

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit one STAR feature per catalog star within the magnitude limit.

        Stars with body or ring conflicts are emitted with the matching
        ``in_body_silhouette`` / ``in_saturation_or_cosmic_mask`` flags
        set; the reliability gate decides whether to keep them.  Stars
        fainter than ``obs.star_max_usable_vmag()`` (or with no catalog
        magnitude) are skipped.  Detectability is expressed through a
        magnitude-margin-derived effective SNR rather than a DN-based
        photometric SNR, so the gate carries no dependence on any
        DN-to-image-unit scale.

        Parameters:
            context: Per-image NavContext.

        Returns:
            List of STAR ``NavFeature`` instances.
        """
        with self.log_section('EMIT STARS FEATURES'):
            return self._emit_features(context)

    def _emit_features(self, context: NavContext) -> list[NavFeature]:
        if not self._stars:
            self._logger.info('Catalog is empty — emitting no features')
            return []
        psf = self.obs.star_psf()
        sigma_psf = psf_sigma_px(psf)
        image_noise_sigma = float(context.image_noise_sigma)
        if image_noise_sigma <= 0.0:
            self._logger.info(
                'Image noise sigma is non-positive (%.3f) — emitting no features',
                image_noise_sigma,
            )
            return []
        mag_limit = float(self.obs.star_max_usable_vmag())
        min_snr = float(getattr(self._stars_config, 'min_predicted_snr', 0.0))
        max_smear = float(getattr(self._stars_config, 'max_smear', math.inf))
        mag_offset_value = _resolve_mag_offset(self.obs)
        self._logger.debug(
            'image_noise_sigma = %.3g, star_max_usable_vmag = %.3f, '
            'sigma_psf = %.3f px, mag_offset = %.3f, '
            'min_predicted_snr = %.2f, max_smear = %.2f px',
            image_noise_sigma,
            mag_limit,
            sigma_psf,
            mag_offset_value,
            min_snr,
            max_smear,
        )

        features: list[NavFeature] = []
        skipped_faint = 0
        skipped_smear = 0
        in_body_count = 0
        in_ring_count = 0
        for star in self._stars:
            # Magnitude gate: drop stars with no catalog magnitude and stars
            # fainter than the per-image limiting magnitude.  This is mostly
            # redundant with the catalog reduction's magnitude cap, but it
            # keeps the model self-contained and handles ``vmag is None``.  A
            # ``photometry_saturated`` star is exempt from the faint cut: its
            # magnitude is an untrusted, too-faint saturated reading, so it
            # must not be rejected on that reading.
            is_saturated = bool(star.photometry_saturated)
            if star.vmag is None or (float(star.vmag) > mag_limit and not is_saturated):
                skipped_faint += 1
                continue
            # Magnitude-margin-derived effective SNR: how far the star sits
            # below the limiting magnitude, expressed as a flux ratio.  A
            # star at the limit gets ``SNR_REF``; each brighter magnitude
            # multiplies by one Pogson ratio.  No DN units are involved.  A
            # saturated star's recorded magnitude is a lower bound on
            # brightness, so it is treated as at least as bright as the
            # limiting magnitude without fabricating a specific brighter value.
            eff_vmag = float(star.vmag)
            if is_saturated:
                eff_vmag = min(eff_vmag, mag_limit)
            snr_eff = SNR_REF * (2.512 ** (mag_limit - eff_vmag))
            snr_eff = max(snr_eff, SNR_FLOOR)
            smear_len = smear_length_px(star.move_v, star.move_u)
            # Per the design's "max_smear" gate — stars with a smear longer
            # than the per-instrument cap are unfittable even with the
            # smear-aware kernel, so we drop them from the feature list
            # rather than emit a structurally-broken feature.
            if smear_len > max_smear:
                skipped_smear += 1
                continue
            v_extfov, u_extfov = self._extfov_indices(star)
            in_body = bool((star.conflicts or '').startswith('BODY'))
            in_ring = bool((star.conflicts or '').startswith('RING'))
            # Saturation / cosmic-ray contamination is NOT determined here.  The
            # mask is image content at the *actual* star position, but the model
            # only knows the *predicted* (unshifted) position; with any pointing
            # offset the two differ, so a predicted-position lookup is meaningless
            # -- and a sharp star peak is itself flagged by ``cosmic_ray_mask``,
            # which would gate every star whenever the offset is near zero.
            # Real contamination is handled at detection time, where the matched
            # peak's actual position and shape are known.
            if in_body:
                in_body_count += 1
            if in_ring:
                in_ring_count += 1
            cov = _crlb_covariance(
                snr=snr_eff,
                sigma_psf=sigma_psf,
                move_v=star.move_v,
                move_u=star.move_u,
            )
            box_half = (star.psf_size[0] // 2 + 2, star.psf_size[1] // 2 + 2)
            bbox = (
                round(v_extfov - box_half[0]),
                round(u_extfov - box_half[1]),
                round(v_extfov + box_half[0] + 1),
                round(u_extfov + box_half[1] + 1),
            )
            features.append(
                NavFeature(
                    feature_id=_star_feature_id(star),
                    feature_type=NavFeatureType.STAR,
                    source_model=self.name,
                    geometry=StarGeometry(
                        predicted_vu=(float(v_extfov), float(u_extfov)),
                        catalog_vu=(float(v_extfov), float(u_extfov)),
                        bbox_extfov_vu=bbox,
                    ),
                    subject_range_km=float('inf'),
                    position_cov_px=cov,
                    intensity_sigma_rel=0.0,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=_reliability_from_snr(
                        snr=snr_eff,
                        in_body=in_body,
                        in_ring=in_ring,
                        in_saturation=False,
                    ),
                    reliability_reasons=NavReliabilityBreakdown(
                        predicted_snr=_snr_reason_score(snr_eff, min_snr),
                        in_body_silhouette=in_body or in_ring,
                        in_saturation_or_cosmic=False,
                        smear_length_ok=True,
                    ),
                    usable_types=frozenset({NavFeatureType.STAR}),
                    flags=StarFlags(
                        saturated=False,
                        smear_length_px=smear_len,
                        in_body_silhouette=in_body or in_ring,
                        in_saturation_or_cosmic_mask=False,
                        predicted_snr=float(snr_eff),
                        vmag=(None if star.vmag is None else float(star.vmag)),
                        photometry_corrected=bool(star.photometry_corrected),
                        photometry_saturated=is_saturated,
                    ),
                )
            )
        self._logger.info(
            'Emitted %d STAR feature(s) (faint skipped %d, smear skipped %d)',
            len(features),
            skipped_faint,
            skipped_smear,
        )
        if features:
            self._logger.debug(
                'Flagged in-body = %d, in-ring = %d',
                in_body_count,
                in_ring_count,
            )
        return features

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit star-box overlays plus name/magnitude labels."""
        del context
        if not self._stars:
            return Annotations()
        return self._build_annotations()

    def _build_annotations(self) -> Annotations:
        """Render star-box overlays + per-star labels for the summary PNG."""
        obs = self.obs
        text_info_list: list[AnnotationTextInfo] = []
        star_avoid_mask = obs.make_extfov_false()
        star_overlay = obs.make_extfov_false()
        stretch_boxes: list[tuple[int, int, int, int]] = []
        for star in self._stars:
            if star.conflicts and star.conflicts != 'STAR':
                # Skip body/ring-blocked stars; they are not labelled.
                continue
            v_idx, u_idx = self._extfov_indices(star)
            v_int = int(v_idx)
            u_int = int(u_idx)
            v_half = (star.psf_size[0] // 2) + 2
            u_half = (star.psf_size[1] // 2) + 2
            u_min, v_min = obs.clip_extfov(u_int - u_half, v_int - v_half)
            u_max, v_max = obs.clip_extfov(u_int + u_half, v_int + v_half)
            star_avoid_mask[v_min : v_max + 1, u_min : u_max + 1] = True
            # The detection box doubles as a local-contrast-stretch box in the
            # summary PNG: a faint star a few DN above a bright background is
            # otherwise indistinguishable from it after the whole-frame stretch.
            stretch_boxes.append((v_min, u_min, v_max + 1, u_max + 1))
            draw_rect(
                star_overlay,
                True,
                u_int,
                v_int,
                u_half,
                v_half,
                dot_spacing=1,
            )
            label_margin = u_half + 3
            text_info_list.append(
                _star_label(
                    star=star,
                    config=self._stars_config,
                    v_int=v_int,
                    u_int=u_int,
                    label_margin=label_margin,
                )
            )
        annotations = Annotations()
        annotations.add_annotations(
            Annotation(
                obs,
                star_overlay,
                self._stars_config.label_star_color,
                thicken_overlay=0,
                avoid_mask=star_avoid_mask,
                stretch_boxes=stretch_boxes,
                text_info=text_info_list,
            )
        )
        return annotations

    def _extfov_indices(self, star: MutableStar) -> tuple[float, float]:
        """Return ``(v, u)`` of ``star`` in extfov coordinates."""
        return (
            float(star.v) + float(self.obs.extfov_margin_v),
            float(star.u) + float(self.obs.extfov_margin_u),
        )


def _resolve_mag_offset(obs: Observation) -> float:
    """Return the per-camera-per-filter magnitude offset for a navigation.

    Reads ``obs.inst_config.mag_offset.{fallback_combo, mag_offset_table}``
    populated by ``ObsInst.from_file``.  When no per-instrument
    ``mag_offset`` block is configured (test fixtures, simulated obs
    that skip the ``from_file`` path), the offset is ``0.0`` — the
    legacy hard-coded value.

    The resolved offset is read from
    ``mag_offset_table[fallback_combo]['default']``; the per-color-bin
    sub-keys are reserved for phase-10 calibration.
    """
    inst_config = getattr(obs, 'inst_config', None)
    if inst_config is None:
        return 0.0
    block = inst_config.get('mag_offset')
    if not block:
        return 0.0
    fallback = block.get('fallback_combo')
    table = block.get('mag_offset_table') or {}
    entry = table.get(fallback) if fallback is not None else None
    if entry is None and table:
        entry = next(iter(table.values()), None)
    if entry is None:
        return 0.0
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return float(entry)
    if isinstance(entry, dict):
        default = entry.get('default', 0.0)
        return float(default) if isinstance(default, (int, float)) else 0.0
    return 0.0


def _safe_mask_lookup(
    mask: Any,
    v: float,
    u: float,
) -> bool:
    """Return ``mask[round(v), round(u)]`` clamped to bounds, defaulting False."""
    if mask is None:
        return False
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return False
    rows, cols = arr.shape
    vi = int(np.clip(round(v), 0, rows - 1))
    ui = int(np.clip(round(u), 0, cols - 1))
    return bool(arr[vi, ui])


def _crlb_covariance(
    *,
    snr: float,
    sigma_psf: float,
    move_v: float,
    move_u: float,
) -> np.ndarray:
    """Return the 2x2 anisotropic CRLB centroid covariance for a star.

    Implements the formula from the design's STAR section:

    ::

        sigma_along  = sqrt(L^2 / 12 + sigma_PSF^2) / sqrt(SNR)
        sigma_across = sigma_PSF / sqrt(SNR)

    rotated so the major axis aligns with the smear vector ``(my, mx)``.
    Below ``MIN_ANISOTROPIC_SMEAR_PX``, the covariance collapses to an
    isotropic ``(sigma_PSF / sqrt(SNR))^2 * I``.

    Parameters:
        snr: Predicted integrated SNR.
        sigma_psf: Per-pixel PSF Gaussian sigma in pixels.
        move_v: Smear vector V component (pixels).
        move_u: Smear vector U component (pixels).

    Returns:
        2x2 numpy float array with rows / columns in (v, u) order.
    """
    if snr <= 0.0:
        return np.eye(2, dtype=np.float64) * 1e6
    if sigma_psf <= 0.0:
        raise ValueError(f'sigma_psf must be > 0; got {sigma_psf!r}')
    smear = math.hypot(move_v, move_u)
    s_across_sq = (sigma_psf * sigma_psf) / snr
    if smear < MIN_ANISOTROPIC_SMEAR_PX:
        return np.eye(2, dtype=np.float64) * s_across_sq
    s_along_sq = (smear * smear / 12.0 + sigma_psf * sigma_psf) / snr
    cos_t = move_v / smear
    sin_t = move_u / smear
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    diag = np.diag([s_along_sq, s_across_sq])
    cov = rot @ diag @ rot.T
    return 0.5 * (cov + cov.T)


def _snr_reason_score(snr: float, min_snr: float) -> float:
    """Map an effective SNR onto the reliability-breakdown ``predicted_snr`` field.

    The ``NavReliabilityBreakdown.predicted_snr`` is a [0, 1] contribution
    score, not the raw SNR.  ``snr`` here is the magnitude-margin-derived
    effective SNR (``SNR_REF * 2.512 ** (mag_limit - vmag)``), not a
    photometric DN SNR.  This helper folds the configured floor in: when
    ``min_snr > 0``, the score is ``snr / min_snr`` capped at 1; when no
    floor is configured, an SNR of 50 saturates the score (the same centre
    the reliability sigmoid uses).

    Parameters:
        snr: Effective magnitude-margin SNR.
        min_snr: Configured floor (``stars.min_predicted_snr``); 0
            when unset.

    Returns:
        Contribution score in ``[0, 1]``.
    """
    if min_snr > 0.0:
        return float(min(1.0, snr / min_snr))
    return float(min(1.0, snr / 50.0))


def _reliability_from_snr(
    *,
    snr: float,
    in_body: bool,
    in_ring: bool,
    in_saturation: bool,
) -> float:
    """Return the [0, 1] reliability for a star.

    Sigmoid-of-SNR core multiplied by hard zero terms for occlusion and
    saturation.  Matches the design's STAR formula:
    ``sigmoid(alpha_0 + alpha_1*(snr - threshold))`` with the
    ``not_in_body_silhouette`` / ``not_in_saturation_or_cosmic`` factors
    applied multiplicatively.  Excessive-smear stars are excluded from
    the feature list before this helper runs.

    Parameters:
        snr: Predicted SNR.
        in_body: True if the star is occluded by a body silhouette.
        in_ring: True if the star is occluded by a ring annulus.
        in_saturation: True if the predicted pixel is saturated or
            tagged as a cosmic-ray hit.

    Returns:
        Reliability scalar in ``[0, 1]``.
    """
    if in_body or in_ring or in_saturation:
        return 0.0
    # Sigmoid centred near snr=10 with a span of ~5 — stars at snr=20 are
    # saturated to ~1, stars at snr=5 are around 0.27.  Coefficients are
    # the calibration starting point; phase-5 fitting will refine them.
    z = 0.2 * (snr - 10.0)
    return float(1.0 / (1.0 + math.exp(-z)))


def _star_feature_id(star: MutableStar) -> str:
    """Return the ``feature_id`` for ``star``: ``star:<catalog>:<id>``."""
    cat = (star.catalog_name or 'unknown').upper()
    uid = star.unique_number if star.unique_number is not None else star.pretty_name
    return f'star:{cat}:{uid}'


def _star_short_info(star: MutableStar) -> str:
    """Return a one-line text summary of a star, suitable for INFO logging.

    Mirrors the shape of the legacy ``NavModelStars._star_short_info`` line so
    the per-image log keeps a familiar listing that operators can grep.
    """
    jb = getattr(star, 'johnson_mag_b', None) or 0.0
    jv = getattr(star, 'johnson_mag_v', None) or 0.0
    temp = getattr(star, 'temperature', None) or 0.0
    return (
        f'Star {star.catalog_name:>6s}/{star.pretty_name:>9s} '
        f'U {star.u:9.3f}+/-{abs(star.move_u):7.3f} '
        f'V {star.v:9.3f}+/-{abs(star.move_v):7.3f} '
        f'VMAG {(-1.0 if star.vmag is None else star.vmag):6.3f} '
        f'JBMAG {jb:6.3f} '
        f'JVMAG {jv:6.3f} '
        f'SCLASS {clean_sclass(star.spectral_class or ""):>3s} '
        f'TEMP {temp:6.0f} '
        f'CONFLICT {star.conflicts}'
    )


def _star_summary(star: MutableStar) -> dict[str, Any]:
    """Return a compact JSON-friendly summary of a star (for metadata)."""
    return {
        'catalog_name': star.catalog_name,
        'unique_number': star.unique_number,
        'pretty_name': star.pretty_name,
        'ra_deg': float(np.degrees(star.ra_pm)),
        'dec_deg': float(np.degrees(star.dec_pm)),
        'vmag': star.vmag,
        'photometry_corrected': star.photometry_corrected,
        'photometry_saturated': star.photometry_saturated,
        'u': star.u,
        'v': star.v,
        'move_u': star.move_u,
        'move_v': star.move_v,
        'spectral_class': star.spectral_class,
        'conflicts': star.conflicts,
    }


def _star_label(
    *,
    star: MutableStar,
    config: Any,
    v_int: int,
    u_int: int,
    label_margin: int,
) -> AnnotationTextInfo:
    """Return the label info for one star (8 candidate placements)."""
    sclass = clean_sclass(star.spectral_class or '')
    line1 = star.pretty_name[:10]
    line2 = f'{star.vmag:.3f} {sclass}' if star.vmag is not None else sclass
    text_loc: list[TextLocInfo] = [
        TextLocInfo(TEXTINFO_BOTTOM, v_int + label_margin, u_int),
        TextLocInfo(TEXTINFO_TOP, v_int - label_margin, u_int),
        TextLocInfo(TEXTINFO_LEFT, v_int, u_int - label_margin),
        TextLocInfo(TEXTINFO_RIGHT, v_int, u_int + label_margin),
        TextLocInfo(TEXTINFO_TOP_LEFT, v_int - label_margin, u_int - label_margin),
        TextLocInfo(TEXTINFO_TOP_RIGHT, v_int - label_margin, u_int + label_margin),
        TextLocInfo(TEXTINFO_BOTTOM_LEFT, v_int + label_margin, u_int - label_margin),
        TextLocInfo(TEXTINFO_BOTTOM_RIGHT, v_int + label_margin, u_int + label_margin),
    ]
    return AnnotationTextInfo(
        f'{line1}\n{line2}',
        ref_vu=(v_int, u_int),
        text_loc=text_loc,
        font=config.label_font,
        font_size=config.label_font_size,
        color=config.label_font_color,
    )
