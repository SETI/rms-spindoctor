"""Catalog-driven star NavModel package.

The package is organized as a small set of helper modules around a thin
orchestrator class:

- ``catalog`` — multi-catalog reduction, stellar aberration, proper
  motion, FOV projection, dedup.
- ``conflicts`` — body and ring occlusion checks for catalog stars.
- ``predicted_snr`` — per-star integrated-SNR estimate plus the
  ``SCLASS_TO_B_MINUS_V`` spectral-class color lookup.
- ``smeared_psf`` — smear-aware PSF rendering and per-image smear vector.
- ``detection`` — DAOPHOT-style source detection (matched filter,
  centroid fit, shape cuts) used by downstream techniques.
- ``nav_model_stars`` — ``NavModelStars`` orchestrator implementing the
  ``NavModel`` ABC.
"""

from spindoctor.nav_model.stars.catalog import (
    CATALOG_MAGNITUDE_BINS,
    aberrate_star,
    reduce_catalogs,
    select_radec_list,
    stars_in_extfov,
)
from spindoctor.nav_model.stars.conflicts import (
    mark_body_and_ring_conflicts,
    parse_ring_occlusion_annuli,
)
from spindoctor.nav_model.stars.detection import (
    DAOPHOT_DEFAULT_DETECTION_SIGMA,
    DAOPHOT_DEFAULT_ROUNDNESS_BOUND,
    DAOPHOT_DEFAULT_SHARPNESS_MAX,
    DAOPHOT_DEFAULT_SHARPNESS_MIN,
    DetectedSource,
    apply_shape_cuts,
    centroid_gaussian_fit,
    centroid_saturated,
    detect_ccd_bloom_columns,
    detect_sources,
    matched_filter_image,
)
from spindoctor.nav_model.stars.nav_model_stars import NavModelStars
from spindoctor.nav_model.stars.nav_model_stars_simulated import NavModelStarsSimulated
from spindoctor.nav_model.stars.predicted_snr import (
    SCLASS_TO_B_MINUS_V,
    integrated_signal_dn,
    predicted_snr,
    psf_aperture_pixels,
    psf_sigma_px,
)
from spindoctor.nav_model.stars.saturation import (
    UCAC4_SATURATION_VMAG_LIMIT,
    correct_saturated_vmags,
    correct_star_photometry,
)
from spindoctor.nav_model.stars.smeared_psf import (
    compute_smear_vector_px,
    movement_granularity_px,
    render_smeared_psf,
    smear_length_px,
)

__all__ = [
    'CATALOG_MAGNITUDE_BINS',
    'DAOPHOT_DEFAULT_DETECTION_SIGMA',
    'DAOPHOT_DEFAULT_ROUNDNESS_BOUND',
    'DAOPHOT_DEFAULT_SHARPNESS_MAX',
    'DAOPHOT_DEFAULT_SHARPNESS_MIN',
    'SCLASS_TO_B_MINUS_V',
    'UCAC4_SATURATION_VMAG_LIMIT',
    'DetectedSource',
    'NavModelStars',
    'NavModelStarsSimulated',
    'aberrate_star',
    'apply_shape_cuts',
    'centroid_gaussian_fit',
    'centroid_saturated',
    'compute_smear_vector_px',
    'correct_saturated_vmags',
    'correct_star_photometry',
    'detect_ccd_bloom_columns',
    'detect_sources',
    'integrated_signal_dn',
    'mark_body_and_ring_conflicts',
    'matched_filter_image',
    'movement_granularity_px',
    'parse_ring_occlusion_annuli',
    'predicted_snr',
    'psf_aperture_pixels',
    'psf_sigma_px',
    'reduce_catalogs',
    'render_smeared_psf',
    'select_radec_list',
    'smear_length_px',
    'stars_in_extfov',
]
