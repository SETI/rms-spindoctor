"""NavContext — per-image global state shared across extractors and techniques.

Created once per navigation by the orchestrator.  Every member is computed
without knowing where any feature lives in the image: global statistics,
sensor-vs-extfov masks, shared image-side derivatives, and provenance.

The context is frozen.  Pass-2 techniques receive a copy with the pass-1
ensemble's prior offset and covariance attached via ``with_prior``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.support.filters import NavFilterSpec
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['NavContext']


@dataclass(frozen=True, eq=False)
class NavContext:
    """Per-image global state shared across feature extraction and techniques.

    The context is frozen; ``with_prior`` returns a new instance via
    ``dataclasses.replace`` rather than mutating in place.

    Parameters:
        obs: The observation snapshot under navigation.  Typed loosely as
            ``object`` to avoid an import cycle; concrete value is an
            ``ObsSnapshotInst`` subclass.
        image_ext: The extended-FOV image array (post source-image filter).
        sensor_mask_ext: ``True`` where the pixel is real sensor data,
            ``False`` for extfov padding.
        image_noise_sigma: Robust MAD-based noise sigma in the image's
            native units (DN for ``raw_dn`` instruments, I/F for
            ``calibrated_if``), computed over the entire sensor area.
            Pixel-threshold consumers (cosmic-ray mask, body-blob noise
            floor, star detection) use this value directly because they
            compare against pixel intensities in the same units.
        saturation_mask_ext: ``True`` where pixels at or above the
            instrument's full-well DN.
        cosmic_ray_mask_ext: ``True`` where single-pixel cosmic-ray spikes
            were detected.
        image_classifier: The image-quality classifier's verdict.
        image_gradient_ext: Optional shared Sobel-of-Gaussian magnitude
            (computed once, reused by every DT-based technique).
        image_gradient_vu_ext: Optional ``(H, W, 2)`` per-pixel
            gradient-vector image (``[..., 0]`` is ``g_v``,
            ``[..., 1]`` is ``g_u``).  Sampled by the polarity filter to
            compare each model vertex's outward normal against the image's
            edge direction.
        image_edge_dt_ext: Optional shared Euclidean distance transform of
            the thresholded gradient image.  It is not signed: the values are
            non-negative everywhere and the zero locus is the edge pixels
            themselves, at their own centers, not an oriented boundary
            running along pixel edges half a pixel away.
        prior_offset_px: Prior offset from pass 1, ``None`` on pass 1.
        prior_covariance_px2: Prior offset covariance from pass 1.
        pre_filter_applied: NavFilterSpec applied to the source image (for
            diagnostic provenance), ``None`` if none.
        provenance: Provenance metadata; populated at context creation.
        fit_camera_rotation: Per-instrument flag enabling 3-DoF technique
            fits.  When True every technique adds in-plane camera rotation
            as a third parameter and reports a 3x3 covariance; when False
            (the default) techniques produce 2-DoF results.
        max_rotation_deg: Maximum allowed rotation magnitude (degrees)
            when ``fit_camera_rotation`` is True; rotation outside the
            bound triggers ``at_edge=True``.  Ignored when
            ``fit_camera_rotation`` is False.
    """

    obs: object
    image_ext: NDArrayFloatType
    sensor_mask_ext: NDArrayBoolType
    image_noise_sigma: float
    saturation_mask_ext: NDArrayBoolType
    cosmic_ray_mask_ext: NDArrayBoolType
    image_classifier: NavImageClassifierResult
    provenance: Provenance
    image_gradient_ext: NDArrayFloatType | None = None
    image_gradient_vu_ext: NDArrayFloatType | None = None
    image_edge_dt_ext: NDArrayFloatType | None = None
    prior_offset_px: tuple[float, float] | None = None
    prior_covariance_px2: NDArrayFloatType | None = None
    pre_filter_applied: NavFilterSpec | None = None
    fit_camera_rotation: bool = False
    max_rotation_deg: float = 5.0

    def with_prior(
        self,
        *,
        offset_px: tuple[float, float],
        covariance_px2: NDArrayFloatType,
    ) -> NavContext:
        """Return a new NavContext with pass-1 prior attached.

        ``with_prior`` is non-mutating; the existing instance is unchanged.

        Parameters:
            offset_px: ``(dv, du)`` offset to install as the pass-2 prior.
            covariance_px2: 2x2 covariance of that offset, or 3x3 when the
                pass-1 ensemble produced a rotation-aware result; the
                top-left 2x2 block is consumed.

        Returns:
            New ``NavContext`` with ``prior_offset_px`` and
            ``prior_covariance_px2`` populated.  Only the 2x2 translation
            block is kept; pass-2 techniques re-derive any rotation prior
            from the per-instrument flag.

        Raises:
            TypeError: if ``offset_px`` entries are non-numeric (cannot be
                coerced to ``float``).
            ValueError: if ``offset_px`` is not a length-2 sequence, contains
                non-finite entries, or ``covariance_px2`` is not square
                2x2 / 3x3 or contains non-finite entries.
        """
        if len(offset_px) != 2:
            raise ValueError(f'offset_px must be a length-2 sequence; got length {len(offset_px)}')
        try:
            dv, du = float(offset_px[0]), float(offset_px[1])
        except (TypeError, ValueError) as exc:
            raise TypeError(f'offset_px entries must be numeric; got {offset_px!r}') from exc
        if not (math.isfinite(dv) and math.isfinite(du)):
            raise ValueError(f'offset_px must be finite; got {offset_px!r}')
        cov_in = np.asarray(covariance_px2, np.float64)
        if cov_in.shape not in ((2, 2), (3, 3)):
            raise ValueError(f'covariance_px2 must have shape (2, 2) or (3, 3); got {cov_in.shape}')
        if not np.isfinite(cov_in).all():
            raise ValueError('covariance_px2 must contain only finite entries')
        # Pass-2 techniques consume the 2x2 translation block only; the
        # rotation prior carries no information across the pass boundary
        # (each technique re-derives it from its own geometry).
        cov_in = cov_in[:2, :2]
        # Take an independent copy and mark it read-only so the caller
        # cannot mutate the prior covariance after the NavContext is built.
        cov = cov_in.copy()
        cov.setflags(write=False)
        return dataclasses.replace(
            self,
            prior_offset_px=(dv, du),
            prior_covariance_px2=cov,
        )
