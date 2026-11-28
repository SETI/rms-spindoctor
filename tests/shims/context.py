"""Minimal ``NavContext`` factory for feature-emission tests.

``NavModel.to_features`` takes a ``NavContext`` so extractors can read
per-image statistics (the BODY_BLOB detection SNR reads ``image_ext``,
the validity masks, and ``image_noise_sigma``).  Tests that only assert
which feature types are emitted, or want to exercise the detection SNR
against a synthetic frame, can build one here instead of hand-rolling
masks, classifier verdicts, and provenance.
"""

from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.provenance import Provenance

__all__ = ['bare_nav_context']


def bare_nav_context(
    obs: Any,
    image_ext: np.ndarray | None = None,
    *,
    image_noise_sigma: float = 1.0,
) -> NavContext:
    """Build a minimal ``NavContext`` sized to ``obs``'s extfov.

    All masks are trivially populated (full-sensor, no saturation, no
    cosmic rays), the classifier reports a clean image, and provenance
    carries deterministic values.  The default frame is all-zero, so a
    BODY_BLOB feature's detection SNR is simply 0 against it; pass
    ``image_ext`` to exercise the SNR against real content.

    Parameters:
        obs: Any obs exposing ``extdata_shape_vu`` (``FakeObs``,
            ``ObsSim``, or a real snapshot).
        image_ext: Optional extended-FOV image; must match
            ``obs.extdata_shape_vu``.  Defaults to zeros.
        image_noise_sigma: Global noise sigma to install on the context.

    Returns:
        A fully-populated ``NavContext`` (without image derivatives).
    """
    shape = tuple(int(s) for s in obs.extdata_shape_vu)
    if image_ext is None:
        image_ext = np.zeros(shape, dtype=np.float64)
    if image_ext.shape != shape:
        raise ValueError(f'image_ext shape {image_ext.shape} != extfov shape {shape}')
    return NavContext(
        obs=obs,
        image_ext=image_ext,
        sensor_mask_ext=np.ones(shape, dtype=bool),
        image_noise_sigma=image_noise_sigma,
        saturation_mask_ext=np.zeros(shape, dtype=bool),
        cosmic_ray_mask_ext=np.zeros(shape, dtype=bool),
        missing_data_mask_ext=np.zeros(shape, dtype=bool),
        image_classifier=NavImageClassifierResult(
            image_class='clean',
            saturation_frac=0.0,
            missing_frac=0.0,
            noise_sigma=image_noise_sigma,
            max_dn=float(image_ext.max()) if image_ext.size else 0.0,
            flags=[],
        ),
        provenance=Provenance(
            spindoctor_version='0.0.0',
            image_et=0.0,
            pipeline_run_iso8601='2026-07-10T00:00:00Z',
            technique_names=(),
            extractor_names=(),
        ),
    )
