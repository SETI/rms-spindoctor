"""Tests for ``spindoctor.nav_orchestrator.nav_context.NavContext``."""

from typing import Any

import numpy as np

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_orchestrator.provenance import Provenance


def _minimal_context(**overrides: Any) -> NavContext:
    """Build a context with the smallest valid set of fields.

    Keyword arguments override the corresponding ``NavContext`` field so
    callers can flip a single property (``fit_camera_rotation=True``,
    custom ``max_rotation_deg``, etc.) without rebuilding the entire
    fixture.
    """
    image = np.zeros((4, 4), np.float64)
    mask = np.ones((4, 4), bool)
    classifier = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )
    provenance = Provenance(
        spindoctor_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )
    defaults: dict[str, Any] = {
        'obs': object(),
        'image_ext': image,
        'sensor_mask_ext': mask,
        'image_noise_sigma': 1.0,
        'saturation_mask_ext': np.zeros((4, 4), bool),
        'cosmic_ray_mask_ext': np.zeros((4, 4), bool),
        'missing_data_mask_ext': np.zeros((4, 4), bool),
        'image_classifier': classifier,
        'provenance': provenance,
    }
    defaults.update(overrides)
    return NavContext(**defaults)


def test_navcontext_constructs_with_minimal_fields() -> None:
    """A minimal NavContext is constructible without optional fields."""
    ctx = _minimal_context()
    assert ctx.image_noise_sigma == 1.0
    assert ctx.prior_offset_px is None
    assert ctx.prior_covariance_px2 is None


def test_navcontext_with_prior_returns_new_instance() -> None:
    """``with_prior`` creates a new NavContext with the prior populated."""
    ctx = _minimal_context()
    cov = np.eye(2, dtype=np.float64) * 0.5
    new_ctx = ctx.with_prior(offset_px=(1.0, 2.0), covariance_px2=cov)
    assert new_ctx is not ctx
    assert new_ctx.prior_offset_px == (1.0, 2.0)
    assert ctx.prior_offset_px is None  # original unchanged


def test_navcontext_with_prior_preserves_other_fields() -> None:
    """``with_prior`` only changes the prior fields."""
    ctx = _minimal_context()
    cov = np.eye(2, dtype=np.float64)
    new_ctx = ctx.with_prior(offset_px=(0.0, 0.0), covariance_px2=cov)
    assert new_ctx.image_noise_sigma == ctx.image_noise_sigma
    assert new_ctx.image_classifier is ctx.image_classifier


def test_navcontext_rotation_fields_default_off() -> None:
    """Default NavContext disables rotation fitting at the standard 5 degree cap."""
    ctx = _minimal_context()
    assert ctx.fit_camera_rotation is False
    assert ctx.max_rotation_deg == 5.0


def test_navcontext_has_no_signal_scale_field() -> None:
    """The DN-to-image-unit scale was removed with the magnitude-based gate.

    The star gate now limits by magnitude against
    ``obs.star_max_usable_vmag()``, so no DN-to-image-unit scale is
    propagated through the NavContext.
    """
    ctx = _minimal_context()
    assert not hasattr(ctx, 'signal_dn_to_image_unit_scale')


def test_navcontext_rotation_fields_propagate() -> None:
    """Explicit rotation flags survive construction and ``with_prior``."""
    ctx = _minimal_context(fit_camera_rotation=True, max_rotation_deg=3.5)
    assert ctx.fit_camera_rotation is True
    assert ctx.max_rotation_deg == 3.5
    new_ctx = ctx.with_prior(
        offset_px=(0.5, -0.5),
        covariance_px2=np.eye(2, dtype=np.float64),
    )
    assert new_ctx.fit_camera_rotation is True
    assert new_ctx.max_rotation_deg == 3.5


def test_navcontext_with_prior_accepts_3x3_covariance() -> None:
    """``with_prior`` accepts a 3x3 prior covariance and keeps the 2x2 block."""
    ctx = _minimal_context()
    cov_3x3 = np.diag([0.1, 0.2, 1.0]).astype(np.float64)
    new_ctx = ctx.with_prior(offset_px=(0.0, 0.0), covariance_px2=cov_3x3)
    assert new_ctx.prior_covariance_px2 is not None
    assert new_ctx.prior_covariance_px2.shape == (2, 2)
    assert new_ctx.prior_covariance_px2[0, 0] == 0.1
    assert new_ctx.prior_covariance_px2[1, 1] == 0.2
