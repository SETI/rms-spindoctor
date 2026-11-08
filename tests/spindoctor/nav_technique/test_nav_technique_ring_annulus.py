"""End-to-end tests for ``RingAnnulusNav``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import NavContextFactory

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import RingAnnulusFlags
from spindoctor.feature.geometry import RingAnnulusGeometry
from spindoctor.nav_technique.diagnostics import RingAnnulusDiagnostics
from spindoctor.nav_technique.nav_technique import NCCCovarianceTuning
from spindoctor.nav_technique.nav_technique_ring_annulus import (
    RingAnnulusNav,
    _annulus_orbit_channel,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType


def _annulus_template(
    *,
    bbox_extfov_vu: tuple[int, int, int, int],
    center_in_template_vu: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Build a postage-stamp RING_ANNULUS template (anti-aliased annulus).

    The annulus carries bright pixels in the radial band
    ``[inner_radius, outer_radius]`` and zero elsewhere.  The 1-pixel
    anti-aliased ramp at each edge gives the NCC a clean sub-pixel
    convergence target without integer-grid bias.
    """
    h = bbox_extfov_vu[2] - bbox_extfov_vu[0]
    w = bbox_extfov_vu[3] - bbox_extfov_vu[1]
    vs, us = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    rr = np.hypot(vs - center_in_template_vu[0], us - center_in_template_vu[1])
    inside = (rr >= inner_radius - 0.5) & (rr <= outer_radius + 0.5)
    in_band = (rr >= inner_radius + 0.5) & (rr <= outer_radius - 0.5)
    inner_ramp = np.clip(rr - (inner_radius - 0.5), 0.0, 1.0)
    outer_ramp = np.clip((outer_radius + 0.5) - rr, 0.0, 1.0)
    template_img = np.where(in_band, 100.0, 0.0).astype(np.float64)
    edge_ramp = 100.0 * np.minimum(inner_ramp, outer_ramp)
    template_img = np.where(inside & ~in_band, edge_ramp, template_img)
    template_mask: NDArrayBoolType = template_img > 1e-6
    return template_img, template_mask


def _render_annulus_image(
    shape: tuple[int, int],
    center_vu: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> NDArrayFloatType:
    """Render an anti-aliased bright annulus image at full extfov shape."""
    vs, us = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    rr = np.hypot(vs - center_vu[0], us - center_vu[1])
    in_band = (rr >= inner_radius + 0.5) & (rr <= outer_radius - 0.5)
    inside = (rr >= inner_radius - 0.5) & (rr <= outer_radius + 0.5)
    inner_ramp = np.clip(rr - (inner_radius - 0.5), 0.0, 1.0)
    outer_ramp = np.clip((outer_radius + 0.5) - rr, 0.0, 1.0)
    image = np.where(in_band, 100.0, 0.0).astype(np.float64)
    image = np.where(inside & ~in_band, 100.0 * np.minimum(inner_ramp, outer_ramp), image)
    return image


def _make_annulus_feature(
    planet_name: str,
    *,
    extfov_shape: tuple[int, int],
    image_center_vu: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    planted_offset_vu: tuple[float, float] = (0.0, 0.0),
    subject_range_km: float = 1.5e9,
    constituent_count: int = 4,
    orbit_normals_vu: NDArrayFloatType | None = None,
    sigma_orbit_radial_px: float = 0.0,
) -> NavFeature:
    """Build a RING_ANNULUS feature whose template is shifted by a planted offset.

    The template is a bright anti-aliased annulus placed inside a
    postage-stamp bbox.  ``planted_offset_vu`` shifts the predicted
    center relative to the actual image center, so the technique should
    report ``offset_px = planted_offset_vu`` (predicted + offset =
    actual).  ``orbit_normals_vu`` and ``sigma_orbit_radial_px`` populate
    the radial orbit-uncertainty channel the technique widens its
    covariance from.
    """
    if orbit_normals_vu is None:
        orbit_normals_vu = np.zeros((0, 2), dtype=np.float64)
    pred_v = image_center_vu[0] - planted_offset_vu[0]
    pred_u = image_center_vu[1] - planted_offset_vu[1]
    half_extent = round(outer_radius) + 8
    v_min = max(0, round(pred_v - half_extent))
    u_min = max(0, round(pred_u - half_extent))
    v_max = min(extfov_shape[0], round(pred_v + half_extent))
    u_max = min(extfov_shape[1], round(pred_u + half_extent))
    bbox = (v_min, u_min, v_max, u_max)
    center_in_template = (pred_v - v_min, pred_u - u_min)
    template_img, template_mask = _annulus_template(
        bbox_extfov_vu=bbox,
        center_in_template_vu=center_in_template,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
    return NavFeature(
        feature_id=f'ring_annulus:{planet_name}',
        feature_type=NavFeatureType.RING_ANNULUS,
        source_model='rings',
        geometry=RingAnnulusGeometry(
            bbox_extfov_vu=bbox,
            predicted_center_vu=(pred_v, pred_u),
            orbit_normals_vu=orbit_normals_vu,
            sigma_orbit_radial_px=sigma_orbit_radial_px,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.05,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.55,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
        flags=RingAnnulusFlags(
            planet_name=planet_name,
            constituent_edge_count=constituent_count,
        ),
        template_img=template_img,
        template_mask=template_mask,
    )


def test_ring_annulus_recovers_planted_offset_single_annulus(
    make_nav_context: NavContextFactory,
) -> None:
    """One RING_ANNULUS against an anti-aliased annulus image converges below 1 px."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    inner_radius = 14.0
    outer_radius = 28.0
    image = _render_annulus_image(shape, image_center, inner_radius, outer_radius)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        planted_offset_vu=(2.0, -3.0),
    )
    technique = RingAnnulusNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    assert feasibility.consumed_feature_count == 1
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(2.0, abs=1.0)
    assert result.offset_px[1] == pytest.approx(-3.0, abs=1.0)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.annulus_count == 1


def test_ring_annulus_multi_planet_z_buffer_paint(
    make_nav_context: NavContextFactory,
) -> None:
    """Two RING_ANNULUS features fuse via Z-buffer paint and recover one offset.

    Multi-planet scenes (rare) emit one RING_ANNULUS per detectable
    ring system and the technique handles ``len(features) > 1``.
    """
    shape = (240, 240)
    inner_radius = 12.0
    outer_radius = 24.0
    centers = [(60.0, 70.0), (170.0, 160.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in centers:
        image += _render_annulus_image(shape, c, inner_radius, outer_radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_annulus_feature(
            f'PLANET_{i}',
            extfov_shape=shape,
            image_center_vu=c,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            planted_offset_vu=planted,
            # Vary subject_range so the depth ordering is well-defined.
            subject_range_km=1.5e9 * (i + 1),
        )
        for i, c in enumerate(centers)
    ]
    technique = RingAnnulusNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=1.0)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=1.0)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.annulus_count == 2


def test_ring_annulus_infeasible_on_empty_input() -> None:
    technique = RingAnnulusNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_ring_annulus_features' in report.reason


def test_ring_annulus_infeasible_when_no_template() -> None:
    """A RING_ANNULUS feature without a template payload is rejected."""
    feature = NavFeature(
        feature_id='ring_annulus:no_template',
        feature_type=NavFeatureType.RING_ANNULUS,
        source_model='rings',
        geometry=RingAnnulusGeometry(
            bbox_extfov_vu=(0, 0, 10, 10),
            predicted_center_vu=(5.0, 5.0),
        ),
        subject_range_km=1.0e9,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.4,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
        flags=RingAnnulusFlags(planet_name='no_template', constituent_edge_count=0),
    )
    technique = RingAnnulusNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False


@pytest.fixture
def at_edge_annulus_result(
    make_nav_context: NavContextFactory,
) -> NavTechniqueResult:
    """Build a RingAnnulusNav result whose offset hits the search-window edge.

    Plants an annulus at exactly the search-window axis bound, runs
    ``RingAnnulusNav``, and returns the resulting
    :class:`NavTechniqueResult` for the at-edge / hard-zero assertions
    to consume.
    """
    shape = (160, 160)
    image_center = (80.0, 80.0)
    inner_radius = 10.0
    outer_radius = 20.0
    image = _render_annulus_image(shape, image_center, inner_radius, outer_radius)
    margin = 5
    feature = _make_annulus_feature(
        'edge_planet',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        planted_offset_vu=(float(margin), 0.0),
    )
    technique = RingAnnulusNav()
    context = make_nav_context(image, extfov_margin_vu=(margin, margin))
    return technique.navigate([feature], context)


def test_ring_annulus_marks_at_edge_when_offset_hits_window(
    at_edge_annulus_result: NavTechniqueResult,
) -> None:
    """The pyramid wrapper flags the boundary peak as ``at_edge``."""
    assert at_edge_annulus_result.at_edge is True


def test_ring_annulus_at_edge_forces_zero_confidence(
    at_edge_annulus_result: NavTechniqueResult,
) -> None:
    """The ``hard_zero_if={'at_edge': True}`` gate drives confidence to 0."""
    assert at_edge_annulus_result.confidence == pytest.approx(0.0)


def test_ring_annulus_registered_with_navtechnique_registry() -> None:
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert RingAnnulusNav in NavTechnique._registry


def test_ring_annulus_diagnostics_records_quality_and_count(
    make_nav_context: NavContextFactory,
) -> None:
    """A clean planted-offset case reports positive ncc_peak and the right annulus_count."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    inner_radius = 14.0
    outer_radius = 28.0
    image = _render_annulus_image(shape, image_center, inner_radius, outer_radius)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        planted_offset_vu=(1.0, 1.0),
    )
    technique = RingAnnulusNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.ncc_peak > 0.0
    assert result.diagnostics.annulus_count == 1


def test_ring_annulus_diagnostics_records_peak_to_runner_up_ratio(
    make_nav_context: NavContextFactory,
) -> None:
    """A clean single-annulus scene reports peak-to-runner-up ratio > 1.0."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    inner_radius = 14.0
    outer_radius = 28.0
    image = _render_annulus_image(shape, image_center, inner_radius, outer_radius)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        planted_offset_vu=(1.0, 1.0),
    )
    technique = RingAnnulusNav()
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.peak_to_runner_up_ratio > 1.0


def test_ring_annulus_navigate_raises_when_no_eligible_features(
    make_nav_context: NavContextFactory,
) -> None:
    """``navigate`` raises ``ValueError`` if every input feature lacks a template.

    The orchestrator gates with ``is_feasible`` before invoking
    ``navigate``, but a direct caller (a debugger, a manual harness)
    can skip that step.  The boundary check makes the failure mode
    explicit instead of letting downstream code fail with an opaque
    array-shape error.
    """
    feature = NavFeature(
        feature_id='ring_annulus:no_template',
        feature_type=NavFeatureType.RING_ANNULUS,
        source_model='rings',
        geometry=RingAnnulusGeometry(
            bbox_extfov_vu=(0, 0, 10, 10),
            predicted_center_vu=(5.0, 5.0),
        ),
        subject_range_km=1.0e9,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.4,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
        flags=RingAnnulusFlags(planet_name='no_template', constituent_edge_count=0),
    )
    technique = RingAnnulusNav()
    context = make_nav_context(np.zeros((100, 100), dtype=np.float64), extfov_margin_vu=(16, 16))
    with pytest.raises(ValueError, match=r'No usable RING_ANNULUS templates available'):
        technique.navigate([feature], context)


def test_upsample_factor_returns_default_when_offset_block_missing() -> None:
    """``_upsample_factor`` falls back to the module default when ``config.offset`` is None."""
    from spindoctor.nav_technique.nav_technique_ring_annulus import _DEFAULT_UPSAMPLE_FACTOR

    technique = RingAnnulusNav()

    class _StubConfig:
        offset = None

    technique._config = _StubConfig()  # type: ignore[assignment]
    assert technique._upsample_factor() == _DEFAULT_UPSAMPLE_FACTOR


def test_upsample_factor_returns_default_when_value_missing() -> None:
    """``_upsample_factor`` falls back when the offset block has no upsample key."""
    from spindoctor.nav_technique.nav_technique_ring_annulus import _DEFAULT_UPSAMPLE_FACTOR

    technique = RingAnnulusNav()

    class _StubOffset:
        pass

    class _StubConfig:
        offset = _StubOffset()

    technique._config = _StubConfig()  # type: ignore[assignment]
    assert technique._upsample_factor() == _DEFAULT_UPSAMPLE_FACTOR


def test_upsample_factor_rejects_bool_value() -> None:
    """Boolean values are rejected even though Python treats ``bool`` as ``int``."""
    technique = RingAnnulusNav()

    class _StubOffset:
        correlation_fft_upsample_factor = True

    class _StubConfig:
        offset = _StubOffset()

    technique._config = _StubConfig()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r'must be a real \(non-bool\) number'):
        technique._upsample_factor()


def test_upsample_factor_rejects_string_value() -> None:
    """Non-numeric values raise ``ValueError`` naming the config key."""
    technique = RingAnnulusNav()

    class _StubOffset:
        correlation_fft_upsample_factor = 'abc'

    class _StubConfig:
        offset = _StubOffset()

    technique._config = _StubConfig()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r'correlation_fft_upsample_factor must be a real'):
        technique._upsample_factor()


def test_upsample_factor_rejects_zero() -> None:
    """Values below 1 are out of range and raise ``ValueError``."""
    technique = RingAnnulusNav()

    class _StubOffset:
        correlation_fft_upsample_factor = 0

    class _StubConfig:
        offset = _StubOffset()

    technique._config = _StubConfig()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r'must lie in \[1,'):
        technique._upsample_factor()


def test_upsample_factor_rejects_overlarge_value() -> None:
    """Values above the upper bound raise ``ValueError`` to prevent FFT overflow."""
    from spindoctor.nav_technique.nav_technique_ring_annulus import _MAX_UPSAMPLE_FACTOR

    technique = RingAnnulusNav()

    class _StubOffset:
        correlation_fft_upsample_factor = _MAX_UPSAMPLE_FACTOR + 1

    class _StubConfig:
        offset = _StubOffset()

    technique._config = _StubConfig()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r'must lie in \[1,'):
        technique._upsample_factor()


def test_ring_annulus_model_error_floor_inflates_covariance(
    make_nav_context: NavContextFactory,
) -> None:
    """model_error_floor_px adds exactly its square to the covariance diagonal.

    The peak-curvature covariance measures statistical precision only; the
    floor (calibrated against the simulated-scene campaign) carries the
    template model error.  Isolate the floor by zeroing the localization and
    size terms in both runs.
    """
    shape = (180, 180)
    image_center = (90.0, 90.0)
    image = _render_annulus_image(shape, image_center, 14.0, 28.0)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=14.0,
        outer_radius=28.0,
        planted_offset_vu=(2.0, -3.0),
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    bare = RingAnnulusNav()
    bare._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )
    floored = RingAnnulusNav()
    floored._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=1.5
    )
    cov_bare = bare.navigate([feature], context).covariance_px2
    cov_floored = floored.navigate([feature], context).covariance_px2
    for axis in (0, 1):
        assert cov_floored[axis, axis] == pytest.approx(cov_bare[axis, axis] + 1.5**2, rel=1e-9)


def _closed_ring_normals(count: int) -> NDArrayFloatType:
    """Return ``count`` outward unit normals evenly spaced around a full ring."""
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _orbit_channel_fixture(
    make_nav_context: NavContextFactory,
    *,
    orbit_normals_vu: NDArrayFloatType,
    sigma_orbit_radial_px: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Navigate one annulus with and without the orbit channel; return both covariances.

    Both runs disable the size and localization covariance terms so the
    only difference between them is the orbit-uncertainty inflation.
    """
    shape = (180, 180)
    image_center = (90.0, 90.0)
    image = _render_annulus_image(shape, image_center, 14.0, 28.0)
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )

    def _cov(sigma: float) -> NDArrayFloatType:
        feature = _make_annulus_feature(
            'SATURN',
            extfov_shape=shape,
            image_center_vu=image_center,
            inner_radius=14.0,
            outer_radius=28.0,
            planted_offset_vu=(2.0, -3.0),
            orbit_normals_vu=orbit_normals_vu,
            sigma_orbit_radial_px=sigma,
        )
        technique = RingAnnulusNav()
        technique._cov_tuning = tuning
        return np.asarray(technique.navigate([feature], context).covariance_px2)

    return _cov(0.0), _cov(sigma_orbit_radial_px)


def test_orbit_channel_inflates_isotropically_on_closed_ring(
    make_nav_context: NavContextFactory,
) -> None:
    """A closed ring's normals cancel, so the orbit sigma widens every axis equally.

    A uniform radial error of a full ring is a dilation, not a
    translation, so the absorbed direction is near zero and the bound is
    reported on both axes (the direction the nonlinear NCC would lock is
    unpredictable).
    """
    sigma = 5.0
    cov_off, cov_on = _orbit_channel_fixture(
        make_nav_context, orbit_normals_vu=_closed_ring_normals(360), sigma_orbit_radial_px=sigma
    )
    assert cov_on[0, 0] == pytest.approx(cov_off[0, 0] + sigma**2, rel=1e-6)
    assert cov_on[1, 1] == pytest.approx(cov_off[1, 1] + sigma**2, rel=1e-6)


def test_orbit_channel_directional_on_parallel_normals(
    make_nav_context: NavContextFactory,
) -> None:
    """Normals pointing one way widen only that axis (a short arc absorbs it fully)."""
    sigma = 5.0
    normals = np.tile(np.array([1.0, 0.0]), (200, 1))
    cov_off, cov_on = _orbit_channel_fixture(
        make_nav_context, orbit_normals_vu=normals, sigma_orbit_radial_px=sigma
    )
    assert cov_on[0, 0] == pytest.approx(cov_off[0, 0] + sigma**2, rel=1e-6)
    assert cov_on[1, 1] == pytest.approx(cov_off[1, 1], rel=1e-6)


def test_orbit_channel_absent_when_sigma_zero(
    make_nav_context: NavContextFactory,
) -> None:
    """No declared orbit sigma leaves the covariance and diagnostic untouched."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    image = _render_annulus_image(shape, image_center, 14.0, 28.0)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=14.0,
        outer_radius=28.0,
        planted_offset_vu=(2.0, -3.0),
        orbit_normals_vu=_closed_ring_normals(360),
        sigma_orbit_radial_px=0.0,
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    technique = RingAnnulusNav()
    technique._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.sigma_orbit_radial_px == 0.0


def test_orbit_channel_records_effective_sigma_in_diagnostics(
    make_nav_context: NavContextFactory,
) -> None:
    """The consumed orbit sigma is echoed on the diagnostics for auditability."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    image = _render_annulus_image(shape, image_center, 14.0, 28.0)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=14.0,
        outer_radius=28.0,
        orbit_normals_vu=_closed_ring_normals(360),
        sigma_orbit_radial_px=3.5,
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    result = RingAnnulusNav().navigate([feature], context)
    assert isinstance(result.diagnostics, RingAnnulusDiagnostics)
    assert result.diagnostics.sigma_orbit_radial_px == pytest.approx(3.5)


def test_annulus_orbit_channel_combines_features_by_vertex_share() -> None:
    """The helper concatenates normals and vertex-weights the per-feature sigmas."""

    feat_a = _make_annulus_feature(
        'A',
        extfov_shape=(80, 80),
        image_center_vu=(40.0, 40.0),
        inner_radius=10.0,
        outer_radius=18.0,
        orbit_normals_vu=np.tile(np.array([1.0, 0.0]), (30, 1)),
        sigma_orbit_radial_px=2.0,
    )
    feat_b = _make_annulus_feature(
        'B',
        extfov_shape=(80, 80),
        image_center_vu=(40.0, 40.0),
        inner_radius=10.0,
        outer_radius=18.0,
        orbit_normals_vu=np.tile(np.array([1.0, 0.0]), (10, 1)),
        sigma_orbit_radial_px=6.0,
    )
    _g, sigma = _annulus_orbit_channel([feat_a, feat_b])
    # (30 * 2 + 10 * 6) / 40 = 3.0
    assert sigma == pytest.approx(3.0)


def test_annulus_orbit_channel_empty_when_no_sigma() -> None:
    """A feature with normals but no declared sigma contributes nothing."""

    feat = _make_annulus_feature(
        'A',
        extfov_shape=(80, 80),
        image_center_vu=(40.0, 40.0),
        inner_radius=10.0,
        outer_radius=18.0,
        orbit_normals_vu=np.tile(np.array([1.0, 0.0]), (30, 1)),
        sigma_orbit_radial_px=0.0,
    )
    g, sigma = _annulus_orbit_channel([feat])
    assert sigma == 0.0
    assert float(np.linalg.norm(g)) == 0.0


def test_ring_annulus_camera_floor_replaces_tuning_floor(
    make_nav_context: NavContextFactory,
) -> None:
    """A camera floor on the context replaces the tuning floor rather than adding to it."""
    shape = (180, 180)
    image_center = (90.0, 90.0)
    image = _render_annulus_image(shape, image_center, 14.0, 28.0)
    feature = _make_annulus_feature(
        'SATURN',
        extfov_shape=shape,
        image_center_vu=image_center,
        inner_radius=14.0,
        outer_radius=28.0,
        planted_offset_vu=(2.0, -3.0),
    )
    context = make_nav_context(image, extfov_margin_vu=(16, 16))
    camera_context = dataclasses.replace(context, model_error_floor_px={'RingAnnulusNav': 1.5})
    bare = RingAnnulusNav()
    bare._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.0
    )
    tuned = RingAnnulusNav()
    tuned._cov_tuning = NCCCovarianceTuning(
        localization_uncertainty_scale=0.0, model_error_size_frac=0.0, model_error_floor_px=0.5
    )
    cov_bare = bare.navigate([feature], context).covariance_px2
    cov_camera = tuned.navigate([feature], camera_context).covariance_px2
    for axis in (0, 1):
        assert cov_camera[axis, axis] == pytest.approx(cov_bare[axis, axis] + 1.5**2, rel=1e-9)
