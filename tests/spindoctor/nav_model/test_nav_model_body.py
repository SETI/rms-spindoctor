"""Tests for ``spindoctor.nav_model.nav_model_body`` helpers and emission gates.

The full ``NavModelBody.create_model`` path requires a live ``oops``
backplane and so is exercised by integration tests against the image
library.  These unit tests cover the pure helper functions
(``_incidence_factor_array``, ``_sigma_normal_per_vertex``, the
reliability sigmoids) plus the class-level constants.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from spindoctor.feature.constants import MAX_INCIDENCE_FACTOR_CAP
from spindoctor.nav_model.body_shape import DEFAULT_BODY_SHAPE
from spindoctor.nav_model.nav_model_body import (
    BODY_DISC_MAX_OVERFLOW_FRACTION,
    BODY_DISC_MIN_VISIBLE_LIT_FRACTION,
    BODY_POSITION_SLOP_FRAC,
    LIMB_ARC_MAX_UNCERTAINTY_PX,
    TERMINATOR_MIN_PHASE_FACTOR,
    TERMINATOR_MIN_VERTICES,
    _build_polyline_sampler,
    _disc_reliability,
    _incidence_factor_array,
    _PolylineSampler,
    _sigma_normal_per_vertex,
    _visible_arc_fraction,
    limb_reliability,
    terminator_reliability,
)
from spindoctor.nav_model.nav_model_body_base import _blob_reliability, _sigmoid


def test_constants_have_design_values() -> None:
    """Module-level constants match the design's defaults."""
    assert pytest.approx(0.05) == BODY_POSITION_SLOP_FRAC
    assert pytest.approx(3.0) == LIMB_ARC_MAX_UNCERTAINTY_PX
    assert pytest.approx(0.4) == BODY_DISC_MIN_VISIBLE_LIT_FRACTION
    assert pytest.approx(0.3) == BODY_DISC_MAX_OVERFLOW_FRACTION
    assert TERMINATOR_MIN_VERTICES == 8
    assert pytest.approx(0.05) == TERMINATOR_MIN_PHASE_FACTOR


def test_incidence_factor_zero_at_subsolar() -> None:
    """At i=0 the incidence factor is zero (no softening)."""
    arr = _incidence_factor_array(np.array([0.0]))
    assert arr[0] == pytest.approx(0.0, abs=1e-12)


def test_incidence_factor_one_at_60deg() -> None:
    """At i=60 the incidence factor is 1 (cos=0.5; 1/0.5 - 1 = 1)."""
    arr = _incidence_factor_array(np.array([math.radians(60.0)]))
    assert arr[0] == pytest.approx(1.0, abs=1e-12)


def test_incidence_factor_saturates_above_80deg() -> None:
    """Above 80 deg the formula uses the i=80 value (capped by the angle cap)."""
    arr_80 = _incidence_factor_array(np.array([math.radians(80.0)]))
    arr_89 = _incidence_factor_array(np.array([math.radians(89.0)]))
    expected_at_80 = 1.0 / math.cos(math.radians(80.0)) - 1.0
    assert arr_80[0] == pytest.approx(expected_at_80, rel=1e-12)
    assert arr_89[0] == pytest.approx(arr_80[0], rel=1e-12)
    # The cap constant bounds the factor from above (numerically slack).
    assert arr_89[0] <= MAX_INCIDENCE_FACTOR_CAP + 1e-9


def test_incidence_factor_array_returns_float64() -> None:
    """The output dtype is float64 even when inputs are integer-typed."""
    arr = _incidence_factor_array(np.array([0.0, 1.0]))
    assert arr.dtype == np.float64


def _make_sampler(*, n: int, incidence_deg: float, km_per_pixel: float) -> _PolylineSampler:
    """Build a ``_PolylineSampler`` with constant-incidence vertices."""
    vertices = np.zeros((n, 2), dtype=np.float64)
    normals = np.zeros((n, 2), dtype=np.float64)
    incidence = np.full(n, math.radians(incidence_deg), dtype=np.float64)
    km = np.full(n, km_per_pixel, dtype=np.float64)
    return _PolylineSampler(
        vertices_vu=vertices,
        normals_vu=normals,
        incidence_rad=incidence,
        km_per_pixel=km,
        total_vertices=n,
    )


def test_sigma_normal_uses_quadrature_sum() -> None:
    """``sigma_normal_per_vertex`` matches the design's quadrature formula."""
    shape = DEFAULT_BODY_SHAPE
    sampler = _make_sampler(n=4, incidence_deg=0.0, km_per_pixel=10.0)
    sigma_px = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=False
    )
    expected_km = math.sqrt(
        shape.ellipsoid_rms_residual_km**2
        + shape.crater_scale_km**2
        + 0.0  # incidence_factor=0 at i=0
        + shape.spice_orbital_residual_km**2
    )
    expected_px = expected_km / 10.0
    assert sigma_px[0] == pytest.approx(expected_px, rel=1e-12)


def test_sigma_normal_albedo_term_increases_when_enabled() -> None:
    """``include_albedo=True`` adds the albedo + photometric quadrature terms."""
    shape = DEFAULT_BODY_SHAPE
    sampler = _make_sampler(n=2, incidence_deg=30.0, km_per_pixel=5.0)
    no_albedo = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=False
    )
    with_albedo = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=1.0, include_albedo=True
    )
    assert with_albedo[0] > no_albedo[0]


def test_visible_arc_fraction_reports_unity_when_vertices_present() -> None:
    """A non-empty polyline reports ``visible_arc_fraction = 1.0``."""
    sampler = _make_sampler(n=5, incidence_deg=0.0, km_per_pixel=1.0)
    assert _visible_arc_fraction(sampler) == 1.0


def test_visible_arc_fraction_reports_survivors_over_total() -> None:
    """CODE-NAV-MODEL-006: the fraction is survivors / pre-drop total, not 1.0."""
    sampler = _PolylineSampler(
        vertices_vu=np.zeros((3, 2), dtype=np.float64),
        normals_vu=np.zeros((3, 2), dtype=np.float64),
        incidence_rad=np.zeros(3, dtype=np.float64),
        km_per_pixel=np.ones(3, dtype=np.float64),
        total_vertices=12,
    )
    assert _visible_arc_fraction(sampler) == pytest.approx(0.25)


def test_visible_arc_fraction_reports_zero_when_empty() -> None:
    """An empty polyline reports zero arc fraction."""
    sampler = _PolylineSampler(
        vertices_vu=np.empty((0, 2), dtype=np.float64),
        normals_vu=np.empty((0, 2), dtype=np.float64),
        incidence_rad=np.empty(0, dtype=np.float64),
        km_per_pixel=np.empty(0, dtype=np.float64),
        total_vertices=0,
    )
    assert _visible_arc_fraction(sampler) == 0.0


def test_sigmoid_is_monotone() -> None:
    """The sigmoid is strictly increasing across a wide range of inputs."""
    samples = [_sigmoid(x) for x in (-5.0, -1.0, 0.0, 1.0, 5.0)]
    for prev, nxt in itertools.pairwise(samples):
        assert nxt > prev


def test_sigmoid_at_zero() -> None:
    """``_sigmoid(0)`` equals 0.5 — sanity check the symmetry."""
    assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-12)


def test_limb_reliability_increases_with_visible_arc_fraction() -> None:
    """Reliability is monotone in ``visible_arc_fraction`` for fixed arc length."""
    low = limb_reliability(visible_arc_fraction=0.2, visible_arc_px=20.0)
    high = limb_reliability(visible_arc_fraction=0.9, visible_arc_px=20.0)
    assert high > low


def test_limb_reliability_increases_with_arc_length() -> None:
    """Longer arcs score higher (the ``visible_arc_px`` sigmoid)."""
    short = limb_reliability(visible_arc_fraction=0.9, visible_arc_px=5.0)
    long = limb_reliability(visible_arc_fraction=0.9, visible_arc_px=200.0)
    assert long > short


def test_limb_reliability_passes_gate_for_fully_lit_geometry() -> None:
    """A fully visible, well-sampled limb scores well above the 0.30 gate.

    Per-vertex softness already lives in ``_sigma_normal_per_vertex``;
    the reliability score is a feature-existence gate, not a precision
    estimate, so a textbook-good limb (Dione at low phase) must clear
    it.
    """
    score = limb_reliability(visible_arc_fraction=1.0, visible_arc_px=300.0)
    assert score > 0.5


def test_terminator_reliability_zero_at_zero_phase() -> None:
    """Sub-solar-illuminated images produce zero terminator reliability."""
    out = terminator_reliability(visible_arc_fraction=1.0, albedo_variation=0.0, phase_factor=0.0)
    assert out == 0.0


def test_disc_reliability_increases_with_diameter() -> None:
    """Reliability is monotone in body diameter for fixed visibility."""
    low = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=0.0, diameter_px=10.0)
    high = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=0.0, diameter_px=200.0)
    assert high > low


def test_disc_reliability_zero_when_overflow_complete() -> None:
    """A fully off-frame disc has zero reliability."""
    out = _disc_reliability(visible_lit_fraction=1.0, overflow_fraction=1.0, diameter_px=100.0)
    assert out == 0.0


def test_blob_reliability_capped_at_0_4() -> None:
    """Blob reliability cannot exceed the 0.4 design cap."""
    out = _blob_reliability(snr=1e6, diameter_px=1e6)
    assert out <= 0.4 + 1e-12


def test_blob_reliability_gated_without_detection() -> None:
    """Zero detection SNR sits below the 0.20 gate at any body size."""
    out = _blob_reliability(snr=0.0, diameter_px=1e6)
    assert out < 0.2


def test_blob_reliability_admits_small_detected_body() -> None:
    """A clearly detected 10 px body clears the 0.20 reliability gate.

    This is the issue #209 regression: the blob's design regime
    (below-resolution bodies under ~15 px) must be admitted when the
    body is genuinely bright against the image noise.
    """
    out = _blob_reliability(snr=10.0, diameter_px=10.0)
    assert out >= 0.2


def test_blob_reliability_monotone_in_snr() -> None:
    """Reliability increases with detection SNR at fixed size."""
    low = _blob_reliability(snr=1.0, diameter_px=12.0)
    high = _blob_reliability(snr=8.0, diameter_px=12.0)
    assert high > low


def _synthetic_disc_masks(
    *, size: int, center: tuple[float, float], radius: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Build a filled-disc silhouette mask plus its 1-px limb ridge.

    Returns ``(silhouette_mask, limb_ridge_mask, center_vu)`` where the
    silhouette is True inside the disc and the ridge is the inner ring of
    silhouette pixels adjacent to space.
    """
    vv, uu = np.indices((size, size), dtype=np.float64)
    dist = np.hypot(vv - center[0], uu - center[1])
    silhouette = dist <= radius
    # 1-px ridge: silhouette pixels with at least one space (False) neighbor.
    space = ~silhouette
    ridge = silhouette & (
        np.roll(space, 1, axis=0)
        | np.roll(space, -1, axis=0)
        | np.roll(space, 1, axis=1)
        | np.roll(space, -1, axis=1)
    )
    return silhouette, ridge, center


def test_polyline_normal_points_outward_for_lit_disc() -> None:
    """The limb normal points away from the disc center at >=95% of vertices.

    Constructs a synthetic filled disc, extracts its limb ridge, and
    checks ``dot(normal_i, vertex_i - center) > 0`` for the vast majority
    of vertices.  This guards against the ridge-orientation sign bug
    where the per-vertex normal was derived from the 1-px ridge rather
    than from the body silhouette.
    """
    size = 41
    center = (20.0, 20.0)
    radius = 14.0
    silhouette, ridge, center_vu = _synthetic_disc_masks(size=size, center=center, radius=radius)
    sampler = _build_polyline_sampler(
        local_mask=ridge,
        region_mask=silhouette,
        incidence_local=np.zeros((size, size), dtype=np.float64),
        km_per_pixel_local=np.ones((size, size), dtype=np.float64),
        ext_v0=0,
        ext_u0=0,
    )
    assert sampler.vertices_vu.shape[0] > 0
    radial = sampler.vertices_vu - np.array(center_vu, dtype=np.float64)
    dots = np.sum(sampler.normals_vu * radial, axis=1)
    outward_fraction = float(np.count_nonzero(dots > 0)) / dots.shape[0]
    assert outward_fraction >= 0.95


def test_polyline_sampler_drops_zero_resolution_vertices() -> None:
    """CODE-NAV-MODEL-001: ridge vertices with km/px <= 0 are dropped.

    Off the resolved body the resolution backplane is masked / filled with
    0.0; such vertices must be excluded at construction so they cannot reach
    the LM fit with a silently-floored fallback sigma.
    """
    size = 41
    silhouette, ridge, _ = _synthetic_disc_masks(size=size, center=(20.0, 20.0), radius=14.0)
    full = _build_polyline_sampler(
        local_mask=ridge,
        region_mask=silhouette,
        incidence_local=np.zeros((size, size), dtype=np.float64),
        km_per_pixel_local=np.ones((size, size), dtype=np.float64),
        ext_v0=0,
        ext_u0=0,
    )
    km = np.ones((size, size), dtype=np.float64)
    km[:, : size // 2] = 0.0  # left half is unresolved (km/px == 0)
    partial = _build_polyline_sampler(
        local_mask=ridge,
        region_mask=silhouette,
        incidence_local=np.zeros((size, size), dtype=np.float64),
        km_per_pixel_local=km,
        ext_v0=0,
        ext_u0=0,
    )
    assert partial.vertices_vu.shape[0] < full.vertices_vu.shape[0]
    assert partial.vertices_vu.shape[0] > 0
    assert np.all(partial.km_per_pixel > 0.0)


def test_polyline_normal_is_unit_length() -> None:
    """Each computed normal is unit length (or zero for isolated vertices)."""
    size = 31
    silhouette, ridge, _ = _synthetic_disc_masks(size=size, center=(15.0, 15.0), radius=10.0)
    sampler = _build_polyline_sampler(
        local_mask=ridge,
        region_mask=silhouette,
        incidence_local=np.zeros((size, size), dtype=np.float64),
        km_per_pixel_local=np.ones((size, size), dtype=np.float64),
        ext_v0=0,
        ext_u0=0,
    )
    lengths = np.hypot(sampler.normals_vu[:, 0], sampler.normals_vu[:, 1])
    assert np.all(lengths == pytest.approx(1.0, abs=1e-12))
