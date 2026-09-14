"""Integration tests for ``NavModelBody`` exercising ``to_features`` /
``to_annotations`` end-to-end against pre-populated state.

Driving ``create_model`` with a real ``oops.Backplane(obs,
meshgrid=...)`` is integration-test scope (Part 9 / Part 10 image
library).  These tests instead build a NavModelBody, monkeypatch
``_render`` to populate the polyline samplers and silhouette masks
directly, then verify that the public-API methods emit features and
annotations matching the design's emission gates.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from tests.shims import FakeObs, bare_nav_context

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import (
    BodyBlobFlags,
    BodyDiscFlags,
    LimbArcFlags,
    TerminatorArcFlags,
)
from spindoctor.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    LimbPolyline,
    TerminatorPolyline,
)
from spindoctor.nav_model.nav_model_body import (
    BODY_DISC_MAX_OVERFLOW_FRACTION,
    BODY_DISC_MIN_VISIBLE_LIT_FRACTION,
    NavModelBody,
    _PolylineSampler,
)


def _make_polyline_sampler(*, n: int, km_per_pixel: float = 1.0) -> _PolylineSampler:
    """Build a ``_PolylineSampler`` carrying ``n`` vertices on a horizontal line."""
    vs = np.full(n, 30.0, dtype=np.float64)
    us = np.linspace(20.0, 20.0 + n - 1, n, dtype=np.float64)
    vertices = np.stack([vs, us], axis=1)
    normals = np.tile(np.array([0.0, 1.0]), (n, 1))
    incidence = np.zeros(n, dtype=np.float64)
    km = np.full(n, km_per_pixel, dtype=np.float64)
    return _PolylineSampler(
        vertices_vu=vertices,
        normals_vu=normals,
        incidence_rad=incidence,
        km_per_pixel=km,
        total_vertices=n,
    )


def _build_body(
    *,
    obs: FakeObs,
    body_name: str = 'MIMAS',
    diameter_px: float = 30.0,
    visible_lit_fraction: float = 0.8,
    overflow_fraction: float = 0.1,
    phase_factor: float = 0.5,
    km_per_pixel_at_limb: float = 1.0,
    limb_vertices: int = 40,
    terminator_vertices: int = 20,
) -> NavModelBody:
    """Build a NavModelBody with internal state populated for ``to_features``.

    The model bypasses ``_render`` and instead has its private state
    set directly so the test can exercise ``to_features`` /
    ``to_annotations`` against a deterministic body-shape posture.
    """
    inv = {
        'u_min_unclipped': 0,
        'u_max_unclipped': int(diameter_px),
        'v_min_unclipped': 0,
        'v_max_unclipped': int(diameter_px),
        'u_pixel_size': diameter_px,
        'v_pixel_size': diameter_px,
        'range': 1.0e6,
        'center_uv': (diameter_px / 2.0, diameter_px / 2.0),
    }
    obs.inventory_records = {body_name: inv}
    model = NavModelBody(f'body:{body_name}', cast(Any, obs), body_name, inventory=inv)
    rows, cols = obs.extdata_shape_vu
    body_mask = np.zeros((rows, cols), dtype=bool)
    center_v = rows // 2
    center_u = cols // 2
    radius = int(diameter_px // 2)
    vv, uu = np.meshgrid(
        np.arange(rows, dtype=np.float64),
        np.arange(cols, dtype=np.float64),
        indexing='ij',
    )
    body_mask[(vv - center_v) ** 2 + (uu - center_u) ** 2 <= radius * radius] = True
    model_img = body_mask.astype(np.float64)
    limb_mask = body_mask.copy()
    terminator_mask = np.zeros_like(body_mask)
    model._model_img = model_img
    model._body_mask = body_mask
    model._limb_mask = limb_mask
    model._terminator_mask = terminator_mask
    model._limb_sampler = _make_polyline_sampler(n=limb_vertices, km_per_pixel=km_per_pixel_at_limb)
    model._terminator_sampler = _make_polyline_sampler(
        n=terminator_vertices, km_per_pixel=km_per_pixel_at_limb
    )
    model._km_per_pixel_at_limb = km_per_pixel_at_limb
    model._predicted_diameter_px = diameter_px
    model._predicted_center_vu = (float(center_v), float(center_u))
    model._bbox_extfov_vu = (
        center_v - radius,
        center_u - radius,
        center_v + radius + 1,
        center_u + radius + 1,
    )
    model._subject_range_km = 1.0e6
    model._visible_lit_fraction = visible_lit_fraction
    # The helper's model_img renders the whole silhouette as lit, so the
    # lit-pixel count (which gates BODY_BLOB emission) is the mask area.
    model._lit_pixel_count = int(np.count_nonzero(body_mask))
    model._overflow_fraction = overflow_fraction
    model._phase_angle_factor = phase_factor
    return model


@pytest.fixture
def fake_obs() -> FakeObs:
    """Provide a small obs with non-zero extfov margin and a configured PSF."""
    return FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(5, 5),
        closest_planet='SATURN',
    )


def test_to_features_emits_limb_arc_when_uncertainty_is_low(fake_obs: FakeObs) -> None:
    """A well-resolved body emits a LIMB_ARC with the expected polyline payload."""
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=10.0)  # high resolution
    features = model.to_features(bare_nav_context(fake_obs))
    by_type = {f.feature_type: f for f in features}
    assert NavFeatureType.LIMB_ARC in by_type
    limb = by_type[NavFeatureType.LIMB_ARC]
    assert isinstance(limb.geometry, LimbPolyline)
    assert isinstance(limb.flags, LimbArcFlags)
    assert limb.flags.body_name == 'MIMAS'
    assert limb.geometry.vertices_vu.shape == (40, 2)
    assert limb.geometry.sigma_normal_per_vertex_px.shape == (40,)


def test_to_features_emits_blob_instead_of_limb_when_uncertainty_high(
    fake_obs: FakeObs,
) -> None:
    """A poorly-resolved body emits BODY_BLOB instead of LIMB_ARC."""
    # km_per_pixel_at_limb=1000 -> limb_uncertainty_px = 1.0/1000 = 0.001 (low),
    # so to force the high-uncertainty branch we lower the resolution.
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=0.001)
    features = model.to_features(bare_nav_context(fake_obs))
    types = {f.feature_type for f in features}
    assert NavFeatureType.LIMB_ARC not in types
    assert NavFeatureType.BODY_BLOB in types
    blob = next(f for f in features if f.feature_type is NavFeatureType.BODY_BLOB)
    assert isinstance(blob.geometry, BodyBlobGeometry)
    assert isinstance(blob.flags, BodyBlobFlags)
    assert blob.flags.predicted_diameter_px == pytest.approx(30.0)


def test_to_features_emits_blob_when_limb_arc_too_short(fake_obs: FakeObs) -> None:
    """A limb polyline below the vertex floor falls through to BODY_BLOB.

    The per-vertex uncertainty is excellent (high resolution), but the arc
    is shorter than ``BodyLimbNav``'s feasibility floor -- the distant
    small-body posture where the uncertainty test alone would emit a
    guaranteed-infeasible LIMB_ARC and starve the body of its blob.
    """
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=10.0, limb_vertices=9)
    features = model.to_features(bare_nav_context(fake_obs))
    types = {f.feature_type for f in features}
    assert NavFeatureType.LIMB_ARC not in types
    assert NavFeatureType.BODY_BLOB in types


def test_to_features_emits_disc_alongside_limb_when_visibility_high(
    fake_obs: FakeObs,
) -> None:
    """BODY_DISC is emitted when LIMB_ARC fired and visibility gates pass."""
    model = _build_body(
        obs=fake_obs,
        km_per_pixel_at_limb=10.0,
        visible_lit_fraction=BODY_DISC_MIN_VISIBLE_LIT_FRACTION + 0.1,
        overflow_fraction=BODY_DISC_MAX_OVERFLOW_FRACTION - 0.1,
    )
    features = model.to_features(bare_nav_context(fake_obs))
    types = {f.feature_type for f in features}
    assert NavFeatureType.BODY_DISC in types
    disc = next(f for f in features if f.feature_type is NavFeatureType.BODY_DISC)
    assert isinstance(disc.geometry, BodyDiscGeometry)
    assert isinstance(disc.flags, BodyDiscFlags)
    assert disc.template_img is not None
    assert disc.template_mask is not None


def test_to_features_co_emits_witness_blob_with_limb_and_disc(fake_obs: FakeObs) -> None:
    """A resolved body co-emits BODY_BLOB alongside LIMB_ARC and BODY_DISC.

    This is the reachability proof for the ensemble body-witness veto: on the
    real (SPICE-backed) NavModelBody a single resolved body must yield the
    geometric limb / disc AND the pose-free witness blob together, so the veto
    has an independent cross-check to read on real frames rather than only on
    the co-emitting simulated model.
    """
    model = _build_body(
        obs=fake_obs,
        km_per_pixel_at_limb=10.0,
        visible_lit_fraction=BODY_DISC_MIN_VISIBLE_LIT_FRACTION + 0.1,
        overflow_fraction=BODY_DISC_MAX_OVERFLOW_FRACTION - 0.1,
    )
    types = {f.feature_type for f in model.to_features(bare_nav_context(fake_obs))}
    assert NavFeatureType.LIMB_ARC in types
    assert NavFeatureType.BODY_DISC in types
    assert NavFeatureType.BODY_BLOB in types


def test_to_features_witness_blob_is_singular_with_limb(fake_obs: FakeObs) -> None:
    """Exactly one BODY_BLOB is emitted alongside the limb (no duplicate feature)."""
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=10.0)
    blobs = [
        f
        for f in model.to_features(bare_nav_context(fake_obs))
        if f.feature_type is NavFeatureType.BODY_BLOB
    ]
    assert len(blobs) == 1


def test_to_features_skips_disc_when_overflow_high(fake_obs: FakeObs) -> None:
    """A body with too much overflow does not emit BODY_DISC."""
    model = _build_body(
        obs=fake_obs,
        km_per_pixel_at_limb=10.0,
        visible_lit_fraction=0.9,
        overflow_fraction=BODY_DISC_MAX_OVERFLOW_FRACTION + 0.1,
    )
    types = {f.feature_type for f in model.to_features(bare_nav_context(fake_obs))}
    assert NavFeatureType.BODY_DISC not in types


def test_to_features_emits_terminator_when_phase_factor_high(
    fake_obs: FakeObs,
) -> None:
    """A body with high phase factor emits TERMINATOR_ARC."""
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=10.0, phase_factor=0.8)
    features = model.to_features(bare_nav_context(fake_obs))
    types = {f.feature_type for f in features}
    assert NavFeatureType.TERMINATOR_ARC in types
    term = next(f for f in features if f.feature_type is NavFeatureType.TERMINATOR_ARC)
    assert isinstance(term.geometry, TerminatorPolyline)
    assert isinstance(term.flags, TerminatorArcFlags)


def test_to_features_skips_terminator_at_zero_phase(fake_obs: FakeObs) -> None:
    """A sub-solar-illuminated body emits no TERMINATOR_ARC."""
    model = _build_body(obs=fake_obs, km_per_pixel_at_limb=10.0, phase_factor=0.0)
    types = {f.feature_type for f in model.to_features(bare_nav_context(fake_obs))}
    assert NavFeatureType.TERMINATOR_ARC not in types


def test_to_features_skips_terminator_when_polyline_too_short(fake_obs: FakeObs) -> None:
    """A terminator polyline with fewer than the minimum vertices is suppressed."""
    model = _build_body(
        obs=fake_obs,
        km_per_pixel_at_limb=10.0,
        terminator_vertices=4,  # below TERMINATOR_MIN_VERTICES (8)
    )
    types = {f.feature_type for f in model.to_features(bare_nav_context(fake_obs))}
    assert NavFeatureType.TERMINATOR_ARC not in types


def test_to_features_limb_uncertainty_at_threshold(fake_obs: FakeObs) -> None:
    """The LIMB_ARC vs BODY_BLOB switch is correctly placed around the threshold.

    With ``LIMB_ARC_MAX_UNCERTAINTY_PX = 3.0`` and Saturn's
    ``ellipsoid_rms_residual_km = 50`` (gas-giant default), the threshold
    sits at ``km_per_pixel_at_limb = 50 / 3.0 ~= 16.67``.  This test
    probes either side of that threshold:

    * ``km_per_pixel_at_limb = 17.0`` (uncertainty ~ 2.94 px, below
      threshold) — ``saturn_model.to_features`` must emit a
      ``LIMB_ARC``.
    * ``km_per_pixel_at_limb = 15.0`` (uncertainty ~ 3.33 px, above
      threshold) — the limb branch must drop and the technique falls
      through to ``BODY_BLOB``.
    """
    # km_per_pixel_at_limb=17 yields uncertainty ~ 2.94 < 3.0 -> LIMB_ARC.
    saturn_model = _build_body(obs=fake_obs, body_name='SATURN', km_per_pixel_at_limb=17.0)
    features = saturn_model.to_features(bare_nav_context(fake_obs))
    assert any(f.feature_type is NavFeatureType.LIMB_ARC for f in features)
    # km_per_pixel_at_limb=15 yields uncertainty ~ 3.33 > 3.0 -> blob branch.
    saturn_blob = _build_body(obs=fake_obs, body_name='SATURN', km_per_pixel_at_limb=15.0)
    features_blob = saturn_blob.to_features(bare_nav_context(fake_obs))
    types = {f.feature_type for f in features_blob}
    assert NavFeatureType.LIMB_ARC not in types
    # Uncertainty above threshold and diameter above the body's blob-min
    # threshold -> blob.
    assert NavFeatureType.BODY_BLOB in types


def test_create_model_metadata_records_body_name_and_phase(
    fake_obs: FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_model`` populates the metadata block via the wrapping logic."""
    model = _build_body(obs=fake_obs)

    def _fake_render(self: NavModelBody) -> None:
        # Only the metadata fields populated by the wrapping logic matter
        # for this test; ``to_features`` / ``to_annotations`` are tested
        # separately above.
        self._metadata['body_name'] = self._body_name
        self._metadata['phase_angle_deg'] = 30.0

    monkeypatch.setattr(NavModelBody, '_render', _fake_render)
    model.create_model()
    assert model.metadata['body_name'] == 'MIMAS'
    assert 'start_time' in model.metadata
    assert 'end_time' in model.metadata
    assert model.metadata['elapsed_time_sec'] >= 0.0


def test_to_annotations_returns_empty_when_state_unset(fake_obs: FakeObs) -> None:
    """A NavModelBody with no rendered masks returns an empty Annotations."""
    inv = {
        'u_min_unclipped': 0,
        'u_max_unclipped': 30,
        'v_min_unclipped': 0,
        'v_max_unclipped': 30,
        'u_pixel_size': 30.0,
        'v_pixel_size': 30.0,
        'range': 1.0e6,
        'center_uv': (15.0, 15.0),
    }
    fake_obs.inventory_records = {'MIMAS': inv}
    model = NavModelBody('body:MIMAS', cast(Any, fake_obs), 'MIMAS', inventory=inv)
    annotations = model.to_annotations(cast(Any, None))
    assert len(annotations.annotations) == 0
