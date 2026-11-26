"""Integration tests for ``NavModelRings`` exercising ``to_features`` /
``to_annotations`` end-to-end against pre-populated render results.

Driving ``create_model`` with a real ``oops.Backplane`` and the
catalog-loaded ``RingFeature.render`` is integration-test scope.
These tests instead build a NavModelRings, populate the
``_render_results`` list directly with fake edge masks, and verify
that the public-API methods emit features and annotations matching
the design's emission gates (``RING_EDGE`` vs ``RING_ANNULUS``,
straight-line flagging).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from tests.shims import FakeObs

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import RingAnnulusFlags, RingEdgeFlags
from spindoctor.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.rings.ring_feature import RingFeature
from spindoctor.nav_model.rings.ring_types import RingBaseOrbitMode, RingEdgeData, RingFeatureType


def _curved_edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """Paint a curved (parabolic) edge that is not flagged straight."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    us = np.arange(cols)
    # v = 30 + 8 * sin(pi * u / cols) — a wide arch with significant curvature
    vs = (30 + 8.0 * np.sin(np.pi * us / cols)).astype(int)
    for u, v in zip(us, vs, strict=True):
        if 0 <= v < rows:
            mask[v, u] = True
    return mask


def _straight_edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """Paint a horizontal one-pixel edge that is flagged straight."""
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    mask[rows // 2, :] = True
    return mask


def _ring_feature(*, key: str = 'colombo_gap', name: str | None = 'colombo') -> RingFeature:
    """Build a minimal RingFeature with one outer edge."""
    edge = RingEdgeData(
        base_orbit=RingBaseOrbitMode(a=77_870.0, ae=10.0, long_peri=0.0, rate_peri=0.0, rms=2.0),
        perturbations=(),
    )
    return RingFeature(
        key=key,
        name=name,
        feature_type=RingFeatureType.GAP,
        inner_edge=None,
        outer_edge=edge,
        start_date=None,
        end_date=None,
    )


def _build_rings(
    *,
    obs: FakeObs,
    edge_mask: np.ndarray,
    label: str = 'colombo:outer',
    edge_type: str = 'outer',
    uncertainty_km: float = 2.0,
    km_per_pixel_radial: float = 5.0,
    planet: str = 'SATURN',
    constituent_count: int = 1,
) -> NavModelRings:
    """Build a NavModelRings whose render-results list carries one edge."""
    model = NavModelRings(f'rings:{planet}', cast(Any, obs))
    rows, cols = obs.extdata_shape_vu
    feat = _ring_feature()
    model_img = edge_mask.astype(np.float64)
    model_mask = edge_mask.copy()
    edge_info = [(edge_mask, label, edge_type)] * constituent_count
    model._render_results = [(feat, model_img, model_mask, uncertainty_km, edge_info)]
    model._planet = planet
    model._km_per_pixel_radial = km_per_pixel_radial
    model._extfov_v_size = rows
    model._extfov_u_size = cols
    model._predicted_center_vu = (rows / 2.0, cols / 2.0)
    model._subject_range_km = 1.5e9
    return model


@pytest.fixture
def fake_obs() -> FakeObs:
    """Provide a small obs."""
    return FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(5, 5),
        closest_planet='SATURN',
    )


def test_to_features_emits_ring_edge_for_curved_polyline(fake_obs: FakeObs) -> None:
    """A clearly-curved polyline emits a RING_EDGE feature with the polyline payload."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    features = model.to_features(cast(Any, None))
    assert len(features) == 1
    feat = features[0]
    assert feat.feature_type is NavFeatureType.RING_EDGE
    assert isinstance(feat.geometry, RingEdgePolyline)
    assert isinstance(feat.flags, RingEdgeFlags)
    assert feat.flags.planet_name == 'SATURN'
    assert feat.flags.is_straight_line is False
    # Per-vertex sigma_radial_per_vertex_px = uncertainty_km / km_per_pixel_radial.
    assert feat.geometry.sigma_radial_per_vertex_px[0] == pytest.approx(2.0 / 5.0)


def test_to_features_emits_ring_edge_with_straight_line_flag(fake_obs: FakeObs) -> None:
    """A straight polyline emits a RING_EDGE with ``is_straight_line=True``."""
    model = _build_rings(obs=fake_obs, edge_mask=_straight_edge_mask((110, 110)))
    features = model.to_features(cast(Any, None))
    # A horizontal straight line has zero radial extent (mean normal is zero
    # because the discrete-mask normal-finder cannot pick a side), so
    # ``radial_extent_px <= RING_ANNULUS_MAX_RADIAL_PX`` and the polyline is
    # straight -> the gate falls through to ``_build_edge_feature`` with the
    # straight-line flag set.
    assert features[0].feature_type is NavFeatureType.RING_EDGE
    assert isinstance(features[0].flags, RingEdgeFlags)
    assert features[0].flags.is_straight_line is True


def test_to_features_emits_annulus_when_polyline_compresses_radially(
    fake_obs: FakeObs,
) -> None:
    """A polyline whose radial extent is below the threshold emits RING_ANNULUS.

    The mask carries a shallow parabolic arch over a narrow U range so the
    polyline is curved (deviation > FLAT_CURVATURE_THRESHOLD_PX) yet projects
    to a span of < RING_ANNULUS_MAX_RADIAL_PX along its mean normal.
    """
    rows, cols = 110, 110
    mask = np.zeros((rows, cols), dtype=bool)
    us = np.arange(40, 70)
    vs = (rows // 2 + 2.5 * np.sin(np.pi * (us - 40) / 30)).astype(int)
    for u, v in zip(us, vs, strict=True):
        mask[v, u] = True
    model = _build_rings(obs=fake_obs, edge_mask=mask, constituent_count=2)
    features = model.to_features(cast(Any, None))
    annulus_features = [f for f in features if f.feature_type is NavFeatureType.RING_ANNULUS]
    # Two annulus-eligible polylines collapse into ONE composite per planet.
    assert len(annulus_features) == 1
    annulus = annulus_features[0]
    assert isinstance(annulus.geometry, RingAnnulusGeometry)
    assert isinstance(annulus.flags, RingAnnulusFlags)
    assert annulus.flags.planet_name == 'SATURN'
    # The composite reports the total number of fused constituent edges.
    assert annulus.flags.constituent_edge_count == 2


def test_to_features_emits_annulus_when_kmpp_above_planet_threshold(
    fake_obs: FakeObs,
) -> None:
    """A high km/px scene triggers the system-level annulus gate.

    Even when the per-edge polyline's radial extent would otherwise
    classify as a RING_EDGE, the planet-specific kmpp threshold
    (``feature_emission.ring_annulus.planets.SATURN.kmpp_threshold = 25``)
    forces annulus emission for the entire ring system.
    """
    model = _build_rings(
        obs=fake_obs,
        edge_mask=_curved_edge_mask((110, 110)),
        km_per_pixel_radial=20000.0,
    )
    features = model.to_features(cast(Any, None))
    types = {f.feature_type for f in features}
    assert NavFeatureType.RING_ANNULUS in types
    assert NavFeatureType.RING_EDGE not in types


def test_to_features_collapses_multi_ring_input_into_one_annulus(
    fake_obs: FakeObs,
) -> None:
    """Multiple surviving rings under force_annulus produce ONE composite feature.

    Without the composite step each per-ring annulus would carry
    ``constituent_count=1`` and the reliability formula
    (``min(1, k/5) * 0.7 * sigmoid(extent/50 - 1)``) would gate every
    one of them out (0.14 < 0.30 default threshold).  The composite
    feature carries ``constituent_count = N`` so the formula scales
    with the number of rings and clears the gate on a real ring scene.
    """
    rows, cols = 110, 110
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))
    feat = _ring_feature()
    # Three distinct edge masks at different image rows — each a thin
    # 1-pixel ridge so each polyline qualifies for the per-polyline
    # radial-extent gate as well as the system-level km/px gate.
    masks = []
    for v_row in (40, 50, 60):
        m = np.zeros((rows, cols), dtype=bool)
        m[v_row, 30:80] = True
        masks.append(m)
    render_results = [
        (
            feat,
            m.astype(np.float64),
            m.copy(),
            1.0,
            [(m, f'ring_{i}', 'outer')],
        )
        for i, m in enumerate(masks)
    ]
    model._render_results = cast(Any, render_results)
    model._planet = 'SATURN'
    model._km_per_pixel_radial = 20000.0  # force_annulus
    model._extfov_v_size = rows
    model._extfov_u_size = cols
    model._predicted_center_vu = (rows / 2.0, cols / 2.0)
    model._subject_range_km = 1.5e9
    features = model.to_features(cast(Any, None))
    assert len(features) == 1
    annulus_features = [f for f in features if f.feature_type is NavFeatureType.RING_ANNULUS]
    assert len(annulus_features) == 1
    annulus = annulus_features[0]
    assert isinstance(annulus.flags, RingAnnulusFlags)
    assert annulus.flags.constituent_edge_count == 3
    # The template is a postage stamp local to the composite bbox
    # (rows 40-60 inclusive, columns 30-79 inclusive), carrying every
    # constituent ring's pixels at bbox-local coordinates.
    v_min, u_min, v_max, u_max = annulus.geometry.bbox_extfov_vu
    assert (v_min, u_min, v_max, u_max) == (40, 30, 61, 80)
    assert annulus.template_mask is not None
    assert annulus.template_mask.shape == (v_max - v_min, u_max - u_min)
    for m in masks:
        assert annulus.template_mask[m[v_min:v_max, u_min:u_max]].all()


def test_to_features_emits_edge_when_kmpp_below_planet_threshold(
    fake_obs: FakeObs,
) -> None:
    """Below the planet's km/px threshold the per-polyline gate decides.

    Mirrors :func:`test_to_features_emits_ring_edge_for_curved_polyline`
    but pins the km/px below the Saturn-specific threshold so the
    system-level gate explicitly does not fire.
    """
    model = _build_rings(
        obs=fake_obs,
        edge_mask=_curved_edge_mask((110, 110)),
        km_per_pixel_radial=10.0,
    )
    features = model.to_features(cast(Any, None))
    assert features[0].feature_type is NavFeatureType.RING_EDGE


@pytest.mark.parametrize(
    ('km_per_pixel_radial', 'expected_type', 'absent_type'),
    [
        pytest.param(
            10.0, NavFeatureType.RING_EDGE, NavFeatureType.RING_ANNULUS, id='below-threshold'
        ),
        pytest.param(
            25.0, NavFeatureType.RING_ANNULUS, NavFeatureType.RING_EDGE, id='at-threshold'
        ),
        pytest.param(
            100.0, NavFeatureType.RING_ANNULUS, NavFeatureType.RING_EDGE, id='above-threshold'
        ),
    ],
)
def test_to_features_routes_by_saturn_kmpp_threshold(
    fake_obs: FakeObs,
    km_per_pixel_radial: float,
    expected_type: NavFeatureType,
    absent_type: NavFeatureType,
) -> None:
    """Saturn routing by radial km/px: edge below 25, annulus at and above.

    The 25 km/px case pins the inclusive boundary: the system-level gate is
    ``km_per_pixel_radial >= kmpp_threshold``, so a scene exactly at the
    threshold routes to the annulus composite. 100 km/px sits well inside
    the annulus-routed regime, where the head-to-head measured the annulus
    fit wrong on zero accepted answers; a 10 km/px scene keeps emitting a
    RING_EDGE polyline for the distance-transform fit.

    Parameters:
        fake_obs: Injected small observation fixture.
        km_per_pixel_radial: Scene radial resolution under test.
        expected_type: Feature type the routing must emit at that scale.
        absent_type: Feature type the routing must not emit at that scale.
    """
    model = _build_rings(
        obs=fake_obs,
        edge_mask=_curved_edge_mask((110, 110)),
        km_per_pixel_radial=km_per_pixel_radial,
    )
    features = model.to_features(cast(Any, None))
    types = {f.feature_type for f in features}
    assert expected_type in types
    assert absent_type not in types


def test_to_features_skips_empty_edge_info_list(fake_obs: FakeObs) -> None:
    """A render result with no edges emits no features."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    feat = _ring_feature()
    model._render_results = [
        (feat, np.zeros((110, 110)), np.zeros((110, 110), dtype=bool), 1.0, []),
    ]
    assert model.to_features(cast(Any, None)) == []


def test_to_features_empty_when_no_render_results(fake_obs: FakeObs) -> None:
    """A NavModelRings with empty ``_render_results`` emits no features."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))
    assert model.to_features(cast(Any, None)) == []


def test_to_annotations_returns_empty_collection_when_no_render_results(
    fake_obs: FakeObs,
) -> None:
    """An empty NavModelRings emits an empty Annotations collection."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))
    out = model.to_annotations(cast(Any, None))
    assert len(out.annotations) == 0


def test_to_annotations_skips_empty_edge_info_list(fake_obs: FakeObs) -> None:
    """Render results without edges yield no annotations."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    feat = _ring_feature()
    model._render_results = [
        (feat, np.zeros((110, 110)), np.zeros((110, 110), dtype=bool), 1.0, []),
    ]
    out = model.to_annotations(cast(Any, None))
    assert len(out.annotations) == 0


def test_create_model_records_metadata_block(
    fake_obs: FakeObs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_model`` populates the metadata block via the wrapping logic."""
    model = NavModelRings('rings:SATURN', cast(Any, fake_obs))

    def _fake_render(self: NavModelRings) -> None:
        self._planet = 'SATURN'
        self._metadata['planet'] = 'SATURN'
        self._metadata['feature_count'] = 0

    monkeypatch.setattr(NavModelRings, '_render', _fake_render)
    model.create_model()
    assert model.metadata['planet'] == 'SATURN'
    assert 'start_time' in model.metadata
    assert 'end_time' in model.metadata
    assert model.metadata['elapsed_time_sec'] >= 0.0


def test_instances_for_obs_returns_no_models_without_extdata_shape() -> None:
    """``instances_for_obs`` is graceful for an obs lacking the ext-FOV surface."""

    class _BareObs:
        closest_planet = 'SATURN'

    instances = NavModelRings.instances_for_obs(cast(Any, _BareObs()))
    assert instances == []


def test_to_features_carries_catalog_rms_as_orbit_sigma(fake_obs: FakeObs) -> None:
    """A feature with a catalog rms carries it as the fully-correlated orbit sigma."""
    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)))
    features = model.to_features(cast(Any, None))
    assert isinstance(features[0].geometry, RingEdgePolyline)
    # sigma_orbit_radial_px = uncertainty_km / km_per_pixel_radial.
    assert features[0].geometry.sigma_orbit_radial_px == pytest.approx(2.0 / 5.0)


def test_to_features_rms_less_feature_uses_config_default_orbit_sigma(
    fake_obs: FakeObs,
) -> None:
    """Without a catalog rms the configured default orbit sigma applies."""
    from spindoctor.config import DEFAULT_CONFIG

    model = _build_rings(obs=fake_obs, edge_mask=_curved_edge_mask((110, 110)), uncertainty_km=0.0)
    features = model.to_features(cast(Any, None))
    assert isinstance(features[0].geometry, RingEdgePolyline)
    DEFAULT_CONFIG.read_config()
    default_km = DEFAULT_CONFIG.rings.default_orbit_radial_sigma_km
    assert default_km == pytest.approx(2.0)
    assert features[0].geometry.sigma_orbit_radial_px == pytest.approx(default_km / 5.0)
