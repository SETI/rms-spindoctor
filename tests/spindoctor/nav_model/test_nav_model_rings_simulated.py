"""Simulated-rings NavModel feature emission.

``NavModelRingsSimulated`` predicts a navigable ring_system ringlet and
emits a ``RING_ANNULUS`` (for the correlation path) plus one ``RING_EDGE``
per catalog edge -- a radial-normal polyline ``RingEdgeNav`` fits.  These
tests cover that both feature kinds are emitted and that the edge polyline
carries outward radial unit normals.
"""

from typing import Any, cast

import numpy as np
import pytest

from spindoctor.feature.composition import compose_template_features
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_model.sim_ring import predict_ring_feature
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim

_SIZE = 220


def _feature_params() -> dict[str, Any]:
    """A centered navigable ringlet with curved inner and outer edges."""
    return {
        'name': 'SATURN',
        'kind': 'ringlet',
        'tau': 2.0,
        'width_px': 25.0,
        'navigable': True,
        'orbit': {'a_px': 60.0, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0},
    }


def _ring_system() -> dict[str, Any]:
    """A face-on ring_system block carrying the ringlet."""
    return {
        'geometry': {
            'center_v': _SIZE / 2.0,
            'center_u': _SIZE / 2.0,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [_feature_params()],
    }


def _obs() -> ObsSim:
    """A coiss_nac sim obs carrying the ringlet."""
    return ObsSim.from_file(
        '/tmp/ring_sim.json',
        sim_params={
            'size_v': _SIZE,
            'size_u': _SIZE,
            'instrument': 'coiss_nac',
            'ring_system': _ring_system(),
        },
    )


def _features() -> list[Any]:
    """Build the ring model and return its emitted features."""
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', _feature_params(), _ring_system())
    model.create_model()
    return model.to_features(cast(NavContext, None))


def test_emits_ring_annulus() -> None:
    """The correlation-path RING_ANNULUS feature is still emitted."""
    types = [f.feature_type.name for f in _features()]
    assert 'RING_ANNULUS' in types


def test_emits_ring_edge_per_edge() -> None:
    """One RING_EDGE is emitted for each catalog edge (inner + outer)."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    assert len(edges) == 2


def test_ring_edge_is_curved() -> None:
    """A circular ring arc is flagged as not straight (a 2-DoF constraint)."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    assert geometry.is_straight_line is False


def test_ring_edge_normals_are_unit_radial() -> None:
    """The edge polyline carries unit-length normals."""
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    norms = np.hypot(geometry.normals_vu[:, 0], geometry.normals_vu[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_ring_edge_normals_point_outward() -> None:
    """Each normal points away from the ring center (increasing radius)."""
    obs = _obs()
    center_v = _SIZE / 2.0 + obs.extfov_margin_v
    center_u = _SIZE / 2.0 + obs.extfov_margin_u
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    geometry = edges[0].geometry
    assert isinstance(geometry, RingEdgePolyline)
    radial = geometry.vertices_vu - np.array([[center_v, center_u]])
    dots = np.sum(radial * geometry.normals_vu, axis=1)
    assert bool((dots > 0.0).all())


def test_ring_annulus_template_is_bbox_local_postage_stamp() -> None:
    """The RING_ANNULUS template payload is exactly the size of its bbox.

    The compose_template_features convention anchors the template's (0, 0)
    at the bbox origin, so an ext-FOV-sized template with an interior bbox
    paints the ring displaced by the bbox origin (the annulus NCC then
    recovers an offset wrong by exactly the ext-FOV margin).
    """
    annuli = [f for f in _features() if f.feature_type.name == 'RING_ANNULUS']
    assert len(annuli) == 1
    annulus = annuli[0]
    bbox = annulus.geometry.bbox_extfov_vu
    expected_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    assert annulus.template_img is not None
    assert annulus.template_mask is not None
    assert annulus.template_img.shape == expected_shape
    assert annulus.template_mask.shape == expected_shape
    assert annulus.template_mask.any()


def test_ring_annulus_template_paints_at_ring_radius() -> None:
    """Composing the annulus paints pixels at the ring's radius from its center.

    Re-composes the emitted feature into an ext-FOV canvas (the same path
    RingAnnulusNav uses) and checks every painted pixel sits between the
    ringlet's inner and outer radii from the predicted center -- the
    placement invariant the displaced-template defect broke.

    The bound is exact rather than slack.  A pixel is painted when its
    anti-aliasing shade exceeds zero, and that shade ramps over one pixel
    centered on the edge, so a painted pixel's center lies at most half a pixel
    outside the 60-to-85 px band.  Measuring that from array indices only
    gives the same number the renderer used because the predicted center is
    pixel centric too, the system those indices are in; a center carried in
    pixel corner coordinates would put the whole measurement half a pixel off
    on each axis and paint pixels 1.2 px outside the band.
    """
    obs = _obs()
    annuli = [f for f in _features() if f.feature_type.name == 'RING_ANNULUS']
    annulus = annuli[0]
    extfov_shape = (
        _SIZE + 2 * obs.extfov_margin_v,
        _SIZE + 2 * obs.extfov_margin_u,
    )
    _, mask = compose_template_features([annulus], extfov_shape)
    assert mask.any()
    center_v, center_u = annulus.geometry.predicted_center_vu
    vs, us = np.where(mask)
    radii = np.hypot(vs - center_v, us - center_u)
    # Inner edge at 60 px, outer at 85 px, plus the half pixel the
    # anti-aliasing ramp reaches past each of them.
    assert radii.min() >= 60.0 - 0.5
    assert radii.max() <= 85.0 + 0.5


def test_ring_edge_carries_declared_orbit_sigma() -> None:
    """A declared orbit uncertainty lands on the emitted geometry unfloored.

    The semimajor-axis and radial-amplitude sigmas combine in QUADRATURE:
    the eccentric term enters the radial displacement as
    ``-delta(ae) cos(theta - peri)``, so the max-over-longitude 1-sigma of
    the two independent terms is their quadrature sum, not their arithmetic
    sum (which would overstate the envelope by up to sqrt(2) and double the
    variance at equal terms).
    """
    params = _feature_params()
    params['declared_orbit_sigma'] = {'sigma_a_px': 2.0, 'sigma_ae_px': 0.5}
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', params, _ring_system())
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    edges = [f for f in features if f.feature_type.name == 'RING_EDGE']
    assert edges
    expected = float(np.hypot(2.0, 0.5))
    for edge in edges:
        geometry = edge.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert geometry.sigma_orbit_radial_px == pytest.approx(expected, rel=1e-5)


def test_ring_edge_orbit_sigma_zero_when_undeclared() -> None:
    """Without declared_orbit_sigma the systematic channel carries nothing.

    The per-vertex radial sigma keeps its one-pixel sampling floor, but the
    fully-correlated orbit sigma must stay exactly zero -- the floor is a
    statistical scale, not a coherent displacement bound.
    """
    edges = [f for f in _features() if f.feature_type.name == 'RING_EDGE']
    assert edges
    for edge in edges:
        geometry = edge.geometry
        assert isinstance(geometry, RingEdgePolyline)
        assert geometry.sigma_orbit_radial_px == 0.0
        assert float(geometry.sigma_radial_per_vertex_px.min()) == 1.0


def _eccentric_edges(declared: dict[str, float | None]) -> list[RingEdgePolyline]:
    """The edge geometries of an eccentric ringlet (ae 6 px) declaring ``declared``."""
    params = _feature_params()
    params['orbit'] = {**params['orbit'], 'ae_px': 6.0, 'long_peri_deg': 30.0}
    params['declared_orbit_sigma'] = declared
    system = _ring_system()
    system['features'] = [params]
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', params, system)
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    geometries = [f.geometry for f in features if f.feature_type.name == 'RING_EDGE']
    assert geometries
    return [cast(RingEdgePolyline, g) for g in geometries]


def _conic_peri_envelope_px(a_px: float, ae_px: float) -> float:
    """The largest normal shift per radian of pericenter on the face-on exact conic.

    For ``r = a (1 - e**2) / (1 + e cos(theta))`` a pericenter change moves the
    edge radially by ``|dr/dtheta|``; the edge's normal tilts from radial by
    ``atan(dr/dtheta / r)``, so the move along the normal is that over
    ``sqrt(1 + (dr/dtheta / r)**2)``.
    """
    e = ae_px / a_px
    theta = np.linspace(0.0, 2.0 * np.pi, 100_001)
    r = a_px * (1.0 - e * e) / (1.0 + e * np.cos(theta))
    slope = a_px * (1.0 - e * e) * e * np.sin(theta) / (1.0 + e * np.cos(theta)) ** 2
    return float(np.max(np.abs(slope) / np.sqrt(1.0 + (slope / r) ** 2)))


def test_a_declared_pericenter_sigma_widens_the_orbit_envelope() -> None:
    """The envelope adds the pericenter term in quadrature where it beats sigma_ae."""
    edges = _eccentric_edges({'sigma_a_px': 1.0, 'sigma_long_peri_deg': 10.0})
    peri_term = _conic_peri_envelope_px(60.0, 6.0) * np.radians(10.0)
    expected = float(np.sqrt(1.0 + peri_term**2))
    # The normal shift is read from the distance field's one-pixel gradient
    # stencil, good to a few parts in a thousand on a 60 px edge.
    assert [g.sigma_orbit_radial_px for g in edges] == pytest.approx(
        [expected] * len(edges), rel=5e-3
    )


def test_the_pericenter_envelope_follows_the_exact_conic() -> None:
    """At e = 0.5 the envelope follows the exact conic, well above the linear ``ae``."""
    params = _feature_params()
    params['orbit'] = {**params['orbit'], 'ae_px': 30.0, 'long_peri_deg': 30.0}
    params['declared_orbit_sigma'] = {'sigma_long_peri_deg': 1.0}
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', params, _ring_system())
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    edge = next(f for f in features if f.feature_type.name == 'RING_EDGE')
    geometry = cast(RingEdgePolyline, edge.geometry)
    expected = _conic_peri_envelope_px(60.0, 30.0) * np.radians(1.0)
    assert geometry.sigma_orbit_radial_px == pytest.approx(expected, rel=5e-3)


def test_a_declared_pericenter_sigma_widens_the_vertices_off_the_apsides() -> None:
    """Each vertex adds ``ae |sin(lam - peri)| sigma_peri``, most at quadrature, none at apsides."""
    edges = _eccentric_edges({'sigma_a_px': 2.0, 'sigma_long_peri_deg': 10.0})
    peri_term = 6.0 * np.radians(10.0)
    assert [float(g.sigma_radial_per_vertex_px.max()) for g in edges] == pytest.approx(
        [2.0 + peri_term] * len(edges), rel=0.02
    )
    assert [float(g.sigma_radial_per_vertex_px.min()) for g in edges] == pytest.approx(
        [2.0] * len(edges), abs=0.02 * peri_term
    )


def test_the_envelope_adds_amplitude_and_pericenter_terms_in_quadrature() -> None:
    """With both declared, the envelope peaks at the apsides at ``hypot(sigma_a, sigma_ae)``.

    There the pericenter partial vanishes and the amplitude partial is exactly 1,
    and ``ae sigma_peri`` (about 1.05 px) stays below ``sigma_ae`` elsewhere, so the
    quadrature envelope is ``sqrt(1 + 2**2)``; a linear sum would give 3.
    """
    edges = _eccentric_edges({'sigma_a_px': 1.0, 'sigma_ae_px': 2.0, 'sigma_long_peri_deg': 10.0})
    expected = float(np.sqrt(1.0 + 2.0**2))
    assert [g.sigma_orbit_radial_px for g in edges] == pytest.approx(
        [expected] * len(edges), rel=5e-3
    )


def test_each_vertex_sigma_follows_the_exact_conic_from_pericenter() -> None:
    """Each vertex's sigma is ``sigma_peri |dr/dperi|`` along its normal, at its own longitude.

    The ringlet's inner edge follows the orbit (a 60 px, ae 6 px, pericenter 30
    degrees) and its outer edge the same orbit widened to a 85 px.  On the face-on
    ring a vertex's longitude is its sky angle from the center, and the exact conic
    ``r = a (1 - e**2) / (1 + e cos theta)`` gives its pericenter partial.  The
    match holds to the distance field's gradient stencil; the low-eccentricity form
    ``ae sin theta`` misses by over 15%, and a 10 degree phase error by over 50%.
    """
    sigma_peri = np.radians(30.0)
    edges = _eccentric_edges({'sigma_long_peri_deg': 30.0})
    obs = _obs()
    center_v = _SIZE / 2.0 + obs.extfov_margin_v
    center_u = _SIZE / 2.0 + obs.extfov_margin_u
    assert len(edges) == 2
    for geometry, a_px in zip(edges, (60.0, 85.0), strict=True):
        dv = geometry.vertices_vu[:, 0] - center_v
        du = geometry.vertices_vu[:, 1] - center_u
        theta = np.arctan2(-dv, du) - np.radians(30.0)
        e = 6.0 / a_px
        r = a_px * (1.0 - e * e) / (1.0 + e * np.cos(theta))
        slope = a_px * (1.0 - e * e) * e * np.sin(theta) / (1.0 + e * np.cos(theta)) ** 2
        normal_shift = np.abs(slope) / np.sqrt(1.0 + (slope / r) ** 2)
        expected = np.maximum(1.0, sigma_peri * normal_shift)
        np.testing.assert_allclose(geometry.sigma_radial_per_vertex_px, expected, rtol=0.05)


def test_a_pericenter_sigma_on_a_circular_ring_changes_nothing() -> None:
    """A circular edge has no pericenter, so its declared sigma leaves both sigmas alone."""
    params = _feature_params()
    params['declared_orbit_sigma'] = {'sigma_a_px': 2.0, 'sigma_long_peri_deg': 10.0}
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', params, _ring_system())
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    edges = [
        cast(RingEdgePolyline, f.geometry) for f in features if f.feature_type.name == 'RING_EDGE'
    ]
    assert [g.sigma_orbit_radial_px for g in edges] == pytest.approx([2.0] * len(edges), rel=1e-3)
    sigmas = np.concatenate([g.sigma_radial_per_vertex_px for g in edges])
    np.testing.assert_allclose(sigmas, 2.0, rtol=1e-3)


def test_a_null_declared_sigma_counts_as_zero() -> None:
    """A sigma key set to null contributes nothing, exactly as an omitted one."""
    edges = _eccentric_edges({'sigma_a_px': 2.0, 'sigma_long_peri_deg': None})
    omitted = _eccentric_edges({'sigma_a_px': 2.0})
    assert [g.sigma_orbit_radial_px for g in edges] == [g.sigma_orbit_radial_px for g in omitted]


def _edge_vertices(ring_system: dict[str, Any]) -> list[Any]:
    """The ringlet's predicted edge vertices under the given ring_system block."""
    model = NavModelRingsSimulated('rings', _obs(), 'SATURN', _feature_params(), ring_system)
    model.create_model()
    features = model.to_features(cast(NavContext, None))
    return [
        cast(RingEdgePolyline, f.geometry).vertices_vu
        for f in features
        if f.feature_type.name == 'RING_EDGE'
    ]


@pytest.mark.parametrize('key', ['center_v', 'center_u', 'opening_deg_obs', 'node_deg'])
def test_a_null_geometry_key_predicts_the_same_edges_as_an_omitted_one(key: str) -> None:
    """A geometry key set to null reads as unset, as the scene validator reads it."""
    null_system = _ring_system()
    null_system['geometry'][key] = None
    omitted_system = _ring_system()
    del omitted_system['geometry'][key]
    null_edges = _edge_vertices(null_system)
    omitted_edges = _edge_vertices(omitted_system)
    assert len(null_edges) == 2
    for null_vertices, omitted_vertices in zip(null_edges, omitted_edges, strict=True):
        np.testing.assert_array_equal(null_vertices, omitted_vertices)


def test_a_ramp_with_a_null_side_predicts_its_default_outer_edge() -> None:
    """A ramp's null side takes the ramp default, so its sharp end is the outer edge."""
    ramp: dict[str, Any] = {
        'kind': 'ramp',
        'width_px': 20.0,
        'orbit': {'a_px': 60.0, 'ae_px': 0.0, 'long_peri_deg': 0.0},
    }
    grid = {
        'center_v': _SIZE / 2.0,
        'center_u': _SIZE / 2.0,
        'opening_deg_obs': 90.0,
        'node_deg': 0.0,
    }
    null_side = predict_ring_feature((_SIZE, _SIZE), {**ramp, 'side': None}, **grid)
    omitted = predict_ring_feature((_SIZE, _SIZE), ramp, **grid)
    assert [edge.edge_type for edge in null_side.edges] == ['outer']
    np.testing.assert_array_equal(null_side.edges[0].vertices_vu, omitted.edges[0].vertices_vu)
