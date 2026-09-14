"""Navigator-side ring_system predictions: projection, boundary, honesty.

``NavModelRingsSimulated`` consumes only the filtered idealized view: the
shared projection geometry and the navigable features' catalog orbits.
These tests pin that a predicted edge lands on the rendered feature
boundary for an inclined, rotated system (both sides project through the
shared helpers), that the planted ``orbit_error`` never reaches the
prediction (the navigator predicts catalog positions), and that
non-navigable features produce no model at all.
"""

from typing import Any, cast

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import validate_sim_params
from spindoctor.support.types import NDArrayFloatType

_SIZE = 160
_CENTER = _SIZE / 2.0

# Rows either side of a vertex that the column scan reads.  Three is enough to
# reach clear of the band's edge on both sides without meeting its other edge.
_SCAN_HALF_ROWS = 3

# Bound on the mean signed distance between the predicted edge and the
# rendered boundary.  The per-vertex scatter is about half a pixel and the
# edges carry hundreds of vertices, so the mean settles two orders of
# magnitude inside this; half a pixel of convention error lands five times
# outside it.
_EDGE_MEAN_BOUND_PX = 0.1


def _scene(feature: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A noise-free calibrated scene with one inclined, rotated feature."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'ring_nav_probe',
        'instrument': 'coiss_calib_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 5,
        'ring_system': {
            'geometry': {
                'center_v': _CENTER,
                'center_u': _CENTER,
                'opening_deg_obs': 35.0,
                'opening_deg_sun': 25.0,
                'node_deg': 25.0,
            },
            'features': [feature],
        },
    }
    scene.update(extra)
    return validate_sim_params(scene, source='ring_nav_probe')


def _ringlet(**extra: Any) -> dict[str, Any]:
    feature: dict[str, Any] = {
        'name': 'BAND',
        'kind': 'ringlet',
        'tau': 2.0,
        'width': 12.0,
        'navigable': True,
        'orbit': {'a': 45.0, 'ae': 4.0, 'long_peri': 30.0, 'rate_peri': 0.0},
    }
    feature.update(extra)
    return feature


def _obs(scene: dict[str, Any]) -> ObsSim:
    return ObsSim.from_file('/tmp/ring_nav_probe.yaml', sim_params=scene)


def _nav_features(obs: ObsSim) -> list[NavFeature]:
    models = NavModelRingsSimulated.instances_for_obs(obs)
    assert len(models) == 1
    models[0].create_model()
    return models[0].to_features(cast(NavContext, None))


def _edge_vertices_data(obs: ObsSim) -> dict[str, Any]:
    """Predicted edge vertices keyed by edge type, in data coordinates."""
    out: dict[str, Any] = {}
    for feature in _nav_features(obs):
        if feature.feature_type.name != 'RING_EDGE':
            continue
        geometry = feature.geometry
        assert isinstance(geometry, RingEdgePolyline)
        vertices = geometry.vertices_vu.copy()
        vertices[:, 0] -= obs.extfov_margin_v
        vertices[:, 1] -= obs.extfov_margin_u
        out[feature.feature_id.rsplit(':', 1)[-1]] = vertices
    return out


def _coverage_fractions(obs: ObsSim) -> NDArrayFloatType:
    """Return the rendered band's per-pixel covered fraction.

    The band renders at one brightness, so dividing by that plateau turns the
    frame into the fraction of each pixel the band covers, which is what makes
    a sub-pixel measurement of its boundary possible at all.

    Parameters:
        obs: Observation carrying the rendered frame.
    """
    data = np.asarray(obs.data, dtype=np.float64)
    plateau = float(np.median(data[data > 0.9 * data.max()]))
    fractions: NDArrayFloatType = np.clip(data / plateau, 0.0, 1.0)
    return fractions


def _edge_residuals_px(coverage: NDArrayFloatType, vertices: np.ndarray) -> list[float]:
    """Return each vertex's signed row distance from the rendered boundary.

    A column through a near-horizontal stretch of the band runs from fully
    covered to fully empty, and the covered length in that column is the sum
    of its coverage fractions, which puts the boundary at a definite sub-pixel
    row.  Columns where more than one pixel is partly covered are where the
    band runs steeply and a column is the wrong direction to measure along, so
    they are left out rather than measured badly.

    Parameters:
        coverage: Per-pixel covered fraction of the rendered frame.
        vertices: Predicted ``(v, u)`` vertices in data coordinates.
    """
    residuals: list[float] = []
    for v, u in vertices:
        v_i = round(float(v))
        u_i = round(float(u))
        low = v_i - _SCAN_HALF_ROWS
        high = v_i + _SCAN_HALF_ROWS
        if low < 0 or high >= _SIZE or not 0 <= u_i < _SIZE:
            continue
        column = coverage[low : high + 1, u_i]
        if int(((column > 0.001) & (column < 0.999)).sum()) > 1:
            continue
        if column[0] > 0.999 and column[-1] < 0.001:
            boundary = low - 0.5 + float(column.sum())
        elif column[0] < 0.001 and column[-1] > 0.999:
            boundary = high + 0.5 - float(column.sum())
        else:
            continue
        residuals.append(float(v) - boundary)
    return residuals


def test_predicted_edge_vertices_are_whole_pixel_centric_positions() -> None:
    """Each predicted edge vertex names a pixel, so it carries no fraction.

    The edges are read off a rasterized border mask, so a vertex is a row and
    a column stated in pixel centric coordinates.  A fraction in one means a
    conversion reached the polyline that the technique side would then have to
    undo.
    """
    obs = _obs(_scene(_ringlet()))
    for vertices in _edge_vertices_data(obs).values():
        assert np.array_equal(vertices, np.rint(vertices))


def test_predicted_edges_land_on_the_projected_render() -> None:
    """At B = 35, node = 25 the prediction sits on the rendered boundary.

    Both sides project through the shared opening-angle helpers, so a
    predicted edge vertex must sit on the boundary of the rendered
    (foreshortened, rotated) band.  The band renders with partly covered
    pixels at its edge, so the boundary has a sub-pixel row and the
    prediction can be measured against it instead of being truncated to a
    pixel and asked only to land near a transition.

    A single column scan across a curving edge is noisy, so the statistic is
    the mean signed distance over the edge: its scatter is about half a pixel
    per vertex and there are hundreds of them, which puts the mean well
    inside a tenth of a pixel and leaves the half pixel this bound exists to
    catch five times outside it.
    """
    obs = _obs(_scene(_ringlet()))
    coverage = _coverage_fractions(obs)
    edges = _edge_vertices_data(obs)
    assert set(edges) == {'inner', 'outer'}
    for vertices in edges.values():
        residuals = _edge_residuals_px(coverage, vertices)
        assert len(residuals) > 50
        assert abs(float(np.mean(residuals))) < _EDGE_MEAN_BOUND_PX


def test_orbit_error_never_reaches_the_prediction() -> None:
    """The planted ephemeris error displaces the render, not the prediction.

    The same scene with and without a 6 px semimajor-axis error predicts
    byte-identical edges (the navigator sees catalog values only) while the
    rendered images differ -- the misplacement the navigator must absorb.
    """
    clean = _obs(_scene(_ringlet()))
    planted = _obs(_scene(_ringlet(orbit_error={'delta_a_px': 6.0})))
    clean_edges = _edge_vertices_data(clean)
    planted_edges = _edge_vertices_data(planted)
    np.testing.assert_array_equal(planted_edges['inner'], clean_edges['inner'])
    np.testing.assert_array_equal(planted_edges['outer'], clean_edges['outer'])
    assert not np.array_equal(np.asarray(planted.data), np.asarray(clean.data))


def test_non_navigable_feature_builds_no_model() -> None:
    """A rendered confounder feature produces no navigator-side model."""
    obs = _obs(_scene({**_ringlet(), 'navigable': False}))
    assert float(np.asarray(obs.data).max()) > 0.0
    assert NavModelRingsSimulated.instances_for_obs(obs) == []


def test_edge_kind_predicts_a_single_boundary() -> None:
    """An 'edge' feature emits one RING_EDGE and no annulus template."""
    feature = {
        'name': 'SHEETEDGE',
        'kind': 'edge',
        'tau': 1.5,
        'side': 'in',
        'navigable': True,
        'orbit': {'a': 45.0},
    }
    features = _nav_features(_obs(_scene(feature)))
    kinds = [f.feature_type.name for f in features]
    assert kinds == ['RING_EDGE']
    assert features[0].feature_id == 'ring_edge:SHEETEDGE:edge'


def test_ringlet_emits_annulus_template_and_two_edges() -> None:
    """A navigable ringlet emits the correlation template plus both edges."""
    features = _nav_features(_obs(_scene(_ringlet())))
    kinds = sorted(f.feature_type.name for f in features)
    assert kinds == ['RING_ANNULUS', 'RING_EDGE', 'RING_EDGE']
    annulus = next(f for f in features if f.feature_type.name == 'RING_ANNULUS')
    assert annulus.template_img is not None
    assert float(annulus.template_img.max()) == 1.0


def test_declared_orbit_sigma_raises_the_radial_sigma() -> None:
    """declared_orbit_sigma widens the per-vertex radial error bars."""
    features = _nav_features(
        _obs(_scene(_ringlet(declared_orbit_sigma={'sigma_a_px': 2.0, 'sigma_ae_px': 0.5})))
    )
    edge = next(f for f in features if f.feature_type.name == 'RING_EDGE')
    geometry = edge.geometry
    assert isinstance(geometry, RingEdgePolyline)
    assert float(geometry.sigma_radial_per_vertex_px[0]) == 2.5
