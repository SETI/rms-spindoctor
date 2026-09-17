"""Navigator-side simulated body model, including mesh prediction (B7).

``NavModelBodySimulated`` renders the *predicted* body silhouette from its own
params, which need not match what was rendered into the image.  These tests
cover the mesh-vs-ellipsoid prediction split, the pose-disagreement fixture, and
that the predicted mesh reproduces the rendered data when the params agree.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context

from spindoctor.annotation import Annotations
from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX

_SIZE = 96


def _obs() -> ObsSim:
    """A coiss_nac sim obs sized for the body scenes below."""
    return ObsSim.from_file(
        '/tmp/body_sim.json',
        sim_params={'size_v': _SIZE, 'size_u': _SIZE, 'instrument': 'coiss_nac'},
    )


def _body_params(**overrides: Any) -> dict[str, Any]:
    """Centered irregular-body params for prediction and rendering."""
    params = {
        'name': 'HYPERION',
        'center_v': _SIZE / 2.0,
        'center_u': _SIZE / 2.0,
        'axis1': 60.0,
        'axis2': 46.0,
        'axis3': 46.0,
        'illumination_angle': 30.0,
        'phase_angle': 45.0,
    }
    params.update(overrides)
    return params


def _predicted_mask(obs: ObsSim, body_params: dict[str, Any]) -> np.ndarray:
    """Build the model and return its predicted body mask (extfov coords)."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    assert model._body_mask is not None
    return np.asarray(model._body_mask)


def _data_region(obs: ObsSim, full: np.ndarray) -> np.ndarray:
    """Crop an extfov-coordinate mask to the data region."""
    ev = int(obs.extfov_margin_v)
    eu = int(obs.extfov_margin_u)
    return full[ev : ev + _SIZE, eu : eu + _SIZE]


def _mesh_params(**overrides: Any) -> dict[str, Any]:
    """Irregular-body params using the polyhedral mesh shape."""
    return _body_params(
        shape_model='polyhedral_mesh', mesh_lumpiness=0.45, mesh_seed=2, **overrides
    )


def test_the_label_anchors_on_the_pixel_the_body_centre_falls_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The label is anchored on the row and column holding the body's center.

    A scene states a center as a pixel corner, so row 40 spans 40.0 to 41.0
    and a center of 40.7 falls inside it.  The anchor addresses a pixel of the
    image, so it is that row, whatever the fraction is.

    Parameters:
        monkeypatch: Records the anchor the annotation builder is handed.
    """
    anchors: list[tuple[int, int]] = []

    def _record(
        self: NavModelBodySimulated,
        u_center: int,
        v_center: int,
        model: np.ndarray,
        limb_mask: np.ndarray,
        body_mask: np.ndarray,
    ) -> Annotations:
        """Stand in for the annotation builder, keeping its anchor."""
        anchors.append((v_center, u_center))
        return Annotations()

    monkeypatch.setattr(NavModelBodySimulated, '_create_annotations', _record)
    obs = _obs()
    params = _body_params(center_v=40.7, center_u=52.7)
    model = NavModelBodySimulated('body', obs, params['name'], params)
    model.create_model()
    model.to_annotations(bare_nav_context(obs))
    assert anchors[0] == (math.floor(params['center_v']), math.floor(params['center_u']))


def test_mesh_prediction_differs_from_ellipsoid() -> None:
    """Predicting a mesh vs an ellipsoid of equal axes gives different masks."""
    obs = _obs()
    mesh = _predicted_mask(obs, _mesh_params())
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int((mesh != ellipsoid).sum()) > 50


def test_mesh_prediction_pose_disagreement_changes_mask() -> None:
    """A different assumed pose changes the predicted mesh (chaotic-rotator)."""
    obs = _obs()
    true_pose = _predicted_mask(obs, _mesh_params(pose_euler_deg=[0.0, 0.0, 0.0]))
    wrong_pose = _predicted_mask(obs, _mesh_params(pose_euler_deg=[0.0, 90.0, 30.0]))
    assert int((true_pose != wrong_pose).sum()) > 50


def test_mesh_prediction_matches_rendered_data() -> None:
    """With identical params the predicted mesh is the rendered shape, pixel for pixel.

    Both sides call the same mesh renderer, so this is a round trip: it pins
    that the model passes the scene's params through unaltered and lands the
    result on the same grid, and it is blind to a coordinate convention the
    two sides get wrong together.  Being a round trip, it can be exact, and an
    overlap ratio would only hide the pixels that a placement error moves.
    """
    obs = _obs()
    body = _mesh_params()
    predicted = _data_region(obs, _predicted_mask(obs, body))
    img, _ = render_combined_model(
        {
            'size_v': _SIZE,
            'size_u': _SIZE,
            'random_seed': 1,
            'instrument': 'coiss_nac',
            'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
            'bodies': [body],
        }
    )
    rendered = img > 0
    assert np.array_equal(predicted, rendered)


def test_prediction_centres_on_the_centre_the_scene_states() -> None:
    """The predicted silhouette's centroid is the scene's own center.

    The anchor is outside both sides of the round trip above: the scene states
    a pixel corner center, the mask is addressed by rows and columns, and the
    ellipsoid is symmetric about its center, so the centroid of the marked
    pixels is the stated number less the half pixel that converts between the
    two coordinate systems, exactly.  A model reading the scene's center pixel
    centric fails here by that half pixel.
    """
    obs = _obs()
    predicted = _data_region(obs, _predicted_mask(obs, _body_params()))
    vs, us = np.where(predicted)
    expected = _SIZE / 2.0 - PIXEL_CENTER_TO_CORNER_PX
    assert float(vs.mean()) == pytest.approx(expected, abs=1e-12)
    assert float(us.mean()) == pytest.approx(expected, abs=1e-12)


def test_mesh_prediction_is_deterministic() -> None:
    """The predicted mesh mask is reproduced exactly across builds."""
    obs = _obs()
    first = _predicted_mask(obs, _mesh_params())
    second = _predicted_mask(obs, _mesh_params())
    assert np.array_equal(first, second)


def test_ellipsoid_prediction_still_renders() -> None:
    """The default ellipsoid prediction path remains non-empty."""
    obs = _obs()
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int(ellipsoid.sum()) > 0


_LIMB_SIZE = 220


def _limb_obs() -> ObsSim:
    """A coiss_nac sim obs sized for a well-resolved limb body."""
    return ObsSim.from_file(
        '/tmp/limb_sim.json',
        sim_params={'size_v': _LIMB_SIZE, 'size_u': _LIMB_SIZE, 'instrument': 'coiss_nac'},
    )


def _body_feature_types(obs: ObsSim, body_params: dict[str, Any]) -> list[str]:
    """Build the model and return the feature-type names it emits."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    features = model.to_features(bare_nav_context(obs))
    return [f.feature_type.name for f in features]


def _large_body(**overrides: Any) -> dict[str, Any]:
    """A well-resolved sphere (diameter 130 px) at low phase."""
    params = {
        'name': 'RHEA',
        'center_v': _LIMB_SIZE / 2.0,
        'center_u': _LIMB_SIZE / 2.0,
        'axis1': 130.0,
        'axis2': 130.0,
        'axis3': 130.0,
        'illumination_angle': 25.0,
        'phase_angle': 30.0,
    }
    params.update(overrides)
    return params


def test_large_low_phase_body_emits_limb_arc() -> None:
    """A well-resolved low-phase body emits a LIMB_ARC feature."""
    assert 'LIMB_ARC' in _body_feature_types(_limb_obs(), _large_body())


def test_small_body_emits_no_limb_arc() -> None:
    """A body below the limb-resolution floor emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in _body_feature_types(
        _limb_obs(), _large_body(axis1=40.0, axis2=40.0, axis3=40.0)
    )


def test_high_phase_body_emits_no_limb_arc() -> None:
    """A high-phase body (mostly terminator) emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in _body_feature_types(_limb_obs(), _large_body(phase_angle=120.0))


def test_limb_arc_polyline_has_vertices_and_unit_normals() -> None:
    """The emitted LIMB_ARC carries a vertex polyline with unit outward normals."""
    from spindoctor.feature.geometry import LimbPolyline

    obs = _limb_obs()
    model = NavModelBodySimulated('body', obs, 'RHEA', _large_body())
    model.create_model()
    limb = next(
        f for f in model.to_features(bare_nav_context(obs)) if f.feature_type.name == 'LIMB_ARC'
    )
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    assert geometry.vertices_vu.shape[0] >= 30
    norms = np.hypot(geometry.normals_vu[:, 0], geometry.normals_vu[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)


def _ungated_limb_model(body_params: dict[str, Any]) -> NavModelBodySimulated:
    """Build a model with the limb emission gates disabled (measurement path)."""
    obs = _limb_obs()
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.apply_limb_emission_gates = False
    model.create_model()
    return model


def _ungated_limb_vertices(body_params: dict[str, Any]) -> np.ndarray:
    """The gates-off LIMB_ARC vertex array for the given body."""
    from spindoctor.feature.geometry import LimbPolyline

    model = _ungated_limb_model(body_params)
    limb = next(
        f
        for f in model.to_features(bare_nav_context(model.obs))
        if f.feature_type.name == 'LIMB_ARC'
    )
    geometry = limb.geometry
    assert isinstance(geometry, LimbPolyline)
    return np.asarray(geometry.vertices_vu)


def test_gates_off_high_phase_body_emits_limb_arc() -> None:
    """With the gates off, a high-phase body still exposes its limb geometry."""
    vertices = _ungated_limb_vertices(_large_body(phase_angle=120.0))
    assert vertices.shape[0] > 0


def test_gates_off_small_body_emits_limb_arc() -> None:
    """With the gates off, a body below the resolution floor still emits."""
    vertices = _ungated_limb_vertices(_large_body(axis1=40.0, axis2=40.0, axis3=40.0))
    assert vertices.shape[0] > 0


def test_gates_off_limb_is_geometric_not_terminator() -> None:
    """Gates-off vertices trace the geometric limb, not the lit-region boundary.

    For a phase-120 sphere the lit-region boundary includes the terminator,
    which cuts through the disc interior; the lit geometric limb keeps every
    vertex on the silhouette outline (constant radius from the center).
    """
    vertices = _ungated_limb_vertices(_large_body(phase_angle=120.0))
    obs = _limb_obs()
    cv = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    cu = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_u)
    radii = np.hypot(vertices[:, 0] - cv, vertices[:, 1] - cu)
    assert float(radii.min()) > 62.0


def _terminator_feature(obs: ObsSim, body_params: dict[str, Any]) -> Any:
    """Build the model and return its emitted TERMINATOR_ARC feature, or None."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    for feature in model.to_features(bare_nav_context(obs)):
        if feature.feature_type.name == 'TERMINATOR_ARC':
            return feature
    return None


def test_high_phase_body_emits_terminator_arc() -> None:
    """A resolved body at appreciable phase emits a TERMINATOR_ARC."""
    assert 'TERMINATOR_ARC' in _body_feature_types(_limb_obs(), _large_body(phase_angle=90.0))


def test_zero_phase_body_emits_no_terminator_arc() -> None:
    """A fully-lit (zero-phase) body has no terminator and emits none."""
    assert 'TERMINATOR_ARC' not in _body_feature_types(_limb_obs(), _large_body(phase_angle=0.0))


def test_near_zero_phase_body_emits_no_terminator_arc() -> None:
    """Below the phase-factor floor the terminator is indistinct from the limb."""
    assert 'TERMINATOR_ARC' not in _body_feature_types(_limb_obs(), _large_body(phase_angle=1.0))


def test_highly_irregular_resolved_body_emits_no_terminator_arc() -> None:
    """A resolved highly_irregular body suppresses TERMINATOR_ARC.

    The shared ``shape_features_suppressed`` policy: a chaotic rotator's
    rendered terminator does not match the real body once it is resolved,
    so the SPICE-backed model emits none -- and the simulated model must
    not either.
    """
    assert 'TERMINATOR_ARC' not in _body_feature_types(
        _limb_obs(), _large_body(name='HYPERION', phase_angle=90.0)
    )


def test_regular_body_same_geometry_emits_terminator_arc() -> None:
    """The identical geometry on a regular body still emits TERMINATOR_ARC."""
    assert 'TERMINATOR_ARC' in _body_feature_types(
        _limb_obs(), _large_body(name='RHEA', phase_angle=90.0)
    )


def test_terminator_arc_geometry_is_terminator_polyline() -> None:
    """The emitted TERMINATOR_ARC carries a TerminatorPolyline with enough vertices."""
    from spindoctor.feature.geometry import TerminatorPolyline

    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    assert isinstance(terminator.geometry, TerminatorPolyline)
    assert terminator.geometry.vertices_vu.shape[0] >= 8


def test_terminator_arc_is_interior_not_geometric_limb() -> None:
    """The terminator lies inside the disc, distinct from the silhouette limb.

    At 90-degree phase the lit/unlit boundary is a great circle cutting through
    the disc center, so its vertices span radii from near the center out to the
    limb -- unlike the geometric limb, whose vertices all sit near the disc
    edge.  The mean terminator radius therefore sits well inside the limb.
    """
    obs = _limb_obs()
    terminator = _terminator_feature(obs, _large_body(phase_angle=90.0))
    assert terminator is not None
    vertices = np.asarray(terminator.geometry.vertices_vu)
    cv = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    cu = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_u)
    radii = np.hypot(vertices[:, 0] - cv, vertices[:, 1] - cu)
    disc_radius = 130.0 / 2.0
    assert float(radii.min()) < 0.5 * disc_radius
    assert float(radii.mean()) < 0.85 * disc_radius


def test_terminator_arc_normals_are_unit_length() -> None:
    """The terminator polyline carries unit outward normals."""
    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    normals = np.asarray(terminator.geometry.normals_vu)
    norms = np.hypot(normals[:, 0], normals[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_terminator_arc_stays_provisional_via_flags() -> None:
    """The terminator carries its phase-angle factor for the (uncalibrated) technique."""
    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    assert terminator.flags.body_name == 'RHEA'
    assert terminator.flags.phase_angle_factor == pytest.approx(1.0, abs=0.02)


def test_terminator_reliability_matches_shared_formula() -> None:
    """The sim terminator scores through the shared body-model reliability.

    ``terminator_reliability`` applies the albedo-variation and sin(phase)
    penalties the SPICE-backed model applies; the sim feature must carry
    exactly that score for its own computed arc fraction.
    """
    import math

    from spindoctor.nav_model.body_shape import load_body_shape
    from spindoctor.nav_model.nav_model_body import terminator_reliability

    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    shape = load_body_shape('RHEA')
    expected = terminator_reliability(
        visible_arc_fraction=float(terminator.flags.visible_arc_fraction),
        albedo_variation=shape.albedo_variation,
        phase_factor=math.sin(math.radians(90.0)),
    )
    assert terminator.reliability == pytest.approx(expected)


def test_terminator_reliability_is_not_pinned_at_one() -> None:
    """The albedo/phase penalties keep the reliability strictly below 1.0."""
    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    assert terminator.reliability < 1.0


def test_terminator_arc_fraction_fully_framed_is_near_one() -> None:
    """A fully framed terminator sees almost all of the unclipped ridge."""
    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    assert terminator.flags.visible_arc_fraction > 0.95


def test_terminator_arc_fraction_drops_when_frame_clips_it() -> None:
    """A body whose terminator runs off the frame edge scores a lower fraction.

    Center the body near the frame edge so part of the lit/unlit boundary
    falls outside the render; the visible-arc fraction must report the
    surviving portion, not 1.0 (the honest input BodyTerminatorNav's
    confidence needs).
    """
    clipped = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0, center_v=20.0))
    assert clipped is not None
    assert clipped.flags.visible_arc_fraction < 0.9
    assert clipped.flags.visible_arc_fraction > 0.2


def test_terminator_reliability_reasons_carry_albedo_penalty() -> None:
    """The reliability breakdown mirrors the real model's terminator fields."""
    from spindoctor.nav_model.body_shape import load_body_shape

    terminator = _terminator_feature(_limb_obs(), _large_body(phase_angle=90.0))
    assert terminator is not None
    shape = load_body_shape('RHEA')
    assert terminator.reliability_reasons.albedo_penalty == pytest.approx(
        min(1.0, shape.albedo_variation)
    )


def test_metadata_records_predicted_diameter() -> None:
    """The model metadata carries the rendered silhouette diameter."""
    obs = _limb_obs()
    model = NavModelBodySimulated('body', obs, 'RHEA', _large_body())
    model.create_model()
    assert model.metadata['predicted_diameter_px'] == pytest.approx(130.0, abs=3.0)


# ---------------------------------------------------------------------------
# BODY_BLOB detection SNR (issue #209)
# ---------------------------------------------------------------------------


def _blob_feature(obs: ObsSim, body_params: dict[str, Any], context: NavContext) -> Any:
    """Build the model and return its emitted BODY_BLOB feature."""
    model = NavModelBodySimulated('body', obs, body_params['name'], body_params)
    model.create_model()
    features = model.to_features(context)
    return next(f for f in features if f.feature_type.name == 'BODY_BLOB')


def _small_body(obs: ObsSim) -> dict[str, Any]:
    """A 12 px sphere at low phase, centered in ``obs``'s frame."""
    return {
        'name': 'RHEA',
        'center_v': obs.data_shape_v / 2.0,
        'center_u': obs.data_shape_u / 2.0,
        'axis1': 12.0,
        'axis2': 12.0,
        'axis3': 12.0,
        'illumination_angle': 0.0,
        'phase_angle': 10.0,
    }


def _noise_image(obs: ObsSim, seed: int) -> np.ndarray:
    """Unit-sigma Gaussian noise over the extfov shape."""
    rng = np.random.default_rng(seed)
    shape = (
        int(obs.data_shape_v + 2 * obs.extfov_margin_v),
        int(obs.data_shape_u + 2 * obs.extfov_margin_u),
    )
    return rng.standard_normal(shape)


def test_blob_gated_on_pure_noise_image() -> None:
    """With no body signal anywhere, the blob sits below the 0.20 gate.

    The top-N order statistics of a pure-noise search window exceed 3
    sigma individually; the null-level subtraction must keep the
    detection SNR near zero regardless.
    """
    obs = _obs()
    image = _noise_image(obs, seed=20260710)
    blob = _blob_feature(obs, _small_body(obs), bare_nav_context(obs, image))
    assert blob.reliability < 0.2


def test_blob_admits_small_bright_body_off_prediction() -> None:
    """A bright 12 px body displaced within the search window is admitted.

    Issue #209 regression: the detection SNR must find the body anywhere
    inside the extfov capture range (the pointing error is unknown), and
    the resulting reliability must clear the 0.20 BODY_BLOB gate.
    """
    obs = _obs()
    image = _noise_image(obs, seed=20260711)
    # Plant a bright disc displaced (12, -9) px from the predicted center
    # (extfov coords), well inside the 50 px extfov margins.
    center_v = obs.data_shape_v / 2.0 + obs.extfov_margin_v + 12.0
    center_u = obs.data_shape_u / 2.0 + obs.extfov_margin_u - 9.0
    vv, uu = np.indices(image.shape, dtype=np.float64)
    image[np.hypot(vv - center_v, uu - center_u) <= 6.0] += 25.0
    blob = _blob_feature(obs, _small_body(obs), bare_nav_context(obs, image))
    assert blob.reliability >= 0.2
    assert blob.reliability <= 0.4


def test_blob_admits_mostly_offscreen_body() -> None:
    """Off-sensor predicted flux must not dilute the detection SNR.

    A body whose silhouette extends past the sensor edge only lights up
    its on-sensor pixels; counting the whole predicted lit area would
    push the top-N median into the sky and falsely veto the visible
    part (regression seen on body_mostly_offscreen frames).
    """
    import dataclasses

    obs = _obs()
    # Body centered 2 px inside the frame edge: roughly half the 20 px
    # silhouette hangs past the sensor into the extfov margin.
    params = dict(_small_body(obs))
    params.update(center_v=2.0, axis1=20.0, axis2=20.0, axis3=20.0)
    image = _noise_image(obs, seed=20260712)
    mv, mu = int(obs.extfov_margin_v), int(obs.extfov_margin_u)
    # Sensor mask covers only the data region, as on a real frame.
    sensor = np.zeros(image.shape, dtype=bool)
    sensor[mv : mv + int(obs.data_shape_v), mu : mu + int(obs.data_shape_u)] = True
    # Bright half-disc at the predicted position, clipped to the sensor.
    center_v = 2.0 + mv
    center_u = obs.data_shape_u / 2.0 + mu
    vv, uu = np.indices(image.shape, dtype=np.float64)
    image[(np.hypot(vv - center_v, uu - center_u) <= 10.0) & sensor] += 25.0
    context = dataclasses.replace(bare_nav_context(obs, image), sensor_mask_ext=sensor)
    blob = _blob_feature(obs, params, context)
    assert blob.reliability >= 0.2


# ---------------------------------------------------------------------------
# Honest limb visible-arc fraction and sibling-body occlusion
# ---------------------------------------------------------------------------


def _limb_feature(
    obs: ObsSim,
    body_params: dict[str, Any],
    *,
    sibling_bodies: list[dict[str, Any]] | None = None,
) -> Any:
    """Build the model and return its emitted LIMB_ARC feature, or None."""
    model = NavModelBodySimulated(
        'body', obs, body_params['name'], body_params, sibling_bodies=sibling_bodies
    )
    model.create_model()
    for feature in model.to_features(bare_nav_context(obs)):
        if feature.feature_type.name == 'LIMB_ARC':
            return feature
    return None


def test_limb_arc_fraction_fully_framed_is_near_one() -> None:
    """A fully framed, unoccluded limb sees almost all of the silhouette."""
    limb = _limb_feature(_limb_obs(), _large_body())
    assert limb is not None
    assert limb.flags.visible_arc_fraction > 0.9


def test_limb_arc_fraction_drops_when_frame_clips_it() -> None:
    """A body sliding off the frame edge scores a lower limb fraction.

    Center the body near the frame edge so part of the silhouette boundary
    falls outside the render; the visible-arc fraction must report the
    surviving portion, not 1.0 (the honest input BodyLimbNav's
    visible_limb_arc_fraction confidence term needs).
    """
    clipped = _limb_feature(_limb_obs(), _large_body(center_v=20.0))
    assert clipped is not None
    assert clipped.flags.visible_arc_fraction < 0.85
    assert clipped.flags.visible_arc_fraction > 0.2


def test_limb_reliability_matches_shared_formula() -> None:
    """The sim limb scores through the shared body-model reliability."""
    from spindoctor.nav_model.nav_model_body import limb_reliability

    limb = _limb_feature(_limb_obs(), _large_body())
    assert limb is not None
    expected = limb_reliability(
        visible_arc_fraction=float(limb.flags.visible_arc_fraction),
        visible_arc_px=float(limb.geometry.vertices_vu.shape[0]),
    )
    assert limb.reliability == pytest.approx(expected)


def _near_sibling(**overrides: Any) -> dict[str, Any]:
    """A nearer sibling sphere overlapping the _large_body silhouette."""
    params = {
        'name': 'DIONE',
        'center_v': _LIMB_SIZE / 2.0,
        'center_u': _LIMB_SIZE / 2.0 + 55.0,
        'axis1': 100.0,
        'axis2': 100.0,
        'axis3': 100.0,
        'illumination_angle': 25.0,
        'phase_angle': 30.0,
        'range_km': 500000.0,
    }
    params.update(overrides)
    return params


def test_occluded_limb_fraction_drops() -> None:
    """A nearer sibling hides part of the limb; the fraction reports it."""
    body = _large_body(range_km=700000.0)
    unoccluded = _limb_feature(_limb_obs(), body)
    occluded = _limb_feature(_limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert unoccluded is not None
    assert occluded is not None
    assert occluded.flags.visible_arc_fraction < unoccluded.flags.visible_arc_fraction - 0.1
    assert occluded.reliability < unoccluded.reliability


def test_occluded_limb_vertices_leave_the_polyline() -> None:
    """No surviving limb vertex sits inside the nearer sibling's silhouette.

    A hidden arc has no counterpart edge in the image, so its vertices are
    dropped rather than left for the robust fit to reject.
    """
    body = _large_body(range_km=700000.0)
    occluded = _limb_feature(_limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert occluded is not None
    obs = _limb_obs()
    vertices = np.asarray(occluded.geometry.vertices_vu)
    sib_v = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    sib_u = _LIMB_SIZE / 2.0 + 55.0 + int(obs.extfov_margin_u)
    distances = np.hypot(vertices[:, 0] - sib_v, vertices[:, 1] - sib_u)
    assert float(distances.min()) > 49.0


def test_farther_sibling_does_not_occlude() -> None:
    """A sibling with a larger range hides nothing of this body's limb."""
    body = _large_body(range_km=500000.0)
    behind = _limb_feature(_limb_obs(), body, sibling_bodies=[_near_sibling(range_km=700000.0)])
    alone = _limb_feature(_limb_obs(), body)
    assert behind is not None
    assert alone is not None
    assert behind.flags.visible_arc_fraction == pytest.approx(alone.flags.visible_arc_fraction)


def test_sibling_without_range_does_not_occlude() -> None:
    """Without explicit ranges on both sides, stacking is unknowable.

    Mirrors the renderer's rule that overlapping bodies must all carry an
    explicit range_km before their depth order means anything.
    """
    body = _large_body(range_km=700000.0)
    sibling = _near_sibling()
    del sibling['range_km']
    unknown = _limb_feature(_limb_obs(), body, sibling_bodies=[sibling])
    alone = _limb_feature(_limb_obs(), body)
    assert unknown is not None
    assert alone is not None
    assert unknown.flags.visible_arc_fraction == pytest.approx(alone.flags.visible_arc_fraction)


def test_deeply_occluded_limb_is_not_emitted() -> None:
    """A limb with fewer than 30 surviving vertices emits no LIMB_ARC."""
    body = _large_body(range_km=700000.0)
    # A nearer sibling almost concentric with and larger than the body
    # hides nearly the whole silhouette boundary.
    occluder = _near_sibling(center_u=_LIMB_SIZE / 2.0 + 8.0, axis1=150.0, axis2=150.0, axis3=150.0)
    limb = _limb_feature(_limb_obs(), body, sibling_bodies=[occluder])
    assert limb is None


def test_occluded_terminator_fraction_drops() -> None:
    """Sibling occlusion also reduces the terminator's visible-arc fraction."""
    body = _large_body(phase_angle=90.0, range_km=700000.0)
    obs = _limb_obs()
    model_alone = NavModelBodySimulated('body', obs, body['name'], body)
    model_alone.create_model()
    alone = next(
        f
        for f in model_alone.to_features(bare_nav_context(obs))
        if f.feature_type.name == 'TERMINATOR_ARC'
    )
    model_occ = NavModelBodySimulated(
        'body', obs, body['name'], body, sibling_bodies=[_near_sibling()]
    )
    model_occ.create_model()
    occluded = next(
        f
        for f in model_occ.to_features(bare_nav_context(obs))
        if f.feature_type.name == 'TERMINATOR_ARC'
    )
    from spindoctor.feature.flags import TerminatorArcFlags

    assert isinstance(occluded.flags, TerminatorArcFlags)
    assert isinstance(alone.flags, TerminatorArcFlags)
    assert occluded.flags.visible_arc_fraction < alone.flags.visible_arc_fraction - 0.1


def _disc_feature(
    obs: ObsSim,
    body_params: dict[str, Any],
    *,
    sibling_bodies: list[dict[str, Any]] | None = None,
) -> NavFeature:
    """Build the simulated body model and return its emitted BODY_DISC feature.

    Parameters:
        obs: Simulated observation the model renders against.
        body_params: Idealized parameter dict for the navigated body.
        sibling_bodies: Idealized parameter dicts of the other bodies sharing
            the scene, wired in as occlusion candidates; None for the
            single-body case.

    Returns:
        The single BODY_DISC feature the model emits for this body.
    """
    model = NavModelBodySimulated(
        'body', obs, body_params['name'], body_params, sibling_bodies=sibling_bodies
    )
    model.create_model()
    return next(
        f for f in model.to_features(bare_nav_context(obs)) if f.feature_type.name == 'BODY_DISC'
    )


def test_occluded_disc_template_drops_hidden_pixels() -> None:
    """The BODY_DISC template mask excludes pixels a nearer sibling hides."""
    body = _large_body(range_km=700000.0)
    disc = _disc_feature(_limb_obs(), body, sibling_bodies=[_near_sibling()])
    v_min, u_min, _v_max, _u_max = disc.geometry.bbox_extfov_vu
    obs = _limb_obs()
    template_mask = np.asarray(disc.template_mask, dtype=bool)
    vs, us = np.where(template_mask)
    sib_v = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_v) - v_min
    sib_u = _LIMB_SIZE / 2.0 + 55.0 + int(obs.extfov_margin_u) - u_min
    dist = np.hypot(vs - sib_v, us - sib_u)
    # The sibling is a radius-50 sphere; no template pixel survives deep inside it.
    assert int(np.count_nonzero(dist < 40.0)) == 0


def test_occluded_disc_template_img_zero_under_sibling() -> None:
    """The BODY_DISC template brightness is zero deep inside a nearer sibling."""
    body = _large_body(range_km=700000.0)
    disc = _disc_feature(_limb_obs(), body, sibling_bodies=[_near_sibling()])
    v_min, u_min, _v_max, _u_max = disc.geometry.bbox_extfov_vu
    obs = _limb_obs()
    template_img = np.asarray(disc.template_img, dtype=np.float64)
    rows, cols = template_img.shape
    vv, uu = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    sib_v = _LIMB_SIZE / 2.0 + int(obs.extfov_margin_v) - v_min
    sib_u = _LIMB_SIZE / 2.0 + 55.0 + int(obs.extfov_margin_u) - u_min
    # Deep inside the radius-50 sibling the target disc is fully hidden, so the
    # image-zeroing (not just the mask) must drive the brightness to zero there.
    deep = np.hypot(vv - sib_v, uu - sib_u) < 40.0
    assert float(np.max(template_img[deep])) == 0.0


def test_occluded_disc_reliability_drops() -> None:
    """A nearer sibling lowers the BODY_DISC visible fraction and reliability."""
    body = _large_body(range_km=700000.0)
    alone = _disc_feature(_limb_obs(), body)
    occluded = _disc_feature(_limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert alone.reliability == pytest.approx(1.0)
    assert occluded.reliability < 0.9


def test_farther_sibling_leaves_disc_template_intact() -> None:
    """A sibling with a larger range does not trim the disc template."""
    body = _large_body(range_km=500000.0)
    alone = _disc_feature(_limb_obs(), body)
    behind = _disc_feature(_limb_obs(), body, sibling_bodies=[_near_sibling(range_km=700000.0)])
    alone_pixels = int(np.count_nonzero(np.asarray(alone.template_mask, dtype=bool)))
    behind_pixels = int(np.count_nonzero(np.asarray(behind.template_mask, dtype=bool)))
    assert behind_pixels == alone_pixels


def test_instances_for_obs_wires_siblings() -> None:
    """Each per-body model instance receives the other bodies as siblings."""
    far = _large_body(range_km=700000.0)
    near = _near_sibling()
    obs = ObsSim.from_file(
        '/tmp/mutual_sim.json',
        sim_params={
            'size_v': _LIMB_SIZE,
            'size_u': _LIMB_SIZE,
            'instrument': 'coiss_nac',
            'bodies': [far, near],
        },
    )
    models = NavModelBodySimulated.instances_for_obs(obs)
    assert len(models) == 2
    for model in models:
        assert isinstance(model, NavModelBodySimulated)
        assert len(model._sibling_bodies) == 1
        assert model._sibling_bodies[0]['name'] != model._sim_params['name']
