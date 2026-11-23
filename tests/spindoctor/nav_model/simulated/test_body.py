"""Navigator-side simulated body model, including mesh prediction.

``NavModelBodySimulated`` renders the *predicted* body silhouette from its own
params, which need not match what was rendered into the image.  These tests
cover the mesh-vs-ellipsoid prediction split, the pose-disagreement fixture, that
the predicted mesh reproduces the rendered data when the params agree, and the blob
feature.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context
from tests.spindoctor.nav_model.simulated.body_fixtures import (
    SIZE,
    large_body,
    limb_obs,
    small_obs,
)

from spindoctor.annotation import Annotations
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render.driver import render_scene
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX


def _body_params(**overrides: Any) -> dict[str, Any]:
    """Centered irregular-body params for prediction and rendering."""
    params = {
        'name': 'HYPERION',
        'center_v': SIZE / 2.0,
        'center_u': SIZE / 2.0,
        'semi_a_px': 30.0,
        'semi_b_px': 23.0,
        'semi_c_px': 23.0,
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
    return full[ev : ev + SIZE, eu : eu + SIZE]


def _mesh_params(**overrides: Any) -> dict[str, Any]:
    """Irregular-body params using the polyhedral mesh shape."""
    return _body_params(
        shape_model='polyhedral_mesh', mesh_lumpiness=0.45, mesh_seed=2, **overrides
    )


def test_the_label_anchors_on_the_pixel_the_body_center_falls_in(
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
    obs = small_obs()
    params = _body_params(center_v=40.7, center_u=52.7)
    model = NavModelBodySimulated('body', obs, params['name'], params)
    model.create_model()
    model.to_annotations(bare_nav_context(obs))
    assert anchors[0] == (math.floor(params['center_v']), math.floor(params['center_u']))


def test_mesh_prediction_differs_from_ellipsoid() -> None:
    """Predicting a mesh vs an ellipsoid of equal axes gives different masks."""
    obs = small_obs()
    mesh = _predicted_mask(obs, _mesh_params())
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int((mesh != ellipsoid).sum()) > 50


def test_mesh_prediction_pose_disagreement_changes_mask() -> None:
    """A different assumed pose changes the predicted mesh (chaotic-rotator)."""
    obs = small_obs()
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
    obs = small_obs()
    body = _mesh_params()
    predicted = _data_region(obs, _predicted_mask(obs, body))
    _, meta = render_scene(
        {
            'size_v': SIZE,
            'size_u': SIZE,
            'random_seed': 1,
            'instrument': 'coiss_nac',
            'noise': {'poisson': False, 'read_noise_dn': 0.0},
            'detector': {'bias_dn': 0.0},
            'bodies': [body],
        }
    )
    # The night side renders black, so the rendered shape is the body's mask in the
    # render truth, which covers every pixel the outline touches.
    rendered = meta['body_mask_map']['HYPERION']
    assert np.array_equal(predicted, rendered)


def test_prediction_centers_on_the_center_the_scene_states() -> None:
    """The predicted silhouette's centroid is the scene's own center.

    The anchor is outside both sides of the round trip above: the scene states
    a pixel corner center, the mask is addressed by rows and columns, and the
    ellipsoid is symmetric about its center, so the centroid of the marked
    pixels is the stated number less the half pixel that converts between the
    two coordinate systems, exactly.  A model reading the scene's center pixel
    centric fails here by that half pixel.
    """
    obs = small_obs()
    predicted = _data_region(obs, _predicted_mask(obs, _body_params()))
    vs, us = np.where(predicted)
    expected = SIZE / 2.0 - PIXEL_CENTER_TO_CORNER_PX
    assert float(vs.mean()) == pytest.approx(expected, abs=1e-12)
    assert float(us.mean()) == pytest.approx(expected, abs=1e-12)


def test_mesh_prediction_is_deterministic() -> None:
    """The predicted mesh mask is reproduced exactly across builds."""
    obs = small_obs()
    first = _predicted_mask(obs, _mesh_params())
    second = _predicted_mask(obs, _mesh_params())
    assert np.array_equal(first, second)


def test_ellipsoid_prediction_still_renders() -> None:
    """The default ellipsoid prediction path remains non-empty."""
    obs = small_obs()
    ellipsoid = _predicted_mask(obs, _body_params())
    assert int(ellipsoid.sum()) > 0


def test_metadata_records_predicted_diameter() -> None:
    """The model metadata carries the rendered silhouette diameter."""
    obs = limb_obs()
    model = NavModelBodySimulated('body', obs, 'RHEA', large_body())
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
        'semi_a_px': 6.0,
        'semi_b_px': 6.0,
        'semi_c_px': 6.0,
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
    obs = small_obs()
    image = _noise_image(obs, seed=20260710)
    blob = _blob_feature(obs, _small_body(obs), bare_nav_context(obs, image))
    assert blob.reliability < 0.2


def test_blob_admits_small_bright_body_off_prediction() -> None:
    """A bright 12 px body displaced within the search window is admitted.

    Issue #209 regression: the detection SNR must find the body anywhere
    inside the extfov capture range (the pointing error is unknown), and
    the resulting reliability must clear the 0.20 BODY_BLOB gate.
    """
    obs = small_obs()
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


def test_blob_snr_ignores_off_sensor_flux() -> None:
    """Off-sensor predicted flux must not dilute the detection SNR.

    A body whose silhouette extends past the sensor edge only lights up
    its on-sensor pixels; counting the whole predicted lit area would
    push the top-N median into the sky and falsely veto the visible
    part (regression seen on body_mostly_offscreen frames).  The overflow
    gate withholds this body's blob from emission, so the feature is built
    directly to reach the estimator.
    """
    import dataclasses

    obs = small_obs()
    # Body centered 2 px inside the frame edge: roughly half the 20 px
    # silhouette hangs past the sensor into the extfov margin.
    params = dict(_small_body(obs))
    params.update(center_v=2.0, semi_a_px=10.0, semi_b_px=10.0, semi_c_px=10.0)
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
    model = NavModelBodySimulated('body', obs, params['name'], params)
    model.create_model()
    blob = model._build_blob_feature(load_body_shape(params['name']), context=context)
    assert blob.reliability >= 0.2
