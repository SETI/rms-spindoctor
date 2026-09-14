"""Unit tests for the simulated Titan haze model.

The model's whole job is to build the same
:class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs` the
catalog-driven model builds, from a simulated scene's idealized body
parameters instead of from ``oops``.  These tests pin that translation --
the crossing between the two pixel coordinate systems, the radii, the
symmetry axis and its degenerate branch, and the three contaminant-mask
components -- plus the fact that everything downstream (the emitted feature,
its reliability, the overlay) is inherited rather than reimplemented, so a
simulated haze frame cannot mean something different from a real one.

The two places where the pixel coordinate systems cross -- the predicted disc
center and the masked star disc -- are held against the frame the simulator
drew rather than against the arithmetic that produced them.  The rest pin the
model's own arithmetic and measure nothing rendered.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import TitanHazeGeometry
from spindoctor.nav_model.nav_model_titan import NavModelTitan
from spindoctor.nav_model.nav_model_titan_simulated import (
    NavModelTitanSimulated,
)
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.support.types import NDArrayFloatType

_SIZE = 300
_CENTER = 150.0
_SOLID_RADIUS_PX = 60.0
# Titan's published mean radius over the apparent radius above: the scene's
# pixel scale, and therefore the one the envelope radius is derived through.
_KM_PER_PIXEL = 2575.0 / _SOLID_RADIUS_PX

# A body placed away from the frame center, and at different v and u, so a
# prediction that had reached for either instead of the body's own center
# could not pass.  Stated half past a whole number in the pixel corner
# coordinates a scene is written in, the rendered center falls on a pixel
# center, where the frame's mirror symmetry is exact whole array rows.
_DISC_V = 100.5
_DISC_U = 130.5
# Half-width of the box reflected around the disc: wide enough to hold the
# whole body and its haze, narrow enough to keep the frame edges out.
_DISC_REACH_PX = 80

# A star placed the same way, clear of the body's light.
_STAR_V = 40.5
_STAR_U = 60.5
_STAR_REACH_PX = 8


def _titan(**overrides: Any) -> dict[str, Any]:
    """A simulated Titan carrying every parameter the model needs."""
    return {
        'name': 'TITAN',
        'shape_model': 'ellipsoid',
        'center_v': _CENTER,
        'center_u': _CENTER,
        'axis1': 2.0 * _SOLID_RADIUS_PX,
        'axis2': 2.0 * _SOLID_RADIUS_PX,
        'axis3': 2.0 * _SOLID_RADIUS_PX,
        'illumination_angle': 90.0,
        'phase_angle': 45.0,
        'km_per_pixel': _KM_PER_PIXEL,
        'range_km': 1.2e6,
        'atmosphere': {'scale_height_px': 8.0, 'tau_ref': 3.0, 'g': 0.5},
        **overrides,
    }


def _obs(bodies: list[dict[str, Any]], stars: list[dict[str, Any]] | None = None) -> ObsSim:
    """Build a simulated observation carrying the given bodies and stars."""
    scene: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 3,
        'exposure_sec': 1.0,
        'bodies': bodies,
    }
    if stars is not None:
        scene['stars'] = stars
    return ObsSim.from_file('/tmp/titan_sim.json', sim_params=scene)


def _model_for(obs: ObsSim) -> Any:
    """Build the simulated haze model for an observation, asserting exactly one."""
    instances = NavModelTitanSimulated.instances_for_obs(obs)
    assert len(instances) == 1
    return instances[0]


def _model(bodies: list[dict[str, Any]], stars: list[dict[str, Any]] | None = None) -> Any:
    """Build the simulated haze model for a scene, asserting exactly one."""
    return _model_for(_obs(bodies, stars))


def _rendered_image(obs: ObsSim) -> NDArrayFloatType:
    """The frame the simulator drew for an observation."""
    return np.asarray(obs.data, dtype=np.float64)


def _rendered_disc_center_vu(image: NDArrayFloatType) -> tuple[float, float]:
    """Measure a rendered body's center from the extent of its own light.

    A fully lit sphere renders radially symmetric, so the first and the last
    row its bright half reaches sit the same distance either side of its
    center and their midpoint is that center, in the pixel centric
    coordinates an array index is read in.

    Parameters:
        image: A rendered frame holding one body and nothing else.

    Returns:
        The measured ``(v, u)`` center in data coordinates.
    """
    floor = float(image.min())
    lit = image > floor + 0.5 * (float(image.max()) - floor)
    vs, us = np.nonzero(lit)
    return (0.5 * float(vs.min() + vs.max()), 0.5 * float(us.min() + us.max()))


def _rendered_star_center_vu(
    image: NDArrayFloatType, near_vu: tuple[int, int]
) -> tuple[float, float]:
    """Measure a rendered star's center of light within a box around it.

    The frame carries a constant pedestal, so the box is weighted by its
    brightness above the frame's own floor before the centroid is taken.

    Parameters:
        image: A rendered frame.
        near_vu: Whole-pixel ``(v, u)`` the box is cut around, close enough
            to the star that the box holds all of its light.

    Returns:
        The measured ``(v, u)`` center in data coordinates.
    """
    v_lo = near_vu[0] - _STAR_REACH_PX
    u_lo = near_vu[1] - _STAR_REACH_PX
    box = image[v_lo : near_vu[0] + _STAR_REACH_PX + 1, u_lo : near_vu[1] + _STAR_REACH_PX + 1]
    weights = box - float(image.min())
    vs, us = np.mgrid[v_lo : v_lo + box.shape[0], u_lo : u_lo + box.shape[1]].astype(np.float64)
    total = float(weights.sum())
    return (float((vs * weights).sum() / total), float((us * weights).sum() / total))


def _mirror_residual(
    image: NDArrayFloatType, center_vu: tuple[float, float], *, reach_px: int, axis: int
) -> float:
    """Largest disagreement between a box of a frame and its own reflection.

    Cuts the square box of half-width ``reach_px`` around ``center_vu`` and
    reflects it across one axis.  Zero means the light in the box really is
    symmetric about that row or column, which is a property of the rendered
    frame and of nothing the model computed.

    Parameters:
        image: A rendered frame.
        center_vu: The pixel centric center to reflect about; the scenes
            here place their light on a pixel center, so a measurement that
            came out half a pixel away cuts the box off center and the
            residual stops being zero.
        reach_px: Half-width of the box in pixels.
        axis: 0 to reflect rows, 1 to reflect columns.

    Returns:
        The largest absolute difference between the box and its mirror.
    """
    v0 = int(center_vu[0])
    u0 = int(center_vu[1])
    box = image[v0 - reach_px : v0 + reach_px + 1, u0 - reach_px : u0 + reach_px + 1]
    mirrored = box[::-1, :] if axis == 0 else box[:, ::-1]
    return float(np.abs(box - mirrored).max())


# ---------------------------------------------------------------------------
# Geometry translation.
# ---------------------------------------------------------------------------


def test_predicted_center_is_the_rendered_disc_center() -> None:
    """The predicted center lands on the center of the disc the renderer drew.

    The body is rendered fully lit, so it is radially symmetric and the frame
    is its own witness: reflected about the row and the column its light
    centroids to, it reproduces itself exactly.  The prediction is then held
    against that measured center rather than against the arithmetic that
    produced it, so a model that read the scene's pixel corner center as a
    pixel centric one misses the light by the half pixel that separates the
    two systems.
    """
    obs = _obs([_titan(center_v=_DISC_V, center_u=_DISC_U, phase_angle=0.0)])
    image = _rendered_image(obs)
    measured = _rendered_disc_center_vu(image)
    predicted = _model_for(obs).geometry_inputs.predicted_center_vu
    assert _mirror_residual(image, measured, reach_px=_DISC_REACH_PX, axis=0) == 0.0
    assert _mirror_residual(image, measured, reach_px=_DISC_REACH_PX, axis=1) == 0.0
    assert predicted[0] - float(obs.extfov_margin_v) == pytest.approx(measured[0])
    assert predicted[1] - float(obs.extfov_margin_u) == pytest.approx(measured[1])


def test_solid_radius_is_the_mean_image_plane_semi_axis() -> None:
    """The solid radius averages the two image-plane semi-axes."""
    body = _titan(axis1=100.0, axis2=140.0)
    assert _model([body]).geometry_inputs.r_solid_px == pytest.approx(60.0)


def test_envelope_adds_the_configured_atmosphere_height() -> None:
    """The envelope is the solid radius plus the configured height in pixels."""
    geometry = _model([_titan()]).geometry_inputs
    expected = _SOLID_RADIUS_PX + 700.0 / _KM_PER_PIXEL
    assert geometry.r_env_px == pytest.approx(expected)


def test_symmetry_axis_points_toward_the_sub_solar_side() -> None:
    """``a_hat`` from the reported angle matches the scene's sun direction.

    The scene lights from the right (``illumination_angle`` 90), so the
    sunward image direction is ``+u`` and the fitting library's axis vector
    ``(sin theta, cos theta)`` must be ``(0, 1)``.
    """
    theta = _model([_titan(illumination_angle=90.0)]).geometry_inputs.theta_rad
    assert math.sin(theta) == pytest.approx(0.0, abs=1e-12)
    assert math.cos(theta) == pytest.approx(1.0)


def test_symmetry_axis_follows_the_illumination_angle() -> None:
    """Lighting from the top puts the axis along ``-v``."""
    theta = _model([_titan(illumination_angle=0.0)]).geometry_inputs.theta_rad
    assert math.sin(theta) == pytest.approx(-1.0)
    assert math.cos(theta) == pytest.approx(0.0, abs=1e-12)


def test_axis_is_not_degenerate_at_appreciable_phase() -> None:
    """A resolved body at 45 degrees phase has a well-defined axis."""
    assert not _model([_titan()]).geometry_inputs.axis_degenerate


def test_axis_is_degenerate_near_zero_phase() -> None:
    """Near zero phase the disc is rotationally symmetric and any axis serves.

    The sub-solar point of a sphere at phase ``p`` projects ``R sin(p)`` from
    the disc center, so a small enough phase puts it inside the configured
    minimum axis offset -- the same condition the catalog model tests on its
    incidence backplane.
    """
    geometry = _model([_titan(phase_angle=0.5)]).geometry_inputs
    assert geometry.axis_degenerate
    assert geometry.theta_rad == 0.0


def test_filters_are_empty_for_a_simulated_frame() -> None:
    """A simulated observation carries no filter names."""
    assert _model([_titan()]).geometry_inputs.filters == ()


def test_subject_range_comes_from_the_scene() -> None:
    """The scene's range_km becomes the feature's subject range."""
    assert _model([_titan()]).geometry_inputs.subject_range_km == pytest.approx(1.2e6)


def test_missing_range_is_infinite_not_zero() -> None:
    """A body with no stated range is honestly at unknown distance."""
    body = _titan()
    del body['range_km']
    assert math.isinf(_model([body]).geometry_inputs.subject_range_km)


def test_envelope_bbox_contains_the_envelope() -> None:
    """The reported bbox brackets the envelope disc in extfov coordinates."""
    geometry = _model([_titan()]).geometry_inputs
    v_min, u_min, v_max, u_max = geometry.bbox_extfov_vu
    center_v, center_u = geometry.predicted_center_vu
    assert v_min <= center_v - geometry.r_env_px
    assert v_max >= center_v + geometry.r_env_px
    assert u_min <= center_u - geometry.r_env_px
    assert u_max >= center_u + geometry.r_env_px


# ---------------------------------------------------------------------------
# Contaminant mask.
# ---------------------------------------------------------------------------


def _sibling(center: float, *, range_km: float) -> dict[str, Any]:
    """A small moon beside Titan at an explicit range."""
    return {
        'name': 'MOON',
        'center_v': center,
        'center_u': center,
        'axis1': 20.0,
        'axis2': 20.0,
        'axis3': 20.0,
        'illumination_angle': 90.0,
        'phase_angle': 45.0,
        'range_km': range_km,
    }


def test_lone_titan_masks_nothing() -> None:
    """A frame with nothing but Titan carries no contaminant mask."""
    assert _model([_titan()]).geometry_inputs.contaminant_mask is None


def test_sibling_beside_the_limb_enters_the_mask() -> None:
    """A moon beside the disc is masked even though it hides nothing.

    Its visible sliver sits in the symmetry annulus and in the arc rays, so
    the fits must ignore it; range order is deliberately not consulted.
    """
    geometry = _model([_titan(), _sibling(40.0, range_km=2.0e6)]).geometry_inputs
    assert geometry.contaminant_mask is not None
    assert bool(geometry.contaminant_mask.any())


def test_farther_sibling_does_not_count_as_occlusion() -> None:
    """A moon behind Titan contributes to the mask but not to occlusion."""
    geometry = _model([_titan(), _sibling(40.0, range_km=2.0e6)]).geometry_inputs
    assert geometry.occluded_fraction == 0.0


def test_nearer_sibling_counts_as_occlusion() -> None:
    """A moon in front of Titan hides part of the envelope."""
    geometry = _model([_titan(), _sibling(_CENTER, range_km=0.5e6)]).geometry_inputs
    assert geometry.occluded_fraction > 0.0


def test_bright_star_contributes_a_masked_disc() -> None:
    """A star brighter than the mask limit is masked at its catalog position."""
    star = {'name': 'BRIGHT', 'v': 40.0, 'u': 40.0, 'vmag': 5.0}
    geometry = _model([_titan()], [star]).geometry_inputs
    assert geometry.contaminant_mask is not None
    margin_v = 50
    assert bool(geometry.contaminant_mask[int(40.0) + margin_v, int(40.0) + margin_v])


def test_star_disc_covers_the_pixels_the_star_lit() -> None:
    """The masked disc is centered on the light the renderer gave the star.

    The rendered star is the anchor: its light is symmetric about the row and
    the column it centroids to, and the disc that hides it has to be centered
    on the same place.  A scene states a star's position as a pixel corner
    and the disc is painted against pixel centric grids, so a mask painted at
    the stated number sits half a pixel off the light it is there to cover,
    and that shows up here as a disagreement with the measurement.
    """
    star = {'name': 'BRIGHT', 'v': _STAR_V, 'u': _STAR_U, 'vmag': 5.0}
    obs = _obs([_titan()], [star])
    image = _rendered_image(obs)
    measured = _rendered_star_center_vu(image, (int(_STAR_V), int(_STAR_U)))
    mask = _model_for(obs).geometry_inputs.contaminant_mask
    assert mask is not None
    vs, us = np.nonzero(mask)
    assert _mirror_residual(image, measured, reach_px=_STAR_REACH_PX, axis=0) == 0.0
    assert _mirror_residual(image, measured, reach_px=_STAR_REACH_PX, axis=1) == 0.0
    assert float(vs.mean()) - float(obs.extfov_margin_v) == pytest.approx(measured[0])
    assert float(us.mean()) - float(obs.extfov_margin_u) == pytest.approx(measured[1])


def test_faint_star_is_left_unmasked() -> None:
    """A star fainter than the limit is deliberately not masked.

    Faint point sources are a few pixels against thousands of mirror pairs;
    the planted-truth campaign is what verifies the policy is safe.
    """
    star = {'name': 'FAINT', 'v': 40.0, 'u': 40.0, 'vmag': 12.0}
    assert _model([_titan()], [star]).geometry_inputs.contaminant_mask is None


def test_mask_is_extended_frame_shaped() -> None:
    """The mask ships on the extended frame so fitting needs no box origin."""
    geometry = _model([_titan(), _sibling(40.0, range_km=2.0e6)]).geometry_inputs
    assert geometry.contaminant_mask is not None
    assert geometry.contaminant_mask.shape == geometry.extfov_shape_vu


# ---------------------------------------------------------------------------
# Inherited emission: the sim path must mean the same thing as the real one.
# ---------------------------------------------------------------------------


def test_model_is_a_catalog_model_subclass() -> None:
    """The simulated model reuses the catalog model's emission wholesale.

    Only how the geometry dataclass is obtained differs; inheriting the rest
    is what keeps a simulated feature's meaning identical to a real one's.
    """
    assert issubclass(NavModelTitanSimulated, NavModelTitan)


def test_emits_exactly_one_titan_limb_feature() -> None:
    """The simulated model emits the single haze feature, as the real one does."""
    model = _model([_titan()])
    model.create_model()
    features = model.to_features(None)
    assert len(features) == 1
    assert features[0].feature_type is NavFeatureType.TITAN_LIMB


def test_emitted_geometry_carries_the_haze_payload() -> None:
    """The feature's geometry is the haze payload the technique consumes."""
    model = _model([_titan()])
    model.create_model()
    assert isinstance(model.to_features(None)[0].geometry, TitanHazeGeometry)


def test_well_resolved_titan_clears_the_reliability_gate() -> None:
    """A large, unoccluded, fully framed simulated Titan is usable."""
    model = _model([_titan()])
    model.create_model()
    assert model.to_features(None)[0].reliability > 0.3


def test_tiny_titan_is_hard_zero_reliability() -> None:
    """Below the minimum envelope diameter the feature is emitted but gated.

    Emit-then-gate is the sanctioned terminal state: the frame resolves
    through the standard statuses with an attributing record, rather than
    vanishing.
    """
    body = _titan(axis1=20.0, axis2=20.0, axis3=20.0, km_per_pixel=2575.0 / 10.0)
    model = _model([body])
    model.create_model()
    features = model.to_features(None)
    assert len(features) == 1
    assert features[0].reliability == 0.0


def test_create_model_records_geometry_metadata() -> None:
    """``create_model`` publishes the same metadata the catalog model does."""
    model = _model([_titan()])
    model.create_model()
    assert model.metadata['body'] == 'TITAN'
    assert model.metadata['envelope_diameter_px'] == pytest.approx(
        2.0 * model.geometry_inputs.r_env_px
    )


def test_annotations_paint_the_overlay() -> None:
    """The inherited overlay renders for a simulated frame."""
    model = _model([_titan()])
    model.create_model()
    annotations = model.to_annotations(None)
    assert len(annotations.annotations) == 1


def test_geometry_is_computed_once_and_cached() -> None:
    """Repeated access returns the same evaluated geometry object."""
    model = _model([_titan()])
    assert model.geometry_inputs is model.geometry_inputs


def test_the_catalog_geometry_entry_point_is_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulated path reads operator parameters, never ``oops`` geometry.

    A simulated observation carries no usable backplane, so a model that
    reached for one would degrade to a hard-zero feature rather than raise,
    and the omission would be invisible.  The guard is direct: the geometry
    still evaluates while the catalog-driven entry point is replaced by one
    that fails loudly.
    """

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the simulated model must not call geometry_from_obs')

    monkeypatch.setattr('spindoctor.nav_model.nav_model_titan.geometry_from_obs', _fail)
    assert _model([_titan()]).geometry_inputs.r_env_px > 0.0


def test_window_matches_the_observation_search_margin() -> None:
    """The search half-window is the larger extfov margin, as on a real frame."""
    obs = _obs([_titan()])
    geometry = _model([_titan()]).geometry_inputs
    assert geometry.window_px == float(max(obs.extfov_margin_v, obs.extfov_margin_u))


def test_extfov_shape_matches_the_observation() -> None:
    """The geometry reports the observation's own extended-frame shape."""
    obs = _obs([_titan()])
    geometry = _model([_titan()]).geometry_inputs
    assert geometry.extfov_shape_vu == (
        int(obs.extdata_shape_vu[0]),
        int(obs.extdata_shape_vu[1]),
    )


def test_a_sibling_near_the_frame_origin_enters_the_mask() -> None:
    """A moon in the corner of the data frame still masks pixels.

    This asserts the mask the model built is non-empty and measures nothing
    rendered.
    """
    geometry = _model([_titan(), _sibling(10.0, range_km=2.0e6)]).geometry_inputs
    assert geometry.contaminant_mask is not None
    assert np.count_nonzero(geometry.contaminant_mask) > 0
