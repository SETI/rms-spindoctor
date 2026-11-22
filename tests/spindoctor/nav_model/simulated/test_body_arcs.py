"""Limb and terminator arcs of the navigator-side simulated body model.

The limb is the lit part of the predicted silhouette boundary and the terminator the
interior lit/unlit boundary.  These tests cover which arcs a body emits, their
geometry and lit region, and the terminator's reliability and visible-arc fraction.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context
from tests.spindoctor.nav_model.simulated.body_fixtures import (
    LIMB_SIZE,
    body_feature_types,
    large_body,
    limb_obs,
)

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import LimbPolyline, TerminatorPolyline
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX


def test_large_low_phase_body_emits_limb_arc() -> None:
    """A well-resolved low-phase body emits a LIMB_ARC feature."""
    assert 'LIMB_ARC' in body_feature_types(limb_obs(), large_body())


def test_small_body_emits_no_limb_arc() -> None:
    """A body below the limb-resolution floor emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in body_feature_types(
        limb_obs(), large_body(semi_a_px=20.0, semi_b_px=20.0, semi_c_px=20.0)
    )


def test_high_phase_body_emits_no_limb_arc() -> None:
    """A high-phase body (mostly terminator) emits no LIMB_ARC."""
    assert 'LIMB_ARC' not in body_feature_types(limb_obs(), large_body(phase_angle=120.0))


def test_limb_arc_polyline_has_vertices_and_unit_normals() -> None:
    """The emitted LIMB_ARC carries a vertex polyline with unit outward normals."""
    from spindoctor.feature.geometry import LimbPolyline

    obs = limb_obs()
    model = NavModelBodySimulated('body', obs, 'RHEA', large_body())
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
    obs = limb_obs()
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
    vertices = _ungated_limb_vertices(large_body(phase_angle=120.0))
    assert vertices.shape[0] > 0


def test_gates_off_small_body_emits_limb_arc() -> None:
    """With the gates off, a body below the resolution floor still emits."""
    vertices = _ungated_limb_vertices(large_body(semi_a_px=20.0, semi_b_px=20.0, semi_c_px=20.0))
    assert vertices.shape[0] > 0


def test_gates_off_limb_is_geometric_not_terminator() -> None:
    """Gates-off vertices trace the geometric limb, not the lit-region boundary.

    For a phase-120 sphere the lit-region boundary includes the terminator,
    which cuts through the disc interior; the lit geometric limb keeps every
    vertex on the silhouette outline (constant radius from the center).
    """
    vertices = _ungated_limb_vertices(large_body(phase_angle=120.0))
    obs = limb_obs()
    cv = LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    cu = LIMB_SIZE / 2.0 + int(obs.extfov_margin_u)
    radii = np.hypot(vertices[:, 0] - cv, vertices[:, 1] - cu)
    assert float(radii.min()) > 62.0


def _limb_vertex_incidence_cosines(body_params: dict[str, Any], vertices: np.ndarray) -> np.ndarray:
    """The sun's incidence cosine on the sphere's limb at each vertex's direction.

    On the limb the surface normal lies in the image plane along the direction from
    the body center, so the cosine is that direction against the in-plane light.
    """
    obs = limb_obs()
    cv = LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    cu = LIMB_SIZE / 2.0 + int(obs.extfov_margin_u)
    dv = vertices[:, 0] + PIXEL_CENTER_TO_CORNER_PX - cv
    du = vertices[:, 1] + PIXEL_CENTER_TO_CORNER_PX - cu
    light_v, light_u, _ = illumination_vector(
        illumination_angle=math.radians(float(body_params['illumination_angle'])),
        phase_angle=math.radians(float(body_params['phase_angle'])),
    )
    return np.asarray((dv * light_v + du * light_u) / np.hypot(dv, du))


# How far below 0 a lit limb vertex's in-plane cosine may fall.  A limb pixel lies a
# little inside the outline, where the surface tilts toward the observer, so below 90
# deg phase the sun reaches it slightly past the limb's own terminator.  The unlit limb
# runs on to -sin(phase), -0.87 at both phases tested.
_LIMB_LIT_SLACK = -0.15


def test_gates_off_limb_keeps_only_lit_vertices() -> None:
    """At 120 deg phase no gates-off limb vertex lies on the unlit side."""
    body = large_body(phase_angle=120.0)
    vertices = _ungated_limb_vertices(body)
    assert float(_limb_vertex_incidence_cosines(body, vertices).min()) > _LIMB_LIT_SLACK


def test_navigation_limb_keeps_only_lit_vertices() -> None:
    """At 60 deg phase, the most the limb gate allows, every limb vertex is lit."""
    body = large_body(phase_angle=60.0)
    obs = limb_obs()
    model = NavModelBodySimulated('body', obs, body['name'], body)
    model.create_model()
    limb = next(
        f for f in model.to_features(bare_nav_context(obs)) if f.feature_type.name == 'LIMB_ARC'
    )
    assert isinstance(limb.geometry, LimbPolyline)
    vertices = np.asarray(limb.geometry.vertices_vu)
    assert float(_limb_vertex_incidence_cosines(body, vertices).min()) > _LIMB_LIT_SLACK


_LOW_PHASE_SPHERE = {'semi_a_px': 60.0, 'semi_b_px': 60.0, 'semi_c_px': 60.0, 'phase_angle': 10.0}


def _mean_incidence_lit_limb(obs: ObsSim) -> set[tuple[int, int]]:
    """The low-phase sphere's lit limb pixels by the mean-incidence rule, in extfov.

    The rule is computed here from the sphere's own geometry on a 4 x 4 subpixel
    grid: a silhouette pixel is lit when its angle of incidence, averaged over its
    subpixels on the body, is below 90 deg.
    """
    sub = 4
    centers = (np.arange(LIMB_SIZE * sub) + 0.5) / sub - LIMB_SIZE / 2.0
    cv, cu = np.meshgrid(centers, centers, indexing='ij')
    r2 = (cv**2 + cu**2) / 60.0**2
    inside = r2 < 1.0
    nz = np.sqrt(np.maximum(1.0 - r2, 0.0))
    norm = np.sqrt(cv**2 + cu**2 + (60.0 * nz) ** 2)
    light_v, light_u, light_z = illumination_vector(
        illumination_angle=math.radians(25.0), phase_angle=math.radians(10.0)
    )
    cos_i = (cv * light_v + cu * light_u + 60.0 * nz * light_z) / norm
    angle = np.where(inside, np.arccos(np.clip(cos_i, -1.0, 1.0)), 0.0)
    blocks = (LIMB_SIZE, sub, LIMB_SIZE, sub)
    count = inside.reshape(blocks).sum(axis=(1, 3))
    lit = (count > 0) & (angle.reshape(blocks).sum(axis=(1, 3)) < 0.5 * math.pi * count)
    disc = count > 0
    padded = np.pad(disc, 1)
    boundary = disc & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    vs, us = np.nonzero(boundary & lit)
    margin_v, margin_u = int(obs.extfov_margin_v), int(obs.extfov_margin_u)
    return {(int(v) + margin_v, int(u) + margin_u) for v, u in zip(vs, us, strict=True)}


def _vertex_set(vertices_vu: np.ndarray) -> set[tuple[int, int]]:
    """A polyline's vertices as a set of pixel indices."""
    return {(int(v), int(u)) for v, u in np.asarray(vertices_vu)}


def test_limb_lit_region_is_the_mean_incidence_rule() -> None:
    """The limb keeps the outline pixels whose mean incidence is below 90 deg, no others.

    The SPICE-backed body model counts a pixel lit when its angle of incidence,
    averaged over the pixel's subpixels on the body, is below 90 deg.  At 10 deg phase
    on a 60 px sphere that differs from counting a pixel lit whenever any subpixel is
    lit by 37 of the limb pixels on the unlit side (223 against 260).
    """
    obs = limb_obs()
    limb = _limb_feature(obs, large_body(**_LOW_PHASE_SPHERE))
    assert limb is not None
    assert _vertex_set(limb.geometry.vertices_vu) == _mean_incidence_lit_limb(obs)


def test_gates_off_limb_lit_region_is_the_mean_incidence_rule() -> None:
    """The gates-off limb keeps the same outline pixels as the mean-incidence rule."""
    vertices = _ungated_limb_vertices(large_body(**_LOW_PHASE_SPHERE))
    assert _vertex_set(vertices) == _mean_incidence_lit_limb(limb_obs())


def test_low_phase_limb_fraction_counts_lit_pixels_on_both_sides() -> None:
    """The fully framed low-phase limb sees its whole lit outline.

    The visible-arc fraction divides the surviving lit vertices by the lit outline
    of an unclipped render; counting that outline by a different lit rule would put
    it at 0.86.
    """
    limb = _limb_feature(limb_obs(), large_body(**_LOW_PHASE_SPHERE))
    assert limb is not None
    assert float(limb.flags.visible_arc_fraction) == pytest.approx(1.0, abs=0.02)


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
    assert 'TERMINATOR_ARC' in body_feature_types(limb_obs(), large_body(phase_angle=90.0))


def test_zero_phase_body_emits_no_terminator_arc() -> None:
    """A fully-lit (zero-phase) body has no terminator and emits none."""
    assert 'TERMINATOR_ARC' not in body_feature_types(limb_obs(), large_body(phase_angle=0.0))


def test_near_zero_phase_body_emits_no_terminator_arc() -> None:
    """Below the phase-factor floor the terminator is indistinct from the limb."""
    assert 'TERMINATOR_ARC' not in body_feature_types(limb_obs(), large_body(phase_angle=1.0))


def test_highly_irregular_resolved_body_emits_no_terminator_arc() -> None:
    """A resolved highly_irregular body suppresses TERMINATOR_ARC.

    The shared ``shape_features_suppressed`` policy: a chaotic rotator's
    rendered terminator does not match the real body once it is resolved,
    so the SPICE-backed model emits none -- and the simulated model must
    not either.
    """
    assert 'TERMINATOR_ARC' not in body_feature_types(
        limb_obs(), large_body(name='HYPERION', phase_angle=90.0)
    )


def test_regular_body_same_geometry_emits_terminator_arc() -> None:
    """The identical geometry on a regular body still emits TERMINATOR_ARC."""
    assert 'TERMINATOR_ARC' in body_feature_types(
        limb_obs(), large_body(name='RHEA', phase_angle=90.0)
    )


def test_terminator_arc_geometry_is_terminator_polyline() -> None:
    """The emitted TERMINATOR_ARC carries a TerminatorPolyline with enough vertices."""
    from spindoctor.feature.geometry import TerminatorPolyline

    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
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
    obs = limb_obs()
    terminator = _terminator_feature(obs, large_body(phase_angle=90.0))
    assert terminator is not None
    vertices = np.asarray(terminator.geometry.vertices_vu)
    cv = LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    cu = LIMB_SIZE / 2.0 + int(obs.extfov_margin_u)
    radii = np.hypot(vertices[:, 0] - cv, vertices[:, 1] - cu)
    disc_radius = 130.0 / 2.0
    assert float(radii.min()) < 0.5 * disc_radius
    assert float(radii.mean()) < 0.85 * disc_radius


def test_terminator_arc_normals_are_unit_length() -> None:
    """The terminator polyline carries unit outward normals."""
    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
    assert terminator is not None
    normals = np.asarray(terminator.geometry.normals_vu)
    norms = np.hypot(normals[:, 0], normals[:, 1])
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_terminator_arc_stays_provisional_via_flags() -> None:
    """The terminator carries its phase-angle factor for the (uncalibrated) technique."""
    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
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

    from spindoctor.nav_model.nav_model_body import terminator_reliability

    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
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
    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
    assert terminator is not None
    assert terminator.reliability < 1.0


def test_terminator_arc_fraction_fully_framed_is_near_one() -> None:
    """A fully framed terminator sees almost all of the unclipped ridge."""
    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
    assert terminator is not None
    assert terminator.flags.visible_arc_fraction > 0.95


def test_terminator_arc_fraction_drops_when_frame_clips_it() -> None:
    """A body whose terminator runs off the frame edge scores a lower fraction.

    Center the body near the frame edge so part of the lit/unlit boundary
    falls outside the render; the visible-arc fraction must report the
    surviving portion, not 1.0, because BodyTerminatorNav's confidence
    reads it.
    """
    clipped = _terminator_feature(limb_obs(), large_body(phase_angle=90.0, center_v=20.0))
    assert clipped is not None
    assert clipped.flags.visible_arc_fraction < 0.9
    assert clipped.flags.visible_arc_fraction > 0.2


def test_terminator_reliability_reasons_carry_albedo_penalty() -> None:
    """The reliability breakdown mirrors the real model's terminator fields."""

    terminator = _terminator_feature(limb_obs(), large_body(phase_angle=90.0))
    assert terminator is not None
    shape = load_body_shape('RHEA')
    assert terminator.reliability_reasons.albedo_penalty == pytest.approx(
        min(1.0, shape.albedo_variation)
    )


@pytest.mark.parametrize(
    ('feature_type', 'id_prefix', 'geometry_cls', 'sigma_normal_px', 'sigma_tangent_px'),
    [
        (NavFeatureType.LIMB_ARC, 'limb_arc', LimbPolyline, 1.0, 0.5),
        (NavFeatureType.TERMINATOR_ARC, 'terminator_arc', TerminatorPolyline, 2.0, 0.5),
    ],
    ids=['limb', 'terminator'],
)
def test_each_arc_carries_its_own_family_choices(
    feature_type: NavFeatureType,
    id_prefix: str,
    geometry_cls: type[LimbPolyline] | type[TerminatorPolyline],
    sigma_normal_px: float,
    sigma_tangent_px: float,
) -> None:
    """Each arc carries its own ID prefix, geometry class, per-vertex sigmas, and type."""
    obs = limb_obs()
    model = NavModelBodySimulated('body', obs, 'RHEA', large_body(phase_angle=45.0))
    model.create_model()
    feature = next(
        f for f in model.to_features(bare_nav_context(obs)) if f.feature_type is feature_type
    )
    geometry = feature.geometry
    assert feature.feature_id == f'{id_prefix}:RHEA'
    assert type(geometry) is geometry_cls
    assert isinstance(geometry, LimbPolyline | TerminatorPolyline)
    n = geometry.vertices_vu.shape[0]
    assert np.array_equal(geometry.sigma_normal_per_vertex_px, np.full(n, sigma_normal_px))
    assert np.array_equal(geometry.sigma_tangent_per_vertex_px, np.full(n, sigma_tangent_px))
    assert feature.usable_types == frozenset({feature_type})
