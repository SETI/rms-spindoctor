"""The simulated body model's limb arc fractions and sibling-body occlusion.

An arc's visible fraction counts the vertices that survive the frame and any nearer
sibling body, against the whole predicted arc.  A nearer sibling hides limb and
terminator vertices and the disc template's pixels; a farther one, or one without a
range, hides nothing.
"""

from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context
from tests.spindoctor.nav_model.simulated.test_body import (
    _LIMB_SIZE,
    _large_body,
    _limb_obs,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim

# ---------------------------------------------------------------------------
# Limb visible-arc fraction and sibling-body occlusion
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
    surviving portion, not 1.0, because BodyLimbNav's
    visible_limb_arc_fraction confidence term reads it.
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
        'semi_a_px': 50.0,
        'semi_b_px': 50.0,
        'semi_c_px': 50.0,
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
    occluder = _near_sibling(
        center_u=_LIMB_SIZE / 2.0 + 8.0, semi_a_px=75.0, semi_b_px=75.0, semi_c_px=75.0
    )
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
            'random_seed': 1,
            'bodies': [far, near],
        },
    )
    models = NavModelBodySimulated.instances_for_obs(obs)
    assert len(models) == 2
    for model in models:
        assert isinstance(model, NavModelBodySimulated)
        assert len(model._sibling_bodies) == 1
        assert model._sibling_bodies[0]['name'] != model._nav_body['name']
