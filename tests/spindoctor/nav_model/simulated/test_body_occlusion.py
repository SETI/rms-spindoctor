"""The simulated body model's limb arc fractions and sibling-body occlusion.

An arc's visible fraction counts the vertices that survive the frame and any nearer
sibling body, against the whole predicted arc.  A nearer sibling hides limb and
terminator vertices and the disc template's pixels; a farther one, or one without a
range, hides nothing.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.shims import bare_nav_context
from tests.spindoctor.nav_model.simulated.body_fixtures import (
    LIMB_SIZE,
    large_body,
    limb_feature,
    limb_obs,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_model.simulated.body import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.support.types import NDArrayFloatType

# ---------------------------------------------------------------------------
# Limb visible-arc fraction and sibling-body occlusion
# ---------------------------------------------------------------------------


def test_limb_arc_fraction_fully_framed_is_near_one() -> None:
    """A fully framed, unoccluded limb sees almost all of the silhouette."""
    limb = limb_feature(limb_obs(), large_body())
    assert limb is not None
    assert limb.flags.visible_arc_fraction > 0.9


def test_limb_arc_fraction_drops_when_frame_clips_it() -> None:
    """A body sliding off the frame edge scores a lower limb fraction.

    Center the body near the frame edge so part of the silhouette boundary
    falls outside the render; the visible-arc fraction must report the
    surviving portion, not 1.0, because BodyLimbNav's
    visible_limb_arc_fraction confidence term reads it.
    """
    clipped = limb_feature(limb_obs(), large_body(center_v=20.0))
    assert clipped is not None
    assert clipped.flags.visible_arc_fraction < 0.85
    assert clipped.flags.visible_arc_fraction > 0.2


def test_limb_reliability_matches_shared_formula() -> None:
    """The sim limb scores through the shared body-model reliability."""
    from spindoctor.nav_model.nav_model_body import limb_reliability

    limb = limb_feature(limb_obs(), large_body())
    assert limb is not None
    expected = limb_reliability(
        visible_arc_fraction=float(limb.flags.visible_arc_fraction),
        visible_arc_px=float(limb.geometry.vertices_vu.shape[0]),
    )
    assert limb.reliability == pytest.approx(expected)


def _near_sibling(**overrides: Any) -> dict[str, Any]:
    """A nearer sibling sphere overlapping the large_body silhouette."""
    params = {
        'name': 'DIONE',
        'center_v': LIMB_SIZE / 2.0,
        'center_u': LIMB_SIZE / 2.0 + 55.0,
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
    body = large_body(range_km=700000.0)
    unoccluded = limb_feature(limb_obs(), body)
    occluded = limb_feature(limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert unoccluded is not None
    assert occluded is not None
    assert occluded.flags.visible_arc_fraction < unoccluded.flags.visible_arc_fraction - 0.1
    assert occluded.reliability < unoccluded.reliability


def test_occluded_limb_vertices_leave_the_polyline() -> None:
    """No surviving limb vertex sits inside the nearer sibling's silhouette.

    A hidden arc has no counterpart edge in the image, so its vertices are
    dropped rather than left for the robust fit to reject.
    """
    body = large_body(range_km=700000.0)
    occluded = limb_feature(limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert occluded is not None
    obs = limb_obs()
    vertices = np.asarray(occluded.geometry.vertices_vu)
    sib_v = LIMB_SIZE / 2.0 + int(obs.extfov_margin_v)
    sib_u = LIMB_SIZE / 2.0 + 55.0 + int(obs.extfov_margin_u)
    distances = np.hypot(vertices[:, 0] - sib_v, vertices[:, 1] - sib_u)
    assert float(distances.min()) > 49.0


def test_farther_sibling_does_not_occlude() -> None:
    """A sibling with a larger range hides nothing of this body's limb."""
    body = large_body(range_km=500000.0)
    behind = limb_feature(limb_obs(), body, sibling_bodies=[_near_sibling(range_km=700000.0)])
    alone = limb_feature(limb_obs(), body)
    assert behind is not None
    assert alone is not None
    assert behind.flags.visible_arc_fraction == pytest.approx(alone.flags.visible_arc_fraction)


def test_sibling_without_range_does_not_occlude() -> None:
    """Without explicit ranges on both sides, stacking is unknowable.

    Mirrors the renderer's rule that overlapping bodies must all carry an
    explicit range_km before their depth order means anything.
    """
    body = large_body(range_km=700000.0)
    sibling = _near_sibling()
    del sibling['range_km']
    unknown = limb_feature(limb_obs(), body, sibling_bodies=[sibling])
    alone = limb_feature(limb_obs(), body)
    assert unknown is not None
    assert alone is not None
    assert unknown.flags.visible_arc_fraction == pytest.approx(alone.flags.visible_arc_fraction)


def test_deeply_occluded_limb_is_not_emitted() -> None:
    """A limb with fewer than 30 surviving vertices emits no LIMB_ARC."""
    body = large_body(range_km=700000.0)
    # A nearer sibling almost concentric with and larger than the body
    # hides nearly the whole silhouette boundary.
    occluder = _near_sibling(
        center_u=LIMB_SIZE / 2.0 + 8.0, semi_a_px=75.0, semi_b_px=75.0, semi_c_px=75.0
    )
    limb = limb_feature(limb_obs(), body, sibling_bodies=[occluder])
    assert limb is None


def test_occluded_terminator_fraction_drops() -> None:
    """Sibling occlusion also reduces the terminator's visible-arc fraction."""
    body = large_body(phase_angle=90.0, range_km=700000.0)
    obs = limb_obs()
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


def _sibling_distance(disc: NavFeature) -> NDArrayFloatType:
    """Each template pixel's distance from the center of the :func:`_near_sibling`.

    Parameters:
        disc: The BODY_DISC feature whose template the distances cover.

    Returns:
        The distance in pixels, one per template pixel.
    """
    v_min, u_min, _v_max, _u_max = disc.geometry.bbox_extfov_vu
    obs = limb_obs()
    rows, cols = np.asarray(disc.template_mask).shape
    vv, uu = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    sib_v = LIMB_SIZE / 2.0 + int(obs.extfov_margin_v) - v_min
    sib_u = LIMB_SIZE / 2.0 + 55.0 + int(obs.extfov_margin_u) - u_min
    distance: NDArrayFloatType = np.hypot(vv - sib_v, uu - sib_u)
    return distance


def test_occluded_disc_template_drops_hidden_pixels() -> None:
    """The BODY_DISC template mask drops what a nearer sibling hides and keeps the rest.

    The sibling is a radius-50 sphere, so no template pixel survives deep inside it,
    and beyond it the mask is the unoccluded body's.
    """
    body = large_body(range_km=700000.0)
    alone = _disc_feature(limb_obs(), body)
    disc = _disc_feature(limb_obs(), body, sibling_bodies=[_near_sibling()])
    template_mask = np.asarray(disc.template_mask, dtype=bool)
    distance = _sibling_distance(disc)
    clear = distance > 52.0
    assert int(np.count_nonzero(template_mask[distance < 40.0])) == 0
    np.testing.assert_array_equal(
        template_mask[clear], np.asarray(alone.template_mask, dtype=bool)[clear]
    )


def test_occluded_disc_template_img_zero_under_sibling() -> None:
    """The BODY_DISC template is dark deep inside a nearer sibling and unchanged beyond it.

    Deep inside the radius-50 sibling the target disc is fully hidden, so the brightness
    itself must be zero there, and not just the mask.
    """
    body = large_body(range_km=700000.0)
    alone = _disc_feature(limb_obs(), body)
    disc = _disc_feature(limb_obs(), body, sibling_bodies=[_near_sibling()])
    template_img = np.asarray(disc.template_img, dtype=np.float64)
    alone_img = np.asarray(alone.template_img, dtype=np.float64)
    distance = _sibling_distance(disc)
    clear = distance > 52.0
    assert float(np.max(template_img[distance < 40.0])) == 0.0
    assert float(np.max(alone_img[clear])) > 0.0
    np.testing.assert_array_equal(template_img[clear], alone_img[clear])


def test_occluded_disc_reliability_is_the_visible_fraction() -> None:
    """A nearer sibling lowers the BODY_DISC reliability to the share of the disc it leaves.

    The radius-50 sibling's center lies 55 px from the radius-65 body's, so it covers
    the lens where the two circles overlap, 31.8% of the body's disc.  The rendered
    pixel masks follow the circles to within 0.01.
    """
    radius, sibling_radius, separation = 65.0, 50.0, 55.0
    lens = (
        radius**2
        * math.acos((separation**2 + radius**2 - sibling_radius**2) / (2 * separation * radius))
        + sibling_radius**2
        * math.acos(
            (separation**2 + sibling_radius**2 - radius**2) / (2 * separation * sibling_radius)
        )
        - 0.5
        * math.sqrt(
            (-separation + radius + sibling_radius)
            * (separation + radius - sibling_radius)
            * (separation - radius + sibling_radius)
            * (separation + radius + sibling_radius)
        )
    )
    body = large_body(range_km=700000.0)
    alone = _disc_feature(limb_obs(), body)
    occluded = _disc_feature(limb_obs(), body, sibling_bodies=[_near_sibling()])
    assert alone.reliability == pytest.approx(1.0)
    assert occluded.reliability == pytest.approx(1.0 - lens / (math.pi * radius**2), abs=0.01)


def test_farther_sibling_leaves_disc_template_intact() -> None:
    """A sibling with a larger range does not trim the disc template."""
    body = large_body(range_km=500000.0)
    alone = _disc_feature(limb_obs(), body)
    behind = _disc_feature(limb_obs(), body, sibling_bodies=[_near_sibling(range_km=700000.0)])
    alone_pixels = int(np.count_nonzero(np.asarray(alone.template_mask, dtype=bool)))
    behind_pixels = int(np.count_nonzero(np.asarray(behind.template_mask, dtype=bool)))
    assert behind_pixels == alone_pixels


def test_instances_for_obs_wires_siblings() -> None:
    """Each per-body model instance receives the other bodies as siblings."""
    far = large_body(range_km=700000.0)
    near = _near_sibling()
    obs = ObsSim.from_file(
        '/tmp/mutual_sim.json',
        sim_params={
            'size_v': LIMB_SIZE,
            'size_u': LIMB_SIZE,
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
