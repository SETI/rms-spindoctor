"""Behavioral checks for the mutual-event scene class.

A grazing two-body overlap must still navigate to its planted offset within its
truth tolerance, and the navigator-side simulated body model must predict
occlusion-aware limbs: the near body's limb stays complete, while the far body's
limb loses the arc hidden behind the near body's silhouette and its
``visible_arc_fraction`` reports the loss honestly (the input the limb reliability
and BodyLimbNav's confidence formula consume).  The deeper overlaps' fused outcomes
are pinned by their scenes' ``expected`` blocks (test_sim_expected), their truth
tolerances (test_sim_truth_checks), and their baselines.

Everything renders and navigates in-process (no holdings or SPICE), so this
runs in the default suite.
"""

from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_model import build_models_for_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene
from tests.integration.sim_expected import navigate_scene
from tests.integration.sim_truth import FUSED, assert_matches_truth
from tests.shims.context import bare_nav_context

_SCENES_DIR = Path(__file__).parent / 'sim_scenes' / 'mutual_event'


def _load(scene_name: str) -> dict[str, Any]:
    return load_sim_scene(_SCENES_DIR / f'{scene_name}.yaml')


def test_grazing_overlap_navigates_within_tolerance() -> None:
    """The grazing mutual event recovers its planted offset within its truth tolerance."""
    scene = _load('mutual_grazing')
    result = navigate_scene(scene)
    assert result.status == 'success'
    assert_matches_truth(result.offset_px, scene, FUSED)


def test_navigator_predicts_occlusion_aware_limbs() -> None:
    """The far body's limb loses its hidden arc; the near body's stays whole.

    ``NavModelBodySimulated`` predicts each body from its own idealized
    params but knows its siblings' idealized geometry too (both are
    catalog/SPICE-class knowledge), so the arc of FAR's limb behind NEAR's
    silhouette is dropped from the polyline and ``visible_arc_fraction``
    reports the surviving share -- at this ~206-degree occlusion roughly
    0.4 of the full circle -- instead of claiming 1.0.
    """
    scene = _load('mutual_deep')
    obs = ObsSim.from_file('/tmp/mutual_event.yaml', sim_params=scene)
    context = bare_nav_context(obs)
    limb_features = {}
    for model in build_models_for_obs(obs):
        model.create_model()
        for feature in model.to_features(context):
            if feature.feature_type is NavFeatureType.LIMB_ARC:
                limb_features[feature.feature_id] = feature
    assert set(limb_features) == {'limb_arc:FAR', 'limb_arc:NEAR'}

    near = limb_features['limb_arc:NEAR']
    assert near.flags.visible_arc_fraction > 0.9  # type: ignore[union-attr]
    vertices = near.geometry.vertices_vu  # type: ignore[union-attr]
    dv = vertices[:, 0] - (128.0 + obs.extfov_margin_v)
    du = vertices[:, 1] - (110.0 + obs.extfov_margin_u)
    quadrants = {(bool(a), bool(b)) for a, b in zip(dv >= 0, du >= 0, strict=True)}
    assert quadrants == {(False, False), (False, True), (True, False), (True, True)}, (
        f'NEAR limb does not surround its center: {sorted(quadrants)}'
    )
    # The near limb is a closed circle: azimuth coverage has no gap larger
    # than a few degrees.
    azimuths = np.sort(np.arctan2(du, dv))
    gaps = np.diff(np.concatenate([azimuths, [azimuths[0] + 2.0 * np.pi]]))
    assert float(np.degrees(gaps.max())) < 10.0

    far = limb_features['limb_arc:FAR']
    far_fraction = float(far.flags.visible_arc_fraction)  # type: ignore[union-attr]
    assert far_fraction < 0.55
    assert far_fraction > 0.25
    # No surviving FAR vertex sits inside NEAR's silhouette (radius 65 px
    # at center (128, 110)).
    far_vertices = far.geometry.vertices_vu  # type: ignore[union-attr]
    distance_from_near = np.hypot(
        far_vertices[:, 0] - (128.0 + obs.extfov_margin_v),
        far_vertices[:, 1] - (110.0 + obs.extfov_margin_u),
    )
    assert float(distance_from_near.min()) > 64.0
