"""Pose-disagreement behavioral test for an irregular body.

A chaotic rotator's orientation is genuinely unknown, so the useful question is
what the technique ladder does when the navigator's assumed pose is wrong.  This
test renders one irregular body at its true pose and navigates it twice: once
with the predicted pose agreeing with the render (the correct-pose body) and once
with the predicted body given a wrong in-plane roll (the wrong-pose body).  It
asserts the *decision*, per technique, not an exact recovery:

* The orientation-dependent ``BodyLimbNav`` recovers the planted offset at the
  correct pose but lands far off at the wrong pose -- a wrong-pose limb the
  navigator should not trust (here it also self-flags spurious).
* The orientation-free ``BodyBlobNav`` stays accurate at both poses, because the
  body's relief is low: its bulk shape is close to its triaxial ellipsoid, which
  is centrally symmetric, so its lit-weighted centroid barely moves under
  rotation.

A wrong in-plane roll is used because it swings the elongated body's silhouette
the most for the least centroid motion, giving the cleanest separation between
the limb (which depends on the silhouette) and the blob (which does not).  The
assertion is per-technique because the ensemble's demote-to-pose-free choice is
confidence-driven and the confidence alphas are uncalibrated placeholders; the
geometry/error gap asserted here is real and placeholder-independent.  The fused
"prefers the pose-free answer" claim only becomes clean after the real-data
confidence calibration.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_SCENE = (
    Path(__file__).parent
    / 'sim_scenes'
    / 'phase_sweep_irregular_body'
    / 'hyperion_pose_disagree.yaml'
)
# The wrong-pose body predicts the mesh with a 20-degree in-plane roll (the Z
# Euler angle) added to the true pose [10, 35, 0].
_WRONG_POSE_ROLL_DEG = 20.0


def _navigate(sim_params: dict[str, Any], technique: str) -> Any:
    obs = ObsSim.from_file('/tmp/pose.json', sim_params=sim_params)
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=technique
    )
    return orchestrator.navigate(obs)


def _technique_error_px(sim_params: dict[str, Any], technique: str) -> float:
    """Return the technique's own offset error against the planted offset.

    Reads the per-technique offset directly (including a spurious flag) because
    the wrong-pose limb's *geometry* error is the quantity under test, not whether
    the placeholder-alpha confidence happened to keep it in the fused result.
    """
    result = _navigate(sim_params, technique)
    matches = [t for t in result.per_technique if t.technique_name == technique]
    assert matches, f'{technique} produced no technique result'
    offset = matches[0].offset_px
    assert offset is not None
    planted_v = float(sim_params['offset_v'])
    planted_u = float(sim_params['offset_u'])
    return math.hypot(offset[0] - planted_v, offset[1] - planted_u)


def _params(*, wrong_pose: bool) -> dict[str, Any]:
    params = load_sim_scene(_SCENE)
    if wrong_pose:
        params = copy.deepcopy(params)
        params['bodies'][0]['nav_override']['pose_euler_deg'] = [10.0, 35.0, _WRONG_POSE_ROLL_DEG]
    return params


def test_correct_pose_limb_is_accurate() -> None:
    """At the true predicted pose the limb recovers the planted offset."""
    assert _technique_error_px(_params(wrong_pose=False), 'BodyLimbNav') < 1.0


def test_wrong_pose_limb_degrades() -> None:
    """A wrong in-plane roll drives the limb far off the planted offset."""
    assert _technique_error_px(_params(wrong_pose=True), 'BodyLimbNav') > 4.0


def test_blob_stays_accurate_at_correct_pose() -> None:
    """The pose-free blob recovers the planted offset at the true pose."""
    assert _technique_error_px(_params(wrong_pose=False), 'BodyBlobNav') < 1.0


def test_blob_stays_accurate_at_wrong_pose() -> None:
    """The pose-free blob holds near the planted offset despite the wrong pose."""
    assert _technique_error_px(_params(wrong_pose=True), 'BodyBlobNav') < 2.0


def test_wrong_pose_limb_much_worse_than_blob() -> None:
    """The wrong-pose limb error dwarfs the blob error -- the demote-worthy gap."""
    limb = _technique_error_px(_params(wrong_pose=True), 'BodyLimbNav')
    blob = _technique_error_px(_params(wrong_pose=True), 'BodyBlobNav')
    assert limb > 3.0 * blob
