"""Planted-offset algorithmic invariants (Phase T4).

Each scene under ``sim_scenes/algorithmic_invariants/`` carries a ground-truth
offset that is applied as the rendered offset.  A navigator predicting the
unshifted geometry must recover it.  These tests render each scene into an
ObsSim, navigate it, and assert success plus the scene's truth check: the error as a
distance in two dimensions, held to a tolerance measured over noise seeds plus a margin,
or to a band around a known error (:mod:`tests.integration.sim_truth`).  Unlike a
baseline, a truth check compares with the planted offset, so regenerating the
baselines cannot hide a regression.  Everything is in-process, so this runs in the
default suite.

Scenes split into three kinds:

* **Disc scenes** navigate by ``BodyDiscCorrelateNav`` and the *full ensemble*
  recovers the planted offset stably, so they exercise the generic full-ensemble
  recovery assertions.
* **Blob scenes** (``planted_offset_blob*``) are designed so ``BodyBlobNav`` --
  consuming the ``BODY_BLOB`` feature ``NavModelBodySimulated`` emits -- is the
  load-bearing technique.  Their recovery is asserted with ``only_techniques``
  pinned to the blob, which proves the simulated body's blob feature is produced
  and consumable and is deterministic.  The full ensemble on these scenes still
  navigates to success (a disc-spurious high-phase crescent falls to the blob),
  but its recovered offset sits near the weak fallback's limit and jitters
  across processes under parallel BLAS load, so the *tolerance* assertion is made
  blob-only rather than on the full ensemble.
* **Star scenes** (``planted_offset_star*``) carry no body or rings, so the
  ``STAR`` features ``NavModelStarsSimulated`` emits are the only signal and the
  star techniques (``StarFieldFromCatalogNav``, ``StarUniqueMatchNav``,
  ``StarRefineNav``) are the load-bearing ones.  The recovery is asserted on the
  full ensemble: with no body there is no disc-spurious fallback to jitter, and
  the recovered offset sits a few hundredths of a pixel from truth, far from the
  tolerance bound.
* **Technique-pinned star scenes** isolate one star technique each.
  ``planted_unique_star*`` is a single star -- too sparse for the field matcher,
  so ``StarUniqueMatchNav``'s one-star path carries it, pinned with
  ``only_techniques``.  ``planted_refine_star*`` is a multi-star field whose
  pass-1 ensemble reaches 'success' and installs a prior, so the pass-2
  ``StarRefineNav`` runs; it is read from the *full*-ensemble per-technique result
  because a prior-requiring pass-2 technique cannot be isolated with
  ``only_techniques``.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene
from spindoctor.support.status_reason import NavStatusReason
from tests.integration.sim_truth import (
    FUSED,
    RANK1_NORMAL_ERROR,
    ROLL_TOLERANCE_DEG,
    assert_matches_truth,
)

_INVARIANTS_DIR = Path(__file__).parent / 'sim_scenes' / 'algorithmic_invariants'
_SCENE_PATHS = iter_scene_paths(_INVARIANTS_DIR.parent)
_INVARIANT_PATHS = [p for p in _SCENE_PATHS if p.parent.name == 'algorithmic_invariants']
_INVARIANT_IDS = [p.stem for p in _INVARIANT_PATHS]

# Scenes whose stem starts with this are blob-designed: recovery is verified
# blob-only because the full ensemble falls to the weak blob fallback and its
# recovered offset jitters across processes near the technique's limit.
_BLOB_STEM_PREFIX = 'planted_offset_blob'
# Star-designed scenes carry no body/rings; the star techniques are load-bearing
# and the full ensemble recovers stably.
_STAR_STEM_PREFIX = 'planted_offset_star'
# Rotation-designed scenes carry a planted camera roll.  The roll is verified on
# the StarFieldFromCatalogNav per-technique result, which recovers it
# geometrically (non-spurious).  The fused answer is a success, but its translation
# is fitted about the extended field of view's corner, so it carries a known error
# that the scene's expected block holds to a band (#434).
_ROTATION_STEM_PREFIX = 'planted_rotation'
# Limb scenes route through a LIMB_ARC the simulated body emits; the full ensemble
# still recovers via the disc correlation (so they double as disc scenes) and the
# limb path is asserted per-technique.
_LIMB_STEM_PREFIX = 'planted_offset_limb'
# Ring scenes route through a RING_EDGE the simulated ring emits.  Recovery is
# asserted on the RingEdgeNav per-technique result, so the ring path is checked in
# isolation; the fused answer is a high-tier success.
_RING_STEM_PREFIX = 'planted_offset_ring'
# Unique-match scenes are a single unambiguous star: StarFieldFromCatalogNav's
# pattern match is infeasible, so StarUniqueMatchNav's one-star path is the
# load-bearing technique.  Recovery is asserted on its per-technique result
# (navigated with only that technique, which needs no prior).
_UNIQUE_MATCH_STEM_PREFIX = 'planted_unique_star'
# Refine scenes are a pure multi-star field whose pass-1 ensemble reaches
# 'success' and installs a prior, so the pass-2 StarRefineNav runs.  Recovery is
# asserted on its per-technique result read from the full-ensemble navigate
# (only_techniques cannot isolate a prior-requiring pass-2 technique).
_REFINE_STEM_PREFIX = 'planted_refine_star'
_BLOB_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_BLOB_STEM_PREFIX)]
_BLOB_IDS = [p.stem for p in _BLOB_PATHS]
_STAR_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_STAR_STEM_PREFIX)]
_STAR_IDS = [p.stem for p in _STAR_PATHS]
_ROTATION_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_ROTATION_STEM_PREFIX)]
_ROTATION_IDS = [p.stem for p in _ROTATION_PATHS]
_LIMB_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_LIMB_STEM_PREFIX)]
_LIMB_IDS = [p.stem for p in _LIMB_PATHS]
_RING_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_RING_STEM_PREFIX)]
_RING_IDS = [p.stem for p in _RING_PATHS]
_UNIQUE_MATCH_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_UNIQUE_MATCH_STEM_PREFIX)]
_UNIQUE_MATCH_IDS = [p.stem for p in _UNIQUE_MATCH_PATHS]
_REFINE_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_REFINE_STEM_PREFIX)]
_REFINE_IDS = [p.stem for p in _REFINE_PATHS]
# Rank-1 scenes are all-straight flat-ring frames: only the offset component
# along the edge normal is observable, so full 2-D recovery is undefined and
# the assertions are rank-1-specific (status_reason, exact covariance rank
# deficiency, and recovery of the normal component only).
_RANK1_STEM_PREFIX = 'planted_rank1'
_RANK1_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_RANK1_STEM_PREFIX)]
_RANK1_IDS = [p.stem for p in _RANK1_PATHS]
# Disc scenes assert full-ensemble offset recovery.  Blob, star, rotation, ring,
# and the technique-pinned star scenes recover on a pinned technique instead (the
# fused offset is weak, absent, or per-technique), so they are held out; limb
# scenes stay in (they recover via the disc correlation).
_DISC_EXCLUDE = (
    _BLOB_STEM_PREFIX,
    _STAR_STEM_PREFIX,
    _ROTATION_STEM_PREFIX,
    _RING_STEM_PREFIX,
    _RANK1_STEM_PREFIX,
    _UNIQUE_MATCH_STEM_PREFIX,
    _REFINE_STEM_PREFIX,
)
_DISC_PATHS = [p for p in _INVARIANT_PATHS if not p.stem.startswith(_DISC_EXCLUDE)]
_DISC_IDS = [p.stem for p in _DISC_PATHS]
# Scenes whose full ensemble reaches 'success'.  The unique-match (single star) and
# refine scenes are pinned per-technique, so they are held out of the success
# assertion.
_NAVIGATES_EXCLUDE = (
    _UNIQUE_MATCH_STEM_PREFIX,
    _REFINE_STEM_PREFIX,
)
_NAVIGATES_SUCCESS_PATHS = [
    p for p in _INVARIANT_PATHS if not p.stem.startswith(_NAVIGATES_EXCLUDE)
]
_NAVIGATES_SUCCESS_IDS = [p.stem for p in _NAVIGATES_SUCCESS_PATHS]


def _navigate(scene: dict[str, Any], *, only_techniques: str = '*') -> Any:
    obs = ObsSim.from_file('/tmp/invariant.yaml', sim_params=scene)
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=only_techniques
    )
    return orchestrator.navigate(obs)


def test_there_are_invariant_scenes() -> None:
    """The algorithmic-invariants class is populated."""
    assert _INVARIANT_PATHS


def test_there_are_blob_scenes() -> None:
    """At least one blob-designed scene exists for the blob-only coverage."""
    assert _BLOB_PATHS


def test_there_are_star_scenes() -> None:
    """At least one star-designed scene exists for the star coverage."""
    assert _STAR_PATHS


def test_there_are_rotation_scenes() -> None:
    """At least one planted-roll scene exists for the rotation coverage."""
    assert _ROTATION_PATHS


def test_there_are_limb_scenes() -> None:
    """At least one limb-designed scene exists for the BodyLimbNav coverage."""
    assert _LIMB_PATHS


def test_there_are_ring_scenes() -> None:
    """At least one ring-designed scene exists for the RingEdgeNav coverage."""
    assert _RING_PATHS


def test_there_are_unique_match_scenes() -> None:
    """At least one single-star scene exists for the StarUniqueMatchNav coverage."""
    assert _UNIQUE_MATCH_PATHS


def test_there_are_refine_scenes() -> None:
    """At least one star-field scene exists for the StarRefineNav coverage."""
    assert _REFINE_PATHS


def test_there_are_rank1_scenes() -> None:
    """At least one flat-ring scene exists for the rank-1 coverage."""
    assert _RANK1_PATHS


@pytest.mark.parametrize('path', _RANK1_PATHS, ids=_RANK1_IDS)
def test_rank1_scene_reports_rank_1_only(path: Path) -> None:
    """A flat-ring scene fuses to success with the rank_1_only status reason.

    The unobservable axis is surfaced through both the discrete reason and
    the ``sigma_along_unobservable_px`` sentinel; the fused covariance is
    exactly rank-deficient (the technique projects it onto the edge normal).
    """
    result = _navigate(load_sim_scene(path))
    assert result.status == 'success'
    assert result.status_reason == NavStatusReason.RANK_1_ONLY
    assert result.sigma_along_unobservable_px == math.inf
    cov = np.asarray(result.covariance_px2, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov[:2, :2])
    assert float(eigvals.min()) == pytest.approx(0.0, abs=1.0e-9)
    assert float(eigvals.max()) > 0.0


@pytest.mark.parametrize('path', _RANK1_PATHS, ids=_RANK1_IDS)
def test_rank1_scene_holds_its_known_normal_error(path: Path) -> None:
    """A flat-ring scene's edge-normal error stays inside its known band.

    The along-edge component is physically unobservable (the fused offset
    is the minimum-norm representative), so the check projects the
    recovery error onto the observable axis -- the fused covariance's only
    non-null eigenvector -- and holds that component's magnitude to the
    known band: a worse miss fails, and so does a fix.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    cov = np.asarray(result.covariance_px2, dtype=np.float64)
    _eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
    n_hat = eigvecs[:, -1]
    error_vu = np.array(
        [
            result.offset_px[0] - scene['offset_v'],
            result.offset_px[1] - scene['offset_u'],
        ]
    )
    # The observable component misses by about 0.52 px (#873).
    normal_error = abs(float(error_vu @ n_hat))
    assert RANK1_NORMAL_ERROR.contains(normal_error), (
        f'normal-component error {normal_error:.3f} px left the known band '
        f'[{RANK1_NORMAL_ERROR.low}, {RANK1_NORMAL_ERROR.high}] px'
    )


@pytest.mark.parametrize('path', _NAVIGATES_SUCCESS_PATHS, ids=_NAVIGATES_SUCCESS_IDS)
def test_invariant_scene_navigates(path: Path) -> None:
    """Each planted-offset scene navigates to a success status (full ensemble)."""
    result = _navigate(load_sim_scene(path))
    assert result.status == 'success'


@pytest.mark.parametrize('path', _DISC_PATHS, ids=_DISC_IDS)
def test_disc_scene_recovers_planted_offset(path: Path) -> None:
    """Each disc scene's fused offset meets its truth check.

    Three disc scenes carry a known error at the edge of the correlation's sub-pixel
    window and are held to a band around it; the rest are held to a tolerance.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert_matches_truth(result.offset_px, scene, FUSED)


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_scene_navigates_via_blob_alone(path: Path) -> None:
    """Each blob scene navigates with BodyBlobNav as the only technique."""
    result = _navigate(load_sim_scene(path), only_techniques='BodyBlobNav')
    assert result.status == 'success'


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_alone_recovers_planted_offset(path: Path) -> None:
    """BodyBlobNav alone recovers each blob scene's planted offset within tolerance.

    Guards the BODY_BLOB feature emission, the background-pedestal subtraction,
    and the sky-noise threshold; without the latter two the thresholded observed
    centroid diverges from the model's continuous predicted centroid by several
    pixels on the high-phase crescent scene.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert_matches_truth(result.offset_px, scene, 'BodyBlobNav')


@pytest.mark.parametrize('path', _STAR_PATHS, ids=_STAR_IDS)
def test_star_scene_recovers_planted_offset(path: Path) -> None:
    """Each star scene recovers its planted offset within its truth tolerance.

    Guards the ``STAR`` feature emission from ``NavModelStarsSimulated`` and the
    half-pixel-correct star rendering: without the eval-offset correction the
    rendered star centroid sits half a pixel from the model's prediction and the
    recovered offset carries a constant bias.  With no body in the frame the
    star techniques are the only contributors, so this is a direct star-path
    invariant on the full ensemble.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert_matches_truth(result.offset_px, scene, FUSED)


def _starfield_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with StarFieldFromCatalogNav alone; return its result.

    Reads the per-technique result directly rather than the fused answer because
    the roll is the quantity under test.
    """
    result = _navigate(scene, only_techniques='StarFieldFromCatalogNav')
    matches = [t for t in result.per_technique if t.technique_name == 'StarFieldFromCatalogNav']
    assert matches, 'StarFieldFromCatalogNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _ROTATION_PATHS, ids=_ROTATION_IDS)
def test_rotation_scene_recovers_planted_roll(path: Path) -> None:
    """Each rotation scene recovers its planted camera roll within tolerance.

    Guards the camera-roll rendering (``offset_rotation_deg``) and the
    ``fit_camera_rotation`` scene override: the renderer rotates each star about
    the boresight, and StarFieldFromCatalogNav's similarity fit recovers the
    planted roll.  Asserted on the StarField per-technique result, where the
    technique recovers the roll geometrically (non-spurious).
    """
    scene = load_sim_scene(path)
    technique = _starfield_result(scene)
    assert not technique.spurious
    assert technique.rotation_rad is not None
    recovered_deg = math.degrees(technique.rotation_rad)
    assert abs(recovered_deg - scene['offset_rotation_deg']) < ROLL_TOLERANCE_DEG


@pytest.mark.parametrize('path', _LIMB_PATHS, ids=_LIMB_IDS)
def test_limb_alone_recovers_planted_offset(path: Path) -> None:
    """BodyLimbNav alone recovers each limb scene's planted offset within tolerance.

    Guards the LIMB_ARC emission from the simulated body: the silhouette boundary
    is sampled into a vertex polyline with outward normals, and BodyLimbNav's
    distance-transform fit aligns it to the image edge.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyLimbNav')
    assert_matches_truth(result.offset_px, scene, 'BodyLimbNav')


def _ring_edge_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with RingEdgeNav alone; return its per-technique result.

    The offset is read from the per-technique result so the ring path is checked in
    isolation from the rest of the ensemble.
    """
    result = _navigate(scene, only_techniques='RingEdgeNav')
    matches = [t for t in result.per_technique if t.technique_name == 'RingEdgeNav']
    assert matches, 'RingEdgeNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _RING_PATHS, ids=_RING_IDS)
def test_ring_scene_recovers_planted_offset(path: Path) -> None:
    """Each ring scene recovers its planted offset within its truth tolerance.

    Guards the RING_EDGE emission from the simulated ring: each rendered edge is
    sampled into a radial-normal polyline that RingEdgeNav fits against the
    image-edge distance transform.  Two curved arcs constrain both axes.
    """
    scene = load_sim_scene(path)
    technique = _ring_edge_result(scene)
    assert not technique.spurious
    assert_matches_truth(technique.offset_px, scene, 'RingEdgeNav')


def _unique_match_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with StarUniqueMatchNav alone; return its result.

    The technique needs no prior (pass 1), so it can be isolated.  The offset is
    read from the per-technique result because the one-star confidence cap and
    the placeholder-alpha star confidences keep the fused status off a stable
    'success' on a clean single-star field.
    """
    result = _navigate(scene, only_techniques='StarUniqueMatchNav')
    matches = [t for t in result.per_technique if t.technique_name == 'StarUniqueMatchNav']
    assert matches, 'StarUniqueMatchNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _UNIQUE_MATCH_PATHS, ids=_UNIQUE_MATCH_IDS)
def test_unique_match_scene_recovers_planted_offset(path: Path) -> None:
    """Each single-star scene recovers its planted offset via StarUniqueMatchNav.

    Guards the one-star path: with no companion star the brightness-uniqueness
    gate passes trivially and the brightest detection inside the lone star's
    search window is its unambiguous match, so centroid-minus-prediction is the
    offset.
    """
    scene = load_sim_scene(path)
    technique = _unique_match_result(scene)
    assert not technique.spurious
    assert_matches_truth(technique.offset_px, scene, 'StarUniqueMatchNav')


def _star_refine_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with the full ensemble; return its StarRefineNav result.

    StarRefineNav requires the pass-1 prior, so it cannot be isolated with
    ``only_techniques`` (that would starve it of a prior and it would fail).  The
    full ensemble runs pass 1 (StarFieldFromCatalogNav + StarUniqueMatchNav) to a
    'success' that installs the prior, then pass 2 runs StarRefineNav -- its
    presence here confirms the prior path fired.
    """
    result = _navigate(scene, only_techniques='*')
    matches = [t for t in result.per_technique if t.technique_name == 'StarRefineNav']
    assert matches, 'StarRefineNav produced no technique result (pass-1 prior did not install)'
    return matches[0]


@pytest.mark.parametrize('path', _REFINE_PATHS, ids=_REFINE_IDS)
def test_refine_scene_recovers_planted_offset(path: Path) -> None:
    """Each refine scene recovers its planted offset via StarRefineNav.

    Guards the pass-2 prior-refinement path: StarRefineNav re-centroids each
    predicted star around the prior-shifted position and reports the
    inverse-variance-weighted refined absolute offset.  Asserting on the
    per-technique result also confirms the prior installed (the technique only
    runs in pass 2 after a 'success' pass-1 ensemble).
    """
    scene = load_sim_scene(path)
    technique = _star_refine_result(scene)
    assert not technique.spurious
    assert_matches_truth(technique.offset_px, scene, 'StarRefineNav')
