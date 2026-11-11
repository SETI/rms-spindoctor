"""Fast regression tests for specific navigation defects surfaced by the sweeps.

Each test pins the affected technique on one dedicated scene under
``sim_scenes/regression/`` and asserts the correct behavior, so the defect is
guarded in the normal pytest run without executing the full (slow,
runner-only) sweep suite.  Each recovered offset is held to the scene's truth
tolerance for the pinned technique (:mod:`tests.integration.sim_truth`).  These
tests are in-process and unmarked, so they run in the default suite.
"""

from pathlib import Path

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene
from tests.integration.sim_truth import assert_within_tolerance

_REGRESSION_DIR = Path(__file__).parent / 'sim_scenes' / 'regression'


def _pin_technique(scene_name: str, technique: str) -> NavTechniqueResult:
    """Navigate a regression scene with one pinned technique; return its result.

    Parameters:
        scene_name: Regression scene stem under ``sim_scenes/regression``.
        technique: The single technique to run.

    Returns:
        The pinned technique's :class:`NavTechniqueResult`.
    """
    scene = load_sim_scene(_REGRESSION_DIR / f'{scene_name}.yaml')
    obs = ObsSim.from_file('/tmp/regression.yaml', sim_params=scene)
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=technique
    ).navigate(obs)
    pinned = next((t for t in result.per_technique if t.technique_name == technique), None)
    assert pinned is not None
    return pinned


def _assert_technique_recovers(scene_name: str, technique: str) -> None:
    """Pin one technique on a regression scene; assert it recovers the planted offset."""
    scene = load_sim_scene(_REGRESSION_DIR / f'{scene_name}.yaml')
    pinned = _pin_technique(scene_name, technique)
    assert not pinned.spurious
    assert_within_tolerance(pinned.offset_px, scene, technique)


def test_disc_recovers_whole_pixel_offset() -> None:
    """The disc correlation recovers a whole-pixel offset to within a fraction of a pixel.

    Guards the gradient-magnitude NCC sub-pixel bias: the integer peak is found on
    the gradient surfaces but the sub-pixel offset is refined on raw intensity, so
    a whole-pixel offset no longer carries the ~0.3 px-per-axis rectification bias.
    """
    _assert_technique_recovers('disc_subpixel_offset', 'BodyDiscCorrelateNav')


def test_star_field_recovers_near_zero_offset() -> None:
    """The star field recovers a near-zero offset, where the prediction sits on each star.

    Guards the star reliability-gate defect: the model no longer gates a star on
    the saturation / cosmic-ray mask at its predicted position, so a sharp star
    flagged by the cosmic-ray detector is not killed when the offset is near zero.
    """
    _assert_technique_recovers('star_zero_offset', 'StarFieldFromCatalogNav')


def test_star_field_recovers_offset_on_equal_brightness_field() -> None:
    """The star field recovers the offset when every star shares one magnitude.

    Guards the triplet-canonicalisation seed lottery: with equal magnitudes the
    brightness tie-break was decided differently on the catalog and detection
    sides, so the pattern match flipped with the noise realization. The geometric
    (opposite-side-length) canonicalisation is invariant to the tie, so recovery
    no longer depends on which equal-brightness star wins an arbitrary tie-break.
    """
    _assert_technique_recovers('star_equal_brightness', 'StarFieldFromCatalogNav')


def test_low_phase_limb_misconvergence_is_flagged_spurious() -> None:
    """A low-phase limb mis-lock is rejected, never a confident wrong offset.

    Guards the coarse-seed mis-lock gate: at phase 10 deg the under-conditioned
    limb geometry lets a false overlap peak seed the fit in the wrong basin, so
    the Levenberg-Marquardt stage ends unconverged and pinned against the trust
    region several pixels from the true limb.  This pins the BodyLimbNav result
    only: it must be flagged spurious so it contributes no confident multi-pixel
    offset (the disc technique still navigates the scene, so the fused ensemble
    result is a success).
    """
    pinned = _pin_technique('low_phase_limb_misconverge', 'BodyLimbNav')
    assert pinned.spurious
