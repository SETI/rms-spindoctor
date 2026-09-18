"""Fast regression tests for specific navigation defects surfaced by the sweeps.

Each test pins the affected technique on one dedicated scene under
``sim_scenes/regression/`` and asserts the correct behavior, so the defect is
guarded in the normal pytest run without executing the full (slow,
runner-only) sweep suite. These tests are in-process and unmarked, so they run in
the default suite.
"""

import math
from pathlib import Path

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene

_REGRESSION_DIR = Path(__file__).parent / 'sim_scenes' / 'regression'

# A well-resolved disc must recover a whole-pixel offset to a small fraction of a
# pixel.  The correlator upsamples to 1/128 px and reaches that on raw intensity,
# so 0.1 px is a generous bound that the correct behavior clears comfortably.
_DISC_SUBPIXEL_TOLERANCE_PX = 0.1
# A star field must recover a zero offset; the moment centroid clears 0.1 px today
# and the PSF-fit refinement tightens it further.
_STAR_ZERO_OFFSET_TOLERANCE_PX = 0.1


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


def _technique_offset_error(scene_name: str, technique: str) -> float:
    """Pin one technique on a regression scene; return its recovered-offset error."""
    scene = load_sim_scene(_REGRESSION_DIR / f'{scene_name}.yaml')
    pinned = _pin_technique(scene_name, technique)
    assert not pinned.spurious
    assert pinned.offset_px is not None
    return math.hypot(
        pinned.offset_px[0] - scene['offset_v'],
        pinned.offset_px[1] - scene['offset_u'],
    )


def test_disc_recovers_whole_pixel_offset() -> None:
    """The disc correlation recovers a whole-pixel offset to within a fraction of a pixel.

    Guards the gradient-magnitude NCC sub-pixel bias: the integer peak is found on
    the gradient surfaces but the sub-pixel offset is refined on raw intensity, so
    a whole-pixel offset no longer carries the ~0.3 px-per-axis rectification bias.
    """
    assert _technique_offset_error('disc_subpixel_offset', 'BodyDiscCorrelateNav') < (
        _DISC_SUBPIXEL_TOLERANCE_PX
    )


def test_star_field_recovers_zero_offset() -> None:
    """The star field recovers a zero offset, where the prediction sits on each star.

    Guards the star reliability-gate defect: the model no longer gates a star on
    the saturation / cosmic-ray mask at its predicted position, so a sharp star
    flagged by the cosmic-ray detector is not killed when the offset is near zero.
    """
    assert _technique_offset_error('star_zero_offset', 'StarFieldFromCatalogNav') < (
        _STAR_ZERO_OFFSET_TOLERANCE_PX
    )


def test_star_field_recovers_zero_offset_on_equal_brightness_field() -> None:
    """The star field recovers the offset when every star shares one magnitude.

    Guards the triplet-canonicalisation seed lottery: with equal magnitudes the
    brightness tie-break was decided differently on the catalog and detection
    sides, so the pattern match flipped with the noise realization. The geometric
    (opposite-side-length) canonicalisation is invariant to the tie, so recovery
    no longer depends on which equal-brightness star wins an arbitrary tie-break.
    """
    assert _technique_offset_error('star_equal_brightness', 'StarFieldFromCatalogNav') < (
        _STAR_ZERO_OFFSET_TOLERANCE_PX
    )


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
