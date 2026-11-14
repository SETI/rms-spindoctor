"""Expected-outcome assertion machinery for sim scenes (the ``expected`` block).

A sim scene may carry a scene-level ``expected`` block declaring the outcome the
navigator should produce: a ``status`` (``success`` / ``failed`` / ``conflicted``),
an optional ``confidence_tier`` (one of the five navigation ranks, or null to
assert the status only), an optional ``status_reason``, and an optional
known-error check, the ``known_offset_error_px`` / ``known_offset_error_tol_px``
pair.  This module loads that block and asserts a
:class:`~spindoctor.nav_orchestrator.nav_result.NavResult` against it with clear
failure messages.

It is the sim analog of the image-library sidecar's expected-outcome regression,
modeled on that taxonomy (status / confidence_tier / status_reason and its
cross-field rules) but implemented independently: a sim scene is not a sidecar,
so nothing here imports ``tests.integration.sidecar``.  The ``expected`` block is
a test-only scene key -- read here, fed to neither the renderer nor the
navigator, and stripped from ``nav_params`` by the information boundary.

The machinery asserts two distinct patterns:

- **Expected-fail scenes.**  When a scene renders every star in the wrong
  place, or an overwhelming confounder field, the CORRECT navigation outcome
  is a failed / low-confidence result, and the status / tier assertion turns
  that requirement into a test.
- **Known-error checks.**  A scene can produce a confidently wrong offset the
  ensemble cannot remove.  ``orbit_error_ringlet`` is the one in the catalog:
  its planted radial catalog error is absorbed into a ring lock, and the fused
  result is a low-tier success about 1.7 px wrong.  The ``status: success``
  such a scene asserts is NOT an endorsement -- it is a documented hazard held
  in view.  For those scenes ``known_offset_error_px``
  states the measured fused error magnitude and ``known_offset_error_tol_px``
  its band, and the assertion fails when the measured error leaves the band in
  either direction: a worsening regression fails, and a genuine fix also fails
  loudly, prompting someone to re-measure the known error instead of a silent
  behavior change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene


@dataclass(frozen=True)
class ExpectedOutcome:
    """A scene's declared expected navigation outcome.

    Parameters:
        status: The expected fused status (``success`` / ``failed`` /
            ``conflicted``).
        confidence_tier: The expected confidence rank, or ``None`` to leave the
            tier unasserted (assert the status only).
        status_reason: The expected ``status_reason`` value, or ``None`` to
            leave it unasserted.
        known_offset_error_px: For a known-error check, the measured fused
            offset-error magnitude the scene documents as a standing hazard;
            ``None`` leaves the error unasserted.
        known_offset_error_tol_px: Tolerance band half-width around
            ``known_offset_error_px``; present exactly when that value is.
    """

    status: str
    confidence_tier: str | None
    status_reason: str | None
    known_offset_error_px: float | None
    known_offset_error_tol_px: float | None


def expected_from_scene(sim_params: dict[str, Any]) -> ExpectedOutcome | None:
    """Return the scene's :class:`ExpectedOutcome`, or None when it has no block.

    Parameters:
        sim_params: A validated scene mapping (the ``expected`` block, if
            present, was validated by the sim schema).

    Returns:
        The parsed expected outcome, or ``None`` for a scene with no ``expected``
        block.
    """
    block = sim_params.get('expected')
    if not isinstance(block, dict):
        return None
    known_error = block.get('known_offset_error_px')
    known_tol = block.get('known_offset_error_tol_px')
    return ExpectedOutcome(
        status=str(block['status']),
        confidence_tier=block.get('confidence_tier'),
        status_reason=block.get('status_reason'),
        known_offset_error_px=None if known_error is None else float(known_error),
        known_offset_error_tol_px=None if known_tol is None else float(known_tol),
    )


def iter_expected_scene_paths(root: Path) -> list[Path]:
    """Return every catalog scene under ``root`` that carries an ``expected`` block."""
    return [path for path in iter_scene_paths(root) if load_sim_scene(path).get('expected')]


def navigate_scene(sim_params: dict[str, Any]) -> NavResult:
    """Render and navigate a scene in-process, returning its NavResult.

    Parameters:
        sim_params: A validated scene mapping.

    Returns:
        The orchestrator's fused result for the rendered frame.
    """
    obs = ObsSim.from_file('/tmp/sim_expected.yaml', sim_params=sim_params)
    orchestrator = NavOrchestrator(build_models_for_obs(obs), only_models='*', only_techniques='*')
    return orchestrator.navigate(obs)


def assert_result_matches_expected(
    *,
    scene_name: str,
    expected: ExpectedOutcome,
    result: NavResult,
    planted_offset_vu: tuple[float, float] | None = None,
) -> None:
    """Assert a NavResult matches a scene's expected outcome, with clear messages.

    The status is always checked; the confidence tier and the status_reason are
    checked only when the scene asserts them (a null tier or an omitted reason
    leaves that field unconstrained).  When the scene carries a known-error
    check (``known_offset_error_px``), the measured fused error magnitude
    against the planted offset must sit inside the check's tolerance band.
    Below the band means the hazard quietly improved (re-measure the known
    error deliberately), and above it means a worsening regression.

    Parameters:
        scene_name: The scene name, used in the failure messages.
        expected: The scene's declared expected outcome.
        result: The navigator's fused result for the scene.
        planted_offset_vu: The scene's planted ``(offset_v, offset_u)`` truth,
            required when ``expected.known_offset_error_px`` is set (the test
            harness reads it from the scene file; the navigator never sees it).

    Raises:
        AssertionError: On any mismatch, naming the scene and the fields.
    """
    assert result.status == expected.status, (
        f'{scene_name}: expected status {expected.status!r}, got {result.status!r} '
        f'(status_reason {result.status_reason.value!r})'
    )
    if expected.confidence_tier is not None:
        assert result.confidence_rank == expected.confidence_tier, (
            f'{scene_name}: expected confidence tier {expected.confidence_tier!r}, '
            f'got {result.confidence_rank!r}'
        )
    if expected.status_reason is not None:
        assert result.status_reason.value == expected.status_reason, (
            f'{scene_name}: expected status_reason {expected.status_reason!r}, '
            f'got {result.status_reason.value!r}'
        )
    if expected.known_offset_error_px is None:
        return
    assert expected.known_offset_error_tol_px is not None, (
        f'{scene_name}: known_offset_error_px without its tolerance (schema bug)'
    )
    assert planted_offset_vu is not None, (
        f'{scene_name}: known_offset_error_px asserted but no planted offset supplied'
    )
    assert result.offset_px is not None, (
        f'{scene_name}: known_offset_error_px asserted on a result with no offset'
    )
    error_px = math.hypot(
        result.offset_px[0] - planted_offset_vu[0],
        result.offset_px[1] - planted_offset_vu[1],
    )
    lower = expected.known_offset_error_px - expected.known_offset_error_tol_px
    upper = expected.known_offset_error_px + expected.known_offset_error_tol_px
    assert error_px >= lower, (
        f'{scene_name}: measured offset error {error_px:.2f} px fell below the '
        f'known-error band [{lower:.2f}, {upper:.2f}] px -- the documented '
        f'confident-wrong behavior improved; re-measure the known error and update '
        f'(or retire) the known_offset_error_px value'
    )
    assert error_px <= upper, (
        f'{scene_name}: measured offset error {error_px:.2f} px exceeds the '
        f'known-error band [{lower:.2f}, {upper:.2f}] px -- the documented '
        f'confident-wrong behavior worsened'
    )
