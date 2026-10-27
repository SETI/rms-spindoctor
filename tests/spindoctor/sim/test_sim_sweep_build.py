"""Every sweep spec and every step it builds validates, without navigating.

A sweep (``tests/integration/sim_sweeps/*.yaml``) varies one scene parameter
across steps.  These checks load each spec and build each step's scene
parameters, re-validating the stepped ``artifacts`` block, so a sweep that
addresses a retired or misspelled key fails here instead of rendering a clean
frame.  They navigate nothing, so they run with the unit suite; the sweeps'
navigation invariants live in ``tests/integration/test_sim_sweeps.py``.
"""

import re
from pathlib import Path

import pytest
from tests.integration.sim_sweep import build_sweep_params, iter_sweep_paths, load_sweep

from spindoctor.sim.scene import SimSceneValidationError

_SWEEPS_ROOT = Path(__file__).parents[2] / 'integration' / 'sim_sweeps'


def test_every_sweep_validates() -> None:
    """Every sweep spec parses and validates."""
    paths = iter_sweep_paths(_SWEEPS_ROOT)
    assert paths
    for path in paths:
        load_sweep(path)


def test_every_sweep_step_builds_and_validates() -> None:
    """Every step of every sweep builds, its artifacts block re-validated."""
    paths = iter_sweep_paths(_SWEEPS_ROOT)
    steps = [step for path in paths for step in build_sweep_params(load_sweep(path))]
    assert len(steps) >= 2 * len(paths)


def _artifact_sweep_on(tmp_path: Path, parameter: str) -> Path:
    """Write a copy of the missing-lines sweep that addresses ``parameter``."""
    text = (_SWEEPS_ROOT / 'artifact_missing_lines.yaml').read_text()
    path = tmp_path / 'artifact_missing_lines.yaml'
    path.write_text(text.replace('artifacts.missing_lines.expected_events', parameter))
    return path


def test_sweep_on_a_retired_key_is_refused(tmp_path: Path) -> None:
    """A sweep path still naming incidence fails at build, naming the new key."""
    spec = load_sweep(_artifact_sweep_on(tmp_path, 'artifacts.missing_lines.incidence'))
    message = "artifacts.missing_lines: 'incidence' was renamed; use 'expected_events'"
    with pytest.raises(SimSceneValidationError, match=re.escape(message)):
        build_sweep_params(spec)


def test_sweep_on_an_unknown_mode_key_is_refused(tmp_path: Path) -> None:
    """A misspelled artifact-mode key fails at build instead of rendering clean."""
    spec = load_sweep(_artifact_sweep_on(tmp_path, 'artifacts.missing_lines.expected_event'))
    with pytest.raises(SimSceneValidationError, match=r"unknown keys: \['expected_event'\]"):
        build_sweep_params(spec)
