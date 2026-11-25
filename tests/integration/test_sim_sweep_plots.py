"""The sweep figures are drawn only from complete sweep results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.integration import sim_sweep_plots


@pytest.fixture
def plot_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the plots at empty result and figure directories."""
    results = tmp_path / 'results'
    figures = tmp_path / 'figures'
    results.mkdir()
    monkeypatch.setattr(sim_sweep_plots, '_RESULTS_ROOT', results)
    monkeypatch.setattr(sim_sweep_plots, '_FIGURES_ROOT', figures)
    return results, figures


def _write_result(results: Path, name: str) -> None:
    """Write a two-step sweep result with a recovered offset at each step."""
    rows: list[dict[str, Any]] = [
        {'value': value, 'offset_error_px': 0.1, 'confidence': 0.9} for value in (0.0, 1.0)
    ]
    (results / f'{name}.json').write_text(json.dumps({'sweep_name': name, 'rows': rows}))


def test_a_partial_run_leaves_the_mismatch_figure_alone(plot_dirs: tuple[Path, Path]) -> None:
    """With one mismatch axis's result, the mismatch figure isn't drawn."""
    results, _figures = plot_dirs
    _write_result(results, 'atmosphere_haze_albedo')
    assert sim_sweep_plots.generate_plots() == []


def test_the_mismatch_figure_is_drawn_once_every_axis_ran(plot_dirs: tuple[Path, Path]) -> None:
    """With every mismatch axis's result, the mismatch figure alone is drawn."""
    results, figures = plot_dirs
    for axis in sim_sweep_plots._MISMATCH_AXES:
        _write_result(results, axis.name)
    assert sim_sweep_plots.generate_plots() == [figures / 'mismatch_axes.png']


def _mismatch_axis(name: str) -> sim_sweep_plots._MismatchAxis:
    """The mismatch figure's axis for one sweep."""
    return next(axis for axis in sim_sweep_plots._MISMATCH_AXES if axis.name == name)


def test_a_floor_axis_marks_its_recovered_first_step() -> None:
    """An axis starting from a self-consistency floor marks its first step there."""
    rows: list[dict[str, Any]] = [
        {'value': 0.0, 'offset_error_px': 0.02},
        {'value': 1.0, 'offset_error_px': 0.5},
    ]
    assert sim_sweep_plots._floor_point(_mismatch_axis('psf_limb_mismatch'), rows) == (0.0, 0.02)


def test_a_floor_axis_marks_no_floor_when_its_first_step_failed() -> None:
    """A floor axis whose first step recovered no offset marks no floor."""
    rows: list[dict[str, Any]] = [
        {'value': 0.0, 'offset_error_px': None},
        {'value': 1.0, 'offset_error_px': 0.5},
    ]
    assert sim_sweep_plots._floor_point(_mismatch_axis('psf_limb_mismatch'), rows) is None


def test_the_haze_brightness_axis_marks_no_floor() -> None:
    """A recovered offset on the haze brightness sweep's blank first step is no floor."""
    rows: list[dict[str, Any]] = [
        {'value': 0.0, 'offset_error_px': 52.0},
        {'value': 0.6, 'offset_error_px': 0.5},
    ]
    assert sim_sweep_plots._floor_point(_mismatch_axis('atmosphere_haze_albedo'), rows) is None


@pytest.mark.parametrize('which', ['fine', 'wide'])
def test_an_offset_figure_waits_for_every_technique(
    plot_dirs: tuple[Path, Path], which: str
) -> None:
    """One technique's offset sweep short of the set leaves its offset figure undrawn."""
    results, _figures = plot_dirs
    for _key, fine, wide, _marker, _label in sim_sweep_plots._OFFSET_TECHNIQUES[1:]:
        _write_result(results, fine if which == 'fine' else wide)
    assert sim_sweep_plots.generate_plots() == []
