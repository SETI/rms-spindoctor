"""The sweep figures are drawn only from complete sweep results."""

from __future__ import annotations

import json
from pathlib import Path

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
    rows = [{'value': value, 'offset_error_px': 0.1, 'confidence': 0.9} for value in (0.0, 1.0)]
    (results / f'{name}.json').write_text(json.dumps({'sweep_name': name, 'rows': rows}))


def test_a_partial_run_leaves_the_mismatch_figure_alone(plot_dirs: tuple[Path, Path]) -> None:
    """With one mismatch axis's result, the mismatch figure isn't drawn."""
    results, _figures = plot_dirs
    _write_result(results, 'atmosphere_haze_albedo')
    assert sim_sweep_plots.generate_plots() == []


def test_the_mismatch_figure_is_drawn_once_every_axis_ran(plot_dirs: tuple[Path, Path]) -> None:
    """With every mismatch axis's result, the mismatch figure alone is drawn."""
    results, figures = plot_dirs
    for name, _xlabel, _title in sim_sweep_plots._MISMATCH_AXES:
        _write_result(results, name)
    assert sim_sweep_plots.generate_plots() == [figures / 'mismatch_axes.png']


@pytest.mark.parametrize('which', ['fine', 'wide'])
def test_an_offset_figure_waits_for_every_technique(
    plot_dirs: tuple[Path, Path], which: str
) -> None:
    """One technique's offset sweep short of the set leaves its offset figure undrawn."""
    results, _figures = plot_dirs
    for _key, fine, wide, _marker, _label in sim_sweep_plots._OFFSET_TECHNIQUES[1:]:
        _write_result(results, fine if which == 'fine' else wide)
    assert sim_sweep_plots.generate_plots() == []
