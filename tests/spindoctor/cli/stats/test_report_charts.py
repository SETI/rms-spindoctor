"""Tests for the pointing-correction heat map of the statistics report."""

from pathlib import Path

import numpy as np
import pytest
from filecache import FCPath

from spindoctor.cli.stats import report_charts
from spindoctor.cli.stats.report_charts import write_offset_heatmap


def test_offset_heatmap_writes_nothing_for_an_empty_group(tmp_path: Path) -> None:
    """A camera with no successful image gets no chart, and says so."""
    path = tmp_path / 'heatmap.png'
    assert not write_offset_heatmap(FCPath(path), [], [], title='empty')
    assert not path.exists()


def test_offset_heatmap_output_is_deterministic(tmp_path: Path) -> None:
    """The same offsets write the same PNG bytes."""
    rng = np.random.default_rng(1)
    dv = list(rng.normal(0.0, 3.0, 200))
    du = list(rng.normal(1.0, 2.0, 200))
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    assert write_offset_heatmap(FCPath(first), dv, du, title='heat map')
    write_offset_heatmap(FCPath(second), dv, du, title='heat map')
    assert first.read_bytes() == second.read_bytes()


def test_offset_heatmap_draws_an_all_zero_group(tmp_path: Path) -> None:
    """Corrections that are all zero still give the panel a nonzero extent."""
    path = tmp_path / 'heatmap.png'
    assert write_offset_heatmap(FCPath(path), [0.0, 0.0], [0.0, 0.0], title='zero')
    assert path.exists()


def test_offset_heatmap_limit_leaves_the_largest_out() -> None:
    """The half-width follows the 98th percentile, not the largest correction."""
    offsets = np.array([[1.0, 0.0]] * 99 + [[100.0, 0.0]])
    assert report_charts._heatmap_limit(offsets) == pytest.approx(1.1)


def test_offset_heatmap_caption_of_one_image() -> None:
    """One image is named in the singular."""
    offsets = np.array([[3.0, 4.0]])
    assert report_charts._heatmap_caption(offsets, 10.0) == '1 image     RMS 5.00 px'


def test_offset_heatmap_caption_inside_the_panel() -> None:
    """A group wholly inside the panel states its count and RMS and nothing else."""
    offsets = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert report_charts._heatmap_caption(offsets, 10.0) == '2 images     RMS 3.54 px'


def test_offset_heatmap_caption_counts_the_images_outside() -> None:
    """Corrections past the panel edge on either axis are counted, and are in the RMS."""
    offsets = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, -12.0]])
    assert report_charts._heatmap_caption(offsets, 10.0) == (
        '3 images     RMS 7.51 px     1 outside the panel'
    )


def test_offset_heatmap_counts_rows_by_dv_and_columns_by_du() -> None:
    """A correction down and to the left lands below and left of the center bin."""
    offsets = np.array([[0.0, 0.0]] * 10 + [[5.0, -5.0]])
    counts, _ = report_charts._heatmap_counts(offsets, 10.0)
    center = len(counts) // 2
    rows, columns = np.nonzero(counts == 1.0)
    assert counts[center, center] == 10.0
    assert rows[0] > center
    assert columns[0] < center


def test_offset_heatmap_bins_follow_the_image_count() -> None:
    """About the square root of the count, odd, and within the bounds."""
    assert report_charts._heatmap_bins(40) == report_charts._HEATMAP_MIN_BINS
    assert report_charts._heatmap_bins(900) == 31
    assert report_charts._heatmap_bins(1024) == 33
    assert report_charts._heatmap_bins(10**6) == report_charts._HEATMAP_MAX_BINS


def test_count_ticks_are_1_2_5_up_to_the_largest_count() -> None:
    """The colorbar labels plain counts from 1 to the largest bin."""
    assert report_charts._count_ticks(2.0) == [1.0, 2.0]
    assert report_charts._count_ticks(60.0) == [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def test_ring_radii_are_round_and_inside_the_panel() -> None:
    """Two to four rings on a 1-2-5 step, all inside the half-width."""
    assert report_charts._ring_radii(9.0) == [2.0, 4.0, 6.0, 8.0]
    assert report_charts._ring_radii(30.0) == [10.0, 20.0]
    assert report_charts._ring_radii(0.35) == pytest.approx([0.1, 0.2, 0.3])
