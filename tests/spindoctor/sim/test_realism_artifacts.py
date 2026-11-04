"""Unit tests for the FOM 6 artifact-incidence detectors."""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.sim.realism.artifact_incidence import (
    measure_artifact_incidence,
    split_stationary_spikes,
)


def _noise_frame(seed: int, *, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(100.0, 3.0, shape))


def test_missing_line_detected_when_constant() -> None:
    """A constant row inside a noisy frame counts as a lost line."""
    frame = _noise_frame(31)
    frame[20, :] = 0.0
    incidence = measure_artifact_incidence(frame)
    assert incidence.missing_line_fraction == pytest.approx(1.0 / 62.0)


def test_interpolated_line_detected() -> None:
    """A row replaced by the exact mean of its neighbors counts as lost."""
    frame = _noise_frame(32)
    frame[30, :] = 0.5 * (frame[29, :] + frame[31, :])
    incidence = measure_artifact_incidence(frame)
    assert incidence.missing_line_fraction == pytest.approx(1.0 / 62.0)


def test_clean_frame_has_no_lost_lines() -> None:
    """A pure-noise frame reports zero line loss."""
    incidence = measure_artifact_incidence(_noise_frame(33))
    assert incidence.missing_line_fraction == 0.0


def test_blank_frame_not_counted_as_all_lost() -> None:
    """An all-constant (blank) frame is not 100% line loss."""
    incidence = measure_artifact_incidence(np.zeros((64, 64)))
    assert incidence.missing_line_fraction == 0.0


def test_spikes_detected_at_planted_positions() -> None:
    """Planted single-pixel spikes are found at their positions."""
    frame = _noise_frame(34)
    frame[10, 12] += 500.0
    frame[40, 50] += 500.0
    incidence = measure_artifact_incidence(frame)
    positions = {tuple(int(x) for x in row) for row in incidence.spike_positions_vu}
    assert (10, 12) in positions
    assert (40, 50) in positions
    assert incidence.spike_fraction == pytest.approx(2.0 / (64 * 64))


def test_negative_spikes_not_counted() -> None:
    """The detector counts positive spikes only (hot pixels, hits)."""
    frame = _noise_frame(35)
    frame[10, 12] -= 500.0
    incidence = measure_artifact_incidence(frame)
    positions = {tuple(int(x) for x in row) for row in incidence.spike_positions_vu}
    assert (10, 12) not in positions


def test_split_stationary_identifies_recurring_position() -> None:
    """A spike recurring across frames splits as stationary; others transient."""
    frames = []
    for seed in (41, 42, 43):
        frame = _noise_frame(seed)
        frame[5, 5] += 500.0  # stationary hot pixel
        frame[seed % 60, 30] += 500.0  # moving transient
        frames.append(measure_artifact_incidence(frame))
    stationary, transient = split_stationary_spikes(frames)
    assert stationary == pytest.approx(1.0 / (64 * 64), rel=0.01)
    assert transient == pytest.approx(1.0 / (64 * 64), rel=0.01)


def test_split_stationary_keeps_frames_of_different_shapes_apart() -> None:
    """A position shared by frames of two shapes isn't a recurring hot pixel."""
    frames = []
    for seed, shape in ((45, (64, 64)), (46, (48, 48))):
        frame = _noise_frame(seed, shape=shape)
        frame[5, 5] += 500.0
        frames.append(measure_artifact_incidence(frame))
    stationary, transient = split_stationary_spikes(frames)
    assert stationary == 0.0
    assert transient == pytest.approx((1.0 / (64 * 64) + 1.0 / (48 * 48)) / 2.0)


def test_split_stationary_single_frame_is_nan() -> None:
    """One frame cannot split hot pixels from transients."""
    stationary, transient = split_stationary_spikes([measure_artifact_incidence(_noise_frame(44))])
    assert np.isnan(stationary)
    assert np.isnan(transient)
