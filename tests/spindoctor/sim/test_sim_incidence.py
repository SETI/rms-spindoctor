"""Artifact-incidence measurement: planted truth equals image-measured count.

Each structural marker-based loss mode is rendered in isolation onto a biased
disc frame (a non-zero floor so the marker stands out from the sky), and the
image-only estimator is asserted to recover exactly the count the telemetry
stage recorded in the frame truth.  The planted side is exact by construction;
these tests pin the image side to it.  Modes that plant wrong values rather than
the marker (garble, spikes, detector electronics) are out of scope for the
image estimators and are covered only through :func:`planted_incidence`.
"""

import copy
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.incidence import (
    marker_mask,
    measured_missing_blocks,
    measured_missing_lines,
    measured_partial_lines,
    measured_pixel_dropouts,
    planted_incidence,
)
from spindoctor.sim.render import render_combined_model

# A biased, well-resolved disc: the bias floor lifts the sky above the zero
# marker so a dropped pixel is exactly distinguishable from dark sky.
_BASE_SCENE: dict[str, Any] = {
    'schema_version': 2,
    'scene_name': 'incidence_base',
    'instrument': 'coiss_nac',
    'size_v': 200,
    'size_u': 200,
    'random_seed': 7,
    'exposure_sec': 1.0,
    'bodies': [
        {
            'name': 'RHEA',
            'center_v': 100.0,
            'center_u': 100.0,
            'axis1': 80.0,
            'axis2': 80.0,
            'axis3': 80.0,
            'illumination_angle': 25.0,
            'phase_angle': 30.0,
        }
    ],
    'noise': {'poisson': True, 'read_noise_dn': 4.0, 'bias_dn': 20.0},
}


def _render_with(artifacts: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Render the base scene with one artifacts block; return image and truth."""
    scene = copy.deepcopy(_BASE_SCENE)
    scene['artifacts'] = artifacts
    return render_combined_model(scene)


def test_planted_incidence_empty_without_artifacts() -> None:
    """A frame with no artifacts records no planted incidence."""
    _img, truth = render_combined_model(copy.deepcopy(_BASE_SCENE))
    assert planted_incidence(truth) == {}


def test_planted_missing_lines_counts_lost_lines() -> None:
    """The planted count is the number of recorded lost lines."""
    _img, truth = _render_with({'missing_lines': {'expected_events': 6.0}})
    recorded = len(truth['artifacts']['missing_lines']['lines'])
    assert planted_incidence(truth)['missing_lines'] == recorded


def test_measured_missing_lines_matches_planted() -> None:
    """The image estimator recovers exactly the planted lost-line count."""
    img, truth = _render_with({'missing_lines': {'expected_events': 6.0}})
    planted = planted_incidence(truth)['missing_lines']
    assert measured_missing_lines(img) == planted


def test_measured_missing_lines_nonzero() -> None:
    """The rendered scene actually planted at least one missing line."""
    img, _truth = _render_with({'missing_lines': {'expected_events': 6.0}})
    assert measured_missing_lines(img) > 0


def test_measured_partial_lines_matches_planted() -> None:
    """The image estimator recovers exactly the planted truncated-line count."""
    img, truth = _render_with(
        {'partial_lines': {'expected_events': 6.0, 'max_surviving_segments': 2}}
    )
    planted = planted_incidence(truth)['partial_lines']
    assert measured_partial_lines(img) == planted


def test_measured_partial_lines_nonzero() -> None:
    """The rendered scene actually planted at least one truncated line."""
    img, _truth = _render_with({'partial_lines': {'expected_events': 6.0}})
    assert measured_partial_lines(img) > 0


def test_measured_missing_blocks_matches_planted() -> None:
    """The image estimator recovers exactly the planted lost-block count."""
    img, truth = _render_with({'missing_blocks': {'expected_events': 5.0, 'block_lines': 8}})
    record = truth['artifacts']['missing_blocks']
    planted = len(record['blocks'])
    assert measured_missing_blocks(img, block_lines=record['block_lines']) == planted


def test_measured_pixel_dropouts_matches_planted() -> None:
    """The image estimator recovers exactly the planted dead-pixel count."""
    img, truth = _render_with({'dead_pixels': {'expected_events': 30.0}})
    planted = len(truth['artifacts']['dead_pixels']['pixels'])
    assert measured_pixel_dropouts(img) == planted


def test_pixel_dropouts_excludes_missing_lines() -> None:
    """A whole missing line is not counted as isolated pixel dropouts."""
    img, _truth = _render_with({'missing_lines': {'expected_events': 4.0}})
    assert measured_pixel_dropouts(img) == 0


def test_missing_lines_ignores_isolated_dropouts() -> None:
    """Scattered dead pixels do not register as whole missing lines."""
    img, _truth = _render_with({'dead_pixels': {'expected_events': 30.0}})
    assert measured_missing_lines(img) == 0


def test_marker_mask_matches_zero_marker() -> None:
    """The marker mask flags exactly the zero-valued pixels on the raw-DN path."""
    image = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    mask = marker_mask(image, 0.0)
    assert mask.tolist() == [[True, False], [False, True]]


def test_marker_mask_matches_nan_marker() -> None:
    """The marker mask flags NaN pixels on the calibrated path."""
    image = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float64)
    mask = marker_mask(image, float('nan'))
    assert mask.tolist() == [[True, False], [False, True]]


def test_measured_missing_blocks_rejects_nonpositive_block_lines() -> None:
    """A non-positive block height is a caller error."""
    image = np.zeros((16, 16), dtype=np.float64)
    with pytest.raises(ValueError, match='block_lines must be positive'):
        measured_missing_blocks(image, block_lines=0)


def test_planted_incidence_commanded_mode_counts_activation() -> None:
    """A commanded on/off mode reports 1 when active, via its truth flag."""
    truth = {'artifacts': {'cutout_window': {'active': True, 'rect': [0, 4, 0, 4]}}}
    assert planted_incidence(truth)['cutout_window'] == 1


def test_planted_incidence_commanded_mode_inactive_is_zero() -> None:
    """An inactive commanded mode reports 0."""
    truth = {'artifacts': {'cutout_window': {'active': False}}}
    assert planted_incidence(truth)['cutout_window'] == 0
