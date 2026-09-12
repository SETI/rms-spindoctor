"""Unit tests for the per-frame comparison's arithmetic."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from util.nav_verification.compare_pointing import (
    FrameComparison,
    compare,
    consensus_spread,
    remove_common_offset,
)

from spindoctor.nav_records import METADATA_SUFFIX

OBSERVATION = 'ISS_006RI_LPHRLFMOV001_PRIME'


def navigation(*techniques: dict[str, Any], excluded: list[str] | None = None) -> dict[str, Any]:
    """A navigation result holding the given per-technique answers.

    Parameters:
        techniques: The per-technique entries.
        excluded: Technique names the ensemble threw out as outliers.
    """
    return {
        'per_technique': list(techniques),
        'excluded_from_consensus': excluded or [],
    }


def technique(name: str, offset: list[float], *, spurious: bool = False) -> dict[str, Any]:
    """One per-technique answer.

    Parameters:
        name: The technique's name.
        offset: The offset it proposed.
        spurious: Whether the pass called it spurious.
    """
    return {'technique_name': name, 'offset_px': offset, 'spurious': spurious}


def test_spread_of_one_contributor_is_zero() -> None:
    """A frame one technique answered has nothing to disagree with."""
    assert consensus_spread(navigation(technique('A', [1.0, 2.0])))[0] == 0.0


def test_spread_counts_the_contributors() -> None:
    """The count is of techniques that formed the answer."""
    result = navigation(technique('A', [0.0, 0.0]), technique('B', [3.0, 4.0]))
    assert consensus_spread(result)[1] == 2


def test_spread_measures_the_widest_pair() -> None:
    """The spread is the largest distance between any two contributors."""
    result = navigation(
        technique('A', [0.0, 0.0]), technique('B', [3.0, 4.0]), technique('C', [0.0, 1.0])
    )
    assert consensus_spread(result)[0] == pytest.approx(5.0)


def test_an_outlier_the_ensemble_threw_out_does_not_widen_the_spread() -> None:
    """A technique excluded from consensus is not part of the answer.

    Without this, one rejected technique hundreds of pixels away puts a spread
    of hundreds of pixels on a frame that one technique answered cleanly.
    """
    result = navigation(
        technique('A', [0.0, 0.0]),
        technique('B', [-281.5, -519.6]),
        excluded=['B'],
    )
    assert consensus_spread(result)[0] == 0.0


def test_a_spurious_technique_does_not_widen_the_spread() -> None:
    """A technique the pass called spurious is not part of the answer either."""
    result = navigation(technique('A', [0.0, 0.0]), technique('B', [100.0, 100.0], spurious=True))
    assert consensus_spread(result)[0] == 0.0


def comparison(x: float, y: float, error: float) -> FrameComparison:
    """A comparison carrying one measured difference.

    Parameters:
        x: The difference along the camera's first axis.
        y: The difference along its second.
        error: The unsigned difference.
    """
    return FrameComparison(image='N1', error_px=error, error_x_px=x, error_y_px=y)


def test_a_constant_offset_is_measured_and_removed() -> None:
    """A difference every frame shares is a datum, and comes out of the errors."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    common = remove_common_offset(rows, tolerance_px=2.0)
    assert common == pytest.approx((0.5, 0.5))


def test_what_is_left_after_a_constant_is_the_per_frame_difference() -> None:
    """A frame that carries only the constant disagrees by nothing."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[0].residual_px == pytest.approx(0.0, abs=1e-9)


def test_a_badly_navigated_frame_does_not_move_the_constant() -> None:
    """The constant is a median over frames that already agree."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(comparison(60.0, 60.0, 84.85))
    common = remove_common_offset(rows, tolerance_px=2.0)
    assert common == pytest.approx((0.5, 0.5))


def test_a_badly_navigated_frame_keeps_its_error() -> None:
    """Removing the constant does not excuse a frame that is genuinely wrong."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(comparison(60.0, 60.0, 84.85))
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[-1].residual_px == pytest.approx(84.15, rel=1e-3)


def test_too_few_agreeing_frames_measure_no_constant() -> None:
    """A median over three numbers is three numbers, not a datum."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(3)]
    assert remove_common_offset(rows, tolerance_px=2.0) is None


def test_with_no_constant_the_residual_is_the_error() -> None:
    """A run too small to measure a datum reports what it measured."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(3)]
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[0].residual_px == pytest.approx(0.707)


def test_a_frame_with_nothing_to_compare_gets_no_residual() -> None:
    """A frame the bundle holds no answer for is left alone."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(FrameComparison(image='N2'))
    remove_common_offset(rows, tolerance_px=2.0)
    assert rows[-1].residual_px is None


@pytest.mark.parametrize(
    'tolerance', [pytest.param(-1.0, id='negative'), pytest.param(math.nan, id='not-a-number')]
)
def test_a_tolerance_that_is_not_a_distance_is_refused(tolerance: float) -> None:
    """No frame is inside such a tolerance, so every statistic drawn from it is nonsense."""
    with pytest.raises(ValueError, match='a tolerance is a distance in pixels'):
        remove_common_offset([comparison(0.5, 0.5, 0.707)], tolerance_px=tolerance)


def results_tree(root: Path, stubs: dict[str, str]) -> Path:
    """Write a results tree holding one document per stub.

    Parameters:
        root: The directory to write the tree under.
        stubs: The results path stub of each document, against the image name
            the document records.

    Returns:
        The results root.
    """
    for stub, image in stubs.items():
        path = root / f'{stub}{METADATA_SUFFIX}'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({'status': 'success', 'observation': {'image_name': image}}),
            encoding='utf-8',
        )
    return root


def bundle(root: Path, images: list[str]) -> Path:
    """Write a bundle collection holding a supplementary file per image.

    Parameters:
        root: The directory to write the collection under.
        images: The images to hold a frame for.

    Returns:
        The collection.
    """
    collection = root / 'bundle'
    directory = collection / OBSERVATION.lower()
    directory.mkdir(parents=True, exist_ok=True)
    for image in images:
        suppl = directory / f'{image[1:]}{image[0].lower()}_reproj_img_suppl.txt'
        suppl.write_text('Navigation Type = Stars\n', encoding='utf-8')
    return collection


def test_no_selection_keeps_what_the_bundle_holds(tmp_path: Path) -> None:
    """The default is every record the bundle also holds a frame for."""
    tree = results_tree(tmp_path / 'nav', {'VOL/N1000000001_1_CALIB': 'N1000000001_1_CALIB'})
    rows, _, _ = compare(
        tree,
        observation_id=OBSERVATION,
        bundle_dir=bundle(tmp_path, ['N1000000001']),
        images=None,
    )
    assert [r.image for r in rows] == ['N1000000001']


def test_an_empty_selection_keeps_nothing(tmp_path: Path) -> None:
    """An empty list of images is a selection of none of them, not of all of them."""
    tree = results_tree(tmp_path / 'nav', {'VOL/N1000000001_1_CALIB': 'N1000000001_1_CALIB'})
    rows, _, _ = compare(
        tree,
        observation_id=OBSERVATION,
        bundle_dir=bundle(tmp_path, ['N1000000001']),
        images=[],
    )
    assert rows == []


def test_an_image_with_two_records_is_compared_by_neither(tmp_path: Path) -> None:
    """The stream promises no order, so there is no basis for preferring either record."""
    tree = results_tree(
        tmp_path / 'nav',
        {
            'VOL1/N1000000001_1_CALIB': 'N1000000001_1_CALIB',
            'VOL2/N1000000001_1_CALIB': 'N1000000001_1_CALIB',
        },
    )
    rows, _, _ = compare(
        tree,
        observation_id=OBSERVATION,
        bundle_dir=bundle(tmp_path, ['N1000000001']),
        images=None,
    )
    assert rows == []


def test_an_image_with_two_records_is_reported(tmp_path: Path) -> None:
    """An image left out of the comparison is named rather than silently dropped."""
    tree = results_tree(
        tmp_path / 'nav',
        {
            'VOL1/N1000000001_1_CALIB': 'N1000000001_1_CALIB',
            'VOL2/N1000000001_1_CALIB': 'N1000000001_1_CALIB',
        },
    )
    _, duplicated, _ = compare(
        tree,
        observation_id=OBSERVATION,
        bundle_dir=bundle(tmp_path, ['N1000000001']),
        images=None,
    )
    assert duplicated == ['N1000000001']
