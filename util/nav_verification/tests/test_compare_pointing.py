"""Unit tests for the per-frame comparison's arithmetic."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from util.nav_verification.bundle_cassini_fring import NAC_PLATE_SCALE_URAD
from util.nav_verification.compare_pointing import (
    COMMON_OFFSET_TOLERANCE_PX,
    FrameComparison,
    compare,
    consensus_spread,
    measure_common_offset,
    report,
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


def test_a_shared_difference_is_measured() -> None:
    """A difference every agreeing frame shares is measured as their median."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    assert measure_common_offset(rows, agree_px=1.0) == pytest.approx((0.5, 0.5))


def test_measuring_a_shared_difference_leaves_every_frame_alone() -> None:
    """The shared difference is a measurement, never subtracted from a frame."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    measure_common_offset(rows, agree_px=1.0)
    assert rows[0].error_px == pytest.approx(0.707)


def test_a_badly_navigated_frame_does_not_move_the_shared_difference() -> None:
    """The median is over frames that already agree."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(10)]
    rows.append(comparison(60.0, 60.0, 84.85))
    assert measure_common_offset(rows, agree_px=1.0) == pytest.approx((0.5, 0.5))


def test_too_few_agreeing_frames_measure_no_shared_difference() -> None:
    """A median over three numbers is three numbers, not a measurement."""
    rows = [comparison(0.5, 0.5, 0.707) for _ in range(3)]
    assert measure_common_offset(rows, agree_px=1.0) is None


@pytest.mark.parametrize(
    'tolerance', [pytest.param(-1.0, id='negative'), pytest.param(math.nan, id='not-a-number')]
)
def test_a_tolerance_that_is_not_a_distance_is_refused(tolerance: float) -> None:
    """No frame is inside such a tolerance, so every statistic drawn from it is nonsense."""
    with pytest.raises(ValueError, match='a tolerance is a distance in pixels'):
        measure_common_offset([comparison(0.5, 0.5, 0.707)], agree_px=tolerance)


def reported(rows: list[FrameComparison]) -> None:
    """Measure the shared difference over the given comparisons and print the report.

    The caller reads what was printed from ``capsys``.

    Parameters:
        rows: The comparisons to report on.
    """
    common = measure_common_offset(rows, agree_px=1.0)
    report(rows, common=common, agree_px=1.0, wrong_px=2.0)


def test_a_frame_within_the_agreement_bound_agrees(capsys: pytest.CaptureFixture[str]) -> None:
    """A frame 0.9 px off is counted as agreeing."""
    reported([comparison(0.9, 0.0, 0.9)])
    assert 'agree (within 1.0 px): 1' in capsys.readouterr().out


def test_a_frame_between_the_bounds_is_marginal(capsys: pytest.CaptureFixture[str]) -> None:
    """A frame 1.5 px off is neither agreeing nor wrong."""
    reported([comparison(1.5, 0.0, 1.5)])
    assert 'marginal (1.0-2.0 px): 1' in capsys.readouterr().out


def test_a_frame_beyond_the_wrong_bound_is_wrong(capsys: pytest.CaptureFixture[str]) -> None:
    """A frame 2.5 px off is wrong."""
    reported([comparison(2.5, 0.0, 2.5)])
    assert 'WRONG (over 2.0 px): 1' in capsys.readouterr().out


def test_a_shared_difference_is_not_excused(capsys: pytest.CaptureFixture[str]) -> None:
    """Frames that all share a 1.5 px difference are all marginal, whatever they share."""
    rows = [comparison(1.5, 0.0, 1.5) for _ in range(10)]
    common = measure_common_offset(rows, agree_px=2.0)
    report(rows, common=common, agree_px=1.0, wrong_px=2.0)
    assert 'marginal (1.0-2.0 px): 10' in capsys.readouterr().out


def test_a_shared_difference_over_the_tolerance_is_called_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A difference every frame shares reads as loudly as a wrong frame."""
    step = COMMON_OFFSET_TOLERANCE_PX * 2.0
    reported([comparison(step, 0.0, step) for _ in range(10)])
    assert f'WRONG (over {COMMON_OFFSET_TOLERANCE_PX} px in an axis)' in capsys.readouterr().out


def test_the_difference_measured_on_the_f_ring_bundle_is_called_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The half pixel this tool found on ISS_006RI must not pass unremarked.

    The numbers are the ones the run reported before the star model's positions
    were corrected, +0.494 and +0.491 px along the camera's two axes.
    """
    rows = [comparison(0.494, 0.491, math.hypot(0.494, 0.491)) for _ in range(10)]
    reported(rows)
    assert f'WRONG (over {COMMON_OFFSET_TOLERANCE_PX} px in an axis)' in capsys.readouterr().out


def test_a_shared_difference_in_one_axis_alone_is_called_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One axis is enough: a row-versus-column error moves only one of them."""
    step = COMMON_OFFSET_TOLERANCE_PX * 2.0
    reported([comparison(0.0, step, step) for _ in range(10)])
    assert f'WRONG (over {COMMON_OFFSET_TOLERANCE_PX} px in an axis)' in capsys.readouterr().out


def test_a_shared_difference_inside_the_tolerance_is_not_called_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The median of a run whose pipelines agree lands inside its own noise."""
    step = COMMON_OFFSET_TOLERANCE_PX / 2.0
    reported([comparison(step, step, step * math.sqrt(2.0)) for _ in range(10)])
    assert 'in an axis' not in capsys.readouterr().out


def test_an_unmeasured_shared_difference_is_said_to_be_unmeasured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Too few agreeing frames prints no shared difference, which must not read as zero."""
    reported([comparison(0.5, 0.5, 0.707) for _ in range(3)])
    assert 'shared difference not measured' in capsys.readouterr().out


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


def test_a_record_the_bundle_holds_no_frame_for_is_left_out(tmp_path: Path) -> None:
    """With no selection, a record with nothing to compare against is not counted."""
    tree = results_tree(
        tmp_path / 'nav',
        {
            'VOL/N1000000001_1_CALIB': 'N1000000001_1_CALIB',
            'VOL/N1000000002_1_CALIB': 'N1000000002_1_CALIB',
        },
    )
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


def test_an_answer_behind_the_camera_is_still_measured(tmp_path: Path) -> None:
    """A bundle answer pointing the other way is a wrong frame, not a reason to stop."""
    document = {
        'status': 'success',
        'observation': {
            'image_name': 'N1000000001_1_CALIB',
            'camera': 'NAC',
            'image_shape': [1024, 1024],
        },
        'navigation_result': {
            'pointing': {'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        },
    }
    record = tmp_path / 'nav' / f'VOL/N1000000001_1_CALIB{METADATA_SUFFIX}'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(document), encoding='utf-8')
    collection = bundle(tmp_path, ['N1000000001'])
    (collection / OBSERVATION.lower() / '1000000001n_reproj_img_suppl.txt').write_text(
        'Navigation Type = Stars\n'
        'Navigated Boresight RA = 0.0 deg\n'
        'Navigated Boresight Dec = -90.0 deg\n',
        encoding='utf-8',
    )
    rows, _, _ = compare(
        tmp_path / 'nav', observation_id=OBSERVATION, bundle_dir=collection, images=None
    )
    assert rows[0].error_px == pytest.approx(math.pi / (NAC_PLATE_SCALE_URAD * 1e-6))
    assert rows[0].error_x_px is None
