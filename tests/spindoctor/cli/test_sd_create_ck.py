"""End-to-end tests for the products of a clean ``sd_create_ck`` run.

One run, one mission, four images: two that navigate against different original
kernels, one that did not navigate at all, and one whose recorded baseline no
candidate reproduces.  What is asserted is everything the run leaves behind --
the corrected kernels and their comment areas, the meta-kernel's load order, the
report's one row per image, and the two logs -- because those are the products,
and each of them can be wrong on its own while the others look right.  What the
driver refuses, and with what exit status, is the business of
``test_sd_create_ck_refusals``.

Every kernel is written by the test.  The driver furnishes the pool itself and,
being a program, does not unload it again, so the pool fixture here records what
was furnished on the way in and unloads whatever the run added on the way out.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from tests.kernel_pool import isolated_kernel_pool
from tests.spindoctor.cli.ck.ck_helpers import CASSINI_CK_FRAME_ID, write_ck
from tests.spindoctor.cli.sd_create_ck_helpers import (
    BASELINE_A,
    BASELINE_B,
    IMAGE_A_ET,
    report_rows,
    run_driver,
    run_log,
    utc_of,
)

from spindoctor.cli import sd_create_ck
from spindoctor.cli.ck.comments import read_comment_area
from spindoctor.cli.ck.segment import CkSegment
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import open_record_source


@pytest.fixture(scope='module', autouse=True)
def empty_kernel_pool() -> Iterator[None]:
    """Run this module against an emptied SPICE pool, and put the pool back.

    A real C-kernel another test file left furnished in the worker defines the
    same objects the hermetic kernels here do, so it would answer the driver's
    reproduction lookups beside the candidate under test -- which the driver
    rightly refuses to run with.  The isolation is the same loan the ck
    package's conftest makes.

    Yields:
        Nothing; the module's tests run against an empty pool.
    """
    with isolated_kernel_pool():
        yield


# ---------------------------------------------------------------------------
# The corrected kernels
# ---------------------------------------------------------------------------


def test_one_corrected_kernel_per_original(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Each original an image navigated against gets a mirror of its own."""
    run_driver(run_tree, monkeypatch)
    written = sorted(path.name for path in run_tree['output'].glob('*.bc'))
    assert written == ['orig_a_nav.bc', 'orig_b_nav.bc']


def test_no_kernel_is_written_for_an_original_nothing_reproduces(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The drifted image writes nothing at all rather than an empty file."""
    run_driver(run_tree, monkeypatch)
    assert not (run_tree['output'] / 'orig_a_nav_nav.bc').exists()


def test_the_corrected_kernel_describes_the_object(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A plain reader finds the bus in the file, with no writer code present."""
    run_driver(run_tree, monkeypatch)
    path = run_tree['output'] / 'orig_a_nav.bc'
    assert [int(value) for value in cspyce.ckobj(str(path))] == [CASSINI_CK_FRAME_ID]


# ---------------------------------------------------------------------------
# The comment area
# ---------------------------------------------------------------------------


def test_the_comment_area_names_the_baseline(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Read back through the same DAF interface a consumer would use."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(BASELINE_A in line for line in lines)


def test_the_comment_area_names_the_clock_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the one the run chose from the image's own provenance."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any('test.tsc' in line for line in lines)


def test_the_comment_area_names_the_image_it_carries(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And only that image, not every image the run considered."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(line.startswith('A_CALIB') for line in lines)
    assert not any(line.startswith('B_CALIB') for line in lines)


def test_each_file_names_the_image_it_actually_carries(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Checked on the second file too, where taking the run's first image would
    read as a plausible comment area on a file that does not hold that image."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_b_nav.bc')
    assert any(line.startswith('B_CALIB') for line in lines)
    assert not any(line.startswith('A_CALIB') for line in lines)


def test_the_comment_area_names_the_generator_version(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A file says what wrote it, so a regeneration can be told from an old one."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(line.startswith('Generator version:') for line in lines)


def test_the_comment_area_names_the_configuration_hash(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And under what configuration it ran."""
    run_driver(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    stated = [line for line in lines if line.startswith('Configuration hash:')]
    assert len(stated) == 1
    assert len(stated[0].split()[-1]) == 64


# ---------------------------------------------------------------------------
# The meta-kernel
# ---------------------------------------------------------------------------


def test_the_meta_kernel_furnishes_originals_before_corrections(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Furnished for real, because load order is the only thing it decides."""
    run_driver(run_tree, monkeypatch)
    meta = run_tree['output'] / 'coiss_nav.tm'
    cspyce.furnsh(str(meta))
    try:
        loaded = [str(cspyce.kdata(at, 'CK')[0]) for at in range(int(cspyce.ktotal('CK')))]
    finally:
        cspyce.unload(str(meta))
    assert loaded == [
        str(run_tree['kernels'] / BASELINE_A),
        str(run_tree['kernels'] / BASELINE_B),
        str(run_tree['output'] / 'orig_a_nav.bc'),
        str(run_tree['output'] / 'orig_b_nav.bc'),
    ]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_every_image_considered_appears_exactly_once(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Four images considered, four rows, whatever became of each."""
    run_driver(run_tree, monkeypatch)
    rows = report_rows(run_tree)
    assert sorted(rows) == ['A_CALIB', 'B_CALIB', 'C_CALIB', 'D_CALIB']


def test_a_corrected_image_names_the_file_carrying_its_segment(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And it is the mirror of the original that image navigated against."""
    run_driver(run_tree, monkeypatch)
    rows = report_rows(run_tree)
    assert rows['A_CALIB']['source_bc'] == 'orig_a_nav.bc'
    assert rows['B_CALIB']['source_bc'] == 'orig_b_nav.bc'


def test_a_corrected_image_carries_no_omission_reason(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A row says one thing or the other, never both."""
    run_driver(run_tree, monkeypatch)
    assert report_rows(run_tree)['A_CALIB']['omission_reason'] == ''


def test_an_image_that_did_not_navigate_is_reported_as_not_eligible(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With no source file, so it cannot be read as corrected."""
    run_driver(run_tree, monkeypatch)
    row = report_rows(run_tree)['C_CALIB']
    assert row['omission_reason'] == 'not_eligible'
    assert row['source_bc'] == ''


def test_an_image_whose_baseline_no_candidate_reproduces_is_reported(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the detector for a kernel set that changed since navigation."""
    run_driver(run_tree, monkeypatch)
    assert report_rows(run_tree)['D_CALIB']['omission_reason'] == 'no_reproducing_baseline'


def test_the_report_carries_the_measurement_the_metadata_recorded(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The offset column is the recorded one, unrounded."""
    run_driver(run_tree, monkeypatch)
    row = report_rows(run_tree)['A_CALIB']
    assert row['offset_dv'] == '-3.25'
    assert row['confidence_rank'] == 'high'


# ---------------------------------------------------------------------------
# Time selection
# ---------------------------------------------------------------------------


def test_a_time_range_selects_the_images_inside_it(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An image outside the range was never considered and gets no row."""
    stop = utc_of(run_tree, IMAGE_A_ET + 100.0)
    run_driver(run_tree, monkeypatch, '--stop-time', stop)
    assert 'B_CALIB' not in report_rows(run_tree)


def test_a_time_range_still_writes_the_images_inside_it(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Only the original the surviving images navigated against is mirrored."""
    stop = utc_of(run_tree, IMAGE_A_ET + 100.0)
    run_driver(run_tree, monkeypatch, '--stop-time', stop)
    assert sorted(path.name for path in run_tree['output'].glob('*.bc')) == ['orig_a_nav.bc']


# ---------------------------------------------------------------------------
# Both logs
# ---------------------------------------------------------------------------


def test_an_omission_reaches_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An operator watching a batch sees it without opening a per-image log."""
    run_driver(run_tree, monkeypatch)
    assert 'D_CALIB: no corrected segment written (no_reproducing_baseline)' in run_log(run_tree)


def test_the_run_log_counts_each_omission_reason(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """One line per reason at the end, so a batch's shape is readable at a glance."""
    run_driver(run_tree, monkeypatch)
    assert 'Images omitted, no_reproducing_baseline: 1' in run_log(run_tree)


def test_a_directory_the_walk_declined_reaches_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The walk says this, and it only reaches an operator through the run's own log.

    The results root is given a second path to a directory it already holds, so
    the walk meets that directory twice and declines the second visit.  The root
    is still wholly listed, which is why the run goes on -- and why a run told
    nothing would read as one that met no such thing at all.  The record source
    holds no logger of its own, so the line appears only because the program
    lends the source the log it is already writing.
    """
    (run_tree['results'] / 'vol' / 'again').symlink_to(
        run_tree['results'] / 'vol', target_is_directory=True
    )
    run_driver(run_tree, monkeypatch)
    assert 'reached a second way' in run_log(run_tree)


def test_the_run_log_counts_the_images_corrected(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Against which the omissions can be read as a fraction of the batch."""
    run_driver(run_tree, monkeypatch)
    assert 'Images corrected 2' in run_log(run_tree)


def test_the_run_log_names_an_object_whose_coverage_was_skipped(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The operator sees which objects the index could not clock, and the run goes on.

    The kernel directory gains a file describing only an object with no
    spacecraft clock, the shape a merged New Horizons pointing file has.  Its
    coverage cannot be expressed in TDB, so the file can never supply a
    baseline, and the run says so once instead of aborting the scan.
    """
    ticks = np.asarray([0.0, 1.0e6, 2.0e6], dtype=np.float64)
    quats = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (3, 1))
    write_ck(
        run_tree['kernels'] / 'merged_clockless_v001.bc',
        [CkSegment(ck_frame_id=-1, segid='clockless only', sclkdp=ticks, quats=quats, avvs=None)],
    )
    run_driver(run_tree, monkeypatch)
    assert 'Skipped the coverage of CK object(s) -1' in run_log(run_tree)


def test_an_omission_reaches_the_image_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With the detail, in the log of the image it happened to."""
    run_driver(run_tree, monkeypatch)
    logs = list((run_tree['output'] / 'logs').rglob('*D_CALIB*'))
    assert len(logs) == 1
    assert 'no_reproducing_baseline' in logs[0].read_text()


def test_a_corrected_image_records_its_file_in_its_own_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """So an image's own log answers what became of it either way."""
    run_driver(run_tree, monkeypatch)
    logs = list((run_tree['output'] / 'logs').rglob('*A_CALIB*'))
    assert len(logs) == 1
    assert 'orig_a_nav.bc' in logs[0].read_text()


# ---------------------------------------------------------------------------
# Paths a consumer has to be able to resolve
# ---------------------------------------------------------------------------


def test_the_meta_kernel_names_absolute_paths(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """SPICE resolves a relative name against the consumer's directory, not ours."""
    monkeypatch.chdir(run_tree['output'].parent)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            'results',
            '--kernel-dir',
            'kernels',
            '--output-dir',
            'output',
            '--log-root',
            'output/logs',
        ],
    )
    with pytest.raises(SystemExit):
        sd_create_ck.main()
    meta = str(run_tree['output'] / 'coiss_nav.tm')
    cspyce.furnsh(meta)
    try:
        named = [str(cspyce.kdata(at, 'CK')[0]) for at in range(int(cspyce.ktotal('CK')))]
    finally:
        cspyce.unload(meta)
    assert [name for name in named if not name.startswith('/')] == []


def test_a_meta_kernel_written_from_a_relative_run_still_furnishes(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None, tmp_path: Path
) -> None:
    """Furnished from somewhere else entirely, which is where a consumer is."""
    monkeypatch.chdir(run_tree['output'].parent)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            'results',
            '--kernel-dir',
            'kernels',
            '--output-dir',
            'output',
            '--log-root',
            'output/logs',
        ],
    )
    with pytest.raises(SystemExit):
        sd_create_ck.main()
    monkeypatch.chdir(tmp_path.parent)
    meta = str(run_tree['output'] / 'coiss_nav.tm')
    cspyce.furnsh(meta)
    try:
        assert int(cspyce.ktotal('CK')) == 4
    finally:
        cspyce.unload(meta)


def test_a_remote_directory_is_left_as_it_was_given() -> None:
    """It is already absolute, and there is no local directory to resolve it against."""
    assert sd_create_ck.absolute_directory('gs://bucket/kernels') == 'gs://bucket/kernels'


def test_the_records_are_read_at_the_tuning_the_configuration_names(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver resolves the tuning once, at its top, and opens its source with it."""
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    handed: list[Any] = []

    def recording(roots: Any, **kwargs: Any) -> Any:
        handed.append(kwargs.get('tuning'))
        return open_record_source(roots, **kwargs)

    monkeypatch.setattr(sd_create_ck, 'get_results_tree_tuning', lambda config: configured)
    monkeypatch.setattr(sd_create_ck, 'open_record_source', recording)
    run_driver(run_tree, monkeypatch)
    assert handed == [configured]
