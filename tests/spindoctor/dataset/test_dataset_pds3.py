import argparse
import json
import random
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from tests.spindoctor.conftest import metadata_document
from tests.spindoctor.dataset.conftest import coiss_filespecs, install_fake_index

from spindoctor.cli.results_index import ingest_metadata_files
from spindoctor.dataset import dataset_pds3 as dataset_pds3_module
from spindoctor.dataset.dataset_pds3 import DataSetPDS3
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.dataset.dataset_pds3_galileo_ssi import DataSetPDS3GalileoSSI
from spindoctor.dataset.results_filter import ResultsFilter, SelectionError
from spindoctor.nav_records import TreeTuning, UnlistableDirectoryError
from spindoctor.results_index import open_index


def _yielded_names(groups: list[Any]) -> list[str]:
    """Base image names (no suffix) of the yielded single-image groups."""
    return [g.image_files[0].image_file_name.split('_')[0] for g in groups]


def test_last_image_num_keeps_wac_frames(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # COISS indexes sort all N* rows before all W* rows, so the image-number
    # sequence resets partway through the volume. In-range WAC frames after the
    # reset must still be yielded (issue #136).
    numbers = [1000000100, 1000000101, 1000000102, 1000000103, 1000000104]
    install_fake_index(
        ds,
        monkeypatch,
        {'COISS_2001': coiss_filespecs('N', numbers) + coiss_filespecs('W', numbers)},
    )

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001'], img_end_num=1000000102))

    assert _yielded_names(groups) == [
        'N1000000100',
        'N1000000101',
        'N1000000102',
        'W1000000100',
        'W1000000101',
        'W1000000102',
    ]


def test_a_galileo_name_outside_the_product_rule_drops_its_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unnavigable product drops its row and leaves the enumeration running."""
    # GO_0016's SL9 directory stores a G product beside the R product of each of
    # its images. A G name carries no image number, and reading one raised
    # instead of dropping the row, which ended the enumeration of every
    # remaining volume (issue #17).
    ds = DataSetPDS3GalileoSSI()
    install_fake_index(
        ds,
        monkeypatch,
        {'GO_0016': ['SL9/C0248806645G.LBL', 'SL9/C0248806645R.LBL']},
    )

    groups = list(ds.yield_image_files_index(volumes=['GO_0016']))

    assert len(groups) == 1
    assert groups[0].image_files[0].label_file_name == 'C0248806645R.LBL'


def test_scan_stops_after_first_volume_fully_past_range(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Volumes are chronological, so the scan stops after the first volume whose
    # rows are all past img_end_num -- without reading any later volume's index.
    in_range = [1000000100, 1000000101]
    past_range = [1000000200, 1000000201]
    far_past_range = [1000000300, 1000000301]
    volumes_read = install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': coiss_filespecs('N', in_range) + coiss_filespecs('W', in_range),
            'COISS_2002': coiss_filespecs('N', past_range),
            'COISS_2003': coiss_filespecs('N', far_past_range),
        },
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002', 'COISS_2003'], img_end_num=1000000101
        )
    )

    assert len(groups) == 4
    assert volumes_read == ['COISS_2001', 'COISS_2002']


def test_img_name_list_range_clamp_still_stops_scan(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit image-name list clamps the number range; rows rejected by the
    # name list must still report past-the-range so the volume scan terminates.
    numbers = [1000000100, 1000000101, 1000000102]
    later = [1000000200, 1000000201]
    volumes_read = install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': coiss_filespecs('N', numbers) + coiss_filespecs('W', numbers),
            'COISS_2002': coiss_filespecs('N', later),
        },
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002'], img_name_list=['N1000000101']
        )
    )

    assert _yielded_names(groups) == ['N1000000101']
    assert volumes_read == ['COISS_2001', 'COISS_2002']


# --- Results-based filters (--has-offset-file and friends) ---


_FILTER_NUMS = [1000000100, 1000000101, 1000000102]


def _install_two_camera_index(ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a COISS_2001 index with three NAC and three WAC frames."""
    install_fake_index(
        ds,
        monkeypatch,
        {'COISS_2001': coiss_filespecs('N', _FILTER_NUMS) + coiss_filespecs('W', _FILTER_NUMS)},
    )


def _write_result_file(
    results_root: Path,
    volume: str,
    numbers: list[int],
    camera: str,
    num: int,
    suffix: str,
    content: str = 'x',
) -> None:
    """Write one synthetic result file where the pipeline would put it.

    The Cassini results path stub comes from the label filespec, so the
    filename carries the ``_CALIB`` label suffix.
    """
    range_dir = f'{numbers[0]:010d}_{numbers[-1]:010d}'
    path = results_root / volume / 'data' / range_dir / f'{camera}{num:010d}_1_CALIB{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _navigation_document(image_name: str) -> str:
    """Serialize the smallest document the ingest reads as a navigation result.

    Parameters:
        image_name: The image the document is about.

    Returns:
        The document as JSON text.
    """
    return json.dumps(
        {
            'status': 'success',
            'offset': [1.5, -2.5],
            'observation': {'image_name': image_name, 'instrument': 'coiss', 'camera': 'NAC'},
            'navigation_result': {'status_reason': 'ok'},
        }
    )


def _ingest_results_tree(results_root: Path, index_path: Path) -> str:
    """Build a results index over one tree and return its connection URL.

    Parameters:
        results_root: The results root to walk.
        index_path: Path of the index file to create.

    Returns:
        The connection URL of the index.
    """
    url = f'sqlite:///{index_path.as_posix()}'
    engine = open_index(url, create=True)
    try:
        ingest_metadata_files(
            engine, [results_root.as_posix()], logger=pdslogger.NullLogger(), tuning=TreeTuning()
        )
    finally:
        engine.dispose()
    return url


def _write_navigated_pair(results_root: Path) -> None:
    """Write the two navigated frames an ingest of this tree records.

    Parameters:
        results_root: The results root to write into.
    """
    _write_result_file(
        results_root,
        'COISS_2001',
        _FILTER_NUMS,
        'N',
        1000000101,
        '_metadata.json',
        _navigation_document('N1000000101_1_CALIB.IMG'),
    )
    _write_result_file(
        results_root,
        'COISS_2001',
        _FILTER_NUMS,
        'W',
        1000000100,
        '_metadata.json',
        _navigation_document('W1000000100_1_CALIB.IMG'),
    )


def _write_document_the_index_does_not_hold(results_root: Path) -> None:
    """Navigate a third frame after the ingest, so the two paths disagree.

    The index is a snapshot of the last ingest, so it does not hold this frame
    and the tree does. Which of the two answered the enumeration is then
    readable from the selection itself, which is the only thing that makes
    either path's test able to fail.

    Parameters:
        results_root: The results root to write into.
    """
    _write_result_file(
        results_root,
        'COISS_2001',
        _FILTER_NUMS,
        'N',
        1000000100,
        '_metadata.json',
        _navigation_document('N1000000100_1_CALIB.IMG'),
    )


def _indexed_tree_and_late_document(tmp_path: Path) -> tuple[Path, str]:
    """Build a results tree, index it, and then navigate one more frame.

    Parameters:
        tmp_path: Directory the tree and the index live under.

    Returns:
        The results root, and the connection URL of the index.
    """
    results_root = tmp_path / 'results'
    _write_navigated_pair(results_root)
    url = _ingest_results_tree(results_root, tmp_path / 'index.sqlite3')
    _write_document_the_index_does_not_hold(results_root)
    return results_root, url


def _program_arguments(
    ds: DataSetPDS3CassiniISS, *, declares_results_index_db: bool, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse the command line of a program with or without the index option.

    The namespace is built by the dataset's own selection parser rather than
    written out, because what is under test is the difference one declared
    option makes to it and nothing else about its shape.

    Parameters:
        ds: The dataset whose selection arguments the program offers.
        declares_results_index_db: Whether the program declares ``--results-index-db``.
        argv: The command line to parse, defaulting to an empty one.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    ds.add_selection_arguments(parser)
    if declares_results_index_db:
        parser.add_argument('--results-index-db', default=None)
    return parser.parse_args(argv or [])


def test_a_results_index_answers_the_offset_file_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The enumeration is answered by one query over the index rather than by a
    # walk of the tree, so the frame navigated after the ingest is not selected.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            results_index_db_url=url,
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'W1000000100']


def test_the_results_index_url_is_resolved_from_the_environment(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A program that declares --results-index-db and is given no value gets the one
    # the environment names, exactly as it gets the results root from
    # NAV_RESULTS_ROOT.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(ds, declares_results_index_db=True),
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'W1000000100']


def test_a_program_that_declares_no_index_flag_reads_the_results_tree(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An exported URL reaches the programs that declare the option and no
    # others. A program whose selection is meant to read files -- one this work
    # deliberately leaves reading them -- passes arguments that name no index,
    # and answers from the tree however the machine is configured. Its
    # arguments are otherwise those of a program that does declare it, so what
    # is under test is the declaration and nothing else.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(ds, declares_results_index_db=False),
        )
    )

    assert _yielded_names(groups) == ['N1000000100', 'N1000000101', 'W1000000100']


def test_an_exported_index_does_not_answer_the_resume_idiom_for_such_a_program(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --has-no-offset-file is how a run is resumed, and a snapshot answers it
    # with every image navigated since the last ingest. A program that declares
    # no index option must therefore not be handed one by the environment: the
    # tree holds a document for N1000000100, so the resume does not offer it
    # again, and only the three frames nothing was ever written for are left.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_no_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(ds, declares_results_index_db=False),
        )
    )

    assert _yielded_names(groups) == ['N1000000102', 'W1000000101', 'W1000000102']


def test_a_stale_index_re_selects_a_navigated_frame_for_a_program_that_declares_it(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The other half of the same rule, stated so the cost of declaring the
    # option is visible: a program that does declare it answers from the
    # snapshot, so N1000000100 -- navigated after the last ingest -- is offered
    # for navigation all over again.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_no_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(ds, declares_results_index_db=True),
        )
    )

    assert _yielded_names(groups) == [
        'N1000000100',
        'N1000000102',
        'W1000000101',
        'W1000000102',
    ]


def test_the_none_sentinel_reads_the_results_tree(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An exported index URL would otherwise reach every enumeration of every
    # program that declares the option. The sentinel opts one out: the tree is
    # read, so the frame the index does not hold is selected too.
    _install_two_camera_index(ds, monkeypatch)
    results_root, _url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', 'none')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(ds, declares_results_index_db=True),
        )
    )

    assert _yielded_names(groups) == ['N1000000100', 'N1000000101', 'W1000000100']


def test_the_none_sentinel_on_the_command_line_overrides_a_working_url(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The command-line opt-out, which is the only one an operator can see in
    # --help: the exported URL opens and answers, and passing the sentinel
    # still reads the tree.
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            arguments=_program_arguments(
                ds, declares_results_index_db=True, argv=['--results-index-db', 'none']
            ),
        )
    )

    assert _yielded_names(groups) == ['N1000000100', 'N1000000101', 'W1000000100']


def test_an_empty_results_index_url_refuses_the_selection(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An index named with no value stops the enumeration instead of steering it.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index and the exported variable are set through.
        tmp_path: Directory the tree and the index are built under.
    """
    _install_two_camera_index(ds, monkeypatch)
    results_root, _url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', '')

    with pytest.raises(SelectionError, match='empty value'):
        list(
            ds.yield_image_files_index(
                volumes=['COISS_2001'],
                has_offset_file=True,
                nav_results_root=str(results_root),
                arguments=_program_arguments(ds, declares_results_index_db=True),
            )
        )


def test_an_empty_results_index_url_names_the_level_that_set_it(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal a program prints is the resolver's, so it still says what to unset.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index and the exported variable are set through.
        tmp_path: Directory the tree and the index are built under.
    """
    _install_two_camera_index(ds, monkeypatch)
    results_root, _url = _indexed_tree_and_late_document(tmp_path)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', '')

    with pytest.raises(SelectionError, match='NAV_RESULTS_INDEX_DB'):
        list(
            ds.yield_image_files_index(
                volumes=['COISS_2001'],
                has_offset_file=True,
                nav_results_root=str(results_root),
                arguments=_program_arguments(ds, declares_results_index_db=True),
            )
        )


def test_has_offset_file_keeps_only_navigated_in_order(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only the frames with an existing metadata file are yielded, in normal
    # enumeration order (NAC rows before WAC rows within the volume). A summary
    # PNG alone must not satisfy the offset-file filter.
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'W', 1000000100, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000100, '_summary.png')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_file=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'W1000000100']


def test_has_no_offset_file_excludes_navigated(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'W', 1000000100, '_metadata.json')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_no_offset_file=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == [
        'N1000000100',
        'N1000000102',
        'W1000000101',
        'W1000000102',
    ]


def _error_document(num: int, **fields: Any) -> str:
    """Serialize one navigation document for a NAC frame of the filter volume.

    Parameters:
        num: The frame's image number.
        fields: What the document records, passed on to
            :func:`~tests.spindoctor.conftest.metadata_document`.

    Returns:
        The document as JSON text.
    """
    return json.dumps(metadata_document(image_name=f'N{num:010d}_1_CALIB.IMG', **fields))


def _write_error_metadata(tmp_path: Path) -> None:
    """Write metadata files: one success, one SPICE error, one non-SPICE error.

    Parameters:
        tmp_path: Directory the results root is written under.
    """
    contents = {
        1000000100: _error_document(1000000100),
        1000000101: _error_document(
            1000000101, status='error', status_error='missing_spice_data', offset=None
        ),
        1000000102: _error_document(
            1000000102, status='error', status_error='image_read_error', offset=None
        ),
    }
    for num, document in contents.items():
        _write_result_file(
            tmp_path,
            'COISS_2001',
            _FILTER_NUMS,
            'N',
            num,
            '_metadata.json',
            document,
        )


def test_has_offset_error_matches_any_fatal_error(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # WAC frames have no metadata file at all and are excluded by the implied
    # presence filter; the success metadata is excluded by its status.
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'N1000000102']


def test_has_no_offset_error_matches_only_documents_recording_none(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The filter keeps a document recording no fatal error and nothing else.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index is installed through.
        tmp_path: Directory the results root is written under.
    """
    # The one success is kept and the two fatal errors are not. The WAC frames
    # and the NAC frames outside the three written here have no metadata file
    # at all: a document that does not exist records no error, and this filter
    # asks what a document records, so the implied presence filter drops them.
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_no_offset_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000100']


def test_has_offset_spice_error_matches_only_spice(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_spice_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101']


def test_has_offset_nonspice_error_matches_only_nonspice(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_nonspice_error=True,
            nav_results_root=str(tmp_path),
        )
    )

    assert _yielded_names(groups) == ['N1000000102']


def _write_result_file_bytes(
    results_root: Path,
    volume: str,
    numbers: list[int],
    camera: str,
    num: int,
    suffix: str,
    content: bytes,
) -> None:
    """Write one synthetic result file with raw byte content (e.g. bad UTF-8)."""
    range_dir = f'{numbers[0]:010d}_{numbers[-1]:010d}'
    path = results_root / volume / 'data' / range_dir / f'{camera}{num:010d}_1_CALIB{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.mark.parametrize(
    'bad_content',
    [
        pytest.param(b'[1, 2, 3]', id='json-list'),
        pytest.param(b'null', id='json-null'),
        pytest.param(b'\xff\xfe not valid utf-8', id='invalid-utf8'),
    ],
)
def test_has_offset_error_excludes_malformed_metadata(
    ds: DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_content: bytes,
) -> None:
    # A metadata file that is valid JSON but not an object, or is not decodable
    # UTF-8, must exclude only its own image without crashing enumeration.
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(
        tmp_path,
        'COISS_2001',
        _FILTER_NUMS,
        'N',
        1000000100,
        '_metadata.json',
        _error_document(1000000100, status='error', offset=None),
    )
    _write_result_file_bytes(
        tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json', bad_content
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000100']


_NOT_A_NAVIGATION_DOCUMENT = json.dumps({'status': 'error', 'status_error': 'missing_spice_data'})
"""A JSON object that reads perfectly and is no navigation result of any schema.

It carries the two fields the error filters name and nothing else a document has
-- no image, no mission, no navigation result -- so a filter that read those two
fields out of whatever it could parse would select it, and one that reads what a
document records about its image finds nothing recorded.
"""


def test_an_object_that_is_no_navigation_document_has_an_offset_file(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Presence is a question about the file, so whatever is in it, it is there.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index is installed through.
        tmp_path: Directory the results root is written under.
    """
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(
        tmp_path,
        'COISS_2001',
        _FILTER_NUMS,
        'N',
        1000000100,
        '_metadata.json',
        _NOT_A_NAVIGATION_DOCUMENT,
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_file=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000100']


@pytest.mark.parametrize(
    'flag',
    [
        'has_offset_error',
        'has_no_offset_error',
        'has_offset_spice_error',
        'has_offset_nonspice_error',
    ],
)
def test_an_object_that_is_no_navigation_document_matches_no_error_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str
) -> None:
    """What such a file records is unknown, which is neither an error nor the lack of one.

    The filter phrased in the negative is one of the four for exactly that
    reason: selecting this image would claim its navigation ran to an outcome,
    and nothing in the file says an image was navigated at all.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index is installed through.
        tmp_path: Directory the results root is written under.
        flag: The error filter, one per flag that reads a document.
    """
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(
        tmp_path,
        'COISS_2001',
        _FILTER_NUMS,
        'N',
        1000000100,
        '_metadata.json',
        _NOT_A_NAVIGATION_DOCUMENT,
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], nav_results_root=str(tmp_path), **{flag: True}
        )
    )

    assert _yielded_names(groups) == []


def test_results_scan_propagates_non_missing_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directory that is there and will not be read is not an empty directory.

    Permission denied, a share that has gone away, a backend that stopped
    answering: under every one of them there may be documents the filter cannot
    see, and answering from what it did see selects images it has no evidence
    about.  The volume's directory is written first, so the refusal is about
    reading it rather than about its absence.

    Parameters:
        monkeypatch: Fixture the directory listing is replaced through.
        tmp_path: Directory the results root is written under.
    """
    (tmp_path / 'COISS_2001').mkdir()

    def boom(_self: FCPath) -> object:
        raise PermissionError('results scan denied')

    monkeypatch.setattr(FCPath, 'iterdir_metadata', boom)

    with pytest.raises(UnlistableDirectoryError, match='results scan denied'):
        ResultsFilter(
            ['COISS_2001'],
            str(tmp_path),
            has_offset_file=True,
            logger=pdslogger.NullLogger(),
        )


def test_results_scan_passes_over_a_volume_with_no_results_directory(tmp_path: Path) -> None:
    """A volume nobody has navigated has no directory, which ends no enumeration.

    It is the ordinary state of a results root part way through a campaign, and
    the volumes after it are the ones a run is about: a refusal here would leave
    an operator unable to select against a root until every volume they named had
    been navigated at least once.

    Parameters:
        tmp_path: Directory the results root is written under.
    """
    _write_result_file(
        tmp_path,
        'COISS_2002',
        _FILTER_NUMS,
        'N',
        1000000100,
        '_metadata.json',
        _error_document(1000000100),
    )

    results_filter = ResultsFilter(
        ['COISS_2001', 'COISS_2002'],
        str(tmp_path),
        has_offset_file=True,
        logger=pdslogger.NullLogger(),
    )

    assert results_filter.passes('COISS_2002/data/1000000100_1000000102/N1000000100_1_CALIB')


CONTRADICTION_REFUSAL = r'mutually exclusive|cannot be combined with'
"""The two shapes a refusal of contradictory selection flags takes.

Two flags that exclude each other and nothing else are mutually exclusive.  One
flag that excludes several which are satisfiable together is named against the
ones it excludes instead, since "mutually exclusive" over a set claims an
exclusion between every pair in it.
"""


@pytest.mark.parametrize(
    'flags',
    [
        {'has_offset_file': True, 'has_no_offset_file': True},
        {'has_offset_spice_error': True, 'has_offset_nonspice_error': True},
        {'has_offset_error': True, 'has_no_offset_file': True},
        {'has_offset_error': True, 'has_no_offset_error': True},
        {'has_offset_spice_error': True, 'has_no_offset_error': True},
        {'has_offset_nonspice_error': True, 'has_no_offset_error': True},
        {'has_no_offset_error': True, 'has_no_offset_file': True},
    ],
)
def test_contradictory_results_flags_raise(
    ds: DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: dict[str, bool],
) -> None:
    """A pair of selection flags no image could satisfy is refused, not answered.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index is installed through.
        tmp_path: Directory the results root is written under.
        flags: The contradictory pair under test.
    """
    _install_two_camera_index(ds, monkeypatch)

    with pytest.raises(ValueError, match=CONTRADICTION_REFUSAL):
        list(
            ds.yield_image_files_index(
                volumes=['COISS_2001'], nav_results_root=str(tmp_path), **flags
            )
        )


# --- Uniform random sampling (--choose-random-images) ---


def test_choose_random_images_pool_spans_all_volumes(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool is built from every selected volume before sampling. With
    # shuffle replaced by reverse, the yields are deterministic and cross the
    # volume boundary, in shuffled (not enumeration) order.
    nums1 = [1000000100, 1000000101, 1000000102]
    nums2 = [1000000200, 1000000201, 1000000202]
    volumes_read = install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': coiss_filespecs('N', nums1),
            'COISS_2002': coiss_filespecs('N', nums2),
        },
    )
    monkeypatch.setattr(random, 'shuffle', lambda pool: pool.reverse())

    groups = list(
        ds.yield_image_files_index(volumes=['COISS_2001', 'COISS_2002'], choose_random_images=4)
    )

    assert _yielded_names(groups) == [
        'N1000000202',
        'N1000000201',
        'N1000000200',
        'N1000000102',
    ]
    assert volumes_read == ['COISS_2001', 'COISS_2002']


def test_choose_random_images_returns_requested_count(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_two_camera_index(ds, monkeypatch)

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001'], choose_random_images=2))

    names = _yielded_names(groups)
    assert len(names) == 2
    assert len(set(names)) == 2
    all_names = {f'{camera}{num}' for camera in 'NW' for num in _FILTER_NUMS}
    assert set(names) <= all_names


def test_choose_random_images_with_offset_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Random sampling combined with a results filter: only navigated images are
    # candidates, and asking for more than exist returns exactly the candidates.
    nums1 = [1000000100, 1000000101, 1000000102]
    nums2 = [1000000200, 1000000201, 1000000202]
    install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': coiss_filespecs('N', nums1),
            'COISS_2002': coiss_filespecs('N', nums2),
        },
    )
    _write_result_file(tmp_path, 'COISS_2001', nums1, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2002', nums2, 'N', 1000000201, '_metadata.json')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002'],
            choose_random_images=10,
            has_offset_file=True,
            nav_results_root=str(tmp_path),
        )
    )

    assert sorted(_yielded_names(groups)) == ['N1000000101', 'N1000000201']


@pytest.mark.parametrize('bad', [0, -1, -5])
def test_choose_random_images_rejects_non_positive_programmatic(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, bad: int
) -> None:
    # 0 would silently disable sampling (yielding everything) and a negative
    # value would yield exactly one frame; both are rejected at the boundary.
    _install_two_camera_index(ds, monkeypatch)

    with pytest.raises(ValueError, match='positive integer'):
        list(ds.yield_image_files_index(volumes=['COISS_2001'], choose_random_images=bad))


@pytest.mark.parametrize('bad', ['0', '-3'])
def test_choose_random_images_argparse_rejects_non_positive(
    bad: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(['--choose-random-images', bad])

    assert 'positive integer' in capsys.readouterr().err


def test_choose_random_images_argparse_accepts_positive() -> None:
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    arguments = parser.parse_args(['--choose-random-images', '5'])

    assert arguments.choose_random_images == 5


def test_selection_arguments_include_results_filters() -> None:
    """The command-line parser carries the presence and error filters."""
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    arguments = parser.parse_args(['--has-offset-file', '--has-offset-error'])

    assert arguments.has_offset_file is True
    assert arguments.has_offset_error is True
    assert arguments.has_offset_spice_error is False
    assert arguments.has_no_offset_error is False


def test_the_negative_error_filter_is_one_of_the_selection_arguments() -> None:
    """The parser carries the negative error filter beside the positive ones."""
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    arguments = parser.parse_args(['--has-offset-file', '--has-no-offset-error'])

    assert arguments.has_no_offset_error is True
    assert arguments.has_offset_error is False


def test_yielded_imagefile_carries_label_resolver(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_index(
        ds, monkeypatch, {'COISS_2001': coiss_filespecs('N', [1000000100, 1000000100])[:1]}
    )

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001']))

    assert len(groups) == 1
    assert groups[0].image_files[0].image_url_resolver is not None


def _closes_counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count every time an enumeration closes the results filter it built.

    Parameters:
        monkeypatch: Fixture the count is installed through.

    Returns:
        A one-element list holding the count, which the caller reads after the
        enumeration it is watching has ended.
    """
    closes = [0]
    original = ResultsFilter.close

    def counting(self: ResultsFilter) -> None:
        closes[0] += 1
        original(self)

    monkeypatch.setattr(ResultsFilter, 'close', counting)
    return closes


def test_an_enumeration_that_runs_to_the_end_closes_its_results_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An error filter holds a storage open between batches and must give it back.

    Over a results index that storage is a connection pool, and a long run makes
    one enumeration after another, so a filter nobody closes is a pool per
    enumeration held until the interpreter collects it.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index and the count are installed through.
        tmp_path: Directory the results root is written under.
    """
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)
    closes = _closes_counted(monkeypatch)

    list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert closes[0] == 1


def test_an_enumeration_abandoned_part_way_closes_its_results_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller is free to stop reading, and stopping must not leak the storage.

    Taking the first image of a selection and walking away is what every run
    with a result limit does, and a generator closed part way is the case a
    filter released only at the end of the loop would leak.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index and the count are installed through.
        tmp_path: Directory the results root is written under.
    """
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)
    closes = _closes_counted(monkeypatch)

    groups = ds.yield_image_files_index(
        volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
    )
    next(groups)
    groups.close()

    assert closes[0] == 1


def test_the_shared_enumeration_abandoned_part_way_closes_its_results_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every dataset without a grouping of its own gets this one, so it is tested.

    The instrument that groups its frames overrides the enumeration, so a test
    that only ever runs the override leaves the implementation every other
    dataset inherits uncovered.  This one drives the shared implementation
    directly.

    Parameters:
        ds: The dataset under test, driven through the shared enumeration.
        monkeypatch: Fixture the index and the count are installed through.
        tmp_path: Directory the results root is written under.
    """
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)
    closes = _closes_counted(monkeypatch)

    groups = DataSetPDS3.yield_image_files_index(
        ds, volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
    )
    next(groups)
    groups.close()

    assert closes[0] == 1


def test_the_enumeration_hands_the_filter_the_tuning_the_configuration_names(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dataset is a program's front end onto the tree, so it resolves the tuning.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the configuration door and the filter are replaced through.
        tmp_path: Directory the tree and the index are written under.
    """
    _install_two_camera_index(ds, monkeypatch)
    results_root, url = _indexed_tree_and_late_document(tmp_path)
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    handed: list[Any] = []

    class Recording(ResultsFilter):
        """A filter that notes the tuning it was handed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            handed.append(kwargs.get('tuning'))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(dataset_pds3_module, 'get_results_tree_tuning', lambda config: configured)
    monkeypatch.setattr(dataset_pds3_module, 'ResultsFilter', Recording)
    list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_file=True,
            nav_results_root=str(results_root),
            results_index_db_url=url,
        )
    )
    assert handed == [configured]
