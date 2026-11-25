"""What a run that selected on an error filter already knows about each image.

An error filter decides by reading what a document records, so by the time the
enumeration yields an image it has retrieved and parsed that image's navigation
document.  The record travels with the image, and the stage that goes on to
process it reads no document of its own.

The measurement these tests make is the point of the change, and the reason a
test comparing only the output would be worthless: the two readers hold
``FileCache`` instances of their own, in separate directories, so a second read
of a document on a cloud results root is a second download rather than a cache
hit.  A local tree cannot show the download, but it can show the read, and the
read is what the download follows from.  So every assertion below counts reads
of the document rather than comparing what came out of it.

Only the document path carries anything.  A run answered out of a results index
narrows on columns, and the record its per-image stage wants is rebuilt from a
different column set, so the two reads want different things and stay separate;
the index-backed cases here assert that nothing is carried and that the index is
still what answers.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import index_url, ingest_tree, metadata_document, write_metadata
from tests.spindoctor.dataset.conftest import (
    coiss_filespecs,
    install_fake_index,
    null_logger,
)

from spindoctor.cli.reproj.pointing_source import FilePointingSource
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.dataset.results_filter import ResultsFilter

VOLUME = 'COISS_2001'
"""The one volume the fixture index and the fixture results root hold."""

NUMBERS = [1000000100, 1000000101, 1000000102]
"""The image numbers the fixture index serves, for one camera."""

RANGE_DIR = f'{NUMBERS[0]:010d}_{NUMBERS[-1]:010d}'
"""The range directory Cassini results paths put those numbers under."""

SUCCEEDED = f'{VOLUME}/data/{RANGE_DIR}/N1000000100_1_CALIB'
"""The image whose document records a run that succeeded, with a pointing."""

SPICE_ERROR = f'{VOLUME}/data/{RANGE_DIR}/N1000000101_1_CALIB'
"""The image whose document records a fatal SPICE error."""

OTHER_ERROR = f'{VOLUME}/data/{RANGE_DIR}/N1000000102_1_CALIB'
"""The image whose document records a fatal error of another kind."""

POINTING: dict[str, Any] = {
    'cmatrix': [
        0.9636758075215185,
        0.24394452671199518,
        0.10871985046444949,
        -0.23632717093513517,
        0.5892492544213016,
        0.7726155476313796,
        0.1244122432702933,
        -0.7702443664521031,
        0.6254959709488565,
    ],
    'cmatrix_original': [
        0.963676611721357,
        0.24393954782302113,
        0.1087238935521779,
        -0.23632299640079324,
        0.5892234094236003,
        0.772636534962837,
        0.1244139437257574,
        -0.7702657144097279,
        0.6254693436224319,
    ],
    'camera_frame': 'CASSINI_ISS_NAC',
    'camera_frame_id': -82360,
    'ck_frame_id': -82000,
}
"""A complete recorded pointing block, so the kept image supplies a C-matrix."""

TIMES: dict[str, Any] = {
    'start_et': 136576860.0424845,
    'stop_et': 136576860.30248448,
    'midtime_et': 136576860.1724845,
    'exposure_s': 0.26,
    'sclk_start': '1/1461997416.044',
    'sclk_midtime': '1/1461997416.078',
    'sclk_stop': '1/1461997416.111',
}
"""The exposure epochs recorded beside that pointing block."""


def documents() -> dict[str, dict[str, Any]]:
    """Build the three documents the fixture results root holds.

    Returns:
        Results path stub mapped to the document recorded there.
    """
    return {
        SUCCEEDED: metadata_document(
            image_name='N1000000100_1.IMG',
            offset=[5.6005, 1.0788],
            times=TIMES,
            pointing=POINTING,
        ),
        SPICE_ERROR: metadata_document(
            image_name='N1000000101_1.IMG',
            status='error',
            status_error='missing_spice_data',
            offset=None,
        ),
        OTHER_ERROR: metadata_document(
            image_name='N1000000102_1.IMG',
            status='error',
            status_error='image_read_error',
            offset=None,
        ),
    }


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    """Write the fixture results root and return it.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for stub, document in documents().items():
        write_metadata(root, stub, document)
    return root


@pytest.fixture
def ds_with_an_index(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> DataSetPDS3CassiniISS:
    """Return a dataset whose volume index serves the three fixture frames.

    Parameters:
        ds: The dataset under test.
        monkeypatch: Fixture the index reads are replaced through.

    Returns:
        The same dataset, reading no holdings.
    """
    install_fake_index(ds, monkeypatch, {VOLUME: coiss_filespecs('N', NUMBERS)})
    return ds


def enumerate_images(
    dataset: DataSetPDS3CassiniISS,
    results_root: Path,
    *,
    results_index_db_url: str | None = None,
    **flags: bool,
) -> list[ImageFile]:
    """Run the enumeration exactly as a program does, and collect what it yields.

    Parameters:
        dataset: The dataset, with its volume index already installed.
        results_root: The navigation results root the filter reads.
        results_index_db_url: An index to answer the filter from, or None to
            read the documents.
        flags: The selection flags to apply.

    Returns:
        The images the enumeration yielded, in the order it yielded them.
    """
    groups = dataset.yield_image_files_index(
        volumes=[VOLUME],
        nav_results_root=str(results_root),
        results_index_db_url=results_index_db_url,
        **flags,
    )
    return [image for group in groups for image in group.image_files]


@pytest.fixture
def count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count reads of a document's bytes, wherever in the run they happen.

    Wrapped on ``FCPath.read_text`` rather than on any one reader, because the
    two readers under test are different code: the seam reads a retrieved
    document and the per-image stage reads one under the results root.  Both end
    at this call, and on a cloud root each of them is a download.

    Parameters:
        monkeypatch: Fixture the reader is wrapped through.

    Returns:
        A list that grows by one entry per document read, holding the file's
        name.
    """
    read: list[str] = []
    real_read_text = FCPath.read_text

    def counting(self: FCPath, **kwargs: Any) -> str:
        name = self.as_posix().rsplit('/', 1)[-1]
        if name.endswith('_metadata.json'):
            read.append(name)
        return real_read_text(self, **kwargs)

    monkeypatch.setattr(FCPath, 'read_text', counting)
    return read


def document_name(stub: str) -> str:
    """Return the file name of one stub's document.

    Parameters:
        stub: The image's results path stub.

    Returns:
        The document's file name, which is what :func:`count_reads` records.
    """
    return f'{stub.rsplit("/", 1)[-1]}_metadata.json'


@pytest.fixture
def indexed(results_root: Path, tmp_path: Path) -> str:
    """Ingest the fixture results root and return the index URL.

    Parameters:
        results_root: The results root to ingest.
        tmp_path: Directory the index file is written into.

    Returns:
        The connection URL of the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [results_root], logger=null_logger())
    return url


# ---------------------------------------------------------------------------
# What an error filter leaves on the images it kept
# ---------------------------------------------------------------------------


def test_an_image_an_error_filter_kept_carries_the_record_of_its_document(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path
) -> None:
    """The record travels with the image, and it is the document's own content."""
    kept = enumerate_images(ds_with_an_index, results_root, has_no_offset_error=True)

    assert [image.nav_record for image in kept] == [documents()[SUCCEEDED]]


def test_the_carried_record_is_what_the_file_on_disk_holds(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path
) -> None:
    """Compared against the bytes rather than against the factory that wrote them."""
    kept = enumerate_images(ds_with_an_index, results_root, has_no_offset_error=True)
    on_disk = json.loads((results_root / f'{SUCCEEDED}_metadata.json').read_text())

    assert kept[0].nav_record == on_disk


def test_every_image_an_error_filter_kept_carries_one(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path
) -> None:
    """Two kept images, two records: nothing is carried for the first alone."""
    kept = enumerate_images(ds_with_an_index, results_root, has_offset_error=True)

    assert [image.results_path_stub for image in kept if image.nav_record is not None] == [
        SPICE_ERROR,
        OTHER_ERROR,
    ]


def test_a_filter_that_reads_no_document_carries_nothing(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path
) -> None:
    """A presence filter is settled by a listing, so no document is parsed."""
    kept = enumerate_images(ds_with_an_index, results_root, has_offset_file=True)

    assert [image.nav_record for image in kept] == [None, None, None]


def test_a_run_with_no_results_filter_at_all_carries_nothing(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path
) -> None:
    """Nothing asked what any document records, so nothing read one."""
    kept = enumerate_images(ds_with_an_index, results_root)

    assert [image.nav_record for image in kept] == [None, None, None]


# ---------------------------------------------------------------------------
# How many times the run read each document
# ---------------------------------------------------------------------------


def test_the_enumeration_reads_each_candidates_document_once(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, count_reads: list[str]
) -> None:
    """The baseline the count below is measured against: one read per candidate.

    Stated on its own so that an assertion of one read after the per-image stage
    means one read in total rather than one read that never happened.
    """
    enumerate_images(ds_with_an_index, results_root, has_no_offset_error=True)

    assert sorted(count_reads) == sorted(document_name(stub) for stub in documents())


def test_reading_the_record_of_a_kept_image_reads_nothing_further(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, count_reads: list[str]
) -> None:
    """What ``sd_backplanes`` does per image, counted across the whole run."""
    kept = enumerate_images(ds_with_an_index, results_root, has_no_offset_error=True)
    source = FilePointingSource(FCPath(results_root))
    for image in kept:
        source.read_record(image)

    assert count_reads.count(document_name(SUCCEEDED)) == 1


def test_loading_the_pointing_of_a_kept_image_reads_nothing_further(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, count_reads: list[str]
) -> None:
    """What ``sd_mosaic`` does per image, counted across the whole run."""
    kept = enumerate_images(ds_with_an_index, results_root, has_no_offset_error=True)
    source = FilePointingSource(FCPath(results_root))
    for image in kept:
        source.load_pointing(image)

    assert count_reads.count(document_name(SUCCEEDED)) == 1


def test_an_image_carrying_no_record_is_still_read_from_storage(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, count_reads: list[str]
) -> None:
    """The other side of the count: nothing carried means the document is read.

    Without it, "one read" could be satisfied by a per-image stage that had
    stopped reading documents altogether.
    """
    kept = enumerate_images(ds_with_an_index, results_root, has_offset_file=True)
    source = FilePointingSource(FCPath(results_root))
    for image in kept:
        source.load_pointing(image)

    assert count_reads.count(document_name(SUCCEEDED)) == 1


# ---------------------------------------------------------------------------
# The index path, which carries nothing
# ---------------------------------------------------------------------------


def test_an_index_backed_run_carries_no_record(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, indexed: str
) -> None:
    """A row is narrowed on columns, and the record a consumer wants is another read."""
    kept = enumerate_images(
        ds_with_an_index, results_root, results_index_db_url=indexed, has_no_offset_error=True
    )

    assert [image.nav_record for image in kept] == [None]


def test_an_index_backed_run_selects_the_same_images(
    ds_with_an_index: DataSetPDS3CassiniISS, results_root: Path, indexed: str
) -> None:
    """Stated so that the assertion above is about an image the run kept."""
    kept = enumerate_images(
        ds_with_an_index, results_root, results_index_db_url=indexed, has_no_offset_error=True
    )

    assert [image.results_path_stub for image in kept] == [SUCCEEDED]


def test_an_index_backed_run_reads_no_document_at_all(
    ds_with_an_index: DataSetPDS3CassiniISS,
    results_root: Path,
    indexed: str,
    count_reads: list[str],
) -> None:
    """The filter reads rows, so the enumeration opens none of the documents."""
    enumerate_images(
        ds_with_an_index, results_root, results_index_db_url=indexed, has_no_offset_error=True
    )

    assert count_reads == []


# ---------------------------------------------------------------------------
# What the filter leaves on the candidates it did not keep
# ---------------------------------------------------------------------------


def candidates(results_root: Path) -> list[ImageFile]:
    """Build the three candidates an enumeration would offer the filter.

    Parameters:
        results_root: The results root, so the stand-in URLs point somewhere.

    Returns:
        One image per fixture document, in enumeration order.
    """
    return [
        ImageFile(
            image_file_url=FCPath(results_root / f'{stub}.IMG'),
            label_file_url=FCPath(results_root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in (SUCCEEDED, SPICE_ERROR, OTHER_ERROR)
    ]


def test_a_candidate_the_filter_dropped_is_left_carrying_nothing(results_root: Path) -> None:
    """The record goes onto the images kept and onto no others.

    An attach made before the keep decision would put a record on every
    candidate the batch was asked about, including the ones the run will never
    process.
    """
    offered = candidates(results_root)
    results_filter = ResultsFilter(
        [VOLUME],
        str(results_root),
        logger=null_logger(),
        has_no_offset_error=True,
        tuning=packaged_tree_tuning(),
    )
    results_filter.filter_batch(offered)

    assert [image.nav_record for image in offered[1:]] == [None, None]


def test_the_candidate_the_filter_kept_does_carry_one(results_root: Path) -> None:
    """Stated so the assertion above is about a batch that read anything at all."""
    offered = candidates(results_root)
    results_filter = ResultsFilter(
        [VOLUME],
        str(results_root),
        logger=null_logger(),
        has_no_offset_error=True,
        tuning=packaged_tree_tuning(),
    )
    results_filter.filter_batch(offered)

    assert offered[0].nav_record == documents()[SUCCEEDED]
