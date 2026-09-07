"""The one fixture tree both pointing sources are measured against.

The point of these tests is equivalence, so the documents are written once and
both sources are pointed at the same ones: the file-backed source reads the
tree, and the index-backed source reads an index built from that same tree by
the real ingest.  A factory that built two trees, or an index written by hand,
could agree with itself while the two paths a program takes disagreed.

The recorded attitudes are real: they are what the pointing computation produced
for a Cassini NAC frame at its own navigated offset.  A hand-written rotation of
zeros and ones round-trips through JSON whatever the storage does with it, and
the reader's flip gate holds the recovered rotation to 1e-9, so only a value
carrying a full float64 mantissa can show that the storage kept one.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from sqlalchemy.engine import Engine
from tests.spindoctor.conftest import (
    metadata_document,
    write_metadata,
)
from tests.spindoctor.results_index.conftest import (
    postgres_decoy_schema,
    postgres_schema,
    postgres_server_url,
    postgres_url,
)

from spindoctor.cli.reproj.pointing_source import (
    FilePointingSource,
    IndexPointingSource,
    PointingSource,
)
from spindoctor.cli.results_index import ingest_metadata_files
from spindoctor.dataset.dataset import ImageFile
from spindoctor.nav_records import TreeTuning
from spindoctor.results_index import normalize_root_url, open_index

# The pointing postgres tier runs against a schema of its own, exactly as the
# results-index tier does; re-exporting rather than restating keeps one
# definition of how that schema is created and dropped.
__all__ = [
    'postgres_decoy_schema',
    'postgres_schema',
    'postgres_server_url',
    'postgres_url',
]

CMATRIX_ORIGINAL = [
    0.963676611721357,
    0.24393954782302113,
    0.1087238935521779,
    -0.23632299640079324,
    0.5892234094236003,
    0.772636534962837,
    0.1244139437257574,
    -0.7702657144097279,
    0.6254693436224319,
]
"""The as-flown attitude recorded for a real Cassini NAC frame, row-major."""

CMATRIX = [
    0.9636758075215185,
    0.24394452671199518,
    0.10871985046444949,
    -0.23632717093513517,
    0.5892492544213016,
    0.7726155476313796,
    0.1244122432702933,
    -0.7702443664521031,
    0.6254959709488565,
]
"""The corrected attitude recorded for that frame at its navigated offset."""

MIDTIME_ET = 136576860.1724845
"""That frame's exposure midtime, which the reader's gate holds to 1e-6 s."""

TIMES: dict[str, Any] = {
    'start_et': 136576860.0424845,
    'stop_et': 136576860.30248448,
    'midtime_et': MIDTIME_ET,
    'exposure_s': 0.26,
    'sclk_start': '1/1461997416.044',
    'sclk_midtime': '1/1461997416.078',
    'sclk_stop': '1/1461997416.111',
}
"""The exposure epochs recorded beside those attitudes."""

POINTING: dict[str, Any] = {
    'cmatrix': CMATRIX,
    'cmatrix_original': CMATRIX_ORIGINAL,
    'camera_frame': 'CASSINI_ISS_NAC',
    'camera_frame_id': -82360,
    'ck_frame_id': -82000,
}
"""A complete recorded pointing block, in the shape its producer writes."""

FITTED_ROTATION_POINTING: dict[str, Any] = {
    'cmatrix_original': CMATRIX_ORIGINAL,
    'camera_frame': 'CASSINI_ISS_NAC',
    'camera_frame_id': -82360,
    'ck_frame_id': -82000,
}
"""What a result that fitted a camera rotation records: a baseline, no cmatrix."""


OFFSET = [5.6005, 1.0788]
"""The navigated offset those attitudes were computed from."""

VOLUME = 'COISS_2001/data/1461994336_1462054659'
"""Volume and range directory every stub of the fixture tree sits under."""

CMATRIX_STUB = f'{VOLUME}/N100_1_CALIB'
FITTED_STUB = f'{VOLUME}/N101_1_CALIB'
NO_POINTING_STUB = f'{VOLUME}/N102_1_CALIB'
FAILED_STUB = f'{VOLUME}/N103_1_CALIB'
NO_STATUS_ERROR_STUB = f'{VOLUME}/N115_1_CALIB'
REFUSED_DOCUMENT_STUB = f'{VOLUME}/N133_1_CALIB'
ZERO_OFFSET_STUB = f'{VOLUME}/N134_1_CALIB'
ZERO_EPOCH_STUB = f'{VOLUME}/N135_1_CALIB'
UNNAVIGATED_STUB = f'{VOLUME}/N999_1_CALIB'
"""Stubs of the fixture tree; the last is deliberately never written."""

ZERO_OFFSET = [0.0, 0.0]
"""A navigated offset of exactly no pixels.

A recorded value that is present and false at once, which is what separates
asking whether a column holds a value from asking whether the value is true.
The pair is applied like any other, so a rebuild that read it as no pair at all
would build an offset-corrected product through the document and an uncorrected
one through the row.
"""

ZERO_TIMES: dict[str, Any] = {
    'start_et': 0.0,
    'stop_et': 0.0,
    'midtime_et': 0.0,
    'exposure_s': 0.0,
    'sclk_start': '1/0000000000.000',
    'sclk_midtime': '1/0000000000.000',
    'sclk_stop': '1/0000000000.000',
}
"""Exposure epochs at the J2000 epoch itself, every number among them a zero.

``midtime_et`` is the one the pointing ladder cannot run without, so a rebuild
that dropped a present zero would report a clean corrected attitude as a
malformed pointing block.
"""

ZERO_FRAME_ID_POINTING: dict[str, Any] = {**POINTING, 'camera_frame_id': 0, 'ck_frame_id': 0}
"""A complete pointing block whose two frame identities are recorded zeros."""

REFUSED_DOCUMENT_REASON = 'no observation.instrument'
"""Why the ingest refuses the one document in the tree it cannot read.

Named here because the refusal a consumer reports has to carry it: an operator
told only that the index cannot answer for an image has nothing to fix.
"""


def image_file(stub: str) -> ImageFile:
    """Build an image file naming the stub its record is looked up under.

    Parameters:
        stub: The results path stub.

    Returns:
        The image file.
    """
    name = stub.rsplit('/', 1)[-1]
    return ImageFile(
        image_file_url=FCPath(f'/holdings/{name}.IMG'),
        label_file_url=FCPath(f'/holdings/{name}.LBL'),
        results_path_stub=stub,
        index_file_row={},
    )


def document(stub: str, **overrides: Any) -> dict[str, Any]:
    """Build one navigation document, named for the image it records.

    Parameters:
        stub: The results path stub, whose basename names the image.
        overrides: Fields passed through to the document factory.

    Returns:
        The document.
    """
    name = stub.rsplit('/', 1)[-1]
    return metadata_document(image_name=f'{name}.IMG', **overrides)


def reason_tree() -> dict[str, dict[str, Any]]:
    """Build one document per classification the two sources must agree about.

    A fresh tree of fresh documents per call, rather than one shared mapping:
    every document here is mutable, and a test that edited a nested one would
    otherwise change what every later test in the run was given.

    Returns:
        Results path stub mapped to the document recorded there.
    """
    tree: dict[str, dict[str, Any]] = {
        CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET, times=TIMES, pointing=POINTING),
        FITTED_STUB: document(
            FITTED_STUB, offset=OFFSET, times=TIMES, pointing=FITTED_ROTATION_POINTING
        ),
        NO_POINTING_STUB: document(NO_POINTING_STUB, offset=OFFSET),
        FAILED_STUB: document(
            FAILED_STUB, status='error', status_error='missing_spice_data', offset=None
        ),
        # An unsuccessful outcome naming no error, which is every failed and
        # conflicted navigation: the field is written only by a document
        # recording an image that would not load.
        NO_STATUS_ERROR_STUB: document(NO_STATUS_ERROR_STUB, status='failed', offset=None),
        # A document the ingest refuses whole, carrying a usable corrected
        # attitude all the same.  Read as a file it supplies that attitude;
        # the index holds no record of it and must say so rather than report
        # the image as one nothing navigated.
        REFUSED_DOCUMENT_STUB: document(
            REFUSED_DOCUMENT_STUB,
            instrument=None,
            offset=OFFSET,
            times=TIMES,
            pointing=POINTING,
        ),
        # An offset of two zeros: a navigation that found the pointing already
        # right records one, and it is a pair like any other.  Every column of
        # the row it becomes holds a value that is false when it is read for
        # its truth rather than for its presence.
        ZERO_OFFSET_STUB: document(ZERO_OFFSET_STUB, offset=ZERO_OFFSET),
        # And the same on the other half of the rebuild: exposure epochs at the
        # J2000 epoch and frame identities of zero, beside a corrected attitude
        # the ladder does apply.  A rebuild reading presence as truth loses the
        # midtime and reports this record as a malformed pointing block.
        ZERO_EPOCH_STUB: document(
            ZERO_EPOCH_STUB, offset=OFFSET, times=ZERO_TIMES, pointing=ZERO_FRAME_ID_POINTING
        ),
    }
    return tree


@pytest.fixture
def quiet_ingest_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps ingest chatter out of the test output.

    Returns:
        A logger of its own, named uniquely for the life of the process, so
        raising its level cannot affect another test.
    """
    logger = pdslogger.PdsLogger(f'pointing_source_test_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    return logger


def build_tree(root: Path, documents: dict[str, dict[str, Any]]) -> None:
    """Write a results tree of navigation documents.

    Parameters:
        root: The results root to write under.
        documents: Results path stub mapped to the document recorded there.
    """
    for stub, content in documents.items():
        write_metadata(root, stub, content)


def index_for(roots: list[Path], database: Path, *, logger: pdslogger.PdsLogger) -> Engine:
    """Ingest one or more results trees and return the open index.

    Parameters:
        roots: The results roots to walk.
        database: Path of the SQLite file to create.
        logger: Logger the ingest reports through.

    Returns:
        The open index, which the caller disposes of.
    """
    engine = open_index(f'sqlite:///{database.as_posix()}', create=True)
    ingest_metadata_files(
        engine, [root.as_posix() for root in roots], logger=logger, tuning=TreeTuning()
    )
    return engine


@pytest.fixture
def sources(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> Iterator[dict[str, PointingSource]]:
    """Yield both sources over one results tree covering every reachable reason.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
        quiet_ingest_logger: Logger the ingest reports through.

    Yields:
        The file-backed source under ``'file'`` and the index-backed one under
        ``'index'``, both answering for the same root.
    """
    root = tmp_path / 'nav'
    build_tree(root, reason_tree())
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        yield {
            'file': FilePointingSource(FCPath(root)),
            'index': IndexPointingSource(engine, normalize_root_url(root)),
        }
    finally:
        engine.dispose()
