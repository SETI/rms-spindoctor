"""A record the enumeration already read, and what the two sources do with it.

A run that selected its images on what their documents record has read and
parsed every kept image's document, and the enumeration hands that record on
with the image.  The file-backed source answers from it; the index-backed source
does not, because a row is a different read of a different column set.

What has to hold, and is asserted here one condition at a time: a carried record
classifies exactly as the same document read from storage does -- same
mechanism, same matrices, same midtime, same offset and same reason -- and a
carried record does not reach the index path at all.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath
from tests.spindoctor.cli.reproj.conftest import (
    CMATRIX_STUB,
    OFFSET,
    POINTING,
    TIMES,
    build_tree,
    document,
    image_file,
    reason_tree,
)

from spindoctor.cli.reproj.offsets import PointingMechanism, PointingSelection
from spindoctor.cli.reproj.pointing_source import FilePointingSource, PointingSource
from spindoctor.dataset.dataset import ImageFile

CARRIED_OFFSET = [11.5, -13.25]
"""An offset no document of the fixture tree records.

What a carried record has to be answered from rather than looked past: a source
that read its storage instead would report the tree's offset, and the two are
told apart by their values rather than by their presence.
"""


def carrying(stub: str, record: dict[str, Any]) -> ImageFile:
    """Build an image arriving with a record already read for it.

    Parameters:
        stub: The results path stub.
        record: The record the enumeration read for it.

    Returns:
        The image file.
    """
    image = image_file(stub)
    image.nav_record = record
    return image


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Write the fixture results tree and return its root.

    Parameters:
        tmp_path: Directory the tree is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'nav'
    build_tree(root, reason_tree())
    return root


@pytest.fixture
def file_source(tree: Path) -> FilePointingSource:
    """Return a file-backed source over the fixture tree.

    Parameters:
        tree: The results root.

    Returns:
        The source.
    """
    return FilePointingSource(FCPath(tree))


def carried_document() -> dict[str, Any]:
    """Build a complete record recording an offset the tree does not.

    Returns:
        The record, in the shape the navigator writes it.
    """
    return document(CMATRIX_STUB, offset=CARRIED_OFFSET, times=TIMES, pointing=POINTING)


# ---------------------------------------------------------------------------
# The file-backed source answers from what it was given
# ---------------------------------------------------------------------------


def test_the_whole_record_comes_back_as_it_was_carried(
    file_source: FilePointingSource,
) -> None:
    """``read_record`` hands back the carried record itself."""
    record = carried_document()

    assert file_source.read_record(carrying(CMATRIX_STUB, record)) == record


def test_the_carried_record_is_answered_rather_than_the_document(
    file_source: FilePointingSource,
) -> None:
    """The tree holds a different offset for this stub, and it is not read."""
    carried = file_source.read_record(carrying(CMATRIX_STUB, carried_document()))

    assert carried['offset'] == CARRIED_OFFSET


def test_the_document_is_still_read_for_an_image_carrying_nothing(
    file_source: FilePointingSource,
) -> None:
    """The other half of the pair: nothing carried, so storage answers."""
    read = file_source.read_record(image_file(CMATRIX_STUB))

    assert read['offset'] == OFFSET


def test_a_carried_record_supplies_the_pointing_it_records(
    file_source: FilePointingSource,
) -> None:
    """``load_pointing`` classifies the carried record rather than the document."""
    selection = file_source.load_pointing(carrying(CMATRIX_STUB, carried_document()))

    assert selection.offset == (CARRIED_OFFSET[0], CARRIED_OFFSET[1])


def test_a_record_carried_for_a_deleted_document_is_still_answered(
    file_source: FilePointingSource, tree: Path
) -> None:
    """The declared behavior change: the run reads what it read at selection time.

    A document that leaves the tree between the selection and the per-image
    stage is no longer noticed, which narrows a window every run has rather than
    opening a new one.
    """
    (tree / f'{CMATRIX_STUB}_metadata.json').unlink()

    selection = file_source.load_pointing(carrying(CMATRIX_STUB, carried_document()))

    assert selection.mechanism is PointingMechanism.CMATRIX


def test_that_deleted_document_has_no_record_for_an_image_carrying_none(
    file_source: FilePointingSource, tree: Path
) -> None:
    """Stated so the assertion above is about the carry and not about the tree."""
    (tree / f'{CMATRIX_STUB}_metadata.json').unlink()

    selection = file_source.load_pointing(image_file(CMATRIX_STUB))

    assert selection.reason == 'no_metadata'


# ---------------------------------------------------------------------------
# A carried record classifies exactly as the same document read from storage
# ---------------------------------------------------------------------------


def _both_ways(file_source: FilePointingSource) -> tuple[PointingSelection, PointingSelection]:
    """Classify one image's record twice: as carried, and as read from the tree.

    Parameters:
        file_source: The source over the fixture tree.

    Returns:
        The selection from the carried record, and the one from the document.
    """
    tree_document = document(CMATRIX_STUB, offset=OFFSET, times=TIMES, pointing=POINTING)
    return (
        file_source.load_pointing(carrying(CMATRIX_STUB, tree_document)),
        file_source.load_pointing(image_file(CMATRIX_STUB)),
    )


def test_the_comparison_is_over_a_record_that_supplies_a_cmatrix(
    file_source: FilePointingSource,
) -> None:
    """Without it, every comparison below is satisfied by two empty selections.

    A pair of ``NONE`` selections agrees about the mechanism, and their matrices
    and midtimes are all None, which compares equal.  So the shape of what is
    being compared is stated first.
    """
    carried, _read = _both_ways(file_source)

    assert carried.mechanism is PointingMechanism.CMATRIX


def test_a_carried_record_selects_the_same_mechanism(file_source: FilePointingSource) -> None:
    """The ladder ends in the same arm whichever way the record arrived."""
    carried, read = _both_ways(file_source)

    assert carried.mechanism is read.mechanism


def test_a_carried_record_carries_the_same_cmatrix(file_source: FilePointingSource) -> None:
    """Bit for bit, since a corrected attitude is what a product is built on."""
    carried, read = _both_ways(file_source)
    assert carried.cmatrix is not None
    assert read.cmatrix is not None

    assert np.array_equal(carried.cmatrix, read.cmatrix)


def test_a_carried_record_carries_the_same_baseline(file_source: FilePointingSource) -> None:
    """Bit for bit, since the gates compare the baseline to the furnished pool."""
    carried, read = _both_ways(file_source)
    assert carried.cmatrix_original is not None
    assert read.cmatrix_original is not None

    assert np.array_equal(carried.cmatrix_original, read.cmatrix_original)


def test_a_carried_record_carries_the_same_midtime(file_source: FilePointingSource) -> None:
    """Exactly, because the C-matrix gates compare it to a microsecond."""
    carried, read = _both_ways(file_source)

    assert carried.midtime_et == read.midtime_et


def test_a_carried_record_carries_the_same_offset(file_source: FilePointingSource) -> None:
    """The fallback a gate refusal degrades to is the same pair either way."""
    carried, read = _both_ways(file_source)

    assert carried.offset == read.offset


def test_a_carried_record_carries_the_same_reason(file_source: FilePointingSource) -> None:
    """What a run-level tally counts this image under does not change."""
    carried, read = _both_ways(file_source)

    assert carried.reason == read.reason


# ---------------------------------------------------------------------------
# The index path is not the one that carries anything
# ---------------------------------------------------------------------------


def test_the_index_source_reads_its_row_rather_than_a_carried_record(
    sources: dict[str, PointingSource],
) -> None:
    """A record on the image does not reach the index path, which reads its row."""
    record = sources['index'].read_record(carrying(CMATRIX_STUB, carried_document()))

    assert record['offset'] == OFFSET


def test_the_index_source_classifies_its_row_rather_than_a_carried_record(
    sources: dict[str, PointingSource],
) -> None:
    """And the classified selection is the row's, not the carried record's."""
    selection = sources['index'].load_pointing(carrying(CMATRIX_STUB, carried_document()))

    assert selection.offset == (OFFSET[0], OFFSET[1])
