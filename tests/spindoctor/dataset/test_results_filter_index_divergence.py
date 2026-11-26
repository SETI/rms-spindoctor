"""Every answer the index gives differently from the tree, one test each.

Two of them: a file the ingest pass could not retrieve, and a document rewritten
in place keeping the length and the modification time it had before.  Each is a
property of what one pass could read and record rather than of a query, and each
is enumerated in :mod:`spindoctor.dataset.results_filter`, in the navigation
guide, in the plan, and here.  They are tested in both directions -- what the
tree answers and what the index answers -- because what makes each a divergence
is that the two differ, and an assertion about one of them alone would pass if
the other silently changed to match.

A member added to the enumeration is added here.  One that stops being a
divergence keeps a test and loses this file: what made it worth a test is that
the two answers could differ, which is as true of the answer that matches as of
the one that did not, so the test moves to the parity file and asserts the
agreement there.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.dataset.conftest import (
    SPICE_ERROR,
    VOLUMES,
    null_logger,
    one_image_tree,
    select_from,
)

from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import SPICE_STATUS_ERROR, ResultsFilter


def test_a_file_the_pass_could_not_retrieve_reads_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is recorded for it, deliberately, so the next pass tries it again.

    A recorded refusal would be skipped for as long as the file did not change,
    and a download that failed once says nothing that will still be true then.
    The cost is that the file reads as absent until a pass reads it, which is one
    of the answers the index gives differently.
    """
    root, images = one_image_tree(tmp_path)

    def refuse(self: FCPath, *args: Any, **kwargs: Any) -> list[Exception]:
        return [OSError('the backend did not answer') for _ in args[0]]

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(FCPath, 'retrieve', refuse)
    ingest_tree(url, [root], logger=null_logger())
    monkeypatch.undo()
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_index_db_url=url,
        has_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert select_from(results_filter, images) == []


OTHER_STATUS_ERROR = 'some_other_problem'
"""A fatal error of the same text length as the SPICE one.

A document rewritten from one to the other is a different document of exactly
the same size, which is half of what makes the pass skip it.
"""


def _fatal_error_document(status_error: str) -> dict[str, Any]:
    """Build the fatal-error document the rewritten stub carries.

    Parameters:
        status_error: The fatal error the document records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1000000004_1.IMG', status='error', status_error=status_error, offset=None
    )


def _index_after_a_document_was_rewritten_in_place(
    tmp_path: Path, *, keeping_its_size: bool = True
) -> tuple[Path, list[ImageFile], str]:
    """Ingest a root, rewrite one document in place, and ingest it again.

    The rewrite keeps the document's modification time, and by default its
    length: a tree restored with ``cp -p`` or ``rsync --times``, a document
    patched in place and stamped back, or a backend whose listing reports the
    same time for two writes.  Those are the two metrics the pass compares, so
    the second pass has nothing to tell it that the file it already read is not
    the file that is there now.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        keeping_its_size: Whether the rewritten document is the same length as
            the one it replaces.  False makes the rewrite visible again, which
            is what says the divergence is about the metrics and not about
            rewriting.

    Returns:
        The root, the one candidate image, and the connection URL of the index.
    """
    root = tmp_path / 'results'
    document = write_metadata(root, SPICE_ERROR, _fatal_error_document(SPICE_STATUS_ERROR))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    before = document.stat()
    replacement = OTHER_STATUS_ERROR if keeping_its_size else f'{OTHER_STATUS_ERROR}_and_then_some'
    write_metadata(root, SPICE_ERROR, _fatal_error_document(replacement))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (document.stat().st_size == before.st_size) is keeping_its_size
    ingest_tree(url, [root], logger=null_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / f'{SPICE_ERROR}.IMG'),
            label_file_url=FCPath(root / f'{SPICE_ERROR}.LBL'),
            results_path_stub=SPICE_ERROR,
        )
    ]
    return root, images, url


def test_a_document_rewritten_in_place_is_not_read_again(tmp_path: Path) -> None:
    """The mechanism, stated on its own: the pass skips a file it has already read.

    Nothing about a listing distinguishes a document from another of the same
    length written at the same recorded time, and reading it to find out is the
    download the skip exists to avoid.
    """
    root = tmp_path / 'results'
    document = write_metadata(root, SPICE_ERROR, _fatal_error_document(SPICE_STATUS_ERROR))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    before = document.stat()
    write_metadata(root, SPICE_ERROR, _fatal_error_document(OTHER_STATUS_ERROR))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert ingest_tree(url, [root], logger=null_logger()).files_skipped == 1


def test_a_document_rewritten_in_place_reads_as_it_is_now_in_the_tree(tmp_path: Path) -> None:
    """The walk opens the file, so it answers from the document that is there."""
    root, images, _url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    )
    assert select_from(results_filter, images) == []


def test_a_document_rewritten_in_place_reads_as_its_previous_self_in_the_index(
    tmp_path: Path,
) -> None:
    """The row still records what the document said when it was last read.

    No number of completed passes corrects this one, which is what makes it a
    member of the enumeration rather than the snapshot's age: a pass that
    finished a second ago answers from the document before the rewrite.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_index_db_url=url,
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


def test_a_rewrite_that_changes_the_length_is_read_again(tmp_path: Path) -> None:
    """The divergence is the equal metrics, not the rewrite.

    A document rewritten to a different length is one the pass has evidence
    about, and the row it leaves is what the tree would answer.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(
        tmp_path, keeping_its_size=False
    )
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_index_db_url=url,
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    )
    assert select_from(results_filter, images) == []


def test_a_forced_pass_corrects_a_document_rewritten_in_place(tmp_path: Path) -> None:
    """Reading every document regardless is what an operator has to reach for.

    It is the remedy because the alternative is reading the file to find out
    whether it needs reading, which is the cost the skip exists to avoid.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    ingest_tree(url, [root], logger=null_logger(), force=True)
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_index_db_url=url,
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    )
    assert select_from(results_filter, images) == []
