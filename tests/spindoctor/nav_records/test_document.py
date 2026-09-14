"""What a stub may be, where its document lives, and what makes a file a document.

Every reader of a results tree turns a stub into a path, reads a document and
decides what an unreadable one means through this module, whichever program it
belongs to and whether or not that program can read a results index.  So the
rules are tested once, here, rather than once per reader.

A stub is a key rather than a path, and the rule about it is applied where a key
is written rather than where one is read: a results root is normalized to an
absolute, resolved location where it is spelled, so joining a key onto one has a
single answer and nothing left to check.  What a mission-filtered or a
time-bounded read does with a document it cannot place is tested beside the read
itself, in ``test_tree_records.py``.
"""

import json
from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.nav_records import (
    ABSOLUTE_PATH_FRAGMENT,
    METADATA_SUFFIX,
    NOT_A_SINGLE_COMPONENT,
    NULL_BYTE_IN_PATH,
    PARENT_SEGMENT_IN_PATH,
    document_path,
    read_document,
    stub_for_document,
    stub_refusal,
    subtree_refusal,
)

# ---------------------------------------------------------------------------
# What makes a stub a key
# ---------------------------------------------------------------------------


def test_an_ordinary_stub_is_a_key() -> None:
    """The case the refusals below are the exceptions to."""
    assert stub_refusal('COISS_2001/N1454725799') is None


def test_a_stub_carrying_a_null_byte_is_refused() -> None:
    """It would join perfectly well and then fail at the filesystem."""
    assert stub_refusal('COISS_2001/N145\x004725799') == NULL_BYTE_IN_PATH


def test_an_absolute_stub_is_refused() -> None:
    """Joining an absolute fragment discards the root, so it names a file under none."""
    assert stub_refusal('/etc/passwd') == ABSOLUTE_PATH_FRAGMENT


def test_a_stub_spelled_as_a_cloud_url_is_refused() -> None:
    """A URL discards the root exactly as a leading separator does.

    No local path test calls it absolute, so a rule written against the
    filesystem alone would let a stub naming another bucket through.
    """
    assert stub_refusal('gs://somebody-elses-bucket/N1454725799') == ABSOLUTE_PATH_FRAGMENT


def test_a_stub_naming_a_parent_directory_is_refused() -> None:
    """A key that walks upwards names a file the root it was asked under does not hold."""
    assert stub_refusal('../../elsewhere/N1454725799') == PARENT_SEGMENT_IN_PATH


def test_a_parent_segment_in_the_middle_of_a_stub_is_refused() -> None:
    """It resolves back inside the root and is still a second name for one document."""
    assert stub_refusal('VOL1/../BARESCENE') == PARENT_SEGMENT_IN_PATH


def test_a_stub_whose_name_merely_starts_with_two_dots_is_a_key() -> None:
    """The rule is about a path component, not about a pair of characters."""
    assert stub_refusal('VOL1/..hidden_scene') is None


# ---------------------------------------------------------------------------
# What makes a subtree one directory under a root
# ---------------------------------------------------------------------------


def test_an_ordinary_subtree_is_one_directory() -> None:
    """The first component of every stub beneath it, which is what an index stores."""
    assert subtree_refusal('COISS_2001') is None


@pytest.mark.parametrize(
    'subtree',
    [
        pytest.param('', id='empty'),
        pytest.param('.', id='this-directory'),
        pytest.param('..', id='the-parent'),
        pytest.param('COISS_2001/data', id='two-components'),
        pytest.param('COISS_2001/', id='trailing-separator'),
        pytest.param('/COISS_2001', id='absolute'),
        pytest.param('gs://bucket/COISS_2001', id='a-url'),
    ],
)
def test_a_subtree_that_is_not_one_component_is_refused(subtree: str) -> None:
    """A walk joins it and a query compares it, and only one component means one thing.

    Enumerating only some of these lets the others through, and each of them
    makes the two storages answer one selection two ways: a fragment the walk
    descends and the query finds no row for, or a spelling that builds stubs no
    consumer's lookup matches.

    Parameters:
        subtree: The spelling to refuse.
    """
    assert subtree_refusal(subtree) == NOT_A_SINGLE_COMPONENT


def test_a_subtree_carrying_a_null_byte_is_refused() -> None:
    """It is charged to the word that caused it rather than to a directory listing."""
    assert subtree_refusal('COISS\x002001') == NULL_BYTE_IN_PATH


# ---------------------------------------------------------------------------
# Where a document lives
# ---------------------------------------------------------------------------


def test_the_join_is_the_stub_under_the_root_with_the_suffix_back_on(
    tmp_path: Path,
) -> None:
    """The one join in the seam: where the document of this stub lives."""
    joined = document_path(tmp_path, 'COISS_2001/N1454725799')
    assert joined.as_posix() == (tmp_path / f'COISS_2001/N1454725799{METADATA_SUFFIX}').as_posix()


def test_a_documents_stub_is_its_path_under_the_root(tmp_path: Path) -> None:
    """The stub and the path are two spellings of one identity, and must agree.

    This is the inverse of the join above: the walk turns a path into a stub, an
    index row records that stub, and a reader turns it back into a path.  A
    disagreement between the two would make an ingested image unfindable by the
    key it was ingested under.
    """
    root = FCPath(str(tmp_path))
    doc = root / 'COISS_2001' / f'N1454725799{METADATA_SUFFIX}'
    stub = stub_for_document(root, doc)
    assert stub == 'COISS_2001/N1454725799'


def test_a_stub_round_trips_through_the_path_and_back(tmp_path: Path) -> None:
    """Joining a stub and taking the stub of the result returns the stub."""
    root = FCPath(str(tmp_path))
    assert stub_for_document(root, document_path(root, 'COISS_2001/N1454725799')) == (
        'COISS_2001/N1454725799'
    )


def test_a_root_that_merely_spells_the_start_of_a_neighbor_takes_nothing_off(
    tmp_path: Path,
) -> None:
    """``/data/res`` is a character prefix of ``/data/results`` and a parent of none of it.

    A root taken off character by character turns the neighbor's document into
    something that looks like a key, reads like a key and names a file this root
    does not hold.  Nothing downstream can tell such a stub from a real one, so
    it becomes a wrong index key and stays one.
    """
    root = FCPath(str(tmp_path / 'res'))
    doc = FCPath(str(tmp_path / 'results' / 'VOL1' / f'N1454725799{METADATA_SUFFIX}'))
    assert (
        stub_for_document(root, doc) == (tmp_path / 'results' / 'VOL1' / 'N1454725799').as_posix()
    )


def test_a_root_spelled_with_a_trailing_separator_names_the_same_stub(tmp_path: Path) -> None:
    """The two spellings are one root, so they cannot give a document two keys."""
    doc = FCPath(str(tmp_path / 'VOL1' / f'N1454725799{METADATA_SUFFIX}'))
    assert stub_for_document(FCPath(f'{tmp_path.as_posix()}/'), doc) == 'VOL1/N1454725799'


# ---------------------------------------------------------------------------
# What makes a file a document
# ---------------------------------------------------------------------------


def test_a_document_is_read_as_the_object_it_holds(tmp_path: Path) -> None:
    """The ordinary case, which the refusals below are the exceptions to."""
    path = tmp_path / f'N1454725799{METADATA_SUFFIX}'
    path.write_text(json.dumps({'status': 'success'}))
    assert read_document(FCPath(str(path))) == {'status': 'success'}


def test_a_file_holding_json_that_is_not_an_object_is_not_a_document(tmp_path: Path) -> None:
    """Reading a field off an array would fail later and further away."""
    path = tmp_path / f'listy{METADATA_SUFFIX}'
    path.write_text('[1, 2]')
    with pytest.raises(ValueError, match='not a JSON object'):
        read_document(FCPath(str(path)))


def test_a_file_that_is_not_there_raises_the_error_a_caller_distinguishes(
    tmp_path: Path,
) -> None:
    """An unnavigated image and an unreadable document are reported differently."""
    absent = tmp_path / f'absent{METADATA_SUFFIX}'
    with pytest.raises(FileNotFoundError, match=r'absent_metadata\.json'):
        read_document(FCPath(str(absent)))
