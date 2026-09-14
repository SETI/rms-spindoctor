"""The per-image facts a source over the documents answers with.

The fourth question on the seam, over the storage that has no database.  What is
tested here is what the tree alone owes: that a document is read once and the
walk's own metrics ride along with what came out of it, that a file which is no
navigation document is reported rather than raised on, and that a selection is
honored the way the stream of records honors it.  That the answer matches the
index's is tested where both storages can be driven over one tree, in
:mod:`tests.spindoctor.results_index.test_facts_stream_agreement`.
"""

import json
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from tests.spindoctor.nav_records.conftest import (
    FIRST_STUB,
    MISSION,
    OTHER_MISSION,
    SECOND_STUB,
    count_reads,
    count_retrievals,
    document,
    failing_retrievals,
    tree_source,
    two_volume_tree,
    write_document,
    write_text,
)

from spindoctor.nav_records import (
    COULD_NOT_RETRIEVE,
    NOT_A_NAVIGATION_DOCUMENT,
    NOT_VALID_JSON,
    ImageFacts,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)
from spindoctor.results_index import IMAGES

REFUSED_STUB = 'VOL1/not_a_navigation_document'
"""Where a file that reads as JSON and is no navigation result sits."""

REFUSED_DOCUMENT = '{"edges": []}'
"""A JSON object of some other tool's shape."""


def _stub_of(found: ImageFacts | UnreadableFile) -> str:
    """Return the stub of one thing a facts stream yielded.

    Parameters:
        found: The facts, or the file no facts came out of.

    Returns:
        The image's results path stub, which both shapes carry.
    """
    if isinstance(found, UnreadableFile):
        return found.stub
    return str(found.image['results_path_stub'])


def _facts_of(found: list[ImageFacts | UnreadableFile], stub: str) -> ImageFacts:
    """Return the facts of one image out of what a stream yielded.

    Parameters:
        found: What the stream yielded.
        stub: The image to pick out.

    Returns:
        Its facts.
    """
    picked = [one for one in found if isinstance(one, ImageFacts) and _stub_of(one) == stub]
    assert len(picked) == 1
    return picked[0]


def _reasons(found: list[ImageFacts | UnreadableFile]) -> dict[str, str]:
    """Return why each file that yielded no facts yielded none.

    Parameters:
        found: What the stream yielded.

    Returns:
        The reason of each unreadable file, by stub.
    """
    return {one.stub: one.reason for one in found if isinstance(one, UnreadableFile)}


def test_the_facts_of_a_document_name_its_image(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The shape a consumer reads, built out of the document the walk found.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _facts_of(found, FIRST_STUB).image['image_name'] == 'N1454725799_1.IMG'


def test_the_facts_record_the_root_the_document_is_under(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Half the key, and the half a query answering about a stub alone drops.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _facts_of(found, FIRST_STUB).image['root_url'] == root.resolve().as_posix()


def test_the_facts_name_the_document_they_were_read_from(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """So a message about an image names a file an operator can open.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    expected = (root / f'{FIRST_STUB}_metadata.json').resolve().as_posix()
    assert _facts_of(found, FIRST_STUB).image['source_file'] == expected


def test_the_facts_carry_the_modification_time_the_walk_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Threaded through from the listing rather than asked of the file again.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    listed = {entry.stub: entry for entry in source.listing(Selection())}
    found = list(source.facts(Selection()))
    assert _facts_of(found, FIRST_STUB).image['mtime_ns'] == listed[FIRST_STUB].mtime_ns


def test_the_facts_carry_the_size_the_walk_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The other half of what decides whether a document has changed.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    listed = {entry.stub: entry for entry in source.listing(Selection())}
    found = list(source.facts(Selection()))
    assert _facts_of(found, FIRST_STUB).image['size_bytes'] == listed[FIRST_STUB].size_bytes


def test_the_metrics_are_the_files_own_and_not_a_placeholder(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Without which the two tests above would pass over two absent values.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _facts_of(found, FIRST_STUB).image['size_bytes'] is not None


def test_the_facts_of_a_document_carry_every_column_the_index_holds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The shape is the row shape, so the tree owes the whole of it and not a part.

    A consumer cannot see which storage answered, so a key the documents never
    put in is a field that reads as absent over one storage and as a value over
    the other.  Compared against the table rather than against a list restated
    here: a column added to the index and not to this reader is exactly the
    drift being guarded against, and a restated list would be updated by the
    same change that caused it.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert set(_facts_of(found, FIRST_STUB).image) == {column.name for column in IMAGES.columns}


def test_the_facts_carry_the_record_they_were_read_out_of(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The document is already read and already parsed, so it travels with the facts.

    A consumer that narrows on what a document says and then wants the document
    itself reads it once.  This is the one place the field is filled, so what is
    handed back is the record this source built.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    carried = _facts_of(found, FIRST_STUB).record
    assert carried is not None
    assert carried.metadata == document()


def test_the_record_the_facts_carry_names_the_document_it_came_from(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """So a consumer handed the record can still say which file it is."""
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    carried = _facts_of(found, FIRST_STUB).record
    assert carried is not None
    assert carried.stub == FIRST_STUB


def test_each_image_carries_its_own_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One record per image rather than one for the whole stream."""
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    carried = _facts_of(found, SECOND_STUB).record
    assert carried is not None
    assert carried.stub == SECOND_STUB


def test_each_document_is_read_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record and the facts come off one read of one file."""
    root = two_volume_tree(tmp_path)
    read = count_reads(monkeypatch)
    list(tree_source(root, quiet_logger).facts(Selection()))
    assert len(read) == 2


def test_each_document_is_retrieved_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One batched retrieval of both documents, not one pass per question.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
        monkeypatch: Fixture the retrieval is wrapped through.
    """
    root = two_volume_tree(tmp_path)
    retrieved = count_retrievals(monkeypatch)
    list(tree_source(root, quiet_logger).facts(Selection()))
    assert retrieved == [2]


def test_a_file_that_is_no_navigation_document_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A results tree holds them, and a pass that passed over one covers less.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    write_text(root, REFUSED_STUB, REFUSED_DOCUMENT)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _reasons(found)[REFUSED_STUB].startswith(NOT_A_NAVIGATION_DOCUMENT)


def test_a_file_that_is_not_json_at_all_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The reasons a document reader states are the reasons the facts state.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    write_text(root, 'VOL1/torn', '{"observation":')
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _reasons(found)['VOL1/torn'] == NOT_VALID_JSON


def test_a_file_that_never_arrived_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrieval that delivered nothing costs its own file and no other.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
        monkeypatch: Fixture the retrieval is wrapped through.
    """
    root = two_volume_tree(tmp_path)
    failing_retrievals(monkeypatch)
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _reasons(found) == {FIRST_STUB: COULD_NOT_RETRIEVE, SECOND_STUB: COULD_NOT_RETRIEVE}


def test_the_child_rows_of_a_document_come_out_of_the_same_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The technique entries a record does not carry and the facts do.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(
        root,
        FIRST_STUB,
        document(
            navigation_result={
                'times': {'midtime_et': 100.0},
                'per_technique': [
                    {'technique_name': 'StarFieldFromCatalogNav', 'offset_px': [1.0, 2.0]},
                    {'technique_name': 'BodyLimbNav', 'offset_px': [1.5, 2.5]},
                ],
            }
        ),
    )
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert [row['technique_name'] for row in _facts_of(found, FIRST_STUB).techniques] == [
        'StarFieldFromCatalogNav',
        'BodyLimbNav',
    ]


def test_the_feature_inventory_comes_out_of_the_same_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Aggregated per feature type and source, as the index stores it.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(
        root,
        FIRST_STUB,
        document(
            navigation_result={
                'times': {'midtime_et': 100.0},
                'feature_inventory': [
                    {
                        'feature_id': 'star:UCAC4:1',
                        'feature_type': 'STAR',
                        'source_model': 'stars',
                        'gated': True,
                    },
                    {
                        'feature_id': 'star:UCAC4:2',
                        'feature_type': 'STAR',
                        'source_model': 'stars',
                        'gated': False,
                    },
                ],
            }
        ),
    )
    found = list(tree_source(root, quiet_logger).facts(Selection()))
    assert _facts_of(found, FIRST_STUB).feature_sources == [
        {
            'root_url': root.resolve().as_posix(),
            'results_path_stub': FIRST_STUB,
            'feature_type': 'STAR',
            'source_model': 'stars',
            'source_name': 'UCAC4',
            'n_features': 2,
            'n_gated': 1,
        }
    ]


def _mission_tree(tmp_path: Path) -> Path:
    """Write a tree holding one document of each of two missions.

    Parameters:
        tmp_path: Directory the tree is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    write_document(
        root,
        SECOND_STUB,
        document(observation={'instrument': OTHER_MISSION, 'image_name': 'C1454725.IMG'}),
    )
    return root


def test_a_mission_filter_keeps_one_missions_images(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The restriction a stream of records honors, honored the same way.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = _mission_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection(instrument=MISSION)))
    assert [_stub_of(one) for one in found] == [FIRST_STUB]


def test_a_selection_naming_stubs_is_answered_in_the_order_it_names_them(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming an image is not a narrowing, so the answer lines up with the ask.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    named = Selection(stubs=(SECOND_STUB, FIRST_STUB))
    found = list(tree_source(root, quiet_logger).facts(named))
    assert [_stub_of(one) for one in found] == [SECOND_STUB, FIRST_STUB]


def test_a_selection_naming_stubs_walks_nothing_and_so_reports_no_metrics(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The metrics are a listing's, and naming an image reads no listing.

    Both of them, because both come from the same listing entry: one of them
    filled from somewhere else would say the file had been walked when it had
    not.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    found = list(tree_source(root, quiet_logger).facts(Selection(stubs=(FIRST_STUB,))))
    assert _facts_of(found, FIRST_STUB).image['mtime_ns'] is None
    assert _facts_of(found, FIRST_STUB).image['size_bytes'] is None


def test_a_selection_naming_a_root_this_source_does_not_hold_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Refused where a caller asked, rather than partway through its loop.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
    """
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    with pytest.raises(ValueError, match='does not hold'):
        source.facts(Selection(roots=(str(tmp_path / 'elsewhere'),)))


def test_a_selection_naming_stubs_under_two_roots_is_refused(tmp_path: Path) -> None:
    """A stub is a key under one root and says nothing about which.

    Parameters:
        tmp_path: Directory the two trees are written under.
    """
    first = two_volume_tree(tmp_path / 'a')
    second = two_volume_tree(tmp_path / 'b')
    source = TreeRecordSource([first, second])
    with pytest.raises(ValueError, match='keys under one root'):
        source.facts(Selection(stubs=(FIRST_STUB,)))


def test_two_roots_are_answered_under_their_own_root_urls(tmp_path: Path) -> None:
    """One stub under two roots is two images, told apart by the root alone.

    Parameters:
        tmp_path: Directory the two trees are written under.
    """
    first = two_volume_tree(tmp_path / 'a')
    second = two_volume_tree(tmp_path / 'b')
    found = list(TreeRecordSource([first, second]).facts(Selection()))
    roots = {str(one.image['root_url']) for one in found if isinstance(one, ImageFacts)}
    assert roots == {first.resolve().as_posix(), second.resolve().as_posix()}


def test_the_facts_of_two_roots_are_not_one_roots_read_twice(tmp_path: Path) -> None:
    """Each root's own values, which is what a root-blind read would lose.

    Parameters:
        tmp_path: Directory the two trees are written under.
    """
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    write_document(first, FIRST_STUB, _named('N1454725799_1.IMG'))
    write_document(second, FIRST_STUB, _named('N9999999999_1.IMG'))
    found = list(TreeRecordSource([first, second]).facts(Selection()))
    named = {
        str(one.image['root_url']): one.image['image_name']
        for one in found
        if isinstance(one, ImageFacts)
    }
    assert named == {
        first.resolve().as_posix(): 'N1454725799_1.IMG',
        second.resolve().as_posix(): 'N9999999999_1.IMG',
    }


def _named(image_name: str) -> dict[str, Any]:
    """Build a document naming one image.

    Parameters:
        image_name: The recorded image name.

    Returns:
        The document.
    """
    return document(observation={'instrument': MISSION, 'image_name': image_name})


def test_a_document_is_parsed_once_for_a_whole_pass(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The facts are built from the document already in hand, not from a re-read.

    Parameters:
        tmp_path: Directory the tree is written under.
        quiet_logger: Logger the walk reports through.
        monkeypatch: Fixture the parser is wrapped through.
    """
    root = two_volume_tree(tmp_path)
    parsed: list[int] = []
    real_loads = json.loads

    def counting(text: Any, **kwargs: Any) -> Any:
        parsed.append(1)
        return real_loads(text, **kwargs)

    monkeypatch.setattr(json, 'loads', counting)
    list(tree_source(root, quiet_logger).facts(Selection()))
    assert len(parsed) == 2
