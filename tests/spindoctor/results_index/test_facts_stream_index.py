"""What the index alone owes a stream of facts.

Read against the index rather than against the two storages together: these are
guarantees a reader of documents has for nothing and an index has to be made to
keep.  The facts are the whole row whatever columns a consumer selected; a
selection narrows a stream of facts exactly as it narrows a stream of records; a
read naming its own stubs is answered in the order it named them, in batches, and
still honours everything else the selection says; and every one of those reads
carries the root half of the key.  The connection the pass borrows comes back.
"""

from pathlib import Path

import pytest
import sqlalchemy
from tests.spindoctor.results_index.conftest import (
    BOTH_ROOTS,
    COLUMNS,
    ERROR_STUB,
    EXTRA_STUB,
    FIRST_VALUES,
    MISSION,
    OTHER_MISSION_STUB,
    OTHER_ROOT_REFUSED_STUB,
    REFUSED_STUB,
    SUCCESS_STUB,
    TwoRoots,
    facts_from_index,
    facts_from_tree,
    facts_of,
    named_facts_from_index,
    stub_of,
    technique_key,
)

from spindoctor.nav_records import (
    Selection,
    UnreadableFile,
    normalize_root_url,
)
from spindoctor.results_index import (
    IMAGES,
    STUBS_PER_STATEMENT,
    IndexRecordSource,
    open_index,
    open_record_source,
)


def test_the_facts_carry_every_column_whatever_the_consumer_selected(
    two_roots: TwoRoots,
) -> None:
    """The columns narrow a record; the facts are the whole row by definition.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), SUCCESS_STUB)
    assert set(found.image) == {column.name for column in IMAGES.columns}


def test_the_consumers_columns_are_fewer_than_the_whole_row() -> None:
    """Without which the test above would hold whatever the statement selected."""
    assert len(COLUMNS) < len(IMAGES.columns)


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_mission_filter_keeps_one_missions_images(two_roots: TwoRoots, which: str) -> None:
    """The restriction both storages honour the same way.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    selection = Selection(instrument=MISSION)
    from_tree = [stub_of(one) for one in facts_from_tree(two_roots, which, selection)]
    from_index = [stub_of(one) for one in facts_from_index(two_roots, which, selection)]
    assert from_index == from_tree


def test_the_children_of_a_dropped_image_are_not_merged_onto_a_kept_one(
    two_roots: TwoRoots,
) -> None:
    """The dropped image sorts first, so its rows are the first the merge meets.

    A merge reading every child row under the root would hold that image's
    techniques against a key it never yields, and every image after it would
    come back with none.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(
        facts_from_index(two_roots, 'first', Selection(instrument=MISSION)), SUCCESS_STUB
    )
    assert [technique_key(row) for row in sorted(found.techniques, key=technique_key)] == [
        'BodyLimbNav',
        'StarFieldFromCatalogNav',
    ]


def test_the_dropped_image_really_holds_child_rows(two_roots: TwoRoots) -> None:
    """Without which the test above would prove nothing about the merge.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), OTHER_MISSION_STUB)
    assert [technique_key(row) for row in found.techniques] == ['StarUniqueMatchNav']


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_subtree_filter_keeps_one_subtrees_images(two_roots: TwoRoots, which: str) -> None:
    """Answered from a column on one storage and from a walk on the other.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    selection = Selection(subtrees=('VOL1',))
    from_tree = [stub_of(one) for one in facts_from_tree(two_roots, which, selection)]
    from_index = [stub_of(one) for one in facts_from_index(two_roots, which, selection)]
    assert from_index == from_tree


def test_a_selection_naming_stubs_is_answered_in_the_order_it_names_them(
    two_roots: TwoRoots,
) -> None:
    """Naming an image is not a narrowing, so the answer lines up with the ask.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(ERROR_STUB, SUCCESS_STUB, REFUSED_STUB))
    found = named_facts_from_index(two_roots, 'first', named)
    assert [stub_of(one) for one in found] == [ERROR_STUB, SUCCESS_STUB, REFUSED_STUB]


def test_a_selection_naming_stubs_reads_the_selected_roots_values(two_roots: TwoRoots) -> None:
    """The other root holds the same stub, recording something else.

    Read against the *first* root, whose rows were written first.  A batch read
    builds what it found with a dictionary update over a stream ordered by the
    key, so a query that dropped the root half would be answered by whichever
    row came back last -- which is the root whose URL sorts last, and asking for
    that one would be answered correctly by the defect.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB,))
    found = named_facts_from_index(two_roots, 'first', named)
    assert facts_of(found, SUCCESS_STUB).image['covariance_px2'] == FIRST_VALUES.twist_covariance


def test_the_root_read_first_is_not_the_one_a_root_blind_read_would_answer_with(
    two_roots: TwoRoots,
) -> None:
    """Without which the direction of the test above would be the wrong way round.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    assert normalize_root_url(two_roots.first) < normalize_root_url(two_roots.second)


def test_a_selection_naming_stubs_carries_the_child_rows(two_roots: TwoRoots) -> None:
    """The merge runs per batch, so a named read has to reach it too.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB,))
    found = named_facts_from_index(two_roots, 'first', named)
    rows = sorted(facts_of(found, SUCCESS_STUB).techniques, key=technique_key)
    assert [technique_key(row) for row in rows] == ['BodyLimbNav', 'StarFieldFromCatalogNav']


_NAMES_CROSSING_A_BATCH = tuple([ERROR_STUB, SUCCESS_STUB] * (STUBS_PER_STATEMENT // 2 + 1))
"""More stub names than one retrieval batch binds, at whatever that batch is sized.

Derived from the constant rather than written out, so that raising the batch
moves this with it instead of quietly leaving the test below inside one batch,
testing nothing.
"""


def test_a_selection_naming_more_stubs_than_one_batch_answers_every_one(
    two_roots: TwoRoots,
) -> None:
    """A caller is free to name a mission's worth, and the read is cut into batches.

    Asked for in more names than one batch binds, so a read that answered the
    first batch and dropped the rest would hand a queue task back short.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = named_facts_from_index(two_roots, 'first', Selection(stubs=_NAMES_CROSSING_A_BATCH))
    assert [stub_of(one) for one in found] == list(_NAMES_CROSSING_A_BATCH)


def test_the_named_stubs_really_do_cross_a_batch_boundary() -> None:
    """Without which the test above would hold whatever the batching did."""
    assert len(_NAMES_CROSSING_A_BATCH) > STUBS_PER_STATEMENT


def test_named_stubs_still_honour_the_mission(two_roots: TwoRoots) -> None:
    """A selection is a narrowing whatever else it names.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB, OTHER_MISSION_STUB), instrument=MISSION)
    found = named_facts_from_index(two_roots, 'first', named)
    assert [stub_of(one) for one in found] == [SUCCESS_STUB]


def test_named_stubs_still_honour_a_time_bound(two_roots: TwoRoots) -> None:
    """The other half of what a selection restricts a named read by.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(SUCCESS_STUB, ERROR_STUB), start_et=FIRST_VALUES.midtime + 5.0)
    found = named_facts_from_index(two_roots, 'first', named)
    assert [stub_of(one) for one in found] == [ERROR_STUB]


def test_a_named_stub_only_the_other_root_holds_yields_nothing(two_roots: TwoRoots) -> None:
    """Naming a key does not stop it being a key under one root.

    The stub is one only the other root holds, so a query that bound the keys
    and dropped the root would hand this run an image nobody asked for -- and
    do it whichever order the server returned its rows in, which naming a stub
    both roots hold cannot show.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = named_facts_from_index(two_roots, 'first', Selection(stubs=(EXTRA_STUB,)))
    assert found == []


def test_a_named_stub_only_the_other_root_refused_yields_nothing(two_roots: TwoRoots) -> None:
    """The refusal half of a named read carries its own root term.

    The other root refused a file at a stub this root holds nothing at, so a
    refusal query blind to the root would report this run a shortfall that is
    not its own.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(OTHER_ROOT_REFUSED_STUB,))
    found = named_facts_from_index(two_roots, 'first', named)
    assert found == []


def test_the_other_roots_refused_file_really_is_refused_there(two_roots: TwoRoots) -> None:
    """Without which the test above would hold over a file nothing recorded.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(OTHER_ROOT_REFUSED_STUB,))
    found = named_facts_from_index(two_roots, 'second', named)
    assert [type(one) for one in found] == [UnreadableFile]


def test_a_selection_naming_a_root_the_source_does_not_hold_is_refused(
    two_roots: TwoRoots, tmp_path: Path
) -> None:
    """Refused where a caller asked, rather than partway through its loop.

    Parameters:
        two_roots: The two ingested roots and their index.
        tmp_path: Directory the unheld root would be under.
    """
    with (
        open_record_source(
            [two_roots.first], results_index_db_url=two_roots.url, columns=COLUMNS
        ) as source,
        pytest.raises(ValueError, match='does not hold'),
    ):
        source.facts(Selection(roots=(str(tmp_path / 'elsewhere'),)))


def test_a_stream_of_facts_gives_its_connection_back(two_roots: TwoRoots) -> None:
    """Three cursors on one connection, all of them released when it is done.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events: list[str] = []
    engine = open_index(two_roots.url)
    sqlalchemy.event.listen(engine, 'checkout', lambda *_args: events.append('out'))
    sqlalchemy.event.listen(engine, 'checkin', lambda *_args: events.append('in'))
    with IndexRecordSource(engine, [two_roots.first], two_roots.url, COLUMNS) as source:
        list(source.facts(Selection()))
    assert events.count('out') == events.count('in')


def test_a_stream_of_facts_takes_a_connection_out_of_the_pool(two_roots: TwoRoots) -> None:
    """Without which the balance above would hold over nothing at all.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events: list[str] = []
    engine = open_index(two_roots.url)
    sqlalchemy.event.listen(engine, 'checkout', lambda *_args: events.append('out'))
    with IndexRecordSource(engine, [two_roots.first], two_roots.url, COLUMNS) as source:
        list(source.facts(Selection()))
    assert events.count('out') == 1
