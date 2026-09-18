"""The stored fixture tree is what the writer emits, and says what it claims to.

``data/results_tree`` is the input the statistics ingest is measured over and
the frozen report output is derived from, so a document there that no writer
could have produced puts the whole measurement outside the schema: key sets,
vocabularies and value shapes the ingest and the report then read from nothing
the pipeline writes.  Nothing held these documents against the writer before,
which is how they came to diverge without anyone noticing.

This is what holds them.  The ``tests.mini_nav_results`` package builds each
document through the writer itself, and the first test here compares the bytes
it produces against the bytes on disk, so the stored tree is writer output by
construction.  A writer change is then reported here, and the fix is to run
that package and re-ratify the frozen report output against what it wrote.

The rest are the properties the tree is relied on for that the frozen report
cannot see: the attitude and exposure blocks a host with SPICE frames always
records, and the internal
agreements a hand-authored document is free to break -- a technique citing a
feature the inventory does not hold, a spurious result reported as excluded
from consensus, a recorded midtime that is not the recorded epoch, a spacecraft
clock triple that spans a fraction of the exposure it was read over.  What
only one host's documents can say, such as the shutter modes a host's labels
record, is held in a test module named for the host.
"""

from typing import Any

import pytest
from tests.mini_nav_results import (
    RESULTS_TREE,
    results_tree_documents,
    stored_documents,
)
from tests.sclk_readings import triples_disagreeing_with_their_epochs

from spindoctor.support.file import json_as_string

_SIMULATED = 'sim_scene_000042'
"""The stub of the simulated scene, the one host with no SPICE camera frame."""


@pytest.fixture(scope='module')
def built() -> dict[str, dict[str, Any]]:
    """Build every document once for the whole module.

    Returns:
        Stub to the document the writer produces for it.
    """
    return results_tree_documents()


@pytest.fixture(scope='module')
def stored() -> dict[str, dict[str, Any]]:
    """Read every document the tree holds, once for the whole module.

    Returns:
        Stub to the parsed document.
    """
    return stored_documents()


def test_the_tree_holds_exactly_the_documents_the_corpus_names(
    built: dict[str, dict[str, Any]], stored: dict[str, dict[str, Any]]
) -> None:
    """A document nobody builds, or one nobody wrote, is a tree out of step."""
    assert sorted(stored) == sorted(built)


def test_every_stored_document_is_byte_for_byte_what_the_writer_emits(
    built: dict[str, dict[str, Any]],
) -> None:
    """The bytes on disk are the writer's own serialization of its own output."""
    differing = [
        stub
        for stub, document in built.items()
        if (RESULTS_TREE / f'{stub}_metadata.json').read_text(encoding='utf-8')
        != json_as_string(document)
    ]
    assert differing == []


def test_every_navigated_image_with_spice_frames_records_its_attitude_and_times(
    stored: dict[str, dict[str, Any]],
) -> None:
    """Both blocks are stamped for every result of such a host, failures included.

    Only an image that never loaded, whose document records status ``error``, has
    no result to stamp.
    """
    missing = sorted(
        stub
        for stub, document in stored.items()
        if document['status'] != 'error'
        if stub != _SIMULATED
        if not {'pointing', 'times'} <= set(document['navigation_result'])
    )
    assert missing == []


def test_only_a_navigation_that_produced_an_offset_records_a_corrected_attitude(
    stored: dict[str, dict[str, Any]],
) -> None:
    """A failed navigation has no offset, so it has no correction to record."""
    wrong = sorted(
        stub
        for stub, document in stored.items()
        if 'pointing' in document.get('navigation_result', {})
        if ('cmatrix' in document['navigation_result']['pointing'])
        != (document['status'] == 'success')
    )
    assert wrong == []


def test_the_simulated_scene_records_no_attitude_and_no_times(
    stored: dict[str, dict[str, Any]],
) -> None:
    """It has no spacecraft and no furnished camera frame, so it records neither."""
    navigation_result = stored[_SIMULATED]['navigation_result']
    assert 'pointing' not in navigation_result
    assert 'times' not in navigation_result


def test_every_recorded_midtime_is_the_recorded_epoch(
    stored: dict[str, dict[str, Any]],
) -> None:
    """An image's epoch is its observation's midtime, and a reader gates on it."""
    disagreeing = sorted(
        stub
        for stub, document in stored.items()
        if 'times' in document.get('navigation_result', {})
        if document['navigation_result']['times']['midtime_et']
        != document['navigation_result']['provenance']['image_et']
    )
    assert disagreeing == []


def test_every_clock_triple_spans_the_exposure_it_was_read_over(
    stored: dict[str, dict[str, Any]],
) -> None:
    """A triple narrower than its own epochs is one no clock conversion returns.

    The readings come from the epochs beside them, so the interval between two
    of them is the interval between those epochs, to within the tick the clock
    counts in.  A triple that says the shutter was open for a fifth of the time
    the epochs say is a hand-authored one, and every reader that subtracts two
    of its readings measures that fifth.
    """
    assert triples_disagreeing_with_their_epochs(stored) == []


def test_every_technique_cites_features_the_inventory_holds(
    stored: dict[str, dict[str, Any]],
) -> None:
    """A technique consumes features that were extracted and survived the gate."""
    unknown: list[str] = []
    for stub, document in stored.items():
        navigation_result = document.get('navigation_result', {})
        kept = {
            str(entry['feature_id'])
            for entry in navigation_result.get('feature_inventory', [])
            if not entry['gated']
        }
        for entry in navigation_result.get('per_technique', []):
            unknown += [
                f'{stub}: {entry["technique_name"]} cites {feature_id}'
                for feature_id in entry['feature_ids']
                if feature_id not in kept
            ]
    assert unknown == []


def test_no_spurious_result_is_reported_as_excluded_from_consensus(
    stored: dict[str, dict[str, Any]],
) -> None:
    """The ensemble drops a spurious result before it selects a consensus.

    So a spurious technique is never one the consensus left out, and a document
    naming it in both places describes an ensemble run that cannot happen.
    """
    both: list[str] = []
    for stub, document in stored.items():
        navigation_result = document.get('navigation_result', {})
        excluded = set(navigation_result.get('excluded_from_consensus', []))
        both += [
            f'{stub}: {entry["technique_name"]}'
            for entry in navigation_result.get('per_technique', [])
            if entry['spurious']
            if str(entry['technique_name']) in excluded
        ]
    assert both == []
