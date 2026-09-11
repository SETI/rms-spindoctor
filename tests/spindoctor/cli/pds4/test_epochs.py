"""Tests for the rule a navigation document's epochs are held to before a label states them.

Each document here differs from one that passes in one respect, and the description the
check gives for it is asserted in full, since that description is what the log line
naming the image carries.  The last two hold the reason a scan of such documents gives
when one of them leaves it no range.
"""

from typing import Any

import pytest

from spindoctor.cli.pds4.epochs import EpochRangeScan, NoEpochRange, unrecorded_epoch

_TIMES: dict[str, Any] = {
    'start_et': 129399999.77,
    'stop_et': 129400000.23,
    'midtime_et': 129400000.0,
}
"""Epochs a success document records, the stop after the start."""


def _document(**times: Any) -> dict[str, Any]:
    """Return a success document recording the passing epochs with ``times`` over them.

    Parameters:
        **times: Epochs to record in place of the passing ones.

    Returns:
        The document.
    """
    return {'status': 'success', 'navigation_result': {'times': {**_TIMES, **times}}}


def _document_without(key: str) -> dict[str, Any]:
    """Return a success document recording every passing epoch but one.

    Parameters:
        key: The epoch left out.

    Returns:
        The document.
    """
    document = _document()
    del document['navigation_result']['times'][key]
    return document


def test_a_document_recording_all_three_epochs_passes() -> None:
    """Finite epochs, the stop after the start, give a label everything it states."""
    assert unrecorded_epoch(_document()) is None


def test_a_stop_equal_to_its_start_passes() -> None:
    """An interval of no length is still one whose stop is not before its start."""
    assert unrecorded_epoch(_document(stop_et=129399999.77)) is None


@pytest.mark.parametrize(
    ('document', 'expected'),
    [
        ({'status': 'success'}, 'records no navigation_result block'),
        (
            {'status': 'success', 'navigation_result': []},
            'records no navigation_result block',
        ),
        (
            {'status': 'success', 'navigation_result': {}},
            'records no navigation_result.times block',
        ),
        (
            {'status': 'success', 'navigation_result': {'times': []}},
            'records no navigation_result.times block',
        ),
        (_document_without('stop_et'), 'records no navigation_result.times.stop_et'),
        (
            _document(start_et=float('nan')),
            'records a navigation_result.times.start_et of nan, which is not a finite number',
        ),
        (
            _document(midtime_et=float('inf')),
            'records a navigation_result.times.midtime_et of inf, which is not a finite number',
        ),
        (
            _document(stop_et='129400000.23'),
            "records a navigation_result.times.stop_et of '129400000.23', which is not a "
            'finite number',
        ),
        (
            _document(midtime_et=None),
            'records a navigation_result.times.midtime_et of None, which is not a finite number',
        ),
        (
            _document(start_et=True),
            'records a navigation_result.times.start_et of True, which is not a finite number',
        ),
        (
            _document(start_et=10**400),
            'records a navigation_result.times.start_et of an integer 401 digits long, which '
            'is too large for a float',
        ),
        (
            _document(stop_et=129399999.0),
            'records a navigation_result.times.stop_et of 129399999.0, earlier than its '
            'start_et of 129399999.77',
        ),
    ],
    ids=[
        'no navigation result',
        'a navigation result that is not an object',
        'no times block',
        'a times block that is not an object',
        'no stop',
        'a NaN start',
        'an infinite midtime',
        'a stop recorded as a string',
        'a null midtime',
        'a boolean start',
        'an integer start too large for a float',
        'a stop before its start',
    ],
)
def test_a_document_lacking_an_epoch_a_label_can_state_is_described(
    document: dict[str, Any], expected: str
) -> None:
    """Every way a document can fail to give a label its times, named as the log names it.

    Parameters:
        document: The navigation document checked.
        expected: The description the check gives for it.
    """
    assert unrecorded_epoch(document) == expected


def test_the_first_refused_document_is_the_reason_a_scan_gives() -> None:
    """Of two refused documents, the reason names the one read first.

    One refusal is enough to leave no range, so the first is the one kept.  A document
    that passes is read before either, so the range it began does not hide them.
    """
    scan = EpochRangeScan()
    scan.include('supplemental file A', _document())
    scan.include('supplemental file B', {'status': 'success'})
    scan.include('supplemental file C', {'status': 'success', 'navigation_result': {}})
    expected = NoEpochRange(
        'supplemental file B records no navigation_result block, so no range can be taken '
        'that contains every product'
    )
    assert scan.result() == expected


def test_a_scan_whose_every_document_is_refused_names_the_refusal() -> None:
    """With no document passing, the reason is the refusal, not an absence of products.

    The file was read, so the collection holds its product; what is missing is a time
    for it, and the reason says which file lacks one.
    """
    scan = EpochRangeScan()
    scan.include('supplemental file B', {'status': 'success'})
    expected = NoEpochRange(
        'supplemental file B records no navigation_result block, so no range can be taken '
        'that contains every product'
    )
    assert scan.result() == expected
