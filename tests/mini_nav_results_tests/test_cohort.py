"""Self-tests for the bundle cohort the PDS4 phases are asserted against.

The cohort is built from each image's epoch and nothing else, so what is worth
holding it to is that everything derived from that epoch still agrees with it:
the clock readings a document records, and the number the image is named for.
An image whose name and epoch disagree is the defect the epoch-first
constructor exists to make unreachable, and it is invisible to any test that
reads one of them alone.

The rest is the layout the bundle stage is built over: two images that shard
into different bundle directories, one with ring backplanes and one without.
"""

from typing import Any

import pytest

from spindoctor.dataset import DataSetPDS3CassiniISSSaturn
from tests.mini_nav_results import cohort_documents
from tests.mini_nav_results.cohort_cassini import (
    LIMB_IMAGE_NAME,
    RINGS_IMAGE_NAME,
    cohort_images,
)

_SCLK_TICK_S = 1.0 / 256.0
"""How long one tick of the Cassini clock is.

Written out rather than taken from the builders: a test that asks the code what
its own clock counts in agrees with every answer the code gives.  This is the
tick the mission clock kernel records, so a triple that disagrees with its own
epochs by more than one of them is a triple no conversion could have returned.
"""


def _sclk_seconds(reading: str) -> float:
    """Read a Cassini clock string back as a number of seconds on that clock.

    Parameters:
        reading: The clock string, partition and all.

    Returns:
        The reading in seconds, on the clock's own origin.  Only differences
        between two readings mean anything.
    """
    seconds, fraction = reading.split('/', 1)[1].split('.')
    return int(seconds) + int(fraction) * _SCLK_TICK_S


@pytest.fixture(scope='module')
def documents() -> dict[str, dict[str, Any]]:
    """Build every cohort document once for the whole module.

    Returns:
        Stub to the document the writer produces for it.
    """
    return cohort_documents()


def test_every_clock_triple_spans_the_epochs_beside_it(
    documents: dict[str, dict[str, Any]],
) -> None:
    """A reading that is not the one its epoch converts to is an invented one.

    Every reader that subtracts two readings, or converts one back into an
    epoch, reads whatever a hand-authored triple happened to say.
    """
    disagreeing: list[str] = []
    for stub, document in documents.items():
        times = document['navigation_result']['times']
        opened = _sclk_seconds(str(times['sclk_start']))
        for reading, epoch in (('sclk_midtime', 'midtime_et'), ('sclk_stop', 'stop_et')):
            on_the_clock = _sclk_seconds(str(times[reading])) - opened
            between_the_epochs = float(times[epoch]) - float(times['start_et'])
            if abs(on_the_clock - between_the_epochs) > _SCLK_TICK_S:
                disagreeing.append(
                    f'{stub}: {reading} is {on_the_clock} s after sclk_start, against '
                    f'{between_the_epochs} s between the epochs'
                )
    assert disagreeing == []


def test_every_image_is_named_for_the_reading_its_shutter_opened_at(
    documents: dict[str, dict[str, Any]],
) -> None:
    """The name and the epoch are two spellings of one moment, or the document lies.

    A Cassini image is named for the whole-second field of the clock reading at
    shutter open, so the name is derivable from the epoch and any disagreement
    is a number that came from somewhere else.
    """
    disagreeing: list[str] = []
    for image in cohort_images():
        recorded = str(documents[image.stub]['navigation_result']['times']['sclk_start'])
        named = image.image_name[1:].split('_', 1)[0]
        if recorded.split('/', 1)[1].split('.')[0] != named:
            disagreeing.append(f'{image.image_name} carries sclk_start {recorded}')
    assert disagreeing == []


def test_the_two_navigated_images_shard_into_different_bundle_directories() -> None:
    """A cohort that shards into one directory cannot show a per-directory defect."""
    shards = [
        DataSetPDS3CassiniISSSaturn.pds4_bundle_path_for_image(name.split('_', 1)[0])
        for name in (LIMB_IMAGE_NAME, RINGS_IMAGE_NAME)
    ]
    assert shards[0] != shards[1]
