"""The statistics fixture tree's Cassini documents record the shutter modes they claim.

Cassini labels record a shutter mode, and the tree is relied on for two of its
values: the BOTSIM pair, one shutter over two cameras, which is a pair only while
both record the mode that makes it one and share an epoch and a clock reading;
and the images whose labels record a single camera.  No other host's labels carry
a mode, so no other host's document records one, and neither does the Cassini
image whose load failed.
"""

from typing import Any

import pytest
from tests.mini_nav_results import stored_documents

_BOTSIM_PAIR = (
    'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    'COISS_2001/data/1294561143_1295221348/W1294561202_1_CALIB',
)
"""The two stubs of the pair whose cameras were shuttered together."""


_SINGLE_CAMERA = {
    'COISS_2001/data/1294561143_1295221348/N1294562000_1_CALIB': 'NACONLY',
    'COISS_2001/data/1294561143_1295221348/N1294564000_1_CALIB': 'NACONLY',
}
"""The Cassini stubs whose labels record one camera, and which mode they record."""


_LOAD_ERROR = 'COISS_2001/data/1294561143_1295221348/N1294563000_1_CALIB'
"""The stub of the image whose load failed before an observation existed."""


@pytest.fixture(scope='module')
def stored() -> dict[str, dict[str, Any]]:
    """Read every document the tree holds, once for the whole module.

    Returns:
        Stub to the parsed document.
    """
    return stored_documents()


def test_the_botsim_pair_records_the_mode_that_makes_it_a_pair(
    stored: dict[str, dict[str, Any]],
) -> None:
    """Without the shutter mode the pair is only two images sharing a number."""
    modes = [stored[stub]['observation']['shutter_mode'] for stub in _BOTSIM_PAIR]
    assert modes == ['BOTSIM', 'BOTSIM']


def test_the_botsim_pair_shares_one_shutter(stored: dict[str, dict[str, Any]]) -> None:
    """One shutter is one epoch and one clock reading, on both cameras."""
    times = [stored[stub]['navigation_result']['times'] for stub in _BOTSIM_PAIR]
    assert times[0]['midtime_et'] == times[1]['midtime_et']
    assert times[0]['sclk_midtime'] == times[1]['sclk_midtime']


def test_the_single_camera_images_record_their_own_shutter_mode(
    stored: dict[str, dict[str, Any]],
) -> None:
    """A column that only ever held one value would not tell the modes apart."""
    found = {stub: stored[stub]['observation']['shutter_mode'] for stub in _SINGLE_CAMERA}
    assert found == _SINGLE_CAMERA


def test_the_hosts_whose_labels_carry_no_shutter_mode_record_none(
    stored: dict[str, dict[str, Any]],
) -> None:
    """Voyager, the simulated scene, and an image that never loaded record none."""
    carrying = sorted(
        stub
        for stub, document in stored.items()
        if str(document['observation']['instrument']) != 'coiss'
        and 'shutter_mode' in document['observation']
    )
    assert carrying == []
    assert 'shutter_mode' not in stored[_LOAD_ERROR]['observation']
