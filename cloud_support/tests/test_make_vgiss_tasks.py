"""Unit tests for the Voyager generator's encounter selection.

Which encounters a run covers is the only thing this generator decides for
itself; the rest is the shared machinery, tested beside it.

Run these with ``pytest cloud_support/tests``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import make_vgiss_tasks as vgiss


def test_the_four_encounters_come_back_in_archive_order() -> None:
    """The encounters are ordered as the archive orders them, not as typed."""
    assert vgiss.selected_planets('neptune,jupiter,uranus,saturn') == [
        'jupiter',
        'saturn',
        'uranus',
        'neptune',
    ]


def test_an_encounter_named_twice_is_one_encounter() -> None:
    """A repeated name writes one file and counts its tasks once."""
    # Writing its file twice would leave one file and count its tasks twice.
    assert vgiss.selected_planets('jupiter,jupiter') == ['jupiter']


def test_spacing_and_case_do_not_make_a_second_encounter() -> None:
    """Spacing and case are normalized before the duplicate is dropped."""
    assert vgiss.selected_planets(' Jupiter , jupiter ') == ['jupiter']


def test_a_name_that_is_no_encounter_is_refused() -> None:
    """An unknown name is refused, with the four valid ones named."""
    with pytest.raises(ValueError) as exc:
        vgiss.selected_planets('jupiter,mars')
    assert 'Unknown planet(s) mars' in str(exc.value)


def test_each_encounter_knows_its_volumes() -> None:
    """An encounter resolves to the volumes of its volume set."""
    volumes = vgiss.planet_volumes('uranus')
    assert volumes[0] == 'VGISS_7201'
    assert all(volume[6] == '7' for volume in volumes)
