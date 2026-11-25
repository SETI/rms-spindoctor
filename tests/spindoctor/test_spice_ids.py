"""Tests for the C-kernel writer's CK-object-to-spacecraft-clock mapping.

The mapping exists because ``cspyce.ckmeta`` computes a clock id from a CK
object id instead of validating one, so the C-kernel writer checks what it
resolves against a recorded table rather than trusting it.  These tests pin the
table against SPICE and pin the writer to the table.

``ckmeta`` needs no furnished kernels, so all of this is hermetic.
"""

from typing import cast

import cspyce
import pytest

from spindoctor.cli.ck.segment import resolve_sclk_id
from spindoctor.spice_ids import (
    CK_OBJECT_SCLK_ID,
    FROZEN_ATTITUDE_CK_IDS,
    VOYAGER_CK_OBJECT_ID,
)

# The recorded pairs, in a stable order so a failure names the same case run
# to run.  Voyager 1 is the pair integer division gets wrong: ``-31100 //
# 1000`` is -32 in Python, which is the other spacecraft.
_RECORDED_PAIRS = sorted(CK_OBJECT_SCLK_ID.items())

# A CK object no mission owns, used to pin what ``ckmeta`` does with one.
_NONEXISTENT_CK_FRAME_ID = -999999


@pytest.mark.parametrize(('ck_frame_id', 'sclk_id'), _RECORDED_PAIRS)
def test_every_recorded_pair_agrees_with_spice(ck_frame_id: int, sclk_id: int) -> None:
    """Each recorded CK object resolves to the clock recorded beside it.

    A typo in either half of a pair would otherwise only surface as time tags
    encoding another spacecraft's clock, on every record of every kernel.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected kernel targets.
        sclk_id: Spacecraft clock that object's time tags are encoded against.
    """
    assert int(cspyce.ckmeta(ck_frame_id, 'SCLK')) == sclk_id


@pytest.mark.parametrize(('ck_frame_id', 'sclk_id'), _RECORDED_PAIRS)
def test_the_kernel_writer_reads_the_recorded_pair(ck_frame_id: int, sclk_id: int) -> None:
    """The writer's resolved clock for an object is the recorded one.

    The writer additionally cross-checks what ``ckmeta`` computes, so this
    fails both for a writer that stopped reading the table and for a table
    that stopped agreeing with SPICE.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected kernel targets.
        sclk_id: Spacecraft clock that object's time tags are encoded against.
    """
    assert resolve_sclk_id(ck_frame_id) == sclk_id


def test_the_frozen_attitude_set_is_the_two_voyager_platforms() -> None:
    """The objects whose navigated attitude is one snapped lookup, pinned by id.

    Both the writer and the step that pairs an image with its baseline branch
    on this set, and neither can tell a wrong id from a mission that simply
    never appears.
    """
    assert set(FROZEN_ATTITUDE_CK_IDS) == {-31100, -32100}


def test_every_voyager_platform_has_a_recorded_clock() -> None:
    """A platform with no clock would encode no time tags at all."""
    assert set(VOYAGER_CK_OBJECT_ID.values()) <= set(CK_OBJECT_SCLK_ID)


def test_ckmeta_computes_a_clock_for_a_ck_object_that_does_not_exist() -> None:
    """``ckmeta`` computes rather than validates, which is why the table exists.

    A nonexistent CK object still yields a plausible clock id and no error, so
    trusting the round trip would write time tags from the wrong clock while
    every call reported success.
    """
    assert int(cspyce.ckmeta(_NONEXISTENT_CK_FRAME_ID, 'SCLK')) == -999


def test_the_recorded_table_is_read_only() -> None:
    """No consumer can edit the table the writer checks against."""
    with pytest.raises(TypeError, match='does not support item assignment'):
        cast(dict[int, int], CK_OBJECT_SCLK_ID)[-82000] = -1
