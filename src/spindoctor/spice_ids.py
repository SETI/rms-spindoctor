"""SPICE ids the C-kernel writer reads.

The first fact recorded here is which spacecraft clock a C-kernel object's time
tags are encoded against.  ``cspyce.ckmeta`` computes that mapping rather than
validating it -- it answers -999 for the nonexistent object -999999 without
raising -- so a wrong CK object id yields a plausible clock id, a successful
encoding, and silently wrong time tags on every record.  The writer therefore
checks what ``ckmeta`` resolves against the mapping below instead of trusting
it.  The attitude computation makes the same check against the clock each host
names in an observation's ``ck_sclk_id`` subfield; the writer reads navigation
records and kernel files rather than observations, so it keeps this copy.

The second is which CK object each Voyager spacecraft's scan platform is, and
the set of objects whose navigated attitude is frozen across the exposure.
Those are one fact stated twice: the Voyager host builds its observation frame
from a single pointing lookup at the image's clock stop count, within 20 ticks,
rather than an evaluated frame chain, so the frozen set is exactly the Voyager
platforms and is derived from them rather than written out again.

The clock is not derivable from the CK object by arithmetic.  Python's
``-31100 // 1000`` is -32, which is the other Voyager, so each object states
its clock explicitly.

This module holds constants and imports only the standard library.
"""

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    'CK_OBJECT_SCLK_ID',
    'FROZEN_ATTITUDE_CK_IDS',
    'VOYAGER_CK_OBJECT_ID',
]

# Read-only so that no consumer can edit the table the other consumer checks
# against.
CK_OBJECT_SCLK_ID: Mapping[int, int] = MappingProxyType(
    {
        -82000: -82,  # Cassini bus
        -77001: -77,  # Galileo scan platform
        -98000: -98,  # New Horizons spacecraft
        -31100: -31,  # Voyager 1 scan platform
        -32100: -32,  # Voyager 2 scan platform
    }
)
"""The spacecraft clock each supported CK object's time tags are encoded against."""

# Keyed by the spacecraft digit the Voyager image labels carry, because one
# instrument key serves two spacecraft.  Written out rather than computed from
# the digit: ``-31100`` and ``-32100`` are not derivable from ``1`` and ``2``
# by any arithmetic a reader can check at a glance.
VOYAGER_CK_OBJECT_ID: Mapping[str, int] = MappingProxyType(
    {
        '1': -31100,  # Voyager 1 scan platform
        '2': -32100,  # Voyager 2 scan platform
    }
)
"""The scan platform each Voyager spacecraft's corrected C-kernel targets."""

FROZEN_ATTITUDE_CK_IDS: frozenset[int] = frozenset(VOYAGER_CK_OBJECT_ID.values())
"""The CK objects whose navigated attitude is one snapped lookup, constant across the exposure.

Derived from the Voyager platforms rather than restated, because that is what
makes an object frozen: the Voyager host builds its observation frame from one
pointing lookup at the image's clock stop count instead of evaluating a frame
chain, so a segment for one of these objects carries a single attitude and the
step that identifies which baseline kernel an image navigated against has to
make the same lookup rather than evaluating a chain.
"""
# #1085: that step still makes its lookup at the exposure midtime
