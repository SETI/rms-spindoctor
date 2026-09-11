"""The Cassini ISS host: its camera frames, its exposure, and its clock.

What stamping an attitude solution onto a Cassini result needs to know, as the
:class:`~tests.mini_nav_results.shared.Host` :data:`CASSINI_ISS`, and the
conversions a Cassini document's epochs, clock readings and image number come
from.  The statistics fixture tree's Cassini documents and the Cassini ISS Saturn
cohort are both built from them.

What the clock conversions here are held to, and by what.  The two tests over
the cohort's readings -- that a triple spans the epochs beside it, and that an
image is named for the reading its shutter opened at -- compare two quantities
this module derived from one function, so they report a triple written out by
hand beside one it counted, which is the state they exist to make unreachable.
They cannot report a conversion that is wrong the same way everywhere: move the
anchors below by an hour and every one of them still passes, with every image
renamed.  What reports that is an integration test that furnishes the mission
clock kernel and converts each epoch again, and it is excluded from the default
run because the kernel is not there to furnish.
"""

from __future__ import annotations

import numpy as np

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.support.types import NDArrayFloatType

from .shared import Host, elapsed_ticks, exposure_span, sclk_triple, with_pointing

COISS_KERNELS = (
    '05138_05159ra.bc',
    'cas00172.tsc',
    'cpck15Dec2017.tpc',
    'naif0012.tls',
    'sat428.bsp',
)
"""Kernels loaded for the Cassini images."""


_CASSINI_OOPS_FROM_SPICE: NDArrayFloatType = np.diag([-1.0, -1.0, 1.0])
"""The constant rotation between the oops and SPICE Cassini ISS camera frames."""


_CASSINI_CAMERA_FRAME_IDS = {'NAC': -82360, 'WAC': -82361}
"""SPICE frame id of each Cassini ISS camera frame."""


CASSINI_EXPOSURE_S = 0.46
"""Exposure the Cassini images were taken with."""


CASSINI_EXPOSURE_MS = CASSINI_EXPOSURE_S * 1000.0
"""The same exposure, in the milliseconds a PDS3 index records it in.

Derived rather than written out again beside the index row that carries it: a
row whose exposure is one number while the clock triple beside it is counted
over another spans two different exposures, and no reader of it holds both.
"""


_CASSINI_SCLK_TICKS_PER_SECOND = 256
"""Ticks in one second of the Cassini clock, the modulus of its second field.

The clock is two fields, whole seconds and a fractional field counting ticks of
one 256th of a second, so a reading is a count of those ticks and the fields
are its quotient and its remainder by this.  It is the width of the field, not
the rate the clock runs at; how long a tick lasts is measured against the
kernel further down.
"""


def _cassini_sclk_reading(ticks: int) -> str:
    """Spell a Cassini clock tick count the way the conversion spells it.

    The two fields are written behind the clock partition and separated by a
    period, each zero padded to the digits its own modulus needs: ten for the
    seconds and three for the 256 ticks of the fraction.  A tick count past the
    fraction's modulus therefore carries into the seconds field rather than
    widening the fraction.

    Parameters:
        ticks: The reading, as a count of ticks of one 256th of a second.

    Returns:
        The clock string.
    """
    seconds, fraction = divmod(ticks, _CASSINI_SCLK_TICKS_PER_SECOND)
    return f'1/{seconds:010d}.{fraction:03d}'


def cassini_sclk_open(image_number: int, tick: int) -> int:
    """Return a Cassini image's clock reading at shutter open, as a tick count.

    A Cassini image is named for the whole-second field of the reading its
    shutter opened at: a label carrying ``IMAGE_NUMBER = "1454725799"`` carries
    ``SPACECRAFT_CLOCK_START_COUNT = "1454725799.102"`` beside it.  So the
    image number and the tick the shutter opened on are the two fields of that
    reading.

    Parameters:
        image_number: The image number, which is the whole-second field.
        tick: The fractional field, in ticks of one 256th of a second.

    Returns:
        The reading, as a count of ticks.
    """
    return image_number * _CASSINI_SCLK_TICKS_PER_SECOND + tick


def cassini_exposure_span(midtime_et: float) -> tuple[float, float, float]:
    """Return the start, midtime and stop epochs of one Cassini exposure.

    Parameters:
        midtime_et: The exposure midtime, which is the image's epoch.

    Returns:
        The three epochs, in that order.
    """
    return exposure_span(midtime_et, CASSINI_EXPOSURE_S)


# ---------------------------------------------------------------------------
# Epochs first, everything else derived
# ---------------------------------------------------------------------------
#
# An image's epoch, the clock readings recorded beside it and the number it is
# named for are three spellings of one moment, and a document that spells them
# from three sources is free to disagree with itself: the reading says the
# shutter opened years from where the epoch says it did, and every reader that
# converts one into the other reads a document no run could have written.  So a
# document is built from its epoch alone, through the constructors below, and
# there is nowhere in that path for a second answer to enter.


_CASSINI_SCLK_ANCHORS = ((1454725799, 129305290.24137056), (1456120518, 130700000.15065941))
"""Two Cassini clock readings and the epochs the mission clock kernel gives them.

Both pairs are correlation points read out of ``cas00172.tsc`` rather than
numbers chosen here, and they bracket every epoch the cohort uses.  A line
through two of them calibrates the rate the clock runs at as well as where it
started, which one of them cannot: the clock gains 6.5 parts per million on
ephemeris time, so a conversion anchored at one point alone reads 0.6 s off a
day away and 9.1 s off at the far end of the cohort's own span, and named one
cohort image for a second the kernel puts nine seconds later.

A mission clock kernel is linear in pieces, each with its own rate, so a single
line cannot be right everywhere.  Measured against the kernel over the 16 days
these two span, this one is never more than half a tick out, and each of the
three cohort epochs converts to exactly the tick the kernel returns for it.
"""


_CASSINI_SCLK_ANCHOR_TICKS = _CASSINI_SCLK_ANCHORS[0][0] * _CASSINI_SCLK_TICKS_PER_SECOND
"""The first anchor's reading, as a tick count, which the conversion counts from."""


_CASSINI_SCLK_ANCHOR_ET = _CASSINI_SCLK_ANCHORS[0][1]
"""The epoch of that reading."""


_CASSINI_SCLK_TICK_S = (_CASSINI_SCLK_ANCHORS[1][1] - _CASSINI_SCLK_ANCHOR_ET) / (
    _CASSINI_SCLK_ANCHORS[1][0] * _CASSINI_SCLK_TICKS_PER_SECOND - _CASSINI_SCLK_ANCHOR_TICKS
)
"""How long one tick of the Cassini clock lasts, as the two anchors measure it.

Slightly less than one 256th of a second, which is the whole point of taking
two of them.
"""


def cassini_sclk_at(epoch_et: float) -> int:
    """Return the Cassini clock's reading at an epoch, as a tick count.

    The conversion is the line through the two kernel correlation points above,
    which is what a mission clock kernel is over any short enough span.

    Parameters:
        epoch_et: The epoch to read the clock at.

    Returns:
        The reading, as a count of ticks of the clock's fractional field.
    """
    return _CASSINI_SCLK_ANCHOR_TICKS + elapsed_ticks(
        epoch_et - _CASSINI_SCLK_ANCHOR_ET, _CASSINI_SCLK_TICK_S
    )


def cassini_image_number(midtime_et: float) -> int:
    """Return the number a Cassini image taken at this epoch is named for.

    The name is the whole-second field of the reading the shutter opened at, so
    an image built from its epoch cannot be named for a moment its own clock
    readings do not cover.

    Parameters:
        midtime_et: The exposure midtime, which is the image's epoch.

    Returns:
        The image number.
    """
    start_et, _midtime_et, _stop_et = cassini_exposure_span(midtime_et)
    return cassini_sclk_at(start_et) // _CASSINI_SCLK_TICKS_PER_SECOND


def cassini_sclk_triple(midtime_et: float) -> tuple[str, str, str]:
    """Return the Cassini clock readings at the three epochs of one exposure.

    Parameters:
        midtime_et: The exposure midtime, which is the image's epoch.

    Returns:
        The readings at start, midtime and stop, spelled as the conversion
        spells them, partition and all.
    """
    start_et, _midtime_et, stop_et = cassini_exposure_span(midtime_et)
    return sclk_triple(
        cassini_sclk_at(start_et),
        start_et=start_et,
        midtime_et=midtime_et,
        stop_et=stop_et,
        tick_s=_CASSINI_SCLK_TICK_S,
        spell=_cassini_sclk_reading,
    )


CASSINI_ISS = Host(
    camera_frames={
        camera: (f'CASSINI_ISS_{camera}', frame_id)
        for camera, frame_id in _CASSINI_CAMERA_FRAME_IDS.items()
    },
    ck_frame_id=-82000,
    oops_from_spice=_CASSINI_OOPS_FROM_SPICE,
    exposure_s=CASSINI_EXPOSURE_S,
    tick_s=_CASSINI_SCLK_TICK_S,
    spell=_cassini_sclk_reading,
)
"""The Cassini ISS host: its cameras' frames, its exposure, and its clock.

The clock's tick is the one the two kernel anchors measure, not the nominal
256th of a second its fractional field counts.
"""


def with_pointing_from_epoch(
    result: NavResult,
    *,
    camera: str,
    midtime_et: float,
    original: NDArrayFloatType,
    corrected: NDArrayFloatType | None,
) -> NavResult:
    """Stamp a Cassini attitude solution derived from the image's epoch alone.

    The clock triple comes from :func:`cassini_sclk_triple`, so the solution a
    document carries is the one that epoch converts to, on :data:`CASSINI_ISS`'s
    frames, exposure and clock.

    Parameters:
        result: The result to stamp.
        camera: The camera that took the image, which names its frame.
        midtime_et: Exposure midtime, which is also the image's epoch.
        original: The uncorrected attitude at midtime.
        corrected: The corrected attitude, or None for a result with no offset.

    Returns:
        The same result, carrying the solution.
    """
    start_et, _midtime_et, _stop_et = cassini_exposure_span(midtime_et)
    return with_pointing(
        result,
        host=CASSINI_ISS,
        camera=camera,
        midtime_et=midtime_et,
        sclk_open=cassini_sclk_at(start_et),
        original=original,
        corrected=corrected,
    )
