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

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.obs.obs_inst_cassini_iss import _label_metadata, _published_sclk
from spindoctor.support.cmatrix import AttitudeBaseline
from spindoctor.support.time import et_to_utc
from spindoctor.support.types import NDArrayFloatType

from .shared import (
    Host,
    elapsed_ticks,
    exposure_span,
    holdings_path,
    published_times,
    recorded_exposure,
    sclk_triple,
    with_pointing,
)

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


_GAIN_MODE_IDS = {
    0: '215 ELECTRONS PER DN',
    1: '95 ELECTRONS PER DN',
    2: '29 ELECTRONS PER DN',
    3: '12 ELECTRONS PER DN',
}
"""A label's gain text for each gain state oops reads out of it."""

_SHARED_LABEL_ITEMS: dict[str, Any] = {
    'DATA_SET_ID': 'CO-S-ISSNA/ISSWA-2-EDR-V1.0',
    'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
    'MISSION_NAME': 'CASSINI-HUYGENS',
    'MISSION_PHASE_NAME': 'TOUR',
    'ANTIBLOOMING_STATE_FLAG': 'OFF',
    'BIAS_STRIP_MEAN': 7.32844,
    'COMMAND_FILE_NAME': 'trigger_24820_2.ioi',
    'COMMAND_SEQUENCE_NUMBER': 24820,
    'DARK_STRIP_MEAN': 0.300024,
    'DATA_CONVERSION_TYPE': 'TABLE',
    'DELAYED_READOUT_FLAG': 'NO',
    'DETECTOR_TEMPERATURE': -89.3184,
    'ELECTRONICS_BIAS': 112,
    'EXPECTED_MAXIMUM': [50.6578, 55.8509],
    'EXPECTED_PACKETS': 390,
    'FILTER_TEMPERATURE': -0.468354,
    'FLIGHT_SOFTWARE_VERSION_ID': '1.4',
    'SOFTWARE_VERSION_ID': 'ISS 11.00 05-03-2005',
    'IMAGE_OBSERVATION_TYPE': 'SCIENCE',
    'INSTRUMENT_DATA_RATE': 182.784,
    'INST_CMPRS_TYPE': 'LOSSLESS',
    'INST_CMPRS_PARAM': ['N/A', 'N/A', 'N/A', 'N/A'],
    'INST_CMPRS_RATE': [2.7, 1.56508],
    'INST_CMPRS_RATIO': 5.11156,
    'LIGHT_FLOOD_STATE_FLAG': 'ON',
    'MISSING_LINES': 0,
    'MISSING_PACKET_FLAG': 'NO',
    'ORDER_NUMBER': 12,
    'PARALLEL_CLOCK_VOLTAGE_INDEX': 9,
    'PRODUCT_VERSION_TYPE': 'FINAL',
    'TARGET_DESC': 'Iapetus',
    'TARGET_LIST': 'N/A',
    'TARGET_NAME': 'IAPETUS',
    'PREPARE_CYCLE_INDEX': 0,
    'READOUT_CYCLE_INDEX': 10,
    'RECEIVED_PACKETS': 231,
    'SENSOR_HEAD_ELEC_TEMPERATURE': 1.63302,
    'SEQUENCE_ID': 'S11',
    'SEQUENCE_NUMBER': 12,
    'SEQUENCE_TITLE': 'IAPETUS',
    'SHUTTER_STATE_ID': 'ENABLED',
    'TELEMETRY_FORMAT_ID': 'S&ER3',
    'VALID_MAXIMUM': [4095, 4095],
}
"""The label items every Cassini image of this tree shares.

They are N1635282917_1_CALIB's, a narrow angle frame of 2009, except for six items a 2005
image of Iapetus writes otherwise: ``MISSION_PHASE_NAME``, ``SOFTWARE_VERSION_ID``,
``SEQUENCE_ID``, ``SEQUENCE_TITLE``, ``TARGET_DESC`` and ``TARGET_NAME``.
``ANTIBLOOMING_STATE_FLAG`` is ``OFF`` rather than ``ON`` as well, so that it differs from
``LIGHT_FLOOD_STATE_FLAG``, as it does on many real labels.  ``DATA_SET_ID``,
``INSTRUMENT_HOST_NAME`` and ``MISSION_NAME`` are one value across the archive.
"""

_CAMERA_LABEL_ITEMS: dict[str, dict[str, Any]] = {
    'NAC': {
        'CALIBRATION_LAMP_STATE_FLAG': 'N/A',
        'INSTRUMENT_ID': 'ISSNA',
        'INSTRUMENT_NAME': 'IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
        'OPTICS_TEMPERATURE': [0.712693, 1.90571],
    },
    'WAC': {
        'CALIBRATION_LAMP_STATE_FLAG': 'OFF',
        'INSTRUMENT_ID': 'ISSWA',
        'INSTRUMENT_NAME': 'IMAGING SCIENCE SUBSYSTEM WIDE ANGLE',
        'OPTICS_TEMPERATURE': [6.93953, -999.0],
    },
}
"""The label items a camera decides.

The narrow angle camera has no calibration lamp and the wide angle camera no rear optics
temperature sensor, so their labels write ``N/A`` and ``-999.0`` there, and each camera
states its own ``INSTRUMENT_ID`` and ``INSTRUMENT_NAME``.  The values are
N1635282917_1_CALIB's and W1521598221_1_CALIB's.
"""

_EARTH_RECEIVED_AFTER_S = 48600.0
"""How long after its shutter closed an image of this tree began to reach Earth."""

_DOWNLINK_S = 17.5
"""How long an image of this tree took to reach Earth."""

_BUILT_AFTER_S = 62000.0
"""How long after its shutter closed an image of this tree was built on the ground."""


def _label_time(et: float) -> str:
    """Spell an epoch the way a Cassini ISS VICAR label writes a time.

    Parameters:
        et: The epoch, in TDB seconds.

    Returns:
        The UTC year, day of year and time of day to the millisecond, then ``Z``.
    """
    utc = datetime.strptime(et_to_utc(et), '%Y-%m-%dT%H:%M:%S.%f')
    return utc.strftime('%Y-%jT%H:%M:%S.%f')[:-3] + 'Z'


def _build_time(et: float) -> str:
    """Spell a build time the way a tour label writes its ``PRODUCT_CREATION_TIME``.

    Parameters:
        et: The epoch, in TDB seconds.

    Returns:
        The UTC year, day of year and time of day to the whole second, then ``.000``,
        with no ``Z``.
    """
    return _label_time(et)[: -len('.000Z')] + '.000'


def _label(
    exposure: AttitudeBaseline,
    *,
    camera: str,
    gain_mode: int,
    observation_id: str,
    description: str,
    filters: tuple[str, str],
    sampling: str,
    shutter_mode: str,
) -> dict[str, Any]:
    """Return the VICAR label items one Cassini image of this tree carries.

    What the document chooses for itself is written the way a Cassini ISS label writes
    it: its clock counts are the recorded clock strings without their partition, its
    image number is the whole seconds of its stop count, its product id is its camera
    letter and its stop count behind the clock partition, its exposure is in
    milliseconds, its gain is the text oops reads the gain state out of, and its shutter
    open, midtime and shutter close are the recorded epochs in the label's day-of-year
    text.  It reached Earth and was built on the ground after its shutter
    closed, and the label writes its build time to the whole second, as a tour label
    does (see :func:`_build_time`).
    The rest is :data:`_SHARED_LABEL_ITEMS` and the camera's :data:`_CAMERA_LABEL_ITEMS`.

    Parameters:
        exposure: The recorded exposure.
        camera: ``NAC`` or ``WAC``.
        gain_mode: The gain state oops reads out of the label's gain mode.
        observation_id: The label's observation id, which its method description names.
        description: The label's free text about the image.
        filters: The two filter wheel positions, in the label's order.
        sampling: The label's instrument mode: ``FULL``, ``SUM2`` or ``SUM4``.
        shutter_mode: The shutter mode the exposure was commanded in.

    Returns:
        The label items.
    """
    partition, _, start_count = exposure.sclk_start.partition('/')
    stop_count = exposure.sclk_stop.partition('/')[2]
    received = exposure.stop_et + _EARTH_RECEIVED_AFTER_S
    return {
        **_SHARED_LABEL_ITEMS,
        **_CAMERA_LABEL_ITEMS[camera],
        'SPACECRAFT_CLOCK_CNT_PARTITION': int(partition),
        'SPACECRAFT_CLOCK_START_COUNT': start_count,
        'SPACECRAFT_CLOCK_STOP_COUNT': stop_count,
        'IMAGE_NUMBER': int(stop_count.partition('.')[0]),
        'PRODUCT_ID': f'{partition}_{camera[0]}{stop_count}',
        'DESCRIPTION': description,
        'FILTER_NAME': list(filters),
        'INSTRUMENT_MODE_ID': sampling,
        'OBSERVATION_ID': observation_id,
        'SHUTTER_MODE_ID': shutter_mode,
        'EXPOSURE_DURATION': round(exposure.exposure_s * 1000.0, 3),
        'GAIN_MODE_ID': _GAIN_MODE_IDS[gain_mode],
        'METHOD_DESC': f'ISSPT2.5.4;Iapetus;{observation_id}_1',
        'START_TIME': _label_time(exposure.start_et),
        'IMAGE_MID_TIME': _label_time(exposure.midtime_et),
        'IMAGE_TIME': _label_time(exposure.stop_et),
        'STOP_TIME': _label_time(exposure.stop_et),
        'EARTH_RECEIVED_START_TIME': _label_time(received),
        'EARTH_RECEIVED_STOP_TIME': _label_time(received + _DOWNLINK_S),
        'PRODUCT_CREATION_TIME': _build_time(exposure.stop_et + _BUILT_AFTER_S),
    }


def cassini_public_metadata(
    result: NavResult,
    *,
    image_name: str,
    camera: str,
    image_shape: tuple[int, int],
    filters: tuple[str, str],
    sampling: str,
    gain_mode: int,
    observation_id: str,
    description: str,
    shutter_mode: str,
    image_path: Path | FCPath | None = None,
) -> dict[str, Any]:
    """Return what the Cassini ISS host publishes about one image.

    The host converts the label's start and stop clock counts to seconds of the
    clock and publishes them with their exact mean, through its own conversion,
    which is used here too.  A fixture's clock strings are counted from its label's
    reading at shutter open (see :func:`cassini_sclk_open`), so here the label's
    counts are the recorded strings without their partition.  On a real image the
    two can differ: the counts are the instrument's own, and the strings are SPICE's
    conversion of the exposure epochs.

    The metadata the host reads out of the image comes last, in a ``label_metadata``
    block under the Cassini data dictionary's own attribute names, from the label
    :func:`_label` writes for the image and from the image's file name.

    Parameters:
        result: The image's result, carrying its attitude solution.
        image_name: Basename of the source image.
        camera: ``NAC`` or ``WAC``.
        image_shape: The loaded image's ``(v, u)`` pixel dimensions.
        filters: The two filter wheel positions the label records.
        sampling: The label's instrument mode: ``FULL``, ``SUM2`` or ``SUM4``.
        gain_mode: The gain state oops reads out of the label's gain mode.
        observation_id: The label's observation id.
        description: The label's description.
        shutter_mode: The shutter mode the exposure was commanded in.
        image_path: The file the run read.  Defaults to the basename directly under
            a holdings root.

    Returns:
        The published facts, in the host's own key order.
    """
    exposure = recorded_exposure(result)
    path = image_path if image_path is not None else holdings_path(image_name)
    label = _label(
        exposure,
        camera=camera,
        gain_mode=gain_mode,
        observation_id=observation_id,
        description=description,
        filters=filters,
        sampling=sampling,
        shutter_mode=shutter_mode,
    )
    return {
        'image_path': path.as_posix(),
        'image_name': image_name,
        'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
        'instrument_lid': f'urn:nasa:pds:context:instrument:iss{camera[0].lower()}a.co',
        **published_times(exposure),
        **_published_sclk(
            exposure.sclk_start.partition('/')[2], exposure.sclk_stop.partition('/')[2]
        ),
        'image_shape_xy': (image_shape[1], image_shape[0]),
        'camera': camera,
        'exposure_time': exposure.exposure_s,
        'filters': list(filters),
        'label_metadata': _label_metadata(label, image_name),
    }
