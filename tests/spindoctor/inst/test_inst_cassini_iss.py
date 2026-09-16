"""Tests for ``spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS``."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import vicar
from filecache import FCPath
from tests.config import (
    REQUIRES_EXTERNAL_DATA,
    URL_CASSINI_ISS_CRUISE_01,
    URL_CASSINI_ISS_RHEA_01,
)
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)
from tests.spindoctor.public_metadata_cassini_iss import CASSINI_ISS_PUBLIC_METADATA

import spindoctor.obs.obs_inst_cassini_iss as obstcoiss
from spindoctor.obs.obs_inst_cassini_iss import (
    _LABEL_METADATA,
    ObsCassiniISS,
    _label_metadata,
    _sclk_count,
)

# The marker is applied per test rather than module-wide: the shutter-mode
# label tests build a bare observation and fetch nothing, so they run even
# where the external trees are absent.


def _obs_with_label(label: dict[str, Any]) -> ObsCassiniISS:
    """Build a bare ObsCassiniISS carrying only the given label dict.

    ``shutter_mode`` is a pure function of ``self.dict``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsCassiniISS)
    obs.dict = label
    return obs


def _cassini_observation(label: VicarLabelStandIn) -> ObsCassiniISS:
    """Build a bare narrow-angle observation whose public metadata can be read.

    Parameters:
        label: The image's VICAR label items.

    Returns:
        The observation.
    """
    return bare_observation(
        ObsCassiniISS,
        label,
        detector='NAC',
        filter1='CL1',
        filter2='CL2',
        sampling='FULL',
        gain_mode=2,
    )


def _is_marker(key: Any, name: str) -> bool:
    """Return whether a VICAR label key is one occurrence of a repeated marker.

    A label carrying several such sections has each marker keyed by its name and its
    occurrence number.  One carrying a single section, as the earliest cruise labels do,
    keys it by its name alone, so both forms have to be recognized.

    Parameters:
        key: One key of the label.
        name: The marker's name, such as ``PROPERTY``.

    Returns:
        True when the key is that marker, in either form.
    """
    return key == name or (isinstance(key, tuple) and key[0] == name)


def _property_block_keywords(label: Mapping[str, Any]) -> list[str]:
    """Return the keywords a VICAR label states in its property blocks.

    A label's items run in three parts: the file layout, then one group per ``PROPERTY``
    marker, then the history the calibration wrote behind its first ``TASK`` marker.  The
    property blocks' keywords are the plainly keyed items between the first marker of each
    kind, the markers themselves excluded.

    Parameters:
        label: The image's VICAR label items.

    Returns:
        The keywords, in the label's own order, or an empty list for a label that marks no
        property block at all.
    """
    keys = list(label.keys())
    properties = [i for i, key in enumerate(keys) if _is_marker(key, 'PROPERTY')]
    if not properties:
        return []
    tasks = [i for i, key in enumerate(keys) if _is_marker(key, 'TASK')]
    end = tasks[0] if tasks else len(keys)
    return [
        key
        for key in keys[properties[0] : end]
        if isinstance(key, str) and not _is_marker(key, 'PROPERTY')
    ]


def _w1573251410_label() -> VicarLabelStandIn:
    """Return the label items the host reads from W1573251410_1_CALIB (COISS_2039).

    They are the VICAR label's, as rms-vicar reads them.  The frame is a lossy wide angle
    one, so its compression parameters are numbers and its missing-line count is the text
    ``N/A`` a lossy label writes.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        ANTIBLOOMING_STATE_FLAG='ON',
        BIAS_STRIP_MEAN=22.0,
        CALIBRATION_LAMP_STATE_FLAG='OFF',
        COMMAND_FILE_NAME='trigger_7192_2.ioi',
        COMMAND_SEQUENCE_NUMBER=7192,
        DARK_STRIP_MEAN=19.5,
        DATA_CONVERSION_TYPE='TABLE',
        DATA_SET_ID='CO-S-ISSNA/ISSWA-2-EDR-V1.0',
        DELAYED_READOUT_FLAG='YES',
        DESCRIPTION='N/A',
        DETECTOR_TEMPERATURE=-87.8952,
        EARTH_RECEIVED_START_TIME='2007-313T14:43:33.041Z',
        EARTH_RECEIVED_STOP_TIME='2007-313T14:43:36.276Z',
        ELECTRONICS_BIAS=112,
        EXPECTED_MAXIMUM=[56.0227, 61.7658],
        EXPECTED_PACKETS=28,
        EXPOSURE_DURATION=25.0,
        FILTER_NAME=['CL1', 'GRN'],
        FILTER_TEMPERATURE=3.19298,
        FLIGHT_SOFTWARE_VERSION_ID='1.4',
        GAIN_MODE_ID='29 ELECTRONS PER DN',
        IMAGE_MID_TIME='2007-312T21:41:14.934Z',
        IMAGE_NUMBER=1573251410,
        IMAGE_OBSERVATION_TYPE='SCIENCE',
        IMAGE_TIME='2007-312T21:41:14.946Z',
        INSTRUMENT_DATA_RATE=182.784,
        INSTRUMENT_HOST_NAME='CASSINI ORBITER',
        INSTRUMENT_ID='ISSWA',
        INSTRUMENT_MODE_ID='FULL',
        INSTRUMENT_NAME='IMAGING SCIENCE SUBSYSTEM WIDE ANGLE',
        INST_CMPRS_PARAM=[1, 1, 41, 0],
        INST_CMPRS_RATE=[0.194248, 0.360077],
        INST_CMPRS_RATIO=22.2175,
        INST_CMPRS_TYPE='LOSSY',
        LIGHT_FLOOD_STATE_FLAG='ON',
        METHOD_DESC='ISSPT2.6.5;Saturn;ISS_052SA_STRMOVIA001_PRIME_2',
        MISSING_LINES='N/A',
        MISSING_PACKET_FLAG='NO',
        MISSION_NAME='CASSINI-HUYGENS',
        MISSION_PHASE_NAME='TOUR',
        OBSERVATION_ID='ISS_052SA_STRMOVIA001_PRIME',
        OPTICS_TEMPERATURE=[6.93953, -999.0],
        ORDER_NUMBER=2,
        PARALLEL_CLOCK_VOLTAGE_INDEX=9,
        PREPARE_CYCLE_INDEX=3,
        PRODUCT_CREATION_TIME='2007-313T17:04:33.000',
        PRODUCT_ID='1_W1573251410.122',
        PRODUCT_VERSION_TYPE='FINAL',
        READOUT_CYCLE_INDEX=15,
        RECEIVED_PACKETS=51,
        SENSOR_HEAD_ELEC_TEMPERATURE=2.98847,
        SEQUENCE_ID='S35',
        SEQUENCE_NUMBER=148,
        SEQUENCE_TITLE='ISS_052SA_STRMOVIA001_PRIME_2',
        SHUTTER_MODE_ID='BOTSIM',
        SHUTTER_STATE_ID='ENABLED',
        SOFTWARE_VERSION_ID='ISS 11.00 05-24-2006',
        SPACECRAFT_CLOCK_CNT_PARTITION=1,
        SPACECRAFT_CLOCK_START_COUNT='1573251410.115',
        SPACECRAFT_CLOCK_STOP_COUNT='1573251410.122',
        START_TIME='2007-312T21:41:14.921Z',
        STOP_TIME='2007-312T21:41:14.946Z',
        TARGET_DESC='Saturn',
        TARGET_LIST='N/A',
        TARGET_NAME='SATURN',
        TELEMETRY_FORMAT_ID='S&ER3',
        VALID_MAXIMUM=[4095, 4095],
    )


_W1573251410_METADATA: dict[str, Any] = {
    'ANTIBLOOMING_STATE_FLAG': 'ON',
    'BIAS_STRIP_MEAN': 22.0,
    'CALIBRATION_LAMP_STATE_FLAG': 'OFF',
    'COMMAND_FILE_NAME': 'trigger_7192_2.ioi',
    'COMMAND_SEQUENCE_NUMBER': 7192,
    'DARK_STRIP_MEAN': 19.5,
    'DATA_CONVERSION_TYPE': 'TABLE',
    'DATA_SET_ID': 'CO-S-ISSNA/ISSWA-2-EDR-V1.0',
    'DELAYED_READOUT_FLAG': 'YES',
    'DESCRIPTION': 'N/A',
    'DETECTOR_TEMPERATURE': -87.8952,
    'EARTH_RECEIVED_START_TIME': '2007-313T14:43:33.041Z',
    'EARTH_RECEIVED_STOP_TIME': '2007-313T14:43:36.276Z',
    'ELECTRONICS_BIAS': 112,
    'EXPECTED_MAXIMUM': [56.0227, 61.7658],
    'EXPECTED_PACKETS': 28,
    'EXPOSURE_DURATION': 25.0,
    'FILTER_NAME': ['CL1', 'GRN'],
    'FILTER_TEMPERATURE': 3.19298,
    'FLIGHT_SOFTWARE_VERSION_ID': '1.4',
    'GAIN_MODE_ID': '29 ELECTRONS PER DN',
    'IMAGE_MID_TIME': '2007-312T21:41:14.934Z',
    'IMAGE_NUMBER': 1573251410,
    'IMAGE_OBSERVATION_TYPE': 'SCIENCE',
    'IMAGE_TIME': '2007-312T21:41:14.946Z',
    'INSTRUMENT_DATA_RATE': 182.784,
    'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
    'INSTRUMENT_ID': 'ISSWA',
    'INSTRUMENT_MODE_ID': 'FULL',
    'INSTRUMENT_NAME': 'IMAGING SCIENCE SUBSYSTEM WIDE ANGLE',
    'INST_CMPRS_PARAM': [1, 1, 41, 0],
    'INST_CMPRS_RATE': [0.194248, 0.360077],
    'INST_CMPRS_RATIO': 22.2175,
    'INST_CMPRS_TYPE': 'LOSSY',
    'LIGHT_FLOOD_STATE_FLAG': 'ON',
    'METHOD_DESC': 'ISSPT2.6.5;Saturn;ISS_052SA_STRMOVIA001_PRIME_2',
    'MISSING_LINES': 'N/A',
    'MISSING_PACKET_FLAG': 'NO',
    'MISSION_NAME': 'CASSINI-HUYGENS',
    'MISSION_PHASE_NAME': 'TOUR',
    'OBSERVATION_ID': 'ISS_052SA_STRMOVIA001_PRIME',
    'OPTICS_TEMPERATURE': [6.93953, -999.0],
    'ORDER_NUMBER': 2,
    'PARALLEL_CLOCK_VOLTAGE_INDEX': 9,
    'PREPARE_CYCLE_INDEX': 3,
    'PRODUCT_CREATION_TIME': '2007-313T17:04:33.000',
    'PRODUCT_ID': '1_W1573251410.122',
    'PRODUCT_VERSION_TYPE': 'FINAL',
    'READOUT_CYCLE_INDEX': 15,
    'RECEIVED_PACKETS': 51,
    'SENSOR_HEAD_ELEC_TEMPERATURE': 2.98847,
    'SEQUENCE_ID': 'S35',
    'SEQUENCE_NUMBER': 148,
    'SEQUENCE_TITLE': 'ISS_052SA_STRMOVIA001_PRIME_2',
    'SHUTTER_MODE_ID': 'BOTSIM',
    'SHUTTER_STATE_ID': 'ENABLED',
    'SOFTWARE_VERSION_ID': 'ISS 11.00 05-24-2006',
    'SPACECRAFT_CLOCK_CNT_PARTITION': 1,
    'SPACECRAFT_CLOCK_START_COUNT': '1573251410.115',
    'SPACECRAFT_CLOCK_STOP_COUNT': '1573251410.122',
    'START_TIME': '2007-312T21:41:14.921Z',
    'STOP_TIME': '2007-312T21:41:14.946Z',
    'TARGET_DESC': 'Saturn',
    'TARGET_LIST': 'N/A',
    'TARGET_NAME': 'SATURN',
    'TELEMETRY_FORMAT_ID': 'S&ER3',
    'VALID_MAXIMUM': [4095, 4095],
}
"""The label metadata W1573251410_1_CALIB's label states, under its own keyword names."""


def _n1454725799_label() -> VicarLabelStandIn:
    """Return the label items the host reads from N1454725799_1_CALIB (COISS_2001).

    They are the VICAR label's, as rms-vicar reads them.  The frame's mission phase is
    written with an underscore, ``APPROACH_SCIENCE``, and its antiblooming was off while
    its light flood was on.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        ANTIBLOOMING_STATE_FLAG='OFF',
        BIAS_STRIP_MEAN=14.8699,
        CALIBRATION_LAMP_STATE_FLAG='N/A',
        COMMAND_FILE_NAME='OPNAV_848_3.ioi',
        COMMAND_SEQUENCE_NUMBER=8,
        DARK_STRIP_MEAN=0.0,
        DATA_CONVERSION_TYPE='12BIT',
        DATA_SET_ID='CO-S-ISSNA/ISSWA-2-EDR-V1.0',
        DELAYED_READOUT_FLAG='NO',
        DESCRIPTION='N/A',
        DETECTOR_TEMPERATURE=-89.2435,
        EARTH_RECEIVED_START_TIME='2004-039T01:35:53.622Z',
        EARTH_RECEIVED_STOP_TIME='2004-039T01:36:55.067Z',
        ELECTRONICS_BIAS=112,
        EXPECTED_MAXIMUM=[50.0, 75.0],
        EXPECTED_PACKETS=1143,
        EXPOSURE_DURATION=80.0,
        FILTER_NAME=['CL1', 'CL2'],
        FILTER_TEMPERATURE=-0.468354,
        FLIGHT_SOFTWARE_VERSION_ID='1.3',
        GAIN_MODE_ID='29 ELECTRONS PER DN',
        IMAGE_MID_TIME='2004-037T02:07:06.458Z',
        IMAGE_NUMBER=1454725799,
        IMAGE_OBSERVATION_TYPE='OPNAV',
        IMAGE_TIME='2004-037T02:07:06.498Z',
        INSTRUMENT_DATA_RATE=365.568,
        INSTRUMENT_HOST_NAME='CASSINI ORBITER',
        INSTRUMENT_ID='ISSNA',
        INSTRUMENT_MODE_ID='FULL',
        INSTRUMENT_NAME='IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
        INST_CMPRS_PARAM=['N/A', 'N/A', 'N/A', 'N/A'],
        INST_CMPRS_RATE=[6.0, 2.11688],
        INST_CMPRS_RATIO=7.55829,
        INST_CMPRS_TYPE='LOSSLESS',
        LIGHT_FLOOD_STATE_FLAG='ON',
        METHOD_DESC='OPNAV MAN.',
        MISSING_LINES=0,
        MISSING_PACKET_FLAG='NO',
        MISSION_NAME='CASSINI-HUYGENS',
        MISSION_PHASE_NAME='APPROACH_SCIENCE',
        OBSERVATION_ID='NAV_C42SK_OPNAV371_PRIME',
        OPTICS_TEMPERATURE=[0.712693, 1.82047],
        ORDER_NUMBER=0,
        PARALLEL_CLOCK_VOLTAGE_INDEX=9,
        PREPARE_CYCLE_INDEX=3,
        PRODUCT_CREATION_TIME='2004-038T19:26:35.000Z',
        PRODUCT_ID='1_N1454725799.122',
        PRODUCT_VERSION_TYPE='FINAL',
        READOUT_CYCLE_INDEX=5,
        RECEIVED_PACKETS=309,
        SENSOR_HEAD_ELEC_TEMPERATURE=1.63302,
        SEQUENCE_ID='C42',
        SEQUENCE_NUMBER=1,
        SEQUENCE_TITLE='--',
        SHUTTER_MODE_ID='NACONLY',
        SHUTTER_STATE_ID='ENABLED',
        SOFTWARE_VERSION_ID='ISS 9.00 05-22-2003',
        SPACECRAFT_CLOCK_CNT_PARTITION=1,
        SPACECRAFT_CLOCK_START_COUNT='1454725799.102',
        SPACECRAFT_CLOCK_STOP_COUNT='1454725799.122',
        START_TIME='2004-037T02:07:06.418Z',
        STOP_TIME='2004-037T02:07:06.498Z',
        TARGET_DESC='RHEA',
        TARGET_LIST='N/A',
        TARGET_NAME='SKY',
        TELEMETRY_FORMAT_ID='UNK',
        VALID_MAXIMUM=[4095, 4095],
    )


_N1454725799_METADATA: dict[str, Any] = {
    'ANTIBLOOMING_STATE_FLAG': 'OFF',
    'BIAS_STRIP_MEAN': 14.8699,
    'CALIBRATION_LAMP_STATE_FLAG': 'N/A',
    'COMMAND_FILE_NAME': 'OPNAV_848_3.ioi',
    'COMMAND_SEQUENCE_NUMBER': 8,
    'DARK_STRIP_MEAN': 0.0,
    'DATA_CONVERSION_TYPE': '12BIT',
    'DATA_SET_ID': 'CO-S-ISSNA/ISSWA-2-EDR-V1.0',
    'DELAYED_READOUT_FLAG': 'NO',
    'DESCRIPTION': 'N/A',
    'DETECTOR_TEMPERATURE': -89.2435,
    'EARTH_RECEIVED_START_TIME': '2004-039T01:35:53.622Z',
    'EARTH_RECEIVED_STOP_TIME': '2004-039T01:36:55.067Z',
    'ELECTRONICS_BIAS': 112,
    'EXPECTED_MAXIMUM': [50.0, 75.0],
    'EXPECTED_PACKETS': 1143,
    'EXPOSURE_DURATION': 80.0,
    'FILTER_NAME': ['CL1', 'CL2'],
    'FILTER_TEMPERATURE': -0.468354,
    'FLIGHT_SOFTWARE_VERSION_ID': '1.3',
    'GAIN_MODE_ID': '29 ELECTRONS PER DN',
    'IMAGE_MID_TIME': '2004-037T02:07:06.458Z',
    'IMAGE_NUMBER': 1454725799,
    'IMAGE_OBSERVATION_TYPE': 'OPNAV',
    'IMAGE_TIME': '2004-037T02:07:06.498Z',
    'INSTRUMENT_DATA_RATE': 365.568,
    'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
    'INSTRUMENT_ID': 'ISSNA',
    'INSTRUMENT_MODE_ID': 'FULL',
    'INSTRUMENT_NAME': 'IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
    'INST_CMPRS_PARAM': ['N/A', 'N/A', 'N/A', 'N/A'],
    'INST_CMPRS_RATE': [6.0, 2.11688],
    'INST_CMPRS_RATIO': 7.55829,
    'INST_CMPRS_TYPE': 'LOSSLESS',
    'LIGHT_FLOOD_STATE_FLAG': 'ON',
    'METHOD_DESC': 'OPNAV MAN.',
    'MISSING_LINES': 0,
    'MISSING_PACKET_FLAG': 'NO',
    'MISSION_NAME': 'CASSINI-HUYGENS',
    'MISSION_PHASE_NAME': 'APPROACH_SCIENCE',
    'OBSERVATION_ID': 'NAV_C42SK_OPNAV371_PRIME',
    'OPTICS_TEMPERATURE': [0.712693, 1.82047],
    'ORDER_NUMBER': 0,
    'PARALLEL_CLOCK_VOLTAGE_INDEX': 9,
    'PREPARE_CYCLE_INDEX': 3,
    'PRODUCT_CREATION_TIME': '2004-038T19:26:35.000Z',
    'PRODUCT_ID': '1_N1454725799.122',
    'PRODUCT_VERSION_TYPE': 'FINAL',
    'READOUT_CYCLE_INDEX': 5,
    'RECEIVED_PACKETS': 309,
    'SENSOR_HEAD_ELEC_TEMPERATURE': 1.63302,
    'SEQUENCE_ID': 'C42',
    'SEQUENCE_NUMBER': 1,
    'SEQUENCE_TITLE': '--',
    'SHUTTER_MODE_ID': 'NACONLY',
    'SHUTTER_STATE_ID': 'ENABLED',
    'SOFTWARE_VERSION_ID': 'ISS 9.00 05-22-2003',
    'SPACECRAFT_CLOCK_CNT_PARTITION': 1,
    'SPACECRAFT_CLOCK_START_COUNT': '1454725799.102',
    'SPACECRAFT_CLOCK_STOP_COUNT': '1454725799.122',
    'START_TIME': '2004-037T02:07:06.418Z',
    'STOP_TIME': '2004-037T02:07:06.498Z',
    'TARGET_DESC': 'RHEA',
    'TARGET_LIST': 'N/A',
    'TARGET_NAME': 'SKY',
    'TELEMETRY_FORMAT_ID': 'UNK',
    'VALID_MAXIMUM': [4095, 4095],
}
"""The label metadata N1454725799_1_CALIB's label states, under its own keyword names."""


def _n1737255524_label() -> VicarLabelStandIn:
    """Return the label items the host reads from N1737255524_1_CALIB (COISS_2080).

    They are the VICAR label's, as rms-vicar reads them.  The exposure spans a second: its
    counts run from 1737255523.232 to 1737255524.122, and its image number is the stop
    count's seconds.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        ANTIBLOOMING_STATE_FLAG='OFF',
        BIAS_STRIP_MEAN=5.66667,
        CALIBRATION_LAMP_STATE_FLAG='N/A',
        COMMAND_FILE_NAME='trigger_31305_1.ioi',
        COMMAND_SEQUENCE_NUMBER=31305,
        DARK_STRIP_MEAN=2.625,
        DATA_CONVERSION_TYPE='TABLE',
        DATA_SET_ID='CO-S-ISSNA/ISSWA-2-EDR-V1.0',
        DELAYED_READOUT_FLAG='NO',
        DESCRIPTION='N/A',
        DETECTOR_TEMPERATURE=-89.3184,
        EARTH_RECEIVED_START_TIME='2013-019T16:30:55.609Z',
        EARTH_RECEIVED_STOP_TIME='2013-019T16:31:06.169Z',
        ELECTRONICS_BIAS=112,
        EXPECTED_MAXIMUM=[58.3402, 64.3209],
        EXPECTED_PACKETS=674,
        EXPOSURE_DURATION=560.0,
        FILTER_NAME=['CL1', 'CL2'],
        FILTER_TEMPERATURE=-0.468354,
        FLIGHT_SOFTWARE_VERSION_ID='1.4',
        GAIN_MODE_ID='29 ELECTRONS PER DN',
        IMAGE_MID_TIME='2013-019T02:04:35.696Z',
        IMAGE_NUMBER=1737255524,
        IMAGE_OBSERVATION_TYPE='SCIENCE',
        IMAGE_TIME='2013-019T02:04:35.976Z',
        INSTRUMENT_DATA_RATE=182.784,
        INSTRUMENT_HOST_NAME='CASSINI ORBITER',
        INSTRUMENT_ID='ISSNA',
        INSTRUMENT_MODE_ID='FULL',
        INSTRUMENT_NAME='IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
        INST_CMPRS_PARAM=[0, 0, 1, 0],
        INST_CMPRS_RATE=[4.7, 0.981995],
        INST_CMPRS_RATIO=8.14668,
        INST_CMPRS_TYPE='LOSSY',
        LIGHT_FLOOD_STATE_FLAG='ON',
        METHOD_DESC='ISSPT2.8;Saturn-Rings;ISS_179RI_MOONLETC001_PIE_1',
        MISSING_LINES='N/A',
        MISSING_PACKET_FLAG='NO',
        MISSION_NAME='CASSINI-HUYGENS',
        MISSION_PHASE_NAME='EXTENDED-EXTENDED MISSION',
        OBSERVATION_ID='ISS_179RI_MOONLETC001_PIE',
        OPTICS_TEMPERATURE=[0.712693, 1.90571],
        ORDER_NUMBER=1,
        PARALLEL_CLOCK_VOLTAGE_INDEX=9,
        PREPARE_CYCLE_INDEX=3,
        PRODUCT_CREATION_TIME='2013-039T12:10:08.000',
        PRODUCT_ID='1_N1737255524.122',
        PRODUCT_VERSION_TYPE='FINAL',
        READOUT_CYCLE_INDEX=5,
        RECEIVED_PACKETS=140,
        SENSOR_HEAD_ELEC_TEMPERATURE=1.63302,
        SEQUENCE_ID='S77',
        SEQUENCE_NUMBER=108,
        SEQUENCE_TITLE='--',
        SHUTTER_MODE_ID='NACONLY',
        SHUTTER_STATE_ID='ENABLED',
        SOFTWARE_VERSION_ID='ISS 11.00 05-24-2006',
        SPACECRAFT_CLOCK_CNT_PARTITION=1,
        SPACECRAFT_CLOCK_START_COUNT='1737255523.232',
        SPACECRAFT_CLOCK_STOP_COUNT='1737255524.122',
        START_TIME='2013-019T02:04:35.416Z',
        STOP_TIME='2013-019T02:04:35.976Z',
        TARGET_DESC='Saturn-Rings',
        TARGET_LIST='N/A',
        TARGET_NAME='SATURN',
        TELEMETRY_FORMAT_ID='S&ER3',
        VALID_MAXIMUM=[4095, 4095],
    )


_N1737255524_METADATA: dict[str, Any] = {
    'ANTIBLOOMING_STATE_FLAG': 'OFF',
    'BIAS_STRIP_MEAN': 5.66667,
    'CALIBRATION_LAMP_STATE_FLAG': 'N/A',
    'COMMAND_FILE_NAME': 'trigger_31305_1.ioi',
    'COMMAND_SEQUENCE_NUMBER': 31305,
    'DARK_STRIP_MEAN': 2.625,
    'DATA_CONVERSION_TYPE': 'TABLE',
    'DATA_SET_ID': 'CO-S-ISSNA/ISSWA-2-EDR-V1.0',
    'DELAYED_READOUT_FLAG': 'NO',
    'DESCRIPTION': 'N/A',
    'DETECTOR_TEMPERATURE': -89.3184,
    'EARTH_RECEIVED_START_TIME': '2013-019T16:30:55.609Z',
    'EARTH_RECEIVED_STOP_TIME': '2013-019T16:31:06.169Z',
    'ELECTRONICS_BIAS': 112,
    'EXPECTED_MAXIMUM': [58.3402, 64.3209],
    'EXPECTED_PACKETS': 674,
    'EXPOSURE_DURATION': 560.0,
    'FILTER_NAME': ['CL1', 'CL2'],
    'FILTER_TEMPERATURE': -0.468354,
    'FLIGHT_SOFTWARE_VERSION_ID': '1.4',
    'GAIN_MODE_ID': '29 ELECTRONS PER DN',
    'IMAGE_MID_TIME': '2013-019T02:04:35.696Z',
    'IMAGE_NUMBER': 1737255524,
    'IMAGE_OBSERVATION_TYPE': 'SCIENCE',
    'IMAGE_TIME': '2013-019T02:04:35.976Z',
    'INSTRUMENT_DATA_RATE': 182.784,
    'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
    'INSTRUMENT_ID': 'ISSNA',
    'INSTRUMENT_MODE_ID': 'FULL',
    'INSTRUMENT_NAME': 'IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
    'INST_CMPRS_PARAM': [0, 0, 1, 0],
    'INST_CMPRS_RATE': [4.7, 0.981995],
    'INST_CMPRS_RATIO': 8.14668,
    'INST_CMPRS_TYPE': 'LOSSY',
    'LIGHT_FLOOD_STATE_FLAG': 'ON',
    'METHOD_DESC': 'ISSPT2.8;Saturn-Rings;ISS_179RI_MOONLETC001_PIE_1',
    'MISSING_LINES': 'N/A',
    'MISSING_PACKET_FLAG': 'NO',
    'MISSION_NAME': 'CASSINI-HUYGENS',
    'MISSION_PHASE_NAME': 'EXTENDED-EXTENDED MISSION',
    'OBSERVATION_ID': 'ISS_179RI_MOONLETC001_PIE',
    'OPTICS_TEMPERATURE': [0.712693, 1.90571],
    'ORDER_NUMBER': 1,
    'PARALLEL_CLOCK_VOLTAGE_INDEX': 9,
    'PREPARE_CYCLE_INDEX': 3,
    'PRODUCT_CREATION_TIME': '2013-039T12:10:08.000',
    'PRODUCT_ID': '1_N1737255524.122',
    'PRODUCT_VERSION_TYPE': 'FINAL',
    'READOUT_CYCLE_INDEX': 5,
    'RECEIVED_PACKETS': 140,
    'SENSOR_HEAD_ELEC_TEMPERATURE': 1.63302,
    'SEQUENCE_ID': 'S77',
    'SEQUENCE_NUMBER': 108,
    'SEQUENCE_TITLE': '--',
    'SHUTTER_MODE_ID': 'NACONLY',
    'SHUTTER_STATE_ID': 'ENABLED',
    'SOFTWARE_VERSION_ID': 'ISS 11.00 05-24-2006',
    'SPACECRAFT_CLOCK_CNT_PARTITION': 1,
    'SPACECRAFT_CLOCK_START_COUNT': '1737255523.232',
    'SPACECRAFT_CLOCK_STOP_COUNT': '1737255524.122',
    'START_TIME': '2013-019T02:04:35.416Z',
    'STOP_TIME': '2013-019T02:04:35.976Z',
    'TARGET_DESC': 'Saturn-Rings',
    'TARGET_LIST': 'N/A',
    'TARGET_NAME': 'SATURN',
    'TELEMETRY_FORMAT_ID': 'S&ER3',
    'VALID_MAXIMUM': [4095, 4095],
}
"""The label metadata N1737255524_1_CALIB's label states, under its own keyword names."""


def _wide_angle_observation(label: VicarLabelStandIn) -> ObsCassiniISS:
    """Build a bare wide angle observation, taken as W1573251410_1_CALIB was.

    Parameters:
        label: The image's VICAR label items.

    Returns:
        The observation.
    """
    return bare_observation(
        ObsCassiniISS,
        label,
        detector='WAC',
        filter1='CL1',
        filter2='GRN',
        sampling='FULL',
        gain_mode=2,
    )


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_basic() -> None:
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.midtime == 196177280.54761


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_calib_filename_selects_calib_inst_config() -> None:
    """A ``_CALIB.IMG`` filename selects the calibrated_if config block.

    Regression: CALIB I/F products were previously loaded with the raw_dn
    config block, causing the image-quality classifier to flag every
    CALIB image as ``blank`` (max I/F < 1.0 against the 5.0 DN floor).
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.inst_config is not None
    assert obs.inst_config['data_units'] == 'calibrated_if'
    # Calibrated_if blocks expose the I/F-keyed thresholds, not DN-keyed
    # ones.  Saturation is intentionally NOT keyed in I/F (Phase 10 §F):
    # calibration is exposure-/filter-/gain-dependent, so a single I/F
    # threshold cannot identify physically saturated pixels.  The
    # orchestrator leaves the per-pixel saturation mask empty for
    # calibrated_if input.
    iqt = obs.inst_config['image_quality_thresholds']
    assert 'saturation_threshold_if' not in iqt
    assert 'blank_max_if' in iqt
    assert 'noisy_threshold_if' in iqt


@REQUIRES_EXTERNAL_DATA
def test_cassini_iss_reports_shutter_mode() -> None:
    """The shutter mode is read from the image label.

    The Rhea test frame was taken with both cameras exposed at once, so it
    reports the simultaneous mode rather than a single-camera one.
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert obs.shutter_mode == 'BOTSIM'


def test_shutter_mode_absent_from_the_label_reads_as_none() -> None:
    """A label carrying no SHUTTER_MODE_ID reports no shutter mode."""
    assert _obs_with_label({}).shutter_mode is None


def test_shutter_mode_null_label_value_reads_as_none() -> None:
    """A SHUTTER_MODE_ID present but null reports no shutter mode, not 'None'."""
    assert _obs_with_label({'SHUTTER_MODE_ID': None}).shutter_mode is None


def test_shutter_mode_non_text_label_value_is_refused() -> None:
    """A non-string SHUTTER_MODE_ID raises rather than serializing the object.

    ``str()`` would render any object without complaint, and the result would
    pass downstream as a legible shutter mode.
    """
    with pytest.raises(ValueError, match='SHUTTER_MODE_ID is not text'):
        _ = _obs_with_label({'SHUTTER_MODE_ID': 42}).shutter_mode


def test_the_published_counts_are_fractional_seconds_and_their_exact_mean() -> None:
    """The label's counts are published as clock seconds, and their mean as the midtime.

    N1459552248_1_CALIB's exposure crosses a second: its label counts, 1459552247.012
    and 1459552248.137, are 1459552247 + 12/256 and 1459552248 + 137/256 seconds, and
    the midtime count is exactly halfway between them.  Each count is a whole number of
    1/512-second steps, which a float holds exactly, so each is compared exactly.
    """
    label = VicarLabelStandIn(
        SPACECRAFT_CLOCK_START_COUNT='1459552247.012',
        SPACECRAFT_CLOCK_STOP_COUNT='1459552248.137',
    )
    start, midtime, end = published_clock_counts(_cassini_observation(label))
    assert start == 1459552247.046875
    assert midtime == 1459552247.791015625
    assert end == 1459552248.53515625


def test_a_tick_field_that_lost_its_trailing_zeros_is_padded_back() -> None:
    """A count whose tick field lost its trailing zeros is read with them restored.

    The COISS index writes N1347929382_3's start count, 1347929382.110, as
    1347929382.11.
    """
    assert _sclk_count('1347929382.11') == 1347929382 + 110 / 256


def test_a_label_without_clock_counts_publishes_null_counts() -> None:
    """A label carrying no clock counts publishes all three as null.

    Some labels, W1294561143_1_CALIB's among them, carry neither
    SPACECRAFT_CLOCK_START_COUNT nor SPACECRAFT_CLOCK_STOP_COUNT.
    """
    start, midtime, end = published_clock_counts(_cassini_observation(VicarLabelStandIn()))
    assert start is None
    assert midtime is None
    assert end is None


@pytest.mark.parametrize(
    ('build', 'metadata'),
    [
        pytest.param(
            lambda: _wide_angle_observation(_w1573251410_label()),
            _W1573251410_METADATA,
            id='W1573251410_1_CALIB',
        ),
        pytest.param(
            lambda: _cassini_observation(_n1454725799_label()),
            _N1454725799_METADATA,
            id='N1454725799_1_CALIB',
        ),
        pytest.param(
            lambda: _cassini_observation(_n1737255524_label()),
            _N1737255524_METADATA,
            id='N1737255524_1_CALIB',
        ),
    ],
)
def test_the_label_metadata_is_published_as_the_label_states_it(
    build: Callable[[], ObsCassiniISS], metadata: dict[str, Any]
) -> None:
    """Each keyword is published under its own name, with the value the label states.

    W1573251410_1_CALIB's numbers stay numbers, its times are its own day-of-year
    spellings with their trailing Z, its gain mode is its text, its missing-line count is
    the text N/A, and each of its sequence keywords stays the sequence the label states.
    N1454725799_1_CALIB's mission phase keeps its underscore, and its antiblooming flag,
    OFF, is published apart from its light flood flag, ON.  N1737255524_1_CALIB's exposure
    spans a second, so its image number, the stop count's seconds, is not its start
    count's.

    Parameters:
        build: Builds the observation carrying the image's label.
        metadata: The label metadata that label states.
    """
    public = build().get_public_metadata()
    assert {key: public[key] for key in metadata} == metadata


def test_the_metadata_fixture_holds_the_keys_the_host_publishes() -> None:
    """The metadata chapter's Cassini fixture holds the host's keys, in the host's order.

    The chapter's staleness guard takes the Cassini metadata from that fixture, so a
    keyword the host gains has to reach the fixture, and through it the chapter, rather
    than pass unexamined.
    """
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert list(public) == list(CASSINI_ISS_PUBLIC_METADATA)


def test_a_keyword_the_label_lacks_is_published_as_null() -> None:
    """A label carrying none of the keywords publishes every one of them as null."""
    public = _cassini_observation(VicarLabelStandIn()).get_public_metadata()
    assert {key: public[key] for key in _W1573251410_METADATA} == dict.fromkeys(
        _W1573251410_METADATA
    )


@REQUIRES_EXTERNAL_DATA
def test_a_real_image_publishes_its_vicar_label_metadata() -> None:
    """The label metadata is read from the VICAR label inside the calibrated image file.

    That label writes its times with a trailing Z, keeps a sequence keyword's elements in
    one sequence, and gives a wide angle frame's back optics temperature as -999.0, since
    the wide angle camera has no rear optics sensor.
    """
    public = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01).get_public_metadata()
    assert {
        key: public[key]
        for key in (
            'IMAGE_TIME',
            'GAIN_MODE_ID',
            'EXPOSURE_DURATION',
            'FILTER_NAME',
            'OPTICS_TEMPERATURE',
            'INST_CMPRS_PARAM',
        )
    } == {
        'IMAGE_TIME': '2006-080T01:40:16.112Z',
        'GAIN_MODE_ID': '29 ELECTRONS PER DN',
        'EXPOSURE_DURATION': 1500.0,
        'FILTER_NAME': ['CL1', 'VIO'],
        'OPTICS_TEMPERATURE': [6.93953, -999.0],
        'INST_CMPRS_PARAM': ['N/A', 'N/A', 'N/A', 'N/A'],
    }


def test_a_label_marking_no_property_block_reports_no_keywords() -> None:
    """A label that marks no property block reports none rather than raising.

    The walk reads from the first property marker, and a label with none has no section
    to read; taking the first of no markers would raise instead of reporting.
    """
    assert _property_block_keywords({'LBLSIZE': 1024, 'FORMAT': 'BYTE'}) == []


def test_a_plain_property_marker_delimits_the_block_without_the_marker() -> None:
    """A marker written as an ordinary keyword names the section without being in it.

    The earliest cruise labels write ``PROPERTY`` as a plain keyword naming the section
    rather than in the numbered form.  A walk that looked only for the numbered form would
    report no keywords at all on such a label, and one that kept the marker would report
    it as a keyword the label states.
    """
    label = {
        'LBLSIZE': 1024,
        'PROPERTY': 'CASSINI-ISS2',
        'FILTER1_NAME': 'CL1',
        'TASK': 'CISSCAL',
        'GAIN_CORRECTION': 1.0,
    }
    assert _property_block_keywords(label) == ['FILTER1_NAME']


@REQUIRES_EXTERNAL_DATA
def test_a_tour_era_label_states_exactly_the_published_keywords() -> None:
    """A tour-era label's property blocks hold exactly the keywords the host publishes.

    The published list is the archive's PDS3 keyword set, written out rather than read
    from the label, so a tour-era label carrying a keyword the list omits would drop that
    keyword in silence.  This is what says so instead.  An earlier label states fewer of
    them, which is the case below.
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    assert sorted(_property_block_keywords(obs.dict)) == sorted(_LABEL_METADATA)


@REQUIRES_EXTERNAL_DATA
def test_a_cruise_era_label_publishes_what_it_states_and_nulls_the_rest() -> None:
    """An earlier label states fewer of the keywords, and items of its own besides.

    N1294562651_1_CALIB, of the earliest cruise volume, marks its property section with a
    plain ``PROPERTY`` keyword rather than the numbered form, states only some of the
    published keywords, and carries items the archive's label does not state, among them
    differently named equivalents such as ``FILTER1_NAME`` and ``SENSOR_HEAD_ELEC_TEMP``.
    Its label is read directly rather than through the host, because a cruise epoch has no
    camera frame in the local kernel set and loading the image would raise.
    """
    path = cast(Path, FCPath(URL_CASSINI_ISS_CRUISE_01).retrieve())
    label = vicar.VicarImage.from_file(path, strict=False).label
    section = _property_block_keywords(label)
    published = _label_metadata(label)
    stated = [keyword for keyword in _LABEL_METADATA if keyword in section]
    absent = [keyword for keyword in _LABEL_METADATA if keyword not in section]
    outside = [keyword for keyword in section if keyword not in _LABEL_METADATA]
    assert absent != [], 'this label is expected to state fewer than the published keywords'
    assert outside != [], 'this label is expected to carry items outside the published list'
    assert {key: published[key] for key in absent} == dict.fromkeys(absent)
    assert [key for key in outside if key in published] == []
    assert {key: published[key] for key in stated} == {key: label.get(key, None) for key in stated}
