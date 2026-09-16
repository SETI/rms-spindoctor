"""Tests for ``spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS``."""

from collections.abc import Mapping
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

_UNREAD_KEYWORDS = (
    'DATA_SET_ID',
    'INSTRUMENT_HOST_NAME',
    'INSTRUMENT_ID',
    'INSTRUMENT_NAME',
    'MISSION_NAME',
    'PRODUCT_ID',
)
"""The label keywords no published attribute holds.

Each states the archive, the spacecraft, the camera or the raw product rather than the
exposure, and the Cassini data dictionary gives the same facts elsewhere in a label: the
camera in the instrument LID the block already states, the rest in the bundle's own
identification.  A tour-era label states these six beyond the keywords the attributes
read, and nothing else.
"""


def _obs_with_label(label: dict[str, Any]) -> ObsCassiniISS:
    """Build a bare ObsCassiniISS carrying only the given label dict.

    ``shutter_mode`` is a pure function of ``self.dict``, so a fully
    constructed observation (and an external image fetch) is unnecessary for
    testing it.
    """
    obs = object.__new__(ObsCassiniISS)
    obs.dict = label
    return obs


def _cassini_observation(
    label: VicarLabelStandIn, *, image_name: str = 'image_0001.img'
) -> ObsCassiniISS:
    """Build a bare narrow-angle observation whose public metadata can be read.

    Parameters:
        label: The image's VICAR label items.
        image_name: Basename of the image file, which is where the version number is
            stated.  The default states none.

    Returns:
        The observation.
    """
    return bare_observation(
        ObsCassiniISS,
        label,
        detector='NAC',
        filter1='CL1',
        filter2='CL2',
        abspath=Path('/cache') / image_name,
    )


def _wide_angle_observation(label: VicarLabelStandIn) -> ObsCassiniISS:
    """Build a bare wide angle observation, taken and named as W1573251410_1_CALIB was.

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
        abspath=Path('/cache/W1573251410_1_CALIB.IMG'),
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


def _read_keywords() -> set[str]:
    """Return every label keyword a published attribute reads."""
    return {keyword for _attribute, keyword, _element in _LABEL_METADATA if keyword is not None}


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
    'cassini:mission_phase_name': 'TOUR',
    'cassini:spacecraft_clock_count_partition': 1,
    'cassini:spacecraft_clock_start_count': '1573251410.115',
    'cassini:spacecraft_clock_stop_count': '1573251410.122',
    'cassini:limitations': 'N/A',
    'cassini:antiblooming_state_flag': 'ON',
    'cassini:bias_strip_mean': 22.0,
    'cassini:calibration_lamp_state_flag': 'OFF',
    'cassini:command_file_name': 'trigger_7192_2.ioi',
    'cassini:command_sequence_number': 7192,
    'cassini:dark_strip_mean': 19.5,
    'cassini:data_conversion_type': 'TABLE',
    'cassini:delayed_readout_flag': 'YES',
    'cassini:detector_temperature': -87.8952,
    'cassini:electronics_bias': 112,
    'cassini:earth_received_start_time': '2007-313T14:43:33.041Z',
    'cassini:earth_received_stop_time': '2007-313T14:43:36.276Z',
    'cassini:expected_maximum_full_well': 56.0227,
    'cassini:expected_maximum_DN_sat': 61.7658,
    'cassini:expected_packets': 28,
    'cassini:exposure_duration': 25.0,
    'cassini:filter_name_1': 'CL1',
    'cassini:filter_name_2': 'GRN',
    'cassini:filter_temperature': 3.19298,
    'cassini:flight_software_version_id': '1.4',
    'cassini:gain_mode_id': '29 ELECTRONS PER DN',
    'cassini:ground_software_version_id': 'ISS 11.00 05-24-2006',
    'cassini:image_mid_time': '2007-312T21:41:14.934Z',
    'cassini:image_number': 1573251410,
    'cassini:image_time': '2007-312T21:41:14.946Z',
    'cassini:image_observation_type': 'SCIENCE',
    'cassini:instrument_data_rate': 182.784,
    'cassini:instrument_mode_id': 'FULL',
    'cassini:inst_cmprs_type': 'LOSSY',
    'cassini:inst_cmprs_param_malgo': 1,
    'cassini:inst_cmprs_param_tb': 1,
    'cassini:inst_cmprs_param_blocks': 41,
    'cassini:inst_cmprs_param_quant': 0,
    'cassini:inst_cmprs_rate_expected_bits': 0.194248,
    'cassini:inst_cmprs_rate_actual_bits': 0.360077,
    'cassini:inst_cmprs_ratio': 22.2175,
    'cassini:light_flood_state_flag': 'ON',
    'cassini:method_description': 'ISSPT2.6.5;Saturn;ISS_052SA_STRMOVIA001_PRIME_2',
    'cassini:missing_lines': 'N/A',
    'cassini:missing_packet_flag': 'NO',
    'cassini:observation_id': 'ISS_052SA_STRMOVIA001_PRIME',
    'cassini:optics_temperature_front': 6.93953,
    'cassini:optics_temperature_back': -999.0,
    'cassini:order_number': 2,
    'cassini:parallel_clock_voltage_index': 9,
    'cassini:pds3_product_creation_time': '2007-313T17:04:33.000',
    'cassini:pds3_product_version_type': 'FINAL',
    'cassini:pds3_target_desc': 'Saturn',
    'cassini:pds3_target_list': 'N/A',
    'cassini:pds3_target_name': 'SATURN',
    'cassini:pre-pds_version_number': 1,
    'cassini:prepare_cycle_index': 3,
    'cassini:readout_cycle_index': 15,
    'cassini:received_packets': 51,
    'cassini:sensor_head_electronics_temperature': 2.98847,
    'cassini:sequence_id': 'S35',
    'cassini:sequence_number': 148,
    'cassini:sequence_title': 'ISS_052SA_STRMOVIA001_PRIME_2',
    'cassini:shutter_mode_id': 'BOTSIM',
    'cassini:shutter_state_id': 'ENABLED',
    'cassini:start_time_doy': '2007-312T21:41:14.921Z',
    'cassini:stop_time_doy': '2007-312T21:41:14.946Z',
    'cassini:telemetry_format_id': 'S&ER3',
    'cassini:valid_maximum_full_well': 4095,
    'cassini:valid_maximum_DN_sat': 4095,
}
"""What W1573251410_1_CALIB publishes: its label's values, under the dictionary's names.

The frame's file name states version 1, and its label states every keyword the attributes
read.
"""


def _n1454725799_label() -> VicarLabelStandIn:
    """Return the label items the host reads from N1454725799_1_CALIB (COISS_2001).

    They are the VICAR label's, as rms-vicar reads them.  The frame's mission phase is
    written with an underscore, ``APPROACH_SCIENCE``, and its antiblooming was off while
    its light flood was on.  Its product creation time ends in ``Z``, which the archive's
    Pacific local time is written with on some labels and not on others.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        ANTIBLOOMING_STATE_FLAG='OFF',
        LIGHT_FLOOD_STATE_FLAG='ON',
        MISSION_PHASE_NAME='APPROACH_SCIENCE',
        PRODUCT_CREATION_TIME='2004-038T19:26:35.000Z',
        SEQUENCE_TITLE='--',
        SPACECRAFT_CLOCK_START_COUNT='1454725799.102',
        SPACECRAFT_CLOCK_STOP_COUNT='1454725799.122',
        IMAGE_NUMBER=1454725799,
        TELEMETRY_FORMAT_ID='UNK',
    )


def _n1737255524_label() -> VicarLabelStandIn:
    """Return the label items the host reads from N1737255524_1_CALIB (COISS_2080).

    They are the VICAR label's, as rms-vicar reads them.  The exposure spans a second: its
    counts run from 1737255523.232 to 1737255524.122, and its image number is the stop
    count's seconds.  The frame was lossy compressed, so its compression parameters are
    the numbers a lossy label states rather than the ``N/A`` of a label with none.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        IMAGE_NUMBER=1737255524,
        INST_CMPRS_PARAM=[0, 0, 1, 0],
        MISSION_PHASE_NAME='EXTENDED-EXTENDED MISSION',
        SPACECRAFT_CLOCK_START_COUNT='1737255523.232',
        SPACECRAFT_CLOCK_STOP_COUNT='1737255524.122',
    )


def _sequence_label() -> VicarLabelStandIn:
    """Return a label whose sequence keywords state a different value in each position.

    A real label repeats a value inside a sequence -- W1573251410's compression
    parameters are ``[1, 1, 41, 0]`` -- which would hide a swap of two positions holding
    the same value.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        EXPECTED_MAXIMUM=[11.0, 12.0],
        FILTER_NAME=['F1', 'F2'],
        INST_CMPRS_PARAM=[21, 22, 23, 24],
        INST_CMPRS_RATE=[31.0, 32.0],
        OPTICS_TEMPERATURE=[41.0, 42.0],
        VALID_MAXIMUM=[51, 52],
    )


_SEQUENCE_ELEMENTS: tuple[tuple[str, Any], ...] = (
    ('cassini:expected_maximum_full_well', 11.0),
    ('cassini:expected_maximum_DN_sat', 12.0),
    ('cassini:filter_name_1', 'F1'),
    ('cassini:filter_name_2', 'F2'),
    ('cassini:inst_cmprs_param_malgo', 21),
    ('cassini:inst_cmprs_param_tb', 22),
    ('cassini:inst_cmprs_param_blocks', 23),
    ('cassini:inst_cmprs_param_quant', 24),
    ('cassini:inst_cmprs_rate_expected_bits', 31.0),
    ('cassini:inst_cmprs_rate_actual_bits', 32.0),
    ('cassini:optics_temperature_front', 41.0),
    ('cassini:optics_temperature_back', 42.0),
    ('cassini:valid_maximum_full_well', 51),
    ('cassini:valid_maximum_DN_sat', 52),
)
"""Each element of :func:`_sequence_label`'s sequences, and the attribute holding it."""


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


def test_every_attribute_is_published_with_the_value_its_keyword_states() -> None:
    """A label stating every keyword publishes all of the attributes, with its values.

    W1573251410_1_CALIB's numbers stay numbers, its times are its own day-of-year
    spellings with their trailing Z, its gain mode is its text, its missing-line count is
    the text N/A, and each of its sequence keywords is split into the attributes its
    elements hold.
    """
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert public['label_metadata'] == _W1573251410_METADATA


def test_a_label_stating_none_of_the_keywords_publishes_every_attribute_as_null() -> None:
    """A label carrying none of them publishes the same attributes, each null.

    That is what makes the block one shape for every image: a reader finds the attribute
    whether or not the image states it, and a null says the image does not state it.
    """
    block = _cassini_observation(VicarLabelStandIn()).get_public_metadata()['label_metadata']
    assert list(block) == list(_W1573251410_METADATA)
    assert set(block.values()) == {None}


@pytest.mark.parametrize(('attribute', 'value'), _SEQUENCE_ELEMENTS)
def test_each_sequence_keyword_lands_in_its_elements_by_position(
    attribute: str, value: Any
) -> None:
    """Each element of a sequence keyword reaches the attribute its position names.

    The label's own order is what is followed.  The COISS index table writes the four
    compression parameters in another order -- blocks per group, algorithm, quantization
    factor, block type, against the label's algorithm, block type, blocks per group,
    quantization factor -- so a reader taking them from an index row would publish them
    shuffled.

    Parameters:
        attribute: The attribute one element is published under.
        value: The element the label states for it.
    """
    block = _cassini_observation(_sequence_label()).get_public_metadata()['label_metadata']
    assert block[attribute] == value


def test_a_keyword_stating_one_value_fills_the_first_of_its_attributes() -> None:
    """A single value fills the first attribute of its keyword and nulls the later ones.

    Some labels state ``OPTICS_TEMPERATURE`` as one reading rather than two. The reading
    they state is the front one and they carry no rear one, which is what the two-value
    form says as well when the rear reading is the ``-999.0`` of a camera with no rear
    sensor. Reading a single value as no sequence at all would publish the front reading
    the label does state as null.
    """
    label = VicarLabelStandIn(OPTICS_TEMPERATURE=0.71)
    block = _cassini_observation(label).get_public_metadata()['label_metadata']
    assert block['cassini:optics_temperature_front'] == 0.71
    assert block['cassini:optics_temperature_back'] is None


def test_the_keywords_the_block_once_stated_itself_are_published_as_attributes() -> None:
    """The four items the block used to state under names of its own are attributes now.

    Each is published once, in the block below, under the dictionary's name for it.
    """
    label = VicarLabelStandIn(
        DESCRIPTION='Saturn and its rings.',
        INSTRUMENT_MODE_ID='SUM2',
        OBSERVATION_ID='ISS_052SA_STRMOVIA001_PRIME',
        SHUTTER_MODE_ID='BOTSIM',
    )
    block = _cassini_observation(label).get_public_metadata()['label_metadata']
    assert block['cassini:limitations'] == 'Saturn and its rings.'
    assert block['cassini:instrument_mode_id'] == 'SUM2'
    assert block['cassini:observation_id'] == 'ISS_052SA_STRMOVIA001_PRIME'
    assert block['cassini:shutter_mode_id'] == 'BOTSIM'


def test_the_items_published_twice_are_gone_from_the_top_level() -> None:
    """Each of the four is stated once, in the block, and no longer beside it.

    A program wanting the summing mode reads the image size; the label's own mode is
    ``cassini:instrument_mode_id``.
    """
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert 'sampling' not in public
    assert 'gain_mode' not in public
    assert 'description' not in public
    assert 'observation_id' not in public


def test_the_filters_stay_beside_the_block() -> None:
    """``filters`` is published at the top level as well, as every host publishes it."""
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert public['filters'] == ['CL1', 'GRN']


@pytest.mark.parametrize(
    'image_name',
    ['N1521598221_1.IMG', 'N1521598221_1_CALIB.IMG'],
    ids=['raw', 'calibrated'],
)
def test_the_version_number_is_read_from_the_image_file_name(image_name: str) -> None:
    """No keyword states the version number; the file name does, and both names agree.

    It is the segment after the image number, which the calibrated file keeps, so a raw
    file and the calibrated product made from it publish the same version.

    Parameters:
        image_name: Basename of the image file.
    """
    obs = _cassini_observation(VicarLabelStandIn(), image_name=image_name)
    block = obs.get_public_metadata()['label_metadata']
    assert block['cassini:pre-pds_version_number'] == 1


def test_a_file_name_stating_no_version_number_publishes_null() -> None:
    """A name the version rule does not fit states no version, like any unstated value."""
    obs = _cassini_observation(VicarLabelStandIn(), image_name='N1521598221.IMG')
    block = obs.get_public_metadata()['label_metadata']
    assert block['cassini:pre-pds_version_number'] is None


def test_the_mission_phase_keeps_the_spelling_its_label_uses() -> None:
    """N1454725799_1_CALIB's phase keeps its underscore, and its two flags stay apart.

    The value published is the label's own, not a tidied one, and a flag is published
    from its own keyword rather than from another flag's.
    """
    block = _cassini_observation(_n1454725799_label()).get_public_metadata()['label_metadata']
    assert block['cassini:mission_phase_name'] == 'APPROACH_SCIENCE'
    assert block['cassini:antiblooming_state_flag'] == 'OFF'
    assert block['cassini:light_flood_state_flag'] == 'ON'


def test_the_image_number_is_the_label_value_not_the_start_count() -> None:
    """N1737255524_1_CALIB's exposure spans a second, so the two differ.

    The dictionary derives this number from the start count; the archive takes it from
    the clock at shutter close, and the label's own value is what is published.
    """
    block = _cassini_observation(_n1737255524_label()).get_public_metadata()['label_metadata']
    assert block['cassini:image_number'] == 1737255524
    assert block['cassini:spacecraft_clock_start_count'] == '1737255523.232'


def test_a_lossy_label_states_its_compression_parameters_as_numbers() -> None:
    """N1737255524_1_CALIB was lossy compressed, so its four parameters are numbers.

    A label with no lossy compression states ``N/A`` in all four instead. Both forms are
    published as the label states them, so the four attributes are not numbers by type.
    """
    block = _cassini_observation(_n1737255524_label()).get_public_metadata()['label_metadata']
    assert block['cassini:inst_cmprs_param_malgo'] == 0
    assert block['cassini:inst_cmprs_param_tb'] == 0
    assert block['cassini:inst_cmprs_param_blocks'] == 1
    assert block['cassini:inst_cmprs_param_quant'] == 0


def test_the_product_creation_time_keeps_the_trailing_z_its_label_writes() -> None:
    """The creation time is published exactly as the label writes it, ``Z`` and all.

    The archive states this time in Pacific local time rather than UTC, and writes it
    with a trailing ``Z`` on some labels and without one on others. Neither is corrected,
    so a reader sees which form its own image carries.
    """
    block = _cassini_observation(_n1454725799_label()).get_public_metadata()['label_metadata']
    assert block['cassini:pds3_product_creation_time'] == '2004-038T19:26:35.000Z'


def test_the_metadata_fixture_holds_the_keys_the_host_publishes() -> None:
    """The metadata chapter's Cassini fixture holds the host's keys, in the host's order.

    The chapter's staleness guard takes the Cassini metadata from that fixture, so an
    attribute the host gains has to reach the fixture, and through it the chapter, rather
    than pass unexamined.
    """
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert list(public) == list(CASSINI_ISS_PUBLIC_METADATA)
    assert list(public['label_metadata']) == list(CASSINI_ISS_PUBLIC_METADATA['label_metadata'])


@REQUIRES_EXTERNAL_DATA
def test_a_real_image_publishes_its_vicar_label_metadata() -> None:
    """The metadata is read from the VICAR label inside the calibrated image file.

    That label writes its times with a trailing Z, and gives a wide angle frame's back
    optics temperature as -999.0, since the wide angle camera has no rear optics sensor.
    Its two-element and four-element keywords reach the attributes their positions name,
    and W1521598221_1_CALIB.IMG's own name states its version.
    """
    public = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01).get_public_metadata()
    block = public['label_metadata']
    assert {
        key: block[key]
        for key in (
            'cassini:image_time',
            'cassini:gain_mode_id',
            'cassini:exposure_duration',
            'cassini:filter_name_1',
            'cassini:filter_name_2',
            'cassini:optics_temperature_front',
            'cassini:optics_temperature_back',
            'cassini:inst_cmprs_param_malgo',
            'cassini:pre-pds_version_number',
        )
    } == {
        'cassini:image_time': '2006-080T01:40:16.112Z',
        'cassini:gain_mode_id': '29 ELECTRONS PER DN',
        'cassini:exposure_duration': 1500.0,
        'cassini:filter_name_1': 'CL1',
        'cassini:filter_name_2': 'VIO',
        'cassini:optics_temperature_front': 6.93953,
        'cassini:optics_temperature_back': -999.0,
        'cassini:inst_cmprs_param_malgo': 'N/A',
        'cassini:pre-pds_version_number': 1,
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
def test_a_tour_era_label_states_every_keyword_the_attributes_read() -> None:
    """A tour-era label states every keyword an attribute reads, and six beyond them.

    The attributes are written out rather than read from the label, so a tour-era label
    carrying a keyword no attribute holds would drop it in silence.  This is what says so
    instead: the six are the ones that do not describe the exposure, and a seventh would
    fail here.
    """
    obs = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01)
    stated = set(_property_block_keywords(obs.dict))
    read = _read_keywords()
    assert sorted(read - stated) == []
    assert sorted(stated - read) == sorted(_UNREAD_KEYWORDS)


@REQUIRES_EXTERNAL_DATA
def test_a_cruise_era_label_publishes_the_same_attributes_and_nulls_the_rest() -> None:
    """An earlier label states fewer of the keywords, and items of its own besides.

    N1294562651_1_CALIB, of the earliest cruise volume, marks its property section with a
    plain ``PROPERTY`` keyword rather than the numbered form, states only some of the
    keywords the attributes read, and carries items no attribute reads, among them
    differently named equivalents such as ``FILTER1_NAME`` and ``SENSOR_HEAD_ELEC_TEMP``.
    Its label is read directly rather than through the host, because a cruise epoch has no
    camera frame in the local kernel set and loading the image would raise.  The block it
    publishes is the same shape as a tour image's all the same.

    This label states ``OPTICS_TEMPERATURE`` as one number rather than two, so the split
    attributes are checked here and not only the plainly mapped ones: reading a single
    value as no sequence would publish the front reading this label does state as null.
    """
    path = cast(Path, FCPath(URL_CASSINI_ISS_CRUISE_01).retrieve())
    label = vicar.VicarImage.from_file(path, strict=False).label
    published = _label_metadata(label, 'N1294562651_1_CALIB.IMG')
    section = _property_block_keywords(label)
    absent = [
        (attribute, keyword)
        for attribute, keyword, _element in _LABEL_METADATA
        if keyword is not None and keyword not in section
    ]
    outside = [keyword for keyword in section if keyword not in _read_keywords()]
    optics = label.get('OPTICS_TEMPERATURE', None)
    assert list(published) == list(_W1573251410_METADATA)
    assert absent != [], 'this label is expected to state fewer than the keywords read'
    assert outside != [], 'this label is expected to carry items no attribute reads'
    assert not isinstance(optics, list), 'this label is expected to state one optics reading'
    assert published['cassini:optics_temperature_front'] == optics
    assert published['cassini:optics_temperature_back'] is None
    assert {
        f'cassini:{attribute}': published[f'cassini:{attribute}'] for attribute, _ in absent
    } == dict.fromkeys(f'cassini:{attribute}' for attribute, _ in absent)
    assert {
        f'cassini:{attribute}': published[f'cassini:{attribute}']
        for attribute, keyword, element in _LABEL_METADATA
        if keyword is not None and keyword in section and element is None
    } == {
        f'cassini:{attribute}': label.get(keyword, None)
        for attribute, keyword, element in _LABEL_METADATA
        if keyword is not None and keyword in section and element is None
    }
