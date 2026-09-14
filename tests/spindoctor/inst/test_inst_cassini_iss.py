"""Tests for ``spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS``."""

from typing import Any

import pytest
from tests.config import REQUIRES_EXTERNAL_DATA, URL_CASSINI_ISS_RHEA_01
from tests.spindoctor.inst.conftest import (
    VicarLabelStandIn,
    bare_observation,
    published_clock_counts,
)

import spindoctor.obs.obs_inst_cassini_iss as obstcoiss
from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS, _sclk_count

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


def _w1573251410_label() -> VicarLabelStandIn:
    """Return the label items the host reads from W1573251410_1_CALIB (COISS_2039).

    They are the VICAR label's, as rms-vicar reads them.  The frame is a lossy wide angle
    one, so its compression parameters are numbers and its missing-line count is the text
    ``N/A`` a lossy label writes.

    Returns:
        The label items.
    """
    return VicarLabelStandIn(
        MISSION_PHASE_NAME='TOUR',
        SPACECRAFT_CLOCK_CNT_PARTITION=1,
        SPACECRAFT_CLOCK_START_COUNT='1573251410.115',
        SPACECRAFT_CLOCK_STOP_COUNT='1573251410.122',
        DESCRIPTION='N/A',
        ANTIBLOOMING_STATE_FLAG='ON',
        BIAS_STRIP_MEAN=22.0,
        CALIBRATION_LAMP_STATE_FLAG='OFF',
        COMMAND_FILE_NAME='trigger_7192_2.ioi',
        COMMAND_SEQUENCE_NUMBER=7192,
        DARK_STRIP_MEAN=19.5,
        DATA_CONVERSION_TYPE='TABLE',
        DELAYED_READOUT_FLAG='YES',
        DETECTOR_TEMPERATURE=-87.8952,
        ELECTRONICS_BIAS=112,
        EARTH_RECEIVED_START_TIME='2007-313T14:43:33.041Z',
        EARTH_RECEIVED_STOP_TIME='2007-313T14:43:36.276Z',
        EXPECTED_MAXIMUM=[56.0227, 61.7658],
        EXPECTED_PACKETS=28,
        EXPOSURE_DURATION=25.0,
        FILTER_NAME=['CL1', 'GRN'],
        FILTER_TEMPERATURE=3.19298,
        FLIGHT_SOFTWARE_VERSION_ID='1.4',
        GAIN_MODE_ID='29 ELECTRONS PER DN',
        SOFTWARE_VERSION_ID='ISS 11.00 05-24-2006',
        IMAGE_MID_TIME='2007-312T21:41:14.934Z',
        IMAGE_NUMBER=1573251410,
        IMAGE_TIME='2007-312T21:41:14.946Z',
        IMAGE_OBSERVATION_TYPE='SCIENCE',
        INSTRUMENT_DATA_RATE=182.784,
        INSTRUMENT_MODE_ID='FULL',
        INST_CMPRS_TYPE='LOSSY',
        INST_CMPRS_PARAM=[1, 1, 41, 0],
        INST_CMPRS_RATE=[0.194248, 0.360077],
        INST_CMPRS_RATIO=22.2175,
        LIGHT_FLOOD_STATE_FLAG='ON',
        METHOD_DESC='ISSPT2.6.5;Saturn;ISS_052SA_STRMOVIA001_PRIME_2',
        MISSING_LINES='N/A',
        MISSING_PACKET_FLAG='NO',
        OBSERVATION_ID='ISS_052SA_STRMOVIA001_PRIME',
        OPTICS_TEMPERATURE=[6.93953, -999.0],
        ORDER_NUMBER=2,
        PARALLEL_CLOCK_VOLTAGE_INDEX=9,
        PRODUCT_CREATION_TIME='2007-313T17:04:33.000',
        PRODUCT_VERSION_TYPE='FINAL',
        TARGET_DESC='Saturn',
        TARGET_LIST='N/A',
        TARGET_NAME='SATURN',
        PREPARE_CYCLE_INDEX=3,
        READOUT_CYCLE_INDEX=15,
        RECEIVED_PACKETS=51,
        SENSOR_HEAD_ELEC_TEMPERATURE=2.98847,
        SEQUENCE_ID='S35',
        SEQUENCE_NUMBER=148,
        SEQUENCE_TITLE='ISS_052SA_STRMOVIA001_PRIME_2',
        SHUTTER_MODE_ID='BOTSIM',
        SHUTTER_STATE_ID='ENABLED',
        START_TIME='2007-312T21:41:14.921Z',
        STOP_TIME='2007-312T21:41:14.946Z',
        TELEMETRY_FORMAT_ID='S&ER3',
        VALID_MAXIMUM=[4095, 4095],
    )


_W1573251410_FACTS: dict[str, Any] = {
    'mission_phase_name': 'TOUR',
    'spacecraft_clock_count_partition': 1,
    'spacecraft_clock_start_count': '1573251410.115',
    'spacecraft_clock_stop_count': '1573251410.122',
    'antiblooming_state_flag': 'ON',
    'bias_strip_mean': 22.0,
    'calibration_lamp_state_flag': 'OFF',
    'command_file_name': 'trigger_7192_2.ioi',
    'command_sequence_number': 7192,
    'dark_strip_mean': 19.5,
    'data_conversion_type': 'TABLE',
    'delayed_readout_flag': 'YES',
    'detector_temperature': -87.8952,
    'electronics_bias': 112,
    'earth_received_start_time': '2007-313T14:43:33.041Z',
    'earth_received_stop_time': '2007-313T14:43:36.276Z',
    'expected_maximum_full_well': 56.0227,
    'expected_maximum_DN_sat': 61.7658,
    'expected_packets': 28,
    'exposure_duration': 25.0,
    'filter_temperature': 3.19298,
    'flight_software_version_id': '1.4',
    'gain_mode_id': '29 ELECTRONS PER DN',
    'ground_software_version_id': 'ISS 11.00 05-24-2006',
    'image_mid_time': '2007-312T21:41:14.934Z',
    'image_number': 1573251410,
    'image_time': '2007-312T21:41:14.946Z',
    'image_observation_type': 'SCIENCE',
    'instrument_data_rate': 182.784,
    'inst_cmprs_type': 'LOSSY',
    'inst_cmprs_param_malgo': 1,
    'inst_cmprs_param_tb': 1,
    'inst_cmprs_param_blocks': 41,
    'inst_cmprs_param_quant': 0,
    'inst_cmprs_rate_expected_bits': 0.194248,
    'inst_cmprs_rate_actual_bits': 0.360077,
    'inst_cmprs_ratio': 22.2175,
    'light_flood_state_flag': 'ON',
    'method_description': 'ISSPT2.6.5;Saturn;ISS_052SA_STRMOVIA001_PRIME_2',
    'missing_lines': 'N/A',
    'missing_packet_flag': 'NO',
    'optics_temperature_front': 6.93953,
    'optics_temperature_back': -999.0,
    'order_number': 2,
    'parallel_clock_voltage_index': 9,
    'pds3_product_creation_time': '2007-313T17:04:33.000',
    'pds3_product_version_type': 'FINAL',
    'pds3_target_desc': 'Saturn',
    'pds3_target_list': 'N/A',
    'pds3_target_name': 'SATURN',
    'prepare_cycle_index': 3,
    'readout_cycle_index': 15,
    'received_packets': 51,
    'sensor_head_electronics_temperature': 2.98847,
    'sequence_id': 'S35',
    'sequence_number': 148,
    'sequence_title': 'ISS_052SA_STRMOVIA001_PRIME_2',
    'shutter_state_id': 'ENABLED',
    'start_time_doy': '2007-312T21:41:14.921Z',
    'stop_time_doy': '2007-312T21:41:14.946Z',
    'telemetry_format_id': 'S&ER3',
    'valid_maximum_full_well': 4095,
    'valid_maximum_DN_sat': 4095,
}
"""The label facts W1573251410_1_CALIB's label states, under their dictionary names."""


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


def test_the_label_facts_are_published_as_the_label_states_them() -> None:
    """Each label fact is published under its dictionary name, as the label states it.

    W1573251410_1_CALIB's numbers stay numbers, its times are its own day-of-year
    spellings with their trailing Z, its gain mode is its text, and its missing-line count
    is the text N/A.
    """
    public = _wide_angle_observation(_w1573251410_label()).get_public_metadata()
    assert {key: public[key] for key in _W1573251410_FACTS} == _W1573251410_FACTS


def test_a_keyword_the_label_lacks_is_published_as_null() -> None:
    """A label carrying none of the keywords publishes every label fact as null.

    That covers a scalar keyword and each element of a sequence keyword alike.
    """
    public = _cassini_observation(VicarLabelStandIn()).get_public_metadata()
    assert {key: public[key] for key in _W1573251410_FACTS} == dict.fromkeys(_W1573251410_FACTS)


@pytest.mark.parametrize(
    ('keyword', 'values', 'names'),
    [
        pytest.param(
            'EXPECTED_MAXIMUM',
            [56.0227, 61.7658],
            ('expected_maximum_full_well', 'expected_maximum_DN_sat'),
            id='EXPECTED_MAXIMUM',
        ),
        pytest.param(
            'INST_CMPRS_PARAM',
            [1, 0, 41, 11],
            (
                'inst_cmprs_param_malgo',
                'inst_cmprs_param_tb',
                'inst_cmprs_param_blocks',
                'inst_cmprs_param_quant',
            ),
            id='INST_CMPRS_PARAM',
        ),
        pytest.param(
            'INST_CMPRS_RATE',
            [0.194248, 0.360077],
            ('inst_cmprs_rate_expected_bits', 'inst_cmprs_rate_actual_bits'),
            id='INST_CMPRS_RATE',
        ),
        pytest.param(
            'OPTICS_TEMPERATURE',
            [0.712693, 1.90571],
            ('optics_temperature_front', 'optics_temperature_back'),
            id='OPTICS_TEMPERATURE',
        ),
        pytest.param(
            'VALID_MAXIMUM',
            [16380, 4095],
            ('valid_maximum_full_well', 'valid_maximum_DN_sat'),
            id='VALID_MAXIMUM',
        ),
    ],
)
def test_a_sequence_keyword_is_split_into_its_attributes_in_order(
    keyword: str, values: list[Any], names: tuple[str, ...]
) -> None:
    """Each element of a sequence keyword is published under the attribute it is.

    The values are real labels' where a real label's elements all differ.  No archive
    label's four compression parameters all differ, so those four are made distinct here;
    the label's order is malgo, block type, blocks per group, quantization factor.

    Parameters:
        keyword: The sequence keyword.
        values: Its elements, in the label's order.
        names: The attributes the elements are, in the same order.
    """
    public = _cassini_observation(VicarLabelStandIn({keyword: values})).get_public_metadata()
    assert [public[name] for name in names] == values


def test_a_sequence_keyword_of_the_wrong_length_is_refused() -> None:
    """A sequence with more elements than it names raises rather than dropping one."""
    label = VicarLabelStandIn(OPTICS_TEMPERATURE=[0.712693, 1.90571, 3.0])
    with pytest.raises(ValueError, match='argument 2 is longer than argument 1'):
        _cassini_observation(label).get_public_metadata()


@REQUIRES_EXTERNAL_DATA
def test_a_real_image_publishes_its_vicar_label_facts() -> None:
    """The label facts are read from the VICAR label inside the calibrated image file.

    That label writes its times with a trailing Z and a wide angle frame's back optics
    temperature as -999.0, since the wide angle camera has no rear optics sensor.
    """
    public = obstcoiss.ObsCassiniISS.from_file(URL_CASSINI_ISS_RHEA_01).get_public_metadata()
    assert {
        key: public[key]
        for key in (
            'image_time',
            'gain_mode_id',
            'exposure_duration',
            'optics_temperature_front',
            'optics_temperature_back',
            'inst_cmprs_param_malgo',
        )
    } == {
        'image_time': '2006-080T01:40:16.112Z',
        'gain_mode_id': '29 ELECTRONS PER DN',
        'exposure_duration': 1500.0,
        'optics_temperature_front': 6.93953,
        'optics_temperature_back': -999.0,
        'inst_cmprs_param_malgo': 'N/A',
    }
