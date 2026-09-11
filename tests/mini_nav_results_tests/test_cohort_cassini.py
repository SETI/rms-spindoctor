"""Self-tests of the Cassini ISS Saturn cohort: what only its bundle's images can say.

The cohort is built from each image's epoch and nothing else, so what is worth
holding it to is that everything derived from that epoch still agrees with it:
the clock readings a document records, and the number the image is named for.
An image whose name and epoch disagree is the defect the epoch-first
constructor exists to make unreachable, and it is invisible to any test that
reads one of them alone.

The rest is what the bundle stage reads off this cohort and cannot check for
itself: the layout, which is two navigated images that shard into different
bundle directories; the holdings layout below a holdings root, which a document
and an enumeration share; and the index row, whose column names are the one
thing a label variable is read by.  What every cohort is held to is in
``test_cohort.py``.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from spindoctor.dataset import DataSetPDS3CassiniISSSaturn
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import (
    LIMB_IMAGE_NAME,
    RINGS_IMAGE_NAME,
    CassiniISSSaturnCohort,
)
from tests.sclk_readings import triples_disagreeing_with_their_epochs


@pytest.fixture(scope='module')
def documents() -> dict[str, dict[str, Any]]:
    """Build every cohort document once for the whole module.

    Returns:
        Stub to the document the writer produces for it.
    """
    return CassiniISSSaturnCohort.documents()


@pytest.fixture(scope='module')
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CassiniISSSaturnCohort:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CassiniISSSaturnCohort)


def test_every_clock_triple_spans_the_epochs_beside_it(
    documents: dict[str, dict[str, Any]],
) -> None:
    """A reading that is not the one its epoch converts to is an invented one.

    Every reader that subtracts two readings, or converts one back into an
    epoch, reads whatever a hand-authored triple happened to say.
    """
    assert triples_disagreeing_with_their_epochs(documents) == []


def test_every_image_is_named_for_the_reading_its_shutter_opened_at(
    documents: dict[str, dict[str, Any]],
) -> None:
    """The name and the epoch are two spellings of one moment, or the document lies.

    A Cassini image is named for the whole-second field of the clock reading at
    shutter open, so the name is derivable from the epoch and any disagreement
    is a number that came from somewhere else.
    """
    disagreeing: list[str] = []
    for image in CassiniISSSaturnCohort.images():
        recorded = str(documents[image.stub]['navigation_result']['times']['sclk_start'])
        named = image.image_name[1:].split('_', 1)[0]
        if recorded.split('/', 1)[1].split('.')[0] != named:
            disagreeing.append(f'{image.image_name} carries sclk_start {recorded}')
    assert disagreeing == []


def test_every_index_row_column_is_one_the_real_index_has() -> None:
    """A column name the index does not have populates nothing, silently.

    The label stage reads sixty-six of its variables out of the row by name,
    and pdstemplate renders a name it cannot resolve as an empty value; a
    misspelling here would therefore look exactly like the defect that made the
    row worth carrying, and a later phase would set out to fix a variable that
    was never broken.
    """
    unknown: list[str] = []
    for image in CassiniISSSaturnCohort.images():
        unknown += [
            f'{image.image_name}: {column}'
            for column in sorted(set(image.index_file_row) - _COISS_INDEX_COLUMNS)
        ]
    assert unknown == []


def test_the_two_navigated_images_shard_into_different_bundle_directories() -> None:
    """A cohort that shards into one directory cannot show a per-directory defect."""
    shards = [
        DataSetPDS3CassiniISSSaturn.pds4_bundle_path_for_image(name.split('_', 1)[0])
        for name in (LIMB_IMAGE_NAME, RINGS_IMAGE_NAME)
    ]
    assert shards[0] != shards[1]


_COISS_INDEX_COLUMNS = frozenset(
    (
        'ANTIBLOOMING_STATE_FLAG',
        'BIAS_STRIP_MEAN',
        'CALIBRATION_LAMP_STATE_FLAG',
        'CENTER_LATITUDE',
        'CENTER_LONGITUDE',
        'CENTRAL_BODY_DISTANCE',
        'COMMAND_FILE_NAME',
        'COMMAND_SEQUENCE_NUMBER',
        'COORDINATE_SYSTEM_NAME',
        'DARK_STRIP_MEAN',
        'DATA_CONVERSION_TYPE',
        'DATA_SET_ID',
        'DATA_SET_NAME',
        'DECLINATION',
        'DELAYED_READOUT_FLAG',
        'DESCRIPTION',
        'DETECTOR_TEMPERATURE',
        'EARTH_RECEIVED_START_TIME',
        'EARTH_RECEIVED_STOP_TIME',
        'ELECTRONICS_BIAS',
        'EMISSION_ANGLE',
        'EXPECTED_MAXIMUM',
        'EXPECTED_PACKETS',
        'EXPOSURE_DURATION',
        'FILE_NAME',
        'FILE_SPECIFICATION_NAME',
        'FILTER_NAME',
        'FILTER_TEMPERATURE',
        'FLIGHT_SOFTWARE_VERSION_ID',
        'GAIN_MODE_ID',
        'IMAGE_MID_TIME',
        'IMAGE_NUMBER',
        'IMAGE_OBSERVATION_TYPE',
        'IMAGE_TIME',
        'INCIDENCE_ANGLE',
        'INST_CMPRS_PARAM',
        'INST_CMPRS_RATE',
        'INST_CMPRS_RATIO',
        'INST_CMPRS_TYPE',
        'INSTRUMENT_DATA_RATE',
        'INSTRUMENT_HOST_ID',
        'INSTRUMENT_HOST_NAME',
        'INSTRUMENT_ID',
        'INSTRUMENT_MODE_ID',
        'INSTRUMENT_NAME',
        'LIGHT_FLOOD_STATE_FLAG',
        'LOWER_LEFT_LATITUDE',
        'LOWER_LEFT_LONGITUDE',
        'LOWER_RIGHT_LATITUDE',
        'LOWER_RIGHT_LONGITUDE',
        'MAXIMUM_RING_RADIUS',
        'METHOD_DESC',
        'MINIMUM_RING_RADIUS',
        'MISSING_LINES',
        'MISSING_PACKET_FLAG',
        'MISSION_NAME',
        'MISSION_PHASE_NAME',
        'NORTH_AZIMUTH_CLOCK_ANGLE',
        'OBSERVATION_ID',
        'OPTICS_TEMPERATURE',
        'ORDER_NUMBER',
        'PARALLEL_CLOCK_VOLTAGE_INDEX',
        'PHASE_ANGLE',
        'PIXEL_SCALE',
        'PLANET_CENTER',
        'PREPARE_CYCLE_INDEX',
        'PRODUCT_CREATION_TIME',
        'PRODUCT_ID',
        'PRODUCT_TYPE',
        'PRODUCT_VERSION_TYPE',
        'READOUT_CYCLE_INDEX',
        'RECEIVED_PACKETS',
        'RIGHT_ASCENSION',
        'RING_CENTER_LATITUDE',
        'RING_CENTER_LONGITUDE',
        'RING_EMISSION_ANGLE',
        'RING_INCIDENCE_ANGLE',
        'RINGS_FLAG',
        'SC_PLANET_POSITION_VECTOR',
        'SC_PLANET_VELOCITY_VECTOR',
        'SC_SUN_POSITION_VECTOR',
        'SC_SUN_VELOCITY_VECTOR',
        'SC_TARGET_POSITION_VECTOR',
        'SC_TARGET_VELOCITY_VECTOR',
        'SENSOR_HEAD_ELEC_TEMPERATURE',
        'SEQUENCE_ID',
        'SEQUENCE_NUMBER',
        'SEQUENCE_TITLE',
        'SHUTTER_MODE_ID',
        'SHUTTER_STATE_ID',
        'SOFTWARE_VERSION_ID',
        'SPACECRAFT_CLOCK_CNT_PARTITION',
        'SPACECRAFT_CLOCK_START_COUNT',
        'SPACECRAFT_CLOCK_STOP_COUNT',
        'SPICE_PRODUCT_ID',
        'STANDARD_DATA_PRODUCT_ID',
        'START_TIME',
        'STOP_TIME',
        'SUB_SOLAR_LATITUDE',
        'SUB_SOLAR_LONGITUDE',
        'SUB_SPACECRAFT_LATITUDE',
        'SUB_SPACECRAFT_LONGITUDE',
        'TARGET_DESC',
        'TARGET_DISTANCE',
        'TARGET_EASTERNMOST_LONGITUDE',
        'TARGET_LIST',
        'TARGET_NAME',
        'TARGET_NORTH_CLOCK_ANGLE',
        'TARGET_NORTHERNMOST_LATITUDE',
        'TARGET_SOUTHERNMOST_LATITUDE',
        'TARGET_WESTERNMOST_LONGITUDE',
        'TELEMETRY_FORMAT_ID',
        'TWIST_ANGLE',
        'UPPER_LEFT_LATITUDE',
        'UPPER_LEFT_LONGITUDE',
        'UPPER_RIGHT_LATITUDE',
        'UPPER_RIGHT_LONGITUDE',
        'VALID_MAXIMUM',
        'VOLUME_ID',
    )
)
"""Every column the Cassini index tables name, transcribed from a real label.

The 119 COLUMN NAME values of COISS_2001_index.lbl, which is the
index of the volume the cohort's rows say they came from.  Sixty-six of the
seventy cassini:* label variables are read out of a row of that table by
name, and a name the table does not have reads as an absent value rather than
as an error -- indistinguishable, in a rendered label, from a variable that is
genuinely empty.  So the row is held against the table here rather than against
itself.
"""


def _below_the_holdings_root(path: str) -> str:
    """Return the part of an image's path a holdings root is not.

    Parameters:
        path: The path, holdings root and all.

    Returns:
        Everything below the last ``holdings`` directory in it.
    """
    return path.rsplit('/holdings/', 1)[-1]


def test_a_document_and_its_image_file_name_one_file(cassini_cohort: Cohort) -> None:
    """The path a run recorded and the URL an enumeration hands on are one file.

    Two roots, deliberately -- a document records the machine that navigated
    the image and a bundle run is given holdings of its own -- but one layout
    below them, since a phase deriving a volume or a collection out of either
    has only the layout to derive it from.  The index row is the odd one out on
    purpose: it names the raw product on its own volume, which is what a real
    index names and a different file.
    """
    found: dict[str, tuple[str, str]] = {}
    expected: dict[str, tuple[str, str]] = {}
    for image, image_file in zip(
        CassiniISSSaturnCohort.images(), cassini_cohort.image_files, strict=True
    ):
        document = json.loads(
            Path(f'{cassini_cohort.nav_results_root / image.stub}_metadata.json').read_text(
                encoding='utf-8'
            )
        )
        found[image.stub] = (
            _below_the_holdings_root(str(document['observation']['image_path'])),
            _below_the_holdings_root(image_file.image_file_url.as_posix()),
        )
        one_file = f'{CassiniISSSaturnCohort.HOLDINGS_SUBTREE}/{image.stub}.IMG'
        expected[image.stub] = (one_file, one_file)
    assert found == expected
