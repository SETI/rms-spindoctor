"""Self-tests for the bundle cohort the PDS4 phases are asserted against.

The cohort is built from each image's epoch and nothing else, so what is worth
holding it to is that everything derived from that epoch still agrees with it:
the clock readings a document records, and the number the image is named for.
An image whose name and epoch disagree is the defect the epoch-first
constructor exists to make unreachable, and it is invisible to any test that
reads one of them alone.

The rest is what the bundle stage reads off the cohort and cannot check for
itself: the layout, which is two images that shard into different bundle
directories and one with ring backplanes and one without; the products, which
are a real FITS whose HDUs a reader finds where a label says they are; and the
index row, whose column names are the one thing a label variable is read by.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from astropy.io import fits

from spindoctor.dataset import DataSetPDS3CassiniISSSaturn
from tests.mini_nav_results import cohort_documents
from tests.mini_nav_results.cohort import Cohort
from tests.mini_nav_results.cohort_cassini import (
    HOLDINGS_SUBTREE,
    LIMB_IMAGE_NAME,
    RINGS_IMAGE_NAME,
    cohort_images,
)
from tests.sclk_readings import triples_disagreeing_with_their_epochs


@pytest.fixture(scope='module')
def documents() -> dict[str, dict[str, Any]]:
    """Build every cohort document once for the whole module.

    Returns:
        Stub to the document the writer produces for it.
    """
    return cohort_documents()


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
    for image in cohort_images():
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
    for image in cohort_images():
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


_BODY_HDU_NAMES = (
    'BODY_COARSEST_RESOLUTION',
    'BODY_EMISSION_ANGLE',
    'BODY_FINEST_RESOLUTION',
    'BODY_INCIDENCE_ANGLE',
    'BODY_LATITUDE',
    'BODY_LONGITUDE',
    'BODY_PHASE_ANGLE',
)
"""The image HDUs a frame with body backplanes carries, in the order written.

Written out rather than sorted here: the order is what every array's byte
offset in the file is stated against, and a test that sorts the names it
expects agrees with a merge that stopped sorting.  This is the order both real
products on this machine carry.
"""

_RING_HDU_NAMES = (
    'RING_EMISSION_ANGLE',
    'RING_LONGITUDE',
    'RING_LONGITUDINAL_RESOLUTION',
    'RING_PHASE_ANGLE',
    'RING_RADIAL_RESOLUTION',
    'RING_RADIUS',
)
"""The image HDUs a frame with ring backplanes carries as well, after them all."""


def test_each_fits_carries_the_hdus_its_backplanes_imply(mini_nav_cohort: Cohort) -> None:
    """A reader opens a real FITS and finds the planes the image was navigated on.

    The byte blob a stand-in writes has no HDUs to find, and the label states
    where in the file each array begins.
    """
    found: dict[str, tuple[str, ...]] = {}
    for image in cohort_images():
        if not image.navigated:
            continue
        path = mini_nav_cohort.backplane_results_root / f'{image.stub}_backplanes.fits'
        with fits.open(path) as hdul:
            found[image.image_name] = tuple(hdu.name for hdu in hdul)
    assert found == {
        LIMB_IMAGE_NAME: ('PRIMARY', 'BODY_ID_MAP', *_BODY_HDU_NAMES),
        RINGS_IMAGE_NAME: ('PRIMARY', 'BODY_ID_MAP', *_BODY_HDU_NAMES, *_RING_HDU_NAMES),
    }


def test_each_backplane_document_names_the_planes_its_fits_carries(
    mini_nav_cohort: Cohort,
) -> None:
    """The index columns come from one and the arrays from the other.

    A document naming a plane the FITS does not carry, or missing one it does,
    puts a global index column beside an array that is not the one it measures.
    A frame with no ring pixels in view is read the same way, through the same
    key: the ring stage returns a result holding nothing rather than no result,
    so ``rings`` names an empty ``backplanes`` rather than being empty itself,
    and a document that has to be read defensively is one no run wrote.
    """
    disagreeing: list[str] = []
    for image in cohort_images():
        if not image.navigated:
            continue
        stem = mini_nav_cohort.backplane_results_root / image.stub
        with fits.open(Path(f'{stem}_backplanes.fits')) as hdul:
            in_the_fits = {hdu.name for hdu in hdul} - {'PRIMARY', 'BODY_ID_MAP'}
        document = json.loads(Path(f'{stem}_backplane_metadata.json').read_text(encoding='utf-8'))
        named: set[str] = set()
        for body in document['bodies'].values():
            named |= {name.upper() for name in body['backplanes']}
        named |= {name.upper() for name in document['rings']['backplanes']}
        if named != in_the_fits:
            disagreeing.append(
                f'{image.image_name}: the document names {sorted(named)} and the FITS '
                f'carries {sorted(in_the_fits)}'
            )
    assert disagreeing == []


def test_each_body_is_placed_down_the_frame_and_sized_across_it(
    mini_nav_cohort: Cohort,
) -> None:
    """The writer states a body's center and its extent in opposite axis orders.

    ``center_uv`` is written down the frame first and ``size_uv`` across it
    first, which is a convention nothing in a document declares and everything
    that draws a body over an image depends on.  Both are read here against the
    geometry the cohort declared, so a transposition on either side is reported
    rather than absorbed by a body a swap describes just as well.
    """
    found: dict[str, tuple[list[float], list[float]]] = {}
    expected: dict[str, tuple[list[float], list[float]]] = {}
    for image in cohort_images():
        if not image.navigated:
            continue
        stem = mini_nav_cohort.backplane_results_root / image.stub
        document = json.loads(Path(f'{stem}_backplane_metadata.json').read_text(encoding='utf-8'))
        for body in image.bodies:
            entry = document['bodies'][body.name]
            found[body.name] = (entry['center_uv'], entry['size_uv'])
            center_v, center_u = body.center_vu
            radius_v, radius_u = body.radii_vu
            expected[body.name] = ([center_v, center_u], [2.0 * radius_u, 2.0 * radius_v])
    assert found == expected


def _below_the_holdings_root(path: str) -> str:
    """Return the part of an image's path a holdings root is not.

    Parameters:
        path: The path, holdings root and all.

    Returns:
        Everything below the last ``holdings`` directory in it.
    """
    return path.rsplit('/holdings/', 1)[-1]


def test_a_document_and_its_image_file_name_one_file(mini_nav_cohort: Cohort) -> None:
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
    for image, image_file in zip(cohort_images(), mini_nav_cohort.image_files, strict=True):
        document = json.loads(
            Path(f'{mini_nav_cohort.nav_results_root / image.stub}_metadata.json').read_text(
                encoding='utf-8'
            )
        )
        found[image.stub] = (
            _below_the_holdings_root(str(document['observation']['image_path'])),
            _below_the_holdings_root(image_file.image_file_url.as_posix()),
        )
        one_file = f'{HOLDINGS_SUBTREE}/{image.stub}.IMG'
        expected[image.stub] = (one_file, one_file)
    assert found == expected


def _paths_git_reports(repository: Path) -> list[str]:
    """Return every path ``git status`` reports, read without quoting.

    Asked for in the NUL-separated form, because the readable form quotes a
    path holding a space or a byte outside ASCII and a reader that does not
    unquote it then compares a name with a quotation mark on the end of it,
    which matches nothing -- so the one file most likely to be somewhere it
    should not be is the one a guard over the readable form cannot see.

    Parameters:
        repository: The checkout to ask about.

    Returns:
        The paths, with a renamed entry contributing both of its names.
    """
    reported = subprocess.run(
        ['git', 'status', '--porcelain', '-z', '--untracked-files=all'],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    entries = iter(reported.split('\0'))
    paths: list[str] = []
    for entry in entries:
        if not entry:
            continue
        paths.append(entry[3:])
        if 'R' in entry[:2] or 'C' in entry[:2]:
            # A rename or a copy names where it went and then, as a record of
            # its own, where it came from.
            paths.append(next(entries, ''))
    return paths


def test_no_cohort_product_reaches_the_working_tree(mini_nav_cohort: Cohort) -> None:
    """The cohort is built where it is torn down, and nothing it writes is committed.

    Its products are named for their images, so a file of one of those names
    anywhere in the repository is a build that escaped the temporary directory
    -- reported here rather than committed by whoever runs ``git add`` next.
    """
    repository = Path(__file__).resolve().parents[2]
    if not (repository / '.git').exists():
        pytest.skip('not a git checkout')
    product_names = {path.name for path in mini_nav_cohort.written}
    escaped = [path for path in _paths_git_reports(repository) if Path(path).name in product_names]
    assert escaped == []
