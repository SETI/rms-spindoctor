"""Self-tests for the bundle cohort the PDS4 phases are asserted against.

The cohort is built from each image's epoch and nothing else, so what is worth
holding it to is that everything derived from that epoch still agrees with it:
the clock readings a document records, and the number the image is named for.
An image whose name and epoch disagree is the defect the epoch-first
constructor exists to make unreachable, and it is invisible to any test that
reads one of them alone.

The rest is the layout the bundle stage is built over: two images that shard
into different bundle directories, one with ring backplanes and one without.
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
    LIMB_IMAGE_NAME,
    RINGS_IMAGE_NAME,
    cohort_images,
)

_SCLK_TICK_S = 1.0 / 256.0
"""How long one tick of the Cassini clock is.

Written out rather than taken from the builders: a test that asks the code what
its own clock counts in agrees with every answer the code gives.  This is the
tick the mission clock kernel records, so a triple that disagrees with its own
epochs by more than one of them is a triple no conversion could have returned.
"""


def _sclk_seconds(reading: str) -> float:
    """Read a Cassini clock string back as a number of seconds on that clock.

    Parameters:
        reading: The clock string, partition and all.

    Returns:
        The reading in seconds, on the clock's own origin.  Only differences
        between two readings mean anything.
    """
    seconds, fraction = reading.split('/', 1)[1].split('.')
    return int(seconds) + int(fraction) * _SCLK_TICK_S


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
    disagreeing: list[str] = []
    for stub, document in documents.items():
        times = document['navigation_result']['times']
        opened = _sclk_seconds(str(times['sclk_start']))
        for reading, epoch in (('sclk_midtime', 'midtime_et'), ('sclk_stop', 'stop_et')):
            on_the_clock = _sclk_seconds(str(times[reading])) - opened
            between_the_epochs = float(times[epoch]) - float(times['start_et'])
            if abs(on_the_clock - between_the_epochs) > _SCLK_TICK_S:
                disagreeing.append(
                    f'{stub}: {reading} is {on_the_clock} s after sclk_start, against '
                    f'{between_the_epochs} s between the epochs'
                )
    assert disagreeing == []


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


def test_the_two_navigated_images_shard_into_different_bundle_directories() -> None:
    """A cohort that shards into one directory cannot show a per-directory defect."""
    shards = [
        DataSetPDS3CassiniISSSaturn.pds4_bundle_path_for_image(name.split('_', 1)[0])
        for name in (LIMB_IMAGE_NAME, RINGS_IMAGE_NAME)
    ]
    assert shards[0] != shards[1]


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

    Parameters:
        mini_nav_cohort: The session's cohort.
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

    Parameters:
        mini_nav_cohort: The session's cohort.
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
        named |= {name.upper() for name in document['rings'].get('backplanes', {})}
        if named != in_the_fits:
            disagreeing.append(
                f'{image.image_name}: the document names {sorted(named)} and the FITS '
                f'carries {sorted(in_the_fits)}'
            )
    assert disagreeing == []


def test_no_cohort_product_reaches_the_working_tree(mini_nav_cohort: Cohort) -> None:
    """The cohort is built where it is torn down, and nothing it writes is committed.

    Its products are named for their images, so a file of one of those names
    anywhere in the repository is a build that escaped the temporary directory
    -- reported here rather than committed by whoever runs ``git add`` next.

    Parameters:
        mini_nav_cohort: The session's cohort.
    """
    repository = Path(__file__).resolve().parents[2]
    if not (repository / '.git').exists():
        pytest.skip('not a git checkout')
    reported = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    product_names = {path.name for path in mini_nav_cohort.written}
    escaped = [line for line in reported if Path(line[3:]).name in product_names]
    assert escaped == []
