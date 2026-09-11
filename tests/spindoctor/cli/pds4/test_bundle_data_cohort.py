"""The labels pass over the cohort, through the shipped Cassini templates.

These tests ask what a label *says*, which only the registered dataset, the shipped
template set and the products a navigation run and the backplane stage leave behind
can answer (``CohortBundleEnv`` in the package's ``conftest``).  The plumbing of the
same pass -- which file goes where, which variable reaches which template -- is
tested over stand-ins in ``test_bundle_data.py``.
"""

import hashlib
import re
from pathlib import Path
from xml.etree import ElementTree

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort
from tests.mini_nav_results.cohort_cassini import (
    GATED_STUB,
    LIMB_IMAGE_NAME,
    LIMB_STUB,
    RINGS_IMAGE_NAME,
    RINGS_STUB,
)

from spindoctor.cli.pds4.bundle_data import BundleDataOutcome, generate_bundle_data_files
from spindoctor.config import MAIN_LOGGER

from .conftest import make_cohort_bundle_env


def test_the_cohort_image_that_did_not_navigate_is_skipped(
    mini_nav_cohort: Cohort, tmp_path: Path
) -> None:
    """An image the bundle has nothing to describe is skipped, not failed.

    Over the registered dataset and the shipped templates rather than
    stand-ins, because a selection made by volume routinely names images that
    did not navigate, and what the bundle does with one is a property of the
    run rather than of a fixture's ``status`` key.
    """
    env = make_cohort_bundle_env(mini_nav_cohort, tmp_path)
    outcome = generate_bundle_data_files(
        env.dataset,
        mini_nav_cohort.batch(GATED_STUB),
        nav_results_root=FCPath(mini_nav_cohort.nav_results_root),
        backplane_results_root=FCPath(mini_nav_cohort.backplane_results_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    assert outcome is BundleDataOutcome.SKIPPED
    assert not env.bundle_dir.exists()


def _product_stem(image_name: str) -> str:
    """Return where one navigated image's products sit, below a collection directory.

    The sharded directories and the product stem are spelled out here rather
    than asked of the dataset, since what they are is what the tests over the
    cohort are for: a bundle is read by walking those directories, and an image
    that lands in the wrong one is found by whoever cannot find it.

    Parameters:
        image_name: The calibrated image's name, camera letter and all.

    Returns:
        The two shard directories and the product stem, without a suffix.
    """
    number = image_name[1:11]
    return f'{number[:4]}xxxxxx/{number[:6]}xxxx/{number}{image_name[0].lower()}'


def _bundle_products(image_name: str) -> set[str]:
    """Return every bundle file the labels pass writes for one navigated image.

    Parameters:
        image_name: The calibrated image's name, camera letter and all.

    Returns:
        The paths, relative to the bundle's own directory.
    """
    stem = _product_stem(image_name)
    return {
        f'data/{stem}_backplanes.lblx',
        f'data/{stem}_backplanes.fits',
        f'data/{stem}_supplemental.txt',
        f'browse/{stem}_summary.lblx',
        f'browse/{stem}_summary.png',
    }


def test_the_cohort_s_navigated_images_are_written_into_the_bundle(
    mini_nav_cohort: Cohort, tmp_path: Path
) -> None:
    """The shipped Cassini templates render over the cohort, into their shards.

    This is the assertion the cohort exists to make possible and the one every
    phase after this builds on: the registered dataset, the shipped template
    set and a real backplane FITS, with nothing standing in for anything.  A
    template the fixture cannot satisfy, a product written under the wrong
    number, or a render that fails and leaves half a bundle behind is reported
    here, in the phase that owns the fixture.
    """
    env = make_cohort_bundle_env(mini_nav_cohort, tmp_path)
    outcomes = {
        stub: generate_bundle_data_files(
            env.dataset,
            mini_nav_cohort.batch(stub),
            nav_results_root=FCPath(mini_nav_cohort.nav_results_root),
            backplane_results_root=FCPath(mini_nav_cohort.backplane_results_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )
        for stub in (LIMB_STUB, RINGS_STUB)
    }
    assert outcomes == {
        LIMB_STUB: BundleDataOutcome.WRITTEN,
        RINGS_STUB: BundleDataOutcome.WRITTEN,
    }
    written = {
        path.relative_to(env.bundle_dir).as_posix()
        for path in env.bundle_dir.rglob('*')
        if path.is_file()
    }
    assert written == _bundle_products(LIMB_IMAGE_NAME) | _bundle_products(RINGS_IMAGE_NAME)


@pytest.mark.parametrize(
    ('stub', 'image_name', 'start', 'stop'),
    [
        (LIMB_STUB, LIMB_IMAGE_NAME, '2004-02-07T04:25:35.585Z', '2004-02-07T04:25:36.045Z'),
        (RINGS_STUB, RINGS_IMAGE_NAME, '2004-02-22T05:32:15.895Z', '2004-02-22T05:32:16.355Z'),
    ],
    ids=['limb image', 'ring image'],
)
def test_a_cohort_data_label_states_its_exposure_s_start_and_stop(
    mini_nav_cohort: Cohort, tmp_path: Path, stub: str, image_name: str, start: str, stop: str
) -> None:
    """The shipped data label states the document's start and stop, to the millisecond.

    The expected strings are SPICE's.  With the leapseconds kernel furnished,
    ``et2utc`` writes the limb image's recorded start and stop at three decimals as
    ``04:25:35.585`` and ``04:25:36.045``, and the ring image's as ``05:32:15.895``
    and ``05:32:16.355``: each at the nearest millisecond.  At nine decimals the
    ring image's start is ``05:32:15.894745827`` and the limb image's stop
    ``04:25:36.045069233``, so the first is a millisecond from what rounding down
    would write and the second a millisecond from what rounding up would.

    Parameters:
        mini_nav_cohort: The session's cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which cohort image, by its results path stub.
        image_name: That image's calibrated name.
        start: What its data label's ``start_date_time`` has to say.
        stop: What its data label's ``stop_date_time`` has to say.
    """
    env = make_cohort_bundle_env(mini_nav_cohort, tmp_path)
    generate_bundle_data_files(
        env.dataset,
        mini_nav_cohort.batch(stub),
        nav_results_root=FCPath(mini_nav_cohort.nav_results_root),
        backplane_results_root=FCPath(mini_nav_cohort.backplane_results_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    label = env.bundle_dir / 'data' / f'{_product_stem(image_name)}_backplanes.lblx'
    text = label.read_text(encoding='utf-8')
    assert re.findall(r'<start_date_time>(.*)</start_date_time>', text) == [start]
    assert re.findall(r'<stop_date_time>(.*)</stop_date_time>', text) == [stop]


# ---------------------------------------------------------------------------
# The backplane FITS the data label describes
# ---------------------------------------------------------------------------

NAVIGATED_IMAGES = [(LIMB_STUB, LIMB_IMAGE_NAME), (RINGS_STUB, RINGS_IMAGE_NAME)]
"""The cohort's two navigated images, by results path stub and calibrated name."""

NAVIGATED_IDS = ['limb image', 'ring image']
"""Test ids for :data:`NAVIGATED_IMAGES`, in the same order."""

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the paths below use."""


def _label_cohort_image(cohort: Cohort, tmp_path: Path, stub: str, image_name: str) -> Path:
    """Run the labels pass over one cohort image, returning where its data label goes.

    Parameters:
        cohort: The session's cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which cohort image, by its results path stub.
        image_name: That image's calibrated name.

    Returns:
        The path of the image's data label in the bundle.
    """
    env = make_cohort_bundle_env(cohort, tmp_path)
    generate_bundle_data_files(
        env.dataset,
        cohort.batch(stub),
        nav_results_root=FCPath(cohort.nav_results_root),
        backplane_results_root=FCPath(cohort.backplane_results_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    return env.bundle_dir / 'data' / f'{_product_stem(image_name)}_backplanes.lblx'


def _text(element: ElementTree.Element, path: str) -> str:
    """Return the text of the one element at a path below another, stripped.

    Parameters:
        element: The element the path starts from.
        path: An ElementTree path whose steps carry the ``pds`` prefix.

    Returns:
        The element's text, without surrounding whitespace.

    Raises:
        LookupError: If no element is at the path, or it holds no text.
    """
    found = element.find(path, PDS4_NAMESPACES)
    if found is None or found.text is None:
        raise LookupError(f'no text at {path!r}')
    return found.text.strip()


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_the_files_a_cohort_data_label_names_are_the_ones_beside_it(
    mini_nav_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """A data label names its FITS and supplemental file beside it, and states the copy.

    A PDS4 label names a file with no directory part, so every file it names has to
    be in the label's own directory; and the size and checksum it states for the FITS
    are the ones the copy in the bundle has, read from the copy here.
    """
    label = _label_cohort_image(mini_nav_cohort, tmp_path, stub, image_name)
    root = ElementTree.parse(label).getroot()
    names = [element.text for element in root.iterfind('.//pds:file_name', PDS4_NAMESPACES)]
    product = _product_stem(image_name).rsplit('/', 1)[1]
    assert names == [f'{product}_backplanes.fits', f'{product}_supplemental.txt']
    assert [name for name in names if not (label.parent / str(name)).is_file()] == []
    fits_copy = label.parent / f'{product}_backplanes.fits'
    fits_file = 'pds:File_Area_Observational/pds:File'
    assert int(_text(root, f'{fits_file}/pds:file_size')) == fits_copy.stat().st_size
    md5 = hashlib.md5(fits_copy.read_bytes(), usedforsecurity=False).hexdigest()
    assert _text(root, f'{fits_file}/pds:md5_checksum') == md5
