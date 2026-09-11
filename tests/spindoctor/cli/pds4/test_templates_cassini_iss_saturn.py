"""The shipped Cassini ISS Saturn templates, rendered by the registered dataset.

What a label rendered from the templates the package ships for the Cassini ISS
Saturn bundle says, over the bundle's cohort or over hand-made inputs: that the
shipped draft templates render without substitution errors, that the cohort's
navigated images land in their shards and the one that did not navigate is
skipped, and that the collection inventory names the LIDs the labels do.  The
plumbing these rest on is tested over stand-in templates in
``test_bundle_data.py`` and ``test_collections.py``.

The shipped templates are drafts: these tests assert what renders and where it
lands, never PDS4-standard content correctness of the draft labels.
"""

import json
import re
from pathlib import Path

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import (
    GATED_STUB,
    LIMB_IMAGE_NAME,
    LIMB_STUB,
    RINGS_IMAGE_NAME,
    RINGS_STUB,
    CohortCassiniISSSaturn,
)

from spindoctor.cli.pds4.bundle_data import BundleDataOutcome, generate_bundle_data_files
from spindoctor.cli.pds4.collections import generate_collection_files
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .conftest import (
    A_RANGE,
    make_cohort_bundle_env,
    make_image_file,
    navigated_document,
    read_tab,
    touch_label,
    write_backplane_fits,
)


def _cassini_dataset(tmp_path: Path) -> DataSetPDS3CassiniISSSaturn:
    """Construct the registered Cassini ISS Saturn dataset over an empty holdings root.

    Parameters:
        tmp_path: Base temporary directory the holdings root is made under.

    Returns:
        The dataset.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def _label_with_the_shipped_templates(
    tmp_path: Path, *, fits_shape: tuple[int, int] = (2, 2)
) -> Path:
    """Run the labels pass over one Cassini image with the shipped templates.

    Parameters:
        tmp_path: Base temporary directory for every root the pass reads and writes.
        fits_shape: The lines and samples of the backplane FITS's one plane.

    Returns:
        The bundle's own directory.
    """
    dataset = _cassini_dataset(tmp_path)
    stub = 'COISS_2001/N1454725799_1'
    image_file = make_image_file('N1454725799_1', results_path_stub=stub, base_dir=tmp_path)
    nav_root = tmp_path / 'nav'
    backplane_root = tmp_path / 'backplanes'
    bundle_results_root = tmp_path / 'bundle'
    (nav_root / 'COISS_2001').mkdir(parents=True)
    (backplane_root / 'COISS_2001').mkdir(parents=True)
    bundle_results_root.mkdir()
    nav_metadata = navigated_document()
    (nav_root / f'{stub}_metadata.json').write_text(json.dumps(nav_metadata), encoding='utf-8')
    (backplane_root / f'{stub}_backplane_metadata.json').write_text(
        json.dumps({'bodies': {}, 'rings': {}}), encoding='utf-8'
    )
    write_backplane_fits(backplane_root / f'{stub}_backplanes.fits', shape=fits_shape)
    (nav_root / f'{stub}_summary.png').write_bytes(b'\x89PNG fake bytes')

    generate_bundle_data_files(
        dataset,
        ImageFiles(image_files=[image_file]),
        nav_results_root=FCPath(nav_root),
        backplane_results_root=FCPath(backplane_root),
        bundle_results_root=FCPath(bundle_results_root),
        logger=MAIN_LOGGER,
    )
    return bundle_results_root / 'cassini_iss_saturn_backplanes_rsfrench2027'


def test_cassini_end_to_end_with_shipped_draft_templates(tmp_path: Path) -> None:
    """Phase 1 renders the shipped draft Cassini templates without substitution errors.

    Structural only: asserts the output files exist, the LID substitution took,
    and no pdstemplate error markers ([[[...]]]) are embedded.  PDS4-standard
    content correctness of the draft templates is out of scope until the
    templates are finalized.
    """
    bundle_dir = _label_with_the_shipped_templates(tmp_path)
    label = bundle_dir / 'data' / '1454xxxxxx' / '145472xxxx' / '1454725799n_backplanes.lblx'
    assert label.is_file()
    text = label.read_text(encoding='utf-8')
    lid = 'urn:nasa:pds:cassini_iss_saturn_backplanes_rsfrench2027:data:1454725799n'
    assert lid in text
    assert '[[[' not in text
    browse_label = bundle_dir / 'browse' / '1454xxxxxx' / '145472xxxx' / '1454725799n_summary.lblx'
    assert browse_label.is_file()
    browse_text = browse_label.read_text(encoding='utf-8')
    assert '[[[' not in browse_text
    suppl = bundle_dir / 'data' / '1454xxxxxx' / '145472xxxx' / '1454725799n_supplemental.txt'
    assert suppl.is_file()


def test_the_shipped_data_label_states_a_plane_s_lines_and_samples_as_its_fits_does(
    tmp_path: Path,
) -> None:
    """Over a plane 2 lines by 3 samples, the shipped label's axes say 2 and 3, in order.

    Every frame of the cohort is square, so an exchange of the two axes in the
    template would pass there unnoticed; this plane is not.
    """
    bundle_dir = _label_with_the_shipped_templates(tmp_path, fits_shape=(2, 3))
    label = bundle_dir / 'data' / '1454xxxxxx' / '145472xxxx' / '1454725799n_backplanes.lblx'
    text = label.read_text(encoding='utf-8')
    axes = re.findall(r'<axis_name>(\w+)</axis_name>\s*<elements>(\d+)</elements>', text)
    assert axes == [('Line', '2'), ('Sample', '3')]


def test_the_cohort_image_that_did_not_navigate_is_skipped(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """An image the bundle has nothing to describe is skipped, not failed.

    Over the registered dataset and the shipped templates rather than
    stand-ins, because a selection made by volume routinely names images that
    did not navigate, and what the bundle does with one is a property of the
    run rather than of a fixture's ``status`` key.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    outcome = generate_bundle_data_files(
        env.dataset,
        cassini_cohort.batch(GATED_STUB),
        nav_results_root=FCPath(cassini_cohort.nav_results_root),
        backplane_results_root=FCPath(cassini_cohort.backplane_results_root),
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
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """The shipped Cassini templates render over the cohort, into their shards.

    This is the assertion the cohort exists to make possible and the one every
    phase after this builds on: the registered dataset, the shipped template
    set and a real backplane FITS, with nothing standing in for anything.  A
    template the fixture cannot satisfy, a product written under the wrong
    number, or a render that fails and leaves half a bundle behind is reported
    here, in the phase that owns the fixture.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    outcomes = {
        stub: generate_bundle_data_files(
            env.dataset,
            cassini_cohort.batch(stub),
            nav_results_root=FCPath(cassini_cohort.nav_results_root),
            backplane_results_root=FCPath(cassini_cohort.backplane_results_root),
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
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str, start: str, stop: str
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
        cassini_cohort: The session's Cassini ISS Saturn cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which cohort image, by its results path stub.
        image_name: That image's calibrated name.
        start: What its data label's ``start_date_time`` has to say.
        stop: What its data label's ``stop_date_time`` has to say.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    generate_bundle_data_files(
        env.dataset,
        cassini_cohort.batch(stub),
        nav_results_root=FCPath(cassini_cohort.nav_results_root),
        backplane_results_root=FCPath(cassini_cohort.backplane_results_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    label = env.bundle_dir / 'data' / f'{_product_stem(image_name)}_backplanes.lblx'
    text = label.read_text(encoding='utf-8')
    assert re.findall(r'<start_date_time>(.*)</start_date_time>', text) == [start]
    assert re.findall(r'<stop_date_time>(.*)</stop_date_time>', text) == [stop]


def test_a_navigated_image_that_recorded_no_exposure_times_fails_with_nothing_written(
    cassini_cohort: Cohort, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A success document with no times fails its image, and the bundle stays as it was.

    A navigation that recorded no pointing recorded no exposure times either, and a data
    label states when its exposure began and ended.  The document is the cohort's limb
    image's as written, its times and pointing taken out, under a navigation root of the
    test's own.
    """
    cohort_document = cassini_cohort.nav_results_root / f'{LIMB_STUB}_metadata.json'
    document = json.loads(cohort_document.read_text(encoding='utf-8'))
    del document['navigation_result']['times']
    del document['navigation_result']['pointing']
    written = tmp_path / 'nav' / f'{LIMB_STUB}_metadata.json'
    written.parent.mkdir(parents=True)
    written.write_text(json.dumps(document), encoding='utf-8')
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    outcome = generate_bundle_data_files(
        env.dataset,
        cassini_cohort.batch(LIMB_STUB),
        nav_results_root=FCPath(tmp_path / 'nav'),
        backplane_results_root=FCPath(cassini_cohort.backplane_results_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )
    assert outcome is BundleDataOutcome.FAILED
    assert not env.bundle_dir.exists()
    assert 'its navigation recorded no exposure times' in capsys.readouterr().out


def test_cassini_inventory_lidvid_matches_label_lid(tmp_path: Path) -> None:
    """The Cassini collection inventory LIDVID matches the label's DATA_LID."""
    dataset = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')
    bundle_results_root = tmp_path / 'bundle'
    bundle_dir = bundle_results_root / dataset.pds4_bundle_name()
    touch_label(bundle_dir / 'data', '1454xxxxxx/145472xxxx/1454725799n')
    generate_collection_files(FCPath(bundle_results_root), dataset, MAIN_LOGGER, epochs=A_RANGE)
    rows = read_tab(bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = rows[1][1].split('::')[0]
    label_lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert inventory_lid == label_lid
