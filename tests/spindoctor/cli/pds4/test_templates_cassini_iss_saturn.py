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
from pathlib import Path
from typing import Any

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

from .conftest import make_cohort_bundle_env, make_image_file, read_tab, touch_label


def _cassini_dataset(tmp_path: Path) -> DataSetPDS3CassiniISSSaturn:
    """Construct the registered Cassini ISS Saturn dataset on a local, empty holdings root.

    Parameters:
        tmp_path: Base temporary directory the holdings root is made under.

    Returns:
        The dataset.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def test_cassini_end_to_end_with_shipped_draft_templates(tmp_path: Path) -> None:
    """Phase 1 renders the shipped draft Cassini templates without substitution errors.

    Structural only: asserts the output files exist, the LID substitution took,
    and no pdstemplate error markers ([[[...]]]) are embedded.  PDS4-standard
    content correctness of the draft templates is out of scope until the
    templates are finalized.
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
    nav_metadata: dict[str, Any] = {
        'status': 'success',
        'observation': {
            'start_time': '2007-01-01T00:00:00Z',
            'stop_time': '2007-01-01T00:00:10Z',
            'mid_time': '2007-01-01T00:00:05Z',
        },
    }
    (nav_root / f'{stub}_metadata.json').write_text(json.dumps(nav_metadata), encoding='utf-8')
    (backplane_root / f'{stub}_backplane_metadata.json').write_text(
        json.dumps({'bodies': {}, 'rings': {}}), encoding='utf-8'
    )
    (backplane_root / f'{stub}_backplanes.fits').write_bytes(b'FAKE FITS BYTES')
    (nav_root / f'{stub}_summary.png').write_bytes(b'\x89PNG fake bytes')

    generate_bundle_data_files(
        dataset,
        ImageFiles(image_files=[image_file]),
        nav_results_root=FCPath(nav_root),
        backplane_results_root=FCPath(backplane_root),
        bundle_results_root=FCPath(bundle_results_root),
        logger=MAIN_LOGGER,
    )

    bundle_dir = bundle_results_root / 'cassini_iss_saturn_backplanes_rsfrench2027'
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


def _bundle_products(image_name: str) -> set[str]:
    """Return every bundle file the labels pass writes for one navigated image.

    The sharded directories and the product stem are spelled out here rather
    than asked of the dataset, since what they are is what this test is for: a
    bundle is read by walking those directories, and an image that lands in the
    wrong one is found by whoever cannot find it.

    This is what the pass writes, not everything the bundle finally holds: the
    data label names a ``_backplanes.fits`` beside it that nothing copies yet,
    which ``test_backplane_fits_copied_into_bundle_data_tree`` pins as expected
    to fail.  A set that included the FITS would fail here rather than there.

    Parameters:
        image_name: The calibrated image's name, camera letter and all.

    Returns:
        The paths, relative to the bundle's own directory.
    """
    number = image_name[1:11]
    stem = f'{number[:4]}xxxxxx/{number[:6]}xxxx/{number}{image_name[0].lower()}'
    return {
        f'data/{stem}_backplanes.lblx',
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


def test_cassini_inventory_lidvid_matches_label_lid(tmp_path: Path) -> None:
    """The Cassini collection inventory LIDVID matches the label's DATA_LID."""
    dataset = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')
    bundle_results_root = tmp_path / 'bundle'
    bundle_dir = bundle_results_root / dataset.pds4_bundle_name()
    touch_label(bundle_dir / 'data', '1454xxxxxx/145472xxxx/1454725799n')
    generate_collection_files(FCPath(bundle_results_root), dataset, MAIN_LOGGER)
    rows = read_tab(bundle_dir / 'data' / 'collection_data.tab')
    inventory_lid = rows[1][1].split('::')[0]
    label_lid = dataset.pds4_image_name_to_data_lid('N1454725799')
    assert inventory_lid == label_lid
