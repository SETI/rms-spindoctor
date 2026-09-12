"""The shipped Cassini ISS Saturn templates, rendered by the registered dataset.

What a label rendered from the templates the package ships for the Cassini ISS
Saturn bundle says, over the bundle's cohort or over hand-made inputs: that the
shipped draft templates render without substitution errors, that the cohort's
navigated images land in their shards and the one that did not navigate is
skipped, that a data label states its exposure's start and stop and an image
whose navigation recorded none fails, that a data label describes the backplane
FITS beside it, HDU by HDU and byte for byte, and that the collection inventory
names the LIDs the labels do.  The plumbing these rest on is tested over stand-in
templates in ``test_bundle_data.py`` and ``test_collections.py``.

The shipped templates are drafts.  Apart from the data objects describing the FITS,
which are held to the file's bytes and to the schema's order, these tests assert
what renders and where it lands, not the PDS4-standard correctness of the draft
labels.
"""

import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from astropy.io import fits
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
from spindoctor.config import DEFAULT_CONFIG, MAIN_LOGGER
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
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """A data label names its FITS and supplemental file beside it, and states the copy.

    A PDS4 label names a file with no directory part, so every file it names has to
    be in the label's own directory; and the size and checksum it states for the FITS
    are the ones the copy in the bundle has, read from the copy here.
    """
    label = _label_cohort_image(cassini_cohort, tmp_path, stub, image_name)
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


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_a_cohort_data_label_describes_its_supplemental_file_as_the_text_it_is(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """The supplemental file is one text stream, as its bytes say it is.

    The label's Stream_Text starts at the file's first byte and runs its whole length;
    every byte is 7-bit ASCII, as its parsing standard says; each line ends in a line
    feed alone, as its record delimiter says; and the text is one JSON object.
    """
    label = _label_cohort_image(cassini_cohort, tmp_path, stub, image_name)
    root = ElementTree.parse(label).getroot()
    area = 'pds:File_Area_Observational_Supplemental'
    raw = (label.parent / _text(root, f'{area}/pds:File/pds:file_name')).read_bytes()
    stream = f'{area}/pds:Stream_Text'
    assert int(_text(root, f'{stream}/pds:offset')) == 0
    assert int(_text(root, f'{stream}/pds:object_length')) == len(raw)
    assert _text(root, f'{stream}/pds:parsing_standard_id') == '7-Bit ASCII Text'
    assert max(raw) < 128
    assert _text(root, f'{stream}/pds:record_delimiter') == 'Line-Feed'
    assert b'\r' not in raw
    assert isinstance(json.loads(raw), dict)


# ---------------------------------------------------------------------------
# The data objects: each checked against the FITS bytes and astropy, not the builder
# ---------------------------------------------------------------------------

FITS_BLOCK = 2880
"""The length of a FITS record; every header and every data unit fills whole ones."""

END_CARD = b'END' + b' ' * 77
"""The card that ends a FITS header."""

DATA_TYPES = {-32: 'IEEE754MSBSingle', 32: 'SignedMSB4'}
"""The PDS4 data type of a FITS array, by its BITPIX: big-endian, as FITS stores it."""

NUMPY_TYPES = {'IEEE754MSBSingle': '>f4', 'SignedMSB4': '>i4'}
"""The numpy type each PDS4 data type names."""


def _labelled_fits(
    cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> tuple[ElementTree.Element, Path]:
    """Label one cohort image, returning the label's root and the FITS it names.

    Parameters:
        cohort: The session's cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which cohort image, by its results path stub.
        image_name: That image's calibrated name.

    Returns:
        The data label's root element, and the path of the FITS beside it.
    """
    label = _label_cohort_image(cohort, tmp_path, stub, image_name)
    root = ElementTree.parse(label).getroot()
    file_name = _text(root, 'pds:File_Area_Observational/pds:File/pds:file_name')
    return root, label.parent / file_name


def _data_objects(root: ElementTree.Element, tag: str) -> list[ElementTree.Element]:
    """Return a label's data objects of one kind, in the order the label states them.

    Parameters:
        root: The label's root element.
        tag: The object's element name in the PDS4 namespace, ``Header`` for one.

    Returns:
        Every such element directly under ``File_Area_Observational``.
    """
    return root.findall(f'pds:File_Area_Observational/pds:{tag}', PDS4_NAMESPACES)


def _read_array(raw: bytes, array: ElementTree.Element) -> np.ndarray:
    """Read one array out of a file's bytes, as its ``Array_2D_Image`` describes it.

    Parameters:
        raw: The whole file.
        array: The array's element in the label.

    Returns:
        The ``Line`` by ``Sample`` elements at the array's offset, of its data type.
    """
    axes = {
        _text(axis, 'pds:axis_name'): int(_text(axis, 'pds:elements'))
        for axis in array.iterfind('pds:Axis_Array', PDS4_NAMESPACES)
    }
    data_type = NUMPY_TYPES[_text(array, 'pds:Element_Array/pds:data_type')]
    count = axes['Line'] * axes['Sample']
    offset = int(_text(array, 'pds:offset'))
    return np.frombuffer(raw, dtype=data_type, count=count, offset=offset).reshape(
        axes['Line'], axes['Sample']
    )


def _end_card_record(header: bytes) -> int:
    """Return which record of a header region holds its first ``END`` card.

    Parameters:
        header: The bytes a label says one header takes.

    Returns:
        The record's index from zero, or -1 when no card of the region is ``END``.
    """
    cards = [header[start : start + len(END_CARD)] for start in range(0, len(header), 80)]
    if END_CARD not in cards:
        return -1
    return cards.index(END_CARD) * 80 // FITS_BLOCK


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_a_cohort_data_label_describes_each_hdu_where_the_fits_holds_it(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """One header per HDU and one array per image HDU, each where its bytes are.

    At each header's offset the file reads ``SIMPLE`` or ``XTENSION``, and the length
    stated for it runs to the end of the record holding its ``END`` card.  Read out of
    the file's bytes at each array's offset, as its stated lines, samples and type, each
    array is the one astropy reads for its HDU, and its type and unit are the HDU's.
    """
    root, fits_copy = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    raw = fits_copy.read_bytes()
    with fits.open(fits_copy) as hdul:
        hdu_count = len(hdul)
        names = [hdu.name.lower() for hdu in hdul[1:]]
        values = [hdu.data.tolist() for hdu in hdul[1:]]
        data_types = [DATA_TYPES[hdu.header['BITPIX']] for hdu in hdul[1:]]
        units = [hdu.header.get('BUNIT') for hdu in hdul[1:]]
    headers = [
        raw[offset : offset + length]
        for offset, length in (
            (int(_text(header, 'pds:offset')), int(_text(header, 'pds:object_length')))
            for header in _data_objects(root, 'Header')
        )
    ]
    arrays = _data_objects(root, 'Array_2D_Image')
    assert [header[:8] for header in headers] == [b'SIMPLE  '] + [b'XTENSION'] * (hdu_count - 1)
    assert [len(header) % FITS_BLOCK for header in headers] == [0] * hdu_count
    last_records = [len(header) // FITS_BLOCK - 1 for header in headers]
    assert [_end_card_record(header) for header in headers] == last_records
    assert [_text(array, 'pds:local_identifier') for array in arrays] == names
    assert [_read_array(raw, array).tolist() for array in arrays] == values
    assert [_text(array, 'pds:Element_Array/pds:data_type') for array in arrays] == data_types
    stated_units = [
        array.findtext('pds:Element_Array/pds:unit', namespaces=PDS4_NAMESPACES) for array in arrays
    ]
    assert stated_units == units


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_a_cohort_data_label_declares_the_masked_value_its_float_arrays_hold(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """Every float array declares the configured masked value, and holds it where masked.

    The body identity map declares none.  Which pixels a plane measured nothing at is
    read from the identity map rather than from the label: in a cohort frame one body
    claims a disc and its rings every other pixel, so a body plane is masked where no
    body claimed the pixel and a ring plane where one did.  Every masked pixel holds the
    declared value, and no measured one does.
    """
    root, fits_copy = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    raw = fits_copy.read_bytes()
    masked_value = float(DEFAULT_CONFIG.backplanes.masked_value)
    arrays = {
        _text(array, 'pds:local_identifier'): array
        for array in _data_objects(root, 'Array_2D_Image')
    }
    body_id_map = arrays.pop('body_id_map')
    assert body_id_map.find('pds:Special_Constants', PDS4_NAMESPACES) is None
    floats = {
        name: array
        for name, array in arrays.items()
        if _text(array, 'pds:Element_Array/pds:data_type') == 'IEEE754MSBSingle'
    }
    assert sorted(floats) == sorted(arrays)
    constants = {
        name: float(_text(array, 'pds:Special_Constants/pds:missing_constant'))
        for name, array in floats.items()
    }
    assert constants == dict.fromkeys(floats, masked_value)
    claimed = _read_array(raw, body_id_map) != 0
    planes = {name: _read_array(raw, array) for name, array in floats.items()}
    masked = {name: claimed if name.startswith('ring_') else ~claimed for name in floats}
    held = {name: sorted(set(planes[name][masked[name]].tolist())) for name in floats}
    assert held == {name: [masked_value] for name in floats}
    measured_as_masked = [
        name for name in floats if bool((planes[name][~masked[name]] == masked_value).any())
    ]
    assert measured_as_masked == []


@pytest.mark.parametrize(
    ('stub', 'image_name', 'has_rings'),
    [(LIMB_STUB, LIMB_IMAGE_NAME, False), (RINGS_STUB, RINGS_IMAGE_NAME, True)],
    ids=NAVIGATED_IDS,
)
def test_only_the_ring_image_s_data_label_describes_ring_arrays(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str, has_rings: bool
) -> None:
    """The ring image's label has one array per configured ring plane, the limb's none.

    Parameters:
        cassini_cohort: The session's Cassini ISS Saturn cohort.
        tmp_path: Base temporary directory for this test's bundle.
        stub: Which cohort image, by its results path stub.
        image_name: That image's calibrated name.
        has_rings: Whether the cohort gives the image ring backplanes.
    """
    root, _ = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    identifiers = [
        _text(array, 'pds:local_identifier') for array in _data_objects(root, 'Array_2D_Image')
    ]
    # The writer never writes a ring plane named distance as an array of its own.
    configured = [
        entry['name'] for entry in DEFAULT_CONFIG.backplanes.rings if entry['name'] != 'distance'
    ]
    expected = sorted(configured) if has_rings else []
    assert sorted(name for name in identifiers if name.startswith('ring_')) == expected


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_each_array_of_a_cohort_data_label_has_display_settings_that_resolve(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """One display settings block per array, each naming an identifier the label defines.

    Every ``local_identifier`` in the label is its only one of that name, every
    ``local_identifier_reference`` names one of them, and the references are the
    arrays', one each, in the order the arrays are described.
    """
    root, _ = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    identifiers = [
        element.text for element in root.iterfind('.//pds:local_identifier', PDS4_NAMESPACES)
    ]
    references = [
        element.text
        for element in root.iterfind('.//pds:local_identifier_reference', PDS4_NAMESPACES)
    ]
    arrays = [
        _text(array, 'pds:local_identifier') for array in _data_objects(root, 'Array_2D_Image')
    ]
    assert len(identifiers) == len(set(identifiers))
    assert [reference for reference in references if reference not in identifiers] == []
    assert references == arrays


def _children(element: ElementTree.Element) -> list[str]:
    """Return the names of an element's children, without their namespace.

    Parameters:
        element: The element.

    Returns:
        Each child's local name, in document order.
    """
    return [child.tag.rsplit('}', 1)[-1] for child in element]


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_each_data_object_of_a_cohort_data_label_is_in_the_schema_s_shape(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """Each header and array holds the children PDS4_PDS_1O00 allows, in its order.

    The order and the fixed values are the schema's: a ``Header`` is its name, offset,
    length and parsing standard; an ``Array_2D_Image`` is its identifier, offset, axes,
    index order, a description where it has one, its element, two axes and its special
    constants where it has them; the parsing standard is one the Schematron names for
    FITS and the index order the one it allows.  The ``File`` comes first.
    """
    root, _ = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    file_area = root.find('pds:File_Area_Observational', PDS4_NAMESPACES)
    assert file_area is not None
    assert _children(file_area)[0] == 'File'
    headers = _data_objects(root, 'Header')
    assert [_children(header) for header in headers] == [
        ['name', 'offset', 'object_length', 'parsing_standard_id']
    ] * len(headers)
    assert {_text(header, 'pds:parsing_standard_id') for header in headers} == {'FITS 3.0'}
    arrays = _data_objects(root, 'Array_2D_Image')
    expected = []
    for array in arrays:
        is_body_id_map = _text(array, 'pds:local_identifier') == 'body_id_map'
        expected.append(
            ['local_identifier', 'offset', 'axes', 'axis_index_order', 'description']
            + ['Element_Array', 'Axis_Array', 'Axis_Array']
            + ([] if is_body_id_map else ['Special_Constants'])
        )
    assert [_children(array) for array in arrays] == expected
    assert {_text(array, 'pds:axis_index_order') for array in arrays} == {'Last Index Fastest'}
    assert {_text(array, 'pds:axes') for array in arrays} == {'2'}
    axes = [
        [(_text(axis, 'pds:axis_name'), _text(axis, 'pds:sequence_number')) for axis in each]
        for each in (array.findall('pds:Axis_Array', PDS4_NAMESPACES) for array in arrays)
    ]
    assert axes == [[('Line', '1'), ('Sample', '2')]] * len(arrays)
    units = {
        element.get('unit')
        for element in root.iterfind('pds:File_Area_Observational/*/pds:offset', PDS4_NAMESPACES)
    } | {
        element.get('unit')
        for element in root.iterfind(
            'pds:File_Area_Observational/pds:Header/pds:object_length', PDS4_NAMESPACES
        )
    }
    assert units == {'byte'}
    supplemental = root.find('pds:File_Area_Observational_Supplemental', PDS4_NAMESPACES)
    assert supplemental is not None
    assert _children(supplemental) == ['File', 'Stream_Text']
    stream = supplemental.find('pds:Stream_Text', PDS4_NAMESPACES)
    assert stream is not None
    assert _children(stream) == [
        'offset',
        'object_length',
        'parsing_standard_id',
        'description',
        'record_delimiter',
    ]


@pytest.mark.parametrize(('stub', 'image_name'), NAVIGATED_IMAGES, ids=NAVIGATED_IDS)
def test_every_float_array_of_a_cohort_data_label_says_what_it_holds(
    cassini_cohort: Cohort, tmp_path: Path, stub: str, image_name: str
) -> None:
    """Each float array's description names its method and unit, and the missing constant.

    The method is the one the shipped configuration names for the plane, the unit the
    HDU's ``BUNIT`` as astropy reads it, and the constant the configured masked value.
    """
    root, fits_copy = _labelled_fits(cassini_cohort, tmp_path, stub, image_name)
    methods = {
        entry['name']: entry['method']
        for entry in [*DEFAULT_CONFIG.backplanes.bodies, *DEFAULT_CONFIG.backplanes.rings]
    }
    with fits.open(fits_copy) as hdul:
        units = {hdu.name.lower(): hdu.header.get('BUNIT') for hdu in hdul[1:]}
    descriptions = {
        _text(array, 'pds:local_identifier'): _text(array, 'pds:description')
        for array in _data_objects(root, 'Array_2D_Image')
        if _text(array, 'pds:Element_Array/pds:data_type') == 'IEEE754MSBSingle'
    }
    assert sorted(descriptions) == sorted(name for name in units if name != 'body_id_map')
    without_method = [
        name
        for name, text in descriptions.items()
        if f'from the oops backplane method {methods[name]},' not in text
    ]
    assert without_method == []
    without_unit = [
        name for name, text in descriptions.items() if f', in {units[name]}.' not in text
    ]
    assert without_unit == []
    constant = float(DEFAULT_CONFIG.backplanes.masked_value)
    without_constant = [
        name
        for name, text in descriptions.items()
        if f'holds the missing constant, {constant!r}.' not in text
    ]
    assert without_constant == []
