"""Spec-first tests for PDS4 per-image bundle data generation (phase 1).

Contract under test (docs/user_guide/user_guide_pds4_bundle.rst "Labels Pass" /
"Inputs and Outputs" and docs/dev_guide/dev_guide_pds4.rst "Pipeline overview"):
``generate_bundle_data_files`` takes a one-image batch, reads the image's
``_metadata.json`` and ``_backplane_metadata.json``, and writes into
``<bundle_results_root>/<bundle name>/`` a ``data/<stub>_backplanes.lblx`` label
rendered from the dataset's ``data.lblx`` template, a
``data/<stub>_supplemental.txt`` JSON file combining both metadata dicts, and
(browse products being optional) a ``browse/<stub>_summary.png`` copy plus a
``browse/<stub>_summary.lblx`` label when the navigation summary PNG exists.
Non-navigated images (``status`` != ``success``) are skipped with a warning.
The per-dataset ``pds4_*`` hooks parameterize the layout, the LIDs, and the
template variables; datasets without PDS4 support raise ``NotImplementedError``.

The shipped Cassini templates are drafts: tests below assert substitution and
layout plumbing, never PDS4-standard content correctness of the draft labels.
"""

import json
import re
from pathlib import Path
from typing import Any

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
from spindoctor.config import MAIN_LOGGER, Config
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn
from spindoctor.dataset.dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS

from .conftest import (
    DATA_TEMPLATE,
    BundleEnv,
    NoPds4DataSet,
    make_bundle_env,
    make_cohort_bundle_env,
    make_image_file,
    write_nav_inputs,
)

BROKEN_TEMPLATE_BODY = '$COMPLETELY_UNSET_VARIABLE$'
"""An expression naming a variable no caller defines, so the render errors."""


def _generate(env: BundleEnv) -> BundleDataOutcome:
    """Run generate_bundle_data_files over the environment's one-image batch.

    Parameters:
        env: The hermetic bundle environment to process.

    Returns:
        What the generation came to for the environment's one image.
    """
    return generate_bundle_data_files(
        env.dataset.as_dataset(),
        env.image_files,
        nav_results_root=FCPath(env.nav_root),
        backplane_results_root=FCPath(env.backplane_root),
        bundle_results_root=FCPath(env.bundle_results_root),
        logger=MAIN_LOGGER,
    )


# ---------------------------------------------------------------------------
# Batch cardinality
# ---------------------------------------------------------------------------


def test_empty_batch_raises(tmp_path: Path) -> None:
    """A zero-image batch is rejected with a ValueError."""
    env = make_bundle_env(tmp_path)
    with pytest.raises(ValueError, match='Expected exactly one image per batch; got 0'):
        generate_bundle_data_files(
            env.dataset.as_dataset(),
            ImageFiles(image_files=[]),
            nav_results_root=FCPath(env.nav_root),
            backplane_results_root=FCPath(env.backplane_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )


def test_two_image_batch_raises(tmp_path: Path) -> None:
    """A two-image batch is rejected with a ValueError."""
    env = make_bundle_env(tmp_path)
    batch = ImageFiles(image_files=[make_image_file('1111111111n'), make_image_file('2222222222w')])
    with pytest.raises(ValueError, match='Expected exactly one image per batch; got 2'):
        generate_bundle_data_files(
            env.dataset.as_dataset(),
            batch,
            nav_results_root=FCPath(env.nav_root),
            backplane_results_root=FCPath(env.backplane_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )


# ---------------------------------------------------------------------------
# Success path: supplemental file, data label, browse products
# ---------------------------------------------------------------------------


def test_supplemental_file_written_under_bundle_data(tmp_path: Path) -> None:
    """The supplemental file lands at data/<pds4 stub>_supplemental.txt."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    assert suppl.is_file()


def test_supplemental_combines_navigation_and_backplane_metadata(tmp_path: Path) -> None:
    """The supplemental JSON has 'navigation' and 'backplanes' sections verbatim."""
    env = make_bundle_env(tmp_path)
    nav_metadata, backplane_metadata = write_nav_inputs(
        env,
        nav_extra={'offset': {'dv': 1.5, 'du': -2.0}},
        backplane_metadata={'bodies': {'MIMAS': {'backplanes': {}}}, 'rings': {}},
    )
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    combined = json.loads(suppl.read_text(encoding='utf-8'))
    assert combined['navigation'] == nav_metadata
    assert combined['backplanes'] == backplane_metadata


def test_data_label_rendered_with_substituted_variables(tmp_path: Path) -> None:
    """The data label renders with all variables substituted, and the image is written."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.WRITTEN
    label = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.lblx'
    text = label.read_text(encoding='utf-8')
    assert 'urn:nasa:pds:fake_bundle:data:1234567890w' in text
    assert '$' not in text


def test_injected_file_path_template_variables(tmp_path: Path) -> None:
    """The stage injects the BACKPLANE_*/BROWSE_FULL_* file variables into the dict."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    _generate(env)
    variables = env.dataset.template_variables
    assert variables['BACKPLANE_FILENAME'] == '1234567890w_backplanes.fits'
    assert variables['BACKPLANE_SUPPL_FILENAME'] == '1234567890w_supplemental.txt'
    assert variables['BROWSE_FULL_FILENAME'] == '1234567890w_summary.png'
    expected_fits = str(FCPath(env.backplane_root) / f'{env.results_path_stub}_backplanes.fits')
    assert variables['BACKPLANE_PATH'] == expected_fits
    expected_suppl = str(FCPath(env.bundle_dir) / 'data' / f'{env.pds4_path_stub}_supplemental.txt')
    assert variables['BACKPLANE_SUPPL_PATH'] == expected_suppl
    expected_png = str(FCPath(env.bundle_dir) / 'browse' / f'{env.pds4_path_stub}_summary.png')
    assert variables['BROWSE_FULL_PATH'] == expected_png


def test_template_variables_hook_receives_both_metadata_dicts(tmp_path: Path) -> None:
    """pds4_template_variables gets the image file and both parsed metadata dicts."""
    env = make_bundle_env(tmp_path)
    nav_metadata, backplane_metadata = write_nav_inputs(env)
    _generate(env)
    assert len(env.dataset.template_variables_calls) == 1
    call = env.dataset.template_variables_calls[0]
    assert call['image_file'] is env.image_file
    assert call['nav_metadata'] == nav_metadata
    assert call['backplane_metadata'] == backplane_metadata


def test_browse_png_copied_byte_identical(tmp_path: Path) -> None:
    """The navigation summary PNG is copied into browse/ byte-for-byte."""
    env = make_bundle_env(tmp_path)
    png_bytes = b'\x89PNG distinctive payload'
    write_nav_inputs(env, summary_png=png_bytes)
    _generate(env)
    copied = env.bundle_dir / 'browse' / f'{env.pds4_path_stub}_summary.png'
    assert copied.read_bytes() == png_bytes


def test_browse_label_rendered(tmp_path: Path) -> None:
    """The browse label is rendered from browse.lblx next to the copied PNG."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    _generate(env)
    label = env.bundle_dir / 'browse' / f'{env.pds4_path_stub}_summary.lblx'
    text = label.read_text(encoding='utf-8')
    assert 'urn:nasa:pds:fake_bundle:browse:1234567890w' in text
    assert '1234567890w_summary.png' in text


def test_missing_summary_png_fails_the_image_and_keeps_the_data_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A success document with no summary PNG beside it fails the image.

    The navigation stage writes the PNG before the document that records the
    success, and both under one condition, so there is no run in which a
    success document legitimately has no PNG beside it.  One that has none is a
    broken input and the bundle stage says so.  The data half succeeded, so its
    label stays, exactly as when a browse template will not render.
    """
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, summary_png=None)
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.FAILED
    data_label = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.lblx'
    assert data_label.is_file()
    browse_label = env.bundle_dir / 'browse' / f'{env.pds4_path_stub}_summary.lblx'
    assert not browse_label.exists()
    browse_png = env.bundle_dir / 'browse' / f'{env.pds4_path_stub}_summary.png'
    assert not browse_png.exists()
    assert 'ERROR | No summary PNG at' in capsys.readouterr().out


def test_unicode_template_variables_round_trip(tmp_path: Path) -> None:
    """Unicode template-variable values survive into the rendered label."""
    note = '\u00c5ngstr\u00f6m \u03bc test'
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'data.lblx': '<Product>\n  <note>$NOTE$</note>\n</Product>\n',
            'browse.lblx': '<Browse>\n  <note>$NOTE$</note>\n</Browse>\n',
        },
        template_variables={'NOTE': note},
    )
    write_nav_inputs(env)
    _generate(env)
    label = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.lblx'
    assert note in label.read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# Skip and error paths
# ---------------------------------------------------------------------------


def test_non_success_status_skips_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-success navigation status skips the image with a warning, and is not a failure."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, status='failure', nav_extra={'status_error': 'no offset found'})
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.SKIPPED
    assert not env.bundle_dir.exists()
    out = capsys.readouterr().out
    assert 'Skipping bundle generation' in out
    assert 'no offset found' in out


def test_missing_status_key_skips_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Navigation metadata without a status key is treated as not navigated."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, status=None)
    _generate(env)
    assert not env.bundle_dir.exists()
    assert 'status=None' in capsys.readouterr().out


def test_missing_nav_metadata_skips_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image with no _metadata.json was never navigated, so it is skipped.

    Absence of a navigation document says the same thing the status field says
    when it is not success, in the other spelling, and a selection made by
    volume names far more images than have been navigated.
    """
    env = make_bundle_env(tmp_path)
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.SKIPPED
    assert 'no navigation metadata at' in capsys.readouterr().out


def test_missing_backplane_metadata_skips_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A navigated image with no backplanes has nothing for the bundle to describe."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    bp_file = env.backplane_root / f'{env.results_path_stub}_backplane_metadata.json'
    bp_file.unlink()
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.SKIPPED
    assert 'no backplane metadata at' in capsys.readouterr().out


def test_malformed_nav_metadata_raises(tmp_path: Path) -> None:
    """Unparseable navigation metadata propagates a JSON decode error.

    A document that is there but will not parse is a defect in that document,
    not an image the bundle has nothing to say about, so it is not a skip.
    """
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    nav_file = env.nav_root / f'{env.results_path_stub}_metadata.json'
    nav_file.write_text('not json at all', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError, match='Expecting value'):
        _generate(env)


def test_missing_template_dir_raises_after_supplemental(tmp_path: Path) -> None:
    """A missing template directory raises, but the supplemental file is already written."""
    env = make_bundle_env(tmp_path)
    env.dataset._template_dir = tmp_path / 'no_such_templates'
    write_nav_inputs(env)
    with pytest.raises(FileNotFoundError, match='File does not exist'):
        _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    assert suppl.is_file()


def test_undefined_template_variable_fails_the_product(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A data template referencing an unset variable writes no label and fails the image.

    pdstemplate reports the NameError through its return value rather than by
    raising, so the returned outcome is the only thing that can tell a caller the
    product is not there.  The log must not say otherwise either.
    """
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'data.lblx': f'<Product>{BROKEN_TEMPLATE_BODY}</Product>\n',
            'browse.lblx': '<Browse>ok</Browse>\n',
        },
        template_variables={},
    )
    write_nav_inputs(env)
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.FAILED
    label = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.lblx'
    assert not label.exists()
    assert 'Generated PDS4 label' not in capsys.readouterr().out


def test_broken_browse_template_fails_the_product_and_keeps_the_data_label(
    tmp_path: Path,
) -> None:
    """A browse template that cannot render is its own failure; the data label stays.

    Both labels are attempted, so one run reports every label it could not write.
    """
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'data.lblx': DATA_TEMPLATE,
            'browse.lblx': f'<Browse>{BROKEN_TEMPLATE_BODY}</Browse>\n',
        },
    )
    write_nav_inputs(env)
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.FAILED
    data_label = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.lblx'
    assert data_label.is_file()
    browse_label = env.bundle_dir / 'browse' / f'{env.pds4_path_stub}_summary.lblx'
    assert not browse_label.exists()


def test_dataset_without_pds4_support_raises(tmp_path: Path) -> None:
    """A dataset without pds4_* hook implementations propagates NotImplementedError."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    with pytest.raises(NotImplementedError, match='not supported for this dataset'):
        generate_bundle_data_files(
            NoPds4DataSet().as_dataset(),
            env.image_files,
            nav_results_root=FCPath(env.nav_root),
            backplane_results_root=FCPath(env.backplane_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )


@pytest.mark.xfail(
    strict=False,
    reason='dev_guide_pds4.rst "Pipeline overview" says the backplane FITS file is '
    'copied (or symlinked) into the bundle data/ tree, but generate_bundle_data_files '
    'never copies it; user_guide_pds4_bundle.rst omits it from the outputs, so this '
    'may be deliberately deferred to template/bundle finalization',
)
def test_backplane_fits_copied_into_bundle_data_tree(tmp_path: Path) -> None:
    """Dev guide: the backplane FITS product is placed in the bundle data/ tree."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    fits_source = env.backplane_root / f'{env.results_path_stub}_backplanes.fits'
    fits_source.write_bytes(b'FAKE FITS BYTES')
    _generate(env)
    bundled_fits = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.fits'
    assert bundled_fits.is_file()


# ---------------------------------------------------------------------------
# Dataset pds4_* hook contract (reference Cassini implementation + walls)
# ---------------------------------------------------------------------------


def _cassini_dataset(
    tmp_path: Path, *, config: Config | None = None
) -> DataSetPDS3CassiniISSSaturn:
    """Construct the reference Cassini Saturn dataset on a local holdings root.

    Parameters:
        tmp_path: Base temporary directory used as the (empty) holdings root.
        config: Optional Config override; DEFAULT_CONFIG when None.
    """
    return DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings', config=config)


def test_cassini_bundle_path_for_image_shards_by_image_number(tmp_path: Path) -> None:
    """Cassini image names shard into 1234xxxxxx/123456xxxx/ directories."""
    dataset = _cassini_dataset(tmp_path)
    assert dataset.pds4_bundle_path_for_image('N1454725799') == '1454xxxxxx/145472xxxx/'


def test_cassini_bundle_path_rejects_short_image_name(tmp_path: Path) -> None:
    """A too-short Cassini image name raises instead of building a malformed path."""
    dataset = _cassini_dataset(tmp_path)
    with pytest.raises(ValueError, match='invalid Cassini image name'):
        dataset.pds4_bundle_path_for_image('N123')


def test_cassini_path_stub_appends_lid_part(tmp_path: Path) -> None:
    """The path stub is the shard path plus the rotated lowercase image LID part."""
    dataset = _cassini_dataset(tmp_path)
    image_file = make_image_file('N1454725799_1')
    assert dataset.pds4_path_stub(image_file) == '1454xxxxxx/145472xxxx/1454725799n'


def test_cassini_default_bundle_name_from_config(tmp_path: Path) -> None:
    """The bundle name comes from the shipped pds4.coiss_saturn config block."""
    dataset = _cassini_dataset(tmp_path)
    assert dataset.pds4_bundle_name() == 'cassini_iss_saturn_backplanes_rsfrench2027'


def test_cassini_default_template_dir_is_shipped_package_data(tmp_path: Path) -> None:
    """The default template dir resolves inside the shipped templates package data."""
    dataset = _cassini_dataset(tmp_path)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert template_dir.name == 'cassini_iss_saturn_1.0'
    assert template_dir.parent.name == 'templates'
    assert (template_dir / 'data.lblx').is_file()
    assert (template_dir / 'browse.lblx').is_file()
    assert (template_dir / 'collection_data.lblx').is_file()
    assert (template_dir / 'global_index_bodies.lblx').is_file()


def test_cassini_config_overrides_template_dir_and_bundle_name(tmp_path: Path) -> None:
    """config pds4.<dataset>.template_dir/bundle_name override the defaults."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        'pds4:\n'
        '  coiss_saturn:\n'
        '    template_dir: /absolute/custom/templates\n'
        '    bundle_name: custom_bundle_name\n',
        encoding='utf-8',
    )
    config = Config()
    config.update_config(override)
    dataset = _cassini_dataset(tmp_path, config=config)
    assert dataset.pds4_bundle_template_dir() == '/absolute/custom/templates'
    assert dataset.pds4_bundle_name() == 'custom_bundle_name'


def test_cassini_relative_template_dir_override_resolves_under_templates(
    tmp_path: Path,
) -> None:
    """A bare-name template_dir override resolves under the packaged templates dir."""
    override = tmp_path / 'override.yaml'
    override.write_text(
        'pds4:\n  coiss_saturn:\n    template_dir: my_custom_set\n', encoding='utf-8'
    )
    config = Config()
    config.update_config(override)
    dataset = _cassini_dataset(tmp_path, config=config)
    template_dir = Path(dataset.pds4_bundle_template_dir())
    assert template_dir.name == 'my_custom_set'
    assert template_dir.parent.name == 'templates'


def test_voyager_pds4_hooks_not_implemented(tmp_path: Path) -> None:
    """The Voyager dataset's per-image PDS4 hooks are NotImplementedError walls.

    The base-class walls raise a bare NotImplementedError, so the assertions
    pin the empty message: a messaged NotImplementedError escaping from deeper
    code would fail them.
    """
    dataset = DataSetPDS3VoyagerISS(tmp_path / 'holdings')
    image_file = make_image_file('C1234567')
    with pytest.raises(NotImplementedError) as stub_exc:
        dataset.pds4_path_stub(image_file)
    assert str(stub_exc.value) == ''
    with pytest.raises(NotImplementedError) as lidvid_exc:
        dataset.pds4_image_name_to_data_lidvid('C1234567')
    assert str(lidvid_exc.value) == ''


# ---------------------------------------------------------------------------
# End-to-end phase 1 against the shipped (draft) Cassini templates
# ---------------------------------------------------------------------------


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


def test_cassini_data_label_lid_matches_dataset_builder(tmp_path: Path) -> None:
    """The DATA_LID template variable equals pds4_image_name_to_data_lid's output."""
    dataset = _cassini_dataset(tmp_path)
    image_file = make_image_file('N1454725799_1')
    variables = dataset.pds4_template_variables(
        image_file=image_file, nav_metadata={}, backplane_metadata={}
    )
    assert variables['DATA_LID'] == dataset.pds4_image_name_to_data_lid('N1454725799_1')
    assert variables['BROWSE_LID'] == dataset.pds4_image_name_to_browse_lid('N1454725799_1')


def test_cassini_camera_variables_from_image_name(tmp_path: Path) -> None:
    """The camera template variables derive from the image name's leading letter."""
    dataset = _cassini_dataset(tmp_path)
    variables = dataset.pds4_template_variables(
        image_file=make_image_file('W1454725799_1'), nav_metadata={}, backplane_metadata={}
    )
    assert variables['CAMERA_WIDTH'] == 'Wide'
    assert variables['CAMERA_WN_UC'] == 'W'
    assert variables['CAMERA_WN_LC'] == 'w'


def test_cassini_lid_charset_is_pds4_legal(tmp_path: Path) -> None:
    """Cassini LIDs are lowercase urn:nasa:pds identifiers with a legal charset."""
    dataset = _cassini_dataset(tmp_path)
    lid = dataset.pds4_image_name_to_data_lid('N1454725799_1.IMG')
    assert lid == lid.lower()
    assert re.fullmatch(r'urn:nasa:pds(:[a-z0-9_-]+)+', lid) is not None


# ---------------------------------------------------------------------------
# A record the enumeration already read
# ---------------------------------------------------------------------------


def test_a_carried_record_is_used_in_place_of_the_document(tmp_path: Path) -> None:
    """A record carried with the image is what the supplemental file records."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, nav_extra={'marker': 'from the document'})
    env.image_file.nav_record = {'status': 'success', 'marker': 'from the enumeration'}
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    combined = json.loads(suppl.read_text(encoding='utf-8'))
    assert combined['navigation']['marker'] == 'from the enumeration'


def test_a_carried_record_is_used_when_the_document_has_gone(tmp_path: Path) -> None:
    """An image carrying a record needs no document under the results root."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    (env.nav_root / f'{env.results_path_stub}_metadata.json').unlink()
    env.image_file.nav_record = {'status': 'success', 'marker': 'from the enumeration'}
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    combined = json.loads(suppl.read_text(encoding='utf-8'))
    assert combined['navigation']['marker'] == 'from the enumeration'


def test_an_image_carrying_no_record_reads_the_document(tmp_path: Path) -> None:
    """With nothing carried, the document under the results root is what is read."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, nav_extra={'marker': 'from the document'})
    assert env.image_file.nav_record is None
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    combined = json.loads(suppl.read_text(encoding='utf-8'))
    assert combined['navigation']['marker'] == 'from the document'


def test_an_image_carrying_no_record_needs_the_document(tmp_path: Path) -> None:
    """With nothing carried and no document, no products are written at all.

    Nothing stands in for the record: an image the enumeration handed over
    without one and whose document is not under the results root is an image
    the run has read no navigation for, so it writes none of its products.
    """
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    (env.nav_root / f'{env.results_path_stub}_metadata.json').unlink()
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.SKIPPED
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    assert not suppl.exists()


def test_the_cohort_image_that_did_not_navigate_is_skipped(
    mini_nav_cohort: Cohort, tmp_path: Path
) -> None:
    """An image the bundle has nothing to describe is skipped, not failed.

    Over the registered dataset and the shipped templates rather than
    stand-ins, because a selection made by volume routinely names images that
    did not navigate, and what the bundle does with one is a property of the
    run rather than of a fixture's ``status`` key.

    Parameters:
        mini_nav_cohort: The session's cohort.
        tmp_path: pytest-provided temporary directory for this test's bundle.
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


def _bundle_products(image_name: str) -> set[str]:
    """Return every bundle file one navigated Cassini image calls for.

    The sharded directories and the product stem are spelled out here rather
    than asked of the dataset, since what they are is what this test is for: a
    bundle is read by walking those directories, and an image that lands in the
    wrong one is found by whoever cannot find it.

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
    mini_nav_cohort: Cohort, tmp_path: Path
) -> None:
    """The shipped Cassini templates render over the cohort, into their shards.

    This is the assertion the cohort exists to make possible and the one every
    phase after this builds on: the registered dataset, the shipped template
    set and a real backplane FITS, with nothing standing in for anything.  A
    template the fixture cannot satisfy, a product written under the wrong
    number, or a render that fails and leaves half a bundle behind is reported
    here, in the phase that owns the fixture.

    Parameters:
        mini_nav_cohort: The session's cohort.
        tmp_path: pytest-provided temporary directory for this test's bundle.
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
