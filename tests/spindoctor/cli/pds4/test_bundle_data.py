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

The templates here are the tests' own stand-ins, so the tests assert substitution
and layout plumbing.  What a bundle's shipped templates render is tested in a
module named for that bundle.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.bundle_data import BundleDataOutcome, generate_bundle_data_files
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset import ImageFiles

from .conftest import (
    DATA_TEMPLATE,
    BundleEnv,
    NoPds4DataSet,
    make_bundle_env,
    make_image_file,
    navigated_document,
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
        backplane_metadata={'bodies': {'MOON_A': {'backplanes': {}}}, 'rings': {}},
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
    """The stage injects the BACKPLANE_*/BROWSE_FULL_* file variables into the dict.

    The image's results path stub names it otherwise than its bundle path stub, so
    a file name taken from the one where the other belongs is seen.
    """
    env = make_bundle_env(tmp_path, results_path_stub='res/1234567890w_CALIB')
    write_nav_inputs(env)
    _generate(env)
    variables = env.dataset.template_variables
    assert variables['BACKPLANE_FILENAME'] == '1234567890w_backplanes.fits'
    assert variables['BACKPLANE_SUPPL_FILENAME'] == '1234567890w_supplemental.txt'
    assert variables['BROWSE_FULL_FILENAME'] == '1234567890w_summary.png'
    expected_fits = str(FCPath(env.bundle_dir) / 'data' / f'{env.pds4_path_stub}_backplanes.fits')
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


RING_RESOLUTION_PLANE: list[dict[str, Any]] = [
    {'name': 'longitudinal_resolution', 'units': 'rad/pixel'}
]
"""A ring plane declared in radians per pixel, whose statistic is in degrees per pixel."""


def _ring_resolution_document(units: str) -> dict[str, Any]:
    """Build backplane metadata holding one ring longitudinal resolution statistic.

    Parameters:
        units: The unit the statistic records.

    Returns:
        The document, in the shape the backplane writer leaves on disk.
    """
    statistic: dict[str, Any] = {'min': 1.4e-05, 'max': 3.9e-05, 'units': units}
    return {'bodies': {}, 'rings': {'backplanes': {'longitudinal_resolution': statistic}}}


def test_a_statistic_in_another_unit_fails_the_image(tmp_path: Path) -> None:
    """A document recording a plane in a unit the configuration does not give it is failed.

    Every index column is in its plane's configured unit, so indexing the
    statistic would put its column in two units with nothing saying so.
    """
    env = make_bundle_env(tmp_path, rings=RING_RESOLUTION_PLANE)
    write_nav_inputs(env, backplane_metadata=_ring_resolution_document('rad/pixel'))
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.FAILED


def test_a_statistic_in_another_unit_writes_nothing(tmp_path: Path) -> None:
    """The unit is checked before any product is written, so nothing is on disk."""
    env = make_bundle_env(tmp_path, rings=RING_RESOLUTION_PLANE)
    write_nav_inputs(env, backplane_metadata=_ring_resolution_document('rad/pixel'))
    _generate(env)
    assert not env.bundle_dir.exists()


def test_a_unit_disagreement_names_the_plane_and_both_units(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error names the plane, the unit the document records and the unit expected."""
    env = make_bundle_env(tmp_path, rings=RING_RESOLUTION_PLANE)
    write_nav_inputs(env, backplane_metadata=_ring_resolution_document('rad/pixel'))
    _generate(env)
    out = capsys.readouterr().out
    assert 'longitudinal_resolution' in out
    assert 'in rad/pixel' in out
    assert 'expects deg/pixel' in out


def test_a_foreign_unit_on_an_undeclared_plane_is_ignored(tmp_path: Path) -> None:
    """A plane the configuration does not declare is not compared, whatever it records."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, backplane_metadata=_ring_resolution_document('furlong/pixel'))
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.WRITTEN


RESOLUTION_PLANE: list[dict[str, Any]] = [{'name': 'resolution', 'units': 'km/pixel'}]
"""A body plane in kilometers per pixel, whose statistic keeps that unit."""


def _resolution_document(maximum: float) -> dict[str, Any]:
    """Build backplane metadata holding one body resolution statistic with this maximum.

    Parameters:
        maximum: The maximum the statistic records, in the unit the plane takes.

    Returns:
        The document, in the shape the backplane writer leaves on disk.
    """
    statistic = {'min': 60.0, 'max': maximum, 'units': 'km/pixel'}
    return {'bodies': {'PLANET': {'backplanes': {'resolution': statistic}}}, 'rings': {}}


def test_a_statistic_that_is_not_a_finite_number_fails_the_image(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A maximum of NaN fails the image, and the log names the plane and the value.

    No index column can hold it, and a blank in its place would say the plane
    measured nothing, so it is refused as a statistic in another unit is.
    """
    env = make_bundle_env(tmp_path, bodies=RESOLUTION_PLANE)
    write_nav_inputs(env, backplane_metadata=_resolution_document(math.nan))
    outcome = _generate(env)
    assert outcome is BundleDataOutcome.FAILED
    assert 'records a resolution maximum of nan' in capsys.readouterr().out


def test_a_statistic_that_is_not_a_finite_number_writes_nothing(tmp_path: Path) -> None:
    """The values are checked before any product is written, so nothing is on disk."""
    env = make_bundle_env(tmp_path, bodies=RESOLUTION_PLANE)
    write_nav_inputs(env, backplane_metadata=_resolution_document(math.nan))
    _generate(env)
    assert not env.bundle_dir.exists()


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


def test_backplane_fits_copied_into_bundle_data_tree(tmp_path: Path) -> None:
    """The backplane FITS is copied into the bundle beside its data label, byte for byte."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    fits_source = env.backplane_root / f'{env.results_path_stub}_backplanes.fits'
    _generate(env)
    bundled_fits = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_backplanes.fits'
    assert bundled_fits.read_bytes() == fits_source.read_bytes()


# ---------------------------------------------------------------------------
# A record the enumeration already read
# ---------------------------------------------------------------------------


def test_a_carried_record_is_used_in_place_of_the_document(tmp_path: Path) -> None:
    """A record carried with the image is what the supplemental file records."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env, nav_extra={'marker': 'from the document'})
    env.image_file.nav_record = navigated_document(marker='from the enumeration')
    _generate(env)
    suppl = env.bundle_dir / 'data' / f'{env.pds4_path_stub}_supplemental.txt'
    combined = json.loads(suppl.read_text(encoding='utf-8'))
    assert combined['navigation']['marker'] == 'from the enumeration'


def test_a_carried_record_is_used_when_the_document_has_gone(tmp_path: Path) -> None:
    """An image carrying a record needs no document under the results root."""
    env = make_bundle_env(tmp_path)
    write_nav_inputs(env)
    (env.nav_root / f'{env.results_path_stub}_metadata.json').unlink()
    env.image_file.nav_record = navigated_document(marker='from the enumeration')
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
