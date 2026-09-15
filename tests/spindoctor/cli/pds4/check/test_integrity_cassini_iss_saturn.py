"""The Cassini ISS Saturn bundle's tree checked as a whole, broken on purpose.

Each test copies the cohort's bundle, changes one file, and holds the integrity check to
finding the change: a file a label names that is not there, names by a path, names twice
or describes otherwise than it is; a file no label names or two labels name; a template's
marker; an empty element; two labels declaring one product; and a reference to a product
of the bundle, or to a version of one, that the tree does not hold.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check.elements import child_text
from spindoctor.cli.pds4.check.findings import CheckName, Finding, Severity
from spindoctor.cli.pds4.check.integrity import integrity_findings

from ..cohort_bundle import write_cohort_bundle
from .controls import copy_bundle, line_of, parse, parsed_labels, substitute_once

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

KERNELS = 'spice_kernels/kernels.lblx'
"""The metakernel's label."""

KERNELS_TITLE = '/Product_SPICE_Kernel/Identification_Area/title'
"""Where the metakernel's label gives its title."""

SUPPLEMENTAL = '/Product_Observational/File_Area_Observational_Supplemental/File'
"""Where a data label describes its supplemental file."""


@pytest.fixture(scope='module')
def plain_bundle(
    mini_nav_cohorts: WrittenCohorts, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Return the cohort's bundle, written once for the module, without a user guide.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.
        tmp_path_factory: Factory the bundle's directory is made under.

    Returns:
        The bundle's directory, which no test changes.
    """
    cohort = mini_nav_cohorts(CohortCassiniISSSaturn)
    return write_cohort_bundle(cohort, tmp_path_factory.mktemp('plain'), NAVIGATED_STUBS).bundle_dir


def _check(bundle_dir: Path) -> list[Finding]:
    """Check a bundle's tree as a whole.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The integrity check's findings.
    """
    return integrity_findings(bundle_dir, parsed_labels(bundle_dir))


def _data_label(bundle_dir: Path) -> str:
    """Return the first data label of a bundle.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The label's path relative to it.
    """
    label = sorted((bundle_dir / 'data').rglob('*_backplanes.lblx'))[0]
    return label.relative_to(bundle_dir).as_posix()


def _supplemental(bundle_dir: Path, label: str) -> Path:
    """Return the supplemental file a data label names.

    Parameters:
        bundle_dir: The bundle's directory.
        label: The data label's path relative to it.

    Returns:
        The file.
    """
    name = child_text(
        parse(bundle_dir / label).getroot(),
        'File_Area_Observational_Supplemental',
        'File',
        'file_name',
    )
    return (bundle_dir / label).with_name(name or '')


def _browse_label(bundle_dir: Path) -> Path:
    """Return the first browse label of a bundle.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The label.
    """
    return sorted((bundle_dir / 'browse').rglob('*_summary.lblx'))[0]


def _bundle_lid(bundle_dir: Path) -> str:
    """Return the logical identifier a bundle's label declares.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        The identifier.
    """
    root = parse(bundle_dir / 'bundle.lblx').getroot()
    return child_text(root, 'Identification_Area', 'logical_identifier') or ''


def test_a_file_its_label_names_that_is_not_beside_it_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A data label whose supplemental file is gone names a file that is not there."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _data_label(bundle)
    supplemental = _supplemental(bundle, label)
    supplemental.unlink()
    expected = Finding(
        label,
        CheckName.INTEGRITY,
        f'{SUPPLEMENTAL}/file_name',
        f"names '{supplemental.name}', which is not beside it",
    )
    assert expected in _check(bundle)


def test_a_file_name_holding_a_directory_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``file_name`` reaching into a directory names no file beside its label."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(
        bundle / KERNELS,
        r'<file_name>kernels\.ker</file_name>',
        '<file_name>../spice_kernels/kernels.ker</file_name>',
    )
    expected = Finding(
        KERNELS,
        CheckName.INTEGRITY,
        '/Product_SPICE_Kernel/File_Area_SPICE_Kernel/File/file_name',
        "names '../spice_kernels/kernels.ker', which holds a directory, where a file beside "
        'it is named',
    )
    assert expected in _check(bundle)


def test_a_file_one_label_names_twice_is_found_once(plain_bundle: Path, tmp_path: Path) -> None:
    """A browse label repeating its file area names its image twice, and that alone."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _browse_label(bundle)
    text = label.read_text(encoding='utf-8')
    start = text.index('<File_Area_Browse>')
    end = text.index('</File_Area_Browse>') + len('</File_Area_Browse>')
    label.write_text(text[:end] + '\n  ' + text[start:end] + text[end:], encoding='utf-8')
    image = label.with_suffix('.png')
    about = [
        finding
        for finding in _check(bundle)
        if finding.file == image.relative_to(bundle).as_posix()
        or 'File_Area_Browse' in finding.location
    ]
    expected = Finding(
        label.relative_to(bundle).as_posix(),
        CheckName.INTEGRITY,
        '/Product_Browse/File_Area_Browse[2]/File/file_name',
        f"names '{image.name}' a second time",
    )
    assert about == [expected]


def test_a_file_rewritten_after_its_label_is_found_by_its_size(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A statistic dropped from a supplemental file changes the size its label states."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _data_label(bundle)
    supplemental = _supplemental(bundle, label)
    stated = child_text(
        parse(bundle / label).getroot(),
        'File_Area_Observational_Supplemental',
        'File',
        'file_size',
    )
    document = json.loads(supplemental.read_text(encoding='ascii'))
    for body in document['backplanes']['bodies'].values():
        body['backplanes'].pop('body_phase_angle', None)
    supplemental.write_text(json.dumps(document, indent=2), encoding='ascii')
    expected = Finding(
        label,
        CheckName.INTEGRITY,
        f'{SUPPLEMENTAL}/file_size',
        f'file_size is {stated}, but {supplemental.name} is {supplemental.stat().st_size} bytes',
    )
    assert expected in _check(bundle)


def test_a_file_rewritten_after_its_label_is_found_by_its_checksum(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A supplemental file with one digit changed keeps its size but not its checksum."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _data_label(bundle)
    supplemental = _supplemental(bundle, label)
    stated = child_text(
        parse(bundle / label).getroot(),
        'File_Area_Observational_Supplemental',
        'File',
        'md5_checksum',
    )
    text = supplemental.read_text(encoding='ascii')
    changed = re.sub(r'\d', lambda digit: str((int(digit.group()) + 1) % 10), text, count=1)
    supplemental.write_text(changed, encoding='ascii')
    digest = hashlib.md5(supplemental.read_bytes()).hexdigest()
    expected = Finding(
        label,
        CheckName.INTEGRITY,
        f'{SUPPLEMENTAL}/md5_checksum',
        f'md5_checksum is {stated}, but the MD5 checksum of {supplemental.name} is {digest}',
    )
    assert expected in _check(bundle)


def test_a_file_no_label_names_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A file in the tree that is not a label and that no label names is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    (bundle / 'data' / 'stray.txt').write_text('stray\n', encoding='ascii')
    expected = Finding('data/stray.txt', CheckName.INTEGRITY, '', 'no label names it')
    assert expected in _check(bundle)


def test_a_file_two_labels_name_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A FITS a second label beside the first also names is named by two labels."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _data_label(bundle)
    second = (bundle / label).with_name('second_backplanes.lblx')
    shutil.copy(bundle / label, second)
    fits = label.replace('_backplanes.lblx', '_backplanes.fits')
    naming = sorted([label, second.relative_to(bundle).as_posix()])
    expected = Finding(fits, CheckName.INTEGRITY, '', f'2 labels name it: {", ".join(naming)}')
    assert expected in _check(bundle)


def test_a_template_marker_left_in_a_label_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A ``[[[`` marker in a label is found, at its line."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(bundle / KERNELS, '<title>', '<title>[[[unrendered]]] ')
    expected = Finding(
        KERNELS,
        CheckName.INTEGRITY,
        f'line {line_of(bundle / KERNELS, "[[[")}',
        'holds a [[[ marker, which a template leaves where it could not fill in a value',
    )
    assert expected in _check(bundle)


def test_an_empty_element_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """An element holding nothing, without ``xsi:nil``, is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(bundle / KERNELS, r'<title>[^<]*</title>', '<title></title>')
    expected = Finding(
        KERNELS, CheckName.INTEGRITY, KERNELS_TITLE, 'is empty: it holds no element and no text'
    )
    assert expected in _check(bundle)


@pytest.mark.parametrize('nil', ['true', '1', ' true '])
def test_an_empty_element_carrying_xsi_nil_is_not_found(
    plain_bundle: Path, tmp_path: Path, nil: str
) -> None:
    """An empty element whose ``xsi:nil`` is true, however spelled, is empty on purpose.

    Parameters:
        plain_bundle: The cohort's bundle.
        tmp_path: The test's directory.
        nil: The ``xsi:nil`` the element carries, a spelling of true.
    """
    bundle = copy_bundle(plain_bundle, tmp_path)
    substitute_once(bundle / KERNELS, r'<title>[^<]*</title>', f'<title xsi:nil="{nil}"></title>')
    at_title = [
        finding
        for finding in _check(bundle)
        if finding.file == KERNELS and finding.location == KERNELS_TITLE
    ]
    assert at_title == []


def test_two_labels_declaring_one_product_are_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A second browse label declaring the first's identifier, over a copy of its PNG."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    label = _browse_label(bundle)
    image = label.with_suffix('.png')
    twin = label.with_name('twin_summary.lblx')
    shutil.copy(image, image.with_name('twin_summary.png'))
    twin.write_text(
        label.read_text(encoding='utf-8').replace(
            f'<file_name>{image.name}</file_name>', '<file_name>twin_summary.png</file_name>'
        ),
        encoding='utf-8',
    )
    lid = child_text(parse(label).getroot(), 'Identification_Area', 'logical_identifier')
    expected = Finding(
        twin.relative_to(bundle).as_posix(),
        CheckName.INTEGRITY,
        '/Product_Browse/Identification_Area/logical_identifier',
        f'declares {lid}, which {label.relative_to(bundle).as_posix()} declares too',
    )
    assert expected in _check(bundle)


def test_a_reference_to_a_product_of_the_bundle_not_in_the_tree_is_a_warning(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A member entry naming a collection of the bundle no label declares is a warning."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    lid = _bundle_lid(bundle)
    substitute_once(
        bundle / 'bundle.lblx',
        f'<lid_reference>{re.escape(lid)}:browse</lid_reference>',
        f'<lid_reference>{lid}:nonesuch</lid_reference>',
    )
    expected = Finding(
        'bundle.lblx',
        CheckName.INTEGRITY,
        '/Product_Bundle/Bundle_Member_Entry[1]/lid_reference',
        f'refers to {lid}:nonesuch, which no label of the tree declares',
        Severity.WARNING,
    )
    assert expected in _check(bundle)


def test_a_reference_to_a_version_the_tree_does_not_hold_is_a_warning(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A member entry naming a collection of the bundle at another version warns."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    lid = _bundle_lid(bundle)
    version = child_text(
        parse(bundle / 'browse' / 'collection_browse.lblx').getroot(),
        'Identification_Area',
        'version_id',
    )
    substitute_once(
        bundle / 'bundle.lblx',
        f'<lid_reference>{re.escape(lid)}:browse</lid_reference>',
        f'<lidvid_reference>{lid}:browse::9.9</lidvid_reference>',
    )
    expected = Finding(
        'bundle.lblx',
        CheckName.INTEGRITY,
        '/Product_Bundle/Bundle_Member_Entry[1]/lidvid_reference',
        f'refers to {lid}:browse::9.9, but the tree holds {lid}:browse at version {version}',
        Severity.WARNING,
    )
    assert expected in _check(bundle)


def test_a_tree_with_no_bundle_label_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A tree with no bundle label at its top has nothing to resolve its references by."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    (bundle / 'bundle.lblx').unlink()
    expected = Finding(
        '.',
        CheckName.INTEGRITY,
        '',
        'the tree holds no bundle label at its top, so no reference to a product of the '
        'bundle can be resolved',
    )
    assert expected in _check(bundle)
