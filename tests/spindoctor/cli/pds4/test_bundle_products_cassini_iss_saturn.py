"""The bundle's run-level products, written over the Cassini ISS Saturn cohort.

What the summary pass writes into the Cassini ISS Saturn bundle from the templates the
package ships: every file of each collection the bundle holds; a bundle label each of
whose member entries names a collection label in the bundle; one LID for the user guide
wherever a label or the document inventory names it, the LID its own label declares; and
a SPICE kernel inventory naming the metakernel by its label's LID and version.  The
plumbing is tested over stand-in templates in ``test_bundle_products.py``.
"""

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from .conftest import (
    label_cohort_images,
    make_cohort_bundle_env,
    read_csv_rows,
    summarize_bundle,
    write_cohort_bundle,
)

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the paths below use."""

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

USER_GUIDE = 'cassini-iss-saturn-backplanes-user-guide'
"""The stem of the bundle's user guide and of its label."""

BUNDLE_TOP_LEVEL = {
    'bundle.lblx',
    'readme.txt',
    'browse',
    'context',
    'data',
    'document',
    'spice_kernels',
    'xml_schema',
}
"""What the bundle's own directory holds once both passes have run."""

SUMMARY_LAYOUT = {
    'bundle.lblx',
    'readme.txt',
    'browse/collection_browse.csv',
    'browse/collection_browse.lblx',
    'context/collection_context.csv',
    'context/collection_context.lblx',
    'data/collection_data.csv',
    'data/collection_data.lblx',
    'document/collection_document.csv',
    'document/collection_document.lblx',
    'document/supplemental/global_index_bodies.tab',
    'document/supplemental/global_index_bodies.lblx',
    'document/supplemental/global_index_rings.tab',
    'document/supplemental/global_index_rings.lblx',
    'spice_kernels/collection_spice_kernels.csv',
    'spice_kernels/collection_spice_kernels.lblx',
    'spice_kernels/kernels.ker',
    'spice_kernels/kernels.lblx',
    'xml_schema/collection_xml_schema.csv',
    'xml_schema/collection_xml_schema.lblx',
}
"""Every file the summary pass writes into a bundle whose template directory has no guide.

The global index tables are under ``document/supplemental/`` until they have a collection
of their own.
"""


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def _files(root: Path) -> set[str]:
    """Return every file under a directory, relative to it.

    Parameters:
        root: The directory.

    Returns:
        Each file's path relative to ``root``, in POSIX form.
    """
    return {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}


def _lid(label: Path) -> str:
    """Return the logical identifier a label declares.

    Parameters:
        label: The label.

    Returns:
        Its ``Identification_Area/logical_identifier``, stripped.
    """
    root = ElementTree.parse(label).getroot()
    lid = root.findtext('pds:Identification_Area/pds:logical_identifier', '', PDS4_NAMESPACES)
    return lid.strip()


def test_the_summary_pass_writes_every_file_of_each_collection_the_bundle_holds(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The summary pass adds exactly the files of section 3.1's tree it has collections for.

    The template directory holds no user guide, so there is no ``document/user_guide/``.
    The labels pass's products stay as they were.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    label_cohort_images(env, NAVIGATED_STUBS)
    labeled = _files(env.bundle_dir)
    summarize_bundle(env)
    written = _files(env.bundle_dir)
    assert {path.name for path in env.bundle_dir.iterdir()} == BUNDLE_TOP_LEVEL
    assert labeled <= written
    assert written - labeled == SUMMARY_LAYOUT


def test_every_member_entry_of_the_bundle_label_names_a_collection_label_in_the_bundle(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The bundle label's member entries are exactly the collection labels' own LIDs.

    A collection label is one directory below the bundle's root.  The bundle label names
    each of the six the pass writes by the logical identifier that label declares, and
    names nothing else.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    root = ElementTree.parse(env.bundle_dir / 'bundle.lblx').getroot()
    declared = [
        entry.findtext('pds:lid_reference', '', PDS4_NAMESPACES).strip()
        for entry in root.iterfind('pds:Bundle_Member_Entry', PDS4_NAMESPACES)
    ]
    held = [_lid(label) for label in env.bundle_dir.glob('*/collection_*.lblx')]
    assert sorted(declared) == sorted(held)


def _documents_named(label: Path, bundle_lid: str) -> set[str]:
    """Return the documents of a bundle that one of its labels names.

    Parameters:
        label: The label.
        bundle_lid: The bundle's LID, which each of its products' LIDs begins with.

    Returns:
        Each LID an ``Internal_Reference`` of a ``..._to_document`` type names that is a
        product of the bundle.
    """
    if label.stat().st_size == 0:
        # The global index labels render from empty templates, so they name nothing.
        return set()
    references = (
        ElementTree.parse(label).getroot().iter(f'{{{PDS4_NAMESPACES["pds"]}}}Internal_Reference')
    )
    return {
        reference.findtext('pds:lid_reference', '', PDS4_NAMESPACES).strip()
        for reference in references
        if reference.findtext('pds:reference_type', '', PDS4_NAMESPACES).endswith('_to_document')
        and reference.findtext('pds:lid_reference', '', PDS4_NAMESPACES).startswith(
            f'{bundle_lid}:'
        )
    }


def test_every_label_names_the_user_guide_by_the_lid_its_own_label_declares(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The labels, the document inventory and the readme name the guide by its own LID.

    The bundle is built from a copy of the shipped template directory holding a stand-in
    guide, so that the guide's label is rendered and its LID can be read from it.  Every
    label naming a document of the bundle names that one LID, the data labels and the
    bundle label among them; it is the document inventory's one primary member; and the
    readme gives it on a line of its own.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    templates = tmp_path / 'templates'
    shutil.copytree(env.dataset.pds4_bundle_template_dir(), templates)
    (templates / f'{USER_GUIDE}.pdf').write_bytes(b'%PDF-1.4 a stand-in user guide\n')
    monkeypatch.setattr(env.dataset, 'pds4_bundle_template_dir', lambda: str(templates))
    label_cohort_images(env, NAVIGATED_STUBS)
    summarize_bundle(env)
    guide_lid = _lid(env.bundle_dir / 'document' / 'user_guide' / f'{USER_GUIDE}.lblx')
    bundle_lid = _lid(env.bundle_dir / 'bundle.lblx')
    naming = {
        label.relative_to(env.bundle_dir).as_posix(): documents
        for label in env.bundle_dir.rglob('*.lblx')
        if len(documents := _documents_named(label, bundle_lid)) > 0
    }
    data_labels = {
        label.relative_to(env.bundle_dir).as_posix()
        for label in (env.bundle_dir / 'data').rglob('*_backplanes.lblx')
    }
    assert {lid for documents in naming.values() for lid in documents} == {guide_lid}
    assert {'bundle.lblx', *data_labels} <= naming.keys()
    rows = read_csv_rows(env.bundle_dir / 'document' / 'collection_document.csv')
    assert [lidvid.split('::')[0] for status, lidvid in rows if status == 'P'] == [guide_lid]
    readme = (env.bundle_dir / 'readme.txt').read_text(encoding='ascii').splitlines()
    assert guide_lid in readme


def test_both_passes_take_no_file_the_dataset_does_not_declare(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over a template directory holding only what the dataset declares, both passes run.

    Each pass checks the files its dataset declares before it processes anything, so a
    file a pass takes from the template directory and the dataset does not declare is
    one that check cannot report, and a run over a tree missing it would stop part way.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    shipped = Path(env.dataset.pds4_bundle_template_dir())
    declared = tmp_path / 'declared'
    declared.mkdir()
    names = [
        *env.dataset.pds4_required_templates('labels'),
        *env.dataset.pds4_required_templates('summary'),
    ]
    for name in names:
        shutil.copy(shipped / name, declared / name)
    # A stand-in guide, so that the guide's label renders from the template declared for it.
    guide = declared / env.dataset.pds4_user_guide_file_name()
    guide.write_bytes(b'%PDF-1.4 a stand-in user guide\n')
    monkeypatch.setattr(env.dataset, 'pds4_bundle_template_dir', lambda: str(declared))
    label_cohort_images(env, NAVIGATED_STUBS)
    summarize_bundle(env)
    assert (env.bundle_dir / 'bundle.lblx').is_file()


def test_the_metakernel_label_describes_the_metakernel_beside_it(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """kernels.lblx states the size, the checksum and the time of the kernels.ker beside it.

    The file it names is the bundle's copy, so its time is the copy's, to the second.
    """
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    metakernel = env.bundle_dir / 'spice_kernels' / 'kernels.ker'
    root = ElementTree.parse(metakernel.with_suffix('.lblx')).getroot()
    stated = 'pds:File_Area_SPICE_Kernel/pds:File/pds:'
    size = root.findtext(f'{stated}file_size', '', PDS4_NAMESPACES)
    md5 = root.findtext(f'{stated}md5_checksum', '', PDS4_NAMESPACES)
    created = root.findtext(f'{stated}creation_date_time', '', PDS4_NAMESPACES)
    assert int(size) == metakernel.stat().st_size
    assert md5.strip() == hashlib.md5(metakernel.read_bytes(), usedforsecurity=False).hexdigest()
    modified = datetime.fromtimestamp(metakernel.stat().st_mtime, UTC)
    assert created.strip() == modified.strftime('%Y-%m-%dT%H:%M:%SZ')


def test_the_spice_kernel_inventory_lists_the_metakernel_by_its_label_s_lid_and_version(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """The SPICE kernel collection's one member is the metakernel, as its label names it."""
    env = write_cohort_bundle(cassini_cohort, tmp_path, NAVIGATED_STUBS)
    label = env.bundle_dir / 'spice_kernels' / 'kernels.lblx'
    version = (
        ElementTree.parse(label)
        .getroot()
        .findtext('pds:Identification_Area/pds:version_id', '', PDS4_NAMESPACES)
    )
    rows = read_csv_rows(env.bundle_dir / 'spice_kernels' / 'collection_spice_kernels.csv')
    assert rows == [['P', f'{_lid(label)}::{version.strip()}']]
