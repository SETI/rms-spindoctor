"""The Cassini ISS Saturn bundle's data inventory held to its tree, broken on purpose.

Each test copies the cohort's bundle, changes the data collection's inventory, and holds
the integrity check to finding the change: a member no label declares, a member at a
version the tree does not hold, a product the inventory leaves out, and a product it
lists twice.
"""

from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check.elements import child_text
from spindoctor.cli.pds4.check.findings import CheckName, Finding, Severity
from spindoctor.cli.pds4.check.integrity import integrity_findings

from ..cohort_bundle import write_cohort_bundle
from .controls import copy_bundle, parsed_labels

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

DATA_COLLECTION = 'data/collection_data.lblx'
"""The data collection's label."""

DATA_INVENTORY = 'data/collection_data.csv'
"""The data collection's inventory."""

INVENTORY = '/Product_Collection/File_Area_Inventory/Inventory'
"""Where a collection's label describes its inventory."""


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


def _members(bundle_dir: Path) -> list[str]:
    """Return the data inventory's records.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        Each record, as in ``P,<LIDVID>``, without its line feed.
    """
    return (bundle_dir / DATA_INVENTORY).read_text(encoding='ascii').splitlines()


def _write_members(bundle_dir: Path, members: list[str]) -> None:
    """Write the data inventory's records.

    Parameters:
        bundle_dir: The bundle's directory.
        members: Each record, without its line feed.
    """
    text = ''.join(f'{member}\n' for member in members)
    (bundle_dir / DATA_INVENTORY).write_text(text, encoding='ascii')


def test_a_member_no_label_declares_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A primary member whose LID no label of the tree declares is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    members = _members(bundle)
    lid, _, version = members[0].removeprefix('P,').partition('::')
    members[0] = f'P,{lid[:-1]}x::{version}'
    _write_members(bundle, members)
    expected = Finding(
        DATA_COLLECTION,
        CheckName.INTEGRITY,
        INVENTORY,
        f'record 1 lists {lid[:-1]}x::{version}, which no label of the tree declares',
    )
    assert expected in _check(bundle)


def test_a_member_at_a_version_the_tree_does_not_hold_is_found(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A primary member at a version no label of the tree declares is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    members = _members(bundle)
    lid, _, version = members[0].removeprefix('P,').partition('::')
    members[0] = f'P,{lid}::9.9'
    _write_members(bundle, members)
    expected = Finding(
        DATA_COLLECTION,
        CheckName.INTEGRITY,
        INVENTORY,
        f'record 1 lists {lid}::9.9, but the tree holds {lid} at version {version}',
    )
    assert expected in _check(bundle)


def test_a_product_its_collections_inventory_leaves_out_is_a_warning(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A data product the data inventory does not list warns, at the product's label."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    members = _members(bundle)
    lid = members.pop().removeprefix('P,').partition('::')[0]
    _write_members(bundle, members)
    label = next(
        file
        for file, document in parsed_labels(bundle).items()
        if child_text(document.getroot(), 'Identification_Area', 'logical_identifier') == lid
    )
    expected = Finding(
        label,
        CheckName.INTEGRITY,
        '',
        f'the inventory of {DATA_COLLECTION}, in whose directory it lies, does not list it',
        Severity.WARNING,
    )
    assert expected in _check(bundle)


def test_a_product_its_inventory_lists_twice_is_found(plain_bundle: Path, tmp_path: Path) -> None:
    """A data product the data inventory lists a second time is found."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    members = _members(bundle)
    members.append(members[0])
    _write_members(bundle, members)
    lid = members[0].removeprefix('P,').partition('::')[0]
    expected = Finding(
        DATA_COLLECTION,
        CheckName.INTEGRITY,
        INVENTORY,
        f'lists {lid} 2 times, in records 1, {len(members)}',
    )
    assert expected in _check(bundle)
