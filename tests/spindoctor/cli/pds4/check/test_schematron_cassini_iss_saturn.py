"""The Cassini ISS Saturn bundle's labels against their Schematron rules, broken on purpose.

The Schematron holds a value to its rule's vocabulary, not to what the product is: it
refuses a SPICE kernel type, a collection type or a member entry's reference type the
vocabulary does not hold, and accepts one it holds even where it is wrong for the
product.  Each test copies the cohort's bundle and changes one label.
"""

from pathlib import Path
from typing import Any

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check.elements import element_path
from spindoctor.cli.pds4.check.schematron import schematron_findings

from ..cohort_bundle import write_cohort_bundle
from .controls import copy_bundle, parse, substitute_once

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

KERNEL_TYPE = ('spice_kernels/kernels.lblx', 'kernel_type', 'MK')
"""The metakernel label's SPICE kernel type: the label, the element and its value."""

COLLECTION_TYPE = ('spice_kernels/collection_spice_kernels.lblx', 'collection_type', 'SPICE Kernel')
"""The SPICE kernel collection's type: the label, the element and its value."""

REFERENCE_TYPE = ('bundle.lblx', 'reference_type', 'bundle_has_spice_kernel_collection')
"""The bundle label's member entry for the SPICE kernel collection: label, element, value."""


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


def _retype(bundle_dir: Path, file: str, element: str, original: str, value: str) -> Any:
    """Change the value of one element of a label, and parse the label.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The label's path relative to it.
        element: The element's local name.
        original: Its value as written.
        value: The value it is given.

    Returns:
        The changed label, parsed.
    """
    substitute_once(
        bundle_dir / file, f'<{element}>{original}</{element}>', f'<{element}>{value}</{element}>'
    )
    return parse(bundle_dir / file)


@pytest.mark.parametrize(
    ('file', 'element', 'original', 'value'),
    [
        pytest.param(*KERNEL_TYPE, 'XX', id='kernel_type XX'),
        pytest.param(*COLLECTION_TYPE, 'XX', id='collection_type XX'),
        pytest.param(*REFERENCE_TYPE, 'bundle_has_XX_collection', id='reference_type XX'),
    ],
)
def test_the_schematron_refuses_a_type_its_vocabulary_does_not_hold(
    plain_bundle: Path, tmp_path: Path, file: str, element: str, original: str, value: str
) -> None:
    """A type outside its rule's vocabulary is one finding, at the element holding it.

    Parameters:
        plain_bundle: The cohort's bundle.
        tmp_path: The test's directory, which the bundle is copied into.
        file: The label changed.
        element: The element changed.
        original: Its value as written.
        value: A value its rule's vocabulary does not hold.
    """
    bundle = copy_bundle(plain_bundle, tmp_path)
    document = _retype(bundle, file, element, original, value)
    changed = next(
        node for node in document.getroot().iter(f'{{*}}{element}') if node.text == value
    )
    findings = schematron_findings(file, document)
    assert [finding.location for finding in findings] == [element_path(changed)]


def test_a_rule_of_several_steps_refuses_a_kernel_type_no_kernel_has(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """``kernel_type`` ``XX`` is refused by the rule on ``pds:SPICE_Kernel/pds:kernel_type``."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    document = _retype(bundle, *KERNEL_TYPE, 'XX')
    findings = schematron_findings(KERNEL_TYPE[0], document)
    rules = [finding.message.rsplit(' (', 1)[1] for finding in findings]
    assert rules == ['assert of the rule on pds:SPICE_Kernel/pds:kernel_type in PDS4_PDS_1O00.sch)']


@pytest.mark.parametrize(
    ('file', 'element', 'original', 'value'),
    [
        pytest.param(*KERNEL_TYPE, 'FK', id='kernel_type FK'),
        pytest.param(*COLLECTION_TYPE, 'Document', id='collection_type Document'),
        pytest.param(
            *REFERENCE_TYPE, 'bundle_has_document_collection', id='reference_type document'
        ),
    ],
)
def test_the_schematron_accepts_a_wrong_type_its_vocabulary_holds(
    plain_bundle: Path, tmp_path: Path, file: str, element: str, original: str, value: str
) -> None:
    """A type its rule's vocabulary holds passes, though it is wrong for the product.

    Parameters:
        plain_bundle: The cohort's bundle.
        tmp_path: The test's directory, which the bundle is copied into.
        file: The label changed.
        element: The element changed.
        original: Its value as written.
        value: A value the vocabulary holds and the product is not.
    """
    bundle = copy_bundle(plain_bundle, tmp_path)
    document = _retype(bundle, file, element, original, value)
    assert schematron_findings(file, document) == []


def test_the_schematron_refuses_an_inventory_offset_other_than_zero(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """An inventory whose ``offset`` is 1 is one finding, at the inventory."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    file = 'data/collection_data.lblx'
    substitute_once(
        bundle / file, '<offset unit="byte">0</offset>', '<offset unit="byte">1</offset>'
    )
    findings = schematron_findings(file, parse(bundle / file))
    locations = [finding.location for finding in findings]
    assert locations == ['/Product_Collection/File_Area_Inventory/Inventory']
