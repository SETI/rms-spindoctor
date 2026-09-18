"""The Cassini ISS Saturn bundle's labels against their XML schemas, broken on purpose.

Each test copies the cohort's bundle, changes one label, and holds the XML schema check
to what the change does or does not break; one builds the Cassini schema without the
cartography schema it imports.
"""

import shutil
from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check import check_bundle
from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import SchemaSource, label_schema, xsd_findings
from spindoctor.config import DEFAULT_CONFIG

from ..cohort_bundle import write_cohort_bundle
from .controls import SCHEMA_COPIES, SCHEMAS, bare_bundle_label, copy_bundle, parse, substitute_once

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""


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


def _xsd(bundle_dir: Path, file: str) -> list[Finding]:
    """Hold one label of a bundle to the XML schemas it declares.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The label's path relative to it.

    Returns:
        What resolving its schemas found, and its XML schema errors.
    """
    document = parse(bundle_dir / file)
    resolved = label_schema(file, document, SCHEMAS)
    findings = list(resolved.findings)
    if resolved.schema is not None:
        findings.extend(xsd_findings(file, document, resolved.schema))
    return findings


def test_the_xml_schema_refuses_a_table_of_no_records(plain_bundle: Path, tmp_path: Path) -> None:
    """A table label stating ``records`` 0 breaks its schema's minimum, a facet's kind.

    The finding is held by the kind of check that failed, the minimum facet, and not by
    the words xmlschema gives it.
    """
    bundle = copy_bundle(plain_bundle, tmp_path)
    file = 'miscellaneous/global_bodies_index.lblx'
    substitute_once(bundle / file, r'<records>\d+</records>', '<records>0</records>')
    records = '/Product_Ancillary/File_Area_Ancillary/Table_Character/records'
    kinds = {(finding.location, finding.kind) for finding in _xsd(bundle, file)}
    assert (records, 'XsdMinInclusiveFacet') in kinds


def test_the_xml_schema_accepts_a_kernel_type_no_kernel_has(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """The XML schema lets ``kernel_type`` be ``XX``; only the Schematron refuses it."""
    bundle = copy_bundle(plain_bundle, tmp_path)
    file = 'spice_kernels/kernels.lblx'
    substitute_once(bundle / file, '<kernel_type>MK</kernel_type>', '<kernel_type>XX</kernel_type>')
    assert _xsd(bundle, file) == []


def test_an_import_with_no_local_copy_is_a_finding(tmp_path: Path) -> None:
    """An import finding no shipped schema warns as the set is built, and that is found.

    The schema directory holds the common dictionary's schema, the geometry build the
    Cassini schema imports and the Cassini schema, and not the cartography dictionary's,
    which the Cassini schema imports too.
    """
    for name in ('PDS4_PDS_1O00.xsd', 'PDS4_GEOM_1O00_19A0.xsd', 'PDS4_CASSINI_1O00_1800.xsd'):
        shutil.copy(SCHEMA_COPIES / name, tmp_path / name)
    location = (
        'http://pds.nasa.gov/pds4/pds/v1 https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.xsd '
        'http://pds.nasa.gov/pds4/mission/cassini/v1 '
        'https://pds.nasa.gov/pds4/mission/cassini/v1/PDS4_CASSINI_1O00_1800.xsd'
    )
    resolved = label_schema('bundle.lblx', bare_bundle_label(location), SchemaSource(tmp_path))
    cart = 'https://pds.nasa.gov/pds4/cart/v1/PDS4_CART_1O00_1970.xsd'
    assert [cart in finding.message for finding in resolved.findings] == [True]


def test_a_label_whose_schemas_cannot_be_built_is_one_finding_and_the_check_goes_on(
    plain_bundle: Path, tmp_path: Path
) -> None:
    """A label pairing the common namespace with a Schematron is one finding.

    The check does not stop there: it returns every finding, this one among them.
    """
    bundle = copy_bundle(plain_bundle, tmp_path)
    file = 'miscellaneous/global_bodies_index.lblx'
    substitute_once(
        bundle / file, r'(http://pds\.nasa\.gov/pds4/pds/v1 \S+PDS4_PDS_1O00)\.xsd', r'\1.sch'
    )
    expected = Finding(
        file,
        CheckName.XSD,
        '',
        'its XML schemas cannot be built, so it is not validated against them: '
        "'{http://purl.oclc.org/dsdl/schematron}schema' is not an element of the schema",
    )
    assert expected in check_bundle(bundle, config=DEFAULT_CONFIG, schema_dir=SCHEMA_COPIES)


def test_a_relative_schema_directory_serves_the_imports_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema directory named relative to where the check runs serves every import too."""
    monkeypatch.chdir(SCHEMA_COPIES.parent)
    location = (
        'http://pds.nasa.gov/pds4/pds/v1 https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.xsd '
        'http://pds.nasa.gov/pds4/mission/cassini/v1 '
        'https://pds.nasa.gov/pds4/mission/cassini/v1/PDS4_CASSINI_1O00_1800.xsd'
    )
    source = SchemaSource(Path(SCHEMA_COPIES.name))
    assert label_schema('bundle.lblx', bare_bundle_label(location), source).findings == ()
