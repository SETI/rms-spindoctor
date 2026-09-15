"""The schemas the package ships, and a label's XML schemas resolved to them."""

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import LabelSchema, label_schema, shipped_copy
from spindoctor.config import DEFAULT_CONFIG

from .controls import bare_bundle_label

UNSHIPPED_PDS_SCHEMA = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1N00.xsd'
"""The URL of a build of the common dictionary's XML schema the package does not ship."""

PDS_SCHEMATRON = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.sch'
"""The URL of the common dictionary's Schematron, which the package ships."""


def test_every_schema_the_configuration_names_is_shipped() -> None:
    """Each dataset's configured schemas have their XML schema and Schematron shipped."""
    missing = [
        url
        for entry in DEFAULT_CONFIG.pds4.values()
        for schema in entry['schemas'].values()
        for url in (f'{schema["location"]}.xsd', f'{schema["location"]}.sch')
        if shipped_copy(url) is None
    ]
    assert missing == []


def test_a_label_declaring_an_xml_schema_the_package_does_not_ship_is_a_finding() -> None:
    """A URL with no shipped copy is a finding that names it, and nothing is validated."""
    location = f'http://pds.nasa.gov/pds4/pds/v1 {UNSHIPPED_PDS_SCHEMA}'
    resolved = label_schema('bundle.lblx', bare_bundle_label(location))
    expected = Finding(
        'bundle.lblx',
        CheckName.XSD,
        '',
        f'declares the XML schema {UNSHIPPED_PDS_SCHEMA}, of which the package ships no copy',
    )
    assert resolved.findings == (expected,)


def test_a_schema_set_that_cannot_be_built_is_one_finding() -> None:
    """A namespace paired with a file that is not an XML schema leaves no set, and one finding."""
    location = f'http://pds.nasa.gov/pds4/pds/v1 {PDS_SCHEMATRON}'
    resolved = label_schema('bundle.lblx', bare_bundle_label(location))
    expected = Finding(
        'bundle.lblx',
        CheckName.XSD,
        '',
        'its XML schemas cannot be built, so it is not validated against them: '
        "'{http://purl.oclc.org/dsdl/schematron}schema' is not an element of the schema",
    )
    assert resolved == LabelSchema(schema=None, findings=(expected,))
