"""The schemas a label names, resolved by URL to the local copies the tests keep."""

from lxml import etree

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import LabelSchema, label_schema
from spindoctor.config import DEFAULT_CONFIG

from .controls import SCHEMA_COPIES, SCHEMAS, bare_bundle_label

UNCOPIED_PDS_SCHEMA = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1N00.xsd'
"""The URL of a build of the common dictionary's XML schema the tests keep no copy of."""

PDS_SCHEMATRON = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.sch'
"""The URL of the common dictionary's Schematron, of which the tests keep a copy."""

XS_IMPORT = '{http://www.w3.org/2001/XMLSchema}import'
"""The element an XML schema imports another's by, with its URL."""


def test_every_schema_the_configuration_names_has_a_local_copy() -> None:
    """Each dataset's configured schemas have their XML schema and Schematron copied."""
    missing = [
        url
        for entry in DEFAULT_CONFIG.pds4.values()
        for schema in entry['schemas'].values()
        for url in (f'{schema["location"]}.xsd', f'{schema["location"]}.sch')
        if not (SCHEMA_COPIES / url.rsplit('/', 1)[-1]).is_file()
    ]
    assert missing == []


def test_every_schema_a_local_copy_imports_has_a_local_copy() -> None:
    """Each XML schema a copy imports, by the URL of the import, is copied too."""
    missing = []
    for copy in sorted(SCHEMA_COPIES.glob('*.xsd')):
        for element in etree.parse(str(copy)).getroot().iter(XS_IMPORT):
            url = str(element.get('schemaLocation'))
            if not (SCHEMA_COPIES / url.rsplit('/', 1)[-1]).is_file():
                missing.append(url)
    assert missing == []


def test_a_url_with_no_local_copy_is_a_finding_that_names_it() -> None:
    """A URL the schema directory holds no file for is a finding, and nothing is validated."""
    location = f'http://pds.nasa.gov/pds4/pds/v1 {UNCOPIED_PDS_SCHEMA}'
    resolved = label_schema('bundle.lblx', bare_bundle_label(location), SCHEMAS)
    expected = Finding(
        'bundle.lblx',
        CheckName.XSD,
        '',
        f'declares the XML schema {UNCOPIED_PDS_SCHEMA}, but no file of its name is in '
        f'{SCHEMA_COPIES}',
    )
    assert resolved.findings == (expected,)


def test_a_schema_set_that_cannot_be_built_is_one_finding() -> None:
    """A namespace paired with a Schematron file builds no set, and is one finding."""
    location = f'http://pds.nasa.gov/pds4/pds/v1 {PDS_SCHEMATRON}'
    resolved = label_schema('bundle.lblx', bare_bundle_label(location), SCHEMAS)
    expected = Finding(
        'bundle.lblx',
        CheckName.XSD,
        '',
        'its XML schemas cannot be built, so it is not validated against them: '
        "'{http://purl.oclc.org/dsdl/schematron}schema' is not an element of the schema",
    )
    assert resolved == LabelSchema(schema=None, findings=(expected,))
