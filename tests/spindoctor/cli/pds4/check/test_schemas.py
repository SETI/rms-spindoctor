"""The schemas a label names, resolved by URL to the local copies the tests keep."""

import shutil
import warnings
from pathlib import Path

import pytest
from lxml import etree

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import LabelSchema, SchemaSource, label_schema
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


def test_a_warning_not_xmlschemas_in_a_build_reaches_the_caller_and_is_no_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warning not xmlschema's, raised as a set is built, is passed on untouched.

    The garbage collector can raise a ``ResourceWarning`` for a connection something else
    left open at any moment, a build's among them; one is raised here from inside the
    build, through the hook xmlschema reads each URL by.  The schema is copied to the
    test's own directory, so that the set is built afresh.
    """
    shutil.copy(SCHEMA_COPIES / 'PDS4_PDS_1O00.xsd', tmp_path / 'PDS4_PDS_1O00.xsd')
    mapped = SchemaSource.mapped

    def leaking(self: SchemaSource, url: str) -> str:
        """Raise another's warning, as the garbage collector can, and map a URL.

        Parameters:
            url: The URL xmlschema is about to read.

        Returns:
            The local file the URL resolves to.
        """
        warnings.warn('unclosed database in <sqlite3.Connection>', ResourceWarning, stacklevel=1)
        return mapped(self, url)

    monkeypatch.setattr(SchemaSource, 'mapped', leaking)
    location = 'http://pds.nasa.gov/pds4/pds/v1 https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.xsd'
    with pytest.warns(ResourceWarning, match='unclosed database'):
        resolved = label_schema('bundle.lblx', bare_bundle_label(location), SchemaSource(tmp_path))
    assert resolved.findings == ()


def test_a_relative_schema_directory_is_the_one_the_source_was_built_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two sources of one relative spelling in two directories are two sources."""
    for name in ('one', 'two'):
        (tmp_path / name / 'copies').mkdir(parents=True)
        (tmp_path / name / 'copies' / 'example.xsd').write_text(name, encoding='utf-8')
    monkeypatch.chdir(tmp_path / 'one')
    one = SchemaSource(Path('copies'))
    monkeypatch.chdir(tmp_path / 'two')
    two = SchemaSource(Path('copies'))
    url = 'https://example.invalid/v1/example.xsd'
    # Unequal is the property the schema set's cache is keyed on; the files are what
    # that buys, and one source answering for both is the defect.
    assert one != two
    assert one.locate(url) == (tmp_path / 'one' / 'copies' / 'example.xsd').resolve()
    assert two.locate(url) == (tmp_path / 'two' / 'copies' / 'example.xsd').resolve()
