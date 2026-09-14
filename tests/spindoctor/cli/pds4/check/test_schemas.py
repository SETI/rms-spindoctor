"""The schemas the package ships, and a label's XML schemas resolved to them."""

import shutil
from pathlib import Path

import pytest
from lxml import etree

from spindoctor.cli.pds4.check import schemas
from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.schemas import SCHEMA_DIRECTORY, label_schema, shipped_copy
from spindoctor.config import DEFAULT_CONFIG

UNSHIPPED_PDS_SCHEMA = 'https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1N00.xsd'
"""The URL of a build of the common dictionary's XML schema the package does not ship."""


def _label(schema_location: str) -> etree._ElementTree:
    """Return a bundle label that declares the given XML schemas and holds nothing.

    Parameters:
        schema_location: Its ``xsi:schemaLocation``, each namespace followed by a URL.

    Returns:
        The label, parsed.
    """
    text = (
        '<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="{schema_location}"/>'
    )
    return etree.fromstring(text.encode('ascii')).getroottree()


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
    resolved = label_schema('bundle.lblx', _label(location))
    expected = Finding(
        'bundle.lblx',
        CheckName.XSD,
        '',
        f'declares the XML schema {UNSHIPPED_PDS_SCHEMA}, of which the package ships no copy',
    )
    assert resolved.findings == (expected,)


def test_an_import_with_no_shipped_schema_is_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building a set whose import finds no shipped schema warns, and the warning is found.

    The shipped directory is stood in for by a copy holding the common, geometry and
    Cassini dictionaries' schemas and not the cartography dictionary's, which the Cassini
    schema imports.
    """
    for name in ('PDS4_PDS_1O00.xsd', 'PDS4_GEOM_1O00_19B0.xsd', 'PDS4_CASSINI_1O00_1800.xsd'):
        shutil.copy(SCHEMA_DIRECTORY / name, tmp_path / name)
    monkeypatch.setattr(schemas, 'SCHEMA_DIRECTORY', tmp_path)
    location = (
        'http://pds.nasa.gov/pds4/pds/v1 https://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1O00.xsd '
        'http://pds.nasa.gov/pds4/mission/cassini/v1 '
        'https://pds.nasa.gov/pds4/mission/cassini/v1/PDS4_CASSINI_1O00_1800.xsd'
    )
    resolved = label_schema('bundle.lblx', _label(location))
    messages = [finding.message for finding in resolved.findings]
    assert messages == [
        "building its XML schemas warned: Import of namespace 'http://pds.nasa.gov/pds4/cart/v1' "
        "from ['https://pds.nasa.gov/pds4/cart/v1/PDS4_CART_1O00_1970.xsd'] failed: block "
        'access to remote resource https://pds.nasa.gov/pds4/cart/v1/PDS4_CART_1O00_1970.xsd.'
    ]
