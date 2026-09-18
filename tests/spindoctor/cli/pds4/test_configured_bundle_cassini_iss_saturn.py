"""The Cassini ISS Saturn bundle as its configuration names, versions and declares it.

The bundle's name and version, the information model its labels are written against and
the schema of each dictionary they declare are each set in one place, the dataset's
``pds4`` configuration block.  These tests build the cohort's bundle under a
configuration that changes them, from the templates the package ships, and read every
file it wrote; and hold the shipped configuration's schemas to the shipped templates.
"""

import re
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from ruamel.yaml import YAML
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .cohort_bundle import CohortBundleEnv, label_cohort_images, summarize_bundle
from .conftest import read_csv_rows

PDS4_NAMESPACES = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
"""The PDS4 common dictionary's namespace, under the prefix the paths below use."""

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

OTHER_NAME = 'saturn_backplanes_other_bundle'
"""A bundle name the tests configure in place of the shipped one."""

OTHER_VERSION = '3.7'
"""A bundle version the tests configure in place of the shipped one."""

TEXT_SUFFIXES = {'.csv', '.ker', '.lblx', '.tab', '.txt'}
"""The suffixes of a bundle's text files: its labels, inventories, index tables,
metakernel, readme and supplemental files.  The rest are FITS, PNG and PDF files."""

LIDVID = re.compile(r'(urn:nasa:pds:[a-z0-9._:-]+?)::([0-9]+\.[0-9]+)')
"""A LIDVID, its LID and its version captured."""

UNRENDERED = re.compile(r'\$[A-Za-z_][A-Za-z0-9_]*\$')
"""A template variable a render left in place."""

NAMESPACES = {
    'pds': 'http://pds.nasa.gov/pds4/pds/v1',
    'disp': 'http://pds.nasa.gov/pds4/disp/v1',
    'geom': 'http://pds.nasa.gov/pds4/geom/v1',
    'cassini': 'http://pds.nasa.gov/pds4/mission/cassini/v1',
}
"""The namespace of each dictionary the bundle's labels declare, by its prefix."""

XML_MODEL = re.compile(r'<\?xml-model href="([^"]+)"')
"""The Schematron an ``xml-model`` instruction names."""

XSI_SCHEMA_LOCATION = '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'
"""The attribute pairing each namespace a label declares with its XML schema."""

SCHEMA_VARIABLE = re.compile(r'\$PDS4_([A-Z]+)_SCHEMA(?:_XSD)?\$')
"""A template variable naming a dictionary's schema, the dictionary's prefix captured."""

OTHER_INFORMATION_MODEL_VERSION = '9.8.7.6'
"""An information model version the tests configure in place of the shipped one."""

VERSION_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
"""The digits a PDS4 schema's file name writes the parts of a version in, one each."""


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


def _bundle_under(
    cohort: CohortCassiniISSSaturn, tmp_path: Path, overrides: dict[str, Any]
) -> CohortBundleEnv:
    """Build the cohort's bundle under a configuration changed in its ``pds4`` block.

    The bundle is built from a copy of the shipped template directory holding a stand-in
    user guide, which the configuration names as the dataset's template directory, so
    that the guide's label and the document inventory's line for it are written too.

    Parameters:
        cohort: The session's cohort, holding the navigation and backplane roots the
            run reads.
        tmp_path: Base temporary directory for the templates, the configuration file and
            the bundle.
        overrides: Keys merged into ``pds4.coiss_saturn``, beside the template directory.

    Returns:
        The environment the bundle was written into.
    """
    shipped = cohort.dataset()
    templates = tmp_path / 'templates'
    shutil.copytree(shipped.pds4_bundle_template_dir(), templates)
    guide = templates / shipped.pds4_user_guide_file_name()
    guide.write_bytes(b'%PDF-1.4 a stand-in user guide\n')
    block = {'template_dir': str(templates), **overrides}
    override = tmp_path / 'override.yaml'
    YAML(typ='safe').dump({'pds4': {'coiss_saturn': block}}, override)
    config = Config()
    config.update_config(override)
    dataset = DataSetPDS3CassiniISSSaturn(cohort.holdings_root, config=config)
    bundle_results_root = tmp_path / 'bundle'
    bundle_results_root.mkdir()
    env = CohortBundleEnv(
        dataset=dataset,
        cohort=cohort,
        bundle_results_root=bundle_results_root,
        bundle_dir=bundle_results_root / dataset.pds4_bundle_name(),
    )
    label_cohort_images(env, NAVIGATED_STUBS)
    summarize_bundle(env)
    return env


def _texts(bundle_dir: Path) -> dict[str, str]:
    """Return every text file of a bundle, by its path relative to the bundle.

    Parameters:
        bundle_dir: The bundle's own directory.

    Returns:
        The content of each file whose suffix is in :data:`TEXT_SUFFIXES`.
    """
    return {
        path.relative_to(bundle_dir).as_posix(): path.read_text(encoding='utf-8')
        for path in sorted(bundle_dir.rglob('*'))
        if path.suffix in TEXT_SUFFIXES
    }


def _is_under(lid: str, bundle_lid: str) -> bool:
    """Return whether a LID is a bundle's own or one of its collections' or products'.

    Parameters:
        lid: The LID.
        bundle_lid: The bundle's LID.

    Returns:
        True when ``lid`` is ``bundle_lid`` or begins with it and a colon.
    """
    return lid == bundle_lid or lid.startswith(f'{bundle_lid}:')


def _external_lidvids(texts: dict[str, str], bundle_lid: str) -> dict[str, list[str]]:
    """Return the LIDVIDs each file of a bundle names of products outside the bundle.

    Parameters:
        texts: The bundle's text files, by path relative to the bundle.
        bundle_lid: The bundle's LID.

    Returns:
        For each file, every LIDVID in it whose LID is not under ``bundle_lid``, sorted.
    """
    return {
        path: sorted(
            match.group(0)
            for match in LIDVID.finditer(text)
            if not _is_under(match.group(1), bundle_lid)
        )
        for path, text in texts.items()
    }


def _labels(bundle_dir: Path) -> dict[str, str]:
    """Return every label of a bundle, by its path relative to the bundle.

    Parameters:
        bundle_dir: The bundle's own directory.

    Returns:
        The text of each ``.lblx`` file under it.
    """
    return {
        label.relative_to(bundle_dir).as_posix(): label.read_text(encoding='utf-8')
        for label in sorted(bundle_dir.rglob('*.lblx'))
    }


def _schema_locations(label: Path) -> dict[str, str]:
    """Return the XML schema a label's ``xsi:schemaLocation`` gives each namespace.

    Parameters:
        label: The label.

    Returns:
        Each namespace the attribute names, mapped to the schema it names beside it.
    """
    pairs = ElementTree.parse(label).getroot().get(XSI_SCHEMA_LOCATION, '').split()
    return dict(zip(pairs[::2], pairs[1::2], strict=True))


def _version_code(version: str) -> str:
    """Return the code a PDS4 schema's file name gives a four-part version.

    Parameters:
        version: The version, as ``1.24.0.0``.

    Returns:
        Each part written as one digit of :data:`VERSION_DIGITS`, as ``1O00``.
    """
    return ''.join(VERSION_DIGITS[int(part)] for part in version.split('.'))


def test_the_bundle_takes_the_configured_name_and_version(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """Under another name and version, no file of the bundle holds the shipped ones.

    The bundle is written under the configured name, holding every collection its label
    declares.  No label, inventory, table, metakernel or readme holds the shipped name.
    Every LIDVID naming one of the bundle's own products, in a label or an inventory,
    carries the configured version; every label states it as each of its ``version_id``
    elements, and has a logical identifier under the configured name.  No template
    variable is left unrendered, in the readme and the inventories as in the labels.
    And every LIDVID naming a product outside the bundle -- a context product, the ISS
    data user guide, a schema -- is, file by file, the one the bundle built under the
    shipped configuration names: an external reference keeps its own version.
    """
    shipped = _bundle_under(cassini_cohort, tmp_path / 'shipped', {})
    env = _bundle_under(
        cassini_cohort,
        tmp_path / 'other',
        {'bundle_name': OTHER_NAME, 'bundle_version': OTHER_VERSION},
    )
    shipped_name = DEFAULT_CONFIG.pds4['coiss_saturn']['bundle_name']
    bundle_lid = f'urn:nasa:pds:{OTHER_NAME}'
    texts = _texts(env.bundle_dir)
    shipped_externals = _external_lidvids(
        _texts(shipped.bundle_dir), f'urn:nasa:pds:{shipped_name}'
    )
    labels = {
        path: ElementTree.parse(env.bundle_dir / path).getroot()
        for path in texts
        if path.endswith('.lblx')
    }
    versions = [
        (path, match.group(0))
        for path, text in texts.items()
        for match in LIDVID.finditer(text)
        if _is_under(match.group(1), bundle_lid) and match.group(2) != OTHER_VERSION
    ]
    version_ids = [
        (path, element.text)
        for path, root in labels.items()
        for element in root.iter(f'{{{PDS4_NAMESPACES["pds"]}}}version_id')
        if element.text != OTHER_VERSION
    ]
    identifiers = [
        (path, lid)
        for path, root in labels.items()
        if not _is_under(
            lid := root.findtext(
                'pds:Identification_Area/pds:logical_identifier', '', PDS4_NAMESPACES
            ).strip(),
            bundle_lid,
        )
    ]
    assert (env.bundle_results_root / OTHER_NAME / 'bundle.lblx').is_file()
    assert [path for path, text in texts.items() if shipped_name in text] == []
    assert versions == []
    assert version_ids == []
    assert identifiers == []
    assert [path for path, text in texts.items() if UNRENDERED.search(text)] == []
    assert [lidvid for lidvids in shipped_externals.values() for lidvid in lidvids] != []
    assert _external_lidvids(texts, bundle_lid) == shipped_externals


@pytest.mark.parametrize('prefix', list(NAMESPACES))
def test_a_schema_moved_in_the_configuration_moves_in_every_label_declaring_it(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path, prefix: str
) -> None:
    """A dictionary's schema moved in the configuration moves in every label declaring it.

    The schema's location and LIDVID are changed in the one place the configuration
    gives them.  Every label declaring the dictionary's namespace names the moved
    Schematron in an ``xml-model`` instruction and the moved XML schema for that
    namespace in its ``xsi:schemaLocation``; no label names the shipped location; and the
    XML schema inventory lists the moved LIDVID in the shipped one's place.

    Parameters:
        cassini_cohort: The cohort the bundle is built over.
        tmp_path: Base temporary directory.
        prefix: The dictionary whose schema is moved, by its namespace's prefix.
    """
    shipped = DEFAULT_CONFIG.pds4['coiss_saturn']['schemas']
    moved = {
        'location': f'https://example.invalid/{prefix}/v1/PDS4_MOVED',
        'lidvid': f'urn:nasa:pds:system_bundle:xml_schema:{prefix}-moved::9.9',
    }
    env = _bundle_under(cassini_cohort, tmp_path, {'schemas': {prefix: moved}})
    labels = _labels(env.bundle_dir)
    declaring = [path for path, text in labels.items() if f'="{NAMESPACES[prefix]}"' in text]
    without_schematron = [
        path
        for path in declaring
        if f'{moved["location"]}.sch' not in XML_MODEL.findall(labels[path])
    ]
    without_schema = [
        path
        for path in declaring
        if _schema_locations(env.bundle_dir / path).get(NAMESPACES[prefix])
        != f'{moved["location"]}.xsd'
    ]
    inventory = read_csv_rows(env.bundle_dir / 'xml_schema' / 'collection_xml_schema.csv')
    assert declaring != []
    assert without_schematron == []
    assert without_schema == []
    assert [path for path, text in labels.items() if shipped[prefix]['location'] in text] == []
    assert [lidvid for _, lidvid in inventory] == [
        moved['lidvid'] if name == prefix else entry['lidvid'] for name, entry in shipped.items()
    ]


def test_every_label_states_the_configured_information_model_version(
    cassini_cohort: CohortCassiniISSSaturn, tmp_path: Path
) -> None:
    """Every label states the information model version the configuration gives."""
    env = _bundle_under(
        cassini_cohort, tmp_path, {'information_model_version': OTHER_INFORMATION_MODEL_VERSION}
    )
    stated = {
        path: ElementTree.parse(env.bundle_dir / path)
        .getroot()
        .findtext('pds:Identification_Area/pds:information_model_version', '', PDS4_NAMESPACES)
        for path in _labels(env.bundle_dir)
    }
    assert (env.bundle_dir / 'bundle.lblx').is_file()
    assert {path: v for path, v in stated.items() if v != OTHER_INFORMATION_MODEL_VERSION} == {}


def test_the_shipped_configuration_gives_a_schema_for_each_dictionary_the_templates_declare(
    tmp_path: Path,
) -> None:
    """The shipped configuration gives a schema for exactly the dictionaries declared.

    The XML schema collection lists every schema the configuration gives, so one no
    template declares would be listed for no label, and a template declaring a dictionary
    the configuration gives no schema for would not render.
    """
    dataset = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings')
    template_dir = Path(dataset.pds4_bundle_template_dir())
    declared = {
        prefix.lower()
        for template in template_dir.glob('*.lblx')
        for prefix in SCHEMA_VARIABLE.findall(template.read_text(encoding='utf-8'))
    }
    assert set(DEFAULT_CONFIG.pds4['coiss_saturn']['schemas']) == declared


def test_the_shipped_pds_schema_is_of_the_shipped_information_model() -> None:
    """The shipped schemas are of the information model build the entry names.

    The ``pds`` Schematron requires every label to state the information model version
    its own build is of, so the two move together: the ``pds`` schema's file name carries
    the version's code, ``1O00`` for ``1.24.0.0``, and its LIDVID the version's first two
    parts, ``::1.24``.  Every other dictionary's file name carries the same code, the build
    it was generated from, before its own.
    """
    entry = DEFAULT_CONFIG.pds4['coiss_saturn']
    version = entry['information_model_version']
    code = _version_code(version)
    schemas = entry['schemas']
    of_another_build = [
        name
        for name, schema in schemas.items()
        if name != 'pds'
        and not schema['location'].rsplit('/', 1)[-1].startswith(f'PDS4_{name.upper()}_{code}_')
    ]
    assert schemas['pds']['location'].rsplit('/', 1)[-1] == f'PDS4_PDS_{code}'
    assert schemas['pds']['lidvid'].rsplit('::', 1)[-1] == '.'.join(version.split('.')[:2])
    assert of_another_build == []
