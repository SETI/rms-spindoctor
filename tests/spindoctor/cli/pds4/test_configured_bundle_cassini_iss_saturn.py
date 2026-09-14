"""The Cassini ISS Saturn bundle as its configuration names and versions it.

The bundle's name and version are each set in one place, the dataset's ``pds4``
configuration block.  These tests build the cohort's bundle under a configuration that
changes them, from the templates the package ships, and read every file it wrote.
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

from .conftest import CohortBundleEnv, label_cohort_images, summarize_bundle

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
    """
    env = _bundle_under(
        cassini_cohort,
        tmp_path,
        {'bundle_name': OTHER_NAME, 'bundle_version': OTHER_VERSION},
    )
    shipped_name = DEFAULT_CONFIG.pds4['coiss_saturn']['bundle_name']
    bundle_lid = f'urn:nasa:pds:{OTHER_NAME}'
    texts = _texts(env.bundle_dir)
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
