"""The bundle check over the Cassini ISS Saturn cohort's bundle: the gate.

The cohort's bundle is built twice, as ``sd_create_bundle`` builds it from the templates
the package ships -- once plain, and once from a copy of the template directory holding
a stand-in user guide -- and the check is run over each.  What it finds is exactly what
is known of the bundle, each finding's file, check, location and message -- or, for an
XML schema error, its kind, since xmlschema's wording is not the check's -- and nothing
else.  The errors:

- the ``TODO DOI`` placeholder of the bundle label, in both builds, and the two of the
  user guide's label, in the build with the guide, until the DOIs are registered;
- each data label's empty ``cassini:ISS_Specific_Attributes``, until the Cassini ISS
  label facts it states are recorded by the navigation.

The warnings, in the plain build alone: each reference to the user guide, which a bundle
written without the guide's PDF does not hold, and which the PDS ``validate`` tool
reports as a warning too.  The build with the guide has none.

A finding the check makes over these bundles that is not among these fails the test, and
so does one of these the check stops making.  The check reads the schemas from the local
copies the tests keep, as ``--schema-dir`` has it do, and a test holds it to opening no
connection while it does; one more, marked ``integration``, has it fetch the schemas by
their URLs and holds it to the same findings.
"""

import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest
from filecache import FileCache
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check import CheckName, Finding, Severity, check_bundle, schemas
from spindoctor.cli.pds4.check.elements import child_text

from ..cohort_bundle import (
    CohortBundleEnv,
    label_cohort_images,
    make_cohort_bundle_env,
    stand_in_guide_templates,
    summarize_bundle,
    write_cohort_bundle,
)
from .controls import SCHEMA_COPIES, parse

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

GUIDE_LABEL = 'document/user_guide/cassini-iss-saturn-backplanes-user-guide.lblx'
"""The user guide's label, in a bundle written with the guide."""

GUIDE_LID_PART = 'document:backplanes-user-guide'
"""What the user guide's logical identifier adds to the bundle's."""

BUNDLE_DOI = '/Product_Bundle/Identification_Area/Citation_Information/doi'
"""The bundle label's DOI."""

GUIDE_DOIS = (
    '/Product_Document/Identification_Area/Citation_Information/doi',
    '/Product_Document/Document/doi',
)
"""The user guide label's two DOIs."""

DOI_KIND = 'XsdPatternFacets'
"""The kind of XML schema error a DOI placeholder is: its pattern facet fails."""

ISS_ATTRIBUTES = (
    '/Product_Observational/Observation_Area/Mission_Area/cassini:Cassini/'
    'cassini:ISS_Specific_Attributes'
)
"""The block of a data label the Cassini ISS label facts go in."""

EMPTY = 'is empty: it holds no element and no text'
"""What the integrity check says of an element holding nothing."""

DATA_GUIDE_REFERENCE = '/Product_Observational/Reference_List/Internal_Reference[1]/lid_reference'
"""Where a data label refers to the user guide."""

BROWSE_GUIDE_REFERENCE = '/Product_Browse/Reference_List/Internal_Reference[1]/lid_reference'
"""Where a browse label refers to the user guide."""

RUN_LEVEL_GUIDE_REFERENCES = (
    ('bundle.lblx', '/Product_Bundle/Reference_List/Internal_Reference[1]/lid_reference'),
    (
        'data/collection_data.lblx',
        '/Product_Collection/Reference_List/Internal_Reference[1]/lid_reference',
    ),
    (
        'spice_kernels/kernels.lblx',
        '/Product_SPICE_Kernel/Reference_List/Internal_Reference/lid_reference',
    ),
)
"""Where the bundle, data collection and metakernel labels refer to the user guide."""

_Known = tuple[str, str, str, str]
"""A finding as the gate compares it: its file, check, location, and kind or message."""


@dataclass(frozen=True)
class _Checked:
    """A bundle, what the check found over it, and its files before and after the check.

    Attributes:
        env: The environment the bundle was written in.
        findings: What the check found.
        before: Each path under the bundle's directory, with its modification time, before
            the check.
        after: The same, after it.
    """

    env: CohortBundleEnv
    findings: list[Finding]
    before: dict[str, int]
    after: dict[str, int]


def _snapshot(bundle_dir: Path) -> dict[str, int]:
    """Return every path under a directory, with its modification time.

    Parameters:
        bundle_dir: The directory.

    Returns:
        Each path's modification time in nanoseconds, by its path relative to the
        directory.
    """
    return {
        path.relative_to(bundle_dir).as_posix(): path.stat().st_mtime_ns
        for path in bundle_dir.rglob('*')
    }


def _check(env: CohortBundleEnv) -> _Checked:
    """Run the check over an environment's bundle.

    Parameters:
        env: The environment the bundle was written in.

    Returns:
        What it found, and the bundle's files before and after.
    """
    before = _snapshot(env.bundle_dir)
    findings = check_bundle(env.bundle_dir, config=env.dataset.config, schema_dir=SCHEMA_COPIES)
    return _Checked(env=env, findings=findings, before=before, after=_snapshot(env.bundle_dir))


@pytest.fixture(scope='module')
def plain(mini_nav_cohorts: WrittenCohorts, tmp_path_factory: pytest.TempPathFactory) -> _Checked:
    """Return the cohort's bundle written without a user guide, checked once.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.
        tmp_path_factory: Factory the bundle's directory is made under.

    Returns:
        The bundle and what the check found over it.
    """
    cohort = mini_nav_cohorts(CohortCassiniISSSaturn)
    return _check(write_cohort_bundle(cohort, tmp_path_factory.mktemp('plain'), NAVIGATED_STUBS))


@pytest.fixture(scope='module')
def with_guide(
    mini_nav_cohorts: WrittenCohorts, tmp_path_factory: pytest.TempPathFactory
) -> _Checked:
    """Return the cohort's bundle written with a stand-in user guide, checked once.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.
        tmp_path_factory: Factory the bundle's directory is made under.

    Returns:
        The bundle and what the check found over it.
    """
    cohort = mini_nav_cohorts(CohortCassiniISSSaturn)
    root = tmp_path_factory.mktemp('guide')
    env = make_cohort_bundle_env(cohort, root)
    templates = stand_in_guide_templates(env.dataset, root / 'templates')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(env.dataset, 'pds4_bundle_template_dir', lambda: str(templates))
        label_cohort_images(env, NAVIGATED_STUBS)
        summarize_bundle(env)
    return _check(env)


def _errors(env: CohortBundleEnv, *, guide: bool) -> list[_Known]:
    """Return the errors known of the bundle.

    Parameters:
        env: The environment the bundle was written in.
        guide: Whether the bundle was written with its user guide.

    Returns:
        The known errors, sorted.
    """
    stubs = [
        env.dataset.pds4_path_stub(env.cohort.batch(stub).image_files[0])
        for stub in NAVIGATED_STUBS
    ]
    known: list[_Known] = [('bundle.lblx', CheckName.XSD, BUNDLE_DOI, DOI_KIND)]
    known += [
        (f'data/{stub}_backplanes.lblx', CheckName.INTEGRITY, ISS_ATTRIBUTES, EMPTY)
        for stub in stubs
    ]
    if guide:
        known += [(GUIDE_LABEL, CheckName.XSD, location, DOI_KIND) for location in GUIDE_DOIS]
    return sorted(known)


def _warnings(env: CohortBundleEnv) -> list[_Known]:
    """Return the warnings known of the bundle written without its user guide.

    Parameters:
        env: The environment the bundle was written in.

    Returns:
        The known warnings, sorted: each reference to the user guide.
    """
    stubs = [
        env.dataset.pds4_path_stub(env.cohort.batch(stub).image_files[0])
        for stub in NAVIGATED_STUBS
    ]
    bundle_lid = child_text(
        parse(env.bundle_dir / 'bundle.lblx').getroot(), 'Identification_Area', 'logical_identifier'
    )
    message = f'refers to {bundle_lid}:{GUIDE_LID_PART}, which no label of the tree declares'
    references = [
        *((f'data/{stub}_backplanes.lblx', DATA_GUIDE_REFERENCE) for stub in stubs),
        *((f'browse/{stub}_summary.lblx', BROWSE_GUIDE_REFERENCE) for stub in stubs),
        *RUN_LEVEL_GUIDE_REFERENCES,
    ]
    return sorted((file, CheckName.INTEGRITY, location, message) for file, location in references)


def _found(checked: _Checked, severity: Severity) -> list[_Known]:
    """Return what the check found of one severity.

    Parameters:
        checked: The bundle and what the check found over it.
        severity: The severity.

    Returns:
        The findings of that severity, sorted, each by its kind where it has one and by
        its message otherwise.
    """
    return sorted(
        (finding.file, finding.check, finding.location, finding.kind or finding.message)
        for finding in checked.findings
        if finding.severity is severity
    )


def test_the_check_over_the_plain_bundle_finds_the_known_errors_and_no_other(
    plain: _Checked,
) -> None:
    """Over the bundle written without a guide, the known errors and no other."""
    assert _found(plain, Severity.ERROR) == _errors(plain.env, guide=False)


def test_the_check_over_the_plain_bundle_warns_of_each_guide_reference_and_nothing_else(
    plain: _Checked,
) -> None:
    """Over the bundle written without a guide, a warning for each guide reference."""
    assert _found(plain, Severity.WARNING) == _warnings(plain.env)


def test_the_check_over_the_bundle_with_a_guide_finds_the_known_errors_and_no_other(
    with_guide: _Checked,
) -> None:
    """Over the bundle written with a stand-in guide, the known errors and no other."""
    assert _found(with_guide, Severity.ERROR) == _errors(with_guide.env, guide=True)


def test_the_check_over_the_bundle_with_a_guide_warns_of_nothing(with_guide: _Checked) -> None:
    """Over the bundle written with a stand-in guide, no warning."""
    assert _found(with_guide, Severity.WARNING) == []


def test_the_check_writes_nothing(plain: _Checked) -> None:
    """The check leaves every file of the bundle as it was, and adds none."""
    assert plain.after == plain.before


def test_the_check_reading_the_local_copies_opens_no_connection(
    plain: _Checked, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading the schemas from a directory, the check fetches nothing.

    The copies are copied afresh, so that the check builds every set of schemas and
    compiles every Schematron again; the schema cache is an empty one of the test's own,
    so that a fetch could not be met from an earlier one; and every connection the
    process tries is refused and counted.
    """
    copies = tmp_path / 'schemas'
    shutil.copytree(SCHEMA_COPIES, copies)
    (tmp_path / 'cache').mkdir()
    empty = FileCache('schemas', cache_root=tmp_path / 'cache')
    monkeypatch.setattr(schemas, 'schema_cache', lambda: empty)
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        """Count a connection, and refuse it.

        Parameters:
            address: Where the connection was to go.

        Raises:
            OSError: Always.
        """
        attempts.append(address)
        raise OSError('the check tried to open a connection')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    check_bundle(plain.env.bundle_dir, config=plain.env.dataset.config, schema_dir=copies)
    assert attempts == []


@pytest.mark.integration
def test_the_check_fetching_the_schemas_finds_what_it_finds_from_the_local_copies(
    plain: _Checked, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetched by their URLs, the schemas give the findings the local copies give.

    The fetches go through a file cache of the test's own, so that each schema is fetched
    here rather than found in a cache an earlier check left.
    """
    own = FileCache('schemas', cache_root=tmp_path)
    monkeypatch.setattr(schemas, 'schema_cache', lambda: own)
    assert check_bundle(plain.env.bundle_dir, config=plain.env.dataset.config) == plain.findings
