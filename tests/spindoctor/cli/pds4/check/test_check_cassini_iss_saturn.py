"""The bundle check over the Cassini ISS Saturn cohort's bundle: the gate.

The cohort's bundle is built twice, as ``sd_create_bundle`` builds it from the templates
the package ships -- once plain, and once from a copy of the template directory holding
a stand-in user guide -- and the check is run over each.  What it finds is exactly what
is known of the bundle, and nothing else:

- the ``TODO DOI`` placeholder of the bundle label, in both builds, and the two of the
  user guide's label, in the build with the guide, until the DOIs are registered;
- each data label's empty ``cassini:ISS_Specific_Attributes``, until the Cassini ISS
  label facts it states are recorded by the navigation;
- in the plain build, each reference to the user guide, which a bundle written without
  the guide's PDF does not hold.

A finding the check makes over these bundles that is not among these fails the test, and
so does one of these the check stops making.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.check import CheckName, Finding, check_bundle

from ..cohort_bundle import (
    CohortBundleEnv,
    label_cohort_images,
    make_cohort_bundle_env,
    stand_in_guide_templates,
    summarize_bundle,
    write_cohort_bundle,
)

NAVIGATED_STUBS = (LIMB_STUB, RINGS_STUB)
"""The cohort's two navigated images, by results path stub."""

GUIDE_LABEL = 'document/user_guide/cassini-iss-saturn-backplanes-user-guide.lblx'
"""The user guide's label, in a bundle written with the guide."""

BUNDLE_DOI = '/Product_Bundle/Identification_Area/Citation_Information/doi'
"""The bundle label's DOI."""

GUIDE_DOIS = (
    '/Product_Document/Identification_Area/Citation_Information/doi',
    '/Product_Document/Document/doi',
)
"""The user guide label's two DOIs."""

ISS_ATTRIBUTES = (
    '/Product_Observational/Observation_Area/Mission_Area/cassini:Cassini/'
    'cassini:ISS_Specific_Attributes'
)
"""The block of a data label the Cassini ISS label facts go in."""

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
"""Where the bundle label, the data collection label and the metakernel label refer to it."""


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
        Each path's modification time in nanoseconds, by its path relative to the directory.
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
    findings = check_bundle(env.bundle_dir, config=env.dataset.config)
    return _Checked(env=env, findings=findings, before=before, after=_snapshot(env.bundle_dir))


@pytest.fixture(scope='module')
def plain(mini_nav_cohorts: WrittenCohorts, tmp_path_factory: pytest.TempPathFactory) -> _Checked:
    """Return the cohort's bundle written without a user guide, checked once for the module.

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


def _known(env: CohortBundleEnv, *, guide: bool) -> list[tuple[str, str, str]]:
    """Return what is known of the bundle, as each finding's file, check and location.

    Parameters:
        env: The environment the bundle was written in.
        guide: Whether the bundle was written with its user guide.

    Returns:
        The known findings, sorted.
    """
    stubs = [
        env.dataset.pds4_path_stub(env.cohort.batch(stub).image_files[0])
        for stub in NAVIGATED_STUBS
    ]
    data_labels = [f'data/{stub}_backplanes.lblx' for stub in stubs]
    known: list[tuple[str, str, str]] = [('bundle.lblx', CheckName.XSD, BUNDLE_DOI)]
    known += [(label, CheckName.INTEGRITY, ISS_ATTRIBUTES) for label in data_labels]
    if guide:
        known += [(GUIDE_LABEL, CheckName.XSD, location) for location in GUIDE_DOIS]
    else:
        known += [(label, CheckName.INTEGRITY, DATA_GUIDE_REFERENCE) for label in data_labels]
        known += [
            (f'browse/{stub}_summary.lblx', CheckName.INTEGRITY, BROWSE_GUIDE_REFERENCE)
            for stub in stubs
        ]
        known += [
            (file, CheckName.INTEGRITY, location) for file, location in RUN_LEVEL_GUIDE_REFERENCES
        ]
    return sorted(known)


def _found(checked: _Checked) -> list[tuple[str, str, str]]:
    """Return what the check found, as each finding's file, check and location.

    Parameters:
        checked: The bundle and what the check found over it.

    Returns:
        The findings, sorted.
    """
    return sorted((finding.file, finding.check, finding.location) for finding in checked.findings)


def test_the_check_over_the_plain_bundle_finds_what_is_known_and_nothing_else(
    plain: _Checked,
) -> None:
    """Over the bundle written without a guide, the known findings and no other."""
    assert _found(plain) == _known(plain.env, guide=False)


def test_the_check_over_the_bundle_with_a_guide_finds_what_is_known_and_nothing_else(
    with_guide: _Checked,
) -> None:
    """Over the bundle written with a stand-in guide, the known findings and no other."""
    assert _found(with_guide) == _known(with_guide.env, guide=True)


def test_the_check_writes_nothing(plain: _Checked) -> None:
    """The check leaves every file of the bundle as it was, and adds none."""
    assert plain.after == plain.before
