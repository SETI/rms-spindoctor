"""The bundle environment over a cohort and the templates its dataset ships.

Where the plumbing tests of this package run a duck-typed dataset over templates the
test itself wrote, the environment here runs the registered dataset a cohort's bundle is
built with over the templates that dataset ships, on the products the cohort writes.  It
is how a question about what a label *says* is asked -- an epoch, a target, a described
data object, a name or a version.
"""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort

from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.bundle_products import generate_bundle_products
from spindoctor.cli.pds4.collections import generate_collection_files
from spindoctor.cli.pds4.global_index import generate_global_index_files
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset import DataSet

STAND_IN_USER_GUIDE = b'%PDF-1.4 a stand-in user guide\n'
"""What stands in for a bundle's user guide in a test: enough for the guide's label to render."""


@dataclass
class CohortBundleEnv:
    """A bundle-generation environment over a cohort and its bundle's shipped templates.

    Where the plumbing environment controls every variable so that the substitution
    plumbing can be asserted on, this one controls none of them: the dataset is
    the registered one the cohort's bundle is built with, the templates are the
    ones that dataset ships, and the inputs are the documents and products a
    navigation run and the backplane stage leave behind.  What it is for is
    asserting what a label says, which nothing built out of stand-ins can
    answer.

    Attributes:
        dataset: The registered dataset the cohort's bundle is built with,
            serving its own ``pds4_*`` hooks and its own template directory.
        cohort: The written cohort, holding both input roots and every image.
        bundle_results_root: Where this test's bundle goes.
        bundle_dir: ``bundle_results_root / <the dataset's bundle name>``.
    """

    dataset: DataSet
    cohort: Cohort
    bundle_results_root: Path
    bundle_dir: Path


def make_cohort_bundle_env(cohort: Cohort, tmp_path: Path) -> CohortBundleEnv:
    """Build a bundle environment over the cohort, writing into ``tmp_path``.

    The cohort is read-only and shared by the session; only the bundle the run
    writes belongs to one test.

    Parameters:
        cohort: The session's cohort, holding the navigation and backplane
            roots the run reads.
        tmp_path: Base temporary directory for this test's bundle output.

    Returns:
        The populated :class:`CohortBundleEnv`.
    """
    dataset = cohort.dataset()
    bundle_results_root = tmp_path / 'bundle'
    bundle_results_root.mkdir(parents=True, exist_ok=True)
    return CohortBundleEnv(
        dataset=dataset,
        cohort=cohort,
        bundle_results_root=bundle_results_root,
        bundle_dir=bundle_results_root / dataset.pds4_bundle_name(),
    )


def stand_in_guide_templates(dataset: DataSet, directory: Path) -> Path:
    """Copy a dataset's template directory, with a stand-in user guide in the copy.

    A bundle holds its user guide, and the guide's label, only when the template directory
    holds the guide's PDF, which the shipped directory does not; a test hands the dataset
    the copy as its template directory.

    Parameters:
        dataset: The dataset whose template directory is copied.
        directory: Where the copy goes; it must not exist yet.

    Returns:
        The copy.
    """
    shutil.copytree(dataset.pds4_bundle_template_dir(), directory)
    (directory / dataset.pds4_user_guide_file_name()).write_bytes(STAND_IN_USER_GUIDE)
    return directory


def label_cohort_images(env: CohortBundleEnv, stubs: Sequence[str]) -> None:
    """Run the labels pass over some of the environment's cohort images, in turn.

    Parameters:
        env: The environment whose cohort the images are from and whose bundle the
            labels go into.
        stubs: The images to label, by results path stub.
    """
    for stub in stubs:
        generate_bundle_data_files(
            env.dataset,
            env.cohort.batch(stub),
            nav_results_root=FCPath(env.cohort.nav_results_root),
            backplane_results_root=FCPath(env.cohort.backplane_results_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )


def summarize_bundle(env: CohortBundleEnv) -> None:
    """Run the summary pass's three generators over the environment's bundle.

    They run in the order the pass runs them: the global index first, whose range of
    the products' epochs and whose targets the collection generator and the run-level
    products are handed, and the run-level products last.

    Parameters:
        env: The environment whose bundle is summarized.
    """
    bundle_results_root = FCPath(env.bundle_results_root)
    index = generate_global_index_files(bundle_results_root, env.dataset, MAIN_LOGGER)
    generate_collection_files(
        bundle_results_root, env.dataset, MAIN_LOGGER, epochs=index.epochs, targets=index.targets
    )
    generate_bundle_products(
        bundle_results_root, env.dataset, MAIN_LOGGER, epochs=index.epochs, targets=index.targets
    )


def write_cohort_bundle(cohort: Cohort, tmp_path: Path, stubs: Sequence[str]) -> CohortBundleEnv:
    """Build a bundle over some of a cohort's images, running both passes.

    Parameters:
        cohort: The session's cohort, holding the navigation and backplane roots the
            run reads.
        tmp_path: Base temporary directory for this test's bundle output.
        stubs: The images to bundle, by results path stub.

    Returns:
        The environment the bundle was written into.
    """
    env = make_cohort_bundle_env(cohort, tmp_path)
    label_cohort_images(env, stubs)
    summarize_bundle(env)
    return env
