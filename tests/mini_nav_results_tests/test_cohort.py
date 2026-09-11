"""Self-tests every bundle's cohort is held to.

Every registered cohort is written once for the session, and each test here runs
over each of them.  What they check is what the bundle stage reads off a cohort
and cannot check for itself: every clock triple a document records spanning the
epochs beside it, a real FITS whose HDUs a reader finds where a label says they
are, a metadata document naming the planes that FITS carries, every statistic in
the unit the index tables state, and every body placed where its cohort put it.
Nothing a cohort writes may reach the working tree.

What only one bundle's cohort can say -- the names its images take, its index
columns, its bundle directories, its holdings layout -- is held in a test module
named for that bundle.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from astropy.io import fits
from filecache import FCPath

from spindoctor.config import DEFAULT_CONFIG
from tests.mini_nav_results import COHORTS
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.sclk_readings import triples_disagreeing_with_their_epochs


@pytest.fixture(scope='module', params=sorted(COHORTS))
def cohort(request: pytest.FixtureRequest, mini_nav_cohorts: WrittenCohorts) -> Cohort:
    """Return one registered cohort, as the session wrote it, so a test runs over each.

    Parameters:
        request: Names, as its parameter, the cohort this run of a test is over.
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(COHORTS[request.param])


@pytest.fixture(scope='module')
def first_cohort(mini_nav_cohorts: WrittenCohorts) -> Cohort:
    """Return the first registered cohort by name, for a test that needs one and no more.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(COHORTS[min(COHORTS)])


def _hdu_names(entries: list[dict[str, Any]]) -> list[str]:
    """Return the HDU names one source's configured planes are written under, in order.

    The merge inserts each source's planes in sorted order and the writer names each
    HDU for its plane in upper case; the ring source's ``distance`` entry orders the
    merge and is not written.

    Parameters:
        entries: One source's configured backplane entries.

    Returns:
        The HDU names, in the order the writer writes them.
    """
    return sorted(entry['name'].upper() for entry in entries if entry['name'] != 'distance')


def test_every_clock_triple_spans_the_epochs_beside_it(cohort: Cohort) -> None:
    """A reading that is not the one its epoch converts to is an invented one.

    Every reader that subtracts two readings, or converts one back into an
    epoch, reads whatever a hand-authored triple happened to say.  The readers
    are the ones ``tests.sclk_readings`` registers, so a cohort whose host has no
    reader there fails here as well.
    """
    assert triples_disagreeing_with_their_epochs(cohort.documents()) == []


def test_each_fits_carries_the_hdus_the_configuration_implies(cohort: Cohort) -> None:
    """A reader opens a real FITS and finds the planes the configuration declares.

    Every body plane for an image with bodies, and every ring plane after them for
    an image with rings, each source's planes in sorted order: that is the merge's
    order, which every array's byte offset in the file is stated against, and not
    the order the configuration lists them in, so a merge that stopped sorting is
    reported.  The byte blob a stand-in writes has no HDUs to find.
    """
    body_hdus = _hdu_names(DEFAULT_CONFIG.backplanes.bodies)
    ring_hdus = _hdu_names(DEFAULT_CONFIG.backplanes.rings)
    found: dict[str, tuple[str, ...]] = {}
    expected: dict[str, tuple[str, ...]] = {}
    for image in cohort.images():
        if not image.navigated:
            continue
        with fits.open(cohort.backplane_results_root / f'{image.stub}_backplanes.fits') as hdul:
            found[image.image_name] = tuple(hdu.name for hdu in hdul)
        expected[image.image_name] = (
            'PRIMARY',
            'BODY_ID_MAP',
            *(body_hdus if len(image.bodies) > 0 else []),
            *(ring_hdus if image.rings else []),
        )
    assert found == expected


def test_each_backplane_document_names_the_planes_its_fits_carries(
    cohort: Cohort,
) -> None:
    """The index columns come from one and the arrays from the other.

    A document naming a plane the FITS does not carry, or missing one it does,
    puts a global index column beside an array that is not the one it measures.
    A frame with no ring pixels in view is read the same way, through the same
    key: the ring stage returns a result holding nothing rather than no result,
    so ``rings`` names an empty ``backplanes`` rather than being empty itself,
    and a document that has to be read defensively is one no run wrote.
    """
    disagreeing: list[str] = []
    for image in cohort.images():
        if not image.navigated:
            continue
        stem = cohort.backplane_results_root / image.stub
        with fits.open(Path(f'{stem}_backplanes.fits')) as hdul:
            in_the_fits = {hdu.name for hdu in hdul} - {'PRIMARY', 'BODY_ID_MAP'}
        metadata_path = FCPath(f'{stem}_backplane_metadata.json')
        document = json.loads(metadata_path.read_text(encoding='utf-8'))
        named: set[str] = set()
        for body in document['bodies'].values():
            named |= {name.upper() for name in body['backplanes']}
        named |= {name.upper() for name in document['rings']['backplanes']}
        if named != in_the_fits:
            disagreeing.append(
                f'{image.image_name}: the document names {sorted(named)} and the FITS '
                f'carries {sorted(in_the_fits)}'
            )
    assert disagreeing == []


def test_no_backplane_statistic_is_left_in_radians(cohort: Cohort) -> None:
    """Every statistic a document records is in the unit the index tables state.

    The tables are read by a person, and everything angular in them is degrees.
    A plane whose statistic is still in radians is a column of radians beside
    columns of degrees, saying nothing about which it is unless the reader
    happens to know the plane.  A plane declared in radians per pixel is the
    one exposed to it, its unit not being one an equality against ``rad``
    recognises.
    """
    in_radians: list[str] = []
    for image in cohort.images():
        if not image.navigated:
            continue
        stem = cohort.backplane_results_root / image.stub
        metadata_path = FCPath(f'{stem}_backplane_metadata.json')
        document = json.loads(metadata_path.read_text(encoding='utf-8'))
        planes = dict(document['rings']['backplanes'])
        for body in document['bodies'].values():
            planes |= body['backplanes']
        for name, statistics in planes.items():
            measure = statistics['units'].partition('/')[0]
            if measure.lower().startswith('rad'):
                in_radians.append(f'{image.image_name}: {name} is in {statistics["units"]}')
    assert in_radians == []


# This pins the sidecar writing center_uv down the frame first, v before u,
# which #253 records as a characterization of the writer rather than a decision
# taken; the fix there updates this test.
def test_each_body_is_placed_down_the_frame_and_sized_across_it(
    cohort: Cohort,
) -> None:
    """The writer states a body's center and its extent in opposite axis orders.

    ``center_uv`` is written down the frame first and ``size_uv`` across it
    first, which is a convention nothing in a document declares and everything
    that draws a body over an image depends on.  Both are read here against the
    geometry the cohort declared, so a transposition on either side is reported
    rather than absorbed by a body a swap describes just as well.
    """
    found: dict[str, tuple[list[float], list[float]]] = {}
    expected: dict[str, tuple[list[float], list[float]]] = {}
    for image in cohort.images():
        if not image.navigated:
            continue
        stem = cohort.backplane_results_root / image.stub
        metadata_path = FCPath(f'{stem}_backplane_metadata.json')
        document = json.loads(metadata_path.read_text(encoding='utf-8'))
        for body in image.bodies:
            entry = document['bodies'][body.name]
            found[body.name] = (entry['center_uv'], entry['size_uv'])
            center_v, center_u = body.center_vu
            radius_v, radius_u = body.radii_vu
            expected[body.name] = ([center_v, center_u], [2.0 * radius_u, 2.0 * radius_v])
    assert found == expected


def _paths_git_reports(repository: Path) -> list[str]:
    """Return every file in the checkout, tracked or not, read without quoting.

    Tracked files are listed whether or not they have changed, so a cohort
    product that was committed, and is therefore clean in every checkout after,
    is listed as surely as one left untracked; ``git status`` reports neither a
    clean file nor anything about it, so a guard over what it reports passes in
    every clean checkout of a repository holding one.  Ignored files are not
    listed, since ``git add`` passes over them.

    Asked for in the NUL-separated form, because the readable form quotes a
    path holding a space or a byte outside ASCII and a reader that does not
    unquote it then compares a name with a quotation mark on the end of it,
    which matches nothing -- so the one file most likely to be somewhere it
    should not be is the one a guard over the readable form cannot see.

    Parameters:
        repository: The checkout to ask about.

    Returns:
        Each file's path relative to the checkout.
    """
    reported = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [path for path in reported.split('\0') if path]


def test_no_cohort_product_reaches_the_working_tree(cohort: Cohort) -> None:
    """The cohort is built where it is torn down, and nothing it writes is committed.

    Its products are named for their images, so a file of one of those names
    anywhere in the repository is a build that escaped the temporary directory
    -- reported here rather than committed by whoever runs ``git add`` next.
    """
    repository = Path(__file__).resolve().parents[2]
    if not (repository / '.git').exists():
        pytest.skip('not a git checkout')
    product_names = {path.name for path in cohort.written}
    escaped = [path for path in _paths_git_reports(repository) if Path(path).name in product_names]
    assert escaped == []


def _run_git(repository: Path, *arguments: str) -> None:
    """Run one git command in a throwaway repository and nowhere else.

    A git hook exports the variables locating its own repository and index to
    whatever it runs, and with them set a write here would land in that one, so
    every ``GIT_`` variable is dropped from the command's environment.

    Parameters:
        repository: The throwaway repository's directory.
        *arguments: The command's arguments after ``git``.
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    subprocess.run(
        ['git', *arguments], cwd=repository, env=environment, capture_output=True, check=True
    )


@pytest.mark.parametrize('committed', [True, False], ids=['committed', 'untracked'])
def test_the_guard_lists_a_committed_or_untracked_cohort_product(
    first_cohort: Cohort, tmp_path: Path, committed: bool
) -> None:
    """A cohort product is one the guard sees, whether it was committed or never added.

    ``git status`` says nothing about a clean file, so a guard built on it passes
    in every clean checkout of a repository holding a committed product; and an
    untracked one, left by a build that escaped its temporary directory, is the
    likeliest way a product reaches the working tree at all.  Both cases first
    commit another file, as a real checkout has history beside the product.  The
    repository is a throwaway one, with its committer configured in it alone.

    Parameters:
        first_cohort: A registered cohort, whose products name the file.
        tmp_path: Directory the throwaway repository is built in.
        committed: Whether the product is added and committed, or written after
            the first commit and left unadded.
    """
    name = min(path.name for path in first_cohort.written)
    repository = tmp_path / 'repository'
    repository.mkdir()
    _run_git(repository, 'init', '--quiet')
    _run_git(repository, 'config', 'user.name', 'Cohort Guard')
    _run_git(repository, 'config', 'user.email', 'cohort-guard@example.invalid')
    _run_git(repository, 'config', 'commit.gpgsign', 'false')
    (repository / 'README').write_text('No cohort product.\n', encoding='utf-8')
    _run_git(repository, 'add', '--all')
    _run_git(repository, 'commit', '--quiet', '--message', 'Commit something else')
    (repository / 'data').mkdir()
    (repository / 'data' / name).write_bytes(b'a cohort product')
    if committed:
        _run_git(repository, 'add', '--all')
        _run_git(repository, 'commit', '--quiet', '--message', 'Commit a cohort product')
    assert f'data/{name}' in _paths_git_reports(repository)
