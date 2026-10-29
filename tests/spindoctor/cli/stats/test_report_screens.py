"""The report sections whose fixtures a tree of ordinary images cannot supply.

Four sections say something only about images built to make them say it, and the
frozen regression tree holds none of those images.  It has one image size per
camera, so the per-camera statistics are never a pool of anything.  Every one of
its successes is far inside the search box and no two of its Cassini frames
share a spacecraft-clock count, so no table it produces is ever capped and no
reduction it makes ever has two candidates to choose between.  And every
document in it either succeeded or failed outright, so nothing in it is an image
the navigator could not decide about.

Each fixture here is written for one of those, and each is small enough that the
number the section prints can be worked out by hand from the offsets that went
in.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.spindoctor.conftest import metadata_document, write_metadata

from spindoctor.cli.stats.report import build_report
from spindoctor.nav_records import ImageFacts, Selection, TreeRecordSource, UnreadableFile

from .conftest import ReplayedFacts

_SUSPECT_TABLE_HEADER = '| image | instrument | dV | dU | magnitude | limit (v, u) |'
"""The header row of the suspect-offset table, for finding the rows under it."""


def _tree_of(root: Path, documents: dict[str, dict[str, Any]]) -> Path:
    """Write one document per stub into a results root.

    Parameters:
        root: The results root to write under.
        documents: Results path stub to the document written at it.

    Returns:
        The root, written.
    """
    for stub, document in documents.items():
        write_metadata(root, stub, document)
    return root


def _report_text(tmp_path: Path, documents: dict[str, dict[str, Any]], **options: Any) -> str:
    """Write one report over a tree of the given documents and read it back.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        documents: Results path stub to the document written at it.
        options: Report options, passed through to ``build_report``.

    Returns:
        The Markdown the report wrote.
    """
    root = _tree_of(tmp_path / 'results', documents)
    out = tmp_path / 'report'
    out.mkdir(parents=True, exist_ok=True)
    with TreeRecordSource([root.as_posix()]) as source:
        build_report(source, out, **options)
    return (out / 'report.md').read_text(encoding='utf-8')


def _suspect_rows(text: str) -> list[str]:
    """The data rows of the suspect-offset table, in the order it printed them.

    Parameters:
        text: The whole report.

    Returns:
        One row per image the table listed, and an empty list when it listed
        none.

    Raises:
        ValueError: If the report holds no suspect table at all, which is a
            section that stopped being printed rather than a table with no rows.
    """
    lines = text.splitlines()
    if _SUSPECT_TABLE_HEADER not in lines:
        raise ValueError('the report prints no suspect-offset table')
    start = lines.index(_SUSPECT_TABLE_HEADER) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith('|'):
            break
        rows.append(line)
    return rows


# ---------------------------------------------------------------------------
# An image the navigator could not decide about
# ---------------------------------------------------------------------------


def _conflicted_documents() -> dict[str, dict[str, Any]]:
    """Return one Cassini success and one Cassini image the ensemble could not settle.

    The conflicted image records an offset, because that is what the ensemble
    writes when its techniques agree on nothing: a pointing it arrived at and
    will not stand behind.  Its offset is close enough to the WAC search limit
    to be screened as suspect, and its feature inventory names a body, so an
    image counted as a success would show up in the offset statistics, in the
    suspect screen and on the successful side of the per-body shares.

    Returns:
        Results path stub to document.
    """
    return {
        'COISS_2001/N2000000001_1_CALIB': metadata_document(
            image_name='N2000000001_1_CALIB.IMG',
            camera='NAC',
            image_shape=[1024, 1024],
            offset=[1.5, -2.5],
        ),
        'COISS_2001/W2000000002_1_CALIB': metadata_document(
            image_name='W2000000002_1_CALIB.IMG',
            camera='WAC',
            image_shape=[1024, 1024],
            status='conflicted',
            status_reason='techniques_disagree',
            offset=[4.9, 0.0],
            confidence=0.2,
            confidence_rank='conflicted',
        ),
    }


def test_a_conflicted_image_is_no_part_of_the_successful_offsets(tmp_path: Path) -> None:
    """A pointing the ensemble would not stand behind is not one of the offsets it did.

    The offset statistics, the per-camera histograms and the suspect screen are
    all headed "successful images".  Counting an image the ensemble could not
    settle would put a WAC row into a table with no navigated WAC image in it,
    and would screen a second image against a limit it was never asked about.
    """
    text = _report_text(tmp_path, _conflicted_documents())
    assert 'Suspect images: 0 (0.0%) of 1 screened.' in text
    assert '| coiss | WAC |' not in text


def test_a_conflicted_image_counts_as_a_failure_against_its_body(tmp_path: Path) -> None:
    """Every outcome that is not success is a failure, and only success is success.

    The per-body shares divide the images naming a body into the ones that
    worked and the ones that did not, so a bucket keyed on one named failure
    status rather than on the absence of success would count this image as a
    body the navigator handled.
    """
    text = _report_text(tmp_path, _conflicted_documents())
    assert '| IAPETUS | coiss | 1 (50.0%) | 1 (50.0%) | 0.500 |' in text


# ---------------------------------------------------------------------------
# The tables an option caps
# ---------------------------------------------------------------------------

_CAPPED_OFFSETS: dict[str, tuple[float, float]] = {
    'COISS_2001/N1000000001_1_CALIB': (49.0, 0.0),
    'COISS_2001/W1000000001_1_CALIB': (4.7, 0.0),
    'COISS_2001/N1000000002_1_CALIB': (47.5, 0.0),
    'COISS_2001/W1000000002_1_CALIB': (4.6, 0.0),
    'COISS_2001/N1000000003_1_CALIB': (10.0, 130.0),
    'COISS_2001/W1000000003_1_CALIB': (4.55, 0.0),
}
"""Three simultaneously shuttered Cassini pairs, every frame near its search limit.

The configured CALIB limits are ``(50, 140)`` for a 1024 px NAC frame and
``(5, 10)`` for a 256 px WAC frame, so every one of the six reaches at least 0.91
of a limit and no two of them reach the same fraction of one: the suspect table
is six rows deep and its order is decided by the offsets rather than by a tie.
The third pair is the one that reaches its limit on the U axis alone, and it is
also what makes the three BOTSIM residuals -- 2.000, 1.500 and 134.760 -- far
enough apart that a percentile over them is not the smallest of them.
"""

_CAPPED_TOP_N = 2
"""What the capped runs below pass as ``--top-n``, which is fewer than six."""


def _capped_documents() -> dict[str, dict[str, Any]]:
    """Return the six Cassini frames of :data:`_CAPPED_OFFSETS`.

    Returns:
        Results path stub to document.
    """
    documents = {}
    for stub, offset in _CAPPED_OFFSETS.items():
        name = stub.rsplit('/', 1)[-1]
        documents[stub] = metadata_document(
            image_name=f'{name}.IMG',
            camera='NAC' if name.startswith('N') else 'WAC',
            image_shape=[1024, 1024] if name.startswith('N') else [256, 256],
            offset=list(offset),
        )
    return documents


_MANY_SUSPECTS = 30
"""How many suspect images the uncapped table is measured over.

More than the twenty-five a bounded top-N structure elsewhere in the report
holds, so that a cap borrowed from one of those and applied quietly here is a
cap this measurement can see.
"""


def _many_suspect_documents() -> dict[str, dict[str, Any]]:
    """Return :data:`_MANY_SUSPECTS` Cassini NAC frames, every one near its search limit.

    The offsets run from 46.0 to 48.9 pixels against a V-axis limit of 50, so
    each of them reaches between 0.92 and 0.98 of the limit and every one is
    screened as suspect.

    Returns:
        Results path stub to document.
    """
    documents = {}
    for index in range(_MANY_SUSPECTS):
        name = f'N400000{4000 + index}_1_CALIB'
        documents[f'COISS_2001/{name}'] = metadata_document(
            image_name=f'{name}.IMG',
            camera='NAC',
            image_shape=[1024, 1024],
            offset=[46.0 + index / 10.0, 0.0],
        )
    return documents


def test_the_suspect_table_lists_every_suspect_by_default(tmp_path: Path) -> None:
    """``--top-n 0`` means uncapped here, which is the opposite of what it means elsewhere.

    An operator screening a run needs the whole list rather than the worst few
    of it, so this is the one always-on section that prints a row per image.  A
    quiet cap on it would be invisible wherever fewer images are suspect than
    the cap allows, so this is measured over more images than any bounded
    structure in the report holds.
    """
    text = _report_text(tmp_path, _many_suspect_documents(), top_n=0)
    assert len(_suspect_rows(text)) == _MANY_SUSPECTS


def test_top_n_caps_the_suspect_table(tmp_path: Path) -> None:
    """Given, the option caps the table it says it caps, worst first."""
    text = _report_text(tmp_path, _capped_documents(), top_n=_CAPPED_TOP_N)
    assert len(_suspect_rows(text)) == _CAPPED_TOP_N


def test_top_n_caps_the_names_a_drilldown_lists(tmp_path: Path) -> None:
    """The line promises up to N names per category, and printing more would break it.

    Six images are suspect and two may be named, so a drill-down that listed
    what it held would print a line its own heading contradicts.  Compared
    against the whole line rather than searched for inside it, since the capped
    line is a prefix of the uncapped one.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=_CAPPED_TOP_N)
    assert '- suspect / coiss: N1000000001, N1000000002' in text.splitlines()


def test_top_n_caps_the_worst_botsim_pairs(tmp_path: Path) -> None:
    """Three pairs are identified and two are listed, so the cap is what decides.

    The count in the heading is the number of rows under it, so a cap that did
    not bind would leave the heading naming two and three rows beneath it.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=_CAPPED_TOP_N)
    assert f'Worst {_CAPPED_TOP_N} pair(s):' in text


def test_the_worst_botsim_pair_is_the_one_with_the_largest_residual(tmp_path: Path) -> None:
    """A cap that kept the wrong two would still print two rows and say nothing true.

    The third pair's residual is sixty times the others', so the capped table
    that leaves it out is one an operator would act on the wrong frames from.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=_CAPPED_TOP_N)
    assert '| 1000000003 | N1000000003 | W1000000003 | -35.500 | 130.000 | 134.760 |' in text


def test_the_p95_residual_is_taken_over_the_whole_population(tmp_path: Path) -> None:
    """Three residuals, and the 95th percentile of them is the largest, not the smallest.

    Every percentile in either frozen report runs over a population of one, so
    a percentile that returned the first ordered value would agree with both of
    them.  Here the three are 1.500, 2.000 and 134.760.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=_CAPPED_TOP_N)
    assert '| p95 residual (px) | 134.760 |' in text


def test_the_u_axis_is_screened_as_well_as_the_v_axis(tmp_path: Path) -> None:
    """An offset pinned to the U edge of the search box is as suspect as one on V.

    The Cassini NAC limits are far apart -- 50 pixels on V against 140 on U --
    and the third NAC frame reaches 0.93 of the U limit while reaching 0.20 of
    the V limit, so a screen reading the V axis alone would pass it.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=0)
    assert '| N1000000003 | coiss | 10.000 | 130.000 | 130.384 | (50.0, 140.0) |' in text


def test_the_suspect_fraction_decides_what_is_screened_as_suspect(tmp_path: Path) -> None:
    """The option is what the ratio is compared against, not a number spelled twice.

    At the default fraction all six frames are suspect; at 0.945 only the two
    that reach 0.95 of a limit are.  A screen holding its own constant would
    print six rows either way.
    """
    text = _report_text(tmp_path, _capped_documents(), top_n=0, suspect_fraction=0.945)
    assert len(_suspect_rows(text)) == 2


# ---------------------------------------------------------------------------
# One camera at more than one image size
# ---------------------------------------------------------------------------

_POOLED_SIZES: dict[str, tuple[list[int], float]] = {
    'COISS_2001/N3000000001_1_CALIB': ([1024, 1024], 10.0),
    'COISS_2001/N3000000002_1_CALIB': ([1024, 1024], 20.0),
    'COISS_2001/N3000000003_1_CALIB': ([512, 512], 3.0),
}
"""One Cassini camera at two image sizes, and the V-axis offset each frame records.

Every real Cassini run holds several sizes of one camera -- the configured
search margins are keyed by 256, 512 and 1024 for the NAC -- so the per-camera
statistics are a pool of the per-size ones on every run there is.  The three
offsets are chosen so that no size on its own has the pooled mean, median,
minimum or maximum: over all three the mean is 11.000 and the median 10.000,
where the 1024 frames alone give 15.000 and the 512 frame alone gives 3.000.
"""


def _pooled_size_documents() -> dict[str, dict[str, Any]]:
    """Return the three Cassini NAC frames of :data:`_POOLED_SIZES`.

    Returns:
        Results path stub to document.
    """
    documents = {}
    for stub, (shape, offset_dv) in _POOLED_SIZES.items():
        name = stub.rsplit('/', 1)[-1]
        documents[stub] = metadata_document(
            image_name=f'{name}.IMG',
            camera='NAC',
            image_shape=shape,
            offset=[offset_dv, 0.0],
        )
    return documents


def test_the_per_camera_offsets_pool_the_image_sizes(tmp_path: Path) -> None:
    """The per-camera row is every offset that camera recorded, whatever size it was.

    The finer breakdown keys on the size as well, and the coarser one is that
    breakdown pooled rather than a second copy of the values.  A pool that lost
    a size would print one of the sizes' statistics under a heading that names
    the camera, and the two frozen reports could not tell: neither tree holds
    one camera at two sizes.
    """
    text = _report_text(tmp_path, _pooled_size_documents())
    assert '| coiss | NAC | dV | 3 (100.0%) | 11.000 | 10.000 |' in text
    assert '| coiss | NAC | 512x512 | 1 (33.3%) |' in text


# ---------------------------------------------------------------------------
# Two frames of one camera on one spacecraft-clock count
# ---------------------------------------------------------------------------

_COLLIDING_CALIB = 'VOL/N1454725799_1_CALIB'
"""The calibrated product of the colliding Cassini image."""

_COLLIDING_RAW = 'VOL/N1454725799_1'
"""The raw product of the same image, which holdings carry beside the calibrated one."""

_COLLIDING_WAC = 'VOL/W1454725799_1_CALIB'
"""The WAC frame shuttered with it, which makes the two a BOTSIM pair."""


def _colliding_documents() -> dict[str, dict[str, Any]]:
    """Return one WAC frame and both products of the NAC frame beside it.

    The version suffix and the ``_CALIB`` marker are both outside the clock
    count a BOTSIM pair is keyed on, so the raw and the calibrated product of
    one image are two frames of one camera on one clock count.  Their offsets
    are far apart on purpose: the calibrated one gives the pair a residual of
    30.000 and the raw one gives it 0.000, so which of them stands for the NAC
    is visible in the number the section prints.

    Returns:
        Results path stub to document.
    """
    offsets = {_COLLIDING_CALIB: 35.0, _COLLIDING_RAW: 5.0, _COLLIDING_WAC: 0.5}
    documents = {}
    for stub, offset_dv in offsets.items():
        name = stub.rsplit('/', 1)[-1]
        documents[stub] = metadata_document(
            image_name=f'{name}.IMG',
            camera='WAC' if name.startswith('W') else 'NAC',
            image_shape=[1024, 1024],
            offset=[offset_dv, 0.0],
        )
    return documents


def _facts_by_stub(tmp_path: Path, documents: dict[str, dict[str, Any]]) -> dict[str, ImageFacts]:
    """Read the facts of a tree of documents, keyed by the stub each came from.

    Parameters:
        tmp_path: Directory the tree lives under.
        documents: Results path stub to the document written at it.

    Returns:
        The facts, one per document.

    Raises:
        ValueError: If any document yielded no facts, which would make the
            replay below one image short without saying so.
    """
    root = _tree_of(tmp_path / 'results', documents)
    with TreeRecordSource([root.as_posix()]) as source:
        found = list(source.facts(Selection()))
    refused = [one.stub for one in found if isinstance(one, UnreadableFile)]
    if len(refused) > 0:
        raise ValueError(f'{root} holds documents that yielded no facts: {refused}')
    return {
        str(one.image['results_path_stub']): one for one in found if isinstance(one, ImageFacts)
    }


@pytest.mark.parametrize(
    'order',
    [
        pytest.param(
            (_COLLIDING_CALIB, _COLLIDING_RAW, _COLLIDING_WAC), id='calibrated-arrives-first'
        ),
        pytest.param(
            (_COLLIDING_RAW, _COLLIDING_CALIB, _COLLIDING_WAC), id='calibrated-arrives-last'
        ),
    ],
)
def test_the_frame_standing_for_a_camera_is_the_one_with_the_smallest_identity(
    order: Sequence[str], tmp_path: Path
) -> None:
    """Which of two colliding frames stands for a camera is a reduction, not an arrival.

    A source promises no order, so a section that kept whichever frame it met
    first -- or last -- would print one residual over a results tree and another
    over an index ingested from it.  The raw product has the smaller identity,
    so the residual is 0.000 whichever of the two the report meets first, and
    the calibrated one would give 30.000.

    Parameters:
        order: The order the three frames are handed to the report in.
        tmp_path: Directory the tree and the report live under.
    """
    facts = _facts_by_stub(tmp_path, _colliding_documents())
    out = tmp_path / 'report'
    out.mkdir(parents=True, exist_ok=True)
    with ReplayedFacts([facts[stub] for stub in order]) as source:
        build_report(source, out)
    assert '| median residual (px) | 0.000 |' in (out / 'report.md').read_text(encoding='utf-8')
