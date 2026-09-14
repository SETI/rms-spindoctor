"""Spec-first tests for PDS4 collection inventories and their labels (phase 2).

Contract under test (docs/user_guide/user_guide_pds4_bundle.rst "Summary Pass" /
"Summary Pass Outputs" and docs/dev_guide/dev_guide_pds4.rst "Pipeline overview"):
``generate_collection_files`` scans the bundle's ``data/`` tree for
``*_backplanes.lblx`` labels and its ``browse/`` tree for ``*_summary.lblx``
labels, sorts each by product name, and writes from them the
``collection_data.csv`` / ``collection_browse.csv`` inventories (no header, one
``P,<lidvid>`` line per product ending in a line feed alone, LIDVIDs from the
dataset's ``pds4_image_name_to_*_lidvid`` builders) plus the matching
``.lblx`` labels.  A collection with no label of its kind on disk gets neither
file and counts as a label not written.  It also holds each image's products
against each other: a data label with no browse label, or a browse label or
supplemental file with no data label, is an image whose products disagree,
logged by name and counted.  Every template the generator renders is required:
the driver checks the ones its dataset declares before processing anything, so
one that is missing raises here rather than being passed over.

The global index files, and the LIDs the two generators build against each
other, are tested in ``test_global_index.py``.
"""

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4 import collections as collections_module

from .conftest import (
    BROKEN_TEMPLATE,
    COLLECTION_BROWSE_TEMPLATE,
    COLLECTION_DATA_TEMPLATE,
    make_bundle_env,
    read_csv_rows,
    run_collections,
    touch_browse_label,
    touch_label,
    write_supplemental,
)

COLLECTION_LABELS = {
    'collection_data.lblx': ('data', COLLECTION_DATA_TEMPLATE),
    'collection_browse.lblx': ('browse', COLLECTION_BROWSE_TEMPLATE),
}
"""Each collection label's bundle subdirectory and its intact template body."""
COLLECTION_PRODUCTS = (
    'data/collection_data.csv',
    'data/collection_data.lblx',
    'browse/collection_browse.csv',
    'browse/collection_browse.lblx',
)
"""Every file the collection generator writes, relative to the bundle's directory."""


# ---------------------------------------------------------------------------
# generate_collection_files: inventories
# ---------------------------------------------------------------------------


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    """A bundle without a data directory is rejected with FileNotFoundError."""
    env = make_bundle_env(tmp_path)
    with pytest.raises(FileNotFoundError, match='Data directory does not exist'):
        run_collections(env)


def test_an_empty_data_tree_writes_neither_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no data label and no supplemental file, neither collection is written.

    A collection label states at least one record, so a collection with no member
    gets no label, and so no inventory either.  The data collection has no range to
    state as well, and still counts once.  The generator removes whatever an earlier
    run left at the four paths itself, and not only through the index generator that
    clears them first in the summary pass, so that a collection on disk is always one
    this run wrote.  The data collection's error gives both of its reasons.
    """
    env = make_bundle_env(tmp_path)
    for product in COLLECTION_PRODUCTS:
        earlier = env.bundle_dir / product
        earlier.parent.mkdir(parents=True, exist_ok=True)
        earlier.write_text('an earlier run\n', encoding='utf-8')
    failed = run_collections(env, epochs=None).failed_labels
    assert failed == 2
    assert [name for name in COLLECTION_PRODUCTS if (env.bundle_dir / name).exists()] == []
    out = capsys.readouterr().out
    data_errors = [
        line for line in out.splitlines() if 'The data collection was not written' in line
    ]
    assert len(data_errors) == 1
    assert 'the collection has no member' in data_errors[0]
    assert 'the data tree holds no supplemental file' in data_errors[0]


def test_a_collection_with_no_member_is_not_written_beside_one_that_has(tmp_path: Path) -> None:
    """Each collection is judged by its own members: no browse label, no browse collection.

    The data collection has a member and a range, so it is written; the browse
    collection has no member of its own, so neither its inventory nor its label is,
    and it counts as one label not written.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    failed = run_collections(env).failed_labels
    assert failed == 1
    assert (env.bundle_dir / 'data' / 'collection_data.lblx').is_file()
    browse_products = ['collection_browse.csv', 'collection_browse.lblx']
    assert [name for name in browse_products if (env.bundle_dir / 'browse' / name).exists()] == []


def test_data_inventory_row_per_label_with_primary_status(tmp_path: Path) -> None:
    """One *_backplanes.lblx yields one P row, its data LIDVID, and no header row."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    assert rows == [['P', 'urn:nasa:pds:fake_bundle:data:1234567890w::1.0']]


def test_the_browse_inventory_lists_the_browse_labels_on_disk(tmp_path: Path) -> None:
    """The browse inventory lists the browse labels in the browse tree, by browse LIDVID.

    It is built from the browse labels rather than the data labels, so an image with a
    data label and no browse label is not in it, and one with a browse label and no
    data label is.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_label(env.bundle_dir / 'data', 'shard0/2222222222w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/3333333333n')
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'browse' / 'collection_browse.csv')
    assert rows == [
        ['P', 'urn:nasa:pds:fake_bundle:browse:1111111111n::1.0'],
        ['P', 'urn:nasa:pds:fake_bundle:browse:3333333333n::1.0'],
    ]


def test_inventory_rows_sorted_by_product_name_not_path(tmp_path: Path) -> None:
    """Inventory rows sort by product name, the last part of each LID, not by path."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'zz9/1111111111n')
    touch_label(env.bundle_dir / 'data', 'aa0/2222222222w')
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    assert rows[0][1] == 'urn:nasa:pds:fake_bundle:data:1111111111n::1.0'
    assert rows[1][1] == 'urn:nasa:pds:fake_bundle:data:2222222222w::1.0'


def test_duplicate_image_names_produce_duplicate_rows(tmp_path: Path) -> None:
    """Characterization: the same image name in two shards is listed twice."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_label(env.bundle_dir / 'data', 'shard1/1234567890w')
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    assert len(rows) == 2
    assert rows[0] == rows[1]


def test_non_backplane_label_files_ignored(tmp_path: Path) -> None:
    """Only *_backplanes.lblx files are inventoried from the data tree."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    other = env.bundle_dir / 'data' / 'shard0' / '1111111111n_other.lblx'
    other.write_text('<x/>\n', encoding='utf-8')
    run_collections(env)
    rows = read_csv_rows(env.bundle_dir / 'data' / 'collection_data.csv')
    assert rows == [['P', 'urn:nasa:pds:fake_bundle:data:1234567890w::1.0']]


def test_every_inventory_line_ends_in_a_line_feed_alone(tmp_path: Path) -> None:
    """Every line of each inventory, the last included, ends in a line feed and no CR.

    The collection labels declare their records delimited by a line feed, and a
    delimited table's last record is delimited like every other.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_label(env.bundle_dir / 'data', 'shard0/2222222222w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/2222222222w')
    run_collections(env)
    raws = [
        (env.bundle_dir / 'data' / 'collection_data.csv').read_bytes(),
        (env.bundle_dir / 'browse' / 'collection_browse.csv').read_bytes(),
    ]
    endings = [[line[-1:] for line in raw.splitlines(keepends=True)] for raw in raws]
    assert endings == [[b'\n', b'\n'], [b'\n', b'\n']]
    assert [raw for raw in raws if b'\r' in raw] == []


# ---------------------------------------------------------------------------
# generate_collection_files: each image's products agree
# ---------------------------------------------------------------------------


def test_a_data_label_with_no_browse_label_is_an_image_whose_products_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image with a data label and no browse label is logged by name and counted once.

    Every data product has a browse product.  The other image has both labels, and is
    not counted.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    touch_label(env.bundle_dir / 'data', 'shard0/2222222222w')
    outcome = run_collections(env)
    assert outcome.disagreeing_images == 1
    missing = FCPath(env.bundle_dir) / 'browse' / 'shard0' / '2222222222w_summary.lblx'
    out = capsys.readouterr().out
    assert 'The products of image 2222222222w disagree' in out
    assert f'and no browse label at {missing}' in out


def test_a_browse_label_with_no_data_label_is_an_image_whose_products_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image with a browse label and no data label is logged by name and counted once."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/3333333333n')
    outcome = run_collections(env)
    assert outcome.disagreeing_images == 1
    missing = FCPath(env.bundle_dir) / 'data' / 'shard0' / '3333333333n_backplanes.lblx'
    out = capsys.readouterr().out
    assert 'The products of image 3333333333n disagree' in out
    assert f'and no data label at {missing}' in out


def test_a_supplemental_file_with_no_data_label_is_an_image_whose_products_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image with a supplemental file and no data label is logged and counted once.

    The labels pass writes an image's supplemental file before its data label, so this
    is what a data label that failed to render leaves.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    write_supplemental(env.bundle_dir / 'data', 'shard0/4444444444n')
    outcome = run_collections(env)
    assert outcome.disagreeing_images == 1
    missing = FCPath(env.bundle_dir) / 'data' / 'shard0' / '4444444444n_backplanes.lblx'
    out = capsys.readouterr().out
    assert 'The products of image 4444444444n disagree' in out
    assert f'and no data label at {missing}' in out


def test_an_image_holding_a_browse_label_and_a_supplemental_file_counts_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image with a browse label and a supplemental file and no data label counts once.

    That is what a data label that failed to render, or was removed, leaves.  Its one
    error names both files the image has.
    """
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/1111111111n')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1111111111n')
    browse_label = touch_browse_label(env.bundle_dir / 'browse', 'shard0/5555555555n')
    supplemental = write_supplemental(env.bundle_dir / 'data', 'shard0/5555555555n')
    outcome = run_collections(env)
    assert outcome.disagreeing_images == 1
    errors = [
        line
        for line in capsys.readouterr().out.splitlines()
        if 'The products of image 5555555555n disagree' in line
    ]
    assert len(errors) == 1
    assert str(browse_label) in errors[0]
    assert str(supplemental) in errors[0]


def test_two_images_whose_products_disagree_count_two(tmp_path: Path) -> None:
    """Each image whose products disagree is counted, not only whether any does."""
    env = make_bundle_env(tmp_path)
    touch_label(env.bundle_dir / 'data', 'shard0/2222222222w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/3333333333n')
    assert run_collections(env).disagreeing_images == 2


# ---------------------------------------------------------------------------
# generate_collection_files: labels
# ---------------------------------------------------------------------------


def test_collection_labels_rendered_when_templates_exist(tmp_path: Path) -> None:
    """The collection labels carry the CSV path, the data label the range handed it."""
    env = make_bundle_env(
        tmp_path,
        template_contents={
            'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
            'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
        },
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1234567890w')
    failed = run_collections(env).failed_labels
    assert failed == 0
    data_label = env.bundle_dir / 'data' / 'collection_data.lblx'
    text = data_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'data' / 'collection_data.csv') in text
    assert '<start>2004-02-07T04:25:35Z</start>' in text
    assert '<stop>2004-02-22T05:32:17Z</stop>' in text
    browse_label = env.bundle_dir / 'browse' / 'collection_browse.lblx'
    browse_text = browse_label.read_text(encoding='utf-8')
    assert str(FCPath(env.bundle_dir) / 'browse' / 'collection_browse.csv') in browse_text


@pytest.mark.parametrize(
    ('label', 'subdir'),
    [('collection_data.lblx', 'data'), ('collection_browse.lblx', 'browse')],
    ids=['data collection', 'browse collection'],
)
def test_a_collection_label_is_written_to_the_bundle_not_to_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, subdir: str
) -> None:
    """Each collection label writer is handed the bundle's own path for the label.

    The location an ``FCPath`` names and the location of its local cache copy
    are the same file only while the bundle root is local.  On a cloud bundle
    root they are two different files, so a caller that hands over the cache
    path writes the label into the cache rather than into the bundle, and names
    in its error report a path that is nowhere in the bundle.  A local bundle
    root cannot tell those apart by the file that appears, so what the writer is
    handed is what says which one it was.  Each label is a case of its own
    because each is handed over by a statement of its own.

    Parameters:
        tmp_path: Base temporary directory.
        monkeypatch: Fixture the recording label writer is installed through.
        label: The collection label this case checks.
        subdir: The bundle subdirectory that label belongs in.
    """
    handed_over: dict[str, Any] = {}

    def _record(
        template: Any, template_vars: dict[str, Any], label_path: Any, *, logger: Any
    ) -> bool:
        """Record the path handed over and report the label as written.

        Parameters:
            template: The parsed template, unused.
            template_vars: The template variables, unused.
            label_path: The path whose type and value are under test.
            logger: The logger the caller passed, unused.

        Returns:
            True, so the caller counts the label as written.
        """
        handed_over[FCPath(label_path).name] = label_path
        return True

    env = make_bundle_env(
        tmp_path,
        template_contents={
            'collection_data.lblx': COLLECTION_DATA_TEMPLATE,
            'collection_browse.lblx': COLLECTION_BROWSE_TEMPLATE,
        },
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1234567890w')
    monkeypatch.setattr(collections_module, 'write_label', _record)
    run_collections(env)
    assert isinstance(handed_over[label], FCPath)
    assert handed_over[label] == FCPath(env.bundle_dir) / subdir / label


@pytest.mark.parametrize(
    ('broken', 'intact'),
    [
        ('collection_data.lblx', 'collection_browse.lblx'),
        ('collection_browse.lblx', 'collection_data.lblx'),
    ],
    ids=['data collection', 'browse collection'],
)
def test_a_broken_collection_template_is_counted_and_leaves_the_other(
    tmp_path: Path, broken: str, intact: str
) -> None:
    """One unrenderable collection label is counted; the other still gets written.

    Failing on the first would hide the second, so a run reports every broken
    collection template rather than one per run.  The broken label's inventory,
    written before the label is rendered, stays.  Each label is a case of its own
    because each is counted by a statement of its own.

    Parameters:
        tmp_path: Base temporary directory.
        broken: Template whose render errors in this case.
        intact: Template that still renders in this case.
    """
    broken_dir, _ = COLLECTION_LABELS[broken]
    intact_dir, intact_body = COLLECTION_LABELS[intact]
    env = make_bundle_env(
        tmp_path, template_contents={broken: BROKEN_TEMPLATE, intact: intact_body}
    )
    touch_label(env.bundle_dir / 'data', 'shard0/1234567890w')
    touch_browse_label(env.bundle_dir / 'browse', 'shard0/1234567890w')
    failed = run_collections(env).failed_labels
    assert failed == 1
    assert not (env.bundle_dir / broken_dir / broken).exists()
    assert (env.bundle_dir / broken_dir / broken.replace('.lblx', '.csv')).is_file()
    assert (env.bundle_dir / intact_dir / intact).is_file()


@pytest.mark.parametrize(
    'template', ['collection_data.lblx', 'collection_browse.lblx'], ids=['data', 'browse']
)
def test_a_missing_collection_template_raises_over_an_empty_tree(
    tmp_path: Path, template: str
) -> None:
    """A collection template the dataset declares and does not have ends the run.

    The driver checks every declared template before it processes anything, so
    one that is missing this far in is a template tree that does not carry what
    its dataset says it does.  Passing over it would leave the bundle with an
    inventory no label describes, and nothing saying so.  It raises even over a
    data tree holding no label, where neither collection can be written.  Each
    template is a case of its own because each is parsed by a statement of its own.

    Parameters:
        tmp_path: Base temporary directory.
        template: The collection template missing in this case.
    """
    env = make_bundle_env(tmp_path)
    (Path(env.dataset.pds4_bundle_template_dir()) / template).unlink()
    (env.bundle_dir / 'data').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=template.replace('.', r'\.')):
        run_collections(env)
