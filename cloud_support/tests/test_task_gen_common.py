"""Unit tests for the shared machinery behind the cloud-task generators.

Two things here decide something: how volumes are divided into groups, and how
a URL the enumeration wrote for this machine is re-rooted for the workers.  The
rest of the module runs sd_offset and copies what it wrote, and is exercised by
running it.

Run these with ``pytest cloud_support/tests``; the project's default test paths
cover the package alone.
"""

import argparse
import sys
from pathlib import Path

import pytest
from filecache import FCPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import task_gen_common as common

# The Cassini archive's shape: 125 volumes of wildly uneven size, a few of them
# tiny, summing to about 443,000 images.  Sizes are what makes grouping hard, so
# the pattern here spans the real range rather than a uniform one.
UNEVEN_VOLUMES = [
    (f'COISS_{2000 + index:04d}', size)
    for index, size in enumerate([4285, 40, 5886, 129, 3745, 1021, 4705, 625, 3296, 890], start=1)
]


def test_a_volume_is_never_split_and_never_lost() -> None:
    """Every volume lands in exactly one group, in archive order."""
    groups = common.group_volumes(UNEVEN_VOLUMES, 5000)

    grouped = [volume for volumes, _ in groups for volume in volumes]
    assert grouped == [volume for volume, _ in UNEVEN_VOLUMES]


def test_each_group_reports_the_count_of_the_volumes_in_it() -> None:
    """A group's reported count is the sum of its volumes."""
    groups = common.group_volumes(UNEVEN_VOLUMES, 5000)

    sizes = dict(UNEVEN_VOLUMES)
    assert [count for _, count in groups] == [
        sum(sizes[volume] for volume in volumes) for volumes, _ in groups
    ]


def test_the_groups_come_out_near_the_target_rather_than_near_each_extreme() -> None:
    """The images spread evenly rather than leaving the remainder in the last group."""
    # 443,177 images at a target of 50,000 is the whole Cassini archive. Filling
    # each group to the target in turn leaves the remainder in the last one; the
    # even spread is what keeps the last group from being half or double the rest.
    volumes = [(f'COISS_{2000 + index:04d}', 3545) for index in range(125)]

    groups = common.group_volumes(volumes, 50000)

    assert len(groups) == 9
    # No two groups differ by more than the one volume that separates them.
    assert max(count for _, count in groups) - min(count for _, count in groups) <= 3545


def test_a_volume_that_would_overshoot_is_held_for_the_next_group() -> None:
    """A group closes short of the target when that lands closer than going over."""
    # Closing at 900 is nearer 1000 than closing at 1800 would be, so the third
    # volume starts the next group instead of finishing this one.
    volumes = [('A', 500), ('B', 400), ('C', 900), ('D', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A', 'B'], ['C'], ['D']]


def test_a_remainder_too_small_for_a_queue_joins_the_group_before_it() -> None:
    """A tail under a fifth of a group is folded into its predecessor."""
    volumes = [('A', 1000), ('B', 1000), ('C', 30)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B', 'C']]


def test_a_remainder_worth_a_queue_stands_on_its_own() -> None:
    """A tail large enough to be worth starting instances for stays its own group."""
    volumes = [('A', 1000), ('B', 1000), ('C', 600)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B'], ['C']]


def test_a_volume_holding_no_images_is_passed_over() -> None:
    """A volume the selection matched nothing in joins no group."""
    volumes = [('A', 1000), ('EMPTY', 0), ('B', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B']]


def test_a_volume_larger_than_a_whole_group_is_a_group() -> None:
    """A volume bigger than the target is a group by itself, never split."""
    volumes = [('HUGE', 5000), ('A', 1000), ('B', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['HUGE'], ['A'], ['B']]


def test_no_volumes_makes_no_groups() -> None:
    """An empty selection produces no groups rather than one empty one."""
    assert common.group_volumes([], 1000) == []


def test_a_target_of_zero_is_refused() -> None:
    """A target that could never close a group is refused."""
    with pytest.raises(ValueError) as exc:
        common.group_volumes([('A', 10)], 0)
    assert 'must be positive' in str(exc.value)


def a_task(*urls: str) -> common.Task:
    """A task holding one file, with the given image and label URLs.

    Parameters:
        urls: The image URL and the label URL, in that order.

    Returns:
        The task.
    """
    image_url, label_url = urls
    return {
        'task_id': 'a-task-0',
        'data': {
            'dataset_name': 'coiss',
            'files': [{'image_file_url': image_url, 'label_file_url': label_url}],
        },
    }


LOCAL_IMAGE = '/local/pds/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N1_CALIB.IMG'
LOCAL_LABEL = '/local/pds/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N1_CALIB.LBL'


def test_both_urls_of_a_file_are_re_rooted() -> None:
    """The image and the label URL both move to the holdings the workers read."""
    tasks = [a_task(LOCAL_IMAGE, LOCAL_LABEL)]

    common.retarget_urls(
        tasks, volumes_dir='calibrated', holdings_root='gs://bucket/holdings', path=FCPath('t.json')
    )

    assert tasks[0]['data']['files'][0]['image_file_url'] == (
        'gs://bucket/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N1_CALIB.IMG'
    )
    assert tasks[0]['data']['files'][0]['label_file_url'] == (
        'gs://bucket/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N1_CALIB.LBL'
    )


def test_the_root_that_was_replaced_is_reported() -> None:
    """The local root that was exchanged is handed back to be reported."""
    tasks = [a_task(LOCAL_IMAGE, LOCAL_LABEL)]

    local_root = common.retarget_urls(
        tasks, volumes_dir='calibrated', holdings_root='gs://bucket/holdings', path=FCPath('t.json')
    )

    assert local_root == '/local/pds/holdings'


def test_a_trailing_slash_on_the_target_does_not_double() -> None:
    """A holdings root written with a trailing separator yields no doubled slash."""
    tasks = [a_task(LOCAL_IMAGE, LOCAL_LABEL)]

    common.retarget_urls(
        tasks,
        volumes_dir='calibrated',
        holdings_root='gs://bucket/holdings/',
        path=FCPath('t.json'),
    )

    assert '//calibrated/' not in tasks[0]['data']['files'][0]['image_file_url']


def test_a_local_root_spelled_like_the_volumes_directory_splits_at_the_last_one() -> None:
    """The seam is the last volumes directory, since only a root can repeat the word."""
    # Nothing below a holdings root repeats the volumes directory, so the last
    # occurrence is the seam even when the root itself contains the word.
    image = '/data/volumes/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C1R.IMG'
    label = '/data/volumes/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C1R.LBL'
    tasks = [a_task(image, label)]

    local_root = common.retarget_urls(
        tasks, volumes_dir='volumes', holdings_root='gs://bucket/h', path=FCPath('t.json')
    )

    assert local_root == '/data/volumes/holdings'
    assert tasks[0]['data']['files'][0]['image_file_url'] == (
        'gs://bucket/h/volumes/GO_0xxx/GO_0002/RAW_CAL/C1R.IMG'
    )


def test_a_url_under_no_volumes_directory_is_refused() -> None:
    """A URL whose root cannot be told from the rest of it stops the run."""
    tasks = [a_task('/somewhere/else/N1_CALIB.IMG', '/somewhere/else/N1_CALIB.LBL')]

    with pytest.raises(SystemExit) as exc:
        common.retarget_urls(
            tasks, volumes_dir='calibrated', holdings_root='gs://bucket/h', path=FCPath('t.json')
        )
    assert 'lies under no "calibrated" directory' in str(exc.value)


def test_two_local_roots_in_one_file_are_refused() -> None:
    """One root cannot replace two, so a file holding both stops the run."""
    tasks = [
        a_task(LOCAL_IMAGE, LOCAL_LABEL),
        a_task(
            '/other/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N2_CALIB.IMG',
            '/other/holdings/calibrated/COISS_2xxx/COISS_2001/data/a/N2_CALIB.LBL',
        ),
    ]

    with pytest.raises(SystemExit) as exc:
        common.retarget_urls(
            tasks, volumes_dir='calibrated', holdings_root='gs://bucket/h', path=FCPath('t.json')
        )
    assert 'more than one holdings root' in str(exc.value)


def test_no_tasks_re_roots_nothing() -> None:
    """A file with no tasks reports no root rather than inventing one."""
    assert (
        common.retarget_urls(
            [], volumes_dir='calibrated', holdings_root='gs://bucket/h', path=FCPath('t.json')
        )
        is None
    )


def test_the_volumes_directory_comes_from_the_dataset() -> None:
    """Cassini is navigated from its calibrated products."""
    assert common.volumes_dir_name('coiss') == 'calibrated'


def test_an_instrument_read_from_the_volumes_tree_says_so() -> None:
    """The other instruments are read from the volumes tree."""
    assert common.volumes_dir_name('gossi') == 'volumes'


def test_a_holdings_root_loses_a_trailing_separator() -> None:
    """A trailing separator is dropped so it cannot double in a URL."""
    assert common.holdings_root_argument('gs://bucket/holdings/') == 'gs://bucket/holdings'


def test_a_blank_holdings_root_is_refused() -> None:
    """A root that names nowhere is refused at the command line."""
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        common.holdings_root_argument('   ')
    assert 'names no holdings' in str(exc.value)


def test_re_rooting_under_a_blank_root_is_refused() -> None:
    """A caller reaching the function directly is refused the same way."""
    # argparse turns this away at the command line; a caller reaching the
    # function directly would otherwise write URLs with no root at all.
    tasks = [a_task(LOCAL_IMAGE, LOCAL_LABEL)]

    with pytest.raises(SystemExit) as exc:
        common.retarget_urls(
            tasks, volumes_dir='calibrated', holdings_root='  ', path=FCPath('t.json')
        )
    assert 'names nowhere' in str(exc.value)


def test_a_remote_output_directory_is_left_for_the_write_to_make(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud prefix is made by the write, so nothing is created for it here."""
    # Run somewhere empty: a remote path turned into a local one lands under the
    # working directory as "gs:/bucket/tasks", which is the regression to catch.
    monkeypatch.chdir(tmp_path)

    common.make_directory(FCPath('gs://bucket/tasks'))

    assert list(tmp_path.iterdir()) == []
