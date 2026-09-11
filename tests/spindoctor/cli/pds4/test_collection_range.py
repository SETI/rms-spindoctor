"""Tests for the time range the data collection label states, and where it is taken.

The summary pass reads every supplemental file once, in the global index generator, and
takes the range of the products' exposure epochs in that read.  The collection
generator, run after it, states the range in the data collection label, or, with no
range to state, does not write that label.  These run the two generators in that order,
over plumbing supplemental files and over the cohort, and hold the range's rounding
against the one a data label writes a product's own times with.
"""

import datetime
import re
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB

from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.collections import (
    GlobalIndexOutcome,
    generate_collection_files,
    generate_global_index_files,
)
from spindoctor.cli.pds4.epochs import EpochRange, NoEpochRange
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .conftest import (
    BundleEnv,
    make_bundle_env,
    make_cohort_bundle_env,
    make_image_file,
    read_tab,
    write_supplemental,
)


def _navigation(start_et: float, stop_et: float) -> dict[str, Any]:
    """Return a success navigation document recording an exposure between two epochs.

    Parameters:
        start_et: When the exposure began.
        stop_et: When it ended.

    Returns:
        The document, its midtime halfway between the two.
    """
    times = {'start_et': start_et, 'stop_et': stop_et, 'midtime_et': (start_et + stop_et) / 2}
    return {'status': 'success', 'navigation_result': {'times': times}}


def _summarize(env: BundleEnv) -> tuple[GlobalIndexOutcome, int]:
    """Run the two summary generators over the environment's bundle, in the pass's order.

    Parameters:
        env: The hermetic bundle environment to process.

    Returns:
        What the global index generation came to, and the number of collection labels
        not written by the collection generator, which is handed the index's range.
    """
    bundle_results_root = FCPath(env.bundle_results_root)
    dataset = env.dataset.as_dataset()
    index = generate_global_index_files(bundle_results_root, dataset, MAIN_LOGGER)
    failed = generate_collection_files(
        bundle_results_root, dataset, MAIN_LOGGER, epochs=index.epochs
    )
    return index, failed


def test_the_range_is_the_earliest_start_and_the_latest_stop_over_every_file(
    tmp_path: Path,
) -> None:
    """Over three supplemental files, the range is the least start and the greatest stop.

    The least start is in the second file read and the greatest stop in the first,
    so a range taken from any one file, or from the first file's start and the last
    file's stop, is reported.
    """
    env = make_bundle_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    write_supplemental(data_dir, 'shard0/1111111111n', navigation=_navigation(200.0, 900.0))
    write_supplemental(data_dir, 'shard0/2222222222w', navigation=_navigation(100.0, 300.0))
    write_supplemental(data_dir, 'shard0/3333333333n', navigation=_navigation(500.0, 600.0))
    index, _ = _summarize(env)
    assert index.epochs == EpochRange(start_et=100.0, stop_et=900.0)


def test_one_file_whose_epochs_cannot_be_had_leaves_no_range(tmp_path: Path) -> None:
    """A document recording no epochs beside a good one leaves no range, naming its file.

    A range taken over the others could leave that file's product outside it.  A file
    that cannot be read at all never reaches the range: the index refuses the run on it.
    """
    env = make_bundle_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    write_supplemental(data_dir, 'shard0/1111111111n', navigation=_navigation(100.0, 200.0))
    broken = write_supplemental(data_dir, 'shard0/2222222222w', navigation={})
    expected = NoEpochRange(
        f'supplemental file {FCPath(broken)} records no navigation_result block, so no '
        'range can be taken that contains every product'
    )
    index, _ = _summarize(env)
    assert index.epochs == expected


def test_with_no_supplemental_file_the_data_collection_label_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No products, no range: the label is counted as not written, and the inventory still is.

    A label stating empty dates is not one PDS4 accepts, and an earlier run's label
    at the path would state a range this run did not take, so it is removed.
    """
    env = make_bundle_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    data_dir.mkdir(parents=True)
    earlier = data_dir / 'collection_data.lblx'
    earlier.write_text('<an earlier run/>\n', encoding='utf-8')
    _, failed = _summarize(env)
    assert failed == 1
    assert not earlier.exists()
    assert read_tab(data_dir / 'collection_data.tab') == [['Member Status', 'LIDVID_LID']]
    assert 'there are no products to take a range from' in capsys.readouterr().out


def test_the_range_is_stated_at_the_whole_seconds_outside_it(tmp_path: Path) -> None:
    """A start at .6 of a second is stated at the second before, a stop at .35 at the one after.

    Rounded to the nearer second each would fall inside the product it bounds: the
    start at 04:25:36, after the product's own start of 04:25:35.600, and the stop at
    05:32:16, before its stop of 05:32:16.355.  The expected strings are SPICE's
    ``et2utc`` for the two epochs, ``2004-02-07T04:25:35.599999994`` and
    ``2004-02-22T05:32:16.354745835``, taken to the whole second outside each.
    """
    env = make_bundle_env(tmp_path)
    data_dir = env.bundle_dir / 'data'
    write_supplemental(
        data_dir, 'shard0/1234567890w', navigation=_navigation(129399999.78493077, 130700000.54)
    )
    _summarize(env)
    text = (data_dir / 'collection_data.lblx').read_text(encoding='utf-8')
    assert '<start>2004-02-07T04:25:35Z</start>' in text
    assert '<stop>2004-02-22T05:32:17Z</stop>' in text


def _instant(pds4_utc: str) -> datetime.datetime:
    """Read a time a label writes as the instant it names, so that two can be compared.

    Parameters:
        pds4_utc: The time in the PDS4 spelling, with or without decimals, and not in
            a leap second, which a datetime cannot hold.

    Returns:
        The instant, as a naive datetime in UTC.
    """
    return datetime.datetime.fromisoformat(pds4_utc.removesuffix('Z'))


def test_the_range_contains_a_data_label_s_times_where_they_meet_its_seconds(
    tmp_path: Path,
) -> None:
    """A product's written times lie inside the range even where they meet its seconds.

    A data label writes a product's start and stop at the nearest millisecond, and the
    range is written at the whole second at or before the least start and the one at
    or after the greatest stop.  The nearest millisecond of an epoch is never before
    the whole second at or before the epoch, nor after the one at or after it: a whole
    second is itself a millisecond, and rounding to the nearest carries no epoch past
    one.  So the range contains every product's written start and stop.  These epochs
    are where it is tightest: SPICE's ``et2utc`` writes them at nine decimals as
    ``2004-02-07T04:25:35.000400007`` and ``2004-02-22T05:32:16.999599993``, whose
    nearest milliseconds are the whole seconds the range starts and stops at.
    """
    start_et = 129399999.18533078
    stop_et = 130700001.18485416
    written = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings').pds4_template_variables(
        image_file=make_image_file('N1454725799_1'),
        nav_metadata=_navigation(start_et, stop_et),
        backplane_metadata={},
    )
    stated = EpochRange(start_et=start_et, stop_et=stop_et).template_variables()
    assert _instant(stated['EARLIEST_START_DATE_TIME']) <= _instant(written['START_DATE_TIME'])
    assert _instant(written['STOP_DATE_TIME']) <= _instant(stated['LATEST_STOP_DATE_TIME'])


def test_the_index_refuses_a_bundle_with_no_data_directory_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The index runs first in the summary pass, so it refuses a bundle no labels pass wrote.

    Written into, such a root would hold an index beside which the labels pass then
    refuses to write, since it writes only into an empty bundle directory.
    """
    env = make_bundle_env(tmp_path)
    refusal = r'Data directory does not exist: .*/fake_bundle/data'
    with pytest.raises(FileNotFoundError, match=refusal):
        _summarize(env)
    assert not env.bundle_dir.exists()


def test_the_cohort_s_data_collection_label_states_the_range_of_its_images(
    mini_nav_cohort: Cohort, tmp_path: Path
) -> None:
    """The shipped label over the cohort's two navigated images states their range.

    SPICE's ``et2utc`` writes the limb image's recorded start as
    ``2004-02-07T04:25:35.585069`` and the ring image's recorded stop, fifteen days
    later, as ``2004-02-22T05:32:16.354746``, and the range states each at the whole
    second outside it.  The two generators run in the order the summary pass runs
    them, the global index first.
    """
    env = make_cohort_bundle_env(mini_nav_cohort, tmp_path)
    for stub in (LIMB_STUB, RINGS_STUB):
        generate_bundle_data_files(
            env.dataset,
            mini_nav_cohort.batch(stub),
            nav_results_root=FCPath(mini_nav_cohort.nav_results_root),
            backplane_results_root=FCPath(mini_nav_cohort.backplane_results_root),
            bundle_results_root=FCPath(env.bundle_results_root),
            logger=MAIN_LOGGER,
        )
    index = generate_global_index_files(FCPath(env.bundle_results_root), env.dataset, MAIN_LOGGER)
    generate_collection_files(
        FCPath(env.bundle_results_root), env.dataset, MAIN_LOGGER, epochs=index.epochs
    )
    text = (env.bundle_dir / 'data' / 'collection_data.lblx').read_text(encoding='utf-8')
    assert re.findall(r'<start_date_time>(.*)</start_date_time>', text) == ['2004-02-07T04:25:35Z']
    assert re.findall(r'<stop_date_time>(.*)</stop_date_time>', text) == ['2004-02-22T05:32:17Z']
