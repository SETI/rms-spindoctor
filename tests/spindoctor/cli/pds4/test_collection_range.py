"""Tests for the time range the data collection label states, and where it is taken.

The summary pass reads every supplemental file once, in the global index generator, and
takes the range of the products' exposure epochs in that read.  The collection
generator, run after it, states the range in the data collection label, or, with no
range to state, does not write that label.  These run the two generators in that order,
over plumbing supplemental files.  What the range comes to over a bundle's shipped
templates is tested in a module named for that bundle.
"""

from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.collections import (
    GlobalIndexOutcome,
    generate_collection_files,
    generate_global_index_files,
)
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.config import MAIN_LOGGER

from .conftest import (
    BundleEnv,
    make_bundle_env,
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


def test_with_no_supplemental_file_the_data_collection_label_is_not_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No products, no range: the label counts as not written; the inventory is written.

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
    expected = 'the data tree holds no supplemental file, so there is no time range'
    assert expected in capsys.readouterr().out


def test_the_range_is_stated_at_the_whole_seconds_outside_it(tmp_path: Path) -> None:
    """A start at .6 s is stated at the second before, a stop at .35 s at the one after.

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


def test_the_index_refuses_a_bundle_with_no_data_directory_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Running first in the summary pass, the index refuses a bundle no labels pass wrote.

    Written into, such a root would hold an index beside which the labels pass then
    refuses to write, since it writes only into an empty bundle directory.
    """
    env = make_bundle_env(tmp_path)
    refusal = r'Data directory does not exist: .*/fake_bundle/data'
    with pytest.raises(FileNotFoundError, match=refusal):
        _summarize(env)
    assert not env.bundle_dir.exists()
