"""The time range the Cassini ISS Saturn bundle's data collection label states.

Over the bundle's cohort and its shipped templates, the data collection label states
the range of the cohort's navigated images; and the range, written to the whole
seconds outside it, contains the start and stop a Cassini data label writes to the
nearest millisecond, even where they meet its seconds.  The range itself is tested
over stand-in templates in ``test_collection_range.py``.
"""

import datetime
import re
from pathlib import Path

import pytest
from filecache import FCPath
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CassiniISSSaturnCohort

from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.collections import generate_collection_files, generate_global_index_files
from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.config import MAIN_LOGGER
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .conftest import make_cohort_bundle_env, make_image_file


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CassiniISSSaturnCohort:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CassiniISSSaturnCohort)


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
    times = {'start_et': start_et, 'stop_et': stop_et, 'midtime_et': (start_et + stop_et) / 2}
    written = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings').pds4_template_variables(
        image_file=make_image_file('N1454725799_1'),
        nav_metadata={'status': 'success', 'navigation_result': {'times': times}},
        backplane_metadata={},
    )
    stated = EpochRange(start_et=start_et, stop_et=stop_et).template_variables()
    assert _instant(stated['EARLIEST_START_DATE_TIME']) <= _instant(written['START_DATE_TIME'])
    assert _instant(written['STOP_DATE_TIME']) <= _instant(stated['LATEST_STOP_DATE_TIME'])


def test_the_cohort_s_data_collection_label_states_the_range_of_its_images(
    cassini_cohort: Cohort, tmp_path: Path
) -> None:
    """The shipped label over the cohort's two navigated images states their range.

    SPICE's ``et2utc`` writes the limb image's recorded start as
    ``2004-02-07T04:25:35.585069`` and the ring image's recorded stop, fifteen days
    later, as ``2004-02-22T05:32:16.354746``, and the range states each at the whole
    second outside it.  The two generators run in the order the summary pass runs
    them, the global index first.
    """
    env = make_cohort_bundle_env(cassini_cohort, tmp_path)
    for stub in (LIMB_STUB, RINGS_STUB):
        generate_bundle_data_files(
            env.dataset,
            cassini_cohort.batch(stub),
            nav_results_root=FCPath(cassini_cohort.nav_results_root),
            backplane_results_root=FCPath(cassini_cohort.backplane_results_root),
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
