"""The time range the Cassini ISS Saturn bundle's data collection label states.

Over the bundle's cohort and its shipped templates, the data collection label states
the range of the cohort's navigated images, and the range contains the start and stop
a Cassini data label writes, even where they meet its whole seconds.  The range itself
is tested over stand-in templates in ``test_collection_range.py``.
"""

import datetime
import re
from pathlib import Path

import pytest
from tests.mini_nav_results.cohort import Cohort, WrittenCohorts
from tests.mini_nav_results.cohort_cassini import LIMB_STUB, RINGS_STUB, CohortCassiniISSSaturn

from spindoctor.cli.pds4.epochs import EpochRange
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISSSaturn

from .cohort_bundle import write_cohort_bundle
from .conftest import make_image_file


@pytest.fixture
def cassini_cohort(mini_nav_cohorts: WrittenCohorts) -> CohortCassiniISSSaturn:
    """Return the Cassini ISS Saturn cohort, as the session wrote it.

    Parameters:
        mini_nav_cohorts: What the session's cohorts are written by.

    Returns:
        The written cohort.
    """
    return mini_nav_cohorts(CohortCassiniISSSaturn)


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

    These epochs are where it is tightest: SPICE's ``et2utc`` writes them at nine
    decimals as ``2004-02-07T04:25:35.000400007`` and ``2004-02-22T05:32:16.999599993``,
    whose nearest milliseconds, the times a data label writes, are the whole seconds the
    range starts and stops at.
    """
    start_et = 129399999.18533078
    stop_et = 130700001.18485416
    observation = {'start_time_et': start_et, 'end_time_et': stop_et}
    written = DataSetPDS3CassiniISSSaturn(tmp_path / 'holdings').pds4_template_variables(
        image_file=make_image_file('N1454725799_1'),
        nav_metadata={'status': 'success', 'observation': observation},
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
    env = write_cohort_bundle(cassini_cohort, tmp_path, (LIMB_STUB, RINGS_STUB))
    text = (env.bundle_dir / 'data' / 'collection_data.lblx').read_text(encoding='utf-8')
    assert re.findall(r'<start_date_time>(.*)</start_date_time>', text) == ['2004-02-07T04:25:35Z']
    assert re.findall(r'<stop_date_time>(.*)</stop_date_time>', text) == ['2004-02-22T05:32:17Z']
