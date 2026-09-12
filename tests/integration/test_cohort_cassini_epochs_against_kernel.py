"""The Cassini cohort's epochs as a PDS4 label writes them, held against the kernel.

The unit tests of :func:`~spindoctor.support.time.et_to_pds4_utc` compare it with
strings SPICE wrote once, for epochs chosen when they were written.  This converts
every epoch the cohort's navigation documents record again, through ``et2utc`` and the
leapseconds kernel a navigation run of those images would have loaded, so a
leap-second table or a model of TDB that moved under the conversion is reported
against the very epochs the bundle tests assert labels on.

The kernel is the arbiter of all three roundings.  The nearer millisecond is
``et2utc``'s own three-decimal answer.  The floor is its nine-decimal answer cut to
three, and the ceiling is the same cut taken a millisecond later, which is the next
millisecond for an epoch not exactly on one and the epoch's own millisecond for an
epoch that is.
"""

from collections.abc import Iterator

import pytest

from tests.oops_resources import retrieved_kernel

pytestmark = pytest.mark.integration

_LSK = retrieved_kernel(
    'SPICE/General/LSK/naif0012.tls',
    what='the leapseconds kernel',
    tests='the Cassini cohort epoch kernel tests',
)

import cspyce  # noqa: E402  (guarded import)

from spindoctor.support.time import Pds4Rounding, et_to_pds4_utc  # noqa: E402  (guarded import)
from tests.kernel_pool import isolated_kernel_pool  # noqa: E402  (guarded import)
from tests.mini_nav_results.cohort_cassini import (  # noqa: E402  (guarded import)
    CohortCassiniISSSaturn,
)

_EPOCH_KEYS = ('start_et', 'midtime_et', 'stop_et')
"""The epochs a navigation document records for an exposure."""

_MILLISECOND = 0.001
"""One step of the last digit a product's time is written to, in seconds."""

_ISO_TO_MILLISECONDS = len('YYYY-MM-DDThh:mm:ss.sss')
"""How much of an ISO calendar string runs to its third decimal."""


@pytest.fixture
def leapseconds() -> Iterator[None]:
    """Furnish the leapseconds kernel, and nothing else.

    Yields:
        Nothing; the kernel is furnished for the body of the test.
    """
    with isolated_kernel_pool():
        cspyce.furnsh(str(_LSK))
        yield


def _from_the_kernel(et: float, rounding: Pds4Rounding) -> str:
    """Return what the kernel says an epoch rounds to, in the PDS4 spelling.

    Parameters:
        et: The epoch.
        rounding: Which way it is rounded.

    Returns:
        The millisecond ``et2utc`` puts on that side of the epoch, with the Z.
    """
    if rounding == 'nearest':
        return f'{cspyce.et2utc(et, "ISOC", 3)}Z'
    nine = str(cspyce.et2utc(et, 'ISOC', 9))
    on_a_millisecond = set(nine[_ISO_TO_MILLISECONDS:]) == {'0'}
    if rounding == 'down' or on_a_millisecond:
        return f'{nine[:_ISO_TO_MILLISECONDS]}Z'
    later = str(cspyce.et2utc(et + _MILLISECOND, 'ISOC', 9))
    return f'{later[:_ISO_TO_MILLISECONDS]}Z'


@pytest.mark.parametrize('rounding', ['nearest', 'down', 'up'])
def test_every_cohort_epoch_is_written_as_the_kernel_rounds_it(
    leapseconds: None, rounding: Pds4Rounding
) -> None:
    """Every epoch of every cohort document, each way it can be rounded.

    Parameters:
        leapseconds: Fixture furnishing the leapseconds kernel.
        rounding: Which way the epochs are rounded.
    """
    disagreeing: list[str] = []
    for image in CohortCassiniISSSaturn.images():
        times = image.document['navigation_result']['times']
        for key in _EPOCH_KEYS:
            et = float(times[key])
            written = et_to_pds4_utc(et, rounding=rounding)
            expected = _from_the_kernel(et, rounding)
            if written != expected:
                disagreeing.append(
                    f'{image.image_name} {key} {et!r}: written {written}, the kernel gives '
                    f'{expected}'
                )
    assert disagreeing == []
