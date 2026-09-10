"""The bundle cohort's spacecraft clock, held against the kernel that defines it.

The cohort converts an epoch to a Cassini clock reading with a line through two
correlation points read out of the mission clock kernel, because a fixture that
called SPICE would not run where the fixture has to run.  A line is not what a
clock kernel is, so how far the two part is a measured quantity rather than an
assumed one, and this is where it is measured.

Nothing in the default suite can measure it.  The cohort's own self-tests
compare a reading to another reading from the same function and an image name
to the reading it was derived from, so they hold the fixture to itself: they
catch a hand-authored triple added later, which is their job, and an anchor
moved by an hour would leave every one of them green.  This is the test that
would not be.

The kernels are the ones a navigation run of these images would have loaded.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')
_LSK = Path(_RESOURCES) / 'SPICE' / 'General' / 'LSK' / 'naif0012.tls'
_SCLK = Path(_RESOURCES) / 'SPICE' / 'Cassini' / 'SCLK' / 'cas00172.tsc'

if len(_RESOURCES) == 0 or not _LSK.is_file() or not _SCLK.is_file():
    pytest.skip(
        'OOPS_RESOURCES does not name a local SPICE tree holding the leapseconds and '
        'Cassini clock kernels; skipping the cohort clock kernel tests',
        allow_module_level=True,
    )

import cspyce  # noqa: E402  (guarded import)

from tests.kernel_pool import isolated_kernel_pool  # noqa: E402  (guarded import)
from tests.mini_nav_results.cohort_cassini import cohort_images  # noqa: E402  (guarded import)

_CASSINI_SCLK_ID = -82
"""The clock the kernel defines, and the one the readings are on."""

_TICKS_PER_SECOND = 256
"""The modulus of the reading's fractional field, which is what a tick is."""


def _ticks(reading: str) -> int:
    """Read a Cassini clock string as a count of ticks.

    Parameters:
        reading: The clock string, partition and all.

    Returns:
        The reading, as a count of ticks of the fractional field.
    """
    seconds, fraction = reading.split('/', 1)[1].split('.')
    return int(seconds) * _TICKS_PER_SECOND + int(fraction)


@pytest.fixture
def clock_kernels() -> Iterator[None]:
    """Furnish the leapseconds and Cassini clock kernels, and nothing else.

    Yields:
        Nothing; the two kernels are furnished for the body of the test.
    """
    with isolated_kernel_pool():
        cspyce.furnsh(str(_LSK))
        cspyce.furnsh(str(_SCLK))
        yield


def test_every_cohort_reading_is_the_one_the_kernel_returns(clock_kernels: None) -> None:
    """A reading the kernel does not return for the epoch beside it is invented.

    Every reading of all three cohort documents is converted again here, from
    the epoch the document records, by the kernel the document says was loaded.
    A tick is as close as a reading of that clock can come to an epoch, so a
    tick is the tolerance.
    """
    disagreeing: list[str] = []
    for image in cohort_images():
        times = image.document['navigation_result']['times']
        for reading, epoch in (
            ('sclk_start', 'start_et'),
            ('sclk_midtime', 'midtime_et'),
            ('sclk_stop', 'stop_et'),
        ):
            from_the_kernel = cspyce.sce2s(_CASSINI_SCLK_ID, float(times[epoch]))
            apart = _ticks(str(times[reading])) - _ticks(str(from_the_kernel))
            if abs(apart) > 1:
                disagreeing.append(
                    f'{image.image_name}: {reading} is {times[reading]} where the kernel '
                    f'converts {times[epoch]} to {from_the_kernel}, {apart} ticks apart'
                )
    assert disagreeing == []


def test_every_cohort_image_is_named_for_the_second_the_kernel_gives_it(
    clock_kernels: None,
) -> None:
    """The name is the whole-second field of the reading at shutter open.

    An image whose number is not the one the kernel puts its own epoch in is an
    image no volume would hold under that name, and a bundle names every product
    it writes after it.
    """
    misnamed: list[str] = []
    for image in cohort_images():
        start_et = float(image.document['navigation_result']['times']['start_et'])
        from_the_kernel = str(cspyce.sce2s(_CASSINI_SCLK_ID, start_et))
        named_second = from_the_kernel.split('/', 1)[1].split('.')[0]
        if image.image_name[1:].split('_', 1)[0] != named_second:
            misnamed.append(
                f'{image.image_name} opened at {start_et}, which the kernel reads as '
                f'{from_the_kernel}'
            )
    assert misnamed == []
