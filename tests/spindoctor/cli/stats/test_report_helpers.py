"""Tests for the pure helpers the report is assembled from.

None of these reads a record. They cover the values the report derives from an
image name or an epoch, the search limit it resolves from configuration, and how
one column value is rendered into a CSV cell.
"""

from typing import Any

import pytest

from spindoctor.cli.stats.report_common import count_pct, image_name_from_filename
from spindoctor.cli.stats.report_sections import _csv_value, resolve_offset_limit
from spindoctor.nav_records.derived import (
    date_from_image_et,
    datetime_from_image_et,
    image_number_from_name,
)

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------


def test_date_from_image_et_j2000() -> None:
    """The epoch itself, as the date filters compare it."""
    assert date_from_image_et(0.0) == '2000-01-01'


def test_date_from_image_et_none() -> None:
    """An image with no epoch gets no date rather than a wrong one."""
    assert date_from_image_et(None) is None


def test_date_from_image_et_dates_a_day_s_last_half_second_the_day_after() -> None:
    """An instant 0.3 s before midnight is rounded into, and dated, the next day.

    SPICE's ``et2utc`` writes the epoch at three decimals as
    ``2004-06-30T23:59:59.700`` and at none as ``2004-07-01T00:00:00``.
    """
    assert date_from_image_et(141912063.883703) == '2004-07-01'


def test_datetime_from_image_et_keeps_the_time() -> None:
    """The selection table shows a time; a bare date collapses a whole day."""
    assert datetime_from_image_et(0.0) == '2000-01-01T11:58:56'


@pytest.mark.parametrize(
    ('image_name', 'expected'),
    [
        ('N1454725799_1_CALIB.IMG', 1454725799),
        ('/some/dir/W1728613298_8.IMG', 1728613298),
        ('lor_0003103486_0x630_sci.fit', 3103486),
        ('1454725799', 1454725799),
        ('no-digits-here', None),
        (None, None),
    ],
)
def test_image_number_from_name(image_name: str | None, expected: int | None) -> None:
    """The value ingest stores in the column the range filter compares.

    Parameters:
        image_name: The name to read.
        expected: The number it holds.
    """
    assert image_number_from_name(image_name) == expected


@pytest.mark.parametrize(
    ('instrument', 'filename', 'expected'),
    [
        ('coiss', 'N1454725799_1_CALIB.IMG', 'N1454725799'),
        ('coiss', '/holdings/data/W1728613298_8.IMG', 'W1728613298'),
        ('vgiss', 'C3250013_GEOMED.IMG', 'C3250013'),
        ('gossi', 'C0349632000R.IMG', 'C0349632000R'),
        ('nhlorri', 'lor_0003103486_0x630_sci.fit', 'lor_0003103486'),
        # An unregistered instrument only loses its extension.
        ('mystery', 'X9999999.IMG', 'X9999999'),
    ],
)
def test_image_name_from_filename(instrument: str, filename: str, expected: str) -> None:
    """Every printed name is the token --image-filelist selects on.

    Parameters:
        instrument: The instrument whose naming rule applies.
        filename: The recorded image name.
        expected: The dataset-level image name.
    """
    assert image_name_from_filename(instrument, filename) == expected


def test_image_name_from_filename_is_idempotent() -> None:
    """Re-deriving a name that is already an image name changes nothing."""
    assert image_name_from_filename('coiss', 'N1454725799') == 'N1454725799'


def test_count_pct_formats_share() -> None:
    """Every count in the report carries its percentage."""
    assert count_pct(5, 158) == '5 (3.2%)'


def test_count_pct_zero_total() -> None:
    """An empty denominator renders 0.0% rather than dividing by zero."""
    assert count_pct(0, 0) == '0 (0.0%)'


# ---------------------------------------------------------------------------
# Offset-limit resolution
# ---------------------------------------------------------------------------


def test_resolve_offset_limit_coiss_nac_by_size() -> None:
    """Cassini NAC CALIB limits come from the per-size margin table."""
    assert resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', 1024) == (50.0, 140.0)


def test_resolve_offset_limit_coiss_wac() -> None:
    """Cassini WAC limits use the wac detector block."""
    assert resolve_offset_limit('coiss', 'W1454725799_1_CALIB.IMG', 512) == (5.0, 10.0)


def test_resolve_offset_limit_requires_shape_for_size_tables() -> None:
    """A size-keyed margin table cannot resolve without a recorded shape."""
    result = resolve_offset_limit('coiss', 'N1454725799_1_CALIB.IMG', None)
    assert result == 'image shape not recorded'


def test_resolve_offset_limit_unknown_instrument() -> None:
    """An unregistered instrument has no configured limit to screen against."""
    result = resolve_offset_limit('mystery', 'X123.IMG', 1024)
    assert 'no configured search limit' in str(result)


def test_resolve_offset_limit_missing_size_entry() -> None:
    """A size with no margin entry reports the failure instead of guessing."""
    result = resolve_offset_limit('vgiss', 'C3250013_GEOMED.IMG', 1024)
    assert 'no extfov_margin_vu entry for image size 1024' in str(result)


# ---------------------------------------------------------------------------
# One CSV cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ([1.0, -2.5], '[1.0, -2.5]'),
        ([], '[]'),
        ([[1.0, 0.0], [0.0, 4.0]], '[[1.0, 0.0], [0.0, 4.0]]'),
        ({'iterations': 4}, '{"iterations": 4}'),
        ({}, '{}'),
        ('BodyLimbNav', 'BodyLimbNav'),
        (1.5, 1.5),
        (None, None),
    ],
    ids=[
        'a list',
        'an empty list',
        'a matrix',
        'an object',
        'an empty object',
        'text',
        'a number',
        'nothing',
    ],
)
def test_a_json_container_is_written_as_json_text_and_nothing_else_is_touched(
    value: Any, expected: Any
) -> None:
    """A cell holding a Python container's repr is a cell nothing can read back.

    Driven directly rather than through an export, so that every shape a column
    can hold is covered rather than the few the fixture documents happen to
    carry.  A structured column reaches the export as the container it holds,
    whichever storage answered, so this is what a reader of the file gets.

    Parameters:
        value: The value the facts carry for a column.
        expected: The cell it becomes.
    """
    assert _csv_value(value) == expected
