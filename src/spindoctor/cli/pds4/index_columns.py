"""The columns of a bundle's global index tables, and how each statistic column is written.

A column is an :class:`IndexColumn`: its field name, its PDS4 data type and its
description, and, for a statistic, its unit, the format its values are written in and the
text a missing value is written as.  Each statistic is written to a fixed number of
decimals or of significant figures, by its unit (:data:`INDEX_VALUE_FORMATS`).

The statistic columns are built from the configuration's ``backplanes`` section.  Each
configured plane gives its table the columns of its least and its greatest value, and a
plane whose index block gives them, as a plane of longitudes' can, the two of its range
wrapped at zero (:class:`IndexPlane`); ``backplanes.ring_incidence_angle`` gives the
rings table the incidence angle's least, greatest and mean (:class:`RingIncidence`).
Where an image records no value for a cell, the cell holds the configured masked value
written in the column's format, which the column declares as its missing constant.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.config import Config


@dataclass(frozen=True)
class IndexValueFormat:
    """How a minimum or maximum is written into a global index table.

    Exactly one of the two fields is set.  ``decimals`` writes every value to
    that many decimals, so three decimals write ``1.235`` and ``-89.999``.
    ``significant`` writes every value to that many significant figures,
    positionally: the number of decimals is chosen per value from its
    magnitude, so five figures write ``0.00060000``, ``6.1343``, ``4200.0``,
    ``70853`` and ``123456``.  A value is never written with an exponent or a
    trailing point, and its integer part is never rounded away, so a value with
    more integer digits than figures is written to all of them.  Neither field
    fixes a column's width: values under one format differ in length, so the
    width of a column is the widest value written in it, known only once the
    column is.

    Attributes:
        decimals: The number of decimals every value is written to, or None
            when ``significant`` is set.
        significant: The number of significant figures every value is written
            to, or None when ``decimals`` is set.
    """

    decimals: int | None = None
    significant: int | None = None

    def __post_init__(self) -> None:
        """Refuse a format that sets both fields or neither.

        Raises:
            ValueError: If ``decimals`` and ``significant`` are both set or both
                None.
        """
        if (self.decimals is None) == (self.significant is None):
            raise ValueError(
                'An index value format sets exactly one of decimals and significant; '
                f'got decimals={self.decimals!r}, significant={self.significant!r}'
            )

    def render(self, value: float) -> str:
        """Write one value the way this format says.

        Parameters:
            value: The statistic to write.

        Returns:
            The value as a plain decimal number to a fixed number of decimals,
            with no exponent and no trailing point.

        Raises:
            ValueError: If ``value`` is NaN or infinite, which has no decimal
                form.  The message names the value.
        """
        if not math.isfinite(value):
            raise ValueError(f'An index value has to be a finite number; got {value!r}')
        if self.significant is None:
            return format(value, f'.{self.decimals}f')
        magnitude = 0 if value == 0 else math.floor(math.log10(abs(value)))
        decimals = max(0, self.significant - 1 - magnitude)
        return format(value, f'.{decimals}f')


INDEX_VALUE_FORMATS: dict[str, IndexValueFormat] = {
    'deg': IndexValueFormat(decimals=3),
    'km': IndexValueFormat(decimals=1),
    'deg/pixel': IndexValueFormat(decimals=8),
    'km/pixel': IndexValueFormat(significant=5),
}
"""The format each min and max in the global index tables is written in, by unit.

The key is the unit the statistic is in: for a plane in radians, the degrees unit
:func:`~spindoctor.cli.backplanes.statistics.statistics_units` gives it.  Each format
prints about what one pixel resolves, within the roughly seven significant digits a
float32 backplane array carries.  ``km/pixel`` values span orders of magnitude, so
that format is five significant figures rather than a fixed number of decimals.  No
value is written with an exponent.  The bodies and rings tables share the mapping.
"""


def index_value_format(units: str) -> IndexValueFormat:
    """Return the format a statistic of a plane in these units is written in.

    Parameters:
        units: The unit the plane's values carry, as the configuration declares
            it.  An angular plane's statistic is in degrees whatever the plane
            is in, so what is looked up is the statistic's unit rather than
            this one.

    Returns:
        The format, from :data:`INDEX_VALUE_FORMATS`.

    Raises:
        KeyError: If the statistic's unit has no format in the table.
    """
    return INDEX_VALUE_FORMATS[statistics_units(units)]


@dataclass(frozen=True)
class IndexColumn:
    """One column of a global index table: how its values are written, and its label.

    Attributes:
        name: The field's name, as the header line and the label give it: a PDS4
            dictionary attribute's name with the dictionary's prefix, as in
            ``pds:logical_identifier``, where the column holds that attribute, and a
            name of its own otherwise.
        data_type: The PDS4 data type of its values, as in ``ASCII_Real``.
        description: What its values are.
        unit: The unit of its values, or None for a column that has none.
        value_format: The format each statistic in the column is written in, or None
            for a column of text.  A statistic is right-justified in its field, and
            text left-justified.
        missing_constant: What a cell of the column holds where an image has no
            statistic for its plane, spelled as the cell spells it, or None for a column
            no cell of which is ever missing.
    """

    name: str
    data_type: str
    description: str
    unit: str | None = None
    value_format: IndexValueFormat | None = None
    missing_constant: str | None = None


def _column(
    index: Mapping[str, Any],
    key: str,
    *,
    unit: str,
    value_format: IndexValueFormat,
    missing: str,
) -> IndexColumn:
    """Return one statistic column, as its configuration's index block describes it.

    Parameters:
        index: The index block.
        key: The column's key in the block, as in ``minimum`` or ``wrapped_maximum``,
            under which the block gives its name and its description.
        unit: The unit of the column's values.
        value_format: The format they are written in.
        missing: What a cell of the column holds where an image records no value for it.

    Returns:
        The column, of the data type the block gives all its columns.
    """
    return IndexColumn(
        name=index[key]['name'],
        data_type=index['data_type'],
        description=index[key]['description'],
        unit=unit,
        value_format=value_format,
        missing_constant=missing,
    )


def _recorded_cell(
    recorded: Mapping[str, Any], key: str, *, value_format: IndexValueFormat, missing: str
) -> str:
    """Write the value a mapping records under a key, or the missing constant.

    Parameters:
        recorded: What a supplemental file records, as a statistic or the incidence angle.
        key: The value's key in it.
        value_format: The format the value is written in.
        missing: What the cell holds where the mapping records no value under the key.

    Returns:
        The value, rendered, or ``missing``.

    Raises:
        ValueError: If the value is not a finite number.
    """
    if key not in recorded:
        return missing
    return value_format.render(recorded[key])


_WRAPPED_COLUMNS = ('wrapped_minimum', 'wrapped_maximum')
"""The keys of an index block giving a plane of longitudes its two wrapped columns."""

_WRAPPED_VALUES = ('wrapped_min', 'wrapped_max')
"""Where a statistic records its range wrapped at zero: its start and its end."""


@dataclass(frozen=True)
class IndexPlane:
    """One configured plane, and the columns of an index table its statistic fills.

    Attributes:
        name: The plane's name, under which a supplemental file records its statistic.
        value_format: The format its statistic is written in.
        missing: What each of its cells holds where an image records no value for it: the
            masked value, written in ``value_format``.
        minimum: The column of its least value.
        maximum: The column of its greatest value.
        wrapped: The columns of its range wrapped at zero, where the range starts and where
            it ends, when its entry gives them, as a plane of longitudes' can; none
            otherwise.
    """

    name: str
    value_format: IndexValueFormat
    missing: str
    minimum: IndexColumn
    maximum: IndexColumn
    wrapped: tuple[IndexColumn, ...] = ()

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], *, masked_value: float) -> Self:
        """Return a configured plane and its columns, as its entry describes them.

        Each column takes its name and its description from the entry's ``index``
        block, and the data type the block gives all of them.  Its unit is the unit the
        plane's statistic is in, the entry's ``units`` restated through
        :func:`~spindoctor.cli.backplanes.statistics.statistics_units`, and its format
        the one :data:`INDEX_VALUE_FORMATS` gives that unit, so that a label states the
        unit and the format its column's values are written in.  Its missing constant
        is the masked value written in that format.  A block giving ``wrapped_minimum``
        and ``wrapped_maximum`` gives the plane two more columns, of its range wrapped
        at zero, in the same unit and format.

        Parameters:
            entry: The plane's configuration entry, from ``backplanes.bodies`` or
                ``backplanes.rings``.
            masked_value: The configured masked value, ``backplanes.masked_value``.

        Returns:
            The plane.

        Raises:
            KeyError: If the plane's statistic is in a unit the index has no format for.
        """
        value_format = index_value_format(entry['units'])
        unit = statistics_units(entry['units'])
        # The value every masked pixel of the arrays holds, in the column's own format,
        # so every value of a column, a missing one included, is written in one form
        # and the label declares it in that form (#601).
        missing = value_format.render(masked_value)
        index = entry['index']
        wrapped = _WRAPPED_COLUMNS if _WRAPPED_COLUMNS[0] in index else ()
        minimum, maximum, *wrapped_columns = (
            _column(index, key, unit=unit, value_format=value_format, missing=missing)
            for key in ('minimum', 'maximum', *wrapped)
        )
        return cls(
            name=entry['name'],
            value_format=value_format,
            missing=missing,
            minimum=minimum,
            maximum=maximum,
            wrapped=tuple(wrapped_columns),
        )

    @property
    def columns(self) -> tuple[IndexColumn, ...]:
        """The plane's columns, in the order a row gives them.

        Returns:
            The column of its least value and that of its greatest, and then those of
            its range wrapped at zero, when it has them.
        """
        return (self.minimum, self.maximum, *self.wrapped)

    def cells(self, statistic: Mapping[str, Any] | None) -> list[str]:
        """Write the plane's statistic as the cells an index row gives it.

        Parameters:
            statistic: The plane's statistic as a supplemental file records it, its
                minimum and maximum already checked as finite numbers, or None when
                the file records none for the plane.

        Returns:
            The minimum and the maximum, rendered, or :attr:`missing` twice when there
            is no statistic, which is what a plane that measured nothing leaves; then,
            for a plane with wrapped columns, where its range wrapped at zero starts and
            where it ends, each rendered, or :attr:`missing` where the statistic records
            no such value, as backplanes an earlier version generated do not.

        Raises:
            ValueError: If a value written is not a finite number.
        """
        if statistic is None:
            cells = [self.missing, self.missing]
        else:
            cells = [
                self.value_format.render(statistic['min']),
                self.value_format.render(statistic['max']),
            ]
        if len(self.wrapped) > 0:
            recorded: Mapping[str, Any] = {} if statistic is None else statistic
            cells.extend(
                _recorded_cell(recorded, key, value_format=self.value_format, missing=self.missing)
                for key in _WRAPPED_VALUES
            )
        return cells


_INCIDENCE_COLUMNS = (('minimum', 'min'), ('maximum', 'max'), ('mean', 'mean'))
"""Each ring incidence column's key in its index block, and the key of its value in the
rings block's ``incidence_angle``, in the order the rings table gives them."""


@dataclass(frozen=True)
class RingIncidence:
    """The rings table's three columns of the incidence angle of sunlight on the rings.

    The backplane metadata's rings block records the angle once for an image, as
    ``incidence_angle``, rather than as a plane: its ``value`` at the ring system's center
    and, when a ring pixel has an angle, its ``min``, ``max`` and ``mean`` over the ring
    pixels, which the three columns hold.

    Attributes:
        value_format: The format each value is written in.
        missing: What a cell holds where the image records no value for it: the masked
            value, written in ``value_format``.
        columns: The columns of the least, the greatest and the mean angle.
    """

    value_format: IndexValueFormat
    missing: str
    columns: tuple[IndexColumn, ...]

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any], *, masked_value: float) -> Self:
        """Return the columns ``backplanes.ring_incidence_angle`` describes.

        Each takes its name and its description from the entry's ``index`` block, and the
        data type the block gives all three; its unit is the entry's ``units`` restated
        through :func:`~spindoctor.cli.backplanes.statistics.statistics_units`, and its
        format and its missing constant are the ones a plane's columns in that unit take.

        Parameters:
            entry: The configuration's ``backplanes.ring_incidence_angle``.
            masked_value: The configured masked value, ``backplanes.masked_value``.

        Returns:
            The columns.

        Raises:
            KeyError: If the angle is in a unit the index has no format for.
        """
        value_format = index_value_format(entry['units'])
        missing = value_format.render(masked_value)
        return cls(
            value_format=value_format,
            missing=missing,
            columns=tuple(
                _column(
                    entry['index'],
                    key,
                    unit=statistics_units(entry['units']),
                    value_format=value_format,
                    missing=missing,
                )
                for key, _ in _INCIDENCE_COLUMNS
            ),
        )

    def cells(self, incidence: Mapping[str, Any]) -> list[str]:
        """Write the incidence angle a rings block records as the cells a row gives it.

        Each member is written on its own, and the block's ``units`` is not read, so a
        block recording some members and not others gives the members it records.

        Parameters:
            incidence: The rings block's ``incidence_angle``, as a supplemental file
                records it; empty when the block records none.

        Returns:
            The least, the greatest and the mean angle, each rendered, or
            :attr:`missing` where the block records no such member, as backplanes an
            earlier version generated record the angle at the center alone.

        Raises:
            ValueError: If a member written is not a finite number.
        """
        return [
            _recorded_cell(incidence, key, value_format=self.value_format, missing=self.missing)
            for _, key in _INCIDENCE_COLUMNS
        ]


def plane_columns(planes: Sequence[IndexPlane]) -> list[IndexColumn]:
    """Return the columns the configured planes give an index table, in order.

    Parameters:
        planes: The configured planes, in the configuration's order.

    Returns:
        Each plane's columns: its minimum, its maximum, and any wrapped ones.
    """
    return [column for plane in planes for column in plane.columns]


def configured_columns(
    config: Config,
) -> tuple[list[IndexPlane], list[IndexPlane], RingIncidence]:
    """Return the planes each table gives columns to, and the ring incidence angle's.

    Parameters:
        config: The configuration, whose ``backplanes`` section declares the body and the
            ring planes, each with its index block, the ring incidence angle's index
            block, and the masked value.

    Returns:
        The body planes and the ring planes, each in the configuration's order, and the
        ring incidence angle's columns.

    Raises:
        KeyError: If a column's values are in a unit the index has no format for.
    """
    backplanes = config.backplanes
    masked_value = float(backplanes.masked_value)
    return (
        [IndexPlane.from_entry(entry, masked_value=masked_value) for entry in backplanes.bodies],
        [IndexPlane.from_entry(entry, masked_value=masked_value) for entry in backplanes.rings],
        RingIncidence.from_entry(backplanes.ring_incidence_angle, masked_value=masked_value),
    )


def statistic_index_columns(config: Config) -> list[IndexColumn]:
    """Return every statistic column the configuration gives the global index tables.

    The summary pass writes the tables from the same columns, so a label can be held to
    them.

    Parameters:
        config: The configuration, whose ``backplanes`` section declares them.

    Returns:
        The bodies table's statistic columns, in the order the table gives them, and then
        the rings table's.

    Raises:
        KeyError: If a column's values are in a unit the index has no format for.
    """
    body_planes, ring_planes, ring_incidence = configured_columns(config)
    return [*plane_columns(body_planes), *plane_columns(ring_planes), *ring_incidence.columns]
