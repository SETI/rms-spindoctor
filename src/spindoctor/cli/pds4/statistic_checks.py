"""Why both bundle passes hold every statistic to what an index column can hold.

Every statistic in a backplane metadata document records the unit it is in beside
its minimum and maximum, and each column of the global index tables is the
configured unit of its plane restated through
:func:`~spindoctor.cli.backplanes.statistics.statistics_units`, each value written
in the format that unit calls for.  Two things about a statistic keep it out of
such a column.  One in some other unit, or in none, was written before the
statistics recorded their unit or under another configuration, and a column built
from it beside the others would be in two units with nothing saying so.  A minimum
or maximum that is not a finite number -- NaN, an infinity, or no number at all --
has no decimal form a column can hold, and a blank in its place would say the plane
measured nothing, which the document does not say either.

Both passes hold a document to both properties, since either can reach a table
through either pass.  The labels pass fails the image, before writing anything for
it, since the backplane root it reads can hold such a document beside regenerated
ones.  The summary pass fails the run, before writing either table, since the
bundle tree it reads can hold supplemental files a labels pass wrote before it
held a document to them: the labels pass refuses a bundle root that is not empty,
so it never mixes a tree itself, but it is not the only writer of one.  The check
is the same in both passes and lives here so that neither can drift from the
other.
"""

import math
from dataclasses import dataclass
from typing import Any

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.config import Config

__all__ = ['UnindexableStatistic', 'unindexable_statistic']


_UNIT_REASON = (
    'a backplane document says that when it was recorded before the statistics '
    'carried their unit, or under another configuration'
)
"""Why a statistic in another unit is in it, which is all a message can say of it."""

_VALUE_REASON = (
    'an index column holds only finite numbers, and a blank in one would say the '
    'plane measured nothing'
)
"""Why a minimum or maximum that is not a finite number cannot be indexed."""


@dataclass(frozen=True)
class UnindexableStatistic:
    """A statistic of a configured plane that no global index column can hold.

    Attributes:
        plane: The name of the configured plane the statistic belongs to.
        description: What the document records for the plane, as a clause beginning
            with ``records`` that a message completes by putting the document before
            it: ``records the ring_radius statistic in m where the configuration
            expects km`` (``in no unit at all`` when none is recorded), ``records a
            ring_radius minimum of nan, which is not a finite number``, or
            ``records no ring_radius maximum``.
        reason: Why a column cannot take the statistic as recorded, as a clause a
            message puts after the description.  For a unit it names the two ways a
            document comes to record another one; for a value it says what a
            column holds instead.
    """

    plane: str
    description: str
    reason: str


def unindexable_statistic(
    backplane_metadata: dict[str, Any], config: Config
) -> UnindexableStatistic | None:
    """Find a statistic in the document that no global index column can hold.

    Every configured body and ring plane the document holds a statistic for is
    checked.  Its recorded unit has to be the one the configuration gives the
    plane, restated as the statistic's unit, and its minimum and maximum each
    have to be a finite number.  A plane the document holds that the
    configuration does not declare is not checked, and a plane the configuration
    declares that the document lacks is no concern of this check.

    Parameters:
        backplane_metadata: The backplane metadata document as read, which is also
            what the ``backplanes`` member of a supplemental file holds.
        config: The configuration the run is under, whose declared planes and units
            the document is held to.

    Returns:
        The first statistic no column can hold, over the bodies in document order
        with the rings after them, the planes of each in configuration order, and
        within a plane its unit before its minimum and its minimum before its
        maximum.  None when every statistic can be indexed.

    Raises:
        TypeError: If the configuration entry of a plane the document holds has no
            ``units``, or a ``units`` that is not a string.
        ValueError: If that entry's ``units`` is blank.
    """
    bodies = backplane_metadata.get('bodies', {})
    rings = backplane_metadata.get('rings', {})
    recorded: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [
        (body.get('backplanes', {}), config.backplanes.bodies) for body in bodies.values()
    ]
    recorded.append((rings.get('backplanes', {}), config.backplanes.rings))
    for planes, entries in recorded:
        for entry in entries:
            name = entry['name']
            if name not in planes:
                continue
            # The entry is read from YAML, so its units can be anything, a
            # missing key included, and statistics_units refuses whatever is not
            # a string with a TypeError rather than this raising a KeyError.
            units: Any = entry.get('units')
            finding = _unindexable_part(name, planes[name], statistics_units(units))
            if finding is not None:
                return finding
    return None


def _unindexable_part(
    name: str, statistic: dict[str, Any], expected: str
) -> UnindexableStatistic | None:
    """Check one plane's statistic: its unit, then its minimum, then its maximum.

    Parameters:
        name: The plane's name.
        statistic: The plane's statistic as the document records it.
        expected: The unit the configuration gives a statistic of the plane.

    Returns:
        The first part of the statistic no column can hold, or None when none is.
    """
    found = statistic.get('units')
    if found != expected:
        recorded = 'no unit at all' if found is None else found
        return UnindexableStatistic(
            plane=name,
            description=(
                f'records the {name} statistic in {recorded} where the configuration '
                f'expects {expected}'
            ),
            reason=_UNIT_REASON,
        )
    for key, word in (('min', 'minimum'), ('max', 'maximum')):
        if key not in statistic:
            return UnindexableStatistic(
                plane=name, description=f'records no {name} {word}', reason=_VALUE_REASON
            )
        value = statistic[key]
        if not _is_finite_number(value):
            return UnindexableStatistic(
                plane=name,
                description=f'records a {name} {word} of {value!r}, which is not a finite number',
                reason=_VALUE_REASON,
            )
    return None


def _is_finite_number(value: Any) -> bool:
    """Report whether a value read from a document is a finite number.

    Parameters:
        value: The value as the JSON reader returned it.

    Returns:
        True for an integer, and for a float that is neither NaN nor infinite.
        False for anything else, a boolean included.
    """
    # A JSON true or false reads as a bool, which Python counts as an int, and is
    # not a measurement.  An integer is finite whatever its size, so it is not
    # handed to isfinite, which raises on one too large for a float.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)
