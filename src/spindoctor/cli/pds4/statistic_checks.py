"""What both bundle passes require of every statistic they index.

Each column of the global index tables is in one unit, the configured unit of its
plane restated through :func:`~spindoctor.cli.backplanes.statistics.statistics_units`,
and holds only finite numbers, each written in the format that unit calls for.  So a
statistic of a configured plane can be indexed only when the unit it records is that
unit and neither its minimum nor its maximum is NaN or infinite.  A statistic in
another unit would put its column in two units with nothing saying so; a NaN or an
infinity has no decimal form, and a blank in its place would say the plane measured
nothing.

The labels pass fails an image whose backplane metadata holds such a statistic,
before writing anything for it, and the summary pass fails the run over a
supplemental file that holds one, before writing either index table.  Both use the
check here, so the two cannot differ.
"""

import math
from dataclasses import dataclass
from typing import Any

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.config import Config

__all__ = ['UnindexableStatistic', 'unindexable_statistic']


_UNIT_REASON = (
    'every column of an index table is in one unit, the one the configuration gives its plane'
)
"""Why a statistic in another unit cannot be indexed."""

_VALUE_REASON = (
    'an index column holds only finite numbers, and a blank in one would say the plane '
    'measured nothing'
)
"""Why a NaN or infinite minimum or maximum cannot be indexed."""


@dataclass(frozen=True)
class UnindexableStatistic:
    """A statistic of a configured plane that no global index column can hold.

    Attributes:
        plane: The name of the configured plane the statistic belongs to.
        description: What the document records for the plane, as a clause beginning
            with ``records`` that a message completes by putting the document before
            it: ``records the ring_radius statistic in m where the configuration
            expects km``, or ``records a ring_radius minimum of nan, which is not a
            finite number``.
        reason: Why a column cannot take the statistic as recorded, as a clause a
            message puts after the description: for a unit, that every column is in
            the one unit the configuration gives its plane; for a value, what a
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
    plane, restated as the statistic's unit, and neither its minimum nor its
    maximum may be NaN or infinite.  A plane the
    document holds that the configuration does not declare is not checked, and a
    plane the configuration declares that the document lacks is no concern of
    this check.

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
        return UnindexableStatistic(
            plane=name,
            description=(
                f'records the {name} statistic in {found} where the configuration '
                f'expects {expected}'
            ),
            reason=_UNIT_REASON,
        )
    for key, word in (('min', 'minimum'), ('max', 'maximum')):
        value = statistic[key]
        if not math.isfinite(value):
            return UnindexableStatistic(
                plane=name,
                description=f'records a {name} {word} of {_described(value)}',
                reason=_VALUE_REASON,
            )
    return None


def _described(value: Any) -> str:
    """Describe a minimum or maximum no column can hold, for a message.

    Parameters:
        value: A NaN or infinite value, as the JSON reader returned it.

    Returns:
        The value's representation and why no column can hold it: ``nan, which is
        not a finite number``.
    """
    return f'{value!r}, which is not a finite number'
