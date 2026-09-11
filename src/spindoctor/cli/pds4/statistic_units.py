"""Why both bundle passes hold a document to the unit its configuration declares.

Every statistic in a backplane metadata document records the unit it is in,
and each column of the global index tables is the configured unit of its
plane restated through
:func:`~spindoctor.cli.backplanes.statistics.statistics_units`.  A document
whose statistic is in some other unit, or in none, was written before the
statistics recorded their unit or under another configuration, and a column
built from it beside the others would be in two units with nothing saying so.
Both passes refuse it.  The labels pass fails the image, before writing
anything for it, since the backplane root it reads can hold such a document
beside regenerated ones.  The summary pass fails the run, before writing
either table, since the bundle tree it reads can hold supplemental files a
labels pass wrote before the unit was recorded or held to: the labels pass
refuses a bundle root that is not empty, so it never mixes a tree itself, but
it is not the only writer of one.  The comparison is the same in both passes
and lives here so that neither can drift from the other.
"""

from typing import Any

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.config import Config

__all__ = ['statistic_in_another_unit']


def statistic_in_another_unit(
    backplane_metadata: dict[str, Any], config: Config
) -> tuple[str, str | None, str] | None:
    """Find a statistic the document records in a unit other than its configured one.

    Every configured body and ring plane the document holds a statistic for is
    compared.  A plane the document holds that the configuration does not
    declare is ignored, and a plane the configuration declares that the
    document lacks is no concern of this check.

    Parameters:
        backplane_metadata: The backplane metadata document as read, which is
            also what the ``backplanes`` member of a supplemental file holds.
        config: The configuration the run is under, whose declared planes and
            units the document is held to.

    Returns:
        For the first disagreeing statistic, over the bodies in document order
        with the rings after them, and over the planes of each in
        configuration order: the plane's name, the unit the document records
        for it (None when it records none), and the unit a statistic of that
        plane takes under the configuration.  None when every statistic
        agrees.
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
            expected = statistics_units(entry['units'])
            found = planes[name].get('units')
            if found != expected:
                return name, found, expected
    return None
