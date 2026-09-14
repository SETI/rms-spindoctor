"""The statistic columns of a bundle's global index tables, held to the configuration.

The configuration declares each backplane plane with the unit its array carries and, in
its ``index`` block, the names of the two columns a global index table gives its
statistic, the least and the greatest value.  A label describing such a column -- a
``Field_Character`` of that name -- has to state the unit the plane's statistic is in,
which :func:`~spindoctor.cli.backplanes.statistics.statistics_units` derives from the
plane's unit, and has to declare as its missing constant the configured masked value
written in the format of that unit,
:func:`~spindoctor.cli.pds4.global_index.index_value_format`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.cli.backplanes.statistics import statistics_units
from spindoctor.cli.pds4.check.elements import child_text, element_path
from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.global_index import index_value_format
from spindoctor.config import Config


@dataclass(frozen=True)
class StatisticColumn:
    """What the configuration says one statistic column of a global index table holds.

    Attributes:
        unit: The unit of its values: the unit its plane's statistic is in.
        missing_constant: The configured masked value, written in the column's format.
    """

    unit: str
    missing_constant: str


def statistic_columns(config: Config) -> dict[str, StatisticColumn]:
    """Return every statistic column the configuration declares, by its name.

    Parameters:
        config: The configuration, whose ``backplanes`` section declares the planes.

    Returns:
        Each plane's minimum and maximum columns, by the names its ``index`` block gives
        them, over the body planes and the ring planes.

    Raises:
        KeyError: If a plane's statistic is in a unit the index tables have no format for.
    """
    backplanes = config.backplanes
    masked_value = float(backplanes.masked_value)
    columns: dict[str, StatisticColumn] = {}
    for entry in [*backplanes.bodies, *backplanes.rings]:
        column = StatisticColumn(
            unit=statistics_units(entry['units']),
            missing_constant=index_value_format(entry['units']).render(masked_value),
        )
        for end in ('minimum', 'maximum'):
            columns[str(entry['index'][end]['name'])] = column
    return columns


def statistic_column_findings(
    file: str, document: Any, columns: Mapping[str, StatisticColumn]
) -> list[Finding]:
    """Hold each statistic column a label describes to the configuration.

    Parameters:
        file: The label's path relative to the bundle's directory, which the findings name.
        document: The label, parsed by lxml.
        columns: The configured statistic columns, as :func:`statistic_columns` gives them.

    Returns:
        One finding for each column stating a unit other than its plane's statistic's,
        and one for each declaring a missing constant other than the masked value in the
        column's format.
    """
    findings: list[Finding] = []
    for field in document.getroot().iter('{*}Field_Character'):
        name = child_text(field, 'name')
        if name is None or name not in columns:
            continue
        expected = columns[name]
        unit = child_text(field, 'unit')
        if unit != expected.unit:
            stated = 'no unit' if unit is None else unit
            findings.append(
                Finding(
                    file,
                    CheckName.TABLE,
                    element_path(field),
                    f"{name} states {stated}, but its plane's statistic is in {expected.unit}",
                )
            )
        missing = child_text(field, 'Special_Constants', 'missing_constant')
        if missing != expected.missing_constant:
            if missing is None:
                stated = 'no missing constant'
            else:
                stated = f'the missing constant {missing!r}'
            findings.append(
                Finding(
                    file,
                    CheckName.TABLE,
                    element_path(field),
                    f'{name} declares {stated}, but the masked value in its format is '
                    f'{expected.missing_constant!r}',
                )
            )
    return findings
