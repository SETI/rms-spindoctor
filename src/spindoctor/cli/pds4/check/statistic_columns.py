"""The statistic columns of a bundle's global index tables, held to the configuration.

The summary pass builds every statistic column of the global index tables from the
configuration, through :func:`~spindoctor.cli.pds4.index_columns.statistic_index_columns`:
each configured plane's least and greatest value, the range wrapped at zero of a plane
whose index block gives it columns, and the ring incidence angle's least, greatest and
mean.  A label describing such a column -- a ``Field_Character`` of that name -- has to
state the unit its plane's statistic is in, and has to declare as its missing constant
the configured masked value written in the column's format.

The columns are the summary pass's own, taken from the same function, so that each rule
is stated once: the check holds a label to the configuration as the summary pass reads
it, and a defect in that function would pass it.  The summary pass's tests pin its parts
themselves, in ``tests/spindoctor/cli/backplanes/test_statistics.py`` and
``tests/spindoctor/cli/pds4/test_global_index.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.cli.pds4.check.elements import child_text, element_path
from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.index_columns import statistic_index_columns
from spindoctor.config import Config


@dataclass(frozen=True)
class StatisticColumn:
    """What the configuration says one statistic column of a global index table holds.

    Attributes:
        unit: The unit of its values.
        missing_constant: The configured masked value, written in the column's format.
    """

    unit: str
    missing_constant: str


def statistic_columns(config: Config) -> dict[str, StatisticColumn]:
    """Return every statistic column the configuration declares, by its name.

    Parameters:
        config: The configuration, whose ``backplanes`` section declares the columns.

    Returns:
        Each statistic column of either global index table, by the name its index block
        gives it: each plane's minimum and maximum, a plane's wrapped pair, and the ring
        incidence angle's three.

    Raises:
        KeyError: If a plane's statistic is in a unit the index tables have no format for.
    """
    return {
        column.name: StatisticColumn(unit=column.unit, missing_constant=column.missing_constant)
        for column in statistic_index_columns(config)
        if column.unit is not None and column.missing_constant is not None
    }


def statistic_column_findings(
    file: str, document: Any, columns: Mapping[str, StatisticColumn]
) -> list[Finding]:
    """Hold each statistic column a label describes to the configuration.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        document: The label, parsed by lxml.
        columns: The configured statistic columns, which :func:`statistic_columns` gives.

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
