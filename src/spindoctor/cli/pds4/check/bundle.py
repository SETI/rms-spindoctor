"""A bundle tree held to PDS4, reading only the tree and the schemas the package ships.

:func:`check_bundle` holds every label of a tree to the XML schemas and the Schematron
it declares, reads every table through its label, holds the global index tables to the
layout the summary pass writes them in and their statistic columns to the configuration,
and checks the tree as a whole.  It writes nothing.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lxml import etree

from spindoctor.cli.pds4.check.findings import CheckName, Finding
from spindoctor.cli.pds4.check.index_tables import index_layout_findings
from spindoctor.cli.pds4.check.integrity import LABEL_SUFFIX, integrity_findings
from spindoctor.cli.pds4.check.schemas import label_schema, xsd_findings
from spindoctor.cli.pds4.check.schematron import schematron_findings
from spindoctor.cli.pds4.check.statistic_columns import (
    StatisticColumn,
    statistic_column_findings,
    statistic_columns,
)
from spindoctor.cli.pds4.check.tables import table_findings
from spindoctor.config import Config


def check_label(
    file: str, label: Path, document: Any, *, columns: Mapping[str, StatisticColumn]
) -> list[Finding]:
    """Check one label on its own: its schemas, Schematron, tables and columns.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        label: The label's file, beside which its tables are.
        document: The label, parsed by lxml.
        columns: The configured statistic columns, as
            :func:`~spindoctor.cli.pds4.check.statistic_columns.statistic_columns` gives
            them.

    Returns:
        What resolving its XML schemas found, its XML schema errors, its failed
        Schematron asserts and fired reports, the ways its tables depart from it, its
        statistic columns that depart from the configuration, and the ways a global index
        table departs from the layout the summary pass writes.
    """
    resolved = label_schema(file, document)
    findings = list(resolved.findings)
    if resolved.schema is not None:
        findings.extend(xsd_findings(file, document, resolved.schema))
    findings.extend(schematron_findings(file, document))
    findings.extend(table_findings(file, label, document, resolved.schema))
    findings.extend(statistic_column_findings(file, document, columns))
    findings.extend(index_layout_findings(file, label, document))
    return findings


def check_bundle(bundle_dir: Path, *, config: Config) -> list[Finding]:
    """Check a bundle tree that has already been written.

    Every file under the directory whose name ends in ``.lblx`` is a label.  A label that
    is not well-formed XML is one finding and is checked no further; every other is
    checked on its own, by :func:`check_label`, and then the tree as a whole, by
    :func:`~spindoctor.cli.pds4.check.integrity.integrity_findings`.

    Parameters:
        bundle_dir: The bundle's directory, on the local file system.
        config: The configuration the bundle was written under, whose ``backplanes``
            section declares the statistic columns of its global index tables.

    Returns:
        Every finding, those about one file together, in the order of the files' paths.
    """
    columns = statistic_columns(config)
    labels: dict[str, Any] = {}
    findings: list[Finding] = []
    for label in sorted(bundle_dir.rglob(f'*{LABEL_SUFFIX}')):
        file = label.relative_to(bundle_dir).as_posix()
        try:
            document = etree.parse(str(label))
        except etree.XMLSyntaxError as exc:
            findings.append(Finding(file, CheckName.XML, '', str(exc)))
            continue
        labels[file] = document
        findings.extend(check_label(file, label, document, columns=columns))
    findings.extend(integrity_findings(bundle_dir, labels))
    return sorted(findings, key=lambda finding: finding.file)
