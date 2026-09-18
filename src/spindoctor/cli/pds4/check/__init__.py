"""The bundle check: a PDS4 bundle tree that has been written, held to the PDS4
standard.

``sd_create_bundle check`` runs :func:`~spindoctor.cli.pds4.check.bundle.check_bundle`
over a bundle's directory.  It reads only the tree and the schemas the package ships, and
reports each way the tree departs from the PDS4 standard as a
:class:`~spindoctor.cli.pds4.check.findings.Finding`, an error or a warning:

- each label against the XML schemas it declares
  (:mod:`~spindoctor.cli.pds4.check.schemas`) and the Schematron rules it declares
  (:mod:`~spindoctor.cli.pds4.check.schematron`);
- each table read through its label alone (:mod:`~spindoctor.cli.pds4.check.tables`),
  each global index table against the layout the summary pass writes
  (:mod:`~spindoctor.cli.pds4.check.index_tables`), and each statistic column of one
  against the configuration (:mod:`~spindoctor.cli.pds4.check.statistic_columns`);
- the tree as a whole (:mod:`~spindoctor.cli.pds4.check.integrity`): the files its labels
  name, the identifiers they declare and refer to, the collections' inventories
  (:mod:`~spindoctor.cli.pds4.check.inventories`), and the global index tables' records.
"""

from spindoctor.cli.pds4.check.bundle import check_bundle, check_label
from spindoctor.cli.pds4.check.findings import CheckName, Finding, Severity

__all__ = ['CheckName', 'Finding', 'Severity', 'check_bundle', 'check_label']
