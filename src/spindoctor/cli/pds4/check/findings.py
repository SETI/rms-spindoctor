"""What the bundle check reports: one finding for each way a bundle tree departs from PDS4.

Every check of the package returns :class:`Finding` objects, and ``sd_create_bundle
check`` prints each as one line, naming the file, the check that found it, where in the
file, and what is wrong.
"""

from dataclasses import dataclass
from enum import StrEnum


class CheckName(StrEnum):
    """Which part of the bundle check found a finding.

    Attributes:
        XML: The label is not well-formed XML, so nothing else in it is checked.
        XSD: The label against the XML schemas it declares, or the declaration itself.
        SCHEMATRON: The label against the Schematron rules it declares, or the
            declaration itself.
        TABLE: A table read through its label alone, and a statistic column of a global
            index table against the configuration.
        INTEGRITY: The tree as a whole: the files its labels name, the files no label
            names, the markers and the empty elements a label holds, and the references
            to the bundle's own products.
    """

    XML = 'xml'
    XSD = 'xsd'
    SCHEMATRON = 'schematron'
    TABLE = 'table'
    INTEGRITY = 'integrity'


@dataclass(frozen=True)
class Finding:
    """One way a bundle tree departs from what a PDS4 bundle has to be.

    Attributes:
        file: The file the finding is about, relative to the bundle's directory, in
            POSIX form, or ``.`` for the bundle's directory itself.
        check: Which check found it.
        location: Where in the file: the path of an element, as in
            ``/Product_Bundle/Identification_Area/Citation_Information/doi``, a table's
            record and field, or the empty string for the file as a whole.
        message: What is wrong.
    """

    file: str
    check: CheckName
    location: str
    message: str

    def line(self) -> str:
        """Return the finding as the one line ``sd_create_bundle check`` prints for it.

        Returns:
            ``<file>: [<check>] <location>: <message>``, the location and the colon after
            it left out when the finding is about the file as a whole.
        """
        where = f'{self.location}: ' if self.location != '' else ''
        return f'{self.file}: [{self.check}] {where}{self.message}'
