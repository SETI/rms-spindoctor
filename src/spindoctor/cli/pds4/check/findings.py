"""What the bundle check reports: a finding for each way a bundle tree departs from PDS4.

Every check of the package returns :class:`Finding` objects, and ``sd_create_bundle
check`` prints each as one line, naming the file, whether it is an error or a warning,
the check that found it, where in the file, and what is wrong.  Errors decide the exit
status; warnings are printed and counted.
"""

from dataclasses import dataclass
from enum import StrEnum

BUNDLE_DIRECTORY = '.'
"""What a finding about the tree as a whole names as its file."""


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


class Severity(StrEnum):
    """Whether a finding fails the check.

    Attributes:
        ERROR: The tree is not what PDS4 requires of it; any error fails the check.
        WARNING: A finding the PDS ``validate`` tool also reports as a warning: an
            unresolved reference to a product of the bundle, a product no inventory
            lists, and a Schematron rule whose ``role`` marks it a warning.  It is
            printed and counted, and does not fail the check.
    """

    ERROR = 'error'
    WARNING = 'warning'


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
        severity: Whether it is an error, which fails the check, or a warning.
    """

    file: str
    check: CheckName
    location: str
    message: str
    severity: Severity = Severity.ERROR

    def line(self) -> str:
        """Return the finding as the one line ``sd_create_bundle check`` prints for it.

        Returns:
            ``<file>: <severity> [<check>] <location>: <message>``, the location and the
            colon after it left out when the finding is about the file as a whole.
        """
        where = f'{self.location}: ' if self.location != '' else ''
        return f'{self.file}: {self.severity} [{self.check}] {where}{self.message}'


class RecordFindings:
    """Findings about one file's records, a problem recurring across records told once."""

    def __init__(self, file: str, check: CheckName) -> None:
        """Start with no finding.

        Parameters:
            file: The path, relative to the bundle's directory, of the label the
                findings name.
            check: The check making them.
        """
        self._file = file
        self._check = check
        self._found: list[Finding] = []
        self._recurring: dict[tuple[str, str], tuple[str, int]] = {}

    def add(self, location: str, message: str) -> None:
        """Record one finding, an error.

        Parameters:
            location: The path of the element it is about.
            message: What is wrong.
        """
        self._found.append(Finding(self._file, self._check, location, message))

    def add_recurring(self, location: str, kind: str, message: str) -> None:
        """Record an error in one record, keeping the first such record's message.

        Parameters:
            location: The path of the element it is about.
            kind: What kind of problem, which with the location says whether it recurs.
            message: What is wrong in this record.
        """
        key = (location, kind)
        if key in self._recurring:
            first, count = self._recurring[key]
            self._recurring[key] = (first, count + 1)
        else:
            self._recurring[key] = (message, 1)

    def findings(self) -> list[Finding]:
        """Return every finding: those recorded once, then each recurring problem.

        Returns:
            The findings, a recurring problem's message counting the records after the
            first that show it.
        """
        found = list(self._found)
        for (location, _), (message, count) in self._recurring.items():
            more = '' if count == 1 else f', and {count - 1} more record(s) like it'
            found.append(Finding(self._file, self._check, location, f'{message}{more}'))
        return found
