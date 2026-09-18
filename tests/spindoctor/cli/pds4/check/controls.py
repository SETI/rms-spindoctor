"""Controls for the bundle check: a copy of a written bundle, broken in one place.

A control is a bundle the cohort build wrote, copied into a test's own directory and
changed in exactly one place, so that the test can hold the check to finding the change.
"""

import re
import shutil
from pathlib import Path
from typing import Any

from lxml import etree

from spindoctor.cli.pds4.check.elements import child_text, element_path
from spindoctor.cli.pds4.check.schemas import SchemaSource

SCHEMA_COPIES = Path(__file__).resolve().parent / 'schemas'
"""Local copies of the PDS4 schemas the cohort's labels name and those schemas import.

Each is byte for byte the file its URL serves, under the name the URL ends in, so that
the check's tests read the schemas from here and fetch nothing.
"""

SCHEMAS = SchemaSource(SCHEMA_COPIES)
"""The schema source the check's tests take the schemas from: the local copies."""


def copy_bundle(bundle_dir: Path, destination: Path) -> Path:
    """Copy a written bundle into a test's own directory.

    Parameters:
        bundle_dir: The bundle's directory, which is not changed.
        destination: The test's directory.

    Returns:
        The copy's bundle directory, with the original's name.
    """
    copy = destination / bundle_dir.name
    shutil.copytree(bundle_dir, copy)
    return copy


def substitute_once(
    path: Path, pattern: str, replacement: str, *, within: str | None = None
) -> None:
    """Replace the one match of a pattern in a file, or in one field of a table's label.

    A pattern that does not match exactly once fails the test making the control, by
    assertion: the bundle no longer holds what the control was written to change.

    Parameters:
        path: The file.
        pattern: A regular expression that matches exactly once where it is looked for.
        replacement: What replaces the match; ``\\1`` and the like name its groups.
        within: The name of the field whose element the match is looked for in, from its
            ``name`` to the end of the element, or None for the whole file.
    """
    text = path.read_text(encoding='utf-8')
    start = 0
    end = len(text)
    if within is not None:
        start = text.index(f'<name>{within}</name>')
        end = text.index('</Field_', start)
    matches = len(re.findall(pattern, text[start:end]))
    assert matches == 1, f'{pattern!r} matches {matches} time(s) in {path}, not once'
    changed = re.sub(pattern, replacement, text[start:end], count=1)
    path.write_text(text[:start] + changed + text[end:], encoding='utf-8')


def parse(path: Path) -> Any:
    """Parse a label.

    Parameters:
        path: The label.

    Returns:
        The label, parsed by lxml.
    """
    return etree.parse(str(path))


def parsed_labels(bundle_dir: Path) -> dict[str, Any]:
    """Parse every label of a bundle.

    Parameters:
        bundle_dir: The bundle's directory.

    Returns:
        Each label, parsed, by its path relative to the bundle's directory, in the order
        of the paths.
    """
    return {
        label.relative_to(bundle_dir).as_posix(): parse(label)
        for label in sorted(bundle_dir.rglob('*.lblx'))
    }


def line_of(path: Path, text: str) -> int:
    """Return the line of a file its first occurrence of a text is on.

    Parameters:
        path: The file.
        text: The text.

    Returns:
        The line, counted from 1.
    """
    content = path.read_text(encoding='utf-8')
    return content[: content.index(text)].count('\n') + 1


def table_field(bundle_dir: Path, file: str, name: str) -> tuple[str, int, int]:
    """Return where a field of a character table lies, as its label says.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The table's label, relative to it.
        name: The field's name.

    Returns:
        The path of the field's element, and where it begins and ends in a record, counted
        from 0, the end exclusive.
    """
    element = next(
        field
        for field in parse(bundle_dir / file).getroot().iter('{*}Field_Character')
        if child_text(field, 'name') == name
    )
    location = int(child_text(element, 'field_location') or '0')
    length = int(child_text(element, 'field_length') or '0')
    return element_path(element), location - 1, location - 1 + length


def table_field_names(bundle_dir: Path, file: str) -> list[str]:
    """Return the names of a character table's fields, in the order its label gives them.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The table's label, relative to it.

    Returns:
        The names.
    """
    root = parse(bundle_dir / file).getroot()
    return [child_text(field, 'name') or '' for field in root.iter('{*}Field_Character')]


def table_records(bundle_dir: Path, file: str) -> tuple[int, list[bytes]]:
    """Return a global index table's header length and its records.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The table's label, relative to it.

    Returns:
        The header line's length, its line feed included, and each record, its line feed
        included.
    """
    header, _, body = (bundle_dir / file).with_suffix('.tab').read_bytes().partition(b'\n')
    return len(header) + 1, body.splitlines(keepends=True)


def bare_bundle_label(schema_location: str) -> etree._ElementTree:
    """Return a bundle label that declares the given XML schemas and holds nothing.

    Parameters:
        schema_location: Its ``xsi:schemaLocation``, each namespace followed by a URL.

    Returns:
        The label, parsed.
    """
    text = (
        '<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        f'xsi:schemaLocation="{schema_location}"/>'
    )
    return etree.fromstring(text.encode('ascii')).getroottree()
