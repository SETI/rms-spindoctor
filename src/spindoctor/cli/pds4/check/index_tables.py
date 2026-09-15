"""The global index tables, held to the layout the summary pass writes and to the tree.

The summary pass writes the global index tables, ``global_bodies_index.tab`` and
``global_rings_index.tab`` in the miscellaneous collection, through
:mod:`spindoctor.cli.pds4.global_index`: a header line naming the columns, separated by
commas, and then one record per row, each field padded to its length and followed by a
comma, the last by the record delimiter.  Each table's label describes the header line
as a ``Header`` and the records as a ``Table_Character`` of one ``Field_Character`` per
column.  PDS4 does not constrain what a header holds or the bytes between two fields, so
the table reader (:mod:`~spindoctor.cli.pds4.check.tables`) reads neither;
:func:`index_layout_findings` holds each index table to its layout:

- the header line names the label's fields, in order, separated by commas;
- a comma alone lies between two fields, and the last field ends where the record
  delimiter begins;
- in every record, the byte after each field but the last is a comma.

:func:`index_row_findings` holds each table's records to the tree:

- each record's ``pds:logical_identifier`` names a product a label of the tree
  declares, and its ``file_spec`` is that label's path;
- each other column named for an attribute of the PDS4 common dictionary, as
  ``pds:start_date_time`` is, holds what that label gives the attribute;
- each product whose label names a supplemental file has the records the file calls
  for and no others: one in the bodies table for each body the file gives geometry, as
  :func:`~spindoctor.cli.pds4.targets.has_geometry` decides, and one in the rings table
  when the file gives ring statistics.

The statistics a record holds are not compared with the supplemental file's: they are
written in the summary pass's own formats, which its tests pin, and the statistic column
check (:mod:`~spindoctor.cli.pds4.check.statistic_columns`) holds the format each column
declares to the configuration.  The layout and the columns are those of
:mod:`spindoctor.cli.pds4.global_index`, which every dataset's bundle shares.
"""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from lxml import etree

from spindoctor.cli.pds4.check.elements import (
    child,
    child_integer,
    child_text,
    children,
    element_path,
)
from spindoctor.cli.pds4.check.findings import (
    BUNDLE_DIRECTORY,
    CheckName,
    Finding,
    RecordFindings,
)
from spindoctor.cli.pds4.check.tables import RECORD_DELIMITERS
from spindoctor.cli.pds4.global_index import (
    BODIES_INDEX,
    BODY_COLUMN,
    FILE_COLUMN,
    LID_COLUMN,
    MISCELLANEOUS_COLLECTION,
    RINGS_INDEX,
)
from spindoctor.cli.pds4.targets import has_geometry
from spindoctor.dataset.dataset import pds4_label_name

INDEX_LABELS = {
    f'{MISCELLANEOUS_COLLECTION}/{pds4_label_name(f"{name}.tab")}': name
    for name in (BODIES_INDEX, RINGS_INDEX)
}
"""Each global index table's name, by its label's path in the bundle."""

PDS_PREFIX = 'pds:'
"""The prefix of a column named for an attribute of the PDS4 common dictionary."""

SEPARATOR = b','
"""What lies between two fields of an index record, and two names of its header line."""

_Called = Counter[tuple[str, str]]
"""How many records each product calls for, by its logical identifier and a body's name,
the name empty in the rings table."""


@dataclass(frozen=True)
class _IndexField:
    """One field of an index table's records, as its label places it.

    Attributes:
        name: The field's name.
        location: The path of its element in the label.
        start: Where it begins in a record, counted from 0.
        stop: Where it ends, exclusive.
    """

    name: str
    location: str
    start: int
    stop: int


@dataclass(frozen=True)
class _IndexTable:
    """An index table read through its label, as far as its layout and its records go.

    Attributes:
        header_location: The path of the label's ``Header``.
        header: The bytes the ``Header`` describes, the header line.
        table_location: The path of the label's ``Table_Character``.
        fields: Its fields, in the order the label gives them.
        usable: The bytes of a record before its delimiter.
        delimiter: The record delimiter's bytes.
        records: Each whole record the file holds where the label places the records,
            its delimiter included.
    """

    header_location: str
    header: bytes
    table_location: str
    fields: tuple[_IndexField, ...]
    usable: int
    delimiter: bytes
    records: tuple[bytes, ...]


def _read_index_table(label: Path, document: Any) -> _IndexTable | None:
    """Read an index table through its label.

    Parameters:
        label: The label's file, beside which the table is.
        document: The label, parsed by lxml.

    Returns:
        The table, or None when the label describes no ``Header`` and ``Table_Character``
        of a file beside it, or leaves out a number the reading needs; the table reader
        and the integrity check report those.
    """
    for area in document.getroot():
        if not isinstance(area.tag, str):
            continue
        header = child(area, 'Header')
        table = child(area, 'Table_Character')
        name = child_text(area, 'File', 'file_name')
        if header is None or table is None or name is None:
            continue
        record = child(table, 'Record_Character')
        delimiter = RECORD_DELIMITERS.get(child_text(table, 'record_delimiter') or '')
        header_offset = child_integer(header, 'offset')
        header_length = child_integer(header, 'object_length')
        table_offset = child_integer(table, 'offset')
        records = child_integer(table, 'records')
        length = None if record is None else child_integer(record, 'record_length')
        path = label.parent / name
        if (
            record is None
            or delimiter is None
            or header_offset is None
            or header_length is None
            or table_offset is None
            or records is None
            or length is None
            or length < 1
            or not path.is_file()
        ):
            return None
        fields: list[_IndexField] = []
        for element in children(record, 'Field_Character'):
            location = child_integer(element, 'field_location')
            field_length = child_integer(element, 'field_length')
            if location is None or field_length is None:
                return None
            fields.append(
                _IndexField(
                    name=child_text(element, 'name') or '',
                    location=element_path(element),
                    start=location - 1,
                    stop=location - 1 + field_length,
                )
            )
        data = path.read_bytes()
        whole = max(0, min(records, (len(data) - table_offset) // length))
        return _IndexTable(
            header_location=element_path(header),
            header=data[header_offset : header_offset + header_length],
            table_location=element_path(table),
            fields=tuple(fields),
            usable=length - len(delimiter),
            delimiter=delimiter,
            records=tuple(
                data[table_offset + index * length : table_offset + (index + 1) * length]
                for index in range(whole)
            ),
        )
    return None


def _cell(record: bytes, field: _IndexField) -> str:
    """Return a field's value in one record, its padding stripped.

    Parameters:
        record: The record.
        field: The field.

    Returns:
        The value.
    """
    return record[field.start : field.stop].decode('ascii', errors='replace').strip(' ')


def _header_findings(found: RecordFindings, table: _IndexTable) -> None:
    """Hold an index table's header line to the names of its fields.

    Parameters:
        found: Where the findings go.
        table: The table.
    """
    if not table.header.endswith(table.delimiter):
        found.add(table.header_location, 'the header line does not end in the record delimiter')
    given = table.header.removesuffix(table.delimiter).decode('ascii', errors='replace')
    headings = given.split(SEPARATOR.decode('ascii'))
    names = [field.name for field in table.fields]
    if len(headings) != len(names):
        found.add(
            table.header_location,
            f'the header line names {len(headings)} field(s), but the label describes {len(names)}',
        )
        return
    for number, (heading, name) in enumerate(zip(headings, names, strict=True), start=1):
        if heading != name:
            found.add(
                table.header_location,
                f'the header line names field {number} {heading!r}, but the label names it '
                f'{name!r}',
            )


def index_layout_findings(file: str, label: Path, document: Any) -> list[Finding]:
    """Hold a global index table to the layout the summary pass writes it in.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        label: The label's file, beside which the table is.
        document: The label, parsed by lxml.

    Returns:
        One finding for a header line that does not name the fields, each field that a
        comma alone does not separate from the one before it, a last field that does not
        end where the record delimiter begins, and each field that a record does not
        follow with a comma; none for a label of another product.
    """
    if file not in INDEX_LABELS:
        return []
    table = _read_index_table(label, document)
    if table is None:
        return []
    found = RecordFindings(file, CheckName.TABLE)
    _header_findings(found, table)
    placed = sorted(table.fields, key=lambda field: field.start)
    separated: list[_IndexField] = []
    for before, after in pairwise(placed):
        gap = after.start - before.stop
        if gap == 1:
            separated.append(before)
        elif gap >= 0:
            # A field overlapping the one before it is the table reader's finding.
            found.add(
                after.location,
                f'begins {gap} byte(s) after the field before it, {before.name}, where a '
                'comma alone lies between two fields',
            )
    if len(placed) > 0 and placed[-1].stop < table.usable:
        found.add(
            placed[-1].location,
            f'ends {table.usable - placed[-1].stop} byte(s) before the record delimiter, '
            'where the last field ends at it',
        )
    for number, record in enumerate(table.records, start=1):
        for field in separated:
            byte = record[field.stop : field.stop + 1]
            if byte != SEPARATOR:
                found.add_recurring(
                    field.location,
                    'separator',
                    f'record {number} holds {byte.decode("ascii", errors="replace")!r} after '
                    'the field, not a comma',
                )
    return found.findings()


def _called_for(
    bundle_dir: Path, labels: Mapping[str, Any]
) -> tuple[dict[str, _Called], list[Finding]]:
    """Return the records of each index table the tree's supplemental files call for.

    Parameters:
        bundle_dir: The bundle's directory.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, parsed by lxml.

    Returns:
        The records each table's name calls for, from each product whose label names a
        supplemental file beside it; and one finding for each such file that is not a
        JSON document, whose records cannot be known.
    """
    bodies: _Called = Counter()
    rings: _Called = Counter()
    findings: list[Finding] = []
    for file, document in labels.items():
        root = document.getroot()
        lid = child_text(root, 'Identification_Area', 'logical_identifier')
        name = child_text(root, 'File_Area_Observational_Supplemental', 'File', 'file_name')
        path = (bundle_dir / file).parent / (name or '')
        if lid is None or name is None or not path.is_file():
            continue
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
        except (UnicodeDecodeError, ValueError) as exc:
            findings.append(
                Finding(
                    (PurePosixPath(file).parent / name).as_posix(),
                    CheckName.INTEGRITY,
                    '',
                    f'is not a JSON document, so the index records it calls for are not '
                    f'known: {exc}',
                )
            )
            continue
        backplanes = metadata.get('backplanes', {}) if isinstance(metadata, dict) else {}
        for body, entry in backplanes.get('bodies', {}).items():
            if has_geometry(entry):
                bodies[(lid, body)] += 1
        if backplanes.get('rings', {}).get('backplanes'):
            rings[(lid, '')] += 1
    return {BODIES_INDEX: bodies, RINGS_INDEX: rings}, findings


def _table_record_findings(
    file: str,
    table: _IndexTable,
    labels: Mapping[str, Any],
    products: Mapping[str, Mapping[str, str]],
    called: _Called,
) -> list[Finding]:
    """Hold one index table's records to the tree.

    Parameters:
        file: The table's label, relative to the bundle's directory.
        table: The table.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, parsed by lxml.
        products: The label declaring each version of each product of the tree, by its
            logical identifier and then its version.
        called: The records the tree's supplemental files call for.

    Returns:
        The findings.
    """
    fields = {field.name: field for field in table.fields}
    lid_field = fields.get(LID_COLUMN.name)
    if lid_field is None:
        return []
    body_field = fields.get(BODY_COLUMN.name)
    file_field = fields.get(FILE_COLUMN.name)
    copied = [
        field
        for field in table.fields
        if field.name.startswith(PDS_PREFIX) and field.name != LID_COLUMN.name
    ]
    found = RecordFindings(file, CheckName.INTEGRITY)
    held: _Called = Counter()
    for number, record in enumerate(table.records, start=1):
        lid = _cell(record, lid_field)
        if lid not in products:
            message = f'record {number} names {lid}, which no label of the tree declares'
            found.add_recurring(lid_field.location, 'undeclared', message)
            continue
        held[(lid, '' if body_field is None else _cell(record, body_field))] += 1
        declaring = sorted(products[lid].values())
        if file_field is not None and _cell(record, file_field) not in declaring:
            message = (
                f'record {number} gives {_cell(record, file_field)!r}, but the label '
                f'declaring {lid} is {declaring[0]}'
            )
            found.add_recurring(file_field.location, 'file_spec', message)
        root = labels[declaring[0]].getroot()
        for field in copied:
            attribute = field.name.removeprefix(PDS_PREFIX)
            element = next(root.iter(f'{{{etree.QName(root).namespace}}}{attribute}'), None)
            value = _cell(record, field)
            if element is None:
                message = (
                    f'record {number} gives {value!r}, but {declaring[0]} states no {attribute}'
                )
            elif value != str(element.text or '').strip():
                stated = str(element.text or '').strip()
                message = f'record {number} gives {value!r}, but {declaring[0]} states {stated!r}'
            else:
                continue
            found.add_recurring(field.location, 'copied', message)
    for key in sorted(set(called) | set(held)):
        if held[key] != called[key]:
            lid, body = key
            what = lid if body_field is None else f'{body} in {lid}'
            message = (
                f'holds {held[key]} record(s) for {what}, where its supplemental file calls '
                f'for {called[key]}'
            )
            found.add_recurring(table.table_location, 'records', message)
    return found.findings()


def index_row_findings(
    bundle_dir: Path, labels: Mapping[str, Any], products: Mapping[str, Mapping[str, str]]
) -> list[Finding]:
    """Hold each global index table's records to the tree.

    Parameters:
        bundle_dir: The bundle's directory.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, parsed by lxml.
        products: The label declaring each version of each product of the tree, by its
            logical identifier and then its version.

    Returns:
        One finding for each record naming a product no label declares, each giving a
        ``file_spec`` or a PDS4 attribute otherwise than the product's label does, and
        each product holding more or fewer records of a table than its supplemental file
        calls for; one for a table the supplemental files call for records of that the
        tree does not hold; and one for each supplemental file that is not JSON.
    """
    called, findings = _called_for(bundle_dir, labels)
    for file, name in INDEX_LABELS.items():
        if file not in labels:
            wanted = sum(called[name].values())
            if wanted > 0 and not (bundle_dir / file).exists():
                findings.append(
                    Finding(
                        BUNDLE_DIRECTORY,
                        CheckName.INTEGRITY,
                        '',
                        f'the tree holds no {file}, but its supplemental files call for '
                        f'{wanted} record(s) of {name}',
                    )
                )
            continue
        table = _read_index_table(bundle_dir / file, labels[file])
        if table is not None:
            findings.extend(_table_record_findings(file, table, labels, products, called[name]))
    return findings
