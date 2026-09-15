"""Each table of a bundle read through its label alone.

Neither the XML schema nor the Schematron reads a table, so neither can see whether a
table agrees with the label that describes it.  This module reads each table as a user of
the bundle would, from what its label says, and reports where the two disagree.

The objects of a file area holding a table are held to tile its file: the first begins at
byte 0, each begins where the one before it ends, and the last ends where the file does.
A ``Header`` ends where its ``object_length`` says.  A ``Table_Character`` is its
``records`` records of ``record_length`` bytes, each ending in its record delimiter.  A
``Table_Delimited`` or an ``Inventory`` is its ``records`` records, each ending in its
record delimiter, from its offset to the object after it or to the end of the file.  A
delimiter is matched byte for byte: a delimited record holding a carriage return or a
line feed that is not its delimiter is a finding, so that records ending in a carriage
return and a line feed are not read as records ending in a line feed.

In each record every field is cut out where its label places it -- by ``field_location``
and ``field_length`` in a character table, and by the field delimiter in a delimited one
-- and its value is held to its ``data_type``: the simple type of that name in the PDS4
common dictionary the label declares.  A delimited field's value is also held to its
``maximum_field_length``, and a value equal as a number to the field's
``missing_constant`` is held to be spelled as the constant is.  In a record holding no
group, each field's ``field_number`` is its position among the record's fields.  A
problem that recurs across records is reported once, at the first record showing it,
with a count of the rest.

This module reads ``Table_Character``, ``Table_Delimited`` and ``Inventory`` objects,
with the ``Header`` objects beside them.  It leaves to the XML schema and to the PDS
``validate`` tool the fields of a group (``Group_Field_Character`` and
``Group_Field_Delimited``), a ``Table_Binary``, and every object of a file area that also
holds an object of another class, whose extent it cannot know.
"""

import csv
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import xmlschema
from lxml import etree

from spindoctor.cli.pds4.check.elements import (
    child,
    child_integer,
    child_text,
    children,
    element_path,
    local_name,
)
from spindoctor.cli.pds4.check.findings import CheckName, Finding, RecordFindings

TABLE_CLASSES = frozenset({'Table_Character', 'Table_Delimited', 'Inventory'})
"""The classes of table this module reads."""

_TILED_CLASSES = TABLE_CLASSES | {'Header'}
"""The classes of object whose extent this module knows, which it holds to tile a file."""

RECORD_DELIMITERS = {'Carriage-Return Line-Feed': b'\r\n', 'Line-Feed': b'\n'}
"""The bytes each ``record_delimiter`` a PDS4 table can declare stands for."""

FIELD_DELIMITERS = {'Comma': ',', 'Horizontal Tab': '\t', 'Semicolon': ';', 'Vertical Bar': '|'}
"""The character each ``field_delimiter`` of a PDS4 delimited table stands for."""


@dataclass(frozen=True)
class _Field:
    """One field of a table, as its label describes it.

    Attributes:
        location: The path of the field's element in the label.
        name: The field's name.
        data_type: Its data type, as the label writes it.
        type_name: The data type's name in the namespace of the field's element.
        missing_constant: Its declared missing constant, or None.
        start: Where it begins in a record of a character table, counted from 0.
        stop: Where it ends in a record of a character table, exclusive.
        maximum_length: The longest value a delimited field may hold, or None.
    """

    location: str
    name: str
    data_type: str
    type_name: str
    missing_constant: str | None
    start: int = 0
    stop: int = 0
    maximum_length: int | None = None


def _field(element: Any) -> _Field:
    """Return what a field's label says of it, less where it lies.

    Parameters:
        element: The ``Field_Character`` or ``Field_Delimited`` element.

    Returns:
        The field.
    """
    data_type = child_text(element, 'data_type') or ''
    namespace = etree.QName(element).namespace
    return _Field(
        location=element_path(element),
        name=child_text(element, 'name') or '',
        data_type=data_type,
        type_name=data_type if namespace is None else f'{{{namespace}}}{data_type}',
        missing_constant=child_text(element, 'Special_Constants', 'missing_constant'),
    )


def _same_number(value: str, constant: str) -> bool:
    """Say whether two values are the same number.

    Parameters:
        value: A field's value.
        constant: The field's missing constant.

    Returns:
        True when both are numbers and equal.
    """
    try:
        return float(value) == float(constant)
    except ValueError:
        return False


def _hold_value(
    found: RecordFindings,
    field: _Field,
    number: int,
    raw: bytes,
    schema: xmlschema.XMLSchema | None,
) -> None:
    """Hold one value of one record to what its field's label says.

    Parameters:
        found: Where the findings go.
        field: The field.
        number: The record's number, counted from 1.
        raw: The value's bytes, as the record holds them.
        schema: The XML schemas the label declares, whose simple types the data types
            are; None when they could not be built, when no value is held to its type.
    """
    try:
        value = raw.decode('ascii')
    except UnicodeDecodeError:
        found.add_recurring(
            field.location, 'ascii', f'record {number} holds a byte outside 7-bit ASCII'
        )
        return
    if field.maximum_length is not None and len(value) > field.maximum_length:
        found.add_recurring(
            field.location,
            'length',
            f'record {number} holds {value!r}, longer than its maximum_field_length '
            f'{field.maximum_length}',
        )
    text = value.strip(' ')
    xsd_type = None if schema is None else schema.maps.types.get(field.type_name)
    if xsd_type is not None and not xsd_type.is_valid(text):
        found.add_recurring(
            field.location, 'type', f'record {number} holds {text!r}, not {field.data_type}'
        )
    constant = field.missing_constant
    if constant is not None and text != constant and _same_number(text, constant):
        found.add_recurring(
            field.location,
            'missing',
            f'record {number} holds {text!r}, the missing constant {constant!r} spelled otherwise',
        )


def _count(found: RecordFindings, record: Any, name: str, actual: int) -> None:
    """Hold a record's ``fields`` or ``groups`` to the number of them it describes.

    Parameters:
        found: Where the findings go.
        record: The ``Record_Character`` or ``Record_Delimited`` element.
        name: ``fields`` or ``groups``.
        actual: How many fields or groups the record describes.
    """
    stated = child_integer(record, name)
    if stated is not None and stated != actual:
        found.add(element_path(record), f'{name} is {stated}, but the record describes {actual}')


def _field_numbers(found: RecordFindings, record: Any, elements: list[Any]) -> None:
    """Hold each field's ``field_number`` to its position among its record's fields.

    A record holding a group is left alone, as the rest of a group is.

    Parameters:
        found: Where the findings go.
        record: The ``Record_Character`` or ``Record_Delimited`` element.
        elements: The record's fields, in the order its label gives them.
    """
    if any(local_name(part).startswith('Group_Field_') for part in record):
        return
    for position, element in enumerate(elements, start=1):
        number = child_integer(element, 'field_number')
        if number is not None and number != position:
            found.add(
                element_path(element),
                f'field_number is {number}, but the field is number {position} of its record',
            )


def _character_fields(found: RecordFindings, record: Any, usable: int) -> list[_Field]:
    """Place the fields of a character table's record, holding each within the record.

    Parameters:
        found: Where the findings go.
        record: The ``Record_Character`` element.
        usable: The bytes of a record before its delimiter.

    Returns:
        Each field lying within the record, in the order they lie in it.
    """
    placed: list[_Field] = []
    for element in children(record, 'Field_Character'):
        location = child_integer(element, 'field_location')
        length = child_integer(element, 'field_length')
        if location is None or length is None:
            continue
        field = _field(element)
        start = location - 1
        stop = start + length
        if start < 0 or stop > usable:
            found.add(
                field.location,
                f'field_location {location} and field_length {length} do not lie within '
                f'the {usable} bytes of a record before its delimiter',
            )
            continue
        placed.append(replace(field, start=start, stop=stop))
    placed.sort(key=lambda field: field.start)
    for before, after in pairwise(placed):
        if after.start < before.stop:
            found.add(after.location, f'overlaps the field before it, {before.name}')
    return placed


def _read_character_table(
    found: RecordFindings,
    table: Any,
    extent: bytes,
    between: str,
    schema: xmlschema.XMLSchema | None,
) -> None:
    """Read a ``Table_Character`` through its label.

    Parameters:
        found: Where the findings go.
        table: The ``Table_Character`` element.
        extent: The bytes from its offset to the next object or the end of the file.
        between: Where those bytes lie, for a finding to say.
        schema: The XML schemas the label declares.
    """
    location = element_path(table)
    records = child_integer(table, 'records')
    record = child(table, 'Record_Character')
    length = None if record is None else child_integer(record, 'record_length')
    if records is None or record is None or length is None or length < 1:
        return
    if records * length != len(extent):
        found.add(
            location,
            f'{records} record(s) of {length} bytes are {records * length} bytes, but '
            f'{len(extent)} lie {between}',
        )
    elements = children(record, 'Field_Character')
    _count(found, record, 'fields', len(elements))
    _count(found, record, 'groups', len(children(record, 'Group_Field_Character')))
    _field_numbers(found, record, elements)
    delimiter = RECORD_DELIMITERS.get(child_text(table, 'record_delimiter') or '', b'')
    fields = _character_fields(found, record, length - len(delimiter))
    for index in range(min(records, len(extent) // length)):
        raw = extent[index * length : (index + 1) * length]
        if not raw.endswith(delimiter):
            found.add_recurring(
                location, 'delimiter', f'record {index + 1} does not end in its record delimiter'
            )
            continue
        for field in fields:
            _hold_value(found, field, index + 1, raw[field.start : field.stop], schema)


def _read_delimited_table(
    found: RecordFindings,
    table: Any,
    extent: bytes,
    between: str,
    schema: xmlschema.XMLSchema | None,
) -> None:
    """Read a ``Table_Delimited`` or an ``Inventory`` through its label.

    Parameters:
        found: Where the findings go.
        table: The table's element.
        extent: The bytes from its offset to the next object or the end of the file.
        between: Where those bytes lie, for a finding to say.
        schema: The XML schemas the label declares.
    """
    location = element_path(table)
    records = child_integer(table, 'records')
    delimiter = RECORD_DELIMITERS.get(child_text(table, 'record_delimiter') or '')
    separator = FIELD_DELIMITERS.get(child_text(table, 'field_delimiter') or '')
    record = child(table, 'Record_Delimited')
    if records is None or delimiter is None or separator is None or record is None:
        return
    if not extent.endswith(delimiter):
        found.add(location, f'the bytes {between} do not end in its record delimiter')
    rows = extent.split(delimiter)
    if rows[-1] == b'':
        rows.pop()
    if len(rows) != records:
        found.add(location, f'records is {records}, but {len(rows)} record(s) lie {between}')
    elements = children(record, 'Field_Delimited')
    _count(found, record, 'fields', len(elements))
    _count(found, record, 'groups', len(children(record, 'Group_Field_Delimited')))
    _field_numbers(found, record, elements)
    fields = [
        replace(_field(element), maximum_length=child_integer(element, 'maximum_field_length'))
        for element in elements
    ]
    for number, row in enumerate(rows, start=1):
        if b'\r' in row or b'\n' in row:
            # Checked before the record is parsed, since the parser takes a carriage
            # return or a line feed for the end of a line and drops it.
            message = (
                f'record {number} holds a carriage return or a line feed that is not its '
                'record delimiter'
            )
            found.add_recurring(location, 'line end', message)
            continue
        try:
            text = row.decode('ascii')
        except UnicodeDecodeError:
            message = f'record {number} holds a byte outside 7-bit ASCII'
            found.add_recurring(location, 'ascii', message)
            continue
        values = next(csv.reader([text], delimiter=separator))
        if len(values) != len(fields):
            message = f'record {number} holds {len(values)} field(s), not {len(fields)}'
            found.add_recurring(location, 'fields', message)
            continue
        for field, value in zip(fields, values, strict=True):
            _hold_value(found, field, number, value.encode('ascii'), schema)


def _read_file_area(
    found: RecordFindings, data: bytes, objects: list[Any], schema: xmlschema.XMLSchema | None
) -> None:
    """Read the objects of one file area holding a table, and hold them to tile its file.

    Parameters:
        found: Where the findings go.
        data: The file's bytes.
        objects: The file area's objects, every one a ``Header`` or a table.
        schema: The XML schemas the label declares.
    """
    placed: list[tuple[int, Any]] = []
    for element in objects:
        offset = child_integer(element, 'offset')
        if offset is None:
            return
        placed.append((offset, element))
    placed.sort(key=lambda pair: pair[0])
    if placed[0][0] > 0:
        found.add(
            element_path(placed[0][1]),
            f'the file begins with {placed[0][0]} byte(s) no object describes',
        )
    for index, (offset, element) in enumerate(placed):
        last = index + 1 == len(placed)
        end = len(data) if last else placed[index + 1][0]
        between = f'between its offset {offset} and ' + (
            'the end of the file' if last else 'the next object'
        )
        name = local_name(element)
        if name == 'Header':
            length = child_integer(element, 'object_length')
            if length is not None and length != end - offset:
                found.add(
                    element_path(element),
                    f'object_length is {length}, but {end - offset} bytes lie {between}',
                )
        elif name == 'Table_Character':
            _read_character_table(found, element, data[offset:end], between, schema)
        else:
            _read_delimited_table(found, element, data[offset:end], between, schema)


def table_findings(
    file: str, label: Path, document: Any, schema: xmlschema.XMLSchema | None
) -> list[Finding]:
    """Read every table a label describes through the label alone.

    A file the label names that is not beside it is not read; the integrity check reports
    it.

    Parameters:
        file: The label's path relative to the bundle's directory, which findings name.
        label: The label's file, beside which its tables are.
        document: The label, parsed by lxml.
        schema: The XML schemas the label declares, as
            :func:`~spindoctor.cli.pds4.check.schemas.label_schema` built them, or
            None when they could not be built.

    Returns:
        One finding for each way a table departs from its label.
    """
    found = RecordFindings(file, CheckName.TABLE)
    for area in document.getroot():
        if not isinstance(area.tag, str):
            continue
        objects = [obj for obj in area if isinstance(obj.tag, str) and local_name(obj) != 'File']
        if not any(local_name(obj) in TABLE_CLASSES for obj in objects):
            continue
        if not all(local_name(obj) in _TILED_CLASSES for obj in objects):
            continue
        name = child_text(area, 'File', 'file_name')
        if name is None or not (label.parent / name).is_file():
            continue
        _read_file_area(found, (label.parent / name).read_bytes(), objects, schema)
    return found.findings()
