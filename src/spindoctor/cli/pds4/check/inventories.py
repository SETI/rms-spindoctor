"""Each collection's inventory held to the tree: what it lists, and what it should.

A collection's label describes its inventory, a delimited table of two fields: a member's
status, ``P`` for a primary member and ``S`` for a secondary one, and its LIDVID or LID.
The table reader (:mod:`~spindoctor.cli.pds4.check.tables`) holds the inventory to its
label; this module holds it to the tree, as the PDS ``validate`` tool does:

- each primary member names a product a label of the tree declares, and a LIDVID that
  product's version; one that does not is an error, as ``validate``'s
  ``member_not_found`` is;
- each product whose label lies in the collection's directory, or below it, is listed as
  a primary member once: listed more than once is an error, and not listed a warning, as
  ``validate``'s ``unreferenced_member`` is.

A secondary member names a product outside the bundle, a context product or another
bundle's, and is not checked, nor is the version it names: neither this check nor
``validate`` holds those versions to the products registered with the PDS.
"""

import csv
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from spindoctor.cli.pds4.check.elements import (
    child,
    child_integer,
    child_text,
    element_path,
    local_name,
)
from spindoctor.cli.pds4.check.findings import CheckName, Finding, RecordFindings, Severity
from spindoctor.cli.pds4.check.tables import FIELD_DELIMITERS, RECORD_DELIMITERS

PRIMARY = 'P'
"""The status of a collection's primary member, one of its own products."""

NOT_MEMBERS = frozenset({'Product_Bundle', 'Product_Collection'})
"""The classes of product no collection lists as one of its members."""


def _primary_members(
    bundle_dir: Path, file: str, inventory: Any, name: str
) -> list[tuple[int, str]]:
    """Return each primary member an inventory lists, with the number of its record.

    Parameters:
        bundle_dir: The bundle's directory.
        file: The collection label's path relative to it.
        inventory: The label's ``Inventory`` element.
        name: The inventory's file name, beside the label.

    Returns:
        Each ``P`` record's LIDVID or LID, its white space trimmed, and the record's
        number, counted from 1.  A record the table reader reports as not ASCII or not of
        two fields is passed over; one holding a carriage return or a line feed that is
        not its delimiter, which the table reader reports too, is still read, so that
        its member is resolved.
    """
    path = (bundle_dir / file).parent / name
    delimiter = RECORD_DELIMITERS.get(child_text(inventory, 'record_delimiter') or '')
    separator = FIELD_DELIMITERS.get(child_text(inventory, 'field_delimiter') or '')
    offset = child_integer(inventory, 'offset')
    if delimiter is None or separator is None or offset is None or not path.is_file():
        return []
    members: list[tuple[int, str]] = []
    for number, row in enumerate(path.read_bytes()[offset:].split(delimiter), start=1):
        try:
            text = row.decode('ascii')
        except UnicodeDecodeError:
            continue
        values = next(csv.reader([text], delimiter=separator), [])
        if len(values) == 2 and values[0].strip() == PRIMARY:
            members.append((number, values[1].strip()))
    return members


def _collection_of(file: str, directories: Mapping[PurePosixPath, str]) -> str | None:
    """Return the collection whose directory a label lies in, the nearest above it.

    Parameters:
        file: The label's path relative to the bundle's directory.
        directories: Each collection's label, by the directory it lies in.

    Returns:
        The collection's label, or None when the label lies in no collection's directory.
    """
    for parent in PurePosixPath(file).parents:
        if parent in directories:
            return directories[parent]
    return None


def inventory_findings(
    bundle_dir: Path, labels: Mapping[str, Any], products: Mapping[str, Mapping[str, str]]
) -> list[Finding]:
    """Hold each collection's inventory to the tree.

    Parameters:
        bundle_dir: The bundle's directory.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, parsed by lxml.
        products: The label declaring each version of each product of the tree, by its
            logical identifier and then its version.

    Returns:
        At a collection's label, one error for each primary member no label of the tree
        declares or declares at the version it names, and one for each product listed
        more than once; and at a product's label, one warning for a product in a
        collection's directory that the collection's inventory does not list.
    """
    findings: list[Finding] = []
    listed: dict[str, set[str]] = {}
    directories: dict[PurePosixPath, str] = {}
    for file, document in labels.items():
        root = document.getroot()
        area = child(root, 'File_Area_Inventory')
        if local_name(root) != 'Product_Collection' or area is None:
            continue
        inventory = child(area, 'Inventory')
        name = child_text(area, 'File', 'file_name')
        if inventory is None or name is None:
            continue
        directories[PurePosixPath(file).parent] = file
        location = element_path(inventory)
        found = RecordFindings(file, CheckName.INTEGRITY)
        records: dict[str, list[int]] = {}
        for number, reference in _primary_members(bundle_dir, file, inventory, name):
            lid, _, version = reference.partition('::')
            records.setdefault(lid, []).append(number)
            if lid not in products:
                message = f'record {number} lists {reference}, which no label of the tree declares'
                found.add_recurring(location, 'absent', message)
            elif version != '' and version not in products[lid]:
                held = ', '.join(sorted(products[lid]))
                message = (
                    f'record {number} lists {reference}, but the tree holds {lid} at version {held}'
                )
                found.add_recurring(location, 'version', message)
        for lid, numbers in records.items():
            if len(numbers) > 1:
                where = ', '.join(str(number) for number in numbers)
                found.add(location, f'lists {lid} {len(numbers)} times, in records {where}')
        findings.extend(found.findings())
        listed[file] = set(records)
    for file, document in labels.items():
        root = document.getroot()
        collection = _collection_of(file, directories)
        declared = child_text(root, 'Identification_Area', 'logical_identifier')
        if local_name(root) in NOT_MEMBERS or collection is None or declared is None:
            continue
        if declared not in listed[collection]:
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    '',
                    f'the inventory of {collection}, in whose directory it lies, does not list it',
                    Severity.WARNING,
                )
            )
    return findings
