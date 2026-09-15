"""A bundle tree checked as a whole: the files its labels name, and what they declare.

The checks, over every label of the tree:

- every ``file_name`` a label holds names a file beside the label -- a name holding a
  directory is a finding of its own -- names it once, and states its ``file_size`` and
  ``md5_checksum`` as the file has them;
- every file of the tree is a label or is named by exactly one label;
- no label holds a leftover template marker, ``[[[``, which a template leaves where it
  could not fill in a value;
- no element of a label is empty -- holding no child element and no text -- unless it
  carries ``xsi:nil``;
- no two labels declare the same logical identifier;
- every ``lid_reference`` and ``lidvid_reference`` to a product of the bundle itself, one
  whose logical identifier extends the one the bundle label declares, names a product a
  label of the tree declares, and a ``lidvid_reference`` that product's version; one
  that does not is a warning, as the PDS ``validate`` tool's ``reference_not_found`` is.

A reference to a product outside the bundle is not checked here.  The PDS ``validate``
tool checks the references a label makes to context products against the products
registered with the PDS; neither it nor this check holds to the registry the versions a
collection's inventory names for products outside the bundle.
"""

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

from spindoctor.cli.pds4.check.elements import child, child_text, element_path, local_name
from spindoctor.cli.pds4.check.findings import BUNDLE_DIRECTORY, CheckName, Finding, Severity

LABEL_SUFFIX = '.lblx'
"""The suffix of a PDS4 label in a bundle."""

MARKER = b'[[['
"""What a template leaves in a label where it could not fill in a value."""

XSI_NIL = '{http://www.w3.org/2001/XMLSchema-instance}nil'
"""The attribute saying an element is empty on purpose."""

NIL_TRUE = frozenset({'true', '1'})
"""The ``xs:boolean`` spellings of true, which ``xsi:nil`` takes, white space trimmed."""


def _relative(path: Path, bundle_dir: Path) -> str:
    """Return a path relative to the bundle's directory, in POSIX form.

    Parameters:
        path: A path in or below the bundle's directory.
        bundle_dir: The bundle's directory.

    Returns:
        The relative path.
    """
    return PurePath(os.path.relpath(path, bundle_dir)).as_posix()


def _size_and_checksum(file: str, described: Any, target: Path, name: str) -> list[Finding]:
    """Hold the size and the checksum a label states for a file to the file's own.

    Parameters:
        file: The label's path relative to the bundle's directory.
        described: The element describing the file, a ``File`` or a ``Document_File``.
        target: The file.
        name: The file's name, as the label gives it.

    Returns:
        One finding for a ``file_size`` other than the file's size in bytes, and one for
        an ``md5_checksum`` other than its MD5 checksum.  A value that is not a number
        or not hexadecimal is left to the XML schema.
    """
    findings: list[Finding] = []
    size = child(described, 'file_size')
    if size is not None:
        stated = str(size.text or '').strip()
        actual = target.stat().st_size
        if stated.isdigit() and int(stated) != actual:
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    element_path(size),
                    f'file_size is {stated}, but {name} is {actual} bytes',
                )
            )
    checksum = child(described, 'md5_checksum')
    if checksum is not None:
        stated = str(checksum.text or '').strip()
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'md5').hexdigest()
        if stated.lower() != digest:
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    element_path(checksum),
                    f'md5_checksum is {stated}, but the MD5 checksum of {name} is {digest}',
                )
            )
    return findings


def _named_files(
    file: str, root: Any, bundle_dir: Path, named: dict[str, list[str]]
) -> list[Finding]:
    """Look for the file each ``file_name`` of a label names, beside the label.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.
        bundle_dir: The bundle's directory.
        named: The labels naming each file found so far, by the file's path relative to
            the bundle's directory; each file this label names is added, once.

    Returns:
        One finding for each ``file_name`` holding a directory, each naming a file that
        is not beside the label, each naming a file the label has named already, and
        each file whose size or checksum the label states otherwise than it is.
    """
    findings: list[Finding] = []
    directory = (bundle_dir / file).parent
    own: set[str] = set()
    for element in root.iter('{*}file_name'):
        name = str(element.text or '').strip()
        location = element_path(element)
        if '/' in name or '\\' in name:
            message = f'names {name!r}, which holds a directory, where a file beside it is named'
            findings.append(Finding(file, CheckName.INTEGRITY, location, message))
            continue
        target = directory / name
        if name == '' or not target.is_file():
            message = f'names {name!r}, which is not beside it'
            findings.append(Finding(file, CheckName.INTEGRITY, location, message))
            continue
        relative = _relative(target, bundle_dir)
        if relative in own:
            message = f'names {name!r} a second time'
            findings.append(Finding(file, CheckName.INTEGRITY, location, message))
            continue
        own.add(relative)
        named.setdefault(relative, []).append(file)
        findings.extend(_size_and_checksum(file, element.getparent(), target, name))
    return findings


def _marker(file: str, label: Path) -> list[Finding]:
    """Look for a marker a template left in a label where it could not fill in a value.

    Parameters:
        file: The label's path relative to the bundle's directory.
        label: The label's file.

    Returns:
        One finding, at the line of the first marker, or none.
    """
    data = label.read_bytes()
    index = data.find(MARKER)
    if index < 0:
        return []
    line = data.count(b'\n', 0, index) + 1
    return [
        Finding(
            file,
            CheckName.INTEGRITY,
            f'line {line}',
            'holds a [[[ marker, which a template leaves where it could not fill in a value',
        )
    ]


def _empty_elements(file: str, root: Any) -> list[Finding]:
    """Look for the elements of a label that hold nothing and do not carry ``xsi:nil``.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.

    Returns:
        One finding for each element holding no child element and no text but white
        space, a comment in it notwithstanding, whose ``xsi:nil``, if it carries one, is
        not true.
    """
    findings: list[Finding] = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        if any(isinstance(part.tag, str) for part in element):
            continue
        text = str(element.text or '') + ''.join(str(part.tail or '') for part in element)
        if text.strip() != '' or str(element.get(XSI_NIL, '')).strip() in NIL_TRUE:
            continue
        findings.append(
            Finding(
                file,
                CheckName.INTEGRITY,
                element_path(element),
                'is empty: it holds no element and no text',
            )
        )
    return findings


def _unnamed_files(bundle_dir: Path, named: Mapping[str, list[str]]) -> list[Finding]:
    """Look for the files of the tree that are not labels and not named by one label.

    Parameters:
        bundle_dir: The bundle's directory.
        named: The labels naming each file, by the file's path relative to the bundle's
            directory.

    Returns:
        One finding for each file no label names and one for each that several name.
    """
    findings: list[Finding] = []
    for path in sorted(bundle_dir.rglob('*')):
        if not path.is_file() or path.suffix == LABEL_SUFFIX:
            continue
        file = _relative(path, bundle_dir)
        naming = sorted(named.get(file, []))
        if len(naming) == 0:
            findings.append(Finding(file, CheckName.INTEGRITY, '', 'no label names it'))
        elif len(naming) > 1:
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    '',
                    f'{len(naming)} labels name it: {", ".join(naming)}',
                )
            )
    return findings


def _declared_products(
    labels: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], list[Finding]]:
    """Return the products the labels of a tree declare, and each one declared twice.

    Parameters:
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, in the order of the paths.

    Returns:
        The path of the label declaring each version of each product, by its logical
        identifier and then its version, the first such label where two declare one;
        and one finding for each label declaring a logical identifier a label before it
        declares, as the PDS ``validate`` tool's ``duplicate_identifier`` is.
    """
    products: dict[str, dict[str, str]] = {}
    first: dict[str, str] = {}
    findings: list[Finding] = []
    for file, document in labels.items():
        area = child(document.getroot(), 'Identification_Area')
        if area is None:
            continue
        lid = child_text(area, 'logical_identifier')
        version = child_text(area, 'version_id')
        if lid is None or version is None:
            continue
        if lid in first:
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    element_path(child(area, 'logical_identifier')),
                    f'declares {lid}, which {first[lid]} declares too',
                )
            )
        else:
            first[lid] = file
        products.setdefault(lid, {}).setdefault(version, file)
    return products, findings


def _own_references(
    file: str, root: Any, bundle_lid: str, products: Mapping[str, Mapping[str, str]]
) -> list[Finding]:
    """Resolve each reference of a label to a product of the bundle itself.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.
        bundle_lid: The logical identifier the bundle label declares.
        products: The label declaring each version of each product of the tree, by its
            logical identifier and then its version.

    Returns:
        One warning for each reference to a product of the bundle that no label of the
        tree declares, and one for each naming a version the tree does not hold.
    """
    findings: list[Finding] = []
    for element in root.iter('{*}lid_reference', '{*}lidvid_reference'):
        reference = str(element.text or '').strip()
        lid, _, version = reference.partition('::')
        if lid != bundle_lid and not lid.startswith(f'{bundle_lid}:'):
            continue
        if lid not in products:
            message = f'refers to {reference}, which no label of the tree declares'
        elif version != '' and version not in products[lid]:
            held = ', '.join(sorted(products[lid]))
            message = f'refers to {reference}, but the tree holds {lid} at version {held}'
        else:
            continue
        findings.append(
            Finding(file, CheckName.INTEGRITY, element_path(element), message, Severity.WARNING)
        )
    return findings


def integrity_findings(bundle_dir: Path, labels: Mapping[str, Any]) -> list[Finding]:
    """Check a bundle tree as a whole.

    Parameters:
        bundle_dir: The bundle's directory.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, in the order of the paths, parsed by lxml.

    Returns:
        One finding for each way the tree departs from what this module's checks hold it
        to; and one for a tree with no bundle label at its top, over which no reference
        to a product of the bundle can be resolved.
    """
    findings: list[Finding] = []
    named: dict[str, list[str]] = {}
    products, duplicates = _declared_products(labels)
    findings.extend(duplicates)
    bundle_lid: str | None = None
    for file, document in labels.items():
        root = document.getroot()
        findings.extend(_named_files(file, root, bundle_dir, named))
        findings.extend(_marker(file, bundle_dir / file))
        findings.extend(_empty_elements(file, root))
        if local_name(root) == 'Product_Bundle' and '/' not in file:
            bundle_lid = child_text(root, 'Identification_Area', 'logical_identifier')
    findings.extend(_unnamed_files(bundle_dir, named))
    if bundle_lid is None:
        findings.append(
            Finding(
                BUNDLE_DIRECTORY,
                CheckName.INTEGRITY,
                '',
                'the tree holds no bundle label at its top, so no reference to a product '
                'of the bundle can be resolved',
            )
        )
        return findings
    for file, document in labels.items():
        findings.extend(_own_references(file, document.getroot(), bundle_lid, products))
    return findings
