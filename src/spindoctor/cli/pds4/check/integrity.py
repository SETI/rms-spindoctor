"""A bundle tree checked as a whole: the files its labels name, and what its labels refer to.

Five checks, over every label of the tree:

- every ``file_name`` a label holds names a file beside the label;
- every file of the tree is a label or is named by exactly one label;
- no label holds a ``[[[`` marker, which a template leaves where an expression failed;
- no element of a label is empty -- holding no child element and no text -- unless it
  carries ``xsi:nil``;
- every ``lid_reference`` and ``lidvid_reference`` to a product of the bundle itself, one
  whose logical identifier extends the one the bundle label declares, names a product a
  label of the tree declares, and a ``lidvid_reference`` that product's version.

A reference to a product outside the bundle is left to the PDS ``validate`` tool, which
checks it against the context products registered with the PDS.
"""

import os
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any

from spindoctor.cli.pds4.check.elements import child_text, element_path, local_name
from spindoctor.cli.pds4.check.findings import CheckName, Finding

LABEL_SUFFIX = '.lblx'
"""The suffix of a PDS4 label in a bundle."""

MARKER = b'[[['
"""What a template leaves in a label where an expression failed."""

XSI_NIL = '{http://www.w3.org/2001/XMLSchema-instance}nil'
"""The attribute saying an element is empty on purpose."""

BUNDLE_DIRECTORY = '.'
"""What a finding about the tree as a whole names as its file."""


def _relative(path: Path, bundle_dir: Path) -> str:
    """Return a path relative to the bundle's directory, in POSIX form.

    Parameters:
        path: A path in or below the bundle's directory.
        bundle_dir: The bundle's directory.

    Returns:
        The relative path.
    """
    return PurePath(os.path.relpath(path, bundle_dir)).as_posix()


def _named_files(
    file: str, root: Any, bundle_dir: Path, named: dict[str, list[str]]
) -> list[Finding]:
    """Look for the file each ``file_name`` of a label names, beside the label.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.
        bundle_dir: The bundle's directory.
        named: The labels naming each file found so far, by the file's path relative to
            the bundle's directory; each file this label names is added.

    Returns:
        One finding for each ``file_name`` naming a file that is not beside the label.
    """
    findings: list[Finding] = []
    directory = (bundle_dir / file).parent
    for element in root.iter('{*}file_name'):
        name = str(element.text or '').strip()
        target = directory / name
        if name == '' or not target.is_file():
            findings.append(
                Finding(
                    file,
                    CheckName.INTEGRITY,
                    element_path(element),
                    f'names {name!r}, which is not beside it',
                )
            )
            continue
        named.setdefault(_relative(target, bundle_dir), []).append(file)
    return findings


def _marker(file: str, label: Path) -> list[Finding]:
    """Look for a marker a failed template expression left in a label.

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
            'holds a [[[ marker, which a template leaves where an expression failed',
        )
    ]


def _empty_elements(file: str, root: Any) -> list[Finding]:
    """Look for the elements of a label that hold nothing and do not carry ``xsi:nil``.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.

    Returns:
        One finding for each element holding no child element and no text but white
        space, a comment in it notwithstanding, that does not carry ``xsi:nil``.
    """
    findings: list[Finding] = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        if any(isinstance(part.tag, str) for part in element):
            continue
        text = str(element.text or '') + ''.join(str(part.tail or '') for part in element)
        if text.strip() != '' or element.get(XSI_NIL) in ('true', '1'):
            continue
        findings.append(
            Finding(
                file,
                CheckName.INTEGRITY,
                element_path(element),
                'is empty, and does not carry xsi:nil',
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


def _own_references(
    file: str, root: Any, bundle_lid: str, products: Mapping[str, str]
) -> list[Finding]:
    """Resolve each reference of a label to a product of the bundle itself.

    Parameters:
        file: The label's path relative to the bundle's directory.
        root: The label's root element.
        bundle_lid: The logical identifier the bundle label declares.
        products: The version of each product a label of the tree declares, by its
            logical identifier.

    Returns:
        One finding for each reference to a product of the bundle that no label of the
        tree declares, and one for each naming a version other than the product's.
    """
    findings: list[Finding] = []
    for element in root.iter('{*}lid_reference', '{*}lidvid_reference'):
        reference = str(element.text or '').strip()
        lid, _, version = reference.partition('::')
        if lid != bundle_lid and not lid.startswith(f'{bundle_lid}:'):
            continue
        if lid not in products:
            message = f'refers to {reference}, which no label of the tree declares'
        elif version != '' and version != products[lid]:
            message = f'refers to {reference}, but the tree holds {lid} at version {products[lid]}'
        else:
            continue
        findings.append(Finding(file, CheckName.INTEGRITY, element_path(element), message))
    return findings


def integrity_findings(bundle_dir: Path, labels: Mapping[str, Any]) -> list[Finding]:
    """Check a bundle tree as a whole.

    Parameters:
        bundle_dir: The bundle's directory.
        labels: Every label of the tree that parses, by its path relative to the bundle's
            directory, parsed by lxml.

    Returns:
        One finding for each ``file_name`` naming a file that is not beside its label,
        each file that is neither a label nor named by exactly one label, each label
        holding a ``[[[`` marker, each empty element without ``xsi:nil``, and each
        reference to a product of the bundle that does not resolve in the tree; and one
        finding for a tree with no bundle label at its top, over which no such reference
        can be resolved.
    """
    findings: list[Finding] = []
    named: dict[str, list[str]] = {}
    products: dict[str, str] = {}
    bundle_lid: str | None = None
    for file, document in labels.items():
        root = document.getroot()
        findings.extend(_named_files(file, root, bundle_dir, named))
        findings.extend(_marker(file, bundle_dir / file))
        findings.extend(_empty_elements(file, root))
        lid = child_text(root, 'Identification_Area', 'logical_identifier')
        version = child_text(root, 'Identification_Area', 'version_id')
        if lid is not None and version is not None:
            products[lid] = version
        if local_name(root) == 'Product_Bundle' and '/' not in file:
            bundle_lid = lid
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
