"""Controls for the bundle check: a copy of a written bundle, broken in one place.

A control is a bundle the cohort build wrote, copied into a test's own directory and
changed in exactly one place, so that the test can hold the check to finding the change.
"""

import re
import shutil
from pathlib import Path
from typing import Any

from lxml import etree


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

    Parameters:
        path: The file.
        pattern: A regular expression that matches exactly once where it is looked for.
        replacement: What replaces the match; ``\\1`` and the like name its groups.
        within: The name of the field whose element the match is looked for in, from its
            ``name`` to the end of the element, or None for the whole file.

    Raises:
        ValueError: If the pattern does not match exactly once where it is looked for.
    """
    text = path.read_text(encoding='utf-8')
    start = 0
    end = len(text)
    if within is not None:
        start = text.index(f'<name>{within}</name>')
        end = text.index('</Field_', start)
    matches = len(re.findall(pattern, text[start:end]))
    if matches != 1:
        raise ValueError(f'{pattern!r} matches {matches} time(s) in {path}, not once')
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
        Each label, parsed, by its path relative to the bundle's directory.
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
