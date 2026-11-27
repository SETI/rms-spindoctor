"""Write the scene schema reference page from the typed scene's field declarations.

The documentation build calls :func:`write_scene_reference` before it reads its
sources, so the page lists exactly the keys the typed scene declares: each key's type
and range, its default, its side of the information boundary, and what it means.
The blocks the typed scene does not hold yet are listed at the end, with a pointer to
the developer guide's tables for them.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import AfterValidator

from spindoctor.schema import (
    Bool,
    Float,
    Int,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    Str,
    Unit,
)
from spindoctor.sim.scene.declare import IntWithin, NonEmptyStr, declaration_of
from spindoctor.sim.scene.truth import UNTYPED_BLOCKS, Scene

__all__ = ['scene_reference', 'write_scene_reference']

# The words for each schema alias, as the validator's messages use them.
_ALIAS_WORDS: tuple[tuple[Any, str], ...] = (
    (Positive, 'a positive number'),
    (NonNeg, 'a non-negative number'),
    (Unit, 'a number in [0, 1]'),
    (PositiveInt, 'a positive integer'),
    (NonNegInt, 'a non-negative integer'),
    (NonEmptyStr, 'a non-empty string'),
    (Float, 'a number'),
    (Int, 'an integer'),
    (Bool, 'true or false'),
    (Str, 'a string'),
)

# The boundary classes as the page writes them.
_SIDES: dict[str, str] = {'idealized': 'idealized', 'truth': 'truth', 'test': 'test-only'}

_TITLE = 'Scene Schema Reference'

_INTRO = """\
This page lists every key of the typed scene blocks, generated from the scene's
field declarations when the documentation is built.  For each key it gives the type
and range a scene may write, what a scene that leaves the key out gets, which side of
the information boundary the key is on (see the simulator chapter of the developer
guide), and what the key means.  A scene that writes a value outside its range, or a
key not listed, is refused when it is loaded.
"""

_UNTYPED_INTRO = """\
These top-level blocks are not typed yet.  The developer guide's simulator chapter
tabulates their keys under "Scene parameter reference".
"""


def _without_none(annotation: Any) -> Any:
    """An annotation with ``None`` taken out of its union."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _record(annotation: Any) -> type | None:
    """The dataclass a field holds, alone or as the items of a list; else None."""
    base = _without_none(annotation)
    if get_origin(base) is tuple:
        base = get_args(base)[0]
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        return base
    return None


def _describe(annotation: Any) -> str:
    """The type and range of a field, in words.

    Raises:
        TypeError: If the page has no words for the annotation.
    """
    base = _without_none(annotation)
    for alias, words in _ALIAS_WORDS:
        if base == alias:
            return words
    origin = get_origin(base)
    if origin is Literal:
        return 'one of ' + ', '.join(f'``{choice}``' for choice in get_args(base))
    if origin is tuple:
        item = get_args(base)[0]
        if isinstance(item, type) and dataclasses.is_dataclass(item):
            return 'a list of mappings, below'
        return f'a list of {_plural(_describe(item))}'
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        return 'a mapping, below'
    if origin is Annotated:
        for extra in get_args(base)[1:]:
            if isinstance(extra, AfterValidator) and isinstance(extra.func, IntWithin):
                return extra.func.expected
    raise TypeError(f'the scene reference has no words for {annotation!r}')


def _plural(words: str) -> str:
    """Turn ``a mapping`` into ``mappings`` and ``a number`` into ``numbers``."""
    for article in ('a ', 'an '):
        if words.startswith(article):
            noun = words[len(article) :]
            head, _, rest = noun.partition(' ')
            return f'{head}s {rest}'.rstrip()
    return words


def _default_text(value: float | str | bool) -> str:
    """A default value as a scene file writes it."""
    if isinstance(value, bool):
        return '``true``' if value else '``false``'
    if isinstance(value, str):
        return f'``{value}``'
    return str(value)


def _table(cls: type) -> list[str]:
    """The rows of one block's table, in the field order of the block's type."""
    hints = get_type_hints(cls, include_extras=True)
    lines = [
        '.. list-table::',
        '   :widths: 22 18 18 10 32',
        '   :header-rows: 1',
        '',
        '   * - Key',
        '     - Type',
        '     - When left out',
        '     - Side',
        '     - Meaning',
    ]
    for item in dataclasses.fields(cls):
        declared = declaration_of(item)
        if declared.required:
            absent = 'required'
        elif declared.default is not None:
            absent = _default_text(declared.default)
        else:
            absent = str(declared.when_absent)
        lines += [
            f'   * - ``{item.name}``',
            f'     - {_describe(hints[item.name])}',
            f'     - {absent}',
            f'     - {_SIDES[declared.boundary]}',
            f'     - {declared.description}',
        ]
    return lines


def _heading(text: str, underline: str) -> list[str]:
    """A section heading."""
    return [text, underline * len(text)]


def scene_reference() -> str:
    """The scene schema reference page, in reStructuredText.

    Returns:
        The page.
    """
    lines = ['=' * len(_TITLE), _TITLE, '=' * len(_TITLE), '', _INTRO]
    queue: list[tuple[str, type, bool]] = [('', Scene, False)]
    while queue:
        path, cls, is_list = queue.pop(0)
        if not path:
            lines += [*_heading('The top level', '='), '']
        elif is_list:
            lines += [*_heading(f'Each entry of ``{path}``', '='), '']
        else:
            lines += [*_heading(f'The ``{path}`` block', '='), '']
        lines += [*_table(cls), '']
        hints = get_type_hints(cls, include_extras=True)
        for item in dataclasses.fields(cls):
            record = _record(hints[item.name])
            if record is not None:
                inner = _without_none(hints[item.name])
                child = f'{path}.{item.name}' if path else item.name
                queue.append((child, record, get_origin(inner) is tuple))
    lines += [*_heading('Blocks not typed yet', '='), '', _UNTYPED_INTRO]
    lines += [
        '.. list-table::',
        '   :widths: 40 60',
        '   :header-rows: 1',
        '',
        '   * - Block',
        '     - Side',
    ]
    for name, boundary in UNTYPED_BLOCKS.items():
        lines += [f'   * - ``{name}``', f'     - {_SIDES[boundary]}']
    return '\n'.join(lines) + '\n'


def write_scene_reference(path: Path) -> None:
    """Write the reference page, leaving the file alone if its text has not changed.

    Parameters:
        path: The page's path.
    """
    text = scene_reference()
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    path.write_text(text, encoding='utf-8')
