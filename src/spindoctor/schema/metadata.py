"""Collect the markers that a schema's fields carry for other code.

A schema field can carry markers in ``dataclasses.field(metadata={...})``: the
simulator marks which side of its information boundary a key belongs to, and the
scene editor reads hints for its controls.  The schema core never reads these
markers.  :func:`walk_metadata` gathers them from a whole schema, keyed by each
field's path.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

__all__ = ['ELEMENT', 'walk_metadata']

#: The path segment that stands for every item of a list or tuple and every value of
#: a mapping.
ELEMENT = '*'

# The containers whose items are walked: for a mapping only the value type, its last
# argument, holds fields.
_SEQUENCES: frozenset[Any] = frozenset({list, tuple})
_MAPPINGS: frozenset[Any] = frozenset({dict, Mapping})


def walk_metadata(cls: type) -> dict[str, Mapping[str, Any]]:
    """Map the path of every field that carries markers to its markers.

    A path joins field names with dots.  An item of a list or tuple, and a value of
    a mapping, is written :data:`ELEMENT`, so ``bodies.*.radius`` is the ``radius``
    field of every item of ``bodies``.  The members of a union, and the items of a
    fixed tuple, share their path, so a tagged union's members contribute their fields
    side by side.  Two types at one path must mark each field they share the same way,
    and a field one of them leaves unmarked must be unmarked in the other too.

    Parameters:
        cls: The schema's root dataclass.

    Returns:
        The markers of each field that carries any, keyed by path.

    Raises:
        TypeError: If two types at one path mark a field with the same path
            differently, or one marks it and the other does not.
    """
    seen: dict[str, Mapping[str, Any]] = {}
    _walk(cls, prefix='', seen=seen, visiting=())
    return {path: markers for path, markers in seen.items() if markers}


def _walk(
    annotation: Any, *, prefix: str, seen: dict[str, Mapping[str, Any]], visiting: tuple[type, ...]
) -> None:
    """Collect the markers of every field under one annotation, which sits at ``prefix``.

    ``seen`` gets every field path, with empty markers for an unmarked field, so two
    types that disagree about a path are found even when one of them marks nothing.
    """
    base = get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    origin = get_origin(base)
    if origin is Union or origin is types.UnionType:
        for member in get_args(base):
            _walk(member, prefix=prefix, seen=seen, visiting=visiting)
        return
    if origin in _SEQUENCES or origin in _MAPPINGS:
        arguments = get_args(base)
        items = arguments[-1:] if origin in _MAPPINGS else arguments
        for item in items:
            if item is not Ellipsis:
                _walk(item, prefix=_join(prefix, ELEMENT), seen=seen, visiting=visiting)
        return
    if not (isinstance(base, type) and dataclasses.is_dataclass(base)) or base in visiting:
        return
    hints = get_type_hints(base, include_extras=True)
    for item in dataclasses.fields(base):
        path = _join(prefix, item.name)
        if path in seen and seen[path] != item.metadata:
            raise TypeError(
                f'{path}: two types at this path mark the field differently '
                f'({dict(seen[path])} and {dict(item.metadata)})'
            )
        seen[path] = item.metadata
        _walk(hints[item.name], prefix=path, seen=seen, visiting=(*visiting, base))


def _join(prefix: str, segment: str) -> str:
    """Extend a dotted field path by one segment."""
    return f'{prefix}.{segment}' if prefix else segment
