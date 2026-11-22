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
    field of every item of ``bodies``.  The members of a union share their path, so a
    tagged union's members contribute their fields side by side.

    Parameters:
        cls: The schema's root dataclass.

    Returns:
        The markers of each field that carries any, keyed by path.

    Raises:
        TypeError: If two members of a union carry different markers on a field with
            the same path.
    """
    found: dict[str, Mapping[str, Any]] = {}
    _walk(cls, prefix='', found=found, visiting=())
    return found


def _walk(
    annotation: Any, *, prefix: str, found: dict[str, Mapping[str, Any]], visiting: tuple[type, ...]
) -> None:
    """Collect the markers under one annotation, which sits at ``prefix``."""
    base = get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    origin = get_origin(base)
    if origin is Union or origin is types.UnionType:
        for member in get_args(base):
            _walk(member, prefix=prefix, found=found, visiting=visiting)
        return
    if origin in _SEQUENCES or origin in _MAPPINGS:
        arguments = get_args(base)
        items = arguments[-1:] if origin in _MAPPINGS else arguments
        for item in items:
            if item is not Ellipsis:
                _walk(item, prefix=_join(prefix, ELEMENT), found=found, visiting=visiting)
        return
    if not (isinstance(base, type) and dataclasses.is_dataclass(base)) or base in visiting:
        return
    hints = get_type_hints(base, include_extras=True)
    for item in dataclasses.fields(base):
        path = _join(prefix, item.name)
        if item.metadata:
            if path in found and found[path] != item.metadata:
                raise TypeError(
                    f'{path}: union members carry different markers '
                    f'({dict(found[path])} and {dict(item.metadata)})'
                )
            found[path] = item.metadata
        _walk(hints[item.name], prefix=path, found=found, visiting=(*visiting, base))


def _join(prefix: str, segment: str) -> str:
    """Extend a dotted field path by one segment."""
    return f'{prefix}.{segment}' if prefix else segment
