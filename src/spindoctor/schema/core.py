"""Parse a merged YAML mapping into a typed, frozen dataclass.

A schema is a plain frozen dataclass whose fields carry type annotations.  The
annotations say everything about a key: its type, its range, its default, and whether
it is required.  :func:`parse` checks a mapping against such a class through pydantic
and returns an instance of it.  On failure it raises one :class:`SchemaError` that
lists every problem at once, one line each, in the form ``file: key.path: message``.

The conventions every schema follows:

- **Declare a schema as a frozen stdlib dataclass** that sets
  ``__pydantic_config__ = SCHEMA_CONFIG``.  pydantic is the parser, never a base class:
  no type subclasses pydantic's ``BaseModel``.  The configuration refuses unknown
  keys, so a misspelled key is always an error.
- **Use the aliases below for scalars.**  :data:`Float`, :data:`Int`, :data:`Bool`, and
  :data:`Str` are strict: a number written in quotes is a string and fails, a boolean
  is never a number, and a float must be finite.  Strictness lives on the fields
  through these aliases, never on a dataclass's configuration, because a strict
  dataclass accepts only an instance of itself and refuses a mapping.
- **Use the range aliases for common ranges**: :data:`Positive`, :data:`NonNeg`, and
  :data:`Unit` for floats, and :data:`PositiveInt` and :data:`NonNegInt` for integers.
  Any other range is an ``annotated_types`` constraint, such as
  ``Annotated[Float, Le(90)]``.
- **Type an integer-keyed mapping** as ``dict[DigitInt, ...]``.  A mapping that passed
  through JSON has string keys, and :data:`DigitInt` reads ``'2'`` as ``2``.
- **Type a path or URL** as :data:`FCPath`.
- **Write a rule that ties two fields together** as ``__post_init__`` on the type that
  owns both fields.  It raises ``ValueError``, and the message becomes the problem's
  message at that type's key path.
- **Write a polymorphic entry as a tagged union**: every member has a ``Literal`` tag
  field, and the union is ``Annotated[A | B, tagged('kind')]``.
- **Carry markers for other code** in ``dataclasses.field(metadata={...})``.  The core
  never reads them; :func:`spindoctor.schema.metadata.walk_metadata` collects them.

The supported annotations are the aliases above, ``Literal``, ``Enum`` subclasses,
``X | None``, unions, ``list[X]``, ``tuple[X, ...]``, fixed tuples, ``dict[K, V]``,
``Mapping[K, V]``, and nested schema dataclasses.  :func:`check_schema` refuses any
other annotation, a bare ``float``, ``int``, ``bool``, or ``str``, and a dataclass that
is not frozen or does not use :data:`SCHEMA_CONFIG`.  :func:`parse` runs that check the
first time it sees a class.  Adding a new form or convention needs a reason, written
here.

A null is never data in the files this core reads.  The layer merge
(:mod:`spindoctor.schema.layers`) applies each file family's rule for null before
parsing, so ``X | None`` only describes an optional key that is absent.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import re
import types
from collections.abc import Hashable, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import filecache
from annotated_types import Ge, Gt, Le
from pydantic import (
    AllowInfNan,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    Strict,
    TypeAdapter,
    ValidationError,
)

__all__ = [
    'SCHEMA_CONFIG',
    'Bool',
    'DigitInt',
    'FCPath',
    'Float',
    'Int',
    'Issue',
    'KeyPath',
    'NonNeg',
    'NonNegInt',
    'Positive',
    'PositiveInt',
    'SchemaError',
    'Str',
    'Unit',
    'check_schema',
    'describe_value',
    'format_key_path',
    'parse',
    'tagged',
]

T = TypeVar('T')

#: A key path into a parsed YAML tree: mapping keys, and integer positions in lists.
KeyPath: TypeAlias = tuple[Any, ...]

#: The pydantic configuration every schema dataclass sets as ``__pydantic_config__``.
SCHEMA_CONFIG = ConfigDict(extra='forbid')

#: A finite float.  An integer is accepted and becomes a float.
Float = Annotated[float, Strict(), AllowInfNan(False)]
#: An integer; a float or a boolean fails.
Int = Annotated[int, Strict()]
#: A boolean; a number or a string fails.
Bool = Annotated[bool, Strict()]
#: A string; a number fails.
Str = Annotated[str, Strict()]
#: A finite float above zero.
Positive = Annotated[Float, Gt(0)]
#: A finite float at or above zero.
NonNeg = Annotated[Float, Ge(0)]
#: A finite float in [0, 1].
Unit = Annotated[Float, Ge(0), Le(1)]
#: An integer above zero.
PositiveInt = Annotated[Int, Gt(0)]
#: An integer at or above zero.
NonNegInt = Annotated[Int, Ge(0)]

# A string of ASCII digits, which DigitInt reads as the integer it spells.
_DIGITS = re.compile(r'[0-9]+')


def _digits_to_int(value: object) -> object:
    """Turn a string of digits into its integer; return anything else unchanged."""
    if isinstance(value, str) and _DIGITS.fullmatch(value) is not None:
        return int(value)
    return value


def _to_fcpath(value: object) -> filecache.FCPath:
    """Turn a path or URL string into an ``FCPath``.

    Raises:
        ValueError: If ``value`` is neither a string nor an ``FCPath``.
    """
    if isinstance(value, filecache.FCPath):
        return value
    if isinstance(value, str):
        return filecache.FCPath(value)
    raise ValueError(f'expected a path or URL, got {describe_value(value)}')


#: An integer, also written as a string of digits.  Use it for mapping keys.
DigitInt = Annotated[int, BeforeValidator(_digits_to_int), Strict()]
#: A local path or a URL, read from a string.
FCPath = Annotated[filecache.FCPath, PlainValidator(_to_fcpath)]

# The scalar types a field must not use bare, and the alias that replaces each.
_STRICT_ALIASES: dict[type, str] = {float: 'Float', int: 'Int', bool: 'Bool', str: 'Str'}

# The generic containers a schema may use; a mapping's last argument is its value.
_CONTAINERS: frozenset[Any] = frozenset({list, tuple, dict, Mapping})

# pydantic error types, by how this module reports them.  A type error states what
# the value should have been.
_EXPECTED: dict[str, str] = {
    'bool_parsing': 'a boolean',
    'bool_type': 'a boolean',
    'dataclass_type': 'a mapping',
    'dict_type': 'a mapping',
    'finite_number': 'a finite number',
    'float_parsing': 'a number',
    'float_type': 'a number',
    'int_from_float': 'an integer',
    'int_parsing': 'an integer',
    'int_type': 'an integer',
    'list_type': 'a list',
    'mapping_type': 'a mapping',
    'model_attributes_type': 'a mapping',
    'string_type': 'a string',
    'tuple_type': 'a list',
}
_MISSING = frozenset({'missing'})
_UNKNOWN = frozenset({'extra_forbidden', 'unexpected_keyword_argument'})
_RULE = frozenset({'assertion_error', 'value_error'})
# The location segment pydantic adds for a problem with a mapping key itself.
_KEY_SEGMENT = '[key]'


@dataclasses.dataclass(frozen=True)
class Issue:
    """One problem found in a file.

    Parameters:
        source: The file (or file and entry) that set the value.
        path: The dotted key path of the value, with ``[n]`` for a list item; empty
            for the file as a whole.
        message: What is wrong.
    """

    source: str
    path: str
    message: str

    def __str__(self) -> str:
        """The problem as one line: ``source: path: message``."""
        location = f'{self.source}: {self.path}' if self.path else self.source
        return f'{location}: {self.message}'


class SchemaError(ValueError):
    """A file breaks its schema.  The message has one line per problem.

    Parameters:
        issues: Every problem found, in the order found.
    """

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues: tuple[Issue, ...] = tuple(issues)
        super().__init__('\n'.join(str(issue) for issue in self.issues))

    def __reduce__(self) -> tuple[type[SchemaError], tuple[tuple[Issue, ...]]]:
        """Rebuild the error from its issues, so it survives a trip between processes."""
        return (SchemaError, (self.issues,))


def tagged(key: str) -> Any:
    """The marker that makes a union of schema dataclasses a tagged union.

    Use it as ``Annotated[A | B, tagged('kind')]``, where each member declares the
    field ``key`` as a ``Literal`` of its own tag.  The tag picks the member, so a
    problem is reported against that member alone.

    Parameters:
        key: The name of the tag field.

    Returns:
        The annotation marker.
    """
    return Field(discriminator=key)


def describe_value(value: object) -> str:
    """Describe a value for an error message: its type and its value.

    A string that reads as a number gets a hint, since quoting a number in YAML is
    the usual way to write a string where a number belongs.

    Parameters:
        value: The value read from a file.

    Returns:
        For example ``"int 3"``, or ``"str '0.5' (a number written so YAML reads it as a
        string)"``.
    """
    text = f'{type(value).__name__} {value!r}'
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return text
        return f'{text} (a number written so YAML reads it as a string)'
    return text


def format_key_path(root: object, keys: KeyPath) -> str:
    """Write a key path the way error messages show it.

    Mapping keys join with dots, and a position in a list is written ``[n]``.

    Parameters:
        root: The tree the path leads into, which tells list positions from mapping
            keys.
        keys: The keys from ``root`` down.

    Returns:
        For example ``'bodies[0].radius'``; empty for an empty path.
    """
    text = ''
    node: object = root
    for key in keys:
        if isinstance(node, list | tuple) and isinstance(key, int):
            text = f'{text}[{key}]'
            node = node[key] if 0 <= key < len(node) else None
        else:
            text = f'{text}.{key}' if text else str(key)
            node = node.get(key) if isinstance(node, Mapping) else None
    return text


def check_schema(cls: object) -> None:
    """Check that a schema follows the conventions in this module's docstring.

    Parameters:
        cls: The schema's root type, normally a dataclass.

    Raises:
        TypeError: If a dataclass in the schema is not frozen or does not set
            ``__pydantic_config__ = SCHEMA_CONFIG``, if a field uses a bare
            ``float``, ``int``, ``bool``, or ``str``, or if an annotation is not one
            of the supported forms.
    """
    _check_annotation(cls, where=getattr(cls, '__qualname__', repr(cls)), seen=set())


def _check_annotation(annotation: Any, *, where: str, seen: set[type]) -> None:
    """Check one annotation and everything inside it; ``where`` locates it in messages."""
    base = annotation
    extras: tuple[Any, ...] = ()
    if get_origin(annotation) is Annotated:
        base, *rest = get_args(annotation)
        extras = tuple(rest)
    if base in _STRICT_ALIASES:
        if not any(isinstance(extra, Strict) for extra in extras):
            raise TypeError(
                f'{where}: a bare {base.__name__} reads values leniently; use the schema '
                f'alias {_STRICT_ALIASES[base]}'
            )
        return
    origin = get_origin(base)
    if base is filecache.FCPath or base is type(None) or origin is Literal:
        return
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return
    if origin is Union or origin is types.UnionType or origin in _CONTAINERS:
        for argument in get_args(base):
            if argument is not Ellipsis:
                _check_annotation(argument, where=where, seen=seen)
        return
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        _check_dataclass(base, seen=seen)
        return
    raise TypeError(f'{where}: {base!r} is not a supported schema annotation')


def _check_dataclass(cls: type, *, seen: set[type]) -> None:
    """Check one schema dataclass and the annotation of each of its fields."""
    if cls in seen:
        return
    seen.add(cls)
    params = vars(cls).get('__dataclass_params__')
    if params is None or not params.frozen:
        raise TypeError(f'{cls.__qualname__}: a schema dataclass must be frozen')
    if getattr(cls, '__pydantic_config__', None) != SCHEMA_CONFIG:
        raise TypeError(
            f'{cls.__qualname__}: a schema dataclass sets __pydantic_config__ = SCHEMA_CONFIG'
        )
    hints = get_type_hints(cls, include_extras=True)
    for item in dataclasses.fields(cls):
        _check_annotation(hints[item.name], where=f'{cls.__qualname__}.{item.name}', seen=seen)


@functools.cache
def _adapter(cls: Hashable) -> TypeAdapter[Any]:
    """The checked, cached pydantic adapter for one schema root."""
    check_schema(cls)
    return TypeAdapter(cls)


def parse(
    cls: type[T], raw: object, *, source: str, origins: Mapping[KeyPath, str] | None = None
) -> T:
    """Check a mapping against a schema and build the typed value.

    Parameters:
        cls: The schema's root type.
        raw: The mapping to check, normally the data of a merged set of layers.
        source: The file (or file and entry) to name in a message when ``origins``
            says nothing about the key.
        origins: The source of each value, keyed by the value's key path, as the layer
            merge records it.  A problem with a value, or with an unknown key, gives
            the source that set it.  A problem with a key that is absent, or with a
            whole mapping, gives every source that set a value inside that mapping.

    Returns:
        The instance of ``cls``.

    Raises:
        SchemaError: Listing every problem: each unknown key, missing required key,
            value of the wrong type or out of range, and broken cross-field rule.
        TypeError: If ``cls`` breaks the schema conventions (see :func:`check_schema`).
    """
    # A class is hashable; typeshed's type[T] does not say so.
    adapter = _adapter(cast(Hashable, cls))
    try:
        value: T = adapter.validate_python(raw)
    except ValidationError as exc:
        reporter = _Reporter(raw, source=source, origins=origins or {})
        raise SchemaError(reporter.issues(exc.errors(include_url=False))) from exc
    return value


@dataclasses.dataclass
class _Found:
    """One problem, located in the input, before problems at one place are combined.

    Parameters:
        keys: The key path of the value the problem is about.
        kind: ``'missing'``, ``'unknown'``, ``'expected'``, or ``'message'``.
        words: For ``'missing'`` and ``'unknown'``, the keys; for ``'expected'``, what
            the value should have been; for ``'message'``, the message.
        got: For ``'expected'``, how the value is described.
        named: For ``'unknown'``, the key path of each unknown key, whose sources the
            issue gives; otherwise empty, and the issue gives the sources of ``keys``.
    """

    keys: KeyPath
    kind: str
    words: list[str]
    got: str = ''
    named: list[KeyPath] = dataclasses.field(default_factory=list)


class _Reporter:
    """Turn pydantic's errors for one input into issues in the house format."""

    def __init__(self, raw: object, *, source: str, origins: Mapping[KeyPath, str]) -> None:
        self._raw = raw
        self._source = source
        self._origins = origins

    def issues(self, errors: Sequence[Mapping[str, Any]]) -> list[Issue]:
        """Every issue, with problems at one place combined into one line.

        Missing keys and unknown keys of one mapping each become one line, and the
        failed members of a union of scalars become one line that lists every member.
        """
        combined: dict[tuple[KeyPath, str], _Found] = {}
        lines: list[_Found] = []
        for error in errors:
            found = self._locate(error)
            slot = (found.keys, found.kind)
            if found.kind != 'message' and slot in combined:
                previous = combined[slot]
                previous.words.extend(w for w in found.words if w not in previous.words)
                previous.named.extend(found.named)
                continue
            combined[slot] = found
            lines.append(found)
        return [self._issue(found) for found in lines]

    def _locate(self, error: Mapping[str, Any]) -> _Found:
        """Place one pydantic error on the input's keys and classify it."""
        kind = error['type']
        ctx = error.get('ctx') or {}
        keys, about_key = self._keys(tuple(error['loc']), keep_last=kind in _MISSING)
        if kind in _MISSING:
            return _Found(keys[:-1], 'missing', [str(keys[-1])])
        if kind in _UNKNOWN:
            return _Found(keys[:-1], 'unknown', [str(keys[-1])], named=[keys])
        if kind == 'union_tag_not_found':
            return _Found(keys, 'missing', [_unquote(ctx['discriminator'])])
        if kind == 'union_tag_invalid':
            expected = f'one of {ctx["expected_tags"]}'
            return _Found(
                (*keys, _unquote(ctx['discriminator'])), 'expected', [expected], repr(ctx['tag'])
            )
        value = error['input']
        if kind in _RULE:
            return _Found(keys, 'message', [str(ctx['error'])])
        bound = _bound_words(kind, ctx)
        if bound is not None:
            return _Found(keys, 'expected', [bound], repr(value))
        if kind in _EXPECTED or kind == 'literal_error' or kind == 'enum':
            expected = _EXPECTED[kind] if kind in _EXPECTED else str(ctx['expected'])
            if about_key:
                expected = f'{expected} as the key'
            return _Found(keys, 'expected', [expected], describe_value(value))
        message = str(error['msg'])
        return _Found(keys, 'message', [message[:1].lower() + message[1:]])

    def _keys(self, loc: tuple[Any, ...], *, keep_last: bool) -> tuple[KeyPath, bool]:
        """Map a pydantic location onto the input's keys.

        A segment that leads nowhere in the input is a union member's tag, and is
        dropped, except a missing key's own name at the end.

        Returns:
            The key path, and whether the problem is with a mapping key itself.
        """
        keys: list[Any] = []
        node: object = self._raw
        about_key = False
        for index, segment in enumerate(loc):
            if segment == _KEY_SEGMENT:
                about_key = True
                continue
            if isinstance(node, Mapping):
                key = _mapping_key(node, segment)
                if key is not _ABSENT:
                    keys.append(key)
                    node = node[key]
                    continue
            elif isinstance(node, list | tuple) and isinstance(segment, int):
                if 0 <= segment < len(node):
                    keys.append(segment)
                    node = node[segment]
                    continue
            if keep_last and index == len(loc) - 1:
                keys.append(segment)
        return tuple(keys), about_key

    def _issue(self, found: _Found) -> Issue:
        """Write one combined problem as an issue."""
        if found.kind == 'missing':
            message = f'missing required keys: {sorted(found.words)}'
        elif found.kind == 'unknown':
            message = f'unknown keys: {sorted(found.words)}'
        elif found.kind == 'expected':
            message = f'expected {" or ".join(found.words)}, got {found.got}'
        else:
            message = found.words[0]
        sources = dict.fromkeys(self._origin(keys) for keys in found.named or [found.keys])
        return Issue(', '.join(sources), format_key_path(self._raw, found.keys), message)

    def _origin(self, keys: KeyPath) -> str:
        """The source to name for a problem at ``keys``.

        The source that set the value at ``keys``, or else the sources that set a
        value inside it; failing both, the same for the enclosing mapping, and so on
        up to the root.
        """
        for depth in range(len(keys), -1, -1):
            prefix = keys[:depth]
            if prefix in self._origins:
                return self._origins[prefix]
            inside = [label for path, label in self._origins.items() if path[:depth] == prefix]
            if inside:
                return ', '.join(dict.fromkeys(inside))
        return self._source


# What _mapping_key returns when no key of the mapping matches.
_ABSENT = object()


def _mapping_key(node: Mapping[Any, Any], segment: Any) -> Any:
    """The key of ``node`` that a location segment stands for, or ``_ABSENT``.

    pydantic writes an integer key that arrived as digits as a string, and the reverse,
    so both spellings are tried.
    """
    candidates = [segment]
    if isinstance(segment, str) and _DIGITS.fullmatch(segment) is not None:
        candidates.append(int(segment))
    elif isinstance(segment, int):
        candidates.append(str(segment))
    for candidate in candidates:
        if candidate in node:
            return candidate
    return _ABSENT


def _unquote(text: str) -> str:
    """A pydantic context value without the quotes pydantic wraps around a name."""
    return str(text).strip("'")


def _bound_words(kind: str, ctx: Mapping[str, Any]) -> str | None:
    """What a value out of range should have been, or None for another kind of error."""
    if kind == 'greater_than':
        return 'a positive number' if ctx['gt'] == 0 else f'a number greater than {ctx["gt"]}'
    if kind == 'greater_than_equal':
        return 'a non-negative number' if ctx['ge'] == 0 else f'a number of at least {ctx["ge"]}'
    if kind == 'less_than':
        return f'a number less than {ctx["lt"]}'
    if kind == 'less_than_equal':
        return f'a number of at most {ctx["le"]}'
    return None
