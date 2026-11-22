"""Parse a merged YAML mapping into a typed, frozen dataclass.

A schema is a plain frozen dataclass whose fields carry type annotations.  The
annotations say everything about a key: its type, its range, its default, and whether
it is required.  :func:`parse` checks a mapping against such a class through pydantic
and returns an instance of it.  On failure it raises one
:class:`~spindoctor.schema.errors.SchemaError` that lists every problem at once, one
line each, in the form ``file: key.path: message``.

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
  Each reports its own range in the catalog's words ("a number in [0, 1]").  Any other
  range is an ``annotated_types`` constraint, such as ``Annotated[Float, Le(90)]``.
- **Type an integer-keyed mapping** as ``dict[DigitInt, ...]``.  A mapping that passed
  through JSON has string keys, and :data:`DigitInt` reads ``'2'`` as ``2``.
- **Type a path or URL** as :data:`FCPath`.
- **Write a choice** as a ``Literal`` or an ``Enum`` of strings.
- **Write a rule that ties two fields together** as ``__post_init__`` on the type that
  owns both fields.  It raises ``ValueError``, and the message becomes the problem's
  message at that type's key path.  Any other exception is a bug in the rule, and it
  propagates.
- **Write a polymorphic entry as a tagged union**: every member is a schema dataclass
  with a ``Literal`` tag field, and the union is ``Annotated[A | B, tagged('kind')]``.
  A union of two or more dataclasses must be tagged.
- **Carry markers for other code** in ``dataclasses.field(metadata={...})``.  The core
  never reads them; :func:`spindoctor.schema.metadata.walk_metadata` collects them.
- **Build a schema once, at import.**  :func:`parse` keeps the checked pydantic adapter
  of every class it has seen, keyed by the class, for the life of the process.

The supported annotations are the aliases above, ``Literal`` and ``Enum`` of strings,
``X | None``, unions, ``list[X]``, ``tuple[X, ...]``, fixed tuples, ``dict[K, V]``,
``Mapping[K, V]``, nested schema dataclasses (a schema may refer to itself), and
``annotated_types`` constraints such as ``Le`` and ``Len``.  :func:`check_schema`
refuses everything else, and also refuses what would read a value leniently: a bare
``float``, ``int``, ``bool``, or ``str``, a ``Strict(False)`` or ``AllowInfNan(True)``
marker, and a validator that runs before the type check.  :func:`parse` runs that
check the first time it sees a class.  Adding a new form or convention needs a reason,
written here.

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
from collections.abc import Callable, Hashable, Mapping
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import filecache
from pydantic import (
    AfterValidator,
    AllowInfNan,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    Strict,
    TypeAdapter,
    ValidationError,
    WrapValidator,
)
from pydantic.fields import FieldInfo

from spindoctor.schema.errors import (
    KeyPath,
    RangeError,
    SchemaError,
    describe_value,
    doubled_keys,
    report,
)

__all__ = [
    'SCHEMA_CONFIG',
    'Bool',
    'DigitInt',
    'FCPath',
    'Float',
    'Int',
    'NonNeg',
    'NonNegInt',
    'Positive',
    'PositiveInt',
    'Str',
    'Unit',
    'check_schema',
    'parse',
    'tagged',
]

T = TypeVar('T')

#: The pydantic configuration every schema dataclass sets as ``__pydantic_config__``.
SCHEMA_CONFIG = ConfigDict(extra='forbid')


def _range(expected: str, test: Callable[[Any], bool]) -> AfterValidator:
    """A validator that refuses a value outside a named range, in the catalog's words.

    Parameters:
        expected: What the value should be, such as ``'a positive number'``.
        test: Whether a value lies in the range.

    Returns:
        The validator, to put in an ``Annotated``.
    """

    def check(value: Any) -> Any:
        """Return ``value`` if it lies in the range; raise the range error if not."""
        if not test(value):
            raise RangeError(expected)
        return value

    return AfterValidator(check)


#: A finite float.  An integer is accepted and becomes a float.
Float = Annotated[float, Strict(), AllowInfNan(False)]
#: An integer; a float or a boolean fails.
Int = Annotated[int, Strict()]
#: A boolean; a number or a string fails.
Bool = Annotated[bool, Strict()]
#: A string; a number fails.
Str = Annotated[str, Strict()]
#: A finite float above zero.
Positive = Annotated[Float, _range('a positive number', lambda value: value > 0)]
#: A finite float at or above zero.
NonNeg = Annotated[Float, _range('a non-negative number', lambda value: value >= 0)]
#: A finite float in [0, 1].
Unit = Annotated[Float, _range('a number in [0, 1]', lambda value: 0 <= value <= 1)]
#: An integer above zero.
PositiveInt = Annotated[Int, _range('a positive integer', lambda value: value > 0)]
#: An integer at or above zero.
NonNegInt = Annotated[Int, _range('a non-negative integer', lambda value: value >= 0)]

# A string that spells a non-negative integer in ASCII digits, with no leading zero, so
# no two such strings stand for one integer.
_DIGITS = re.compile(r'0|[1-9][0-9]*')


def _digits_to_int(value: object) -> object:
    """Turn a string of digits into its integer; return anything else unchanged."""
    if isinstance(value, str) and _DIGITS.fullmatch(value) is not None:
        return int(value)
    return value


def _to_fcpath(value: object) -> filecache.FCPath:
    """Turn a path or URL string into an ``FCPath``.

    Raises:
        ValueError: If ``value`` is neither a non-empty string nor an ``FCPath``.
    """
    if isinstance(value, filecache.FCPath):
        return value
    if isinstance(value, str) and value != '':
        return filecache.FCPath(value)
    raise ValueError(f'expected a path or URL, got {describe_value(value)}')


# The validators of DigitInt and FCPath, which the schema check recognizes by identity.
_DIGIT_VALIDATOR = BeforeValidator(_digits_to_int)
_FCPATH_VALIDATOR = PlainValidator(_to_fcpath)

#: An integer, also written as a string of digits with no leading zero.  Use it for
#: mapping keys; :func:`parse` refuses a mapping that writes one key both ways.
DigitInt = Annotated[int, _DIGIT_VALIDATOR, Strict()]
#: A local path or a URL, read from a non-empty string.
FCPath = Annotated[filecache.FCPath, _FCPATH_VALIDATOR]

# The scalar types a field must not use bare, and the alias that replaces each.
_STRICT_ALIASES: dict[type, str] = {float: 'Float', int: 'Int', bool: 'Bool', str: 'Str'}

# The generic containers a schema may use.
_CONTAINERS: frozenset[Any] = frozenset({list, tuple, dict, Mapping})

# Validators that run before the type check, and so can make a lenient read.
_EARLY_VALIDATORS = (BeforeValidator, PlainValidator, WrapValidator)


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


def check_schema(cls: object) -> None:
    """Check that a schema follows the conventions in this module's docstring.

    Parameters:
        cls: The schema's root type, normally a dataclass.

    Raises:
        TypeError: If a dataclass in the schema is not frozen or does not set
            ``__pydantic_config__ = SCHEMA_CONFIG``; if a scalar would be read
            leniently (a bare ``float``, ``int``, ``bool``, or ``str``, a
            ``Strict(False)`` or ``AllowInfNan(True)`` marker, a float that may be
            infinite, or a validator that runs before the type check); if a choice
            has a value that is not a string; if a union of two or more dataclasses
            is not tagged, or a tag does not name a ``Literal`` field of every
            member; if a path is a bare ``FCPath``; or if an annotation is not one of
            the supported forms.
    """
    _check_annotation(cls, where=getattr(cls, '__qualname__', repr(cls)), seen=set())


def _flatten(extras: tuple[Any, ...]) -> list[Any]:
    """``Annotated`` extras, with each ``Field(...)`` replaced by the markers it holds."""
    flat: list[Any] = []
    for extra in extras:
        flat.append(extra)
        if isinstance(extra, FieldInfo):
            flat.extend(extra.metadata)
    return flat


def _check_annotation(annotation: Any, *, where: str, seen: set[type]) -> None:
    """Check one annotation and everything inside it; ``where`` locates it in messages."""
    base = annotation
    extras: list[Any] = []
    if get_origin(annotation) is Annotated:
        base, *rest = get_args(annotation)
        extras = _flatten(tuple(rest))
    tag: str | None = None
    for extra in extras:
        if isinstance(extra, FieldInfo) and isinstance(extra.discriminator, str):
            tag = extra.discriminator
    origin = get_origin(base)
    if base in _STRICT_ALIASES:
        _check_scalar(base, extras, where=where)
        return
    if any(isinstance(extra, Strict | AllowInfNan) for extra in extras):
        raise TypeError(f'{where}: Strict and AllowInfNan apply only to the scalar aliases')
    is_union = origin is Union or origin is types.UnionType
    if tag is not None and not is_union:
        raise TypeError(f'{where}: tagged() marks a union of schema dataclasses')
    if base is filecache.FCPath:
        if _FCPATH_VALIDATOR not in extras:
            raise TypeError(f'{where}: a bare FCPath cannot be read; use the schema alias FCPath')
        return
    if base is type(None):
        return
    if origin is Literal:
        _check_choices(get_args(base), where=where)
        return
    if isinstance(base, type) and issubclass(base, enum.Enum):
        _check_choices(tuple(member.value for member in base), where=where)
        return
    if is_union:
        _check_union(get_args(base), tag, where=where, seen=seen)
        return
    if origin in _CONTAINERS:
        for argument in get_args(base):
            if argument is not Ellipsis:
                _check_annotation(argument, where=where, seen=seen)
        return
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        _check_dataclass(base, seen=seen)
        return
    raise TypeError(f'{where}: {base!r} is not a supported schema annotation')


def _check_scalar(base: type, extras: list[Any], *, where: str) -> None:
    """Check that a scalar annotation reads its values strictly."""
    alias = _STRICT_ALIASES[base]
    for extra in extras:
        if getattr(extra, 'strict', None) is False:
            raise TypeError(f'{where}: Strict(False) reads values leniently; use {alias}')
        if getattr(extra, 'allow_inf_nan', None) is True:
            raise TypeError(f'{where}: AllowInfNan(True) lets a float be infinite; use {alias}')
        if isinstance(extra, _EARLY_VALIDATORS) and extra is not _DIGIT_VALIDATOR:
            raise TypeError(
                f'{where}: a validator that runs before the type check reads values '
                f'leniently; check the value after the type with AfterValidator'
            )
    if not any(isinstance(extra, Strict) for extra in extras):
        raise TypeError(
            f'{where}: a bare {base.__name__} reads values leniently; use the schema alias {alias}'
        )
    if base is float and not any(isinstance(extra, AllowInfNan) for extra in extras):
        raise TypeError(f'{where}: a float must be finite; use the schema alias Float')


def _check_choices(values: tuple[Any, ...], *, where: str) -> None:
    """Check that every value of a Literal or an Enum is a string.

    A non-string choice would accept a value of another type that compares equal to
    it, such as ``True`` or ``1.0`` for ``1``.
    """
    others = [value for value in values if not isinstance(value, str)]
    if others:
        raise TypeError(f'{where}: a choice holds only strings, and {others!r} are not')


def _check_union(members: tuple[Any, ...], tag: str | None, *, where: str, seen: set[type]) -> None:
    """Check a union: its members, and that a union of dataclasses is tagged by a Literal."""
    for member in members:
        _check_annotation(member, where=where, seen=seen)
    records = [member for member in members if _is_dataclass(member)]
    if tag is None:
        if len(records) > 1:
            raise TypeError(
                f'{where}: a union of two or more schema dataclasses is tagged; write it '
                f"as Annotated[A | B, tagged('kind')]"
            )
        return
    if len(records) != len(members):
        raise TypeError(f'{where}: every member of a tagged union is a schema dataclass')
    for record in records:
        hint = get_type_hints(record, include_extras=True).get(tag)
        if get_origin(hint) is not Literal:
            raise TypeError(
                f'{where}: the tag {tag!r} is not a Literal field of {record.__qualname__}'
            )


def _is_dataclass(annotation: Any) -> bool:
    """Whether an annotation, without its ``Annotated`` extras, is a dataclass."""
    base = get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    return isinstance(base, type) and dataclasses.is_dataclass(base)


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
        cls: The schema's root type, built once at import (the checked adapter of each
            class is kept for the life of the process).
        raw: The mapping to check, normally the data of a merged set of layers.
        source: The file (or file and entry) to give in a message when ``origins``
            says nothing about the key.
        origins: The source of each value, keyed by the value's key path, as the layer
            merge records it.  A problem with a value, or with an unknown key, gives
            the source that set it.  A problem with a key that is absent, or with a
            whole mapping, gives every source that set a value inside that mapping.

    Returns:
        The instance of ``cls``.

    Raises:
        SchemaError: Listing every problem: each unknown key, missing required key,
            value of the wrong type or out of range, broken cross-field rule, and
            integer key written twice (as ``2`` and as ``'2'``) in one mapping.
        TypeError: If ``cls`` breaks the schema conventions (see :func:`check_schema`).
    """
    # A class is hashable; typeshed's type[T] does not say so.
    adapter = _adapter(cast(Hashable, cls))
    doubled = doubled_keys(raw, source=source, origins=origins or {})
    try:
        value: T = adapter.validate_python(raw)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        found = report(errors, raw, source=source, origins=origins or {})
        raise SchemaError([*doubled, *found]) from exc
    if doubled:
        raise SchemaError(doubled)
    return value
