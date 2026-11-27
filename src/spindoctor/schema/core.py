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
  is never a number, and a float must be finite.  :data:`FloatOrNan` is the one float
  that may also be NaN, for a value where NaN has a meaning of its own.  Strictness
  lives on the fields through these aliases, never on a dataclass's configuration,
  because a strict dataclass accepts only an instance of itself and refuses a mapping.
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
  message at that type's key path.  To put the problem on the key at fault instead,
  it raises :class:`~spindoctor.schema.errors.RuleError` with that key's path relative
  to the type, as in ``RuleError('not in the table', at=('filters', 'clear'))``.  The
  line then gives the file that set that key, so a problem every entry inherits from
  one base reads the same in each.  A rule that compares keys passes all of them as
  ``reads``, and the line gives the file that set or removed one of them last.  Any
  other exception is a bug in the rule, and it propagates.
- **Give a field an alias** with ``Annotated[X, Field(alias='from')]`` only when the
  key a file writes cannot be a field name, because it is a Python keyword such as
  ``from``.  The field takes another name, such as ``from_``, and a mapping writes the
  alias.
- **Write a polymorphic entry as a tagged union**: every member is a schema dataclass
  with a ``Literal`` tag field, and the union is ``Annotated[A | B, tagged('kind')]``.
  A union of two or more dataclasses must be tagged.
- **Carry markers for other code** in ``dataclasses.field(metadata={...})``.  The core
  never reads them; :func:`spindoctor.schema.metadata.walk_metadata` collects them.
- **Build a schema once, at import.**  :func:`parse` keeps the checked pydantic adapter
  of every class it has seen, keyed by the class, for the life of the process.

The supported annotations are the aliases above, ``Literal`` and ``Enum`` of strings,
``X | None``, unions, ``list[X]``, ``tuple[X, ...]``, fixed tuples, ``dict[K, V]``,
``Mapping[K, V]``, nested schema dataclasses (a schema may refer to itself),
``annotated_types`` constraints such as ``Le`` and ``Len``, and ``Field(alias=...)``
for a key that is a Python keyword.  :func:`check_schema` refuses everything else, and
also refuses what would read a value leniently: a bare ``float``, ``int``, ``bool``, or
``str``, a ``Strict(False)`` or ``AllowInfNan(True)`` marker other than the one
:data:`FloatOrNan` carries, and a validator that runs before the type check, on any
type.  A default that does not fit its field, and an
``InitVar``, are refused too.  :func:`parse` runs that check the first time it sees a
class.
Adding a new form or convention needs a reason, written here.

A null is never data in the files this core reads.  The layer merge
(:mod:`spindoctor.schema.layers`) applies each file family's rule for null before
parsing, so ``X | None`` only describes an optional key that is absent.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
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
    'FloatOrNan',
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

# The markers of FloatOrNan, which the schema check recognizes by identity: NaN is let
# through the type check, and infinity is refused after it.
_ALLOW_NAN = AllowInfNan(True)
_NOT_INFINITE = _range('a finite number or NaN', lambda value: not math.isinf(value))
#: A finite float or NaN, for a value where NaN has a meaning of its own.  An integer is
#: accepted and becomes a float, and an infinite float fails.
FloatOrNan = Annotated[float, Strict(), _ALLOW_NAN, _NOT_INFINITE]
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
_CORE_VALIDATORS = (_DIGIT_VALIDATOR, _FCPATH_VALIDATOR)

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
            ``Strict(False)`` or ``AllowInfNan(True)`` marker other than
            :data:`FloatOrNan`'s, a float that may be infinite, or a validator that
            runs before the type check); if a choice has a value that is not a
            string; if a union of two or more dataclasses is not tagged, or a tag does
            not name a ``Literal`` field of every member; if a path is a bare
            ``FCPath``; if an annotation is not one of the supported forms; if a
            dataclass has an ``InitVar``; or if a field's default does not fit its
            annotation.
    """
    seen: set[type] = set()
    _check_annotation(cls, where=getattr(cls, '__qualname__', repr(cls)), seen=seen)
    for record in seen:
        _check_defaults(record)


def _flatten(extras: tuple[Any, ...]) -> list[Any]:
    """``Annotated`` extras, with each ``Field(...)`` replaced by the markers it holds."""
    flat: list[Any] = []
    for extra in extras:
        flat.append(extra)
        if isinstance(extra, FieldInfo):
            flat.extend(extra.metadata)
    return flat


def _unwrap(annotation: Any) -> tuple[Any, list[Any]]:
    """An annotation without its ``Annotated`` extras, and the extras, flattened."""
    if get_origin(annotation) is not Annotated:
        return annotation, []
    base, *extras = get_args(annotation)
    return base, _flatten(tuple(extras))


def _tag_key(extras: list[Any]) -> str | None:
    """The tag field that :func:`tagged` gives among ``Annotated`` extras, or None."""
    for extra in extras:
        if isinstance(extra, FieldInfo) and isinstance(extra.discriminator, str):
            return extra.discriminator
    return None


def _check_annotation(annotation: Any, *, where: str, seen: set[type]) -> None:
    """Check one annotation and everything inside it; ``where`` locates it in messages."""
    base, extras = _unwrap(annotation)
    for extra in extras:
        if isinstance(extra, _EARLY_VALIDATORS) and extra not in _CORE_VALIDATORS:
            raise TypeError(
                f'{where}: a validator that runs before the type check reads values '
                f'leniently; check the value after the type with AfterValidator'
            )
    tag = _tag_key(extras)
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
        if getattr(extra, 'allow_inf_nan', None) is True and not (
            extra is _ALLOW_NAN and _NOT_INFINITE in extras
        ):
            raise TypeError(f'{where}: AllowInfNan(True) lets a float be infinite; use {alias}')
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
    for name, hint in hints.items():
        if isinstance(hint, dataclasses.InitVar):
            raise TypeError(f'{cls.__qualname__}.{name}: a schema has no InitVar; use a field')
    for item in dataclasses.fields(cls):
        _check_annotation(hints[item.name], where=f'{cls.__qualname__}.{item.name}', seen=seen)


def _check_defaults(cls: type) -> None:
    """Check each default of a schema dataclass against its field's annotation.

    pydantic does not check a default, so a key left out would take a value that a file
    could not write.
    """
    defaults: dict[str, Any] = {}
    for key, item in zip(_field_hints(cls), dataclasses.fields(cls), strict=True):
        if item.default is not dataclasses.MISSING:
            defaults[key] = item.default
        elif item.default_factory is not dataclasses.MISSING:
            defaults[key] = item.default_factory()
    try:
        TypeAdapter(cls).validate_python(defaults)
    except ValidationError as exc:
        # A required field is missing here, and a rule may refuse the defaults alone.
        bad = {
            str(error['loc'][0])
            for error in exc.errors(include_url=False)
            if error['loc'] and (error['type'] != 'missing' or len(error['loc']) > 1)
        }
        if bad:
            raise TypeError(
                f'{cls.__qualname__}: the defaults of {sorted(bad)} do not fit their annotations'
            ) from exc


@functools.cache
def _adapter(cls: Any) -> TypeAdapter[Any]:
    """The checked, cached pydantic adapter for one schema root."""
    check_schema(cls)
    return TypeAdapter(cls)


def _types_at(annotation: Any, keys: KeyPath) -> list[tuple[Any, list[Any]]]:
    """Each type ``annotation`` allows at ``keys``, without and with its extras.

    The keys are the input's own: a union is followed into each of its members, since
    the input does not say which member a value was checked against.

    Parameters:
        annotation: A schema annotation.
        keys: A key path into a value of that annotation.

    Returns:
        The types, as :func:`_unwrap` gives them; a union, and then its members.
    """
    base, extras = _unwrap(annotation)
    origin = get_origin(base)
    arguments = get_args(base)
    if origin is Union or origin is types.UnionType:
        members = [found for member in arguments for found in _types_at(member, keys)]
        return members if keys else [(base, extras), *members]
    if not keys:
        return [(base, extras)]
    key, rest = keys[0], keys[1:]
    if isinstance(base, type) and dataclasses.is_dataclass(base):
        hint = _field_hints(base).get(key)
        return [] if hint is None else _types_at(hint, rest)
    if origin in (dict, Mapping):
        return _types_at(arguments[1], rest)
    if origin is tuple and Ellipsis not in arguments:
        inside = isinstance(key, int) and 0 <= key < len(arguments)
        return _types_at(arguments[key], rest) if inside else []
    if origin in (list, tuple):
        return _types_at(arguments[0], rest)
    return []


class _View:
    """A schema's :class:`~spindoctor.schema.errors.SchemaView`.

    Parameters:
        cls: The schema's root type.
    """

    def __init__(self, cls: Any) -> None:
        self._cls = cls

    def lengths(self, keys: KeyPath) -> frozenset[tuple[int, int | None]]:
        """The (fewest, most) items of each list type allowed at ``keys``; None for no most."""
        found: set[tuple[int, int | None]] = set()
        for base, extras in _types_at(self._cls, keys):
            arguments = get_args(base)
            if get_origin(base) not in (list, tuple):
                continue
            if get_origin(base) is tuple and Ellipsis not in arguments:
                found.add((len(arguments), len(arguments)))
                continue
            fewest = max((getattr(extra, 'min_length', 0) or 0 for extra in extras), default=0)
            most = [getattr(extra, 'max_length', None) for extra in extras]
            limits = [int(limit) for limit in most if limit is not None]
            found.add((int(fewest), min(limits, default=None)))
        return frozenset(found)

    def tag_keys(self, keys: KeyPath) -> frozenset[str]:
        """The tag field of each tagged union at ``keys``."""
        tags = [_tag_key(extras) for _, extras in _types_at(self._cls, keys)]
        return frozenset(tag for tag in tags if tag is not None)

    def digit_keyed(self, keys: KeyPath) -> bool:
        """Whether a mapping at ``keys`` reads a string of digits as an integer key."""
        return any(
            get_origin(base) in (dict, Mapping)
            and _DIGIT_VALIDATOR in _unwrap(get_args(base)[0])[1]
            for base, _ in _types_at(self._cls, keys)
        )


def _field_hints(cls: type) -> dict[str, Any]:
    """A schema dataclass's field annotations, keyed by the key a mapping writes.

    That key is the field's name, or its alias when ``Field(alias=...)`` gives one.
    """
    hints = get_type_hints(cls, include_extras=True)
    keyed: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        key = item.name
        hint = hints[item.name]
        if get_origin(hint) is Annotated:
            for extra in get_args(hint)[1:]:
                if isinstance(extra, FieldInfo) and isinstance(extra.alias, str):
                    key = extra.alias
        keyed[key] = hint
    return keyed


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
        origins: The source of each value, and of each key a null removed, keyed by
            key path, as the layer merge records them.  A problem with a value, or with
            an unknown key, gives the source that set it.  A problem with a key that a
            null removed gives the source that removed it.  A problem with any other
            absent key, or with a whole mapping, gives every source that set or
            removed a value inside that mapping.  A rule that compares keys gives the
            source that comes last in ``origins`` among them, so the order of
            ``origins`` is the order of the layers.

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
    view = _View(cls)
    doubled = doubled_keys(raw, source=source, origins=origins or {}, schema=view)
    try:
        value: T = adapter.validate_python(raw)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        found = report(errors, raw, source=source, origins=origins or {}, schema=view)
        raise SchemaError([*doubled, *found]) from exc
    if doubled:
        raise SchemaError(doubled)
    return value
