"""Tests for :mod:`spindoctor.schema.core`: the aliases and the schema conventions.

The tests cover what the core adds on top of pydantic: how strictly each alias reads
a value, how it words a refusal, and which schemas the convention check refuses.
They do not test pydantic's own parsing.
"""

import enum
from dataclasses import InitVar, dataclass, field
from typing import Annotated, Any, Literal

import filecache
import pytest
from pydantic import AllowInfNan, BeforeValidator, Field, PlainValidator, Strict
from tests.spindoctor.schema.schemas import (
    OUTER,
    Aliases,
    Gap,
    Inner,
    Model,
    Outer,
    Ringlet,
    parse_error,
)

from spindoctor.schema.core import (
    SCHEMA_CONFIG,
    Float,
    Int,
    Positive,
    Str,
    check_schema,
    parse,
    tagged,
)


@pytest.mark.parametrize(
    ('raw', 'message'),
    [
        (
            {'float_': '0.5'},
            "test.yaml: float_: expected a number, got str '0.5' (a number written so YAML "
            'reads it as a string)',
        ),
        ({'float_': True}, 'test.yaml: float_: expected a number, got bool True'),
        ({'float_': float('inf')}, 'test.yaml: float_: expected a finite number, got float inf'),
        ({'int_': 1.0}, 'test.yaml: int_: expected an integer, got float 1.0'),
        ({'bool_': 1}, 'test.yaml: bool_: expected a boolean, got int 1'),
        ({'str_': 3}, 'test.yaml: str_: expected a string, got int 3'),
        ({'positive': 0}, 'test.yaml: positive: expected a positive number, got 0'),
        ({'nonneg': -1}, 'test.yaml: nonneg: expected a non-negative number, got -1'),
        ({'unit': -0.5}, 'test.yaml: unit: expected a number in [0, 1], got -0.5'),
        ({'unit': 2}, 'test.yaml: unit: expected a number in [0, 1], got 2'),
        ({'positive_int': 0}, 'test.yaml: positive_int: expected a positive integer, got 0'),
        (
            {'nonneg_int': -1},
            'test.yaml: nonneg_int: expected a non-negative integer, got -1',
        ),
        (
            {'table': {'x': 'a'}},
            "test.yaml: table.x: expected an integer as the key, got str 'x'",
        ),
        (
            {'table': {'02': 'a'}},
            "test.yaml: table.02: expected an integer as the key, got str '02' (a number "
            'written so YAML reads it as a string)',
        ),
        ({'path': 3}, 'test.yaml: path: expected a path or URL, got int 3'),
        ({'path': ''}, "test.yaml: path: expected a path or URL, got str ''"),
        ({'model': 'cmos'}, "test.yaml: model: 'cmos' is not one of ['ccd', 'vidicon']"),
    ],
    ids=[
        'float-string',
        'float-bool',
        'float-infinite',
        'int',
        'bool',
        'str',
        'positive',
        'nonneg',
        'unit-low',
        'unit-high',
        'positive-int',
        'nonneg-int',
        'digit-key',
        'digit-key-leading-zero',
        'path',
        'path-empty',
        'enum',
    ],
)
def test_alias_refuses_value(raw: dict[str, Any], message: str) -> None:
    """Each alias refuses a value outside its type or range, in the catalog's words."""
    assert str(parse_error(Aliases, raw)) == message


@pytest.mark.parametrize(
    ('raw', 'attribute', 'value'),
    [
        ({'nonneg_int': 0}, 'nonneg_int', 0),
        ({'table': {'2': 'a'}}, 'table', {2: 'a'}),
        ({'path': 'gs://bucket/dir'}, 'path', filecache.FCPath('gs://bucket/dir')),
        ({'path': filecache.FCPath('/data')}, 'path', filecache.FCPath('/data')),
        ({'model': 'vidicon'}, 'model', Model.VIDICON),
        ({'names': {'a': 1}}, 'names', {'a': 1}),
    ],
    ids=['nonneg-int-zero', 'digit-key', 'path-url', 'path-instance', 'enum', 'mapping'],
)
def test_alias_reads_value(raw: dict[str, Any], attribute: str, value: object) -> None:
    """Each alias reads the values its type allows."""
    assert getattr(parse(Aliases, raw, source='test.yaml'), attribute) == value


def test_float_reads_an_integer_as_a_float() -> None:
    """A Float field turns a YAML integer into a float."""
    parsed = parse(Aliases, {'float_': 2}, source='test.yaml')
    assert isinstance(parsed.float_, float)


def test_parse_builds_the_dataclass() -> None:
    """A valid mapping becomes an instance of the schema, nested records included."""
    parsed = parse(Outer, {**OUTER, 'features': [{'kind': 'gap'}]}, source='test.yaml')
    assert parsed == Outer(name='n', inner=Inner(low=0.0, high=1.0), features=(Gap('gap'),))


@dataclass(frozen=True)
class Node:
    """A schema that refers to itself."""

    __pydantic_config__ = SCHEMA_CONFIG
    name: Str
    children: tuple['Node', ...] = ()


def test_parse_reads_a_recursive_schema() -> None:
    """A schema that refers to itself passes the check and parses nested data."""
    parsed = parse(Node, {'name': 'a', 'children': [{'name': 'b'}]}, source='test.yaml')
    assert parsed.children[0].name == 'b'


class _IntChoice(enum.Enum):
    """An Enum whose value is not a string."""

    ONE = 1


def _schema(annotation: Any) -> type:
    """A schema dataclass with one optional field ``value`` of the given annotation."""
    namespace = {'__annotations__': {'value': annotation}, '__pydantic_config__': SCHEMA_CONFIG}
    return dataclass(frozen=True)(type('Probe', (), namespace))


@dataclass
class _NotFrozen:
    """A dataclass that is not frozen."""

    __pydantic_config__ = SCHEMA_CONFIG
    value: Float = 0.0


@dataclass(frozen=True)
class _NoConfig:
    """A dataclass that does not refuse unknown keys."""

    value: Float = 0.0


@dataclass(frozen=True)
class _Untagged:
    """A union of two dataclasses without a tag."""

    __pydantic_config__ = SCHEMA_CONFIG
    value: Ringlet | Gap


@dataclass(frozen=True)
class _TagNotLiteral:
    """A tagged union whose tag is not a Literal field of every member."""

    __pydantic_config__ = SCHEMA_CONFIG
    value: Annotated[Ringlet | Inner, tagged('kind')]


@dataclass(frozen=True)
class _BadDefault:
    """A schema whose default is a value a file could not write."""

    __pydantic_config__ = SCHEMA_CONFIG
    required: Int
    count: Positive = 0.0


@dataclass(frozen=True)
class _InitOnly:
    """A schema with an input that is not a field."""

    __pydantic_config__ = SCHEMA_CONFIG
    count: InitVar[int]


@dataclass(frozen=True)
class _NestedBreak:
    """A schema whose nested record breaks a convention."""

    __pydantic_config__ = SCHEMA_CONFIG
    items: list[_NotFrozen] = field(default_factory=list)


@pytest.mark.parametrize(
    ('cls', 'match'),
    [
        (_NotFrozen, r'^_NotFrozen: a schema dataclass must be frozen$'),
        (_NoConfig, r'^_NoConfig: a schema dataclass sets __pydantic_config__ = SCHEMA_CONFIG$'),
        (
            _schema(float),
            r'Probe.value: a bare float reads values leniently; use the schema '
            r'alias Float',
        ),
        (_schema(Annotated[float, Strict()]), r'Probe.value: a float must be finite'),
        (
            _schema(Annotated[float, Strict(False), AllowInfNan(False)]),
            r'Strict\(False\) reads '
            r'values leniently',
        ),
        (
            _schema(Annotated[Float, Field(strict=False)]),
            r'Strict\(False\) reads values '
            r'leniently',
        ),
        (
            _schema(Annotated[Float, AllowInfNan(True)]),
            r'AllowInfNan\(True\) lets a float be '
            r'infinite',
        ),
        (
            _schema(Annotated[Float, BeforeValidator(float)]),
            r'a validator that runs before '
            r'the type check reads values leniently',
        ),
        (
            _schema(Annotated[list[Int], PlainValidator(lambda value: value)]),
            r'Probe.value: a validator that runs before the type check reads values leniently',
        ),
        (_BadDefault, r"^_BadDefault: the defaults of \['count'\] do not fit their annotations$"),
        (_InitOnly, r'^_InitOnly.count: a schema has no InitVar; use a field$'),
        (_schema(Literal['a', 1]), r'a choice holds only strings, and \[1\] are not'),
        (_schema(_IntChoice), r'a choice holds only strings, and \[1\] are not'),
        (
            _schema(Annotated[Literal['a'], Strict()]),
            r'Strict and AllowInfNan apply only to '
            r'the scalar aliases',
        ),
        (_Untagged, r'a union of two or more schema dataclasses is tagged'),
        (_TagNotLiteral, r"the tag 'kind' is not a Literal field of Inner"),
        (
            _schema(Annotated[Ringlet, tagged('kind')]),
            r'tagged\(\) marks a union of schema '
            r'dataclasses',
        ),
        (
            _schema(filecache.FCPath),
            r'a bare FCPath cannot be read; use the schema alias '
            r'FCPath',
        ),
        (_schema(set[Int]), r'Probe.value: set\[.*\] is not a supported schema annotation'),
        (_NestedBreak, r'^_NotFrozen: a schema dataclass must be frozen$'),
    ],
    ids=[
        'not-frozen',
        'no-config',
        'bare-scalar',
        'float-may-be-infinite',
        'strict-false',
        'field-strict-false',
        'allow-inf-nan',
        'early-validator',
        'early-validator-on-a-container',
        'default-out-of-range',
        'init-var',
        'literal-not-string',
        'enum-not-string',
        'strict-on-literal',
        'untagged-union',
        'tag-not-literal',
        'tagged-not-union',
        'bare-fcpath',
        'unsupported',
        'nested',
    ],
)
def test_check_schema_refuses_a_convention_break(cls: type, match: str) -> None:
    """A schema that breaks a convention is refused when checked."""
    with pytest.raises(TypeError, match=match):
        check_schema(cls)


def test_parse_checks_the_schema() -> None:
    """parse refuses a schema that breaks a convention before reading any input."""
    with pytest.raises(TypeError, match='a bare float reads values leniently'):
        parse(_schema(float), {}, source='test.yaml')
