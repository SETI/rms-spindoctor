"""Tests for :mod:`spindoctor.schema.core`: the aliases, the error report, and the checks.

The tests cover what the core adds on top of pydantic: the house error format, where
each problem is placed, which file it names, and the schema conventions.  They do not
test pydantic's own parsing.
"""

import pickle
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import pytest
from filecache import FCPath as FileCachePath

from spindoctor.schema.core import (
    SCHEMA_CONFIG,
    Bool,
    DigitInt,
    FCPath,
    Float,
    Int,
    Issue,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    SchemaError,
    Str,
    Unit,
    check_schema,
    parse,
    tagged,
)


@dataclass(frozen=True)
class Aliases:
    """One optional field per alias, so each test sets only the key it is about."""

    __pydantic_config__ = SCHEMA_CONFIG
    float_: Float = 0.0
    int_: Int = 0
    bool_: Bool = False
    str_: Str = ''
    positive: Positive = 1.0
    nonneg: NonNeg = 0.0
    unit: Unit = 0.0
    positive_int: PositiveInt = 1
    nonneg_int: NonNegInt = 0
    table: dict[DigitInt, Str] = field(default_factory=dict)
    path: FCPath | None = None


@dataclass(frozen=True)
class Ringlet:
    """A tagged-union member with a required field."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['ringlet']
    width: Positive


@dataclass(frozen=True)
class Gap:
    """A second tagged-union member."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['gap']


@dataclass(frozen=True)
class Inner:
    """A nested record with a cross-field rule."""

    __pydantic_config__ = SCHEMA_CONFIG
    low: Float
    high: Float

    def __post_init__(self) -> None:
        """Refuse a range whose ends are reversed."""
        if self.low > self.high:
            raise ValueError(f'low ({self.low}) is above high ({self.high})')


@dataclass(frozen=True)
class Outer:
    """A schema with nesting, a list of tagged entries, a union, and a literal."""

    __pydantic_config__ = SCHEMA_CONFIG
    name: Str
    inner: Inner
    features: tuple[Annotated[Ringlet | Gap, tagged('kind')], ...] = ()
    either: Int | Str = 0
    units: Literal['raw_dn', 'calibrated_if'] = 'raw_dn'


# A valid Outer mapping, which each test breaks in one place.
_OUTER: dict[str, Any] = {'name': 'n', 'inner': {'low': 0.0, 'high': 1.0}}


def _message(cls: type, raw: object, **kwargs: Any) -> str:
    """The message of the SchemaError that parsing ``raw`` raises.

    Parameters:
        cls: The schema.
        raw: The mapping to parse, which must break the schema.
        **kwargs: ``source`` (default ``'test.yaml'``) and ``origins``.

    Returns:
        The error message.
    """
    kwargs.setdefault('source', 'test.yaml')
    with pytest.raises(SchemaError) as caught:
        parse(cls, raw, **kwargs)
    return str(caught.value)


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
        ({'unit': 2}, 'test.yaml: unit: expected a number of at most 1.0, got 2'),
        ({'positive_int': 0}, 'test.yaml: positive_int: expected a positive number, got 0'),
        ({'nonneg_int': 1.5}, 'test.yaml: nonneg_int: expected an integer, got float 1.5'),
        (
            {'table': {'x': 'a'}},
            "test.yaml: table.x: expected an integer as the key, got str 'x'",
        ),
        ({'path': 3}, 'test.yaml: path: expected a path or URL, got int 3'),
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
        'unit',
        'positive-int',
        'nonneg-int',
        'digit-key',
        'path',
    ],
)
def test_alias_refuses_value(raw: dict[str, Any], message: str) -> None:
    """Each alias refuses a value outside its type or range, in the house phrasing."""
    assert _message(Aliases, raw) == message


def test_float_reads_an_integer_as_a_float() -> None:
    """A Float field turns a YAML integer into a float."""
    parsed = parse(Aliases, {'float_': 2}, source='test.yaml')
    assert isinstance(parsed.float_, float)


def test_digit_int_reads_digits_as_an_integer() -> None:
    """A DigitInt key written as digits, as JSON writes every key, becomes an integer."""
    parsed = parse(Aliases, {'table': {'2': 'a'}}, source='test.yaml')
    assert parsed.table == {2: 'a'}


def test_fcpath_reads_a_url() -> None:
    """An FCPath field turns a string into an FCPath."""
    parsed = parse(Aliases, {'path': 'gs://bucket/dir'}, source='test.yaml')
    assert parsed.path == FileCachePath('gs://bucket/dir')


def test_parse_builds_the_dataclass() -> None:
    """A valid mapping becomes an instance of the schema, nested records included."""
    parsed = parse(Outer, {**_OUTER, 'features': [{'kind': 'gap'}]}, source='test.yaml')
    assert parsed == Outer(name='n', inner=Inner(low=0.0, high=1.0), features=(Gap('gap'),))


def test_every_problem_is_reported_at_once() -> None:
    """Two broken keys give one issue each."""
    with pytest.raises(SchemaError) as caught:
        parse(Aliases, {'int_': 'a', 'unit': -1}, source='test.yaml')
    assert len(caught.value.issues) == 2


@pytest.mark.parametrize(
    ('raw', 'message'),
    [
        (
            {**_OUTER, 'features': [{'kind': 'gap'}, {'kind': 'ringlet', 'width': -1}]},
            'test.yaml: features[1].width: expected a positive number, got -1',
        ),
        (
            {**_OUTER, 'features': [{'kind': 'ring'}]},
            "test.yaml: features[0].kind: expected one of 'ringlet', 'gap', got 'ring'",
        ),
        (
            {**_OUTER, 'features': [{'width': 1}]},
            "test.yaml: features[0]: missing required keys: ['kind']",
        ),
        (
            {**_OUTER, 'inner': {'low': 0.0, 'high': 1.0, 'mid': 0.5, 'avg': 0.5}},
            "test.yaml: inner: unknown keys: ['avg', 'mid']",
        ),
        ({'inner': _OUTER['inner']}, "test.yaml: missing required keys: ['name']"),
        ({**_OUTER, 'inner': 3}, 'test.yaml: inner: expected a mapping, got int 3'),
        ({**_OUTER, 'features': 'gap'}, "test.yaml: features: expected a list, got str 'gap'"),
        (
            {**_OUTER, 'either': 1.5},
            'test.yaml: either: expected an integer or a string, got float 1.5',
        ),
        (
            {**_OUTER, 'units': 'dn'},
            "test.yaml: units: expected 'raw_dn' or 'calibrated_if', got str 'dn'",
        ),
        (
            {**_OUTER, 'inner': {'low': 2.0, 'high': 1.0}},
            'test.yaml: inner: low (2.0) is above high (1.0)',
        ),
    ],
    ids=[
        'tagged-member-field',
        'tagged-unknown-tag',
        'tagged-missing-tag',
        'unknown-keys',
        'missing-keys',
        'not-a-mapping',
        'not-a-list',
        'union-of-scalars',
        'literal',
        'cross-field-rule',
    ],
)
def test_problem_is_placed_and_phrased(raw: dict[str, Any], message: str) -> None:
    """Each kind of problem is reported once, at the key path it concerns."""
    assert _message(Outer, raw) == message


def test_problem_gives_the_file_that_set_the_value() -> None:
    """A problem with a value gives the layer that set it."""
    origins = {
        ('name',): 'base.yaml',
        ('inner', 'low'): 'base.yaml',
        ('inner', 'high'): 'user.yaml',
    }
    raw = {'name': 'n', 'inner': {'low': 0.0, 'high': 'x'}}
    assert _message(Outer, raw, origins=origins).startswith('user.yaml: inner.high: ')


def test_problem_inside_a_list_gives_the_file_that_set_the_list() -> None:
    """A problem inside a list gives the layer that set the whole list."""
    origins = {
        ('name',): 'base.yaml',
        ('inner', 'low'): 'base.yaml',
        ('inner', 'high'): 'base.yaml',
        ('features',): 'user.yaml',
    }
    raw = {**_OUTER, 'features': [{'kind': 'ringlet', 'width': 0}]}
    assert _message(Outer, raw, origins=origins).startswith('user.yaml: features[0].width: ')


def test_unknown_key_gives_the_file_that_set_it() -> None:
    """An unknown key gives the layer that set it, though the line is about its mapping."""
    origins = {('name',): 'base.yaml', ('inner', 'low'): 'base.yaml', ('title',): 'user.yaml'}
    raw = {**_OUTER, 'title': 'n'}
    assert _message(Outer, raw, origins=origins) == "user.yaml: unknown keys: ['title']"


def test_missing_key_gives_every_file_that_set_the_mapping() -> None:
    """A key that is absent gives each layer that set a value in its mapping, in order."""
    origins = {('name',): 'base.yaml', ('inner', 'low'): 'base.yaml', ('inner', 'zz'): 'user.yaml'}
    raw = {'name': 'n', 'inner': {'low': 0.0, 'zz': 1}}
    message = _message(Outer, raw, origins=origins)
    assert message.startswith('base.yaml, user.yaml: inner: missing required keys')


def test_schema_error_survives_pickling() -> None:
    """A SchemaError rebuilt from a pickle keeps its message."""
    error = SchemaError([Issue('a.yaml', 'x', 'bad')])
    assert str(pickle.loads(pickle.dumps(error))) == 'a.yaml: x: bad'


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
class _BareFloat:
    """A dataclass with a lenient scalar field."""

    __pydantic_config__ = SCHEMA_CONFIG
    value: float = 0.0


@dataclass(frozen=True)
class _Unsupported:
    """A dataclass with an annotation outside the supported forms."""

    __pydantic_config__ = SCHEMA_CONFIG
    value: set[Int] = field(default_factory=set)


@dataclass(frozen=True)
class _NestedBreak:
    """A schema whose nested record breaks a convention."""

    __pydantic_config__ = SCHEMA_CONFIG
    items: list[_NotFrozen] = field(default_factory=list)


@pytest.mark.parametrize(
    ('cls', 'match'),
    [
        (_NotFrozen, '_NotFrozen: a schema dataclass must be frozen'),
        (_NoConfig, '_NoConfig: a schema dataclass sets __pydantic_config__ = SCHEMA_CONFIG'),
        (
            _BareFloat,
            '_BareFloat.value: a bare float reads values leniently; use the schema alias Float',
        ),
        (_Unsupported, r'_Unsupported.value: set\[.*\] is not a supported schema annotation'),
        (_NestedBreak, '_NotFrozen: a schema dataclass must be frozen'),
    ],
    ids=['not-frozen', 'no-config', 'bare-scalar', 'unsupported', 'nested'],
)
def test_check_schema_refuses_a_convention_break(cls: type, match: str) -> None:
    """A schema that breaks a convention is refused when checked."""
    with pytest.raises(TypeError, match=match):
        check_schema(cls)


def test_parse_checks_the_schema() -> None:
    """parse refuses a schema that breaks a convention before reading any input."""
    with pytest.raises(TypeError, match='a bare float reads values leniently'):
        parse(_BareFloat, {}, source='test.yaml')
