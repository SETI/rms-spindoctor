"""Schemas shared by the schema core's tests, and a helper that returns a parse error."""

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import pytest
from annotated_types import Len

from spindoctor.schema.core import (
    SCHEMA_CONFIG,
    Bool,
    DigitInt,
    FCPath,
    Float,
    Int,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    Str,
    Unit,
    parse,
    tagged,
)
from spindoctor.schema.errors import SchemaError


class Model(enum.Enum):
    """A choice written as an Enum of strings."""

    CCD = 'ccd'
    VIDICON = 'vidicon'


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
    nonneg_int: NonNegInt = 1
    table: dict[DigitInt, Str] = field(default_factory=dict)
    path: FCPath | None = None
    model: Model = Model.CCD
    names: Mapping[Str, Int] = field(default_factory=dict)
    by_either: dict[Int | Str, Str] = field(default_factory=dict)


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
class Wrapper:
    """A record that holds a nested record, as the one dataclass of an untagged union."""

    __pydantic_config__ = SCHEMA_CONFIG
    inner: Inner
    label: Str


@dataclass(frozen=True)
class Nested:
    """A tagged-union member that holds a nested record."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['nested']
    inner: Inner
    label: Str


@dataclass(frozen=True)
class Outer:
    """A schema with nesting, tagged entries, a union, a choice, and keyed tables."""

    __pydantic_config__ = SCHEMA_CONFIG
    name: Str
    inner: Inner
    features: tuple[Annotated[Ringlet | Gap, tagged('kind')], ...] = ()
    either: Int | Str = 0
    units: Literal['raw_dn', 'calibrated_if'] = 'raw_dn'
    gains: dict[PositiveInt, Positive] = field(default_factory=dict)
    modes: dict[DigitInt, Inner] = field(default_factory=dict)
    pair: tuple[Float, Float] = (0.0, 0.0)
    gains_or_name: dict[PositiveInt, Positive] | Str = ''
    offered: Literal['all'] | list[Literal['a', 'b']] = 'all'
    wrapped: Wrapper | Str = ''
    offered_or_count: Annotated[Literal['all'] | list[Literal['a', 'b']], Len(1, 9)] | Int = 0
    kind_or_name: Annotated[Nested | Gap, tagged('kind')] | Str = ''


# A valid Outer mapping, which each test breaks in one place.
OUTER: dict[str, Any] = {'name': 'n', 'inner': {'low': 0.0, 'high': 1.0}}


def parse_error(cls: type, raw: object, **kwargs: Any) -> SchemaError:
    """The SchemaError that parsing ``raw`` raises.

    Parameters:
        cls: The schema.
        raw: The mapping to parse, which must break the schema.
        **kwargs: ``source`` (default ``'test.yaml'``) and ``origins``.

    Returns:
        The error.
    """
    kwargs.setdefault('source', 'test.yaml')
    with pytest.raises(SchemaError) as caught:
        parse(cls, raw, **kwargs)
    return caught.value
