"""Tests for :mod:`spindoctor.schema.errors`: where each problem is placed and how it reads.

Each case breaks a valid mapping in one place and checks the one line the problem
becomes: its key path, its wording, and the file it gives.
"""

import pickle
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest
from annotated_types import Ge, Gt, Le, Len, Lt
from tests.spindoctor.schema.schemas import OUTER, Aliases, Outer, parse_error

from spindoctor.schema.core import SCHEMA_CONFIG, Float
from spindoctor.schema.errors import Issue, SchemaError


@dataclass(frozen=True)
class Bounds:
    """Ad-hoc annotated_types bounds, which keep pydantic's own figures."""

    __pydantic_config__ = SCHEMA_CONFIG
    below: Annotated[Float, Lt(5)] = 0.0
    at_most: Annotated[Float, Le(90)] = 0.0
    over: Annotated[Float, Gt(1)] = 2.0
    over_zero: Annotated[Float, Gt(0)] = 1.0
    above: Annotated[Float, Ge(2)] = 2.0
    above_zero: Annotated[Float, Ge(0)] = 0.0
    pairs: Annotated[list[Float], Len(2, 2)] = field(default_factory=lambda: [0.0, 0.0])


def _with_key(key: object, value: object) -> dict[Any, Any]:
    """A valid Outer mapping with one more key."""
    raw: dict[Any, Any] = dict(OUTER)
    raw[key] = value
    return raw


def test_every_problem_is_reported_at_once() -> None:
    """Two broken keys give one issue each."""
    assert len(parse_error(Aliases, {'int_': 'a', 'unit': -1}).issues) == 2


@pytest.mark.parametrize(
    ('raw', 'message'),
    [
        (
            {**OUTER, 'features': [{'kind': 'gap'}, {'kind': 'ringlet', 'width': -1}]},
            'test.yaml: features[1].width: expected a positive number, got -1',
        ),
        (
            {**OUTER, 'features': [{'kind': 'ring'}]},
            "test.yaml: features[0].kind: 'ring' is not one of ['gap', 'ringlet']",
        ),
        (
            {**OUTER, 'features': [{'width': 1}]},
            "test.yaml: features[0]: missing required keys: ['kind']",
        ),
        (
            {**OUTER, 'inner': {'low': 0.0, 'high': 1.0, 'mid': 0.5, 'avg': 0.5}},
            "test.yaml: inner: unknown keys: ['avg', 'mid']",
        ),
        ({'inner': OUTER['inner']}, "test.yaml: missing required keys: ['name']"),
        (_with_key(3, 'x'), "test.yaml: unknown keys: ['3']"),
        ({**OUTER, 'inner': 3}, 'test.yaml: inner: expected a mapping, got int 3'),
        ({**OUTER, 'features': 'gap'}, "test.yaml: features: expected a list, got str 'gap'"),
        (
            {**OUTER, 'either': 1.5},
            'test.yaml: either: expected an integer or a string, got float 1.5',
        ),
        (
            {**OUTER, 'units': 'dn'},
            "test.yaml: units: 'dn' is not one of ['calibrated_if', 'raw_dn']",
        ),
        (
            {**OUTER, 'inner': {'low': 2.0, 'high': 1.0}},
            'test.yaml: inner: low (2.0) is above high (1.0)',
        ),
        ({**OUTER, 'pair': [1.0]}, 'test.yaml: pair: expected a list of 2 items, got list [1.0]'),
        (
            {**OUTER, 'pair': [1.0, 2.0, 3.0]},
            'test.yaml: pair: expected a list of at most 2 items, got list [1.0, 2.0, 3.0]',
        ),
    ],
    ids=[
        'tagged-member-field',
        'tagged-unknown-tag',
        'tagged-missing-tag',
        'unknown-keys',
        'missing-keys',
        'key-not-a-string',
        'not-a-mapping',
        'not-a-list',
        'union-of-scalars',
        'literal',
        'cross-field-rule',
        'fixed-list-short',
        'fixed-list-long',
    ],
)
def test_problem_is_placed_and_phrased(raw: dict[Any, Any], message: str) -> None:
    """Each kind of problem is reported once, at the key path it concerns."""
    assert str(parse_error(Outer, raw)) == message


@pytest.mark.parametrize(
    ('raw', 'message'),
    [
        ({'below': 5}, 'test.yaml: below: expected a number less than 5, got 5'),
        ({'at_most': 91}, 'test.yaml: at_most: expected a number of at most 90, got 91'),
        ({'over': 1}, 'test.yaml: over: expected a number greater than 1, got 1'),
        ({'over_zero': 0}, 'test.yaml: over_zero: expected a positive number, got 0'),
        ({'above': 1}, 'test.yaml: above: expected a number of at least 2, got 1'),
        ({'above_zero': -1}, 'test.yaml: above_zero: expected a non-negative number, got -1'),
        ({'pairs': [1.0]}, 'test.yaml: pairs: expected a list of at least 2 items, got list [1.0]'),
    ],
    ids=['less-than', 'at-most', 'greater-than', 'positive', 'at-least', 'non-negative', 'length'],
)
def test_ad_hoc_bound_is_phrased(raw: dict[str, Any], message: str) -> None:
    """An annotated_types bound is worded from its own figure."""
    assert str(parse_error(Bounds, raw)) == message


@pytest.mark.parametrize(
    ('tables', 'messages'),
    [
        (
            {'gains': {0: 1.0}},
            ['test.yaml: gains.0: expected a positive integer as the key, got 0'],
        ),
        ({'gains': {1: -1.0}}, ['test.yaml: gains.1: expected a positive number, got -1.0']),
        (
            {'gains': {0: -1.0}},
            [
                'test.yaml: gains.0: expected a positive integer as the key, got 0',
                'test.yaml: gains.0: expected a positive number, got -1.0',
            ],
        ),
        (
            {'modes': {'x': 5}},
            [
                "test.yaml: modes.x: expected an integer as the key, got str 'x'",
                'test.yaml: modes.x: expected a mapping, got int 5',
            ],
        ),
        (
            {'gains_or_name': {0: -1.0}},
            [
                'test.yaml: gains_or_name.0: expected a positive integer as the key, got 0',
                'test.yaml: gains_or_name.0: expected a positive number, got -1.0',
                'test.yaml: gains_or_name: expected a string, got dict {0: -1.0}',
            ],
        ),
    ],
    ids=['bad-key', 'bad-value', 'both-out-of-range', 'both-wrong-type', 'both-in-a-union'],
)
def test_bad_key_and_bad_value_are_two_problems(
    tables: dict[str, Any], messages: list[str]
) -> None:
    """A bad mapping key and a bad value under it each get a line of their own."""
    error = parse_error(Outer, {**OUTER, **tables})
    assert [str(issue) for issue in error.issues] == messages


@pytest.mark.parametrize('key', ['2', 2], ids=['digit-string', 'integer'])
def test_problem_under_an_integer_key_is_placed(key: object) -> None:
    """A problem under an integer key is placed whether the key is written as digits or not."""
    error = parse_error(Outer, {**OUTER, 'modes': {key: {'low': 'x', 'high': 1.0}}})
    assert str(error) == "test.yaml: modes.2.low: expected a number, got str 'x'"


def test_problem_gives_the_file_that_set_the_value() -> None:
    """A problem with a value gives the layer that set it."""
    origins = {
        ('name',): 'base.yaml',
        ('inner', 'low'): 'base.yaml',
        ('inner', 'high'): 'user.yaml',
    }
    raw = {'name': 'n', 'inner': {'low': 0.0, 'high': 'x'}}
    assert str(parse_error(Outer, raw, origins=origins)).startswith('user.yaml: inner.high: ')


def test_problem_inside_a_list_gives_the_file_that_set_the_list() -> None:
    """A problem inside a list gives the layer that set the whole list."""
    origins = {
        ('name',): 'base.yaml',
        ('inner', 'low'): 'base.yaml',
        ('inner', 'high'): 'base.yaml',
        ('features',): 'user.yaml',
    }
    raw = {**OUTER, 'features': [{'kind': 'ringlet', 'width': 0}]}
    message = str(parse_error(Outer, raw, origins=origins))
    assert message.startswith('user.yaml: features[0].width: ')


def test_unknown_key_gives_the_file_that_set_it() -> None:
    """An unknown key gives the layer that set it, though the line is about its mapping."""
    origins = {('name',): 'base.yaml', ('inner', 'low'): 'base.yaml', ('title',): 'user.yaml'}
    raw = {**OUTER, 'title': 'n'}
    assert str(parse_error(Outer, raw, origins=origins)) == "user.yaml: unknown keys: ['title']"


def test_missing_key_gives_every_file_that_set_the_mapping() -> None:
    """A key that is absent gives each layer that set a value in its mapping, in order."""
    origins = {('name',): 'base.yaml', ('inner', 'low'): 'base.yaml', ('inner', 'zz'): 'user.yaml'}
    raw = {'name': 'n', 'inner': {'low': 0.0, 'zz': 1}}
    message = str(parse_error(Outer, raw, origins=origins))
    assert message.startswith('base.yaml, user.yaml: inner: missing required keys')


def test_integer_key_written_twice_is_refused() -> None:
    """A mapping that writes one integer key as 2 and as '2' is refused."""
    message = str(parse_error(Aliases, {'table': {'2': 'a', 2: 'b'}}))
    assert message == "test.yaml: table: the key 2 is written twice, as 2 and '2'"


def test_schema_error_survives_pickling() -> None:
    """A SchemaError rebuilt from a pickle keeps its message."""
    error = SchemaError([Issue('a.yaml', 'x', 'bad')])
    assert str(pickle.loads(pickle.dumps(error))) == 'a.yaml: x: bad'
