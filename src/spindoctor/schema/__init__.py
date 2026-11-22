"""One typed schema core for SpinDoctor's configuration and scene files.

A schema is a plain frozen dataclass, and pydantic parses a mapping into it
(:mod:`spindoctor.schema.core`).  The mapping comes from a stack of YAML files merged
in order under a rule for null (:mod:`spindoctor.schema.layers`).  Markers that fields
carry for other code are collected by :mod:`spindoctor.schema.metadata`.

The package imports nothing else from ``spindoctor``, so both the simulator and the
configuration code can build on it.
"""

from spindoctor.schema.core import (
    SCHEMA_CONFIG,
    Bool,
    DigitInt,
    FCPath,
    Float,
    Int,
    Issue,
    KeyPath,
    NonNeg,
    NonNegInt,
    Positive,
    PositiveInt,
    SchemaError,
    Str,
    Unit,
    check_schema,
    describe_value,
    format_key_path,
    parse,
    tagged,
)
from spindoctor.schema.layers import (
    Layer,
    Merged,
    NullPolicy,
    merge,
    read_layer,
    resolve_bases,
    split_entries,
)
from spindoctor.schema.metadata import ELEMENT, walk_metadata

__all__ = [
    'ELEMENT',
    'SCHEMA_CONFIG',
    'Bool',
    'DigitInt',
    'FCPath',
    'Float',
    'Int',
    'Issue',
    'KeyPath',
    'Layer',
    'Merged',
    'NonNeg',
    'NonNegInt',
    'NullPolicy',
    'Positive',
    'PositiveInt',
    'SchemaError',
    'Str',
    'Unit',
    'check_schema',
    'describe_value',
    'format_key_path',
    'merge',
    'parse',
    'read_layer',
    'resolve_bases',
    'split_entries',
    'tagged',
    'walk_metadata',
]
