"""Tests for :mod:`spindoctor.schema.metadata`."""

from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest

from spindoctor.schema.core import SCHEMA_CONFIG, DigitInt, Float, tagged
from spindoctor.schema.metadata import walk_metadata


@dataclass(frozen=True)
class Noise:
    """A record reached through a mapping."""

    __pydantic_config__ = SCHEMA_CONFIG
    read_noise: Float = field(default=0.0, metadata={'boundary': 'truth'})


@dataclass(frozen=True)
class Disc:
    """A tagged-union member."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['disc']
    radius: Float = field(default=1.0, metadata={'boundary': 'idealized'})


@dataclass(frozen=True)
class Blob:
    """A second tagged-union member, sharing a field path with the first."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['blob']
    radius: Float = field(default=1.0, metadata={'boundary': 'idealized'})
    sigma: Float = field(default=1.0, metadata={'editor': 'px'})


@dataclass(frozen=True)
class Scene:
    """A root that reaches records through a mapping, a tuple, and an optional field."""

    __pydantic_config__ = SCHEMA_CONFIG
    seed: Float = field(default=0.0, metadata={'boundary': 'idealized', 'editor': 'seed'})
    unmarked: Float = 0.0
    by_binning: dict[DigitInt, Noise] = field(default_factory=dict)
    bodies: tuple[Annotated[Disc | Blob, tagged('kind')], ...] = ()
    noise: Noise | None = None


def test_walk_metadata_maps_each_marked_field_path() -> None:
    """Every marked field appears under its path; containers add the element segment."""
    assert walk_metadata(Scene) == {
        'seed': {'boundary': 'idealized', 'editor': 'seed'},
        'by_binning.*.read_noise': {'boundary': 'truth'},
        'bodies.*.radius': {'boundary': 'idealized'},
        'bodies.*.sigma': {'editor': 'px'},
        'noise.read_noise': {'boundary': 'truth'},
    }


@dataclass(frozen=True)
class _Ring:
    """A union member that marks ``radius`` differently from :class:`Disc`."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['ring']
    radius: Float = field(default=1.0, metadata={'boundary': 'truth'})


@dataclass(frozen=True)
class _Clash:
    """A root whose union members disagree about one field's markers."""

    __pydantic_config__ = SCHEMA_CONFIG
    body: Annotated[Disc | _Ring, tagged('kind')]


@dataclass(frozen=True)
class _Plain:
    """A union member that leaves ``radius`` unmarked."""

    __pydantic_config__ = SCHEMA_CONFIG
    kind: Literal['plain']
    radius: Float = 1.0


@dataclass(frozen=True)
class _HalfMarked:
    """A root whose union members disagree about whether a field is marked."""

    __pydantic_config__ = SCHEMA_CONFIG
    body: Annotated[Disc | _Plain, tagged('kind')]


def test_walk_metadata_refuses_a_marker_one_member_leaves_out() -> None:
    """A field marked in one union member and unmarked in another is refused."""
    with pytest.raises(TypeError, match=r'body\.radius: two types at this path mark'):
        walk_metadata(_HalfMarked)


@dataclass(frozen=True)
class Tree:
    """A schema that refers to itself."""

    __pydantic_config__ = SCHEMA_CONFIG
    label: Float = field(default=0.0, metadata={'editor': 'label'})
    children: tuple['Tree', ...] = ()


def test_walk_metadata_reads_a_recursive_schema() -> None:
    """A schema that refers to itself is walked once, without looping."""
    assert walk_metadata(Tree) == {'label': {'editor': 'label'}}


def test_walk_metadata_refuses_disagreeing_union_members() -> None:
    """Two members that mark the same field path differently are refused."""
    with pytest.raises(TypeError, match=r"'idealized'\} and \{'boundary': 'truth'"):
        walk_metadata(_Clash)
