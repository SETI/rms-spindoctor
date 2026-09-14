"""Tests for ``spindoctor.cli.reproj.factories.build_ring_mosaic`` argument validation.

Covers the absolute-vs-offset radius semantics tied to ``--orbit-model``:

- ``--orbit-model none`` requires ``--radius-inner`` / ``--radius-outer`` and
  rejects ``--radius-inner-offset`` / ``--radius-outer-offset``.
- ``--orbit-model f_ring_core_albers_2007`` (or any non-``none`` value) requires
  ``--radius-inner-offset`` / ``--radius-outer-offset`` and rejects
  ``--radius-inner`` / ``--radius-outer``.
- Offsets are passed to ``RingMosaic`` verbatim (no addition of
  ``orbit_model.a``); the stored offsets are signed offsets from the orbital
  radius at each (longitude, time).
"""

import argparse

import pytest

from spindoctor.cli.reproj.args import add_ring_args
from spindoctor.cli.reproj.factories import build_ring_mosaic
from spindoctor.reproj.ring_orbit_model import FRING_CORE
from spindoctor.reproj.rings import RingMosaic


def _build_parser() -> argparse.ArgumentParser:
    """Return a parser pre-populated with ``add_ring_args`` for testing."""
    parser = argparse.ArgumentParser()
    add_ring_args(parser)
    return parser


def test_no_orbit_model_requires_absolute_radii() -> None:
    """``--orbit-model none`` requires ``--radius-inner`` / ``--radius-outer``."""
    parser = _build_parser()
    args = parser.parse_args(['--planet', 'SATURN'])
    with pytest.raises(ValueError, match='--radius-inner and --radius-outer are required'):
        build_ring_mosaic(args)


def test_no_orbit_model_rejects_offset_radii() -> None:
    """``--orbit-model none`` rejects offset radius arguments."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            '--planet',
            'SATURN',
            '--radius-inner',
            '139000',
            '--radius-outer',
            '140000',
            '--radius-inner-offset',
            '-500',
        ],
    )
    with pytest.raises(ValueError, match=r'radius-inner-offset.*must not be used'):
        build_ring_mosaic(args)


def test_no_orbit_model_builds_mosaic_with_absolute_radii() -> None:
    """When ``--orbit-model`` is none, absolute radii flow through verbatim."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            '--planet',
            'SATURN',
            '--radius-inner',
            '139500',
            '--radius-outer',
            '140500',
        ],
    )
    mosaic: RingMosaic = build_ring_mosaic(args)
    sparse = mosaic.to_sparse()
    assert sparse.radius_inner == pytest.approx(139500.0)
    assert sparse.radius_outer == pytest.approx(140500.0)
    assert sparse.orbit_model_name is None


def test_orbit_model_requires_offset_radii() -> None:
    """A non-``none`` orbit model requires offset radius arguments."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            '--planet',
            'SATURN',
            '--orbit-model',
            'f_ring_core_albers_2007',
        ],
    )
    with pytest.raises(
        ValueError, match='radius-inner-offset and --radius-outer-offset are required'
    ):
        build_ring_mosaic(args)


def test_orbit_model_rejects_absolute_radii() -> None:
    """A non-``none`` orbit model rejects ``--radius-inner`` / ``--radius-outer``."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            '--planet',
            'SATURN',
            '--orbit-model',
            'f_ring_core_albers_2007',
            '--radius-inner',
            '139000',
            '--radius-outer',
            '141000',
            '--radius-inner-offset',
            '-1000',
            '--radius-outer-offset',
            '1000',
        ],
    )
    with pytest.raises(ValueError, match='--radius-inner and --radius-outer must not be used'):
        build_ring_mosaic(args)


def test_orbit_model_offset_radii_passed_verbatim() -> None:
    """Offsets reach ``RingMosaic`` unchanged (no shift by ``orbit_model.a``)."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            '--planet',
            'SATURN',
            '--orbit-model',
            'f_ring_core_albers_2007',
            '--radius-inner-offset',
            '-1000',
            '--radius-outer-offset',
            '1000',
        ],
    )
    mosaic: RingMosaic = build_ring_mosaic(args)
    sparse = mosaic.to_sparse()
    # Stored values are the raw offsets, NOT FRING_CORE.a + offset.
    assert sparse.radius_inner == pytest.approx(-1000.0)
    assert sparse.radius_outer == pytest.approx(1000.0)
    assert sparse.orbit_model_name == FRING_CORE.name


def test_unknown_orbit_model_raises() -> None:
    """An unrecognized ``--orbit-model`` value raises ``ValueError`` from argparse."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--planet',
                'SATURN',
                '--orbit-model',
                'made_up',
            ],
        )
