"""Tests for ``spindoctor.support.ck_type3``."""

from collections.abc import Iterator
from pathlib import Path

import cspyce
import numpy as np
import pytest
from tests.kernel_pool import isolated_kernel_pool

from spindoctor.support.ck_type3 import ck_type3_attitudes
from spindoctor.support.types import NDArrayFloatType

_OBJECT_ID = -999000
"""A CK object no real kernel describes."""

_RECORD_TICKS = (0.0, 100.0, 200.0, 300.0, 500.0, 600.0, 700.0)
"""Time tags of the test segment's pointing instances."""

_INTERVAL_STARTS = (0.0, 500.0)
"""Start times of the test segment's two interpolation intervals; 300 to 500 is a gap."""


def _rotation(axis: tuple[float, float, float], angle: float) -> NDArrayFloatType:
    """Return the rotation of ``angle`` radians about ``axis``."""
    unit = np.asarray(axis, dtype=np.float64) / float(np.linalg.norm(axis))
    return np.asarray(cspyce.axisar(unit, angle), dtype=np.float64)


def _attitude(index: int, turn: float) -> NDArrayFloatType:
    """Return the attitude of pointing instance ``index``, turning faster each step."""
    return _rotation((0.3, -0.5, 0.8), turn * index * index) @ _rotation((1.0, 2.0, 3.0), 1.0)


def _write_ck(
    path: Path,
    segments: list[tuple[tuple[float, ...], tuple[float, ...], float]],
    *,
    rates: bool = True,
) -> None:
    """Write type 3 segments for ``_OBJECT_ID``, each ``(ticks, interval starts, turn)``.

    Parameters:
        path: File to create.
        segments: The segments, in order.
        rates: Whether the pointing instances carry angular velocity.
    """
    handle = cspyce.ckopn(str(path), 'test', 0)
    try:
        for ticks, starts, turn in segments:
            quats = np.array([cspyce.m2q(_attitude(i, turn)) for i in range(len(ticks))])
            cspyce.ckw03(
                handle,
                ticks[0],
                ticks[-1],
                _OBJECT_ID,
                'J2000',
                rates,
                'test',
                np.asarray(ticks, dtype=np.float64),
                quats,
                np.zeros((len(ticks), 3)),
                len(starts),
                list(starts),
            )
    finally:
        cspyce.ckcls(handle)


@pytest.fixture
def kernel(tmp_path: Path) -> Path:
    """Return a C-kernel holding the two-interval test segment."""
    path = tmp_path / 'test.bc'
    _write_ck(path, [(_RECORD_TICKS, _INTERVAL_STARTS, 0.1)])
    return path


@pytest.fixture
def empty_pool() -> Iterator[None]:
    """Run the test against an emptied SPICE kernel pool, restored afterwards."""
    with isolated_kernel_pool():
        yield


@pytest.mark.parametrize(
    'tick',
    [0.0, 50.0, 100.0, 150.5, 300.0, 500.0, 550.0, 700.0],
    ids=[
        'first',
        'between',
        'record',
        'between-2',
        'interval-end',
        'interval-start',
        'second-interval',
        'last',
    ],
)
def test_attitude_matches_spice(kernel: Path, empty_pool: None, tick: float) -> None:
    """Inside an interpolation interval the attitude is the one SPICE interpolates."""
    cspyce.furnsh(str(kernel))
    try:
        expected, _clock = cspyce.ckgp(_OBJECT_ID, tick, 0.0, 'J2000')
    finally:
        cspyce.unload(str(kernel))
    (attitude,) = ck_type3_attitudes(kernel, _OBJECT_ID, [tick])
    assert attitude is not None
    np.testing.assert_allclose(attitude, np.asarray(expected), rtol=0.0, atol=1e-14)


_LONG_TICKS = tuple(float(10 * i) for i in range(250))
"""Time tags of a segment long enough to hold time-tag directory entries."""

_LONG_STARTS = (0.0, 1100.0, 2030.0)
"""Three interpolation intervals, the later two beyond the first directory entry."""


@pytest.mark.parametrize('rates', [True, False], ids=['rates', 'no-rates'])
def test_directories_and_intervals_of_a_long_segment_match_spice(
    tmp_path: Path, empty_pool: None, rates: bool
) -> None:
    """Past 100 records, with three intervals, every attitude and gap agrees with SPICE."""
    path = tmp_path / 'long.bc'
    _write_ck(path, [(_LONG_TICKS, _LONG_STARTS, 1.0e-4)], rates=rates)
    ticks = [t + 0.25 for t in range(0, 2480, 7)]
    expected: list[NDArrayFloatType | None] = []
    cspyce.furnsh(str(path))
    try:
        for tick in ticks:
            try:
                expected.append(np.asarray(cspyce.ckgp(_OBJECT_ID, tick, 0.0, 'J2000')[0]))
            except OSError:
                expected.append(None)
    finally:
        cspyce.unload(str(path))
    found = ck_type3_attitudes(path, _OBJECT_ID, ticks)
    assert [a is None for a in found] == [e is None for e in expected]
    assert sum(e is None for e in expected) > 0
    for attitude, spice in zip(found, expected, strict=True):
        if attitude is not None and spice is not None:
            np.testing.assert_allclose(attitude, spice, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize('tick', [-1.0, 400.0, 701.0], ids=['before', 'gap', 'after'])
def test_no_attitude_where_spice_has_none(kernel: Path, empty_pool: None, tick: float) -> None:
    """Before the segment, between its intervals, and after it there is no attitude."""
    cspyce.furnsh(str(kernel))
    try:
        with pytest.raises(OSError, match='CKINSUFFDATA'):
            cspyce.ckgp(_OBJECT_ID, tick, 0.0, 'J2000')
    finally:
        cspyce.unload(str(kernel))
    assert ck_type3_attitudes(kernel, _OBJECT_ID, [tick]) == [None]


def test_another_object_has_no_attitude(kernel: Path) -> None:
    """Segments of other objects are not read."""
    assert ck_type3_attitudes(kernel, _OBJECT_ID - 1, [50.0]) == [None]


def test_later_segment_takes_precedence(tmp_path: Path) -> None:
    """Where two segments cover a time, the later one in the file answers."""
    path = tmp_path / 'two.bc'
    _write_ck(path, [(_RECORD_TICKS, (0.0,), 0.1), (_RECORD_TICKS, (0.0,), 0.2)])
    (attitude,) = ck_type3_attitudes(path, _OBJECT_ID, [200.0])
    assert attitude is not None
    np.testing.assert_allclose(attitude, _attitude(2, 0.2), rtol=0.0, atol=1e-14)


def test_reading_leaves_the_pool_unchanged(kernel: Path, tmp_path: Path, empty_pool: None) -> None:
    """A read furnishes nothing: the pool's own answer is bit-identical afterwards."""
    furnished = tmp_path / 'furnished.bc'
    _write_ck(furnished, [(_RECORD_TICKS, (0.0,), 0.3)])
    cspyce.furnsh(str(furnished))
    try:
        before, _clock = cspyce.ckgp(_OBJECT_ID, 150.5, 0.0, 'J2000')
        count = cspyce.ktotal('ALL')
        (read,) = ck_type3_attitudes(kernel, _OBJECT_ID, [150.5])
        after, _clock = cspyce.ckgp(_OBJECT_ID, 150.5, 0.0, 'J2000')
        assert cspyce.ktotal('ALL') == count
        assert np.array_equal(np.asarray(after), np.asarray(before))
        assert read is not None
        assert not np.allclose(read, np.asarray(before))
    finally:
        cspyce.unload(str(furnished))
