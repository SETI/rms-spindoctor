"""Releasing a finished unit of work's memory."""

import gc
import weakref
from typing import Any

import pytest

from spindoctor.support import memory
from spindoctor.support.memory import (
    peak_resident_bytes,
    release_transient_memory,
    reset_peak_resident,
)


class _Cyclic:
    """An object that refers to itself, so refcounting alone cannot free it."""

    def __init__(self) -> None:
        self.self_ref: Any = self


def _abandoned_cycle() -> weakref.ref[_Cyclic]:
    """Drop the last name bound to a reference cycle.

    Returns:
        A weak reference to the abandoned object, which is still alive: this is
        exactly the case a release exists to reclaim.
    """
    gc.collect()
    obj = _Cyclic()
    tracker = weakref.ref(obj)
    del obj
    return tracker


def test_the_release_collects_an_abandoned_cycle() -> None:
    """What dropping the name does not reclaim, the release does."""
    tracker = _abandoned_cycle()
    release_transient_memory()
    assert tracker() is None


def test_it_runs_where_the_c_library_cannot_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collection still happens on a C library exposing no arena release.

    Parameters:
        monkeypatch: Fixture used to stand in a C library that has none.
    """
    monkeypatch.setattr(memory, '_LIBC', None)
    tracker = _abandoned_cycle()
    release_transient_memory()
    assert tracker() is None


def test_the_located_library_exposes_the_release() -> None:
    """Whatever the lookup accepts is a library that really has the function."""
    located = memory._library_with_malloc_trim()
    if located is None:
        pytest.skip('this C library exposes no malloc_trim')
    assert hasattr(located, 'malloc_trim')


def test_the_release_accepts_the_size_glibc_declares() -> None:
    """The declared signature is the one glibc's malloc_trim actually has."""
    located = memory._library_with_malloc_trim()
    if located is None:
        pytest.skip('this C library exposes no malloc_trim')
    assert located.malloc_trim(0) in (0, 1)


def test_the_peak_is_a_positive_count_of_bytes() -> None:
    """A kernel that publishes a peak publishes a usable one."""
    peak = peak_resident_bytes()
    if peak is None:
        pytest.skip('this kernel publishes no peak resident size')
    assert peak > 0


def test_the_peak_covers_an_allocation_just_made() -> None:
    """Memory taken since the last reset is inside the reported peak."""
    if peak_resident_bytes() is None:
        pytest.skip('this kernel publishes no peak resident size')
    reset_peak_resident()
    before = peak_resident_bytes()
    assert before is not None
    held = bytearray(256 * 1024 * 1024)
    held[::4096] = b'\x01' * len(held[::4096])
    after = peak_resident_bytes()
    del held
    assert after is not None
    assert after - before >= 128 * 1024 * 1024


def test_the_reset_forgets_an_earlier_peak() -> None:
    """What an earlier unit of work reached is not charged to the next one."""
    if peak_resident_bytes() is None:
        pytest.skip('this kernel publishes no peak resident size')
    held = bytearray(256 * 1024 * 1024)
    held[::4096] = b'\x01' * len(held[::4096])
    high = peak_resident_bytes()
    del held
    assert high is not None
    if not reset_peak_resident():
        pytest.skip('this kernel does not allow the peak to be reset')
    assert peak_resident_bytes() is not None
    settled = peak_resident_bytes()
    assert settled is not None
    assert settled < high


def test_the_reset_reports_whether_it_happened() -> None:
    """The caller is told, rather than left assuming the peak is per-image."""
    assert isinstance(reset_peak_resident(), bool)


def test_the_reset_measures_from_what_is_held_now() -> None:
    """The mark goes to the resident size at the reset, not to zero.

    This is what makes a peak recorded by a process handling several units of
    work include the floor the earlier ones left, rather than being that unit's
    allocation on its own.
    """
    if peak_resident_bytes() is None:
        pytest.skip('this kernel publishes no peak resident size')
    held = bytearray(256 * 1024 * 1024)
    held[::4096] = b'\x01' * len(held[::4096])
    if not reset_peak_resident():
        pytest.skip('this kernel does not allow the peak to be reset')
    still_held = peak_resident_bytes()
    assert still_held is not None
    # The block is still referenced, so it is still resident, so the mark the
    # reset just set has to be at least as large as it.
    assert still_held >= 128 * 1024 * 1024
    del held
