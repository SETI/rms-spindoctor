"""Releasing a finished unit of work's memory."""

import gc
import weakref
from typing import Any

import pytest

from spindoctor.support import memory
from spindoctor.support.memory import release_transient_memory


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


def test_a_reference_cycle_outlives_its_last_name() -> None:
    """The premise: dropping the name does not free a cycle."""
    assert _abandoned_cycle()() is not None


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


def test_repeated_releases_are_safe() -> None:
    """Releasing twice in a row is as harmless as releasing once.

    Each call is given a cycle of its own to collect, so the test fails if
    either one stops happening -- which asserting against a cycle abandoned
    after both calls would not.
    """
    first = _abandoned_cycle()
    release_transient_memory()
    assert first() is None
    second = _abandoned_cycle()
    release_transient_memory()
    assert second() is None


def test_the_located_library_exposes_the_release() -> None:
    """Whatever the lookup accepts is a library that really has the function."""
    located = memory._malloc_trim()
    if located is None:
        pytest.skip('this C library exposes no malloc_trim')
    assert hasattr(located, 'malloc_trim')


def test_the_release_accepts_the_size_glibc_declares() -> None:
    """The declared signature is the one glibc's malloc_trim actually has."""
    located = memory._malloc_trim()
    if located is None:
        pytest.skip('this C library exposes no malloc_trim')
    assert located.malloc_trim(0) in (0, 1)
