"""Coverage invariants for the curated image library.

The structural tests in ``test_image_library`` validate each sidecar on its own.
These tests check the library as a whole:

- every declared scene class holds at least ``MIN_SIDECARS_PER_CLASS`` sidecars;
- every shipped autonomous technique is the expected primary technique of at
  least one sidecar.

The library is still being filled, so each check pins its known gaps.  While a
gap list is non-empty the test is reported as an expected failure whose reason
names the gaps, so they stay visible in the test output.  A gap outside the
pinned list fails the test, and so does a pinned gap that has been filled, so the
list can only shrink.  Once both lists are empty the tests pass outright.

The module is marked ``integration`` so it runs in the deliberate tier rather
than on every pull request while the classes are still filling.
"""

from __future__ import annotations

from collections import Counter

import pytest

from spindoctor.nav_technique import NavTechnique
from tests.integration.sidecar import DECLARED_SCENE_CLASSES, LibraryRoot, load_sidecar

pytestmark = pytest.mark.integration

MIN_SIDECARS_PER_CLASS = 2
"""The fewest sidecars a scene class may hold."""

KNOWN_THIN_CLASSES: frozenset[str] = frozenset({'body_mostly_offscreen', 'faint_stars'})
"""Scene classes known to hold fewer than ``MIN_SIDECARS_PER_CLASS`` sidecars."""

KNOWN_UNEXERCISED_TECHNIQUES: frozenset[str] = frozenset({'RingAnnulusNav', 'TitanHazeNav'})
"""Shipped autonomous techniques that no sidecar names as its primary technique."""


def _check_gaps(gaps: set[str], known: frozenset[str], *, what: str) -> None:
    """Fail on a gap outside ``known`` or a filled known gap; xfail on the rest.

    Parameters:
        gaps: The gaps the library has today.
        known: The pinned gap list.
        what: A plural noun for the messages, such as ``'scene classes'``.
    """
    new = gaps - known
    assert not new, f'{what} newly short of coverage: {sorted(new)}'
    filled = known - gaps
    assert not filled, f'{what} now covered; remove them from the list: {sorted(filled)}'
    if gaps:
        pytest.xfail(f'{what} still short of coverage: {sorted(gaps)}')


def test_every_scene_class_holds_enough_sidecars() -> None:
    """Every declared scene class holds at least ``MIN_SIDECARS_PER_CLASS`` sidecars."""
    library = LibraryRoot()
    counts = Counter(load_sidecar(p).primary_scene_tag for p in library.discover_sidecar_paths())
    thin = {c for c in DECLARED_SCENE_CLASSES if counts[c] < MIN_SIDECARS_PER_CLASS}
    _check_gaps(thin, KNOWN_THIN_CLASSES, what='scene classes')


def test_every_technique_is_some_sidecars_primary() -> None:
    """Every shipped autonomous technique is the primary technique of some sidecar."""
    library = LibraryRoot()
    primaries = {
        load_sidecar(p).expected.primary_technique for p in library.discover_sidecar_paths()
    }
    # Test modules may register their own technique subclasses; only the package's count.
    shipped = {
        cls.name for cls in NavTechnique._registry if cls.__module__.startswith('spindoctor.')
    }
    _check_gaps(shipped - primaries, KNOWN_UNEXERCISED_TECHNIQUES, what='techniques')
