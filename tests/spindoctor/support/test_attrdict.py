"""Hermetic tests for ``spindoctor.support.attrdict``.

The class publishes its own dict as its ``__dict__``, which is what makes
attribute access work and is also its one hazard: an attribute set on one of
these objects is a key inserted into configuration data.  oops does that, by
caching its mutability verdict on every object it walks, so the class carries
oops's opt-out marker.  These tests stand in for oops rather than importing it,
so they pin the contract the marker relies on without needing a kernel pool.
"""

from __future__ import annotations

from typing import Any

import pytest

from spindoctor.support.attrdict import AttrDict


def oops_mutability_walk(obj: Any, *, honor_marker: bool) -> bool:
    """Stand in for ``oops.mutable._get_info``.

    That function walks whatever it can reach from an observation and caches its
    verdict by setting ``_MUTABLE_info`` on each object, skipping any object that
    carries ``_IS_IMMUTABLE``.  Reproduced here rather than called, so the two
    names this class depends on are pinned without a kernel pool.

    oops itself always skips a marked object; ``honor_marker=False`` is the
    counterfactual, kept so the tests can show what the walk writes into a
    mapping that carries no marker.

    Parameters:
        obj: The object to inspect.
        honor_marker: True to skip an object carrying ``_IS_IMMUTABLE``, which is
            what oops does; False to write unconditionally.

    Returns:
        True if the verdict was written.
    """
    if honor_marker and hasattr(obj, '_IS_IMMUTABLE'):
        return False
    obj._MUTABLE_info = object()
    return True


def test_attribute_access_reads_a_key() -> None:
    """The point of the class: a key is readable as an attribute."""
    section = AttrDict({'planet': 'SATURN'})
    assert section.planet == 'SATURN'


def test_setting_an_attribute_writes_a_key() -> None:
    """And the inverse, which is what makes oops's cached verdict dangerous."""
    section = AttrDict()
    # setattr rather than ``section.planet =``, which mypy refuses on a class
    # whose attributes are its keys; the runtime path is the one oops takes.
    setattr(section, 'planet', 'SATURN')  # noqa: B010
    assert section['planet'] == 'SATURN'


def test_a_missing_attribute_raises_attribute_error() -> None:
    """A missing key reads as a missing attribute, not a KeyError."""
    section = AttrDict()
    with pytest.raises(AttributeError, match="Attribute 'planet' not found"):
        _ = section.planet


def test_the_oops_marker_is_advertised() -> None:
    """oops can see that this mapping is not its scratch space."""
    assert hasattr(AttrDict(), '_IS_IMMUTABLE')


def test_the_marker_is_not_itself_a_key() -> None:
    """Carrying the marker must not put anything in the data.

    The marker lives on the class, so it is visible to ``hasattr`` without
    occupying the mapping; a marker stored as a key would be the same defect it
    exists to prevent.
    """
    section = AttrDict({'planet': 'SATURN'})
    assert '_IS_IMMUTABLE' not in section


def test_the_marker_leaves_the_mapping_equal_to_its_plain_dict() -> None:
    """Nothing about length, iteration, or equality changes."""
    assert AttrDict({'planet': 'SATURN'}) == {'planet': 'SATURN'}


def test_the_oops_walk_writes_nothing() -> None:
    """The behavior the marker exists to produce.

    oops builds a Backplane from an observation, and the observation holds the
    shared Config, whose sections are these mappings.  Without the marker the
    walk reaches them and every section gains a ``_MUTABLE_info`` key that no
    configuration file declares, so a section read back differs from the file
    that defined it.
    """
    section = AttrDict({'planet': 'SATURN'})
    wrote = oops_mutability_walk(section, honor_marker=True)
    assert not wrote


def test_the_oops_walk_leaves_the_keys_untouched() -> None:
    """Same case, stated as the invariant a caller actually depends on."""
    section = AttrDict({'planet': 'SATURN'})
    oops_mutability_walk(section, honor_marker=True)
    assert list(section) == ['planet']


def test_the_same_walk_without_the_marker_lands_in_the_mapping() -> None:
    """The hazard itself, which is what makes the marker worth carrying.

    Run against a mapping oops would not skip, the same walk puts
    ``_MUTABLE_info`` in the mapping rather than beside it, where a caller
    reading the section back sees a key no configuration file declares.
    """
    section = AttrDict({'planet': 'SATURN'})
    oops_mutability_walk(section, honor_marker=False)
    assert '_MUTABLE_info' in section


def test_an_ordinary_object_is_still_annotated() -> None:
    """The stand-in really does write when nothing carries the marker.

    Without this the two tests above would pass against a stand-in that never
    writes to anything.
    """

    class Plain:
        """An object with an ordinary attribute dict."""

    plain = Plain()
    assert oops_mutability_walk(plain, honor_marker=True)


def test_a_subclass_inherits_the_marker() -> None:
    """A specialized section is protected for the same reason the base is."""

    class Section(AttrDict):
        """A hypothetical AttrDict subclass."""

    assert not oops_mutability_walk(Section({'planet': 'SATURN'}), honor_marker=True)
