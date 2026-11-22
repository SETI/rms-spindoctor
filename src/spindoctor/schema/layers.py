"""Read YAML files and merge them in order, under a stated rule for null.

Configuration rarely lives in one file.  Packaged defaults come first, a user's
overrides come after them, and an entry may set another entry as its ``base`` and
state only what differs.  This module turns such a stack into one mapping, which
:func:`spindoctor.schema.core.parse` then checks against a schema.

A :class:`Layer` is one mapping and the label that error messages give for it: a file
name, or a file name and an entry.  :func:`merge` applies layers in order:

- Mappings merge key by key, so a later layer that sets one key keeps its siblings.
- Any other value replaces the earlier one whole.  A list replaces the earlier list;
  its items are never merged.
- A null follows the merge's :class:`NullPolicy`.  Under ``REMOVE`` a null deletes the
  value an earlier layer set, and a null with nothing to delete is an error.  Under
  ``REFUSE`` every null is an error.  Under both, a null inside a list, or inside a
  mapping no earlier layer has, is an error, because there is nothing under it to
  delete.

The merge records which layer set each value, so a problem found while parsing gives
the file that holds the mistake.  It collects every null problem in all the layers
before it raises.

:func:`resolve_bases` merges each named entry over the chain of entries it inherits
from, and :func:`split_entries` turns a file of named entries into one layer per
entry.  :func:`read_layer` reads one YAML file and drops every mapping key that begins
with an underscore, which is how a file carries notes, such as citations, beside its
values.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from spindoctor.schema.errors import Issue, KeyPath, SchemaError, describe_value, format_key_path

__all__ = [
    'Layer',
    'Merged',
    'NullPolicy',
    'merge',
    'read_layer',
    'resolve_bases',
    'split_entries',
]

# The prefix of a mapping key that holds a note rather than a value.
_NOTE_PREFIX = '_'


class NullPolicy(enum.Enum):
    """What a null in a layer means.

    ``REMOVE``: a null deletes the value an earlier layer set (the simulated
    instrument catalog).  ``REFUSE``: a null is an error (simulated scenes).
    """

    REMOVE = 'remove'
    REFUSE = 'refuse'


# The message for each kind of null a merge refuses.
_NULL_MESSAGES: dict[tuple[NullPolicy, bool], str] = {
    (NullPolicy.REMOVE, False): (
        'null removes the value an earlier layer sets, and no earlier layer sets this key'
    ),
    (NullPolicy.REMOVE, True): (
        'a list item cannot be null: a list replaces the earlier list whole, so an item has '
        'nothing to remove'
    ),
    (NullPolicy.REFUSE, False): 'null is refused; leave the key out to take its default',
    (NullPolicy.REFUSE, True): 'a list item cannot be null; give it a value or remove it',
}


@dataclasses.dataclass(frozen=True)
class Layer:
    """One mapping in a stack of layers.

    Parameters:
        data: The mapping.
        label: What an error message gives as the mapping's source: the file name, or
            the file name and an entry (``'coiss.yaml: coiss_nac'``).
    """

    data: Mapping[Any, Any]
    label: str


@dataclasses.dataclass(frozen=True)
class Merged:
    """The result of merging layers.

    Parameters:
        data: The merged mapping, which shares no objects with the layers.
        origins: The label of the layer that set each value, keyed by the value's key
            path.  A value is a scalar, a list, or an empty mapping; a list's items are
            not recorded separately.  Pass it to
            :func:`spindoctor.schema.core.parse` as ``origins``.
    """

    data: dict[Any, Any]
    origins: dict[KeyPath, str]


def read_layer(path: Path, *, label: str | None = None) -> Layer:
    """Read one YAML file as a layer.

    Every mapping key that begins with an underscore is dropped, at any depth.  An
    empty file is an empty mapping.

    Parameters:
        path: The file.
        label: The layer's label; the file's name when omitted.

    Returns:
        The layer.

    Raises:
        SchemaError: If the file is not valid YAML (a key written twice included), or
            holds something other than a mapping.
    """
    source = path.name if label is None else label
    try:
        loaded = YAML(typ='safe').load(path.read_text(encoding='utf-8'))
    except YAMLError as exc:
        raise SchemaError([Issue(source, '', f'not valid YAML: {exc}')]) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        message = f'expected a mapping at the top level, got {describe_value(loaded)}'
        raise SchemaError([Issue(source, '', message)])
    return Layer(_drop_notes(loaded), source)


def _drop_notes(value: Any) -> Any:
    """``value`` without any mapping key that begins with an underscore, at any depth."""
    if isinstance(value, dict):
        return {
            key: _drop_notes(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.startswith(_NOTE_PREFIX))
        }
    if isinstance(value, list):
        return [_drop_notes(item) for item in value]
    return value


def merge(layers: Sequence[Layer], *, nulls: NullPolicy) -> Merged:
    """Merge layers in order, each over the ones before it.

    Parameters:
        layers: The layers, earliest first.
        nulls: What a null means.

    Returns:
        The merged mapping and the layer that set each value.

    Raises:
        SchemaError: Listing every null the policy refuses, in every layer.
    """
    merger = _Merger(nulls)
    for layer in layers:
        merger.apply(layer)
    if merger.issues:
        raise SchemaError(merger.issues)
    return Merged(merger.data, merger.origins)


class _Merger:
    """The state of one merge: the result so far, its origins, and the problems found."""

    def __init__(self, nulls: NullPolicy) -> None:
        self.nulls = nulls
        self.data: dict[Any, Any] = {}
        self.origins: dict[KeyPath, str] = {}
        self.issues: list[Issue] = []
        self._layer = Layer({}, '')

    def apply(self, layer: Layer) -> None:
        """Merge one layer over the result so far."""
        self._layer = layer
        self._merge_into(self.data, layer.data, ())

    def _merge_into(
        self, target: dict[Any, Any], overlay: Mapping[Any, Any], path: KeyPath
    ) -> None:
        """Merge ``overlay`` into ``target``, which sits at ``path`` in the result."""
        if overlay and path in self.origins:
            # ``target`` was an empty mapping, recorded as a value; it now has keys.
            del self.origins[path]
        for key, value in overlay.items():
            key_path = (*path, key)
            if value is None:
                if self.nulls is NullPolicy.REMOVE and key in target:
                    self._forget(target[key], key_path)
                    del target[key]
                    if not target and path:
                        # An emptied mapping is a value, set by the layer that emptied it.
                        self.origins[path] = self._layer.label
                else:
                    self._refuse_null(key_path, in_list=False)
                continue
            existing = target.get(key)
            if isinstance(value, Mapping) and isinstance(existing, dict):
                self._merge_into(existing, value, key_path)
                continue
            if key in target:
                self._forget(existing, key_path)
            target[key] = self._take(value, key_path)

    def _take(self, value: Any, path: KeyPath) -> Any:
        """A copy of a value no earlier layer has, recording its origin.

        A null inside it is refused, since it has nothing beneath it to remove.
        """
        if isinstance(value, Mapping):
            taken: dict[Any, Any] = {}
            for key, item in value.items():
                if item is None:
                    self._refuse_null((*path, key), in_list=False)
                else:
                    taken[key] = self._take(item, (*path, key))
            if not taken:
                self.origins[path] = self._layer.label
            return taken
        if isinstance(value, list):
            self._refuse_nulls_in(value, path)
            self.origins[path] = self._layer.label
            return copy.deepcopy(value)
        self.origins[path] = self._layer.label
        return value

    def _refuse_nulls_in(self, value: Any, path: KeyPath) -> None:
        """Refuse every null inside a list, at any depth, as the list replaces whole."""
        items = enumerate(value) if isinstance(value, list) else value.items()
        for key, item in items:
            item_path = (*path, key)
            if item is None:
                self._refuse_null(item_path, in_list=isinstance(value, list))
            elif isinstance(item, list | Mapping):
                self._refuse_nulls_in(item, item_path)

    def _refuse_null(self, path: KeyPath, *, in_list: bool) -> None:
        """Record a refused null at ``path`` in the current layer."""
        message = _NULL_MESSAGES[(self.nulls, in_list)]
        shown = format_key_path(self._layer.data, path)
        self.issues.append(Issue(self._layer.label, shown, message))

    def _forget(self, value: Any, path: KeyPath) -> None:
        """Drop the recorded origins of a value that is being replaced or removed."""
        if isinstance(value, dict):
            for recorded in [p for p in self.origins if p[: len(path)] == path]:
                del self.origins[recorded]
        else:
            self.origins.pop(path, None)


def split_entries(layer: Layer) -> dict[str, Layer]:
    """Split a layer whose top-level keys are entry names into one layer per entry.

    Each entry's layer is labeled with the file's label and the entry's name
    (``'coiss.yaml: coiss_nac'``).

    Parameters:
        layer: A layer that maps entry names to entries.

    Returns:
        Each entry's layer, keyed by name.

    Raises:
        SchemaError: Listing each entry name that is not a string and each entry that
            is not a mapping.
    """
    entries: dict[str, Layer] = {}
    issues: list[Issue] = []
    for name, entry in layer.data.items():
        if not isinstance(name, str):
            issues.append(Issue(layer.label, '', f'the entry name {name!r} must be a string'))
        elif not isinstance(entry, Mapping):
            label = f'{layer.label}: {name}'
            issues.append(Issue(label, '', f'expected a mapping, got {describe_value(entry)}'))
        else:
            entries[name] = Layer(entry, f'{layer.label}: {name}')
    if issues:
        raise SchemaError(issues)
    return entries


def resolve_bases(
    entries: Mapping[str, Layer],
    *,
    nulls: NullPolicy,
    base_key: str = 'base',
    not_inherited: frozenset[str] = frozenset(),
) -> dict[str, Merged]:
    """Merge each entry over the chain of entries it inherits from.

    An entry that sets ``base_key`` to another entry's name inherits that entry, which
    may in turn inherit another.  The chain merges from its far end, so the entry's
    own values come last and win.  ``base_key`` itself is never part of a result.

    When a chain stops at a bad base, the part of the chain before it is still merged,
    so the entry's other null problems are reported beside the bad base.  A null that
    removes nothing there is not reported, since the base that could not be read may
    have set the key.

    Parameters:
        entries: Each entry's layer, keyed by name.
        nulls: What a null means.
        base_key: The key that holds the name of the entry to inherit.
        not_inherited: Keys an entry keeps to itself: a result holds them only from
            the entry's own layer.

    Returns:
        Each entry's merged result, keyed by name, in the order of ``entries``.

    Raises:
        SchemaError: Listing each base that is not a string or not an entry, each
            chain of bases that loops, and every null the policy refuses.
    """
    issues: list[Issue] = []
    passed_on = not_inherited | {base_key}
    results: dict[str, Merged] = {}
    for name in entries:
        chain, complete = _base_chain(name, entries, base_key=base_key, issues=issues)
        layers = [_without(entries[base], passed_on) for base in chain[:0:-1]]
        layers.append(_without(entries[name], frozenset({base_key})))
        try:
            merged = merge(layers, nulls=nulls)
        except SchemaError as exc:
            # Behind a bad base, a null may have had a value to remove in the base that
            # could not be read, so only the nulls refused for another reason count.
            issues.extend(
                issue
                for issue in exc.issues
                if complete or issue.message != _NULL_MESSAGES[(NullPolicy.REMOVE, False)]
            )
            continue
        if complete:
            results[name] = merged
    if issues:
        raise SchemaError(list(dict.fromkeys(issues)))
    return results


def _base_chain(
    name: str, entries: Mapping[str, Layer], *, base_key: str, issues: list[Issue]
) -> tuple[list[str], bool]:
    """``name`` and the entries it inherits from, nearest first.

    Returns:
        The chain, and whether it is complete.  After a bad base the chain ends at the
        entry that sets it, and the problem is added to ``issues``.
    """
    chain = [name]
    while base_key in entries[chain[-1]].data:
        current = entries[chain[-1]]
        base = current.data[base_key]
        if not isinstance(base, str):
            message = f'expected the name of an entry, got {describe_value(base)}'
            issues.append(Issue(current.label, base_key, message))
            return chain, False
        if base not in entries:
            issues.append(Issue(current.label, base_key, f'{base!r} is not an entry'))
            return chain, False
        if base in chain:
            loop = ' -> '.join((*chain, base))
            issues.append(Issue(entries[name].label, base_key, f'inheritance loops ({loop})'))
            return chain, False
        chain.append(base)
    return chain, True


def _without(layer: Layer, keys: frozenset[str]) -> Layer:
    """``layer`` without the top-level ``keys``."""
    return Layer({key: value for key, value in layer.data.items() if key not in keys}, layer.label)
