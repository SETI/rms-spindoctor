"""The errors the schema package raises, and how pydantic's errors become them.

Every problem the package finds is an :class:`Issue`: the source that set the value,
the key path, and what is wrong.  A :class:`SchemaError` carries every issue found in
one pass, and its message has one line per issue, ``file: key.path: message``.

:func:`report` turns the errors pydantic raises for one input into issues.  It places
each error on the input's own keys, words it as the simulated instrument catalog
words its problems ("expected a positive number, got -1", "'cmos' is not one of
['ccd', 'vidicon']"), combines the problems that belong on one line, and gives each
line the source that set the value.  :func:`doubled_keys` finds the one problem pydantic
cannot see: a mapping that writes one integer key both as ``2`` and as ``'2'``.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    'Issue',
    'KeyPath',
    'LengthBounds',
    'RangeError',
    'RuleError',
    'SchemaError',
    'describe_value',
    'doubled_keys',
    'format_key_path',
    'report',
]

#: A key path into a parsed YAML tree: mapping keys, and integer positions in lists.
KeyPath = tuple[Any, ...]

#: Given the key path of a list in the input, the (fewest, most) items each list type the
#: schema allows there may hold; None for no most.  :func:`report` asks it whether a
#: list of the wrong length had to have exactly one length.
LengthBounds = Callable[[KeyPath], frozenset[tuple[int, int | None]]]


# pydantic error types that state what the value should have been.
_EXPECTED: dict[str, str] = {
    'bool_parsing': 'a boolean',
    'bool_type': 'a boolean',
    'dataclass_type': 'a mapping',
    'dict_type': 'a mapping',
    'finite_number': 'a finite number',
    'float_parsing': 'a number',
    'float_type': 'a number',
    'int_from_float': 'an integer',
    'int_parsing': 'an integer',
    'int_type': 'an integer',
    'list_type': 'a list',
    'mapping_type': 'a mapping',
    'model_attributes_type': 'a mapping',
    'string_type': 'a string',
    'tuple_type': 'a list',
}
_UNKNOWN = frozenset({'extra_forbidden', 'invalid_key', 'unexpected_keyword_argument'})
_RULE = frozenset({'value_error'})
_CHOICE = frozenset({'enum', 'literal_error'})
# The location segment pydantic adds for a problem with a mapping key itself.
_KEY_SEGMENT = '[key]'
# One quoted string in pydantic's list of a Literal's or an Enum's values.
_QUOTED = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


@dataclasses.dataclass(frozen=True)
class Issue:
    """One problem found in a file.

    Parameters:
        source: The file (or file and entry) that set the value.
        path: The dotted key path of the value, with ``[n]`` for a list item; empty
            for the file as a whole.
        message: What is wrong.
    """

    source: str
    path: str
    message: str

    def __str__(self) -> str:
        """The problem as one line: ``source: path: message``."""
        location = f'{self.source}: {self.path}' if self.path else self.source
        return f'{location}: {self.message}'


class SchemaError(ValueError):
    """A file breaks its schema.  The message has one line per problem.

    Parameters:
        issues: Every problem found, in the order found.
    """

    def __init__(self, issues: Sequence[Issue]) -> None:
        self.issues: tuple[Issue, ...] = tuple(issues)
        super().__init__('\n'.join(str(issue) for issue in self.issues))

    def __reduce__(self) -> tuple[type[SchemaError], tuple[tuple[Issue, ...]]]:
        """Rebuild the error from its issues, so it survives a trip between processes."""
        return (SchemaError, (self.issues,))


class RangeError(ValueError):
    """A value lies outside one of the schema core's named ranges.

    The range validators raise it, and the report words it as a value of the wrong
    range: ``expected a number in [0, 1], got 2``.

    Parameters:
        expected: What the value should be, such as ``'a positive number'``.
    """

    def __init__(self, expected: str) -> None:
        self.expected = expected
        super().__init__(f'expected {expected}')


class RuleError(ValueError):
    """A cross-field rule is broken, at a key inside the type that checks the rule.

    A rule in a schema dataclass's ``__post_init__`` raises it to put the problem on the
    key at fault rather than on the dataclass as a whole.  The line then gives that
    key's path and the file that set its value, so a problem inherited by several
    entries reads the same in each and is reported once.

    A rule that compares keys passes all of them as ``reads``.  Two files may set them,
    each with a value that is fine on its own, so the line gives the file that set or
    removed any of them last, which is the one that broke the rule.

    Each key path is relative to the dataclass: one key, or a tuple of keys for a key
    deeper inside.  A key may be absent, as when a rule asks for a key that is not set.

    Parameters:
        message: What is wrong.
        at: The key path the line gives; empty for the dataclass itself.
        reads: The key path of each key the rule reads, the one at fault included; when
            omitted, the line gives the file that set ``at``.
    """

    def __init__(
        self, message: str, *, at: KeyPath | str, reads: Sequence[KeyPath | str] = ()
    ) -> None:
        self.at: KeyPath = _as_key_path(at)
        self.reads: tuple[KeyPath, ...] = tuple(_as_key_path(keys) for keys in reads)
        super().__init__(message)


def _as_key_path(keys: KeyPath | str) -> KeyPath:
    """A key path, from one key or a tuple of them."""
    return (keys,) if isinstance(keys, str) else tuple(keys)


def describe_value(value: object) -> str:
    """Describe a value for an error message: its type and its value.

    A string that reads as a number gets a hint, since quoting a number in YAML is
    the usual way to write a string where a number belongs.

    Parameters:
        value: The value read from a file.

    Returns:
        For example ``"int 3"``, or ``"str '0.5' (a number written so YAML reads it as a
        string)"``.
    """
    text = f'{type(value).__name__} {value!r}'
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return text
        return f'{text} (a number written so YAML reads it as a string)'
    return text


def format_key_path(root: object, keys: KeyPath) -> str:
    """Write a key path the way error messages show it.

    Mapping keys join with dots, and a position in a list is written ``[n]``.

    Parameters:
        root: The tree the path leads into, which tells list positions from mapping
            keys.
        keys: The keys from ``root`` down.

    Returns:
        For example ``'bodies[0].radius'``; empty for an empty path.
    """
    text = ''
    node: object = root
    for key in keys:
        if isinstance(node, list | tuple) and isinstance(key, int):
            text = f'{text}[{key}]'
            node = node[key] if 0 <= key < len(node) else None
        else:
            text = f'{text}.{key}' if text else str(key)
            node = node.get(key) if isinstance(node, Mapping) else None
    return text


def report(
    errors: Sequence[Mapping[str, Any]],
    raw: object,
    *,
    source: str,
    origins: Mapping[KeyPath, str],
    lengths: LengthBounds | None = None,
) -> list[Issue]:
    """Turn the errors pydantic raised for one input into issues.

    Missing keys and unknown keys of one mapping each share a line, and so do the
    missing positions of one fixed-length list.  When a value fails every member of
    a union, the members' expectations share a line.  When one member's problems lie
    deeper inside the value, the value matched that member's container, so only that
    member's problems are reported.  Every other problem has a line of its own; a bad
    mapping key and a bad value under it are two problems.  A rule that raises
    :class:`RuleError` is placed on the key it gives.

    Parameters:
        errors: pydantic's errors, from ``ValidationError.errors()``.
        raw: The input pydantic checked.
        source: The source to give when ``origins`` says nothing about a key.
        origins: The source of each value, and of each removed key, keyed by key path, in
            the order they were set or removed.
        lengths: The lengths each list in the input may have, from which a list of
            the wrong length is said to need "exactly" so many items when only one
            length is allowed; when omitted, only the bound pydantic gives is stated.

    Returns:
        One issue per line, in the order pydantic found the problems.
    """
    return _Reporter(raw, source=source, origins=origins, lengths=lengths).issues(errors)


def doubled_keys(raw: object, *, source: str, origins: Mapping[KeyPath, str]) -> list[Issue]:
    """Find each mapping that writes one integer key twice, as ``2`` and as ``'2'``.

    A mapping that passed through JSON has string keys, so merging it with one read
    from YAML can leave both spellings side by side.  An integer key type reads both as
    one key, and one value would be lost without a word.

    Parameters:
        raw: The input, at any depth of which a mapping is looked at.
        source: The source to give when ``origins`` says nothing about a key.
        origins: The source of each value, keyed by the value's key path.

    Returns:
        One issue per doubled key.
    """
    reporter = _Reporter(raw, source=source, origins=origins)
    issues: list[Issue] = []

    def visit(node: object, keys: KeyPath) -> None:
        """Look at one value, and everything inside it."""
        if isinstance(node, list | tuple):
            for index, item in enumerate(node):
                visit(item, (*keys, index))
            return
        if not isinstance(node, Mapping):
            return
        for key, item in node.items():
            # type() rather than isinstance(): a boolean key is not an integer key.
            if type(key) is int and str(key) in node:
                sources = dict.fromkeys(
                    reporter.origin((*keys, written)) for written in (key, str(key))
                )
                message = f"the key {key} is written twice, as {key} and '{key}'"
                issues.append(Issue(', '.join(sources), format_key_path(raw, keys), message))
            visit(item, (*keys, key))

    visit(raw, ())
    return issues


@dataclasses.dataclass
class _Found:
    """One problem, located in the input, before problems at one place are combined.

    Parameters:
        keys: The key path of the value the problem is about.
        slot: What a problem must share with another to be combined with it, or None
            when it is never combined.
        kind: ``'missing'``, ``'unknown'``, ``'length'``, ``'expected'``, or
            ``'message'``.
        words: For ``'missing'`` and ``'unknown'``, the keys; for ``'length'``, the
            missing positions; for ``'expected'``, what the value should have been;
            for ``'message'``, the message.
        got: For ``'expected'`` and ``'length'``, how the value is described.
        alone: The message to use when nothing is combined with this problem, or None
            for the message built from ``words``.
        named: For ``'unknown'`` and ``'missing'``, the key path of each key the line
            gives, whose sources the issue gives; otherwise empty, and the issue gives
            the sources of ``keys``.
        reads: For a rule that compares keys, the key path of each key it reads, of
            which the issue gives the source set last; otherwise empty.
        reach: The deepest key path in the input the error was located at, which for a
            missing key includes the key.
        tags: Each union member's tag in the error's location, with the number of
            input keys before it, which is where the union lies.
    """

    keys: KeyPath
    slot: tuple[Any, ...] | None
    kind: str
    words: list[str]
    got: str = ''
    alone: str | None = None
    named: list[KeyPath] = dataclasses.field(default_factory=list)
    reads: list[KeyPath] = dataclasses.field(default_factory=list)
    reach: KeyPath = ()
    tags: tuple[tuple[int, Any], ...] = ()

    def member(self, depth: int) -> tuple[Any, ...]:
        """The tags of the union members at ``depth`` keys into the input, outermost first."""
        return tuple(tag for at, tag in self.tags if at == depth)


class _Reporter:
    """Turn pydantic's errors for one input into issues in the house format."""

    def __init__(
        self,
        raw: object,
        *,
        source: str,
        origins: Mapping[KeyPath, str],
        lengths: LengthBounds | None = None,
    ) -> None:
        self._raw = raw
        self._source = source
        self._origins = origins
        self._lengths = lengths

    def issues(self, errors: Sequence[Mapping[str, Any]]) -> list[Issue]:
        """Every issue, with the problems that share a line combined."""
        combined: dict[tuple[Any, ...], _Found] = {}
        lines: list[_Found] = []
        located = [self._locate(error) for error in errors]
        for found in located:
            if found.tags and _matched_elsewhere(found, located):
                continue
            if found.slot is not None and found.slot in combined:
                previous = combined[found.slot]
                previous.words.extend(w for w in found.words if w not in previous.words)
                previous.named.extend(found.named)
                previous.alone = None
                continue
            if found.slot is not None:
                combined[found.slot] = found
            lines.append(found)
        return [self._issue(found) for found in lines]

    def _locate(self, error: Mapping[str, Any]) -> _Found:
        """Place one pydantic error on the input's keys and classify it."""
        kind = str(error['type'])
        located = self._keys(tuple(error['loc']), keep_last=kind == 'missing')
        found = self._classify(error, located)
        found.reach = max(found.keys, located.keys, key=len)
        found.tags = located.tags
        return found

    def _classify(self, error: Mapping[str, Any], located: _Located) -> _Found:
        """Classify one pydantic error, placed at ``located`` in the input."""
        kind = str(error['type'])
        ctx = error.get('ctx') or {}
        value = error['input']
        keys = located.keys
        if kind == 'missing' and located.at_position:
            return _Found(
                keys[:-1],
                (keys[:-1], 'length'),
                'length',
                [str(keys[-1])],
                describe_value(_value_at(self._raw, keys[:-1])),
            )
        if kind == 'missing':
            return _Found(
                keys[:-1], (keys[:-1], 'missing'), 'missing', [str(keys[-1])], named=[keys]
            )
        if kind in _UNKNOWN:
            word = str(value) if kind == 'invalid_key' else str(keys[-1])
            return _Found(keys[:-1], (keys[:-1], 'unknown'), 'unknown', [word], named=[keys])
        if kind == 'union_tag_not_found':
            tag = _unquote(ctx['discriminator'])
            return _Found(keys, (keys, 'missing'), 'missing', [tag], named=[(*keys, tag)])
        if kind == 'union_tag_invalid':
            name = _unquote(ctx['discriminator'])
            tag = value[name] if isinstance(value, Mapping) and name in value else ctx['tag']
            choices = _choices(str(ctx['expected_tags']))
            message = f'{tag!r} is not one of {choices}'
            return _Found((*keys, name), None, 'message', [message])
        if kind in _RULE and isinstance(ctx['error'], RuleError):
            rule = ctx['error']
            reads = [self._extend(keys, path) for path in rule.reads]
            return _Found(self._extend(keys, rule.at), None, 'message', [str(rule)], reads=reads)
        if kind in _RULE and not isinstance(ctx['error'], RangeError):
            return _Found(keys, None, 'message', [str(ctx['error'])])
        if kind in ('too_short', 'too_long'):
            return _Found(
                keys, None, 'expected', [self._length_words(kind, keys, ctx)], describe_value(value)
            )
        # A failed union member leaves its tag in the location; only then do the
        # members' expectations share one line.
        slot = (keys, 'expected', located.about_key) if located.tags else None
        suffix = ' as the key' if located.about_key else ''
        if kind in _RULE:
            words = f'{ctx["error"].expected}{suffix}'
            return _Found(keys, slot, 'expected', [words], repr(value))
        if kind in _CHOICE:
            choices = _choices(str(ctx['expected']))
            subject = f'the key {value!r}' if located.about_key else repr(value)
            return _Found(
                keys,
                slot,
                'expected',
                [f'one of {choices}{suffix}'],
                describe_value(value),
                alone=f'{subject} is not one of {choices}',
            )
        expected = _expected_words(kind, ctx)
        if expected is not None:
            got = describe_value(value) if kind in _EXPECTED else repr(value)
            return _Found(keys, slot, 'expected', [f'{expected}{suffix}'], got)
        message = str(error['msg'])
        return _Found(keys, None, 'message', [message[:1].lower() + message[1:]])

    def _length_words(self, kind: str, keys: KeyPath, ctx: Mapping[str, Any]) -> str:
        """What a list of the wrong length should have been, such as "a list of at most 2
        items".

        pydantic gives only the bound the list broke.  The list needs exactly that many
        items when every list type the schema allows at ``keys`` with that bound has
        the same bound on its other side.
        """
        bounds = self._bounds(keys)
        if kind == 'too_short':
            count = int(ctx['min_length'])
            others = {most for fewest, most in bounds if fewest == count}
            amount = f'at least {count}'
        else:
            count = int(ctx['max_length'])
            others = {fewest for fewest, most in bounds if most == count}
            amount = f'at most {count}'
        if others == {count}:
            amount = f'exactly {count}'
        return f'a list of {amount} items'

    def _bounds(self, keys: KeyPath) -> frozenset[tuple[int, int | None]]:
        """The lengths the schema allows for the list at ``keys``; empty when not known."""
        return frozenset() if self._lengths is None else self._lengths(keys)

    def _extend(self, keys: KeyPath, at: KeyPath) -> KeyPath:
        """``keys`` followed by a rule's relative key path ``at``.

        A key of ``at`` that the input writes as a string of digits, as a mapping that
        passed through JSON does, is given as the input writes it.
        """
        extended = list(keys)
        node: object = _value_at(self._raw, keys)
        for key in at:
            written = key
            if isinstance(node, Mapping) and key not in node and str(key) in node:
                written = str(key)
            extended.append(written)
            if isinstance(node, Mapping):
                node = node.get(written)
            elif isinstance(node, list | tuple) and _is_position(node, written):
                node = node[written]
            else:
                node = None
        return tuple(extended)

    def _keys(self, loc: tuple[Any, ...], *, keep_last: bool) -> _Located:
        """Map a pydantic location onto the input's keys.

        A segment that leads nowhere in the input is a union member's tag, and is
        dropped, except a missing key's own name or list position at the end.
        """
        keys: list[Any] = []
        node: object = self._raw
        about_key = False
        tags: list[tuple[int, Any]] = []
        at_position = False
        for index, segment in enumerate(loc):
            if segment == _KEY_SEGMENT:
                about_key = True
                continue
            if isinstance(node, Mapping) and segment in node:
                keys.append(segment)
                node = node[segment]
                continue
            if isinstance(node, list | tuple) and _is_position(node, segment):
                keys.append(segment)
                node = node[segment]
                continue
            if keep_last and index == len(loc) - 1:
                at_position = isinstance(node, list | tuple) and isinstance(segment, int)
                keys.append(segment)
            else:
                tags.append((len(keys), segment))
        return _Located(tuple(keys), about_key=about_key, tags=tuple(tags), at_position=at_position)

    def _issue(self, found: _Found) -> Issue:
        """Write one combined problem as an issue."""
        if found.alone is not None:
            message = found.alone
        elif found.kind == 'missing':
            message = f'missing required keys: {sorted(found.words)}'
        elif found.kind == 'unknown':
            message = f'unknown keys: {sorted(found.words)}'
        elif found.kind == 'length':
            count = max(int(word) for word in found.words) + 1
            message = f'expected a list of exactly {count} items, got {found.got}'
        elif found.kind == 'expected':
            message = f'expected {" or ".join(found.words)}, got {found.got}'
        else:
            message = found.words[0]
        sources = [self._last_origin(found.reads)] if found.reads else []
        if not any(sources):
            sources = [self.origin(keys) for keys in found.named or [found.keys]]
        label = ', '.join(dict.fromkeys(sources))
        return Issue(label, format_key_path(self._raw, found.keys), message)

    def _last_origin(self, paths: Sequence[KeyPath]) -> str:
        """The source that set or removed a value at one of ``paths`` last.

        The origins are in the order their layers applied.  A value inside a list has the
        list's origin.  Empty when ``paths`` have no origin.
        """
        last = ''
        for recorded, label in self._origins.items():
            for path in paths:
                depth = min(len(recorded), len(path))
                if recorded[:depth] == path[:depth]:
                    last = label
        return last

    def origin(self, keys: KeyPath) -> str:
        """The source to give for a problem at ``keys``.

        The sources that set the value at ``keys`` or a value inside it; failing
        that, the same for the enclosing mapping, and so on up to the root.
        """
        for depth in range(len(keys), -1, -1):
            prefix = keys[:depth]
            inside = [label for path, label in self._origins.items() if path[:depth] == prefix]
            if inside:
                return ', '.join(dict.fromkeys(inside))
        return self._source


@dataclasses.dataclass(frozen=True)
class _Located:
    """Where in the input one pydantic error lies.

    Parameters:
        keys: The key path.
        about_key: Whether the problem is with a mapping key itself.
        tags: Each union member's tag dropped from the location, with the number of
            keys before it.
        at_position: Whether the last key is a missing position of a fixed-length list.
    """

    keys: KeyPath
    about_key: bool
    tags: tuple[tuple[int, Any], ...]
    at_position: bool


def _matched_elsewhere(found: _Found, located: Sequence[_Found]) -> bool:
    """Whether another member of a union that ``found`` failed got deeper into the value.

    A value fails a union when it fails every member.  When another member's error
    lies deeper inside the value than this member's, the value matched that member's
    container, such as a list given where a list or a string belongs, so this
    member's error says nothing useful.  A union may be a member of another union, so
    a member is told apart by every tag at its depth together.
    """
    for depth in {depth for depth, _ in found.tags}:
        union, member = found.reach[:depth], found.member(depth)
        for other in located:
            if (
                len(other.reach) > len(found.reach)
                and other.reach[:depth] == union
                and other.member(depth) not in ((), member)
            ):
                return True
    return False


def _is_position(items: Sequence[Any], segment: Any) -> bool:
    """Whether a location segment is a position inside ``items``."""
    return isinstance(segment, int) and 0 <= segment < len(items)


def _value_at(root: object, keys: KeyPath) -> object:
    """The value at ``keys`` in ``root``, which every key leads into."""
    node: Any = root
    for key in keys:
        node = node[key]
    return node


def _unquote(text: str) -> str:
    """A pydantic context value without the quotes pydantic wraps around a name."""
    return str(text).strip("'")


def _choices(expected: str) -> list[str]:
    """The sorted values of a Literal or an Enum, from pydantic's list of them.

    The schema core allows only string values, which pydantic writes quoted.
    """
    return sorted(ast.literal_eval(text) for text in _QUOTED.findall(expected))


def _expected_words(kind: str, ctx: Mapping[str, Any]) -> str | None:
    """What a value should have been, or None for an error that does not say."""
    if kind in _EXPECTED:
        return _EXPECTED[kind]
    if kind == 'greater_than':
        return 'a positive number' if ctx['gt'] == 0 else f'a number greater than {ctx["gt"]:g}'
    if kind == 'greater_than_equal':
        return 'a non-negative number' if ctx['ge'] == 0 else f'a number of at least {ctx["ge"]:g}'
    if kind == 'less_than':
        return f'a number less than {ctx["lt"]:g}'
    if kind == 'less_than_equal':
        return f'a number of at most {ctx["le"]:g}'
    return None
