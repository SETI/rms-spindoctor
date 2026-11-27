"""How a typed scene field declares itself: its boundary class, meaning, default, and hint.

Every field of a typed scene block is a frozen-dataclass field whose annotation gives
its type and range (see :mod:`spindoctor.schema.core`) and whose metadata holds the
rest, a :class:`FieldDeclaration` under the key :data:`DECLARATION`.
:func:`required` and :func:`optional` build the field, so each key of a scene is
declared once, on its field: its boundary class, what it means, what a scene that
leaves it out gets, and the scene editor's control for it.

An optional key's dataclass default is None, because the typed scene keeps the
difference between a key a scene leaves out and a key it sets.  The generated schema
reference reads the declarations, through :func:`declaration_of`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args

from pydantic import AfterValidator

from spindoctor.schema import Str
from spindoctor.schema.errors import RangeError

__all__ = [
    'BOUNDARIES',
    'DECLARATION',
    'Boundary',
    'EditorHint',
    'FieldDeclaration',
    'IntWithin',
    'NonEmptyStr',
    'declaration_of',
    'optional',
    'required',
]

#: The side of the information boundary a scene key belongs to.  An ``idealized`` key
#: holds what a real pipeline could know, and the navigator may read it.  A ``truth``
#: key holds what only the test knows, which the renderer reads.  A ``test`` key holds
#: what the integration tests check the navigation against, and neither side reads it.
Boundary = Literal['idealized', 'truth', 'test']
#: Every boundary class.
BOUNDARIES: tuple[str, ...] = get_args(Boundary)

#: The metadata key a scene field's declaration is stored under.
DECLARATION = 'scene'


def _non_empty(value: str) -> str:
    """Return a string if it is not empty, else raise the range error."""
    if not value:
        raise RangeError('a non-empty string')
    return value


#: A string with at least one character.
NonEmptyStr = Annotated[Str, AfterValidator(_non_empty)]


@dataclass(frozen=True)
class IntWithin:
    """A check that an integer lies in a closed range, for an ``AfterValidator``.

    Parameters:
        low: The smallest value allowed.
        high: The largest value allowed; None for no upper limit.
    """

    low: int
    high: int | None = None

    @property
    def expected(self) -> str:
        """The range, in the words of an error message."""
        if self.high is None:
            return f'an integer of at least {self.low}'
        return f'an integer from {self.low} to {self.high}'

    def __call__(self, value: int) -> int:
        """Return ``value`` if it lies in the range.

        Raises:
            RangeError: If it does not.
        """
        if value < self.low or (self.high is not None and value > self.high):
            raise RangeError(self.expected)
        return value


@dataclass(frozen=True)
class EditorHint:
    """The scene editor's numeric control for a key.

    The control's range is what the editor offers, which may be narrower than what the
    field accepts.  Its minimum may also sit below the field's range when the editor
    shows that value for a scene that leaves the key out, as ``noise_seed``'s control
    shows -1.

    Parameters:
        minimum: The smallest value the control offers.
        maximum: The largest value the control offers.
        decimals: The decimal places the control shows; None for a whole-number
            control.
        step: The control's step; None for the editor's default step.
    """

    minimum: float
    maximum: float
    decimals: int | None = None
    step: float | None = None


@dataclass(frozen=True)
class FieldDeclaration:
    """What a scene field declares besides its type and range.

    A required key has neither ``default`` nor ``when_absent``; an optional key has
    exactly one.

    Parameters:
        boundary: The side of the information boundary the key belongs to.
        description: What the key means, in a sentence or two of plain words.
        default: The value a scene that leaves the key out gets, for a key with one
            fixed default.
        when_absent: What leaving the key out means, for a key whose absence means
            something else, such as the camera's own value.
        editor: The scene editor's numeric control for the key, or None for a key the
            editor sets some other way.
    """

    boundary: Boundary
    description: str
    default: float | str | bool | None = None
    when_absent: str | None = None
    editor: EditorHint | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown boundary class, an empty description, or two defaults.

        Raises:
            TypeError: If the declaration is malformed.
        """
        if self.boundary not in BOUNDARIES:
            raise TypeError(
                f'{self.boundary!r} is not a boundary class; use one of {list(BOUNDARIES)}'
            )
        if not self.description:
            raise TypeError('a scene field needs a description')
        if self.default is not None and self.when_absent is not None:
            raise TypeError('a scene field gives default or when_absent, not both')

    @property
    def required(self) -> bool:
        """Whether every scene sets the key."""
        return self.default is None and self.when_absent is None


def required(boundary: Boundary, description: str, *, editor: EditorHint | None = None) -> Any:
    """Declare a key every scene sets.

    Parameters:
        boundary: The key's boundary class.
        description: What the key means.
        editor: The editor's control for the key, if it has a numeric one.

    Returns:
        The dataclass field.
    """
    declared = FieldDeclaration(boundary, description, editor=editor)
    return dataclasses.field(metadata={DECLARATION: declared})


def optional(
    boundary: Boundary,
    description: str,
    *,
    default: float | str | bool | None = None,
    when_absent: str | None = None,
    editor: EditorHint | None = None,
) -> Any:
    """Declare a key a scene may leave out.

    Give exactly one of ``default`` and ``when_absent``.

    Parameters:
        boundary: The key's boundary class.
        description: What the key means.
        default: The value a scene that leaves the key out gets.
        when_absent: What leaving the key out means, for a key with no one default
            value, such as ``"the camera's value"``.
        editor: The editor's control for the key, if it has a numeric one.

    Returns:
        The dataclass field, whose default is None.

    Raises:
        TypeError: If both or neither of ``default`` and ``when_absent`` are given.
    """
    declared = FieldDeclaration(
        boundary, description, default=default, when_absent=when_absent, editor=editor
    )
    if declared.required:
        raise TypeError('an optional scene field gives default or when_absent')
    return dataclasses.field(default=None, metadata={DECLARATION: declared})


def declaration_of(item: dataclasses.Field[Any]) -> FieldDeclaration:
    """The declaration a scene field carries.

    Parameters:
        item: A field of a typed scene block.

    Returns:
        The field's declaration.

    Raises:
        TypeError: If the field was not declared with :func:`required` or
            :func:`optional`.
    """
    declared = item.metadata.get(DECLARATION)
    if not isinstance(declared, FieldDeclaration):
        raise TypeError(f'{item.name}: a scene field is declared with required() or optional()')
    return declared
