"""The ``techniques`` configuration section: each technique's confidence formula and tuning.

The section, in ``config_files/config_510_techniques.yaml``, holds one block for each
technique the package ships.  A block holds the technique's confidence formula
(``alpha0``, ``terms``, ``hard_zero_if``, and ``hard_cap``, the fields of
:class:`~spindoctor.nav_technique.confidence.ConfidenceSpec`) and its ``tuning``
mapping.  :class:`~spindoctor.config.config.Config` parses the section through the
schema core each time it installs a configuration, so a bad value, an unknown key, or a
missing block is refused at load with the file and key path.  It then assigns each
technique class its ``confidence_spec`` and ``tuning``.

The section's type is built from the technique registry, so a technique declares its
own tuning and no list of techniques is written twice.  Each technique declares
``Tuning``, a frozen schema dataclass of its tuning keys, and has its ``tuning`` mapping
parsed into that type.  Its block is named by its snake_case key, the one the
``logging.techniques`` category uses.

A block may name only diagnostic attributes its technique declares in
``confidence_attributes``, in ``terms`` and in ``hard_zero_if``; any other name is
refused at load, since the formula would fail on every image.
"""

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any, ClassVar

from spindoctor.config.logging_keys import is_shipped, log_key_for
from spindoctor.nav_technique.confidence import ConfidenceSpec
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.schema import SCHEMA_CONFIG, RuleError

__all__ = [
    'TechniqueBlock',
    'TechniquesSection',
    'shipped_techniques',
    'technique_block_type',
    'technique_config_key',
    'techniques_section_type',
]


def technique_config_key(technique: type[NavTechnique]) -> str:
    """Return the key of a technique's block in the ``techniques`` section.

    Parameters:
        technique: The technique class.

    Returns:
        The technique's snake_case key (:func:`~spindoctor.config.log_key_for`), the
        one the ``logging.techniques`` category uses.
    """
    return log_key_for(technique)


def shipped_techniques() -> tuple[type[NavTechnique], ...]:
    """Return the registered techniques the package ships, in registration order.

    Test modules register techniques of their own, which set their own confidence
    specification and have no block.

    Returns:
        The technique classes.
    """
    return tuple(technique for technique in NavTechnique._registry if is_shipped(technique))


@dataclass(frozen=True)
class TechniqueBlock(ConfidenceSpec):
    """The base of every technique's block type: its confidence formula.

    :func:`technique_block_type` adds a ``tuning`` field typed by the technique, and
    sets :attr:`technique`.

    Raises:
        RuleError: If a term or a ``hard_zero_if`` key names a diagnostic attribute the
            technique does not declare.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    #: The technique the block configures.
    technique: ClassVar[type[NavTechnique]]

    def __post_init__(self) -> None:
        """Refuse a diagnostic attribute the technique does not declare."""
        super().__post_init__()
        declared = self.technique.confidence_attributes
        name = self.technique.name
        message = f'{name} declares no such attribute; it declares {sorted(declared)}'
        for index, term in enumerate(self.terms):
            if term.feature not in declared:
                raise RuleError(message, at=('terms', index, 'feature'))
        for key in self.hard_zero_if:
            if key not in declared:
                raise RuleError(message, at=('hard_zero_if', key))

    def confidence_spec(self) -> ConfidenceSpec:
        """Return the block's confidence formula.

        Returns:
            The formula, without the block's tuning.
        """
        return ConfidenceSpec(
            alpha0=self.alpha0,
            terms=self.terms,
            hard_zero_if=self.hard_zero_if,
            hard_cap=self.hard_cap,
        )

    def technique_tuning(self) -> Any:
        """Return the block's tuning.

        Returns:
            An instance of the technique's ``Tuning``.
        """
        # Each technique's block type adds the field with its own type, so this base
        # declares none.
        return vars(self)['tuning']


@dataclass(frozen=True)
class TechniquesSection:
    """The base of the ``techniques`` section's type.

    :func:`techniques_section_type` adds one field per shipped technique, named by
    :func:`technique_config_key` and typed by :func:`technique_block_type`.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    def block(self, technique: type[NavTechnique]) -> TechniqueBlock:
        """Return a technique's block.

        Parameters:
            technique: A shipped technique class.

        Returns:
            The technique's parsed block.

        Raises:
            TypeError: If the section holds no block for the technique.
        """
        block = getattr(self, technique_config_key(technique), None)
        if not isinstance(block, TechniqueBlock):
            raise TypeError(f'the techniques section holds no block for {technique.name}')
        return block


@functools.cache
def technique_block_type(technique: type[NavTechnique]) -> type[TechniqueBlock]:
    """Return the type of a technique's block.

    Parameters:
        technique: The technique class.

    Returns:
        A frozen schema dataclass: the fields of :class:`TechniqueBlock` and a
        required ``tuning`` field of the technique's ``Tuning``.
    """
    block_type: type[TechniqueBlock] = dataclasses.make_dataclass(
        f'{technique.__name__}Block',
        [('tuning', technique.Tuning, dataclasses.field(kw_only=True))],
        bases=(TechniqueBlock,),
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG, 'technique': technique},
    )
    return block_type


@functools.cache
def techniques_section_type(
    techniques: tuple[type[NavTechnique], ...],
) -> type[TechniquesSection]:
    """Return the type of the ``techniques`` section for a set of techniques.

    Parameters:
        techniques: The shipped techniques (:func:`shipped_techniques`).

    Returns:
        A frozen schema dataclass with one required field per technique.
    """
    section_type: type[TechniquesSection] = dataclasses.make_dataclass(
        'TechniquesSection',
        [(technique_config_key(cls), technique_block_type(cls)) for cls in techniques],
        bases=(TechniquesSection,),
        frozen=True,
        namespace={'__pydantic_config__': SCHEMA_CONFIG},
    )
    return section_type
