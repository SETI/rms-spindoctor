"""YAML-backed configuration for SpinDoctor.

Reads the packaged configuration files in filename order, then any user override
files, and merges them with :func:`spindoctor.schema.layers.merge` under
:attr:`~spindoctor.schema.layers.NullPolicy.REFUSE`: mappings merge key by key, any
other value replaces the earlier one whole, and a null in any file is refused.  The
typed sections (:mod:`spindoctor.config.sections`) are parsed into frozen dataclasses
each time a configuration is installed.  The other sections are exposed as
:class:`spindoctor.support.attrdict.AttrDict` for attribute-style access.  The
:class:`Config` class is the main public entry; helpers such as :func:`_as_str_list`
validate list-shaped YAML fragments used when building config structures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spindoctor.schema import (
    Issue,
    KeyPath,
    Layer,
    NullPolicy,
    SchemaError,
    describe_value,
    merge,
    parse,
    read_layer,
)
from spindoctor.support.attrdict import AttrDict

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    # nav_records is imported whole so the results_tree annotation is qualified: the
    # documentation lists its class where it is defined and where it is re-exported, and
    # a bare name would match both.
    import spindoctor.nav_records
    from spindoctor.config.logging_keys import LoggingSection
    from spindoctor.config.sections import (
        EnvironmentSection,
        GeneralSection,
        OffsetSection,
        OrchestratorSection,
        TypedSections,
    )
    from spindoctor.nav_technique.confidence_config import TechniquesSection

_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config_files'
"""The directory of the packaged configuration files, merged in filename order."""

_BLOB_MAX_OVERFLOW_KEY = 'blob_max_overflow_fraction'
"""``bodies`` key holding the overflow limit for BODY_BLOB emission."""

_CAMERA_MARKER_KEY = 'data_units'
"""Key every camera block of the instrument configuration declares."""

_ONE_STAR_ELLIPSE_KEY = 'one_star_ellipse_fraction'
"""Camera key: the fraction of the offset window inside which a lone star is believed."""

_PACKAGED_LABEL = 'the packaged configuration'
"""The source a message gives for a packaged value when no file is recorded for it."""

_MERGED_LABEL = 'the merged configuration'
"""The label of the configuration as it stands when an override is merged over it."""


def _typed_sections_type() -> type[TypedSections]:
    """Return the type the typed sections of a configuration are parsed into.

    The section types import modules that import this package, so the import waits
    until a configuration is installed, by which time this package has loaded.

    Returns:
        The type :func:`~spindoctor.config.sections.typed_sections_type` returns.
    """
    from spindoctor.config.sections import typed_sections_type

    return typed_sections_type()


def _configure_techniques(sections: TypedSections) -> None:
    """Assign each shipped technique class its confidence formula and tuning.

    Parameters:
        sections: The typed sections of the configuration being installed.
    """
    from spindoctor.nav_technique.confidence_config import shipped_techniques

    for technique in shipped_techniques():
        block = sections.techniques.block(technique)
        technique.confidence_spec = block.confidence_spec()
        technique.tuning = block.technique_tuning()


def _parse_typed_sections(
    config_dict: Mapping[str, Any], origins: Mapping[KeyPath, str], *, source: str
) -> tuple[TypedSections, dict[KeyPath, str]]:
    """Parse the typed sections of a merged configuration.

    Parameters:
        config_dict: The merged configuration.
        origins: The label of the file that set each value, keyed by key path, in the
            order the files merged.  Values of other sections may be included.
        source: The label a message gives for a value that ``origins`` does not
            cover.

    Returns:
        The typed sections, and the entries of ``origins`` that lie inside them.

    Raises:
        SchemaError: Listing every problem in the typed sections, each line in the
            form ``file: key.path: message``.
    """
    sections_type = _typed_sections_type()
    keys = {item.name for item in dataclasses.fields(sections_type)}
    raw = {key: value for key, value in config_dict.items() if key in keys}
    kept = {path: label for path, label in origins.items() if path[0] in keys}
    return parse(sections_type, raw, source=source, origins=kept), kept


def _refuse_replaced_sections(merged: Mapping[Any, Any], layer: Layer) -> None:
    """Refuse an override that gives a section as anything but a mapping.

    A section is a top-level key whose value in the merged configuration is a
    mapping.  The merge would let a scalar or a list replace it whole, which would
    leave every reader of the section to fail on its own.  A null is left to the
    merge, which refuses every null.

    Parameters:
        merged: The configuration merged so far.
        layer: The override about to be merged over it.

    Raises:
        SchemaError: Naming the file and each section it gives as something else.
    """
    issues = [
        Issue(
            layer.label,
            str(key),
            f'expected a mapping, since {key} is a section of the configuration; '
            f'got {describe_value(value)}',
        )
        for key, value in layer.data.items()
        if value is not None
        and isinstance(merged.get(key), dict)
        and not isinstance(value, Mapping)
    ]
    if issues:
        raise SchemaError(issues)


HASH_EXCLUDED_SECTIONS = frozenset({'environment', 'logging', 'results_tree'})
"""Sections left out of :meth:`Config.resolved_config_hash`.

A configuration section belongs here when it cannot change what the pipeline
computes.  ``logging`` decides what is written down about a run, never what
the run concludes, so two results that differ only in it were produced by the
same configuration and should compare as such.  ``environment`` holds
deployment locations -- where the holdings are read from, where results are
written, which database indexes them -- and moving a directory does not change
a single number the pipeline produces, so a digest that shifted when it moved
would answer its own question wrongly.  ``results_tree`` says how many requests
a pass over a results tree makes at once, which decides how long the pass takes
and nothing about what any document in it says.

Named explicitly rather than inferred, so that a section added later has to be
a deliberate choice rather than inheriting one.  Adding a section here changes
every future digest, and results either side of that change compare as
differently configured -- which is the cost this set exists to stop paying
repeatedly.
"""


def _stringify_keys(value: Any) -> Any:
    """Recursively coerce every mapping key to ``str``.

    Used by :meth:`Config.resolved_config_hash` so the deterministic JSON
    serialization never hits the mixed-key-type comparison error that
    ``json.dumps(..., sort_keys=True)`` raises (per-instrument blocks use
    integer keys such as image sizes and PSF-size thresholds).

    Parameters:
        value: A YAML-parsed value (mapping, list, or scalar).

    Returns:
        ``value`` with every mapping key converted to its ``str`` form;
        lists are walked element-wise and scalars pass through unchanged.
    """

    if isinstance(value, dict):
        return {str(k): _stringify_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_keys(v) for v in value]
    return value


def _camera_blocks(config_dict: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return every camera block of a configuration with its dotted path.

    A camera block is a mapping that declares ``data_units``: a top-level section for a
    single camera, or a mapping one level inside a section that holds several cameras.

    Parameters:
        config_dict: The merged configuration.

    Returns:
        ``(path, block)`` for each camera block, in the configuration's order.
    """
    blocks: list[tuple[str, dict[str, Any]]] = []
    for name, section in config_dict.items():
        if not isinstance(section, dict):
            continue
        if _CAMERA_MARKER_KEY in section:
            blocks.append((str(name), section))
            continue
        for camera, block in section.items():
            if isinstance(block, dict) and _CAMERA_MARKER_KEY in block:
                blocks.append((f'{name}.{camera}', block))
    return blocks


def _as_str_list(value: Any, *, location: str) -> list[str]:
    """Coerce a YAML list value to ``list[str]``.

    Parameters:
        value: Parsed YAML fragment expected to be a list of strings.
        location: Human-readable path (e.g. config key path) for error messages.

    Returns:
        A new ``list[str]`` containing each element of ``value`` as ``str``.

    Raises:
        TypeError: If ``value`` is not a list, or if any element is not a ``str``.
    """

    if not isinstance(value, list):
        raise TypeError(f'{location}: expected a list of strings, got {type(value).__name__}')
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f'{location}[{i}]: expected str, got {type(item).__name__}')
        out.append(item)
    return out


class Config:
    """Manages configuration settings for the navigation system.

    This class handles loading, updating, and accessing configuration settings from YAML files.
    It provides access to various configuration sections through properties and methods.
    """

    def __init__(self) -> None:
        """Initializes a new Config instance with empty configuration containers."""

        self._config_dict: dict[str, Any] = {}
        self._category_cache: dict[str, AttrDict] = {}
        self._override_paths: list[str] = []
        self._sections: TypedSections | None = None
        self._typed_origins: dict[KeyPath, str] = {}
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_body_shape: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_backplanes: dict[str, Any] = AttrDict({})
        self._config_pds4: dict[str, Any] = AttrDict({})

    @property
    def is_loaded(self) -> bool:
        """Whether merged YAML is present (after ``read_config`` / ``update_config``)."""

        return bool(self._config_dict)

    def ensure_loaded(self) -> None:
        """Load bundled default YAML if not already loaded.

        Safe to call repeatedly; delegates to :meth:`read_config` with no path
        (same early-return behavior when data is already present).
        """

        self.read_config()

    def _update_attrdicts(self) -> None:
        """Updates all attribute dictionaries from the main configuration dictionary.

        Converts dictionary sections to AttrDict instances for convenient attribute-style access.
        """

        # The config was (re)loaded; drop any cached per-category AttrDicts so
        # category() rebuilds them from the fresh dict (mirrors the named
        # section properties below being reassigned here).
        self._category_cache = {}
        self._config_bodies = AttrDict(self._config_dict.get('bodies', {}))
        self._config_body_shape = AttrDict(self._config_dict.get('body_shape', {}))
        self._config_rings = AttrDict(self._config_dict.get('rings', {}))
        self._config_stars = AttrDict(self._config_dict.get('stars', {}))
        self._config_titan = AttrDict(self._config_dict.get('titan', {}))
        self._config_backplanes = AttrDict(self._config_dict.get('backplanes', {}))
        self._config_pds4 = AttrDict(self._config_dict.get('pds4', {}))

    def _install(
        self,
        candidate: dict[str, Any],
        override_paths: list[str],
        *,
        origins: Mapping[KeyPath, str],
        source: str,
    ) -> None:
        """Parse and validate a merged configuration, and keep it only if it passes.

        A refused candidate leaves the configuration, its typed sections, and
        :attr:`override_paths` as they were.

        Parameters:
            candidate: The merged configuration to keep.
            override_paths: The override files it includes, in order.
            origins: The label of the file that set each value of the typed sections,
                keyed by key path, in the order the files merged.
            source: The label a message gives for a value that ``origins`` does not
                cover.

        Raises:
            SchemaError: If a typed section breaks its schema.
            ValueError: If the candidate fails :meth:`_validate`.
        """
        sections, typed_origins = _parse_typed_sections(candidate, origins, source=source)
        self._validate(candidate)
        self._config_dict = candidate
        self._override_paths = override_paths
        self._sections = sections
        self._typed_origins = typed_origins
        self._update_attrdicts()
        _configure_techniques(sections)

    def read_config(self, *, reread: bool = False) -> None:
        """Read the packaged configuration files, unless they have been read already.

        The files merge in filename order.  Every key beginning with an underscore is
        dropped as its file is read, so a file can carry notes, such as ``_sources``
        citations, beside its values.  A reread starts again from the packaged files
        alone, so it drops every override applied since.

        Parameters:
            reread: Whether to read the files again if the configuration is loaded.

        Raises:
            SchemaError: If a file is not a YAML mapping, or holds a null, or if a
                typed section breaks its schema, as with a technique block that names
                a diagnostic attribute its technique does not declare.
            ValueError: If ``bodies.blob_max_overflow_fraction`` is not a finite number
                within ``[0, 1]``, or if a camera block sets no
                ``one_star_ellipse_fraction`` within ``(0, 1]``.  Validation runs once
                per ``read_config`` invocation.
        """

        if not reread and self._config_dict:
            return

        layers = [read_layer(path) for path in sorted(_PACKAGED_CONFIG_DIR.glob('*.yaml'))]
        merged = merge(layers, nulls=NullPolicy.REFUSE)
        self._install(merged.data, [], origins=merged.origins, source=_PACKAGED_LABEL)

    @staticmethod
    def _validate(config_dict: dict[str, Any]) -> None:
        """Apply every load-time check to a merged configuration.

        Both paths that load a configuration (:meth:`read_config` and
        :meth:`update_config`) call this one method, so neither can skip a check the
        other applies.

        Parameters:
            config_dict: The merged configuration.
        """
        Config._validate_bodies_block(config_dict)
        Config._validate_camera_blocks(config_dict)

    @staticmethod
    def _validate_bodies_block(config_dict: dict[str, Any]) -> None:
        """Reject a ``bodies.blob_max_overflow_fraction`` that is not a fraction.

        The value is the overflow limit above which a body emits no ``BODY_BLOB``.
        A non-numeric, non-finite, or out-of-range value would silently disable or
        invert that limit, so it is refused when the configuration is loaded.  A
        configuration that carries no such key is left to fail where the key is read.

        Parameters:
            config_dict: The merged configuration.

        Raises:
            ValueError: If the value is not a finite number within ``[0, 1]``; the
                message names the key and the offending value.
        """
        bodies = config_dict.get('bodies')
        if not isinstance(bodies, dict) or _BLOB_MAX_OVERFLOW_KEY not in bodies:
            return
        value = bodies[_BLOB_MAX_OVERFLOW_KEY]
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        # The chained comparison is False for NaN, so a NaN is refused with the rest.
        if not is_number or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f'bodies.{_BLOB_MAX_OVERFLOW_KEY} must be finite and within [0, 1]; got {value!r}'
            )

    @staticmethod
    def _validate_camera_blocks(config_dict: dict[str, Any]) -> None:
        """Require every camera block to set a lone-star ellipse fraction in ``(0, 1]``.

        ``one_star_ellipse_fraction`` is a property of the camera, so no technique-level
        value stands in for a camera that sets none.  A fraction of 0 would admit every
        lone star's offset and one above 1 would reach past the offset window, so a
        fraction outside ``(0, 1]`` is refused when the configuration is loaded.

        Parameters:
            config_dict: The merged configuration.

        Raises:
            ValueError: If a camera block sets no fraction, or one that is not a number
                within ``(0, 1]``; the message names the block and the key.
        """
        for path, block in _camera_blocks(config_dict):
            if _ONE_STAR_ELLIPSE_KEY not in block:
                raise ValueError(f'{path} sets no {_ONE_STAR_ELLIPSE_KEY}')
            value = block[_ONE_STAR_ELLIPSE_KEY]
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            # The chained comparison is False for NaN, so a NaN is refused with the rest.
            if not is_number or not 0.0 < float(value) <= 1.0:
                raise ValueError(
                    f'{path}.{_ONE_STAR_ELLIPSE_KEY} must lie in (0, 1]; got {value!r}'
                )

    def update_config(self, config_path: str | Path) -> None:
        """Merge a user override file over the configuration.

        The packaged files are read first if they have not been.  The merged
        configuration is then parsed and validated again, and each shipped technique is
        assigned the confidence formula and tuning it sets, so per-technique overrides
        take effect.  A refused override, whatever the reason, leaves the
        configuration, the techniques' settings, and :attr:`override_paths` as they
        were.

        Parameters:
            config_path: The override file.  Error messages and :attr:`override_paths`
                give it as written here.

        Raises:
            FileNotFoundError: If the file does not exist.
            SchemaError: If the file is not a YAML mapping, holds a null, or gives a
                section as anything but a mapping, or if a typed section of the merged
                result breaks its schema, as with an unknown key or a value of the
                wrong type or out of range; each line of the message gives the file
                that set the value and the key path.
            ValueError: If the merged configuration fails the validation
                :meth:`read_config` applies.
        """

        self.read_config()
        layer = read_layer(Path(config_path), label=str(config_path))
        _refuse_replaced_sections(self._config_dict, layer)
        # Only the sections the override names are merged again, and the candidate
        # shares the rest with the configuration as it stands, so the cost follows the
        # size of the override and never the number of overrides before it.
        named = {key: self._config_dict[key] for key in layer.data if key in self._config_dict}
        merged = merge([Layer(named, _MERGED_LABEL), layer], nulls=NullPolicy.REFUSE)
        candidate = {**self._config_dict, **merged.data}
        # A value the override leaves alone keeps the file recorded for it before.
        origins = {
            path: label for path, label in self._typed_origins.items() if path[0] not in named
        }
        origins.update(
            (path, self._typed_origins.get(path, label) if label == _MERGED_LABEL else label)
            for path, label in merged.origins.items()
        )
        self._install(
            candidate,
            [*self._override_paths, str(config_path)],
            origins=origins,
            source=str(config_path),
        )

    @property
    def override_paths(self) -> tuple[str, ...]:
        """The user override files applied by :meth:`update_config`, in order.

        The packaged files are not overrides and never appear here.  Each path is
        given as it was passed to :meth:`update_config`.
        """

        return tuple(self._override_paths)

    def resolved_config_hash(self) -> str:
        """Return a stable sha256 hex digest of the fully-resolved config.

        The merged config dict (bundled defaults plus every applied
        override) is serialized deterministically -- mapping keys are
        stringified and sorted, list order is preserved -- and hashed, so
        two configs with identical resolved content produce identical
        digests regardless of load order or comment/whitespace changes in
        the source YAML files.

        Sections in :data:`HASH_EXCLUDED_SECTIONS` are left out, because this
        digest answers "was this result produced by the same configuration",
        and a setting that cannot change a result would answer it wrongly.
        Raising one component's log level to look into a technique would
        otherwise re-stamp every result in that run as differently
        configured, and it compares against an archive.

        Returns:
            64-character sha256 hex digest of the resolved config content.
        """

        self.read_config()
        hashed = {
            key: value
            for key, value in self._config_dict.items()
            if key not in HASH_EXCLUDED_SECTIONS
        }
        canonical = json.dumps(
            _stringify_keys(hashed),
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def category(self, category: str) -> AttrDict:
        """Returns the configuration settings for the specified category.

        The result is cached per category and rebuilt on config reload (see
        ``_update_attrdicts``), matching the caching semantics of the named
        section properties.  Config is treated read-only; mutating the returned
        AttrDict is not supported.
        """

        self.read_config()
        cached = self._category_cache.get(category)
        if cached is None:
            cached = AttrDict(self._config_dict.get(category, {}))
            self._category_cache[category] = cached
        return cached

    def _parsed(self) -> TypedSections:
        """Return the typed sections of the configuration, reading it first if need be.

        Returns:
            The typed sections, parsed when the configuration was installed.

        Raises:
            RuntimeError: If the configuration is read but holds no parsed sections.
        """
        self.read_config()
        if self._sections is None:
            raise RuntimeError('the configuration is read but holds no parsed sections')
        return self._sections

    @property
    def general(self) -> GeneralSection:
        """Returns the general section, the settings no one part of the pipeline owns."""

        return self._parsed().general

    @property
    def environment(self) -> EnvironmentSection:
        """Returns the environment section, which says where a deployment keeps its files.

        The packaged file sets no location, so one no override names is None.  The helpers
        in :mod:`spindoctor.config.config_helper` define the usual order for a location:
        the command line, then here, then an environment variable.  A reader may apply
        its own order; each takes its location from those three places.
        """

        return self._parsed().environment

    @property
    def logging(self) -> LoggingSection:
        """Returns the logging section: the levels and terminal sinks of the two loggers.

        Holds the per-logger defaults (``main``, ``image``), the per-module
        override categories (``techniques``, ``models``, ``other``), the
        ``strict_scope`` switch, and per-program overrides under ``programs``.
        The section is parsed when the configuration is installed, so every key and
        level has been checked.
        """

        return self._parsed().logging

    @property
    def planets(self) -> list[str]:
        """Returns the list of configured planet names."""

        self.read_config()
        return _as_str_list(self._config_dict.get('planets', []), location='config.planets')

    def satellites(self, planet: str) -> list[str]:
        """Returns the list of satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get satellites for.

        Returns:
            A list of satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.satellites[{planet.upper()!r}]',
        )

    def fuzzy_satellites(self, planet: str) -> list[str]:
        """Returns the list of fuzzy satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get fuzzy satellites for.

        Returns:
            A list of fuzzy satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('fuzzy_satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "fuzzy_satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.fuzzy_satellites[{planet.upper()!r}]',
        )

    def ring_satellites(self, planet: str) -> list[str]:
        """Returns the list of ring satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get ring satellites for.

        Returns:
            A list of ring satellite names for the specified planet.
        """

        self.read_config()
        block = self._config_dict.get('ring_satellites', {})
        if not isinstance(block, dict):
            raise TypeError(
                f'config key "ring_satellites" must be a mapping, got {type(block).__name__}'
            )
        return _as_str_list(
            block.get(planet.upper(), []),
            location=f'config.ring_satellites[{planet.upper()!r}]',
        )

    @property
    def offset(self) -> OffsetSection:
        """Returns the offset section, which sets how finely a correlation resolves a peak."""

        return self._parsed().offset

    @property
    def bodies(self) -> Any:
        """Returns the celestial bodies configuration settings."""

        self.read_config()
        return self._config_bodies

    @property
    def body_shape(self) -> Any:
        """Returns the per-body shape catalog (``config_220_body_shape.yaml``).

        Each entry is keyed by SPICE body name (e.g. ``MIMAS``) and exposes
        ``radii_km``, ``ellipsoid_rms_residual_km``, ``crater_scale_km``,
        ``albedo_variation``, and ``shape_class_hint``.
        Missing-body lookups return ``None`` from ``AttrDict.get``; downstream
        code applies the conservative fallback (10% radius default,
        reliability cap 0.3).
        """

        self.read_config()
        return self._config_body_shape

    @property
    def rings(self) -> Any:
        """Returns the planetary rings configuration settings."""

        self.read_config()
        return self._config_rings

    @property
    def stars(self) -> Any:
        """Returns the stars configuration settings."""

        self.read_config()
        return self._config_stars

    @property
    def titan(self) -> Any:
        """Returns the Titan-specific configuration settings."""

        self.read_config()
        return self._config_titan

    @property
    def orchestrator(self) -> OrchestratorSection:
        """Returns the orchestrator section: the ensemble and the reliability gate.

        The section is parsed when the configuration is installed, so every value has
        been checked against its type and range.
        """

        return self._parsed().orchestrator

    @property
    def techniques(self) -> TechniquesSection:
        """Returns the techniques section: each technique's confidence formula and tuning.

        The section is parsed when the configuration is installed.  Its
        :meth:`~spindoctor.nav_technique.confidence_config.TechniquesSection.block`
        method gives one technique's block.
        """

        return self._parsed().techniques

    @property
    def results_tree(self) -> spindoctor.nav_records.TreeTuning:
        """Returns how much of a pass over a results tree runs at once.

        No command-line option or environment variable names these settings, because
        they describe a machine rather than a run.  A program reads the section once
        and passes it to whatever reads the tree.
        """

        return self._parsed().results_tree

    @property
    def backplanes(self) -> Any:
        """Returns backplanes configuration, including bodies, rings, and target LIDs."""

        self.read_config()
        return self._config_backplanes

    @property
    def pds4(self) -> Any:
        """Returns PDS4 bundle generation configuration."""

        self.read_config()
        return self._config_pds4


DEFAULT_CONFIG = Config()
