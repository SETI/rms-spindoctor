"""YAML-backed configuration for SpinDoctor.

Reads the packaged configuration files in filename order, then any user override
files, and merges them with :func:`spindoctor.schema.layers.merge` under
:attr:`~spindoctor.schema.layers.NullPolicy.REFUSE`: mappings merge key by key, any
other value replaces the earlier one whole, and a null in any file is refused.  The
sections are exposed as :class:`spindoctor.support.attrdict.AttrDict` for
attribute-style access.  The :class:`Config` class is the main public entry; helpers
such as :func:`_as_str_list` validate list-shaped YAML fragments used when building
config structures.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spindoctor.schema import (
    Issue,
    Layer,
    NullPolicy,
    SchemaError,
    describe_value,
    merge,
    read_layer,
)
from spindoctor.support.attrdict import AttrDict

_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config_files'
"""The directory of the packaged configuration files, merged in filename order."""

_BLOB_MAX_OVERFLOW_KEY = 'blob_max_overflow_fraction'
"""``bodies`` key holding the overflow limit for BODY_BLOB emission."""

_CAMERA_MARKER_KEY = 'data_units'
"""Key every camera block of the instrument configuration declares."""

_ONE_STAR_ELLIPSE_KEY = 'one_star_ellipse_fraction'
"""Camera key: the fraction of the offset window inside which a lone star is believed."""


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
        self._packaged_layers: list[Layer] = []
        self._override_layers: list[Layer] = []
        self._config_environment: dict[str, Any] = AttrDict({})
        self._config_general: dict[str, Any] = AttrDict({})
        self._config_logging: dict[str, Any] = AttrDict({})
        self._config_offset: dict[str, Any] = AttrDict({})
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_body_shape: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_backplanes: dict[str, Any] = AttrDict({})
        self._config_pds4: dict[str, Any] = AttrDict({})
        self._config_orchestrator: dict[str, Any] = AttrDict({})
        self._config_results_tree: dict[str, Any] = AttrDict({})

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
        self._config_environment = AttrDict(self._config_dict.get('environment', {}))
        self._config_general = AttrDict(self._config_dict.get('general', {}))
        self._config_logging = AttrDict(self._config_dict.get('logging', {}))
        self._config_offset = AttrDict(self._config_dict.get('offset', {}))
        self._config_bodies = AttrDict(self._config_dict.get('bodies', {}))
        self._config_body_shape = AttrDict(self._config_dict.get('body_shape', {}))
        self._config_rings = AttrDict(self._config_dict.get('rings', {}))
        self._config_stars = AttrDict(self._config_dict.get('stars', {}))
        self._config_titan = AttrDict(self._config_dict.get('titan', {}))
        self._config_backplanes = AttrDict(self._config_dict.get('backplanes', {}))
        self._config_pds4 = AttrDict(self._config_dict.get('pds4', {}))
        self._config_orchestrator = AttrDict(self._config_dict.get('orchestrator', {}))
        self._config_results_tree = AttrDict(self._config_dict.get('results_tree', {}))

    def _merge_layers(self, overrides: list[Layer]) -> None:
        """Merge the packaged layers and then ``overrides`` into the configuration.

        Every key beginning with an underscore was dropped when its file was read, so a
        file can carry notes, such as ``_sources`` citations, beside its values.  A
        merge that raises leaves the configuration as it was.

        Parameters:
            overrides: Every override to apply, in order.

        Raises:
            SchemaError: Naming the file and key path of every null in any layer.
        """
        merged = merge([*self._packaged_layers, *overrides], nulls=NullPolicy.REFUSE)
        self._config_dict = merged.data
        self._override_layers = overrides
        self._update_attrdicts()

    def read_config(self, *, reread: bool = False) -> None:
        """Read the packaged configuration files, unless they have been read already.

        The files merge in filename order.  A reread starts again from the packaged
        files alone, so it drops every override applied since.

        Parameters:
            reread: Whether to read the files again if the configuration is loaded.

        Raises:
            SchemaError: If a file is not a YAML mapping, or holds a null.
            ValueError: If any registered NavTechnique's confidence spec
                references an attribute the technique does not declare, if
                ``bodies.blob_max_overflow_fraction`` is not a finite number within
                ``[0, 1]``, or if a camera block sets no ``one_star_ellipse_fraction``
                within ``(0, 1]``.  Validation runs once per ``read_config`` invocation.
        """

        if not reread and self._config_dict:
            return

        self._packaged_layers = [
            read_layer(path) for path in sorted(_PACKAGED_CONFIG_DIR.glob('*.yaml'))
        ]
        self._merge_layers([])
        self._validate()

    def _validate(self) -> None:
        """Apply every load-time check to the merged configuration.

        Both paths that load a configuration (:meth:`read_config` and
        :meth:`update_config`) call this one method, so neither can skip a check the
        other applies.
        """
        self._validate_registered_techniques()
        self._validate_bodies_block()
        self._validate_camera_blocks()

    def _validate_bodies_block(self) -> None:
        """Reject a ``bodies.blob_max_overflow_fraction`` that is not a fraction.

        The value is the overflow limit above which a body emits no ``BODY_BLOB``.
        A non-numeric, non-finite, or out-of-range value would silently disable or
        invert that limit, so it is refused when the configuration is loaded.  A
        configuration that carries no such key is left to fail where the key is read.

        Raises:
            ValueError: If the value is not a finite number within ``[0, 1]``; the
                message names the key and the offending value.
        """
        bodies = self._config_dict.get('bodies')
        if not isinstance(bodies, dict) or _BLOB_MAX_OVERFLOW_KEY not in bodies:
            return
        value = bodies[_BLOB_MAX_OVERFLOW_KEY]
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        # The chained comparison is False for NaN, so a NaN is refused with the rest.
        if not is_number or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f'bodies.{_BLOB_MAX_OVERFLOW_KEY} must be finite and within [0, 1]; got {value!r}'
            )

    def _validate_camera_blocks(self) -> None:
        """Require every camera block to set a lone-star ellipse fraction in ``(0, 1]``.

        ``one_star_ellipse_fraction`` is a property of the camera, so no technique-level
        value stands in for a camera that sets none.  A fraction of 0 would admit every
        lone star's offset and one above 1 would reach past the offset window, so a
        fraction outside ``(0, 1]`` is refused when the configuration is loaded.

        Raises:
            ValueError: If a camera block sets no fraction, or one that is not a number
                within ``(0, 1]``; the message names the block and the key.
        """
        for path, block in _camera_blocks(self._config_dict):
            if _ONE_STAR_ELLIPSE_KEY not in block:
                raise ValueError(f'{path} sets no {_ONE_STAR_ELLIPSE_KEY}')
            value = block[_ONE_STAR_ELLIPSE_KEY]
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            # The chained comparison is False for NaN, so a NaN is refused with the rest.
            if not is_number or not 0.0 < float(value) <= 1.0:
                raise ValueError(
                    f'{path}.{_ONE_STAR_ELLIPSE_KEY} must lie in (0, 1]; got {value!r}'
                )

    def _validate_registered_techniques(self) -> None:
        """Load + validate every registered NavTechnique's confidence spec.

        Builds each technique's :class:`ConfidenceSpec` from the
        ``techniques.<technique_name>`` block in
        ``config_510_techniques.yaml`` and assigns it to the class
        attribute, then asserts every term references an attribute the
        technique has declared in ``confidence_attributes``.  Test-only
        registry entries whose ``name`` starts with ``_`` are exempt
        from the YAML requirement (the test fixture supplies its own
        spec or uses ``None`` to opt out of confidence evaluation).

        Imported inside the method because ``spindoctor.nav_technique.nav_technique``
        imports ``spindoctor.support.nav_base`` which imports ``spindoctor.config``;
        promoting these imports to the module top would fail with
        ``ImportError: cannot import name 'DEFAULT_CONFIG' from
        partially initialized module 'spindoctor.config'``.  This lazy-import
        site is the minimum cycle break — the other modules can keep
        their top-of-file imports.
        """
        from spindoctor.nav_technique.confidence_config import (
            ConfidenceConfigError,
            load_confidence_spec,
            load_technique_tuning,
        )
        from spindoctor.nav_technique.nav_technique import (
            NavTechnique,
            validate_registered_confidence_specs,
        )

        techniques_block = dict(self._config_dict.get('techniques', {}))
        for cls in NavTechnique._registry:
            try:
                cls.confidence_spec = load_confidence_spec(techniques_block, cls.name)
            except ConfidenceConfigError:
                if not cls.name.startswith('_'):
                    raise
                # Test-only technique: leave spec as inherited (None)
                # so the test fixture controls it.  Tuning still resets
                # below so a stale fixture value cannot leak across reloads.
            cls.tuning = load_technique_tuning(techniques_block, cls.name)
        validate_registered_confidence_specs()

    def update_config(self, config_path: str | Path) -> None:
        """Merge a user override file over the configuration.

        The packaged files are read first if they have not been.  The merged
        configuration is then validated again against every registered
        :class:`NavTechnique`, so per-technique overrides (``confidence_spec`` and
        ``tuning``) take effect.

        Parameters:
            config_path: The override file.  Error messages and :attr:`override_paths`
                give it as written here.

        Raises:
            FileNotFoundError: If the file does not exist.
            SchemaError: If the file is not a YAML mapping, holds a null, or gives a
                section as anything but a mapping; the message names the file and
                the key path.
            ValueError: If the merged configuration fails the validation
                :meth:`read_config` applies.
        """

        self.read_config()
        layer = read_layer(Path(config_path), label=str(config_path))
        _refuse_replaced_sections(self._config_dict, layer)
        self._merge_layers([*self._override_layers, layer])
        self._validate()

    @property
    def override_paths(self) -> tuple[str, ...]:
        """The user override files applied by :meth:`update_config`, in order.

        The packaged files are not overrides and never appear here.  Each path is
        given as it was passed to :meth:`update_config`.
        """

        return tuple(layer.label for layer in self._override_layers)

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

    @property
    def general(self) -> Any:
        """Returns the general configuration settings."""

        self.read_config()
        return self._config_general

    @property
    def environment(self) -> Any:
        """Returns the environment configuration settings."""

        self.read_config()
        return self._config_environment

    @property
    def logging(self) -> Any:
        """Returns the logging configuration settings.

        Holds the per-logger defaults (``main``, ``image``), the per-module
        override categories (``techniques``, ``models``, ``other``), the
        ``strict_scope`` switch, and per-program overrides under ``programs``.
        See :func:`spindoctor.config.logging_keys.validate_logging_config` for
        the accepted keys.
        """

        self.read_config()
        return self._config_logging

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
    def offset(self) -> Any:
        """Returns the offset configuration settings."""

        self.read_config()
        return self._config_offset

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
        ``albedo_mean``, ``albedo_variation``, and ``shape_class_hint``.
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
    def orchestrator(self) -> Any:
        """Returns the orchestrator (ensemble + reliability gate) settings."""

        self.read_config()
        return self._config_orchestrator

    @property
    def results_tree(self) -> Any:
        """Returns how much of a pass over a results tree runs at once.

        :func:`~spindoctor.config.get_results_tree_tuning` is what reads this
        section into a :class:`~spindoctor.nav_records.TreeTuning`.
        """

        self.read_config()
        return self._config_results_tree

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
