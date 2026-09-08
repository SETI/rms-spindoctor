"""YAML-backed configuration for SpinDoctor.

Loads and merges settings from bundled defaults and optional user YAML files
using :class:`ruamel.yaml.YAML` (safe typ), then exposes sections as
:class:`spindoctor.support.attrdict.AttrDict` for attribute-style access. The
:class:`Config` class is the main public entry; helpers such as
:func:`_as_str_list` validate list-shaped YAML fragments used when building
config structures.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.support.attrdict import AttrDict


def _strip_underscore_keys(value: Any) -> Any:
    """Recursively drop mapping keys whose name starts with ``_``.

    Used by :meth:`Config._load_yaml` to remove documentation-only
    ``_sources`` blocks before merging.  Lists are walked element-wise;
    scalars pass through unchanged.

    Parameters:
        value: A YAML-parsed value (mapping, list, or scalar).

    Returns:
        ``value`` with every underscore-prefixed mapping key removed.
    """
    if isinstance(value, dict):
        return {
            k: _strip_underscore_keys(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith('_'))
        }
    if isinstance(value, list):
        return [_strip_underscore_keys(v) for v in value]
    return value


def _deep_merge(base: dict[Any, Any], overlay: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new mapping.

    When a key is present in both mappings and both values are dictionaries,
    the values are merged key-by-key so sibling keys in ``base`` are
    preserved.  For any other case (one side is not a mapping, or the key is
    new) the ``overlay`` value replaces the ``base`` value; lists and scalars
    are never merged element-wise.

    Parameters:
        base: The starting mapping (e.g. bundled defaults).
        overlay: The mapping whose values take precedence at each leaf.

    Returns:
        A new ``dict`` containing the deep-merged result.  ``base`` and
        ``overlay`` are not mutated.
    """

    merged: dict[Any, Any] = dict(base)
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = _deep_merge(base_value, overlay_value)
        else:
            merged[key] = overlay_value
    return merged


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
        self._config_environment: dict[str, Any] = AttrDict({})
        self._config_general: dict[str, Any] = AttrDict({})
        self._config_logging: dict[str, Any] = AttrDict({})
        self._config_offset: dict[str, Any] = AttrDict({})
        self._config_bodies: dict[str, Any] = AttrDict({})
        self._config_body_shape: dict[str, Any] = AttrDict({})
        self._config_rings: dict[str, Any] = AttrDict({})
        self._config_stars: dict[str, Any] = AttrDict({})
        self._config_titan: dict[str, Any] = AttrDict({})
        self._config_bootstrap: dict[str, Any] = AttrDict({})
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
        self._config_bootstrap = AttrDict(self._config_dict.get('bootstrap', {}))
        self._config_backplanes = AttrDict(self._config_dict.get('backplanes', {}))
        self._config_pds4 = AttrDict(self._config_dict.get('pds4', {}))
        self._config_orchestrator = AttrDict(self._config_dict.get('orchestrator', {}))
        self._config_results_tree = AttrDict(self._config_dict.get('results_tree', {}))

    def _load_yaml(self, config_path: str | Path) -> dict[str, Any]:
        """Loads a YAML file and returns a dictionary mapping.

        Documentation-only ``_sources`` blocks (and any other top-level or
        nested key that starts with an underscore) are stripped at load
        time so citations can live alongside values for human review
        without bloating the parsed config.
        """

        yaml = YAML(typ='safe')
        with open(config_path, encoding='utf-8') as f:
            loaded = yaml.load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'Config "{config_path}" did not parse to a dictionary mapping')
        stripped = _strip_underscore_keys(loaded)
        # ``_strip_underscore_keys`` returns ``Any`` because it walks
        # arbitrary YAML; the top-level shape is guaranteed dict because
        # ``loaded`` was checked above.
        assert isinstance(stripped, dict)
        return stripped

    def read_config(self, config_path: str | Path | None = None, reread: bool = False) -> None:
        """Reads configuration from the specified YAML file.

        Parameters:
            config_path: Path to the configuration file. If None, uses the default
                config files.
            reread: Whether to reread the configuration file if it has already been read.

        Raises:
            ValueError: If any registered NavTechnique's confidence spec
                references an attribute the technique does not declare.
                Validation runs once per ``read_config`` invocation.
        """

        if not reread and self._config_dict:
            return

        if config_path is None:
            config_dir = Path(__file__).resolve().parent.parent / 'config_files'
            # The no-path branch always rebuilds from the full config_files
            # glob, so start from an empty dict.  On a reread this ensures keys
            # removed from the YAML files since the previous load do not survive
            # (update_config merges *into* the dict, so a stale dict would union
            # old and new keys rather than producing a clean reload).
            self._config_dict = {}
            # A clean rebuild from bundled defaults carries no user overrides.
            self._override_paths = []
            for filename in sorted(config_dir.glob('*.yaml')):
                self.update_config(filename, read_default=False)
            self._validate_registered_techniques()
            return

        self._config_dict = self._load_yaml(config_path)
        # An explicit path replaces the bundled defaults wholesale; record it
        # as the single applied override so provenance can reproduce the load.
        self._override_paths = [str(config_path)]
        self._update_attrdicts()
        self._validate_registered_techniques()

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

    def update_config(self, config_path: str | Path, read_default: bool = True) -> None:
        """Updates the current configuration with values from the specified YAML file.

        When ``read_default`` is true the merged YAML is re-validated against
        every registered :class:`NavTechnique` so per-technique overrides
        (``confidence_spec`` and ``tuning``) take effect.  The internal
        bootstrap path inside :meth:`read_config` calls with
        ``read_default=False`` and validates once after the loop, since
        early default files are loaded before
        ``config_510_techniques.yaml`` has populated the techniques block.

        Parameters:
            config_path: Path to the configuration file containing update values.
            read_default: Whether to read the default configuration file if no config
                has been previously read.
        """

        if read_default:
            self.read_config()
        new_config = self._load_yaml(config_path)
        for key in new_config:
            if key in self._config_dict and isinstance(self._config_dict[key], dict):
                overlay = new_config[key]
                if not isinstance(overlay, dict):
                    # A section written with no body parses to None, which would
                    # otherwise surface as an AttributeError from inside the merge.
                    kind = 'empty' if overlay is None else type(overlay).__name__
                    raise ValueError(
                        f'Config "{config_path}" section "{key}" is {kind}; expected a '
                        f'mapping because "{key}" is a mapping in the merged configuration'
                    )
                self._config_dict[key] = _deep_merge(self._config_dict[key], overlay)
            else:
                self._config_dict[key] = new_config[key]
        if read_default:
            # ``read_default=True`` marks a user/CLI override file (the
            # internal bundled-defaults bootstrap loads with
            # ``read_default=False``); record it in application order so
            # provenance can list exactly which overrides shaped the run.
            self._override_paths.append(str(config_path))
        self._update_attrdicts()
        if read_default:
            self._validate_registered_techniques()

    @property
    def override_paths(self) -> tuple[str, ...]:
        """The user/CLI override config files applied, in application order.

        Bundled default files (the ``config_files`` glob loaded by
        :meth:`read_config` with no path) are not overrides and never
        appear here; only files loaded via :meth:`update_config` with
        ``read_default=True`` (user ``nav_default_config.yaml`` or
        ``--config-file`` paths) or an explicit :meth:`read_config` path
        are recorded.
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
        """Returns the per-body shape catalogue (``config_220_body_shape.yaml``).

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
    def bootstrap(self) -> Any:
        """Returns the bootstrap configuration settings."""

        self.read_config()
        return self._config_bootstrap

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
