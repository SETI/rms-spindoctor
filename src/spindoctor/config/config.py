"""YAML-backed configuration for SpinDoctor.

:func:`load_nav_config` reads the packaged configuration files in filename order, then
any user override files, then any overrides held in memory, and merges them with
:func:`spindoctor.schema.layers.merge` under
:attr:`~spindoctor.schema.layers.NullPolicy.REFUSE`: mappings merge key by key, any
other value replaces the earlier one whole, and a null in any layer is refused.  It then
parses the whole merged configuration into one
:class:`~spindoctor.config.sections.NavConfig`, so an unknown key in any section, and a
top-level section that the navigator does not declare, are refused when the
configuration loads.

:class:`Config` holds the current :class:`~spindoctor.config.sections.NavConfig` and the
overrides it was built from.  Its section properties return the sections of that
object.  :data:`DEFAULT_CONFIG` is the configuration the programs share.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spindoctor.config.sections_camera import CameraBlock
from spindoctor.schema import (
    Issue,
    Layer,
    NullPolicy,
    SchemaError,
    describe_value,
    merge,
    parse,
    read_layer,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    # nav_records is imported whole so the results_tree annotation is qualified: the
    # documentation lists its class where it is defined and where it is re-exported, and
    # a bare name would match both.
    import spindoctor.nav_records
    from spindoctor.config.logging_keys import LoggingSection
    from spindoctor.config.sections import (
        EnvironmentSection,
        GeneralSection,
        NavConfig,
        OffsetSection,
        OrchestratorSection,
    )
    from spindoctor.config.sections_body_shape import BodyShapeEntry
    from spindoctor.config.sections_models import (
        BodiesSection,
        FeatureEmissionSection,
        StarsSection,
        TitanSection,
    )
    from spindoctor.config.sections_products import BackplanesSection, Pds4Section
    from spindoctor.config.sections_rings import RingsSection
    from spindoctor.nav_technique.confidence_config import TechniquesSection

_PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config_files'
"""The directory of the packaged configuration files, merged in filename order."""

_PACKAGED_LABEL = 'the packaged configuration'
"""The source a message gives for a packaged value when no file is recorded for it."""


def _nav_config_type(
    config_dict: Mapping[str, Any], known_cameras: frozenset[str]
) -> type[NavConfig]:
    """Return the type a merged configuration is parsed into.

    The section types import modules that import this package, so the import waits
    until a configuration is loaded, by which time this package has loaded.

    Parameters:
        config_dict: The merged configuration.
        known_cameras: The camera sections of the packaged configuration.

    Returns:
        The type :func:`~spindoctor.config.sections.nav_config_type` returns.
    """
    from spindoctor.config.sections import nav_config_type

    return nav_config_type(config_dict, known_cameras=known_cameras)


def _configure_techniques(nav_config: NavConfig) -> None:
    """Assign each shipped technique class its confidence formula and tuning.

    Parameters:
        nav_config: The configuration being installed.
    """
    from spindoctor.nav_technique.confidence_config import shipped_techniques

    for technique in shipped_techniques():
        block = nav_config.techniques.block(technique)
        technique.confidence_spec = block.confidence_spec()
        technique.tuning = block.technique_tuning()


def _refuse_replaced_sections(packaged: Mapping[Any, Any], layer: Layer) -> None:
    """Refuse an override that gives a section as anything but a mapping.

    A section is a top-level key whose value in the packaged configuration is a
    mapping.  The merge would let a scalar or a list replace it whole, which would
    leave a camera section to be refused as an unknown key rather than as a value of
    the wrong type.  A null is left to the merge, which refuses every null.

    Parameters:
        packaged: The packaged configuration, merged.
        layer: An override to be merged over it.

    Raises:
        SchemaError: Naming the override and each section it gives as something else.
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
        and isinstance(packaged.get(key), dict)
        and not isinstance(value, Mapping)
    ]
    if issues:
        raise SchemaError(issues)


def load_nav_config(
    paths: Iterable[str | Path] = (), *, overrides: Iterable[Layer] = ()
) -> NavConfig:
    """Read, merge, and parse the navigator configuration.

    The packaged files merge first, in filename order, then each file of ``paths``, then
    each layer of ``overrides``, each over the ones before it.  Every key beginning with
    an underscore is dropped as its file is read, so a file can carry notes, such as
    ``_sources`` citations, beside its values.  The whole merged configuration is then
    parsed.  Each call reads the files again and builds a new object, and nothing else
    changes: the technique classes take their settings only when a :class:`Config`
    installs the result.

    Parameters:
        paths: User override files, in the order they merge.  A message gives each one
            as written here.
        overrides: Overrides held in memory, merged after the files; a message gives
            each one by its label.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If a file of ``paths`` does not exist.
        SchemaError: If a file is not a YAML mapping, a layer holds a null or gives a
            section as anything but a mapping, or the merged configuration breaks its
            schema, as with an unknown key or section, a value of the wrong type or out
            of range, or a technique block that names a diagnostic attribute its
            technique does not declare.  Each line of the message gives the file that set
            the value, and the key path.
    """
    packaged = [read_layer(path) for path in sorted(_PACKAGED_CONFIG_DIR.glob('*.yaml'))]
    layers = [*(read_layer(Path(path), label=str(path)) for path in paths), *overrides]
    base = merge(packaged, nulls=NullPolicy.REFUSE)
    for layer in layers:
        _refuse_replaced_sections(base.data, layer)
    merged = merge([*packaged, *layers], nulls=NullPolicy.REFUSE)
    known_cameras = frozenset(_nav_config_type(base.data, frozenset()).CAMERA_SECTIONS)
    nav_config_type = _nav_config_type(merged.data, known_cameras)
    source = layers[-1].label if layers else _PACKAGED_LABEL
    return parse(nav_config_type, merged.data, source=source, origins=merged.origins)


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
        value: A mapping, list, tuple, or scalar.

    Returns:
        ``value`` with every mapping key converted to its ``str`` form; lists and tuples
        are walked element-wise and scalars pass through unchanged.
    """

    if isinstance(value, dict):
        return {str(k): _stringify_keys(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_stringify_keys(v) for v in value]
    return value


class Config:
    """The navigator configuration a program reads, and the overrides it was built from.

    It holds the current :class:`~spindoctor.config.sections.NavConfig`, which
    :func:`load_nav_config` builds, and reads the packaged files the first time a section
    is asked for.  Installing a configuration also assigns each shipped technique class
    its confidence formula and tuning.

    Parameters:
        overrides: Overrides held in memory, merged over the packaged files each time
            the configuration is read, and before any file :meth:`update_config` merges.
    """

    def __init__(self, *, overrides: Iterable[Layer] = ()) -> None:
        """Make a configuration that reads nothing until a section is asked for."""

        self._base_overrides: tuple[Layer, ...] = tuple(overrides)
        self._overrides: tuple[Layer, ...] = self._base_overrides
        self._override_paths: tuple[str, ...] = ()
        self._nav_config: NavConfig | None = None

    @property
    def is_loaded(self) -> bool:
        """Whether a configuration is installed (after ``read_config`` / ``update_config``)."""

        return self._nav_config is not None

    def ensure_loaded(self) -> None:
        """Load bundled default YAML if not already loaded.

        Safe to call repeatedly; delegates to :meth:`read_config` with no path
        (same early-return behavior when data is already present).
        """

        self.read_config()

    def _install(self, overrides: tuple[Layer, ...], override_paths: tuple[str, ...]) -> NavConfig:
        """Load a configuration from a set of overrides, and keep it only if it passes.

        A refused configuration leaves the configuration, the techniques' settings, and
        :attr:`override_paths` as they were.

        Parameters:
            overrides: The overrides to merge over the packaged files, in order.
            override_paths: The override files among them, in order.

        Returns:
            The configuration installed.

        Raises:
            SchemaError: As :func:`load_nav_config` raises it.
        """
        nav_config = load_nav_config(overrides=overrides)
        self._nav_config = nav_config
        self._overrides = overrides
        self._override_paths = override_paths
        _configure_techniques(nav_config)
        return nav_config

    def read_config(self, *, reread: bool = False) -> None:
        """Read the packaged configuration files, unless they have been read already.

        A reread starts again from the packaged files and the overrides given when the
        configuration was made, so it drops every file merged since.

        Parameters:
            reread: Whether to read the files again if the configuration is loaded.

        Raises:
            SchemaError: As :func:`load_nav_config` raises it.
        """

        if reread or self._nav_config is None:
            self._install(self._base_overrides, ())

    def update_config(self, config_path: str | Path) -> None:
        """Merge a user override file over the configuration.

        The file is read once and kept, and the configuration is loaded again from the
        packaged files and every override merged so far, then this one.  Each shipped
        technique is assigned the confidence formula and tuning the result sets, so
        per-technique overrides take effect.  A refused override, whatever the reason,
        leaves the configuration, the techniques' settings, and :attr:`override_paths` as
        they were.

        Parameters:
            config_path: The override file.  Error messages and :attr:`override_paths`
                give it as written here.

        Raises:
            FileNotFoundError: If the file does not exist.
            SchemaError: As :func:`load_nav_config` raises it.
        """

        self.read_config()
        layer = read_layer(Path(config_path), label=str(config_path))
        self._install((*self._overrides, layer), (*self._override_paths, str(config_path)))

    @property
    def override_paths(self) -> tuple[str, ...]:
        """The user override files applied by :meth:`update_config`, in order.

        The packaged files are not overrides and never appear here, and neither do the
        overrides held in memory.  Each path is given as it was passed to
        :meth:`update_config`.
        """

        return self._override_paths

    @property
    def nav_config(self) -> NavConfig:
        """Returns the current configuration, reading the packaged files first if need be."""

        nav_config = self._nav_config
        if nav_config is None:
            nav_config = self._install(self._base_overrides, ())
        return nav_config

    def resolved_config_hash(self) -> str:
        """Return a stable sha256 hex digest of the fully-resolved config.

        The parsed configuration (bundled defaults plus every applied override) is
        serialized deterministically -- mapping keys are stringified and sorted, list
        order is preserved -- and hashed, so two configs with identical resolved
        content produce identical digests regardless of load order or
        comment/whitespace changes in the source YAML files.

        Sections in :data:`HASH_EXCLUDED_SECTIONS` are left out, because this
        digest answers "was this result produced by the same configuration",
        and a setting that cannot change a result would answer it wrongly.
        Raising one component's log level to look into a technique would
        otherwise re-stamp every result in that run as differently
        configured, and it compares against an archive.

        Returns:
            64-character sha256 hex digest of the resolved config content.
        """

        hashed = {
            key: value
            for key, value in dataclasses.asdict(self.nav_config).items()
            if key not in HASH_EXCLUDED_SECTIONS
        }
        canonical = json.dumps(
            _stringify_keys(hashed),
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    @property
    def general(self) -> GeneralSection:
        """Returns the general section: settings no single part of the pipeline owns."""

        return self.nav_config.general

    @property
    def environment(self) -> EnvironmentSection:
        """Returns the environment section, which says where a deployment keeps its files.

        The packaged file leaves every location unset, so a location is None unless an
        override sets it.  The helpers in :mod:`spindoctor.config.config_helper` define
        the usual order for a location: the command line, then here, then an environment
        variable.  A reader may apply its own order, but each takes its location from one
        of those three places.
        """

        return self.nav_config.environment

    @property
    def logging(self) -> LoggingSection:
        """Returns the logging section: the levels and terminal sinks of the two loggers.

        Holds the per-logger defaults (``main``, ``image``), the per-module
        override categories (``techniques``, ``models``, ``other``), the
        ``strict_scope`` switch, and per-program overrides under ``programs``.
        The section is parsed when the configuration loads, so every key and
        level has been checked.
        """

        return self.nav_config.logging

    @property
    def planets(self) -> list[str]:
        """Returns the SPICE names of the configured planets."""

        return list(self.nav_config.planets)

    def satellites(self, planet: str) -> list[str]:
        """Returns the list of satellites for the specified planet.

        Parameters:
            planet: The name of the planet to get satellites for, in any case.

        Returns:
            A list of satellite names for the specified planet; empty for a planet the
            ``satellites`` section leaves out.
        """

        return list(self.nav_config.satellites.get(planet.upper(), ()))

    def camera_block(self, section: str, camera: str | None = None) -> CameraBlock:
        """Returns one camera's block.

        Parameters:
            section: The section that holds the camera's block, as the files name it.
            camera: The camera's name inside a section that holds several cameras; None
                for a section that is one camera's block.

        Returns:
            The camera's block.

        Raises:
            KeyError: If ``section`` is not a camera section of the configuration, does
                not hold a camera named ``camera``, or is one camera's block and
                ``camera`` is given.
        """

        return self.nav_config.camera_block(section, camera)

    def camera_names(self, section: str) -> tuple[str | None, ...]:
        """Returns the names of the cameras a camera section holds.

        Parameters:
            section: The camera section, as the files name it.

        Returns:
            Each camera's name, in the files' order, for a section that holds several
            cameras; ``(None,)`` for a section that is one camera's block.  Each name is
            what :meth:`camera_block` takes as ``camera``.

        Raises:
            KeyError: If ``section`` is not a camera section of the configuration.
        """

        nav_config = self.nav_config
        if section not in nav_config.CAMERA_SECTIONS:
            raise KeyError(f'{section!r} is not a camera section of the configuration')
        value: CameraBlock | dict[str, CameraBlock] = getattr(nav_config, section)
        if isinstance(value, CameraBlock):
            return (None,)
        return tuple(value)

    @property
    def offset(self) -> OffsetSection:
        """Returns the offset section, which sets how finely a correlation resolves a peak."""

        return self.nav_config.offset

    @property
    def bodies(self) -> BodiesSection:
        """Returns the ``bodies`` section: how a body is rendered, gated, and labeled."""

        return self.nav_config.bodies

    @property
    def body_shape(self) -> Mapping[str, BodyShapeEntry]:
        """Returns the body-shape table, keyed by each body's SPICE name in upper case.

        A key that an entry leaves out reads as None, and
        :func:`~spindoctor.nav_model.body_shape.load_body_shape` then takes its value from
        the next tier.
        """

        return self.nav_config.body_shape

    @property
    def rings(self) -> RingsSection:
        """Returns the ``rings`` section: the ring model's settings and the ring catalogs."""

        return self.nav_config.rings

    @property
    def stars(self) -> StarsSection:
        """Returns the ``stars`` section: the catalogs, their reduction, and conflicts."""

        return self.nav_config.stars

    @property
    def titan(self) -> TitanSection:
        """Returns the ``titan`` section: the haze envelope, its feature, and its fit."""

        return self.nav_config.titan

    @property
    def feature_emission(self) -> FeatureEmissionSection:
        """Returns the ``feature_emission`` section: which features the ring model emits."""

        return self.nav_config.feature_emission

    @property
    def orchestrator(self) -> OrchestratorSection:
        """Returns the orchestrator section: the ensemble and the reliability gate.

        The section is parsed when the configuration loads, so every value has been
        checked against its type and range.
        """

        return self.nav_config.orchestrator

    @property
    def techniques(self) -> TechniquesSection:
        """Returns the techniques section: each technique's confidence formula and tuning.

        The section is parsed when the configuration loads.  Its
        :meth:`~spindoctor.nav_technique.confidence_config.TechniquesSection.block`
        method gives one technique's block.
        """

        return self.nav_config.techniques

    @property
    def results_tree(self) -> spindoctor.nav_records.TreeTuning:
        """Returns how much of a pass over a results tree runs at once.

        These settings have no command-line option or environment variable, because they
        describe a machine rather than a run.  A program reads the section once and
        passes it to whatever reads the tree.
        """

        return self.nav_config.results_tree

    @property
    def backplanes(self) -> BackplanesSection:
        """Returns the backplanes section: the planes each image gets, and their targets.

        The section is parsed when the configuration loads, so every value has been
        checked against its type.
        """

        return self.nav_config.backplanes

    @property
    def pds4(self) -> Pds4Section:
        """Returns the pds4 section: each PDS4 bundle's settings, keyed by its bundle key.

        Its :meth:`~spindoctor.config.sections_products.Pds4Section.bundle` method gives
        one bundle's entry.
        """

        return self.nav_config.pds4


DEFAULT_CONFIG = Config()
