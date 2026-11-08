"""The simulated instruments, loaded from the per-mission YAML files.

The config files live in ``spindoctor/sim/instrument_configs/``, one per mission
(``coiss.yaml`` holds the four Cassini ISS instruments, and ``generic.yaml`` holds
the instrument-agnostic entry).  Each top-level key of a file is one simulated
instrument, and the key is the name a scene writes; the file's own name plays no
part.  An entry holds everything the simulator knows about the camera: the
navigator config block it emulates, its detector chain, readout modes, star zero
point, plate scale, PSF, residual distortion, filter throughputs, and which
artifact modes it offers with their shape defaults.  The provenance of each value
is a YAML comment beside it.  Adding a camera is adding an entry (plus the
navigator's own instrument block for the camera it emulates); no code refers to an
instrument by name.

An entry may set another instrument as its ``base``.  The base's entry is then
merged under the entry's own keys: mappings merge key by key, and any other value
replaces the base's.  A calibrated product can therefore state only what differs
from its raw camera.  Setting a key to null removes the key the base would
otherwise pass on.  A base does not pass on ``aliases`` or ``realism_cohort``.

The catalog loads once per directory and is checked against
:mod:`spindoctor.sim.instrument_schema` and :mod:`spindoctor.sim.instrument_checks`
as it loads.  The load fails on a file that is not valid YAML or not a ``.yaml``
file, an instrument defined in two files, an unknown key, a missing required key,
or a value of the wrong type or out of range.  It also fails on a base that
doesn't exist or that loops back on itself, and on a gain state, readout mode,
filter, downlink table, or artifact mode that an entry refers to but does not
define.  A setting that would do nothing fails too.  Each message gives the file,
the entry, and the key path.

The rest of the simulator reads the catalog through the resolvers in
:mod:`spindoctor.sim.instrument_resolvers`, and through
:mod:`spindoctor.sim.instrument_navigator` for the navigator config block an entry
emulates.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from spindoctor.sim.instrument_checks import check_catalog
from spindoctor.sim.instrument_schema import (
    GENERIC_INSTRUMENT,
    NON_INHERITED_KEYS,
    InstrumentConfigError,
    check_own_entry,
    config_error,
)

__all__ = [
    'GENERIC_INSTRUMENT',
    'INSTRUMENT_CONFIG_DIR',
    'InstrumentCatalog',
    'InstrumentConfigError',
    'instrument_entry',
    'load_instrument_catalog',
    'realism_cohort_instruments',
    'sim_instrument_names',
]

# The packaged directory of instrument config files.
INSTRUMENT_CONFIG_DIR = Path(__file__).resolve().parent / 'instrument_configs'


@dataclass(frozen=True)
class InstrumentCatalog:
    """Every simulated instrument's entry, after inheritance and checks.

    Parameters:
        config_dir: The directory the entries were read from.
        entries: Each instrument's entry keyed by name, ``generic`` included, with
            its base merged in and ``base`` and ``aliases`` removed.
        aliases: Each alias mapped to the instrument it stands for.
    """

    config_dir: Path
    entries: Mapping[str, dict[str, Any]]
    aliases: Mapping[str, str]

    def canonical_name(self, instrument: str | None) -> str:
        """The catalog name a scene's instrument name selects.

        Parameters:
            instrument: An instrument name, an alias, or ``None``.

        Returns:
            The instrument's own name; ``generic`` for ``None``.

        Raises:
            ValueError: If the name is neither an instrument nor an alias.
        """
        if instrument is None:
            return GENERIC_INSTRUMENT
        if instrument in self.entries:
            return instrument
        if instrument in self.aliases:
            return self.aliases[instrument]
        raise ValueError(
            f'unknown sim instrument {instrument!r}; expected one of {sorted(self.allowed_names())}'
        )

    def entry(self, instrument: str | None) -> dict[str, Any]:
        """The shared entry of one instrument; callers must not modify it.

        Parameters:
            instrument: An instrument name, an alias, or ``None`` for the generic
                entry.

        Returns:
            The entry, not a copy.

        Raises:
            ValueError: If the name is neither an instrument nor an alias.
        """
        return self.entries[self.canonical_name(instrument)]

    def instrument_names(self) -> tuple[str, ...]:
        """The camera names, sorted, without the generic entry or any alias."""
        return tuple(sorted(name for name in self.entries if name != GENERIC_INSTRUMENT))

    def allowed_names(self) -> frozenset[str]:
        """Every name a scene may write: the instruments, ``generic``, and the aliases."""
        return frozenset(self.entries) | frozenset(self.aliases)


def _read_file(path: Path) -> dict[str, Any]:
    """Read one config file, returning its entries keyed by instrument.

    A YAML syntax error and a key written twice fail with a message that gives the
    file, as every schema violation does.  A file that holds anything but a mapping
    fails too.
    """
    yaml = YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except YAMLError as exc:
        raise InstrumentConfigError(f'{path.name}: not valid YAML: {exc}') from exc
    if not isinstance(raw, dict):
        raise config_error(path.name, '', 'the file must map instrument names to entries')
    return raw


def _read_entries(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Merge the top-level keys of every config file, checking each entry on its own.

    Parameters:
        paths: The config files, in the order to read them.

    Returns:
        Each instrument's own entry keyed by name, and each name mapped to the
        ``file: entry`` prefix its error messages carry.

    Raises:
        InstrumentConfigError: If a file is not valid YAML or holds no mapping, an
            instrument is defined in two files, or an entry breaks the schema.
    """
    raw: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    files: dict[str, str] = {}
    for path in paths:
        for name, entry in _read_file(path).items():
            source = f'{path.name}: {name}'
            if name in files:
                raise InstrumentConfigError(
                    f'{source}: the instrument is already defined in {files[name]}'
                )
            files[name] = path.name
            sources[name] = source
            raw[name] = check_own_entry(entry, source=source)
    return raw, sources


def _deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any], *, source: str, path: str = ''
) -> dict[str, Any]:
    """``base`` with ``overrides`` merged in, key by key.

    Mappings merge key by key, a null removes the base's key, and any other value
    replaces the base's.

    Parameters:
        base: The base entry (or one of its mappings).
        overrides: The entry's own keys at the same place.
        source: The file and the entry, for error messages.
        path: The dotted key path of ``overrides``, for error messages.

    Returns:
        The merged mapping, a fresh copy.

    Raises:
        InstrumentConfigError: If a null is set on a key the base does not have, so
            it would remove nothing.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        key_path = f'{path}.{key}' if path else str(key)
        existing = merged.get(key)
        if value is None:
            if key not in merged:
                raise InstrumentConfigError(
                    f'{source}: {key_path}: null removes an inherited key, and the base has '
                    'no such key'
                )
            del merged[key]
        elif isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge(existing, value, source=source, path=key_path)
        else:
            # The base has nothing under this key to merge with, so a null anywhere
            # inside the entry's mapping would remove nothing.
            _refuse_nulls(value, source=source, path=key_path)
            merged[key] = copy.deepcopy(value)
    return merged


def _refuse_nulls(value: Any, *, source: str, path: str) -> None:
    """Refuse a null inside a mapping the base has no counterpart for.

    Parameters:
        value: An entry's value under a key the base lacks (or holds as a non-mapping).
        source: The file and the entry, for error messages.
        path: The dotted key path of ``value``, for error messages.

    Raises:
        InstrumentConfigError: If a mapping inside ``value`` sets a key to null; the
            message gives that key's path.
    """
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        key_path = f'{path}.{key}'
        if item is None:
            raise InstrumentConfigError(
                f'{source}: {key_path}: null removes an inherited key, and the base has no such key'
            )
        _refuse_nulls(item, source=source, path=key_path)


def _resolve_inheritance(
    raw: Mapping[str, dict[str, Any]], sources: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Merge each entry over its base chain, refusing an unknown base or a cycle.

    Parameters:
        raw: Each instrument's own entry, keyed by name.
        sources: Each name's ``file: entry`` prefix, for error messages.

    Returns:
        Each entry after inheritance, keyed by name.

    Raises:
        InstrumentConfigError: If a base is not in the catalog or the bases loop.
    """
    resolved: dict[str, dict[str, Any]] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict[str, Any]:
        """Merge ``name`` over its base, ``chain`` being the names already visited."""
        if name in resolved:
            return resolved[name]
        own = raw[name]
        base = own.get('base')
        entry = {key: value for key, value in own.items() if key not in NON_INHERITED_KEYS}
        if base is not None:
            if base not in raw:
                raise config_error(
                    sources[name], 'base', f'{base!r} is not an instrument in the catalog'
                )
            if base in chain:
                loop = ' -> '.join((*chain, base))
                raise config_error(sources[name], 'base', f'inheritance loops ({loop})')
            # A resolved entry already lacks the keys a base does not pass on.
            entry = _deep_merge(resolve(base, (*chain, base)), entry, source=sources[name])
        resolved[name] = entry
        return entry

    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(raw):
        entry = copy.deepcopy(resolve(name, (name,)))
        # The entry's own realism cohort, which a base does not pass on.
        if 'realism_cohort' in raw[name]:
            entry['realism_cohort'] = copy.deepcopy(raw[name]['realism_cohort'])
        entries[name] = entry
    return entries


def _collect_aliases(
    raw: Mapping[str, dict[str, Any]], sources: Mapping[str, str]
) -> dict[str, str]:
    """Map each alias to its instrument, refusing an alias already in use as a name."""
    aliases: dict[str, str] = {}
    for name in sorted(raw):
        for alias in raw[name].get('aliases') or []:
            if alias in raw or alias in aliases:
                raise config_error(
                    sources[name], 'aliases', f'{alias!r} is already an instrument or an alias'
                )
            aliases[alias] = name
    return aliases


def _config_paths(config_dir: Path) -> list[Path]:
    """The config files in ``config_dir``, refusing anything that is not one.

    Every entry must be a ``.yaml`` file: a ``.yml`` file, or any other file, would
    otherwise be skipped without a word.
    """
    paths = sorted(config_dir.iterdir())
    strays = [path.name for path in paths if path.suffix != '.yaml' or not path.is_file()]
    if strays:
        raise InstrumentConfigError(
            f'{config_dir}: {strays} are not instrument config files, which are .yaml files'
        )
    return paths


@lru_cache(maxsize=4)
def _load(config_dir: Path) -> InstrumentCatalog:
    """Read, merge, and check every config file in ``config_dir``."""
    raw, sources = _read_entries(_config_paths(config_dir))
    if GENERIC_INSTRUMENT not in raw:
        raise InstrumentConfigError(
            f'{config_dir}: no {GENERIC_INSTRUMENT!r} entry (the instrument-agnostic entry)'
        )
    entries = _resolve_inheritance(raw, sources)
    check_catalog(entries, sources)
    return InstrumentCatalog(
        config_dir=config_dir, entries=entries, aliases=_collect_aliases(raw, sources)
    )


def load_instrument_catalog(*, config_dir: Path | None = None) -> InstrumentCatalog:
    """Load and check the instrument config files, once per directory.

    Parameters:
        config_dir: The directory to read; the packaged
            :data:`INSTRUMENT_CONFIG_DIR` when omitted.

    Returns:
        The checked catalog.  The result is cached, so every caller shares one
        catalog; its entries must not be modified.

    Raises:
        InstrumentConfigError: If any file breaks the schema or defines an
            instrument another file defines, or if an entry sets a base that doesn't
            exist or loops, reuses an alias, or claims another instrument's realism
            cohort.  Also if no file defines the ``generic`` entry, or that entry
            states no ``detector.calibrated_sky_if``.
    """
    return _load((config_dir or INSTRUMENT_CONFIG_DIR).resolve())


def instrument_entry(instrument: str | None) -> dict[str, Any]:
    """The packaged catalog's shared entry of one instrument; do not modify it.

    Parameters:
        instrument: An instrument name, an alias, or ``None`` for the generic entry.

    Returns:
        The entry, not a copy.

    Raises:
        ValueError: If the name is neither an instrument nor an alias.
    """
    return load_instrument_catalog().entry(instrument)


def sim_instrument_names() -> tuple[str, ...]:
    """The packaged catalog's camera names, sorted, without ``generic`` or any alias."""
    return load_instrument_catalog().instrument_names()


def realism_cohort_instruments() -> dict[tuple[str, str], str]:
    """Map each real (mission, camera) pair to the instrument that emulates its products.

    Returns:
        One entry per ``realism_cohort`` pair in the packaged catalog.
    """
    catalog = load_instrument_catalog()
    return {
        (pair['mission'], pair['camera']): name
        for name in catalog.instrument_names()
        for pair in catalog.entry(name).get('realism_cohort') or []
    }
