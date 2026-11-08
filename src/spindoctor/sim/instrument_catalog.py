"""The simulated instruments, loaded from one YAML file per camera.

Every simulated camera is a file in ``spindoctor/sim/instrument_configs/`` whose
stem is the instrument name a scene writes (``generic.yaml`` is the
instrument-agnostic entry).  A file holds everything the simulator knows about the
camera: the navigator config block it emulates, its detector chain, readout
modes, star zero point, plate scale, PSF, residual distortion, filter
throughputs, and which artifact modes it offers with their shape defaults.  The
provenance of each value is a YAML comment beside it.  Adding a camera is adding
a file (plus the navigator's own instrument block for the camera it emulates);
no code names an instrument.

A file may name another instrument as its ``base``: the base's entry is merged
under the file's own keys (mappings merge key by key, any other value replaces
the base's), so a calibrated product can state only what differs from its raw
camera.  ``aliases`` is the one key a base does not pass on.

The catalog loads once per directory and is checked against
:mod:`spindoctor.sim.instrument_schema` as it loads: an unknown key, a missing
required key, a value of the wrong type, a base that doesn't exist or that loops
back on itself, or a gain state, readout mode, filter, or artifact mode an entry
names but does not define fails with the file and the key path.  The resolver
functions in :mod:`spindoctor.sim.forward.artifacts_catalog`,
:mod:`spindoctor.sim.forward.filter_catalog`, :mod:`spindoctor.sim.forward.psf`,
and :mod:`spindoctor.sim.instruments` are how the rest of the simulator reads it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.sim.instrument_schema import (
    GENERIC_INSTRUMENT,
    NON_INHERITED_KEYS,
    InstrumentConfigError,
    check_entry,
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
        aliases: Each alternative name mapped to the instrument it names.
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
    """Read one config file, which must hold a mapping."""
    yaml = YAML(typ='safe')
    raw = yaml.load(path.read_text())
    if not isinstance(raw, dict):
        raise InstrumentConfigError(f'{path.name}: the file must hold a mapping')
    return raw


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overrides`` merged in: mappings key by key, anything else replaced."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_inheritance(raw: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge each entry over its base chain, refusing an unknown base or a cycle."""
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
                raise InstrumentConfigError(
                    f'{name}.yaml: base: {base!r} is not an instrument in the catalog'
                )
            if base in chain:
                loop = ' -> '.join((*chain, base))
                raise InstrumentConfigError(f'{name}.yaml: base: inheritance loops ({loop})')
            # A resolved entry already lacks the keys a base does not pass on.
            entry = _deep_merge(resolve(base, (*chain, base)), entry)
        resolved[name] = entry
        return entry

    for name in sorted(raw):
        resolve(name, (name,))
    return resolved


def _collect_aliases(raw: Mapping[str, dict[str, Any]]) -> dict[str, str]:
    """Map each alias to its instrument, refusing an alias that is already a name."""
    aliases: dict[str, str] = {}
    for name in sorted(raw):
        for alias in raw[name].get('aliases') or []:
            if alias in raw or alias in aliases:
                raise InstrumentConfigError(
                    f'{name}.yaml: aliases: {alias!r} already names an instrument or an alias'
                )
            aliases[alias] = name
    return aliases


def _check_cohorts(entries: Mapping[str, dict[str, Any]]) -> None:
    """Refuse a real (mission, camera) pair claimed by two instruments."""
    owner: dict[tuple[str, str], str] = {}
    for name in sorted(entries):
        for index, pair in enumerate(entries[name].get('realism_cohort') or []):
            key = (pair['mission'], pair['camera'])
            if key in owner:
                raise InstrumentConfigError(
                    f'{name}.yaml: realism_cohort[{index}]: {key} is already the cohort of '
                    f'{owner[key]!r}'
                )
            owner[key] = name


@lru_cache(maxsize=4)
def _load(config_dir: Path) -> InstrumentCatalog:
    """Read, merge, and check every config file in ``config_dir``."""
    paths = sorted(config_dir.glob('*.yaml'))
    raw: dict[str, dict[str, Any]] = {}
    for path in paths:
        entry = _read_file(path)
        check_entry(entry, source=path.name, complete=False)
        raw[path.stem] = entry
    if GENERIC_INSTRUMENT not in raw:
        raise InstrumentConfigError(
            f'{config_dir}: no {GENERIC_INSTRUMENT}.yaml (the instrument-agnostic entry)'
        )
    entries = _resolve_inheritance(raw)
    for name, entry in entries.items():
        check_entry(entry, source=f'{name}.yaml', complete=True)
    _check_cohorts(entries)
    return InstrumentCatalog(config_dir=config_dir, entries=entries, aliases=_collect_aliases(raw))


def load_instrument_catalog(*, config_dir: Path | None = None) -> InstrumentCatalog:
    """Load and check the instrument config files, once per directory.

    Parameters:
        config_dir: The directory to read; the packaged
            :data:`INSTRUMENT_CONFIG_DIR` when omitted.

    Returns:
        The checked catalog.  The result is cached, so every caller shares one
        catalog; its entries must not be modified.

    Raises:
        InstrumentConfigError: If any file breaks the schema, names a base that
            doesn't exist or loops, reuses an alias, or claims another instrument's
            realism cohort, or if the directory has no ``generic.yaml``.
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
