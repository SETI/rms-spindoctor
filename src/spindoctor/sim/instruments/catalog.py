"""The simulated instruments, loaded from the per-mission YAML files.

The config files live in ``spindoctor/sim/instruments/configs/``, one per mission
(``coiss.yaml`` holds the four Cassini ISS instruments, and ``generic.yaml`` holds
the instrument-agnostic entry).  Each top-level key of a file is one simulated
instrument, and the key is the name a scene writes; the file's own name plays no
part.  An entry holds everything the simulator knows about the camera: the units
its frames are written in, the navigator block it emulates, its detector
chain, readout modes, star zero point, plate scale, PSF, residual distortion,
filter throughputs, and which artifact modes it offers with their shape defaults.
The provenance of each value is a YAML comment beside it.  Adding a camera is adding
an entry (plus the navigator's own instrument block for the camera it emulates); no
code refers to an instrument by name.

An entry may set another instrument as its ``base``.  The base's entry is then
merged under the entry's own keys: mappings merge key by key, and any other value
replaces the base's.  A calibrated product can therefore state only what differs
from its raw camera, such as its ``data_units``.  Setting a key to null removes the
key the base would otherwise pass on.  A base does not pass on ``aliases`` or
``realism_cohort``.  The merge is :func:`spindoctor.schema.resolve_bases` under the
``REMOVE`` rule for null.

The catalog loads once per directory.  Each merged entry is parsed into an
:class:`~spindoctor.sim.instruments.schema.InstrumentEntry`, which checks it, and the
checks that span entries run last.  Every problem found in the directory is reported
at once in one :class:`~spindoctor.schema.SchemaError`, one line each, and a problem
inherited by several entries is reported once.  The load fails on a file that is not
valid YAML or not a ``.yaml`` file, an empty file, an instrument defined in two files,
an unknown key, a missing required key, or a value of the wrong type or out of range.  It also
fails on a base that doesn't exist or that loops back on itself, and on a gain state,
readout mode, filter, downlink table, or artifact mode that an entry refers to but
does not define.  A setting that would do nothing fails too.  Every line about an
entry gives the file and the entry that set the value, and the key path.  A line
about a whole file (a YAML error, or a file that holds no entry) gives the file,
and a line about the directory (a stray file, or no ``generic`` entry) gives the
directory.

The rest of the simulator reads the catalog through the resolvers in
:mod:`spindoctor.sim.instruments.resolvers`, and through
:mod:`spindoctor.sim.instruments.navigator_link` for the navigator block an entry
emulates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from spindoctor.schema import (
    Issue,
    Layer,
    NullPolicy,
    SchemaError,
    parse,
    read_layer,
    resolve_bases,
    split_entries,
)
from spindoctor.sim.instruments.schema import (
    GENERIC_INSTRUMENT,
    NON_INHERITED_KEYS,
    InstrumentEntry,
)

__all__ = [
    'GENERIC_INSTRUMENT',
    'INSTRUMENT_CONFIG_DIR',
    'InstrumentCatalog',
    'instrument_entry',
    'load_instrument_catalog',
    'realism_cohort_instruments',
    'sim_instrument_names',
]

# The packaged directory of instrument config files.
INSTRUMENT_CONFIG_DIR = Path(__file__).resolve().parent / 'configs'


@dataclass(frozen=True)
class InstrumentCatalog:
    """Every simulated instrument's entry, after inheritance and checks.

    Parameters:
        config_dir: The directory the entries were read from.
        entries: Each instrument's entry keyed by name, ``generic`` included.
        aliases: Each alias mapped to the instrument it stands for.
        sources: For each instrument, the file and entry name its error messages
            begin with (``'coiss.yaml: coiss_nac'``).
    """

    config_dir: Path
    entries: Mapping[str, InstrumentEntry]
    aliases: Mapping[str, str]
    sources: Mapping[str, str]

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

    def entry(self, instrument: str | None) -> InstrumentEntry:
        """The entry of one instrument.

        Parameters:
            instrument: An instrument name, an alias, or ``None`` for the generic
                entry.

        Returns:
            The entry.

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


def _config_paths(config_dir: Path) -> list[Path]:
    """The config files in ``config_dir``, refusing anything that is not one.

    Every entry must be a ``.yaml`` file: a ``.yml`` file, or any other file, would
    otherwise be skipped without a word.
    """
    paths = sorted(config_dir.iterdir())
    strays = [path.name for path in paths if path.suffix != '.yaml' or not path.is_file()]
    if strays:
        message = f'{strays} are not instrument config files, which are .yaml files'
        raise SchemaError([Issue(str(config_dir), '', message)])
    return paths


def _read_entries(config_dir: Path) -> tuple[dict[str, Layer], dict[str, str]]:
    """Read every config file in ``config_dir`` as one layer per instrument.

    Parameters:
        config_dir: The directory.

    Returns:
        Each instrument's own layer keyed by name, and each name mapped to its file.

    Raises:
        SchemaError: Listing each stray file; or each file that is not valid YAML,
            holds no mapping of instrument names to entries, or is empty, and each
            instrument defined in two files.
    """
    entries: dict[str, Layer] = {}
    files: dict[str, str] = {}
    issues: list[Issue] = []
    for path in _config_paths(config_dir):
        try:
            layers = split_entries(read_layer(path))
        except SchemaError as exc:
            issues.extend(exc.issues)
            continue
        if not layers:
            issues.append(Issue(path.name, '', 'the file holds no instrument entry'))
        for name, layer in layers.items():
            if name in files:
                message = f'the instrument is already defined in {files[name]}'
                issues.append(Issue(layer.label, '', message))
                continue
            files[name] = path.name
            entries[name] = layer
    if issues:
        raise SchemaError(issues)
    return entries, files


def _parse_entries(entries: Mapping[str, Layer]) -> dict[str, InstrumentEntry]:
    """Merge each entry over its bases and parse it, collecting every problem.

    A problem in an entry that others inherit surfaces once for each of them, with
    the same file, key path, and message, so each is reported once.

    Parameters:
        entries: Each instrument's own layer, keyed by name.

    Returns:
        Each instrument's entry, keyed by name, sorted by name.

    Raises:
        SchemaError: Listing every problem in every entry.
    """
    merged = resolve_bases(entries, nulls=NullPolicy.REMOVE, not_inherited=NON_INHERITED_KEYS)
    parsed: dict[str, InstrumentEntry] = {}
    issues: list[Issue] = []
    for name in sorted(merged):
        label = entries[name].label
        data, origins = merged[name].data, merged[name].origins
        try:
            parsed[name] = parse(InstrumentEntry, data, source=label, origins=origins)
        except SchemaError as exc:
            issues.extend(exc.issues)
    if issues:
        raise SchemaError(list(dict.fromkeys(issues)))
    return parsed


def _cross_entry_issues(
    entries: Mapping[str, InstrumentEntry], sources: Mapping[str, str], files: Mapping[str, str]
) -> list[Issue]:
    """The problems that span entries.

    The generic entry must state the calibrated sky residual that every entry without
    its own falls back on.  No real (mission, spacecraft, camera) triple may be claimed
    by two entries' realism cohorts.  An alias may be neither an instrument's name nor
    another entry's alias; the message gives the file that holds the clashing name or
    alias.
    """
    issues: list[Issue] = []
    if entries[GENERIC_INSTRUMENT].detector.calibrated_sky_if is None:
        issues.append(
            Issue(
                sources[GENERIC_INSTRUMENT],
                'detector.calibrated_sky_if',
                'the generic entry states the calibrated sky residual that every entry '
                'without its own falls back on',
            )
        )
    owner: dict[tuple[str, str | None, str], tuple[str, int]] = {}
    aliases: dict[str, str] = {}
    for name in sorted(entries):
        for index, camera in enumerate(entries[name].realism_cohort):
            key = (camera.mission, camera.spacecraft, camera.camera)
            if key in owner:
                other, other_index = owner[key]
                issues.append(
                    Issue(
                        sources[name],
                        f'realism_cohort[{index}]',
                        f'{key} is already claimed by {sources[other]} '
                        f'(realism_cohort[{other_index}]); one camera of the real cohort '
                        'matches one instrument',
                    )
                )
            else:
                owner[key] = (name, index)
        for alias in entries[name].aliases:
            if alias in entries:
                message = f'{alias!r} is already an instrument, defined in {files[alias]}'
                issues.append(Issue(sources[name], 'aliases', message))
            elif alias in aliases:
                other = aliases[alias]
                message = f'{alias!r} is already an alias of {other!r} in {files[other]}'
                issues.append(Issue(sources[name], 'aliases', message))
            else:
                aliases[alias] = name
    return issues


@lru_cache(maxsize=4)
def _load(config_dir: Path) -> InstrumentCatalog:
    """Read, merge, and check every config file in ``config_dir``."""
    layers, files = _read_entries(config_dir)
    if GENERIC_INSTRUMENT not in layers:
        message = f'no {GENERIC_INSTRUMENT!r} entry (the instrument-agnostic entry)'
        raise SchemaError([Issue(str(config_dir), '', message)])
    entries = _parse_entries(layers)
    sources = {name: layers[name].label for name in entries}
    issues = _cross_entry_issues(entries, sources, files)
    if issues:
        raise SchemaError(issues)
    return InstrumentCatalog(
        config_dir=config_dir,
        entries=entries,
        aliases={alias: name for name in entries for alias in entries[name].aliases},
        sources=sources,
    )


def load_instrument_catalog(*, config_dir: Path | None = None) -> InstrumentCatalog:
    """Load and check the instrument config files, once per directory.

    Parameters:
        config_dir: The directory to read; the packaged
            :data:`INSTRUMENT_CONFIG_DIR` when omitted.

    Returns:
        The checked catalog.  The result is cached, so every caller shares one
        catalog.

    Raises:
        ~spindoctor.schema.SchemaError: Listing every problem: each file that breaks
            the schema or defines an instrument another file defines, each entry
            that sets a base that doesn't exist or loops, reuses an alias, or claims
            another instrument's realism cohort.  Also if no file defines the
            ``generic`` entry, or that entry states no ``detector.calibrated_sky_if``.
    """
    return _load((config_dir or INSTRUMENT_CONFIG_DIR).resolve())


def instrument_entry(instrument: str | None) -> InstrumentEntry:
    """The packaged catalog's entry of one instrument.

    Parameters:
        instrument: An instrument name, an alias, or ``None`` for the generic entry.

    Returns:
        The entry.

    Raises:
        ValueError: If the name is neither an instrument nor an alias.
    """
    return load_instrument_catalog().entry(instrument)


def sim_instrument_names() -> tuple[str, ...]:
    """The packaged catalog's camera names, sorted, without ``generic`` or any alias."""
    return load_instrument_catalog().instrument_names()


def realism_cohort_instruments() -> dict[tuple[str, str | None, str], str]:
    """Map each real (mission, spacecraft, camera) to the instrument emulating its products.

    Returns:
        One entry per ``realism_cohort`` camera in the packaged catalog.
    """
    catalog = load_instrument_catalog()
    return {
        (camera.mission, camera.spacecraft, camera.camera): name
        for name in catalog.instrument_names()
        for camera in catalog.entry(name).realism_cohort
    }
