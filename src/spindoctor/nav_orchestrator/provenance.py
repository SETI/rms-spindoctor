"""Provenance — reproducibility metadata attached to every NavResult.

Two navigations with identical inputs produce byte-identical
``Provenance`` *except* for ``pipeline_run_iso8601``, which is wall-clock
by construction; regression-baseline comparison strips that field before
comparing.

The :func:`collect_provenance_metadata` helper produces the per-image
``spindoctor_git_sha``, loaded-SPICE-kernel list, static-data hash
dictionary, resolved-config hash (plus applied override paths), and
star-catalog identifiers at navigate time so the orchestrator can
populate the ``Provenance`` envelope without each caller re-implementing
the lookups.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from ruamel.yaml.error import YAMLError

from spindoctor.config import DEFAULT_CONFIG, IMAGE_LOGGER, Config, logged_section

__all__ = [
    'Provenance',
    'ProvenanceMetadata',
    'collect_provenance_metadata',
]


_STATIC_DATA_PREFIXES: tuple[str, ...] = (
    'config_220_',  # body shape catalogue (Phase 3+)
    'config_3',  # ring catalogues (300_*_rings.yaml)
    'config_4',  # per-instrument blocks (400_inst_coiss.yaml ...)
)
"""Filename prefixes counted as static-data YAML for hashing.

Anything in ``src/nav/config_files`` whose name starts with one of these
prefixes is sha256-hashed and recorded in ``Provenance.static_data_hashes``.
"""


@dataclass(frozen=True)
class Provenance:
    """Reproducibility envelope written into every NavResult.

    Parameters:
        spindoctor_version: ``__version__`` string (e.g. ``'0.5.2'``).
        spindoctor_git_sha: Short git SHA, ``'dirty'``, or ``None`` if neither
            can be determined.
        spice_kernels: Sorted tuple of SPICE kernel filenames actually
            loaded (from ``spice.ktotal`` / ``spice.kdata``).
        static_data_hashes: Mapping ``filename -> sha256(raw bytes)`` for
            static-data YAMLs (``config_220_body_shape.yaml``, every
            ``config_3N0_*_rings.yaml``, every ``config_4N0_inst_*.yaml``).
            Comments and whitespace are included in the hashed bytes.
            Stored as a read-only ``MappingProxyType`` after construction.
        technique_names: Sorted tuple of registered technique class names.
        extractor_names: Sorted tuple of registered extractor class names.
        config_hash: sha256 hex digest of the fully-resolved config content
            (bundled defaults plus applied overrides, deterministically
            serialized with sorted keys), or ``None`` when unavailable.
        config_overrides: Tuple of user/CLI override config file paths in
            application order (order matters -- later files win the merge),
            so it is deliberately not sorted in ``__post_init__``.
        star_catalogs: Mapping ``catalog name -> configured path or URL``
            for every star catalog in ``config.stars.catalogs``.  The value
            is the resolution root (env var or default), or ``''`` when the
            path cannot be determined.  No version numbers are recorded --
            the catalog data carries none.  Stored as a read-only
            ``MappingProxyType`` after construction.
        image_et: Observation midtime ET (TDB seconds past J2000).
        pipeline_run_iso8601: UTC timestamp when the run began.  Excluded
            from byte-identical regression-baseline comparison because it
            varies wall-clock-to-wall-clock for identical inputs.

    The non-init field ``spice_kernel_count`` is derived from
    ``len(spice_kernels)`` in ``__post_init__``.
    """

    spindoctor_version: str
    image_et: float
    pipeline_run_iso8601: str
    spindoctor_git_sha: str | None = None
    spice_kernels: tuple[str, ...] = ()
    static_data_hashes: Mapping[str, str] = field(default_factory=dict)
    technique_names: tuple[str, ...] = ()
    extractor_names: tuple[str, ...] = ()
    config_hash: str | None = None
    config_overrides: tuple[str, ...] = ()
    star_catalogs: Mapping[str, str] = field(default_factory=dict)
    spice_kernel_count: int = field(init=False)

    def __post_init__(self) -> None:
        """Normalize sequences, derive count, freeze the mapping fields."""
        # Normalize the three sequence fields to deterministic sorted
        # tuples so callers' mutable or unsorted inputs cannot leak in.
        # ``config_overrides`` keeps its caller-supplied order because the
        # merge is order-sensitive; it is only coerced to a tuple.
        # ``object.__setattr__`` is required because the dataclass is frozen.
        object.__setattr__(self, 'spice_kernels', tuple(sorted(self.spice_kernels)))
        object.__setattr__(self, 'technique_names', tuple(sorted(self.technique_names)))
        object.__setattr__(self, 'extractor_names', tuple(sorted(self.extractor_names)))
        object.__setattr__(self, 'config_overrides', tuple(self.config_overrides))
        object.__setattr__(self, 'spice_kernel_count', len(self.spice_kernels))
        if not isinstance(self.static_data_hashes, MappingProxyType):
            object.__setattr__(
                self,
                'static_data_hashes',
                MappingProxyType(dict(self.static_data_hashes)),
            )
        if not isinstance(self.star_catalogs, MappingProxyType):
            object.__setattr__(
                self,
                'star_catalogs',
                MappingProxyType(dict(sorted(dict(self.star_catalogs).items()))),
            )


@dataclass(frozen=True)
class ProvenanceMetadata:
    """The per-image runtime-derived provenance fields.

    Parameters:
        git_sha: Short git SHA of the repository, ``'dirty'`` if there are
            uncommitted changes, or ``None`` if not available.
        spice_kernels: Sorted tuple of SPICE kernel filenames actually
            loaded.
        static_data_hashes: Mapping of static-data YAML filename to
            sha256-hex digest of the file's raw bytes.
        config_hash: sha256 hex digest of the fully-resolved config
            content, or ``None`` when the hash could not be computed.
        config_overrides: Applied user/CLI override config file paths in
            application order.
        star_catalogs: Mapping of configured star-catalog name to its
            resolved path or URL (``''`` when unresolvable).
    """

    git_sha: str | None
    spice_kernels: tuple[str, ...]
    static_data_hashes: Mapping[str, str]
    config_hash: str | None
    config_overrides: tuple[str, ...]
    star_catalogs: Mapping[str, str]


@functools.cache
def _resolve_git_sha() -> str | None:
    """Return the short git SHA at the head of the working tree or ``None``.

    Uses ``git rev-parse HEAD`` to read the SHA and ``git status
    --porcelain`` to detect uncommitted changes (returning ``'dirty'`` in
    that case).  Returns ``None`` when the tree is not inside a git
    repository or git is unavailable.

    Process-memoized: the repo SHA does not change mid-run, so the two ``git``
    subprocesses run once per process rather than once per navigated image.
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', '--short', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if not sha:
        return None
    try:
        status = subprocess.run(
            ['git', '-C', str(repo_root), 'status', '--porcelain'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return sha
    return f'{sha}-dirty' if status.strip() else sha


def _resolve_spice_kernels() -> tuple[str, ...]:
    """Return the sorted tuple of currently-loaded SPICE kernel basenames.

    Reads the loaded kernels from ``cspyce``, the SPICE binding shared with
    ``oops``.  A navigation that reached this point ran on cspyce, so an import
    or lookup failure here is a broken installation and reaches the caller
    rather than being recorded as "no kernels" against a run that used plenty.
    The tuple holds *basenames* only so the hash and JSON output stay
    deterministic across machines with different kernel install roots.
    """
    # Imported here rather than at module scope to keep the import graph flat;
    # not guarded, because a navigation that reached this point ran on cspyce,
    # so an ImportError is a broken installation rather than a machine without
    # SPICE -- and answering "no kernels" for one would put that claim in the
    # record of a run that used plenty.
    import cspyce

    # Read straight through. This tuple is the record of which kernels produced
    # the result, so an empty one is not a missing diagnostic but a false
    # statement about the navigation -- and the run that made it is exactly the
    # run somebody would later try to reproduce from it.
    ktotal = int(cspyce.ktotal('ALL'))
    kernels: list[str] = []
    for index in range(ktotal):
        file_name, _, _, _ = cspyce.kdata(index, 'ALL')
        if file_name:
            kernels.append(Path(str(file_name)).name)
    return tuple(sorted(kernels))


@functools.cache
def _resolve_static_data_hashes() -> Mapping[str, str]:
    """Return ``{filename: sha256_hex(raw bytes)}`` for shipped static data.

    Walks ``src/nav/config_files`` and hashes any file whose name starts
    with one of the recognised static-data prefixes
    (``config_220_``, ``config_3``, ``config_4``).  Returns the mapping
    sorted by filename so equality testing is stable.

    Process-memoized: the shipped config files do not change mid-run, so the
    sha256 pass runs once per process rather than once per navigated image.

    Provenance metadata is best-effort: a per-file I/O failure (file
    disappearing between ``glob`` and ``read_bytes``, permission error,
    OS-level read error) is logged at WARNING and the file is skipped
    rather than allowed to abort the navigation run.
    """
    config_dir = Path(__file__).resolve().parent.parent / 'config_files'
    hashes: dict[str, str] = {}
    for path in sorted(config_dir.glob('*.yaml')):
        name = path.name
        if not any(name.startswith(prefix) for prefix in _STATIC_DATA_PREFIXES):
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            IMAGE_LOGGER.warning('static-data hash skipped for %s: %s', name, exc)
            continue
        hashes[name] = digest
    return MappingProxyType(hashes)


def _resolve_config_hash(config: Config) -> str | None:
    """Return the resolved-config sha256 digest, or ``None`` on failure.

    Provenance metadata is best-effort: a config that cannot be loaded or
    serialized (malformed user override, unreadable file) is logged at
    WARNING and reported as ``None`` rather than allowed to abort the
    navigation run.

    Parameters:
        config: Project ``Config`` whose resolved content is hashed.

    Returns:
        64-character sha256 hex digest, or ``None``.
    """
    try:
        return config.resolved_config_hash()
    except (OSError, TypeError, ValueError, YAMLError) as exc:
        IMAGE_LOGGER.warning('resolved-config hash unavailable: %s', exc)
        return None


def _resolve_star_catalogs(config: Config) -> Mapping[str, str]:
    """Return ``{catalog name: path or URL}`` for the configured catalogs.

    Walks ``config.stars.catalogs`` and records, per catalog, the same
    resolution root the catalog constructors use: ``UCAC4_PATH`` for
    UCAC4, ``YBSC_PATH`` for YBSC, and ``SPICE_PATH/Stars`` (falling back
    to ``OOPS_RESOURCES/SPICE/Stars``) for the SPICE-kernel-backed
    Tycho-2 catalog.  An unresolvable path is recorded as ``''`` -- the
    catalog name still appears so the metadata shows what was configured.
    Only names and paths/URLs are recorded; the catalog data carries no
    version identifier to report.

    Parameters:
        config: Project ``Config`` supplying ``stars.catalogs``.

    Returns:
        Read-only mapping sorted by catalog name.
    """
    # Read straight through. A configuration with no star catalogs is one this
    # navigation could not have used, so an empty mapping here is not an
    # unavailable diagnostic but a false one -- and AttributeError in
    # particular would say the same thing about a defect reading the section.
    catalog_names = [str(name).lower() for name in config.stars.catalogs]
    resolved: dict[str, str] = {}
    for name in catalog_names:
        if name == 'ucac4':
            resolved[name] = os.environ.get('UCAC4_PATH', '')
        elif name == 'ybsc':
            resolved[name] = os.environ.get('YBSC_PATH', '')
        elif name == 'tycho2':
            spice_path = os.environ.get('SPICE_PATH')
            oops_resources = os.environ.get('OOPS_RESOURCES')
            if spice_path:
                resolved[name] = f'{spice_path}/Stars'
            elif oops_resources:
                resolved[name] = f'{oops_resources}/SPICE/Stars'
            else:
                resolved[name] = ''
        else:
            resolved[name] = ''
    return MappingProxyType(dict(sorted(resolved.items())))


@logged_section('provenance', 'PROVENANCE')
def collect_provenance_metadata(config: Config | None = None) -> ProvenanceMetadata:
    """Gather process-wide provenance metadata at navigate time.

    Parameters:
        config: Project ``Config`` supplying the resolved-config hash, the
            applied override list, and the star-catalog configuration.
            Defaults to ``DEFAULT_CONFIG``.

    Returns:
        A :class:`ProvenanceMetadata` instance populated with the current
        git SHA, loaded SPICE kernel list, static-data hashes,
        resolved-config hash plus overrides, and star-catalog identifiers.
    """
    config = config or DEFAULT_CONFIG
    return ProvenanceMetadata(
        git_sha=_resolve_git_sha(),
        spice_kernels=_resolve_spice_kernels(),
        static_data_hashes=_resolve_static_data_hashes(),
        config_hash=_resolve_config_hash(config),
        config_overrides=config.override_paths,
        star_catalogs=_resolve_star_catalogs(config),
    )
