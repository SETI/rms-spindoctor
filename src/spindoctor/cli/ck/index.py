"""The pre-index of candidate C-kernels a corrected segment can be built on.

An image's metadata records which SPICE kernels were furnished when it
navigated, but it records sorted basenames only: no load order, no directory,
and in a batch run the list accumulates every kernel earlier images needed, so
it is a superset of the ones this image actually used.  Identifying the
baseline kernel an image navigated against therefore takes two steps, and this
module is the first of them: scan every C-kernel under the kernel directories
once per run, recording which objects each file describes and over what epochs,
so that the second step -- furnishing candidates one at a time and keeping the
ones that reproduce the recorded attitude -- has only a handful of files to try
per image instead of the whole holdings.

The scan uses ``ckobj`` and ``ckcov`` at segment level in TDB.  Both read a
file by name, so no C-kernel has to be furnished to be indexed, but the
leapseconds kernel and the spacecraft clock kernel of every object encountered
must be, since that is what converts a coverage window from clock ticks into
TDB.  An object whose clock is not furnished contributes no coverage and is
recorded as unreadable instead, because a real kernel can name an object no
clock kernel describes and no run should be stopped by an object none of its
images asks about; an image that does ask about one is refused by the
assignment step.  Both calls also need a real local file, so a kernel tree that
lives remotely is fetched as it is scanned; for a tree that is already local
nothing is copied.

Coverage is read with a tolerance for the objects whose attitude was navigated
through a tolerance-snapped pointing lookup.  Such a lookup answers with a
record up to its tolerance away from the epoch asked for, so an exposure just
outside a segment's window is still served by that segment, and a coverage
filter that did not allow for it would drop the only candidate that reproduces.
The filter is only a filter: reproduction, not coverage, decides.
"""

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import cast

import cspyce
from filecache import FCPath

from spindoctor.spice_ids import FROZEN_ATTITUDE_CK_IDS

# The extensions a binary C-kernel is stored under in the holdings.  Text
# kernels are never C-kernels, and the label files sitting beside the binaries
# carry their own extensions, so an extension test is enough to enumerate them.
CK_SUFFIXES = frozenset({'.bc', '.ck'})

# What a corrected file's name carries beyond the original's, before the
# extension.  It lives here because it is a fact about kernel filenames that
# the scan needs: a corrected kernel written back beside its original must not
# be indexed as a candidate for the next run.
OUTPUT_NAME_MARKER = '_nav'

# ``ckcov`` reads coverage at segment level -- the window each segment's
# descriptor advertises -- and reports it in TDB seconds, matching the epochs
# the metadata records.
_COVERAGE_LEVEL = 'SEGMENT'
_COVERAGE_TIME_SYSTEM = 'TDB'
_COVERAGE_NEEDS_ANGULAR_VELOCITY = False

# The widest tolerance, in spacecraft clock ticks, that a snapped pointing
# lookup can be made at: the value oops registers its fallback Voyager frame
# with.  It is declared once and read twice, because the two uses have to
# agree.  The reproduction step asks for pointing this far from an exposure
# midtime, and the scan below widens a frozen-attitude object's coverage by
# the same amount so that every image that lookup can serve survives the
# coverage filter and reaches the reproduction step at all.  Widening by less
# than the lookup reaches drops the only candidate that reproduces and reports
# the image as having no baseline.  The two agree to within half a tick at the
# extreme edge, because the filter measures from the exposure midtime while
# the lookup measures from that midtime rounded to a whole tick; an image that
# far from any pointing record is refused rather than corrected, which is the
# safe direction.
SNAPPED_LOOKUP_TOL_TICKS = 80000.0

# Coverage tolerance for an object whose attitude is read by evaluating a
# frame chain at the epoch itself, which reaches exactly as far as it is asked.
_COVERAGE_TOL_TICKS = 0.0


class KernelClass(Enum):
    """How the producer of a C-kernel described its pointing.

    The holdings carry reconstructed, gapfill and predicted kernels with
    overlapping coverage, so an image can navigate against a baseline that
    several files reproduce.  This is the first key of the tie-break that then
    picks one of them.
    """

    RECONSTRUCTED = 'reconstructed'
    GAPFILL = 'gapfill'
    PREDICTED = 'predicted'
    UNCLASSIFIED = 'unclassified'

    @property
    def preference_rank(self) -> int:
        """Rank in the tie-break, lowest first.

        Returns:
            The class's position in the preference order: reconstructed
            pointing before gapfill before predicted, and a kernel whose name
            declares no class last of all.
        """
        return _CLASS_PREFERENCE.index(self)


# Reconstructed pointing is measured, gapfill fills what reconstruction did not
# cover, and predicted pointing is a plan.  A kernel whose name declares none
# of those is preferred last, so one that does declare a class is always chosen
# over one that does not.  The ranking decides which output file carries a
# segment and never which attitude it carries: every candidate it orders has
# already reproduced the same recorded attitude.  A mission whose names declare
# nothing therefore ties on this key for every candidate and falls through to
# the basename, which is exactly what it did before any class existed.
_CLASS_PREFERENCE: tuple[KernelClass, ...] = (
    KernelClass.RECONSTRUCTED,
    KernelClass.GAPFILL,
    KernelClass.PREDICTED,
    KernelClass.UNCLASSIFIED,
)


@dataclass(frozen=True)
class _MissionNameRules:
    """How one mission's C-kernel basenames declare a kernel class.

    Parameters:
        mission: The mission whose holdings the names come from.  It names the
            source of each match in the message a two-class name is refused
            with.
        rules: One pattern per class the mission's names can declare, each
            matched in full against the lowercased basename.  A mission whose
            basenames declare no class carries no rules, which records that its
            holdings were read and encode nothing rather than that nobody
            looked.
    """

    mission: str
    rules: tuple[tuple[re.Pattern[str], KernelClass], ...]


# Cassini names the class in a release code that follows the two dates a
# kernel's coverage spans: ``p`` for planned pointing, ``r`` for reconstructed,
# plus a letter distinguishing successive releases of the same span.  Two
# patterns carry the reconstructed class because two date conventions are in
# use and the digit counts are what tell them apart: the Saturn tour and the
# cruise stamp YYDOY_YYDOY, the Jupiter flyby stamps YYMMDD_YYMMDD, and the
# earliest flyby release omits the code altogether.  The gapfill kernels are
# ``pa`` names, so the predicted pattern excludes them itself instead of
# relying on being tested second -- the patterns are mutually exclusive by
# construction, not by order.  This is the convention rms-spyceman's Cassini
# rules encode.
_CASSINI_NAME_RULES: tuple[tuple[re.Pattern[str], KernelClass], ...] = (
    (re.compile(r'\d{5}_\d{5}p[a-z]_gapfill_v\d+\.bc'), KernelClass.GAPFILL),
    (re.compile(r'\d{5}_\d{5}p[a-z](?!_gapfill).*\.bc'), KernelClass.PREDICTED),
    (re.compile(r'\d{5}_\d{5}r[a-z]\.bc'), KernelClass.RECONSTRUCTED),
    (re.compile(r'\d{6}_\d{6}(?:r[a-z])?\.bc'), KernelClass.RECONSTRUCTED),
)

# New Horizons marks only the pair of kernels that exist in both forms, with a
# trailing ``_recon`` or ``_pred`` on the mission's ``nh_`` prefix.  Every other
# name in the holdings -- the merged pointing files and the hazard-search
# kernels -- declares nothing, and is left declaring nothing rather than
# guessed at from a prefix that says which product it is and not how its
# pointing was made.
_NEW_HORIZONS_NAME_RULES: tuple[tuple[re.Pattern[str], KernelClass], ...] = (
    (re.compile(r'nh_.+_recon\.bc'), KernelClass.RECONSTRUCTED),
    (re.compile(r'nh_.+_pred\.bc'), KernelClass.PREDICTED),
)

# The encoding is a property of each mission's naming convention, so it is
# declared one mission at a time rather than as a single pattern set that would
# be mission-specific only by the accident of matching one mission's names.
# Voyager and Galileo are listed with no rules: each holds one kind of C-kernel
# and neither names it anywhere in a basename, so ``UNCLASSIFIED`` is the
# correct answer and the tie-break falls through to the basename for them.
_MISSION_NAME_RULES: tuple[_MissionNameRules, ...] = (
    _MissionNameRules(mission='Cassini', rules=_CASSINI_NAME_RULES),
    _MissionNameRules(mission='New Horizons', rules=_NEW_HORIZONS_NAME_RULES),
    _MissionNameRules(mission='Voyager', rules=()),
    _MissionNameRules(mission='Galileo', rules=()),
)


@dataclass(frozen=True)
class CoverageInterval:
    """One span of epochs over which a C-kernel file describes one object.

    Parameters:
        ck_frame_id: SPICE id of the object covered.
        start_et: First covered epoch, TDB seconds past J2000.
        stop_et: Last covered epoch, TDB seconds past J2000.
    """

    ck_frame_id: int
    start_et: float
    stop_et: float

    def __post_init__(self) -> None:
        """Refuse a window that is not a window.

        Raises:
            ValueError: if either endpoint is not finite, or if the interval
                ends before it starts.  An infinite endpoint would report every
                epoch as covered, and a NaN one would report none, in both
                cases without any comparison ever failing.
        """
        for label, value in (('start_et', self.start_et), ('stop_et', self.stop_et)):
            if not math.isfinite(value):
                raise ValueError(
                    f'coverage {label} for object {self.ck_frame_id} is not finite: {value!r}'
                )
        if self.stop_et < self.start_et:
            raise ValueError(
                f'coverage for object {self.ck_frame_id} ends at {self.stop_et!r} before it '
                f'starts at {self.start_et!r}'
            )

    def contains(self, ck_frame_id: int, et: float) -> bool:
        """Report whether this interval covers one object at one epoch.

        Parameters:
            ck_frame_id: SPICE id of the object asked about.
            et: TDB seconds past J2000.

        Returns:
            True when the object matches and the epoch lies within the
            interval, endpoints included.  False for a non-finite epoch: both
            endpoints are finite, refused at construction otherwise, so an
            infinity falls outside the window and a NaN answers every
            comparison with False.
        """
        if ck_frame_id != self.ck_frame_id:
            return False
        return bool(self.start_et <= et <= self.stop_et)


@dataclass(frozen=True)
class CkFile:
    """One C-kernel file the index found, and what it covers.

    Parameters:
        path: Path to the file, local or remote.
        kernel_class: How its producer described its pointing, declared by its
            own basename.
        coverage: One interval per object per segment window, in the order
            SPICE reported them.
        unreadable_objects: The objects the file describes whose coverage
            could not be expressed in TDB, because no furnished kernel defines
            the spacecraft clock their time tags are encoded against.  The file
            offers no coverage for them, so it is never a candidate for one,
            and an image that needs one is refused by the assignment step
            rather than being told its baseline has drifted.
    """

    path: FCPath
    kernel_class: KernelClass
    coverage: tuple[CoverageInterval, ...]
    unreadable_objects: tuple[int, ...] = ()

    @property
    def basename(self) -> str:
        """The file's basename, which is what image metadata records."""
        return self.path.name

    def covers(self, ck_frame_id: int, et: float) -> bool:
        """Report whether this file describes one object at one epoch.

        Parameters:
            ck_frame_id: SPICE id of the object asked about.
            et: TDB seconds past J2000.

        Returns:
            True when any of the file's coverage intervals contains the epoch
            for that object.
        """
        return any(interval.contains(ck_frame_id, et) for interval in self.coverage)


@dataclass(frozen=True)
class CkIndex:
    """Every C-kernel found under a run's kernel directories.

    Parameters:
        files: The indexed files.  A basename may appear more than once, since
            two directories may hold files of the same name; each such file is
            an independent candidate.

    Raises:
        ValueError: if two entries name the same path, which would offer the
            same file twice as a candidate.
    """

    files: tuple[CkFile, ...]
    _by_basename: dict[str, tuple[CkFile, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Group the files by basename, which is how metadata names them."""
        paths = [ck_file.path.as_posix() for ck_file in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError('the index holds the same path more than once')
        grouped: dict[str, list[CkFile]] = {}
        for ck_file in self.files:
            grouped.setdefault(ck_file.basename, []).append(ck_file)
        object.__setattr__(
            self, '_by_basename', {name: tuple(found) for name, found in grouped.items()}
        )

    @property
    def unreadable_objects(self) -> frozenset[int]:
        """The objects whose coverage no indexed file could express in TDB.

        An object appears here when the spacecraft clock its time tags are
        encoded against is not furnished, so its coverage window cannot be
        converted.  Nothing in the index offers coverage for such an object,
        which makes it invisible to :meth:`candidates`; the assignment step
        reads this so an image that needs one is refused for the reason it
        actually has.
        """
        return frozenset(
            ck_frame_id for ck_file in self.files for ck_frame_id in ck_file.unreadable_objects
        )

    def candidates(
        self, *, basenames: Collection[str], ck_frame_id: int, et: float
    ) -> tuple[CkFile, ...]:
        """Return the indexed files that could have supplied one image's baseline.

        A file qualifies when the image's metadata names its basename, when it
        describes the CK object the image's correction targets, and when its
        coverage includes the exposure midtime.  A basename found in more than
        one directory contributes each of those files, since only reproduction
        can tell them apart.

        Parameters:
            basenames: The kernel basenames the image's metadata recorded.
                Repeats are ignored; a basename the index does not hold
                contributes nothing.
            ck_frame_id: SPICE id of the object the correction targets.
            et: The exposure midtime, TDB seconds past J2000.

        Returns:
            The qualifying files, ordered by preference: kernel class first,
            then the greatest basename, then the greatest path, so that the
            first entry is the one the tie-break selects.
        """
        found: list[CkFile] = []
        for basename in set(basenames):
            if basename not in self._by_basename:
                continue
            found.extend(
                ck_file
                for ck_file in self._by_basename[basename]
                if ck_file.covers(ck_frame_id, et)
            )
        return tuple(sorted(found, key=preference_key, reverse=True))


def preference_key(ck_file: CkFile) -> tuple[int, str, str]:
    """Return the sort key that orders candidates by preference, greatest first.

    Several kernels can reproduce one image's baseline, because the holdings
    carry reconstructed, gapfill and predicted sets with overlapping coverage.
    They agree on the attitude by construction -- that is what reproducing it
    means -- so the choice among them decides only which output file carries
    the image's segment, and all it has to be is deterministic.

    Parameters:
        ck_file: The candidate to key.

    Returns:
        A key whose greatest value is the preferred candidate: the negated
        class rank first, so reconstructed beats gapfill beats predicted, then
        the basename, so the lexicographically greatest name wins, then the
        full path, so two directories holding the same basename in the same
        class still resolve the same way on every run.
    """
    return (
        -ck_file.kernel_class.preference_rank,
        ck_file.basename,
        ck_file.path.as_posix(),
    )


def kernel_class_for_basename(basename: str) -> KernelClass:
    """Classify a C-kernel from its own basename.

    The name is matched against the naming convention of every mission whose
    C-kernels the scan reads.  Those conventions are disjoint, each anchored on
    a shape only one mission's names have, so at most one class is ever
    declared; a mission whose names encode no class contributes no pattern and
    leaves its kernels ``UNCLASSIFIED``.

    Pattern matching ignores case, since the holdings spell some kernel names
    in upper case and a name's case says nothing about its class.  The
    corrected-kernel marker below ignores case for the same reason, and the
    scan tests it the same way, so the two agree on exactly which names those
    are: an upper-cased copy of a corrected kernel must not slip past the
    marker and then classify as a baseline candidate through the
    case-blind patterns.

    Parameters:
        basename: The file's own name, extension included, spelled as the
            holdings spell it and as image metadata records it.

    Returns:
        The class the name declares, or ``UNCLASSIFIED`` when it declares none.
        A name carrying no extension, or one under an extension its mission
        does not use, declares none: the conventions include the extension.

    Raises:
        ValueError: if the name is empty, is ``.`` or ``..``, or holds a path
            separator, none of which is a file's own name and each of which
            would otherwise be reported as declaring no class; if the name's
            stem ends in the corrected-kernel marker, since a corrected kernel
            is written by this program and is never a baseline candidate; or if
            the name declares more than one class, which no preference order
            can resolve and which would otherwise be settled silently by the
            order the patterns happen to be tested in.
    """
    if len(basename) == 0 or basename in ('.', '..') or '/' in basename:
        raise ValueError(
            f'{basename!r} is not a C-kernel basename; classify the name of the file itself '
            f'rather than a path or a fragment of one'
        )
    if Path(basename).stem.lower().endswith(OUTPUT_NAME_MARKER):
        raise ValueError(
            f'C-kernel basename {basename!r} names a corrected kernel, which this program '
            f'writes and never classifies as a baseline candidate'
        )
    lowered = basename.lower()
    matched = [
        (mission_rules.mission, kernel_class)
        for mission_rules in _MISSION_NAME_RULES
        for pattern, kernel_class in mission_rules.rules
        if pattern.fullmatch(lowered) is not None
    ]
    declared = {kernel_class for _, kernel_class in matched}
    if len(declared) > 1:
        raise ValueError(
            f'C-kernel basename {basename!r} declares more than one kernel class: '
            f'{sorted(f"{mission} {kernel_class.value}" for mission, kernel_class in matched)}'
        )
    if len(declared) == 0:
        return KernelClass.UNCLASSIFIED
    return matched[0][1]


def build_ck_index(roots: Sequence[str | Path | FCPath]) -> CkIndex:
    """Scan the kernel directories of a run and index every C-kernel in them.

    Each directory is listed without recursion, so a subdirectory of kernels is
    indexed only when it is named as a root of its own.  The caller must have
    furnished the leapseconds kernel and the spacecraft clock kernel of every
    mission whose kernels are scanned, since coverage is reported in TDB and
    converting a clock tick to TDB needs both.

    A corrected kernel is never indexed.  Writing the corrections back beside
    the originals is the natural thing to do, and a corrected kernel reproduces
    its own baseline exactly wherever the correction was the identity, so
    indexing one would offer a correction as the baseline for the next run.

    Parameters:
        roots: The directories to scan.  A directory is only where kernels are
            found; each kernel is classified by its own basename, so how the
            directories are named and split makes no difference to the index.

    Returns:
        The index.

    Raises:
        ValueError: if no directory is given, if one names the same directory
            as another once symbolic links and ``..`` are resolved, if one does
            not exist or is not a directory, if a kernel's basename declares
            more than one kernel class, or if no C-kernel is found under any of
            them -- an empty index would report every image as having no
            reproducing baseline, which is indistinguishable from a genuine
            baseline drift.
        OSError: if a file with a C-kernel extension cannot be read as one.
    """
    if len(roots) == 0:
        raise ValueError('no kernel directory to scan; the index would hold nothing')
    directories = [FCPath(root) for root in roots]
    resolved = [root.resolve().as_posix() for root in directories]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f'a kernel directory is named more than once: '
            f'{[root.as_posix() for root in directories]}'
        )
    files: list[CkFile] = []
    for root in directories:
        files.extend(_index_one_file(path) for path in _kernel_files(root))
    if len(files) == 0:
        raise ValueError(
            f'no C-kernel found under {[root.as_posix() for root in directories]}; every image '
            f'would be reported as having no reproducing baseline'
        )
    return CkIndex(files=tuple(files))


def _kernel_files(root: FCPath) -> list[FCPath]:
    """List the C-kernels of one directory, in a stable order.

    The directory is listed rather than probed first: for a remote tree an
    existence check is a round trip of its own, and the listing answers the
    same question as a side effect.

    Parameters:
        root: The directory to list.

    Returns:
        The C-kernels it holds, excluding corrected kernels, ordered by path.

    Raises:
        ValueError: if the directory does not exist or is not a directory.
    """
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.as_posix())
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(
            f'kernel directory {root.as_posix()!r} does not exist or is not a directory'
        ) from exc
    return [
        entry
        for entry in entries
        if entry.suffix.lower() in CK_SUFFIXES
        and not entry.stem.lower().endswith(OUTPUT_NAME_MARKER)
        and entry.is_file()
    ]


def _index_one_file(path: FCPath) -> CkFile:
    """Read one C-kernel's class, objects and coverage.

    Parameters:
        path: The file to read.

    Returns:
        The indexed file, carrying the coverage of every object whose clock is
        furnished and the ids of the objects whose clock is not.

    Raises:
        ValueError: if the file's basename declares more than one kernel class.
        OSError: if the file cannot be read as a C-kernel.
    """
    kernel_class = kernel_class_for_basename(path.name)
    # SPICE reads a C-kernel by name from the local filesystem, so a remote one
    # is fetched first.  This is a no-op for a kernel that is already local.
    local = str(cast(Path, path.retrieve()))
    coverage: list[CoverageInterval] = []
    unreadable: list[int] = []
    for ck_frame_id in sorted(int(value) for value in cspyce.ckobj(local)):
        tolerance = (
            SNAPPED_LOOKUP_TOL_TICKS
            if ck_frame_id in FROZEN_ATTITUDE_CK_IDS
            else _COVERAGE_TOL_TICKS
        )
        try:
            window = [
                float(value)
                for value in cspyce.ckcov(
                    local,
                    ck_frame_id,
                    _COVERAGE_NEEDS_ANGULAR_VELOCITY,
                    _COVERAGE_LEVEL,
                    tolerance,
                    _COVERAGE_TIME_SYSTEM,
                )
            ]
        except LookupError:
            # A real kernel can describe an object whose spacecraft clock no
            # kernel defines: a merged New Horizons pointing file names object
            # -1 beside the spacecraft, and the clock id SPICE computes for it
            # is 0, which no SCLK kernel supplies.  Converting that object's
            # coverage to TDB is impossible, and refusing the whole scan over
            # it would make every New Horizons directory unindexable for the
            # sake of an object no image will ever ask about.  The object is
            # recorded instead, and an image that does ask about one is refused
            # before any candidate is tried.
            unreadable.append(ck_frame_id)
            continue
        coverage.extend(
            CoverageInterval(ck_frame_id=ck_frame_id, start_et=window[at], stop_et=window[at + 1])
            for at in range(0, len(window), 2)
        )
    return CkFile(
        path=path,
        kernel_class=kernel_class,
        coverage=tuple(coverage),
        unreadable_objects=tuple(unreadable),
    )
