"""Which baseline kernel each image's correction belongs to, and where it goes.

A corrected C-kernel is an overlay on exactly one original: it carries the
segments of the images whose attitude that original supplied, under the
original's own name with ``_nav`` before the extension.  Deciding which
original that is cannot be a guess.  The metadata records sorted kernel
basenames with no load order, accumulated across a batch, so several of them
may describe the right object over the right epochs while only one of them is
the file the navigation actually read.

So the pairing is made by reproduction.  Each candidate is furnished on its own
and asked for the attitude the image navigated against; a candidate that
answers the recorded ``cmatrix_original`` to within a nanoradian supplied that
baseline, and one that does not, did not.  An image no candidate reproduces
receives no segment at all: that is also the detector for a kernel set that has
changed since navigation ran, since a corrected segment measured against a
baseline that no longer exists is worse than no segment.

How the attitude is asked for follows how it was navigated.  Most instruments
evaluate a frame chain at the exposure midtime, which ``pxform`` reproduces.
Voyager does not: its observation frame is frozen from a single
tolerance-snapped pointing lookup, so reproducing it means making the same
lookup, at the same clock tick, with the same tolerance, and composing the same
fixed platform-to-camera rotation on top.  There are two such lookups to try,
because oops falls back to a second, far wider tolerance when the first finds
nothing, and each encodes the epoch its own way.

The caller owns the kernel pool.  The supporting kernels -- leapseconds, the
spacecraft clock navigation used, and the frame kernel defining the camera and
CK object frames -- must be furnished before assignment runs, and no C-kernel
may be, since a stray one answers the same lookups and would make the
reproduction test meaningless.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cspyce
import numpy as np

from spindoctor.cli.ck.images import (
    BOTSIM_WINNING_CAMERA,
    ImageEntry,
    OmissionReason,
    botsim_losers,
)
from spindoctor.cli.ck.index import (
    CK_SUFFIXES,
    OUTPUT_NAME_MARKER,
    SNAPPED_LOOKUP_TOL_TICKS,
    CkFile,
    CkIndex,
)
from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.pool import furnished
from spindoctor.cli.ck.segment import BASE_FRAME, resolve_sclk_id
from spindoctor.spice_ids import FROZEN_ATTITUDE_CK_IDS
from spindoctor.support.types import NDArrayFloatType

# A candidate reproduces a recorded attitude when the rotation between the two
# is smaller than this.  It is a reproduction bound, not a navigation bound:
# the same kernels evaluated the same way answer to floating-point noise, so
# anything larger is a different kernel.
REPRODUCTION_TOL_RAD = 1e-9

# The tolerance oops uses for the snapped pointing lookup that freezes a
# Voyager observation frame, in spacecraft clock ticks: a fixed part plus a
# term in the exposure.  When that lookup finds nothing, oops falls back to a
# frame registered at a far wider tolerance, so a baseline is tried at both.
# That wider one is ``SNAPPED_LOOKUP_TOL_TICKS``, read from the index rather
# than restated here: the index widens a frozen-attitude object's coverage by
# the same tolerance, and an image this lookup can serve but that widening
# does not admit never reaches this code at all.
_SNAPPED_TOL_TICKS = 800.0
_SNAPPED_TOL_EXPOSURE_DIVISOR = 48.0

# The wider tolerance is 4799.995 s of Voyager clock, measured, and the frame
# that uses it caches its answer for that long: two Voyager images navigated
# through the fallback within eighty minutes of each other in one process share
# one attitude.  The second then records an attitude no lookup at its own
# midtime reproduces, and is refused rather than corrected against a
# baseline it did not use.

# The fixed rotation from a frozen-attitude object's frame to the camera frame
# is read at this epoch, as oops reads it: the frame kernel defines it as a
# constant, so the epoch only has to match.
_FIXED_ROTATION_ET = 0.0

# What cspyce raises when the furnished kernels cannot answer a lookup: a
# missing frame connection or an unresolvable clock arrives as a LookupError,
# a kernel it cannot read as an OSError, and other SPICE errors as a
# RuntimeError or a ValueError.  A candidate that cannot answer is a candidate
# that did not supply the baseline, which is an ordinary outcome of asking.
_SPICE_LOOKUP_FAILURES = (LookupError, OSError, RuntimeError, ValueError)


@dataclass(frozen=True)
class Assignment:
    """One image's disposition: the baseline it corrects, or why it has none.

    Parameters:
        entry: The image.
        baseline: The C-kernel file that supplied its uncorrected attitude, or
            ``None`` when it receives no segment.
        omission_reason: Why it receives no segment, or ``None`` when it does.

    Raises:
        ValueError: if a baseline is paired with a reason there is none, if
            neither is present, or if a baseline is assigned to an image that
            carries no pointing.
    """

    entry: ImageEntry
    baseline: CkFile | None
    omission_reason: OmissionReason | None

    def __post_init__(self) -> None:
        """Refuse an assignment that both writes a segment and does not."""
        if (self.baseline is None) == (self.omission_reason is None):
            raise ValueError(
                f'{self.entry.image_name} must be assigned either a baseline kernel or a reason '
                f'it has none, not both and not neither'
            )
        if self.baseline is not None and self.entry.pointing is None:
            raise ValueError(
                f'{self.entry.image_name} is assigned the baseline {self.baseline.basename} but '
                f'carries no pointing to write'
            )

    @property
    def image_name(self) -> str:
        """Basename of the image this assignment is for."""
        return self.entry.image_name

    @property
    def output_name(self) -> str | None:
        """Basename of the corrected file carrying the segment, or None."""
        if self.baseline is None:
            return None
        return output_basename(self.baseline.basename)


@dataclass(frozen=True)
class OutputGroup:
    """One corrected C-kernel file and the images whose segments it carries.

    Parameters:
        baseline: The original C-kernel this file mirrors.
        name: Basename of the corrected file.
        assignments: The images it carries, in the order they were assigned.
    """

    baseline: CkFile
    name: str
    assignments: tuple[Assignment, ...]


def output_basename(basename: str) -> str:
    """Return the name of the corrected file mirroring one original.

    The corrected file carries the original's name with ``_nav`` before the
    extension, so that the pairing is legible without opening either file:
    ``03236_04002ra.bc`` becomes ``03236_04002ra_nav.bc``.

    Parameters:
        basename: Basename of the original C-kernel.

    Returns:
        The corrected file's basename.

    Raises:
        ValueError: if the name is not a bare C-kernel basename -- if it is
            empty, carries a directory component, or does not end in a
            C-kernel extension, which a name that is nothing but an extension
            does not either -- or if it is already a corrected file's name,
            which would otherwise correct a correction and write it to a third
            name.
    """
    path = Path(basename)
    if len(path.parts) != 1 or basename in ('.', '..'):
        raise ValueError(f'{basename!r} is not a bare C-kernel basename')
    if path.suffix.lower() not in CK_SUFFIXES:
        raise ValueError(
            f'{basename!r} does not end in a C-kernel extension; expected one of '
            f'{sorted(CK_SUFFIXES)}'
        )
    stem = path.stem
    # Case-blind, as the index tests it: an upper-cased copy of a corrected
    # kernel is still a corrected kernel.
    if stem.lower().endswith(OUTPUT_NAME_MARKER):
        raise ValueError(
            f'{basename!r} is already a corrected kernel; correcting it again would measure a '
            f'correction against a corrected baseline'
        )
    return f'{stem}{OUTPUT_NAME_MARKER}{path.suffix}'


def rotation_angle_rad(first: NDArrayFloatType, second: NDArrayFloatType) -> float:
    """Return the angle of the rotation carrying one attitude onto another.

    The angle is taken through a quaternion rather than through the trace,
    because the trace form loses half its digits as the angle goes to zero,
    which is the regime a reproduction test lives in.  It is zero if and only
    if the two matrices are equal, so it measures a wrong direction as readily
    as a wrong magnitude.

    Parameters:
        first: A 3x3 rotation.
        second: A 3x3 rotation.

    Returns:
        The angle in radians, in ``[0, pi]``.

    Raises:
        ValueError: if either matrix is not 3x3 or holds a non-finite value.
            A NaN would otherwise answer every comparison with False, which
            reads as agreement in one direction and disagreement in the other
            depending only on how the test is written.
    """
    relative = _as_finite_3x3(first, 'first') @ _as_finite_3x3(second, 'second').T
    quaternion = np.asarray(cspyce.m2q(relative), dtype=np.float64)
    return float(2.0 * np.arctan2(float(np.linalg.norm(quaternion[1:])), abs(float(quaternion[0]))))


def _as_finite_3x3(matrix: NDArrayFloatType, label: str) -> NDArrayFloatType:
    """Return one attitude as a finite 3x3 float array.

    Parameters:
        matrix: The matrix to check.
        label: Name used in the exception messages.

    Returns:
        The matrix as float64.

    Raises:
        ValueError: if it is not 3x3 or holds a non-finite value.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f'{label} attitude is not a 3x3 matrix; got shape {array.shape}')
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f'{label} attitude holds a non-finite value: {array.tolist()!r}')
    return array


def attitudes_reproduce(recorded: NDArrayFloatType, evaluated: NDArrayFloatType) -> bool:
    """Report whether an evaluated attitude reproduces a recorded one.

    Parameters:
        recorded: The attitude the metadata recorded.
        evaluated: The attitude a candidate kernel gives.

    Returns:
        True when the rotation between them is at most
        ``REPRODUCTION_TOL_RAD``.

    Raises:
        ValueError: if either matrix is not a finite 3x3.
    """
    return rotation_angle_rad(recorded, evaluated) <= REPRODUCTION_TOL_RAD


def reproduces_baseline(pointing: ImagePointing) -> bool:
    """Report whether the furnished kernels reproduce an image's baseline.

    Exactly one candidate C-kernel must be furnished, along with the supporting
    kernels; this asks it for the uncorrected attitude the image navigated
    against and compares it against the recorded one.

    The attitudes it is compared against are the ones
    :func:`baseline_attitudes` gives, and any of them reproducing it is enough.

    Parameters:
        pointing: The image's recorded pointing.

    Returns:
        True when the furnished kernels answer the recorded
        ``cmatrix_original`` to within ``REPRODUCTION_TOL_RAD``.  False when
        they answer something else, and False when they cannot answer at all,
        which is what a candidate covering the wrong epochs does.

    Raises:
        ValueError: if the CK object is not one this writer knows, or if the
            spacecraft clock resolved for it is not the expected one.
        KeyError: if a frozen-attitude object has no frame name in the
            furnished kernels, which is a missing frame kernel rather than a
            candidate that does not reproduce.
    """
    return any(
        attitudes_reproduce(pointing.cmatrix_original, evaluated)
        for evaluated in baseline_attitudes(pointing)
    )


def baseline_attitudes(pointing: ImagePointing) -> tuple[NDArrayFloatType, ...]:
    """Return the attitudes the furnished kernels give for one image's baseline.

    There is one per way the navigated observation frame could have been built,
    in the order oops would have built them, and a way the furnished kernels
    cannot answer contributes nothing rather than raising -- so an empty result
    means this candidate answered no lookup at all.

    Most instruments evaluate a frame chain at the exposure midtime, which is
    one way and therefore at most one attitude.  A frozen-attitude object has
    two: a pointing lookup at the whole clock tick of the midtime with the
    tolerance that lookup used, and, since oops falls back to a frame
    registered at a far wider tolerance when that finds nothing, a second
    lookup at that tolerance and at the continuously encoded tick.  Both are
    composed with the fixed rotation from the object's frame to the camera
    frame.  On a baseline that interpolates between its records the two answer
    differently, because a whole tick and a fractional one are different
    epochs; on a discrete baseline the tolerance is what decides whether
    either answers at all.

    Parameters:
        pointing: The image's recorded pointing.

    Returns:
        The attitudes, in the order oops would have tried them.

    Raises:
        ValueError: if the CK object is not one this writer knows, or if the
            spacecraft clock resolved for it is not the expected one.
        KeyError: if a frozen-attitude object has no frame name in the
            furnished kernels.
    """
    if pointing.ck_frame_id not in FROZEN_ATTITUDE_CK_IDS:
        chained = _evaluated_attitude(pointing.camera_frame, pointing.midtime_et)
        return () if chained is None else (chained,)
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    snapped_tol = _SNAPPED_TOL_TICKS + pointing.exposure_s / _SNAPPED_TOL_EXPOSURE_DIVISOR
    # oops encodes the midtime as a whole tick for the primary lookup and
    # continuously for the wider fallback, so each is reproduced as it is
    # made: a whole tick and a fractional one land on different attitudes
    # wherever the baseline interpolates between its records.
    attempts = (
        (float(cspyce.sce2t(sclk_id, pointing.midtime_et)), snapped_tol),
        (float(cspyce.sce2c(sclk_id, pointing.midtime_et)), SNAPPED_LOOKUP_TOL_TICKS),
    )
    snapped = [_snapped_attitude(pointing, tick, tolerance) for tick, tolerance in attempts]
    return tuple(attitude for attitude in snapped if attitude is not None)


def _evaluated_attitude(camera_frame: str, et: float) -> NDArrayFloatType | None:
    """Evaluate the J2000-to-camera rotation from the furnished kernels.

    Parameters:
        camera_frame: SPICE name of the camera frame.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation, or ``None`` when the furnished kernels cannot supply
        it -- which for a candidate under test means it is not the baseline.
    """
    try:
        matrix = cspyce.pxform(BASE_FRAME, camera_frame, et)
    except _SPICE_LOOKUP_FAILURES:
        return None
    return np.asarray(matrix, dtype=np.float64)


def _snapped_attitude(
    pointing: ImagePointing, tick: float, tolerance_ticks: float
) -> NDArrayFloatType | None:
    """Read a frozen observation frame's attitude the way oops built it.

    Parameters:
        pointing: The image's recorded pointing.
        tick: Encoded spacecraft clock time to ask at.
        tolerance_ticks: How far from that tick a pointing record may be found.

    Returns:
        The 3x3 J2000-to-camera rotation the lookup gives, composed with the
        fixed rotation from the CK object's frame to the camera frame, or
        ``None`` when the furnished kernels have no pointing within the
        tolerance.
    """
    try:
        j2000_to_object, _clkout = cspyce.ckgp(
            pointing.ck_frame_id, tick, tolerance_ticks, BASE_FRAME
        )
    except _SPICE_LOOKUP_FAILURES:
        return None
    object_frame = str(cspyce.frmnam(pointing.ck_frame_id))
    object_to_camera = np.asarray(
        cspyce.pxform(object_frame, pointing.camera_frame, _FIXED_ROTATION_ET), dtype=np.float64
    )
    attitude: NDArrayFloatType = object_to_camera @ np.asarray(j2000_to_object, dtype=np.float64)
    return attitude


def assign_images(entries: Sequence[ImageEntry], index: CkIndex) -> tuple[Assignment, ...]:
    """Decide, for every image a run considered, what becomes of its pointing.

    Each eligible image is paired with the indexed C-kernel that reproduces its
    recorded uncorrected attitude; every other image is given a reason.
    Candidates are tried one at a time with nothing else furnished, and images
    sharing a candidate set are tested together, so the kernel pool changes
    once per set rather than once per image.

    Simultaneous exposures are settled after the reproduction test, not
    before: a narrow angle frame suppresses its wide angle partner only when
    its own baseline reproduced and it will actually write.  One that writes
    nothing yields the bus to its partner, whose correction then conflicts
    with nothing.

    Parameters:
        entries: The images the run considered.
        index: The pre-indexed C-kernels their baselines may be among.

    Returns:
        One assignment per image, in the order the images were given.

    Raises:
        ValueError: if two entries name the same image, whose assignments
            could not then be told apart; if a C-kernel is already furnished,
            which would answer the reproduction lookups alongside the candidate
            under test; if a frame an image needs is not defined in the
            furnished kernels, which is a missing frame kernel and not a
            baseline that has drifted; or if the index could not read the
            coverage of an object an image needs, which is a missing clock
            kernel and likewise not drift.
        OSError: if a candidate kernel cannot be furnished.
    """
    names = [entry.image_name for entry in entries]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            f'these images are named more than once, so their assignments cannot be told '
            f'apart: {", ".join(duplicates)}'
        )
    _require_no_furnished_ck()
    _require_frames_defined(entries)
    _require_coverage_readable(entries, index)
    pointed = [entry for entry in entries if entry.pointing is not None]
    baselines = _reproducing_baselines(pointed, index)
    # A winner that will write nothing suppresses nothing, so the pairing sees
    # only the winners whose baselines reproduced; every potential loser stays
    # in, since yielding does not depend on the loser's own baseline.
    losers = botsim_losers(
        [
            entry
            for entry in pointed
            if entry.camera != BOTSIM_WINNING_CAMERA or len(baselines[entry.image_name]) > 0
        ]
    )
    return tuple(_assignment_for(entry, losers, baselines) for entry in entries)


def _assignment_for(
    entry: ImageEntry, losers: frozenset[str], baselines: dict[str, tuple[CkFile, ...]]
) -> Assignment:
    """Decide one image's disposition from what the run has established.

    Parameters:
        entry: The image.
        losers: The images that yielded to a simultaneous exposure.
        baselines: The reproducing candidates of every image that was tested.

    Returns:
        The image's assignment.

    Raises:
        KeyError: if an eligible image was never tested, which cannot happen:
            the assignment step tests every image that carries a pointing,
            the yielded ones included.
    """
    if entry.ineligibility_reason is not None:
        return Assignment(entry=entry, baseline=None, omission_reason=entry.ineligibility_reason)
    if entry.image_name in losers:
        return Assignment(entry=entry, baseline=None, omission_reason=OmissionReason.BOTSIM_LOSER)
    candidates = baselines[entry.image_name]
    if len(candidates) == 0:
        return Assignment(
            entry=entry, baseline=None, omission_reason=OmissionReason.NO_REPRODUCING_BASELINE
        )
    return Assignment(entry=entry, baseline=candidates[0], omission_reason=None)


def _reproducing_baselines(
    entries: Sequence[ImageEntry], index: CkIndex
) -> dict[str, tuple[CkFile, ...]]:
    """Find, for each eligible image, every candidate that reproduces its baseline.

    Parameters:
        entries: The eligible images.  An image carrying no pointing is not
            tested and does not appear in the result.
        index: The pre-indexed C-kernels.

    Returns:
        One entry per tested image, holding its reproducing candidates in
        preference order, so the first is the one to use and an empty tuple
        means no candidate reproduced.

    Raises:
        OSError: if a candidate kernel cannot be furnished.
    """
    tested = [(entry, entry.pointing) for entry in entries if entry.pointing is not None]
    groups: dict[tuple[str, ...], list[tuple[ImageEntry, ImagePointing]]] = {}
    candidates_by_key: dict[tuple[str, ...], tuple[CkFile, ...]] = {}
    reproducing: dict[str, list[CkFile]] = {}
    for entry, pointing in tested:
        candidates = index.candidates(
            basenames=entry.kernel_basenames,
            ck_frame_id=pointing.ck_frame_id,
            et=pointing.midtime_et,
        )
        key = tuple(candidate.path.as_posix() for candidate in candidates)
        candidates_by_key[key] = candidates
        groups.setdefault(key, []).append((entry, pointing))
        reproducing[entry.image_name] = []
    for key, group in groups.items():
        for candidate in candidates_by_key[key]:
            with furnished(candidate.path):
                for entry, pointing in group:
                    if reproduces_baseline(pointing):
                        reproducing[entry.image_name].append(candidate)
    return {name: tuple(found) for name, found in reproducing.items()}


def _require_frames_defined(entries: Sequence[ImageEntry]) -> None:
    """Refuse to run when a frame the images need is not defined.

    A candidate that cannot answer a lookup is reported as not reproducing,
    which is how a baseline that has drifted since navigation is detected.  A
    frame kernel that was never furnished defeats the same lookup for every
    image alike, so it would empty a whole run and report it as drift.  It is
    checked once, before any candidate is furnished, against the frames the
    images themselves name.

    Parameters:
        entries: The images the run considered.  Those carrying no pointing
            name no frames and are skipped.

    Raises:
        ValueError: if a camera frame or a CK object frame is not defined in
            the furnished kernels.
    """
    needed = sorted(
        {
            (entry.pointing.camera_frame, entry.pointing.ck_frame_id)
            for entry in entries
            if entry.pointing is not None
        }
    )
    for camera_frame, ck_frame_id in needed:
        try:
            cspyce.namfrm(camera_frame)
        except LookupError as exc:
            raise ValueError(
                f'the camera frame {camera_frame} is not defined by the furnished kernels; the '
                f'frame kernel that defines it has to be furnished before an image can be '
                f'matched to the baseline it navigated against'
            ) from exc
        try:
            cspyce.frmnam(ck_frame_id)
        except LookupError as exc:
            raise ValueError(
                f'CK object {ck_frame_id} has no frame name in the furnished kernels; the frame '
                f'kernel that names it has to be furnished before an image can be matched to the '
                f'baseline it navigated against'
            ) from exc


def _require_coverage_readable(entries: Sequence[ImageEntry], index: CkIndex) -> None:
    """Refuse to run when the index could not read an object the images need.

    The index expresses each file's coverage in TDB, which needs the clock the
    object's time tags are encoded against; an object whose clock is not
    furnished is recorded as unreadable and offers no coverage at all.  Its
    candidates are then invisible to the coverage filter, so every image
    correcting that object would be reported as having no reproducing
    baseline -- the report that is meant to mean the holdings have changed
    since navigation ran.  The missing clock is named here instead, before any
    candidate is tried, exactly as a missing frame kernel is.

    Parameters:
        entries: The images the run considered.  Those carrying no pointing
            correct no object and are skipped.
        index: The pre-indexed C-kernels.

    Raises:
        ValueError: if any image corrects an object the index could not read.
    """
    unreadable = index.unreadable_objects
    if len(unreadable) == 0:
        return
    blocked = sorted(
        {
            entry.pointing.ck_frame_id
            for entry in entries
            if entry.pointing is not None and entry.pointing.ck_frame_id in unreadable
        }
    )
    if len(blocked) > 0:
        raise ValueError(
            f'the index could not read the coverage of CK object(s) {blocked}; the spacecraft '
            f'clock kernel that encodes their time tags has to be furnished before an image can '
            f'be matched to the baseline it navigated against'
        )


def _require_no_furnished_ck() -> None:
    """Refuse to run the reproduction test with a C-kernel already furnished.

    Raises:
        ValueError: if the kernel pool holds any C-kernel.  Reproduction asks
            the pool for an attitude with one candidate furnished, so anything
            else covering the same object and epoch would answer instead, and
            every image would be paired with whichever kernel happened to be
            loaded first.
    """
    furnished = int(cspyce.ktotal('CK'))
    if furnished != 0:
        raise ValueError(
            f'{furnished} C-kernel(s) are already furnished; the reproduction test needs a pool '
            f'holding only the supporting kernels, since any other C-kernel answers the same '
            f'lookups as the candidate under test'
        )


def group_for_output(assignments: Sequence[Assignment]) -> tuple[OutputGroup, ...]:
    """Group assigned images by the corrected file that will carry them.

    Each corrected file mirrors exactly one original, so its size stays
    proportional to that original's and regeneration is per original file.  An
    original that no image reproduces yields no group and therefore no file:
    SPICE refuses to close a C-kernel holding no segments, and an empty
    corrected kernel would claim a correction it does not carry.

    Parameters:
        assignments: The assignments of a run, in any order.  Those with no
            baseline are skipped.

    Returns:
        One group per corrected file, ordered by name, each holding its
        assignments in the order given.

    Raises:
        ValueError: if two different originals would be written to the same
            corrected name, which happens when two directories hold the same
            basename and different images reproduce each.  One would overwrite
            the other.
    """
    grouped: dict[str, list[Assignment]] = {}
    baselines: dict[str, CkFile] = {}
    for assignment in assignments:
        baseline = assignment.baseline
        if baseline is None:
            continue
        name = output_basename(baseline.basename)
        if name in baselines and baselines[name].path != baseline.path:
            raise ValueError(
                f'{baselines[name].path} and {baseline.path} would both be corrected to {name!r}'
            )
        baselines[name] = baseline
        grouped.setdefault(name, []).append(assignment)
    return tuple(
        OutputGroup(baseline=baselines[name], name=name, assignments=tuple(grouped[name]))
        for name in sorted(grouped)
    )
