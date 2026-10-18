"""Check a navigation pass frame by frame against an independent answer.

For every frame of an observation this reports the angle between the attitude
the pass recorded and the attitude something else navigated the same frame to,
expressed in pixels of the camera that took it, beside the quantities that
might have predicted a bad answer: the reported sigma, the spread between the
techniques that formed the answer, how many star features the models emitted,
and how many features the most confident technique consumed.

The point of reporting those together is that a run's own confidence is a
statement about its fit and not about its answer.  A frame fitted to one star
reports a tight sigma whatever it is pointing at, so the only way to learn
which of a pass's confident answers are wrong is to hold them against an answer
the pass did not produce.

A C-matrix is an absolute attitude, so any difference between two of them is a
difference in where the two answers say the camera pointed.  A frame is judged
on that difference as measured, with nothing subtracted: within ``AGREE_PX`` it
agrees, beyond ``WRONG_PX`` it is wrong, and between the two it is marginal.

A difference shared by every frame is still worth measuring on its own, because
it is what a coordinate defect inside one of the two pipelines looks like: such
a pipeline places every answer it writes a fixed fraction of a pixel away from
where the camera pointed.  The median signed difference over the agreeing frames
is therefore reported, and one larger than ``COMMON_OFFSET_TOLERANCE_PX`` in
either axis is called out as a defect in one of the two pipelines.  It is never
removed from the per-frame numbers, which would hide exactly that defect.

Run after ``source /seti/newnav/setup.sh``, from the repository root::

    python util/nav_verification/compare_pointing.py \\
        --nav-results-root /data/nav-run \\
        --observation ISS_006RI_LPHRLFMOV001_PRIME

With no ``--images`` it compares every record under the root that the bundle
also holds a frame for.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

import numpy as np  # noqa: E402
from filecache import FCPath  # noqa: E402
from util.nav_verification.boresight import (  # noqa: E402
    boresight_from_cmatrix,
    offset_in_camera_px,
    separation_px,
)
from util.nav_verification.bundle_cassini_fring import (  # noqa: E402
    DEFAULT_BUNDLE_DIR,
    bundle_image_names,
    plate_scale_urad,
    read_bundle_pointing,
    suppl_path,
)

from spindoctor.nav_records import (  # noqa: E402
    NavRecord,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)

# Within this many pixels a frame agrees with the bundle; beyond WRONG_PX it is wrong,
# and between the two it is marginal.
AGREE_PX = 1.0
WRONG_PX = 2.0

# How large the median signed difference over the agreeing frames may be, in either
# axis, before the report calls it out.  Two pipelines that place the camera correctly
# leave only the estimator's own noise in that median: the per-frame scatter measured
# on this bundle is about 0.1 px per axis, which puts the standard error of the median
# at 0.06 px over the fewest frames it is ever taken on and under 0.02 px over a few
# hundred.  A larger number is one pipeline writing every attitude rotated by the same
# amount.  Measured against the Cassini F ring bundle it was +0.494, +0.491 px, which
# was this pipeline's star navigation mixing two pixel coordinate systems (fixed).
COMMON_OFFSET_TOLERANCE_PX = 0.1

# Below this many agreeing frames a median is a few numbers rather than a measurement,
# so none is reported.
MIN_FRAMES_FOR_COMMON_OFFSET = 5


@dataclass
class FrameComparison:
    """One frame's navigation, and how far it is from the independent answer.

    Parameters:
        image: The image name without its version or processing suffix.
        status: What the record says became of the navigation.
        reason: What the record says about why, when it did not succeed.
        confidence: The confidence the pass reported.
        rank: The confidence tier the pass reported.
        sigma_px: The larger of the two reported sigmas, in pixels.
        offset_px: The offset the pass settled on.
        stars_emitted: How many star features the models put in front of the
            techniques.
        features_used: How many features the most confident technique consumed.
        top_technique: The most confident technique that was not called
            spurious.  It is not the answer: the answer is the ensemble's, and
            on most frames it differs from this technique's own offset.
        top_technique_to_answer_px: How far this technique's offset is from the
            answer the pass recorded, which says how much the two preceding
            fields describe the frame that was measured.
        techniques: Every technique that returned a result, spurious or not.
        n_consensus: How many techniques' answers were combined into the final
            one.
        technique_spread_px: The largest distance between any two of those,
            which is zero when only one technique contributed.
        error_px: The angle to the independent answer, in pixels, or None when
            there is nothing to compare against.
        error_x_px: That difference along the camera's first axis, signed, or
            None when the independent answer lies behind the camera, where it
            has no place on the detector.
        error_y_px: The same along the camera's second axis.
        bundle_navigation_type: How the independent answer was itself
            navigated, which says how much to believe it.
        bundle_from_cmatrix: Whether that answer was read at full precision.
    """

    image: str
    status: str | None = None
    reason: str | None = None
    confidence: float | None = None
    rank: str | None = None
    sigma_px: float | None = None
    offset_px: list[float] | None = None
    stars_emitted: int = 0
    features_used: int = 0
    top_technique: str | None = None
    top_technique_to_answer_px: float | None = None
    techniques: list[str] = field(default_factory=list)
    n_consensus: int = 0
    technique_spread_px: float = 0.0
    error_px: float | None = None
    error_x_px: float | None = None
    error_y_px: float | None = None
    bundle_navigation_type: str | None = None
    bundle_from_cmatrix: bool | None = None


def bare_image_name(record: NavRecord) -> str:
    """The image name a record is about, without version or suffix.

    Parameters:
        record: The record.

    Returns:
        The name, or the record's stub when the document names no image.
    """
    observation = record.metadata.get('observation') or {}
    named = str(observation.get('image_name') or '')
    if not named:
        return Path(str(record.stub)).name.split('_')[0]
    return named.split('_')[0]


def consensus_spread(navigation: dict[str, Any]) -> tuple[float, int]:
    """How far apart the techniques that formed the answer put the same frame.

    A technique the ensemble threw out as an outlier is not part of the answer,
    so counting it would report the disagreement the ensemble already settled
    rather than the uncertainty in what it settled on.  One such technique --
    reported at confidence zero and hundreds of pixels away -- is enough to put
    a spread of several hundred pixels on a frame that one technique answered
    cleanly.

    Parameters:
        navigation: The record's navigation result.

    Returns:
        The largest distance in pixels between any two contributing offsets,
        and how many techniques contributed.
    """
    excluded = set(navigation.get('excluded_from_consensus') or [])
    offsets = [
        t['offset_px']
        for t in navigation.get('per_technique', [])
        if not t.get('spurious') and t.get('offset_px') and t.get('technique_name') not in excluded
    ]
    pairs = [
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(offsets)
        for b in offsets[i + 1 :]
    ]
    return max([*pairs, 0.0]), len(offsets)


def compare_record(
    record: NavRecord,
    *,
    observation_id: str,
    bundle_dir: str | Path | FCPath,
) -> FrameComparison:
    """Compare one record against the bundle's answer for the same frame.

    Parameters:
        record: The record to compare.
        observation_id: The observation the frame belongs to.
        bundle_dir: The bundle's reprojected-image collection.

    Returns:
        The comparison, with ``error_px`` left None when the bundle holds no
        answer for this frame or the record carries no attitude.
    """
    image = bare_image_name(record)
    navigation = record.metadata.get('navigation_result') or {}
    spread, contributing = consensus_spread(navigation)
    excluded = set(navigation.get('excluded_from_consensus') or [])
    top = max(
        (
            t
            for t in navigation.get('per_technique', [])
            if not t.get('spurious') and t.get('technique_name') not in excluded
        ),
        key=lambda t: t.get('confidence') or 0.0,
        default=None,
    )
    sigmas = navigation.get('sigma_px')
    answer = navigation.get('offset_px')
    row = FrameComparison(
        image=image,
        status=record.metadata.get('status'),
        reason=navigation.get('status_reason'),
        confidence=navigation.get('confidence'),
        rank=navigation.get('confidence_rank'),
        sigma_px=max(sigmas) if sigmas else None,
        offset_px=answer,
        stars_emitted=int((navigation.get('feature_count_by_type') or {}).get('STAR', 0)),
        features_used=len(top.get('feature_ids') or []) if top else 0,
        top_technique=top.get('technique_name') if top else None,
        techniques=list(navigation.get('techniques_used') or []),
        n_consensus=contributing,
        technique_spread_px=round(spread, 3),
    )
    if top is not None and top.get('offset_px') and answer:
        row.top_technique_to_answer_px = round(
            math.hypot(top['offset_px'][0] - answer[0], top['offset_px'][1] - answer[1]), 3
        )

    path = suppl_path(image, observation_id=observation_id, bundle_dir=bundle_dir)
    try:
        independent = read_bundle_pointing(path)
    except FileNotFoundError:
        return row
    if independent is None:
        return row
    row.bundle_navigation_type = independent.navigation_type
    row.bundle_from_cmatrix = independent.from_cmatrix
    cmatrix = (navigation.get('pointing') or {}).get('cmatrix')
    if cmatrix:
        observation = record.metadata['observation']
        scale = plate_scale_urad(observation['camera'], int(observation['image_shape'][1]))
        row.error_px = separation_px(
            boresight_from_cmatrix(cmatrix),
            independent.boresight,
            plate_scale_urad=scale,
        )
        # An answer behind the camera has no place on the detector, so it gets no
        # axis values; its separation alone still marks the frame wrong.
        with contextlib.suppress(ValueError):
            row.error_x_px, row.error_y_px = offset_in_camera_px(
                cmatrix, independent.boresight, plate_scale_urad=scale
            )
    return row


def compare(
    nav_results_root: str | Path | FCPath,
    *,
    observation_id: str,
    bundle_dir: str | Path | FCPath,
    images: list[str] | None,
) -> tuple[list[FrameComparison], list[str], list[str]]:
    """Compare every selected record under a results root.

    Parameters:
        nav_results_root: The results root holding the pass to check.
        observation_id: The observation the frames belong to.
        bundle_dir: The bundle's reprojected-image collection.
        images: The image names to keep, or None to keep every record under the
            root that the bundle also holds a frame for.  An empty list keeps
            nothing.  A results root that holds more than one observation is
            the ordinary case, and counting another observation's frames as
            this one's failures would make every proportion in the report
            wrong, so the default narrows to what there is something to compare
            against.

    Returns:
        One comparison per image exactly one record was found for, sorted by
        image; the names of images more than one record was found for; and the
        names of files that were not records.  An image with more than one
        record is excluded rather than compared: the records would be checked
        against a single bundle answer, and since the stream promises no order
        there is no basis for preferring either, so keeping one would make
        every number in the report depend on the order of the walk.
    """
    if images is None:
        wanted = set(bundle_image_names(observation_id=observation_id, bundle_dir=bundle_dir))
    else:
        wanted = set(images)
    compared: dict[str, FrameComparison] = {}
    duplicated: set[str] = set()
    unreadable: list[str] = []
    with TreeRecordSource([nav_results_root]) as source:
        for found in source.records(Selection()):
            if isinstance(found, UnreadableFile):
                unreadable.append(f'{found.stub}: {found.reason}')
                continue
            image = bare_image_name(found)
            if image not in wanted:
                continue
            if image in compared or image in duplicated:
                duplicated.add(image)
                compared.pop(image, None)
                continue
            compared[image] = compare_record(
                found, observation_id=observation_id, bundle_dir=bundle_dir
            )
    return [compared[image] for image in sorted(compared)], sorted(duplicated), unreadable


def measure_common_offset(
    rows: list[FrameComparison], *, agree_px: float = AGREE_PX
) -> tuple[float, float] | None:
    """Measure the signed difference the agreeing frames share.

    The median over the frames that already agree, so that a handful of badly
    navigated frames cannot move it.  It is a measurement only: nothing is taken
    out of any frame's difference.

    Parameters:
        rows: The comparisons.
        agree_px: How close a frame must be to help measure it.

    Returns:
        The median difference along the camera's two axes, or None when too few
        frames agree for a median to mean anything.

    Raises:
        ValueError: If the tolerance is not a distance, since a negative or
            NaN one is a tolerance no frame is inside and every statistic
            drawn from it would be nonsense rather than empty.
    """
    if not math.isfinite(agree_px) or agree_px < 0.0:
        raise ValueError(f'a tolerance is a distance in pixels, not {agree_px}')
    agreeing = [
        r
        for r in rows
        if r.error_px is not None
        and r.error_x_px is not None
        and r.error_y_px is not None
        and r.error_px <= agree_px
    ]
    if len(agreeing) < MIN_FRAMES_FOR_COMMON_OFFSET:
        return None
    return (
        float(np.median([r.error_x_px for r in agreeing])),
        float(np.median([r.error_y_px for r in agreeing])),
    )


def _px(value: float | None, width: int, places: int) -> str:
    """Format a number that may be missing, without printing a missing one as zero.

    Parameters:
        value: The number, or None.
        width: The column width.
        places: How many decimal places.

    Returns:
        The formatted number, or dashes.
    """
    return f'{value:{width}.{places}f}' if value is not None else '-' * width


def report(
    rows: list[FrameComparison],
    *,
    common: tuple[float, float] | None,
    agree_px: float = AGREE_PX,
    wrong_px: float = WRONG_PX,
    bundle_frames: int | None = None,
    duplicated: list[str] | None = None,
    unreadable: list[str] | None = None,
) -> None:
    """Print what the comparison found.

    Parameters:
        rows: The comparisons.
        common: The signed difference the agreeing frames share, as
            ``measure_common_offset`` measured it, or None when too few frames
            agreed to measure one.  Either way it is reported: one above
            ``COMMON_OFFSET_TOLERANCE_PX`` in an axis is called out the way a
            wrong frame is, and one that could not be measured is said to be
            unmeasured rather than left to read as zero.
        agree_px: Within this many pixels a frame agrees.
        wrong_px: Beyond this many pixels a frame is wrong; between the two it
            is marginal.

    Raises:
        ValueError: If either bound is not a finite distance, or agreement is
            looser than wrong, since a frame could then be both.
        bundle_frames: How many frames the bundle holds for this observation,
            so that frames the pass never attempted are counted rather than
            silently improving the rate.
        duplicated: Images excluded because more than one record named them.
            They are counted as having a record, which they do; what they have
            not got is one record to compare.
        unreadable: Files under the root that were not records at all.
    """
    if not (math.isfinite(agree_px) and math.isfinite(wrong_px) and 0.0 <= agree_px <= wrong_px):
        raise ValueError(
            f'the bounds are distances with agreement inside wrong, not {agree_px} and {wrong_px}'
        )
    navigated = [r for r in rows if r.status == 'success']
    failed = [r for r in rows if r.status != 'success']
    compared = [r for r in rows if r.error_px is not None]
    off = sorted(
        (r for r in compared if (r.error_px or 0.0) > agree_px),
        key=lambda r: -(r.error_px or 0.0),
    )
    errors = np.array([r.error_px for r in compared], dtype=float)

    excluded = duplicated or []
    if bundle_frames is not None:
        with_record = len(rows) + len(excluded)
        print(f'bundle frames     {bundle_frames}')
        print(f'  with a record   {with_record}')
        print(f'  no record       {bundle_frames - with_record}')
    print(f'records compared  {len(rows)}')
    print(f'  navigated       {len(navigated)} ({100 * len(navigated) / max(1, len(rows)):.1f}%)')
    print(f'  failed          {len(failed)}')
    if excluded:
        print(
            f'  more than one   {len(excluded)} (excluded, no basis for either: '
            f'{", ".join(excluded[:3])}{" ..." if len(excluded) > 3 else ""})'
        )
    if unreadable:
        print(f'  not a record    {len(unreadable)} (outside this observation as well as in it)')

    if errors.size:
        rounded = sum(1 for r in compared if r.bundle_from_cmatrix is False)
        print(
            f'  comparable      {len(compared)}'
            + (f' ({rounded} against a rounded boresight)' if rounded else '')
        )

    if common is not None and errors.size:
        print(f'\nshared difference {common[0]:+.3f}, {common[1]:+.3f} px along the camera axes')
        print('                  the median signed difference over the agreeing frames; it is')
        print('                  included in every number below, never subtracted')
        if max(abs(common[0]), abs(common[1])) > COMMON_OFFSET_TOLERANCE_PX:
            print(f'  WRONG (over {COMMON_OFFSET_TOLERANCE_PX} px in an axis): one of the two')
            print('                  pipelines writes every attitude rotated by this amount')
    elif errors.size:
        print(
            f'\nshared difference not measured: fewer than {MIN_FRAMES_FOR_COMMON_OFFSET} '
            'frames agree'
        )

    if errors.size:
        marginal = sum(1 for r in off if (r.error_px or 0.0) <= wrong_px)
        print('\ndifference from the independent answer:')
        print(
            f'  median {np.median(errors):.2f} px, 90th {np.percentile(errors, 90):.2f}, '
            f'max {errors.max():.2f}'
        )
        for threshold in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0):
            within = int((errors <= threshold).sum())
            print(f'    within {threshold:5.2f} px: {within:4d}/{len(errors)}')
        print(f'  agree (within {agree_px} px): {len(compared) - len(off)}')
        print(f'  marginal ({agree_px}-{wrong_px} px): {marginal}')
        print(f'  WRONG (over {wrong_px} px): {len(off) - marginal}')

    if failed:
        print('\nfailure reasons:')
        for reason, count in Counter(r.reason for r in failed).most_common():
            print(f'  {count:5d}  {reason}')

    if off:
        print(f'\nevery frame more than {agree_px} px off:')
        print(
            f'  {"image":16s} {"err px":>7s} {"sigma":>6s} {"spread":>7s} {"conf":>6s} '
            f'{"stars":>5s} {"used":>4s} {"n_cons":>6s}  '
            f'most confident technique / bundle navigation type'
        )
        for r in off:
            print(
                f'  {r.image:16s} {_px(r.error_px, 7, 2)} {_px(r.sigma_px, 6, 2)} '
                f'{r.technique_spread_px:7.2f} {_px(r.confidence, 6, 3)} '
                f'{r.stars_emitted:5d} {r.features_used:4d} {r.n_consensus:6d}  '
                f'{r.top_technique} / {r.bundle_navigation_type}'
            )


def main() -> None:
    """Compare one observation's navigation against the bundle and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--nav-results-root', required=True, help='the results root holding the pass to check'
    )
    parser.add_argument('--observation', required=True, help='the observation the frames belong to')
    parser.add_argument(
        '--images',
        type=Path,
        default=None,
        help='a local file of whitespace-separated image names to keep; '
        'the default keeps every record the bundle also holds '
        'a frame for',
    )
    parser.add_argument(
        '--bundle-dir',
        type=FCPath,
        default=DEFAULT_BUNDLE_DIR,
        help='the bundle collection to compare against, local or remote '
        f'(default {DEFAULT_BUNDLE_DIR})',
    )
    parser.add_argument(
        '--agree-px',
        type=float,
        default=AGREE_PX,
        help=f'within this many pixels a frame agrees (default {AGREE_PX})',
    )
    parser.add_argument(
        '--wrong-px',
        type=float,
        default=WRONG_PX,
        help=f'beyond this many pixels a frame is wrong; between the two it is '
        f'marginal (default {WRONG_PX})',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='a local file to write the per-frame detail to, as JSON',
    )
    args = parser.parse_args()

    images = None
    bundle_frames = None
    if args.images is not None:
        images = [n.split('_')[0] for n in args.images.read_text().split()]
    else:
        # One listing is both the default selection and the report's denominator.
        images = bundle_image_names(observation_id=args.observation, bundle_dir=args.bundle_dir)
        bundle_frames = len(images)

    rows, duplicated, unreadable = compare(
        args.nav_results_root,
        observation_id=args.observation,
        bundle_dir=args.bundle_dir,
        images=images,
    )
    common = measure_common_offset(rows, agree_px=args.agree_px)

    print(f'=== {args.observation} ===')
    report(
        rows,
        common=common,
        agree_px=args.agree_px,
        wrong_px=args.wrong_px,
        bundle_frames=bundle_frames,
        duplicated=duplicated,
        unreadable=unreadable,
    )

    if args.output is not None:
        args.output.write_text(json.dumps([vars(r) for r in rows], indent=1))
        print(f'\nper-frame detail written to {args.output}')


if __name__ == '__main__':
    main()
