"""The turning of the Cassini spacecraft during an ISS exposure.

The star model draws each star's streak from the camera's turning during the exposure.
The reconstructed C-kernels that give Cassini ISS its pointing describe that turning
poorly: they hold one attitude every 4 s, and on 31% of the intervals around the F ring
frames measured the attitude repeats the one before exactly while the next interval
carries twice the commanded motion.  An exposure shorter than an interval therefore
sees either no motion or about double the true motion.  Over 1,796 Cassini NAC frames
whose stars' streaks were measured, streaks predicted from those kernels are a median
0.58 times the measured length and 20.8 degrees off its direction, and 15% lie within
1 px of the measured streak.

The as-flown predicted C-kernels hold one attitude every 6 s at the constant rate the
spacecraft was commanded to turn, and over the same frames the streaks they predict are
a median 0.96 times the measured length, 2.9 degrees off, and 99% within 1 px.  The
motion is therefore read from those kernels, without furnishing them, so that the
absolute pointing stays the reconstructed kernels' everywhere; only the rotation between
the start, middle, and end of the exposure is taken from the as-flown attitude.

Where no as-flown kernel gives an attitude at all three instants, the motion is the
reconstructed kernels' mean rate over a window centered on the exposure, which the same
frames place a median 0.42 px from the measured streak, 96% within 1 px, with a 16 s
window.  A longer window recovers the streak on steady scans but averages across turns:
on the 13 frames within 30 s of a turn a 60 s window is off by up to 10 px.  An exposure
at least as long as the window spans several of the reconstructed kernels' records, and
the observation's own pointing at its start and end describes its turning; so does that
pointing where the reconstructed kernels cannot place the window.  Only the as-flown
motion vouches for its predicted smear (``max_streak_to_predicted``).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cspyce
import julian
import numpy as np
import spicedb

from spindoctor.config import IMAGE_LOGGER
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID
from spindoctor.support.ck_type3 import ck_type3_attitudes
from spindoctor.support.exposure_motion import (
    ExposureMotion,
    motion_from_attitudes,
    motion_from_rate,
)
from spindoctor.support.types import NDArrayFloatType

from .obs_snapshot import ObsSnapshot

__all__ = [
    'EXPOSURE_MOTION_SECTION',
    'PREDICTED_SOURCE_PREFIX',
    'RECONSTRUCTED_RATE_SOURCE',
    'cassini_exposure_motion',
    'predicted_ck_paths',
]

EXPOSURE_MOTION_SECTION = 'cassini_iss_exposure_motion'
"""The configuration section naming the kernels the exposure motion is read from."""

PREDICTED_SOURCE_PREFIX = 'predicted C-kernel '
"""What the motion's source begins with when an as-flown kernel supplied it."""

RECONSTRUCTED_RATE_SOURCE = 'reconstructed C-kernels mean rate over {window:g} s'
"""The motion's source when the reconstructed kernels' mean rate supplied it."""

# What cspyce raises, through oops, when the furnished kernels cannot place a frame at an
# epoch within seconds of an exposure: a missing frame or clock as a LookupError, missing
# C-kernel data as an OSError, and an unconnected frame chain as a RuntimeError.  cspyce
# also raises ValueError for an epoch outside the clock's range, which cannot happen this
# close to an exposure, so a ValueError here is a defect and is left to propagate.
_SPICE_FAILURES = (LookupError, OSError, RuntimeError)


def predicted_ck_paths(
    epoch_et: float, *, spacecraft_id: int, kernel_name: str, kernel_version: str
) -> list[Path]:
    """Return local copies of the predicted C-kernels the SPICE database lists at an epoch.

    The kernels are selected from the database oops furnishes its kernels from,
    restricted to one release of the set, and retrieved to the local cache; none is
    furnished.  A listed kernel that cannot be retrieved is left out and logged.

    Parameters:
        epoch_et: The epoch, TDB seconds past J2000.
        spacecraft_id: SPICE id of the spacecraft the database lists the kernels under.
        kernel_name: The database's name of the kernel set, such as
            ``CAS-CK-PREDICTED``.
        kernel_version: The release of the set to use, such as ``V02``.

    Returns:
        The kernels' local paths in the database's load order, later ones taking
        precedence; empty when the database lists none.
    """
    was_open = spicedb.db_is_open()
    if not was_open:
        spicedb.open_db()
    tai = julian.tai_from_tdb(epoch_et)
    try:
        kernels = spicedb.select_ck(spacecraft_id, name=kernel_name, time=(tai, tai))
    except ValueError:
        # spicedb signals an empty selection by raising rather than by returning nothing;
        # outside the kernel set's span that is the expected answer.
        kernels = []
    finally:
        if not was_open:
            spicedb.close_db()
    if len(kernels) == 0:
        return []
    prefix = spicedb.get_spice_filecache_prefix()
    paths: list[Path] = []
    for kernel in kernels:
        if kernel.kernel_version != kernel_version:
            continue
        for filespec in str(kernel.filespec).split(','):
            local = prefix.retrieve(filespec, exception_on_fail=False)
            if isinstance(local, Path):
                paths.append(local)
            else:
                IMAGE_LOGGER.warning(
                    'Predicted C-kernel %s cannot be retrieved: %s', filespec, local
                )
    return paths


def cassini_exposure_motion(
    obs: ObsSnapshot,
    settings: Mapping[str, Any],
    *,
    kernel_paths: Sequence[Path] | None = None,
) -> ExposureMotion | None:
    """Return the turning of the Cassini spacecraft during one ISS exposure.

    The attitude of ``settings['ck_object_id']`` at the start, middle, and end of the
    exposure is read from the as-flown predicted C-kernels, later kernels taking
    precedence as they do when furnished, and the motion is the rotation between those
    attitudes; it vouches for its predicted smear up to
    ``settings['predicted_max_streak_to_predicted']`` and names the kernel that gave the
    middle attitude.  When the kernels leave any of the three instants without an
    attitude, the motion is the mean rate of the observation's own pointing over the
    window of ``2 * settings['reconstructed_half_window_s']`` seconds centered on
    mid-exposure, which vouches for nothing.  An exposure at least that long, or one
    whose pointing cannot be evaluated across the window, has no motion to supply.

    Parameters:
        obs: The observation.
        settings: The ``cassini_iss_exposure_motion`` configuration section:
            ``spacecraft_id``, ``ck_object_id``, ``predicted_ck_name``,
            ``predicted_ck_version``, ``predicted_max_streak_to_predicted``, and
            ``reconstructed_half_window_s``.
        kernel_paths: The predicted C-kernels to read for every instant, in load order;
            None selects each instant's with :func:`predicted_ck_paths` from
            ``settings['spacecraft_id']``, ``settings['predicted_ck_name']``, and
            ``settings['predicted_ck_version']``.

    Returns:
        The motion, or None to use the observation's own pointing at the exposure's
        start and end.
    """
    start_et, end_et = float(obs.time[0]), float(obs.time[1])
    mid_et = float(obs.midtime)
    epochs = (start_et, mid_et, end_et)
    ck_object_id = int(settings['ck_object_id'])
    if kernel_paths is None:
        # One selection per instant: the database's selection over a span keeps only the
        # kernels covering its start, and each instant is answered by its own kernels'
        # priority, as SPICE would answer it.  An exposure across two kernels is read from
        # both, as furnished kernels would read it, rather than falling back; each V02
        # as-flown file covers one contiguous interval, and none of the 20,584 F ring
        # frames of the 2026-10-22 run lies within 60 s of an interval's end.
        per_instant = [
            predicted_ck_paths(
                et,
                spacecraft_id=int(settings['spacecraft_id']),
                kernel_name=str(settings['predicted_ck_name']),
                kernel_version=str(settings['predicted_ck_version']),
            )
            for et in epochs
        ]
    else:
        per_instant = [list(kernel_paths)] * len(epochs)
    sclk_id = CK_OBJECT_SCLK_ID[ck_object_id]
    attitudes: list[NDArrayFloatType | None] = [None, None, None]
    suppliers: list[Path | None] = [None, None, None]
    for index, (et, paths) in enumerate(zip(epochs, per_instant, strict=True)):
        tick = float(cspyce.sce2c(sclk_id, et))
        for path in reversed(paths):
            (attitude,) = ck_type3_attitudes(path, ck_object_id, [tick])
            if attitude is not None:
                attitudes[index] = attitude
                suppliers[index] = path
                break
    start, middle, end = attitudes
    mid_supplier = suppliers[1]
    if start is not None and middle is not None and end is not None and mid_supplier is not None:
        IMAGE_LOGGER.info('Exposure motion from the predicted C-kernel %s', mid_supplier.name)
        return motion_from_attitudes(
            start,
            middle,
            end,
            source=PREDICTED_SOURCE_PREFIX + mid_supplier.name,
            max_streak_to_predicted=float(settings['predicted_max_streak_to_predicted']),
        )
    return _reconstructed_rate_motion(obs, float(settings['reconstructed_half_window_s']))


def _reconstructed_rate_motion(obs: ObsSnapshot, half_window_s: float) -> ExposureMotion | None:
    """Return the motion at the observation's mean rate over a window about mid-exposure.

    Parameters:
        obs: The observation.
        half_window_s: Half the window's length in seconds.

    Returns:
        The motion, or None when the exposure is at least as long as the window or the
        observation's pointing cannot be evaluated at the window's ends.
    """
    mid_et = float(obs.midtime)
    exposure_s = float(obs.time[1]) - float(obs.time[0])
    window_s = 2.0 * half_window_s
    if exposure_s >= window_s:
        IMAGE_LOGGER.info(
            'No predicted C-kernel covers the exposure, which lasts %.1f s; exposure '
            'motion from the pointing at its ends',
            exposure_s,
        )
        return None
    try:
        before, middle, after = (
            np.asarray(obs.get_spice_cmatrix(time=et).vals, dtype=np.float64)
            for et in (mid_et - half_window_s, mid_et, mid_et + half_window_s)
        )
    except _SPICE_FAILURES as exc:
        IMAGE_LOGGER.info(
            'No predicted C-kernel covers the exposure and the pointing has no attitude '
            '%.1f s either side of it (%s); exposure motion from the pointing at its ends',
            half_window_s,
            exc,
        )
        return None
    IMAGE_LOGGER.info(
        'No predicted C-kernel covers the exposure; exposure motion from the '
        'reconstructed C-kernels mean rate over %.1f s',
        window_s,
    )
    return motion_from_rate(
        before,
        middle,
        after,
        window_s=window_s,
        exposure_s=exposure_s,
        source=RECONSTRUCTED_RATE_SOURCE.format(window=window_s),
    )
