"""The three processes of the corrected-pointing round trip.

Navigation measures a pointing correction; the C-kernel writer turns it into a
kernel; a consumer furnishes that kernel and gets corrected geometry.  This
module runs that loop end to end on one real image, as three steps meant to be
run in three separate processes:

``navigate``
    Navigate the image the way the pipeline does, writing the per-image
    metadata the kernel writer reads, plus the list of kernels oops furnished
    while doing it.
``generate``
    Pair the image with the baseline kernel it navigated against and write the
    corrected kernel, recording the corrected camera attitude the written
    segment claims at the exposure start, midtime and stop.
``renavigate``
    Load the image again in a fresh process, furnish the corrected kernel, read
    back the camera attitude at those three epochs, and navigate again.

Three processes rather than one, because oops caches frames and manages its own
kernel pool: a ``furnsh`` in the middle of a process that has already navigated
is not guaranteed to take effect, and a round trip that quietly measured the
uncorrected pointing twice would pass.  Each step writes its findings to a JSON
file in a shared working directory, and the test that drives the three reads
those files and does the deciding.  Nothing here asserts: a step that cannot do
its job raises, and everything else is a measurement for the caller to judge.

Two facts about the re-navigation step are worth stating, because they are what
make it a real test rather than a formality:

- The corrected kernel is furnished **after** the host's ``from_file`` returns,
  since that is when oops has finished furnishing the originals, and SPICE
  gives precedence to the C-kernel furnished last.  Furnishing it earlier would
  put it underneath the originals, where it would be silently ignored.
- A host that freezes its observation frame while ``from_file`` runs -- which is
  what Voyager ISS does, building a fixed attitude out of one tolerance-snapped
  pointing lookup -- cannot see a kernel furnished after that call returns at
  all.  For those hosts the image is loaded a second time, with the correction
  already furnished, and the second observation is the one navigated.  The
  attitude readback then reports what the whole pool answers, so a correction
  that failed to take effect is visible either way.
"""

import argparse
import contextlib
import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import cspyce
import numpy as np
import oops
from filecache import FCPath

from spindoctor.cli.ck.assignment import assign_images, output_basename
from spindoctor.cli.ck.images import ImageEntry
from spindoctor.cli.ck.index import build_ck_index
from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.segment import (
    CkSegment,
    build_segment,
    resolve_sclk_id,
    write_segment,
)
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator, build_metadata_dict
from spindoctor.navigate_image_files import navigate_image_files
from spindoctor.obs import (
    ObsCassiniISS,
    ObsGalileoSSI,
    ObsNewHorizonsLORRI,
    ObsSnapshotInst,
    ObsVoyagerISS,
)
from spindoctor.spice_ids import FROZEN_ATTITUDE_CK_IDS
from spindoctor.support.types import NDArrayFloatType
from tests.integration.checkout_guard import require_checkout_package
from tests.integration.sidecar import LibraryRoot, Sidecar, load_sidecar

# The three steps, named as the driver names them on the command line.
NAVIGATE = 'navigate'
GENERATE = 'generate'
RENAVIGATE = 'renavigate'

# The epochs a segment always carries a record at, and the only ones this round
# trip makes any claim about.  An exposure longer than 10 s carries more, at a
# 1 s cadence, which are reproduced just as exactly but are deliberately not
# asserted: they exist only on long exposures, so asserting them would make the
# validation's coverage depend on the cohort's exposure lengths.
RECORD_LABELS = ('start', 'midtime', 'stop')

# The tolerance oops makes its snapped Voyager pointing lookup at, in spacecraft
# clock ticks: a fixed part plus a term in the exposure.  Restated here rather
# than imported because this module reproduces the *host's* lookup, and the
# writer's copy of the same numbers exists to reproduce a *recorded* attitude;
# tying the two together would let a change to one silently redefine the other.
_SNAPPED_TOL_TICKS = 800.0
_SNAPPED_TOL_EXPOSURE_DIVISOR = 48.0

# The mission strings the image library uses, and the host class each names.
_MISSION_TO_OBS_CLASS: dict[str, type[ObsSnapshotInst]] = {
    'COISS': ObsCassiniISS,
    'VGISS': ObsVoyagerISS,
    'GOSSI': ObsGalileoSSI,
    'NHLORRI': ObsNewHorizonsLORRI,
}

# A corrected kernel is written with no comment area; the comment area is
# Phase E's business, and a segment reads back the same with or without one.
_COMMENT_CHARS = 0

# The pixel step a scale is measured over, one pixel along each axis from the
# boresight.
_SCALE_STEP_PX = 1.0


def sidecar_for(image_id: str) -> Sidecar:
    """Return the image library's sidecar for one image.

    Parameters:
        image_id: The library's id for the image, which is also its sidecar's
            filename stem.

    Returns:
        The sidecar.

    Raises:
        ValueError: if no sidecar in the library carries that id, naming what
            the library does hold so a typo is obvious.
    """
    for path in LibraryRoot().discover_sidecar_paths():
        if path.stem == image_id:
            return load_sidecar(path)
    raise ValueError(f'no image library sidecar is named {image_id!r}')


def holdings_url(sidecar: Sidecar) -> FCPath:
    """Resolve one sidecar's image URL against the holdings root.

    Parameters:
        sidecar: The image's sidecar.

    Returns:
        The image's URL, with a ``pds3://`` scheme resolved against
        ``PDS3_HOLDINGS_DIR`` and anything else passed through.

    Raises:
        KeyError: if the URL needs the holdings root and it is not set.
    """
    url = sidecar.image_url
    if not url.startswith('pds3://'):
        return FCPath(url)
    root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
    return FCPath(f'{root}/{url[len("pds3://") :]}')


def step_path(work: Path, image_id: str, step: str) -> Path:
    """Return the file one step writes its findings to.

    Parameters:
        work: The working directory the three steps share.
        image_id: The library's id for the image.
        step: Name of the step.

    Returns:
        The path.
    """
    return work / f'{image_id}_{step}.json'


def metadata_path(work: Path, image_id: str) -> Path:
    """Return the per-image navigation metadata the navigate step writes.

    Parameters:
        work: The working directory the three steps share.
        image_id: The library's id for the image.

    Returns:
        The path, which is where ``navigate_image_files`` puts it.
    """
    return work / f'{image_id}_metadata.json'


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON document written by an earlier step.

    Parameters:
        path: The file to read.

    Returns:
        The document.

    Raises:
        ValueError: if the file does not exist, which means the step that
            writes it did not run or did not get that far, or if it does not
            hold a JSON object.
    """
    if not path.is_file():
        raise ValueError(f'{path} does not exist; the step that writes it did not complete')
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f'{path} holds a {type(document).__name__}, not a JSON object')
    return cast(dict[str, Any], document)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    """Write one step's findings.

    Parameters:
        path: The file to write.
        document: What to write.
    """
    path.write_text(json.dumps(document, indent=2, sort_keys=True))


def furnished_kernels() -> list[dict[str, str]]:
    """Report every kernel currently in the SPICE pool, in load order.

    A navigation run's provenance records sorted basenames only, which is not
    enough to furnish the same pool again: it holds no directories and no
    order.  This is read straight from SPICE after a real load, so the
    generate step can furnish exactly the supporting kernels the navigation
    used and index exactly the directories its C-kernels came from.

    Returns:
        One entry per kernel, each holding its ``path`` and the ``kind``
        SPICE reports (``CK``, ``SPK``, ``TEXT`` and so on), in the order they
        were furnished.
    """
    kernels: list[dict[str, str]] = []
    for at in range(int(cspyce.ktotal('ALL'))):
        data = cspyce.kdata(at, 'ALL')
        kernels.append({'path': str(data[0]), 'kind': str(data[1])})
    return kernels


def pixel_scales(fov: Any) -> tuple[float, float]:
    """Measure one FOV's angular scale at the boresight, per axis.

    The scale converts an attitude residual into the pixel error it causes,
    which is the unit the round trip's tolerance is stated in.  It is measured
    on the FOV itself rather than read from a table, since a binned or
    subarrayed frame has a different scale from its instrument's nominal one.

    Parameters:
        fov: The observation's unmodified oops FOV.

    Returns:
        The angle subtended by one pixel at the boresight, in radians, along
        ``u`` and along ``v``.

    Raises:
        ValueError: if either measured scale is not finite and positive, which
            would make every angle-to-pixel conversion meaningless.
    """
    uv_los = fov.uv_los
    boresight = np.asarray(fov.los_from_xy(fov.xy_from_uv(uv_los)).unit().vals, dtype=np.float64)
    scales: list[float] = []
    for step in ((_SCALE_STEP_PX, 0.0), (0.0, _SCALE_STEP_PX)):
        stepped_uv = oops.Pair((uv_los.vals[0] + step[0], uv_los.vals[1] + step[1]))
        direction = fov.los_from_xy(fov.xy_from_uv(stepped_uv)).unit()
        stepped = np.asarray(direction.vals, dtype=np.float64)
        cross = float(np.linalg.norm(np.cross(boresight, stepped)))
        scales.append(float(np.arctan2(cross, float(np.dot(boresight, stepped)))))
    for axis, scale in zip(('u', 'v'), scales, strict=True):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f'the FOV scale along {axis} is {scale!r}, not a positive angle')
    return scales[0], scales[1]


def angle_to_pixels(angle_rad: float, scale_rad_px: float) -> float:
    """Convert an angular residual into the pixel error it causes.

    Parameters:
        angle_rad: The angle, in radians.
        scale_rad_px: The angle one pixel subtends, in radians.

    Returns:
        The angle expressed in pixels.

    Raises:
        ValueError: if the angle is not finite, or if the scale is not finite
            and positive.  A NaN divided by a scale is a NaN, which compares
            False against every tolerance and so reads as a pass.
    """
    if not math.isfinite(angle_rad):
        raise ValueError(f'the angle is {angle_rad!r}, not a finite number of radians')
    if not math.isfinite(scale_rad_px) or scale_rad_px <= 0.0:
        raise ValueError(f'the pixel scale is {scale_rad_px!r}, not a positive angle')
    return angle_rad / scale_rad_px


def record_index_for_tick(ticks: Sequence[float], tick: float) -> int:
    """Return which of a segment's records sits at one encoded clock time.

    The round trip claims the record epochs and nothing between them, so a
    readback has to be compared against the record that is actually there.
    The match is exact: a segment carries a record at the epoch or it does
    not, and comparing a readback against the nearest record instead would
    silently turn an assertion about a record into an assertion about an
    interpolation.

    Parameters:
        ticks: The segment's encoded SCLK time tags, in record order.
        tick: The encoded SCLK time to find.

    Returns:
        The index of the record at that time.

    Raises:
        ValueError: if no record sits exactly there, or if either the time or
            the tags hold a non-finite value, which no comparison would
            report.
    """
    if not math.isfinite(tick):
        raise ValueError(f'the encoded clock time is {tick!r}, not a finite tick')
    if len(ticks) == 0:
        raise ValueError('the segment holds no records')
    # Every tag is validated before any is compared, so a non-finite tag is
    # refused even when it sits after the record that would have matched.
    for at, value in enumerate(ticks):
        if not math.isfinite(value):
            raise ValueError(f'record {at} has a non-finite time tag: {value!r}')
    for at, value in enumerate(ticks):
        if value == tick:
            return at
    raise ValueError(f'no record sits at encoded clock time {tick!r}; the tags are {list(ticks)!r}')


def _as_matrix(values: Sequence[float]) -> NDArrayFloatType:
    """Return nine row-major floats as a 3x3 array.

    Parameters:
        values: The nine elements.

    Returns:
        The 3x3 array.

    Raises:
        ValueError: if there are not exactly nine of them.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (9,):
        raise ValueError(f'expected nine row-major floats; got shape {array.shape}')
    return array.reshape(3, 3)


def _flatten(matrix: NDArrayFloatType) -> list[float]:
    """Return a 3x3 rotation as nine row-major floats.

    Parameters:
        matrix: The rotation.

    Returns:
        Its elements, unrounded, since they must reproduce to a nanoradian.
    """
    return [float(value) for value in np.asarray(matrix, dtype=np.float64).reshape(9)]


def _snapped_tolerance(exposure_s: float) -> float:
    """Return the tolerance a frozen-attitude host makes its lookup at.

    Parameters:
        exposure_s: The exposure duration in seconds.

    Returns:
        The tolerance in spacecraft clock ticks.
    """
    return _SNAPPED_TOL_TICKS + exposure_s / _SNAPPED_TOL_EXPOSURE_DIVISOR


def camera_attitude_from_pool(pointing: ImagePointing, et: float) -> NDArrayFloatType:
    """Read the camera attitude the furnished kernels now give at one epoch.

    The lookup is the one the image's own host makes, so that what is measured
    is what a consumer of these kernels would see.  Most hosts evaluate the
    frame chain, which ``pxform`` reproduces.  A frozen-attitude host makes a
    tolerance-snapped pointing lookup on the CK object and composes the fixed
    rotation to the camera, and its camera frame cannot be evaluated by chain
    at all, so the same lookup is made here.

    Parameters:
        pointing: The image's recorded pointing, naming the frames and the
            exposure.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 J2000-to-camera rotation.

    Raises:
        OSError: if the pool holds no pointing there.
        RuntimeError: if the frame chain cannot be evaluated there.
    """
    if pointing.ck_frame_id not in FROZEN_ATTITUDE_CK_IDS:
        return np.asarray(cspyce.pxform('J2000', pointing.camera_frame, et), dtype=np.float64)
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    j2000_to_object, _clkout = cspyce.ckgp(
        pointing.ck_frame_id,
        float(cspyce.sce2t(sclk_id, et)),
        _snapped_tolerance(pointing.exposure_s),
        'J2000',
    )
    object_to_camera = _object_to_camera(pointing, et)
    attitude: NDArrayFloatType = object_to_camera @ np.asarray(j2000_to_object, dtype=np.float64)
    return attitude


def _object_to_camera(pointing: ImagePointing, et: float) -> NDArrayFloatType:
    """Return the fixed rotation from the CK object's frame to the camera frame.

    Parameters:
        pointing: The image's recorded pointing.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation.

    Raises:
        KeyError: if the CK object has no frame name in the furnished kernels.
    """
    ck_frame = str(cspyce.frmnam(pointing.ck_frame_id))
    return np.asarray(cspyce.pxform(ck_frame, pointing.camera_frame, et), dtype=np.float64)


def _claimed_camera_attitudes(
    pointing: ImagePointing, segment: CkSegment
) -> dict[str, dict[str, Any]]:
    """Return the camera attitude the written segment claims at each record.

    Taken from the segment's own quaternions rather than recomputed from the
    baseline and the correction, so that what the re-navigation is compared
    against is what was written rather than what was meant.

    Parameters:
        pointing: The image's recorded pointing.
        segment: The segment written for it.

    Returns:
        One entry per record epoch, each holding the epoch, the encoded clock
        time, the record index, and the claimed 3x3 camera attitude as nine
        row-major floats.

    Raises:
        ValueError: if the segment carries no record at one of the epochs.
    """
    sclk_id = resolve_sclk_id(pointing.ck_frame_id)
    epochs = (pointing.start_et, pointing.midtime_et, pointing.stop_et)
    claimed: dict[str, dict[str, Any]] = {}
    for label, et in zip(RECORD_LABELS, epochs, strict=True):
        tick = float(cspyce.sce2c(sclk_id, et))
        at = record_index_for_tick([float(value) for value in segment.sclkdp], tick)
        object_attitude = np.asarray(cspyce.q2m(segment.quats[at]), dtype=np.float64)
        claimed[label] = {
            'et': et,
            'tick': tick,
            'record_index': at,
            'cmatrix': _flatten(_object_to_camera(pointing, et) @ object_attitude),
        }
    return claimed


def step_navigate(image_id: str, work: Path) -> None:
    """Navigate one library image and record what the kernel writer will need.

    Parameters:
        image_id: The library's id for the image.
        work: The working directory the three steps share.
    """
    sidecar = sidecar_for(image_id)
    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    url = holdings_url(sidecar)
    image_files = ImageFiles(
        image_files=[ImageFile(image_file_url=url, label_file_url=url, results_path_stub=image_id)]
    )
    navigate_image_files(obs_class, image_files, FCPath(str(work)), write_output_files=True)
    _write_json(step_path(work, image_id, NAVIGATE), {'kernels': furnished_kernels()})


def step_generate(image_id: str, work: Path) -> None:
    """Write the corrected kernel for one navigated image.

    The pool is rebuilt from what the navigation run furnished: every kernel
    but the C-kernels, since the assignment step needs a pool holding none, and
    the directories those C-kernels came from as the candidate index.

    Parameters:
        image_id: The library's id for the image.
        work: The working directory the three steps share.

    Raises:
        ValueError: if the image is not eligible for a segment, or if no
            candidate kernel reproduces the baseline it navigated against.
    """
    metadata = read_json(metadata_path(work, image_id))
    kernels = read_json(step_path(work, image_id, NAVIGATE))['kernels']
    for kernel in kernels:
        if kernel['kind'] != 'CK':
            cspyce.furnsh(kernel['path'])
    roots = sorted(
        {str(Path(kernel['path']).parent) for kernel in kernels if kernel['kind'] == 'CK'}
    )
    entry = ImageEntry.from_metadata(metadata)
    if entry.pointing is None:
        raise ValueError(
            f'{image_id} carries no corrected attitude to write: '
            f'{entry.ineligibility_reason.value if entry.ineligibility_reason else "unknown"}'
        )
    assignment = assign_images([entry], build_ck_index(roots))[0]
    if assignment.baseline is None:
        raise ValueError(f'{image_id} has no reproducing baseline among {roots}')
    local = str(cast(Path, assignment.baseline.path.retrieve()))
    cspyce.furnsh(local)
    try:
        segment = build_segment(entry.pointing)
    finally:
        cspyce.unload(local)
    output = work / output_basename(assignment.baseline.basename)
    handle = int(cspyce.ckopn(str(output), output.stem, _COMMENT_CHARS))
    try:
        write_segment(handle, segment)
        cspyce.ckcls(handle)
    except Exception:
        # A failure between ckopn and ckcls leaves an open DAF handle and a
        # partial kernel behind.  Mirror the production writer: close through
        # the plain DAF interface (whose own failure is suppressed so the
        # exception worth reading survives), remove the unusable file, and
        # re-raise.
        with contextlib.suppress(Exception):
            cspyce.dafcls(handle)
        output.unlink(missing_ok=True)
        raise
    _write_json(
        step_path(work, image_id, GENERATE),
        {
            'baseline_path': str(assignment.baseline.path),
            'ck_frame_id': entry.pointing.ck_frame_id,
            'claimed': _claimed_camera_attitudes(entry.pointing, segment),
            'has_angular_velocity': segment.has_angular_velocity,
            'output_path': str(output),
            'record_count': segment.record_count,
            'segid': segment.segid,
        },
    )


def step_renavigate(image_id: str, work: Path) -> None:
    """Furnish the corrected kernel and navigate the same image again.

    Parameters:
        image_id: The library's id for the image.
        work: The working directory the three steps share.
    """
    metadata = read_json(metadata_path(work, image_id))
    generated = read_json(step_path(work, image_id, GENERATE))
    pointing = ImagePointing.from_metadata(metadata)
    sidecar = sidecar_for(image_id)
    obs_class = _MISSION_TO_OBS_CLASS[sidecar.mission]
    url = holdings_url(sidecar)
    obs = cast(ObsSnapshotInst, obs_class.from_file(url))
    cspyce.furnsh(generated['output_path'])
    if pointing.ck_frame_id in FROZEN_ATTITUDE_CK_IDS:
        # This host built its observation frame out of a pointing lookup while
        # from_file ran, so the frame it returned cannot see a kernel furnished
        # afterwards.  Loading again with the correction in the pool is what
        # "furnish before any geometry is computed" means for such a host.
        obs = cast(ObsSnapshotInst, obs_class.from_file(url))
    measured = {
        label: _flatten(camera_attitude_from_pool(pointing, float(claim['et'])))
        for label, claim in generated['claimed'].items()
    }
    scale_u, scale_v = pixel_scales(obs.fov)
    orchestrator = NavOrchestrator(build_models_for_obs(obs))
    result = orchestrator.navigate(obs)
    _write_json(
        step_path(work, image_id, RENAVIGATE),
        {
            'measured': measured,
            'navigation_result': build_metadata_dict(result),
            'scale_u_rad_px': scale_u,
            'scale_v_rad_px': scale_v,
        },
    )


_STEPS: dict[str, Callable[[str, Path], None]] = {
    NAVIGATE: step_navigate,
    GENERATE: step_generate,
    RENAVIGATE: step_renavigate,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Run one step of the round trip.

    Parameters:
        argv: The command line, or ``None`` to read ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=sorted(_STEPS))
    parser.add_argument('image_id', help='the image library id of the image to run on')
    parser.add_argument('work_dir', help='the directory the three steps share')
    args = parser.parse_args(argv)
    _STEPS[args.step](args.image_id, Path(args.work_dir))


if __name__ == '__main__':
    require_checkout_package(__file__)
    main()
