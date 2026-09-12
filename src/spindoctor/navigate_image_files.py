"""Top-level driver that navigates a single image and writes results.

Given an observation class and an ``ImageFiles`` batch of size one, this
module reads the image, builds a ``NavOrchestrator`` configured with the
caller's model and technique filters, runs ``orchestrator.navigate(obs)``,
and writes the curated metadata (and a summary PNG when requested) to
``nav_results_root``.

This is the function ``sd_offset`` and ``sd_offset_cloud_tasks`` invoke
once per image.  Every failure of one image is recorded in that image's
document and the run goes on to the next image, wherever the failure arises:
the label read that resolves the image's URL, its retrieval into the cache,
the image load, missing SPICE coverage, a navigation contract violation,
provenance or context construction in the orchestrator, the corrected-pointing
computation, the summary PNG, or a defect anywhere in between.  A failure the
orchestrator classifies is a ``failed`` document with the reason in
``status_reason``; one that raises past it is an ``error`` document with
``status_error`` ``internal_error``, the exception in ``status_exception`` and
its traceback in ``status_traceback``, which the error selection filters pick
up on a rerun.  A fault before the
image's own log section opens is recorded from the run's log and names the
image by its URL.  Only an interrupt stops the run.

The PNG is written before the document, so a document never says success for
an image whose products are incomplete.  Nothing an earlier run left for an
image is removed before this run navigates it: each product is replaced only
when this run writes its own, so a run interrupted at an image leaves the
earlier products in place.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pdslogger
from filecache import FCPath
from PIL import Image

from spindoctor.config import (
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    run_logging_for_root,
)
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import (
    NavOrchestrator,
    NavResult,
    build_metadata_dict,
)
from spindoctor.obs import ObsSnapshotInst, obs_class_to_inst_name
from spindoctor.support.file import json_as_string
from spindoctor.support.memory import peak_resident_bytes, reset_peak_resident
from spindoctor.support.misc import log_run_environment
from spindoctor.support.summary_png import (
    SummaryMetadata,
    render_annotated_summary_rgb,
)

__all__ = [
    'build_metadata_from_result',
    'build_timing_section',
    'log_final_result_to_run',
    'navigate_image_files',
    'write_summary_png',
]


_RESTATED_PUBLIC_METADATA = frozenset({'image_shape_xy'})
"""Keys a host publishes that the ``observation`` block already states in another form.

``image_shape_xy`` is ``image_shape`` in the other axis order.  A published key the block
itself holds -- the image's path, its name, its camera -- is skipped by name instead.
"""

_SPICE_DATA_HINTS = (
    'SPICE(CKINSUFFDATA)',
    'SPICE(SPKINSUFFDATA)',
    'SPICE(NOFRAMECONNECT)',
)


def _iso8601_utc(moment: datetime) -> str:
    """UTC ISO8601 string (``Z`` suffix) for a timezone-aware datetime.

    Parameters:
        moment: Timezone-aware datetime to format.

    Returns:
        The moment rendered in UTC as ``YYYY-MM-DDTHH:MM:SS.ffffffZ``.
    """
    return moment.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def log_final_result_to_run(image_name: str, nav_result: NavResult) -> None:
    """Report one image's answer to the run's log, in a single line.

    The detail of how the answer was reached belongs to the image and stays in
    the image's log, which is where the orchestrator writes it.  What the
    answer *was* is the run's business too: it is the thing an operator
    watching a batch is waiting for, and following a run otherwise means
    opening a file per image to find out whether any of them worked.

    Deliberately one line, and deliberately a summary rather than a move of
    the orchestrator's records -- the per-image log keeps the offset, the
    sigmas, the confidence rank and the per-technique breakdown in full.

    Parameters:
        image_name: The image's file name, which names the line.
        nav_result: The navigation result to report.
    """
    if nav_result.offset_px is None:
        MAIN_LOGGER.info(
            '%s: status=%s, no offset (%s)',
            image_name,
            nav_result.status,
            nav_result.status_reason,
        )
        return
    MAIN_LOGGER.info(
        '%s: status=%s, offset (dv, du) = (%.3f, %.3f) px, confidence %.3f (%s)',
        image_name,
        nav_result.status,
        nav_result.offset_px[0],
        nav_result.offset_px[1],
        nav_result.confidence,
        nav_result.confidence_rank,
    )


def build_timing_section(start: datetime, end: datetime, *, peak_measured: bool) -> dict[str, Any]:
    """Build the ``timing`` metadata section from run start and end moments.

    Every metadata document carries this section so downstream statistics
    (``sd_results_index`` / ``sd_stats_report``) can aggregate per-image run
    times and the memory each image needed.

    The peak is the largest resident size the navigating process reached while
    this image was being navigated, which is the figure an out-of-memory kill is
    decided against.  A process handling one image reports that image's whole
    memory usage.  A process handling several reports, for each, what it reached
    while that image ran, measured from a floor that includes whatever earlier
    images left resident: the memory a worker has to hold at that moment rather
    than the memory the image would use on its own.  It is absent where the kernel
    publishes no peak at all.

    Parameters:
        start: Timezone-aware run start (captured before the image load).
        end: Timezone-aware run end (captured after navigation, or at
            error time).
        peak_measured: What :func:`~spindoctor.support.memory.reset_peak_resident`
            returned for this image.  A kernel that publishes the mark but will
            not let it be reset would otherwise have the whole process's
            lifetime high-water mark recorded as this image's peak, which is a
            number about a different thing rather than a missing one.

    Returns:
        Dict with ``start_iso8601`` and ``end_iso8601`` (UTC ISO8601
        strings), ``elapsed_s`` (float seconds) and ``peak_memory_bytes``
        (int, or None where the kernel publishes no peak or the mark could
        not be reset ahead of this image).
    """
    return {
        'start_iso8601': _iso8601_utc(start),
        'end_iso8601': _iso8601_utc(end),
        'elapsed_s': (end - start).total_seconds(),
        'peak_memory_bytes': peak_resident_bytes() if peak_measured else None,
    }


def navigate_image_files(
    obs_class: type[ObsSnapshotInst],
    image_files: ImageFiles,
    nav_results_root: FCPath,
    *,
    nav_models: list[str] | None = None,
    nav_techniques: list[str] | None = None,
    write_output_files: bool = True,
    run_logging: RunLogging | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Navigate one image batch and optionally write the result files.

    Top-level driver.  Reads the image, asks every registered
    ``NavModel`` subclass to construct whatever instances apply to the
    observation, runs the orchestrator with the caller's model and
    technique filters, and writes ``_metadata.json`` plus
    ``_summary.png`` if requested.  Whatever fails for the image is
    recorded in its metadata and returned rather than raised, so a caller
    working through a batch goes on to its next image; only an interrupt
    stops the run.

    Parameters:
        obs_class: Concrete ``ObsSnapshotInst`` subclass for the mission.
        image_files: ``ImageFiles`` batch.  Exactly one image per batch is
            supported; calling code should split larger batches.
        nav_results_root: Directory to write ``_metadata.json`` and
            ``_summary.png`` results to.  May be a ``FileCache`` URL.
        nav_models: Glob-pattern list selecting which ``NavModel`` instances
            run; ``None`` means all.  Patterns may use a leading ``!`` for
            exclusion.
        nav_techniques: Glob-pattern list selecting which ``NavTechnique``
            instances run.  ``None`` means all.
        write_output_files: When True, write the metadata JSON and summary
            PNG; when False, perform a dry run and return the metadata
            dict only.
        run_logging: This run's resolved logging, giving the level and sinks
            the per-image log is written with.  ``None`` resolves the
            configuration's defaults, for a caller outside a configured run.

    Returns:
        Tuple ``(success, metadata)`` where ``success`` is True for a
        ``success`` ``NavResult.status`` and False otherwise.  ``metadata``
        is the curated JSON-friendly dict, or an error document (``status``
        ``'error'`` with ``status_error``, ``status_exception`` and
        ``status_traceback``) when the image could not be loaded or navigating
        it raised.
    """
    logger = IMAGE_LOGGER
    # Ahead of any work this image causes, so that the peak the section
    # records is this image's own rather than one an earlier image reached.
    # What it answers is carried to the section: a mark that could not be reset
    # is a mark about the whole process.
    peak_measured = reset_peak_resident()
    run_start = datetime.now(UTC)

    if len(image_files.image_files) != 1:
        # A malformed batch is a caller error, reported before any image scope
        # exists and belonging to no image, so it goes to the run's log.
        MAIN_LOGGER.error(
            'Expected exactly one image per batch; got %d',
            len(image_files.image_files),
        )
        return False, {
            'status': 'error',
            'status_error': 'expected_one_image_per_batch',
            'status_exception': (
                f'Expected exactly one image per batch; got {len(image_files.image_files)}'
            ),
            'observation': {'instrument': obs_class_to_inst_name(obs_class)},
            'timing': build_timing_section(
                run_start, datetime.now(UTC), peak_measured=peak_measured
            ),
        }

    image_file = image_files.image_files[0]
    instrument = obs_class_to_inst_name(obs_class)
    public_metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = nav_results_root / (image_file.results_path_stub + '_summary.png')
    try:
        # resolve_image_url may correct the URL from the label contents, so it must
        # run before the URL is read
        image_url = image_file.resolve_image_url().absolute()
        image_name = image_url.name
        extra_params = image_file.extra_params

        if run_logging is None:
            # Derive the log root from the results root this call was given rather
            # than re-resolving one: a caller that named its results root has
            # already said where its output belongs, and resolving afresh both
            # ignores that and fails outright when nothing else names a root.
            run_logging = run_logging_for_root(nav_results_root / 'logs')
        try:
            local_handlers, image_log_path = build_image_log_handlers(
                'nav',
                image_file.results_path_stub,
                run_logging.sinks,
                run_logging.levels,
                timestamp=run_logging.timestamp,
            )
        except ValueError as exc:
            # A stub that would put the log outside the log root is a bad image
            # entry.  It fails its own image and returns like any other per-image
            # error, rather than raising through the driver and taking the rest of
            # the batch with it.
            MAIN_LOGGER.error('Refusing to navigate %s: %s', image_url, exc)
            return False, {
                'status': 'error',
                'status_error': 'invalid_results_path_stub',
                'status_exception': str(exc),
                'observation': {'instrument': instrument},
                'timing': build_timing_section(
                    run_start, datetime.now(UTC), peak_measured=peak_measured
                ),
            }

        try:
            with logger.open(
                str(image_url),
                handler=local_handlers,
                level=run_logging.levels.image_section_level(),
            ):
                log_run_environment(logger, sys.argv[1:])
                try:
                    snapshot = obs_class.from_file(image_url, **extra_params)
                except (OSError, RuntimeError) as exc:
                    metadata = _metadata_for_load_error(
                        image_url,
                        image_name,
                        exc,
                        logger=logger,
                        instrument=instrument,
                        camera=image_file.camera,
                        timing=build_timing_section(
                            run_start, datetime.now(UTC), peak_measured=peak_measured
                        ),
                    )
                    if write_output_files:
                        public_metadata_file.write_text(json_as_string(metadata))
                    if image_log_path is not None:
                        MAIN_LOGGER.info('Wrote log to %s', image_log_path)
                    return False, metadata
                snapshot_inst = cast(ObsSnapshotInst, snapshot)
                try:
                    orchestrator = NavOrchestrator(
                        build_models_for_obs(snapshot_inst),
                        only_models=nav_models or '*',
                        only_techniques=nav_techniques or '*',
                    )
                    nav_result = orchestrator.navigate(snapshot_inst)
                    data_shape = snapshot_inst.data.shape
                    metadata = build_metadata_from_result(
                        nav_result,
                        image_url,
                        image_name,
                        instrument=instrument,
                        camera=snapshot_inst.camera,
                        shutter_mode=snapshot_inst.shutter_mode,
                        image_shape=(int(data_shape[0]), int(data_shape[1])),
                        public_metadata=snapshot_inst.get_public_metadata(),
                        timing=build_timing_section(
                            run_start, datetime.now(UTC), peak_measured=peak_measured
                        ),
                    )
                    if write_output_files:
                        # The PNG is written before the document so a fault in it
                        # is recorded as this image's failure, not as a success
                        # document beside no PNG.
                        write_summary_png(snapshot_inst, nav_result, summary_png_file, logger)
                        logger.info('Writing metadata to %s', public_metadata_file)
                        public_metadata_file.write_text(json_as_string(metadata))
                    log_final_result_to_run(image_name, nav_result)
                    success = nav_result.status == 'success'
                except Exception as exc:
                    # This is the top level of one image's run, and the one place
                    # a catch-all belongs: whatever raised, the image is failed and
                    # recorded, and the run goes on to the next image.  The
                    # orchestrator has already turned every model and technique
                    # failure into a failed result, so what arrives here is a
                    # fault outside them or a defect.  The camera is the index's,
                    # read without touching the observation, since reading the
                    # observation may be what raised.  An interrupt is not an
                    # Exception and still stops the run.
                    logger.exception(
                        'INTERNAL ERROR: navigating %s raised; failing this image with '
                        'status error internal_error',
                        image_name,
                    )
                    metadata = _metadata_for_internal_error(
                        image_url,
                        image_name,
                        exc,
                        instrument=instrument,
                        camera=image_file.camera,
                        timing=build_timing_section(
                            run_start, datetime.now(UTC), peak_measured=peak_measured
                        ),
                    )
                    if write_output_files:
                        public_metadata_file.write_text(json_as_string(metadata))
                    success = False
                if image_log_path is not None:
                    MAIN_LOGGER.info('Wrote log to %s', image_log_path)
                return success, metadata
        finally:
            for handler in local_handlers:
                if handler is not pdslogger.NULL_HANDLER:
                    handler.close()
    except Exception as exc:
        # A fault before or around the image's own log section -- the label
        # read that resolves the URL, the retrieval into the cache, the log
        # handlers, the section itself, or a document write inside it -- is
        # this image's failure too.  It is recorded from the run's log, since
        # the image's log may not exist, and named by the URL, since the local
        # path may not.  An interrupt is not an Exception and still stops the
        # run.
        MAIN_LOGGER.exception(
            'INTERNAL ERROR: preparing %s raised; failing this image with '
            'status error internal_error',
            image_file.image_file_url,
        )
        metadata = _metadata_for_internal_error(
            image_file.image_file_url,
            image_file.image_file_url.name,
            exc,
            instrument=instrument,
            camera=image_file.camera,
            timing=build_timing_section(run_start, datetime.now(UTC), peak_measured=peak_measured),
        )
        if write_output_files:
            try:
                public_metadata_file.write_text(json_as_string(metadata))
            except Exception:
                # The document write itself may be what is failing, and a
                # results root that refuses one image's document must not
                # stop the run either; the document still reaches the caller.
                MAIN_LOGGER.exception(
                    'INTERNAL ERROR: writing %s raised; the error document is returned unwritten',
                    public_metadata_file,
                )
        return False, metadata


def _metadata_for_load_error(
    image_path: FCPath,
    image_name: str,
    exc: BaseException,
    *,
    logger: Any,
    instrument: str,
    camera: str | None,
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Build a metadata dict for an image-load or kernel-coverage failure.

    No image shape is recorded because the load never produced pixel data,
    and no epoch because that comes from the observation the load never
    built.  The camera is recorded when the caller could read it from the
    index, so an image that failed for want of a SPICE kernel is still
    attributed to its camera.
    """
    message = str(exc)
    if any(hint in message for hint in _SPICE_DATA_HINTS):
        logger.exception('No SPICE kernel available for "%s": %s', image_path, message)
        status_error = 'missing_spice_data'
    else:
        logger.exception('Error reading image "%s": %s', image_path, message)
        status_error = 'image_read_error'
    return _error_metadata(
        image_path,
        image_name,
        status_error=status_error,
        status_exception=message,
        status_traceback=_traceback_text(exc),
        instrument=instrument,
        camera=camera,
        timing=timing,
    )


def _metadata_for_internal_error(
    image_path: FCPath,
    image_name: str,
    exc: Exception,
    *,
    instrument: str,
    camera: str | None,
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Build a metadata dict for an image whose navigation raised.

    This is the document of a failure the driver's catch-all recorded rather than
    one the orchestrator classified: a fault in provenance or context
    construction, in the corrected-pointing computation, in the summary PNG, or a
    defect anywhere between loading the image and writing its document.  The
    image loaded, but nothing it would report is read here, since reading it may
    be what raised; so, as for a load error, no image shape and no epoch are
    recorded.

    Parameters:
        image_path: Where the run read the source image from: its URL for
            remote holdings, its absolute path for local ones.
        image_name: Basename of the source image.
        exc: What navigating the image raised.
        instrument: Registered instrument name for the observation class.
        camera: The camera the index attributed the image to, or ``None`` for an
            image with no index row.
        timing: Run-timing section from :func:`build_timing_section`.

    Returns:
        A dict with ``status`` ``'error'``, ``status_error`` ``'internal_error'``,
        ``status_exception`` naming the exception's type and text as
        ``'RuntimeError: message'``, ``status_traceback`` carrying its traceback,
        an ``observation`` block with the image path, image name, instrument and
        (when given) camera, and the ``timing`` section.
    """
    return _error_metadata(
        image_path,
        image_name,
        status_error='internal_error',
        status_exception=f'{type(exc).__name__}: {exc}',
        status_traceback=_traceback_text(exc),
        instrument=instrument,
        camera=camera,
        timing=timing,
    )


def _traceback_text(exc: BaseException) -> str:
    """Format an exception's traceback the way the document records it.

    The text is the interpreter's own rendering, so a chained exception
    brings the chain with it and the reader of the document sees exactly
    what the log shows.

    Parameters:
        exc: The exception that failed the image.

    Returns:
        The traceback as a single string, without a trailing newline.
    """
    return ''.join(traceback.format_exception(exc)).rstrip('\n')


def _error_metadata(
    image_path: FCPath,
    image_name: str,
    *,
    status_error: str,
    status_exception: str,
    status_traceback: str,
    instrument: str,
    camera: str | None,
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Build the document of an image this run could not navigate.

    Parameters:
        image_path: Where the run read the source image from: its URL for
            remote holdings, its absolute path for local ones.
        image_name: Basename of the source image.
        status_error: Machine-readable classification of what went wrong.
        status_exception: The failure's text, for an operator reading the document.
        status_traceback: The failure's traceback, from :func:`_traceback_text`.
        instrument: Registered instrument name for the observation class.
        camera: The camera that took the image, or ``None`` to omit the field.
        timing: Run-timing section from :func:`build_timing_section`.

    Returns:
        A dict with ``status`` ``'error'``, the ``status_error``,
        ``status_exception`` and ``status_traceback`` given, an ``observation``
        block with the image path, image name, instrument and (when given)
        camera, and the ``timing`` section.
    """
    observation: dict[str, Any] = {
        'image_path': image_path.as_posix(),
        'image_name': image_name,
        'instrument': instrument,
    }
    if camera is not None:
        observation['camera'] = camera
    return {
        'status': 'error',
        'status_error': status_error,
        'status_exception': status_exception,
        'status_traceback': status_traceback,
        'observation': observation,
        'timing': timing,
    }


def build_metadata_from_result(
    result: NavResult,
    image_path: str | Path | FCPath,
    image_name: str,
    *,
    instrument: str,
    camera: str | None = None,
    shutter_mode: str | None = None,
    image_shape: tuple[int, int] | None = None,
    public_metadata: Mapping[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON metadata dict from a NavResult.

    Used by both the autonomous pipeline and the manual-nav driver so a
    manually-picked offset writes the same ``_metadata.json`` schema.

    The ``observation`` block records what the observation says about the
    image whether or not its navigation succeeded and whether or not a pointing
    was recorded: the identity the parameters below give it, followed by every
    fact in ``public_metadata`` the block does not already state.

    Parameters:
        result: NavResult to curate.
        image_path: Where the run read the source image from, its URL for
            remote holdings or its absolute path for local ones, as a string,
            a ``Path`` or an ``FCPath``; written to the ``observation.image_path``
            field.
        image_name: Basename of the source image; written to the
            ``observation.image_name`` field.
        instrument: Registered instrument name for the observation class
            (see :func:`spindoctor.obs.obs_class_to_inst_name`); written to
            the ``observation.instrument`` field.
        camera: The camera that took the image (``ObsInst.camera``, e.g.
            ``'NAC'``); written to the ``observation.camera`` field.  None
            omits the field.
        shutter_mode: The shutter mode the image was taken in
            (``ObsInst.shutter_mode``, e.g. ``'BOTSIM'``); written to the
            ``observation.shutter_mode`` field.  None omits the field.
        image_shape: ``(v, u)`` pixel dimensions of the loaded image data;
            written to the ``observation.image_shape`` field.  None omits
            the field.
        public_metadata: What the observation's instrument host publishes about the
            image (``ObsInst.get_public_metadata``): its start, midtime and end times,
            its exposure time, its filters, and whatever else that host states.  Each
            fact is written to the ``observation`` block under the host's own key and
            as the host states it, after the fields above, unless the block already
            states it: a key the block already holds keeps the block's value, and
            ``image_shape_xy`` is omitted because it is ``image_shape`` in the other
            axis order.  None records no published fact.
        timing: Run-timing section from :func:`build_timing_section`;
            written to the top-level ``timing`` field.  None omits the
            field.
    """
    location = FCPath(image_path)
    observation: dict[str, Any] = {
        'image_path': location.as_posix(),
        'image_name': image_name,
        'instrument': instrument,
    }
    if camera is not None:
        observation['camera'] = camera
    if shutter_mode is not None:
        observation['shutter_mode'] = shutter_mode
    if image_shape is not None:
        observation['image_shape'] = [int(image_shape[0]), int(image_shape[1])]
    if public_metadata is not None:
        observation.update(
            (key, value)
            for key, value in public_metadata.items()
            if key not in observation and key not in _RESTATED_PUBLIC_METADATA
        )
    metadata: dict[str, Any] = {
        'status': result.status,
        'observation': observation,
        'navigation_result': build_metadata_dict(result),
    }
    if timing is not None:
        metadata['timing'] = timing
    if result.offset_px is not None:
        metadata['offset'] = list(result.offset_px)
    metadata['confidence'] = result.confidence
    return metadata


def _summary_metadata_from_obs_result(obs: ObsSnapshotInst, result: NavResult) -> SummaryMetadata:
    """Assemble the summary-PNG header block from an obs and its NavResult.

    Reads the image name, filter, and exposure from the observation's public
    metadata (degrading gracefully when a field is absent) and the status,
    contributing techniques, and confidence from the NavResult.

    Parameters:
        obs: Observation snapshot the summary is rendered from.
        result: Navigation result whose status / techniques / confidence
            are reported.

    Returns:
        A populated :class:`~spindoctor.support.summary_png.SummaryMetadata`.
    """
    exposure_s: float | None = None
    # get_public_metadata is called without a guard.  Substituting an empty
    # dict when it raised would produce a PNG with a blank caption that looks
    # finished; letting the exception reach the driver records the failure in
    # the image's document instead.  The PNG is written before that document,
    # so no document ever describes a PNG with a blank caption.
    public = obs.get_public_metadata()
    abspath = getattr(obs, 'abspath', None)
    image_name = str(public.get('image_name') or (abspath.name if abspath is not None else ''))
    filters = [str(f) for f in (public.get('filters') or []) if f]
    filter_name = '+'.join(filters)
    exposure_raw = public.get('exposure_time')
    if exposure_raw is not None:
        try:
            exposure_s = float(exposure_raw)
        except (TypeError, ValueError):
            IMAGE_LOGGER.warning('Unparsable exposure_time %r; omitting from summary', exposure_raw)
    # The techniques that actually contributed to the reported offset are the
    # ensemble's consensus subset, not every technique that ran (outliers the
    # ensemble rejected still appear in per_technique).  Fall back to the full
    # per-technique set only if a success left the consensus list unstamped.
    if result.status == 'success':
        contributing = result.consensus_techniques or [
            tr.technique_name for tr in result.per_technique
        ]
        techniques = tuple(sorted(set(contributing)))
    else:
        techniques = ()
    return SummaryMetadata(
        image_name=image_name,
        filter_name=filter_name,
        exposure_s=exposure_s,
        status=result.status,
        techniques=techniques,
        confidence=result.confidence,
        confidence_rank=result.confidence_rank,
    )


def write_summary_png(
    obs: ObsSnapshotInst,
    result: NavResult,
    png_path: FCPath,
    logger: Any,
) -> None:
    """Composite the source image with the orchestrator's annotation overlay.

    Thin driver around
    :func:`spindoctor.support.summary_png.render_annotated_summary_rgb`; writes
    the resulting RGB to ``png_path`` as a PNG.

    Parameters:
        obs: Observation snapshot used as the background.
        result: Navigation result; ``offset_px`` shifts the overlay to the
            best-fit pose, ``annotations`` carries every NavModel's overlay.
        png_path: Destination path; supports ``FCPath`` URLs.
        logger: ``pdslogger`` to emit one INFO line on success.
    """
    overlay_offset = result.offset_px if result.offset_px is not None else (0.0, 0.0)
    summary_metadata = _summary_metadata_from_obs_result(obs, result)
    rgb = render_annotated_summary_rgb(
        obs, result.annotations, overlay_offset, metadata=summary_metadata
    )
    buf = BytesIO()
    Image.fromarray(rgb, mode='RGB').save(buf, format='PNG')
    png_path.write_bytes(buf.getvalue())
    logger.info('Wrote summary PNG to %s', png_path)
