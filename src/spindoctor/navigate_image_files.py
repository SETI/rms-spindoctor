"""Top-level driver that navigates a single image and writes results.

Given an observation class and an ``ImageFiles`` batch of size one, this
module reads the image, builds a ``NavOrchestrator`` configured with the
caller's model and technique filters, runs ``orchestrator.navigate(obs)``,
and writes the curated metadata (and a summary PNG when requested) to
``nav_results_root``.

This is the function ``sd_offset`` and ``sd_offset_cloud_tasks`` invoke
once per image.  Errors from image loading, missing SPICE coverage, and
navigation contract violations are captured into the output metadata so
the driver returns a structured result for them rather than crashing the
worker process.  One class of exception deliberately propagates: a defect
inside the corrected-pointing computation (anything it raises other than
``NavPointingError``) fails the run on its first image, because absorbing
it would silently drop pointing from a whole batch while every image
still reported success.
"""

from __future__ import annotations

import sys
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


def build_timing_section(start: datetime, end: datetime) -> dict[str, Any]:
    """Build the ``timing`` metadata section from run start and end moments.

    Every metadata document carries this section so downstream statistics
    (``sd_results_index`` / ``sd_stats_report``) can aggregate per-image run
    times.

    Parameters:
        start: Timezone-aware run start (captured before the image load).
        end: Timezone-aware run end (captured after navigation, or at
            error time).

    Returns:
        Dict with ``start_iso8601`` and ``end_iso8601`` (UTC ISO8601
        strings) and ``elapsed_s`` (float seconds).
    """
    return {
        'start_iso8601': _iso8601_utc(start),
        'end_iso8601': _iso8601_utc(end),
        'elapsed_s': (end - start).total_seconds(),
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
    ``_summary.png`` if requested.

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
        Tuple ``(success, metadata)`` where ``success`` is True for an
        ``ok`` ``NavResult.status`` and False otherwise.  ``metadata`` is
        the curated JSON-friendly dict.
    """
    logger = IMAGE_LOGGER
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
            'timing': build_timing_section(run_start, datetime.now(UTC)),
        }

    image_file = image_files.image_files[0]
    # resolve_image_url may correct the URL from the label contents, so it must
    # run before the URL is read
    image_url = image_file.resolve_image_url()
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    instrument = obs_class_to_inst_name(obs_class)
    extra_params = image_file.extra_params
    public_metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = nav_results_root / (image_file.results_path_stub + '_summary.png')

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
            'timing': build_timing_section(run_start, datetime.now(UTC)),
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
                    image_path,
                    image_name,
                    exc,
                    logger=logger,
                    instrument=instrument,
                    camera=image_file.camera,
                    timing=build_timing_section(run_start, datetime.now(UTC)),
                )
                if write_output_files:
                    public_metadata_file.write_text(json_as_string(metadata))
                if image_log_path is not None:
                    MAIN_LOGGER.info('Wrote log to %s', image_log_path)
                return False, metadata
            snapshot_inst = cast(ObsSnapshotInst, snapshot)
            orchestrator = NavOrchestrator(
                build_models_for_obs(snapshot_inst),
                only_models=nav_models or '*',
                only_techniques=nav_techniques or '*',
            )
            nav_result = orchestrator.navigate(snapshot_inst)
            data_shape = snapshot_inst.data.shape
            metadata = build_metadata_from_result(
                nav_result,
                image_path,
                image_name,
                instrument=instrument,
                camera=snapshot_inst.camera,
                shutter_mode=snapshot_inst.shutter_mode,
                image_shape=(int(data_shape[0]), int(data_shape[1])),
                timing=build_timing_section(run_start, datetime.now(UTC)),
            )
            if write_output_files:
                logger.info('Writing metadata to %s', public_metadata_file)
                public_metadata_file.write_text(json_as_string(metadata))
                write_summary_png(snapshot_inst, nav_result, summary_png_file, logger)
            log_final_result_to_run(image_name, nav_result)
            if image_log_path is not None:
                MAIN_LOGGER.info('Wrote log to %s', image_log_path)
            return nav_result.status == 'success', metadata
    finally:
        for handler in local_handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()


def _metadata_for_load_error(
    image_path: Path,
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
    observation: dict[str, Any] = {
        'image_path': str(image_path),
        'image_name': image_name,
        'instrument': instrument,
    }
    if camera is not None:
        observation['camera'] = camera
    return {
        'status': 'error',
        'status_error': status_error,
        'status_exception': message,
        'observation': observation,
        'timing': timing,
    }


def build_metadata_from_result(
    result: NavResult,
    image_path: Path,
    image_name: str,
    *,
    instrument: str,
    camera: str | None = None,
    shutter_mode: str | None = None,
    image_shape: tuple[int, int] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON metadata dict from a NavResult.

    Used by both the autonomous pipeline and the manual-nav driver so a
    manually-picked offset writes the same ``_metadata.json`` schema.

    Parameters:
        result: NavResult to curate.
        image_path: Absolute path to the source image; written to the
            ``observation.image_path`` field.
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
        timing: Run-timing section from :func:`build_timing_section`;
            written to the top-level ``timing`` field.  None omits the
            field.
    """
    observation: dict[str, Any] = {
        'image_path': str(image_path),
        'image_name': image_name,
        'instrument': instrument,
    }
    if camera is not None:
        observation['camera'] = camera
    if shutter_mode is not None:
        observation['shutter_mode'] = shutter_mode
    if image_shape is not None:
        observation['image_shape'] = [int(image_shape[0]), int(image_shape[1])]
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
    # Read straight through. These are the labels the summary PNG is captioned
    # with, and an empty dict does not produce a PNG that says it is missing
    # them -- it produces an unlabelled one, which reads as a product rather
    # than as a fault. The accessor is the observation describing itself; if it
    # raises, this image is not one whose product should be shipped.
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
