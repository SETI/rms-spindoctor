"""Generate per-pixel geometry backplanes for one navigated image.

The stage reads the image's navigation record, applies the pointing that record
supplies, computes the body and ring backplanes on the pointed observation, and
writes them as one FITS file.  Where the record comes from is the caller's
choice: a :class:`~spindoctor.cli.reproj.pointing_source.PointingSource` reads it
either from the ``_metadata.json`` document the navigator wrote or from one row
of an ingested results index.  Both supply the same record fields and both
classify the pointing with the same classifier, so the backplanes an image gets
do not depend on which of them the run was pointed at.  The record shapes whose
degradation the two storages *name* differently, none of which changes the
product, are stated in that module.

An image whose navigation did not succeed produces no backplanes, and an image
nothing navigated at all raises rather than producing them on uncorrected
pointing.  The distinction is the point: a backplane computed on a pointing
nobody navigated is geometry for a place the camera was not quite looking, and
the product carries no sign of it.
"""

from typing import Any

import pdslogger
from filecache import FCPath

from spindoctor.cli.reproj.offsets import apply_pointing_to_obs, select_pointing
from spindoctor.cli.reproj.pointing_source import PointingSource
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    run_logging_for_root,
)
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.nav_record import record_status, record_status_error

from .backplanes_bodies import create_body_backplanes
from .backplanes_rings import create_ring_backplanes
from .merge import merge_sources_into_master
from .writer import write_fits


def generate_backplanes_image_files(
    obs_class: type[ObsSnapshotInst],
    image_files: ImageFiles,
    *,
    pointing_source: PointingSource,
    backplane_results_root: FCPath,
    write_output_files: bool = True,
    run_logging: RunLogging | None = None,
) -> dict[str, Any]:
    """Generate backplanes for a single image batch using prior navigation metadata.

    Parameters:
        obs_class: Observation snapshot class for the instrument.
        image_files: List of images; must have exactly one image in the batch.
        pointing_source: Where this image's navigation record is read from --
            the documents the navigator wrote, or an ingested results index.
        backplane_results_root: Destination root for FITS and label files.
        write_output_files: Whether to write outputs to storage.
        run_logging: This run's resolved logging, giving the level and sinks
            the per-image log is written with.  ``None`` resolves the
            configuration's defaults against the backplane results root.

    Returns:
        ``{'status': 'success'}`` -- carrying ``pointing_source`` (one of
        ``'cmatrix'``, ``'pool'``, ``'offset'``, ``'none'``) naming which
        recorded pointing the product was built on, plus ``pointing_reason``
        when the outcome carries one, plus
        ``uncorrected_pointing`` when that source is ``'none'`` and the
        backplanes were computed on the camera's uncorrected pointing -- or
        ``{'status': 'skipped'}`` with the navigation status that caused the
        skip.  An image can be skipped or degraded for a reason its caller
        has no other way to learn: the reason is reported to the run's log,
        and a cloud task has no run log, so returning it is what keeps a
        batch that quietly skipped or degraded everything distinguishable
        from one that processed it.

    Raises:
        ValueError: if more than one image is batched, which is a defect in the
            caller rather than in any record.
    """

    logger = IMAGE_LOGGER
    config = DEFAULT_CONFIG
    if run_logging is None:
        run_logging = run_logging_for_root(backplane_results_root / 'logs')

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}'
        )

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    fits_file_path = backplane_results_root / (image_file.results_path_stub + '_backplanes.fits')

    # Decide whether there is work before opening a log for it, so a skipped
    # image does not leave behind a file containing only its own header.  This
    # raises if nothing recorded the image; the caller reports that.
    nav_metadata = pointing_source.read_record(image_file)

    # Read through the functions every consumer of a record reads them
    # through, so that a document and the row it was ingested into report the
    # same outcome and the same error for the same image.
    status = record_status(nav_metadata)
    if status != 'success':
        nav_error = record_status_error(nav_metadata)
        MAIN_LOGGER.warning(
            'Skipping backplanes for "%s": status=%s error=%s',
            image_path,
            status,
            nav_error,
        )
        return {
            'status': 'skipped',
            'status_error': 'nav_status_not_success',
            'nav_status': status,
            'nav_status_error': nav_error,
        }

    applied_source = 'none'
    pointing_reason: str | None = None
    local_handlers, image_log_path = build_image_log_handlers(
        'backplanes',
        image_file.results_path_stub,
        run_logging.sinks,
        run_logging.levels,
        timestamp=run_logging.timestamp,
    )
    try:
        with logger.open(
            f'Processing image: {image_path!s}',
            handler=local_handlers,
            level=run_logging.levels.image_section_level(),
        ):
            # The traceback belongs to the image, and the image's own log is
            # attached only inside this window; a failure recorded after it
            # closes leaves the one log a reader would open with no account of
            # how the image ended.  The caller still meets the exception and
            # decides whether it costs the image or the run.
            try:
                # Build observation in original FOV
                # TODO We only support snapshots for backplane generation for now
                obs = obs_class.from_file(image_path, extfov_margin_vu=(0, 0))
                if not isinstance(obs, ObsSnapshotInst):
                    raise TypeError(f'Expected ObsSnapshotInst, got {type(obs).__name__}')
                snapshot = obs

                # Apply the recorded pointing: the corrected C-matrix when the
                # record carries a usable one, else the (dv, du) offset via
                # OffsetFOV, else nothing.
                selection = select_pointing(nav_metadata, subject=str(image_path))
                applied = apply_pointing_to_obs(snapshot, selection, subject=str(image_path))
                applied_source = applied.source
                pointing_reason = applied.reason
                if applied.source == 'none':
                    # Backplanes computed on uncorrected pointing are geometry for
                    # a place the camera was not quite looking, and the product
                    # carries no sign of it.  The image's log gets the account; the
                    # run's gets told it happened, because someone watching a batch
                    # would otherwise have to open every log to find out.
                    MAIN_LOGGER.warning(
                        '%s: computing backplanes on uncorrected pointing (%s)',
                        image_path,
                        applied.reason,
                    )

                # Compute bodies backplanes
                bodies_result = create_body_backplanes(snapshot, config, logger=logger)
                # Compute rings backplanes (if enabled/configured)
                rings_result = create_ring_backplanes(snapshot, config, logger=logger)

                # Merge all sources (distance-aware)
                master_by_type, body_id_map = merge_sources_into_master(
                    snapshot,
                    bodies_result=bodies_result,
                    rings_result=rings_result,
                )

                if write_output_files:
                    write_fits(
                        fits_file_path=fits_file_path,
                        snapshot=snapshot,
                        master_by_type=master_by_type,
                        body_id_map=body_id_map,
                        config=config,
                        bodies_result=bodies_result,
                        rings_result=rings_result,
                        logger=logger,
                    )
            except Exception:
                logger.exception('Backplane generation failed for %s', image_path)
                raise
    finally:
        for handler in local_handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
        if image_log_path is not None:
            MAIN_LOGGER.info('Wrote log to %s', image_log_path)

    # Returned as well as logged: a cloud task has no run log, so this is
    # the only way the facts leave the worker.
    result: dict[str, Any] = {'status': 'success', 'pointing_source': applied_source}
    if pointing_reason is not None:
        result['pointing_reason'] = pointing_reason
    if applied_source == 'none':
        result['uncorrected_pointing'] = True
    return result
