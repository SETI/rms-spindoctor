#!/usr/bin/env python3
################################################################################
# sd_offset.py
#
# This is the main top-level driver for offset finding. It enumerates one or
# more images either from scanning an index file, a holdings directory,
# or from an AWS SQS queue, and for each computes the offset and saves the
# resulting offset and preview image files. TODO
################################################################################

import argparse
import cProfile
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import islice
from typing import cast

import pdslogger
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    build_run_logging,
    get_nav_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_OFFSET
from spindoctor.dataset import dataset_name_to_class, dataset_name_to_inst_name, dataset_names
from spindoctor.dataset.dataset import DataSet, ImageFiles
from spindoctor.dataset.results_filter import SelectionError
from spindoctor.nav_technique import run_manual_nav
from spindoctor.navigate_image_files import (
    build_metadata_from_result,
    build_timing_section,
    navigate_image_files,
    write_summary_png,
)
from spindoctor.obs import ObsSnapshotInst, inst_name_to_obs_class, obs_class_to_inst_name
from spindoctor.support.file import json_as_string
from spindoctor.support.memory import reset_peak_resident
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_OFFSET
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

DATASET: DataSet | None = None
DATASET_NAME: str | None = None
NUM_FILES_PROCESSED: int = 0
NUM_FILES_SKIPPED: int = 0
NUM_FILES_COMPLETED: int = 0
START_TIME: float = 0.0


################################################################################
#
# ARGUMENT PARSING
#
################################################################################


def parse_args(command_list: list[str]) -> argparse.Namespace:
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: sd_offset <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_offset <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_offset <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Navigation & Backplane Main Interface for Offsets',
        epilog="""Default behavior is to perform an offset pass on all Cassini images
                that don't have associated offset files""",
    )

    # Arguments about the environment
    environment_group = cmdparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--pds3-holdings-root',
        type=str,
        default=None,
        help="""The root directory of the PDS3 holdings; overrides the PDS3_HOLDINGS_DIR
        environment variable and the pds3_holdings_root configuration variable""",
    )
    environment_group.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help="""The root directory of the navigation results; overrides the NAV_RESULTS_ROOT
        environment variable and the nav_results_root configuration variable""",
    )
    environment_group.add_argument(
        '--results-index-db',
        type=str,
        default=None,
        metavar='URL',
        help="""Connection URL of the results index written by sd_results_index (a sqlite:
        URL naming a local path, or a postgresql+psycopg: URL naming a server); overrides
        the environment.results_index_db configuration variable and NAV_RESULTS_INDEX_DB. The image
        selection options that read the navigation results are then answered by one query
        over that index instead of by reading the results tree. The index is a snapshot of
        its last ingest, so an image navigated since is one it does not hold; pass
        --results-index-db none to read the tree instead, spelled exactly that way in lower
        case, since anything else is read as the URL of an index.""",
    )

    # Arguments about the general navigation process
    nav_group = cmdparser.add_argument_group('Navigation')
    nav_group.add_argument(
        '--nav-models',
        type=str,
        default=None,
        help='Comma-separated list of model names to use',
    )
    nav_group.add_argument(
        '--nav-techniques',
        type=str,
        default=None,
        help='Comma-separated list of navigation technique names to use',
    )
    nav_group.add_argument(
        '--manual',
        action='store_true',
        default=False,
        help="""Open the interactive manual-navigation dialog instead of running
        the autonomous pipeline.  Requires the selection to resolve to exactly
        one image.  On accept, the chosen offset is printed to stdout and the
        same _metadata.json and _summary.png files the autonomous pipeline
        writes are produced under nav_results_root (suppress with
        --no-write-output-files); use the dialog's "Save as Library Entry..."
        button to write a sidecar.""",
    )

    # Arguments about output file generation
    output_group = cmdparser.add_argument_group('Output')
    # output_group.add_argument(
    #     '--write-offset-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate an offset file; no implies --no-overlay-file')
    # output_group.add_argument(
    #     '--write-overlay-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate an overlay file')
    # output_group.add_argument(
    #     '--write-png-file', action=argparse.BooleanOptionalAction, default=True,
    #     help='Generate a PNG file')
    output_group.add_argument(
        '--output-cloud-tasks-file',
        type=str,
        default=None,
        help="""Write a JSON file containing task descriptions for all selected images that
        is suitable for loading into a cloud_tasks queue; do not perform any other processing.""",
    )
    output_group.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Don't actually process the images, just print what would be done",
    )
    output_group.add_argument(
        '--no-write-output-files',
        action='store_true',
        default=False,
        help="Don't write any output files",
    )

    # Add all the arguments related to selecting files
    DATASET.add_selection_arguments(cmdparser)

    add_logging_arguments(cmdparser)

    # Misc arguments
    misc_group = cmdparser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Do performance profiling',
    )

    arguments = cmdparser.parse_args(command_list[1:])

    return arguments


def exit_processing() -> None:
    MAIN_LOGGER.info('Total files processed %d', NUM_FILES_PROCESSED)
    MAIN_LOGGER.info('Total files skipped %d', NUM_FILES_SKIPPED)
    MAIN_LOGGER.info('Total elapsed time %.2f sec', time.time() - START_TIME)

    sys.exit(0)


def _run_manual_pass(
    obs_class: type['ObsSnapshotInst'],
    arguments: argparse.Namespace,
    nav_results_root: FCPath,
    run_logging: 'RunLogging',
    *,
    write_output_files: bool = True,
) -> None:
    """Open the manual-navigation dialog on a single selected image.

    Requires the dataset selection arguments to resolve to exactly one
    one-image batch.  On accept, the chosen offset is printed to stdout
    and the same ``_metadata.json`` and ``_summary.png`` files the
    autonomous pipeline writes are produced under ``nav_results_root``;
    set ``write_output_files=False`` to skip the writes.  The dialog's
    "Save as Library Entry..." button independently writes a sidecar to
    the image library when the operator wants one.

    The orchestrator's per-image log lines (image classifier verdict,
    NavModel build, feature extraction, manual-nav skip warnings) are
    routed through the same per-image ``IMAGE_LOGGER.open(...)`` context
    that ``navigate_image_files`` uses, so this run's per-image handlers are
    attached during prepare + dialog.
    """
    # Bound the dataset traversal to at most six items: we only need to
    # distinguish the {0, 1, >1} cases and to surface up to five filespecs
    # in the multi-match diagnostic.  Larger datasets used to scan the
    # whole tree just to print a count.
    selected_preview = list(islice(_selected_image_files(arguments), 6))
    if not selected_preview:
        MAIN_LOGGER.error('No images matched the selection arguments')
        sys.exit(1)
    if len(selected_preview) > 1:
        names = ', '.join(b.image_files[0].image_file_url.as_posix() for b in selected_preview[:5])
        MAIN_LOGGER.error(
            '--manual requires exactly one image; selection matched at least %d '
            '(first few: %s).  Tighten the selection flags.',
            len(selected_preview),
            names,
        )
        sys.exit(1)

    image_files = selected_preview[0]
    if len(image_files.image_files) != 1:
        MAIN_LOGGER.error(
            '--manual requires a one-image batch; got %d.', len(image_files.image_files)
        )
        sys.exit(1)

    image_file = image_files.image_files[0]
    # resolve_image_url may correct the URL from the label contents, so it must
    # run before the URL is read
    image_url = image_file.resolve_image_url().absolute()
    image_name = image_url.name
    extra_params = image_file.extra_params
    public_metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = nav_results_root / (image_file.results_path_stub + '_summary.png')
    MAIN_LOGGER.info('Manual nav: loading image %s', image_url.as_posix())

    local_handlers, image_log_path = build_image_log_handlers(
        'nav',
        image_file.results_path_stub,
        run_logging.sinks,
        run_logging.levels,
        timestamp=run_logging.timestamp,
    )

    try:
        with IMAGE_LOGGER.open(
            str(image_url),
            handler=local_handlers,
            level=run_logging.levels.image_section_level(),
        ):
            # Ahead of any work this image causes, so that the peak the
            # timing section records is this image's own.
            reset_peak_resident()
            run_start = datetime.now(UTC)
            obs = cast(ObsSnapshotInst, obs_class.from_file(image_url, **extra_params))
            result = run_manual_nav(obs, config=DEFAULT_CONFIG)
            if result is None:
                # ``run_manual_nav`` already logged the precise reason
                # (no renderable features, empty composed overlay, or
                # operator cancelled).  No metadata or PNG is written.
                sys.exit(2)

            assert result.offset_px is not None  # status='success' guarantees offset
            dv, du = result.offset_px
            IMAGE_LOGGER.info('Manual nav: offset_dv_px=%.4f, offset_du_px=%.4f', dv, du)
            if write_output_files:
                # The timing section's elapsed time is the manual-nav wall
                # time: image load + dialog interaction until accept.
                metadata = build_metadata_from_result(
                    result,
                    image_url,
                    image_name,
                    instrument=obs_class_to_inst_name(obs_class),
                    camera=obs.camera,
                    shutter_mode=obs.shutter_mode,
                    image_shape=(int(obs.data.shape[0]), int(obs.data.shape[1])),
                    timing=build_timing_section(run_start, datetime.now(UTC)),
                )
                IMAGE_LOGGER.info('Writing metadata to %s', public_metadata_file)
                public_metadata_file.write_text(json_as_string(metadata))
                write_summary_png(obs, result, summary_png_file, IMAGE_LOGGER)
            if image_log_path is not None:
                MAIN_LOGGER.info('Wrote log to %s', image_log_path)
    finally:
        for handler in local_handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    # The dv / du print statements are the CLI's machine-parsable contract;
    # they go to stdout regardless of the per-image log routing.
    print(f'offset_dv_px={dv:.4f}')
    print(f'offset_du_px={du:.4f}')


###############################################################################
#
# MAIN
#
###############################################################################


def _selected_image_files(arguments: argparse.Namespace) -> Iterator[ImageFiles]:
    """Yield the image batches the selection arguments name, reporting a refusal.

    The enumeration is where the selection arguments are finally read, so it is
    where a contradictory pair of them, or a results index that cannot be
    opened, cannot be read, or does not cover this root, is first diagnosed.
    Each of those is a run that is misconfigured rather than one that went
    wrong, and each already carries a message saying what to change; a traceback
    would bury it, and would print an index URL's password into the terminal
    along the way.  Only those are reported this way: an enumeration that fails
    for any other reason has gone wrong rather than been misconfigured, and its
    traceback is what says where.

    Parameters:
        arguments: The parsed command line.

    Yields:
        One batch of image files at a time, in enumeration order.
    """
    assert DATASET is not None  # just for type checking
    try:
        yield from DATASET.yield_image_files_from_arguments(arguments)
    except SelectionError as exc:
        MAIN_LOGGER.error('%s', exc)
        sys.exit(1)


def main() -> None:

    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    if arguments.profile:
        pr = cProfile.Profile()
        pr.enable()

    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    # Derive the results root
    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    with reporting_configuration_errors():
        run_logging = build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    global START_TIME, NUM_FILES_PROCESSED, NUM_FILES_SKIPPED, NUM_FILES_COMPLETED
    START_TIME = time.time()
    NUM_FILES_PROCESSED = 0
    NUM_FILES_SKIPPED = 0
    NUM_FILES_COMPLETED = 0

    MAIN_LOGGER.info('**********************************')
    MAIN_LOGGER.info('*** BEGINNING MAIN OFFSET PASS ***')
    MAIN_LOGGER.info('**********************************')
    MAIN_LOGGER.info('')
    log_run_environment(MAIN_LOGGER, command_list)

    try:
        INST_NAME = dataset_name_to_inst_name(cast(str, DATASET_NAME))
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        sys.exit(1)

    obs_class = inst_name_to_obs_class(INST_NAME)

    nav_models = arguments.nav_models.split(',') if arguments.nav_models is not None else None
    nav_techniques = (
        arguments.nav_techniques.split(',') if arguments.nav_techniques is not None else None
    )

    assert DATASET is not None  # just for type checking

    if arguments.manual:
        _run_manual_pass(
            obs_class,
            arguments,
            nav_results_root,
            run_logging,
            write_output_files=not arguments.no_write_output_files,
        )
        sys.exit(0)

    if arguments.output_cloud_tasks_file:
        MAIN_LOGGER.info('Writing cloud_tasks file to %s', arguments.output_cloud_tasks_file)
        task_arguments = {
            'nav_models': nav_models,
            'nav_techniques': nav_techniques,
        }
        tasks_json = []
        for imagefile_idx, imagefiles in enumerate(_selected_image_files(arguments)):
            task_id = f'{DATASET_NAME}-{imagefiles.image_files[0].label_file_name}-{imagefile_idx}'
            task_files = []
            for image_file in imagefiles.image_files:
                task_files.append(
                    {
                        'image_file_url': image_file.image_file_url.as_posix(),
                        'label_file_url': image_file.label_file_url.as_posix(),
                        'results_path_stub': image_file.results_path_stub,
                        'index_file_row': image_file.index_file_row,
                    }
                )
            task_info = {
                'task_id': task_id,
                'data': {
                    'arguments': task_arguments,
                    'dataset_name': DATASET_NAME,
                    'files': task_files,
                },
            }
            tasks_json.append(task_info)

        cloud_tasks_path = FCPath(arguments.output_cloud_tasks_file)
        with cloud_tasks_path.open('w') as f:
            json_string = json_as_string(tasks_json)
            f.write(json_string)
        MAIN_LOGGER.info('Wrote cloud_tasks file to %s', arguments.output_cloud_tasks_file)
        sys.exit(0)

    for imagefiles in _selected_image_files(arguments):
        assert len(imagefiles.image_files) == 1
        if arguments.dry_run:
            MAIN_LOGGER.info(
                'Would process: %s', imagefiles.image_files[0].label_file_url.as_posix()
            )
            continue

        MAIN_LOGGER.info(
            'Processing: %s',
            ', '.join(f.image_file_url.as_posix() for f in imagefiles.image_files),
        )
        success, metadata = navigate_image_files(
            obs_class,
            imagefiles,
            nav_results_root=nav_results_root,
            nav_models=nav_models,
            nav_techniques=nav_techniques,
            write_output_files=not arguments.no_write_output_files,
            run_logging=run_logging,
        )
        if success:
            NUM_FILES_PROCESSED += 1
        else:
            NUM_FILES_SKIPPED += 1
            MAIN_LOGGER.debug(
                'Navigation failed; metadata keys: %s',
                sorted(metadata.keys(), key=str),
            )

    exit_processing()


if __name__ == '__main__':
    main()
