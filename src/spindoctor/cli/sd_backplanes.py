#!/usr/bin/env python3
################################################################################
# sd_backplanes.py
#
# Top-level driver for backplane generation. Enumerates images via datasets and
# for each, generates body and ring backplanes based on prior offset metadata.
################################################################################

import argparse
import os
import sys
import traceback
from typing import cast

from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.backplanes.backplanes import generate_backplanes_image_files
from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.cli.reproj.pointing_source import build_pointing_source
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_run_logging,
    get_backplane_results_root,
    get_nav_results_root,
    get_results_index_db_url,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_BACKPLANES
from spindoctor.dataset import dataset_name_to_class, dataset_name_to_inst_name, dataset_names
from spindoctor.dataset.dataset import DataSet
from spindoctor.obs import inst_name_to_obs_class
from spindoctor.results_index import masked_url
from spindoctor.support.file import json_as_string
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_BACKPLANES
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

DATASET: DataSet | None = None
DATASET_NAME: str | None = None


def parse_args(command_list: list[str]) -> argparse.Namespace:
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: sd_backplanes <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_backplanes <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_backplanes <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Backplanes Main Interface',
        epilog="""Default behavior is to generate body/ring backplanes for selected images
                  using prior navigation offsets stored in metadata_root.""",
    )

    # Environment
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
        '--nav-results-root',
        type=str,
        default=None,
        help="""Root directory for prior navigation metadata files (_metadata.json);
        overrides NAV_RESULTS_ROOT and the nav_results_root configuration variable""",
    )
    environment_group.add_argument(
        '--backplane-results-root',
        type=str,
        default=None,
        help="""Root directory for backplane results; overrides the BACKPLANE_RESULTS_ROOT
        environment variable and the backplane_results_root configuration variable""",
    )
    environment_group.add_argument(
        '--results-index-db',
        type=str,
        default=None,
        metavar='URL',
        help="""Connection URL of the results index written by sd_results_index (a
        sqlite: URL naming a local path, or a postgresql+psycopg: URL naming a
        server); overrides the environment.results_index_db configuration variable
        and NAV_RESULTS_INDEX_DB. Each image's navigation record is then read as one
        row instead of one file. Pass --results-index-db none to read the files even where
        an index is configured. Without an index the navigation results tree is read
        directly, which is the default.""",
    )

    # Output
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--output-cloud-tasks-file',
        type=str,
        default=None,
        help="""Write a JSON task descriptions file suitable for loading into a
        cloud_tasks queue (consumed by sd_backplanes_cloud_tasks); do not
        generate any backplanes.""",
    )
    output_group.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Don't process images, just print what would be done",
    )
    output_group.add_argument(
        '--no-write-output-files',
        action='store_true',
        default=False,
        help="Don't write any output files",
    )

    # Dataset selection
    DATASET.add_selection_arguments(cmdparser)

    add_logging_arguments(cmdparser)

    # Misc
    misc_group = cmdparser.add_argument_group('Miscellaneous')
    misc_group.add_argument(
        '--profile', action='store_true', default=False, help='Enable profiling'
    )

    arguments = cmdparser.parse_args(command_list[1:])
    return arguments


def main() -> None:
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    # Read configuration files
    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)
    with reporting_configuration_errors():
        run_logging = build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    # Derive roots
    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    results_index_db_url = get_results_index_db_url(arguments, DEFAULT_CONFIG)

    MAIN_LOGGER.info('Starting backplanes generation')
    MAIN_LOGGER.info('Dataset: %s', DATASET_NAME)
    MAIN_LOGGER.info('Nav results root: %s', nav_results_root.as_posix())
    MAIN_LOGGER.info('Backplane results root: %s', backplane_results_root.as_posix())
    MAIN_LOGGER.info(
        'Results index: %s',
        masked_url(results_index_db_url)
        if results_index_db_url is not None
        else 'none (reading files)',
    )
    MAIN_LOGGER.info('Dry run: %s', arguments.dry_run)
    MAIN_LOGGER.info('No write output files: %s', arguments.no_write_output_files)
    # Routed through the run-environment block rather than logged directly:
    # that is the one place a command line is recorded, and it masks the value
    # of every connection-URL option before the line reaches a log file or a
    # bug report.  An index URL can carry a database password.
    log_run_environment(MAIN_LOGGER, command_list)

    assert DATASET is not None
    inst_name = dataset_name_to_inst_name(cast(str, DATASET_NAME))
    obs_class = inst_name_to_obs_class(inst_name)

    if arguments.output_cloud_tasks_file:
        MAIN_LOGGER.info('Writing cloud_tasks file to %s', arguments.output_cloud_tasks_file)
        tasks_json = []
        for imagefile_idx, imagefiles in enumerate(
            DATASET.yield_image_files_from_arguments(arguments)
        ):
            task_id = f'{DATASET_NAME}-{imagefiles.image_files[0].label_file_name}-{imagefile_idx}'
            task_files = [
                {
                    'image_file_url': f.image_file_url.as_posix(),
                    'label_file_url': f.label_file_url.as_posix(),
                    'results_path_stub': f.results_path_stub,
                    'index_file_row': f.index_file_row,
                }
                for f in imagefiles.image_files
            ]
            tasks_json.append(
                {
                    'task_id': task_id,
                    'data': {
                        'dataset_name': DATASET_NAME,
                        'files': task_files,
                    },
                }
            )

        cloud_tasks_path = FCPath(arguments.output_cloud_tasks_file)
        with cloud_tasks_path.open('w') as f:
            f.write(json_as_string(tasks_json))
        MAIN_LOGGER.info('Wrote cloud_tasks file to %s', arguments.output_cloud_tasks_file)
        return

    # A dry run reads no navigation record, so it opens no index: failing it
    # for want of a working index would fail it for something it never touches.
    if arguments.dry_run:
        for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
            # The same batching the real loop requires: a dry run that reported
            # a batch the real run would refuse would report a run that cannot
            # happen.
            assert len(imagefiles.image_files) == 1
            MAIN_LOGGER.info(
                'Would process: %s', imagefiles.image_files[0].label_file_url.as_posix()
            )
        return

    # A resolved index that will not open, or a root it has not fully ingested,
    # fails the run here.  Falling back to reading files would turn a
    # misconfigured run into a slow, silently different one.
    pointing_source = build_pointing_source(
        nav_results_root, results_index_db_url=results_index_db_url
    )
    n_done = 0
    n_skipped = 0
    n_failed = 0
    try:
        for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
            assert len(imagefiles.image_files) == 1
            label_url = imagefiles.image_files[0].label_file_url.as_posix()
            try:
                result = generate_backplanes_image_files(
                    obs_class,
                    imagefiles,
                    pointing_source=pointing_source,
                    backplane_results_root=backplane_results_root,
                    write_output_files=not arguments.no_write_output_files,
                    run_logging=run_logging,
                )
            except FileNotFoundError as e:
                # An expected outcome rather than a defect: nothing navigated
                # this image, so there is nothing to build geometry from.
                MAIN_LOGGER.error('Skipped due to missing metadata: %s (%s)', label_url, str(e))
                n_skipped += 1
                continue
            except Exception as exc:
                # Backplane generation is per-image work with no cross-image
                # state, so one image's failure is that image's failure and not
                # the run's.  Nothing here knows where the failure came from --
                # a lookup before the image's own log exists leaves no other
                # account of it -- so the run's log carries the traceback as
                # well as the image and the message.  ``stacktrace=False`` plus
                # ``more=`` is used because PdsLogger's own stack omits the
                # final "SomeError: ..." line and would duplicate the frames.
                MAIN_LOGGER.exception(
                    'Failed to generate backplanes for %s: %s',
                    label_url,
                    exc,
                    stacktrace=False,
                    more=traceback.format_exc(),
                )
                n_failed += 1
                continue
            if result['status'] == 'skipped':
                n_skipped += 1
            else:
                n_done += 1
    finally:
        pointing_source.close()
    MAIN_LOGGER.info(
        'Backplane pass complete: %d done, %d skipped, %d failed', n_done, n_skipped, n_failed
    )


if __name__ == '__main__':
    main()
