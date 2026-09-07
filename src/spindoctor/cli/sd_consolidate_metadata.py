#!/usr/bin/env python3
################################################################################
# sd_consolidate_metadata.py
#
# Iterate through a dataset selection in the same way sd_offset.py does and
# copy each image's metadata JSON, summary PNG, or both to a single flat
# destination directory so the results are easy to browse without descending
# the per-volume / per-rev path hierarchy.
################################################################################

import argparse
import os
import sys
import time
from pathlib import PurePosixPath

from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_run_logging,
    get_nav_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_CONSOLIDATE_METADATA
from spindoctor.dataset import dataset_name_to_class, dataset_names
from spindoctor.dataset.dataset import DataSet
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_CONSOLIDATE_METADATA
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

DATASET: DataSet | None = None
DATASET_NAME: str | None = None
NUM_FILES_COPIED: int = 0
NUM_FILES_MISSING: int = 0
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
        print('Usage: sd_consolidate_metadata <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_consolidate_metadata <dataset_name> [args]')
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_consolidate_metadata <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='Copy per-image navigation metadata files into a single flat '
        'destination directory.',
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
        help="""The root directory of the navigation results to read from; overrides the
        NAV_RESULTS_ROOT environment variable and the nav_results_root configuration
        variable""",
    )

    # Arguments about what to copy and where
    output_group = cmdparser.add_argument_group('Consolidation')
    output_group.add_argument(
        '--dest-dir',
        type=str,
        required=True,
        help="""Destination directory.  All copied files land directly here; no
        per-volume sub-hierarchy is preserved.  May be a local path or any
        FCPath-supported URL (gs://, s3://, https://...); the directory and
        any missing parents are materialized on first write.""",
    )
    output_group.add_argument(
        '--copy-metadata',
        action='store_true',
        default=False,
        help='Copy per-image _metadata.json files',
    )
    output_group.add_argument(
        '--copy-png',
        action='store_true',
        default=False,
        help='Copy per-image _summary.png files',
    )
    output_group.add_argument(
        '--copy-all',
        action='store_true',
        default=False,
        help='Copy both metadata JSON and summary PNG (equivalent to --copy-metadata --copy-png).',
    )
    output_group.add_argument(
        '--add-numerical-prefix',
        action='store_true',
        default=False,
        help="""Prefix each destination filename with a six-digit increasing
        index so the directory listing matches the iteration order.""",
    )
    output_group.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite destination files if they already exist.',
    )
    output_group.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Don't actually copy files, just print what would be done",
    )

    # Add all the arguments related to selecting files
    DATASET.add_selection_arguments(cmdparser)

    add_logging_arguments(cmdparser, has_image_logger=False)

    arguments = cmdparser.parse_args(command_list[1:])

    if arguments.copy_all:
        arguments.copy_metadata = True
        arguments.copy_png = True

    if not (arguments.copy_metadata or arguments.copy_png):
        cmdparser.error('Specify at least one of --copy-metadata, --copy-png, or --copy-all')

    return arguments


def _flat_basename(results_path_stub: str) -> str:
    """Return the leaf filename of a results-path stub for a flat destination.

    ``results_path_stub`` carries directory components that mirror the
    PDS3 / PDS4 hierarchy (e.g. ``COISS_2090/data/.../N1777325846_1_CALIB``);
    the consolidated destination is flat, so only the leaf name survives.
    """
    return PurePosixPath(results_path_stub).name


def _copy_one(
    src: FCPath,
    dest: FCPath,
    label: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    """Copy ``src`` to ``dest`` if it exists; return True on success.

    Logs the outcome at INFO; missing-source and skip-existing are
    INFO-level too because they are expected during partial runs.
    Existence of ``src`` is determined as a side effect of
    :meth:`FCPath.read_bytes` (catching :class:`FileNotFoundError`)
    rather than a pre-flight ``exists()`` to avoid the second
    round-trip on remote backends.
    """
    global NUM_FILES_MISSING
    if dest.exists() and not overwrite:
        MAIN_LOGGER.info(
            '%s file destination exists, skipping (use --overwrite to replace): %s',
            label,
            dest.as_posix(),
        )
        return False
    if dry_run:
        MAIN_LOGGER.info('Would copy %s -> %s', src.as_posix(), dest.as_posix())
        return True
    try:
        payload = src.read_bytes()
    except FileNotFoundError:
        MAIN_LOGGER.info('%s file not present: %s', label, src.as_posix())
        NUM_FILES_MISSING += 1
        return False
    dest.write_bytes(payload)
    MAIN_LOGGER.info('Copied %s -> %s', src.as_posix(), dest.as_posix())
    return True


###############################################################################
#
# MAIN
#
###############################################################################


def main() -> None:
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)
    # No pre-flight mkdir: FCPath.write_bytes materializes the necessary
    # parent directories on first write.
    dest_root = FileCache(None).new_path(arguments.dest_dir)

    with reporting_configuration_errors():
        build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    global START_TIME, NUM_FILES_COPIED, NUM_FILES_MISSING
    START_TIME = time.time()
    NUM_FILES_COPIED = 0
    NUM_FILES_MISSING = 0

    MAIN_LOGGER.info('***************************************')
    MAIN_LOGGER.info('*** BEGINNING METADATA CONSOLIDATION ***')
    MAIN_LOGGER.info('***************************************')
    MAIN_LOGGER.info('')
    log_run_environment(MAIN_LOGGER, command_list)

    assert DATASET is not None  # parse_args populated it
    image_index = 0
    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        for image_file in imagefiles.image_files:
            image_index += 1
            stub = image_file.results_path_stub
            leaf = _flat_basename(stub)
            if arguments.add_numerical_prefix:
                leaf = f'{image_index:06d}_{leaf}'
            copied_any = False
            if arguments.copy_metadata:
                src = nav_results_root / (stub + '_metadata.json')
                dest = dest_root / (leaf + '_metadata.json')
                if _copy_one(
                    src,
                    dest,
                    'Metadata',
                    overwrite=arguments.overwrite,
                    dry_run=arguments.dry_run,
                ):
                    copied_any = True
            if arguments.copy_png:
                src = nav_results_root / (stub + '_summary.png')
                dest = dest_root / (leaf + '_summary.png')
                if _copy_one(
                    src,
                    dest,
                    'Summary PNG',
                    overwrite=arguments.overwrite,
                    dry_run=arguments.dry_run,
                ):
                    copied_any = True
            if copied_any:
                NUM_FILES_COPIED += 1

    MAIN_LOGGER.info('Total images for which something was copied %d', NUM_FILES_COPIED)
    MAIN_LOGGER.info('Total source files missing %d', NUM_FILES_MISSING)
    MAIN_LOGGER.info('Total elapsed time %.2f sec', time.time() - START_TIME)
    sys.exit(0)


if __name__ == '__main__':
    main()
