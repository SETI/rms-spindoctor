#!/usr/bin/env python3
################################################################################
# sd_create_bundle.py
#
# Top-level driver for PDS4 bundle generation. Enumerates images via datasets
# and, for each, generates PDS4 labels and metadata files. Also supports
# generating collection files and global index files.
################################################################################

import argparse
import os
import sys

import pdstemplate
from filecache import FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.cli.pds4.bundle_data import generate_bundle_data_files
from spindoctor.cli.pds4.collections import (
    generate_collection_files,
    generate_global_index_files,
)
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    build_run_logging,
    get_backplane_results_root,
    get_nav_results_root,
    get_pds4_bundle_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_CREATE_BUNDLE
from spindoctor.dataset import dataset_name_to_class, dataset_names
from spindoctor.dataset.dataset import DataSet

PROGRAM_NAME = SD_CREATE_BUNDLE
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""

DATASET: DataSet | None = None
DATASET_NAME: str | None = None


def add_common_arguments(parser: argparse.ArgumentParser, *, for_labels: bool = False) -> None:
    """Add common arguments to an argument parser.

    Parameters:
        parser: The argument parser to add arguments to.
    """
    add_logging_arguments(parser, has_image_logger=False)
    environment_group = parser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--bundle-results-root',
        type=str,
        default=None,
        required=False,
        help="""Root directory for bundle results; overrides the BUNDLE_RESULTS_ROOT
        environment variable and the bundle_results_root configuration variable""",
    )

    if for_labels:
        environment_group.add_argument(
            '--pds3-holdings-root',
            type=str,
            default=None,
            help='Root directory of PDS3 holdings; overrides PDS3_HOLDINGS_DIR or config',
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


def parse_args_labels(command_list: list[str]) -> argparse.Namespace:
    """Parse arguments for the labels subcommand."""
    global DATASET
    global DATASET_NAME

    if len(command_list) < 1:
        print('Usage: sd_create_bundle labels <dataset_name> [args]')
        sys.exit(1)

    DATASET_NAME = command_list[0].lower()

    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_create_bundle labels <dataset_name> [args]')
        sys.exit(1)

    DATASET = dataset_name_to_class(DATASET_NAME)()

    cmdparser = argparse.ArgumentParser(
        description='PDS4 Bundle Generation - Labels',
        epilog="""Generate PDS4 labels and metadata files for selected images.""",
    )

    # Common arguments
    add_common_arguments(cmdparser, for_labels=True)

    # Output
    output_group = cmdparser.add_argument_group('Output')
    output_group.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help="Don't process images, just print what would be done",
    )

    # Dataset selection
    DATASET.add_selection_arguments(cmdparser)

    arguments = cmdparser.parse_args(command_list[1:])
    return arguments


def parse_args_summary(command_list: list[str]) -> argparse.Namespace:
    """Parse arguments for the summary subcommand."""
    if len(command_list) < 1:
        print('Usage: sd_create_bundle summary <dataset_name> [args]')
        sys.exit(1)

    dataset_name = command_list[0].lower()

    if dataset_name not in dataset_names():
        print(f'Unknown dataset "{dataset_name}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: sd_create_bundle summary <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(
        description='PDS4 Bundle Generation - Summary',
        epilog="""Generate collection files and global index files for completed bundle.""",
    )

    add_common_arguments(cmdparser)

    arguments = cmdparser.parse_args(command_list[1:])
    # Store dataset_name in arguments for use in main_summary
    arguments.dataset_name = dataset_name.lower()
    return arguments


def main_labels() -> None:
    """Main function for labels subcommand."""
    command_list = sys.argv[2:]  # Skip 'labels'
    arguments = parse_args_labels(command_list)

    # Read configuration files
    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    with reporting_configuration_errors():
        build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    # Derive roots
    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    bundle_results_root_str = get_pds4_bundle_results_root(arguments, DEFAULT_CONFIG)
    bundle_results_root = FileCache(None).new_path(bundle_results_root_str)

    pdstemplate.PdsTemplate.set_logger(MAIN_LOGGER)

    assert DATASET is not None

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        if len(imagefiles.image_files) != 1:
            MAIN_LOGGER.error(
                'Expected 1 image file, got %d for %s',
                len(imagefiles.image_files),
                imagefiles,
            )
            continue
        if arguments.dry_run:
            MAIN_LOGGER.info(
                'Would process: %s', imagefiles.image_files[0].label_file_url.as_posix()
            )
            continue

        generate_bundle_data_files(
            dataset=DATASET,
            image_files=imagefiles,
            nav_results_root=nav_results_root,
            backplane_results_root=backplane_results_root,
            bundle_results_root=bundle_results_root,
            logger=MAIN_LOGGER,
        )


def main_summary() -> None:
    """Main function for summary subcommand."""
    command_list = sys.argv[2:]  # Skip 'summary'
    arguments = parse_args_summary(command_list)

    # Read configuration files via the shared loader (single source of truth
    # for config / CLI / env precedence), matching main_labels.
    with reporting_configuration_errors():
        load_default_and_user_config(arguments, DEFAULT_CONFIG)

    with reporting_configuration_errors():
        build_run_logging(PROGRAM_NAME, arguments, DEFAULT_CONFIG)

    bundle_results_root_str = get_pds4_bundle_results_root(arguments, DEFAULT_CONFIG)
    bundle_results_root = FileCache(None).new_path(bundle_results_root_str)

    pdstemplate.PdsTemplate.set_logger(MAIN_LOGGER)

    dataset_name = arguments.dataset_name
    dataset = dataset_name_to_class(dataset_name)()

    # Generate collection files
    try:
        generate_collection_files(
            bundle_results_root=bundle_results_root,
            dataset=dataset,
            logger=MAIN_LOGGER,
        )
    except Exception:
        MAIN_LOGGER.exception('Failed to generate collection files')
        sys.exit(1)

    # Generate global index files
    try:
        generate_global_index_files(
            bundle_results_root=bundle_results_root,
            dataset=dataset,
            logger=MAIN_LOGGER,
        )
    except Exception:
        MAIN_LOGGER.exception('Failed to generate global index files')
        sys.exit(1)

    MAIN_LOGGER.info('Summary generation complete')


def main() -> None:
    """Main entry point with subparsers."""
    if len(sys.argv) < 2:
        print('Usage: sd_create_bundle <labels|summary> [args]')
        sys.exit(1)

    subcommand = sys.argv[1].lower()

    if subcommand == 'labels':
        main_labels()
    elif subcommand == 'summary':
        main_summary()
    else:
        print(f'Unknown subcommand "{subcommand}"')
        print('Usage: sd_create_bundle <labels|summary> [args]')
        sys.exit(1)


if __name__ == '__main__':
    main()
