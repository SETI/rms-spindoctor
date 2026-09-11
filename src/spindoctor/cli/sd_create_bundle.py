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
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.logging_args import add_logging_arguments, reporting_configuration_errors
from spindoctor.cli.pds4.bundle_data import BundleDataOutcome, generate_bundle_data_files
from spindoctor.cli.pds4.collections import (
    generate_collection_files,
    generate_global_index_files,
    index_value_format,
)
from spindoctor.config import (
    DEFAULT_CONFIG,
    MAIN_LOGGER,
    Config,
    build_run_logging,
    get_backplane_results_root,
    get_nav_results_root,
    get_pds4_bundle_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_CREATE_BUNDLE
from spindoctor.dataset import dataset_name_to_class, dataset_names
from spindoctor.dataset.dataset import DataSet, Pds4Pass

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
    """Parse arguments for the labels subcommand.

    The selection arguments are the dataset class's own, declared and read by
    it, so the parser adds them without an instance and reads none of them
    itself; the dataset is constructed once the command line is parsed.

    Sets the module globals ``DATASET`` and ``DATASET_NAME`` as a side effect,
    because every later stage of the subcommand reads the dataset from there.

    Parameters:
        command_list: The subcommand's arguments, the dataset name first.

    Returns:
        The parsed arguments.

    Raises:
        SystemExit: With status 1, and a usage line on stdout, when no dataset
            name was given or when the name is not a known dataset.  These end
            the program rather than raising to a caller because this is a
            command line being read, and there is nothing above it to recover.
    """
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

    dataset_class = dataset_name_to_class(DATASET_NAME)

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
    dataset_class.add_selection_arguments(cmdparser)

    arguments = cmdparser.parse_args(command_list[1:])

    DATASET = dataset_class()

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


def _exit_on_missing_templates(dataset: DataSet, pds4_pass: Pds4Pass) -> None:
    """Report every template the pass needs and cannot find, and stop if any.

    Every product of a pass renders from the same template directory, so a
    template that is not there is not there for every image; the pass says so
    once, before it has written anything, rather than failing identically for
    thousands of images.

    Parameters:
        dataset: The dataset whose template directory the pass renders from.
        pds4_pass: Which pass's declared templates to look for.

    Raises:
        SystemExit: If any declared template is not a file in the template
            directory.  A directory at that name satisfies ``exists`` and is
            not a template, so the check is that the path is a file: the point
            of the preflight is that nothing is written before a render that
            cannot happen.
    """
    template_dir = FCPath(dataset.pds4_bundle_template_dir())
    missing = [
        template_dir / name
        for name in dataset.pds4_required_templates(pds4_pass)
        if not (template_dir / name).is_file()
    ]
    if len(missing) == 0:
        return
    for template_path in missing:
        MAIN_LOGGER.error('PDS4 template not found: %s', template_path)
    MAIN_LOGGER.error(
        'The %s pass needs %d template(s) that are not in %s; nothing was written',
        pds4_pass,
        len(missing),
        template_dir,
    )
    sys.exit(1)


def _exit_on_unusable_units(config: Config) -> None:
    """Report every configured backplane in a unit the bundle cannot use, and stop if any.

    Both passes hold a backplane to the unit its configuration entry declares:
    the labels pass compares each document's statistics against it, and the
    summary pass writes each index column in the format that unit calls for.
    A unit neither can use is unusable for every image, so the pass says so
    once, before it has read anything, rather than failing identically for
    thousands of images or after the collection files are on disk.

    Parameters:
        config: The configuration the pass runs under, whose body and ring
            backplane entries are checked.

    Raises:
        SystemExit: If any entry's ``units`` is missing or is not a string, is
            blank, or names a unit the index tables have no format for.  Every
            such entry is reported, with the reason, before the exit.
    """
    unusable: list[tuple[str, str]] = []
    for entry in [*config.backplanes.bodies, *config.backplanes.rings]:
        try:
            index_value_format(entry.get('units'))
        except (TypeError, ValueError) as exc:
            unusable.append((entry['name'], str(exc)))
    if len(unusable) == 0:
        return
    for name, reason in unusable:
        MAIN_LOGGER.error('Backplane %s declares a unit the bundle cannot use: %s', name, reason)
    MAIN_LOGGER.error(
        'The configuration declares %d backplane(s) in a unit the bundle cannot use; '
        'nothing was written',
        len(unusable),
    )
    sys.exit(1)


def _bundle_root_holds_anything(bundle_root: FCPath) -> bool:
    """Report whether the bundle's own directory already holds something.

    Parameters:
        bundle_root: The bundle's directory under the bundle results root.

    Returns:
        True if anything at all is in that directory, False if it is empty or
        is not there at all.
    """
    try:
        return next(iter(bundle_root.iterdir()), None) is not None
    except FileNotFoundError:
        return False


def main_labels() -> None:
    """Main function for labels subcommand.

    Three preconditions are checked before any image is processed, and each
    ends the run with exit status 1 having written nothing.  Every template the
    dataset declares for this pass must be in its template directory, since one
    that is not would otherwise fail identically for every image.  Every
    backplane the configuration declares must be in a unit the bundle can use,
    for the same reason.  And the bundle root must be empty or absent: a bundle
    is written into an empty directory rather than assembled out of two runs.
    A dry run is refused the same way, because what it reports on is a run
    that would be.
    """
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

    _exit_on_missing_templates(DATASET, 'labels')
    _exit_on_unusable_units(DATASET.config)

    bundle_root = bundle_results_root / DATASET.pds4_bundle_name()
    if _bundle_root_holds_anything(bundle_root):
        MAIN_LOGGER.error(
            'The bundle root %s already holds files; a bundle is written into an empty '
            'directory. Clear it, or name another bundle results root, and run again',
            bundle_root,
        )
        sys.exit(1)

    written_images = 0
    skipped_images = 0
    failed_images = 0
    listed_images = 0
    malformed_empty_batches = 0

    for imagefiles in DATASET.yield_image_files_from_arguments(arguments):
        if len(imagefiles.image_files) != 1:
            # A batch of any other size is images the run did not write labels
            # for, so they count against the run the same way a broken label
            # does -- every one of them, since a batch of two is two images
            # without labels.  An empty batch is no images and still a run that
            # failed, so it is counted apart from them rather than passing for
            # a run that wrote everything it meant to.  A dry run writes
            # nothing, so it reports the batch it cannot process without
            # counting a label it never set out to write.
            MAIN_LOGGER.error(
                'Expected 1 image file, got %d for %s',
                len(imagefiles.image_files),
                imagefiles,
            )
            if not arguments.dry_run:
                failed_images += len(imagefiles.image_files)
                malformed_empty_batches += len(imagefiles.image_files) == 0
            continue
        if arguments.dry_run:
            MAIN_LOGGER.info(
                'Would process: %s', imagefiles.image_files[0].label_file_url.as_posix()
            )
            listed_images += 1
            continue

        try:
            outcome = generate_bundle_data_files(
                dataset=DATASET,
                image_files=imagefiles,
                nav_results_root=nav_results_root,
                backplane_results_root=backplane_results_root,
                bundle_results_root=bundle_results_root,
                logger=MAIN_LOGGER,
            )
        except Exception:
            # One image whose inputs cannot be read or whose template cannot be
            # found is one image without a label, not a run without a report:
            # the images after it are still processed and the run still says at
            # the end how many labels it did not write.
            MAIN_LOGGER.exception(
                'Failed to generate bundle data files for %s',
                imagefiles.image_files[0].image_file_url.as_posix(),
            )
            failed_images += 1
            continue

        if outcome is BundleDataOutcome.FAILED:
            failed_images += 1
        elif outcome is BundleDataOutcome.SKIPPED:
            skipped_images += 1
        else:
            written_images += 1

    if failed_images > 0 or malformed_empty_batches > 0:
        MAIN_LOGGER.error(
            'Label generation incomplete: %d image(s) labeled, %d skipped, '
            '%d whose labels were not written, %d empty batch(es)',
            written_images,
            skipped_images,
            failed_images,
            malformed_empty_batches,
        )
        sys.exit(1)

    if arguments.dry_run:
        # A dry run counts nothing against itself, so it never reaches the
        # report above; what it has to say is what it would have processed.
        MAIN_LOGGER.info('Dry run complete: %d image(s) would be processed', listed_images)
        return

    MAIN_LOGGER.info(
        'Label generation complete: %d image(s) labeled, %d skipped',
        written_images,
        skipped_images,
    )


def main_summary() -> None:
    """Main function for summary subcommand.

    Every template the dataset declares for this pass must be in its template
    directory, and every backplane the configuration declares must be in a
    unit the index tables can write; either failing ends the run with exit
    status 1 before anything is read or written.  The bundle root is not
    checked for emptiness here: this pass reads the tree the labels pass wrote.
    """
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

    _exit_on_missing_templates(dataset, 'summary')
    _exit_on_unusable_units(dataset.config)

    # Generate collection files
    try:
        failed_labels = generate_collection_files(
            bundle_results_root=bundle_results_root,
            dataset=dataset,
            logger=MAIN_LOGGER,
        )
    except Exception:
        MAIN_LOGGER.exception('Failed to generate collection files')
        sys.exit(1)

    # Generate global index files
    try:
        failed_labels += generate_global_index_files(
            bundle_results_root=bundle_results_root,
            dataset=dataset,
            logger=MAIN_LOGGER,
        )
    except Exception as exc:
        # The logger's exception() writes the message it is handed and the
        # frames, not the exception's own text, and a supplemental file in
        # another unit is refused with a message naming the file and both
        # units that the frames alone do not carry.
        MAIN_LOGGER.exception('Failed to generate global index files: %s', exc)
        sys.exit(1)

    if failed_labels > 0:
        MAIN_LOGGER.error(
            'Summary generation incomplete: %d label(s) were not written', failed_labels
        )
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
