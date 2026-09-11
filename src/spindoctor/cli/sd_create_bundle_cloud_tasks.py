#!/usr/bin/env python3
################################################################################
# sd_create_bundle_cloud_tasks.py
#
# PDS4 bundle generator when image batches are provided by cloud_tasks.
################################################################################

import argparse
import asyncio
import os
import sys
from typing import Any, cast

from cloud_tasks.worker import Worker, WorkerData
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.pds4.bundle_data import BundleDataOutcome, generate_bundle_data_files
from spindoctor.cli.pds4.collections import unusable_units
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    get_backplane_results_root,
    get_nav_results_root,
    get_pds4_bundle_results_root,
    load_default_and_user_config,
)
from spindoctor.dataset import dataset_name_to_class
from spindoctor.dataset.dataset import ImageFile, ImageFiles


def process_task(
    _task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Generate bundle files for a single batch of image files.

    The configured backplane units are checked once the dataset is constructed,
    before any image is read.  A unit no global index column has a format for is
    unusable for every image, and the check each document gets covers only the
    planes that document holds, so a task that did not make this one would write
    labels the summary pass then refuses to index.

    Parameters:
        _task_id: The queue's identifier for the task, unused.
        task_data: The task: ``dataset_name`` and ``files``, each file carrying
            ``image_file_url``, ``label_file_url`` and ``results_path_stub``, and
            optionally ``index_file_row``.
        worker_data: The worker's data, whose ``args`` is the parsed command line
            the configuration and the results roots are read from.

    Returns:
        ``(retry, result)``, where ``retry`` is always False.  ``result`` is
        ``{'status': 'success'}`` when the image's products were written or the
        image was skipped as one the bundle has nothing to describe, and
        otherwise ``{'status': 'error', 'status_error': ...}``:
        ``unusable_unit`` when a configured backplane declares no unit or one no
        index column has a format for, with every such backplane and its reason
        in ``status_exception`` and nothing generated; ``label_not_written`` when
        a product could not be written; and ``no_nav_root``,
        ``no_backplane_root``, ``no_bundle_root``, ``no_dataset_name``,
        ``unknown_dataset`` (with ``status_exception``), ``no_files``,
        ``no_image_file_url``, ``no_label_file_url`` or ``no_results_path_stub``
        when the task names too little to run.

    Raises:
        Exception: Whatever generation raises, for a document it cannot read or a
            template it cannot find; the task does not catch it.
    """

    arguments = cast(argparse.Namespace, worker_data.args)
    load_default_and_user_config(arguments, DEFAULT_CONFIG)

    # Derive roots
    try:
        nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {'status': 'error', 'status_error': 'no_nav_root'}
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    try:
        backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {'status': 'error', 'status_error': 'no_backplane_root'}
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    try:
        bundle_results_root_str = get_pds4_bundle_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {'status': 'error', 'status_error': 'no_bundle_root'}
    bundle_results_root = FileCache(None).new_path(bundle_results_root_str)

    dataset_name = task_data.get('dataset_name')
    if dataset_name is None:
        return False, {'status': 'error', 'status_error': 'no_dataset_name'}
    try:
        dataset = dataset_name_to_class(dataset_name)()
    except KeyError:
        return False, {
            'status': 'error',
            'status_error': 'unknown_dataset',
            'status_exception': f'Unknown dataset "{dataset_name}"',
        }

    # The two sd_create_bundle passes refuse such a configuration before the
    # run reads anything; a task is the whole of what this worker holds, so it
    # refuses it per task, and no retry, since the configuration will not change.
    unusable = unusable_units(dataset.config)
    if len(unusable) > 0:
        return False, {
            'status': 'error',
            'status_error': 'unusable_unit',
            'status_exception': '; '.join(
                f'Backplane {name} declares no units'
                if reason is None
                else f'Backplane {name} declares a unit the bundle cannot use: {reason}'
                for name, reason in unusable
            ),
        }

    files = task_data.get('files')
    if files is None:
        return False, {'status': 'error', 'status_error': 'no_files'}
    image_files = []
    for file in files:
        image_file_url = file.get('image_file_url', None)
        label_file_url = file.get('label_file_url', None)
        results_path_stub = file.get('results_path_stub', None)
        index_file_row = file.get('index_file_row', None)
        if image_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_image_file_url'}
        if label_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_label_file_url'}
        if results_path_stub is None:
            return False, {'status': 'error', 'status_error': 'no_results_path_stub'}
        image_file = ImageFile(
            image_file_url=FCPath(image_file_url),
            label_file_url=FCPath(label_file_url),
            results_path_stub=results_path_stub,
            index_file_row=index_file_row,
        )
        image_files.append(image_file)

    outcome = generate_bundle_data_files(
        dataset=dataset,
        image_files=ImageFiles(image_files=image_files),
        nav_results_root=nav_results_root,
        backplane_results_root=backplane_results_root,
        bundle_results_root=bundle_results_root,
        logger=IMAGE_LOGGER,
    )
    # Neither result asks for a retry, under any circumstances: a product whose
    # labels are on disk has nothing left to do, and a label the template could
    # not render will not render on a second attempt either.
    if outcome is BundleDataOutcome.FAILED:
        return False, {'status': 'error', 'status_error': 'label_not_written'}

    return False, {'status': 'success'}


async def async_main() -> None:
    argparser = argparse.ArgumentParser(
        description='PDS4 Bundle Generation Main Interface (Cloud Tasks version)'
    )

    environment_group = argparser.add_argument_group('Environment')
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
        help='Root directory for prior navigation results (metadata, offsets)',
    )
    environment_group.add_argument(
        '--backplane-results-root',
        type=str,
        default=None,
        help='Root directory for backplane results',
    )
    environment_group.add_argument(
        '--bundle-results-root',
        type=str,
        default=None,
        help='Root directory for bundle results',
    )

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    """Synchronous entry point; runs ``asyncio.run(async_main())``."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
