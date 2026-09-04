#!/usr/bin/env python3
"""Write the cloud-tasks file for every New Horizons LORRI image.

New Horizons LORRI is small enough to navigate as one batch, so this writes a single
task file covering all NH..LO_2001 volumes.

Usage:
    make_nhlorri_tasks.py --holdings-root gs://BUCKET/holdings [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task_gen_common as common

DATASET_NAME = 'nhlorri'


def main() -> None:
    """Write the New Horizons LORRI task file."""
    parser = argparse.ArgumentParser(
        description='Write the cloud-tasks file for every New Horizons LORRI image'
    )
    common.add_common_arguments(parser)
    arguments = parser.parse_args()

    output_path = arguments.output_dir / f'{DATASET_NAME}_tasks.json'
    print(f'Enumerating {DATASET_NAME} into {output_path}')
    written = common.write_task_file(output_path, arguments=arguments, dataset_name=DATASET_NAME)
    common.report_files(
        [(output_path, written.count)],
        local_root=written.local_root,
        holdings_root=arguments.holdings_root,
    )


if __name__ == '__main__':
    main()
