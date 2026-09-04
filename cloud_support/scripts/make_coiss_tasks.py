#!/usr/bin/env python3
"""Write the cloud-tasks files for Cassini ISS in groups of whole volumes.

Cassini ISS is far too large to navigate as one batch, so its volumes are
divided into consecutive groups holding roughly a fixed number of images each
(50,000 by default).  A volume is never split across groups, so the group
sizes are approximate; each group is one queue's worth of work, named for the
first and last volume it covers.

The image count of a volume is not recorded anywhere that can be read cheaply,
so each volume is enumerated in turn and counted, and the groups are formed
from the counts that enumeration produced.

Usage:
    make_coiss_tasks.py --holdings-root gs://BUCKET/holdings [--output-dir DIR]
                        [--group-size N] [--volumes VOL,VOL]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from filecache import FCPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task_gen_common as common

DEFAULT_GROUP_SIZE = 50000


def main() -> None:
    """Write the Cassini ISS task files, one per volume group."""
    parser = argparse.ArgumentParser(
        description='Write the cloud-tasks files for Cassini ISS in groups of whole volumes'
    )
    parser.add_argument(
        '--dataset',
        default='coiss',
        choices=['coiss', 'coiss_cruise', 'coiss_saturn'],
        help="""The Cassini dataset to cover: all volumes, the cruise volumes
        (COISS_1001-1009) or the Saturn volumes (COISS_2001-2116)
        (default: coiss, meaning all of them)""",
    )
    parser.add_argument(
        '--group-size',
        type=int,
        default=DEFAULT_GROUP_SIZE,
        metavar='N',
        help=f'Images to aim for in each group (default: {DEFAULT_GROUP_SIZE:,})',
    )
    parser.add_argument(
        '--volumes',
        default=None,
        metavar='VOL,VOL',
        help="""Comma-separated volumes to restrict the run to, for making a
        small task file to try the pipeline with (default: every volume of the
        dataset)""",
    )
    parser.add_argument(
        '--work-dir',
        default=None,
        metavar='DIR',
        help="""Directory for the per-volume task files the groups are built
        from; it is kept after the run (default: a temporary directory that is
        deleted)""",
    )
    common.add_common_arguments(parser)
    arguments = parser.parse_args()

    if arguments.group_size <= 0:
        parser.error('--group-size must be positive')

    all_volumes = common.volume_names(arguments.dataset)
    if arguments.volumes is None:
        volumes = list(all_volumes)
    else:
        wanted = [name.strip().upper() for name in arguments.volumes.split(',') if name.strip()]
        unknown = [name for name in wanted if name not in all_volumes]
        if unknown:
            parser.error(f'Not {arguments.dataset} volumes: {", ".join(unknown)}')
        volumes = [name for name in all_volumes if name in wanted]
    if not volumes:
        parser.error('--volumes selected no volumes')

    if arguments.work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix='coiss_tasks_'))
        keep_work_dir = False
    else:
        work_dir = Path(arguments.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        keep_work_dir = True

    try:
        volume_counts, local_root = _enumerate_volumes(
            volumes, arguments=arguments, work_dir=work_dir
        )
        groups = common.group_volumes(volume_counts, arguments.group_size)
        written = _write_groups(groups, arguments=arguments, work_dir=work_dir)
    finally:
        if not keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    common.report_files(written, local_root=local_root, holdings_root=arguments.holdings_root)


def _enumerate_volumes(
    volumes: list[str], *, arguments: argparse.Namespace, work_dir: Path
) -> tuple[list[tuple[str, int]], str | None]:
    """Enumerate each volume into its own task file and count its images.

    Parameters:
        volumes: The volumes to enumerate, in archive order.
        arguments: The parsed command line.
        work_dir: The directory to write the per-volume task files into.

    Returns:
        Each volume with the number of images enumerated for it, in the order
        the volumes were given, and the holdings root they were enumerated
        from.
    """
    volume_counts: list[tuple[str, int]] = []
    local_root = None
    running_total = 0
    for index, volume in enumerate(volumes, start=1):
        task_file = common.write_task_file(
            FCPath(work_dir / f'{volume}.json'),
            arguments=arguments,
            dataset_name=arguments.dataset,
            volumes=[volume],
        )
        volume_counts.append((volume, task_file.count))
        local_root = task_file.local_root or local_root
        running_total += task_file.count
        print(
            f'  [{index:>3}/{len(volumes)}] {volume}: {task_file.count:>6,} images '
            f'({running_total:,} so far)'
        )
    return volume_counts, local_root


def _write_groups(
    groups: list[tuple[list[str], int]], *, arguments: argparse.Namespace, work_dir: Path
) -> list[tuple[FCPath, int]]:
    """Merge the per-volume task files of each group into one task file.

    Parameters:
        groups: Each group's volume names and image count.
        arguments: The parsed command line.
        work_dir: The directory holding the per-volume task files.

    Returns:
        Each file written, with the number of tasks in it.
    """
    written: list[tuple[Path, int]] = []
    task_number = 0
    for index, (group_volumes, count) in enumerate(groups, start=1):
        tasks: list[common.Task] = []
        for volume in group_volumes:
            tasks += common.read_task_file(FCPath(work_dir / f'{volume}.json'))
        task_number = common.renumber_tasks(tasks, first_number=task_number)
        output_path = arguments.output_dir / (
            f'{arguments.dataset}_tasks_{index:02d}_{group_volumes[0]}_{group_volumes[-1]}.json'
        )
        print(
            f'Group {index}: {len(group_volumes)} volumes, {count:,} images -> {output_path.name}'
        )
        common.save_task_file(tasks, output_path)
        written.append((output_path, len(tasks)))
    return written


if __name__ == '__main__':
    main()
