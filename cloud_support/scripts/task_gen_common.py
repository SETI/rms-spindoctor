"""Shared machinery for the cloud-task generator scripts.

The generators do not enumerate images themselves.  Each one runs
``sd_offset --output-cloud-tasks-file``, which is the program that defines the
task JSON, and then copies or divides what that program wrote.  A change to
the task schema reaches these scripts without their being touched, and a task
file is the same document whether it was made here or by hand.

Every generator requires the holdings root a cloud worker will read, and that
root governs the task file alone.  Enumeration reads the holdings this machine
is already configured for, through PDS3_HOLDINGS_DIR or the configuration, the
same as any other sd_offset run; the URLs that come back are pointed at the
worker's holdings before the file is written.  The two are rarely the same
place, and a task carries its URLs absolute, so writing the local ones would
leave a file of paths no instance can resolve.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

from filecache import FCPath

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run from a source tree without requiring an installed package, the same way
# the sd_* dispatch modules do.
sys.path.insert(0, str(REPO_ROOT / 'src'))

from spindoctor.dataset import dataset_name_to_class  # noqa: E402

Task = dict[str, Any]
"""One entry of a cloud-tasks JSON array."""


class TaskFile(NamedTuple):
    """What one written task file amounts to.

    Attributes:
        count: The number of tasks in it.
        local_root: The holdings root the enumeration read, which the tasks'
            URLs were re-rooted away from.  None when the file holds no tasks
            and nothing was re-rooted.
    """

    count: int
    local_root: str | None


def volume_names(dataset_name: str) -> tuple[str, ...]:
    """The PDS3 volumes a dataset covers, in archive order.

    The dataset classes carry the authoritative list, so it is read from them
    rather than repeated here; a volume added to the archive is picked up by
    editing the dataset class alone.

    Parameters:
        dataset_name: The dataset name, as spelled on an sd_* command line.

    Returns:
        The volume names, in the order the dataset class declares them.

    Raises:
        ValueError: If the dataset declares no volumes.
        KeyError: If no dataset goes by that name.  Each generator names its
            own dataset, so this reports a mistake in a script rather than
            anything a caller typed.
    """
    dataset_class = dataset_name_to_class(dataset_name)
    volumes: tuple[str, ...] = getattr(dataset_class, '_ALL_VOLUME_NAMES', ())
    if not volumes:
        raise ValueError(f'Dataset "{dataset_name}" declares no volumes')
    return tuple(volumes)


def volumes_dir_name(dataset_name: str) -> str:
    """The holdings subdirectory a dataset's products live under.

    ``volumes`` for most instruments and ``calibrated`` for Cassini, which is
    navigated from its calibrated products.  It is the first path segment below
    the holdings root, and so the seam at which one holdings root can be
    exchanged for another.

    Parameters:
        dataset_name: The dataset name, as spelled on an sd_* command line.

    Returns:
        The subdirectory name.

    Raises:
        ValueError: If the dataset declares none.
        KeyError: If no dataset goes by that name.
    """
    dataset_class = dataset_name_to_class(dataset_name)
    name: str = getattr(dataset_class, '_VOLUMES_DIR_NAME', '')
    if not name:
        raise ValueError(f'Dataset "{dataset_name}" declares no volumes directory')
    return name


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments every generator accepts.

    Parameters:
        parser: The parser to add the arguments to.
    """
    parser.add_argument(
        '--holdings-root',
        required=True,
        type=holdings_root_argument,
        metavar='URL',
        help="""The PDS3 holdings root the cloud workers will read, for example
        a gs:// mirror or https://pds-rings.seti.org/holdings.  Required, and it
        governs the task file alone: every image and label URL is written under
        it, while the enumeration reads the holdings this machine is configured
        for through PDS3_HOLDINGS_DIR or the configuration.""",
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        type=FCPath,
        metavar='DIR',
        help="""Directory to write the task file(s) into; a cloud URL is
        allowed, since a task file is loaded into a queue by URL as readily as
        by path (default: the current directory)""",
    )
    parser.add_argument(
        '--nav-models',
        default=None,
        metavar='PATTERNS',
        help='Value for sd_offset --nav-models, recorded in each task',
    )
    parser.add_argument(
        '--nav-techniques',
        default=None,
        metavar='PATTERNS',
        help='Value for sd_offset --nav-techniques, recorded in each task',
    )
    parser.add_argument(
        '--sd-offset',
        default='sd_offset',
        metavar='PROGRAM',
        help='The sd_offset program to run (default: sd_offset from PATH)',
    )
    parser.add_argument(
        'sd_offset_args',
        nargs='*',
        metavar='SD_OFFSET_ARG',
        help="""Further arguments passed through to sd_offset after a bare --,
        for example "-- --has-no-offset-file" to select only images that have
        not been navigated yet""",
    )


def write_task_file(
    output_path: FCPath,
    *,
    arguments: argparse.Namespace,
    dataset_name: str,
    volumes: list[str] | None = None,
) -> TaskFile:
    """Write one cloud-task file by running sd_offset, re-rooted for the workers.

    Parameters:
        output_path: The task file to write.
        arguments: The parsed command line, carrying the common arguments.
        dataset_name: The dataset name to enumerate.
        volumes: The volumes to restrict the enumeration to.  None enumerates
            every volume the dataset covers.

    Returns:
        The number of tasks written and the holdings root they were enumerated
        from.

    Raises:
        SystemExit: If sd_offset fails, reporting the command that failed -- a
            generator has nothing to write once an enumeration has failed, and
            the command is what an operator needs to see, not a traceback
            through this module -- or if the URLs it wrote cannot be re-rooted.
    """
    command = [
        arguments.sd_offset,
        dataset_name,
        '--output-cloud-tasks-file',
        output_path.as_posix(),
        # Enumeration is the whole job here, and its progress is reported by
        # this script; a log file would also demand a results root that a
        # generating run has no other use for.
        '--no-log-main-to-console',
        '--no-log-main-to-file',
    ]
    if volumes is not None:
        command += ['--volumes', ','.join(volumes)]
    if arguments.nav_models is not None:
        command += ['--nav-models', arguments.nav_models]
    if arguments.nav_techniques is not None:
        command += ['--nav-techniques', arguments.nav_techniques]
    command += list(arguments.sd_offset_args)

    make_directory(output_path.parent)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        printable = ' '.join(shlex.quote(part) for part in command)
        raise SystemExit(
            f'sd_offset exited with status {result.returncode}; the command was:\n  {printable}'
        )
    tasks = read_task_file(output_path)
    local_root = retarget_urls(
        tasks,
        volumes_dir=volumes_dir_name(dataset_name),
        holdings_root=arguments.holdings_root,
        path=output_path,
    )
    save_task_file(tasks, output_path)
    return TaskFile(count=len(tasks), local_root=local_root)


def retarget_urls(
    tasks: list[Task],
    *,
    volumes_dir: str,
    holdings_root: str,
    path: FCPath,
) -> str | None:
    """Point every URL in the tasks at the holdings a cloud worker reads.

    The enumeration reads this machine's holdings and writes their paths into
    the tasks.  A worker reads neither, so each URL is re-rooted here: the part
    from the volumes directory onward is what identifies the file, and the part
    before it is one holdings root being exchanged for another.

    The local root is read back from the URLs rather than resolved from the
    configuration, so it is what the enumeration actually did and not what this
    script believes it should have done.  Every URL has to agree on it; two
    roots in one file would mean the run enumerated from somewhere this cannot
    account for, and re-rooting only some of them would produce a file that is
    half wrong and looks whole.

    Parameters:
        tasks: The tasks to re-root, modified in place.
        volumes_dir: The holdings subdirectory the dataset's products live
            under, which is where the root ends.
        holdings_root: The holdings root to write.
        path: The file the tasks came from, named in any message.

    Returns:
        The local holdings root that was replaced, or None if there were no
        tasks to re-root.

    Raises:
        SystemExit: If a URL does not lie under a volumes directory, or if the
            URLs do not agree on one local root.
    """
    marker = f'/{volumes_dir}/'
    target = holdings_root.strip().rstrip('/')
    if not target:
        raise SystemExit(f'{path} cannot be written under a holdings root that names nowhere')
    local_roots: set[str] = set()
    for task in tasks:
        for task_file in task['data']['files']:
            for key in ('image_file_url', 'label_file_url'):
                url = str(task_file[key])
                # From the right: a holdings root may itself contain a path
                # segment spelled like the volumes directory, while nothing
                # below one ever does.
                local_root, separator, remainder = url.rpartition(marker)
                if not separator:
                    raise SystemExit(
                        f'{path} holds a URL that lies under no "{volumes_dir}" '
                        f'directory, so its holdings root cannot be told from the '
                        f'rest of it:\n  {url}'
                    )
                local_roots.add(local_root)
                task_file[key] = f'{target}{marker}{remainder}'
    if len(local_roots) > 1:
        raise SystemExit(
            f'{path} holds URLs under more than one holdings root, which no single '
            f'root can replace:\n  ' + '\n  '.join(sorted(local_roots))
        )
    return local_roots.pop() if local_roots else None


def make_directory(directory: FCPath) -> None:
    """Create a local directory, and leave a remote one alone.

    A cloud object store has no directories to create; writing the object is
    what makes its prefix exist.

    Parameters:
        directory: The directory to create.
    """
    if directory.is_local():
        directory.mkdir(parents=True, exist_ok=True)


def holdings_root_argument(value: str) -> str:
    """Accept a holdings root that can front a URL.

    Parameters:
        value: The root as typed.

    Returns:
        The root with any trailing separator removed.

    Raises:
        argparse.ArgumentTypeError: If the root is blank.  argparse enforces
            that the option is given; only this can say that what was given
            names nowhere, and an empty root would otherwise be written into
            every URL as no root at all.
    """
    root = value.strip().rstrip('/')
    if not root:
        raise argparse.ArgumentTypeError('--holdings-root names no holdings')
    return root


def read_task_file(path: FCPath) -> list[Task]:
    """Read a cloud-task file.

    Parameters:
        path: The task file to read.

    Returns:
        The tasks it holds.
    """
    with path.open(mode='r', encoding='utf-8') as file:
        tasks: list[Task] = json.load(file)
    return tasks


def save_task_file(tasks: list[Task], path: FCPath) -> None:
    """Write tasks as a cloud-tasks JSON file, checking their ids are unique.

    A queue addresses a task by its id, so two tasks sharing one is a defect
    that must not reach the queue.

    Parameters:
        tasks: The tasks to write.
        path: The file to write them to.

    Raises:
        ValueError: If two tasks share a task id.
    """
    task_ids = [task['task_id'] for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f'Duplicate task ids in {path}')
    make_directory(path.parent)
    with path.open(mode='w', encoding='utf-8') as file:
        json.dump(tasks, file, indent=2)


def renumber_tasks(tasks: list[Task], *, first_number: int = 0) -> int:
    """Renumber task ids so that they are unique across merged files.

    sd_offset numbers the tasks of one invocation from zero, so merging the
    output of several invocations can pair one image name with a number
    already used by another.  The trailing number is replaced with a count
    running across everything one generator writes.

    Parameters:
        tasks: The tasks to renumber, modified in place.
        first_number: The number to give the first task.

    Returns:
        The number the next call should start from.
    """
    number = first_number
    for task in tasks:
        stem = str(task['task_id']).rsplit('-', 1)[0]
        task['task_id'] = f'{stem}-{number}'
        number += 1
    return number


def group_volumes(volume_counts: list[tuple[str, int]], target: int) -> list[tuple[list[str], int]]:
    """Divide volumes into consecutive groups of roughly ``target`` images.

    A volume is never split, so a group holds whole volumes and its size is
    only approximately the target.  Two things keep the groups near each other
    in size rather than merely near the target: the images are spread evenly
    over as many groups as the target implies, so that the last group is not
    left with whatever remains; and a volume that would overshoot a group is
    held back for the next one whenever stopping short lands closer to the
    target than going over.  A remainder too small to be worth a queue of its
    own, under a fifth of a group, joins the group before it.

    Parameters:
        volume_counts: The volumes in archive order, each with its image count.
        target: The image count to aim at for each group.

    Returns:
        One entry per group, each the group's volume names and its image count.

    Raises:
        ValueError: If the target is not positive.
    """
    if target <= 0:
        raise ValueError(f'Group size must be positive, got {target}')

    counted = [(volume, count) for volume, count in volume_counts if count > 0]
    if not counted:
        return []
    total = sum(count for _, count in counted)
    even_target = total / max(1, round(total / target))

    groups: list[tuple[list[str], int]] = []
    volumes: list[str] = []
    count = 0
    for volume, volume_count in counted:
        if volumes and abs(count - even_target) <= abs(count + volume_count - even_target):
            groups.append((volumes, count))
            volumes = []
            count = 0
        volumes.append(volume)
        count += volume_count
        if count >= even_target:
            groups.append((volumes, count))
            volumes = []
            count = 0
    if volumes:
        if groups and count < even_target / 5:
            last_volumes, last_count = groups[-1]
            groups[-1] = (last_volumes + volumes, last_count + count)
        else:
            groups.append((volumes, count))
    return groups


def report_files(
    written: list[tuple[FCPath, int]],
    *,
    local_root: str | None = None,
    holdings_root: str | None = None,
) -> None:
    """Print what was written, one line per file plus a total.

    Parameters:
        written: Each file written, with the number of tasks in it.
        local_root: The holdings root the enumeration read.
        holdings_root: The holdings root the tasks were written under.
    """
    if not written:
        print('No task files written (the selection matched no images)')
        return
    width = max(len(path.name) for path, _ in written)
    print()
    if local_root is not None and holdings_root is not None:
        print(f'Enumerated from {local_root}')
        print(f'URLs written as {holdings_root.rstrip("/")}')
        print()
    print('Task files written:')
    for path, count in written:
        print(f'  {path.name:<{width}}  {count:>7,} tasks  ({path.parent})')
    print(f'  {"TOTAL":<{width}}  {sum(count for _, count in written):>7,} tasks')
