#!/usr/bin/env python3
"""Write the cloud-tasks files for Voyager ISS, one per planetary encounter.

The four encounters are navigated as four separate batches because their
volumes are disjoint and their images differ enough in character that an
operator usually wants to run, watch and re-run one encounter at a time.  The
encounter is spelled by the volume set: VGISS_5xxx is Jupiter, 6xxx Saturn,
7xxx Uranus and 8xxx Neptune, each holding both spacecraft's volumes.

Usage:
    make_vgiss_tasks.py --holdings-root gs://BUCKET/holdings [--output-dir DIR]
                        [--planets jupiter,saturn,uranus,neptune]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task_gen_common as common

DATASET_NAME = 'vgiss'

PLANET_VOLUME_DIGIT = {
    'jupiter': '5',
    'saturn': '6',
    'uranus': '7',
    'neptune': '8',
}
"""The digit of a VGISS volume name that names the encounter."""


def selected_planets(names: str) -> list[str]:
    """The encounters named on the command line, each once, in archive order.

    A name repeated on the command line is one encounter, not two: writing its
    file twice would leave one file and count its tasks twice.  The order is
    the archive's rather than the caller's, so that two spellings of the same
    selection report the same thing.

    Parameters:
        names: The comma-separated encounter names, as typed.

    Returns:
        The encounter names, deduplicated, in the order PLANET_VOLUME_DIGIT
        declares them.

    Raises:
        ValueError: If a name is not one of the four encounters.
    """
    wanted = {name.strip().lower() for name in names.split(',') if name.strip()}
    unknown = sorted(wanted - set(PLANET_VOLUME_DIGIT))
    if unknown:
        valid = ', '.join(PLANET_VOLUME_DIGIT)
        raise ValueError(f'Unknown planet(s) {", ".join(unknown)}; valid names: {valid}')
    return [planet for planet in PLANET_VOLUME_DIGIT if planet in wanted]


def planet_volumes(planet: str) -> list[str]:
    """The Voyager volumes of one encounter, in archive order.

    Parameters:
        planet: The encounter name, a key of PLANET_VOLUME_DIGIT.

    Returns:
        The volume names of that encounter.

    Raises:
        ValueError: If the encounter is not one of the four, or the dataset
            declares no volume for it.
    """
    try:
        digit = PLANET_VOLUME_DIGIT[planet]
    except KeyError:
        valid = ', '.join(PLANET_VOLUME_DIGIT)
        raise ValueError(f'Unknown planet "{planet}"; valid names: {valid}') from None
    volumes = [name for name in common.volume_names(DATASET_NAME) if name[6] == digit]
    if not volumes:
        raise ValueError(f'No VGISS volumes for {planet}')
    return volumes


def main() -> None:
    """Write one Voyager ISS task file per selected encounter."""
    parser = argparse.ArgumentParser(
        description='Write the cloud-tasks files for Voyager ISS, one per planet'
    )
    parser.add_argument(
        '--planets',
        default=','.join(PLANET_VOLUME_DIGIT),
        metavar='NAMES',
        help="""Comma-separated encounters to write files for (default: all four:
        jupiter, saturn, uranus, neptune)""",
    )
    common.add_common_arguments(parser)
    arguments = parser.parse_args()

    try:
        planets = selected_planets(arguments.planets)
    except ValueError as exc:
        parser.error(str(exc))
    if not planets:
        parser.error('--planets selected no encounters')

    written = []
    local_root = None
    for planet in planets:
        volumes = planet_volumes(planet)
        output_path = arguments.output_dir / f'{DATASET_NAME}_tasks_{planet}.json'
        print(
            f'Enumerating {planet} ({len(volumes)} volumes, '
            f'{volumes[0]} to {volumes[-1]}) into {output_path}'
        )
        task_file = common.write_task_file(
            output_path,
            arguments=arguments,
            dataset_name=DATASET_NAME,
            volumes=volumes,
        )
        written.append((output_path, task_file.count))
        local_root = task_file.local_root or local_root

    common.report_files(written, local_root=local_root, holdings_root=arguments.holdings_root)


if __name__ == '__main__':
    main()
