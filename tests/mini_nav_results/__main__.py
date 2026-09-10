"""Write one of the package's document sets, as the regeneration entry point.

Which set, then where to write it, both required for both sets alike: an
argument that changes *which* files are written depending on whether a later
argument is present is exactly the surprise a fixture tool should not hold, and
spelling the statistics path out is what makes regenerating a checked-in
fixture tree something the operator asked for by name.

Every path written is printed, which is the step a change to a writer or to a
document is re-ratified by.  Standard output carries those paths and nothing
else, so the list can be redirected to a file or read by another program; the
writers' own diagnostics, which the run log takes, go to standard error along
with everything else that is not the list.
"""

import argparse
import sys
from pathlib import Path

import pdslogger

from spindoctor.config import MAIN_LOGGER

from . import write_cohort, write_results_tree

_parser = argparse.ArgumentParser(
    prog='python -m tests.mini_nav_results',
    description='Write a document set of the miniature navigation results package.',
)
_parser.add_argument(
    'document_set',
    choices=('results_tree', 'cohort'),
    help=(
        'results_tree writes the eight documents of the statistics fixture tree; '
        'cohort writes the bundle cohort, documents and products alike'
    ),
)
_parser.add_argument(
    'output_directory',
    type=Path,
    help='the root to write the set under',
)
_arguments = _parser.parse_args()

MAIN_LOGGER.replace_handler(pdslogger.stream_handler(stream=sys.stderr))

if _arguments.document_set == 'results_tree':
    _written = write_results_tree(_arguments.output_directory)
else:
    _written = list(write_cohort(_arguments.output_directory).written)

for _written_path in _written:
    print(_written_path)
