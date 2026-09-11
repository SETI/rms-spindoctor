"""Which programs offer ``--pds3-holdings-root``, and whether what it names is read.

The option belongs to ``DataSetPDS3``, the only class it means anything to.  A
program therefore offers it exactly when the dataset it was asked for enumerates
a PDS3 holdings tree, and what an operator typed lands on the namespace that
dataset reads.

Six programs used to declare it in an environment group of their own and none of
them read it, so every value typed was parsed into a namespace, dropped, and
replaced by whatever the machine's ``PDS3_HOLDINGS_DIR`` said -- with no error,
no warning and no log line, for as long as the run lasted.  What made that
survivable to write and invisible to test is that a machine configured with one
holdings root cannot tell the two apart, so every assertion here names two roots
and says which one won.

The surface is asserted two ways, because either alone passes while the defect
is present.  A scan says the dataset is the only place the option is declared,
which is the half that keeps failing for a program added later.  Each program is
then driven through its own parser and its namespace handed to a real dataset,
because an option that parses proves only that argparse accepted the word: what
a run does with the value is what was missing.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest
from filecache import FCPath
from tests.spindoctor.cli.conftest import help_text
from tests.spindoctor.dataset.conftest import coiss_filespecs, install_fake_index

from spindoctor.cli import (
    sd_backplane_viewer,
    sd_backplanes,
    sd_consolidate_metadata,
    sd_create_bundle,
    sd_mosaic,
    sd_offset,
)
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS

_SOURCE_ROOT = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

_OPTION = '--pds3-holdings-root'

_TYPED_ROOT = '/typed/holdings'
"""The root a command line names, which is what every run below must read."""

_EXPORTED_ROOT = '/exported/holdings'
"""The root the machine names, which every run below must read in preference to."""

_DECLARING_MODULES = {'dataset/dataset_pds3.py'}
"""The one module allowed to declare the option, relative to the package root.

A program declares it by offering the selection arguments of a dataset that
reads PDS3 holdings, never by writing the option itself: an option a program
declares lands on the namespace under the program's own help text, and nothing
downstream is obliged to read it.  That is the shape the defect had.
"""

# Every program whose selection enumerates PDS3 images, named by the argv its
# own parser is driven with.  The dataset name is COISS throughout because what
# is under test is the program's surface rather than any one mission's.
_ENUMERATING: list[tuple[str, list[str]]] = [
    ('sd_offset', ['coiss_saturn']),
    ('sd_backplanes', ['coiss_saturn']),
    ('sd_consolidate_metadata', ['coiss_saturn', '--dest-dir', 'out', '--copy-all']),
    ('sd_create_bundle', ['coiss_saturn']),
    ('sd_backplane_viewer', ['coiss_saturn']),
    ('sd_mosaic', ['rings', 'coiss_saturn', '--output-dir', 'out', '--planet', 'SATURN']),
    ('sd_mosaic', ['body', 'coiss_saturn', '--output-dir', 'out', '--body-name', 'RHEA']),
]

# The same programs, named by the argv that reaches the help text each one
# prints.  Two of them read a subcommand from argv before their dataset, so the
# line a user types is not the line the parser is handed.
_ENUMERATING_HELP: list[tuple[str, list[str]]] = [
    ('sd_offset', ['coiss_saturn']),
    ('sd_backplanes', ['coiss_saturn']),
    ('sd_consolidate_metadata', ['coiss_saturn']),
    ('sd_create_bundle', ['labels', 'coiss_saturn']),
    ('sd_backplane_viewer', ['coiss_saturn']),
    ('sd_mosaic', ['rings', 'coiss_saturn']),
    ('sd_mosaic', ['body', 'coiss_saturn']),
]

# The navigator asked for the one dataset that reads no PDS3 holdings.  A
# simulated scene is named by its own path, so a holdings root would be an
# option promising a tree the run never opens.
_ENUMERATING_NOTHING_HELP: list[tuple[str, list[str]]] = [
    ('sd_offset', ['sim']),
]

_PARSERS: dict[str, Callable[[list[str]], argparse.Namespace]] = {
    'sd_offset': sd_offset.parse_args,
    'sd_backplanes': sd_backplanes.parse_args,
    'sd_consolidate_metadata': sd_consolidate_metadata.parse_args,
    'sd_create_bundle': sd_create_bundle.parse_args_labels,
    'sd_backplane_viewer': sd_backplane_viewer.parse_args,
    'sd_mosaic': lambda argv: sd_mosaic.parse_args(argv)[1],
}
"""Each program's own parser, so a surface is read from what the program builds."""


def _declares_the_option(module: FCPath) -> bool:
    """Whether one module's source declares the option to argparse.

    Parameters:
        module: The module to read.

    Returns:
        True when the source writes the option as a quoted literal, which is
        how ``add_argument`` is given one.  A mention in prose or in a help
        string is not a declaration and does not count.
    """
    with module.open('r', encoding='utf-8') as source:
        text = source.read()
    return f"'{_OPTION}'" in text or f'"{_OPTION}"' in text


def _modules_declaring_the_option() -> set[str]:
    """Return every package module that declares the option.

    Returns:
        Paths relative to the package root, using forward slashes.
    """
    return {
        module.relative_to(_SOURCE_ROOT).as_posix()
        for module in _SOURCE_ROOT.rglob('*.py')
        if _declares_the_option(module)
    }


def _one_line(program: str, argv: list[str]) -> str:
    """Return a program's help with its wrapping removed.

    argparse rewraps every help string to the terminal width, so a phrase is
    only reliably searchable once the line breaks are gone.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.

    Returns:
        The help text as one space-separated line.
    """
    return ' '.join(help_text(program, argv).split())


def _enumerated_image_url(arguments: argparse.Namespace, monkeypatch: pytest.MonkeyPatch) -> str:
    """Return the URL of the one image a program's own namespace enumerates.

    The index reads are served from memory, so the only thing the holdings root
    decides is the URL the enumeration builds -- which is exactly what a run
    would go on to open.

    Parameters:
        arguments: The namespace one program's parser produced.
        monkeypatch: Fixture the index reads are replaced through.

    Returns:
        The image URL, built under whichever root answered.
    """
    ds = DataSetPDS3CassiniISS()
    install_fake_index(ds, monkeypatch, {'COISS_2001': coiss_filespecs('N', [1000000100])})

    groups = list(ds.yield_image_files_from_arguments(arguments))

    return groups[0].image_files[0].image_file_url.as_posix()


def test_only_the_dataset_declares_the_option() -> None:
    """A program that writes the option itself is a program free not to read it.

    Adding a module here means the option was declared somewhere that is not the
    class that resolves the root, which is the defect rather than a variation on
    it.  Asserted as an equality so a declaration that moved is a failure
    whichever way it moved.
    """
    assert _modules_declaring_the_option() == _DECLARING_MODULES


@pytest.mark.parametrize(('program', 'argv'), _ENUMERATING_HELP)
def test_a_program_enumerating_pds3_images_offers_the_option(program: str, argv: list[str]) -> None:
    """Every program that reads a PDS3 holdings tree says where it reads it from.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.
    """
    assert _OPTION in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _ENUMERATING_NOTHING_HELP)
def test_a_program_enumerating_no_pds3_images_does_not_offer_the_option(
    program: str, argv: list[str]
) -> None:
    """The same program asked for a dataset with no holdings offers nothing to name.

    This is what declaring the option on the dataset buys: an option is absent
    where it would mean nothing, rather than accepted and quietly ignored.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.
    """
    assert _OPTION not in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _ENUMERATING_NOTHING_HELP)
def test_the_help_the_absence_is_read_from_is_the_program_s_own(
    program: str, argv: list[str]
) -> None:
    """The control for it, which a usage error would otherwise satisfy.

    A program that reads its dataset from argv before parsing prints a usage
    error instead of its help when it is given something it cannot use, and
    every option is then absent from a string naming none of them.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.
    """
    assert '--nav-results-root' in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _ENUMERATING)
def test_the_root_a_program_parses_is_the_root_it_enumerates(
    program: str, argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What each program's own parser produced is what its enumeration reads.

    The machine names a holdings root too, and a different one, so a namespace
    that reached the dataset carrying nothing usable would still enumerate --
    from the exported root, which is the failure this asserts against rather
    than an error anybody would see.  That the exported root is live, and is
    what answers when nothing else does, is pinned beside the resolution ladder
    itself in the dataset's own tests.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: The program's own command line, before the two options added here.
        monkeypatch: Fixture the environment and the index reads are set through.
    """
    monkeypatch.setenv('PDS3_HOLDINGS_DIR', _EXPORTED_ROOT)
    arguments = _PARSERS[program](
        [*argv, _OPTION, _TYPED_ROOT, '--volumes', 'COISS_2001'],
    )

    assert _enumerated_image_url(arguments, monkeypatch).startswith(f'{_TYPED_ROOT}/')
