"""Tests that ``sd_create_bundle`` refuses, in one line, a dataset it cannot bundle.

Two datasets cannot be bundled: one that makes no PDS4 bundle, and one that names a
bundle the configuration gives no entry.  Every pass ends the same way for each, with
one line on stdout and exit status 1, and these drive the program's entry point to show
it.
"""

from pathlib import Path

import pytest

from spindoctor.cli import sd_create_bundle


def _run(
    subcommand: str, dataset_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the program's entry point on one subcommand and dataset.

    Parameters:
        subcommand: The pass to run.
        dataset_name: The dataset to name on the command line.
        tmp_path: Base temporary directory served as every results root the pass reads.
        monkeypatch: Fixture the command line is installed through.
    """
    root = str(tmp_path)
    arguments = ['--bundle-results-root', root]
    if subcommand == 'labels':
        arguments += ['--nav-results-root', root, '--backplane-results-root', root]
    monkeypatch.setattr('sys.argv', ['sd_create_bundle', subcommand, dataset_name, *arguments])
    sd_create_bundle.main()


@pytest.mark.parametrize('subcommand', ['labels', 'summary', 'check'])
def test_a_dataset_that_makes_no_bundle_is_refused_by_every_pass(
    subcommand: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dataset with no bundle ends the pass with one line naming it and status 1."""
    with pytest.raises(SystemExit) as raised:
        _run(subcommand, 'vgiss', tmp_path, monkeypatch)
    assert raised.value.code == 1
    assert capsys.readouterr().out == 'The dataset vgiss does not produce PDS4 bundles\n'


@pytest.mark.parametrize('subcommand', ['labels', 'summary', 'check'])
def test_a_bundle_with_no_configuration_entry_is_refused_by_every_pass(
    subcommand: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bundle key with no entry ends the pass with one line naming the key and status 1."""
    with pytest.raises(SystemExit) as raised:
        _run(subcommand, 'coiss_cruise', tmp_path, monkeypatch)
    assert raised.value.code == 1
    expected = (
        "the pds4 section holds no entry for the bundle 'coiss_cruise', so no bundle can be made"
    )
    assert capsys.readouterr().out.strip().endswith(expected)
