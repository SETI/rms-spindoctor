"""Tests that ``sd_create_bundle`` refuses, in one line, a dataset it cannot bundle.

Two kinds of dataset cannot be bundled: one that does not make PDS4 bundles, and one
whose bundle has no entry in the configuration.  For each, every pass ends with one line
and exit status 1; the labels pass writes its line through the run logger.  These tests
drive the program's entry point.
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
    """A dataset that does not make bundles ends the pass with one line giving its name."""
    with pytest.raises(SystemExit) as raised:
        _run(subcommand, 'vgiss', tmp_path, monkeypatch)
    assert raised.value.code == 1
    assert capsys.readouterr().out == 'The dataset vgiss does not produce PDS4 bundles\n'


@pytest.mark.parametrize('subcommand', ['summary', 'check'])
def test_a_bundle_with_no_configuration_entry_is_refused_by_the_quiet_passes(
    subcommand: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bundle key that has no entry ends the pass with one line giving the key."""
    with pytest.raises(SystemExit) as raised:
        _run(subcommand, 'coiss_cruise', tmp_path, monkeypatch)
    assert raised.value.code == 1
    expected = (
        "the pds4 section has no entry for the bundle 'coiss_cruise', so no bundle can be made"
    )
    assert capsys.readouterr().out.strip().endswith(expected)


def test_the_labels_run_log_records_the_missing_entry_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The labels pass writes the missing-entry refusal into its run log and exits 1."""
    root = tmp_path.as_posix()
    log_root = tmp_path / 'logs'
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_bundle',
            'labels',
            'coiss_cruise',
            '--bundle-results-root',
            root,
            '--nav-results-root',
            root,
            '--backplane-results-root',
            root,
            '--log-root',
            log_root.as_posix(),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        sd_create_bundle.main()
    assert raised.value.code == 1
    refusal = "has no entry for the bundle 'coiss_cruise', so no bundle can be made"
    logs = sorted(log_root.rglob('main_*.log'))
    assert len(logs) == 1
    assert refusal in logs[0].read_text()
