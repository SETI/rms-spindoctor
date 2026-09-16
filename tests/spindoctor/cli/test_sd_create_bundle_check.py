"""The ``check`` subcommand, stood up on stubs.

The subcommand writes nothing and reports through what it prints and its exit status,
which is what is under test here; the check itself is tested in
``tests/spindoctor/cli/pds4/check``.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, NoReturn

import pytest
from tests.spindoctor.cli.sd_create_bundle_helpers import BUNDLE_NAME, refuse, stub_dataset

from spindoctor.cli import sd_create_bundle
from spindoctor.cli.pds4.check import CheckName, Finding, Severity

FINDINGS = [
    Finding(
        'bundle.lblx',
        CheckName.XSD,
        '/Product_Bundle/Identification_Area/Citation_Information/doi',
        "value doesn't match any pattern (line 17)",
    ),
    Finding(
        'bundle.lblx',
        CheckName.INTEGRITY,
        '/Product_Bundle/Reference_List/Internal_Reference/lid_reference',
        'refers to urn:nasa:pds:bundle:document:guide, which no label of the tree declares',
        Severity.WARNING,
    ),
]
"""Two findings a stand-in check reports: an error and a warning."""


@pytest.fixture
def check_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand the check subcommand up on stubs, over an empty bundle directory.

    Parameters:
        tmp_path: Base temporary directory served as the bundle results root.
        monkeypatch: Fixture the stand-ins are installed through.

    Returns:
        The bundle directory.
    """
    monkeypatch.setattr(
        sd_create_bundle,
        'parse_args_check',
        lambda _: argparse.Namespace(dataset_name='stub', schema_dir=None),
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: str(tmp_path))
    dataset = stub_dataset(tmp_path)
    monkeypatch.setattr(sd_create_bundle, 'dataset_name_to_class', lambda _: lambda: dataset)
    bundle_dir = tmp_path / BUNDLE_NAME
    bundle_dir.mkdir()
    return bundle_dir


def _check_finds(monkeypatch: pytest.MonkeyPatch, findings: list[Finding]) -> None:
    """Make the check report the given findings.

    Parameters:
        monkeypatch: Fixture the stand-in is installed through.
        findings: What the check reports.
    """
    monkeypatch.setattr(sd_create_bundle, 'check_bundle', lambda *a, **k: findings)


def test_the_check_prints_each_finding_then_the_counts_and_exits_one(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every finding is one line, then the counts, and an error ends the run with 1."""
    _check_finds(monkeypatch, FINDINGS)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_check()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out.splitlines() == [
        *(finding.line() for finding in FINDINGS),
        f'Bundle check of {check_run}: 1 error(s), 1 warning(s)',
    ]


def test_the_check_exits_zero_over_a_bundle_with_no_finding(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bundle with no finding ends the run normally, with the counts."""
    _check_finds(monkeypatch, [])
    sd_create_bundle.main_check()
    assert capsys.readouterr().out.splitlines() == [
        f'Bundle check of {check_run}: 0 error(s), 0 warning(s)'
    ]


def test_the_check_exits_zero_over_a_bundle_with_warnings_alone(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Warnings are printed and counted, and do not fail the run."""
    _check_finds(monkeypatch, FINDINGS[1:])
    sd_create_bundle.main_check()
    assert capsys.readouterr().out.splitlines()[-1] == (
        f'Bundle check of {check_run}: 0 error(s), 1 warning(s)'
    )


def test_the_check_exits_one_without_a_bundle_directory(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a bundle directory the run exits 1, naming where it looked."""
    check_run.rmdir()
    monkeypatch.setattr(sd_create_bundle, 'check_bundle', refuse)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_check()
    assert excinfo.value.code == 1
    assert f'No bundle directory at {check_run}' in capsys.readouterr().out


def test_the_check_exits_one_over_a_bundle_results_root_that_is_not_local(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A root that is not local is refused by its own name, and nothing is read."""
    monkeypatch.setattr(
        sd_create_bundle, 'get_pds4_bundle_results_root', lambda *a: 'gs://bucket/root'
    )
    monkeypatch.setattr(sd_create_bundle, 'check_bundle', refuse)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_check()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out.splitlines() == [
        f'The check reads a local tree, and gs://bucket/root/{BUNDLE_NAME} is not one: '
        'name a local bundle results root'
    ]


def test_the_check_is_handed_the_bundles_own_local_directory(
    check_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local root reaches the check as the bundle's directory, a local path."""
    seen: list[Path] = []

    def _record(bundle_dir: Path, **kwargs: Any) -> list[Finding]:
        """Stand in for the check, recording the directory it is handed.

        Parameters:
            bundle_dir: The bundle's directory.
            **kwargs: The configuration and the schema directory.

        Returns:
            No finding.
        """
        seen.append(bundle_dir)
        return []

    monkeypatch.setattr(sd_create_bundle, 'check_bundle', _record)
    sd_create_bundle.main_check()
    assert seen == [check_run]


def test_the_check_exits_one_with_the_traceback_of_a_check_that_stops(
    check_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A check that raises ends the run with status 1 and its traceback, and no count."""

    def _stops(*args: Any, **kwargs: Any) -> NoReturn:
        """Stand in for a check that stops part way.

        Parameters:
            *args: The bundle directory.
            **kwargs: The configuration.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError('the check stopped here')

    monkeypatch.setattr(sd_create_bundle, 'check_bundle', _stops)
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_check()
    assert excinfo.value.code == 1
    assert 'RuntimeError: the check stopped here' in capsys.readouterr().out


def test_the_check_takes_the_configuration_arguments(tmp_path: Path) -> None:
    """The check takes a dataset, configuration files, a bundle results root and schemas."""
    arguments = sd_create_bundle.parse_args_check(
        [
            'sim',
            '--config-file',
            'a.yaml',
            '--bundle-results-root',
            str(tmp_path),
            '--schema-dir',
            'copies',
        ]
    )
    taken = (
        arguments.dataset_name,
        arguments.config_file,
        arguments.bundle_results_root,
        arguments.schema_dir,
    )
    assert taken == ('sim', ['a.yaml'], str(tmp_path), 'copies')


def test_the_check_reads_the_schemas_from_the_directory_it_is_given(
    check_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--schema-dir`` is handed to the check as the directory it reads the schemas from."""
    monkeypatch.setattr(
        sd_create_bundle,
        'parse_args_check',
        lambda _: argparse.Namespace(dataset_name='stub', schema_dir='copies'),
    )
    seen: list[Path | None] = []

    def _record(*args: Any, **kwargs: Any) -> list[Finding]:
        """Stand in for the check, recording the schema directory it is handed.

        Parameters:
            *args: The bundle directory.
            **kwargs: The configuration and the schema directory.

        Returns:
            No finding.
        """
        seen.append(kwargs['schema_dir'])
        return []

    monkeypatch.setattr(sd_create_bundle, 'check_bundle', _record)
    sd_create_bundle.main_check()
    assert seen == [Path('copies')]


def test_the_program_runs_the_check_for_its_check_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sd_create_bundle check`` runs the check subcommand."""
    calls: list[str] = []
    monkeypatch.setattr(sd_create_bundle, 'main_check', lambda: calls.append('check'))
    monkeypatch.setattr(sys, 'argv', ['sd_create_bundle', 'check'])
    sd_create_bundle.main()
    assert calls == ['check']
