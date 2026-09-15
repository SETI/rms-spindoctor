"""The labels subcommand's ``--check-only``, stood up on stubs.

The report writes nothing and says what it found through what it prints and its exit
status, which is what is under test here; the report on one image's inputs is tested in
``tests/spindoctor/cli/pds4/test_image_inputs.py``.
"""

import argparse
import json
from pathlib import Path

import pytest
from tests.spindoctor.cli.sd_create_bundle_helpers import batch_image_name, refuse, stub_dataset

from spindoctor.cli import sd_create_bundle

STUB = f'res/{batch_image_name(0, 0)}'
"""The results path stub of the one image the stub dataset enumerates."""


@pytest.fixture
def check_only_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand the labels subcommand up with ``--check-only``, over one enumerated image.

    Parameters:
        tmp_path: Base temporary directory holding the navigation root ``nav`` and the
            backplane root ``backplanes``.
        monkeypatch: Fixture the stand-ins are installed through.

    Returns:
        The directory holding the two roots.
    """
    monkeypatch.setattr(
        sd_create_bundle,
        'parse_args_labels',
        lambda _: argparse.Namespace(dry_run=False, check_only=True),
    )
    monkeypatch.setattr(sd_create_bundle, 'load_default_and_user_config', lambda *a: None)
    monkeypatch.setattr(sd_create_bundle, 'get_nav_results_root', lambda *a: str(tmp_path / 'nav'))
    monkeypatch.setattr(
        sd_create_bundle, 'get_backplane_results_root', lambda *a: str(tmp_path / 'backplanes')
    )
    monkeypatch.setattr(sd_create_bundle, 'DATASET', stub_dataset(tmp_path))
    return tmp_path


def _write_inputs(root: Path, *, status: str) -> None:
    """Write the four files of the enumerated image under a directory's two roots.

    Parameters:
        root: The directory holding the two roots.
        status: The status the navigation document records.
    """
    nav = root / 'nav' / STUB
    backplanes = root / 'backplanes' / STUB
    nav.parent.mkdir(parents=True)
    backplanes.parent.mkdir(parents=True)
    nav.with_name(f'{nav.name}_metadata.json').write_text(json.dumps({'status': status}))
    nav.with_name(f'{nav.name}_summary.png').write_bytes(b'\x89PNG stand-in')
    backplanes.with_name(f'{backplanes.name}_backplanes.fits').write_bytes(b'SIMPLE  = T')
    backplanes.with_name(f'{backplanes.name}_backplane_metadata.json').write_text('{}')


def test_check_only_exits_zero_when_every_selected_image_is_complete(
    check_only_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A selection whose every image is complete ends the run normally, with the count."""
    _write_inputs(check_only_run, status='success')
    sd_create_bundle.main_labels()
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == 'Input check: 1 image(s) selected, 1 complete, 0 incomplete'


def test_check_only_exits_one_when_a_selected_image_is_incomplete(
    check_only_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A selection with an incomplete image exits 1, after the count."""
    _write_inputs(check_only_run, status='error')
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == 'Input check: 1 image(s) selected, 0 complete, 1 incomplete'


def test_check_only_writes_nothing_and_needs_no_bundle_root_or_log(
    check_only_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report asks for no bundle results root and no logging, and writes no file."""
    _write_inputs(check_only_run, status='success')
    monkeypatch.setattr(sd_create_bundle, 'get_pds4_bundle_results_root', refuse)
    monkeypatch.setattr(sd_create_bundle, 'build_run_logging', refuse)
    before = sorted(check_only_run.rglob('*'))
    sd_create_bundle.main_labels()
    assert sorted(check_only_run.rglob('*')) == before


@pytest.mark.parametrize(
    ('image_count', 'count'),
    [
        pytest.param(
            0,
            'Input check: 0 image(s) selected, 0 complete, 0 incomplete, '
            '1 batch(es) the labels pass refuses',
            id='empty',
        ),
        pytest.param(
            2,
            'Input check: 2 image(s) selected, 0 complete, 2 incomplete, '
            '1 batch(es) the labels pass refuses',
            id='two',
        ),
    ],
)
def test_check_only_exits_one_over_a_batch_the_labels_pass_refuses(
    check_only_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image_count: int,
    count: str,
) -> None:
    """A batch of other than one image is incomplete, as the labels pass would fail it.

    Parameters:
        check_only_run: The directory holding the two roots.
        monkeypatch: Fixture the batch's dataset is installed through.
        capsys: Fixture the report is read through.
        image_count: How many images the batch holds.
        count: The report's last line.
    """
    monkeypatch.setattr(
        sd_create_bundle, 'DATASET', stub_dataset(check_only_run, image_count=image_count)
    )
    with pytest.raises(SystemExit) as excinfo:
        sd_create_bundle.main_labels()
    assert excinfo.value.code == 1
    assert capsys.readouterr().out.splitlines()[-1] == count
