"""What a selected image has of the files the labels pass reads, as ``--check-only`` says it.

The four files are written by hand under the two roots, at the paths the labels pass
reads them from, so that the paths themselves are pinned here.
"""

import json
from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.cli.pds4.image_inputs import image_inputs, report_image_inputs

from .conftest import make_image_file

STUB = 'res/1234567890w'
"""The image's results path stub."""

INPUTS = {
    'navigation document': f'nav/{STUB}_metadata.json',
    'summary PNG': f'nav/{STUB}_summary.png',
    'backplane FITS': f'backplanes/{STUB}_backplanes.fits',
    'backplane metadata': f'backplanes/{STUB}_backplane_metadata.json',
}
"""Each of the four files, by the name the report gives it, under the test's directory."""


def _write_inputs(root: Path, *, status: str = 'success') -> None:
    """Write the four files of the image under a directory's two roots.

    Parameters:
        root: The directory holding the navigation root ``nav`` and the backplane root
            ``backplanes``.
        status: The status the navigation document records.
    """
    for path in INPUTS.values():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / INPUTS['navigation document']).write_text(json.dumps({'status': status}))
    (root / INPUTS['summary PNG']).write_bytes(b'\x89PNG stand-in')
    (root / INPUTS['backplane FITS']).write_bytes(b'SIMPLE  = T stand-in')
    (root / INPUTS['backplane metadata']).write_text('{}')


def _line(root: Path) -> str:
    """Return the line the report gives the image.

    Parameters:
        root: The directory holding the two roots.

    Returns:
        The line.
    """
    report = report_image_inputs(
        make_image_file(results_path_stub=STUB),
        nav_results_root=FCPath(root / 'nav'),
        backplane_results_root=FCPath(root / 'backplanes'),
    )
    return report.line()


def test_an_image_with_every_input_whose_navigation_succeeded_is_complete(tmp_path: Path) -> None:
    """The four files present and a navigation that succeeded make a complete image."""
    _write_inputs(tmp_path)
    assert _line(tmp_path) == (
        f'{STUB}: navigation document present, summary PNG present, backplane FITS present, '
        'backplane metadata present, navigation succeeded: complete'
    )


@pytest.mark.parametrize('missing', list(INPUTS))
def test_an_image_missing_one_input_is_incomplete(tmp_path: Path, missing: str) -> None:
    """An image without one of the four files is incomplete, the report naming the file.

    Parameters:
        tmp_path: The directory holding the two roots.
        missing: The file removed.
    """
    _write_inputs(tmp_path)
    (tmp_path / INPUTS[missing]).unlink()
    files = ', '.join(f'{name} {"absent" if name == missing else "present"}' for name in INPUTS)
    navigation = (
        'no navigation recorded' if missing == 'navigation document' else 'navigation succeeded'
    )
    assert _line(tmp_path) == f'{STUB}: {files}, {navigation}: incomplete'


def test_an_image_whose_navigation_did_not_succeed_is_incomplete(tmp_path: Path) -> None:
    """An image whose navigation document records another status is incomplete."""
    _write_inputs(tmp_path, status='error')
    assert _line(tmp_path) == (
        f'{STUB}: navigation document present, summary PNG present, backplane FITS present, '
        'backplane metadata present, navigation did not succeed (status error): incomplete'
    )


def test_a_root_given_as_a_string_or_a_path_names_the_files_an_fcpath_names(
    tmp_path: Path,
) -> None:
    """The two roots are taken as a string and as a Path, as well as as an FCPath."""
    nav = tmp_path / 'nav'
    backplanes = tmp_path / 'backplanes'
    expected = image_inputs(
        STUB, nav_results_root=FCPath(nav), backplane_results_root=FCPath(backplanes)
    )
    as_strings = image_inputs(
        STUB, nav_results_root=str(nav), backplane_results_root=str(backplanes)
    )
    as_paths = image_inputs(STUB, nav_results_root=nav, backplane_results_root=backplanes)
    assert as_strings == expected
    assert as_paths == expected


def test_a_report_built_from_string_roots_names_the_same_files(tmp_path: Path) -> None:
    """The report over roots given as strings is the report over the same FCPaths."""
    _write_inputs(tmp_path)
    report = report_image_inputs(
        make_image_file(results_path_stub=STUB),
        nav_results_root=str(tmp_path / 'nav'),
        backplane_results_root=str(tmp_path / 'backplanes'),
    )
    assert report.line() == _line(tmp_path)
