"""Unit tests for reading the F ring bundle's per-frame pointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from util.nav_verification.bundle_cassini_fring import (
    read_bundle_pointing,
    suppl_path,
)

SUPPL = """This file contains a C-matrix that describes the rotation from the J2000 reference
frame to the camera pointing based upon analysis of the contents of the image.

Source Data Product ID = 1492052683n_calib
Navigation Type = Stars
Navigated Boresight RA = 83.9924 deg (05h35m58.167s)
Navigated Boresight Dec = 1.59145 deg (+001d35m29.205s)
Navigated Boresight Roll = 2.65089 deg
C-Matrix =
   -0.9935781474    0.1032716250    0.0462324183
    0.0430926715   -0.0324309172    0.9985445695
    0.1046206801    0.9941243419    0.0277724004
"""


def test_suppl_path_moves_the_camera_letter_to_the_end() -> None:
    """The bundle names a product by its number with the camera letter last."""
    path = suppl_path(
        'N1492052683', observation_id='ISS_006RI_LPHRLFMOV001_PRIME', bundle_dir=Path('/b')
    )
    assert path.name == '1492052683n_reproj_img_suppl.txt'


def test_suppl_path_names_the_observation_directory_in_lower_case() -> None:
    """The directory is the observation as the bundle spells it."""
    path = suppl_path(
        'N1492052683', observation_id='ISS_006RI_LPHRLFMOV001_PRIME', bundle_dir=Path('/b')
    )
    assert path.parent.name == 'iss_006ri_lphrlfmov001_prime'


def test_suppl_path_ignores_a_version_suffix() -> None:
    """A name carrying its version names the same product."""
    versioned = suppl_path('N1492052683_1_CALIB', observation_id='ISS_006RI', bundle_dir=Path('/b'))
    bare = suppl_path('N1492052683', observation_id='ISS_006RI', bundle_dir=Path('/b'))
    assert versioned == bare


def test_suppl_path_refuses_what_is_not_an_image_name() -> None:
    """A name the bundle's convention cannot describe is refused where it is given."""
    with pytest.raises(ValueError, match='does not name a Cassini ISS image'):
        suppl_path('mosaic', observation_id='ISS_006RI', bundle_dir=Path('/b'))


def test_pointing_comes_from_the_cmatrix(tmp_path: Path) -> None:
    """The full-precision matrix is preferred over the rounded pair."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.from_cmatrix is True


def test_the_cmatrix_boresight_is_its_third_row(tmp_path: Path) -> None:
    """The boresight read is the matrix's third row, normalized."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.boresight == pytest.approx([0.1046206801, 0.9941243419, 0.0277724004], abs=1e-6)


def test_the_navigation_type_is_carried(tmp_path: Path) -> None:
    """How the bundle navigated the frame says how much to believe it."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.navigation_type == 'Stars'


def test_the_rounded_pair_is_the_fallback(tmp_path: Path) -> None:
    """A file quoting no matrix still answers, at the pair's precision."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL.split('C-Matrix')[0])
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.from_cmatrix is False


def test_the_two_spellings_agree(tmp_path: Path) -> None:
    """The matrix and the rounded pair describe the same direction."""
    full = tmp_path / 'full.txt'
    full.write_text(SUPPL)
    rounded = tmp_path / 'rounded.txt'
    rounded.write_text(SUPPL.split('C-Matrix')[0])
    from_matrix = read_bundle_pointing(full)
    from_pair = read_bundle_pointing(rounded)
    assert from_matrix is not None
    assert from_pair is not None
    assert float(np.dot(from_matrix.boresight, from_pair.boresight)) == pytest.approx(1.0, abs=1e-9)


def test_a_file_with_no_pointing_answers_nothing(tmp_path: Path) -> None:
    """A file recording no boresight is not an answer."""
    path = tmp_path / 'suppl.txt'
    path.write_text('Navigation Type = Manual\n')
    assert read_bundle_pointing(path) is None
