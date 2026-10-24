"""Unit tests for reading the F ring bundle's per-frame pointing."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import oops
import pytest
from util.nav_verification.boresight import offset_in_camera_px, separation_px
from util.nav_verification.bundle_cassini_fring import (
    NAC_PLATE_SCALE_URAD,
    WAC_AS_PUBLISHED,
    WAC_OFFSET_RESTORED,
    WAC_PLATE_SCALE_URAD,
    bundle_image_names,
    plate_scale_urad,
    read_bundle_pointing,
    restore_old_oops_wac_offset,
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

# A wide angle frame from 2013, as the bundle publishes it.
WAC_SUPPL = """Source Data Product ID = 1738087703w_calib
Image Mid Time (UTC) = 2013-01-28T17:14:09.581Z
Navigation Type = Stars
Navigated Boresight RA = 301.882569 deg (20h07m31.817s)
Navigated Boresight Dec = 49.237139 deg (+049d14m13.699s)
Navigated Boresight Roll = 11.502156 deg
C-Matrix =
    0.7523069074    0.6458196113    0.1301973373
   -0.5613401896    0.5249106856    0.6398171330
    0.3448644786   -0.5544238466    0.7574184377
"""
WAC_CMATRIX = np.array(
    [
        [0.7523069074, 0.6458196113, 0.1301973373],
        [-0.5613401896, 0.5249106856, 0.6398171330],
        [0.3448644786, -0.5544238466, 0.7574184377],
    ]
)
# The same file moved to a 2007 date, when the bundle's matrices need no restoration.
WAC_2007_SUPPL = WAC_SUPPL.replace('2013-01-28T17:14:09.581Z', '2007-03-17T19:32:12.015Z')


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


def test_an_observation_the_bundle_does_not_hold_has_no_frames(tmp_path: Path) -> None:
    """Comparing a pass the bundle never covered is a question, not an error."""
    assert bundle_image_names(observation_id='ISS_006RI', bundle_dir=tmp_path) == []


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
    """The matrix and the rounded pair describe the same direction.

    The tolerance is in pixels because that is the unit the disagreement is
    read in, and it is set just above the third of a NAC pixel the pair is
    quantized to at this right ascension, so that anything larger than the
    quantization fails.  Asserting instead that the dot product of the two unit
    vectors is one to within 1e-9 would pass at seven pixels apart.
    """
    full = tmp_path / 'full.txt'
    full.write_text(SUPPL)
    rounded = tmp_path / 'rounded.txt'
    rounded.write_text(SUPPL.split('C-Matrix')[0])
    from_matrix = read_bundle_pointing(full)
    from_pair = read_bundle_pointing(rounded)
    assert from_matrix is not None
    assert from_pair is not None
    separation = separation_px(
        from_matrix.boresight, from_pair.boresight, plate_scale_urad=NAC_PLATE_SCALE_URAD
    )
    assert separation == pytest.approx(0.0, abs=0.5)


def test_a_file_with_no_pointing_answers_nothing(tmp_path: Path) -> None:
    """A file recording no boresight is not an answer."""
    path = tmp_path / 'suppl.txt'
    path.write_text('Navigation Type = Manual\n')
    assert read_bundle_pointing(path) is None


def test_a_matrix_holding_no_finite_number_is_passed_over(tmp_path: Path) -> None:
    """Another project's file can hold a value that is not a number.

    A NaN or an infinity read out of it would spread through every number
    measured from the boresight, so the matrix is declined and the rounded pair
    answers instead.
    """
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL.replace('0.0277724004', '1e999'))
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.from_cmatrix is False


def test_a_pair_holding_no_finite_number_is_no_answer(tmp_path: Path) -> None:
    """With neither spelling usable the file records no boresight this can use."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL.split('C-Matrix')[0].replace('83.9924', '83.99.24'))
    assert read_bundle_pointing(path) is None


def test_a_full_frame_wac_pixel_is_the_wac_scale() -> None:
    """A full-frame WAC pixel spans the WAC plate scale."""
    assert plate_scale_urad('WAC', 1024) == WAC_PLATE_SCALE_URAD


def test_a_summed_pixel_spans_the_summing_factor() -> None:
    """A frame summed four by four has pixels four times as wide."""
    assert plate_scale_urad('NAC', 256) == pytest.approx(4.0 * NAC_PLATE_SCALE_URAD)


def test_an_unknown_camera_is_refused() -> None:
    """Only the two Cassini ISS cameras have a scale here."""
    with pytest.raises(ValueError, match='NAC or WAC'):
        plate_scale_urad('HRI', 1024)


def test_a_width_no_summing_mode_produces_is_refused() -> None:
    """A frame 300 columns wide is not a summed Cassini ISS frame."""
    with pytest.raises(ValueError, match='300 columns'):
        plate_scale_urad('NAC', 300)


def test_a_width_that_divides_the_frame_but_no_mode_produces_is_refused() -> None:
    """A 128-column frame would be 8x8 summing, which Cassini ISS does not have."""
    with pytest.raises(ValueError, match='128 columns'):
        plate_scale_urad('NAC', 128)


def test_a_nac_frame_is_read_as_published(tmp_path: Path) -> None:
    """A narrow angle frame carries no note, so nothing was done to its matrix."""
    path = tmp_path / 'suppl.txt'
    path.write_text(SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.note is None


def test_the_restoration_is_the_old_oops_wac_frame() -> None:
    """The rotation is the one oops builds from the old WAC offset, in SPICE axes.

    The old library defined its WAC frame as Navigation((-7, +4.4) NAC pixels) on its
    own camera frame, which is the SPICE camera frame turned half way round the
    boresight.
    """
    nac_pixel = math.atan(math.tan(math.radians(0.175)) / 512.0)
    navigation = oops.frame.Navigation((-7.0 * nac_pixel, 4.4 * nac_pixel), oops.Frame.J2000)
    rot180 = np.diag([-1.0, -1.0, 1.0])
    expected = rot180 @ navigation.transform_at_time(0.0).matrix.vals @ rot180 @ WAC_CMATRIX
    assert restore_old_oops_wac_offset(WAC_CMATRIX) == pytest.approx(expected, abs=1e-15)


def test_restoring_moves_the_published_boresight_by_the_predicted_amount() -> None:
    """Seen from the restored attitude, the published boresight is (-0.699, -0.440) px."""
    restored = restore_old_oops_wac_offset(WAC_CMATRIX)
    published_from_restored = offset_in_camera_px(
        restored.tolist(), WAC_CMATRIX[2], plate_scale_urad=WAC_PLATE_SCALE_URAD
    )
    assert published_from_restored == pytest.approx((-0.699, -0.440), abs=0.001)


def test_a_wac_frame_has_the_old_offset_restored(tmp_path: Path) -> None:
    """The boresight read is the restored matrix's third row."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.boresight == pytest.approx(restore_old_oops_wac_offset(WAC_CMATRIX)[2], abs=1e-9)


def test_a_wac_frame_says_its_offset_was_restored(tmp_path: Path) -> None:
    """The note says what was done, so a report can count it."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.note == WAC_OFFSET_RESTORED


def test_a_2007_wac_frame_is_read_as_published(tmp_path: Path) -> None:
    """The bundle's 2007 WAC matrices are in the standard frame, so none is restored."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_2007_SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.boresight == pytest.approx(WAC_CMATRIX[2], abs=1e-9)


def test_a_2007_wac_frame_says_it_was_read_as_published(tmp_path: Path) -> None:
    """The exception is named in the note rather than left to look like a NAC frame."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_2007_SUPPL)
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.note == WAC_AS_PUBLISHED


@pytest.mark.parametrize(
    ('mid_time', 'note'),
    [
        ('2006-12-31T23:59:59.999Z', WAC_OFFSET_RESTORED),
        ('2007-01-01T00:00:00.000Z', WAC_AS_PUBLISHED),
        ('2007-12-31T23:59:59.999Z', WAC_AS_PUBLISHED),
        ('2008-01-01T00:00:00.000Z', WAC_OFFSET_RESTORED),
    ],
)
def test_the_2007_window_has_closed_start_and_open_end(
    tmp_path: Path, mid_time: str, note: str
) -> None:
    """The window is calendar 2007: its first instant is in, 2008's first is out.

    Parameters:
        tmp_path: Fixture used to write the supplementary file.
        mid_time: The frame's mid time.
        note: The note the frame must carry.
    """
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL.replace('2013-01-28T17:14:09.581Z', mid_time))
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.note == note


def test_a_wac_frame_quoting_no_mid_time_has_the_offset_restored(tmp_path: Path) -> None:
    """Absent a date, the frame is read like every WAC frame outside 2007."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL.replace('Image Mid Time (UTC) = 2013-01-28T17:14:09.581Z\n', ''))
    answer = read_bundle_pointing(path)
    assert answer is not None
    assert answer.note == WAC_OFFSET_RESTORED


def test_a_wac_frame_needing_the_offset_and_quoting_no_matrix_is_no_answer(
    tmp_path: Path,
) -> None:
    """The rounded pair cannot carry a rotation of the whole attitude."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL.split('C-Matrix')[0])
    assert read_bundle_pointing(path) is None


def test_a_wac_matrix_with_a_bad_first_row_is_no_answer(tmp_path: Path) -> None:
    """The restored boresight mixes all three rows, so a bad first row reaches it."""
    path = tmp_path / 'suppl.txt'
    path.write_text(WAC_SUPPL.replace('0.7523069074', '1e999'))
    assert read_bundle_pointing(path) is None
