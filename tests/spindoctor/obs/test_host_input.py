"""Tests for choosing the file a host reads to load an image."""

from pathlib import Path

from filecache import FCPath

from spindoctor.obs.obs_snapshot_inst import host_input


def test_an_image_with_a_label_beside_it_is_read_through_the_label(tmp_path: Path) -> None:
    """The label with the image's name and the suffix .LBL is the file the host reads."""
    image = tmp_path / 'C2062133_GEOMED.IMG'
    image.write_bytes(b'')
    (tmp_path / 'C2062133_GEOMED.LBL').write_text('END\n')
    assert host_input(str(image)) == FCPath(tmp_path / 'C2062133_GEOMED.LBL')


def test_a_lower_case_image_name_has_a_lower_case_label(tmp_path: Path) -> None:
    """An image whose suffix is lower case has its label's suffix in lower case too."""
    image = tmp_path / 'lor_0299147641_0x630_sci.fit'
    image.write_bytes(b'')
    (tmp_path / 'lor_0299147641_0x630_sci.lbl').write_text('END\n')
    assert host_input(FCPath(image)) == FCPath(tmp_path / 'lor_0299147641_0x630_sci.lbl')


def test_an_image_with_no_label_beside_it_is_read_alone(tmp_path: Path) -> None:
    """With no label beside it, the host reads the image file itself."""
    image = tmp_path / 'C0368558239R.IMG'
    image.write_bytes(b'')
    assert host_input(image) == FCPath(image)
