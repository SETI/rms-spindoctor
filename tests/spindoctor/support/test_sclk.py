"""Tests for reading an image's clock counts from the PDS3 label beside it."""

from pathlib import Path

from filecache import FCPath

from spindoctor.support.sclk import pds3_label_clock_counts


def test_the_counts_come_from_the_label_beside_the_image(tmp_path: Path) -> None:
    """The counts are those of the label named for the image, whose path is text."""
    image = tmp_path / 'image_0001.fit'
    image.write_bytes(b'')
    (tmp_path / 'image_0001.lbl').write_text(
        'PDS_VERSION_ID               = PDS3\n'
        'SPACECRAFT_CLOCK_START_COUNT = "0000001234:04321"\n'
        'SPACECRAFT_CLOCK_STOP_COUNT  = "0000001235:00000"\n'
        'END\n'
    )
    assert pds3_label_clock_counts(str(image)) == ('0000001234:04321', '0000001235:00000')


def test_an_image_with_no_label_beside_it_has_no_counts(tmp_path: Path) -> None:
    """An image with no label beside it has neither count."""
    assert pds3_label_clock_counts(FCPath(tmp_path / 'image_0001.IMG')) == (None, None)
