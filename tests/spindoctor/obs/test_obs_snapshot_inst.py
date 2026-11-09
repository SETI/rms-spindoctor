"""Unit tests for the helpers in ``spindoctor.obs.obs_snapshot_inst``."""

from pathlib import Path

from filecache import FCPath

from spindoctor.obs.obs_snapshot_inst import local_image_path


def test_an_image_reached_through_a_renamed_link_keeps_the_link_name(tmp_path: Path) -> None:
    """The local path names the link the caller gave, not the file it points to."""
    target = tmp_path / 'C3446143_GEOMED.IMG'
    target.write_bytes(b'')
    link = tmp_path / 'renamed_GEOMED.IMG'
    link.symlink_to(target)

    assert local_image_path(FCPath(link)) == link.absolute()
