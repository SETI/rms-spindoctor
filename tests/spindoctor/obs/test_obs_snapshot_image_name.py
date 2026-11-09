"""Unit tests for ``ObsSnapshot.image_name``."""

from pathlib import Path

from spindoctor.obs.obs_snapshot import ObsSnapshot


def test_an_image_reached_through_a_renamed_link_is_named_for_the_link(tmp_path: Path) -> None:
    """The name comes from the image as the caller gave it, not the resolved local file."""
    target = tmp_path / 'C3446143_GEOMED.IMG'
    target.write_bytes(b'')
    link = tmp_path / 'renamed_GEOMED.IMG'
    link.symlink_to(target)
    obs = object.__new__(ObsSnapshot)
    obs.image_url = link.absolute().as_posix()
    obs.abspath = link.resolve()

    assert obs.image_name == 'renamed_GEOMED.IMG'
