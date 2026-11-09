from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from oops.observation import Snapshot

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


def local_image_path(image: FCPath) -> Path:
    """Return the image file on local storage, under the name the caller gave it.

    An oops host records the local path with every symlink resolved, so an image reached
    through a link with a name of its own would be published under its target's name,
    which disagrees with the published image path.  This path keeps the link.

    Parameters:
        image: The image file as the caller named it.

    Returns:
        The absolute local path of the image, fetched into the cache when remote.
    """
    return cast(Path, image.get_local_path()).absolute()


class HostLabel:
    """Mix-in for an observation whose host keeps the image's label on the snapshot.

    The label is declared here rather than on the instrument's class, whose methods
    annotate their own results as ``dict`` and would otherwise name this attribute.
    """

    dict: dict[Any, Any]
    """The image's label items, keyed by keyword, as the host read them."""


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        """Initializes a new ObsSnapshotInst instance.

        Parameters:
            snapshot: The Snapshot object to wrap.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(snapshot, **kwargs)
