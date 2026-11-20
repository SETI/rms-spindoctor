from typing import Any

from filecache import FCPath
from oops.observation import Snapshot

from spindoctor.support.types import PathLike

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


def host_input(image: PathLike) -> FCPath:
    """Return the file to give a host to read an image: its PDS3 label, if it has one.

    A host given an image file reads that file alone, so it lacks every value that only
    the PDS3 label gives, such as a Voyager image's start clock count or the clock counts
    of an early Cassini cruise image.  Given the label, the host reads the label and the
    image file together and merges what the two say.  The data are the same either way.

    The label is the file beside the image with the image's name and the suffix
    ``.LBL``, or ``.lbl`` when the image's own suffix is lower case, as a PDS3 volume
    names it.  If there is no such file, the image is read alone.

    Parameters:
        image: The image file, local or remote.

    Returns:
        The label beside the image if it exists, otherwise the image file.
    """
    image_path = FCPath(image)
    suffix = image_path.suffix
    label = image_path.with_suffix('.LBL' if suffix == suffix.upper() else '.lbl')
    return label if label.exists() else image_path


class HostLabel:
    """Mix-in for an observation whose host keeps the image's metadata on the snapshot.

    The metadata is declared here rather than on the instrument's class, whose methods
    annotate their own results as ``dict`` and would otherwise name this attribute.
    """

    dict: dict[Any, Any]
    """The image's metadata, merged from every label the host read, keyed by keyword.

    A value that no label gives has no key.
    """


class ObsSnapshotInst(ObsSnapshot, ObsInst):
    """Mix-in of ObsSnapshot and ObsInst."""

    _abstract = True

    def __init__(self, snapshot: Snapshot, **kwargs: Any) -> None:
        """Initializes a new ObsSnapshotInst instance.

        Parameters:
            snapshot: The Snapshot object to wrap.
            **kwargs: Additional keyword arguments used by subclasses.
        """
        super().__init__(snapshot, **kwargs)
