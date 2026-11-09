from typing import Any

from oops.observation import Snapshot

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot


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
