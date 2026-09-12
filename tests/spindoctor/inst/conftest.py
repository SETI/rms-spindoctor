"""Shared stand-ins for the instrument host tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from spindoctor.obs import ObsSnapshotInst

_ObsT = TypeVar('_ObsT', bound=ObsSnapshotInst)

CLOCK_KEYS = ('start_time_sclk', 'midtime_sclk', 'end_time_sclk')
"""The three spacecraft clock counts a host publishes, start to end."""


def published_clock_counts(obs: ObsSnapshotInst) -> list[Any]:
    """Return the three clock counts an observation's host publishes, start to end.

    Parameters:
        obs: The observation.

    Returns:
        Its public metadata's ``start_time_sclk``, ``midtime_sclk`` and ``end_time_sclk``.
    """
    public = obs.get_public_metadata()
    return [public[key] for key in CLOCK_KEYS]


def bare_observation(obs_class: type[_ObsT], label: dict[str, Any], **attributes: Any) -> _ObsT:
    """Build an observation of a host class without loading an image.

    The observation carries the label as its ``dict``, and what every host's public
    metadata reads beyond it: an image path and name, a two-second exposure and a
    1024 x 1024 image.  The host's own attributes are given as keywords.

    Parameters:
        obs_class: The host's observation class.
        label: The label items the host reads.
        attributes: Further attributes to set, by name.

    Returns:
        The observation.
    """
    obs = object.__new__(obs_class)
    obs.dict = label
    obs.image_url = '/holdings/image_0001.img'
    obs.abspath = Path('/cache/image_0001.img')
    obs.cadence = SimpleNamespace(time=(0.0, 2.0), midtime=1.0)
    obs.texp = 2.0
    obs._data_shape_uv = (1024, 1024)
    for name, value in attributes.items():
        setattr(obs, name, value)
    return obs


class VicarLabelStandIn(dict[str, Any]):
    """A VICAR label's items, read the way rms-vicar's label reads them.

    ``VicarLabel.get`` takes its default as a required argument, so a host reading an item
    a label may lack has to pass one.  This stand-in refuses a call without it the same
    way, which a plain dict would not.
    """

    def get(self, key: str, default: Any) -> Any:  # type: ignore[override]  # default is required
        """Return an item, or the default when the label lacks it.

        Parameters:
            key: The item's name.
            default: What to return when the label lacks the item.

        Returns:
            The item's value, or the default.
        """
        return super().get(key, default)
