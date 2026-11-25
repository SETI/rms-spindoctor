"""Stand-ins for a host package's observation, for loading an image through a reader.

A reader builds its observation from the Snapshot its host returns, and building it
measures the distance to each configured planet.  These stand-ins give a reader a host
Snapshot without reading a file, and place the planets on fixed paths, so that no SPICE
kernel is needed.
"""

from typing import Any

import numpy as np
import oops
import pytest
from oops.observation import Snapshot

from spindoctor.config import DEFAULT_CONFIG


def place_planets_without_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register every configured planet on a fixed path, in registries of the test's own.

    The planets are registered in copies of the oops registries, which the monkeypatch
    puts back when the test ends, so that no other test sees them.  The frame registries
    are copied too: oops caches a frame without an id under a key that any other such
    frame matches, so a frame an earlier test built would stand in for the test's own.

    Parameters:
        monkeypatch: The test's monkeypatch.
    """
    for owner, registry in (
        (oops.Body, 'BODY_REGISTRY'),
        (oops.path.Path, '_PATH_REGISTRY'),
        (oops.path.Path, '_PATH_CACHE'),
        (oops.path.LinearPath, '_WAYPOINTS'),
        (oops.Frame, '_FRAME_REGISTRY'),
        (oops.Frame, '_FRAME_CACHE'),
    ):
        monkeypatch.setattr(owner, registry, dict(vars(owner)[registry]))
    for au, planet in enumerate(DEFAULT_CONFIG.planets, start=1):
        position = oops.Vector3((1.5e8 * au, 0.0, 0.0))
        path = oops.path.LinearPath(position, 0.0, 'SSB', frame='J2000', path_id=planet)
        oops.Body(planet, path, 'J2000')


def host_snapshot(
    data: np.ndarray, *, frame: oops.Frame | None = None, **subfields: Any
) -> Snapshot:
    """Return a Snapshot as a host returns it: the image and the host's subfields.

    Parameters:
        data: The image array.
        frame: The camera frame; by default one fixed in J2000.
        **subfields: The host's other subfields, by name.

    Returns:
        The Snapshot.
    """
    snapshot = Snapshot(
        axes=('v', 'u'),
        cadence=oops.cadence.SnapCadence(0.0, 1.0),
        fov=oops.fov.FlatFOV((1e-5, 1e-5), data.shape[::-1]),
        path='SSB',
        frame=oops.frame.Cmatrix(oops.Matrix3.IDENTITY) if frame is None else frame,
    )
    snapshot.insert_subfield('data', data)
    for name, value in subfields.items():
        snapshot.insert_subfield(name, value)
    return snapshot


class HostCall:
    """A host's ``from_file`` that records what it was given and returns a Snapshot."""

    def __init__(self, snapshot: Snapshot) -> None:
        """Hold the Snapshot to return.

        Parameters:
            snapshot: What the host returns.
        """
        self.snapshot = snapshot
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Snapshot:
        """Record the call and return the Snapshot.

        Parameters:
            *args: The positional arguments of the call.
            **kwargs: The keyword arguments of the call.

        Returns:
            The Snapshot.
        """
        self.args = args
        self.kwargs = kwargs
        return self.snapshot
