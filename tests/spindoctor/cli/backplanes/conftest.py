"""Shared hermetic fixtures for the backplane pipeline test suite.

The backplane backend is driven by an ``ObsSnapshot`` plus a ``Config``.  Real
observations require SPICE kernels and PDS holdings, neither of which is available
in CI, so these helpers build genuine ``oops`` snapshots on the built-in SSB path
and J2000 frame (no kernels needed) plus duck-typed stand-ins for the pieces that
would otherwise reach SPICE: the per-image body inventory, the full-frame
``oops.Backplane``, and the configuration object.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import oops
from oops.observation.snapshot import Snapshot

from spindoctor.config import Config
from spindoctor.obs import Obs, ObsSnapshotInst
from spindoctor.support.types import PathLike

MASKED_VALUE = -999.0
"""The ``backplanes.masked_value`` the shipped configuration sets.

Bound here so a test asserts against a name rather than a literal, and so a
change to the shipped value is made in one place on the test side.
"""


class HermeticObs(ObsSnapshotInst):
    """Hermetic ``ObsSnapshotInst`` that needs no SPICE kernels or holdings.

    Wraps a real ``oops`` Snapshot on the built-in SSB path and J2000 frame with a
    FlatFOV, so the geometry-free code paths (FOV clipping, meshgrid construction,
    ``oops.Backplane`` instantiation) exercise genuine oops objects.  The
    ``_closest_planet`` attribute is stamped onto the wrapped Snapshot before
    construction, which bypasses the SPICE-driven closest-planet search in
    ``ObsSnapshot.__init__``.

    A canned inventory dict stands in for the SPICE-driven ``Snapshot.inventory``
    so the non-simulated body path can run hermetically; requested body lists are
    recorded on ``inventory_calls``.
    """

    sim_inventory: dict[str, Any]
    sim_body_mask_map: dict[str, Any]
    sim_body_order_near_to_far: list[str]
    sim_body_index_map: Any

    def __init__(
        self,
        snapshot: Snapshot,
        *,
        canned_inventory: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Wrap a Snapshot, then attach the canned inventory and call recorder.

        Parameters:
            snapshot: The oops Snapshot to wrap (its ``__dict__`` is absorbed).
            canned_inventory: Inventory dict returned by :meth:`inventory`; None
                makes any inventory call an error.
            **kwargs: Forwarded to ``ObsSnapshotInst`` (notably ``simulated`` and
                ``extfov_margin_vu``).
        """
        super().__init__(snapshot, **kwargs)
        # Must be set after super().__init__: ObsSnapshot replaces self.__dict__.
        self._canned_inventory = canned_inventory
        self.inventory_calls: list[list[str]] = []

    @staticmethod
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> Obs:
        """Unsupported; tests construct HermeticObs directly via make_snapshot.

        Parameters:
            path: Ignored.
            config: Ignored.
            extfov_margin_vu: Ignored.
            **kwargs: Ignored.
        """
        raise NotImplementedError('HermeticObs is constructed directly in tests')

    @property
    def camera(self) -> str:
        """Return a fixed camera name (unused by backplanes)."""
        return 'HERMETIC'

    def star_min_usable_vmag(self) -> float:
        """Return a fixed lower star magnitude bound (unused by backplanes)."""
        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Return a fixed upper star magnitude bound (unused by backplanes)."""
        return 30.0

    def get_public_metadata(self) -> dict[str, Any]:
        """Return an empty public-metadata dict (unused by backplanes)."""
        return {}

    def inventory(self, bodies: list[str], **kwargs: Any) -> dict[str, Any]:
        """Return the canned inventory, recording the requested body list.

        Parameters:
            bodies: Body names the caller asked to inventory.
            **kwargs: Ignored (e.g. ``return_type``).
        """
        self.inventory_calls.append(list(bodies))
        if self._canned_inventory is None:
            raise RuntimeError('HermeticObs was built without a canned inventory')
        return self._canned_inventory


def make_snapshot(
    *,
    shape_vu: tuple[int, int] = (10, 12),
    closest_planet: str | None = 'SATURN',
    simulated: bool = False,
    sim_inventory: dict[str, Any] | None = None,
    sim_body_mask_map: dict[str, Any] | None = None,
    sim_body_order_near_to_far: list[str] | None = None,
    sim_body_index_map: Any = None,
    canned_inventory: dict[str, Any] | None = None,
) -> HermeticObs:
    """Build a hermetic snapshot observation with zero-filled image data.

    Parameters:
        shape_vu: Image shape as (rows, columns).
        closest_planet: Planet name stamped as ``_closest_planet`` (None for a
            no-planet observation).
        simulated: Whether the observation reports ``is_simulated``.
        sim_inventory: Simulated per-body inventory dict.
        sim_body_mask_map: Simulated body-name to full-frame boolean mask map.
        sim_body_order_near_to_far: Simulated body names ordered near to far.
        sim_body_index_map: Simulated full-frame per-pixel body index map.
        canned_inventory: Inventory dict served by ``HermeticObs.inventory``.

    Returns:
        A fully constructed :class:`HermeticObs`.
    """
    size_v, size_u = shape_vu
    fov = oops.fov.FlatFOV((0.001, 0.001), (size_u, size_v))
    snapshot = Snapshot(
        axes=('v', 'u'),
        tstart=0.0,
        texp=1.0,
        fov=fov,
        path='SSB',
        frame='J2000',
    )
    snapshot.insert_subfield('data', np.zeros(shape_vu, dtype=np.float32))
    # The subfields an oops host declares, which the pointing reader reads to
    # learn the convention and the frame the observation was built on.
    snapshot.insert_subfield('spice_to_frame', oops.Matrix3(np.eye(3)))
    snapshot.insert_subfield('spice_frame_name', 'J2000')
    snapshot.insert_subfield('spice_frame_id', 1)
    snapshot._closest_planet = closest_planet
    obs = HermeticObs(
        snapshot,
        canned_inventory=canned_inventory,
        extfov_margin_vu=(0, 0),
        simulated=simulated,
    )
    obs.sim_inventory = sim_inventory if sim_inventory is not None else {}
    obs.sim_body_mask_map = sim_body_mask_map if sim_body_mask_map is not None else {}
    obs.sim_body_order_near_to_far = (
        sim_body_order_near_to_far if sim_body_order_near_to_far is not None else []
    )
    obs.sim_body_index_map = sim_body_index_map
    return obs


def inventory_entry(
    *,
    u_min: int,
    u_max: int,
    v_min: int,
    v_max: int,
    body_range: float,
    center_uv: tuple[float, float] | None = None,
    u_pixel_size: float | None = None,
    v_pixel_size: float | None = None,
) -> dict[str, Any]:
    """Build a per-body inventory entry in the shape the backplane code consumes.

    Parameters:
        u_min: Unclipped bounding-box minimum u pixel.
        u_max: Unclipped bounding-box maximum u pixel.
        v_min: Unclipped bounding-box minimum v pixel.
        v_max: Unclipped bounding-box maximum v pixel.
        body_range: Distance from the observer to the body in km.
        center_uv: Optional (u, v) body center used by the writer sidecar.
        u_pixel_size: Optional apparent body size in u pixels.
        v_pixel_size: Optional apparent body size in v pixels.

    Returns:
        Inventory dict with the unclipped bounding-box and range keys.
    """
    entry: dict[str, Any] = {
        'u_min_unclipped': u_min,
        'u_max_unclipped': u_max,
        'v_min_unclipped': v_min,
        'v_max_unclipped': v_max,
        'range': body_range,
    }
    if center_uv is not None:
        entry['center_uv'] = [center_uv[0], center_uv[1]]
    if u_pixel_size is not None:
        entry['u_pixel_size'] = u_pixel_size
    if v_pixel_size is not None:
        entry['v_pixel_size'] = v_pixel_size
    return entry


class StubVals:
    """Stand-in for an oops Scalar result exposing only the ``mvals`` masked array."""

    def __init__(self, mvals: Any) -> None:
        """Wrap a masked array.

        Parameters:
            mvals: The ``numpy.ma.MaskedArray`` to expose as ``mvals``.
        """
        self.mvals = mvals


class FakeRingBackplane:
    """Fake full-frame ``oops.Backplane`` whose methods return canned masked arrays.

    Configured with a mapping from method name to masked array; attribute lookup for
    a configured method returns a callable that records ``(method, target, kwargs)``
    in ``calls`` and returns the canned array wrapped in :class:`StubVals`.
    """

    def __init__(self, method_values: dict[str, Any]) -> None:
        """Store the canned per-method masked arrays.

        Parameters:
            method_values: Mapping from oops Backplane method name to the
                ``numpy.ma.MaskedArray`` the method should return.
        """
        self.method_values = method_values
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, method: str) -> Callable[..., StubVals]:
        """Resolve a configured method name to a recording callable.

        Parameters:
            method: The oops Backplane method name being looked up.
        """
        method_values = self.__dict__['method_values']
        if method not in method_values:
            raise AttributeError(method)

        def _call(target: str, **kwargs: Any) -> StubVals:
            """Record the invocation and return the canned array for this method.

            Parameters:
                target: The oops target key the method is evaluated on.
                **kwargs: Additional method keyword arguments, recorded verbatim.
            """
            self.calls.append((method, target, dict(kwargs)))
            return StubVals(method_values[method])

        return _call


BodyValuesFn = Callable[[str, str, tuple[int, int]], Any]


def make_fake_body_backplane_cls(values_fn: BodyValuesFn) -> type:
    """Build a fake ``oops.Backplane`` class for the per-body meshgrid path.

    The returned class mimics ``Backplane(obs, meshgrid=...)``: any method name
    resolves to a callable taking the body name, and the callable returns a
    :class:`StubVals` wrapping ``values_fn(method, body_name, meshgrid_shape_vu)``.

    Parameters:
        values_fn: Callback mapping (method, body_name, meshgrid shape (nv, nu))
            to the ``numpy.ma.MaskedArray`` to return.

    Returns:
        A class suitable for monkeypatching over
        ``spindoctor.cli.backplanes.backplanes_bodies.Backplane``.
    """

    class _FakeBodyBackplane:
        """Fake per-body oops Backplane that answers every method via ``values_fn``."""

        def __init__(self, obs: Any, meshgrid: Any = None) -> None:
            """Capture the meshgrid shape; the observation itself is unused.

            Parameters:
                obs: The observation the real Backplane would wrap (ignored).
                meshgrid: The oops Meshgrid whose shape sizes the returned arrays.
            """
            self.meshgrid_shape = cast('tuple[int, int]', tuple(int(x) for x in meshgrid.shape))

        def __getattr__(self, method: str) -> Callable[[str], StubVals]:
            """Resolve any method name to a callable that evaluates ``values_fn``.

            Parameters:
                method: The oops Backplane method name being looked up.
            """
            shape = self.__dict__['meshgrid_shape']

            def _call(body_name: str) -> StubVals:
                """Return the canned array for this method, body, and meshgrid shape.

                Parameters:
                    body_name: The body the backplane method is evaluated for.
                """
                return StubVals(values_fn(method, body_name, shape))

            return _call

    return _FakeBodyBackplane


class FakeBackplanesConfig:
    """Duck-typed ``Config`` stand-in exposing only what the backplane modules read.

    ``backplanes`` is a namespace whose ``bodies`` / ``rings`` attributes exist only
    when configured, matching the modules' getattr-with-default access pattern.
    Satellite queries are recorded on ``satellites_calls``.
    """

    def __init__(
        self,
        *,
        bodies: list[dict[str, Any]] | None = None,
        rings: list[dict[str, Any]] | None = None,
        satellites: dict[str, list[str]] | None = None,
        masked_value: float = MASKED_VALUE,
    ) -> None:
        """Build the fake config.

        Parameters:
            bodies: ``backplanes.bodies`` entry list, or None to omit the attribute.
            rings: ``backplanes.rings`` entry list, or None to omit the attribute.
            satellites: Planet name to satellite-name-list map for
                :meth:`satellites`.
            masked_value: ``backplanes.masked_value``, the value a pixel carries
                where the backplane has no valid measurement.
        """
        self.backplanes = SimpleNamespace(masked_value=masked_value)
        if bodies is not None:
            self.backplanes.bodies = bodies
        if rings is not None:
            self.backplanes.rings = rings
        self._satellites = satellites if satellites is not None else {}
        self.satellites_calls: list[str] = []

    def satellites(self, planet: str) -> list[str]:
        """Record and answer a satellite-list query.

        Parameters:
            planet: Planet name being queried.
        """
        self.satellites_calls.append(planet)
        return list(self._satellites.get(planet.upper(), []))

    def as_config(self) -> Config:
        """Return self cast to ``Config`` for passing into typed call sites."""
        return cast(Config, self)
