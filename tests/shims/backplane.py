"""Fake ``oops.Backplane`` for unit-testing nav-pipeline code paths.

Real backplane evaluation depends on a working SPICE kernel pool, an
``oops.Observation`` whose camera model can be inverted at sub-pixel
resolution, and a per-image meshgrid.  This shim lets a test plug in
canned per-pixel arrays and per-body / per-ring scalar facts; every
backplane method the navigation pipeline calls returns a real
``polymath.Scalar`` wrapping the configured numpy array.

The shim is table-driven.  Tests construct a :class:`FakeBackplane`
with kwargs that fill the lookup tables; methods that have no entry
raise ``LookupError`` naming the missing key, so missing wiring fails
fast instead of silently returning empty data.

Construction is intentionally explicit so tests can build the smallest
backplane they need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polymath

__all__ = [
    'BodyBackplaneData',
    'FakeBackplane',
    'KeyedScalar',
    'RingBackplaneData',
    'plant_circular_body',
]


class KeyedScalar(polymath.Scalar):
    """A ``polymath.Scalar`` that also carries a backplane ``key``.

    ``oops.Backplane`` tags each result it returns with the key that
    produced it, and the navigation pipeline reads that key back off a
    ``ring_radius`` result to call :meth:`oops.Backplane.border_atop`.
    ``polymath.Scalar`` has no such attribute of its own, so the shim
    declares one here rather than grafting it onto an instance.

    Construction takes the same arguments as ``polymath.Scalar``; the
    ``key`` attribute is not a constructor argument.  It defaults to
    ``None`` and :func:`_scalar` assigns it when a caller supplies one,
    so every arithmetic or indexing result derived from a keyed scalar
    reports ``key is None`` rather than inheriting a key that no longer
    describes it.
    """

    key: tuple[Any, ...] | None = None


def _scalar(
    vals: np.ndarray | float,
    mask: np.ndarray | bool | None = None,
    *,
    key: tuple[Any, ...] | None = None,
) -> KeyedScalar:
    """Build a ``KeyedScalar`` and optionally tag it with a backplane ``key``.

    ``polymath.Scalar`` already covers ``.vals`` / ``.mvals`` /
    ``min`` / ``max`` / ``median`` / ``expand_mask`` / ``mask_where`` /
    ``any`` / ``__getitem__`` natively.  The shim adds only the ``key``
    attribute that ``oops.Backplane`` sets on a registered backplane, so
    that :meth:`FakeBackplane.standardize_backplane_key` answers from it
    the way the real method does.

    Parameters:
        vals: Numpy array or scalar.
        mask: Optional boolean mask of the same shape as ``vals``.
        key: Optional opaque key surfaced as ``.key``.  A result built
            without one reports ``key is None``.

    Returns:
        Configured ``KeyedScalar``.
    """
    if mask is None:
        scalar = KeyedScalar(np.asarray(vals))
    else:
        scalar = KeyedScalar(np.asarray(vals), np.asarray(mask, dtype=bool))
    if key is not None:
        scalar.key = key
    return scalar


@dataclass
class BodyBackplaneData:
    """Per-body backplane arrays and scalar facts.

    Parameters:
        body_mask: 2-D boolean mask, ``True`` where the body silhouette
            is on the pixel grid.  Required.
        incidence_rad: 2-D float array of per-pixel incidence angles in
            radians.  Values outside ``body_mask`` are ignored.
            Required.
        lambert: 2-D float array of per-pixel Lambert reflectance.
            Defaults to ``cos(incidence)`` clipped to ``[0, inf)`` and
            zeroed outside the silhouette.
        resolution_km_px: 2-D float array of per-pixel km/px scale.
            Defaults to a constant fill from ``default_resolution_km_px``.
        default_resolution_km_px: Scalar fallback when
            ``resolution_km_px`` is not supplied.
        sub_solar_lon_rad: scalar; default 0.
        sub_solar_lat_rad: scalar; default 0.
        sub_observer_lon_rad: scalar; default 0.
        sub_observer_lat_rad: scalar; default 0.
        center_phase_rad: scalar; default 0.
        intercepted_mask: 2-D boolean mask returned by
            ``where_intercepted``.  Defaults to ``body_mask``.
    """

    body_mask: np.ndarray
    incidence_rad: np.ndarray
    lambert: np.ndarray | None = None
    resolution_km_px: np.ndarray | None = None
    default_resolution_km_px: float = 1.0
    sub_solar_lon_rad: float = 0.0
    sub_solar_lat_rad: float = 0.0
    sub_observer_lon_rad: float = 0.0
    sub_observer_lat_rad: float = 0.0
    center_phase_rad: float = 0.0
    intercepted_mask: np.ndarray | None = None

    def lambert_array(self) -> np.ndarray:
        """Return the Lambert array, computing the default if absent."""
        if self.lambert is not None:
            return self.lambert
        cos_i = np.clip(np.cos(self.incidence_rad), 0.0, None)
        return np.where(self.body_mask, cos_i, 0.0)

    def resolution_array(self) -> np.ndarray:
        """Return the resolution array, computing the default if absent."""
        if self.resolution_km_px is not None:
            return self.resolution_km_px
        return np.full(self.body_mask.shape, self.default_resolution_km_px)

    def intercept_mask(self) -> np.ndarray:
        """Return the intercepted-mask, defaulting to ``body_mask``."""
        return self.intercepted_mask if self.intercepted_mask is not None else self.body_mask


@dataclass
class RingBackplaneData:
    """Per-planet ring-system backplane arrays.

    Parameters:
        ring_radius_km: 2-D float array of per-pixel ring-plane radius
            in km.  Required.
        ring_mask: 2-D boolean mask, ``True`` where the ring plane is
            sampled (False -> masked out, off the ring).  Required.
        radial_resolution_km_px: 2-D float array of per-pixel radial
            km/px.  Defaults to a constant fill.
        default_radial_resolution_km_px: Scalar fallback when
            ``radial_resolution_km_px`` is not supplied.
        distance_km: 2-D float array of per-pixel distance to the ring
            plane.  Defaults to a constant fill.
        default_distance_km: Scalar fallback for ``distance_km``.
        shadow_mask: Optional 2-D boolean mask, ``True`` where the ring
            plane is in the planet's shadow.  Defaults to all-False.
        border_atop: Optional callable mapping ``(key, a)`` to a 2-D
            boolean mask.  Defaults to a thresholding helper that picks
            pixels whose ``ring_radius_km`` is within 0.5 km of ``a``.
    """

    ring_radius_km: np.ndarray
    ring_mask: np.ndarray
    radial_resolution_km_px: np.ndarray | None = None
    default_radial_resolution_km_px: float = 1.0
    distance_km: np.ndarray | None = None
    default_distance_km: float = 1.0e9
    shadow_mask: np.ndarray | None = None
    border_atop: Callable[[tuple[Any, ...], float], np.ndarray] | None = None

    def radial_resolution_array(self) -> np.ndarray:
        """Return the radial-resolution array, defaulting if absent."""
        if self.radial_resolution_km_px is not None:
            return self.radial_resolution_km_px
        return np.full(self.ring_mask.shape, self.default_radial_resolution_km_px)

    def distance_array(self) -> np.ndarray:
        """Return the distance array, defaulting if absent."""
        if self.distance_km is not None:
            return self.distance_km
        return np.full(self.ring_mask.shape, self.default_distance_km)

    def shadow_array(self) -> np.ndarray:
        """Return the shadow mask, defaulting to all-False."""
        return (
            self.shadow_mask
            if self.shadow_mask is not None
            else np.zeros(self.ring_mask.shape, dtype=bool)
        )

    def border_atop_mask(self, key: tuple[Any, ...], a: float) -> np.ndarray:
        """Return the boolean ``border_atop(key, a)`` mask."""
        if self.border_atop is not None:
            return self.border_atop(key, a)
        return np.abs(self.ring_radius_km - a) < 0.5


@dataclass
class FakeBackplane:
    """Table-driven stand-in for ``oops.Backplane``.

    Each method returns a real ``polymath.Scalar`` whose ``vals`` /
    ``mvals`` / aggregate-reduction methods behave exactly like the
    production code expects, so the navigation pipeline runs against
    standard ``polymath`` objects.

    Parameters:
        per_body: Mapping of upper-case body name to
            :class:`BodyBackplaneData`.  Methods like
            ``incidence_angle(body_name)`` look up the matching entry
            and return a Scalar over the configured array.
        per_ring: Mapping of ring target string (e.g. ``'saturn:ring'``)
            to :class:`RingBackplaneData`.
    """

    per_body: dict[str, BodyBackplaneData] = field(default_factory=dict)
    per_ring: dict[str, RingBackplaneData] = field(default_factory=dict)
    backplanes: dict[tuple[Any, ...], KeyedScalar] = field(default_factory=dict)

    def standardize_backplane_key(self, backplane_key: Any) -> tuple[Any, ...]:
        """Name the key an array is registered under, as ``oops.Backplane`` does.

        This is how production code names a key.  It does not read the array's
        ``key`` attribute, which oops sets on what the registry hands out but
        which is absent from any array computed from one -- correctly so, since
        a computed array is not the registered backplane and a key claiming
        otherwise would resolve to the wrong array without raising.

        Mirrors the real method: an array is answered from ``key`` when it has
        one and by searching the registry for it by identity otherwise, a
        string becomes an upper-case one-tuple, and a tuple passes through.

        Parameters:
            backplane_key: An array obtained from this backplane, or a key
                already in string or tuple form.

        Returns:
            The registered backplane key.

        Raises:
            ValueError: If an array is not one this backplane handed out, or the
                argument is neither an array nor a string nor a tuple.
        """
        if isinstance(backplane_key, polymath.Qube):
            key = getattr(backplane_key, 'key', None)
            if key is not None:
                return cast('tuple[Any, ...]', key)
            for registered_key, array in self.backplanes.items():
                if array is backplane_key:
                    return registered_key
            raise ValueError('illegal backplane key type: ' + type(backplane_key).__name__)
        if isinstance(backplane_key, str):
            return (backplane_key.upper(),)
        if isinstance(backplane_key, tuple):
            return backplane_key
        raise ValueError('illegal backplane key type: ' + type(backplane_key).__name__)

    def _body(self, body_name: str) -> BodyBackplaneData:
        key = body_name.upper()
        if key not in self.per_body:
            raise LookupError(f'FakeBackplane has no entry for body {body_name!r}')
        return self.per_body[key]

    def _ring(self, ring_target: str) -> RingBackplaneData:
        if ring_target not in self.per_ring:
            raise LookupError(f'FakeBackplane has no entry for ring target {ring_target!r}')
        return self.per_ring[ring_target]

    # ------------------------------------------------------------------
    # Body methods
    # ------------------------------------------------------------------

    def incidence_angle(self, body_name: str) -> polymath.Scalar:
        """Return per-pixel incidence angle in radians."""
        data = self._body(body_name)
        return _scalar(data.incidence_rad, ~data.body_mask)

    def lambert_law(self, body_name: str) -> polymath.Scalar:
        """Return per-pixel Lambert reflectance."""
        data = self._body(body_name)
        return _scalar(data.lambert_array(), ~data.body_mask)

    def resolution(self, body_name: str) -> polymath.Scalar:
        """Return per-pixel km/px scale."""
        data = self._body(body_name)
        return _scalar(data.resolution_array(), ~data.body_mask)

    def sub_solar_longitude(self, body_name: str) -> polymath.Scalar:
        """Return scalar sub-solar longitude (radians)."""
        return _scalar(self._body(body_name).sub_solar_lon_rad)

    def sub_solar_latitude(self, body_name: str) -> polymath.Scalar:
        """Return scalar sub-solar latitude (radians)."""
        return _scalar(self._body(body_name).sub_solar_lat_rad)

    def sub_observer_longitude(self, body_name: str) -> polymath.Scalar:
        """Return scalar sub-observer longitude (radians)."""
        return _scalar(self._body(body_name).sub_observer_lon_rad)

    def sub_observer_latitude(self, body_name: str) -> polymath.Scalar:
        """Return scalar sub-observer latitude (radians)."""
        return _scalar(self._body(body_name).sub_observer_lat_rad)

    def center_phase_angle(self, body_name: str) -> polymath.Scalar:
        """Return scalar center-pixel phase angle (radians)."""
        return _scalar(self._body(body_name).center_phase_rad)

    def center_resolution(self, body_name: str, axis: str = 'u') -> polymath.Scalar:
        """Return the scalar km/px scale at the body centre.

        The shim reports the same scale on both axes; ``axis`` is accepted
        for API parity with ``oops.Backplane.center_resolution``.
        """
        del axis
        return _scalar(self._body(body_name).default_resolution_km_px)

    def where_intercepted(self, body_name: str) -> polymath.Scalar:
        """Return a boolean Scalar marking pixels intercepting the body."""
        return _scalar(self._body(body_name).intercept_mask().astype(bool))

    # ------------------------------------------------------------------
    # Ring methods
    # ------------------------------------------------------------------

    def ring_radius(self, ring_target: str) -> KeyedScalar:
        """Return per-pixel ring-plane radius in km, registered under its key.

        The array is registered under ``('ring_radius', ring_target)`` and
        cached, so :meth:`standardize_backplane_key` -- which is how production
        names it before calling :meth:`border_atop` -- can find it by identity.
        Repeat calls answer with the one object, as ``oops.Backplane`` answers
        from its cache.

        Parameters:
            ring_target: Ring target name, as keyed into ``per_ring``.

        Returns:
            Per-pixel radii, masked outside the configured ``ring_mask``.
        """
        key = ('ring_radius', ring_target)
        if key in self.backplanes:
            return self.backplanes[key]
        data = self._ring(ring_target)
        radii = _scalar(data.ring_radius_km, ~data.ring_mask, key=key)
        self.backplanes[key] = radii
        return radii

    def ring_radial_resolution(self, ring_target: str) -> polymath.Scalar:
        """Return per-pixel radial km/px scale."""
        data = self._ring(ring_target)
        return _scalar(data.radial_resolution_array(), ~data.ring_mask)

    def border_atop(self, key: tuple[Any, ...], a: float) -> polymath.Scalar:
        """Return a boolean Scalar marking pixels at ring radius ``a``.

        ``key`` is the tuple production code obtains from
        :meth:`standardize_backplane_key`; the head determines which ring's
        radius array we threshold against ``a``.
        """
        if not key or key[0] != 'ring_radius':
            raise LookupError(f'FakeBackplane.border_atop expected a ring_radius key, got {key!r}')
        ring_target = key[1]
        data = self._ring(ring_target)
        return _scalar(data.border_atop_mask(key, a).astype(bool))

    def distance(self, ring_target: str, *, direction: str = 'dep') -> polymath.Scalar:
        """Return per-pixel ring-plane distance in km."""
        del direction
        data = self._ring(ring_target)
        return _scalar(data.distance_array(), ~data.ring_mask)

    def where_inside_shadow(self, ring_target: str, planet: str) -> polymath.Scalar:
        """Return a boolean Scalar marking ring pixels inside the planet's shadow."""
        del planet
        data = self._ring(ring_target)
        return _scalar(data.shadow_array().astype(bool))

    def where_in_front(self, near_target: str, far_target: str) -> polymath.Scalar:
        """Return a boolean Scalar marking pixels where one surface hides another.

        Nothing occludes anything here: a scene that wants an occluder plants
        one and answers this itself. The method exists because the renderer
        calls it unconditionally, and a stand-in that does not answer a call the
        real backplane answers is a hole in the double rather than a fact about
        the scene.

        Parameters:
            near_target: The surface that might be in front.
            far_target: The surface that might be hidden.
        """
        del near_target
        data = self._ring(far_target)
        return _scalar(np.zeros(data.ring_mask.shape, dtype=bool))


def plant_circular_body(
    *,
    shape: tuple[int, int],
    centre_vu: tuple[float, float],
    radius_px: float,
    sub_solar_lon_deg: float = 0.0,
    sub_solar_lat_deg: float = 0.0,
    sub_observer_lon_deg: float = 0.0,
    sub_observer_lat_deg: float = 0.0,
    phase_angle_deg: float = 30.0,
    resolution_km_px: float = 5.0,
) -> BodyBackplaneData:
    """Build :class:`BodyBackplaneData` for a circular body silhouette.

    Convenience factory that paints a circle of radius ``radius_px``
    centred at ``centre_vu`` and assigns each silhouette pixel an
    incidence angle that varies smoothly across the disc (zero at the
    centre, increasing outward to the limb).  Suitable for end-to-end
    body-NavModel tests where the exact incidence pattern is not the
    focus.

    Parameters:
        shape: ``(rows, cols)`` of the output arrays.
        centre_vu: Body centre in pixel coordinates.
        radius_px: Body radius in pixels.
        sub_solar_lon_deg: Scalar sub-solar longitude in degrees.
        sub_solar_lat_deg: Scalar sub-solar latitude in degrees.
        sub_observer_lon_deg: Scalar sub-observer longitude in degrees.
        sub_observer_lat_deg: Scalar sub-observer latitude in degrees.
        phase_angle_deg: Scalar phase angle in degrees.
        resolution_km_px: Constant per-pixel km/px scale.

    Returns:
        Configured :class:`BodyBackplaneData`.

    Raises:
        ValueError: If ``radius_px`` is not positive (would divide by zero
            in the incidence ramp and seed NaNs into the outputs).
    """
    if not radius_px > 0.0:
        raise ValueError(f'radius_px must be > 0; got {radius_px!r}')
    rows, cols = shape
    vv, uu = np.meshgrid(
        np.arange(rows, dtype=np.float64),
        np.arange(cols, dtype=np.float64),
        indexing='ij',
    )
    dv = vv - centre_vu[0]
    du = uu - centre_vu[1]
    radius = np.sqrt(dv * dv + du * du)
    body_mask = radius <= radius_px
    # Linear ramp from 0 at centre to pi/2 at limb; outside the limb the
    # value is irrelevant because ``body_mask`` is False.
    incidence = np.where(
        body_mask,
        (radius / radius_px) * (np.pi / 2.0),
        np.pi / 2.0,
    )
    return BodyBackplaneData(
        body_mask=body_mask,
        incidence_rad=incidence,
        sub_solar_lon_rad=float(np.radians(sub_solar_lon_deg)),
        sub_solar_lat_rad=float(np.radians(sub_solar_lat_deg)),
        sub_observer_lon_rad=float(np.radians(sub_observer_lon_deg)),
        sub_observer_lat_rad=float(np.radians(sub_observer_lat_deg)),
        center_phase_rad=float(np.radians(phase_angle_deg)),
        default_resolution_km_px=resolution_km_px,
    )
