"""Shared ring-edge geometry for the simulator's two sides.

The image-side ring-system renderer (``spindoctor.sim.forward.ring_system``)
and the navigator-side predicted-ring renderer
(``spindoctor.nav_model.sim_ring`` plus
``spindoctor.nav_model.nav_model_rings_simulated``) must place a feature
edge at byte-identical pixel positions: the same true-anomaly convention,
the same pericenter precession, the same m-mode and edge-wave forms, and
the same pixel-center rasterization.  With shared conventions the planted
scene error is the only error in a recovery measurement.

The ring-plane projection helpers here are the single implementation of the
optical-depth ring system's opening-angle geometry, shared by design: the
forward renderer draws the full system through them and the navigator-side
ring model predicts navigable edges through the same functions, so predicted
edges land in projected positions by construction.  The conventions:

- ``lam`` is ring-plane longitude measured from the ascending node, *in the
  ring plane*, increasing counterclockwise viewed from the north.  Every
  orbital angle (pericenter longitudes and the like) lives in this frame.
- ``node_deg`` is the sky position angle of the ascending node, measured
  counterclockwise from +u toward -v; it enters only the final sky
  rotation, never the orbit model.
- ``opening_deg_obs`` is the observer's ring opening angle B in (-90, 90],
  positive north.  ``|B| = 90`` reduces the projection to sky-plane circles
  (the flat-ring regression identity).
- A point's line-of-sight depth relative to the ring center is
  ``dlos = -y * cos(B)``, positive toward the observer, so for ``B > 0`` the
  near arm is the ``y < 0`` half and the ansae have zero depth.

Every function takes explicit geometry arguments; the only scene fragment
read here is the per-feature ``orbit`` mapping (via
:func:`ring_orbit_from_mapping`), which is an idealized block both sides are
entitled to, so no truth-side information can cross the information boundary
here (see ``spindoctor.sim.scene``).  The renderer applies its planted
``orbit_error`` truth values on its own side, before calling in.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'RingEdgeWave',
    'RingOrbit',
    'RingOrbitMode',
    'compute_antialiasing_shade',
    'compute_edge_radii_array',
    'compute_edge_radius_at_angle',
    'compute_edge_wave_dr',
    'compute_orbit_radii',
    'ring_los_depth',
    'ring_orbit_from_mapping',
    'ring_plane_from_sky',
    'ring_radial_scale',
    'ring_sky_from_plane',
]


def _edge_eccentricity(*, a: float, ae: float) -> float:
    """Eccentricity of a mode 1 ring edge, validated against the ellipse limit.

    Parameters:
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.

    Returns:
        The eccentricity ``ae / a`` (0.0 when ``a`` is not positive).

    Raises:
        ValueError: If the eccentricity is 1 or more, which does not describe
            a closed elliptical edge.
    """
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        raise ValueError(
            f'Ring edge eccentricity e = ae/a = {ae}/{a} = {e} is physically '
            f'impossible; a closed elliptical edge requires e < 1'
        )
    return e


def compute_edge_radius_at_angle(
    angle: float,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> float:
    """Compute edge radius at a specific angle using mode 1 parameters.

    Parameters:
        angle: Angle in radians from center.
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Edge radius in pixels at the given angle.

    Raises:
        ValueError: If ``ae / a`` is an eccentricity of 1 or more, which does
            not describe a closed elliptical edge.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angle - current_long_peri

    # Compute radius using elliptical orbit equation
    e = _edge_eccentricity(a=a, ae=ae)
    r = a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))

    return r


def compute_edge_radii_array(
    angles: NDArrayFloatType,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> NDArrayFloatType:
    """Compute edge radii array for all angles using mode 1 parameters.

    Parameters:
        angles: Array of angles in radians from center.
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Array of edge radii in pixels at the given angles.

    Raises:
        ValueError: If ``ae / a`` is an eccentricity of 1 or more, which does
            not describe a closed elliptical edge.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angles - current_long_peri

    # Compute radius using elliptical orbit equation: r = a(1 - e^2) / (1 + e*cos(v))
    # where e = ae / a
    e = _edge_eccentricity(a=a, ae=ae)
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(true_anomaly))

    return cast(NDArrayFloatType, r)


@dataclass(frozen=True)
class RingOrbitMode:
    """One m >= 2 radial mode of a ring-feature orbit.

    The mode perturbs the edge radius by ``-amp * cos(m * (lam - peri))``
    with ``lam`` the ring-plane longitude from the ascending node, so for a
    pure m-mode on a circular base orbit ``r(lam) = a - amp * cos(m * (lam -
    peri))`` exactly (the normative closed form).  ``amp`` is ``a * e`` in
    the same radial units as the semimajor axis; ``peri`` is the mode's
    pericenter longitude in degrees, in the ring-plane frame (the sky node
    angle never enters the orbit model).

    Parameters:
        m: Mode number (2 or greater; the mode-1 shape is the base ellipse).
        amp: Radial amplitude in the orbit's radial units.
        peri: Pericenter longitude in degrees (ring-plane frame).
    """

    m: int
    amp: float
    peri: float


@dataclass(frozen=True)
class RingEdgeWave:
    """A satellite edge wave (Daphnis/Pan-style) on a ring-feature orbit.

    The radial perturbation is
    ``dr(lam) = amp * exp(-(lam - lam0) / damp) *
    sin(2 * pi * (lam - lam0) * a / wavelength)``
    evaluated ONLY downstream of ``lam0``: the longitude difference is taken
    modulo 2*pi into [0, 2*pi), so the exponential argument is never
    negative.  The clamp is load-bearing, not cosmetic -- the exponential
    grows without bound if evaluated for ``lam < lam0`` -- and the modular
    form is chosen because rings are periodic and the wave physically wraps
    the full turn: immediately upstream of ``lam0`` it carries a wrap-seam
    residual of ``amp * exp(-2*pi/damp)`` of its launch amplitude.  The
    scene validator caps ``damp`` at 2.0 radians, bounding that residual at
    ``exp(-pi)``, about 4.3% of ``amp``.

    Parameters:
        amp: Radial amplitude in the orbit's radial units.
        wavelength: Azimuthal wavelength as an arc length, in the orbit's
            radial units (the sine argument is arc length over wavelength).
        damp: Azimuthal damping constant in RADIANS of downstream longitude.
        lam0: Launch longitude in degrees (ring-plane frame; the perturbing
            moon's longitude).
    """

    amp: float
    wavelength: float
    damp: float
    lam0: float


@dataclass(frozen=True)
class RingOrbit:
    """A ring feature's full orbit model: mode-1 ellipse plus perturbations.

    This is the shared parsed form of the idealized per-feature ``orbit``
    scene mapping: the image-side renderer draws through it (after applying
    its planted orbit error on its own side) and the navigator-side model
    predicts through it unmodified, so both sides interpret the same catalog
    values identically by construction.

    Parameters:
        a: Semimajor axis (pixels; any consistent radial unit).
        ae: Eccentricity times semimajor axis, same units.
        long_peri: Mode-1 pericenter longitude in degrees (ring-plane frame).
        rate_peri: Mode-1 pericenter precession rate in degrees/day.
        modes: The m >= 2 radial modes.
        edge_wave: The satellite edge wave, or None.
    """

    a: float
    ae: float
    long_peri: float
    rate_peri: float
    modes: tuple[RingOrbitMode, ...] = ()
    edge_wave: RingEdgeWave | None = None

    def widened(self, dr: float) -> 'RingOrbit':
        """The same orbit shape displaced outward by ``dr`` radial units.

        A feature's outer edge is its inner-edge orbit widened by the radial
        width: the ellipse grows by ``dr`` while every perturbation keeps its
        amplitude and phase, so the band has constant radial width.

        Parameters:
            dr: Radial displacement in the orbit's radial units.

        Returns:
            The widened orbit.
        """
        return replace(self, a=self.a + dr)

    def scaled(self, factor: float) -> 'RingOrbit':
        """The orbit with every radial quantity scaled by ``factor``.

        Used to move detector-pixel scene values onto an oversampled render
        grid: radii and radial amplitudes scale; angles, rates, and the
        (angular) edge-wave damping do not.

        Parameters:
            factor: The radial scale factor (the oversampling factor).

        Returns:
            The scaled orbit.
        """
        return replace(
            self,
            a=self.a * factor,
            ae=self.ae * factor,
            modes=tuple(replace(m, amp=m.amp * factor) for m in self.modes),
            edge_wave=(
                None
                if self.edge_wave is None
                else replace(
                    self.edge_wave,
                    amp=self.edge_wave.amp * factor,
                    wavelength=self.edge_wave.wavelength * factor,
                )
            ),
        )


def ring_orbit_from_mapping(orbit: Mapping[str, Any]) -> RingOrbit:
    """Parse a validated per-feature ``orbit`` scene mapping into a RingOrbit.

    This is the single interpretation of the idealized orbit block, shared by
    the forward renderer and the navigator-side model so both sides apply
    the same defaults to the same catalog values.

    Parameters:
        orbit: The validated ``ring_system.features[].orbit`` mapping.

    Returns:
        The parsed orbit.
    """
    modes = tuple(
        RingOrbitMode(
            m=int(mode['m']),
            amp=float(mode.get('amp', 0.0)),
            peri=float(mode.get('peri', 0.0)),
        )
        for mode in orbit.get('modes') or []
    )
    wave_map = orbit.get('edge_wave')
    edge_wave = (
        None
        if wave_map is None
        else RingEdgeWave(
            amp=float(wave_map.get('amp', 0.0)),
            wavelength=float(wave_map['wavelength']),
            damp=float(wave_map['damp']),
            lam0=float(wave_map.get('lam0', 0.0)),
        )
    )
    return RingOrbit(
        a=float(orbit.get('a', 0.0)),
        ae=float(orbit.get('ae', 0.0)),
        long_peri=float(orbit.get('long_peri', 0.0)),
        rate_peri=float(orbit.get('rate_peri', 0.0)),
        modes=modes,
        edge_wave=edge_wave,
    )


def compute_orbit_radii(
    lam: NDArrayFloatType,
    orbit: RingOrbit,
    *,
    epoch: float,
    time: float,
) -> NDArrayFloatType:
    """Edge radii of a full orbit model at ring-plane longitudes ``lam``.

    The mode-1 precessing ellipse (exact conic form) minus each m >= 2
    mode's ``amp * cos(m * (lam - peri))``, plus the edge wave's downstream
    perturbation when the orbit carries one.  All longitudes -- ``lam``, the
    pericenters, the wave's launch longitude -- live in the ring-plane frame
    measured from the ascending node; the sky node angle enters only the
    final projection, never here.

    Parameters:
        lam: Ring-plane longitudes from the ascending node, in radians.
        orbit: The parsed orbit model.
        epoch: Ring epoch (TDB seconds) for mode-1 precession.
        time: Scene time (TDB seconds).

    Returns:
        Edge radii at each longitude, in the orbit's radial units.

    Raises:
        ValueError: If ``ae / a`` is an eccentricity of 1 or more.
    """
    r = compute_edge_radii_array(
        lam,
        a=orbit.a,
        ae=orbit.ae,
        long_peri=orbit.long_peri,
        rate_peri=orbit.rate_peri,
        epoch=epoch,
        time=time,
    )
    for mode in orbit.modes:
        r = r - mode.amp * np.cos(mode.m * (lam - math.radians(mode.peri)))
    if orbit.edge_wave is not None:
        r = r + compute_edge_wave_dr(lam, orbit.edge_wave, a=orbit.a)
    return r


def compute_edge_wave_dr(
    lam: NDArrayFloatType,
    wave: RingEdgeWave,
    *,
    a: float,
) -> NDArrayFloatType:
    """The satellite edge wave's radial perturbation at longitudes ``lam``.

    ``dr = amp * exp(-dlam / damp) * sin(2 * pi * dlam * a / wavelength)``
    with ``dlam = (lam - lam0) mod 2*pi`` in [0, 2*pi): the wave exists only
    DOWNSTREAM of the launch longitude, so the exponential argument is never
    negative (evaluating the raw form for ``lam < lam0`` would grow without
    bound -- the clamp is load-bearing).  The wrapped wave carries an
    upstream residual of ``amp * exp(-2*pi/damp)`` just before ``lam0``;
    the validator's cap of ``damp <= 2.0`` radians bounds it at ``exp(-pi)``,
    about 4.3% of ``amp``.  ``a`` is the feature's semimajor axis, making
    the sine argument arc length over wavelength (dimensionless).

    Parameters:
        lam: Ring-plane longitudes from the ascending node, in radians.
        wave: The edge-wave parameters (``damp`` in radians).
        a: The feature's semimajor axis, in the wave's radial units.

    Returns:
        Radial perturbations at each longitude, in the wave's radial units.
    """
    dlam = np.mod(lam - math.radians(wave.lam0), 2.0 * math.pi)
    dr = wave.amp * np.exp(-dlam / wave.damp) * np.sin(2.0 * math.pi * dlam * a / wave.wavelength)
    return cast(NDArrayFloatType, dr)


def ring_sky_from_plane(
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    *,
    opening_deg_obs: float,
    node_deg: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Project ring-plane points ``(r, lam)`` to sky-plane offsets ``(dv, du)``.

    Implements the normative projection: with node-aligned in-plane axes
    ``x = r*cos(lam)``, ``y = r*sin(lam)``,

    - ``du = x*cos(node) - y*sin(B)*sin(node)``
    - ``dv = -(x*sin(node) + y*sin(B)*cos(node))``

    where ``B`` is the observer opening angle and ``node`` the sky position
    angle of the ascending node (see the module docstring for both
    conventions).  ``lam`` is ring-plane longitude from the ascending node;
    ``node_deg`` enters only this final sky rotation.

    Parameters:
        r: Ring-plane radii (pixels; any consistent unit).
        lam: Ring-plane longitudes from the ascending node, in radians.
        opening_deg_obs: Observer ring opening angle B in degrees, (-90, 90].
        node_deg: Sky position angle of the ascending node in degrees,
            counterclockwise from +u toward -v.

    Returns:
        ``(dv, du)`` sky-plane offsets from the ring center, in the units of
        ``r``.
    """
    sin_b = math.sin(math.radians(opening_deg_obs))
    node = math.radians(node_deg)
    x = r * np.cos(lam)
    y = r * np.sin(lam)
    du = x * math.cos(node) - y * sin_b * math.sin(node)
    dv = -(x * math.sin(node) + y * sin_b * math.cos(node))
    return cast(NDArrayFloatType, dv), cast(NDArrayFloatType, du)


def ring_plane_from_sky(
    dv: NDArrayFloatType,
    du: NDArrayFloatType,
    *,
    opening_deg_obs: float,
    node_deg: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Invert the ring projection: sky offsets to in-plane coordinates.

    The exact inverse of :func:`ring_sky_from_plane` for ``B != 0`` (an
    edge-on ring has no invertible projection; callers render nothing at
    ``B = 0``).  At ``|B| = 90`` the mapping is a pure rotation, so ``r``
    equals the sky-plane radius ``hypot(dv, du)`` -- the flat-ring
    regression identity.

    Parameters:
        dv: Sky-plane v offsets from the ring center.
        du: Sky-plane u offsets from the ring center.
        opening_deg_obs: Observer ring opening angle B in degrees, nonzero.
        node_deg: Sky position angle of the ascending node in degrees.

    Returns:
        ``(r, lam, x, y)``: ring-plane radius, longitude from the ascending
        node in radians in [0, 2*pi), and the node-aligned in-plane
        coordinates.

    Raises:
        ValueError: If ``opening_deg_obs`` is 0 (edge-on; not invertible).
    """
    sin_b = math.sin(math.radians(opening_deg_obs))
    if sin_b == 0.0:
        raise ValueError('ring projection is not invertible at opening_deg_obs = 0 (edge-on)')
    node = math.radians(node_deg)
    p = du
    q = -dv
    x = p * math.cos(node) + q * math.sin(node)
    y = (-p * math.sin(node) + q * math.cos(node)) / sin_b
    r = np.hypot(x, y)
    lam = np.mod(np.arctan2(y, x), 2.0 * math.pi)
    return (
        cast(NDArrayFloatType, r),
        cast(NDArrayFloatType, lam),
        cast(NDArrayFloatType, x),
        cast(NDArrayFloatType, y),
    )


def ring_los_depth(y: NDArrayFloatType, *, opening_deg_obs: float) -> NDArrayFloatType:
    """Line-of-sight depth of ring-plane points relative to the ring center.

    ``dlos = -y * cos(B)``, positive toward the observer: for ``B > 0`` the
    near arm is the ``y < 0`` half, the ring's nearest point sits at
    ``lam = 270`` degrees when ``node = 0``, and the ansae (``lam = 0`` and
    ``180``) have zero depth by construction.  Compositing against a body
    orders by observer distance ``range_km - dlos_km``: positive-toward-the-
    observer depth *subtracts*, so the nearer object has the smaller
    distance.

    Parameters:
        y: Node-aligned in-plane y coordinates (from
            :func:`ring_plane_from_sky`).
        opening_deg_obs: Observer ring opening angle B in degrees.

    Returns:
        Depth values in the units of ``y``, positive toward the observer.
    """
    return -y * math.cos(math.radians(opening_deg_obs))


def ring_radial_scale(
    r: NDArrayFloatType,
    x: NDArrayFloatType,
    y: NDArrayFloatType,
    *,
    opening_deg_obs: float,
) -> NDArrayFloatType:
    """Magnitude of the image-plane gradient of the ring-plane radius.

    ``|grad r| = sqrt(x**2 + y**2 / sin(B)**2) / r``: the change in
    ring-plane radius per image pixel of sky-plane displacement.  Dividing a
    ring-plane radial distance by this converts it to image pixels, so an
    anti-aliased edge spans a constant width on the detector regardless of
    the foreshortening direction.  At ``|B| = 90`` the scale is exactly 1
    everywhere (the sky-plane-circle identity); it grows toward the minor
    axis of an inclined ring, where radial structure is foreshortened.

    Parameters:
        r: Ring-plane radii (nonzero where meaningful; a zero radius yields
            a scale of 1 to keep the division benign at the exact center).
        x: Node-aligned in-plane x coordinates.
        y: Node-aligned in-plane y coordinates.
        opening_deg_obs: Observer ring opening angle B in degrees, nonzero.

    Returns:
        The dimensionless radial foreshortening scale, >= 1.
    """
    sin_b = math.sin(math.radians(opening_deg_obs))
    scale = np.sqrt(x * x + (y * y) / (sin_b * sin_b))
    return cast(NDArrayFloatType, np.where(r > 0.0, scale / np.where(r > 0.0, r, 1.0), 1.0))


def compute_antialiasing_shade(edge_dist: NDArrayFloatType, resolution: float) -> NDArrayFloatType:
    """Compute anti-aliasing shade from signed edge distance.

    Callers pass the signed distance with the covered side positive: a ring
    feature passes ``r - r_edge`` (or ``r_outer - r``) so a pixel inside the
    band shades toward 1, and a moonlet disc passes ``radius - dist`` so a
    pixel inside the disc shades toward 1.

    Parameters:
        edge_dist: Signed distance from pixel center to the edge (positive =
            inside the covered feature, negative = outside).
        resolution: Pixel resolution for anti-aliasing (the shade ramps over
            one such window centered on the edge).

    Returns:
        Anti-aliasing shade value [0, 1]; 0.5 means the pixel center sits
        exactly on the edge, 1 fully covered, 0 fully outside.
    """
    shade = 0.5 + edge_dist / resolution
    shade[shade < 0.0] = 0.0
    shade[shade > 1.0] = 1.0
    return shade
