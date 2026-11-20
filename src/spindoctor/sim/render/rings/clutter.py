"""Truth-side clutter of the ring system: azimuthal structure, propellers, and moonlets.

None of this is catalog knowledge, so the navigator never predicts it.  The azimuthal
factor scales the emitted brightness without touching optical depth; a propeller
multiplies the composed optical depth around its moonlet; and a moonlet is an opaque
disc at the ring's depth, drawn with its coverage on the render grid.
"""

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from spindoctor.sim.render.rings.profiles import FOOTPRINT_HALF_WIDTH_PX
from spindoctor.sim.scene.fields import scene_value
from spindoctor.sim.shared.ring import compute_antialiasing_shade, ring_sky_from_plane
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'azimuthal_intensity_factor',
    'moonlet_disc_coverage',
    'propeller_radial_reach',
    'propeller_tau_factor',
]

# exp(-0.5 * x**2) underflows to exactly 0.0 in float64 once x passes about
# 38.7 (0.5 * x**2 > ~745.2), so a propeller's Gaussian lobes are bitwise
# zero beyond this many sigmas; the radial bound uses it to stay exact.
_GAUSSIAN_UNDERFLOW_SIGMAS: float = 40.0


def propeller_radial_reach(
    moonlet: Mapping[str, Any], propeller: Mapping[str, Any], *, os: int
) -> tuple[float, float]:
    """Return a propeller's center radius and how far its lobes reach from it.

    Beyond the reach both Gaussian lobes are exactly 0.0, so the propeller's tau factor
    is exactly 1.0 there.

    Parameters:
        moonlet: The moonlet mapping (placement).
        propeller: The moonlet's ``propeller`` mapping.
        os: The oversampling factor.

    Returns:
        ``(center, reach)`` in render-grid ring-plane units.
    """
    center = float(scene_value(moonlet, 'a_px', 0.0)) * os
    width = float(scene_value(propeller, 'width_px', 0.0)) * os
    return center, width * (1.0 + _GAUSSIAN_UNDERFLOW_SIGMAS)


def _wrap_dlam(lam: NDArrayFloatType, center_rad: float) -> NDArrayFloatType:
    """Longitude differences wrapped into [-pi, pi)."""
    return cast(NDArrayFloatType, np.mod(lam - center_rad + math.pi, 2.0 * math.pi) - math.pi)


def azimuthal_intensity_factor(
    azimuthal: Mapping[str, Any],
    *,
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    os: int,
    spokes_seed: int,
) -> NDArrayFloatType:
    """The truth-side azimuthal brightness factor (intensity only, never tau).

    - ``modulation``: ``1 + amplitude * cos(m * (lam - brightness_peak_longitude_deg))`` --
      the self-gravity-wake quadrant asymmetry.
    - ``shadow``: a multiplicative ``1 - darkness`` wedge over
      ``[start_deg, start_deg + extent_deg)`` -- the planet-shadow boundary,
      a strong non-navigable edge crossing every feature.
    - ``spokes``: ``count`` seeded wedges, azimuthally sharp (quartic wedge
      profile) and radially broad (sin^2 envelope over
      ``r_inner_px..r_outer_px``),
      each with a drawn contrast in ``[0.5, 1] * contrast`` (negative for
      the dark low-phase appearance) and a drawn width in ``[0.5, 1.5] *
      width_deg``.

    The combined factor clips at zero: azimuthal structure can darken to
    black but never drive the emission negative.

    Parameters:
        azimuthal: The validated ``ring_system.azimuthal`` mapping.
        r: Per-pixel ring-plane radii on the render grid.
        lam: Per-pixel ring-plane longitudes (radians).
        os: The oversampling factor (scales the spoke radial band).
        spokes_seed: The spoke field's realization seed.

    Returns:
        The per-pixel intensity factor (non-negative).
    """
    factor = np.ones_like(r)
    modulation = azimuthal.get('modulation')
    if modulation:
        amplitude = float(scene_value(modulation, 'amplitude', 0.0))
        m = int(scene_value(modulation, 'm', 1))
        phase = math.radians(float(scene_value(modulation, 'brightness_peak_longitude_deg', 0.0)))
        factor = factor * (1.0 + amplitude * np.cos(m * (lam - phase)))
    shadow = azimuthal.get('shadow')
    if shadow:
        start = math.radians(float(scene_value(shadow, 'start_deg', 0.0)))
        extent = math.radians(float(scene_value(shadow, 'extent_deg', 0.0)))
        darkness = float(scene_value(shadow, 'darkness', 0.0))
        in_wedge = np.mod(lam - start, 2.0 * math.pi) < extent
        factor = factor * np.where(in_wedge, 1.0 - darkness, 1.0)
    spokes = azimuthal.get('spokes')
    if spokes:
        rng = np.random.default_rng(spokes_seed)
        count = int(scene_value(spokes, 'count', 0))
        r_inner = float(scene_value(spokes, 'r_inner_px', 0.0)) * os
        r_outer = float(scene_value(spokes, 'r_outer_px', 0.0)) * os
        contrast = float(scene_value(spokes, 'contrast', 0.0))
        width_deg = float(scene_value(spokes, 'width_deg', 0.0))
        band = np.clip((r - r_inner) / max(r_outer - r_inner, 1e-12), 0.0, 1.0)
        radial_env = np.sin(math.pi * band) ** 2
        radial_env[(r < r_inner) | (r > r_outer)] = 0.0
        spoke_sum = np.zeros_like(r)
        for _ in range(count):
            lam_k = rng.uniform(0.0, 2.0 * math.pi)
            half_width = math.radians(width_deg * rng.uniform(0.5, 1.5)) / 2.0
            c_k = contrast * rng.uniform(0.5, 1.0)
            dlam = _wrap_dlam(lam, lam_k)
            wedge = np.clip(1.0 - (dlam / half_width) ** 4, 0.0, 1.0)
            spoke_sum += c_k * wedge
        factor = factor * (1.0 + spoke_sum * radial_env)
    clipped: NDArrayFloatType = np.clip(factor, 0.0, None)
    return clipped


def propeller_tau_factor(
    moonlet: Mapping[str, Any],
    propeller: Mapping[str, Any],
    *,
    r: NDArrayFloatType,
    lam: NDArrayFloatType,
    os: int,
) -> NDArrayFloatType:
    """A propeller's multiplicative tau disturbance around its moonlet.

    Two Gaussian lobes straddle the moonlet radially by ``width_px`` and
    azimuthally by ``length_deg / 2`` in opposite senses (the leading lobe
    outward, the trailing lobe inward -- the stylized S shape of a real
    propeller), each with radial sigma ``width_px`` and azimuthal sigma
    ``length_deg / 4``.  ``contrast`` scales the disturbance: negative
    values carve the local partial gaps a physical propeller opens, positive
    values pile density up.  The factor clips at zero.

    Parameters:
        moonlet: The moonlet mapping (placement).
        propeller: The moonlet's ``propeller`` mapping.
        r: Per-pixel ring-plane radii on the render grid.
        lam: Per-pixel ring-plane longitudes (radians).
        os: The oversampling factor.

    Returns:
        The per-pixel multiplicative tau factor (non-negative).
    """
    a_m = float(scene_value(moonlet, 'a_px', 0.0)) * os
    lam_m = math.radians(float(scene_value(moonlet, 'lam_deg', 0.0)))
    length = math.radians(float(scene_value(propeller, 'length_deg', 0.0)))
    width = float(scene_value(propeller, 'width_px', 0.0)) * os
    contrast = float(scene_value(propeller, 'contrast', 0.0))
    sigma_lam = length / 4.0
    disturbance = np.zeros_like(r)
    for radial_sign, azimuthal_sign in ((1.0, 1.0), (-1.0, -1.0)):
        dr = (r - (a_m + radial_sign * width)) / width
        dlam = _wrap_dlam(lam, lam_m + azimuthal_sign * length / 2.0) / sigma_lam
        disturbance += np.exp(-0.5 * (dr * dr + dlam * dlam))
    return cast(NDArrayFloatType, np.clip(1.0 + contrast * disturbance, 0.0, None))


def moonlet_disc_coverage(
    moonlet: Mapping[str, Any],
    shape: tuple[int, int],
    *,
    center_v: float,
    center_u: float,
    b_obs: float,
    node_deg: float,
    os: int,
) -> tuple[tuple[slice, slice], NDArrayFloatType] | None:
    """A moonlet's anti-aliased disc coverage, bounded to its sky box.

    The moonlet sits in the ring plane at polar placement ``(a_px, lam_deg)``,
    projected to the sky through the shared projection; the disc is drawn in
    sky coordinates (a body, not a ring-plane band, so it does not
    foreshorten with the ring).  The coverage ramps across one render-grid
    pixel, so on an oversampled grid the box downsample blurs the disc's rim
    only once.  It is exactly 0.0 wherever the pixel center sits half a
    pixel or more outside the disc radius, so the evaluation is restricted
    to the bounding box of that reach (plus one pixel of slack for the half
    pixel between the coordinates the reach is stated in and the cells the
    box names) and the caller composites only inside it -- an exact
    restriction, not an approximation.

    Parameters:
        moonlet: The moonlet mapping.
        shape: The render-grid shape ``(V, U)``.
        center_v: Ring-system center v on the render grid.
        center_u: Ring-system center u on the render grid.
        b_obs: Observer ring opening angle in degrees.
        node_deg: Sky position angle of the ascending node in degrees.
        os: The oversampling factor.

    Returns:
        ``((v_slice, u_slice), coverage)`` with the box-sized per-pixel
        coverage in [0, 1], or None when the box misses the grid entirely.
    """
    size_v, size_u = shape
    a_m = float(scene_value(moonlet, 'a_px', 0.0)) * os
    lam_m = math.radians(float(scene_value(moonlet, 'lam_deg', 0.0)))
    radius = float(scene_value(moonlet, 'radius_px', 1.0)) * os
    dv, du = ring_sky_from_plane(
        np.asarray([a_m]), np.asarray([lam_m]), opening_deg_obs=b_obs, node_deg=node_deg
    )
    pos_v = center_v + float(dv[0])
    pos_u = center_u + float(du[0])
    reach = radius + FOOTPRINT_HALF_WIDTH_PX
    v0 = max(math.floor(pos_v - reach) - 1, 0)
    v1 = min(math.ceil(pos_v + reach) + 1, size_v)
    u0 = max(math.floor(pos_u - reach) - 1, 0)
    u1 = min(math.ceil(pos_u + reach) + 1, size_u)
    if v0 >= v1 or u0 >= u1:
        return None
    # The box is named by array rows and columns and the moonlet's position is
    # stated in the geometry layer's pixel corner coordinates, so the half
    # pixel goes on here.
    v_centers = np.arange(v0, v1, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    u_centers = np.arange(u0, u1, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
    box_v, box_u = np.meshgrid(v_centers, u_centers, indexing='ij')
    dist = stable_hypot(box_v - pos_v, box_u - pos_u)
    coverage = compute_antialiasing_shade(radius - dist, 2.0 * FOOTPRINT_HALF_WIDTH_PX)
    return (slice(v0, v1), slice(u0, u1)), coverage
