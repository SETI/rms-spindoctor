"""Crater carving for the ellipsoid renderer.

:func:`carve_crater_heights` carves a body's crater height field on a pixel grid,
and :func:`perturb_normals_by_slope` tilts the surface normals by the slopes of that
field.  The ellipsoid renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`)
carves its craters on the detector grid and shades them on its working grid.

The crater centers are drawn among the pixels of a grid that holds the body.
Usually that is the grid the craters are carved on.  A body too large to be drawn
whole is drawn only in part, and its centers are drawn on a larger grid, a
:class:`CraterSite`, that holds the whole body; the carved grid is a box of it.

The crater knobs live only on this side of the information boundary: they are
truth keys the navigator never sees (its best model is the smooth Lambert
ellipsoid).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = ['CraterSite', 'carve_crater_heights', 'mask_site', 'perturb_normals_by_slope']

# Candidate pixels drawn per round of the crater center search
_CENTER_CANDIDATES_PER_ROUND = 256


@dataclass(frozen=True)
class CraterSite:
    """The grid crater centers are drawn on, when it is not the carved grid's own body.

    The site is a larger grid when the carved grid holds only part of the body, and
    the carved grid's pixels inside the unrolled body when a planted roll turned the
    body (see :func:`mask_site`).

    Parameters:
        shape: The grid's rows and columns.
        inside: Takes flat pixel indices on the grid and returns whether each pixel
            lies strictly inside the body.  It must find at least one such pixel
            whenever a crater is placed.
        origin: The grid row and column of the carved grid's first pixel.
    """

    shape: tuple[int, int]
    inside: Callable[[NDArrayIntType], NDArrayBoolType]
    origin: tuple[int, int]


def mask_site(mask: NDArrayBoolType) -> CraterSite:
    """The crater site of a grid's own pixels: those where ``mask`` is True.

    Parameters:
        mask: Whether each pixel of the grid lies strictly inside the body.

    Returns:
        The site, whose origin is the grid's first pixel.
    """
    flat_mask = mask.ravel()
    return CraterSite(
        shape=(mask.shape[0], mask.shape[1]),
        inside=lambda indices: flat_mask[indices],
        origin=(0, 0),
    )


def carve_crater_heights(
    ellipse_mask_nz: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    crater_seed: int,
    n_craters: int,
    R_min: float,
    R_max: float,
    crater_power_law_exponent: float,
    crater_relief_scale: float,
    work_center_v: float,
    work_center_u: float,
    site: CraterSite | None = None,
    turn: float = 0.0,
) -> NDArrayFloatType:
    """Carve the crater height field (bowl + walls + raised rim per crater).

    The ellipsoid renderer carves its craters on the detector grid; the draw rule
    below is its determinism contract for a given seed and grid.

    Each crater draws from its own stream, seeded by
    ``derive_effect_seed(crater_seed, 'crater:<index>')``: one uniform value for the
    radius, then one normal value for the depth scatter, then its center (see
    :func:`_draw_crater_center`).  A crater's values therefore depend only on the
    seed, its index, the dimensions of the grid its center is drawn on, and which
    of the candidate pixels its own stream picks lie inside the body.  Changing one
    interior pixel moves only a crater whose center search reaches that pixel, and
    a change in the crater count adds or drops craters at the end of the list
    without touching the others.

    Parameters:
        ellipse_mask_nz: Boolean mask of pixels strictly inside the ellipse.
        v_coords: Centered v pixel coordinates of the grid.
        u_coords: Centered u pixel coordinates of the grid.
        crater_seed: The body's crater seed, from which each crater derives its stream.
        n_craters: Number of craters to place.
        R_min: Minimum crater radius in pixels.
        R_max: Maximum crater radius in pixels.
        crater_power_law_exponent: Power law exponent for crater radii, above 1.
        crater_relief_scale: Scale factor for crater depth.
        work_center_v: Body center v on the grid.
        work_center_u: Body center u on the grid.
        site: The grid the crater centers are drawn on, or None to draw them
            among the carved grid's own pixels inside ``ellipse_mask_nz``.
        turn: The angle in radians each drawn center is turned by about the body
            center, from ``+v`` toward ``+u``.  A body turned by a planted roll
            draws its centers among the pixels inside the unrolled body, and this
            turns them onto the rolled one, so its craters are the unrolled body's
            turned with it.

    Returns:
        The crater height field over the grid, in its pixels.

    Raises:
        ValueError: If ``crater_power_law_exponent`` is not above 1 and a crater is
            placed.
    """
    height = np.zeros_like(ellipse_mask_nz, dtype=float)
    if n_craters <= 0 or not ellipse_mask_nz.any():
        return height
    if site is None:
        site = mask_site(ellipse_mask_nz)
    if crater_power_law_exponent <= 1:
        raise ValueError('alpha must be > 1 for a proper power law.')

    # Radius distribution: power law p(R) ~ R^(-alpha) on [R_min, R_max]
    power = 1.0 - crater_power_law_exponent
    R_min_a = R_min**power
    R_max_a = R_max**power

    # Distances from each crater center are taken in the frame's own pixel
    # corner coordinates so the two frames cannot disagree.  v_coords and
    # u_coords are pixel corner coordinates with the body center already taken
    # off, so adding it back restores them.
    v_abs = v_coords + work_center_v
    u_abs = u_coords + work_center_u

    for index in range(n_craters):
        rng = np.random.default_rng(derive_effect_seed(crater_seed, f'crater:{index}'))
        crater_radius = float((R_min_a + rng.uniform() * (R_max_a - R_min_a)) ** (1.0 / power))
        # Multiplicative depth scatter around the size-dependent mean of the depth law
        noise = np.exp(rng.normal(loc=0.0, scale=0.25))
        v_crater, u_crater = _draw_crater_center(site, rng)

        # The crater center is an array row and column, so the half pixel goes on here.
        center_v = v_crater + PIXEL_CENTER_TO_CORNER_PX
        center_u = u_crater + PIXEL_CENTER_TO_CORNER_PX
        if turn != 0.0:
            rel_v = center_v - work_center_v
            rel_u = center_u - work_center_u
            center_v = work_center_v + math.cos(turn) * rel_v - math.sin(turn) * rel_u
            center_u = work_center_u + math.sin(turn) * rel_v + math.cos(turn) * rel_u
        v_dist = v_abs - center_v
        u_dist = u_abs - center_u
        crater_dist = np.sqrt(v_dist**2 + u_dist**2)

        # Mask where crater affects height (slightly beyond radius for rim)
        # TODO make config parameter
        crater_mask = (crater_dist < crater_radius * 1.1) & ellipse_mask_nz

        crater_depth = _crater_depth(
            crater_radius, noise, R_max=R_max, crater_relief_scale=crater_relief_scale
        )
        # Add crater relief to global heightmap
        profile = _crater_profile(crater_dist[crater_mask], crater_radius, crater_depth)
        height[crater_mask] += profile

    return height


def _crater_depth(
    crater_radius: float, noise: float, *, R_max: float, crater_relief_scale: float
) -> float:
    """The depth of one crater: small craters are deeper for their size, large ones shallower.

    The depth-to-diameter ratio blends from 0.20 at a 10-pixel diameter to 0.07 at
    the largest diameter, twice ``R_max``, and is scattered by ``noise``.

    Parameters:
        crater_radius: The crater's radius in pixels.
        noise: The crater's multiplicative depth scatter.
        R_max: The largest crater radius in pixels.
        crater_relief_scale: Scale factor for crater depth.

    Returns:
        The crater's depth, in pixels.
    """
    D = 2.0 * crater_radius  # diameter

    # Choose size range where we blend d/D ratio
    D_small = 10.0
    D_large = R_max * 2.0  # roughly largest diameters

    # Blend factor in [0,1]
    t = np.clip((D - D_small) / (D_large - D_small + 1e-9), 0.0, 1.0)
    # d/D: small -> 0.20, large -> 0.07
    d_over_D_mean = (1.0 - t) * 0.20 + t * 0.07
    d_over_D = d_over_D_mean * noise

    return float(crater_relief_scale * d_over_D * D)


def _crater_profile(
    r: NDArrayFloatType, crater_radius: float, crater_depth: float
) -> NDArrayFloatType:
    """The height profile of one crater: a bowl, walls, and a raised rim.

    Parameters:
        r: The distance of each affected pixel from the crater center, in pixels.
        crater_radius: The crater's radius in pixels, where the rim peaks.
        crater_depth: The crater's depth in pixels.

    Returns:
        The height at each distance in ``r``.
    """
    # TODO make config parameters
    R_floor = 0.6 * crater_radius  # flat-ish floor
    R_rim = crater_radius  # inner rim radius
    R_outer = 1.3 * crater_radius  # where rim merges back to surface

    local_profile = np.zeros_like(r)

    # Central bowl / floor (parabolic-ish)
    inside_floor = r <= R_floor
    local_profile[inside_floor] = -crater_depth * (1.0 - (r[inside_floor] / R_floor) ** 2)

    # Wall up to rim
    wall = (r > R_floor) & (r <= R_rim)
    t_wall = (r[wall] - R_floor) / (R_rim - R_floor + 1e-9)
    local_profile[wall] = -crater_depth * (1.0 - t_wall)

    # Raised rim outside
    rim = (r > R_rim) & (r <= R_outer)
    t_rim = (r[rim] - R_rim) / (R_outer - R_rim + 1e-9)
    rim_height = crater_depth * 0.25
    local_profile[rim] = rim_height * (1.0 - t_rim)
    return local_profile


def _draw_crater_center(site: CraterSite, rng: np.random.Generator) -> tuple[int, int]:
    """Draw a crater center uniformly among the pixels of a site inside the body.

    Candidates are drawn in rounds of a fixed number of pixels, each uniform over
    the site's whole grid, and the first candidate inside the body is the center.
    The candidates depend only on the stream and the grid's dimensions, never on
    which pixels lie inside, so changing one pixel changes the center only when one
    of the candidates up to and including the chosen one is that pixel.

    Parameters:
        site: The grid the center is drawn on; at least one of its pixels must lie
            inside the body.
        rng: The crater's own random stream.

    Returns:
        The center's array row and column on the carved grid.
    """
    n_rows, n_cols = site.shape
    while True:
        candidates = rng.integers(n_rows * n_cols, size=_CENTER_CANDIDATES_PER_ROUND)
        hits = candidates[site.inside(candidates)]
        if hits.size:
            row, col = divmod(int(hits[0]), n_cols)
            return row - site.origin[0], col - site.origin[1]


def perturb_normals_by_slope(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    slope_v: NDArrayFloatType,
    slope_u: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Perturb base-surface unit normals by the slopes of a height field carved along z.

    The height field is carved along the viewing axis, so the exact normal
    of the combined surface (base + height) is proportional to
    ``(-d(z + height)/dv, -d(z + height)/du, 1)``.  Scaling that vector by
    the base normal's z component rewrites it as the base normal plus a
    height-gradient term, which stays finite at the limb where the base
    surface gradient diverges.

    Parameters:
        normal_v: V component of the base unit surface normal.
        normal_u: U component of the base unit surface normal.
        normal_z: Z (toward-observer) component of the base unit normal.
        slope_v: The height field's slope along v (height per pixel of the same
            length unit).
        slope_u: The height field's slope along u.

    Returns:
        Tuple of (v, u, z) perturbed unit-normal component arrays.
    """
    perturbed_v = normal_v - normal_z * slope_v
    perturbed_u = normal_u - normal_z * slope_u
    perturbed_mag = np.sqrt(perturbed_v**2 + perturbed_u**2 + normal_z**2)
    perturbed_mag = np.maximum(perturbed_mag, 1e-10)  # Avoid division by zero
    perturbed_v /= perturbed_mag
    perturbed_u /= perturbed_mag
    perturbed_z = normal_z / perturbed_mag
    return perturbed_v, perturbed_u, perturbed_z
