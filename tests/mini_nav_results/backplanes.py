"""The backplane products one cohort image leaves under a backplane root.

A FITS of one image HDU per backplane, a per-pixel body identity map, and the
metadata document beside them whose statistics the global index tables are
built from.  All three are written by
:func:`spindoctor.cli.backplanes.writer.write_fits`, the same call the
backplane stage makes, over arrays synthesized here instead of over arrays a
``Backplane`` computed from SPICE.  A fixture written by a second, parallel
writer stops describing the product the moment the real writer changes, and
what every phase after this one reads is the product.

The writer reads four things from the observation it is handed --
``is_simulated``, ``sim_inventory``, ``closest_planet`` and ``inventory()`` --
and reads the last two only when the first is False, so an observation that
reports itself simulated and carries an inventory dict reaches no SPICE and no
image.  That is what stands in for a snapshot here.

The planes are the ones the shipped configuration declares, and their values
are ramps between the bounds one plane of that name spans.  A plane the
configuration declares and this module has no bounds for stops the build: a
plane written with invented bounds would put a global index column outside the
range its own arrays cover, which is the disagreement these products exist to
be checked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from filecache import FCPath

from spindoctor.cli.backplanes.writer import write_fits
from spindoctor.config import MAIN_LOGGER, Config
from spindoctor.obs import ObsSnapshot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

COHORT_SHAPE_VU = (16, 16)
"""The pixel dimensions of a cohort frame.

A miniature of the 1024 by 1024 frame the camera reads out.  Every plane, every
mask and every browse image is sized from this, so the whole cohort is built
and torn down inside one test session; nothing in the bundle stage reads a
frame's size against the instrument mode its index row records.
"""

_PLANE_BOUNDS = {
    'body_longitude': (0.0, 2.0 * math.pi),
    'body_latitude': (-math.pi / 2.0, math.pi / 2.0),
    'body_incidence_angle': (0.0, math.pi),
    'body_emission_angle': (0.0, math.pi / 2.0),
    'body_phase_angle': (0.0, math.pi),
    'body_finest_resolution': (1.24, 2.35),
    'body_coarsest_resolution': (2.35, 8.06),
    'ring_radius': (74658.0, 136780.0),
    'ring_longitude': (0.0, 2.0 * math.pi),
    'ring_emission_angle': (0.0, math.pi / 2.0),
    'ring_phase_angle': (0.0, math.pi),
    'ring_radial_resolution': (2.11, 9.04),
    'ring_longitudinal_resolution': (1.4e-05, 3.9e-05),
}
"""What one plane of each name spans, in the units the configuration declares.

Radians for the angles and the longitudes, kilometres for the ring radii, and
km or radians per pixel for the resolutions -- the units the arrays carry.  The
ring radii span Saturn's main rings; the resolutions are what a Cassini frame
of a body a few hundred thousand kilometres away resolves.
"""


@dataclass(frozen=True)
class CohortBody:
    """A body one cohort image has backplanes for.

    Attributes:
        name: The body's name, as the inventory and the backplanes key it.
        naif_id: Its NAIF identifier, which is what the body identity map
            carries wherever the body claims a pixel.
        center_vu: Where the body's center sits in the frame, in pixels.
        radius_px: How far from that center the body's disc reaches.
        range_km: How far the body is from the observer.
    """

    name: str
    naif_id: int
    center_vu: tuple[float, float]
    radius_px: float
    range_km: float


class _SimulatedSnapshot:
    """The observation the writer is handed: simulated, with an inventory.

    Attributes:
        is_simulated: True, which is what routes the writer to the inventory
            below rather than to the SPICE-driven one.
        sim_inventory: Per-body inventory entries, keyed by body name.
    """

    def __init__(self, sim_inventory: dict[str, Any]) -> None:
        """Hold the inventory the writer reads.

        Parameters:
            sim_inventory: Per-body inventory entries, keyed by body name.
        """
        self.is_simulated = True
        self.sim_inventory = sim_inventory


def _ramp(
    bounds: tuple[float, float], mask: NDArrayBoolType, masked_value: float
) -> NDArrayFloatType:
    """Return one plane: a ramp across the frame, masked where nothing is seen.

    Parameters:
        bounds: The lowest and highest value the plane reaches.
        mask: True wherever the plane has a measurement.
        masked_value: What a pixel carries where it does not.

    Returns:
        The full-frame plane.
    """
    low, high = bounds
    size_v, size_u = COHORT_SHAPE_VU
    steps = np.linspace(0.0, 1.0, size_v * size_u, dtype=np.float64).reshape(COHORT_SHAPE_VU)
    plane = np.full(COHORT_SHAPE_VU, masked_value, dtype=np.float32)
    plane[mask] = (low + (high - low) * steps).astype(np.float32)[mask]
    return cast(NDArrayFloatType, plane)


def _bounds_for(name: str) -> tuple[float, float]:
    """Return what a plane of this name spans.

    Parameters:
        name: The plane's configured name.

    Returns:
        Its lowest and highest value.

    Raises:
        KeyError: If the configuration declares a plane this module has no
            bounds for.
    """
    try:
        return _PLANE_BOUNDS[name]
    except KeyError:
        raise KeyError(
            f'the cohort has no bounds for backplane {name!r}; add them beside the others'
        ) from None


def _statistics(
    planes: dict[str, NDArrayFloatType],
    masks: dict[str, NDArrayBoolType],
    units: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Return the per-plane statistics, as the backplane stage computes them.

    Over the valid pixels alone, and in degrees wherever the plane's declared
    units are radians, which is the convention the global index tables state.

    Parameters:
        planes: The full-frame planes, keyed by name.
        masks: True wherever a plane has a measurement, keyed by name.
        units: The units each plane's values are in, keyed by name.

    Returns:
        The lowest and highest value of each plane.
    """
    statistics: dict[str, dict[str, float]] = {}
    for name, plane in planes.items():
        values = plane[masks[name]]
        if units[name].lower() == 'rad':
            values = np.degrees(values)
        statistics[name] = {'min': float(np.nanmin(values)), 'max': float(np.nanmax(values))}
    return statistics


def _disc_mask(body: CohortBody) -> NDArrayBoolType:
    """Return the pixels a body's disc claims.

    Parameters:
        body: The body, with the center and radius its disc covers.

    Returns:
        True wherever the body is seen.
    """
    size_v, size_u = COHORT_SHAPE_VU
    grid_v, grid_u = np.mgrid[0:size_v, 0:size_u]
    center_v, center_u = body.center_vu
    distance = np.hypot(grid_v - center_v, grid_u - center_u)
    return cast(NDArrayBoolType, distance <= body.radius_px)


def write_backplanes(
    fits_file_path: FCPath,
    *,
    bodies: tuple[CohortBody, ...],
    rings: bool,
    config: Config,
) -> None:
    """Write one image's backplane FITS and the metadata document beside it.

    The run log takes the writer's own diagnostics: building a fixture is not
    navigating an image, and the image logger has no image open to file them
    under.

    Parameters:
        fits_file_path: Where the FITS goes; the metadata document is named
            from it, by the writer.
        bodies: The bodies the frame has backplanes for, each claiming a disc
            of the frame.
        rings: Whether the frame has ring backplanes, which claim every pixel
            no body does.
        config: The configuration whose declared planes, units and masked value
            the products are built from.
    """
    masked_value = float(config.backplanes.masked_value)
    body_units = {entry['name']: str(entry.get('units', '')) for entry in config.backplanes.bodies}
    ring_units = {
        entry['name']: str(entry.get('units', ''))
        for entry in config.backplanes.rings
        if entry['name'] != 'distance'
    }

    master_by_type: dict[str, NDArrayFloatType] = {}
    body_id_map: NDArrayIntType = np.zeros(COHORT_SHAPE_VU, dtype=np.int32)
    claimed: NDArrayBoolType = np.zeros(COHORT_SHAPE_VU, dtype=np.bool_)
    bodies_result: dict[str, Any] = {}
    sim_inventory: dict[str, Any] = {}

    for body in bodies:
        mask = _disc_mask(body) & ~claimed
        claimed |= mask
        body_id_map[mask] = body.naif_id
        planes = {name: _ramp(_bounds_for(name), mask, masked_value) for name in body_units}
        for name, plane in planes.items():
            master = master_by_type.setdefault(
                name, np.full(COHORT_SHAPE_VU, masked_value, dtype=np.float32)
            )
            master[mask] = plane[mask]
        bodies_result[body.name] = {
            'statistics': _statistics(planes, dict.fromkeys(planes, mask), body_units)
        }
        center_v, center_u = body.center_vu
        sim_inventory[body.name] = {
            'center_uv': [center_u, center_v],
            'range': body.range_km,
            'u_pixel_size': 2.0 * body.radius_px,
            'v_pixel_size': 2.0 * body.radius_px,
        }

    rings_result: dict[str, Any] | None = None
    if rings:
        ring_mask = ~claimed
        ring_planes = {
            name: _ramp(_bounds_for(name), ring_mask, masked_value) for name in ring_units
        }
        ring_masks = dict.fromkeys(ring_planes, ring_mask)
        master_by_type.update(ring_planes)
        rings_result = {'statistics': _statistics(ring_planes, ring_masks, ring_units)}

    write_fits(
        fits_file_path=fits_file_path,
        snapshot=cast(ObsSnapshot, _SimulatedSnapshot(sim_inventory)),
        master_by_type=master_by_type,
        body_id_map=body_id_map,
        config=config,
        bodies_result=bodies_result,
        rings_result=rings_result,
        logger=MAIN_LOGGER,
    )
