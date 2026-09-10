"""The backplane products one cohort image leaves under a backplane root.

A FITS of one image HDU per backplane, a per-pixel body identity map, and the
metadata document beside them whose statistics the global index tables are
built from.  What this module synthesizes is what a ``Backplane`` computes from
SPICE: one array and one mask per plane per source, and the range to each
source.  Everything downstream of that is the backplane stage's own code --
:func:`spindoctor.cli.backplanes.merge.merge_sources_into_master` resolves the
sources into one array per plane and the body identity map, and
:func:`spindoctor.cli.backplanes.writer.write_fits` writes the FITS and the
metadata document from what it returns.

The merge belongs here as much as the writer does, because it is the merge that
decides what order the HDUs are written in: it inserts the body planes in
sorted order and then the ring planes in sorted order, and the writer walks that
dict as it found it.  A fixture that assembled the same dict itself would be a
second implementation of the merge, and the first thing it would get wrong is
the order every array's byte offset in the file is stated against.

The two stages read six things between them from the observation they are
handed -- ``data``, ``config`` and ``is_simulated`` in the merge,
``is_simulated``, ``sim_inventory``, ``closest_planet`` and ``inventory()`` in
the writer -- and read the last two only when ``is_simulated`` is False, so an
observation that reports itself simulated and carries an inventory dict reaches
no SPICE and no image.  That is what stands in for a snapshot here.

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

from spindoctor.cli.backplanes.merge import merge_sources_into_master
from spindoctor.cli.backplanes.writer import write_fits
from spindoctor.config import MAIN_LOGGER, Config
from spindoctor.obs import ObsSnapshot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

COHORT_SHAPE_VU = (16, 16)
"""The pixel dimensions of a cohort frame.

A miniature of the 1024 by 1024 frame the camera reads out.  Every plane, every
mask and every browse image is sized from this, so the whole cohort is built
and torn down inside one test session.

It is smaller than any real Cassini readout, which leaves one thing about the
product untrue: the index row records an instrument mode of ``FULL``, and no
Cassini mode names a 16 pixel frame, so there is no value it could record
instead.  A reader holding a frame's size against the mode beside it therefore
finds a frame this size claiming to be a full one.
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

    The body's NAIF identifier is not stated here.  The merge reads it from the
    name, as it does for a real image, so the identity map cannot carry an
    identifier the name does not resolve to.

    Every value here differs between the two axes, deliberately.  The frame is
    square and the products state a body's center and its extent as pairs, so a
    body centered on the frame with a circular disc is one a swapped pair
    describes exactly as well -- and the writer transposes one of those pairs
    and not the other.  An off-center body with an elliptical disc is what
    makes the axis order load-bearing, and it costs the fixture nothing.

    Attributes:
        name: The body's name, as the inventory and the backplanes key it.
        center_vu: Where the body's center sits in the frame, in pixels, down
            the frame first and across it second.
        radii_vu: How far from that center the body's disc reaches along each
            axis, in the same order.
        range_km: How far the body is from the observer.
    """

    name: str
    center_vu: tuple[float, float]
    radii_vu: tuple[float, float]
    range_km: float


class _SimulatedSnapshot:
    """The observation the merge and the writer are handed.

    Simulated, with an inventory: that is what routes both stages away from
    SPICE and away from the image, and it is the whole of what either reads.

    Attributes:
        is_simulated: True, which is what routes the writer to the inventory
            below rather than to the SPICE-driven one, and what lets the merge
            name a body the kernels do not know.
        sim_inventory: Per-body inventory entries, keyed by body name.
        data: An array of the frame's shape, which is where the merge reads how
            large a plane is.  Its values are never read.
        config: The configuration whose masked value the merge fills with.
    """

    def __init__(self, sim_inventory: dict[str, Any], config: Config) -> None:
        """Hold what the merge and the writer read off an observation.

        Parameters:
            sim_inventory: Per-body inventory entries, keyed by body name.
            config: The configuration the merge reads the masked value from.
        """
        self.is_simulated = True
        self.sim_inventory = sim_inventory
        self.data: NDArrayFloatType = np.zeros(COHORT_SHAPE_VU, dtype=np.float32)
        self.config = config


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
        body: The body, with the center its disc is drawn around and how far it
            reaches along each axis.

    Returns:
        True wherever the body is seen.
    """
    size_v, size_u = COHORT_SHAPE_VU
    grid_v, grid_u = np.mgrid[0:size_v, 0:size_u]
    center_v, center_u = body.center_vu
    radius_v, radius_u = body.radii_vu
    reach = np.hypot((grid_v - center_v) / radius_v, (grid_u - center_u) / radius_u)
    return cast(NDArrayBoolType, reach <= 1.0)


def _ring_distance(bodies: tuple[CohortBody, ...]) -> NDArrayFloatType:
    """Return how far the ring system is at each pixel, as the merge reads it.

    The merge drops a ring pixel a nearer body claims, so what this says about
    the rings decides which of the two the frame keeps where they overlap.  It
    puts the rings behind every body in the frame, which is what leaves each
    body's disc to the body and every other pixel to the rings -- the same
    division the ring mask already carries, stated where the merge reads it so
    the two cannot disagree.

    Parameters:
        bodies: The bodies in the frame, whose ranges the rings sit beyond.

    Returns:
        The full-frame distance.
    """
    beyond = max((body.range_km for body in bodies), default=0.0) + 1.0
    return cast(NDArrayFloatType, np.full(COHORT_SHAPE_VU, beyond, dtype=np.float32))


def write_backplanes(
    fits_file_path: FCPath,
    *,
    bodies: tuple[CohortBody, ...],
    rings: bool,
    config: Config,
) -> None:
    """Write one image's backplane FITS and the metadata document beside it.

    The per-source arrays, masks and ranges are synthesized here; the merge and
    the writer take them from there, so the HDU order, the body identity map and
    the metadata document are the backplane stage's own answers rather than a
    second set that agrees with them until one of the two changes.

    The run log takes the writer's own diagnostics: building a fixture is not
    navigating an image, and the image logger has no image open to file them
    under.

    Parameters:
        fits_file_path: Where the FITS goes; the metadata document is named
            from it, by the writer.
        bodies: The bodies the frame has backplanes for, each claiming a disc
            of the frame.
        rings: Whether the frame has ring backplanes, which claim every pixel
            no body does.  A frame with none still carries a ring result with
            nothing in it, as a real frame whose rings are out of the field
            does.
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

    # Every pixel some body covers, which is where the rings are not seen.
    claimed: NDArrayBoolType = np.zeros(COHORT_SHAPE_VU, dtype=np.bool_)
    bodies_result: dict[str, Any] = {}
    sim_inventory: dict[str, Any] = {}

    for body in bodies:
        # The disc the body covers, whole: which of two overlapping bodies owns
        # a pixel is the merge's decision, taken on their ranges, and deciding
        # it here as well would be a second answer that agrees only while the
        # nearer body happens to be declared first.
        mask = _disc_mask(body)
        claimed |= mask
        planes = {name: _ramp(_bounds_for(name), mask, masked_value) for name in body_units}
        masks = dict.fromkeys(planes, mask)
        bodies_result[body.name] = {
            'arrays': planes,
            'masks': masks,
            'distance': body.range_km,
            'statistics': _statistics(planes, masks, body_units),
        }
        center_v, center_u = body.center_vu
        radius_v, radius_u = body.radii_vu
        sim_inventory[body.name] = {
            'center_uv': [center_u, center_v],
            'range': body.range_km,
            'u_pixel_size': 2.0 * radius_u,
            'v_pixel_size': 2.0 * radius_v,
        }

    # A frame with no ring pixels in view still carries a ring result, holding
    # nothing: that is what the ring stage returns for a real frame whose rings
    # are all out of the field, and it is the shape -- "rings" naming an empty
    # "backplanes" rather than being empty itself -- that the collections read.
    ring_mask = ~claimed
    ring_planes = (
        {name: _ramp(_bounds_for(name), ring_mask, masked_value) for name in ring_units}
        if rings
        else {}
    )
    ring_masks = dict.fromkeys(ring_planes, ring_mask)
    rings_result: dict[str, Any] = {
        'arrays': ring_planes,
        'masks': ring_masks,
        'distance': _ring_distance(bodies),
        'statistics': _statistics(ring_planes, ring_masks, ring_units),
    }

    snapshot = cast(ObsSnapshot, _SimulatedSnapshot(sim_inventory, config))
    master_by_type, body_id_map = merge_sources_into_master(
        snapshot, bodies_result=bodies_result, rings_result=rings_result
    )
    write_fits(
        fits_file_path=fits_file_path,
        snapshot=snapshot,
        master_by_type=master_by_type,
        body_id_map=body_id_map,
        config=config,
        bodies_result=bodies_result,
        rings_result=rings_result,
        logger=MAIN_LOGGER,
    )
