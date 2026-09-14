from typing import Any

import numpy as np
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import PlaneStatistics, plane_statistics
from spindoctor.config import Config
from spindoctor.obs import ObsSnapshot


def ring_target(planet: str) -> str:
    """Return the ring target an image's ring backplanes are computed for.

    Parameters:
        planet: The image's closest planet, as the observation names it.

    Returns:
        ``SATURN_MAIN_RINGS`` for Saturn, whose main rings the ring backplanes cover, and
        ``<PLANET>_RING_SYSTEM`` for any other planet.
    """
    # Saturn's ring backplanes cover its main rings: a rule about one planet in a module
    # whose name names none, which moving the choice into configuration would end (#618).
    if planet == 'SATURN':
        return 'SATURN_MAIN_RINGS'
    return f'{planet}_RING_SYSTEM'


def create_ring_backplanes(
    snapshot: ObsSnapshot, config: Config, *, logger: PdsLogger
) -> dict[str, Any] | None:
    """Create configured ring backplanes over the full image, if applicable.

    Returns an empty dict if no rings are configured or closest planet is None.

    Parameters:
        snapshot: The observation snapshot.
        config: The configuration.
        logger: The logger.

    Returns:
        A dictionary containing the ring backplanes and associated metadata using
        the following, or None if there is no closest planet or ring backplanes are
        not configured.

        - "planet": The closest planet name.
        - "target_key": The target key used for backplane generation.
        - "arrays": The ring backplane arrays.
        - "masks": The ring backplane masks.
        - "distance": The ring backplane distance.
        - "statistics": The ring backplane statistics, each stating the unit it
          is in, which is not the unit of the array it was taken from wherever
          the plane is angular.  See
          :mod:`spindoctor.cli.backplanes.statistics`.
    """

    masked_value = float(config.backplanes.masked_value)

    result: dict[str, Any] = {
        'planet': None,
        'target_key': None,
        'arrays': {},
        'masks': {},
        'distance': None,  # per-pixel distance array
    }

    if snapshot.is_simulated:
        return None

    closest_planet = snapshot.closest_planet
    if closest_planet is None:
        # No planet, no rings
        return None

    rings_cfg = getattr(config.backplanes, 'rings', None)
    if rings_cfg is None:
        raise ValueError('Configuration has no rings section for backplanes')

    target_key = ring_target(closest_planet)

    bp = snapshot.bp

    result['planet'] = closest_planet
    result['target_key'] = target_key
    ring_stats: dict[str, PlaneStatistics] = {}

    for bp_cfg in rings_cfg:
        bp_name = bp_cfg['name']
        method = bp_cfg.get('method')
        if method is None:
            raise ValueError(f'Ring backplane "method" is required: {bp_name}')
        units = bp_cfg.get('units')
        if units is None:
            raise ValueError(f'Ring backplane "units" is required: {bp_name}')

        logger.debug(f'{closest_planet}: Creating ring backplane {bp_name}')
        func = getattr(bp, method)
        vals = func(target_key)

        if method == 'distance':
            # Also save as the per-pixel distance field for merge ordering
            result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

        mvals = vals.mvals
        full = np.asarray(np.ma.filled(mvals, fill_value=masked_value), dtype=np.float32)
        mask = ~np.ma.getmaskarray(mvals)

        if np.any(mask):
            result['arrays'][bp_name] = full
            result['masks'][bp_name] = mask

        # Calculate min/max statistics
        valid_values = full[mask]
        if len(valid_values) > 0:
            ring_stats[bp_name] = plane_statistics(valid_values, units=units)

    result['statistics'] = ring_stats

    # Ensure distance is present
    if result['distance'] is None:
        # If not explicitly configured, compute per-pixel distance for ordering
        vals = bp.distance(target_key, direction='dep')
        result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

    return result
