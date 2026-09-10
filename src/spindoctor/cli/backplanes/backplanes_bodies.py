import hashlib
from typing import Any

import numpy as np
from oops.backplane import Backplane
from oops.meshgrid import Meshgrid
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import PlaneStatistics, plane_statistics
from spindoctor.config import Config
from spindoctor.obs import ObsSnapshot


def _create_simulated_body_backplane(
    snapshot: ObsSnapshot,
    body_name: str,
    backplane_name: str,
    v0: int,
    v1: int,
    u0: int,
    u1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simulated backplane array and mask for a body.

    Parameters:
        snapshot: The observation snapshot.
        body_name: Name of the body.
        backplane_name: Name of the backplane type.
        v0: Minimum v coordinate (inclusive).
        v1: Maximum v coordinate (inclusive).
        u0: Minimum u coordinate (inclusive).
        u1: Maximum u coordinate (inclusive).

    Returns:
        A tuple containing (full_array, full_mask) where full_array is the
        backplane values embedded in a full-frame array and full_mask is the
        corresponding boolean mask.
    """

    masked_value = float(snapshot.config.backplanes.masked_value)
    full = np.full(snapshot.data.shape, masked_value, dtype=np.float32)
    full_mask = np.zeros(snapshot.data.shape, dtype=bool)
    # hashlib, not hash(): str hashing is salted per interpreter run and the
    # simulated values must be deterministic
    digest = hashlib.sha256(f'{body_name}:{backplane_name}'.encode()).digest()
    seed = int.from_bytes(digest[:8], 'big') % (2**32)
    rng = np.random.default_rng(seed)
    val = float(rng.uniform(1.0, 100.0))
    sub_mask = None
    # Prefer body mask map if available
    mask_map = snapshot.sim_body_mask_map
    mask = mask_map.get(str(body_name).upper(), None)
    if mask is not None:
        sub_mask = mask[v0 : v1 + 1, u0 : u1 + 1].astype(bool)
    if sub_mask is None:
        try:
            # Build robust name list for matching
            order_names = [str(n).upper() for n in snapshot.sim_body_order_near_to_far]
            body_idx = order_names.index(str(body_name).upper()) + 1
            sim_map = snapshot.sim_body_index_map
            sub_mask = sim_map[v0 : v1 + 1, u0 : u1 + 1] == body_idx
        except Exception:
            # Fallback: fill entire rect (last resort)
            sub_mask = np.ones((v1 - v0 + 1, u1 - u0 + 1), dtype=bool)
    full_slice = full[v0 : v1 + 1, u0 : u1 + 1]
    # Only fill where this body contributed
    full_slice[sub_mask] = val
    full[v0 : v1 + 1, u0 : u1 + 1] = full_slice
    full_mask[v0 : v1 + 1, u0 : u1 + 1] = sub_mask
    return full, full_mask


def create_body_backplanes(
    snapshot: ObsSnapshot, config: Config, *, logger: PdsLogger
) -> dict[str, Any]:
    """Create configured body backplanes embedded in full-frame arrays.

    Parameters:
        snapshot: The observation snapshot.
        config: The configuration.
        logger: The logger.

    Returns:
        A dictionary keyed by body name containing the body backplanes as a dictionary with
        the following keys:

        - "arrays": The body backplane arrays.
        - "masks": The body backplane masks.
        - "distance": The body backplane distance.
        - "statistics": The body backplane statistics, each stating the unit it
          is in, which is not the unit of the array it was taken from wherever
          the plane is angular.  See
          :mod:`spindoctor.cli.backplanes.statistics`.
    """

    masked_value = float(config.backplanes.masked_value)

    # maps body_name -> arrays/masks/distance
    # per_body is ordered by increasing distance
    result: dict[str, Any] = {}

    # Build inventory and candidate body list
    if snapshot.is_simulated:
        # Guaranteed present for simulated bodies
        inv = snapshot.sim_inventory
    else:
        closest_planet = snapshot.closest_planet
        if closest_planet is None:
            # No planet, no bodies
            return result
        body_list = [closest_planet, *config.satellites(closest_planet)]
        inv = snapshot.inventory(body_list, return_type='full')

    candidate_names = list(inv.keys())
    # Filter and sort by distance
    bodies_by_range = [
        (name, inv[name]) for name in candidate_names if snapshot.inventory_body_in_fov(inv[name])
    ]
    bodies_by_range.sort(key=lambda x: x[1]['range'])

    # Configured backplanes for bodies
    # Each entry consists of "name", "method", and "units"
    # If "method" is not provided, it defaults to the same as "name"
    bodies_cfg = getattr(config.backplanes, 'bodies', None)
    if bodies_cfg is None:
        raise ValueError('Configuration has no bodies section for backplanes')

    # For each body, create a restricted meshgrid and evaluate configured backplanes
    for body_name, inv_info in bodies_by_range:
        u_min = int(inv_info['u_min_unclipped'])
        u_max = int(inv_info['u_max_unclipped'])
        v_min = int(inv_info['v_min_unclipped'])
        v_max = int(inv_info['v_max_unclipped'])
        u0, u1, v0, v1 = snapshot.clip_rect_fov(u_min, u_max, v_min, v_max)
        if u1 < u0 or v1 < v0:
            # Nothing visible
            continue

        # Build restricted meshgrid covering the clipped rectangle (inclusive indices)
        meshgrid = Meshgrid.for_fov(
            snapshot.fov,
            origin=(u0 + 0.5, v0 + 0.5),
            limit=(u1 + 0.5, v1 + 0.5),
            swap=True,
        )
        bp = Backplane(snapshot, meshgrid=meshgrid)

        per_type_arrays: dict[str, np.ndarray] = {}
        per_type_masks: dict[str, np.ndarray] = {}
        body_stats: dict[str, PlaneStatistics] = {}

        for bp_cfg in bodies_cfg:
            bp_name = bp_cfg['name']
            method = bp_cfg.get('method')
            if method is None:
                raise ValueError(f'Body backplane "method" is required: {bp_name}')
            units = bp_cfg.get('units')
            if units is None:
                raise ValueError(f'Body backplane "units" is required: {bp_name}')

            logger.debug(f'{body_name}: Creating body backplane {bp_name}')
            # Evaluate backplane for this body
            if snapshot.is_simulated:
                full, full_mask = _create_simulated_body_backplane(
                    snapshot, body_name, bp_name, v0, v1, u0, u1
                )

            else:
                # This will raise an exception if the backplane is not available
                func = getattr(bp, method)
                vals = func(body_name)
                # Convert to masked array via .mvals to preserve mask
                mvals = vals.mvals  # masked numpy array
                # Embed into full-frame arrays
                full = np.full(snapshot.data.shape, masked_value, dtype=np.float32)
                full_mask = np.zeros(snapshot.data.shape, dtype=bool)
                full[v0 : v1 + 1, u0 : u1 + 1] = np.ma.filled(
                    mvals, fill_value=masked_value
                ).astype(np.float32)
                mask = ~np.ma.getmaskarray(mvals)
                full_mask[v0 : v1 + 1, u0 : u1 + 1] = mask  # True where valid

            per_type_arrays[bp_name] = full
            per_type_masks[bp_name] = full_mask

            # Calculate min/max statistics
            valid_values = full[full_mask]
            if len(valid_values) > 0:
                body_stats[bp_name] = plane_statistics(valid_values, units=units)

        result[body_name] = {
            'arrays': per_type_arrays,
            'masks': per_type_masks,
            'distance': float(inv_info['range']),
            'statistics': body_stats,
        }

    return result
