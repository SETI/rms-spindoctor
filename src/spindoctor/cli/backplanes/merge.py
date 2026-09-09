import hashlib
from typing import Any

import cspyce
import numpy as np

from spindoctor.obs import ObsSnapshot


def fake_naif_id(body_name: str) -> int:
    """Return a deterministic fake NAIF ID for an unknown simulated body.

    The ID is derived from a hashlib digest, not ``hash()``, because str
    hashing is salted per interpreter run and the ID must be reproducible
    across processes.

    Parameters:
        body_name: The (unknown) body name to synthesize an ID for.

    Returns:
        An int in the range [10000, 30000), stable across processes.
    """
    digest = hashlib.sha256(str(body_name).encode('utf-8')).digest()
    return 10000 + (int.from_bytes(digest[:8], 'big') % 20000)


def merge_sources_into_master(
    snapshot: ObsSnapshot,
    *,
    bodies_result: dict[str, Any],
    rings_result: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Merge bodies and rings per-pixel based on nearest distance.

    Parameters:
        snapshot: The observation snapshot.
        bodies_result: The bodies result from create_body_backplanes (dict keyed by body name).
        rings_result: The rings result from create_ring_backplanes (dict keyed by metadata key).

    Returns:
        A tuple containing the master arrays by backplane type and the body id map.
    """

    height, width = snapshot.data.shape
    masked_value = float(snapshot.config.backplanes.masked_value)
    master_by_type: dict[str, np.ndarray] = {}
    # Not masked_value: this is the body identity map, and 0 is not a NAIF ID, so
    # zero already says no body claimed the pixel.
    body_id_map = np.zeros((height, width), dtype=np.int32)

    # Build source list with per-pixel distance and NAIF IDs
    body_sources: list[dict[str, Any]] = []

    # Bodies: scalar distances per body, broadcast within mask; inf elsewhere
    for body_name, entry in bodies_result.items():
        distance_scalar = float(entry['distance'])
        # Create per-pixel distance with inf default and scalar where within any mask
        any_mask: np.ndarray | None = None
        for m in entry['masks'].values():
            if any_mask is None:
                any_mask = m
            else:
                if not np.all(any_mask == m):
                    raise ValueError(f'Masks for body {body_name} are not all the same')
        if any_mask is None:
            any_mask = np.zeros((height, width), dtype=bool)
        distance = np.where(any_mask, distance_scalar, np.inf).astype(np.float32)
        try:
            naif_id = int(cspyce.bodn2c(body_name))
        except Exception:
            if snapshot.is_simulated:
                # Unknown body name; create a deterministic fake NAIF ID in int32 range
                naif_id = fake_naif_id(body_name)
            else:
                raise  # This is a real problem
        body_sources.append(
            {
                'name': body_name,
                'naif_id': naif_id,
                'distance': distance,
                'mask': any_mask,
                'arrays': entry['arrays'],
                'masks': entry['masks'],
            }
        )

    if not body_sources and not rings_result:
        return master_by_type, body_id_map

    body_presence = np.zeros((height, width), dtype=bool)

    if body_sources:
        # Compute nearest body index and nearest body distance (ignoring rings)
        body_dist_stack = np.stack([x['distance'] for x in body_sources], axis=0)
        nearest_body_idx = np.argmin(body_dist_stack, axis=0)
        nearest_body_distance = np.take_along_axis(
            body_dist_stack, nearest_body_idx[None, ...], axis=0
        )[0]

        # Build a union mask indicating if any body has presence at each pixel (any body mask True)
        for body_source in body_sources:
            body_presence |= body_source['mask']

        # Identify available backplane types
        body_types: set[str] = set()
        for body_source in body_sources:
            body_types.update(body_source['arrays'].keys())

        # 1) Body backplanes: use nearest body among bodies only; rings do not affect these.
        # Unclaimed pixels carry the masked value (the writer omits all-masked planes).
        for bp_type in sorted(body_types):
            master = np.full((height, width), masked_value, dtype=np.float32)
            for body_idx, body_source in enumerate(body_sources):
                arrays = body_source['arrays']
                masks = body_source['masks']
                if bp_type not in arrays or bp_type not in masks:
                    raise ValueError(
                        f'Backplane type {bp_type} array or mask not found for body '
                        f'{body_source["name"]}'
                    )
                src_vals = arrays[bp_type]
                src_mask = masks[bp_type]
                take = (nearest_body_idx == body_idx) & src_mask
                master[take] = src_vals[take]
            master_by_type[bp_type] = master

        for body_idx, body_source in enumerate(body_sources):
            mask = body_source['mask']
            take = (nearest_body_idx == body_idx) & mask
            body_id_map[take] = np.int32(body_source['naif_id'])

    else:
        nearest_body_distance = np.full((height, width), np.inf, dtype=np.float32)

    # 2) Ring backplanes: rings fill where valid and not occluded by a closer body
    if rings_result:
        ring_arrays: dict[str, Any] = rings_result['arrays']
        ring_masks: dict[str, Any] = rings_result['masks']
        ring_distance: np.ndarray = rings_result['distance']

        for bp_type in sorted(ring_arrays.keys()):
            if bp_type == 'distance':
                # Merge-ordering input only; never a product backplane HDU
                continue
            if bp_type not in ring_arrays or bp_type not in ring_masks:
                raise ValueError(f'Backplane type {bp_type} array or mask not found for rings')
            master = np.full((height, width), masked_value, dtype=np.float32)
            src_vals = ring_arrays[bp_type]
            src_mask = ring_masks[bp_type]
            # occlusion: if any body is present and nearer than ring, mask out ring
            occluded = body_presence & (nearest_body_distance < ring_distance)
            valid = src_mask & (~occluded)
            master[valid] = src_vals[valid]
            if bp_type in master_by_type:
                raise ValueError(
                    f'Ring backplane type {bp_type} already exists in master_by_type '
                    'because it is also a body backplane type'
                )
            master_by_type[bp_type] = master

    return master_by_type, body_id_map
