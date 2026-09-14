from typing import Any

import numpy as np
from astropy.io import fits
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.config import IMAGE_LOGGER, Config
from spindoctor.obs import ObsSnapshot
from spindoctor.support.file import json_as_string

BODY_ID_MAP_HDU_NAME = 'BODY_ID_MAP'
"""The name of the HDU holding the body identity map, in a FITS that holds one.

Each of its 32-bit integers is the NAIF ID of the body that claimed the pixel, and 0
where no body did.
"""


def write_fits(
    *,
    fits_file_path: FCPath,
    snapshot: ObsSnapshot,
    master_by_type: dict[str, np.ndarray],
    body_id_map: np.ndarray,
    config: Config,
    bodies_result: dict[str, Any],
    rings_result: dict[str, Any] | None = None,
    logger: PdsLogger = IMAGE_LOGGER,
) -> None:
    """Write FITS file and backplane metadata JSON using FCPath.

    Parameters:
        fits_file_path: The FITS file path.
        snapshot: The observation snapshot.
        master_by_type: The master by type.
        body_id_map: The body id map.
        config: The configuration.
        bodies_result: Result from create_body_backplanes containing statistics.
        rings_result: Result from create_ring_backplanes containing statistics.
        logger: Logger for diagnostic messages.
    """

    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = []
    primary = fits.PrimaryHDU()
    hdus.append(primary)

    # BODY_ID_MAP first (after Primary) - only include if not empty
    has_body_id_map = np.any(body_id_map != 0)
    if has_body_id_map:
        id_hdu = fits.ImageHDU(data=body_id_map.astype('int32'), name=BODY_ID_MAP_HDU_NAME)
        hdus.append(id_hdu)

    # Backplane arrays
    units_map: dict[str, str] = {}
    # Try to pull declared units from config
    for bp in getattr(config.backplanes, 'bodies', []):
        units_map[bp['name']] = bp.get('units', '')
    for rp in getattr(config.backplanes, 'rings', []):
        if rp['name'] != 'distance':  # not written as a master backplane type
            units_map[rp['name']] = rp.get('units', '')

    # Filter out backplanes with no valid pixels (unclaimed pixels carry the
    # configured masked value; non-finite values never count as valid)
    masked_value = float(config.backplanes.masked_value)
    filtered_master = {
        k: v for k, v in master_by_type.items() if np.any((v != masked_value) & np.isfinite(v))
    }

    for name, arr in filtered_master.items():
        hdu = fits.ImageHDU(data=arr.astype('float32'), name=name.upper())
        if units_map.get(name):
            hdu.header['BUNIT'] = units_map[name]
        hdus.append(hdu)

    hdul = fits.HDUList(hdus)
    local_path = fits_file_path.get_local_path()
    hdul.writeto(local_path, overwrite=True)
    fits_file_path.upload()

    # Write backplane metadata JSON file
    metadata_file_path = fits_file_path.parent / (
        fits_file_path.stem.replace('_backplanes', '') + '_backplane_metadata.json'
    )
    backplane_metadata: dict[str, Any] = {
        'bodies': {},
        'rings': {},
    }

    # Get inventory information for all bodies
    # TODO Clean this up
    if snapshot.is_simulated:
        inv = snapshot.sim_inventory
    else:
        closest_planet = snapshot.closest_planet
        if closest_planet:
            body_list = [closest_planet, *config.satellites(closest_planet)]
            inv = snapshot.inventory(body_list, return_type='full')
        else:
            inv = {}

    # Extract body statistics and inventory information per body
    for body_name, body_data in bodies_result.items():
        body_entry: dict[str, Any] = {}
        if 'statistics' in body_data:
            body_entry['backplanes'] = body_data['statistics']

        # Add inventory information for this body
        if body_name in inv:
            inv_data = inv[body_name]
            # center_uv is [u, v] but we need [v, u]
            center_uv = inv_data.get('center_uv', None)
            if center_uv is not None:
                body_entry['center_uv'] = [float(center_uv[1]), float(center_uv[0])]
            # center_range from range
            center_range = inv_data.get('range', None)
            if center_range is not None:
                body_entry['center_range'] = float(center_range)
            # size_uv from u_pixel_size and v_pixel_size
            u_pixel_size = inv_data.get('u_pixel_size', None)
            v_pixel_size = inv_data.get('v_pixel_size', None)
            if u_pixel_size is not None and v_pixel_size is not None:
                body_entry['size_uv'] = [float(u_pixel_size), float(v_pixel_size)]

        if body_entry:
            backplane_metadata['bodies'][body_name] = body_entry

    # Extract ring statistics
    if rings_result and 'statistics' in rings_result:
        backplane_metadata['rings'] = {'backplanes': rings_result['statistics']}

    metadata_file_path.write_text(json_as_string(backplane_metadata))
    logger.debug('Wrote backplane metadata: %s', metadata_file_path)
