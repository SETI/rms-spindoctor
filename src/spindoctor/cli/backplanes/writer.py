from collections.abc import Mapping
from typing import Any

import numpy as np
from astropy.io import fits
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.backplanes_bodies import BODY_LONGITUDE, backplane_body_names
from spindoctor.cli.backplanes.backplanes_rings import (
    RING_LONGITUDE,
    RING_LONGITUDINAL_RESOLUTION,
    RingIncidenceAngle,
)
from spindoctor.cli.backplanes.merge import body_naif_id
from spindoctor.cli.backplanes.statistics import PlaneStatistics, longitude_step, plane_statistics
from spindoctor.config import IMAGE_LOGGER, Config
from spindoctor.obs import ObsSnapshot
from spindoctor.support.file import json_as_string
from spindoctor.support.types import NDArrayBoolType

BODY_ID_MAP_HDU_NAME = 'BODY_ID_MAP'
"""The name of the HDU holding the body identity map, in a FITS that holds one.

Each of its 32-bit integers is the NAIF ID of the body that claimed the pixel, and 0
where no body did.
"""


def _plane_statistics(
    planes: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    pixels: NDArrayBoolType,
    *,
    masked_value: float,
) -> dict[str, PlaneStatistics]:
    """Return each plane's statistic over some pixels, from the planes the FITS holds.

    A pixel counts for a plane where the plane has a value there, anything but the masked
    value, so that a statistic summarizes exactly what a reader of the FITS finds at those
    pixels.

    Parameters:
        planes: The planes the merge left, keyed by name, as the FITS holds them.
        units: The unit the configuration gives each plane to be summarized, keyed by
            name, in the order the statistics are recorded in.
        pixels: Where the statistics are taken.
        masked_value: The value a plane holds where it has none.

    Returns:
        The statistic of each of those planes that has a value at one of the pixels at
        least, as :func:`~spindoctor.cli.backplanes.statistics.plane_statistics` takes it.
    """
    statistics: dict[str, PlaneStatistics] = {}
    for name, unit in units.items():
        if name not in planes:
            continue
        plane = planes[name]
        values = plane[pixels & (plane != masked_value)]
        if values.size > 0:
            statistics[name] = plane_statistics(values, units=unit)
    return statistics


def _incidence_angle(
    center: RingIncidenceAngle,
    pixel_incidence: np.ndarray,
    ring_pixels: NDArrayBoolType,
    *,
    masked_value: float,
) -> RingIncidenceAngle:
    """Return the incidence angle the rings block records, at the center and over the rings.

    Parameters:
        center: The incidence angle at the ring system's center, as the ring stage took it.
        pixel_incidence: The incidence angle at each pixel, in radians, the masked value
            where the pixel is not on the rings.
        ring_pixels: The pixels where a ring plane the FITS holds has a value.
        masked_value: The value an array holds where it has none.

    Returns:
        The angle at the center and, when a ring pixel has an incidence angle, the least,
        the greatest and the mean over those pixels, in degrees.
    """
    values = pixel_incidence[ring_pixels & (pixel_incidence != masked_value)]
    if values.size == 0:
        return center
    degrees = np.degrees(values.astype(np.float64))
    return RingIncidenceAngle(
        value=center['value'],
        min=float(degrees.min()),
        max=float(degrees.max()),
        mean=float(degrees.mean()),
        units=center['units'],
    )


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

    Every statistic the metadata records is taken from the planes the FITS holds, after
    the merge, so that it summarizes exactly the pixels where the product's own plane has
    a value: a body's over the pixels the body identity map gives the body, and the
    rings' over every pixel.  A pixel of the rings or of a body that a nearer body covers
    holds the nearer body's value, so it counts for the nearer body alone.  The ring
    longitude's statistic also records its range wrapped at zero, over the same pixels,
    with the coarsest longitudinal size of a pixel on the rings as the widest gap that
    leaves the circle covered.  A body's longitude statistic records its range wrapped at
    zero too, over the body's pixels, with the widest longitude step between two of them
    that neighbor as that gap, as
    :func:`~spindoctor.cli.backplanes.statistics.longitude_step` finds it.  Beside the
    incidence angle at the ring system's center,
    the rings block records the least, the greatest and the mean incidence angle over
    the pixels where a ring plane has a value.

    Parameters:
        fits_file_path: The FITS file path.
        snapshot: The observation snapshot.
        master_by_type: The master by type.
        body_id_map: The body id map.
        config: The configuration.
        bodies_result: Result from create_body_backplanes, each of whose bodies the
            metadata records, with its statistics and its inventory information.
        rings_result: Result from create_ring_backplanes, whose ring target and incidence
            angles the metadata's ``rings`` block records as ``target`` and
            ``incidence_angle``, beside the ring statistics as ``backplanes``.  None
            leaves the block empty.
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
            body_list = backplane_body_names(closest_planet, config)
            inv = snapshot.inventory(body_list, return_type='full')
        else:
            inv = {}

    # The planes each statistic is taken from, in the configuration's order
    body_units = {bp['name']: bp['units'] for bp in getattr(config.backplanes, 'bodies', [])}
    ring_units = {
        rp['name']: rp['units']
        for rp in getattr(config.backplanes, 'rings', [])
        if rp['name'] != 'distance'
    }

    # Each body's statistics and inventory information
    for body_name in bodies_result:
        # The pixels the merge gave the body, where the planes hold its values
        claimed = body_id_map == body_naif_id(snapshot, body_name)
        body_statistics = _plane_statistics(
            master_by_type, body_units, claimed, masked_value=masked_value
        )
        # The body longitude's range wrapped at zero as well, the arc the body's pixels
        # cover, which a gap no wider than the widest longitude step between two of its
        # neighboring pixels does not break: such a gap is one the body's sampling
        # leaves, not longitude out of view
        if BODY_LONGITUDE in body_statistics:
            longitude = master_by_type[BODY_LONGITUDE]
            measured = claimed & (longitude != masked_value)
            units = body_units[BODY_LONGITUDE]
            body_statistics[BODY_LONGITUDE] = plane_statistics(
                longitude[measured],
                units=units,
                longitude_resolution=longitude_step(longitude, measured, units=units),
            )
        body_entry: dict[str, Any] = {'backplanes': body_statistics}

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

        backplane_metadata['bodies'][body_name] = body_entry

    # The ring target the ring backplanes were computed for, the incidence angle of
    # sunlight on its plane, and each ring plane's statistics, over every pixel: where a
    # nearer body covers the rings, the ring planes have no value
    if rings_result is not None:
        everywhere = np.ones(body_id_map.shape, dtype=np.bool_)
        ring_statistics = _plane_statistics(
            master_by_type, ring_units, everywhere, masked_value=masked_value
        )
        # The ring longitude's range wrapped at zero as well, the arc its pixels cover,
        # which a gap no wider than the coarsest pixel does not break.  With no
        # longitudinal resolution to measure a gap against -- a configuration declaring
        # none, or a frame where that plane has no value -- no wrapped range is recorded,
        # and the rest of the document is written as ever
        if RING_LONGITUDE in ring_statistics and RING_LONGITUDINAL_RESOLUTION in ring_statistics:
            longitude = master_by_type[RING_LONGITUDE]
            ring_statistics[RING_LONGITUDE] = plane_statistics(
                longitude[longitude != masked_value],
                units=ring_units[RING_LONGITUDE],
                longitude_resolution=ring_statistics[RING_LONGITUDINAL_RESOLUTION]['max'],
            )
        # The pixels where a ring plane the FITS holds has a value, over which the
        # incidence angle's range is taken
        ring_pixels = np.zeros(body_id_map.shape, dtype=np.bool_)
        for name in ring_statistics:
            ring_pixels |= master_by_type[name] != masked_value
        backplane_metadata['rings'] = {
            'target': rings_result['target_key'],
            'incidence_angle': _incidence_angle(
                rings_result['incidence_angle'],
                rings_result['pixel_incidence'],
                ring_pixels,
                masked_value=masked_value,
            ),
            'backplanes': ring_statistics,
        }

    metadata_file_path.write_text(json_as_string(backplane_metadata))
    logger.debug('Wrote backplane metadata: %s', metadata_file_path)
