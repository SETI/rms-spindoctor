from typing import Any, TypedDict

import numpy as np
from pdslogger import PdsLogger

from spindoctor.cli.backplanes.statistics import DEGREES
from spindoctor.config import Config
from spindoctor.obs import ObsSnapshot

RING_LONGITUDE = 'ring_longitude'
"""The configured ring plane holding the ring longitude.

Its statistic also records its range wrapped at zero, the arc of longitude the image's
ring pixels cover.
"""

RING_LONGITUDINAL_RESOLUTION = 'ring_longitudinal_resolution'
"""The configured ring plane holding the longitudinal size of a pixel on the rings.

Its coarsest value over an image is the widest gap between the image's ring longitudes
that still leaves the circle covered.
"""


class RingIncidenceAngle(TypedDict):
    """The incidence angle of sunlight on the ring plane, as the backplane metadata records it.

    Attributes:
        value: The angle between the direction the sunlight arrives from and the normal to
            the ring plane on its sunlit side, from 0 to 90 degrees.
        units: The unit ``value`` is in, ``deg``.
    """

    value: float
    units: str


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
        - "target_key": The ring target the backplanes are computed for, as
          :func:`ring_target` names it.
        - "incidence_angle": The incidence angle of sunlight on the ring target's
          plane, as a :class:`RingIncidenceAngle` in degrees.  It is one angle over
          the whole image, so no backplane holds it: oops's
          ``ring_center_incidence_angle`` evaluates it once, at the ring system's
          center, for the light that reaches the camera at the observation's
          midtime, measured from the normal on the plane's sunlit side.  It is
          recorded whether or not any pixel of the image is on the rings.
        - "arrays": The ring backplane arrays.
        - "masks": The ring backplane masks.
        - "distance": The ring backplane distance.

        No statistics: the writer takes them once the merge has removed the ring
        pixels a nearer body covers.
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

    # Sunlight falls on the ring plane at one angle over the whole image, so no backplane
    # holds it (#47): it is taken once, at the ring system's center, for the light that
    # reaches the camera at the observation's midtime.
    center_incidence = bp.ring_center_incidence_angle(target_key)
    result['incidence_angle'] = RingIncidenceAngle(
        value=float(np.degrees(center_incidence.vals)), units=DEGREES
    )

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

    # Ensure distance is present
    if result['distance'] is None:
        # If not explicitly configured, compute per-pixel distance for ordering
        vals = bp.distance(target_key, direction='dep')
        result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

    return result
