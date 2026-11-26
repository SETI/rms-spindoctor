"""Facts about bodies that the photometric-law study reads, whichever camera saw them.

A photometric law describes how one surface brightens and darkens with the angles of
the light and the view, so the study (:mod:`tests.integration.sim_realism_laws`) fits a
law to pixels of one surface only.  Two kinds of body fact decide which pixels those are,
and neither depends on the mission that took the frame, so they live here.

**Two-tone moons.**  Iapetus's leading side is covered by a dark terrain, Cassini Regio,
centered near the apex of its orbital motion at 0 deg latitude and 90 deg west longitude;
its trailing side and its poles are about ten times brighter (Porco et al. 2005, Science
307, 1237-1242).  The dark terrain does not fill the leading hemisphere.  Dividing each
pixel's I/F on the cohort's two Iapetus frames by the lunar-Lambert law's value there
maps the albedo: the dark terrain ends about 50 deg from the apex toward the north pole,
but along the equator it reaches about 120 deg from the apex, within 60 deg of the
antapex.  A hemisphere cut at a meridian therefore mixes the two terrains: on one frame
40% of the leading hemisphere's pixels lay above 45 deg latitude and read 3 to 5 times
brighter than the rest.

So each terrain is a cap on the sphere, :class:`TerrainCap`, drawn well inside it: 40 deg
of arc about the apex for the dark terrain, and 40 deg about the antapex (0 deg latitude,
270 deg west) for the bright one.  Each of the two frames shows one cap.  The dark cap's
2,424 counted pixels read an albedo from 0.018 to 0.026 (10th to 90th percentile), and
the bright cap's 8,137 pixels read 0.15 to 0.23.  The rest of the moon is left out.

**Rings that can hide a moon.**  A moon seen through a planet's main rings is dimmed by
them, but the frame's geometry traces the moon alone.  :data:`OCCULTING_RINGS` gives the
ring body whose pixels in front of a moon the study leaves out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'OCCULTING_RINGS',
    'TWO_TONE_TERRAINS',
    'TerrainCap',
    'unit_vectors',
]


def unit_vectors(lat_deg: NDArrayFloatType, west_lon_deg: NDArrayFloatType) -> NDArrayFloatType:
    """Unit vectors in a body's frame from planetocentric latitude and west longitude.

    Parameters:
        lat_deg: Latitudes in degrees, any shape.
        west_lon_deg: West longitudes in degrees, the same shape.

    Returns:
        An array of the inputs' shape with a last axis of 3.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(west_lon_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


@dataclass(frozen=True)
class TerrainCap:
    """The part of a moon's surface taken as one terrain: a cap about a point.

    Parameters:
        name: The terrain's name in the study's records, such as ``'dark terrain'``.
        center_lat_deg: The cap's center, planetocentric latitude in degrees.
        center_west_lon_deg: Its west longitude in degrees, IAU reference.
        radius_deg: The cap's angular radius in degrees.
    """

    name: str
    center_lat_deg: float
    center_west_lon_deg: float
    radius_deg: float

    def contains(
        self, lat_deg: NDArrayFloatType, west_lon_deg: NDArrayFloatType
    ) -> NDArrayBoolType:
        """Which surface points lie inside the cap; NaN coordinates lie outside.

        Parameters:
            lat_deg: Planetocentric latitudes in degrees, any shape.
            west_lon_deg: West longitudes in degrees, the same shape.

        Returns:
            A mask of the inputs' shape.
        """
        center = unit_vectors(np.asarray(self.center_lat_deg), np.asarray(self.center_west_lon_deg))
        points = unit_vectors(lat_deg, west_lon_deg)
        cosine = np.sum(points * center, axis=-1)
        inside = cosine >= math.cos(math.radians(self.radius_deg))
        return np.asarray(inside & np.isfinite(cosine), dtype=bool)


TWO_TONE_TERRAINS: dict[str, tuple[TerrainCap, ...]] = {
    'IAPETUS': (
        TerrainCap('dark terrain', 0.0, 90.0, 40.0),
        TerrainCap('bright terrain', 0.0, 270.0, 40.0),
    ),
}
"""For each moon whose surface holds two terrains, the cap the study measures of each."""

OCCULTING_RINGS: dict[str, str] = {'SATURN': 'SATURN_MAIN_RINGS'}
"""For each planet, the ring body (as oops knows it) whose pixels hide a moon behind them."""
