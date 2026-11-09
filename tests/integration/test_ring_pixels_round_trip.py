"""Ring longitude and radius map back to the pixels they came from, on a real frame.

``RingMosaic.longitude_radius_to_pixels`` traces each ring coordinate to the camera
through ``oops``'s ring surface. Its inverse is the ``oops`` backplane, which gives the
ring radius and longitude seen at each pixel. Feeding a pixel's backplane values back
through the conversion has to land on that pixel's center.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_RESOURCES = os.environ.get('OOPS_RESOURCES', '')

if len(_RESOURCES) == 0 or 'PDS3_HOLDINGS_DIR' not in os.environ:
    pytest.skip(
        'the ring round trip needs SPICE and the holdings; set OOPS_RESOURCES and '
        'PDS3_HOLDINGS_DIR',
        allow_module_level=True,
    )

from oops.backplane import Backplane  # noqa: E402  (guarded import)

from spindoctor.obs.obs_inst_cassini_iss import ObsCassiniISS  # noqa: E402  (guarded import)
from spindoctor.reproj.rings import RingMosaic  # noqa: E402  (guarded import)

# A NAC frame whose field holds the main rings over a wide span of radius.
_FRAME = 'calibrated/COISS_2xxx/COISS_2010/data/1489518473_1490053457/N1489603610_2_CALIB.IMG'

# The main rings; outside them the plane is seen nearly edge-on or from millions of km,
# where the backplane itself is not a fine-grained reference.
_RADIUS_RANGE_KM = (74000.0, 140000.0)

# Measured round-trip error is at most 1.3e-4 px on this frame.
_TOLERANCE_PX = 1e-3


@pytest.fixture(scope='module')
def round_trip_error() -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (du, dv) between six main-ring pixels and their round trip."""
    holdings = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
    obs = ObsCassiniISS.from_file(f'{holdings}/{_FRAME}')
    backplane = Backplane(obs)
    radius = backplane.ring_radius('saturn:ring')
    longitude = backplane.ring_longitude('saturn:ring')
    radius_vals = np.asarray(radius.vals)
    longitude_vals = np.asarray(longitude.vals)
    masked = np.broadcast_to(np.asarray(radius.mask), radius_vals.shape)
    on_rings = ~masked & (radius_vals > _RADIUS_RANGE_KM[0]) & (radius_vals < _RADIUS_RANGE_KM[1])
    v_all, u_all = np.nonzero(on_rings)
    pick = np.linspace(0, len(v_all) - 1, 6).astype(int)
    v_px, u_px = v_all[pick], u_all[pick]

    u, v = RingMosaic.longitude_radius_to_pixels(
        obs, longitude_vals[v_px, u_px], radius_vals[v_px, u_px]
    )
    return u - (u_px + 0.5), v - (v_px + 0.5)


def test_ring_coordinates_return_to_their_pixel_columns(
    round_trip_error: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every sampled pixel comes back to its own column."""
    assert np.max(np.abs(round_trip_error[0])) < _TOLERANCE_PX


def test_ring_coordinates_return_to_their_pixel_rows(
    round_trip_error: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every sampled pixel comes back to its own row."""
    assert np.max(np.abs(round_trip_error[1])) < _TOLERANCE_PX
