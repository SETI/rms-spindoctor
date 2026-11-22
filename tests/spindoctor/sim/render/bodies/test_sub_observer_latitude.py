"""A body seen at a sub-observer latitude, drawn by the image-side renderer.

The latitude turns the ellipsoid as a solid body, so its surface texture turns with
it: the disc center shows the latitude below the observer.  A turned sphere looks
the same, haze included, since turning a sphere moves none of its surface on the
image.  The body's inventory entry reports the extent of the outline it draws.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body

from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.support.types import NDArrayFloatType

_SIZE = 101
# The body center sits on the middle of pixel (50, 50), so pixel centers lie whole
# pixels from it.
_CENTER = 50.5
_RADIUS = 40.0
_BANDS = {'band_amplitude': 0.4, 'band_wavenumber': 5.0}


def _render(**keys: Any) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render one sphere lit from the observer, without supersampling.

    Parameters:
        **keys: Further body keys.

    Returns:
        The image and the body info.
    """
    body = disc_body(
        _CENTER,
        _CENTER,
        _RADIUS,
        illumination_angle=0.0,
        phase_angle=0.0,
        anti_aliasing=0.0,
        **keys,
    )
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _mask, info = render_single_body(img, body_spec_from_params(body, index=0), 0.0, offset_u=0.0)
    return img, info


def _texture_factor(**keys: Any) -> NDArrayFloatType:
    """The banded texture's factor on the sphere: the banded render over the plain one.

    Parameters:
        **keys: Further body keys.

    Returns:
        The factor per pixel, 0 off the body.
    """
    banded, _ = _render(disc_texture=dict(_BANDS), **keys)
    plain, _ = _render(**keys)
    return np.divide(banded, plain, out=np.zeros_like(banded), where=plain > 0.0)


def test_the_disc_center_shows_the_band_at_the_sub_observer_latitude() -> None:
    """At 30 degrees the disc center shows the bands found 30 degrees north equator-on.

    Seen equator-on, the point 30 degrees north on the central meridian lies
    sin(30 deg) = 0.5 of a radius, 20 px, from the center toward +v.
    """
    turned = _texture_factor(sub_observer_lat_deg=30.0)
    equator_on = _texture_factor()
    assert float(turned[50, 50]) == pytest.approx(float(equator_on[70, 50]), abs=1e-12)


def test_the_bands_differ_between_the_disc_center_and_30_degrees_north() -> None:
    """The banded texture is not flat between the two latitudes the test above compares."""
    equator_on = _texture_factor()
    assert abs(float(equator_on[50, 50]) - float(equator_on[70, 50])) > 0.05


def test_the_inventory_reports_the_foreshortened_outline() -> None:
    """A body flattened along its pole and turned 60 degrees spans the rows it draws.

    The pole is 10 px and the equator 20 px, so the outline reaches
    sqrt((10 cos 60)**2 + (20 sin 60)**2) = 18.03 px along the pole.
    """
    img, info = _render(semi_a_px=10.0, sub_observer_lat_deg=60.0)
    drawn_rows = int(np.count_nonzero(np.any(img > 0.0, axis=1)))
    assert abs(info['inventory']['v_pixel_size'] - drawn_rows) <= 1.0


def test_a_turned_hazy_sphere_renders_as_it_does_equator_on() -> None:
    """The haze turns with the body, so a hazy sphere's image and halo do not change."""
    haze = {'scale_height_px': 4.0, 'tau_ref': 2.0, 'g': 0.5}
    turned, _ = _render(atmosphere=dict(haze), sub_observer_lat_deg=40.0)
    equator_on, _ = _render(atmosphere=dict(haze))
    np.testing.assert_allclose(turned, equator_on, rtol=0.0, atol=1e-12)
