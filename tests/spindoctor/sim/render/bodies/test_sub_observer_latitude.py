"""A body seen at a sub-observer latitude, drawn by the image-side renderer.

The latitude turns the ellipsoid as a solid body, so its surface texture turns with
it: the disc center shows the latitude below the observer.  A turned sphere looks
the same, haze included, since turning a sphere moves none of its surface on the
image.  The body's inventory entry reports the extent of the outline it draws.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body

from spindoctor.nav_model.simulated.predicted_body import render_predicted_body
from spindoctor.sim.render.bodies import atmosphere
from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidGeometry,
    ellipsoid_image_normals,
    project_ellipsoid,
)
from spindoctor.sim.shared.lighting import illumination_vector
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
    """The banded texture differs between the two latitudes the test above compares."""
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


def test_the_image_and_the_navigators_prediction_share_the_turned_outline() -> None:
    """A turned triaxial body covers the same pixels in the image and the prediction."""
    body = disc_body(
        _CENTER,
        _CENTER,
        _RADIUS,
        illumination_angle=0.0,
        phase_angle=0.0,
        semi_a_px=25.0,
        semi_c_px=45.0,
        rotation_z=30.0,
        sub_observer_lat_deg=40.0,
    )
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    image_mask, _info = render_single_body(
        img, body_spec_from_params(body, index=0), 0.0, offset_u=0.0
    )
    predicted = render_predicted_body(
        body, 0.0, shape_vu=(_SIZE, _SIZE), center_vu=(_CENTER, _CENTER)
    )
    np.testing.assert_array_equal(image_mask, predicted > 0.0)


def test_the_haze_column_at_the_north_limb_lies_below_the_pole() -> None:
    """On a sphere turned 30 degrees, the limb's northernmost column is at latitude 60.

    The north pole tips toward the observer, so it leaves the limb, and the limb
    point on the pole's side lies 30 degrees short of it.
    """
    tilt = math.radians(30.0)
    geometry = EllipsoidGeometry(
        semi_a=20.0, semi_b=20.0, semi_c=20.0, cos_lat=math.cos(tilt), sin_lat=math.sin(tilt)
    )
    sin_latitude = atmosphere._column_latitude_sine(
        np.array([24.0]), np.array([0.0]), np.array([(24.0 / 20.0) ** 2]), geometry=geometry
    )
    assert float(sin_latitude[0]) == pytest.approx(math.cos(tilt), abs=1e-12)


def test_the_on_disc_haze_reads_the_turned_surfaces_incidence() -> None:
    """The haze's surface incidence on a turned oblate body matches the disc's normals."""
    projection = project_ellipsoid(
        (80, 80), (40.0, 40.0), 18.0, semi_b_px=30.0, semi_c_px=30.0, sub_observer_lat=0.8
    )
    on_disc = projection.ellipse_dist_sq < 0.9
    light_v, light_u, light_z = illumination_vector(illumination_angle=0.4, phase_angle=0.7)
    incidence = atmosphere._disc_incidence(
        projection.v_rot[on_disc],
        projection.u_rot[on_disc],
        projection.ellipse_dist_sq[on_disc],
        geometry=projection.geometry,
        cos_rz=projection.cos_rz,
        sin_rz=projection.sin_rz,
        illum_v=light_v,
        illum_u=light_u,
        illum_z=light_z,
    )
    normal_v, normal_u, normal_z = ellipsoid_image_normals(projection)
    expected = (normal_v * light_v + normal_u * light_u + normal_z * light_z)[on_disc]
    np.testing.assert_allclose(incidence, expected, rtol=0.0, atol=1e-12)


def test_faint_relief_on_a_turned_body_shades_like_the_smooth_body() -> None:
    """Relief a millionth of the radius high changes no pixel of a turned body by 1e-4.

    The relief's normals and its sun are turned with the body, so they match the
    smooth body's to within the relief's own slope.
    """
    keys = {
        'semi_a_px': 25.0,
        'semi_c_px': 45.0,
        'sub_observer_lat_deg': 35.0,
        'phase_angle': 50.0,
        'illumination_angle': 120.0,
        'anti_aliasing': 1.0,
    }
    smooth = _render_lit(**keys)
    faint = _render_lit(limb_relief_rms=1e-6, limb_relief_corr_deg=15.0, **keys)
    np.testing.assert_allclose(faint, smooth, rtol=0.0, atol=1e-4)


def _render_lit(**keys: Any) -> NDArrayFloatType:
    """Render one body at the frame center, lit as the keys set.

    Parameters:
        **keys: The body keys beyond its center and radius.

    Returns:
        The image.
    """
    body = {**disc_body(_CENTER, _CENTER, _RADIUS), 'seed': 5, **keys}
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    render_single_body(img, body_spec_from_params(body, index=0), 0.0, offset_u=0.0)
    return img
