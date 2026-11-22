"""The shared ellipsoid projection, its normals, and the smooth Lambert render.

The projection measures each pixel from its center and rotates it as a hand
computation does.  A sub-observer latitude turns the ellipsoid as a solid body: a
sphere keeps its outline, an oblate body's outline is the projection of the turned
body, and the disc center sees the sub-observer latitude.  A cropped projection keeps
each pixel's values and its position on the whole grid.  The normals face the way the
outline does, are perpendicular to the turned surface, lie in the image plane outside
the limb, and have unit length.
The Lambert render shades each pixel by its cosine of incidence, leaves the sky
dark, and averages its supersampled grid down to the caller's.
"""

import math

import numpy as np
import pytest

from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    ellipsoid_image_normals,
    project_ellipsoid,
    render_lambert_ellipsoid,
    surface_unit_coords,
)

# A hand-computed pixel: row 23, column 15 of a body centered at (20.0, 18.0), whose
# pixel center lies 3.5 px below and 2.5 px left of the body center.
_ROW, _COL = 23, 15
_CENTER = (20.0, 18.0)
_V, _U = 3.5, -2.5
_ROTATION_Z = 0.5
_TILT = 0.6
_SEMI_A, _SEMI_B = 12.0, 7.0


def _hand_body(*, sub_observer_lat: float) -> EllipsoidProjection:
    """The body the hand-computed pixel belongs to, at a chosen tilt."""
    return project_ellipsoid(
        (48, 40),
        _CENTER,
        _SEMI_A,
        semi_b_px=_SEMI_B,
        semi_c_px=9.0,
        rotation_z=_ROTATION_Z,
        sub_observer_lat=sub_observer_lat,
    )


def _tilted_body(*, anti_aliasing: float = 0.0) -> EllipsoidProjection:
    """A rotated, tilted triaxial body on a 48 x 40 frame."""
    return project_ellipsoid(
        (48, 40),
        (22.3, 19.6),
        14.0,
        semi_b_px=10.5,
        semi_c_px=12.0,
        rotation_z=0.6,
        sub_observer_lat=0.4,
        anti_aliasing=anti_aliasing,
    )


def test_projection_measures_from_pixel_centers() -> None:
    """A pixel's coordinates run from the body center to the pixel's center."""
    projection = _hand_body(sub_observer_lat=_TILT)
    assert projection.v_coords[_ROW, _COL] == pytest.approx(_V, abs=1e-12)
    assert projection.u_coords[_ROW, _COL] == pytest.approx(_U, abs=1e-12)


def test_projection_rotates_a_pixel() -> None:
    """The rotated frame turns by rotation_z and not by the sub-observer latitude."""
    projection = _hand_body(sub_observer_lat=_TILT)
    cos_rz, sin_rz = math.cos(_ROTATION_Z), math.sin(_ROTATION_Z)
    expected_v_rot = _V * cos_rz - _U * sin_rz
    expected_u_rot = _V * sin_rz + _U * cos_rz
    assert projection.v_rot[_ROW, _COL] == pytest.approx(expected_v_rot, abs=1e-12)
    assert projection.u_rot[_ROW, _COL] == pytest.approx(expected_u_rot, abs=1e-12)


@pytest.mark.parametrize('latitude_deg', [30.0, 60.0, -75.0])
def test_a_sphere_keeps_its_outline_at_any_sub_observer_latitude(latitude_deg: float) -> None:
    """A turned sphere covers exactly the pixels it covers seen equator-on.

    Parameters:
        latitude_deg: The sub-observer latitude in degrees.
    """
    equator_on = project_ellipsoid((64, 48), (32.0, 24.5), 10.0, semi_b_px=10.0, semi_c_px=10.0)
    turned = project_ellipsoid(
        (64, 48),
        (32.0, 24.5),
        10.0,
        semi_b_px=10.0,
        semi_c_px=10.0,
        sub_observer_lat=math.radians(latitude_deg),
    )
    np.testing.assert_array_equal(turned.ellipse_dist_sq < 1.0, equator_on.ellipse_dist_sq < 1.0)


def test_an_oblate_bodys_outline_is_the_projection_of_the_turned_body() -> None:
    """The outline reaches along the pole as far as the turned body's farthest point.

    The points are sampled over the surface of a body flattened along its pole
    (a = 30, b = c = 40), turned 35 degrees toward the observer about semi-axis b,
    and dropped onto the image plane.
    """
    latitude = math.radians(35.0)
    projection = project_ellipsoid(
        (8, 8), (4.0, 4.0), 30.0, semi_b_px=40.0, semi_c_px=40.0, sub_observer_lat=latitude
    )
    lat, lon = np.meshgrid(
        np.linspace(-np.pi / 2, np.pi / 2, 20001), np.array([np.pi / 2, 3 * np.pi / 2])
    )
    pole = 30.0 * np.sin(lat)
    depth = 40.0 * np.cos(lat) * np.sin(lon)
    first = pole * math.cos(latitude) - depth * math.sin(latitude)
    assert projection.semi_a_outline == pytest.approx(float(np.abs(first).max()), rel=1e-6)


def test_the_disc_center_sees_the_sub_observer_latitude() -> None:
    """A turned sphere's disc center shows the point at the sub-observer latitude."""
    latitude = math.radians(27.0)
    geometry = project_ellipsoid(
        (4, 4), (2.0, 2.0), 20.0, semi_b_px=20.0, semi_c_px=20.0, sub_observer_lat=latitude
    ).geometry
    x, _y, _z = surface_unit_coords(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), geometry=geometry
    )
    assert float(x[0]) == pytest.approx(math.sin(latitude), abs=1e-12)


def test_the_equator_curves_across_a_turned_disc() -> None:
    """The equator crosses the central meridian sin(lat) of a radius from the center.

    Seen equator-on the equator runs straight through the disc center.  With the
    north pole tipped toward the observer it bows away from the north pole.
    """
    latitude = math.radians(27.0)
    radius = 20.0
    geometry = project_ellipsoid(
        (4, 4), (2.0, 2.0), radius, semi_b_px=radius, semi_c_px=radius, sub_observer_lat=latitude
    ).geometry
    v_rot = np.array([-radius * math.sin(latitude)])
    x, _y, _z = surface_unit_coords(
        v_rot, np.array([0.0]), (v_rot / radius) ** 2, geometry=geometry
    )
    assert float(x[0]) == pytest.approx(0.0, abs=1e-12)


def test_the_surface_point_lies_under_its_pixel() -> None:
    """Each on-disc point, turned back into the view, lies on its pixel's sight line."""
    projection = _tilted_body()
    geometry = projection.geometry
    on_disc = projection.ellipse_dist_sq < 1.0
    x, y, z = projection.surface_unit_coords()
    first, _second, _toward = geometry.to_view(
        x * geometry.semi_a, y * geometry.semi_b, z * geometry.semi_c
    )
    np.testing.assert_allclose(first[on_disc], projection.v_rot[on_disc], rtol=0.0, atol=1e-9)


def test_the_surface_point_lies_on_the_turned_body() -> None:
    """Each on-disc point of a turned triaxial body has unit length after scaling."""
    projection = _tilted_body()
    on_disc = projection.ellipse_dist_sq < 1.0
    x, y, z = projection.surface_unit_coords()
    np.testing.assert_allclose((x**2 + y**2 + z**2)[on_disc], 1.0, rtol=0.0, atol=1e-9)


def test_a_turned_spheres_normals_match_the_sphere_seen_equator_on() -> None:
    """Turning a sphere about its center moves no surface normal on the image."""
    equator_on = project_ellipsoid((40, 40), (20.0, 20.0), 15.0, semi_b_px=15.0, semi_c_px=15.0)
    turned = project_ellipsoid(
        (40, 40), (20.0, 20.0), 15.0, semi_b_px=15.0, semi_c_px=15.0, sub_observer_lat=0.9
    )
    np.testing.assert_allclose(
        np.stack(ellipsoid_image_normals(turned)),
        np.stack(ellipsoid_image_normals(equator_on)),
        rtol=0.0,
        atol=1e-9,
    )


def test_normals_are_perpendicular_to_the_turned_surface() -> None:
    """The normal's slope matches the slope of the turned surface's depth.

    The depth toward the observer is read from each pixel's surface point, and its
    slope across neighboring rows, by a central difference, gives a direction along
    the surface the normal must be perpendicular to.
    """
    projection = project_ellipsoid(
        (200, 200), (100.0, 100.0), 60.0, semi_b_px=45.0, semi_c_px=80.0, sub_observer_lat=0.6
    )
    geometry = projection.geometry
    x, y, z = projection.surface_unit_coords()
    toward = geometry.to_view(x * geometry.semi_a, y * geometry.semi_b, z * geometry.semi_c)[2]
    slope_v = (toward[121, 90] - toward[119, 90]) / 2.0
    normal_v, _normal_u, normal_z = ellipsoid_image_normals(projection)
    assert -normal_v[120, 90] / normal_z[120, 90] == pytest.approx(slope_v, rel=1e-3)


def test_normals_face_the_way_the_outline_does() -> None:
    """On an untilted rotated body the in-plane normal follows the ellipse's gradient.

    The gradient of ``(v_rot / a)**2 + (u_rot / b)**2`` with respect to the image
    coordinates comes from the chain rule through the rotation.
    """
    normal_v, normal_u, _ = ellipsoid_image_normals(_hand_body(sub_observer_lat=0.0))
    cos_rz, sin_rz = math.cos(_ROTATION_Z), math.sin(_ROTATION_Z)
    v_rot = _V * cos_rz - _U * sin_rz
    u_rot = _V * sin_rz + _U * cos_rz
    grad_v = v_rot * cos_rz / _SEMI_A**2 + u_rot * sin_rz / _SEMI_B**2
    grad_u = -v_rot * sin_rz / _SEMI_A**2 + u_rot * cos_rz / _SEMI_B**2
    in_plane = math.hypot(normal_v[_ROW, _COL], normal_u[_ROW, _COL])
    grad = math.hypot(grad_v, grad_u)
    assert normal_v[_ROW, _COL] / in_plane == pytest.approx(grad_v / grad, abs=1e-12)
    assert normal_u[_ROW, _COL] / in_plane == pytest.approx(grad_u / grad, abs=1e-12)


@pytest.mark.parametrize('name', ['v_coords', 'u_coords', 'v_rot', 'u_rot', 'ellipse_dist_sq'])
def test_crop_keeps_each_pixels_values(name: str) -> None:
    """A box cropped from a supersampled working grid reads the whole grid's values."""
    whole = _tilted_body(anti_aliasing=1.0)
    rows, cols = slice(37, 141), slice(12, 97)
    cropped = whole.crop(rows, cols)
    np.testing.assert_array_equal(getattr(cropped, name), getattr(whole, name)[rows, cols])


def test_crop_keeps_each_pixels_position_on_the_whole_grid() -> None:
    """Adding the center to a cropped coordinate gives the pixel center on the whole grid."""
    rows, cols = slice(37, 141), slice(12, 97)
    cropped = _tilted_body(anti_aliasing=1.0).crop(rows, cols)
    np.testing.assert_allclose(
        cropped.v_coords[:, 0] + cropped.center[0], np.arange(37, 141) + 0.5, atol=1e-12
    )
    np.testing.assert_allclose(
        cropped.u_coords[0, :] + cropped.center[1], np.arange(12, 97) + 0.5, atol=1e-12
    )


def test_normals_outside_the_limb_lie_in_the_image_plane() -> None:
    """Off the body the normal has no component toward the observer."""
    projection = _tilted_body()
    normal_z = ellipsoid_image_normals(projection)[2]
    assert float(np.abs(normal_z[projection.ellipse_dist_sq > 1.0]).max()) == 0.0


def test_normals_are_unit_length_at_every_pixel() -> None:
    """On and off the body every normal has length 1."""
    normal_v, normal_u, normal_z = ellipsoid_image_normals(_tilted_body())
    length = np.sqrt(normal_v**2 + normal_u**2 + normal_z**2)
    np.testing.assert_allclose(length, 1.0, rtol=0.0, atol=1e-12)


def test_render_lambert_ellipsoid_shades_by_incidence() -> None:
    """Lit head-on, a sphere's pixel is the cosine of incidence at its center."""
    radius = 20.0
    projection = project_ellipsoid(
        (64, 64), (32.0, 32.0), radius, semi_b_px=radius, semi_c_px=radius
    )
    image = render_lambert_ellipsoid(projection, illumination_angle=0.0, phase_angle=0.0)
    r2 = projection.v_coords**2 + projection.u_coords**2
    cos_i = np.sqrt(np.maximum(0.0, 1.0 - r2 / radius**2))
    lit = cos_i > 0.02
    np.testing.assert_allclose(image[lit], cos_i[lit], rtol=0.0, atol=1e-12)


def test_render_lambert_ellipsoid_is_dark_outside_the_limb() -> None:
    """A pixel whose center lies off the body renders 0."""
    projection = _tilted_body()
    image = render_lambert_ellipsoid(projection, illumination_angle=1.1, phase_angle=0.5)
    assert float(image[projection.ellipse_dist_sq > 1.0].max()) == 0.0


def test_render_lambert_ellipsoid_averages_the_supersample() -> None:
    """At full anti-aliasing each pixel is the mean of a 4 x 4 block of a finer render."""
    coarse = render_lambert_ellipsoid(
        _tilted_body(anti_aliasing=1.0), illumination_angle=1.1, phase_angle=0.5
    )
    fine_projection = project_ellipsoid(
        (192, 160),
        (22.3 * 4, 19.6 * 4),
        14.0 * 4,
        semi_b_px=10.5 * 4,
        semi_c_px=12.0 * 4,
        rotation_z=0.6,
        sub_observer_lat=0.4,
    )
    fine = render_lambert_ellipsoid(fine_projection, illumination_angle=1.1, phase_angle=0.5)
    np.testing.assert_allclose(
        coarse, fine.reshape(48, 4, 40, 4).mean(axis=(1, 3)), rtol=0.0, atol=1e-15
    )
