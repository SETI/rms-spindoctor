"""The shared ellipsoid projection, its normals, and the smooth Lambert render.

The projection measures each pixel from its center, rotates and then tilts it as a
hand computation does, and stretches the outline by the tilt.  A cropped projection
keeps each pixel's values and its position on the whole grid.  The normals face the
way the outline does, lie in the image plane outside the limb, and have unit length.
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
)

# A hand-computed pixel: row 23, column 15 of a body centered at (20.0, 18.0), whose
# pixel center lies 3.5 px below and 2.5 px left of the body center.
_ROW, _COL = 23, 15
_CENTER = (20.0, 18.0)
_V, _U = 3.5, -2.5
_ROTATION_Z = 0.5
_TILT = 0.6
_SEMI_A, _SEMI_B = 12.0, 7.0


def _hand_body(*, rotation_tilt: float) -> EllipsoidProjection:
    """The body the hand-computed pixel belongs to, at a chosen tilt."""
    return project_ellipsoid(
        (48, 40),
        _CENTER,
        _SEMI_A,
        semi_b_px=_SEMI_B,
        semi_c_px=9.0,
        rotation_z=_ROTATION_Z,
        rotation_tilt=rotation_tilt,
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
        rotation_tilt=0.4,
        anti_aliasing=anti_aliasing,
    )


def test_projection_measures_from_pixel_centers() -> None:
    """A pixel's coordinates run from the body center to the pixel's center."""
    projection = _hand_body(rotation_tilt=_TILT)
    assert projection.v_coords[_ROW, _COL] == pytest.approx(_V, abs=1e-12)
    assert projection.u_coords[_ROW, _COL] == pytest.approx(_U, abs=1e-12)


def test_projection_rotates_then_tilts_a_pixel() -> None:
    """The rotated frame turns by rotation_z, then the tilt scales its v by cos(tilt)."""
    projection = _hand_body(rotation_tilt=_TILT)
    cos_rz, sin_rz = math.cos(_ROTATION_Z), math.sin(_ROTATION_Z)
    expected_v_rot = (_V * cos_rz - _U * sin_rz) * math.cos(_TILT)
    expected_u_rot = _V * sin_rz + _U * cos_rz
    assert projection.v_rot[_ROW, _COL] == pytest.approx(expected_v_rot, abs=1e-12)
    assert projection.u_rot[_ROW, _COL] == pytest.approx(expected_u_rot, abs=1e-12)


def test_projection_stretches_the_outline_by_the_tilt() -> None:
    """A 10 px sphere tilted by 1 radian reaches 10 / cos(1) = 18.51 px along v.

    One column of pixel centers runs through the body center, and the rows whose
    centers lie 0.5 to 18.5 px above and below it fall within that reach: 38 rows.
    """
    projection = project_ellipsoid(
        (64, 48), (32.0, 24.5), 10.0, semi_b_px=10.0, semi_c_px=10.0, rotation_tilt=1.0
    )
    covered_rows = int(np.any(projection.ellipse_dist_sq <= 1.0, axis=1).sum())
    assert covered_rows == 38


def test_normals_face_the_way_the_outline_does() -> None:
    """On an untilted rotated body the in-plane normal follows the ellipse's gradient.

    The gradient of ``(v_rot / a)**2 + (u_rot / b)**2`` with respect to the image
    coordinates comes from the chain rule through the rotation.
    """
    normal_v, normal_u, _ = ellipsoid_image_normals(_hand_body(rotation_tilt=0.0))
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
        rotation_tilt=0.4,
    )
    fine = render_lambert_ellipsoid(fine_projection, illumination_angle=1.1, phase_angle=0.5)
    np.testing.assert_allclose(
        coarse, fine.reshape(48, 4, 40, 4).mean(axis=(1, 3)), rtol=0.0, atol=1e-15
    )
