"""The shared ellipsoid projection, its normals, and the smooth Lambert render.

A cropped projection keeps each pixel's values, the normals outside the limb lie in
the image plane, and the Lambert render shades each pixel by its cosine of incidence,
leaves the sky dark, and averages its supersampled grid down to the caller's.
"""

import numpy as np
import pytest

from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    ellipsoid_image_normals,
    project_ellipsoid,
    render_lambert_ellipsoid,
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


@pytest.mark.parametrize('name', ['v_coords', 'u_coords', 'v_rot', 'u_rot', 'ellipse_dist_sq'])
def test_crop_keeps_each_pixels_values(name: str) -> None:
    """A box cropped from a supersampled working grid reads the whole grid's values."""
    whole = _tilted_body(anti_aliasing=1.0)
    rows, cols = slice(37, 141), slice(12, 97)
    cropped = whole.crop(rows, cols)
    np.testing.assert_array_equal(getattr(cropped, name), getattr(whole, name)[rows, cols])


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
