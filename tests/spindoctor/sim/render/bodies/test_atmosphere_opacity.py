"""The haze's opacity: the surface under it, the solid limb, and the column's light."""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies.atmosphere import AtmosphereSpec, apply_atmosphere
from spindoctor.sim.shared.ellipsoid import ellipsoid_image_normals, project_ellipsoid
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX

_SIZE = 256
_CENTER = 128.0
_RADIUS = 60.0

# A haze several optical depths thick at every pixel of the disc: H = 3 px with a
# 60-optical-depth tangent column at the surface gives a vertical depth of 7.7.
_THICK = AtmosphereSpec(scale_height_px=3.0, tau_ref=60.0, g=0.6, albedo=0.95)


def _shifted_layers(spec: AtmosphereSpec, body: float, *, shift_u: float, phase_deg: float) -> Any:
    """The haze layers of a sphere of radius ``_RADIUS`` moved ``shift_u`` px along u.

    Parameters:
        spec: The haze spec.
        body: The uniform brightness the sphere is painted with, under its haze.
        shift_u: How far the sphere moves from ``_CENTER`` along u, in pixels.
        phase_deg: The phase angle in degrees.

    Returns:
        The ``AtmosphereLayers`` ``apply_atmosphere`` produces for it.
    """
    center_u = _CENTER + shift_u
    offsets_v = np.arange(_SIZE, dtype=np.float64) + (PIXEL_CENTER_TO_CORNER_PX - _CENTER)
    offsets_u = np.arange(_SIZE, dtype=np.float64) + (PIXEL_CENTER_TO_CORNER_PX - center_u)
    radius2 = offsets_v[:, None] ** 2 + offsets_u[None, :] ** 2
    outline = (radius2 < _RADIUS * _RADIUS).astype(np.float64)
    return apply_atmosphere(
        body * outline,
        spec,
        center_v=_CENTER,
        center_u=center_u,
        semi_a=_RADIUS,
        semi_b=_RADIUS,
        semi_c=_RADIUS,
        rotation_z=0.0,
        sub_observer_lat=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=math.radians(phase_deg),
        silhouette=outline,
    )


def test_a_thick_haze_hides_the_solid_limb() -> None:
    """A thick haze is as bright just inside the solid limb as just outside it.

    One pixel is put 1e-6 px inside the limb, where it is disc, and then 1e-6 px
    outside it, where it is halo.  The disc side sees the near half of the tangent
    column and the halo the whole of it; both are opaque, so they reflect alike and
    the limb of the bright surface beneath does not show.
    """
    pixel = (int(_CENTER), int(_CENTER + _RADIUS))
    pixel_u = pixel[1] + PIXEL_CENTER_TO_CORNER_PX - _CENTER

    def shift_to(radius: float) -> float:
        """The shift along u that puts the pixel's center at ``radius``."""
        return pixel_u - math.sqrt(radius * radius - PIXEL_CENTER_TO_CORNER_PX**2)

    inside = _shifted_layers(_THICK, 0.9, shift_u=shift_to(_RADIUS - 1e-6), phase_deg=155.0)
    outside = _shifted_layers(_THICK, 0.9, shift_u=shift_to(_RADIUS + 1e-6), phase_deg=155.0)
    assert float(inside.disc[pixel]) == pytest.approx(float(outside.halo.emission[pixel]), rel=1e-3)


def test_a_thick_haze_dims_smoothly_across_the_solid_limb() -> None:
    """A thick haze is within 5% as bright 1.5 px outside the solid limb as 1.5 px inside.

    The haze reflects from the level where its tangent depth is 1, about 12 px above
    the limb here, so the solid limb is an ordinary place on that larger disc.  Were
    the haze to take the solid body's emission angles, the brightness would fall by a
    fifth across those three pixels.
    """
    layers = _shifted_layers(_THICK, 0.9, shift_u=0.0, phase_deg=30.0)
    row = int(_CENTER)
    inside = float(layers.disc[row, int(_CENTER + _RADIUS) - 2])
    outside = float(layers.halo.emission[row, int(_CENTER + _RADIUS) + 1])
    assert outside == pytest.approx(inside, rel=0.05)


def test_a_thick_haze_hides_the_surface() -> None:
    """Under a thick haze a bright surface and a black one render the same disc."""
    bright = _shifted_layers(_THICK, 0.9, shift_u=0.0, phase_deg=30.0)
    black = _shifted_layers(_THICK, 0.0, shift_u=0.0, phase_deg=30.0)
    assert float(np.abs(bright.disc - black.disc).max()) < 1e-3


def _disc_center_cosines(spec: AtmosphereSpec, phase_deg: float) -> tuple[float, float]:
    """The effective emission and incidence cosines at the disc-center pixel.

    Parameters:
        spec: The haze spec, whose scale height sets the grazing constant.
        phase_deg: The phase angle in degrees; the sun lies along +u.

    Returns:
        ``(mu, mu0)``, each ``sqrt(cos**2 (1 - c) + c)`` with ``c = 2 H / (pi R)``.
    """
    offset = PIXEL_CENTER_TO_CORNER_PX
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=math.radians(90.0), phase_angle=math.radians(phase_deg)
    )
    cos_emission = math.sqrt(1.0 - 2.0 * (offset / _RADIUS) ** 2)
    cos_incidence = (offset * illum_v + offset * illum_u) / _RADIUS + cos_emission * illum_z
    c = 2.0 * spec.scale_height_px / (math.pi * _RADIUS)
    return (
        math.sqrt(cos_emission**2 * (1.0 - c) + c),
        math.sqrt(cos_incidence**2 * (1.0 - c) + c),
    )


def _vertical_depth(spec: AtmosphereSpec) -> float:
    """The vertical optical depth of a spec referenced at the surface."""
    return spec.tau_ref / math.sqrt(2.0 * math.pi * _RADIUS / spec.scale_height_px)


def test_the_surface_is_dimmed_by_the_haze_column_both_ways() -> None:
    """At disc center the surface shows through exp(-tau_vert (1 / mu + 1 / mu0))."""
    spec = AtmosphereSpec(scale_height_px=8.0, tau_ref=1.5, albedo=0.0)
    layers = _shifted_layers(spec, 1.0, shift_u=0.0, phase_deg=30.0)
    mu, mu0 = _disc_center_cosines(spec, 30.0)
    expected = math.exp(-_vertical_depth(spec) * (1.0 / mu + 1.0 / mu0))
    pixel = (int(_CENTER), int(_CENTER))
    assert float(layers.disc[pixel]) == pytest.approx(expected, rel=1e-9)


def test_a_thin_haze_lights_disc_center_by_its_whole_column() -> None:
    """At disc center a thin haze reflects albedo * P * tau_vert / (4 mu).

    The column the line of sight crosses runs from the top of the haze down to the
    surface, so the haze is not zero where that column is shortest.
    """
    albedo = 0.8
    g = 0.5
    spec = AtmosphereSpec(scale_height_px=8.0, tau_ref=1e-3, g=g, albedo=albedo)
    layers = _shifted_layers(spec, 0.0, shift_u=0.0, phase_deg=30.0)
    mu, _mu0 = _disc_center_cosines(spec, 30.0)
    cos_phase = math.cos(math.radians(30.0))
    phase_function = (1.0 - g * g) / (1.0 + g * g + 2.0 * g * cos_phase) ** 1.5
    expected = albedo * phase_function * _vertical_depth(spec) / (4.0 * mu)
    pixel = (int(_CENTER), int(_CENTER))
    assert float(layers.disc[pixel]) == pytest.approx(expected, rel=1e-3)


def test_haze_on_an_oblate_body_follows_its_surface_normal() -> None:
    """At phase 0 the haze over a strongly oblate body depends only on the normal.

    The body is three times wider than it is tall.  At phase 0 a haze too thin to lift
    its reflecting level lights each pixel by the angle between the surface normal and
    the line of sight, so the short axis matches the long axis wherever their normals
    agree.  Had the haze taken a sphere's emission angle from the outline, the short
    axis would be up to about 30% off.
    """
    semi_a, semi_b = 20.0, 60.0
    projection = project_ellipsoid(
        (_SIZE, _SIZE), (_CENTER, _CENTER), semi_a, semi_b_px=semi_b, semi_c_px=semi_b
    )
    outline = (projection.ellipse_dist_sq < 1.0).astype(np.float64)
    layers = apply_atmosphere(
        np.zeros((_SIZE, _SIZE)),
        AtmosphereSpec(scale_height_px=3.0, tau_ref=0.5, g=0.6, albedo=0.9),
        center_v=_CENTER,
        center_u=_CENTER,
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_b,
        rotation_z=0.0,
        sub_observer_lat=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=0.0,
        silhouette=outline,
    )
    _normal_v, _normal_u, normal_z = ellipsoid_image_normals(projection)
    center = int(_CENTER)
    long_axis = slice(center, center + int(semi_b))
    long_order = np.argsort(normal_z[center, long_axis])
    # The 14 pixels of the short axis whose normals lie within 72 degrees of the line of
    # sight, where the long axis samples the normals finely.
    short_axis = slice(center, center + 14)
    expected = np.interp(
        normal_z[short_axis, center],
        normal_z[center, long_axis][long_order],
        layers.disc[center, long_axis][long_order],
    )
    np.testing.assert_allclose(layers.disc[short_axis, center], expected, rtol=1e-3)


def test_the_glow_over_an_oblate_body_fades_smoothly_past_its_reflecting_level() -> None:
    """Along a diagonal of an oblate body the glow falls at every pixel past its peak.

    The body is three times wider than it is tall and lit from the side at phase 90.  A
    thick haze reflects from a level about 1.3 times the body's size, which the diagonal
    crosses on its way out.  Inside that level the glow takes the scaled surface's
    incidence and outside it the limb's.  Had the limb's incidence followed the
    diagonal itself, the glow would jump back up where the two meet.
    """
    semi_a, semi_b = 30.0, 90.0
    projection = project_ellipsoid(
        (_SIZE, _SIZE), (_CENTER, _CENTER), semi_a, semi_b_px=semi_b, semi_c_px=semi_b
    )
    layers = apply_atmosphere(
        np.zeros((_SIZE, _SIZE)),
        AtmosphereSpec(scale_height_px=6.0, tau_ref=20.0, g=0.3, albedo=0.9),
        center_v=_CENTER,
        center_u=_CENTER,
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_b,
        rotation_z=0.0,
        sub_observer_lat=0.0,
        illumination_angle=math.radians(90.0),
        phase_angle=math.radians(90.0),
        silhouette=(projection.ellipse_dist_sq < 1.0).astype(np.float64),
    )
    steps = np.arange(int(semi_a), int(semi_a) + 30)
    glow = layers.halo.emission[int(_CENTER) - steps, int(_CENTER) + steps]
    past_peak = glow[int(np.argmax(glow)) :]
    assert np.all(np.diff(past_peak) < 0.0)
