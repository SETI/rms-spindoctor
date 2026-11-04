"""Acceptance checks for the planet disc texture (bands/storms) and transits.

The contract: bands are aligned with body-polar latitude and rotate with the
body's in-plane pose, storms land where commanded, a transit shadow darkens
the textured disc by its commanded factor, a transiting moon disc occludes
the texture (rendering from the smooth-law shading), and none of it moves
the silhouette.
"""

import dataclasses
import math

import numpy as np
import pytest

from spindoctor.sim.forward.body_texture import DiscTextureSpec, TransitSpec
from spindoctor.sim.forward.body_topo import TopoBodySpec, create_topographic_body
from spindoctor.support.types import NDArrayFloatType

_SIZE = 200
_CENTER = _SIZE / 2.0
_RADIUS = 80.0
# Moderate phase keeps the shading well below the ceiling so multiplied
# texture never clips and ratio probes stay exact.
_PHASE = 1.0

_BANDS = DiscTextureSpec(band_amplitude=0.3, band_wavenumber=8.0, band_phase_deg=0.0)


def _spec(
    disc_texture: DiscTextureSpec | None,
    *,
    transits: tuple[TransitSpec, ...] = (),
    rotation_z: float = 0.0,
) -> TopoBodySpec:
    """A sphere at moderate phase with optional disc texture and transits.

    The opposition surge (amplitude 1 at phase ~57 deg) halves the whole
    disc's shading, so mu0 -> 1 near the sub-solar limb stays well below the
    signal ceiling and a multiplied texture can never clip: every ratio
    probe below is exact.
    """
    return TopoBodySpec(
        axis1=2.0 * _RADIUS,
        axis2=2.0 * _RADIUS,
        axis3=2.0 * _RADIUS,
        rotation_z=rotation_z,
        phase_angle=_PHASE,
        surge_amplitude=1.0,
        disc_texture=disc_texture,
        transits=transits,
    )


def _render(spec: TopoBodySpec) -> NDArrayFloatType:
    return create_topographic_body((_SIZE, _SIZE), (_CENTER, _CENTER), spec)


def _texture_ratio(spec: TopoBodySpec) -> NDArrayFloatType:
    """The textured/smooth ratio (1 outside the disc)."""
    smooth = _render(dataclasses.replace(spec, disc_texture=None, transits=()))
    textured = _render(spec)
    return np.divide(textured, smooth, out=np.ones_like(smooth), where=smooth > 0)


def test_bands_are_latitude_aligned_at_zero_rotation() -> None:
    """With the pole along +v, the band pattern is constant along each row."""
    ratio = _texture_ratio(_spec(_BANDS))
    # Central disc window, comfortably inside the limb.
    window = ratio[60:140, 60:140]
    within_row_spread = float(np.std(window, axis=1).max())
    across_rows_spread = float(np.std(window.mean(axis=1)))
    assert within_row_spread < 0.02
    assert across_rows_spread > 0.1


def test_bands_rotate_with_the_pose() -> None:
    """At rotation_z = 90 deg the bands run along columns instead of rows."""
    ratio = _texture_ratio(_spec(_BANDS, rotation_z=math.pi / 2.0))
    window = ratio[60:140, 60:140]
    within_col_spread = float(np.std(window, axis=0).max())
    across_cols_spread = float(np.std(window.mean(axis=0)))
    assert within_col_spread < 0.02
    assert across_cols_spread > 0.1


def test_storm_lands_at_commanded_position() -> None:
    """A storm at (lat 0, lon 90) brightens exactly the sub-observer point."""
    storms = DiscTextureSpec(storms=((0.0, 90.0, 8.0, 1.4),))
    ratio = _texture_ratio(_spec(storms))
    center = int(_CENTER)
    assert ratio[center, center] == pytest.approx(1.4, abs=1e-9)
    # Well away from the storm the disc is untouched.
    assert ratio[center + 60, center] == pytest.approx(1.0, abs=1e-12)


def test_transit_shadow_darkens_the_texture() -> None:
    """The shadow multiplies the banded texture by 1 - darkness."""
    shadow = TransitSpec(shadow=(0.0, 20.0, 8.0, 0.8))
    banded = _render(_spec(_BANDS))
    shadowed = _render(_spec(_BANDS, transits=(shadow,)))
    v, u = int(_CENTER), int(_CENTER + 20)
    assert shadowed[v, u] == pytest.approx(banded[v, u] * 0.2, abs=1e-12)
    # The shadow is darker than every textured pixel in its neighborhood ring.
    ring = banded[v - 12 : v + 12, u - 12 : u + 12]
    assert shadowed[v, u] < float(ring.min())


def test_moon_disc_occludes_the_texture() -> None:
    """The transiting moon renders from the smooth-law shading, not the bands."""
    moon = TransitSpec(moon=(0.0, -20.0, 8.0, 1.0))
    spec = _spec(_BANDS, transits=(moon,))
    smooth = _render(dataclasses.replace(spec, disc_texture=None, transits=()))
    banded = _render(_spec(_BANDS))
    transit = _render(spec)
    v, u = int(_CENTER), int(_CENTER - 20)
    # The bands were visibly present at the moon's location...
    assert abs(banded[v, u] - smooth[v, u]) > 0.01
    # ...and the moon disc removed them (albedo_factor 1 = smooth shading).
    assert transit[v, u] == pytest.approx(smooth[v, u], abs=1e-12)


def test_moon_disc_covers_its_own_shadow() -> None:
    """A moon disc placed over a shadow wins: moons render after shadows."""
    entry = TransitSpec(moon=(0.0, 10.0, 6.0, 1.2), shadow=(0.0, 10.0, 6.0, 0.9))
    spec = _spec(_BANDS, transits=(entry,))
    smooth = _render(dataclasses.replace(spec, disc_texture=None, transits=()))
    transit = _render(spec)
    v, u = int(_CENTER), int(_CENTER + 10)
    assert transit[v, u] == pytest.approx(smooth[v, u] * 1.2, abs=1e-12)


def test_transits_render_only_on_the_parent_disc() -> None:
    """A transit disc hanging over the limb affects no off-body pixel."""
    moon = TransitSpec(moon=(0.0, _RADIUS - 2.0, 10.0, 1.5))
    smooth = _render(_spec(None))
    transit = _render(_spec(None, transits=(moon,)))
    off_disc = smooth == 0.0
    assert np.array_equal(transit[off_disc], smooth[off_disc])


def test_a_transit_shadow_reaches_the_relief_outline() -> None:
    """With limb relief, a shadow at the limb darkens the terrain past the smooth limb too."""
    shadow = TransitSpec(shadow=(0.0, _RADIUS, 10.0, 0.8))
    relief = dataclasses.replace(
        _spec(None), anti_aliasing=0.0, limb_relief_rms=0.03, limb_relief_corr_deg=15.0
    )
    lit = _render(relief)
    shadowed = _render(dataclasses.replace(relief, transits=(shadow,)))
    v, u = np.mgrid[0:_SIZE, 0:_SIZE] + 0.5
    past_limb = (v - _CENTER) ** 2 + (u - _CENTER) ** 2 > _RADIUS**2
    under = (v - _CENTER) ** 2 + (u - _CENTER - _RADIUS) ** 2 <= 8.0**2
    probe = past_limb & under & (lit > 0.0)
    assert int(probe.sum()) > 0
    np.testing.assert_allclose(shadowed[probe], 0.2 * lit[probe], rtol=1e-12)


def test_silhouette_is_untouched_by_disc_texture_and_transits() -> None:
    """Texture and transits never move the silhouette."""
    spec = _spec(
        _BANDS,
        transits=(TransitSpec(moon=(5.0, 5.0, 10.0, 0.4), shadow=(-8.0, 0.0, 6.0, 0.9)),),
    )
    smooth = _render(dataclasses.replace(spec, disc_texture=None, transits=()))
    textured = _render(spec)
    assert np.array_equal(textured > 0.0, smooth > 0.0)
