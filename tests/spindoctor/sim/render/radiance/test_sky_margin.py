"""The background sky drawn across the margin around the frame.

The render draws the scene past the frame edge, as far as the optics carry light into
the frame.  These tests check that the sky's stars there do not depend on how wide
that margin is, that the frame's own stars do not depend on it either, that the draw
repeats, and that a rolled sky is the unrolled sky turned with the frame, margin and
all.
"""

from typing import Any

import numpy as np

from spindoctor.sim.render.radiance.stars import clear_star_caches, render_sky_counts

# A frame this size holds a few hundred sky stars at the density below.
_SIZE = 48

# A dense sky: about 250 stars per square degree at 60 arcsec per pixel.
_SKY: dict[str, float] = {
    'a': -1.0,
    'b': 0.34,
    'density_factor': 1.0,
    'pixel_scale_arcsec': 60.0,
    'faint_cutoff_mag': 10.0,
    'zero_point': 1000.0,
    'exposure_sec': 1.0,
    'diffuse_flux_per_px': 0.0,
}


def _sky_plane(margin: int, *, rotation_deg: float = 0.0) -> np.ndarray:
    """The sky field on a frame with a margin, at oversample 1.

    Parameters:
        margin: How many pixels the plane extends past the frame on each side.
        rotation_deg: The camera roll in degrees.

    Returns:
        The point-source plane, the frame and its margin.
    """
    clear_star_caches()
    plane = np.zeros((_SIZE + 2 * margin, _SIZE + 2 * margin), dtype=np.float64)
    sky: dict[str, Any] = dict(_SKY)
    render_sky_counts(plane, seed=7, oversample=1, rotation_deg=rotation_deg, margin=margin, **sky)
    return plane


def test_margin_stars_do_not_depend_on_the_margin_width() -> None:
    """A margin one pixel wider adds stars in that pixel and changes no other star.

    A star just outside the narrow plane still puts part of its light on the plane's
    outermost pixels, so the narrow plane is the wide one cut down, edge pixels
    included.
    """
    narrow = _sky_plane(8)
    wide = _sky_plane(9)
    assert np.array_equal(narrow, wide[1:-1, 1:-1])


def test_rolled_margin_stars_do_not_depend_on_the_margin_width() -> None:
    """A rolled sky with a wider margin holds the narrower rolled sky, edges included."""
    narrow = _sky_plane(8, rotation_deg=30.0)
    wide = _sky_plane(9, rotation_deg=30.0)
    assert np.array_equal(narrow, wide[1:-1, 1:-1])


def test_margin_holds_sky_stars() -> None:
    """The margin gets its own sky stars, beyond the reach of the frame stars' light.

    A frame star's bilinear share reaches one pixel past the frame, so the rows
    farther out hold only the margin's own stars.
    """
    assert float(_sky_plane(8)[:6, :].sum()) > 0.0


def test_frame_stars_do_not_depend_on_the_margin() -> None:
    """The frame's own stars are the ones a frame without a margin draws.

    Margin stars put part of their light on the frame's outermost pixels, so those
    are left out of the comparison.
    """
    framed = _sky_plane(8)[9:-9, 9:-9]
    plain = _sky_plane(0)[1:-1, 1:-1]
    assert np.array_equal(framed, plain)


def test_margin_sky_repeats() -> None:
    """The same scene draws the same margin sky every time."""
    assert np.array_equal(_sky_plane(8), _sky_plane(8))


def test_rolled_sky_is_the_unrolled_sky_turned_margin_included() -> None:
    """A quarter-turn roll renders the unrolled sky, margin and all, turned a quarter turn."""
    unrolled = _sky_plane(8)
    rolled = _sky_plane(8, rotation_deg=90.0)
    difference = np.abs(rolled - np.rot90(unrolled, 1)).max()
    assert difference / unrolled.max() < 1e-9
