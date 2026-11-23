"""A planted pointing offset moves the background sky as it moves every other object.

The offset is the pointing error the navigator is asked to find, so each object in the
scene sits where it would without the offset plus the offset.  A sky star the offset
carries into or out of the frame or its margin is kept or dropped by where it lands,
and the sky's random draw does not depend on the offset, so the same stars move.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.radiance import stars
from spindoctor.sim.render.radiance.stars import clear_star_caches, render_sky_counts
from spindoctor.support.types import NDArrayFloatType

_SIZE = 64
# The sky beyond the frame is drawn in tiles of 32 pixels, so an offset of more than
# a tile reaches stars from tiles the unplanted frame and its neighbors never draw.
# A whole-pixel offset moves each star's bilinear deposit exactly, so planes compare.
_FAR_DV = 37
_FAR_DU = -45
# The unplanted sky drawn this far past the frame holds every star the far offset
# carries onto a frame with a margin of up to 8 pixels, edge shares included.
_WIDE_MARGIN = 56
# A small whole-pixel offset for the rendered scene, whose edges the optics touch.
_DV = 5
_DU = -3
# Pixels cut from every edge of a rendered frame, which the offset's own edge effects
# touch through the optics and the margin's reach.
_EDGE_PX = 12

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


def _plane(
    margin: int, *, offset: tuple[float, float] = (0.0, 0.0), rotation_deg: float = 0.0
) -> NDArrayFloatType:
    """The sky field of a frame with a margin, planted offset, and roll, at oversample 1."""
    clear_star_caches()
    plane = np.zeros((_SIZE + 2 * margin, _SIZE + 2 * margin), dtype=np.float64)
    render_sky_counts(
        plane,
        seed=7,
        oversample=1,
        margin=margin,
        offset_v=offset[0],
        offset_u=offset[1],
        rotation_deg=rotation_deg,
        **_SKY,
    )
    return plane


def _unplanted_moved(margin: int, *, rotation_deg: float = 0.0) -> NDArrayFloatType:
    """The unplanted sky moved by the far offset, cut to a frame with a margin."""
    wide = _plane(_WIDE_MARGIN, rotation_deg=rotation_deg)
    top = _WIDE_MARGIN - margin - _FAR_DV
    left = _WIDE_MARGIN - margin - _FAR_DU
    side = _SIZE + 2 * margin
    return wide[top : top + side, left : left + side]


def test_offset_sky_is_the_unplanted_sky_moved() -> None:
    """The planted frame is the unplanted sky moved by the offset, stars from beyond included."""
    planted = _plane(0, offset=(float(_FAR_DV), float(_FAR_DU)))
    assert np.allclose(planted, _unplanted_moved(0), rtol=0.0, atol=1e-9)


def test_offset_sky_is_the_rolled_sky_moved() -> None:
    """The offset moves the sky after the roll has turned it."""
    planted = _plane(0, offset=(float(_FAR_DV), float(_FAR_DU)), rotation_deg=30.0)
    expected = _unplanted_moved(0, rotation_deg=30.0)
    assert np.allclose(planted, expected, rtol=0.0, atol=1e-9)


def test_offset_sky_with_a_margin_is_the_unplanted_sky_moved() -> None:
    """A star the offset carries out of the plane is dropped and one carried in is kept."""
    planted = _plane(4, offset=(float(_FAR_DV), float(_FAR_DU)))
    assert np.allclose(planted, _unplanted_moved(4), rtol=0.0, atol=1e-9)


def test_tiny_offset_moves_the_sky_a_tiny_way_edges_included() -> None:
    """A frame with no margin or roll changes by a tiny offset's translation alone.

    A star less than a pixel outside the frame lights its edge pixels with no offset as
    with one, so the edge rows and columns change no more than the interior does.  At
    this density the frame's brightest pixel holds about 36 units of flux, and a move of
    1e-6 px changes no pixel by as much as 1e-4.
    """
    still = _plane(0)
    nudged = _plane(0, offset=(1e-6, -1e-6))
    assert np.abs(nudged - still).max() < 1e-3


def test_stars_beyond_an_empty_frame_light_its_edge() -> None:
    """A frame whose own sky draw is empty still shows the edge light of stars past it."""
    clear_star_caches()
    plane = np.zeros((8, 8), dtype=np.float64)
    # On an 8-pixel frame, seed 34 draws no star inside the frame and one star less than
    # a pixel outside it.
    render_sky_counts(plane, seed=34, oversample=1, margin=0, **_SKY)
    assert float(plane.sum()) > 0.0


def _deposited_stars(
    monkeypatch: pytest.MonkeyPatch, margin: int, offset: tuple[float, float]
) -> set[tuple[float, float, float]]:
    """The position and total flux of every sky star deposited on a frame and margin."""
    deposited: set[tuple[float, float, float]] = set()
    deposit = stars._deposit_point_mass

    def record(plane: NDArrayFloatType, v: float, u: float, *, total: float, margin: int) -> None:
        """Note one star's deposit, then make it."""
        deposited.add((v, u, total))
        deposit(plane, v, u, total=total, margin=margin)

    monkeypatch.setattr(stars, '_deposit_point_mass', record)
    _plane(margin, offset=offset)
    monkeypatch.undo()
    return deposited


def test_offset_does_not_change_the_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fractional offset draws the same stars and moves each of them by the offset.

    The planted frame holds exactly the unplanted stars whose moved position lands where
    its bilinear deposit reaches the frame: less than a pixel before the first pixel
    center, and before the end of the frame.
    """
    dv = _FAR_DV + 0.37
    du = _FAR_DU - 0.61
    unplanted = _deposited_stars(monkeypatch, _WIDE_MARGIN, (0.0, 0.0))
    planted = _deposited_stars(monkeypatch, 0, (dv, du))
    expected = {
        (v + dv, u + du, total)
        for v, u, total in unplanted
        if -1.0 < v + dv < _SIZE and -1.0 < u + du < _SIZE
    }
    assert planted == expected


def _sky_scene(offset_v: float, offset_u: float) -> dict[str, Any]:
    """A noiseless frame holding a dense sky field and a planted offset."""
    params = frame_scene(
        size=_SIZE,
        seed=5,
        exposure_sec=1.0,
        noise=quiet_noise(),
        sky_counts={'a': 1.0, 'b': 0.34, 'density_factor': 50.0},
        optics={'psf': {'sigma_v': 0.8}},
    )
    params['offset_v'] = offset_v
    params['offset_u'] = offset_u
    return params


def test_rendered_scene_moves_the_sky_with_the_offset() -> None:
    """A rendered sky-only scene with a planted offset is the unplanted render moved."""
    plain, _ = render_scene(_sky_scene(0.0, 0.0))
    planted, _ = render_scene(_sky_scene(float(_DV), float(_DU)))
    moved = np.asarray(plain, dtype=np.float64)[
        _EDGE_PX - _DV : _SIZE - _EDGE_PX - _DV, _EDGE_PX - _DU : _SIZE - _EDGE_PX - _DU
    ]
    cut = np.asarray(planted, dtype=np.float64)[
        _EDGE_PX : _SIZE - _EDGE_PX, _EDGE_PX : _SIZE - _EDGE_PX
    ]
    assert np.abs(cut - moved).max() <= 1.0
