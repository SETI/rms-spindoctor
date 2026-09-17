"""Inventory bounding boxes follow the projected, rotated silhouette (SIM-7).

A body's inventory bbox uses the per-axis half-extents of the axis1/axis2
ellipse rotated in-plane by rotation_z (axis3 is the depth axis and does not
project): sqrt((a cos t)^2 + (b sin t)^2) along v and
sqrt((a sin t)^2 + (b cos t)^2) along u.  These tests plant an elongated,
tilted body and check the bbox against both the analytic values and the
rendered silhouette.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX

_CENTER = 64.0
_SEMI_MAJOR = 30.0  # axis1 / 2
_SEMI_MINOR = 10.0  # axis2 / 2
_ROTATION_Z_DEG = 30.0

_COS_T = np.cos(np.radians(_ROTATION_Z_DEG))
_SIN_T = np.sin(np.radians(_ROTATION_Z_DEG))
_HALF_V = float(np.hypot(_SEMI_MAJOR * _COS_T, _SEMI_MINOR * _SIN_T))
_HALF_U = float(np.hypot(_SEMI_MAJOR * _SIN_T, _SEMI_MINOR * _COS_T))

# The bbox is stated in the pixel corner coordinates the scene uses; the mask
# is addressed by rows and columns.  Converting between them is the only thing
# separating the two, so the containment bound below is zero: every marked
# pixel's own center lies inside the converted box.  The tightness bound is one
# pixel, which is what a rasterized extent can promise -- the outermost marked
# pixel is the last one whose center the box contains, so the box edge can be
# up to a whole pixel beyond it and no further.
_TIGHTNESS_BOUND_PX = 1.0


def _scene() -> dict[str, Any]:
    """A noiseless scene with one fully lit, elongated, rotated body."""
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 3,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [
            {
                'name': 'SLAB',
                'center_v': _CENTER,
                'center_u': _CENTER,
                'axis1': 2 * _SEMI_MAJOR,
                'axis2': 2 * _SEMI_MINOR,
                'axis3': 2 * _SEMI_MINOR,
                'rotation_z': _ROTATION_Z_DEG,
                'phase_angle': 0.0,
                'anti_aliasing': 0.0,
            }
        ],
    }


def _inventory() -> dict[str, float]:
    """Render the scene and return the planted body's inventory entry."""
    _, meta = render_combined_model(_scene())
    inventory: dict[str, float] = meta['inventory']['SLAB']
    return inventory


def _mask_extents() -> tuple[float, float, float, float]:
    """Render the scene and return (v_min, v_max, u_min, u_max) of the mask."""
    _, meta = render_combined_model(_scene())
    mask = meta['body_masks'][0]
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return float(rows.min()), float(rows.max()), float(cols.min()), float(cols.max())


def _mask_centroid() -> tuple[float, float]:
    """Render the scene and return the silhouette's (v, u) centroid in rows and columns."""
    _, meta = render_combined_model(_scene())
    mask = meta['body_masks'][0]
    vv, uu = np.indices(mask.shape)
    return float(vv[mask].mean()), float(uu[mask].mean())


def _bbox_centric(inventory: dict[str, float], axis: str) -> tuple[float, float]:
    """Return one axis of the bbox converted to the rows and columns of the mask.

    Parameters:
        inventory: The planted body's inventory entry.
        axis: ``'v'`` or ``'u'``.
    """
    low = inventory[f'{axis}_min_unclipped'] - PIXEL_CENTER_TO_CORNER_PX
    high = inventory[f'{axis}_max_unclipped'] - PIXEL_CENTER_TO_CORNER_PX
    return low, high


def test_bbox_v_size_matches_rotated_ellipse() -> None:
    """The v pixel size equals the analytic projected extent along v."""
    assert _inventory()['v_pixel_size'] == pytest.approx(2 * _HALF_V)


def test_bbox_u_size_matches_rotated_ellipse() -> None:
    """The u pixel size equals the analytic projected extent along u."""
    assert _inventory()['u_pixel_size'] == pytest.approx(2 * _HALF_U)


def test_bbox_v_limits_match_rotated_ellipse() -> None:
    """The v min/max limits bracket the center by the analytic half-extent."""
    inventory = _inventory()
    assert inventory['v_min_unclipped'] == pytest.approx(_CENTER - _HALF_V)
    assert inventory['v_max_unclipped'] == pytest.approx(_CENTER + _HALF_V)


def test_bbox_u_limits_match_rotated_ellipse() -> None:
    """The u min/max limits bracket the center by the analytic half-extent."""
    inventory = _inventory()
    assert inventory['u_min_unclipped'] == pytest.approx(_CENTER - _HALF_U)
    assert inventory['u_max_unclipped'] == pytest.approx(_CENTER + _HALF_U)


def test_silhouette_centers_on_the_bbox_center() -> None:
    """The rendered silhouette's centroid is the middle of the inventory bbox.

    This is the one statement here that resolves the conversion the bbox
    spans.  The ellipse is symmetric about its center, so the centroid of the
    marked pixels is exact, and the bbox's own midpoint is the scene's stated
    center; the two differ by exactly the half pixel between a pixel corner
    coordinate and a row or column.  An extent bound cannot see that half
    pixel, because a rasterized extent is only good to the pixel.
    """
    inventory = _inventory()
    centroid_v, centroid_u = _mask_centroid()
    v_low, v_high = _bbox_centric(inventory, 'v')
    u_low, u_high = _bbox_centric(inventory, 'u')
    assert centroid_v == pytest.approx((v_low + v_high) / 2.0, abs=1e-12)
    assert centroid_u == pytest.approx((u_low + u_high) / 2.0, abs=1e-12)


def test_silhouette_falls_inside_bbox() -> None:
    """Every rendered body pixel's center lies within the inventory bbox.

    The bound is zero: a pixel is marked because its own center is inside the
    ellipse, so once the bbox is converted to rows and columns no marked pixel
    can lie outside it at all.
    """
    inventory = _inventory()
    v_min, v_max, u_min, u_max = _mask_extents()
    v_low, v_high = _bbox_centric(inventory, 'v')
    u_low, u_high = _bbox_centric(inventory, 'u')
    assert v_min >= v_low
    assert v_max <= v_high
    assert u_min >= u_low
    assert u_max <= u_high


def test_bbox_hugs_silhouette() -> None:
    """The bbox is tight: the silhouette reaches within a pixel of each bbox edge.

    One pixel is what a rasterized extent can promise.  The outermost marked
    pixel is the last one whose center the box contains, so the box edge lies
    less than a whole pixel beyond it; anything looser would admit an empty
    row or column between the silhouette and the box.
    """
    inventory = _inventory()
    v_min, v_max, u_min, u_max = _mask_extents()
    v_low, v_high = _bbox_centric(inventory, 'v')
    u_low, u_high = _bbox_centric(inventory, 'u')
    assert v_min - v_low < _TIGHTNESS_BOUND_PX
    assert v_high - v_max < _TIGHTNESS_BOUND_PX
    assert u_min - u_low < _TIGHTNESS_BOUND_PX
    assert u_high - u_max < _TIGHTNESS_BOUND_PX
