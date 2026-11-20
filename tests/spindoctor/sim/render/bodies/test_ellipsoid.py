"""Accuracy of the ellipsoid renderer's shading on its working grid.

A setting at an effectively zero value renders the smooth body.  A small body at
oversample 4 matches an 8 times finer render of the same body.  A relief body has
its light where the same body shaded wholly on its working grid has it, although
its relief is laid out on the detector grid, and so does a body smaller than a
pixel.  An albedo spot's edge is drawn on the working grid.  A sample exactly on a
smooth body's limb lies outside the body, and a cratered body renders in a frame
one detector pixel tall or wide.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene

from spindoctor.sim.render.bodies import ellipsoid
from spindoctor.sim.render.bodies.ellipsoid import (
    EllipsoidBodySpec,
    create_ellipsoid_body,
    render_ellipsoid_body,
)
from spindoctor.sim.render.bodies.ellipsoid_grid import limb_band
from spindoctor.sim.render.bodies.relief import synthesize_relief_field
from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.render.bodies.texture import AlbedoTextureSpec
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.shared.ellipsoid import project_ellipsoid, render_lambert_ellipsoid
from spindoctor.support.types import NDArrayFloatType


def _centroid(img: NDArrayFloatType) -> NDArrayFloatType:
    """The intensity-weighted center of an image, in array rows and columns."""
    vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    total = float(img.sum())
    return np.array([float((img * vv).sum()) / total, float((img * uu).sum()) / total])


def _detector(img: NDArrayFloatType, factor: int) -> NDArrayFloatType:
    """Average an image down by ``factor`` onto the detector grid."""
    rows, cols = img.shape
    return img.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


# ----------------------------------------------------------------------------
# Settings at an effectively zero value
# ----------------------------------------------------------------------------

_ZERO_BODY: dict[str, Any] = {
    'name': 'B',
    'center_v': 64.3,
    'center_u': 63.6,
    'semi_a_px': 40.0,
    'semi_b_px': 35.0,
    'semi_c_px': 35.0,
    'illumination_angle': 30.0,
    'phase_angle': 40.0,
}

_ZERO_SETTINGS: dict[str, dict[str, Any]] = {
    'relief': {'limb_relief_rms': 1e-9},
    'surge': {'opposition_surge': {'amplitude': 1e-12}},
    'minnaert': {'photometric_law': 'minnaert', 'minnaert_k': 1.0},
    'texture': {'albedo_texture': {'rms': 0.0}},
}


def _render_body(extra: dict[str, Any]) -> NDArrayFloatType:
    """The zero-test body, plus ``extra`` keys, drawn by the render path at oversample 1."""
    img = np.zeros((128, 128))
    spec = body_spec_from_params({**_ZERO_BODY, **extra}, index=0)
    render_single_body(img, spec, 0.0, offset_u=0.0, ref_center_v=64.0, ref_center_u=64.0)
    return img


@pytest.mark.parametrize('setting', sorted(_ZERO_SETTINGS))
def test_effectively_zero_setting_renders_the_smooth_body(setting: str) -> None:
    """A setting at an effectively zero value renders the smooth body.

    The relief of 1e-9 of the radius changes the shading by about 1e-8; the bound
    of 1e-7 sits above that and far below the 1% step a body showed when such a
    setting switched it onto another shading path.

    Parameters:
        setting: The key into ``_ZERO_SETTINGS``.
    """
    smooth = _render_body({})
    np.testing.assert_allclose(_render_body(_ZERO_SETTINGS[setting]), smooth, rtol=0.0, atol=1e-7)


# ----------------------------------------------------------------------------
# Small bodies against an 8 times finer render
# ----------------------------------------------------------------------------

_SMALL_SIZE = 32
_SMALL_CENTER = (16.3, 15.7)
_SMALL_ILLUMINATION = 0.7

# Bounds per diameter: (relative brightness error, center-of-light error in px).
# A smooth body drawn at 4 times supersampling on this geometry, as the renderer
# draws it at oversample 4 and as the retired smooth Lambert renderer drew it at
# full anti-aliasing, misses the 32 times truth by at most 0.41% and 0.016 px at
# 4 px, 1.6% and 0.022 px at 8 px, and 0.19% and 0.0016 px at 16 px, all at
# 90 deg phase.  Shading a body with relief at detector resolution instead missed
# by 6% and 0.094 px at 4 px, 2.6% at 8 px, and 0.021 px at 16 px.
_SMALL_BOUNDS = {4.0: (0.02, 0.04), 8.0: (0.02, 0.025), 16.0: (0.005, 0.012)}


def _small_body(diameter: float, oversample: int, *, relief: float) -> NDArrayFloatType:
    """A small body at 90 deg phase, averaged onto the detector grid."""
    semi = diameter / 2.0 * oversample
    spec = EllipsoidBodySpec(
        semi_a_px=semi,
        semi_b_px=semi,
        semi_c_px=semi,
        illumination_angle=_SMALL_ILLUMINATION,
        phase_angle=math.radians(90.0),
        oversample=oversample,
        limb_relief_rms=relief,
        relief_seed=3,
    )
    size = _SMALL_SIZE * oversample
    center = (_SMALL_CENTER[0] * oversample, _SMALL_CENTER[1] * oversample)
    return _detector(create_ellipsoid_body((size, size), center, spec), oversample)


@pytest.mark.parametrize('diameter', sorted(_SMALL_BOUNDS))
def test_small_body_at_oversample_4_matches_a_finer_render(diameter: float) -> None:
    """A small half-lit body at oversample 4 matches the smooth body rendered 8 times finer.

    The body carries a relief of 1e-9 of its radius, so its relief is laid out on the
    detector grid while it should render as the smooth body.

    Parameters:
        diameter: The body's diameter in detector pixels.
    """
    brightness_bound, centroid_bound = _SMALL_BOUNDS[diameter]
    truth = _small_body(diameter, 32, relief=0.0)
    rendered = _small_body(diameter, 4, relief=1e-9)
    assert abs(float(rendered.sum()) / float(truth.sum()) - 1.0) < brightness_bound
    assert float(np.hypot(*(_centroid(rendered) - _centroid(truth)))) < centroid_bound


# ----------------------------------------------------------------------------
# Relief against shading wholly on the working grid
# ----------------------------------------------------------------------------

# Relief bodies, each placing its error where one part of the renderer handles it.
# The crescent's light depends on the tilt interpolated over its disc and terminator,
# the small body's on the band along its limb, and the rough body at low sun on its
# long cast shadows.  Each runs at oversample 4; the crescent also runs at
# oversample 1 with full anti-aliasing, the other working grid four times the
# detector's.
_RELIEF_CASES: dict[str, dict[str, Any]] = {
    'crescent': {
        'size': 128,
        'radius': 36.0,
        'sun': 90.0,
        'phase': 120.0,
        'rms': 0.02,
        'corr': 12.0,
        'seed': 5,
    },
    'crescent_anti_aliased': {
        'size': 128,
        'radius': 36.0,
        'sun': 90.0,
        'phase': 120.0,
        'rms': 0.02,
        'corr': 12.0,
        'seed': 5,
        'aa': True,
    },
    'small': {
        'size': 32,
        'radius': 8.0,
        'sun': 70.0,
        'phase': 60.0,
        'rms': 0.02,
        'corr': 12.0,
        'seed': 6,
    },
    'rough_low_sun': {
        'size': 200,
        'radius': 80.0,
        'sun': 57.0,
        'phase': 80.0,
        'rms': 0.05,
        'corr': 5.0,
        'seed': 3,
    },
}

# The light's center may sit this far, along each axis, from where shading wholly on
# the working grid puts it.  Measured: crescent 0.00012 px, small body 0.00008 px,
# rough body 0.00002 px.  The crescent reaches 0.0037 px with the tilt interpolated
# bilinearly and the small body 0.0028 px with a limb band of half a pixel.  The
# rough body reaches 0.21 px when its cast shadows are interpolated, and 0.0050 px
# when the cells around them are left out, where a shadow edge crosses a cell
# without reaching its center.  Shading them on the detector grid and upsampling
# put the light 0.013, 0.032, and 0.047 px off.
_RELIEF_CENTROID_BOUND_PX = 0.0005


def _relief_body(case: dict[str, Any]) -> NDArrayFloatType:
    """A relief case's body, averaged onto the detector grid."""
    oversample, anti_aliasing = (1, 1.0) if case.get('aa') else (4, 0.0)
    scale = oversample
    spec = EllipsoidBodySpec(
        semi_a_px=case['radius'] * scale,
        semi_b_px=case['radius'] * scale,
        semi_c_px=case['radius'] * scale,
        illumination_angle=math.radians(case['sun']),
        phase_angle=math.radians(case['phase']),
        oversample=oversample,
        anti_aliasing=anti_aliasing,
        limb_relief_rms=case['rms'],
        limb_relief_corr_deg=case['corr'],
        relief_seed=case['seed'],
    )
    size = case['size'] * scale
    center = (size / 2.0 + 0.3 * scale, size / 2.0 - 0.65 * scale)
    img = create_ellipsoid_body((size, size), center, spec)
    return _detector(img, scale)


@pytest.mark.parametrize('case', sorted(_RELIEF_CASES))
def test_relief_body_light_matches_working_grid_shading(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relief body's light sits where shading wholly on its working grid puts it.

    Parameters:
        case: The key into ``_RELIEF_CASES``.
        monkeypatch: Switches the reference render to shade everything on the
            working grid.
    """
    rendered = _relief_body(_RELIEF_CASES[case])
    monkeypatch.setattr(ellipsoid, '_lays_detail_on_detector_grid', lambda spec, field: False)
    reference = _relief_body(_RELIEF_CASES[case])
    error = float(np.abs(_centroid(rendered) - _centroid(reference)).max())
    assert error < _RELIEF_CENTROID_BOUND_PX


# ----------------------------------------------------------------------------
# Surface textures on the working grid
# ----------------------------------------------------------------------------


def test_albedo_spot_edge_is_sharp_on_the_working_grid() -> None:
    """At oversample 4 an albedo spot darkens whole working-grid pixels.

    The textured render divided by the smooth one is the spot's factor or 1 at every
    pixel, so no pixel of the working grid blends the two across the spot's edge.
    """
    size, oversample = 64, 4
    smooth = EllipsoidBodySpec(
        semi_a_px=24.0 * oversample,
        semi_b_px=24.0 * oversample,
        semi_c_px=24.0 * oversample,
        illumination_angle=0.6,
        phase_angle=0.4,
        oversample=oversample,
    )
    spotted = dataclasses.replace(
        smooth, albedo_texture=AlbedoTextureSpec(rms=0.0, spots=((20.0, 80.0, 25.0, 0.5),))
    )
    frame = (size * oversample, size * oversample)
    center = (size * oversample / 2.0 + 1.0, size * oversample / 2.0 - 3.0)
    plain = create_ellipsoid_body(frame, center, smooth)
    textured = create_ellipsoid_body(frame, center, spotted)
    lit = plain > 0.0
    ratio = np.round(textured[lit] / plain[lit], 12)
    assert set(np.unique(ratio).tolist()) == {0.5, 1.0}


# ----------------------------------------------------------------------------
# A body smaller than a pixel
# ----------------------------------------------------------------------------


def test_relief_body_smaller_than_a_pixel_renders_its_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relief body 0.4 px across, centered on a pixel corner, renders its light.

    Its silhouette covers parts of four detector pixels, none of whose centers it
    reaches; its light must land there as it does when shaded wholly on the working
    grid.
    """
    oversample = 4
    spec = EllipsoidBodySpec(
        semi_a_px=0.2 * oversample,
        semi_b_px=0.2 * oversample,
        semi_c_px=0.2 * oversample,
        phase_angle=0.3,
        oversample=oversample,
        limb_relief_rms=0.02,
        relief_seed=4,
    )
    size = 16 * oversample
    center = (size / 2.0, size / 2.0)
    rendered = render_ellipsoid_body((size, size), center, spec).intensity
    monkeypatch.setattr(ellipsoid, '_lays_detail_on_detector_grid', lambda spec, field: False)
    reference = render_ellipsoid_body((size, size), center, spec).intensity
    assert float(reference.sum()) > 0.0
    assert float(rendered.sum()) == pytest.approx(float(reference.sum()), rel=1e-9)


# ----------------------------------------------------------------------------
# Samples exactly on the smooth limb
# ----------------------------------------------------------------------------


def test_exact_limb_sample_renders_as_the_predicted_body() -> None:
    """A smooth body with a working sample exactly on its limb renders as predicted.

    A radius-2 sphere at (4.5, 4.5) drawn at anti-aliasing 0.75 puts a sample of its
    3 times supersample grid exactly on each of its four axis limbs.  The surface
    there faces sideways, so at 90 deg phase the navigator's predicted body leaves
    the sunward one dark, and so must the renderer.
    """
    spec = EllipsoidBodySpec(2.0, 2.0, 2.0, phase_angle=math.pi / 2.0, anti_aliasing=0.75)
    rendered = create_ellipsoid_body((9, 9), (4.5, 4.5), spec)
    projection = project_ellipsoid(
        (9, 9), (4.5, 4.5), 2.0, semi_b_px=2.0, semi_c_px=2.0, anti_aliasing=0.75
    )
    predicted = render_lambert_ellipsoid(
        projection, illumination_angle=0.0, phase_angle=math.pi / 2.0
    )
    np.testing.assert_array_equal(rendered, predicted)


def test_limbs_touching_at_one_sample_need_no_ranges() -> None:
    """Two smooth bodies whose limbs meet at one sample do not overlap.

    The sample both limbs pass through lies outside both bodies, so neither claims
    it and the scene renders without the ranges that order overlapping bodies.
    """
    scene = frame_scene(
        size=17,
        bodies=[
            disc_body(8.5, 8.5, 2.0, name='A', anti_aliasing=0.0),
            disc_body(8.5, 12.5, 2.0, name='C', anti_aliasing=0.0),
        ],
    )
    _, meta = render_scene(scene)
    assert not np.any(meta['body_masks'][0] & meta['body_masks'][1])


def test_limb_band_reaching_the_center_includes_it() -> None:
    """A limb band whose inner edge reaches the body center includes the center sample."""
    grid = project_ellipsoid((9, 9), (4.5, 4.5), 2.0, semi_b_px=2.0, semi_c_px=2.0)
    band = limb_band(grid, synthesize_relief_field(0.02, 15.0, 4), width_px=3.0)
    assert bool(band[4, 4])


# ----------------------------------------------------------------------------
# A frame one detector pixel tall or wide
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(('size', 'center'), [((1, 8), (0.5, 4.0)), ((8, 1), (4.0, 0.5))])
def test_cratered_body_renders_on_a_one_pixel_axis(
    size: tuple[int, int], center: tuple[float, float]
) -> None:
    """A cratered body in a frame one detector pixel tall or wide renders its light.

    Parameters:
        size: The frame dimensions (rows, columns).
        center: The body center (v, u).
    """
    spec = EllipsoidBodySpec(
        2.0, 2.0, 2.0, crater_fill=0.25, anti_aliasing=1.0, crater_seed=3, phase_angle=0.3
    )
    assert float(create_ellipsoid_body(size, center, spec).sum()) > 0.0
