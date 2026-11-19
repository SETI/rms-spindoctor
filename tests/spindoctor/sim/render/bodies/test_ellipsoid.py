"""Accuracy of the ellipsoid renderer's shading on its working grid.

Three properties: a setting at an effectively zero value renders the smooth body;
a small body at oversample 4 matches an 8 times finer render of the same body; and a
body with relief, craters, or a texture has its light where the same body shaded
wholly on its working grid has it, although its relief and craters are laid out on
the detector grid.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies import ellipsoid
from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, create_ellipsoid_body
from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.render.bodies.texture import AlbedoTextureSpec
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
# Surface detail against shading wholly on the working grid
# ----------------------------------------------------------------------------

_DETAIL_SIZE = 128
_DETAIL_CENTER = (64.3, 63.6)
_DETAIL_RADIUS = 36.0

_DETAIL_SURFACES: dict[str, dict[str, Any]] = {
    'relief': {'limb_relief_rms': 0.02, 'limb_relief_corr_deg': 12.0, 'relief_seed': 77},
    'craters': {'crater_fill': 0.3, 'crater_seed': 52},
    'texture': {
        'albedo_texture': AlbedoTextureSpec(
            rms=0.25,
            corr_px=12.0,
            spots=((40.0, 30.0, 12.0, 0.55), (10.0, 300.0, 9.0, 1.4)),
            seed=11,
        )
    },
}

# The light's center may sit this far, along each axis, from where shading wholly on
# the working grid puts it.  Measured on these bodies: relief 0.0011 px, and craters
# and texture 0 px, whose detail is the same on both; across three relief seeds at
# phases 30 and 70 deg the relief reached 0.0035 px.  Shading these bodies at
# detector resolution instead put the light 0.0046, 0.026, and 0.0079 px off.
_DETAIL_CENTROID_BOUND_PX = 0.004


def _detail_body(surface: str, *, oversample: int, anti_aliasing: float) -> NDArrayFloatType:
    """A detailed body at 50 deg phase, averaged onto the detector grid."""
    spec = EllipsoidBodySpec(
        semi_a_px=_DETAIL_RADIUS * oversample,
        semi_b_px=_DETAIL_RADIUS * oversample,
        semi_c_px=_DETAIL_RADIUS * oversample,
        illumination_angle=math.radians(40.0),
        phase_angle=math.radians(50.0),
        oversample=oversample,
        anti_aliasing=anti_aliasing,
        **_DETAIL_SURFACES[surface],
    )
    size = _DETAIL_SIZE * oversample
    center = (_DETAIL_CENTER[0] * oversample, _DETAIL_CENTER[1] * oversample)
    return _detector(create_ellipsoid_body((size, size), center, spec), oversample)


def _detail_centroid_error(
    surface: str, monkeypatch: pytest.MonkeyPatch, *, oversample: int, anti_aliasing: float
) -> float:
    """The largest axis offset between the rendered light's center and the reference's."""
    rendered = _detail_body(surface, oversample=oversample, anti_aliasing=anti_aliasing)
    with monkeypatch.context() as patch:
        patch.setattr(ellipsoid, '_lays_detail_on_detector_grid', lambda spec, field: False)
        reference = _detail_body(surface, oversample=oversample, anti_aliasing=anti_aliasing)
    return float(np.abs(_centroid(rendered) - _centroid(reference)).max())


@pytest.mark.parametrize('surface', sorted(_DETAIL_SURFACES))
def test_detailed_body_light_matches_working_grid_shading(
    surface: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The light's center matches the body shaded wholly on its working grid.

    Checked at oversample 4 and at oversample 1 with full anti-aliasing, the two
    working grids four times the detector's.

    Parameters:
        surface: The key into ``_DETAIL_SURFACES``.
        monkeypatch: Switches the reference render to shade everything on the
            working grid.
    """
    oversampled = _detail_centroid_error(surface, monkeypatch, oversample=4, anti_aliasing=0.0)
    anti_aliased = _detail_centroid_error(surface, monkeypatch, oversample=1, anti_aliasing=1.0)
    assert oversampled < _DETAIL_CENTROID_BOUND_PX
    assert anti_aliased < _DETAIL_CENTROID_BOUND_PX
