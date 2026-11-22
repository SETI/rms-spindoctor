"""The range of a rendered body's brightness: a black night side and no cap at 1.

A body's night side renders at the sky level, so its dark limb carries no step.  A
limb-brightened law keeps rising toward the sunlit limb instead of flattening at 1,
and a bright albedo spot brightens the frame until the detector saturates.  A mesh
body's night side is black but still hides what lies behind it.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise, star_entry

from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, render_ellipsoid_body
from spindoctor.sim.render.bodies.texture import AlbedoTextureSpec
from spindoctor.sim.render.driver import render_scene
from spindoctor.support.types import NDArrayFloatType

_LIMB_BRIGHTENED_LAWS = [('lommel_seeliger', 0.5), ('minnaert', 0.7)]


def _night_scene() -> dict[str, Any]:
    """A sphere lit from the top at 120 deg phase, so its lower half is night."""
    return frame_scene(
        size=128,
        noise=quiet_noise(),
        bodies=[disc_body(64.0, 64.0, 40.0, illumination_angle=0.0, phase_angle=120.0)],
    )


def test_night_side_renders_at_the_sky_level() -> None:
    """With noise off, every night-side pixel reads what the sky reads."""
    img, meta = render_scene(_night_scene())
    rows = np.arange(128)[:, np.newaxis] + 0.5
    night = meta['body_mask_map']['B'] & (rows > 70.0)
    assert np.array_equal(np.unique(img[night]), np.array([img[0, 0]]))


def test_dark_limb_carries_no_step() -> None:
    """Just inside the dark limb reads the same as just outside it."""
    img, meta = render_scene(_night_scene())
    v, u = np.mgrid[0:128, 0:128] + 0.5
    radius = np.hypot(v - 64.0, u - 64.0)
    lower = v > 70.0
    inside = meta['body_mask_map']['B'] & lower & (radius > 36.0)
    outside = ~meta['body_mask_map']['B'] & lower & (radius < 44.0)
    assert float(img[inside].mean()) == float(img[outside].mean())


def _law_render(law: str, minnaert_k: float) -> NDArrayFloatType:
    """A 40 px sphere lit from the top at 60 deg phase under ``law``."""
    spec = EllipsoidBodySpec(
        semi_a_px=40.0,
        semi_b_px=40.0,
        semi_c_px=40.0,
        phase_angle=math.radians(60.0),
        photometric_law=law,
        minnaert_k=minnaert_k,
        anti_aliasing=1.0,
    )
    return render_ellipsoid_body((100, 100), (50.0, 50.0), spec).intensity


@pytest.mark.parametrize(('law', 'minnaert_k'), _LIMB_BRIGHTENED_LAWS)
def test_limb_brightened_law_has_no_plateau(law: str, minnaert_k: float) -> None:
    """Almost no lit pixel sits at the disc's brightest value."""
    intensity = _law_render(law, minnaert_k)
    lit = intensity[intensity > 0.0]
    assert float(np.mean(lit >= 0.999 * lit.max())) < 0.01


@pytest.mark.parametrize(('law', 'minnaert_k'), _LIMB_BRIGHTENED_LAWS)
def test_limb_brightened_law_rises_toward_the_sunlit_limb(law: str, minnaert_k: float) -> None:
    """From the disc center to 0.9 radius sunward, every pixel is brighter than the last."""
    intensity = _law_render(law, minnaert_k)
    # Light from the top: sunward is up the center column, from row 49 to row 14.
    profile = intensity[14:50, 50][::-1]
    assert bool(np.all(np.diff(profile) > 0.0))


def test_bright_albedo_spot_multiplies_the_shading() -> None:
    """A spot three times the albedo renders three times as bright, past 1."""
    spot = AlbedoTextureSpec(rms=0.0, spots=((0.0, 90.0, 20.0, 3.0),), seed=0)
    spec = EllipsoidBodySpec(semi_a_px=20.0, semi_b_px=20.0, semi_c_px=20.0)
    plain = render_ellipsoid_body((64, 64), (32.0, 32.0), spec).intensity
    spotted = render_ellipsoid_body(
        (64, 64), (32.0, 32.0), dataclasses.replace(spec, albedo_texture=spot)
    ).intensity
    assert float(spotted[32, 32]) == pytest.approx(3.0 * float(plain[32, 32]), rel=1e-12)


def _spot_scene(instrument: str, albedo_factor: float | None) -> dict[str, Any]:
    """A fully lit sphere, with a spot of ``albedo_factor`` over its center if given."""
    body = disc_body(32.0, 32.0, 20.0, illumination_angle=0.0, phase_angle=0.0)
    if albedo_factor is not None:
        body['albedo_texture'] = {
            'rms': 0.0,
            'spots': [
                {
                    'lat_deg': 0.0,
                    'lon_deg': 90.0,
                    'radius_deg': 20.0,
                    'albedo_factor': albedo_factor,
                }
            ],
        }
    return frame_scene(instrument=instrument, size=64, noise=quiet_noise(), bodies=[body])


@pytest.mark.parametrize('instrument', ['coiss_nac', 'vgiss'])
def test_bright_spot_brightens_the_frame_past_the_unit_signal(instrument: str) -> None:
    """A spot of three times the albedo reads well above the plain disc.

    A signal of 1 fills half the well, so the plain center reads about half the
    detector's ceiling and the spot at least nearly the whole of it.
    """
    plain, _ = render_scene(_spot_scene(instrument, None))
    spotted, _ = render_scene(_spot_scene(instrument, 3.0))
    assert float(spotted[32, 32]) > 1.9 * float(plain[32, 32])


@pytest.mark.parametrize('instrument', ['coiss_nac', 'vgiss'])
def test_bright_spot_saturates_the_detector(instrument: str) -> None:
    """Two spot pixels of different brightness both read the detector's ceiling."""
    spotted, _ = render_scene(_spot_scene(instrument, 3.0))
    assert float(spotted[32, 34]) == float(spotted[32, 32])


def test_mesh_night_side_hides_the_star_behind_it() -> None:
    """A star behind a mesh body's black night side does not show through."""
    body = disc_body(
        32.0,
        32.0,
        20.0,
        illumination_angle=0.0,
        phase_angle=150.0,
        shape_model='polyhedral_mesh',
        mesh_lumpiness=0.0,
    )
    without_star, _ = render_scene(frame_scene(size=64, noise=quiet_noise(), bodies=[body]))
    with_star, _ = render_scene(
        frame_scene(
            size=64, noise=quiet_noise(), bodies=[body], stars=[star_entry(40.5, 32.5, 6.0)]
        )
    )
    assert float(with_star[40, 32]) == float(without_star[40, 32])
