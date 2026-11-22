"""A body's normal albedo: how bright its surface is, and how the validator reads it.

The surface renders as the normal albedo times the photometric law, which is 1 where
the surface is lit and viewed head-on at zero phase.  The key scales an ellipsoid and
a mesh body alike, leaves a haze layer's glow alone, and refuses a value that is not
positive and one that is not finite.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene

from spindoctor.sim.render.bodies.render import clear_ellipsoid_shape_caches, render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params
from spindoctor.support.types import NDArrayFloatType

_SIZE = 64


def _ellipsoid(**keys: Any) -> dict[str, Any]:
    """A sphere 40 px across at 30 deg phase, lit from the upper right."""
    return disc_body(32.0, 32.0, 20.0, illumination_angle=40.0, phase_angle=30.0, **keys)


def _mesh(**keys: Any) -> dict[str, Any]:
    """A lumpy mesh body 40 px across at 30 deg phase."""
    return _ellipsoid(shape_model='polyhedral_mesh', mesh_lumpiness=0.3, **keys)


def _hazy(**keys: Any) -> dict[str, Any]:
    """A sphere with a haze layer whose glow reaches past the limb."""
    return _ellipsoid(atmosphere={'scale_height_px': 2.0, 'tau_ref': 0.5}, **keys)


def _drawn(body: dict[str, Any]) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """The body drawn alone into a black frame, with the body's render info."""
    clear_ellipsoid_shape_caches()
    image = np.zeros((_SIZE, _SIZE))
    _mask, info = render_single_body(
        image, body_spec_from_params(body, index=0), 0.0, shift_u=0.0, body_streams_seed=7
    )
    return image, info


def _refusal(value: Any) -> str:
    """The validator's message for a body whose normal albedo is ``value``."""
    scene = frame_scene(size=_SIZE, bodies=[_ellipsoid(normal_albedo=value)])
    with pytest.raises(SimSceneValidationError) as caught:
        validate_sim_params(scene, source='albedo')
    return str(caught.value)


@pytest.mark.parametrize(
    ('value', 'message'),
    [(0.0, 'must be a positive number'), (float('inf'), 'must be finite')],
    ids=['zero', 'infinite'],
)
def test_a_normal_albedo_that_is_not_a_positive_finite_number_is_refused(
    value: float, message: str
) -> None:
    """Zero and infinity are refused, each with its own reason."""
    assert f'bodies[0].normal_albedo {message}' in _refusal(value)


def test_leaving_the_key_out_draws_the_law_itself() -> None:
    """A body without the key draws the law's own value, 1 at zero phase, at its center.

    At zero phase the Lommel-Seeliger law is 1 everywhere on the disc, so the center
    pixel reads exactly the law.
    """
    body = disc_body(32.5, 32.5, 20.0, illumination_angle=0.0, phase_angle=0.0)
    image, _ = _drawn(body)
    assert image[32, 32] == pytest.approx(1.0, rel=1e-12)


def test_leaving_the_key_out_draws_a_white_surface() -> None:
    """A body without the key draws exactly as one whose normal albedo is 1."""
    absent, _ = _drawn(_ellipsoid())
    white, _ = _drawn(_ellipsoid(normal_albedo=1.0))
    np.testing.assert_array_equal(absent, white)


@pytest.mark.parametrize('body', [_ellipsoid, _mesh], ids=['ellipsoid', 'mesh'])
def test_the_normal_albedo_scales_the_surface(body: Any) -> None:
    """A normal albedo of 0.4 draws every pixel at 0.4 times its white brightness."""
    white, _ = _drawn(body())
    dark, _ = _drawn(body(normal_albedo=0.4))
    np.testing.assert_allclose(dark, 0.4 * white, rtol=1e-12, atol=0.0)


def test_the_haze_glow_keeps_its_brightness() -> None:
    """A darker surface leaves the glow of the haze above the limb as it was."""
    _, white = _drawn(_hazy())
    _, dark = _drawn(_hazy(normal_albedo=0.4))
    np.testing.assert_array_equal(dark['halo'].emission, white['halo'].emission)


def test_a_normal_albedo_above_one_brightens_the_surface() -> None:
    """A normal albedo of 1.5 draws every pixel at 1.5 times its white brightness."""
    white, _ = _drawn(_ellipsoid())
    bright, _ = _drawn(_ellipsoid(normal_albedo=1.5))
    np.testing.assert_allclose(bright, 1.5 * white, rtol=1e-12, atol=0.0)
