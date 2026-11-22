"""Coverage compositing in the scene-radiance stage.

When the near body's outline covers a fraction ``c`` of a limb pixel, the pixel holds
``c`` of the near body and ``1 - c`` of whatever lies behind it: a farther body, the
ring system, or the body's own haze glow.  A star behind the limb keeps the
same open fraction of its light.  The scenes here put the near body's limb
across the middle of one pixel, so the outline covers exactly half of it.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, ring_system_block, star_entry

from spindoctor.sim.render.radiance.compose import compose_scene_radiance
from spindoctor.sim.render.stages import SimFrame, new_sim_frame
from spindoctor.sim.scene.validate import validate_sim_params

_SIZE = 64
# The near body's right limb crosses the middle of pixel (32, 32), at u = 32.5.  Its
# center lies a whole number of pixels from the frame center, where a body is drawn
# before it is moved into place, so the move copies the drawing's coverage unchanged.
_LIMB_PIXEL = (32, 32)
_NEAR_RADIUS = 10.5
_NEAR_CENTER = (32.0, 32.5 - _NEAR_RADIUS)
# The fraction of the limb pixel the near body covers.
_HALF = 0.5


def _near_body(**keys: Any) -> dict[str, Any]:
    """The near body: lit from the observer, its right limb across the limb pixel."""
    return disc_body(
        *_NEAR_CENTER,
        _NEAR_RADIUS,
        name='NEAR',
        illumination_angle=0.0,
        phase_angle=0.0,
        range_km=1.0e6,
        **keys,
    )


def _far_body() -> dict[str, Any]:
    """A large, farther body lit from the observer, bright behind the limb pixel."""
    return disc_body(
        34.0, 34.0, 26.0, name='FAR', illumination_angle=0.0, phase_angle=0.0, range_km=2.0e6
    )


def _ring_system() -> dict[str, Any]:
    """A face-on ringlet behind the near body, its band over the limb pixel."""
    feature = {
        'name': 'R',
        'kind': 'ringlet',
        'tau': 1.5,
        'width_px': 8.0,
        'orbit': {'a_px': 8.0, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0},
    }
    ring_system = ring_system_block(32.5, 40.5, [feature])
    ring_system['range_km'] = 2.0e6
    ring_system['km_per_pixel'] = 1000.0
    return ring_system


def _compose(**blocks: Any) -> SimFrame:
    """Run the radiance stage on a noiseless frame holding ``blocks``."""
    scene = validate_sim_params(frame_scene(size=_SIZE, seed=3, exposure_sec=1.0, **blocks))
    frame = new_sim_frame(_SIZE, _SIZE)
    compose_scene_radiance(frame, params=scene)
    return frame


def test_a_half_covered_limb_pixel_shows_half_of_the_farther_body() -> None:
    """The limb pixel holds the near body's light plus half the far body's."""
    near = _compose(bodies=[_near_body()]).signal
    far = _compose(bodies=[_far_body()]).signal
    stacked = _compose(bodies=[_near_body(), _far_body()]).signal
    assert stacked[_LIMB_PIXEL] == pytest.approx(
        near[_LIMB_PIXEL] + _HALF * far[_LIMB_PIXEL], abs=1e-12
    )


def test_a_half_covered_limb_pixel_shows_half_of_the_ring_behind_it() -> None:
    """The limb pixel holds the near body's light plus half the ring's."""
    near = _compose(bodies=[_near_body()]).signal
    ring = _compose(ring_system=_ring_system()).signal
    stacked = _compose(bodies=[_near_body()], ring_system=_ring_system()).signal
    assert ring[_LIMB_PIXEL] > 0.0
    assert stacked[_LIMB_PIXEL] == pytest.approx(
        near[_LIMB_PIXEL] + _HALF * ring[_LIMB_PIXEL], abs=1e-12
    )


def test_a_star_just_outside_a_limb_loses_only_the_share_the_body_covers() -> None:
    """A star 0.9 px outside the limb keeps all its light but half its limb share.

    The star's bilinear share falls on the limb pixel and the open pixel beyond it,
    so the body covers half of the limb pixel's share and nothing else.
    """
    star = [star_entry(32.5, 33.4, 6.0)]
    alone = _compose(stars=star).point_e
    behind = _compose(stars=star, bodies=[_near_body()]).point_e
    expected = alone.copy()
    expected[_LIMB_PIXEL] *= 1.0 - _HALF
    assert alone[_LIMB_PIXEL] > 0.0
    np.testing.assert_allclose(behind, expected, rtol=1e-12, atol=0.0)


def test_a_mesh_limb_pixel_shows_half_of_the_farther_body() -> None:
    """A mesh body's limb pixel lets the far body through the half it leaves open.

    The unlumpy mesh's outline covers eight of the limb pixel's sixteen supersamples.
    """
    mesh = _near_body(shape_model='polyhedral_mesh', mesh_lumpiness=0.0)
    near = _compose(bodies=[mesh]).signal
    far = _compose(bodies=[_far_body()]).signal
    stacked = _compose(bodies=[mesh, _far_body()]).signal
    assert stacked[_LIMB_PIXEL] == pytest.approx(
        near[_LIMB_PIXEL] + _HALF * far[_LIMB_PIXEL], abs=1e-12
    )


def test_a_haze_limb_at_a_sub_pixel_position_shows_no_dark_seam() -> None:
    """Across a hazy limb placed off the pixel grid, no pixel dips below its neighbors.

    At high phase the forward-scattering glow just above the limb is bright, and so
    is the haze on the disc just inside it, so the pixel the limb crosses blends two
    bright values.  Drawing that pixel from the disc alone, without the glow behind
    its open part, leaves it darker than both its neighbors.

    The limb crosses the row at columns 11.7 and 51.7, so only the columns within 5 px
    of those are checked; the middle of the night side is black.
    """
    body = disc_body(
        32.3,
        31.7,
        20.0,
        name='HAZE',
        illumination_angle=90.0,
        phase_angle=155.0,
        atmosphere={'scale_height_px': 3.0, 'tau_ref': 3.0, 'g': 0.75},
    )
    row = _compose(bodies=[body]).signal[32]
    dips = [
        column
        for column in (*range(7, 17), *range(47, 57))
        if row[column] < 0.9 * min(row[column - 1], row[column + 1])
    ]
    assert dips == []


def test_the_bodies_and_rings_layers_sum_to_the_composite() -> None:
    """The per-class layers of a hazy body over a ring add up to the composite.

    A smear of the stars alone asks for the layers.  The ring lies behind the body,
    its limb pixels, and its glow, so every kind of step in the stack dims the ring.
    """
    hazy = _near_body(atmosphere={'scale_height_px': 2.0, 'tau_ref': 1.0, 'g': 0.4})
    frame = _compose(
        bodies=[hazy],
        ring_system=_ring_system(),
        optics={'smear': [{'dv_px': 2.0, 'du_px': 0.0, 'object_class': 'stars'}]},
    )
    layers = frame.truth['radiance_layers']
    np.testing.assert_allclose(layers['bodies'] + layers['rings'], frame.signal, atol=1e-12)
