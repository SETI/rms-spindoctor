"""A body larger than half the frame renders wherever its center lies.

Every body is drawn at the frame center and translated into place.  A body that
reaches past the frame from there is drawn on a canvas holding the part of it that
lands in the frame, so it must look exactly as the same body does in the matching
window of a frame large enough to hold it.  The tests place such bodies at centers
inside and outside a 64-pixel frame and compare the light, its center, and the
halo with that unclipped reference.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body

from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, render_ellipsoid_body
from spindoctor.sim.render.bodies.placement import body_canvas
from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.support.types import NDArrayFloatType

_SIZE = 64
# The reference frame is the test frame padded by this many pixels on every side, so
# both frames place the body by the same fractional translation.
_PAD = 160
# Rendered without noise or detector effects: only the body's own light.
_SHADING: dict[str, Any] = {'anti_aliasing': 1.0, 'phase_angle': 30.0, 'illumination_angle': 20.0}


def _render(
    body: dict[str, Any], *, pad: int = 0, oversample: int = 1
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render one body onto the test frame, or onto the padded reference frame.

    Parameters:
        body: The body entry, its center in test-frame pixels.
        pad: The padding of the frame on every side, which also moves the body's
            center by the same amount, so the test frame is the window at ``pad``.
        oversample: The render grid's oversampling factor.

    Returns:
        The test frame's window of the rendered image, and the body info.
    """
    size = (_SIZE + 2 * pad) * oversample
    img = np.zeros((size, size))
    shifted = {
        **body,
        'center_v': (body['center_v'] + pad) * oversample,
        'center_u': (body['center_u'] + pad) * oversample,
        'semi_a_px': body['semi_a_px'] * oversample,
        'semi_b_px': body['semi_b_px'] * oversample,
        'semi_c_px': body['semi_c_px'] * oversample,
    }
    _, info = render_single_body(
        img,
        body_spec_from_params(shifted, index=0),
        0.0,
        offset_u=0.0,
        body_streams_seed=11,
        oversample=oversample,
    )
    window = slice(pad * oversample, (pad + _SIZE) * oversample)
    return img[window, window], info


def _light_center(img: NDArrayFloatType) -> tuple[float, float]:
    """The light-weighted center (v, u) of an image, in array coordinates.

    Parameters:
        img: The image.

    Returns:
        The center of its light.
    """
    rows, cols = np.indices(img.shape)
    total = float(img.sum())
    return float((rows * img).sum()) / total, float((cols * img).sum()) / total


@pytest.mark.parametrize(
    ('center_v', 'diameter'),
    [(-25.0, 80.0), (-40.0, 120.0), (-60.0, 200.0), (100.0, 160.0)],
)
def test_large_body_off_center_matches_a_direct_render(center_v: float, diameter: float) -> None:
    """A large body at a whole-pixel position matches the body drawn directly there.

    These are the four bodies first reported cut off or missing.  At a whole-pixel
    position the translation copies pixels, so the frame must hold exactly the
    body drawn at its own center on a frame large enough to hold it.

    Parameters:
        center_v: The body center's row; the column is the frame center.
        diameter: The body's diameter in pixels.
    """
    radius = diameter / 2.0
    img, _ = _render(disc_body(center_v, 32.0, radius, **_SHADING))
    spec = EllipsoidBodySpec(
        radius,
        radius,
        radius,
        anti_aliasing=1.0,
        phase_angle=float(np.radians(30.0)),
        illumination_angle=float(np.radians(20.0)),
    )
    big = _SIZE + 2 * _PAD
    direct = render_ellipsoid_body((big, big), (center_v + _PAD, 32.0 + _PAD), spec).intensity
    expected = direct[_PAD : _PAD + _SIZE, _PAD : _PAD + _SIZE]
    assert float(expected.sum()) > 0.0
    assert np.array_equal(img, expected)


_FRACTIONAL_BODIES = [
    pytest.param(-45.3, 20.7, 60.0, id='center_above_the_frame'),
    pytest.param(100.6, -30.2, 80.0, id='center_below_and_left'),
    pytest.param(31.4, 90.85, 50.0, id='center_right_of_the_frame'),
    pytest.param(-20.25, -20.75, 70.0, id='limb_across_a_corner'),
    pytest.param(40.3, 25.6, 45.0, id='center_on_the_frame'),
]


@pytest.mark.parametrize(('center_v', 'center_u', 'radius'), _FRACTIONAL_BODIES)
def test_large_body_light_matches_the_unclipped_body(
    center_v: float, center_u: float, radius: float
) -> None:
    """A large body's light and its center in the frame match the unclipped body's.

    Parameters:
        center_v: The body center's row.
        center_u: The body center's column.
        radius: The body's radius in pixels.
    """
    body = disc_body(center_v, center_u, radius, **_SHADING)
    img, _ = _render(body)
    reference, _ = _render(body, pad=_PAD)
    assert float(reference.sum()) > 0.0
    assert float(img.sum()) == pytest.approx(float(reference.sum()), rel=1e-12)
    assert _light_center(img) == pytest.approx(_light_center(reference), abs=1e-9)


def test_large_relief_body_at_oversample_2_matches_the_unclipped_body() -> None:
    """A large relief body on an oversampled grid matches the unclipped body.

    Its relief tilt is interpolated by a spline whose ends reach into the canvas,
    so the frame matches the reference to the canvas margin's bound, not exactly.
    """
    body = disc_body(-30.4, 50.3, 60.0, limb_relief_rms=0.03, limb_relief_corr_deg=12.0)
    img, _ = _render(body, oversample=2)
    reference, _ = _render(body, pad=_PAD, oversample=2)
    assert float(reference.sum()) > 0.0
    assert float(np.abs(img - reference).max()) < 1e-6


@pytest.mark.parametrize(
    'keys',
    [
        pytest.param({'semi_a_px': 18.0, 'rotation_tilt': 60.0, 'rotation_z': 20.0}, id='tilt'),
        pytest.param({'limb_relief_rms': 0.04, 'limb_relief_corr_deg': 10.0}, id='relief'),
    ],
)
def test_body_reaching_past_the_frame_by_its_tilt_or_relief_matches_the_unclipped_body(
    keys: dict[str, Any],
) -> None:
    """A body whose tilt or relief alone carries it past the frame matches the unclipped body.

    Without them the body would fit in the frame drawn at the frame center.  The
    tilt stretches the outline along semi-axis a, and the relief lifts the limb.

    Parameters:
        keys: The body keys that carry it past the frame.
    """
    body = {**disc_body(50.3, 10.6, 31.0, **_SHADING), **keys}
    img, _ = _render(body)
    reference, _ = _render(body, pad=_PAD)
    assert float(reference.sum()) > 0.0
    assert float(np.abs(img - reference).max()) < 1e-6


def test_large_halo_matches_the_unclipped_halo() -> None:
    """A large haze body's halo in the frame matches the unclipped halo.

    A drawing cut at a frame-sized box would end the halo in a straight line half a
    frame from the body center, at column 13 here; the unclipped halo glows well
    past it.
    """
    body = disc_body(
        12.4,
        -18.6,
        24.0,
        atmosphere={'scale_height_px': 4.0, 'tau_ref': 1.5, 'g': 0.4},
        **_SHADING,
    )
    _, info = _render(body)
    _, reference_info = _render(body, pad=_PAD)
    window = slice(_PAD, _PAD + _SIZE)
    reference = reference_info['halo'].emission[window, window]
    assert float(reference[:, 14:].max()) > 0.0
    assert float(np.abs(info['halo'].emission - reference).max()) < 1e-12


def test_large_mesh_body_matches_the_unclipped_body() -> None:
    """A large mesh body placed off center matches the unclipped body.

    Its light and the center of its light in the frame match the same body in a
    frame large enough to hold it.
    """
    body = {
        **disc_body(-28.3, 70.6, 56.0, **_SHADING),
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': 0.3,
        'pose_euler_deg': [20.0, 35.0, -10.0],
    }
    img, _ = _render(body)
    reference, _ = _render(body, pad=_PAD)
    assert float(reference.sum()) > 0.0
    assert float(img.sum()) == pytest.approx(float(reference.sum()), rel=1e-12)
    assert _light_center(img) == pytest.approx(_light_center(reference), abs=1e-9)


def test_partly_drawn_body_keeps_its_craters_in_place() -> None:
    """A partly drawn cratered body moved by whole pixels moves with its craters.

    Its craters are laid out over the whole body, so moving the body moves the
    frame's content by the same whole pixels, craters included.
    """
    body = disc_body(-30.25, 40.5, 60.0, crater_fill=0.4, **_SHADING)
    moved = {**body, 'center_v': -28.25, 'center_u': 43.5}
    img, _ = _render(body)
    img_moved, _ = _render(moved)
    assert np.array_equal(img_moved[2:, 3:], img[:-2, :-3])


def test_body_that_misses_the_frame_paints_nothing() -> None:
    """A large body wholly outside the frame paints nothing and keeps its inventory."""
    img = np.zeros((_SIZE, _SIZE))
    mask, info = render_single_body(
        img,
        body_spec_from_params(disc_body(-200.0, 32.0, 50.0, name='far'), index=0),
        0.0,
        offset_u=0.0,
    )
    assert not mask.any()
    assert info['inventory']['v_max_unclipped'] == pytest.approx(-150.0)


def test_body_that_fits_is_drawn_on_the_frame() -> None:
    """A body that fits in the frame from its center is drawn on the frame itself."""
    canvas = body_canvas(
        (_SIZE, _SIZE), center_v=-10.0, center_u=70.0, reach_v=31.5, reach_u=20.0, oversample=1
    )
    assert canvas.on_frame


def test_partial_canvas_edges_lie_on_detector_pixels() -> None:
    """A partial canvas on an oversampled grid starts and ends on detector pixels."""
    canvas = body_canvas(
        (_SIZE * 4, _SIZE * 4),
        center_v=-61.3,
        center_u=150.7,
        reach_v=230.0,
        reach_u=230.0,
        oversample=4,
    )
    edges = [canvas.rows.start, canvas.rows.stop, canvas.cols.start, canvas.cols.stop]
    assert [edge % 4 for edge in edges] == [0, 0, 0, 0]
