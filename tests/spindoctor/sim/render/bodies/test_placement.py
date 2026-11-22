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

from spindoctor.sim.render.bodies import mesh as body_mesh
from spindoctor.sim.render.bodies import render as body_render
from spindoctor.sim.render.bodies.ellipsoid import EllipsoidBodySpec, render_ellipsoid_body
from spindoctor.sim.render.bodies.ellipsoid_grid import CraterPlane, crater_plane_site
from spindoctor.sim.render.bodies.placement import BodyCoverage, body_canvas
from spindoctor.sim.render.bodies.render import render_single_body
from spindoctor.sim.render.bodies.spec import body_spec_from_params
from spindoctor.sim.shared.ellipsoid import project_ellipsoid
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
    """A body carried past the frame only by tilt or relief matches the unclipped body.

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
    halo = info['halo']
    reference = reference_info['halo'].emission[window, window]
    assert float(reference[:, 14:].max()) > 0.0
    assert float(np.abs(halo.emission - reference).max()) < 1e-12
    reference_transmission = reference_info['halo'].transmission[window, window]
    assert float(np.abs(halo.transmission - reference_transmission).max()) < 1e-12
    differs = (halo.emission != 0.0) | (halo.transmission != 1.0)
    in_box = np.zeros_like(differs)
    in_box[halo.box_v, halo.box_u] = halo.mask
    assert not np.any(differs & ~in_box)


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


def test_lumpy_mesh_body_reaching_past_its_semi_axes_matches_the_unclipped_body() -> None:
    """A lumpy mesh body whose bumps alone carry it past the frame matches the reference.

    Its semi-axes would fit in the frame drawn at the frame center, but its farthest
    vertex lies 1.28 times the semi-axis from its center, past the frame edge.
    """
    body = {
        **disc_body(50.3, 12.6, 28.0, **_SHADING),
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': 0.6,
        'mesh_seed': 0,
        'pose_euler_deg': [20.0, 35.0, -10.0],
    }
    img, _ = _render(body)
    reference, _ = _render(body, pad=_PAD)
    assert float(reference.sum()) > 0.0
    assert float(np.abs(img - reference).max()) < 1e-12


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


def _refuse_to_draw(*args: Any, **kwargs: Any) -> NDArrayFloatType:
    """Stand in for a body renderer that must not be called.

    Parameters:
        *args: The renderer's positional arguments.
        **kwargs: The renderer's keyword arguments.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError('a body outside the frame was drawn')


@pytest.mark.parametrize(
    ('semi_a', 'semi_b', 'rotation_z', 'rotation_tilt', 'center'),
    [
        pytest.param(40.0, 40.0, 0.0, 0.0, (61.0, 57.0), id='circle_on_pixel_corners'),
        pytest.param(37.3, 21.9, 0.7, 0.5, (60.41, 57.83), id='rotated_and_tilted'),
        pytest.param(25.0, 24.0, np.pi / 2, 0.0, (61.5, 57.5), id='quarter_turn'),
    ],
)
def test_crater_plane_counts_every_pixel_inside_the_body(
    semi_a: float,
    semi_b: float,
    rotation_z: float,
    rotation_tilt: float,
    center: tuple[float, float],
) -> None:
    """The crater plane counts the pixels inside the body as testing each one does.

    The count tests only pixels near the limb of each row; the reference tests every
    pixel of the same plane with the projection the renderer uses.

    Parameters:
        semi_a: Semi-axis a in pixels.
        semi_b: Semi-axis b in pixels.
        rotation_z: In-plane rotation in radians.
        rotation_tilt: Tilt in radians.
        center: The body center on the plane.
    """
    shape = (122, 115)
    grid = project_ellipsoid(
        shape,
        center,
        semi_a,
        semi_b_px=semi_b,
        semi_c_px=semi_b,
        rotation_z=rotation_z,
        rotation_tilt=rotation_tilt,
    )
    _, count = crater_plane_site(
        CraterPlane(shape=shape, origin=(0, 0)), grid, cos_tilt=float(np.cos(rotation_tilt))
    )
    assert count == int(np.count_nonzero(grid.ellipse_dist_sq < 1.0))


def test_crater_plane_counts_a_pixel_on_the_limb_of_a_long_thin_outline() -> None:
    """A long, thin outline with one pixel inside it by the last bit counts that pixel.

    The outline is 13,600 times as long as it is wide, and pixel (2, 4) lies on its
    limb at a squared distance of 1 less one rounding step, so its row's span of
    pixels below 1 is a sliver that a discriminant canceling to rounding would lose.
    """
    shape = (5, 9)
    grid = project_ellipsoid(
        shape,
        (1054.3219104111986, -1481.041245408661),
        0.13414593030640126,
        semi_b_px=1820.2093722775085,
        semi_c_px=1.0,
        rotation_z=5.667084053647428,
    )
    _, count = crater_plane_site(CraterPlane(shape=shape, origin=(0, 0)), grid, cos_tilt=1.0)
    assert count == 1


@pytest.mark.parametrize(
    ('keys', 'patched'),
    [
        pytest.param({}, (body_render, '_render_partial_ellipsoid_cached'), id='ellipsoid'),
        pytest.param(
            {'shape_model': 'polyhedral_mesh'},
            (body_mesh, '_render_partial_mesh_cached'),
            id='mesh',
        ),
    ],
)
def test_body_that_misses_the_frame_is_never_drawn(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Any], patched: tuple[Any, str]
) -> None:
    """A large body wholly outside the frame is never drawn and keeps its inventory.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.
        keys: The body keys that choose its renderer.
        patched: The module and name of the renderer that must not be called.
    """
    monkeypatch.setattr(*patched, _refuse_to_draw)
    img = np.zeros((_SIZE, _SIZE))
    body = {**disc_body(-200.0, 32.0, 50.0, name='far'), **keys}
    mask, info = render_single_body(img, body_spec_from_params(body, index=0), 0.0, offset_u=0.0)
    assert not mask.any()
    assert info['inventory']['v_max_unclipped'] == pytest.approx(-150.0)


@pytest.mark.parametrize(
    ('reach', 'on_frame'),
    [pytest.param(31.0, True, id='a_pixel_inside'), pytest.param(31.5, False, id='closer')],
)
def test_body_is_drawn_on_the_frame_only_a_pixel_inside_its_edge(
    reach: float, on_frame: bool
) -> None:
    """A body is drawn on the frame only if it stays a pixel inside the frame edge.

    Parameters:
        reach: The body's reach along v from its center.
        on_frame: Whether the body is drawn on the frame.
    """
    canvas = body_canvas(
        (_SIZE, _SIZE), center_v=-10.0, center_u=70.0, reach_v=reach, reach_u=20.0, oversample=1
    )
    assert canvas.on_frame is on_frame


@pytest.mark.parametrize(
    'keys',
    [
        pytest.param({'semi_a_px': 32.0, 'semi_b_px': 32.0, 'semi_c_px': 32.0}, id='edge'),
        pytest.param({'semi_a_px': 31.4, 'semi_b_px': 31.4, 'semi_c_px': 31.4}, id='aa_pixel'),
    ],
)
def test_body_within_a_pixel_of_the_frame_edge_matches_the_unclipped_body(
    keys: dict[str, Any],
) -> None:
    """A body reaching within a pixel of the frame edge matches the unclipped body.

    Drawn on the frame, it would light the drawing's outermost pixels, part of whose
    light the move drops, so the frame would differ from the unclipped body there.

    Parameters:
        keys: The body's semi-axes.
    """
    body = {**disc_body(63.6, 63.2, 1.0, **_SHADING), **keys}
    img, _ = _render(body)
    reference, _ = _render(body, pad=_PAD)
    assert float(reference.sum()) > 0.0
    assert float(np.abs(img - reference).max()) < 1e-12


def test_offset_sweep_of_a_large_body_keeps_one_drawing() -> None:
    """A large body drawn at many positions keeps only its last drawing.

    Its canvas changes with every position, so keeping more would only fill memory.
    """
    body_render.clear_ellipsoid_shape_caches()
    for step in range(6):
        _render(disc_body(-40.0 + 1.3 * step, 32.4, 60.0, **_SHADING))
    assert body_render._render_partial_ellipsoid_cached.cache_info().currsize == 1


def test_large_body_drawn_twice_is_drawn_once() -> None:
    """A large body drawn for the composite and again for the layers is drawn once.

    The radiance stage draws every body of a scene that smears each layer apart
    twice, first for the composite and then for the layers, so the second pass must
    find the drawing cached.
    """
    body = disc_body(-40.0, 20.0, 60.0, name='a', **_SHADING)
    body_render.clear_ellipsoid_shape_caches()
    for _pass in ('composite', 'layers'):
        _render(body)
    assert body_render._render_partial_ellipsoid_cached.cache_info().hits == 1


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


def test_a_pixel_covered_to_the_last_bit_transmits_nothing() -> None:
    """A coverage a rounding step short of 1 lets nothing through from behind.

    Moving a wholly covered pixel by linear interpolation can leave its coverage one
    unit in the last place below 1.
    """
    coverage = BodyCoverage(
        box_v=slice(0, 1), box_u=slice(0, 2), coverage=np.array([[np.nextafter(1.0, 0.0), 0.25]])
    )
    assert coverage.transmission.tolist() == [[0.0, 0.75]]
