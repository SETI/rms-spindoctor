"""Light from beyond the frame edge: the margin the radiance and optics stages share.

The optics stage spreads light sideways, so a frame's edge pixels receive light from
beyond the frame and lose light past it.  The render draws the scene across a margin
around the frame, runs the optics there too, and cuts the frame out afterward.  These
tests check how wide the margin is, that a uniform field stays uniform to the frame
edge under the PSF and under smear, that a frame matches the same part of a larger
frame showing the same scene, that the distortion's margin is as good as a wider one,
that stray light and the truth stay the frame's, and that the margin leaves the
frame's interior alone.
"""

import copy
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, ring_system_block, star_entry

from spindoctor.sim.instruments.resolvers import resolve_psf_truncation_px
from spindoctor.sim.render.driver import clear_render_caches, render_scene, resolve_oversample
from spindoctor.sim.render.optics.stage import optics_margin_px
from spindoctor.sim.render.pipeline import STAGES
from spindoctor.sim.render.stages import SimFrame, new_sim_frame

# The Cassini narrow-angle camera's catalog kernel, in detector pixels.
_NAC_PSF: dict[str, float] = {'sigma_v': 0.7, 'sigma_u': 0.7, 'w': 0.12, 'r0': 3.0, 'n': 4.0}

# A flat sky floor and no sky stars: a field of one value everywhere.
_FLAT_SKY: dict[str, float] = {'density_factor': 0.0, 'diffuse_e_per_px': 100.0}

# How far the larger reference frames extend past the frame under test, in pixels.
_PAD = 6


def _focal_plane(scene: dict[str, Any], *, margin: bool = True, extra_px: int = 0) -> SimFrame:
    """Run a scene through the radiance, optics, and downsample stages.

    Parameters:
        scene: The scene.
        margin: Whether to give the frame the margin the render gives it; False
            renders with no margin, so the optics treat the frame edge as dark.
        extra_px: Detector pixels added to the margin.

    Returns:
        The frame on the detector grid, before the detector stage.
    """
    clear_render_caches()
    oversample = resolve_oversample(scene)
    margin_px = optics_margin_px(scene, oversample=oversample) if margin else 0
    frame = new_sim_frame(
        int(scene['size_v']),
        int(scene['size_u']),
        oversample=oversample,
        margin_px=margin_px + extra_px,
    )
    for stage in STAGES[:3]:
        stage(frame, params=scene)
    return frame


def _larger_frame(scene: dict[str, Any]) -> dict[str, Any]:
    """The same scene on a frame ``_PAD`` pixels larger on each side.

    Parameters:
        scene: A scene whose bodies and stars all state their positions.

    Returns:
        The scene with the frame enlarged and every position moved with it.
    """
    larger = copy.deepcopy(scene)
    larger['size_v'] = scene['size_v'] + 2 * _PAD
    larger['size_u'] = scene['size_u'] + 2 * _PAD
    for body in larger.get('bodies', []):
        body['center_v'] += _PAD
        body['center_u'] += _PAD
    for star in larger.get('stars', []):
        star['v'] += _PAD
        star['u'] += _PAD
    ring_system = larger.get('ring_system')
    if ring_system is not None:
        ring_system['geometry']['center_v'] += _PAD
        ring_system['geometry']['center_u'] += _PAD
    return larger


def _middle(plane: np.ndarray) -> np.ndarray:
    """The part of a larger frame's plane that the frame under test shows."""
    return plane[_PAD:-_PAD, _PAD:-_PAD]


def _ringlet(radius: float) -> dict[str, Any]:
    """A 4 px wide ringlet of optical depth 1 at ``radius`` px from the ring center."""
    return {
        'name': 'RINGLET',
        'kind': 'ringlet',
        'tau': 1.0,
        'width_px': 4.0,
        'orbit': {'a_px': radius},
    }


def _largest_relative_difference(plane: np.ndarray, reference: np.ndarray) -> float:
    """The largest difference between two planes, over the reference's peak."""
    return float(np.max(np.abs(plane - reference)) / np.max(np.abs(reference)))


def test_margin_is_zero_when_no_optics_move_light() -> None:
    """A scene without smear, distortion, a PSF, or ghosts renders with no margin."""
    assert optics_margin_px(frame_scene(), oversample=1) == 0


def test_margin_covers_the_psf_half_width() -> None:
    """A PSF-only scene's margin is the kernel's half-width in detector pixels."""
    scene = frame_scene(optics={'psf': _NAC_PSF})
    assert optics_margin_px(scene, oversample=4) == resolve_psf_truncation_px('coiss_nac')


def test_uniform_field_stays_uniform_to_the_edge_under_the_psf() -> None:
    """The catalog narrow-angle PSF leaves a flat field flat to the edges and corners."""
    frame = _focal_plane(frame_scene(size=16, optics={'psf': _NAC_PSF}, sky_counts=_FLAT_SKY))
    assert float(np.max(np.abs(frame.point_e / 100.0 - 1.0))) < 1e-12


def test_uniform_field_stays_uniform_to_the_edge_under_smear() -> None:
    """A 12 px smear leaves a flat field flat in the edge rows and corners."""
    smear = [{'dv_px': 12.0, 'du_px': 0.0, 'object_class': 'all'}]
    frame = _focal_plane(frame_scene(size=16, optics={'smear': smear}, sky_counts=_FLAT_SKY))
    assert float(np.max(np.abs(frame.point_e / 100.0 - 1.0))) < 1e-12


def test_distortion_margin_is_as_good_as_a_wider_one() -> None:
    """Under a radial distortion, a frame-filling planet's edges need no wider margin.

    A larger frame would have a different distortion, so the reference is the same
    frame rendered with a margin 16 pixels wider.
    """
    distortion = {'k1': 0.01, 'k2': 0.0, 'nonradial_rms_px': 0.0}
    planet = disc_body(8.0, 8.0, 30.0, name='PLANET', phase_angle=60.0)
    scene = frame_scene(size=16, optics={'distortion': distortion}, bodies=[planet])
    frame = _focal_plane(scene)
    reference = _focal_plane(scene, extra_px=16)
    assert _largest_relative_difference(frame.signal, reference.signal) < 1e-12


def test_star_just_outside_the_frame_lights_the_edge_as_a_larger_frame_shows() -> None:
    """A star centered 0.2 px above the frame spreads its light into the edge rows."""
    scene = frame_scene(size=16, optics={'psf': _NAC_PSF}, stars=[star_entry(-0.2, 8.3, 6.0)])
    frame = _focal_plane(scene)
    reference = _focal_plane(_larger_frame(scene))
    assert _largest_relative_difference(frame.point_e, _middle(reference.point_e)) < 1e-12


def test_frame_filling_planet_edges_match_a_larger_frame() -> None:
    """A planet overfilling the frame has the edge pixels a larger frame shows there."""
    planet = disc_body(8.0, 8.0, 60.0, name='PLANET', phase_angle=40.0)
    scene = frame_scene(size=16, optics={'psf': _NAC_PSF}, bodies=[planet])
    frame = _focal_plane(scene)
    reference = _focal_plane(_larger_frame(scene))
    assert _largest_relative_difference(frame.signal, _middle(reference.signal)) < 1e-12


def test_ghost_of_a_body_beyond_the_frame_appears_in_it() -> None:
    """A defocused ghost spreads a body above the frame into it, as a larger frame shows.

    The ghost's offset leaves the body's copy above the frame, so only the defocus
    blur carries its light in.
    """
    ghosts = [{'amplitude': 0.1, 'dv_px': 6.0, 'du_px': 0.0, 'defocus_sigma': 2.0}]
    body = disc_body(-12.0, 8.0, 3.0, name='MOON')
    scene = frame_scene(size=16, optics={'ghosts': ghosts}, bodies=[body])
    frame = _focal_plane(scene)
    reference = _focal_plane(_larger_frame(scene))
    assert _largest_relative_difference(frame.signal, _middle(reference.signal)) < 1e-12


def test_margin_leaves_the_frame_interior_unchanged() -> None:
    """Farther than the optics reach from the edge, the margin changes no pixel."""
    scene = frame_scene(
        size=100,
        optics={'psf': _NAC_PSF},
        oversample=1,
        bodies=[disc_body(44.0, 44.0, 6.0, name='MOON', range_km=1.0e6)],
        stars=[star_entry(60.3, 58.7, 6.0)],
        ring_system=ring_system_block(56.0, 56.0, [_ringlet(8.0)], range_km=2.0e6),
    )
    reach = optics_margin_px(scene, oversample=1)
    interior = (slice(reach, -reach), slice(reach, -reach))
    with_margin = _focal_plane(scene)
    without_margin = _focal_plane(scene, margin=False)
    assert (
        _largest_relative_difference(with_margin.signal[interior], without_margin.signal[interior])
        < 1e-12
    )
    assert (
        _largest_relative_difference(
            with_margin.point_e[interior], without_margin.point_e[interior]
        )
        < 1e-12
    )


def test_margin_leaves_out_the_distortion_in_the_preview() -> None:
    """The editor's fast preview skips the distortion, so its margin is the PSF's alone."""
    distortion = {'k1': 0.01, 'k2': 0.0, 'nonradial_rms_px': 0.0}
    scene = frame_scene(optics={'psf': _NAC_PSF, 'distortion': distortion})
    margin = optics_margin_px(scene, oversample=1, preview=True)
    assert margin == resolve_psf_truncation_px('coiss_nac')


def test_distortion_margin_covers_the_psf_reach() -> None:
    """Under a PSF, the distortion's reach is measured where the PSF reads.

    The warp moves light farther away from the optical center, so a margin sized by
    the frame's own corners is too narrow for the pixels the PSF reads past them.  The
    planet covers the whole margin, so light is missing wherever the margin is short.
    """
    distortion = {'k1': 0.01, 'k2': 0.0, 'nonradial_rms_px': 0.0}
    planet = disc_body(8.0, 8.0, 80.0, name='PLANET', phase_angle=60.0)
    scene = frame_scene(
        size=16, optics={'psf': _NAC_PSF, 'distortion': distortion}, bodies=[planet]
    )
    frame = _focal_plane(scene)
    reference = _focal_plane(scene, extra_px=24)
    assert _largest_relative_difference(frame.signal, reference.signal) < 1e-12


def test_ring_crossing_the_edge_matches_a_larger_frame() -> None:
    """A ringlet crossing the frame edge has the edge pixels a larger frame shows there."""
    ringlet = _ringlet(9.0)
    scene = frame_scene(
        size=16, optics={'psf': _NAC_PSF}, ring_system=ring_system_block(8.0, 8.0, [ringlet])
    )
    frame = _focal_plane(scene)
    reference = _focal_plane(_larger_frame(scene))
    assert _largest_relative_difference(frame.signal, _middle(reference.signal)) < 1e-9


def test_rolled_planet_edges_match_a_larger_rolled_frame() -> None:
    """A rolled frame-filling planet has the edge pixels a larger rolled frame shows."""
    planet = disc_body(6.0, 10.0, 40.0, name='PLANET', phase_angle=40.0, illumination_angle=30.0)
    scene = frame_scene(
        size=16, optics={'psf': _NAC_PSF}, bodies=[planet], offset_rotation_deg=30.0
    )
    frame = _focal_plane(scene)
    reference = _focal_plane(_larger_frame(scene))
    assert _largest_relative_difference(frame.signal, _middle(reference.signal)) < 1e-12


def test_rolled_smear_keeps_a_uniform_field_uniform_to_the_edge() -> None:
    """A smear turned by the roll is as wide as the margin it gets.

    Turned 45 degrees, a (10, 10) px drift lies along one axis and is 14 px long.
    """
    smear = [{'dv_px': 10.0, 'du_px': 10.0, 'object_class': 'all'}]
    scene = frame_scene(
        size=16, optics={'smear': smear}, sky_counts=_FLAT_SKY, offset_rotation_deg=45.0
    )
    frame = _focal_plane(scene)
    assert float(np.max(np.abs(frame.point_e / 100.0 - 1.0))) < 1e-12


def test_stray_light_is_added_to_the_frame_alone() -> None:
    """A frame with a margin gets the stray-light ramp a frame without one gets."""
    stray = {'amplitude': 0.1, 'direction_deg': 30.0, 'model': 'linear'}
    scene = frame_scene(size=16, optics={'psf': _NAC_PSF, 'stray_light': stray})
    frame = _focal_plane(scene)
    reference = _focal_plane(scene, margin=False)
    assert _largest_relative_difference(frame.signal, reference.signal) < 1e-12


@pytest.mark.parametrize('key', ['body_masks', 'ring_masks', 'body_index_map', 'body_mask_map'])
def test_truth_arrays_cover_the_frame_alone(key: str) -> None:
    """Each per-pixel truth array has the frame's shape, not the margin's."""
    ringlet = _ringlet(9.0)
    scene = frame_scene(
        size=16,
        optics={'psf': _NAC_PSF},
        bodies=[disc_body(8.0, 8.0, 3.0, name='MOON', range_km=1.0e6)],
        ring_system=ring_system_block(8.0, 8.0, [ringlet], range_km=2.0e6),
    )
    _, meta = render_scene(scene)
    value = meta[key]
    arrays = list(value.values()) if isinstance(value, dict) else value
    shapes = {np.shape(array) for array in (arrays if isinstance(arrays, list) else [arrays])}
    assert shapes == {(16, 16)}
