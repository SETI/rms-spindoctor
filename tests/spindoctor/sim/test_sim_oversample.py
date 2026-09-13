"""Oversampled radiance rendering and the no-optics identity property.

A scene with an active whole-scene PSF renders its radiance on an oversampled
grid and downsamples back to the detector grid; a scene with no optics block
renders at oversample 1.  These tests cover the oversample resolution, the
determinism of the no-optics path, and that the oversampled render preserves
body flux and position after the downsample.
"""

from typing import Any

import numpy as np

from spindoctor.sim.render import render_combined_model, resolve_oversample
from spindoctor.support.types import STAR_UV_DATUM_PX


def _body_scene(*, oversample: int | None) -> dict[str, Any]:
    """A single centered coiss body, optionally forcing an oversample factor."""
    scene: dict[str, Any] = {
        'size_v': 60,
        'size_u': 60,
        'random_seed': 3,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [
            {
                'name': 'B',
                'center_v': 30.0,
                'center_u': 30.0,
                'axis1': 24.0,
                'axis2': 24.0,
                'axis3': 24.0,
                'phase_angle': 0.0,
            }
        ],
    }
    if oversample is not None:
        scene['oversample'] = oversample
    return scene


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of an image."""
    vv, uu = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    total = float(image.sum())
    return float((image * vv).sum()) / total, float((image * uu).sum()) / total


def test_resolve_oversample_defaults_to_one_without_optics() -> None:
    """A scene with no optics block renders on the detector grid."""
    assert resolve_oversample({'size_v': 8, 'size_u': 8}) == 1


def test_resolve_oversample_is_four_when_psf_active() -> None:
    """An explicit PSF block turns on 4x oversampling by default."""
    scene = {'size_v': 8, 'size_u': 8, 'optics': {'psf': {'sigma_v': 0.6, 'sigma_u': 0.6}}}
    assert resolve_oversample(scene) == 4


def test_resolve_oversample_explicit_key_wins() -> None:
    """An explicit oversample key overrides the PSF-active default."""
    scene = {'oversample': 2, 'optics': {'psf': {'sigma_v': 0.6, 'sigma_u': 0.6}}}
    assert resolve_oversample(scene) == 2


def test_no_optics_render_is_deterministic() -> None:
    """Two renders of the same no-optics scene are bit-identical."""
    first, _ = render_combined_model(_body_scene(oversample=None))
    second, _ = render_combined_model(_body_scene(oversample=None))
    assert np.array_equal(first, second)


def test_oversampled_render_returns_detector_grid() -> None:
    """The downsample returns the image to the detector grid."""
    img, _ = render_combined_model(_body_scene(oversample=4))
    assert img.shape == (60, 60)


def test_oversample_conserves_body_flux() -> None:
    """Oversampling then averaging preserves the body's total signal."""
    plain, _ = render_combined_model(_body_scene(oversample=None))
    fine, _ = render_combined_model(_body_scene(oversample=4))
    assert abs(float(fine.sum()) - float(plain.sum())) < 0.02 * float(plain.sum())


def test_oversample_keeps_body_centered() -> None:
    """The oversampled body lands at the same detector centroid as os 1."""
    plain, _ = render_combined_model(_body_scene(oversample=None))
    fine, _ = render_combined_model(_body_scene(oversample=4))
    cv_plain, cu_plain = _centroid(plain)
    cv_fine, cu_fine = _centroid(fine)
    assert abs(cv_fine - cv_plain) < 0.2
    assert abs(cu_fine - cu_plain) < 0.2


def test_oversample_inventory_is_detector_scale() -> None:
    """The body inventory bbox is reported in detector pixels, not os pixels."""
    _, meta = render_combined_model(_body_scene(oversample=4))
    inv = meta['inventory']['B']
    assert abs(inv['v_pixel_size'] - 24.0) < 1.0


def test_oversample_star_records_are_detector_scale() -> None:
    """The truth star records return to detector units alongside star_info.

    At oversample 4 the radiance stage builds the records from os-scaled scene
    entries, so the downsample must rescale them (position, motion vector, PSF
    window) or they disagree with the detector-unit ``star_info`` entries by a
    factor of the oversample.  A record's position is oops uv, so it rescales
    about the half-pixel datum rather than straight through the divide, and it
    comes back a half pixel above the scene's pixel index.
    """
    scene: dict[str, Any] = {
        'size_v': 60,
        'size_u': 60,
        'random_seed': 3,
        'instrument': 'coiss_nac',
        'oversample': 4,
        'offset_v': 2.0,
        'offset_u': -1.0,
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'stars': [
            {
                'name': 'S',
                'v': 20.0,
                'u': 24.0,
                'vmag': 5.0,
                'move_v': 2.0,
                'move_u': -4.0,
                'psf_size': [11, 11],
            }
        ],
    }
    _, meta = render_combined_model(scene)
    star = meta['stars'][0]
    assert star.v == 20.0 + STAR_UV_DATUM_PX
    assert star.u == 24.0 + STAR_UV_DATUM_PX
    assert star.move_v == 2.0
    assert star.move_u == -4.0
    assert star.psf_size == (11, 11)
    # The records agree with the detector-unit hit-test entries: the rendered
    # centre is the pixel the record's catalog position lands on, plus the
    # planted offset.  ``center_v`` is a pixel index -- it says where the star
    # was drawn -- so the record's datum comes off before they are compared.
    info = meta['star_info'][0]
    assert abs(info['center_v'] - (star.v - STAR_UV_DATUM_PX + 2.0)) < 1e-9
    assert abs(info['center_u'] - (star.u - STAR_UV_DATUM_PX - 1.0)) < 1e-9
    # The hit-test half-window never shrinks below one detector pixel: with
    # no PSF the oversampled record floors at ``oversample`` subsamples, so
    # the downsample's divide lands exactly at the editor's 1-px click floor.
    assert info['psf_half_v'] == 1.0
    assert info['psf_half_u'] == 1.0


def test_oversample_defaulted_psf_size_survives_the_round_trip() -> None:
    """A star with no explicit psf_size keeps the default window at oversample 4.

    The radiance stage materializes and scales the record builder's default
    window alongside the explicit entries, so the downsample's divide returns
    it to the default detector-pixel size instead of (11 // os, 11 // os).
    """
    scene: dict[str, Any] = {
        'size_v': 60,
        'size_u': 60,
        'random_seed': 3,
        'instrument': 'coiss_nac',
        'oversample': 4,
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'stars': [{'name': 'S', 'v': 20.0, 'u': 24.0, 'vmag': 5.0}],
    }
    _, meta = render_combined_model(scene)
    assert meta['stars'][0].psf_size == (11, 11)


def test_optics_scene_renders_deterministically() -> None:
    """A PSF scene (oversample 4) is bit-identical across renders."""
    scene = _body_scene(oversample=None)
    scene['optics'] = {'psf': {'sigma_v': 1.0, 'sigma_u': 1.0, 'w': 0.02}}
    first, _ = render_combined_model(scene)
    second, _ = render_combined_model(scene)
    assert np.array_equal(first, second)
    assert first.shape == (60, 60)
