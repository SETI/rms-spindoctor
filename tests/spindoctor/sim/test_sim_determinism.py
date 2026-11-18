"""Determinism guarantees for the simulator's randomized effects.

These tests pin the seed-derivation contract (process-stable, per-effect
independent sub-seeds), pin the names of the streams the detector and telemetry
stages seed for themselves, and assert that rendering the same scene twice produces
byte-identical pixels even after the render caches are cleared.
"""

from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import frame_scene, star_entry

from spindoctor.sim.render.detector import chain, resolve_detector_params
from spindoctor.sim.render.driver import clear_render_caches, render_scene
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.stage import apply_telemetry
from spindoctor.sim.seeds import derive_effect_seed, stable_param_seed

# Pinned values guard against an accidental switch to a salted hash (e.g. the
# built-in ``hash``): a salted derivation would not reproduce these constants.
_EXPECTED_NOISE_SEED = 2193400513
_EXPECTED_BACKGROUND_STARS_SEED = 2528068292
_EXPECTED_CRATERS_SEED = 4259976418
_EXPECTED_PARAM_SEED = 3733498255

_SEED_CEILING = 2**32


def _clear_render_caches() -> None:
    """Drop every render-path lru_cache so RNG paths re-run."""
    clear_render_caches()


def _scene() -> dict[str, Any]:
    """A scene exercising the background sky and crater RNG paths."""
    body = {
        'name': 'SIM-BODY-1',
        'center_v': 32.0,
        'center_u': 32.0,
        'semi_a_px': 15.0,
        'semi_b_px': 12.0,
        'semi_c_px': 12.0,
        'crater_fill': 0.3,
    }
    return frame_scene(
        instrument='generic', seed=42, sky_counts={'density_factor': 500.0}, bodies=[body]
    )


def test_derive_effect_seed_is_process_stable() -> None:
    """The noise sub-seed matches its pinned, salt-free value."""
    assert derive_effect_seed(42, 'noise') == _EXPECTED_NOISE_SEED


def test_derive_effect_seed_distinct_per_effect() -> None:
    """Different effect names derive different streams from one scene seed."""
    seeds = {
        derive_effect_seed(42, 'noise'),
        derive_effect_seed(42, 'background_stars'),
        derive_effect_seed(42, 'craters'),
    }
    assert len(seeds) == 3


def test_derive_effect_seed_distinct_per_scene_seed() -> None:
    """The same effect derives different streams for different scene seeds."""
    assert derive_effect_seed(42, 'noise') != derive_effect_seed(43, 'noise')


def test_derive_effect_seed_in_randomstate_range() -> None:
    """Derived seeds stay within the RandomState-accepted range."""
    assert 0 <= derive_effect_seed(42, 'noise') < _SEED_CEILING


def test_pinned_effect_seeds_match() -> None:
    """All three effect sub-seeds match their pinned values."""
    assert derive_effect_seed(42, 'background_stars') == _EXPECTED_BACKGROUND_STARS_SEED
    assert derive_effect_seed(42, 'craters') == _EXPECTED_CRATERS_SEED


def test_stable_param_seed_is_process_stable() -> None:
    """The geometry fallback seed matches its pinned, salt-free value."""
    assert stable_param_seed(1.0, 2.0, 3.0, (4.0, 5.0)) == _EXPECTED_PARAM_SEED


def test_stable_param_seed_in_randomstate_range() -> None:
    """The fallback seed stays within the RandomState-accepted range."""
    assert 0 <= stable_param_seed(1.0, 2.0, 3.0, (4.0, 5.0)) < _SEED_CEILING


def test_render_is_byte_identical_across_cache_clears() -> None:
    """Re-rendering a scene after clearing caches reproduces identical pixels."""
    _clear_render_caches()
    img_a, _ = render_scene(_scene())
    _clear_render_caches()
    img_b, _ = render_scene(_scene())
    assert np.array_equal(img_a, img_b)


def test_render_body_index_map_identical_across_cache_clears() -> None:
    """The body index map is also reproduced byte-for-byte."""
    _clear_render_caches()
    _, meta_a = render_scene(_scene())
    _clear_render_caches()
    _, meta_b = render_scene(_scene())
    assert np.array_equal(meta_a['body_index_map'], meta_b['body_index_map'])


def test_different_scene_seed_changes_pixels() -> None:
    """Changing only the scene seed changes the rendered noise field."""
    _clear_render_caches()
    img_a, _ = render_scene(_scene())
    other = _scene()
    other['random_seed'] = 7
    _clear_render_caches()
    img_b, _ = render_scene(other)
    assert not np.array_equal(img_a, img_b)


def _full_stack_scene() -> dict[str, Any]:
    """A scene exercising the whole optics + detector + instrument_defaults stack."""
    return frame_scene(
        size=96,
        seed=5,
        exposure_sec=1.0,
        artifacts={'instrument_defaults': True},
        optics={
            'psf': {'sigma_v': 0.6, 'sigma_u': 0.6, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
            'smear': [{'dv_px': 1.0, 'du_px': 0.5, 'object_class': 'all'}],
        },
        noise={
            'poisson': True,
            'read_noise_dn': 4.0,
            'cosmic_ray_events_per_pixel_per_sec': 0.001,
            'bloom_length': 2,
        },
        bodies=[
            {
                'name': 'B',
                'center_v': 48.0,
                'center_u': 48.0,
                'semi_a_px': 15.0,
                'semi_b_px': 13.0,
                'semi_c_px': 13.0,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
            }
        ],
        stars=[star_entry(20.0, 70.0, 5.0)],
    )


def test_full_stack_render_is_byte_identical_across_cache_clears() -> None:
    """The whole optics + detector + instrument_defaults stack is reproducible."""
    _clear_render_caches()
    img_a, _ = render_scene(_full_stack_scene())
    _clear_render_caches()
    img_b, _ = render_scene(_full_stack_scene())
    assert np.array_equal(img_a, img_b)


def _flat_frame(value: float) -> SimFrame:
    """A 48 x 48 detector-grid frame whose signal plane holds ``value`` everywhere."""
    return SimFrame(
        signal=np.full((48, 48), value, dtype=np.float64),
        point_e=np.zeros((48, 48), dtype=np.float64),
    )


def test_detector_noise_draws_from_the_detector_stream() -> None:
    """The detector stage's shot and read noise come from the stream ``detector``."""
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'random_seed': 11,
        'exposure_sec': 1.0,
        'noise': {'poisson': True, 'read_noise_dn': 5.0},
    }
    staged = _flat_frame(0.3)
    chain.apply_detector(staged, params=params)
    expected = _flat_frame(0.3)
    chain._apply_ccd(
        expected,
        resolve_detector_params(params),
        np.random.default_rng(derive_effect_seed(11, 'detector')),
    )
    assert np.array_equal(staged.signal, expected.signal)


def test_vidicon_noise_draws_from_the_detector_stream() -> None:
    """The vidicon chain's line and pixel read noise come from the stream ``detector``."""
    params: dict[str, Any] = {
        'instrument': 'vgiss',
        'random_seed': 11,
        'exposure_sec': 1.0,
        'noise': {'vidicon': {'read_noise_line_dn': 1.5, 'read_noise_pixel_dn': 3.0}},
    }
    staged = _flat_frame(0.3)
    chain.apply_detector(staged, params=params)
    dp = resolve_detector_params(params)
    expected = _flat_frame(0.3)
    chain._apply_vidicon(expected, dp, np.random.default_rng(derive_effect_seed(11, 'detector')))
    # vgiss renders calibrated I/F, so the stage ends with the calibration inverse.
    chain._apply_calibration_inverse(expected, dp)
    assert np.array_equal(staged.signal, expected.signal)


def test_missing_data_losses_draw_from_the_telemetry_stream() -> None:
    """The per-pixel missing-data losses come from the stream ``telemetry``."""
    rate = 0.3
    frame = _flat_frame(50.0)
    apply_telemetry(
        frame,
        params={'instrument': 'generic', 'random_seed': 11, 'noise': {'missing_data_rate': rate}},
    )
    draws = np.random.default_rng(derive_effect_seed(11, 'telemetry')).random(size=(48, 48))
    assert np.array_equal(frame.signal != 50.0, draws < rate)
