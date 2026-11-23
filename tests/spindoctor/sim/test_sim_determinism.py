"""Determinism guarantees for the simulator's randomized effects.

These tests pin the seed-derivation contract (process-stable, per-effect
independent sub-seeds), pin the names of the streams the detector and telemetry
stages seed for themselves, check what a scene's ``noise_seed`` reseeds and what it
leaves alone, and assert that rendering the same scene twice produces byte-identical
pixels even after the render caches are cleared.
"""

from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise, star_entry, zero_bias

from spindoctor.sim.instruments.navigator_link import navigator_marker
from spindoctor.sim.render.detector import chain, resolve_detector_params
from spindoctor.sim.render.driver import clear_render_caches, render_scene
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.render.telemetry.stage import apply_telemetry
from spindoctor.sim.seeds import derive_effect_seed

# Pinned values guard against an accidental switch to a salted hash (e.g. the
# built-in ``hash``): a salted derivation would not reproduce these constants.
_EXPECTED_NOISE_SEED = 2193400513
_EXPECTED_BACKGROUND_STARS_SEED = 2528068292
_EXPECTED_CRATERS_SEED = 4259976418

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
        artifacts={'instrument_defaults': True, 'bloom': {'bloom_length': 2}},
        optics={
            'psf': {'sigma_v': 0.6, 'sigma_u': 0.6, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
            'smear': [{'dv_px': 1.0, 'du_px': 0.5, 'object_class': 'all'}],
        },
        noise={
            'poisson': True,
            'read_noise_dn': 4.0,
            'cosmic_ray_events_per_pixel_per_sec': 0.001,
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
        'instrument': 'vgiss_1_nac',
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
    chain._apply_calibration_inverse(expected, dp, marker=navigator_marker(params))
    assert np.array_equal(staged.signal, expected.signal)


def test_missing_data_losses_draw_from_the_telemetry_stream() -> None:
    """The per-pixel missing-data losses come from the stream ``telemetry``."""
    rate = 0.3
    frame = _flat_frame(50.0)
    apply_telemetry(
        frame,
        params={
            'instrument': 'generic',
            'random_seed': 11,
            'detector': {'missing_data_rate': rate},
        },
    )
    draws = np.random.default_rng(derive_effect_seed(11, 'telemetry')).random(size=(48, 48))
    assert np.array_equal(frame.signal != 50.0, draws < rate)


def _render(scene: dict[str, Any]) -> np.ndarray:
    """The rendered image of ``scene``, with the render caches cleared first."""
    _clear_render_caches()
    img, _ = render_scene(scene)
    return np.asarray(img)


def test_noise_seed_redraws_the_noise_and_nothing_before_the_detector() -> None:
    """A noise_seed changes a noisy render and leaves the same scene without noise alone.

    The scene's background stars and craters draw from random_seed, so with the
    detector noise turned off the frame is the same with or without the key.
    """
    noisy = {**_scene(), 'noise': {'poisson': True, 'read_noise_dn': 5.0}}
    quiet = {**_scene(), 'noise': quiet_noise(), 'detector': zero_bias()}
    assert not np.array_equal(_render(noisy), _render({**noisy, 'noise_seed': 23}))
    assert np.array_equal(_render(quiet), _render({**quiet, 'noise_seed': 23}))


def test_noise_seed_reseeds_the_detector_stream_under_its_own_name() -> None:
    """With a noise_seed the shot and read noise draw from ``detector`` under it."""
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'random_seed': 11,
        'noise_seed': 23,
        'exposure_sec': 1.0,
        'noise': {'poisson': True, 'read_noise_dn': 5.0},
    }
    staged = _flat_frame(0.3)
    chain.apply_detector(staged, params=params)
    expected = _flat_frame(0.3)
    chain._apply_ccd(
        expected,
        resolve_detector_params(params),
        np.random.default_rng(derive_effect_seed(23, 'detector')),
    )
    assert np.array_equal(staged.signal, expected.signal)


# The structured detector streams, by the name each derives its seed under.  A real
# camera draws the per-frame ones again for every frame, so they follow a scene's
# noise_seed.  The fixed-pattern ones belong to the camera and stay on random_seed.
_PER_FRAME_STREAMS = frozenset(
    {
        'cosmic_rays',
        'calibration_residual',
        'banding',
        'bias_structure',
        'bright_dark_pairs',
        'radiation_transients',
    }
)
_FIXED_PATTERN_STREAMS = frozenset(
    {'hot_pixels', 'fixed_pattern', 'fixed_pattern_dn', 'readout_dark_columns'}
)


def _follows_noise_seed(params: dict[str, Any], name: str) -> bool:
    """Return True when the detector stream ``name`` draws from the scene's noise_seed."""
    drawn = chain._stream_rng(resolve_detector_params(params), name).integers(0, 2**62, size=4)
    seed = derive_effect_seed(params['noise_seed'], f'detector/{name}')
    expected = np.random.default_rng(seed).integers(0, 2**62, size=4)
    return bool(np.array_equal(drawn, expected))


def test_noise_seed_reseeds_every_per_frame_stream_and_no_fixed_pattern() -> None:
    """Every per-frame detector stream draws from the noise_seed, and no other stream does."""
    params: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'random_seed': 11,
        'noise_seed': 23,
        'exposure_sec': 1.0,
        'noise': {'poisson': True, 'read_noise_dn': 5.0},
    }
    streams = _PER_FRAME_STREAMS | _FIXED_PATTERN_STREAMS
    following = {name for name in streams if _follows_noise_seed(params, name)}
    assert following == _PER_FRAME_STREAMS


def _camera_scene(**extra: Any) -> dict[str, Any]:
    """A noiseless disc frame carrying every fixed-pattern detector effect.

    The hot pixels are placed adversarially, so they draw from their stream rather
    than from the detector's fixed map.
    """
    return frame_scene(
        size=96,
        seed=3,
        noise=quiet_noise(),
        detector=zero_bias(),
        bodies=[disc_body(48.0, 48.0, 15.0)],
        artifacts={
            'adversarial': True,
            'hot_pixels': {'pixel_fraction': 0.002, 'amplitude_e': 4.0e4},
            'fixed_pattern': {
                'probability': 1.0,
                'prnu_rms': 0.02,
                'dust_donut_count': 2,
                'jail_bar_dn': 2.0,
            },
            'readout_dark': {'probability': 1.0, 'amplitude': 200.0, 'rbi_column_factor': 0.5},
        },
        **extra,
    )


def test_noise_seed_leaves_the_fixed_pattern_in_place() -> None:
    """The camera's hot pixels, fixed pattern, and warm columns stay where they are."""
    _clear_render_caches()
    img_a, truth_a = render_scene(_camera_scene())
    _clear_render_caches()
    img_b, truth_b = render_scene(_camera_scene(noise_seed=23))
    assert np.array_equal(img_a, img_b)
    assert truth_a['artifacts'] == truth_b['artifacts']


def test_noise_seed_leaves_each_mode_activation_in_place() -> None:
    """Which artifact modes a frame shows is drawn from random_seed, never the noise_seed.

    At random_seed 23 these modes activate differently than at 3, so a draw that
    followed the noise_seed would change the set.
    """
    modes = {
        name: {'probability': 0.5}
        for name in ('fixed_pattern', 'readout_dark', 'banding', 'bias_structure', 'bloom')
    }
    scene = frame_scene(seed=3, artifacts=modes)
    reseeded = resolve_detector_params({**scene, 'noise_seed': 23})
    assert reseeded.detector_modes == resolve_detector_params(scene).detector_modes
