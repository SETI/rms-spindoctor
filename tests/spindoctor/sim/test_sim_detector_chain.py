"""End-to-end detector paths: calibrated I/F, vidicon, and instrument_defaults.

These render whole scenes to check the calibrated round-trip, the Voyager
vidicon DN path, and the ``instrument_defaults`` physical-chain opt-in against the
disabled floor (a scene with neither an artifacts nor a noise block).
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model


def _disc(instrument: str, *, size: int = 64, **extra: Any) -> dict[str, Any]:
    """A centered lit disc scene for the given instrument."""
    scene: dict[str, Any] = {
        'size_v': size,
        'size_u': size,
        'random_seed': 3,
        'instrument': instrument,
        'exposure_sec': 1.0,
        'bodies': [
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'axis1': size * 0.4,
                'axis2': size * 0.35,
                'axis3': size * 0.35,
            }
        ],
    }
    scene.update(extra)
    return scene


def test_calibrated_disc_round_trips_to_if_unity() -> None:
    """A noise-free calibrated disc round-trips its fully-lit peak to I/F ~1."""
    img, _ = render_combined_model(_disc('coiss_calib_nac'))
    assert abs(float(img.max()) - 1.0) < 0.02


def test_calibrated_path_carries_propagated_noise() -> None:
    """A calibrated scene with a noise block carries noise texture in I/F units."""
    clean, _ = render_combined_model(_disc('coiss_calib_nac'))
    noisy, _ = render_combined_model(
        _disc('coiss_calib_nac', noise={'poisson': True, 'read_noise_dn': 4.0})
    )
    assert not np.array_equal(clean, noisy)


def test_calibrated_sky_sits_near_zero_if() -> None:
    """Bias and dark are subtracted before the divide, so sky sits near I/F 0."""
    img, _ = render_combined_model(_disc('coiss_calib_nac'))
    corner = float(img[0, 0])
    assert abs(corner) < 0.05


def test_vidicon_floor_is_clean() -> None:
    """A vgiss scene with no noise block renders without vidicon noise (floor)."""
    plain, _ = render_combined_model(_disc('vgiss'))
    corner = plain[:4, :4]
    assert float(corner.std()) == 0.0


def test_vidicon_defaults_add_dn_domain_noise() -> None:
    """instrument_defaults turns on the vidicon DN-domain noise model."""
    plain, _ = render_combined_model(_disc('vgiss'))
    noisy, _ = render_combined_model(_disc('vgiss', artifacts={'instrument_defaults': True}))
    assert not np.array_equal(plain, noisy)


def test_vidicon_calibrated_if_is_exposure_invariant() -> None:
    """The vgiss calibrated path reports the same I/F at 1, 2, and 4 seconds.

    The vidicon forward mapping carries no exposure term, so the calibration
    inverse must not divide the exposure back out: a scene's I/F is a property
    of the geometry, not of how long the shutter was open.
    """
    peaks = []
    for exposure in (1.0, 2.0, 4.0):
        img, _ = render_combined_model(_disc('vgiss', exposure_sec=exposure))
        peaks.append(float(img.max()))
    assert abs(peaks[1] - peaks[0]) < 0.01
    assert abs(peaks[2] - peaks[0]) < 0.01
    assert peaks[0] > 0.9


def test_floor_scene_has_no_noise() -> None:
    """A scene with neither artifacts nor a noise block renders a clean DN frame."""
    img, _ = render_combined_model(_disc('coiss_nac'))
    corner = img[:8, :8]
    assert float(corner.std()) == 0.0


def test_instrument_defaults_activates_the_physical_chain() -> None:
    """instrument_defaults injects detector noise the floor scene does not carry."""
    floor, _ = render_combined_model(_disc('coiss_nac'))
    physical, _ = render_combined_model(_disc('coiss_nac', artifacts={'instrument_defaults': True}))
    assert not np.array_equal(floor, physical)
    # The physical chain lifts the sky off the flat bias with dark + banding.
    assert float(physical[:8, :8].std()) > 0.0


def test_instrument_defaults_defaults_oversample_when_psf_active() -> None:
    """instrument_defaults implies a PSF, which oversamples the radiance grid."""
    from spindoctor.sim.render import resolve_oversample

    assert resolve_oversample(_disc('coiss_nac', artifacts={'instrument_defaults': True})) == 4
    assert resolve_oversample(_disc('coiss_nac')) == 1


def test_gain_state_selection_changes_dn_scale() -> None:
    """Selecting a lower-gain state raises the DN a given signal reaches."""
    state2, _ = render_combined_model(_disc('coiss_nac', detector={'gain_state': 2}))
    state3, _ = render_combined_model(_disc('coiss_nac', detector={'gain_state': 3}))
    # Gain state 3 (~13 e-/DN) yields more DN per electron than state 2 (~30).
    assert float(state3.max()) > float(state2.max())


def test_instrument_defaults_turns_on_poisson_and_catalog_bloom() -> None:
    """The physical-chain opt-in resolves shot noise on and the catalog bloom."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(_disc('coiss_nac', artifacts={'instrument_defaults': True}))
    assert resolved.poisson is True
    assert resolved.bloom_length == 4
    # The Cassini entries retain a zero cosmic-ray rate (the chain's
    # exposure-scaling stage cannot represent the cohort's
    # exposure-independent transients; see the catalog comment).
    assert resolved.cosmic_ray_rate_per_sec == 0.0


def test_cosmic_ray_retained_zero_for_gossi() -> None:
    """The Galileo entry retains zero: its tuned hot-pixel fraction already
    carries the cohort's measured single-pixel incidence."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(_disc('gossi', artifacts={'instrument_defaults': True}))
    assert resolved.cosmic_ray_rate_per_sec == 0.0


def test_cosmic_ray_scene_value_activates_the_stage() -> None:
    """A scene noise value still turns the cosmic-ray stage on."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(
        _disc(
            'gossi',
            artifacts={'instrument_defaults': True},
            noise={'cosmic_ray_rate_per_sec': 2.0e-4},
        )
    )
    assert resolved.cosmic_ray_rate_per_sec == pytest.approx(2.0e-4)


def test_cosmic_ray_retained_zero_for_lorri() -> None:
    """The LORRI catalog entry explicitly retains a zero cosmic-ray rate."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(_disc('nhlorri', artifacts={'instrument_defaults': True}))
    assert resolved.cosmic_ray_rate_per_sec == 0.0


def test_floor_resolves_poisson_off_and_no_bloom() -> None:
    """Without artifacts or a noise block, shot noise and bloom stay disabled."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(_disc('coiss_nac'))
    assert resolved.poisson is False
    assert resolved.bloom_length == 0
    # No instrument_defaults: the catalog cosmic-ray rate does not activate.
    assert resolved.cosmic_ray_rate_per_sec == 0.0


def test_explicit_noise_override_beats_instrument_defaults() -> None:
    """noise: {poisson: false} wins over the instrument_defaults opt-in."""
    from spindoctor.sim.forward.detector.params import resolve_detector_params

    resolved = resolve_detector_params(
        _disc(
            'coiss_nac',
            artifacts={'instrument_defaults': True},
            noise={'poisson': False, 'bloom_length': 9},
        )
    )
    assert resolved.poisson is False
    assert resolved.bloom_length == 9


def _flat_defaults_scene(**noise_extra: Any) -> dict[str, Any]:
    """A generic instrument_defaults scene whose disc center is flat and bright.

    The generic detector has unit gain (electrons == DN) and a clean catalog
    chain (no dark, hot pixels, or banding), so the variance of the rendered
    frame in the flat center region isolates the shot term.
    """
    scene = _disc('generic', size=128, oversample=1, artifacts={'instrument_defaults': True})
    # A disc much larger than the frame makes the center region flat; the
    # signal clips at 1.0, which only flattens it further.
    scene['bodies'][0].update({'axis1': 400.0, 'axis2': 400.0, 'axis3': 400.0})
    if noise_extra:
        scene['noise'] = dict(noise_extra)
    return scene


def test_instrument_defaults_shot_noise_variance_tracks_mean_electrons() -> None:
    """An instrument_defaults bright flat region has variance ~ mean electrons."""
    noisy, _ = render_combined_model(_flat_defaults_scene())
    clean, _ = render_combined_model(_flat_defaults_scene(poisson=False))
    region = np.s_[32:96, 32:96]
    diff = noisy[region] - clean[region]
    bias_dn = 20.0  # generic catalog bias, present in both renders
    mean_electrons = float(clean[region].mean()) - bias_dn
    ratio = float(diff.var()) / mean_electrons
    assert mean_electrons > 500.0
    assert 0.85 < ratio < 1.15


def test_explicit_poisson_false_render_is_shot_noise_free() -> None:
    """The explicit override renders a flat region with no shot variance."""
    scene_a = _flat_defaults_scene(poisson=False)
    scene_b = _flat_defaults_scene(poisson=False)
    scene_b['random_seed'] = 4
    clean_a, _ = render_combined_model(scene_a)
    clean_b, _ = render_combined_model(scene_b)
    # Differencing two seeds removes the deterministic disc shading; only the
    # generic read noise (1 e-) remains, far below the ~45 e- shot sigma the
    # Poisson term would add at this signal level.
    diff = clean_a[32:96, 32:96] - clean_b[32:96, 32:96]
    assert float(diff.std()) < 5.0
