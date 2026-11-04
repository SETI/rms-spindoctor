"""Binned and summed readout modes: ``detector.binning`` and its catalog.

A scene selects a camera's readout mode with ``detector.binning``; each frame pixel then
sums a ``b`` by ``b`` block of native pixels.  The mode's published gain and read noise
replace the native ones, an extended source, the dark current, and the cosmic-ray rate
deposit ``b**2`` times the native charge, a binned pixel holds ``b**2`` native wells, a
star's total charge stays the same, the hot pixels are the native map summed into binned
pixels, the instrument PSF kernel shrinks by ``b``, and an uncataloged mode is a
validation error.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifacts_catalog import (
    PSF_KERNELS,
    binned_psf_kernel,
    binning_for_frame_size,
    cataloged_binnings,
)
from spindoctor.sim.forward.detector.noise_stages import detector_hot_pixel_map
from spindoctor.sim.forward.detector.params import DetectorParamError, resolve_detector_params
from spindoctor.sim.forward.optics import effective_psf
from spindoctor.sim.forward.scene_radiance import _nominal_read_noise
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError


def _scene(instrument: str, binning: int | None = None, **extra: Any) -> dict[str, Any]:
    """A minimal scene on ``instrument``, binned by ``binning`` when given."""
    scene: dict[str, Any] = {
        'instrument': instrument,
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'exposure_sec': 1.0,
        **extra,
    }
    if binning is not None:
        scene['detector'] = {'binning': binning}
    return scene


@pytest.mark.parametrize(
    ('instrument', 'expected'),
    [
        ('coiss_nac', (1, 2, 4)),
        ('coiss_calib_wac', (1, 2, 4)),
        ('gossi', (1, 2)),
        ('nhlorri', (1, 4)),
        ('vgiss', (1,)),
        (None, (1,)),
    ],
)
def test_each_camera_catalogs_its_readout_modes(
    instrument: str | None, expected: tuple[int, ...]
) -> None:
    """The cataloged binnings are each camera's published modes."""
    assert cataloged_binnings(instrument) == expected


@pytest.mark.parametrize(
    ('instrument', 'size', 'binning'),
    [
        ('coiss_calib_nac', 1024, 1),
        ('coiss_calib_nac', 512, 2),
        ('nhlorri', 256, 4),
        ('gossi', 800, 1),
        ('vgiss', 1000, 1),
    ],
)
def test_a_full_frame_names_its_binning(instrument: str, size: int, binning: int) -> None:
    """A full frame's size over the camera's native size is its binning."""
    assert binning_for_frame_size(instrument, (size, size)) == binning


def test_a_subframe_names_no_binning() -> None:
    """A frame that isn't a full frame in any cataloged mode fails loudly."""
    with pytest.raises(ValueError, match='not a full'):
        binning_for_frame_size('nhlorri', (512, 512))


def test_an_uncataloged_binning_fails_validation() -> None:
    """A LORRI scene asking for 2x2 binning is refused with the cataloged modes."""
    with pytest.raises(SimSceneValidationError, match=r'detector\.binning 2.*\[1, 4\]'):
        validate_sim_params(_scene('nhlorri', 2))


def test_a_cataloged_binning_validates() -> None:
    """A LORRI scene asking for 4x4 binning validates and keeps the key."""
    assert validate_sim_params(_scene('nhlorri', 4))['detector']['binning'] == 4


def test_the_resolver_refuses_an_uncataloged_binning() -> None:
    """A scene that bypasses validation still can't render an uncataloged mode."""
    with pytest.raises(DetectorParamError, match='no cataloged binning 4'):
        resolve_detector_params(_scene('gossi', 4))


@pytest.mark.parametrize('binning', [4.9, 4.0, True], ids=['fraction', 'float', 'bool'])
def test_the_resolver_refuses_a_binning_that_is_not_an_integer(binning: object) -> None:
    """A scene that bypasses validation can't truncate a fractional binning to a mode."""
    scene = _scene('nhlorri')
    scene['detector'] = {'binning': binning}
    with pytest.raises(DetectorParamError, match=r'detector\.binning must be an integer'):
        resolve_detector_params(scene)


def _vidicon_scene() -> dict[str, Any]:
    """A summed NAC scene that asks for the vidicon DN path."""
    scene = _scene('coiss_nac', 2)
    scene['detector']['detector_model'] = 'vidicon'
    return scene


def test_a_binned_vidicon_scene_fails_validation() -> None:
    """The vidicon DN path has no binned readout, so validation refuses the pair."""
    with pytest.raises(SimSceneValidationError, match=r'binning 2 .*vidicon'):
        validate_sim_params(_vidicon_scene())


def test_the_resolver_refuses_a_binned_vidicon_scene() -> None:
    """A scene that bypasses validation can't render a binned vidicon frame either."""
    with pytest.raises(DetectorParamError, match=r'vidicon.*binning 2'):
        resolve_detector_params(_vidicon_scene())


def test_a_binned_lorri_frame_reads_its_published_gain() -> None:
    """LORRI's 4x4 mode reads 19.4 e-/DN against the native 21."""
    assert resolve_detector_params(_scene('nhlorri', 4)).gain_e_per_dn == 19.4


def test_a_summed_galileo_frame_reads_its_published_read_noise() -> None:
    """Galileo's 2x2 summation mode reads 44 e- under the instrument chain."""
    scene = _scene('gossi', 2, artifacts={'instrument_defaults': True})
    assert resolve_detector_params(scene).read_noise_e == 44.0


def test_the_sky_cutoff_reads_the_summed_mode_read_noise() -> None:
    """The faint-sky cutoff uses the readout mode's noise floor: 44 e- for Galileo 2x2."""
    assert _nominal_read_noise('gossi', 'electrons', binning=2) == 44.0


def test_binning_scales_the_native_dark_current() -> None:
    """A 2x2 pixel collects four native pixels' dark charge."""
    extra: dict[str, Any] = {'artifacts': {'instrument_defaults': True}}
    native = resolve_detector_params(_scene('coiss_nac', **extra))
    binned = resolve_detector_params(_scene('coiss_nac', 2, **extra))
    assert binned.dark_current_e_per_sec == 4.0 * native.dark_current_e_per_sec


_LIT_BODY: dict[str, Any] = {
    'bodies': [
        {
            'name': 'B',
            'center_v': 32.0,
            'center_u': 32.0,
            'axis1': 40.0,
            'axis2': 40.0,
            'axis3': 40.0,
            'phase_angle': 0.0,
            'illumination_angle': 0.0,
        }
    ],
    'noise': {'poisson': False},
    # Short enough that four native pixels' charge stays below the ADC clip.
    'exposure_sec': 0.2,
}


def test_a_binned_pixel_collects_the_extended_signal_of_its_block() -> None:
    """A lit disc renders four times the native DN above bias at 2x2."""
    native, _meta = render_combined_model(_scene('coiss_nac', **_LIT_BODY))
    binned, _meta = render_combined_model(_scene('coiss_nac', 2, **_LIT_BODY))
    bias = float(resolve_detector_params(_scene('coiss_nac')).bias_dn)
    ratio = (float(binned[32, 32]) - bias) / (float(native[32, 32]) - bias)
    assert ratio == pytest.approx(4.0, rel=0.01)


def test_a_star_keeps_its_total_charge_when_binned() -> None:
    """Binning concentrates a star into fewer pixels without changing its total."""
    star = {'stars': [{'name': 'S', 'v': 32.5, 'u': 32.5, 'vmag': 6.0, 'psf_sigma': 1.5}]}
    totals = []
    for binning in (None, 2):
        image, _meta = render_combined_model(_scene('coiss_nac', binning, **star))
        totals.append(float(np.sum(image - np.median(image))))
    assert totals[1] == pytest.approx(totals[0], rel=0.02)


def test_the_binned_hot_pixel_map_is_the_native_map_summed() -> None:
    """Each binned hot pixel holds the summed amplitudes of its block's native hot pixels."""
    native_v, native_u, native_amps = detector_hot_pixel_map(
        (256, 256), fraction=0.002, detector_id='coiss_nac'
    )
    binned_v, binned_u, binned_amps = detector_hot_pixel_map(
        (64, 64), fraction=0.002, detector_id='coiss_nac', binning=4
    )
    expected = np.zeros((64, 64))
    np.add.at(expected, (native_v // 4, native_u // 4), native_amps)
    observed = np.zeros((64, 64))
    observed[binned_v, binned_u] = binned_amps
    np.testing.assert_allclose(observed, expected, rtol=1e-12)


def test_the_instrument_psf_shrinks_with_the_binning() -> None:
    """The instrument chain's PSF in 4x4 pixels is the native kernel over 4."""
    scene = _scene('nhlorri', 4, artifacts={'instrument_defaults': True})
    assert effective_psf(scene) == binned_psf_kernel(PSF_KERNELS['nhlorri'], 4)


def test_binned_psf_kernel_scales_only_the_radii() -> None:
    """The radii divide by the binning; the wing fraction and index don't change."""
    kernel = {'sigma_v': 1.2, 'sigma_u': 0.8, 'w': 0.1, 'r0': 3.0, 'n': 4.0}
    assert binned_psf_kernel(kernel, 2) == {
        'sigma_v': 0.6,
        'sigma_u': 0.4,
        'w': 0.1,
        'r0': 1.5,
        'n': 4.0,
    }


def test_a_calibrated_frame_keeps_its_i_over_f_when_summed() -> None:
    """Summing collects four times the charge and calibrates back to the same I/F."""
    native, _meta = render_combined_model(_scene('coiss_calib_nac', **_LIT_BODY))
    summed, _meta = render_combined_model(_scene('coiss_calib_nac', 2, **_LIT_BODY))
    assert float(summed[32, 32]) == pytest.approx(float(native[32, 32]), rel=1e-3)


def test_a_summed_pixel_saturates_at_its_blocks_well() -> None:
    """A summed pixel holds four native wells, so a bright disc keeps its I/F.

    At the full exposure four native pixels of the disc hold twice one native
    well; the lowest gain state keeps that charge below the ADC clip.  The
    tolerance is one DN of the native frame's 236 DN disc.
    """
    bright: dict[str, Any] = {**_LIT_BODY, 'exposure_sec': 1.0}
    native_scene = _scene('coiss_calib_nac', **bright)
    native_scene['detector'] = {'gain_state': 0}
    summed_scene = _scene('coiss_calib_nac', 2, **bright)
    summed_scene['detector']['gain_state'] = 0
    native, _meta = render_combined_model(native_scene)
    summed, _meta = render_combined_model(summed_scene)
    assert float(summed[32, 32]) == pytest.approx(float(native[32, 32]), rel=1.0 / 236.0)
