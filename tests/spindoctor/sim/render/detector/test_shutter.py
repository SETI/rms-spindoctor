"""The shutter shading and the frame-transfer smear."""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, disc_scene, frame_scene, quiet_noise

from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.detector.shutter import apply_frame_transfer_smear, apply_shutter_shading
from spindoctor.sim.render.driver import render_scene
from spindoctor.support.types import NDArrayFloatType

_SIZE = 64


def test_shutter_shading_scales_by_line() -> None:
    """The shutter shading scales line 0 by 1 and the last line by the ratio."""
    electrons = np.full((_SIZE, _SIZE), 10.0, dtype=np.float64)
    apply_shutter_shading(electrons, top_factor=1.5, bottom_factor=1.05)
    assert float(electrons[0].mean()) == pytest.approx(10.0)
    assert float(electrons[-1].mean()) == pytest.approx(10.0 * 1.05 / 1.5)


def test_shutter_shading_leaves_the_dark_current_alone() -> None:
    """The shutter shades the light, and dark charge builds up whether it is open or not."""
    scene = frame_scene(
        instrument='gossi',
        seed=3,
        exposure_sec=1.0,
        noise={**quiet_noise(), 'dark_current_e_per_sec': 2.0e4},
        artifacts={'shutter_shading': {'probability': 1.0}},
    )
    img, _ = render_scene(scene)
    assert float(np.ptp(img)) == 0.0


# Quiet noise for each detector chain, so a rendered frame holds the light alone.
_QUIET_CCD = quiet_noise()
_QUIET_VIDICON = {'vidicon': {'read_noise_line_dn': 0.0, 'read_noise_pixel_dn': 0.0}}

# The shutter's two factors, which the vidicon camera's entry does not supply.
_SHUTTER = {'probability': 1.0, 'top_factor': 1.5, 'bottom_factor': 1.05}


def _lit_frame(
    instrument: str, noise: dict[str, Any], artifacts: dict[str, Any]
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """A quiet 64 by 64 frame filled by a lit body, rendered on ``instrument``."""
    scene = frame_scene(
        instrument=instrument,
        size=64,
        exposure_sec=0.15,
        bodies=[disc_body(32, 32, 300.0)],
        noise=noise,
        detector={'quantization': 'exact'},
        artifacts=artifacts,
    )
    sky = resolve_detector_params(scene).calibrated_sky_if if instrument == 'vgiss' else 0.0
    image, truth = render_scene(scene)
    return image - sky, truth


@pytest.mark.parametrize(
    ('instrument', 'noise'),
    [('gossi', _QUIET_CCD), ('vgiss', _QUIET_VIDICON)],
    ids=['ccd', 'vidicon'],
)
def test_each_chain_shades_the_last_line_by_the_factor_ratio(
    instrument: str, noise: dict[str, Any]
) -> None:
    """The last line keeps 1.05 / 1.5 of the light it has with the shutter shading off."""
    off, _ = _lit_frame(instrument, noise, {})
    on, _ = _lit_frame(instrument, noise, {'shutter_shading': _SHUTTER})
    ratio = on / off
    assert float(ratio[-1].mean() / ratio[0].mean()) == pytest.approx(0.7, abs=0.01)


@pytest.mark.parametrize(
    ('instrument', 'noise'),
    [('gossi', _QUIET_CCD), ('vgiss', _QUIET_VIDICON)],
    ids=['ccd', 'vidicon'],
)
def test_each_chain_records_the_shutter_shading(instrument: str, noise: dict[str, Any]) -> None:
    """The frame truth records the shading's two factors."""
    _image, truth = _lit_frame(instrument, noise, {'shutter_shading': _SHUTTER})
    assert truth['artifacts']['shutter_shading'] == {
        'active': True,
        'top_factor': 1.5,
        'bottom_factor': 1.05,
    }


def test_frame_transfer_smear_adds_a_column_pedestal() -> None:
    """A bright column gains a full-column pedestal; a dark column stays dark."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    electrons[20:30, 32] = 1000.0
    apply_frame_transfer_smear(electrons, t_scrub_sec=0.012, t_transfer_sec=0.011, exposure_sec=0.1)
    assert float(electrons[0, 32]) > 0.0
    assert float(electrons[0, 5]) == pytest.approx(0.0)


def test_frame_transfer_smear_disabled_at_zero_transfer() -> None:
    """A zero scrub and transfer time is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    electrons[20:30, 32] = 1000.0
    before = electrons.copy()
    apply_frame_transfer_smear(electrons, t_scrub_sec=0.0, t_transfer_sec=0.0, exposure_sec=0.1)
    assert np.array_equal(electrons, before)


def test_lorri_instrument_defaults_injects_frame_transfer_smear() -> None:
    """LORRI turns frame_transfer_smear on under instrument_defaults (15.7)."""
    dp = resolve_detector_params(
        {
            'instrument': 'nhlorri',
            'random_seed': 1,
            'exposure_sec': 0.1,
            'artifacts': {'instrument_defaults': True},
        }
    )
    assert 'frame_transfer_smear' in dp.detector_modes
    assert dp.detector_modes['frame_transfer_smear']['t_transfer_sec'] == 0.011


def test_lorri_floor_does_not_inject_frame_transfer_smear() -> None:
    """Without instrument_defaults LORRI carries no injected smear."""
    dp = resolve_detector_params({'instrument': 'nhlorri', 'random_seed': 1, 'exposure_sec': 0.1})
    assert 'frame_transfer_smear' not in dp.detector_modes


def test_generic_instrument_defaults_injects_no_frame_transfer_smear() -> None:
    """The injection is LORRI-only: a generic instrument_defaults scene gets none."""
    dp = resolve_detector_params(
        {
            'instrument': 'generic',
            'random_seed': 1,
            'exposure_sec': 0.1,
            'artifacts': {'instrument_defaults': True},
        }
    )
    assert 'frame_transfer_smear' not in dp.detector_modes


def test_generic_instrument_defaults_records_no_smear_truth() -> None:
    """A generic instrument_defaults render writes no frame_transfer_smear record."""
    _img, truth = render_scene(disc_scene('generic', artifacts={'instrument_defaults': True}))
    assert 'frame_transfer_smear' not in truth.get('artifacts', {})
