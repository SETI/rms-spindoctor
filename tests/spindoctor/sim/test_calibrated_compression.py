"""Lossy DCT compression on a calibrated frame acts on the transmitted DN.

The codec runs on board, so ``compression_dct`` maps a calibrated (I/F) frame back
to the DN the spacecraft sent, quantizes there with the same ``scale_factor`` DN
step a raw frame gets, and maps the result back to I/F.  The pedestal the
calibration subtracts takes part in the quantization, so a dark sky keeps its
level instead of rounding onto the 0 marker.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.detector.params import calibration_map, resolve_detector_params
from spindoctor.sim.forward.telemetry_artifacts import apply_compression_dct
from spindoctor.sim.instruments import read_frame_as_navigator, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model

_COMPRESSION = {'compression_dct': {'probability': 1.0}}


def _disc(instrument: str, **extra: Any) -> dict[str, Any]:
    """A 96 x 96 frame with a lit disc and dark sky, its codec on."""
    scene: dict[str, Any] = {
        'size_v': 96,
        'size_u': 96,
        'instrument': instrument,
        'random_seed': 7,
        'artifacts': dict(_COMPRESSION),
        'bodies': [
            {
                'name': 'RHEA',
                'center_v': 48.0,
                'center_u': 48.0,
                'axis1': 50.0,
                'axis2': 50.0,
                'axis3': 50.0,
                'illumination_angle': 18.0,
                'phase_angle': 30.0,
            }
        ],
    }
    scene.update(extra)
    return scene


_CASES: dict[str, dict[str, Any]] = {
    'floor': {},
    'instrument_defaults': {'artifacts': {**_COMPRESSION, 'instrument_defaults': True}},
    # Under instrument_defaults the NAC's default gain state puts a whole DN on
    # exactly 0 I/F at this exposure (see test_calibrated_sky.py).
    'defaults-at-a-lattice-crossing': {
        'exposure_sec': 75.0 / 7.0,
        'artifacts': {**_COMPRESSION, 'instrument_defaults': True},
    },
}


@pytest.mark.parametrize('case', sorted(_CASES))
@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac', 'vgiss'])
def test_a_compressed_calibrated_frame_has_no_pixel_at_the_marker(
    instrument: str, case: str
) -> None:
    """No pixel of a compressed calibrated frame reads 0 (vgiss has no codec mode)."""
    img, _ = render_combined_model(_disc(instrument, **_CASES[case]))
    assert int((img == 0.0).sum()) == 0


@pytest.mark.parametrize('case', sorted(_CASES))
@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac', 'vgiss'])
def test_a_compressed_calibrated_frame_classifies_clean(instrument: str, case: str) -> None:
    """The navigator reads a compressed calibrated disc frame as clean."""
    params = _disc(instrument, **_CASES[case])
    img, _ = render_combined_model(params)
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, instrument)
    assert read_frame_as_navigator(img, inst_config).image_class == 'clean'


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac'])
def test_the_codec_keeps_a_flat_dark_sky_at_its_level(instrument: str) -> None:
    """On the noise-free floor, a codec block of pure sky keeps the sky's residual level.

    Blocks that cross the limb ring, as on a raw frame; a block wholly in the
    flat sky is a whole DN on board and comes back unchanged.
    """
    plain = _disc(instrument)
    plain['artifacts'] = {}
    before, _ = render_combined_model(plain)
    after, _ = render_combined_model(_disc(instrument))
    sky_blocks = (before == before.min()).reshape(12, 8, 12, 8).all(axis=(1, 3))
    sky = np.repeat(np.repeat(sky_blocks, 8, axis=0), 8, axis=1)
    assert int(sky.sum()) > 1000
    np.testing.assert_allclose(after[sky], before[sky], rtol=0.0, atol=1e-12)


def test_the_codec_compresses_a_calibrated_frame_in_its_transmitted_dn() -> None:
    """Compressing the I/F frame equals compressing its DN and calibrating the result.

    The step is ``scale_factor`` DN on both: the calibrated quantization grid is
    the DN grid mapped through the calibration the inverse uses.
    """
    params = _disc('coiss_calib_nac', exposure_sec=2.5)
    params['artifacts'] = {'instrument_defaults': True}
    calibration = calibration_map(resolve_detector_params(params))
    assert calibration is not None
    dn = np.random.default_rng(3).uniform(15.0, 400.0, size=(32, 32))
    cfg = {'probability': 1.0, 'scale_factor': 8.0, 'block': 8}
    from_dn = dn.copy()
    apply_compression_dct(from_dn, cfg, rng=np.random.default_rng(0), protect=None)
    from_if = calibration.to_if(dn)
    apply_compression_dct(
        from_if, cfg, rng=np.random.default_rng(0), protect=None, calibration=calibration
    )
    np.testing.assert_allclose(from_if, calibration.to_if(from_dn), rtol=0.0, atol=1e-12)
