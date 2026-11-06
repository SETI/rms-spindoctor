"""The calibrated product's residual: a fixed column pattern and a per-frame pixel term.

A real calibration pipeline subtracts per-pixel models from integer DN, so its
product is continuous and carries structure the raw frame does not.  The sim's
calibration inverse reproduces that with ``calibration_column_residual_e`` (a
per-detector column map, the same in every scene) and
``calibration_pixel_residual_e`` (a per-frame Gaussian per pixel), both in
electrons per native pixel.  A raw product never carries either.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifacts_catalog import DETECTOR_DEFAULTS
from spindoctor.sim.forward.detector.noise_stages import detector_column_map
from spindoctor.sim.forward.detector.params import calibration_map, resolve_detector_params
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError

_SIZE = 256


def _scene(instrument: str, noise: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A dark-sky scene on ``instrument`` with ``noise``."""
    scene: dict[str, Any] = {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 11,
        'instrument': instrument,
        'exposure_sec': 0.1,
        'noise': noise,
    }
    scene.update(extra)
    return scene


def _residual_only(column_e: float, pixel_e: float) -> dict[str, Any]:
    """A noise block that leaves only the calibration residual."""
    return {
        'poisson': False,
        'read_noise_dn': 0.0,
        'bias_dn': 0.0,
        'calibration_column_residual_e': column_e,
        'calibration_pixel_residual_e': pixel_e,
    }


def _electrons(scene: dict[str, Any]) -> np.ndarray:
    """Render ``scene`` and convert its calibrated I/F back to electrons."""
    img, _ = render_combined_model(scene)
    dp = resolve_detector_params(scene)
    cal = calibration_map(dp)
    assert cal is not None
    return np.asarray(img - cal.sky_if) * cal.dn_per_if * dp.gain_e_per_dn


# ---- The column map ----


def test_the_column_map_is_standard_normal() -> None:
    """Over many columns the map has mean 0 and sigma 1."""
    values = detector_column_map(200_000, effect='calibration_columns', detector_id='coiss_nac')
    assert float(values.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(values.std()) == pytest.approx(1.0, abs=0.01)


def test_the_column_map_keys_each_column_on_the_detector_alone() -> None:
    """A narrower frame sees the first columns of the same map."""
    wide = detector_column_map(512, effect='calibration_columns', detector_id='coiss_nac')
    narrow = detector_column_map(128, effect='calibration_columns', detector_id='coiss_nac')
    assert np.array_equal(wide[:128], narrow)


@pytest.mark.parametrize(
    ('effect', 'detector_id'),
    [('calibration_columns', 'coiss_wac'), ('other_effect', 'coiss_nac')],
    ids=['other-detector', 'other-effect'],
)
def test_another_detector_or_effect_draws_another_map(effect: str, detector_id: str) -> None:
    """The map is keyed on both the effect and the detector."""
    base = detector_column_map(4096, effect='calibration_columns', detector_id='coiss_nac')
    other = detector_column_map(4096, effect=effect, detector_id=detector_id)
    assert abs(float(np.corrcoef(base, other)[0, 1])) < 0.05


# ---- The residual's amplitude ----


@pytest.mark.parametrize('binning', [1, 2], ids=['full', 'sum2'])
def test_the_column_residual_sigma_is_the_cataloged_electrons(binning: int) -> None:
    """Column means carry sigma column_e x b x sqrt(b) electrons per frame pixel."""
    scene = _scene(
        'coiss_calib_nac',
        _residual_only(15.0, 0.0),
        size_u=4096 // binning,
        size_v=8,
        detector={'binning': binning},
    )
    electrons = _electrons(scene)
    columns = electrons.mean(axis=0)
    assert float(columns.std()) == pytest.approx(15.0 * binning**1.5, rel=0.05)


def test_the_column_residual_is_constant_down_a_column() -> None:
    """Without a pixel term every row carries the same column pattern."""
    electrons = _electrons(_scene('coiss_calib_nac', _residual_only(15.0, 0.0)))
    assert float(np.abs(electrons - electrons[0]).max()) < 1e-6


@pytest.mark.parametrize('binning', [1, 2], ids=['full', 'sum2'])
def test_the_pixel_residual_sigma_is_the_cataloged_electrons(binning: int) -> None:
    """The pixel term has sigma pixel_e x b electrons per frame pixel."""
    scene = _scene('coiss_calib_nac', _residual_only(0.0, 9.4), detector={'binning': binning})
    electrons = _electrons(scene)
    assert float(electrons.std()) == pytest.approx(9.4 * binning, rel=0.03)


# ---- Fixed columns, continuous values ----


def test_two_scenes_share_the_column_pattern() -> None:
    """Different seeds give the same columns (the map belongs to the detector)."""
    noise = _residual_only(15.0, 9.4)
    first = _electrons(_scene('coiss_calib_nac', noise, random_seed=1)).mean(axis=0)
    second = _electrons(_scene('coiss_calib_nac', noise, random_seed=2)).mean(axis=0)
    assert float(np.corrcoef(first, second)[0, 1]) > 0.95


def test_an_instrument_defaults_calibrated_sky_is_continuous() -> None:
    """The cataloged residual takes the calibrated sky off the integer DN lattice."""
    img, _ = render_combined_model(
        _scene('coiss_calib_nac', {}, artifacts={'instrument_defaults': True})
    )
    assert np.unique(img).size > 10_000


@pytest.mark.parametrize('instrument', ['coiss_calib_nac', 'coiss_calib_wac'])
def test_each_calibrated_camera_catalogs_both_terms(instrument: str) -> None:
    """Both calibrated Cassini products carry a measured, positive residual."""
    entry = DETECTOR_DEFAULTS[instrument]
    assert entry['calibration_column_residual_e'] > 0.0
    assert entry['calibration_pixel_residual_e'] > 0.0


def test_a_raw_product_ignores_the_residual() -> None:
    """A raw DN frame has no calibration inverse, so the keys change nothing."""
    plain, _ = render_combined_model(_scene('coiss_nac', _residual_only(0.0, 0.0)))
    keyed, _ = render_combined_model(_scene('coiss_nac', _residual_only(15.0, 9.4)))
    assert np.array_equal(plain, keyed)


def test_the_validator_refuses_a_negative_residual() -> None:
    """A residual sigma is non-negative."""
    with pytest.raises(SimSceneValidationError, match='calibration_column_residual_e'):
        validate_sim_params(_scene('coiss_calib_nac', {'calibration_column_residual_e': -1.0}))
