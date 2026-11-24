"""The bias structure: its pedestal, its gradients, and the cameras that show it."""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene, frame_scene, quiet_noise

from spindoctor.sim.render.detector.bias import add_bias_structure
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene

# A bias structure of the pedestal alone.
_FLAT_BIAS = {'row_gradient_dn': 0.0, 'col_gradient_dn': 0.0}


def test_bias_structure_adds_a_pedestal_and_gradients() -> None:
    """Bias structure changes the DN field when any amplitude is nonzero."""
    dn = np.zeros((32, 48), dtype=np.float64)
    add_bias_structure(
        dn,
        pedestal_sigma_dn=2.0,
        row_gradient_dn=4.0,
        col_gradient_dn=2.0,
        rng=np.random.default_rng(7),
    )
    assert float(dn.std()) > 0.0


def test_bias_structure_disabled_at_zero_amplitudes() -> None:
    """Bias structure with every amplitude zero is a no-op."""
    dn = np.zeros((32, 48), dtype=np.float64)
    add_bias_structure(
        dn,
        pedestal_sigma_dn=0.0,
        row_gradient_dn=0.0,
        col_gradient_dn=0.0,
        rng=np.random.default_rng(7),
    )
    assert np.all(dn == 0.0)


def test_bias_structure_records_the_pedestal_it_draws() -> None:
    """The rendered frame sits off the flat bias by the recorded pedestal."""
    modes = {'bias_structure': {'probability': 1.0, 'pedestal_sigma_dn': 50.0, **_FLAT_BIAS}}
    img, truth = render_scene(
        frame_scene(seed=3, noise=quiet_noise(), detector={'bias_dn': 500.0}, artifacts=modes)
    )
    pedestal = truth['artifacts']['bias_structure']['pedestal_dn']
    assert float(img.mean()) == pytest.approx(500.0 + pedestal, abs=0.5)
    assert abs(pedestal) > 1.0


def test_a_calibrated_cassini_product_has_no_bias_structure() -> None:
    """The instrument defaults of a calibrated Cassini camera leave the bias structure off."""
    dp = resolve_detector_params(
        disc_scene('coiss_calib_nac', artifacts={'instrument_defaults': True})
    )
    assert 'bias_structure' not in dp.detector_modes


def _gradient_frame(
    instrument: str, noise: dict[str, Any], *, bias_dn: float = 0.0, **gradients: float
) -> dict[str, Any]:
    """A quiet empty frame whose bias structure is the given gradients alone."""
    bias = {'probability': 1.0, 'pedestal_sigma_dn': 0.0, **gradients}
    return frame_scene(
        instrument=instrument,
        exposure_sec=0.1,
        noise=noise,
        detector={'quantization': 'exact', 'bias_dn': bias_dn},
        artifacts={'bias_structure': bias},
    )


# A quiet CCD frame on a bias high enough that a falling gradient stays above 0 DN.
_QUIET_CCD = quiet_noise()


def test_the_row_gradient_rises_down_the_frame() -> None:
    """A 20 DN row gradient raises the last line 20 DN above the first."""
    img, _ = render_scene(
        _gradient_frame(
            'coiss_nac', _QUIET_CCD, bias_dn=100.0, row_gradient_dn=20.0, col_gradient_dn=0.0
        )
    )
    rows = img.mean(axis=1)
    assert float(rows[-1] - rows[0]) == pytest.approx(20.0)


def test_the_column_gradient_rises_across_the_frame() -> None:
    """A 20 DN column gradient raises the last column 20 DN above the first."""
    img, _ = render_scene(
        _gradient_frame(
            'coiss_nac', _QUIET_CCD, bias_dn=100.0, row_gradient_dn=0.0, col_gradient_dn=20.0
        )
    )
    columns = img.mean(axis=0)
    assert float(columns[-1] - columns[0]) == pytest.approx(20.0)


def test_the_vidicon_chain_draws_the_bias_structure() -> None:
    """On the vidicon chain a 20 DN row gradient reads through the calibration."""
    quiet = {'vidicon': {'read_noise_line_dn': 0.0, 'read_noise_pixel_dn': 0.0}}
    scene = _gradient_frame(
        'vgiss_1_nac', quiet, bias_dn=100.0, row_gradient_dn=20.0, col_gradient_dn=0.0
    )
    img, _ = render_scene(scene)
    rise_if = 20.0 / (resolve_detector_params(scene).calibration_scale_dn_per_s_per_if * 0.1)
    rows = img.mean(axis=1)
    assert float(rows[-1] - rows[0]) == pytest.approx(rise_if)
