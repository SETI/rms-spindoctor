"""The banding effect: its bands, its dark step, and the cameras that show it."""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene

from spindoctor.sim.render.detector.banding import add_banding
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene

_SIZE = 64


def _rng(seed: int = 0) -> np.random.Generator:
    """A seeded generator for the mechanics under test."""
    return np.random.default_rng(seed)


def test_banding_horizontal_is_row_correlated() -> None:
    """Horizontal banding is constant along a row and varies between rows."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_banding(
        electrons,
        amplitude=100.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert float(np.ptp(electrons, axis=1).max()) < 1e-9
    assert float(np.ptp(electrons, axis=0).max()) > 1.0


def test_banding_vertical_is_column_correlated() -> None:
    """Vertical banding is constant down a column and varies between columns."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_banding(
        electrons,
        amplitude=100.0,
        period_px=8.0,
        orientation='vertical',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert float(np.ptp(electrons, axis=0).max()) < 1e-9
    assert float(np.ptp(electrons, axis=1).max()) > 1.0


def test_banding_dark_step_lifts_the_lower_half() -> None:
    """A dark-level step raises the mean of the rows below the mid-line."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    add_banding(
        electrons,
        amplitude=10.0,
        period_px=16.0,
        orientation='horizontal',
        freq_step_factor=1.0,
        dark_step_dn=50.0,
        gain_e_per_dn=2.0,
        rng=_rng(1),
    )
    top = float(electrons[: _SIZE // 2].mean())
    bottom = float(electrons[_SIZE // 2 :].mean())
    assert bottom - top == pytest.approx(100.0, abs=5.0)


def test_banding_disabled_at_zero_amplitude() -> None:
    """A zero banding amplitude is a no-op."""
    electrons = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    record = add_banding(
        electrons,
        amplitude=0.0,
        period_px=16.0,
        orientation='both',
        freq_step_factor=1.0,
        dark_step_dn=0.0,
        gain_e_per_dn=1.0,
        rng=_rng(5),
    )
    assert record['active'] is False
    assert np.all(electrons == 0.0)


def test_the_calibrated_cassini_products_show_no_banding() -> None:
    """The instrument defaults of a calibrated Cassini camera leave banding off."""
    dp = resolve_detector_params(
        disc_scene('coiss_calib_nac', artifacts={'instrument_defaults': True})
    )
    assert 'banding' not in dp.detector_modes


def test_the_vidicon_bands_are_vertical() -> None:
    """The vidicon chain draws its camera's bands down the columns, in DN."""
    quiet = {'vidicon': {'read_noise_line_dn': 0.0, 'read_noise_pixel_dn': 0.0}}
    blocks: dict[str, Any] = {'noise': quiet, 'detector': {'quantization': 'exact'}}
    on, _ = render_scene(disc_scene('vgiss', artifacts={'banding': {'probability': 1.0}}, **blocks))
    strong, truth = render_scene(
        disc_scene(
            'vgiss', artifacts={'banding': {'probability': 1.0, 'amplitude': 20.0}}, **blocks
        )
    )
    assert truth['artifacts']['banding']['orientation'] == 'vertical'
    sky = strong[:8]
    assert float(np.ptp(sky, axis=0).max()) == 0.0
    assert float(np.ptp(sky)) > 10.0 * float(np.ptp(on[:8]))
