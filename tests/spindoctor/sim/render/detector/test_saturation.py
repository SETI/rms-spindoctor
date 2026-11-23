"""The full well, the column bloom above it, and the serial tail.

A Cassini scene saturates at ``full_well_e / gain`` DN, below the ADC ceiling.  The
bloom and the tail are driven directly, and a bright scene is rendered end to end.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_scene, frame_scene, quiet_noise, zero_bias

from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.detector.saturation import add_serial_tail, apply_saturation
from spindoctor.sim.render.driver import render_scene

_SIZE = 64
_FULL_WELL_E = 110.0e3
_NAC_GAIN = 30.0
_NAC_BIAS = 20.0


def test_clip_caps_at_full_well() -> None:
    """Electrons above the full well are capped down to it."""
    electrons = np.array([[0.0, 5.0e4], [2.0e5, 5.0e5]], dtype=np.float64)
    apply_saturation(electrons, full_well_e=_FULL_WELL_E)
    assert float(electrons.max()) == _FULL_WELL_E


def test_clip_leaves_subfull_pixels_untouched() -> None:
    """Electrons below the full well are unchanged by capping."""
    electrons = np.array([[0.0, 5.0e4], [2.0e5, 5.0e5]], dtype=np.float64)
    apply_saturation(electrons, full_well_e=_FULL_WELL_E)
    assert float(electrons[0, 1]) == 5.0e4


def test_no_bloom_keeps_single_saturated_pixel() -> None:
    """Without bloom, only the originally over-full pixel saturates."""
    electrons = np.zeros((21, 21), dtype=np.float64)
    electrons[10, 10] = 50.0 * _FULL_WELL_E
    apply_saturation(electrons, full_well_e=_FULL_WELL_E, bloom_length=0)
    assert int((electrons >= _FULL_WELL_E).sum()) == 1


def test_bloom_spreads_along_column() -> None:
    """Column bloom turns a saturated pixel into a vertical streak."""
    electrons = np.zeros((21, 21), dtype=np.float64)
    electrons[10, 10] = 50.0 * _FULL_WELL_E
    apply_saturation(electrons, full_well_e=_FULL_WELL_E, bloom_length=3)
    saturated = np.argwhere(electrons >= _FULL_WELL_E)
    distinct_v = {int(v) for v, _ in saturated}
    distinct_u = {int(u) for _, u in saturated}
    assert len(distinct_v) > 1
    assert distinct_u == {10}


def test_bloom_conserves_excess_floor() -> None:
    """Bloom never lowers the count of saturated pixels below the no-bloom case."""
    base = np.zeros((21, 21), dtype=np.float64)
    base[10, 10] = 50.0 * _FULL_WELL_E
    bloomed = base.copy()
    apply_saturation(base, full_well_e=_FULL_WELL_E, bloom_length=0)
    apply_saturation(bloomed, full_well_e=_FULL_WELL_E, bloom_length=4)
    assert int((bloomed >= _FULL_WELL_E).sum()) >= int((base >= _FULL_WELL_E).sum())


def _bright_scene(*, size: int = 64) -> dict[str, Any]:
    """A coiss scene bright enough to drive the lit body into saturation."""
    return frame_scene(
        size=size,
        exposure_sec=1.0,
        detector={'signal_full_scale_frac': 2.0},
        bodies=[
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'semi_a_px': size * 0.3,
                'semi_b_px': size * 0.25,
                'semi_c_px': size * 0.25,
            }
        ],
    )


def test_render_saturates_at_physical_full_well() -> None:
    """A bright scene clips at full_well_e / gain, below the 4095 ADC ceiling."""
    img, _ = render_scene(_bright_scene())
    expected = round(_FULL_WELL_E / _NAC_GAIN + _NAC_BIAS)
    assert float(img.max()) == expected
    assert expected < 4095.0


def test_the_bloom_mode_spreads_a_saturated_source_along_its_column() -> None:
    """A rendered star above the full well saturates more of its column with the mode on."""
    star = {'name': 'S', 'v': 48.5, 'u': 48.5, 'vmag': -1.0}
    blocks: dict[str, Any] = {
        'instrument': 'coiss_nac',
        'size': 96,
        'noise': quiet_noise(),
        'detector': zero_bias(),
    }
    off, _ = render_scene(frame_scene(stars=[star], **blocks))
    on, truth = render_scene(
        frame_scene(
            stars=[star], artifacts={'bloom': {'probability': 1.0, 'bloom_length': 9}}, **blocks
        )
    )
    ceiling = round(_FULL_WELL_E / _NAC_GAIN)
    assert int((on[:, 48] >= ceiling).sum()) > int((off[:, 48] >= ceiling).sum()) > 0
    assert truth['artifacts']['bloom'] == {'active': True, 'bloom_length': 9}


def test_lorri_does_not_bloom_under_instrument_defaults() -> None:
    """LORRI's CCD drains the charge above the well, so its defaults turn no bloom on."""
    dp = resolve_detector_params(disc_scene('nhlorri', artifacts={'instrument_defaults': True}))
    assert 'bloom' not in dp.detector_modes


def test_serial_tail_writes_a_horizontal_tail_off_saturation() -> None:
    """A saturated pixel seeds a bright-then-dark tail in the readout direction."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    dn[32, 20] = 4095.0
    add_serial_tail(
        dn,
        saturation_dn=4095.0,
        saturation_frac=0.99,
        amplitude_dn=12.0,
        length_px=6,
        direction='right',
    )
    assert float(dn[32, 21]) > 0.0  # bright overshoot just right of the source
    assert float(dn[32, 24]) < 0.0  # dark undershoot further along
    assert float(dn[32, 19]) == pytest.approx(0.0)  # nothing to the left


def test_serial_tail_disabled_without_saturation() -> None:
    """With no saturated pixel the serial tail is a no-op."""
    dn = np.full((_SIZE, _SIZE), 100.0, dtype=np.float64)
    before = dn.copy()
    record = add_serial_tail(
        dn,
        saturation_dn=4095.0,
        saturation_frac=0.99,
        amplitude_dn=12.0,
        length_px=6,
        direction='right',
    )
    assert record['sources'] == 0
    assert np.array_equal(dn, before)


def test_serial_tail_length_one_renders() -> None:
    """A 1-px tail renders without error; there is nothing past the source to write."""
    dn = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    dn[32, 20] = 4095.0
    record = add_serial_tail(
        dn,
        saturation_dn=4095.0,
        saturation_frac=0.99,
        amplitude_dn=12.0,
        length_px=1,
        direction='right',
    )
    assert record['active'] is True
    assert record['sources'] == 1
    assert float(dn[32, 21]) == pytest.approx(0.0)
