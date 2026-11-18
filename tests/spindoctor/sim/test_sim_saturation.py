"""Electron-domain full-well bloom and the physical saturation level.

``apply_saturation`` now caps and blooms in electrons against the full well, so a
Cassini scene saturates at ``full_well_e / gain`` DN (below the ADC ceiling), the
antiblooming-free CCD behavior.  These tests drive the bloom mechanics directly
and render a bright scene end-to-end.
"""

from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import frame_scene

from spindoctor.sim.render.detector import apply_saturation
from spindoctor.sim.render.driver import render_combined_model

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
        noise={'signal_full_scale_frac': 2.0},
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
    img, _ = render_combined_model(_bright_scene())
    expected = round(_FULL_WELL_E / _NAC_GAIN + _NAC_BIAS)
    assert float(img.max()) == expected
    assert expected < 4095.0
