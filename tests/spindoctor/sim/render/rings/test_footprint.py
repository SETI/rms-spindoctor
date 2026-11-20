"""How ring pixels are sorted for integration, and that batching changes no value."""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import ring_system_block

from spindoctor.sim.render.rings import footprint as footprint_module
from spindoctor.sim.render.rings.footprint import RingGrid, SlopeMap, split_sloped_pixels
from spindoctor.sim.render.rings.render import render_ring_system


def _orbit(a_px: float) -> dict[str, float]:
    """A circular orbit of radius ``a_px``."""
    return {'a_px': a_px, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0}


def test_propeller_lobe_inside_a_footprint_curves_the_pixel() -> None:
    """A pixel whose footprint crosses a narrow propeller lobe goes to the cell path.

    The propeller sits at radius 100 px with lobes 0.01 px wide, which reach 0.41 px.
    A pixel centered at 100.45 px is beyond that reach, but its footprint, 100.45 px
    plus or minus 0.5, covers the lobe at 100.01 px.
    """
    slopes = SlopeMap.empty((1, 1))
    slopes.mask[0, 0] = True
    grid = RingGrid(r=np.array([[100.45]]), lam=np.array([[0.0]]), radial_scale=np.array([[1.0]]))
    moonlet = {
        'a_px': 100.0,
        'lam_deg': 0.0,
        'propeller': {'length_deg': 10.0, 'width_px': 0.01, 'contrast': -0.5},
    }
    split = split_sloped_pixels(
        slopes,
        np.zeros(0, dtype=np.int64),
        [],
        [moonlet],
        grid=grid,
        unclipped_tau=np.zeros((1, 1)),
        os=1,
    )
    assert split.curved_pixels.tolist() == [0]


def _busy_system() -> dict[str, Any]:
    """Narrow ringlets, a ramp, and a propeller over the ramp, inclined and rotated."""
    features: list[dict[str, Any]] = [
        {'kind': 'ringlet', 'tau': 1.0, 'width_px': 0.7, 'orbit': _orbit(12.0 + 2.5 * i)}
        for i in range(6)
    ]
    features.append(
        {'kind': 'ramp', 'tau': 2.0, 'width_px': 5.0, 'side': 'out', 'orbit': _orbit(30.0)}
    )
    system = ring_system_block(
        32.0, 32.0, features, opening_deg_obs=40.0, opening_deg_sun=30.0, node_deg=25.0
    )
    system['moonlets'] = [
        {
            'a_px': 33.0,
            'lam_deg': 20.0,
            'radius_px': 0.5,
            'amplitude': 0.0,
            'propeller': {'length_deg': 20.0, 'width_px': 1.0, 'contrast': 2.0},
        }
    ]
    return system


def test_cell_batches_change_no_pixel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integrating the cut and curved pixels one pixel's cells at a time is bit-identical."""
    system = _busy_system()
    whole = render_ring_system((64, 64), system, center_v=32.0, center_u=32.0, node_deg=25.0)
    monkeypatch.setattr(footprint_module, '_CELL_BATCH', 1)
    batched = render_ring_system((64, 64), system, center_v=32.0, center_u=32.0, node_deg=25.0)
    np.testing.assert_array_equal(batched.intensity, whole.intensity)
