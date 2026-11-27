"""The LORRI entry's frame transfer and integration time against real raw frames.

The shares come from the raw frame lor_0034849319_0x630_eng (Europa, 5 ms): against a
per-row sky taken from columns away from the disc, each row above the disc (small line
numbers) carries 2.115e-3 of its column's charge and each row below it 1.937e-3.  A
simulated 5 ms frame of a bright disc must show the same shares, which holds only when
the frame-transfer times, their sides, and the extra integration time are all right.
"""

import copy
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.driver import render_scene

# The measured shares of lor_0034849319_0x630_eng, and how far a render may stray.
_REAL_SHARE_ABOVE = 2.115e-3
_REAL_SHARE_BELOW = 1.937e-3
_SHARE_TOLERANCE = 0.03

# The camera's extra integration time, as its entry states it.
_EXTRA_INTEGRATION_SEC = 6.3e-4

_DISC_COLUMNS = slice(490, 535)


def _scene(exposure_sec: float, **extra: Any) -> dict[str, Any]:
    """A full LORRI frame of a bright disc in the upper part of the frame, no defects."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'nhlorri_smear_share',
        'instrument': 'nhlorri',
        'size_v': 1024,
        'size_u': 1024,
        'random_seed': 7,
        'exposure_sec': exposure_sec,
        'bodies': [
            {
                'name': 'EUROPA',
                'center_v': 300.0,
                'center_u': 512.0,
                'semi_a_px': 60.0,
                'semi_b_px': 60.0,
                'semi_c_px': 60.0,
                'illumination_angle': 230.0,
                'phase_angle': 28.0,
            }
        ],
    }
    scene.update(extra)
    return scene


@pytest.fixture(scope='module')
def smear_shares() -> tuple[float, float]:
    """The smear share above and below a disc in a 5 ms frame with the catalog timings."""
    smeared = _scene(0.005, artifacts={'frame_transfer_smear': {'probability': 1.0}})
    clean = copy.deepcopy(smeared)
    del clean['artifacts']
    img, _ = render_scene(smeared)
    img0, _ = render_scene(clean)
    smear = img - img0
    bias = float(np.median(img0[:, :100]))
    charge = float((img0[:, _DISC_COLUMNS] - bias).sum())
    above = float(smear[20:200, _DISC_COLUMNS].mean(axis=0).sum()) / charge
    below = float(smear[420:1000, _DISC_COLUMNS].mean(axis=0).sum()) / charge
    return above, below


def test_rows_above_a_bright_source_carry_the_measured_share(
    smear_shares: tuple[float, float],
) -> None:
    """Each row above the disc carries the real frame's share of its column's charge."""
    assert abs(smear_shares[0] / _REAL_SHARE_ABOVE - 1.0) < _SHARE_TOLERANCE


def test_rows_below_a_bright_source_carry_the_measured_share(
    smear_shares: tuple[float, float],
) -> None:
    """Each row below the disc carries the real frame's share of its column's charge."""
    assert abs(smear_shares[1] / _REAL_SHARE_BELOW - 1.0) < _SHARE_TOLERANCE


def _signal_dn(scene: dict[str, Any]) -> float:
    """The frame's total DN above the bias."""
    img, _ = render_scene(scene)
    return float((img - float(np.median(img[:, :20]))).sum())


def test_a_disc_collects_light_for_the_exposure_plus_the_extra_time() -> None:
    """Doubling the exposure scales a disc's light by the ratio of integration times."""
    ratio = _signal_dn(_scene(0.010)) / _signal_dn(_scene(0.005))
    expected = (0.010 + _EXTRA_INTEGRATION_SEC) / (0.005 + _EXTRA_INTEGRATION_SEC)
    assert ratio == pytest.approx(expected, rel=1e-4)


def test_a_star_collects_light_for_the_exposure_plus_the_extra_time() -> None:
    """Doubling the exposure scales a star's light by the ratio of integration times.

    The unblurred star lands in a few pixels of a few hundred DN each, whose rounding to
    whole DN allows the looser tolerance; leaving out the extra time moves the ratio by
    5.9%.
    """
    star = {'bodies': [], 'stars': [{'name': 'S1', 'v': 600.0, 'u': 600.0, 'vmag': 5.0}]}
    ratio = _signal_dn(_scene(0.010, **star)) / _signal_dn(_scene(0.005, **star))
    expected = (0.010 + _EXTRA_INTEGRATION_SEC) / (0.005 + _EXTRA_INTEGRATION_SEC)
    assert ratio == pytest.approx(expected, rel=5e-3)
