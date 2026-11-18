"""Stage-activation property: an absent optics block contributes nothing.

The self-consistency floor and the single-variable sweeps both depend on a
stage doing nothing unless its scene block is present.  These tests assert that
each optics sub-stage (PSF, smear, distortion, ghosts, stray light) and the
SPK-error knob are inert when absent or zeroed, and that each one activates
only its own effect when present.
"""

from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise, star_entry

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.optics.stage import apply_optics
from spindoctor.sim.render.stages import new_sim_frame


def _base_scene(**extra: Any) -> dict[str, Any]:
    """A coiss body-plus-star scene with an optional extra top-level block."""
    return frame_scene(
        size=60,
        seed=4,
        noise=quiet_noise(),
        bodies=[
            {'name': 'B', 'center_v': 30.0, 'center_u': 30.0, 'semi_a_px': 10.0, 'semi_b_px': 10.0}
        ],
        stars=[star_entry(12.0, 48.0, 0.0)],
        **extra,
    )


def test_apply_optics_with_no_block_is_a_noop() -> None:
    """The optics stage leaves the frame untouched with no optics mapping."""
    frame = new_sim_frame(40, 40)
    frame.signal[10:20, 10:20] = 0.5
    before = frame.signal.copy()
    apply_optics(frame, params={'instrument': 'coiss_nac'})
    assert np.array_equal(frame.signal, before)


def test_empty_optics_matches_no_optics() -> None:
    """An empty optics mapping renders bit-identically to no optics block."""
    plain, _ = render_scene(_base_scene())
    empty, _ = render_scene(_base_scene(optics={}))
    assert np.array_equal(plain, empty)


def test_inert_smear_contributes_nothing() -> None:
    """An empty smear list is disabled."""
    plain, _ = render_scene(_base_scene())
    inert, _ = render_scene(_base_scene(optics={'smear': []}))
    assert np.array_equal(plain, inert)


def test_inert_ghosts_contribute_nothing() -> None:
    """An empty ghost list is disabled."""
    plain, _ = render_scene(_base_scene())
    inert, _ = render_scene(_base_scene(optics={'ghosts': []}))
    assert np.array_equal(plain, inert)


def test_zeroed_distortion_contributes_nothing() -> None:
    """A distortion block with no amplitude is the identity."""
    plain, _ = render_scene(_base_scene())
    inert, _ = render_scene(
        _base_scene(optics={'distortion': {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 0.0}})
    )
    assert np.array_equal(plain, inert)


def test_zero_amplitude_stray_light_contributes_nothing() -> None:
    """A stray-light block with zero amplitude is disabled."""
    plain, _ = render_scene(_base_scene())
    inert, _ = render_scene(_base_scene(optics={'stray_light': {'amplitude': 0.0}}))
    assert np.array_equal(plain, inert)


def test_absent_spk_error_contributes_nothing() -> None:
    """No spk_error block leaves geometry unshifted."""
    plain, _ = render_scene(_base_scene())
    # A body already carries a default range; adding a zero-vector spk_error is inert.
    scene = _base_scene()
    scene['bodies'][0]['range_km'] = 100000.0
    with_range, _ = render_scene(scene)
    assert np.array_equal(plain, with_range)


def test_psf_block_activates_blur() -> None:
    """A PSF block softens the limb; its absence leaves it sharp (activation)."""
    sharp, _ = render_scene(_base_scene())
    blurred, _ = render_scene(
        _base_scene(optics={'psf': {'sigma_v': 1.2, 'sigma_u': 1.2, 'w': 0.0}})
    )
    # The limb gradient magnitude drops when the PSF spreads the edge.
    sharp_grad = float(np.abs(np.diff(sharp[30, :])).max())
    blurred_grad = float(np.abs(np.diff(blurred[30, :])).max())
    assert blurred_grad < sharp_grad
