"""Tests for the limb-navigation bias harness.

The renderer-validation tests run the simulator body renderer directly (no
navigation, no holdings), so they execute in the plain unit suite and guard
the assumption the sim-based diagnosis rests on: the simulator plants a body
at its requested sub-pixel center without embedding a positional bias.  The
navigation smoke test is marked ``integration`` because it drives the full
model / technique stack.
"""

from __future__ import annotations

import pytest

from tests.integration.limb_bias import (
    build_body_scene,
    measure_sim_limb_bias,
    renderer_centroid_offset,
    ridge_inset_phase_zero,
)


@pytest.mark.parametrize('center_v', [100.0, 100.25, 100.5, 100.75])
@pytest.mark.parametrize('center_u', [100.0, 100.4])
def test_renderer_centroid_is_unbiased(center_v: float, center_u: float) -> None:
    """A phase-0 sphere's brightness centroid lands on its geometric center."""
    check = renderer_centroid_offset(center_vu=(center_v, center_u), diameter_px=140.0)
    assert check.centroid_error_mag_px < 0.02


def test_renderer_gradient_ridge_sits_inside_limb() -> None:
    """The brightness gradient ridge is inset from the geometric limb.

    A positive inset is the photometric roll-off signature: the steepest-slope
    point that the edge distance transform localizes lies inside the true
    silhouette boundary, so a limb fit that assumes the edge is at the geometry
    is pulled inward on the lit side.
    """
    inset = ridge_inset_phase_zero(diameter_px=160.0)
    assert inset > 0.1


def test_zero_offset_scene_has_finite_bias() -> None:
    """A noise-free integer-offset scene yields a finite, sub-pixel-scale error.

    Rendered at a whole-pixel offset there is no interpolation of the observed
    image, so any error is the pure edge-model component; it must be small
    (well under a pixel) but is allowed to be non-zero -- that residual is the
    bias under study.
    """
    scene = build_body_scene(
        diameter_px=160.0,
        phase_deg=30.0,
        illumination_deg=25.0,
        offset_vu=(0.0, 0.0),
    )
    sample = measure_sim_limb_bias(scene)
    assert sample.error_vu is not None
    assert sample.error_mag_px is not None
    assert sample.error_mag_px < 0.5


@pytest.mark.integration
def test_sim_limb_navigation_recovers_offset() -> None:
    """The limb technique recovers a planted offset to better than one pixel."""
    scene = build_body_scene(
        diameter_px=160.0,
        phase_deg=30.0,
        illumination_deg=25.0,
        offset_vu=(1.3, -0.7),
    )
    sample = measure_sim_limb_bias(scene)
    assert sample.recovered_offset_vu is not None
    assert sample.error_mag_px is not None
    assert sample.error_mag_px < 1.0
