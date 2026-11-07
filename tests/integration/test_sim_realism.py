"""Integration tests for the Section 7 realism-match runner.

Two layers, under the ``sim_integration`` tier because they render simulated
frames and need no holdings, SPICE, or catalogs:

- matched-scene construction: every cohort scene class produces a valid,
  deterministic scene for every cohort instrument;
- sim-side determinism: the same record yields identical pooled samples
  across two extractions (the whole match is reproducible given the
  cohort).

The end-to-end smoke on real frames lives in ``test_sim_realism_smoke.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.artifacts_catalog import NATIVE_FRAME_SIZE_PX
from spindoctor.sim.forward.detector.chain import COISS_LUT_DN
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from tests.integration.sim_realism import (
    INSTRUMENT_FOR,
    FrameRecord,
    _frame_samples_sim,
    discover_cohort,
)
from tests.integration.sim_realism_scenes import matched_scene

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_ALL_CLASSES = (
    'star_dominated',
    'one_bright_star_no_body',
    'two_bright_stars_no_body',
    'body_full_fov',
    'high_phase_terminator',
    'ring_only_flat',
    'ring_only_curved',
    'ring_plus_body',
    'negative_cases',
    'scattered_light',
)


_NAC_FULL = (1024, 1024)


def _full_frame(instrument: str) -> tuple[int, int]:
    """A full unbinned frame's shape for ``instrument``.

    The shape is ``NATIVE_FRAME_SIZE_PX`` on a side, or 1000 by 1000 (Voyager's
    GEOMED frame) for an instrument that table doesn't list.
    """
    size = NATIVE_FRAME_SIZE_PX.get(instrument, 1000)
    return (size, size)


@pytest.mark.parametrize('instrument', sorted(set(INSTRUMENT_FOR.values())))
def test_matched_scenes_validate_per_instrument(instrument: str) -> None:
    """Every scene class builds a schema-valid scene for every instrument."""
    for scene_class in _ALL_CLASSES:
        scene = matched_scene(
            f'TEST_{scene_class}', scene_class, instrument, 0.68, size_vu=_full_frame(instrument)
        )
        validate_sim_params(scene)


def test_a_binned_cohort_frame_renders_through_its_readout_mode() -> None:
    """A 256 px LORRI frame is a 4x4 full frame: the scene selects that mode at that size."""
    scene = matched_scene('L1', 'star_dominated', 'nhlorri', 1.0, size_vu=(256, 256))
    assert scene['detector']['binning'] == 4
    assert (scene['size_v'], scene['size_u']) == (256, 256)


@pytest.mark.parametrize(
    ('matching', 'expected'),
    [
        ({}, None),
        (
            {'filter_name': 'CL1+GRN', 'gain_state': 3, 'quantization': 'coiss_lut'},
            {'filter': 'CL1+GRN', 'gain_state': 3, 'quantization': 'coiss_lut'},
        ),
    ],
    ids=['unmatched', 'table-frame'],
)
def test_a_matched_scene_renders_the_frame_filter_gain_and_table(
    matching: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    """The real frame's filter, gain state, and downlink table reach the render."""
    scene = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, **matching
    )
    assert scene.get('detector') == expected


@pytest.mark.parametrize(
    ('quantization', 'on_table'),
    [('coiss_lut', True), (None, False)],
    ids=['table-frame', 'full-frame'],
)
def test_a_matched_scene_renders_on_the_cassini_table_only_for_a_table_frame(
    quantization: str | None, on_table: bool
) -> None:
    """A TABLE frame's raw render lands on the table's DN values; a 12-bit one doesn't."""
    scene = matched_scene(
        'N123', 'star_dominated', 'coiss_nac', 1.2, size_vu=(256, 256), quantization=quantization
    )
    img, _ = render_combined_model(scene)
    assert bool(np.isin(img, COISS_LUT_DN).all()) is on_table


def test_matched_scene_is_deterministic() -> None:
    """The same image_id yields byte-identical scene dicts."""
    a = matched_scene('N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL)
    b = matched_scene('N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL)
    assert a == b


def test_matched_scene_seed_varies_with_image_id() -> None:
    """Different frames draw different seeds (independent realizations)."""
    a = matched_scene('N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL)
    b = matched_scene('N456', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL)
    assert a['random_seed'] != b['random_seed']


def test_matched_limb_scene_tracks_diameter_and_phase() -> None:
    """The matched body mirrors the real body's scale and phase."""
    scene = matched_scene(
        'N1',
        'body_full_fov',
        'coiss_calib_nac',
        0.18,
        size_vu=_NAC_FULL,
        diameter_px=155.0,
        phase_angle_deg=17.7,
    )
    body = scene['bodies'][0]
    assert body['axis1'] == pytest.approx(155.0)
    assert body['phase_angle'] == pytest.approx(17.7)


def test_matched_ring_body_tracks_diameter_and_phase() -> None:
    """The ring_plus_body moon mirrors the real body's scale and phase.

    The cohort's ring frames carry small moons at every phase; a
    fixed-size fixed-phase body would land the sim widths in a different
    FOM 3 stratum than the matched real body's.
    """
    scene = matched_scene(
        'N2',
        'ring_plus_body',
        'coiss_calib_nac',
        0.68,
        size_vu=_NAC_FULL,
        diameter_px=55.0,
        phase_angle_deg=126.3,
    )
    body = scene['bodies'][0]
    assert body['axis1'] == pytest.approx(55.0)
    assert body['phase_angle'] == pytest.approx(126.3)


def test_sim_side_extraction_is_deterministic() -> None:
    """Two extractions of one matched record produce identical samples."""
    record = FrameRecord(
        image_id='DETERMINISM_PROBE',
        scene_class='star_dominated',
        instrument='coiss_calib_nac',
        exposure_sec=1.2,
        stratum='0.5s_to_5s',
        offset_vu=(0.0, 0.0),
    )
    first, _inc1 = _frame_samples_sim(record)
    second, _inc2 = _frame_samples_sim(record)
    assert sorted(first.samples) == sorted(second.samples)
    for kind, values in first.samples.items():
        assert np.array_equal(np.asarray(values), np.asarray(second.samples[kind])), kind


def test_flat_top_guard_rejects_clipped_core() -> None:
    """A plateau of core pixels at the cutout max marks a clipped star."""
    from tests.integration.sim_realism_support import _has_flat_top

    image = np.zeros((33, 33))
    image[15:18, 15:18] = 50.0
    image[16, 16] = 100.0
    assert _has_flat_top(image, 16, 16) is False
    image[16, 15:18] = 100.0  # three-pixel flat top
    assert _has_flat_top(image, 16, 16) is True


def test_flat_top_guard_survives_calibration_rescale() -> None:
    """The plateau test is invariant under a multiplicative unit change."""
    from tests.integration.sim_realism_support import _has_flat_top

    image = np.zeros((33, 33))
    image[15:18, 15:18] = 50.0
    image[16, 15:18] = 100.0
    assert _has_flat_top(image * 1.7e-4, 16, 16) is True


def test_pooled_limb_statistic_uses_copopulated_bins_only() -> None:
    """FOM 3 pooling excludes strata that only one side populates."""
    from tests.integration.sim_realism import InstrumentComparison, _aggregate

    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    comparison.real.samples['limb_width_p0_r1'] = [2.0] * 10
    comparison.sim.samples['limb_width_p0_r1'] = [2.5] * 10
    comparison.real.samples['limb_width_p1_r0'] = [9.0] * 10  # real-only stratum
    comparison.sim.samples['limb_width_p2_r2'] = [1.0] * 10  # sim-only stratum
    _aggregate(comparison)
    assert comparison.real.samples['limb_width_copop'] == [2.0] * 10
    assert comparison.sim.samples['limb_width_copop'] == [2.5] * 10
    assert comparison.limb_bins_real_only == ['limb_width_p1_r0']
    assert comparison.limb_bins_sim_only == ['limb_width_p2_r2']


def test_aggregate_emits_density_w1_for_curve_kinds() -> None:
    """Curve kinds present on both sides get a density-W1 scalar."""
    from tests.integration.sim_realism import InstrumentComparison, _aggregate

    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    x = np.linspace(0.0, 8.0, 17)
    real_y = np.exp(-0.5 * (x / 1.0) ** 2)
    sim_y = np.exp(-0.5 * (x / 1.5) ** 2)
    comparison.real.curves['star_profile'] = [(x, real_y)]
    comparison.sim.curves['star_profile'] = [(x, sim_y)]
    comparison.real.curves['sky_psd'] = [(x, real_y)]  # real-only: no scalar
    _aggregate(comparison)
    assert set(comparison.curve_divergences) == {'star_profile'}
    result = comparison.curve_divergences['star_profile']
    assert result.w1 > 0.0
    assert np.isfinite(result.w1_normalized)


def test_cohort_discovery_covers_known_instruments() -> None:
    """The committed library maps onto the expected sim instruments."""
    cohort = discover_cohort()
    assert 'coiss_calib_nac' in cohort
    assert 'nhlorri' in cohort
    for sidecars in cohort.values():
        assert sidecars == sorted(sidecars, key=lambda s: s.image_id)
