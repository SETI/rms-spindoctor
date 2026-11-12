"""Integration tests for the realism-match runner.

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

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.detector.params import resolve_detector_params
from spindoctor.sim.instrument_resolvers import resolve_native_frame_size_px
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import validate_sim_params
from tests.integration.sim_realism import (
    INSTRUMENT_FOR,
    FrameRecord,
    _frame_samples_real,
    _frame_samples_sim,
    discover_cohort,
)
from tests.integration.sim_realism_scenes import matched_scene
from tests.spindoctor.inst.conftest import VicarLabelStandIn

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
    """A full unbinned frame's shape for ``instrument``: its native size on a side."""
    size = resolve_native_frame_size_px(instrument)
    assert size is not None, f'{instrument} states no native_frame_size_px'
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
            {'filter_name': 'CL1+GRN', 'gain_state': 3, 'quantization': 'sqrt_lut'},
            {'filter': 'CL1+GRN', 'gain_state': 3, 'quantization': 'sqrt_lut'},
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
    [('sqrt_lut', True), (None, False)],
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
    table = resolve_detector_params(scene).sqrt_lut
    assert table is not None
    assert bool(np.isin(img, table.levels()).all()) is on_table


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


def test_a_content_filled_frame_leaves_both_sky_pools() -> None:
    """A real frame whose sky patches all lie on content adds no sky noise on either side."""
    from spindoctor.sim.realism.noise import SkyPatch
    from tests.integration.sim_realism import pair_sky
    from tests.integration.sim_realism_support import FrameSamples, FrameSky

    def frame_at(mean: float) -> FrameSamples:
        patch = SkyPatch(v0=0, u0=0, size=32, mean=mean, sigma=1e-4)
        sky = FrameSky(patches=[patch], floor=0.0, psd_freq=np.zeros(16), psd_tiles=[])
        return FrameSamples(sky=sky)

    record = FrameRecord(
        image_id='N1',
        scene_class='ring_only_flat',
        instrument='coiss_calib_nac',
        exposure_sec=0.68,
        stratum='0.5s_to_5s',
        offset_vu=(0.0, 0.0),
    )
    real = frame_at(0.07)
    sim = frame_at(3e-4)
    pair_sky(record, real, sim)
    assert 'sky_sigma' not in real.samples
    assert 'sky_sigma' not in sim.samples


def test_cohort_discovery_covers_known_instruments() -> None:
    """The committed library maps onto the expected sim instruments."""
    cohort = discover_cohort()
    assert 'coiss_calib_nac' in cohort
    assert 'nhlorri' in cohort
    for sidecars in cohort.values():
        assert sidecars == sorted(sidecars, key=lambda s: s.image_id)


@pytest.mark.parametrize(
    ('label_items', 'quantization'),
    [
        ({'DATA_CONVERSION_TYPE': 'TABLE'}, 'sqrt_lut'),
        ({'DATA_CONVERSION_TYPE': '12BIT'}, None),
        ({}, None),
    ],
    ids=['table', '12-bit', 'without-the-item'],
)
def test_a_calibrated_frame_reads_its_downlink_table_from_its_vicar_label(
    monkeypatch: pytest.MonkeyPatch, label_items: dict[str, str], quantization: str | None
) -> None:
    """Only a TABLE frame asks for sqrt_lut, and a label without the item reads as 12-bit.

    The real observation is a stand-in whose label refuses a lookup without a default, as
    rms-vicar's does, and feature extraction is skipped, so no holdings are read.
    """
    sidecar = discover_cohort()['coiss_calib_nac'][0]
    frame = SimpleNamespace(
        dict=VicarLabelStandIn(label_items),
        filter1='CL1',
        filter2='CL2',
        gain_mode=2,
        data=np.zeros((64, 64)),
        data_shape_vu=(64, 64),
        texp=1.0,
    )
    monkeypatch.setattr('tests.integration.sim_realism._load_real_obs', lambda _sidecar: frame)
    monkeypatch.setattr('tests.integration.sim_realism.MODELS_FOR_CLASS', {})
    _, record, _ = _frame_samples_real(sidecar, 'coiss_calib_nac')
    assert record.quantization == quantization
