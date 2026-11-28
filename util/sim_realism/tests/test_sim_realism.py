"""Tests of the realism-match runner's scenes, twin stars, sky pairing, and summary.

The tests render simulated frames and need no holdings, SPICE, or catalogs.  CI does
not run the file.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.inst.conftest import VicarLabelStandIn
from util.sim_realism import sim_realism, sim_realism_report
from util.sim_realism.noise import SkyPatch
from util.sim_realism.sim_realism import (
    INSTRUMENT_FOR,
    FrameRecord,
    InstrumentComparison,
    _aggregate,
    _frame_samples_real,
    discover_cohort,
    pool_frame_pairs,
)
from util.sim_realism.sim_realism_scenes import matched_scene
from util.sim_realism.sim_realism_sky import pair_sky
from util.sim_realism.sim_realism_support import (
    FrameSamples,
    FrameSky,
    add_sky_samples,
)
from util.sim_realism.sim_realism_twin_stars import (
    twin_star_entries,
)

from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.instruments.resolvers import (
    resolve_calibrated_sky_if,
    resolve_native_frame_size_px,
)
from spindoctor.sim.scene.validate import validate_sim_params

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

# A star twin's catalog stars, as the real side builds them.
_STARS: list[dict[str, Any]] = [
    {'name': 'UCAC4-1', 'v': 300.0, 'u': 400.0, 'vmag': 6.5},
    {'name': 'UCAC4-2', 'v': 650.0, 'u': 120.0, 'vmag': 7.2},
    {'name': 'UCAC4-3', 'v': 820.0, 'u': 700.0, 'vmag': 8.0},
]


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
            f'TEST_{scene_class}',
            scene_class,
            instrument,
            0.68,
            size_vu=_full_frame(instrument),
            stars=_STARS,
        )
        validate_sim_params(scene)


def test_a_binned_cohort_frame_renders_through_its_readout_mode() -> None:
    """A 256 px LORRI frame is a 4x4 full frame: the scene selects that mode at that size."""
    scene = matched_scene('L1', 'star_dominated', 'nhlorri', 1.0, size_vu=(256, 256), stars=[])
    assert scene['detector']['binning'] == 4
    assert (scene['size_v'], scene['size_u']) == (256, 256)


@dataclass
class _FakeStar:
    """A navigator catalog star record carrying the fields a twin's star list reads."""

    v: float
    u: float
    vmag: float | None = 9.0
    conflicts: str = ''
    unique_number: int | None = 7
    catalog_name: str = 'ucac4'


def _entries(stars: list[_FakeStar], offset_vu: tuple[float, float] = (0.0, 0.0)) -> list[Any]:
    """The twin entries of ``stars`` in a 1024-pixel frame with a V 15.5 cutoff."""
    return twin_star_entries(stars, offset_vu=offset_vu, size_vu=_NAC_FULL, faint_cutoff_vmag=15.5)


def test_a_twin_star_sits_at_its_prediction_moved_by_the_verified_offset() -> None:
    """A star predicted at (v, u) appears at (v + dv, u + du)."""
    (entry,) = _entries([_FakeStar(v=100.0, u=200.0)], offset_vu=(3.5, -33.75))
    assert (entry['v'], entry['u']) == (103.5, 166.25)


class _FakeBackplane:
    """A backplane over a fake Saturn frame, read at the meshgrid's samples.

    Saturn's disc covers every sample left of u = 300, and the B ring covers every
    sample past v = 900.  Every other sample lies on the ring plane outside the opaque
    annuli, at 150,000 km, where nothing hides a star.  The answers depend only on
    where a sample lies, never on what the star records carry.
    """

    def __init__(self, obs: Any, meshgrid: Any) -> None:
        del obs
        uv = np.asarray(meshgrid.uv.vals, dtype=np.float64)
        self._u = uv[..., 0]
        self._v = uv[..., 1]

    def where_intercepted(self, body_name: str) -> SimpleNamespace:
        """Saturn's disc, and no other body."""
        on_disc = self._u < 300.0 if body_name == 'SATURN' else np.zeros(self._u.shape, bool)
        return SimpleNamespace(vals=on_disc)

    def ring_radius(self, ring_target: str) -> SimpleNamespace:
        """The B ring's radius past v = 900, and 150,000 km short of it."""
        del ring_target
        return SimpleNamespace(vals=np.where(self._v > 900.0, 100_000.0, 150_000.0), mask=False)


def test_matched_scene_is_deterministic() -> None:
    """The same image_id yields byte-identical scene dicts."""
    a = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
    b = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
    assert a == b


def test_a_matched_frame_is_lit_from_its_real_bodys_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matched body takes the light direction measured on the real body."""
    monkeypatch.setattr(ObsSim, 'from_file', staticmethod(lambda _path, sim_params: sim_params))
    record = FrameRecord(
        image_id='N1',
        scene_class='body_full_fov',
        instrument='coiss_calib_nac',
        exposure_sec=0.5,
        stratum='0.5s_to_5s',
        offset_vu=(0.0, 0.0),
        frame_shape_vu=(256, 256),
        light_deg=123.0,
    )
    _obs, scene = sim_realism._sim_obs_for_record(record)
    assert scene['bodies'][0]['illumination_angle'] == 123.0


def test_aggregate_emits_density_w1_for_curve_kinds() -> None:
    """Curve kinds present on both sides get a density-W1 scalar."""
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


# The calibrated Cassini NAC's sky residual, from its config.
_REF = resolve_calibrated_sky_if('coiss_calib_nac')


def _sky_frame_of(patches: list[tuple[float, float]]) -> FrameSamples:
    """A frame's samples holding one sky patch per ``(mean, sigma)``."""
    sky_patches = [
        SkyPatch(v0=0, u0=32 * i, size=32, mean=mean, sigma=sigma)
        for i, (mean, sigma) in enumerate(patches)
    ]
    sky = FrameSky(patches=sky_patches, floor=0.0, psd_freq=np.zeros(16), psd_tiles=[])
    return FrameSamples(sky=sky)


def _sky_frame(mean: float, sigma: float = 1e-4) -> FrameSamples:
    """A frame's samples holding one sky patch at ``mean`` with noise ``sigma``."""
    return _sky_frame_of([(mean, sigma)])


def _sky_record(instrument: str = 'coiss_calib_nac') -> FrameRecord:
    """A frame record on ``instrument`` for the sky-pairing tests."""
    return FrameRecord(
        image_id='N1',
        scene_class='ring_only_flat',
        instrument=instrument,
        exposure_sec=0.68,
        stratum='0.5s_to_5s',
        offset_vu=(0.0, 0.0),
    )


def test_a_dark_sky_pair_adds_sky_noise_on_both_sides() -> None:
    """A pair whose patches both lie on dark sky adds its sky noise to both pools."""
    real = _sky_frame(1e-4)
    sim = _sky_frame(4e-5)
    pair_sky(_sky_record(), real, sim, twins_with_sky=True)
    assert 'sky_sigma' in real.samples
    assert 'sky_sigma' in sim.samples


def test_a_noisy_content_frame_leaves_both_sky_pools() -> None:
    """A real frame on content is dropped however noisy its patches are.

    The patches sit 5e-3 I/F above the residual with a sigma of 3e-3, like the halo
    sky around a bright moon, so a tolerance scaled by the real frame's sigma would
    keep them.
    """
    real = _sky_frame(_REF + 5e-3, sigma=3e-3)
    sim = _sky_frame(_REF)
    pair_sky(_sky_record(), real, sim, twins_with_sky=True)
    assert 'sky_sigma' not in real.samples
    assert 'sky_sigma' not in sim.samples


def _grid_patch(v0: int, u0: int, mean: float = _REF) -> SkyPatch:
    """A 32-px sky patch at ``(v0, u0)`` with mean ``mean``."""
    return SkyPatch(v0=v0, u0=u0, size=32, mean=mean, sigma=1e-4)


def test_a_tile_enters_the_sky_spectrum_only_when_all_its_patches_were_kept() -> None:
    """A tile whose anchor patch was kept but whose neighbor lies on a ring is left out."""
    corner, top, ring = _grid_patch(0, 0), _grid_patch(0, 32), _grid_patch(0, 64, mean=0.076)
    left, middle, right = _grid_patch(32, 0), _grid_patch(32, 32), _grid_patch(32, 64)
    kept = [corner, top, left, middle, right]
    whole_power = np.linspace(1.0, 0.0, 16)
    whole = (corner, top, left, middle)
    beside_ring = (top, ring, middle, right)
    sky = FrameSky(
        patches=[*kept, ring],
        floor=0.0,
        psd_freq=np.linspace(0.0, 0.5, 16),
        psd_tiles=[(whole, whole_power), (beside_ring, np.ones(16))],
    )
    frame = FrameSamples(sky=sky)
    add_sky_samples(frame, kept)
    assert np.array_equal(frame.curves['sky_psd'][0][1], whole_power)


def _pooled_star_comparison(n_real: int, n_sim: int) -> InstrumentComparison:
    """An aggregated comparison of three frame pairs.

    The first ``n_real`` real frames and the first ``n_sim`` matched frames add EE50
    samples.
    """
    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    reals = [FrameSamples(samples={'star_ee50': [1.0] * 4} if i < n_real else {}) for i in range(3)]
    sims = [FrameSamples(samples={'star_ee50': [1.2] * 4} if i < n_sim else {}) for i in range(3)]
    comparison.records.extend(_sky_record() for _ in range(3))
    pool_frame_pairs(comparison, reals, sims)
    _aggregate(comparison)
    return comparison


def test_a_statistic_without_simulated_frames_is_unsupported() -> None:
    """Real frames alone don't support a statistic: support counts both sides."""
    comparison = _pooled_star_comparison(3, 0)
    assert comparison.fom_support['fom2_psf'] == 'unsupported'


def test_the_command_line_runs_the_package_copy_of_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run as a script, the module calls the main of its copy imported by package name.

    The report module counts frame pairs by their status class, which it imports by
    package name, so a run that built them from the ``__main__`` copy counted none.
    """

    def refuse_to_run() -> None:
        raise AssertionError('the __main__ copy of the module started a realism match')

    calls: list[str] = []
    monkeypatch.setattr(sim_realism, 'main', lambda: calls.append('package'))
    # Should the script run its own copy's main instead, that main must stop before it
    # renders anything or writes the tracked figures and summary.  Its first step after
    # parsing the command line is to read the code provenance.
    monkeypatch.setattr(sim_realism_report, 'require_code_provenance', refuse_to_run)
    monkeypatch.setattr(sys, 'argv', ['sim_realism'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        runpy.run_module('util.sim_realism.sim_realism', run_name='__main__')
    assert calls == ['package']


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
    monkeypatch.setattr('util.sim_realism.sim_realism._load_real_obs', lambda _sidecar: frame)
    monkeypatch.setattr('util.sim_realism.sim_realism.MODELS_FOR_CLASS', {})
    monkeypatch.setattr(
        'util.sim_realism.sim_realism_frames_cassini.cassini_clipped_pixels', lambda *_args: None
    )
    # The frame's archive path feeds only the clipped-pixel read stubbed above, so no
    # holdings root is needed.
    monkeypatch.setattr('util.sim_realism.sim_realism._resolve_image_url', lambda url: url)
    _, record, _ = _frame_samples_real(sidecar, 'coiss_calib_nac')
    assert record.quantization == quantization
