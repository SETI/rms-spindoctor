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

import random
import runpy
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from polymath import Pair

from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.obs import ObsSnapshotInst
from spindoctor.sim.instruments.resolvers import (
    resolve_calibrated_sky_if,
    resolve_dark_sky_ceiling_if,
    resolve_native_frame_size_px,
)
from spindoctor.sim.realism.noise import SkyPatch
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_combined_model
from spindoctor.sim.render.radiance.compose import sky_star_faint_cutoff_mag
from spindoctor.sim.scene.validate import validate_sim_params
from tests.integration import sim_realism, sim_realism_report, sim_realism_twin_stars
from tests.integration.sim_realism import (
    INSTRUMENT_FOR,
    FrameRecord,
    InstrumentComparison,
    RealismResults,
    SkyPairStatus,
    _aggregate,
    _frame_samples_real,
    _frame_samples_sim,
    discover_cohort,
    pair_sky,
    pool_frame_pairs,
    twins_have_sky,
)
from tests.integration.sim_realism_report import _sky_comparison, summary_dict
from tests.integration.sim_realism_scenes import matched_scene
from tests.integration.sim_realism_support import (
    FrameSamples,
    FrameSky,
    _has_flat_top,
    add_sky_samples,
    extract_pixel_samples,
)
from tests.integration.sim_realism_twin_stars import (
    catalog_twin_stars,
    twin_faint_cutoff_vmag,
    twin_star_entries,
)
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
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=[], **matching
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
        'N123',
        'star_dominated',
        'coiss_nac',
        1.2,
        size_vu=(256, 256),
        quantization=quantization,
        stars=[],
    )
    img, _ = render_combined_model(scene)
    table = resolve_detector_params(scene).sqrt_lut
    assert table is not None
    assert bool(np.isin(img, table.levels()).all()) is on_table


@pytest.mark.parametrize('stars', [[], _STARS], ids=['no-catalog-star', 'three-stars'])
def test_a_star_twin_holds_its_real_frames_catalog_stars(stars: list[dict[str, Any]]) -> None:
    """A star twin's scene holds exactly the catalog stars it is given, and none for none."""
    scene = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=stars
    )
    assert scene['stars'] == stars


def test_a_star_twin_without_a_catalog_list_is_refused() -> None:
    """A star class can't be matched without its real frame's catalog stars."""
    with pytest.raises(ValueError, match="needs its real frame's catalog star list"):
        matched_scene('N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL)


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


def test_a_twin_star_fainter_than_the_cutoff_is_left_out() -> None:
    """Of two stars either side of the cutoff, only the brighter is kept."""
    entries = _entries(
        [
            _FakeStar(v=100.0, u=100.0, vmag=15.4, unique_number=1),
            _FakeStar(v=200.0, u=200.0, vmag=15.6, unique_number=2),
        ]
    )
    assert [entry['vmag'] for entry in entries] == [15.4]


@pytest.mark.parametrize(
    ('v', 'u', 'kept'),
    [
        (-1.0, 500.0, False),
        (1024.0, 500.0, False),
        (500.0, -1.0, False),
        (500.0, 1024.0, False),
        (0.0, 500.0, True),
        (500.0, 0.0, True),
    ],
    ids=['above', 'below', 'left', 'right', 'on-the-top-edge', 'on-the-left-edge'],
)
def test_a_twin_star_moved_outside_the_frame_is_left_out(v: float, u: float, kept: bool) -> None:
    """Each frame edge drops a star whose moved position lies past it.

    Each star is predicted 10 pixels above and to the right of where it lands.  The
    star on the top edge is predicted outside the frame, and it is kept because the
    offset moves it in.
    """
    entries = _entries([_FakeStar(v=v - 10.0, u=u + 10.0)], offset_vu=(10.0, -10.0))
    assert (len(entries) == 1) is kept


@pytest.mark.parametrize(
    ('second', 'count'),
    [
        (_FakeStar(v=500.0, u=500.0), 1),
        (_FakeStar(v=500.0, u=500.0, catalog_name='tycho2'), 2),
        (_FakeStar(v=500.0, u=500.0, unique_number=8), 2),
    ],
    ids=['listed-twice', 'other-catalog', 'other-number'],
)
def test_a_twin_star_listed_twice_by_its_catalog_is_kept_once(
    second: _FakeStar, count: int
) -> None:
    """A second entry with the same catalog and catalog number is dropped; others aren't."""
    entries = _entries([_FakeStar(v=500.0, u=500.0), second])
    assert len(entries) == count


def test_twin_star_entries_are_rounded_and_named_from_the_catalog() -> None:
    """Positions and magnitudes are rounded to 0.01, and the name carries the catalog id."""
    (entry,) = _entries([_FakeStar(v=100.1234, u=200.5678, vmag=9.8765)])
    assert entry == {'name': 'UCAC4-7', 'v': 100.12, 'u': 200.57, 'vmag': 9.88}


def test_twin_star_entries_do_not_depend_on_the_catalog_order() -> None:
    """The same stars in any order give the same list."""
    stars = [
        _FakeStar(v=float(10 * i), u=float(20 * i), vmag=12.0 - i % 3, unique_number=i)
        for i in range(1, 9)
    ]
    shuffled = list(stars)
    random.Random(5).shuffle(shuffled)
    assert _entries(shuffled) == _entries(stars)


def test_the_twin_cutoff_is_the_renderers_sky_star_cutoff_for_its_clear_scene() -> None:
    """The twin cutoff is the renderer's background-sky-star cutoff for the twin's scene.

    A 400-pixel Galileo frame is binned 2x2, so the scene's read noise is the binned
    mode's and its PSF is the camera's kernel in binned pixels.
    """
    scene = matched_scene('G1', 'star_dominated', 'gossi', 2.0, size_vu=(400, 400), stars=[])
    cutoff = twin_faint_cutoff_vmag('gossi', 2.0, (400, 400))
    assert cutoff == sky_star_faint_cutoff_mag(scene)


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


def _twin_stars_from_fake_catalog(
    monkeypatch: pytest.MonkeyPatch, stars: list[_FakeStar]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run the twin's catalog query on a 1.2 s NAC frame of Saturn against a fake catalog.

    The body and ring check runs on :class:`_FakeBackplane`.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.
        stars: The fake catalog's stars in the frame.  The fake reduction returns at
            most as many of them as the config's ``stars.max_stars`` allows.

    Returns:
        The twin's scene star entries, and the faint limit the reduction was given
        under ``'faint_vmag'``.
    """
    asked: dict[str, float] = {}

    def reduce(obs: ObsSnapshotInst, config: Config, **_: Any) -> list[_FakeStar]:
        asked['faint_vmag'] = obs.star_max_usable_vmag()
        return list(stars[: config.stars.max_stars])

    monkeypatch.setattr(sim_realism_twin_stars, 'reduce_catalogs', reduce)
    monkeypatch.setattr(sim_realism_twin_stars, 'Backplane', _FakeBackplane)
    # The navigator's own faint limit is far brighter than the twin's cutoff.
    obs = SimpleNamespace(
        data_shape_vu=_NAC_FULL,
        star_max_usable_vmag=lambda: 9.5,
        closest_planet='SATURN',
        fov=SimpleNamespace(uv_shape=Pair(_NAC_FULL)),
    )
    entries = catalog_twin_stars(
        cast(ObsSnapshotInst, obs),
        instrument='coiss_calib_nac',
        exposure_sec=1.2,
        offset_vu=(0.0, 0.0),
    )
    return entries, asked


def test_the_twin_catalog_query_reaches_the_twin_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog reduction is asked for stars down to the twin cutoff."""
    _, asked = _twin_stars_from_fake_catalog(monkeypatch, [_FakeStar(v=500.0, u=500.0)])
    assert asked['faint_vmag'] == twin_faint_cutoff_vmag('coiss_calib_nac', 1.2, _NAC_FULL)


def test_the_twin_catalog_query_keeps_more_stars_than_the_navigators_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field denser than the navigator's star cap keeps every star in the twin."""
    count = int(DEFAULT_CONFIG.stars.max_stars) + 50
    stars = [_FakeStar(v=5.0 + 5.0 * i, u=500.0, unique_number=i) for i in range(count)]
    entries, _ = _twin_stars_from_fake_catalog(monkeypatch, stars)
    assert len(entries) == count


@pytest.mark.parametrize(
    ('v', 'u', 'conflicts', 'kept'),
    [
        (500.0, 304.0, '', False),
        (500.0, 304.0, 'STAR', False),
        (898.0, 500.0, 'STAR', False),
        (897.0, 500.0, 'STAR', True),
        (500.0, 500.0, 'STAR', True),
    ],
    ids=[
        'planet-within-the-margin',
        'planet-within-the-margin-beside-another-star',
        'ring-over-3-of-11-rows-beside-another-star',
        'ring-over-2-of-11-rows-beside-another-star',
        'sky-beside-another-star',
    ],
)
def test_a_twin_star_behind_a_body_or_ring_is_left_out(
    monkeypatch: pytest.MonkeyPatch, v: float, u: float, conflicts: str, kept: bool
) -> None:
    """A star whose window a body or an opaque ring covers is left out, whatever its tag.

    A star's window reaches 5 pixels from it along each axis, in 11 rows.  The star at
    u = 304 lies off the disc, but its window reaches u = 299.  The B ring hides a star
    when it covers at least a quarter of the window: 3 rows of 11 do, and 2 don't.  The
    navigator tags a star ``'STAR'`` when it overlaps another star and then skips it in
    its own body and ring check, and the twin checks such a star like any other.
    """
    star = _FakeStar(v=v, u=u, conflicts=conflicts)
    entries, _ = _twin_stars_from_fake_catalog(monkeypatch, [star])
    assert (len(entries) == 1) is kept


@pytest.mark.parametrize('stars_per_backplane', [2000, 2], ids=['one-batch', 'three-batches'])
def test_each_twin_star_reads_its_own_window(
    monkeypatch: pytest.MonkeyPatch, stars_per_backplane: int
) -> None:
    """Hidden and visible stars in mixed order keep exactly the visible ones.

    Five stars lie on Saturn's disc, open sky, open sky, the B ring, and open sky, an
    order that reads differently backward, so a flag read from the wrong star's window,
    a batch read twice, or flags returned in the wrong order keep a hidden star or drop
    a visible one.  With two stars to a backplane the five stars span three batches, the
    last one partial.
    """
    monkeypatch.setattr(sim_realism_twin_stars, '_STARS_PER_BACKPLANE', stars_per_backplane)
    stars = [
        _FakeStar(v=500.0, u=200.0, unique_number=1),
        _FakeStar(v=500.0, u=500.0, unique_number=2),
        _FakeStar(v=600.0, u=600.0, unique_number=3),
        _FakeStar(v=950.0, u=500.0, unique_number=4),
        _FakeStar(v=700.0, u=700.0, unique_number=5),
    ]
    entries, _ = _twin_stars_from_fake_catalog(monkeypatch, stars)
    assert sorted(entry['name'] for entry in entries) == ['UCAC4-2', 'UCAC4-3', 'UCAC4-5']


def test_matched_scene_is_deterministic() -> None:
    """The same image_id yields byte-identical scene dicts."""
    a = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
    b = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
    assert a == b


def test_matched_scene_seed_varies_with_image_id() -> None:
    """Different frames draw different seeds (independent realizations)."""
    a = matched_scene(
        'N123', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
    b = matched_scene(
        'N456', 'star_dominated', 'coiss_calib_nac', 1.2, size_vu=_NAC_FULL, stars=_STARS
    )
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
    assert body['semi_a_px'] == pytest.approx(77.5)
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
    assert body['semi_a_px'] == pytest.approx(27.5)
    assert body['phase_angle'] == pytest.approx(126.3)


def test_sim_side_extraction_is_deterministic() -> None:
    """Two extractions of one matched record produce identical samples and sky patches."""
    record = FrameRecord(
        image_id='DETERMINISM_PROBE',
        scene_class='star_dominated',
        instrument='coiss_calib_nac',
        exposure_sec=1.2,
        stratum='0.5s_to_5s',
        offset_vu=(0.0, 0.0),
        stars=_STARS,
    )
    first, _inc1 = _frame_samples_sim(record)
    second, _inc2 = _frame_samples_sim(record)
    assert sorted(first.samples) == sorted(second.samples)
    for kind, values in first.samples.items():
        assert np.array_equal(np.asarray(values), np.asarray(second.samples[kind])), kind
    assert first.sky is not None
    assert second.sky is not None
    assert first.sky.patches == second.sky.patches


def test_flat_top_guard_rejects_clipped_core() -> None:
    """A plateau of core pixels at the cutout max marks a clipped star."""
    image = np.zeros((33, 33))
    image[15:18, 15:18] = 50.0
    image[16, 16] = 100.0
    assert _has_flat_top(image, 16, 16) is False
    image[16, 15:18] = 100.0  # three-pixel flat top
    assert _has_flat_top(image, 16, 16) is True


def test_flat_top_guard_survives_calibration_rescale() -> None:
    """The plateau test is invariant under a multiplicative unit change."""
    image = np.zeros((33, 33))
    image[15:18, 15:18] = 50.0
    image[16, 15:18] = 100.0
    assert _has_flat_top(image * 1.7e-4, 16, 16) is True


def test_pooled_limb_statistic_uses_copopulated_bins_only() -> None:
    """FOM 3 pooling excludes strata that only one side populates."""
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


# The calibrated Cassini NAC's sky residual and dark-sky ceiling, and the WAC's sky
# residual, from their configs.
_REF = resolve_calibrated_sky_if('coiss_calib_nac')
_CEILING = cast(float, resolve_dark_sky_ceiling_if('coiss_calib_nac'))
_WAC_REF = resolve_calibrated_sky_if('coiss_calib_wac')


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


def test_a_matched_frame_on_content_leaves_both_sky_pools() -> None:
    """If a matched frame's patches all lie on a body, neither frame adds sky noise."""
    real = _sky_frame(1e-4)
    sim = _sky_frame(1e-2)
    pair_sky(_sky_record(), real, sim, twins_with_sky=True)
    assert 'sky_sigma' not in real.samples
    assert 'sky_sigma' not in sim.samples


def test_a_pair_on_content_in_both_frames_is_dropped() -> None:
    """Two frames whose patches both lie on a body are dropped, though their levels agree."""
    pair = pair_sky(
        _sky_record(), _sky_frame(_REF + 1e-2), _sky_frame(_REF + 1e-2), twins_with_sky=True
    )
    assert pair.status is SkyPairStatus.DROPPED


def test_an_offset_table_sky_is_compared_with_a_quiet_twin() -> None:
    """A real sky 2e-3 I/F above the residual is dark sky, though many sigmas above it.

    The matched frame's sky lies at the residual with a sigma of 1e-4, so a tolerance
    scaled by either frame's sigma would drop the real sky.
    """
    pair = pair_sky(
        _sky_record(),
        _sky_frame(_REF + 2e-3, sigma=1.5e-4),
        _sky_frame(_REF, sigma=1e-4),
        twins_with_sky=True,
    )
    assert pair.status is SkyPairStatus.COMPARED


def test_a_real_sky_far_below_the_residual_is_compared() -> None:
    """A real sky below the residual is compared, whatever the two frames' sigmas.

    Modeled on a long WAC star frame whose real sky lies 3.9e-5 I/F below the residual
    with a sigma of 8.7e-8, while its matched frame's sky lies at the residual with a
    sigma of 1.3e-7.
    """
    pair = pair_sky(
        _sky_record('coiss_calib_wac'),
        _sky_frame(_WAC_REF - 3.9e-5, sigma=8.7e-8),
        _sky_frame(_WAC_REF - 1.5e-8, sigma=1.3e-7),
        twins_with_sky=True,
    )
    assert pair.status is SkyPairStatus.COMPARED


def test_a_patch_at_the_ceiling_counts_as_dark_sky() -> None:
    """The ceiling is measured from the calibrated sky residual, and a patch on it is kept."""
    pair = pair_sky(
        _sky_record(), _sky_frame(_REF + _CEILING), _sky_frame(_REF), twins_with_sky=True
    )
    assert pair.status is SkyPairStatus.COMPARED


def test_a_frame_without_real_sky_adds_nothing_on_the_sim_side() -> None:
    """When the real frame has no sky patches, the matched frame's sky isn't added."""
    sim = _sky_frame(_REF)
    pair_sky(_sky_record(), FrameSamples(), sim, twins_with_sky=True)
    assert 'sky_sigma' not in sim.samples


def test_a_constant_patch_adds_no_sky_noise() -> None:
    """A patch whose sigma is 0 is left out of the pool."""
    real = _sky_frame_of([(0.0, 0.0), (_REF, 1e-4)])
    pair_sky(_sky_record(), real, _sky_frame(_REF), twins_with_sky=True)
    assert real.samples['sky_sigma'] == [1e-4]


def test_a_raw_dn_pair_adds_every_patch_unjudged() -> None:
    """On a raw-DN camera the two sides' levels can't be compared, so every patch is kept.

    The matched LORRI frame keeps the simulator's bias pedestal, about 545 DN, which the
    real bias-subtracted product doesn't have.
    """
    real = _sky_frame(2.18, sigma=1.1)
    sim = _sky_frame(544.5, sigma=1.05)
    pair_sky(_sky_record('nhlorri'), real, sim, twins_with_sky=True)
    assert 'sky_sigma' in sim.samples


def test_a_calibrated_camera_without_a_ceiling_is_not_judged() -> None:
    """A calibrated camera that doesn't set a ceiling is handled like a raw-DN one."""
    pair = pair_sky(
        _sky_record('vgiss'), _sky_frame(0.5, sigma=2e-3), _sky_frame(0.012), twins_with_sky=True
    )
    assert pair.status is SkyPairStatus.NOT_JUDGED


def test_real_sky_is_recorded_alone_when_no_twin_has_sky() -> None:
    """When no matched frame of the instrument has sky, the real sky is recorded as it is."""
    real = _sky_frame(1e-4)
    pair_sky(_sky_record(), real, FrameSamples(), twins_with_sky=False)
    assert 'sky_sigma' in real.samples


def test_real_sky_recorded_alone_must_lie_on_dark_sky() -> None:
    """When no matched frame has sky, a real frame on content still adds no sky noise."""
    real = _sky_frame(_REF + 1e-2)
    pair_sky(_sky_record(), real, FrameSamples(), twins_with_sky=False)
    assert 'sky_sigma' not in real.samples


@pytest.mark.parametrize(
    ('sims', 'expected'),
    [
        ([_sky_frame(4e-5), FrameSamples()], True),
        ([_sky_frame(0.0, sigma=0.0)], False),
    ],
    ids=['one-twin-of-two', 'constant-patch-only'],
)
def test_twins_have_sky_when_any_twin_has_a_noisy_sky_patch(
    sims: list[FrameSamples], expected: bool
) -> None:
    """One matched frame with sky is enough, and a constant patch doesn't count as sky."""
    assert twins_have_sky(sims) is expected


@pytest.mark.parametrize(
    ('real_patches', 'expected'),
    [
        ([(1e-4, 1e-4), (0.076, 1e-3)], 1e-4),
        ([(0.0, 0.0), (0.076, 1e-3)], 0.076),
    ],
    ids=['compared', 'dropped'],
)
def test_the_reported_sky_level_follows_the_patches_kept(
    real_patches: list[tuple[float, float]], expected: float
) -> None:
    """A frame reports the level of the patches it kept, or else of all its sky patches.

    The compared frame also has a patch on rings, and the dropped frame has a constant
    patch.  Neither of those patches enters the frame's reported level.
    """
    pair = pair_sky(
        _sky_record(), _sky_frame_of(real_patches), _sky_frame(_REF), twins_with_sky=True
    )
    assert pair.real_level == pytest.approx(expected)


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


def test_the_sky_spectrum_tiles_are_made_wholly_of_sky_patches() -> None:
    """Each spectrum tile lies wholly on sky patches, so none reaches brighter content.

    The darkest patches lie in the second row, beside the bright band below, so a
    tile anchored at a darkest patch would cover the band.
    """
    image = np.random.default_rng(1).normal(1.0, 1e-2, (256, 256))
    image[:32] = np.random.default_rng(2).normal(2e-4, 1e-5, (32, 256))
    image[32:64] = np.random.default_rng(3).normal(1e-4, 1e-5, (32, 256))
    samples, _rates = extract_pixel_samples(
        image, exposure_stratum='0.5s_to_5s', saturation_level=float('nan')
    )
    assert samples.sky is not None
    sky_patches = samples.sky.patches
    assert samples.sky.psd_tiles
    assert all(cell in sky_patches for cells, _ in samples.sky.psd_tiles for cell in cells)


def test_a_zero_filled_block_leaves_the_dark_sky_in_the_sky_quartile() -> None:
    """The sky is the darkest quarter of the noisy patches, whatever constant tiles the frame has.

    The zero-filled block covers 24 of the 64 patches, more than a quarter of the frame.
    """
    image = np.random.default_rng(4).normal(1e-4, 1e-5, (256, 256))
    image[:96] = 0.0
    samples, _rates = extract_pixel_samples(
        image, exposure_stratum='0.5s_to_5s', saturation_level=float('nan')
    )
    assert samples.sky is not None
    assert len(samples.sky.patches) == 10


def test_the_pools_carry_the_sky_samples_each_pair_kept() -> None:
    """Each pair's sky samples are added before its frames enter the pools."""
    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    comparison.records.append(_sky_record())
    pool_frame_pairs(comparison, [_sky_frame(1e-4)], [_sky_frame(4e-5)])
    assert comparison.real.samples['sky_sigma'] == [1e-4]


def test_a_twin_without_sky_is_dropped_when_other_twins_have_sky() -> None:
    """A real frame isn't added alone to a pool that the other pairs compare."""
    real = _sky_frame(1e-4)
    pair_sky(_sky_record(), real, FrameSamples(), twins_with_sky=True)
    assert 'sky_sigma' not in real.samples


def test_the_summary_counts_the_frames_that_added_sky_on_each_side() -> None:
    """Real frames recorded alone count on the real side and not on the sim side."""
    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    for _ in range(2):
        comparison.sky_pairs.append(
            pair_sky(_sky_record(), _sky_frame(1e-4), FrameSamples(), twins_with_sky=False)
        )
    results = RealismResults(comparisons={'coiss_calib_nac': comparison}, runtime_sec=0.0)
    summary = summary_dict(results)
    assert summary['instruments']['coiss_calib_nac']['n_sky_frames'] == {'real': 2, 'sim': 0}


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


def test_the_summary_labels_each_statistic_from_both_sides_frames() -> None:
    """Each divergence entry carries both sides' frame counts and the smaller one's label."""
    comparison = _pooled_star_comparison(3, 2)
    results = RealismResults(comparisons={'coiss_calib_nac': comparison}, runtime_sec=0.0)
    entry = summary_dict(results)['instruments']['coiss_calib_nac']['divergences']['star_ee50']
    support = {k: entry[k] for k in ('n_real_frames', 'n_sim_frames', 'support')}
    assert support == {'n_real_frames': 3, 'n_sim_frames': 2, 'support': 'limited'}


def test_a_frame_off_the_downlink_table_is_grouped_as_native() -> None:
    """A frame that wasn't downlinked through a companding table is grouped as native."""
    comparison = InstrumentComparison(instrument='coiss_calib_nac')
    comparison.sky_pairs.append(
        pair_sky(_sky_record(), _sky_frame(1e-4), _sky_frame(4e-5), twins_with_sky=True)
    )
    assert list(_sky_comparison(comparison)['by_downlink']) == ['native']


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
        runpy.run_module('tests.integration.sim_realism', run_name='__main__')
    assert calls == ['package']


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
