"""Tests of the realism match's photometric-law study, and its regression checks.

The regression checks hold the committed realism record (or the summary that
``SPINDOCTOR_TEST_REALISM_RECORD`` points to): in every phase bin with a verdict, the
default law's twins must keep the real discs' Minnaert exponent within the bin's band,
and wherever the laws separate, the default law must agree at least as well as each
other law.  The other tests build their discs in memory; some render a small twin.
Nothing here reads holdings, so the file runs in the plain unit tier.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.realism.disc_profile import (
    LunarLambertFit,
    MinnaertFit,
    equator_profile,
    select_disc_pixels,
)
from spindoctor.sim.render.bodies.photometry import shade_surface
from tests.integration import sim_realism_laws, sim_realism_laws_discs
from tests.integration.realism_regression import REGENERATE_HINT, load_record, made_at
from tests.integration.sim_realism_body_facts import TerrainCap
from tests.integration.sim_realism_laws import (
    DEFAULT_LAW,
    DiscRow,
    LawStudyInputs,
    RealDisc,
    StudyFrame,
    TwinDisc,
    compare_disc_profiles,
    law_comparison,
    minnaert_line,
    phase_bin,
    run_law_study,
)
from tests.integration.sim_realism_laws_discs import (
    cap_lit_area_px,
    disc_lit_area_px,
    light_direction_deg,
    lit_share,
)
from tests.integration.sim_realism_laws_report import (
    law_study_summary,
    record_disc_profile_problems,
    record_has_law_study,
    record_law_comparison_problems,
    write_law_figures,
)

_RECORD = load_record()
_FRAME = StudyFrame('N0000000000_1_CALIB', 'coiss_calib_nac', (0.0, 0.0), 0.1, (256, 256))
_EMPTY = np.zeros(1)


def _disc(phase_deg: float, k: float, *, body: str = 'DIONE', part: str | None = None) -> RealDisc:
    """A real disc that holds only the facts the statistics read."""
    return RealDisc(
        frame=_FRAME,
        body=body,
        part=part,
        phase_deg=phase_deg,
        diameter_px=200.0,
        center_vu=(128.0, 128.0),
        illumination_deg=90.0,
        intensity=_EMPTY,
        mu0=_EMPTY,
        mu=_EMPTY,
        minnaert=MinnaertFit(k=k, b0=0.5, k_sigma=0.001, n_pixels=1000),
        lunar_lambert=LunarLambertFit(blend=0.5, b0=0.5, blend_sigma=0.01, n_pixels=1000),
        profile=_EMPTY,
        crop=np.zeros((1, 1), dtype=np.float32),
        selection=np.ones((256, 256), dtype=bool),
    )


def _row(phase_deg: float, k: float, twin_k: dict[str, float]) -> DiscRow:
    """A disc with a twin of the given exponent under each law."""
    twins = {
        law: TwinDisc(
            law=law,
            normal_albedo=0.5,
            minnaert_k=None,
            k=value,
            profile_residual=0.05,
            limb_peak=None,
            crop=np.zeros((1, 1), dtype=np.float32),
        )
        for law, value in twin_k.items()
    }
    return DiscRow(disc=_disc(phase_deg, k), twins=twins)


@pytest.mark.parametrize(
    ('phase', 'label'), [(10.0, 'p0'), (59.9, 'p0'), (60.0, 'p1'), (119.0, 'p1'), (120.0, 'p2')]
)
def test_a_disc_takes_the_limb_strata_phase_bin(phase: float, label: str) -> None:
    """The study bins phase as the limb strata do."""
    assert phase_bin(phase) == label


def test_a_bin_sets_its_band_from_the_scatter_of_its_paired_differences() -> None:
    """The band is 2 standard errors of the robust scatter of twin less real."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d}) for d in (0.0, 0.1, 0.2)]
    check = compare_disc_profiles(rows)['p0']
    assert check.band_k == pytest.approx(2.0 * 1.4826 * 0.1 / math.sqrt(3.0))


def test_the_real_exponents_own_scatter_does_not_widen_the_band() -> None:
    """Real exponents spread over 0.2 with twins that follow them get the floor."""
    rows = [_row(30.0, k, {DEFAULT_LAW: k + 0.01}) for k in (0.5, 0.6, 0.7)]
    assert compare_disc_profiles(rows)['p0'].band_k == pytest.approx(0.02)


def test_twins_beyond_the_band_lie_outside_it() -> None:
    """A median difference larger than the band is outside it."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d}) for d in (0.2, 0.25, 0.3)]
    assert compare_disc_profiles(rows)['p0'].within_band is False


def test_twins_within_the_band_lie_within_it() -> None:
    """A median difference smaller than the band is within it."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d}) for d in (-0.1, 0.05, 0.2)]
    assert compare_disc_profiles(rows)['p0'].within_band is True


def test_a_bin_with_two_discs_is_not_judged() -> None:
    """Fewer than 3 discs give no band and no verdict."""
    rows = [_row(30.0, k, {DEFAULT_LAW: k + 0.5}) for k in (0.5, 0.6)]
    assert compare_disc_profiles(rows)['p0'].within_band is None


def test_a_tight_bin_is_held_to_the_band_floor() -> None:
    """Identical real exponents get the floor of 0.02."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6}) for _ in range(3)]
    assert compare_disc_profiles(rows)['p0'].band_k == pytest.approx(0.02)


def test_a_disc_whose_twin_has_no_exponent_is_left_out() -> None:
    """A twin that could not be fitted leaves its disc out of the check."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6}) for _ in range(3)]
    rows[0].twins[DEFAULT_LAW] = TwinDisc(DEFAULT_LAW, 0.5, None, None, None, None, _EMPTY)
    assert compare_disc_profiles(rows)['p0'].n == 2


def _comparison_rows(default_offset: float, other_offset: float) -> list[DiscRow]:
    """Three p0 discs whose default-law and Lambert twins miss by the given amounts."""
    return [
        _row(30.0, k, {DEFAULT_LAW: k + default_offset, 'lambert': k + other_offset})
        for k in (0.5, 0.6, 0.7)
    ]


def test_a_law_that_agrees_worse_by_more_than_the_band_separates() -> None:
    """Lambert missing by 0.3 against the default's 0.01 separates from it."""
    entry = law_comparison(_comparison_rows(0.01, 0.3))['p0']
    assert entry.separates['lambert'] is True


def test_the_default_law_is_best_when_the_others_agree_worse() -> None:
    """The default law agrees best when every law it separates from agrees worse."""
    entry = law_comparison(_comparison_rows(0.01, 0.3))['p0']
    assert entry.default_best is True


def test_a_law_that_agrees_better_by_more_than_the_band_beats_the_default() -> None:
    """Lambert missing by 0.01 against the default's 0.3 beats it."""
    entry = law_comparison(_comparison_rows(0.3, 0.01))['p0']
    assert entry.default_best is False


def test_laws_within_the_band_of_each_other_do_not_separate() -> None:
    """Two laws whose agreement differs by less than the band claim no order."""
    entry = law_comparison(_comparison_rows(0.05, 0.04))['p0']
    assert entry.separates['lambert'] is False


def test_laws_within_the_band_leave_the_default_best() -> None:
    """A slightly better law that does not separate leaves the default best."""
    entry = law_comparison(_comparison_rows(0.05, 0.04))['p0']
    assert entry.default_best is True


def test_the_law_comparison_takes_its_band_from_the_default_laws_pairs() -> None:
    """The band is the disc-profile band of the default law's paired differences."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d, 'lambert': 0.9}) for d in (0.0, 0.1, 0.2)]
    entry = law_comparison(rows)['p0']
    assert entry.band_k == pytest.approx(2.0 * 1.4826 * 0.1 / math.sqrt(3.0))


def test_a_bin_with_two_discs_has_no_law_comparison() -> None:
    """Fewer than 3 discs give no comparison."""
    rows = _comparison_rows(0.01, 0.3)[:2]
    assert law_comparison(rows) == {}


def test_the_minnaert_line_runs_through_the_real_exponents() -> None:
    """Exponents on a straight line give back that line."""
    discs = [_disc(phase, 0.5 + 0.004 * phase) for phase in (10.0, 40.0, 90.0)]
    k0, k1 = minnaert_line(discs)
    assert (k0, k1) == pytest.approx((0.5, 0.004))


def test_the_minnaert_line_reads_a_generator_once() -> None:
    """Discs handed over as a generator give the same line."""
    k0, k1 = minnaert_line(_disc(phase, 0.5 + 0.004 * phase) for phase in (10.0, 40.0, 90.0))
    assert (k0, k1) == pytest.approx((0.5, 0.004))


def test_a_limb_peak_attaches_to_the_whole_disc_of_its_body() -> None:
    """A frame's one limb gradient-peak sample goes to the disc of the measured body."""
    inputs = LawStudyInputs(discs=[_disc(30.0, 0.6), _disc(30.0, 0.6, body='RHEA')])
    inputs.add_limb_peak(_FRAME.image_id, 'DIONE', {'limb_peak_p0_r1': [0.12], 'other': [1.0]})
    assert inputs.discs[0].limb_peak == ('limb_peak_p0_r1', 0.12)


def test_a_limb_peak_leaves_the_other_bodies_alone() -> None:
    """The disc of another body in the frame gets no limb gradient-peak sample."""
    inputs = LawStudyInputs(discs=[_disc(30.0, 0.6), _disc(30.0, 0.6, body='RHEA')])
    inputs.add_limb_peak(_FRAME.image_id, 'DIONE', {'limb_peak_p0_r1': [0.12]})
    assert inputs.discs[1].limb_peak is None


def test_a_limb_peak_leaves_a_terrain_alone() -> None:
    """A terrain of the measured body gets no limb gradient-peak sample."""
    inputs = LawStudyInputs(discs=[_disc(30.0, 0.6, part='dark terrain')])
    inputs.add_limb_peak(_FRAME.image_id, 'DIONE', {'limb_peak_p0_r1': [0.12]})
    assert inputs.discs[0].limb_peak is None


def test_a_frame_with_two_limb_peak_kinds_attaches_none() -> None:
    """A frame whose samples hold two limb gradient-peak kinds attaches neither."""
    inputs = LawStudyInputs(discs=[_disc(30.0, 0.6)])
    inputs.add_limb_peak(
        _FRAME.image_id, 'DIONE', {'limb_peak_p0_r1': [0.12], 'limb_peak_p0_r2': [0.1]}
    )
    assert inputs.discs[0].limb_peak is None


def test_the_light_direction_comes_back_from_the_cosines() -> None:
    """A sphere lit from 70 deg gives back 70 deg."""
    on_body, mu0, _mu = sim_realism_laws._sphere_cosines(
        (120, 120), (60.0, 60.0), 50.0, illumination_deg=70.0, phase_deg=35.0
    )
    found = light_direction_deg(on_body, mu0, (60.0, 60.0), 50.0)
    assert found == pytest.approx(70.0, abs=1e-6)


def test_a_disc_wholly_in_the_frame_has_its_whole_lit_share() -> None:
    """A lit disc of the expected area has a lit share near 1."""
    on_body, mu0, _mu = sim_realism_laws._sphere_cosines(
        (220, 220), (110.0, 110.0), 100.0, illumination_deg=0.0, phase_deg=60.0
    )
    share = lit_share(on_body & (mu0 > 0.0), disc_lit_area_px(100.0, 60.0))
    assert share == pytest.approx(1.0, abs=0.01)


def test_a_disc_cut_by_the_frame_has_a_smaller_lit_share() -> None:
    """Keeping one quarter of the lit disc gives a share near a quarter."""
    on_body, mu0, _mu = sim_realism_laws._sphere_cosines(
        (220, 220), (110.0, 110.0), 100.0, illumination_deg=0.0, phase_deg=0.0
    )
    lit = on_body & (mu0 > 0.0)
    lit[:110, :] = False
    lit[:, :110] = False
    share = lit_share(lit, disc_lit_area_px(100.0, 0.0))
    assert share == pytest.approx(0.25, abs=0.01)


def test_a_cap_over_the_whole_sphere_has_the_lit_discs_area() -> None:
    """A cap of 180 deg is the whole sphere, whose lit area the phase alone sets."""
    cap = TerrainCap('everything', 0.0, 0.0, 180.0)
    area = cap_lit_area_px(cap, 100.0, sub_observer_deg=(0.0, 0.0), sub_solar_deg=(0.0, 60.0))
    assert area == pytest.approx(disc_lit_area_px(100.0, 60.0), rel=0.001)


def _cap_pixels(cap: TerrainCap) -> Any:
    """The pixels of a cap on a sphere of radius 100 px seen from (0, 0), lit head-on.

    The image's top is north and west longitude grows to the right.
    """
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (220, 220), (110.0, 110.0), 100.0, illumination_deg=0.0, phase_deg=0.0
    )
    rows, cols = np.indices(on_body.shape, dtype=np.float64)
    normal_v = (rows + 0.5 - 110.0) / 100.0
    normal_u = (cols + 0.5 - 110.0) / 100.0
    lat = np.degrees(np.arcsin(np.clip(-normal_v, -1.0, 1.0)))
    west = np.degrees(np.arctan2(normal_u, np.where(on_body, mu, 0.0))) % 360.0
    return on_body & (mu0 > 0.0) & cap.contains(lat, west)


def test_a_cap_wholly_in_the_frame_has_its_whole_lit_share() -> None:
    """A cap off the disc's center, all of it in the frame, has a lit share near 1."""
    cap = TerrainCap('dark terrain', 20.0, 40.0, 40.0)
    expected = cap_lit_area_px(cap, 100.0, sub_observer_deg=(0.0, 0.0), sub_solar_deg=(0.0, 0.0))
    assert lit_share(_cap_pixels(cap), expected) == pytest.approx(1.0, abs=0.02)


def test_a_cap_cut_by_the_frame_has_a_smaller_lit_share() -> None:
    """Keeping the cap's pixels in the left half of the image gives a smaller share."""
    cap = TerrainCap('dark terrain', 0.0, 0.0, 40.0)
    expected = cap_lit_area_px(cap, 100.0, sub_observer_deg=(0.0, 0.0), sub_solar_deg=(0.0, 0.0))
    pixels = _cap_pixels(cap)
    pixels[:, 110:] = False
    assert lit_share(pixels, expected) == pytest.approx(0.5, abs=0.02)


def test_a_cap_holds_the_points_within_its_radius() -> None:
    """A point 39 deg from the cap's center lies inside a cap of 40 deg."""
    cap = TerrainCap('dark terrain', 0.0, 90.0, 40.0)
    assert bool(cap.contains(np.array([39.0]), np.array([90.0]))[0])


def test_a_cap_leaves_out_the_points_beyond_its_radius() -> None:
    """A point 41 deg from the cap's center along the equator lies outside it."""
    cap = TerrainCap('dark terrain', 0.0, 90.0, 40.0)
    assert not bool(cap.contains(np.array([0.0]), np.array([131.0]))[0])


def test_a_large_disc_keeps_at_most_sixty_thousand_pixels() -> None:
    """Every third pixel of a 160,000-pixel selection is kept, at most 60,000 of them."""
    mask = np.ones((400, 400), dtype=bool)
    kept = int(sim_realism_laws.decimate(mask).sum())
    assert kept == 53_334


def _hiding_backplane() -> tuple[Any, dict[str, dict[str, Any]]]:
    """A frame holding a near and a far body whose boxes overlap, rings, and a shadow."""
    near = np.zeros((40, 40), dtype=bool)
    near[10:20, 10:20] = True
    far = np.zeros((40, 40), dtype=bool)
    far[15:30, 15:30] = True
    rings = np.zeros((40, 40), dtype=bool)
    rings[28, 28] = True
    shadow = np.zeros((40, 40), dtype=bool)
    shadow[29, 15] = True
    masks = {'NEAR': near, 'FAR': far}
    backplane = SimpleNamespace(
        where_intercepted=lambda name: SimpleNamespace(vals=masks[name]),
        where_in_back=lambda name, ring: SimpleNamespace(
            vals=rings if (name, ring) == ('FAR', 'SATURN_MAIN_RINGS') else rings & False
        ),
        where_inside_shadow=lambda name, planet: SimpleNamespace(
            vals=shadow if (name, planet) == ('FAR', 'SATURN') else shadow & False
        ),
    )
    box = {'u_min': 10, 'u_max': 30, 'v_min': 10, 'v_max': 30}
    entries = {'NEAR': {**box, 'range': 1.0e6}, 'FAR': {**box, 'range': 2.0e6}}
    return backplane, entries


def _hidden(body: str, planet: str | None) -> Any:
    """The pixels of one body of the frame above that something hides."""
    backplane, entries = _hiding_backplane()
    return sim_realism_laws_discs._hidden_pixels(
        backplane, body, entries[body], list(entries.items()), planet=planet
    )


def test_a_nearer_body_hides_its_pixels_and_a_margin() -> None:
    """The far body loses the near body's pixels, widened by the disc clearance."""
    assert _hidden('FAR', None)[22, 15]


def test_a_farther_body_hides_nothing() -> None:
    """The near body loses no pixels to the far body behind it."""
    assert not _hidden('NEAR', None).any()


def test_the_planets_main_rings_hide_the_pixels_behind_them() -> None:
    """The far body loses the pixels the rings lie in front of, with a margin."""
    assert _hidden('FAR', 'SATURN')[28, 31]


def test_the_planets_shadow_hides_its_pixels() -> None:
    """The far body loses the pixels in the planet's shadow, with a margin."""
    assert _hidden('FAR', 'SATURN')[32, 15]


def _sphere_disc(law: str, **kwargs: float) -> RealDisc:
    """A real disc drawn from a sphere under one law, in the twin's own geometry."""
    phase = 30.0
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (256, 256), (128.0, 128.0), 60.0, illumination_deg=90.0, phase_deg=phase
    )
    keep = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    intensity = 0.05 * shade_surface(
        mu0[keep], mu[keep], law=law, phase_angle=math.radians(phase), **kwargs
    )
    disc = _disc(phase, 1.0)
    disc.diameter_px = 120.0
    disc.intensity = intensity
    disc.mu0 = mu0[keep]
    disc.mu = mu[keep]
    disc.profile = equator_profile(intensity, mu0[keep], mu[keep], math.radians(phase))
    disc.selection = keep
    return disc


def test_a_lambert_twin_fits_the_exponent_one() -> None:
    """A rendered Lambert twin, blurred and noisy as the camera makes it, fits k near 1."""
    twin = sim_realism_laws._twin(
        _sphere_disc('lambert'), 'lambert', minnaert_k=0.6, limb_peak=False
    )
    assert twin.k == pytest.approx(1.0, abs=0.03)


def test_a_minnaert_twin_takes_the_exponent_it_is_given() -> None:
    """A rendered Minnaert twin fits the exponent it was rendered with."""
    twin = sim_realism_laws._twin(
        _sphere_disc('minnaert', minnaert_k=0.6), 'minnaert', minnaert_k=0.6, limb_peak=False
    )
    assert twin.k == pytest.approx(0.6, abs=0.03)


def test_a_twin_is_fitted_only_over_the_real_discs_pixels() -> None:
    """A real disc that counted 50 pixels leaves its twin too few to fit."""
    disc = _sphere_disc('lambert')
    rows, cols = np.nonzero(disc.selection)
    disc.selection = np.zeros_like(disc.selection)
    disc.selection[rows[:50], cols[:50]] = True
    twin = sim_realism_laws._twin(disc, 'lambert', minnaert_k=0.6, limb_peak=False)
    assert twin.k is None


def test_a_twin_takes_its_albedo_from_the_real_disc_under_its_own_law() -> None:
    """The twin's albedo is the real disc's measured against the twin's law."""
    twin = sim_realism_laws._twin(
        _sphere_disc('lambert'), 'lambert', minnaert_k=0.6, limb_peak=False
    )
    assert twin.normal_albedo == pytest.approx(0.05)


def _record(within: bool | None, agreement: dict[str, float], separates: bool) -> dict[str, Any]:
    """A record holding one disc-profile bin and one law-comparison bin."""
    return {
        'instruments': {
            'coiss_calib_nac': {
                'disc_profiles': {
                    'p0': {
                        'n': 4,
                        'real_median_k': 0.6,
                        'sim_median_k': 0.7,
                        'median_delta_k': 0.1,
                        'delta_scatter_k': 0.05,
                        'band_k': 0.05,
                        'within_band': within,
                    }
                }
            }
        },
        'law_study': {
            'default_law': 'lunar_lambert',
            'comparison': {
                'p0': {
                    'n': 4,
                    'band_k': 0.05,
                    'agreement': agreement,
                    'separates': {'lambert': separates},
                    'default_best': True,
                }
            },
        },
    }


def test_a_record_bin_outside_its_band_is_a_problem() -> None:
    """A disc-profile bin whose verdict is False is reported."""
    problems = record_disc_profile_problems(_record(False, {}, False))
    assert problems == [
        "coiss_calib_nac p0: the twins' Minnaert exponents differ from the real ones by a "
        'median +0.100, more than the band of 0.050 (4 discs)'
    ]


@pytest.mark.parametrize('within', [True, None])
def test_a_record_bin_within_or_without_a_band_is_no_problem(within: bool | None) -> None:
    """A disc-profile bin within its band, or without a verdict, is not reported."""
    assert record_disc_profile_problems(_record(within, {}, False)) == []


def test_a_law_that_separates_and_agrees_better_is_a_problem() -> None:
    """A law better than the default by more than the band is reported."""
    record = _record(True, {'lunar_lambert': 0.2, 'lambert': 0.05}, True)
    assert record_law_comparison_problems(record) == [
        'p0: lambert agrees with the real discs to a median |dk| of 0.050, better than '
        "lunar_lambert's 0.200 by more than the band of 0.050 (4 discs)"
    ]


def test_a_better_law_that_does_not_separate_is_no_problem() -> None:
    """A slightly better law within the band is not reported."""
    record = _record(True, {'lunar_lambert': 0.2, 'lambert': 0.05}, False)
    assert record_law_comparison_problems(record) == []


def test_a_worse_law_that_separates_is_no_problem() -> None:
    """A law that separates by agreeing worse is not reported."""
    record = _record(True, {'lunar_lambert': 0.05, 'lambert': 0.2}, True)
    assert record_law_comparison_problems(record) == []


@pytest.mark.skipif(
    not record_has_law_study(_RECORD),
    reason='the realism record predates the photometric-law study',
)
def test_the_recorded_disc_profiles_lie_within_their_bands() -> None:
    """In every bin with a verdict, the default law's twins keep the real exponents."""
    problems = record_disc_profile_problems(_RECORD)
    assert not problems, (
        f'The disc profiles of the record made at {made_at(_RECORD)} leave their bands:\n'
        + '\n'.join(problems)
        + f'\n{REGENERATE_HINT}'
    )


@pytest.mark.skipif(
    not record_has_law_study(_RECORD),
    reason='the realism record predates the photometric-law study',
)
def test_the_default_law_agrees_best_where_the_laws_separate() -> None:
    """Wherever another law separates from the default law, it agrees worse."""
    problems = record_law_comparison_problems(_RECORD)
    assert not problems, (
        f'In the record made at {made_at(_RECORD)} another law fits the real discs '
        'better than the default:\n' + '\n'.join(problems) + f'\n{REGENERATE_HINT}'
    )


def test_the_study_writes_its_record_block_and_figures(tmp_path: Path) -> None:
    """A study of three small discs writes its record block and its three figures."""
    discs = [_sphere_disc('lunar_lambert') for _ in range(3)]
    for disc, phase in zip(discs, (20.0, 30.0, 40.0), strict=True):
        disc.phase_deg = phase
    study = run_law_study(LawStudyInputs(discs=discs))
    summary = law_study_summary(study)
    written = write_law_figures(study, tmp_path)
    assert len(summary['discs']) == 3
    assert [p.name for p in written if p.exists()] == [
        'realism_law_fits.png',
        'realism_disc_profiles.png',
        'realism_law_sheet.png',
    ]
