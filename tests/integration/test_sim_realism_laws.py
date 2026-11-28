"""Tests of the realism match's photometric-law study, and its regression checks.

The regression checks hold the committed realism record (or the summary that
``SPINDOCTOR_TEST_REALISM_RECORD`` points to).  The record must hold the study, made
with the code's default law, with at least one judged bin.  In every phase bin with a
verdict, the default law's twins must keep the real discs' Minnaert exponent within the
bin's band, apart from the bins :data:`KNOWN_DISC_PROFILE_FINDINGS` lists, which must
still fail; and wherever the laws separate, the default law must agree at least as well
as each other law.  The other tests build their discs in memory; some render a small
twin.  Nothing here reads holdings, so the file runs in the plain unit tier.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.obs.obs_snapshot import ObsSnapshot
from spindoctor.sim.realism.disc_profile import (
    LunarLambertFit,
    MinnaertFit,
    equator_profile,
    select_disc_pixels,
)
from spindoctor.sim.render.bodies.photometry import PHOTOMETRIC_LAW_DEFAULT, shade_surface
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType
from tests.integration import sim_realism_laws, sim_realism_laws_discs, sim_realism_laws_report
from tests.integration.realism_regression import REGENERATE_HINT, load_record, made_at
from tests.integration.sim_realism_body_facts import TerrainCap
from tests.integration.sim_realism_laws import (
    DEFAULT_LAW,
    DiscProfileBin,
    DiscRow,
    ExcludedDisc,
    LawComparisonBin,
    LawStudyInputs,
    RealDisc,
    StudyFrame,
    TwinDisc,
    compare_disc_profiles,
    crop_image,
    law_comparison,
    minnaert_line,
    minnaert_twin_k,
    phase_bin,
    run_law_study,
)
from tests.integration.sim_realism_laws_discs import (
    box_in_fov_at_offset,
    cap_lit_area_px,
    disc_lit_area_px,
    light_direction_deg,
    lit_share,
)
from tests.integration.sim_realism_laws_report import (
    disc_profile_entries,
    law_study_summary,
    record_disc_profile_failures,
    record_disc_profile_verdicts,
    record_has_law_study,
    record_law_comparison_problems,
    write_law_figures,
)

_FRAME = StudyFrame('N0000000000_1_CALIB', 'coiss_calib_nac', (0.0, 0.0), 0.1, (256, 256))
_EMPTY = np.zeros(1)
_COMPARED = (DEFAULT_LAW, 'lambert')

KNOWN_DISC_PROFILE_FINDINGS: dict[tuple[str, str], str] = {}
"""Disc-profile bins of the record known to lie outside their band, with the finding."""


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
        crop_bounds=(0, 1, 0, 1),
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
            real_k=k,
            profile_residual=0.05,
            limb_peak=None,
            crop=np.zeros((1, 1), dtype=np.float32),
            clipped_px=0,
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
    rows[0].twins[DEFAULT_LAW] = TwinDisc(DEFAULT_LAW, 0.5, None, None, 0.6, None, None, _EMPTY, 0)
    assert compare_disc_profiles(rows)['p0'].n == 2


def _comparison_rows(default_offset: float, other_offset: float) -> list[DiscRow]:
    """Three p0 discs whose default-law and Lambert twins miss by the given amounts."""
    return [
        _row(30.0, k, {DEFAULT_LAW: k + default_offset, 'lambert': k + other_offset})
        for k in (0.5, 0.6, 0.7)
    ]


def test_a_law_that_agrees_worse_by_more_than_the_band_separates() -> None:
    """Lambert missing by 0.3 against the default's 0.01 separates from it."""
    entry = law_comparison(_comparison_rows(0.01, 0.3), laws=_COMPARED)['p0']
    assert entry.separates['lambert'] is True


def test_the_default_law_is_best_when_the_others_agree_worse() -> None:
    """The default law agrees best when every law it separates from agrees worse."""
    entry = law_comparison(_comparison_rows(0.01, 0.3), laws=_COMPARED)['p0']
    assert entry.default_best is True


def test_a_law_that_agrees_better_by_more_than_the_band_beats_the_default() -> None:
    """Lambert missing by 0.01 against the default's 0.3 beats it."""
    entry = law_comparison(_comparison_rows(0.3, 0.01), laws=_COMPARED)['p0']
    assert entry.default_best is False


def test_laws_within_the_band_of_each_other_do_not_separate() -> None:
    """Two laws whose agreement differs by less than the band claim no order."""
    entry = law_comparison(_comparison_rows(0.05, 0.04), laws=_COMPARED)['p0']
    assert entry.separates['lambert'] is False


def test_laws_within_the_band_leave_the_default_best() -> None:
    """A slightly better law that does not separate leaves the default best."""
    entry = law_comparison(_comparison_rows(0.05, 0.04), laws=_COMPARED)['p0']
    assert entry.default_best is True


def test_the_law_comparison_takes_its_band_from_the_default_laws_pairs() -> None:
    """The band is the disc-profile band of the default law's paired differences."""
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d, 'lambert': 0.9}) for d in (0.0, 0.1, 0.2)]
    entry = law_comparison(rows, laws=_COMPARED)['p0']
    assert entry.band_k == pytest.approx(2.0 * 1.4826 * 0.1 / math.sqrt(3.0))


def test_a_bin_with_two_discs_has_no_law_comparison() -> None:
    """Fewer than 3 discs give no comparison."""
    rows = _comparison_rows(0.01, 0.3)[:2]
    assert law_comparison(rows, laws=_COMPARED) == {}


def test_a_law_comparison_of_one_law_is_refused() -> None:
    """The default law alone has nothing to be compared with."""
    with pytest.raises(ValueError, match='at least 2 laws'):
        law_comparison(_comparison_rows(0.01, 0.3), laws=(DEFAULT_LAW,))


def test_a_disc_without_a_twin_under_a_compared_law_is_left_out() -> None:
    """A disc whose twins leave out one of the compared laws does not enter the bin."""
    rows = [*_comparison_rows(0.01, 0.3), _row(30.0, 0.6, {DEFAULT_LAW: 0.61})]
    assert law_comparison(rows, laws=_COMPARED)['p0'].n == 3


def test_the_law_comparisons_band_counts_every_default_law_pair() -> None:
    """Discs that have only a default-law twin set the band with the complete ones.

    Three complete discs miss by 0.10, 0.14, and 0.18 under the default law, and four
    more have only a default-law twin, which matches.  Over all seven the band is the
    floor of 0.02.
    """
    rows = [_row(30.0, 0.6, {DEFAULT_LAW: 0.6 + d, 'lambert': 0.9}) for d in (0.10, 0.14, 0.18)]
    rows += [_row(30.0, 0.6, {DEFAULT_LAW: 0.6}) for _ in range(4)]
    assert law_comparison(rows, laws=_COMPARED)['p0'].band_k == pytest.approx(0.02)


def test_discs_at_one_phase_give_a_flat_minnaert_line() -> None:
    """Two discs at the same phase cannot set a slope, so the line is their mean."""
    k0, k1 = minnaert_line([_disc(30.0, 0.5), _disc(30.0, 0.7)])
    assert (k0, k1) == pytest.approx((0.6, 0.0))


def test_the_minnaert_twins_exponent_is_held_to_its_range() -> None:
    """A line that reaches 1.8 at the disc's phase gives the twin 1.5."""
    assert minnaert_twin_k(0.6, 0.01, 120.0) == pytest.approx(1.5)


def test_a_crop_block_without_a_finite_pixel_is_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2 by 2 block of NaN pixels becomes NaN, and the other blocks keep their mean."""
    monkeypatch.setattr(sim_realism_laws, '_CROP_PX', 2)
    image = np.arange(16.0).reshape(4, 4)
    image[:2, :2] = np.nan
    expected = np.array([[np.nan, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_array_equal(crop_image(image, (0, 4, 0, 4)), expected)


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
    kept = int(sim_realism_laws_discs.decimate(mask).sum())
    assert kept == 53_334


def _hiding_backplane() -> tuple[Any, dict[str, dict[str, Any]]]:
    """A frame holding a near and a far body whose boxes overlap, rings, and a shadow.

    The rings lie in front of the far body at (28, 28), over the sky at (5, 35), and
    behind the near body at (12, 12).
    """
    near = np.zeros((40, 40), dtype=bool)
    near[10:20, 10:20] = True
    far = np.zeros((40, 40), dtype=bool)
    far[15:30, 15:30] = True
    rings = np.zeros((40, 40), dtype=bool)
    rings[28, 28] = True
    ring_plane = rings.copy()
    ring_plane[5, 35] = True
    ring_plane[12, 12] = True
    shadow = np.zeros((40, 40), dtype=bool)
    shadow[29, 15] = True
    masks = {'NEAR': near, 'FAR': far, 'SATURN_MAIN_RINGS': ring_plane}
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
    return sim_realism_laws_discs.hidden_pixels(
        backplane, body, list(entries.items()), planet=planet
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


def test_a_body_behind_the_limb_is_foreign() -> None:
    """The far body's disc lies beyond the near body's limb, so it is foreign to it.

    It hides nothing of the near body, yet a track from the near body's limb that
    reaches it measures its light.
    """
    backplane, entries = _hiding_backplane()
    foreign = sim_realism_laws_discs.foreign_pixels(
        backplane, 'NEAR', list(entries.items()), planet=None
    )
    assert foreign[25, 25]


def test_the_main_rings_beside_a_limb_are_foreign() -> None:
    """Ring light over the sky beside the near body reaches its limb's tracks.

    The rings at (5, 35) lie in front of neither body, yet a track from the near body's
    limb that reaches them, or the clearance around them, measures their light.
    """
    backplane, entries = _hiding_backplane()
    foreign = sim_realism_laws_discs.foreign_pixels(
        backplane, 'NEAR', list(entries.items()), planet='SATURN'
    )
    assert foreign[5, 37]


def test_the_main_rings_behind_a_body_are_not_foreign_to_it() -> None:
    """Where the near body lies in front of the rings, its pixel shows the body."""
    backplane, entries = _hiding_backplane()
    foreign = sim_realism_laws_discs.foreign_pixels(
        backplane, 'NEAR', list(entries.items()), planet='SATURN'
    )
    assert not foreign[12, 12]


def test_the_main_rings_beside_a_moon_hide_none_of_its_disc() -> None:
    """Rings over the sky beside a disc take none of the pixels its fits count."""
    assert not _hidden('NEAR', 'SATURN').any()


def _moon_backplane(
    size: int, radius: float, *, phase_deg: float = 0.0
) -> tuple[Any, NDArrayFloatType]:
    """An offset backplane of one sphere lit from +u, and a Lambert image of it."""
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (size, size), (size / 2.0, size / 2.0), radius, illumination_deg=90.0, phase_deg=phase_deg
    )

    def angles(cosines: NDArrayFloatType) -> Any:
        return SimpleNamespace(
            mvals=np.ma.array(np.arccos(np.clip(cosines, -1.0, 1.0)), mask=~on_body)
        )

    backplane = SimpleNamespace(
        where_intercepted=lambda _body: SimpleNamespace(vals=on_body),
        incidence_angle=lambda _body: angles(mu0),
        emission_angle=lambda _body: angles(mu),
        phase_angle=lambda _body: SimpleNamespace(
            mvals=np.ma.array(np.zeros(on_body.shape), mask=~on_body)
        ),
    )
    return backplane, 0.5 * np.where(on_body, np.clip(mu0, 0.0, None), 0.0)


def _real_orientation() -> Any:
    """The real limb orientation of a sphere at (50, 50), as a pixel corner, lit from +u."""
    backplane, _data = _moon_backplane(100, 40.0, phase_deg=40.0)
    entry = {'center_uv': (50.0, 50.0), 'u_pixel_size': 80.0, 'v_pixel_size': 80.0}
    selection = sim_realism_laws_discs.real_limb_selection(
        backplane, 'DIONE', [('DIONE', entry)], planet=None, offset_vu=(0.0, 0.0)
    )
    return selection.orientation


def test_a_real_limb_is_lit_from_where_its_disc_says() -> None:
    """A sphere lit from +u gives its limb a light direction of 90 degrees."""
    assert _real_orientation().light_deg == pytest.approx(90.0, abs=1e-6)


def test_a_real_limbs_center_is_its_bodys_center_in_pixel_centers() -> None:
    """A body centered at (50, 50) as a pixel corner centers its limb at (49.5, 49.5)."""
    assert _real_orientation().center_vu == pytest.approx((49.5, 49.5))


def _measured(size: int, radius: float, *, clipped_and_hidden: Any = None) -> LawStudyInputs:
    """The study's inputs after measuring one sphere, with some pixels clipped and hidden."""
    backplane, data = _moon_backplane(size, radius)
    marked = np.zeros(data.shape, dtype=bool)
    if clipped_and_hidden is not None:
        marked[clipped_and_hidden] = True
    inputs = LawStudyInputs()
    sim_realism_laws_discs._measure_moon(
        inputs,
        backplane,
        _FRAME,
        'DIONE',
        diameter_px=2.0 * radius,
        center_uv=(size / 2.0, size / 2.0),
        data=data,
        usable=(data > 0.0) & ~marked,
        full_scale=marked,
        hidden=marked,
    )
    return inputs


def test_a_large_disc_keeps_the_thinned_pixels_as_its_selection() -> None:
    """A disc of more than 60,000 eligible pixels keeps at most 60,000, and fits those."""
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (340, 340), (170.0, 170.0), 160.0, illumination_deg=90.0, phase_deg=0.0
    )
    eligible = select_disc_pixels(on_body, mu0, mu, usable=on_body & (mu0 > 0.0))
    disc = _measured(340, 160.0).discs[0]
    assert int(eligible.sum()) > 60_000
    assert int(disc.selection.sum()) <= 60_000
    assert int(disc.selection.sum()) == disc.intensity.size


def test_clipped_pixels_that_a_nearer_body_hides_do_not_clip_the_disc() -> None:
    """A clipped body covering a third of the moon leaves the moon measured."""
    inputs = _measured(100, 40.0, clipped_and_hidden=np.s_[10:40, :])
    assert len(inputs.discs) == 1


def test_a_disc_whose_fit_fails_says_its_pixels_do_not_support_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disc with enough clear pixels whose Minnaert fit fails gives that as the reason."""
    monkeypatch.setattr(sim_realism_laws_discs, 'fit_minnaert', lambda *_args: None)
    reason = _measured(100, 40.0).excluded[0].reason
    assert reason.endswith("do not support a fit of Minnaert's law")


def test_a_body_the_offset_moves_into_the_frame_is_in_view() -> None:
    """A box 3 px past the frame's right edge, moved 5 px left, overlaps the frame."""
    fov = SimpleNamespace(fov_u_min=0, fov_u_max=99, fov_v_min=0, fov_v_max=99)
    obs = SimpleNamespace(
        inventory_body_in_fov=lambda inv: ObsSnapshot.inventory_body_in_fov(cast(Any, fov), inv)
    )
    entry = {
        'u_min_unclipped': 102.0,
        'u_max_unclipped': 120.0,
        'v_min_unclipped': 40.0,
        'v_max_unclipped': 60.0,
    }
    assert box_in_fov_at_offset(obs, entry, (0.0, -5.0)) is True


def _sphere_disc(law: str, *, albedo: float = 0.05, **kwargs: float) -> RealDisc:
    """A real disc drawn from a sphere under one law, in the twin's own geometry.

    Its crop bounds are a box unlike the sphere's own, 100 by 200 px.
    """
    phase = 30.0
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (256, 256), (128.0, 128.0), 60.0, illumination_deg=90.0, phase_deg=phase
    )
    keep = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    intensity = albedo * shade_surface(
        mu0[keep], mu[keep], law=law, phase_angle=math.radians(phase), **kwargs
    )
    disc = _disc(phase, 1.0)
    disc.diameter_px = 120.0
    disc.intensity = intensity
    disc.mu0 = mu0[keep]
    disc.mu = mu[keep]
    disc.profile = equator_profile(intensity, mu0[keep], mu[keep], math.radians(phase))
    disc.selection = keep
    disc.crop_bounds = (10, 110, 20, 220)
    return disc


def _twins_of(
    disc: RealDisc, laws: tuple[str, ...], *, minnaert_k: float, limb_peak: bool
) -> dict[str, TwinDisc]:
    """The twins of a disc that gives a normal albedo under every law."""
    twins = sim_realism_laws._twins(disc, laws, minnaert_k=minnaert_k, limb_peak=limb_peak)
    assert twins is not None
    return twins


@pytest.fixture(scope='module')
def lambert_twin() -> TwinDisc:
    """The rendered Lambert twin of a Lambert disc whose exponent is recorded as 1."""
    return _twins_of(_sphere_disc('lambert'), ('lambert',), minnaert_k=0.6, limb_peak=False)[
        'lambert'
    ]


def test_a_lambert_twin_fits_the_exponent_one(lambert_twin: TwinDisc) -> None:
    """A rendered Lambert twin, blurred and noisy as the camera makes it, fits k near 1."""
    assert lambert_twin.k == pytest.approx(1.0, abs=0.03)


def test_a_twin_well_below_its_ceiling_reads_no_pixel_at_it(lambert_twin: TwinDisc) -> None:
    """A twin of albedo 0.05 leaves every one of the real fits' pixels in its fit."""
    assert lambert_twin.clipped_px == 0


@pytest.fixture(scope='module')
def bright_twin() -> TwinDisc:
    """The Lambert twin of a disc of albedo 3, which fills the camera past its ceiling.

    The real frame measured a limb gradient peak in the twin's limb bin.
    """
    disc = _sphere_disc('lambert', albedo=3.0)
    disc.limb_peak = ('limb_peak_p0_r1', 0.0)
    return _twins_of(disc, ('lambert',), minnaert_k=0.6, limb_peak=True)['lambert']


def test_a_twin_counts_its_pixels_at_the_ceiling(bright_twin: TwinDisc) -> None:
    """A twin past its ceiling reports the real fits' pixels it reads at the ceiling."""
    assert bright_twin.clipped_px > 0


def test_a_twin_leaves_its_pixels_at_the_ceiling_out_of_its_fit(bright_twin: TwinDisc) -> None:
    """Without its clipped pixels, a Lambert twin past its ceiling still fits k near 1."""
    assert bright_twin.k == pytest.approx(1.0, abs=0.03)


def test_a_clipped_twin_gives_no_limb_peak(bright_twin: TwinDisc) -> None:
    """A twin whose disc is clipped, by the test a real disc meets, has no limb peak."""
    assert bright_twin.limb_peak is None


def test_a_clipped_twin_gives_the_clipping_as_its_refusal(bright_twin: TwinDisc) -> None:
    """The refusal of a clipped twin's limb says the clipping caused it."""
    assert bright_twin.limb_peak_refusal == "its disc is clipped at the camera's ceiling"


def test_a_twin_takes_its_albedo_from_the_real_disc_under_its_own_law(
    lambert_twin: TwinDisc,
) -> None:
    """The twin's albedo is the real disc's measured against the twin's law."""
    assert lambert_twin.normal_albedo == pytest.approx(0.05)


def test_a_twin_that_counts_every_real_pixel_pairs_with_the_real_exponent(
    lambert_twin: TwinDisc,
) -> None:
    """A twin over all of the real fits' pixels is compared with the real disc's own fit."""
    assert lambert_twin.real_k == pytest.approx(1.0, abs=1e-9)


def test_a_twin_is_cropped_to_the_real_discs_bounds(lambert_twin: TwinDisc) -> None:
    """A 100 by 200 px crop is halved to 50 by 100, whatever the twin's own extent."""
    assert lambert_twin.crop.shape == (50, 100)


def test_a_minnaert_twin_takes_the_exponent_it_is_given() -> None:
    """A rendered Minnaert twin fits the exponent it was rendered with."""
    twin = _twins_of(
        _sphere_disc('minnaert', minnaert_k=0.6), ('minnaert',), minnaert_k=0.6, limb_peak=False
    )['minnaert']
    assert twin.k == pytest.approx(0.6, abs=0.03)


def _keep_selected(disc: RealDisc, keep: NDArrayBoolType) -> None:
    """Narrow a disc's counted pixels to those of ``keep`` that it counted."""
    kept = keep[disc.selection]
    disc.intensity = disc.intensity[kept]
    disc.mu0 = disc.mu0[kept]
    disc.mu = disc.mu[kept]
    disc.selection = disc.selection & keep


def test_a_twin_is_fitted_only_over_the_real_discs_pixels() -> None:
    """A real disc that counted 99 pixels leaves its twin too few to fit.

    They lie where both cosines are at least 0.3, so they give the twin its albedo.
    """
    disc = _sphere_disc('lambert')
    facing = np.flatnonzero((disc.mu0 >= 0.3) & (disc.mu >= 0.3))[:99]
    rows, cols = np.nonzero(disc.selection)
    first = np.zeros_like(disc.selection)
    first[rows[facing], cols[facing]] = True
    _keep_selected(disc, first)
    twin = _twins_of(disc, ('lambert',), minnaert_k=0.6, limb_peak=False)['lambert']
    assert twin.k is None


def test_a_twin_that_leaves_out_real_pixels_pairs_with_the_real_disc_over_the_rest() -> None:
    """The real disc is fitted again over the pixels its twin also counts.

    The real fits counted every lit pixel out to the limb and the terminator, and the
    pixels the twin's clearance leaves out are three times too bright, so only a fit
    over the shared pixels gives back the Lambert exponent 1.
    """
    phase = 30.0
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (256, 256), (128.0, 128.0), 60.0, illumination_deg=90.0, phase_deg=phase
    )
    lit = on_body & (mu0 > 0.05) & (mu > 0.05)
    clear = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    disc = _sphere_disc('lambert')
    disc.selection = lit
    disc.mu0 = mu0[lit]
    disc.mu = mu[lit]
    disc.intensity = 0.05 * disc.mu0 * np.where(clear[lit], 1.0, 3.0)
    disc.minnaert = MinnaertFit(k=0.3, b0=0.05, k_sigma=0.01, n_pixels=int(lit.sum()))
    twin = _twins_of(disc, ('lambert',), minnaert_k=0.6, limb_peak=False)['lambert']
    assert twin.real_k == pytest.approx(1.0, abs=1e-9)


def test_a_pixel_one_twin_clips_is_left_out_of_every_twins_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every law's twin is fitted over the pixels all of the disc's twins keep.

    The Lambert twin reads the brighter part of the disc at its ceiling, and there the
    other twin falls off more steeply than Lambert's law, so only a fit that leaves out
    the Lambert twin's clipped pixels gives the other twin the Lambert exponent 1.
    """
    phase = 30.0
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (256, 256), (128.0, 128.0), 60.0, illumination_deg=90.0, phase_deg=phase
    )
    lit = on_body & (mu0 > 0.0) & (mu > 0.0)
    lambert = np.zeros(on_body.shape)
    lambert[lit] = 0.05 * shade_surface(
        mu0[lit], mu[lit], law='lambert', phase_angle=math.radians(phase)
    )
    bright = lit & (mu0 > 0.6)

    def render(
        disc: RealDisc,
        law: str,
        *,
        normal_albedo: float,
        minnaert_k: float,
        limb_peak: bool,
        on_body: NDArrayBoolType,
    ) -> Any:
        steep = law == 'lommel_seeliger'
        data = np.where(bright & steep, lambert * (mu0 / 0.6) ** 2, lambert)
        return sim_realism_laws._TwinRender(
            law=law,
            normal_albedo=0.05,
            data=data,
            at_ceiling=bright & (law == 'lambert'),
            limb_peak=None,
            limb_peak_refusal=None,
        )

    monkeypatch.setattr(sim_realism_laws, '_render_twin', render)
    twins = _twins_of(
        _sphere_disc('lambert'), ('lambert', 'lommel_seeliger'), minnaert_k=0.6, limb_peak=False
    )
    assert twins['lommel_seeliger'].k == pytest.approx(1.0, abs=1e-6)


def test_pixels_the_real_fit_drops_are_left_out_of_every_twins_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every twin of a disc is fitted over the pixels the real disc's fit keeps.

    The real disc is Lambert and five times too bright where the disc faces both the
    Sun and the camera most closely, and its fit drops those pixels.  The twin carries
    3% noise and is 5% too bright there, too little for a fit of its own to drop, so
    only a fit that leaves them out gives the twin the Lambert exponent 1.
    """
    phase = 30.0
    on_body, mu0, mu = sim_realism_laws._sphere_cosines(
        (256, 256), (128.0, 128.0), 60.0, illumination_deg=90.0, phase_deg=phase
    )
    lambert = np.where(on_body, 0.05 * np.clip(mu0, 0.0, None), 0.0)
    marked = on_body & (mu0 * mu > 0.8)
    noise = np.exp(0.03 * np.random.default_rng(5).normal(size=on_body.shape))
    disc = _sphere_disc('lambert')
    disc.intensity = disc.intensity * np.where(marked[disc.selection], 5.0, 1.0)

    def render(
        disc: RealDisc,
        law: str,
        *,
        normal_albedo: float,
        minnaert_k: float,
        limb_peak: bool,
        on_body: NDArrayBoolType,
    ) -> Any:
        return sim_realism_laws._TwinRender(
            law=law,
            normal_albedo=normal_albedo,
            data=lambert * noise * np.where(marked, 1.05, 1.0),
            at_ceiling=np.zeros(on_body.shape, dtype=bool),
            limb_peak=None,
            limb_peak_refusal=None,
        )

    monkeypatch.setattr(sim_realism_laws, '_render_twin', render)
    twins = _twins_of(disc, ('lambert',), minnaert_k=0.6, limb_peak=False)
    assert twins['lambert'].k == pytest.approx(1.0, abs=0.002)


def test_a_disc_without_a_normal_albedo_is_left_out_of_the_study() -> None:
    """A disc whose pixels all lie at an incidence cosine of 0.1 is listed as left out."""
    disc = _sphere_disc('lambert')
    disc.mu0 = np.full(disc.mu0.shape, 0.1)
    study = run_law_study(LawStudyInputs(discs=[disc]), laws=('lambert',))
    reason = 'its disc does not give a normal albedo for its twins'
    assert study.excluded == [ExcludedDisc(_FRAME.image_id, 'DIONE', reason)]


def _record(
    median_delta_k: float,
    band_k: float | None,
    within: bool | None,
    *,
    agreement: dict[str, float] | None = None,
    separates: bool = False,
    default_best: bool = True,
) -> dict[str, Any]:
    """A record holding one disc-profile bin and one law-comparison bin.

    The comparison's band is 0.05, and its laws agree to within it unless ``agreement``
    says otherwise.
    """
    return {
        'instruments': {
            'coiss_calib_nac': {
                'disc_profiles': {
                    'p0': {
                        'n': 4,
                        'real_median_k': 0.6,
                        'sim_median_k': 0.6 + median_delta_k,
                        'median_delta_k': median_delta_k,
                        'delta_scatter_k': None if band_k is None else 0.05,
                        'band_k': band_k,
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
                    'agreement': agreement or {'lunar_lambert': 0.06, 'lambert': 0.08},
                    'separates': {'lambert': separates},
                    'default_best': default_best,
                }
            },
        },
    }


def test_a_record_bin_outside_its_band_is_a_failure() -> None:
    """A disc-profile bin whose median difference exceeds its band is reported."""
    failures = record_disc_profile_failures(_record(0.1, 0.05, False))
    assert failures == {
        ('coiss_calib_nac', 'p0'): "coiss_calib_nac p0: the twins' Minnaert exponents "
        'differ from the real ones by a median +0.100, more than the band of 0.050 (4 discs)'
    }


@pytest.mark.parametrize(('delta', 'band', 'within'), [(0.03, 0.05, True), (0.1, None, None)])
def test_a_record_bin_within_or_without_a_band_is_no_failure(
    delta: float, band: float | None, within: bool | None
) -> None:
    """A disc-profile bin within its band, or without a band, is not reported."""
    assert record_disc_profile_failures(_record(delta, band, within)) == {}


@pytest.mark.parametrize(
    ('delta', 'band', 'within', 'count'),
    [(0.03, 0.05, True, 1), (0.1, 0.05, False, 1), (0.1, None, None, 0)],
)
def test_a_record_counts_its_disc_profile_verdicts(
    delta: float, band: float | None, within: bool | None, count: int
) -> None:
    """A bin within or outside its band has a verdict; a bin without a band has none."""
    assert record_disc_profile_verdicts(_record(delta, band, within)) == count


def test_a_recorded_verdict_its_numbers_contradict_is_refused() -> None:
    """A bin recorded within its band, with a median difference twice the band, is refused."""
    with pytest.raises(ValueError, match=r'within_band is True, but its median difference'):
        record_disc_profile_failures(_record(0.1, 0.05, True))


def test_a_law_that_separates_and_agrees_better_is_a_problem() -> None:
    """A law better than the default by more than the band is reported."""
    record = _record(
        0.03,
        0.05,
        True,
        agreement={'lunar_lambert': 0.2, 'lambert': 0.05},
        separates=True,
        default_best=False,
    )
    assert record_law_comparison_problems(record) == [
        'p0: lambert agrees with the real discs to a median |dk| of 0.050, better than '
        "lunar_lambert's 0.200 by more than the band of 0.050 (4 discs)"
    ]


def test_a_better_law_that_does_not_separate_is_no_problem() -> None:
    """A law better than the default by less than the band is not reported."""
    record = _record(0.03, 0.05, True, agreement={'lunar_lambert': 0.08, 'lambert': 0.05})
    assert record_law_comparison_problems(record) == []


def test_a_worse_law_that_separates_is_no_problem() -> None:
    """A law that separates by agreeing worse is not reported."""
    record = _record(
        0.03, 0.05, True, agreement={'lunar_lambert': 0.05, 'lambert': 0.2}, separates=True
    )
    assert record_law_comparison_problems(record) == []


def test_a_recorded_separation_its_numbers_contradict_is_refused() -> None:
    """A law recorded as not separating, 0.15 apart against a band of 0.05, is refused."""
    record = _record(0.03, 0.05, True, agreement={'lunar_lambert': 0.2, 'lambert': 0.05})
    with pytest.raises(ValueError, match=r'separates from lunar_lambert is False, but'):
        record_law_comparison_problems(record)


def test_a_recorded_default_best_its_numbers_contradict_is_refused() -> None:
    """A bin the default wins, 0.08 ahead against a band of 0.05, recorded as lost."""
    record = _record(
        0.03,
        0.05,
        True,
        agreement={'lunar_lambert': 0.02, 'lambert': 0.10},
        separates=True,
        default_best=False,
    )
    with pytest.raises(ValueError, match=r'agrees best is False, but its agreements and band'):
        record_law_comparison_problems(record)


def test_the_blend_axis_widens_to_hold_a_fitted_blend() -> None:
    """A blend of 1.8 with an error of 0.1 lifts the axis's top above 1.9."""
    disc = _disc(30.0, 0.6)
    disc.lunar_lambert = LunarLambertFit(blend=1.8, b0=0.5, blend_sigma=0.1, n_pixels=500)
    row = DiscRow(disc=disc, twins={})
    assert sim_realism_laws_report._blend_limits([row])[1] > 1.9


def test_a_comparison_that_leaves_out_a_laws_separation_is_refused() -> None:
    """A bin that compares Lambert with the default but says nothing of it is refused."""
    record = _record(0.03, 0.05, True, agreement={'lunar_lambert': 0.2, 'lambert': 0.05})
    record['law_study']['comparison']['p0']['separates'] = {}
    with pytest.raises(ValueError, match=r'each law other than the default needs one such'):
        record_law_comparison_problems(record)


def test_a_written_disc_profile_bin_just_outside_its_band_reads_back_outside() -> None:
    """A median difference 3e-8 past the band is written as it is and read as outside."""
    entry = DiscProfileBin(
        n=4,
        real_median_k=0.6,
        sim_median_k=0.62000003,
        median_delta_k=0.02000003,
        delta_scatter_k=0.03,
        band_k=0.02,
        within_band=False,
    )
    record = {
        'instruments': {'coiss_calib_nac': {'disc_profiles': disc_profile_entries({'p0': entry})}}
    }
    assert set(record_disc_profile_failures(record)) == {('coiss_calib_nac', 'p0')}


def test_a_written_law_comparison_just_past_its_band_reads_back_separated() -> None:
    """Agreements 3e-8 farther apart than the band are written as they are and separate."""
    entry = LawComparisonBin(
        n=4,
        band_k=0.02,
        agreement={'lunar_lambert': 0.02000003, 'lambert': 0.0},
        separates={'lambert': True},
        default_best=False,
    )
    record = {
        'law_study': {
            'default_law': 'lunar_lambert',
            'comparison': {'p0': sim_realism_laws_report._comparison_entry(entry)},
        }
    }
    assert len(record_law_comparison_problems(record)) == 1


@pytest.fixture(scope='module')
def realism_record() -> dict[str, Any]:
    """The realism record the regression checks read, read once for the module."""
    return load_record()


def test_the_record_holds_the_law_study(realism_record: dict[str, Any]) -> None:
    """The realism record carries the photometric-law study."""
    assert record_has_law_study(realism_record), (
        f'The realism record made at {made_at(realism_record)} lacks the law study.\n'
        f'{REGENERATE_HINT}'
    )


def test_the_record_was_made_with_the_codes_default_law(realism_record: dict[str, Any]) -> None:
    """The record's default law is the scene default the code renders with."""
    assert realism_record.get('law_study', {}).get('default_law') == PHOTOMETRIC_LAW_DEFAULT, (
        f'The realism record made at {made_at(realism_record)} judged another default law.\n'
        f'{REGENERATE_HINT}'
    )


def test_the_record_judges_at_least_one_disc_profile_bin(realism_record: dict[str, Any]) -> None:
    """At least one disc-profile bin of the record has a verdict."""
    assert record_disc_profile_verdicts(realism_record) >= 1


def test_the_record_compares_the_laws_in_at_least_one_bin(realism_record: dict[str, Any]) -> None:
    """At least one phase bin of the record holds a law comparison."""
    assert realism_record.get('law_study', {}).get('comparison')


def test_the_recorded_disc_profiles_lie_within_their_bands(
    realism_record: dict[str, Any],
) -> None:
    """In every bin with a verdict, the default law's twins keep the real exponents.

    A bin :data:`KNOWN_DISC_PROFILE_FINDINGS` lists is a recorded finding against the
    default law: it must still lie outside its band, and while one does, the test is an
    expected failure that gives the finding.  A listed bin back within its band fails,
    so the list can only shrink.
    """
    failures = record_disc_profile_failures(realism_record)
    new = [line for key, line in failures.items() if key not in KNOWN_DISC_PROFILE_FINDINGS]
    assert not new, (
        f'The disc profiles of the record made at {made_at(realism_record)} leave their '
        'bands:\n' + '\n'.join(new) + f'\n{REGENERATE_HINT}'
    )
    back = sorted(set(KNOWN_DISC_PROFILE_FINDINGS) - set(failures))
    assert not back, f'These known findings now lie within their bands; drop them: {back}'
    if failures:
        pytest.xfail(
            'known disc-profile findings: '
            + '; '.join(KNOWN_DISC_PROFILE_FINDINGS[key] for key in sorted(failures))
        )


def test_the_default_law_agrees_best_where_the_laws_separate(
    realism_record: dict[str, Any],
) -> None:
    """Wherever another law separates from the default law, it agrees worse."""
    problems = record_law_comparison_problems(realism_record)
    assert not problems, (
        f'In the record made at {made_at(realism_record)} another law fits the real discs '
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


def _study_of(rows: list[DiscRow]) -> Any:
    """A study of the given rows under the default law alone, its statistics empty."""
    return sim_realism_laws.LawStudy(
        rows=rows,
        excluded=[],
        minnaert_k0=0.5,
        minnaert_k1=0.0,
        law_bins={DEFAULT_LAW: {}},
        body_bins={DEFAULT_LAW: {}},
        limb_peaks={DEFAULT_LAW: {}},
        disc_check={},
        comparison={},
    )


def test_a_study_without_a_large_disc_still_writes_its_sheet(tmp_path: Path) -> None:
    """A study whose discs are all under 100 px across writes a sheet that says so."""
    rows = [_row(phase, 0.6, {DEFAULT_LAW: 0.6}) for phase in (20.0, 30.0, 40.0)]
    for row in rows:
        row.disc.diameter_px = 50.0
    written = write_law_figures(_study_of(rows), tmp_path)
    assert (tmp_path / 'realism_law_sheet.png') in [p for p in written if p.exists()]


def test_a_study_without_a_twin_exponent_still_writes_its_figures(tmp_path: Path) -> None:
    """A study whose twins could not be fitted writes all three figures."""
    rows = [_row(phase, 0.6, {DEFAULT_LAW: 0.6}) for phase in (20.0, 30.0, 40.0)]
    for row in rows:
        row.disc.diameter_px = 50.0
        row.twins[DEFAULT_LAW] = TwinDisc(DEFAULT_LAW, 0.5, None, None, 0.6, None, None, _EMPTY, 0)
    written = write_law_figures(_study_of(rows), tmp_path)
    assert [p.name for p in written if p.exists()] == [
        'realism_law_fits.png',
        'realism_disc_profiles.png',
        'realism_law_sheet.png',
    ]


def test_the_default_law_panel_plots_each_pair_as_the_check_reads_it() -> None:
    """A pair is plotted with the real exponent over the pixels its twin also counted."""
    row = _row(30.0, 0.6, {DEFAULT_LAW: 0.7})
    row.twins[DEFAULT_LAW] = dataclasses.replace(row.twins[DEFAULT_LAW], real_k=0.65)
    assert sim_realism_laws_report._default_law_pairs([row]) == [(30.0, 0.65, 0.7)]


@pytest.mark.parametrize(
    ('band', 'within', 'line'),
    [
        (None, None, 'coiss_calib_wac p0: n=1, median dk +0.070, too few discs for a verdict'),
        (0.05, False, 'coiss_calib_wac p0: n=1, median dk +0.070, outside band 0.050'),
    ],
)
def test_the_default_law_panel_gives_each_bins_verdict(
    band: float | None, within: bool | None, line: str
) -> None:
    """Each instrument's phase bin is shown with its verdict, or as too small for one."""
    entry = DiscProfileBin(1, 0.6, 0.67, 0.07, None, band, within)
    lines = sim_realism_laws_report._disc_check_lines({'coiss_calib_wac': {'p0': entry}})
    assert lines == [line]
