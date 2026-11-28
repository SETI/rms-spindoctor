"""Tests of the realism match's photometric-law study.

The tests build their discs in memory; some render a small twin.  Nothing here reads
holdings or the committed record, and CI does not run the file.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from util.sim_realism import sim_realism_laws, sim_realism_laws_discs
from util.sim_realism.disc_profile import (
    LunarLambertFit,
    MinnaertFit,
    equator_profile,
    select_disc_pixels,
)
from util.sim_realism.sim_realism_body_facts import TerrainCap
from util.sim_realism.sim_realism_laws import (
    DEFAULT_LAW,
    DiscRow,
    ExcludedDisc,
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
from util.sim_realism.sim_realism_laws_discs import (
    disc_lit_area_px,
    lit_share,
)
from util.sim_realism.sim_realism_laws_report import (
    law_study_summary,
    write_law_figures,
)

from spindoctor.sim.render.bodies.photometry import shade_surface
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

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


def test_a_bin_with_two_discs_is_not_judged() -> None:
    """Fewer than 3 discs give no band and no verdict."""
    rows = [_row(30.0, k, {DEFAULT_LAW: k + 0.5}) for k in (0.5, 0.6)]
    assert compare_disc_profiles(rows)['p0'].within_band is None


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


def test_laws_within_the_band_of_each_other_do_not_separate() -> None:
    """Two laws whose agreement differs by less than the band claim no order."""
    entry = law_comparison(_comparison_rows(0.05, 0.04), laws=_COMPARED)['p0']
    assert entry.separates['lambert'] is False


def test_the_minnaert_line_runs_through_the_real_exponents() -> None:
    """Exponents on a straight line give back that line."""
    discs = [_disc(phase, 0.5 + 0.004 * phase) for phase in (10.0, 40.0, 90.0)]
    k0, k1 = minnaert_line(discs)
    assert (k0, k1) == pytest.approx((0.5, 0.004))


def test_a_disc_wholly_in_the_frame_has_its_whole_lit_share() -> None:
    """A lit disc of the expected area has a lit share near 1."""
    on_body, mu0, _mu = sim_realism_laws._sphere_cosines(
        (220, 220), (110.0, 110.0), 100.0, illumination_deg=0.0, phase_deg=60.0
    )
    share = lit_share(on_body & (mu0 > 0.0), disc_lit_area_px(100.0, 60.0))
    assert share == pytest.approx(1.0, abs=0.01)


def test_a_cap_holds_the_points_within_its_radius() -> None:
    """A point 39 deg from the cap's center lies inside a cap of 40 deg."""
    cap = TerrainCap('dark terrain', 0.0, 90.0, 40.0)
    assert bool(cap.contains(np.array([39.0]), np.array([90.0]))[0])


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


@pytest.fixture(scope='module')
def bright_twin() -> TwinDisc:
    """The Lambert twin of a disc of albedo 3, which fills the camera past its ceiling.

    The real frame measured a limb gradient peak in the twin's limb bin.
    """
    disc = _sphere_disc('lambert', albedo=3.0)
    disc.limb_peak = ('limb_peak_p0_r1', 0.0)
    return _twins_of(disc, ('lambert',), minnaert_k=0.6, limb_peak=True)['lambert']


def test_a_twin_leaves_its_pixels_at_the_ceiling_out_of_its_fit(bright_twin: TwinDisc) -> None:
    """Without its clipped pixels, a Lambert twin past its ceiling still fits k near 1."""
    assert bright_twin.k == pytest.approx(1.0, abs=0.03)


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


def test_a_disc_without_a_normal_albedo_is_left_out_of_the_study() -> None:
    """A disc whose pixels all lie at an incidence cosine of 0.1 is listed as left out."""
    disc = _sphere_disc('lambert')
    disc.mu0 = np.full(disc.mu0.shape, 0.1)
    study = run_law_study(LawStudyInputs(discs=[disc]), laws=('lambert',))
    reason = 'its disc does not give a normal albedo for its twins'
    assert study.excluded == [ExcludedDisc(_FRAME.image_id, 'DIONE', reason)]


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
