"""Integration tests for ``spindoctor.nav_model.stars.catalog`` using shims.

These tests drive ``stars_in_extfov`` and ``reduce_catalogs`` against a
fake catalog (via :func:`tests.shims.install_fake_catalogs`) and a
fake observation (via :class:`tests.shims.FakeObs`), so the catalog
reduction's UCAC4 / Tycho-2 / YBSC branches and the FOV-projection
edge-culling code are exercised without real catalog data or SPICE.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest
from tests.shims import FakeObs, install_fake_catalogs, make_star

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_model.stars import catalog as nav_catalog
from spindoctor.nav_model.stars.catalog import _merge_catalogs, reduce_catalogs, stars_in_extfov
from spindoctor.support.types import STAR_UV_DATUM_PX, MutableStar


@pytest.fixture
def fake_obs() -> FakeObs:
    """Provide a 100x100 obs with extfov margin and the configured PSF."""
    return FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(10, 10),
        midtime=0.0,
        ra_dec_limits_ext_rad=(0.0, 0.5, -0.1, 0.1),
        star_min_vmag=0.0,
        star_max_vmag=15.0,
    )


@pytest.fixture(autouse=True)
def disable_stellar_aberration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``aberrate_star`` with a no-op for these integration tests.

    The production helper builds a real ``oops.Event`` whose constructor
    requires a SPICE-aware ``path`` / ``frame`` on the observation; the
    ``FakeObs`` shim does not provide those.  These tests exercise the
    catalog-reduction call paths, not the aberration math (which has its
    own unit tests), so the no-op is safe.
    """
    monkeypatch.setattr(
        'spindoctor.nav_model.stars.catalog.aberrate_star',
        lambda _obs, _star: None,
    )


def test_stars_in_extfov_returns_records_from_fake_ucac4(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A fake UCAC4 with one star in the box yields one reduced record."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0)],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    star = out[0]
    assert star.unique_number == 1
    assert star.catalog_name == 'ucac4'
    assert star.dn > 0.0
    # The FOV projection populates u, v, move_u, move_v.
    assert star.u != 0.0 or star.v != 0.0


def test_stars_in_extfov_drops_stars_outside_ra_window(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """Stars outside the configured RA limits are excluded by the catalog filter."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=10.0, dec=0.0, vmag=5.0)],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert out == []


def test_stars_in_extfov_drops_unfittable_johnson_b_in_ybsc(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """YBSC's ``b_v``-based johnson-mag fill path is exercised."""
    install_fake_catalogs(
        monkeypatch,
        ybsc=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0, b_v=0.65, name='Sirius')],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ybsc',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    assert out[0].johnson_mag_v == 5.0
    assert out[0].johnson_mag_b == pytest.approx(5.65)


def test_stars_in_extfov_tycho2_drops_johnson_mags(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """Tycho-2's per-catalog ``johnson_mag_*`` clearing path is exercised.

    The shim runs through the production code's tycho2 branch, which
    explicitly clears johnson_mag_v / johnson_mag_b before the
    fill-from-spectral-class path runs.  After reduction those fields
    are populated from the configured ``default_star_class``.
    """
    install_fake_catalogs(
        monkeypatch,
        tycho2=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0)],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='tycho2',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    # The fill-from-default path runs because tycho2 cleared the
    # johnson_mag_* fields; johnson_mag_faked is True after reduction.
    assert out[0].johnson_mag_faked is True


def test_reduce_catalogs_walks_every_configured_catalog(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """``reduce_catalogs`` calls each configured catalog and merges the results.

    Configures one star per catalog at distinct RA values so the
    deduplication path keeps all three.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.00, vmag=5.0)],
        tycho2=[make_star(unique_number=2, ra=0.20, dec=0.01, vmag=6.0)],
        ybsc=[make_star(unique_number=3, ra=0.30, dec=0.02, vmag=7.0, b_v=0.5)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    by_id = {s.unique_number for s in out}
    assert by_id == {1, 2, 3}


def test_reduce_catalogs_dedupes_against_higher_precedence(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A duplicate in a lower-precedence catalog is dropped by the merge."""
    same_ra = 0.10
    same_dec = 0.0
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=same_ra, dec=same_dec, vmag=5.0, name='')],
        tycho2=[make_star(unique_number=2, ra=same_ra, dec=same_dec, vmag=5.0, name='')],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    assert len(out) == 1
    # UCAC4 has higher precedence so its entry survives.
    assert out[0].catalog_name == 'ucac4'


def test_reduce_catalogs_caps_at_max_stars(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """The merged list is capped at ``stars.max_stars``."""
    fake_stars = [
        make_star(unique_number=i, ra=0.1 + 0.001 * i, dec=0.0, vmag=5.0 + i * 0.001)
        for i in range(150)
    ]
    install_fake_catalogs(monkeypatch, ucac4=fake_stars)
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    assert len(out) == DEFAULT_CONFIG.stars.max_stars


def test_stars_in_extfov_skips_records_with_missing_radec(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """Records whose RA / DEC / vmag is ``None`` are skipped by the reduction."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[
            make_star(unique_number=1, ra=0.05, dec=0.05, vmag=5.0),
            # The catalog filter drops records with ``vmag is None`` before
            # they ever reach the reduction; an additional ``ra is None``
            # record is dropped by the ``ra is None`` skip in the reduction
            # body.  Construct a record post-filter by mutating after the
            # fact (the dataclass allows it).
        ],
    )
    # Inject a star whose ``ra`` is None into the FakeStarCatalog list so
    # the post-filter reduction loop hits the early-skip branch.  The
    # patched getter is reached via dynamic attribute lookup on the
    # catalog module so monkeypatch's swap is honoured.
    cat = cast(Any, nav_catalog.get_ucac4_catalog())
    bad = make_star(unique_number=2, ra=0.05, dec=0.05, vmag=5.0)
    bad.ra = None
    cat.stars.append(bad)
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    # Only the well-formed star survives.
    assert len(out) == 1
    assert out[0].unique_number == 1


def test_stars_in_extfov_supplies_default_temperature_when_missing(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A star with ``temperature=None`` is filled with the default star class."""
    star = make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0)
    star.temperature = None
    star.spectral_class = None
    install_fake_catalogs(monkeypatch, ucac4=[star])
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    assert out[0].temperature_faked is True
    assert out[0].temperature is not None
    assert out[0].spectral_class == DEFAULT_CONFIG.stars.default_star_class


def test_stars_in_extfov_handles_star_without_name_attribute(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A star whose ``name`` access raises ``AttributeError`` falls back gracefully.

    The production reduction uses ``try / except AttributeError`` to
    handle catalog records that don't expose a ``name`` field at all
    (vs. simply having an empty string).
    """

    class _NoNameStar:
        unique_number = 99
        catalog_name = ''
        pretty_name = ''
        ra = 0.05
        dec = 0.05
        vmag = 5.0
        b_v = 0.5
        johnson_mag_v: float | None = None
        johnson_mag_b: float | None = None
        johnson_mag_faked = False
        spectral_class = 'G0'
        temperature: float | None = 5800.0
        temperature_faked = False
        ra_pm = 0.0
        dec_pm = 0.0
        psf_size = (5, 5)
        u = 0.0
        v = 0.0
        move_u = 0.0
        move_v = 0.0
        dn = 0.0
        conflicts = ''
        diff_u = 0.0
        diff_v = 0.0

        def ra_dec_with_pm(self, _tdb: float) -> tuple[float, float]:
            return self.ra, self.dec

        # Note: no ``name`` attribute; ``star.name`` raises AttributeError.

    install_fake_catalogs(monkeypatch, ucac4=[cast(Any, _NoNameStar())])
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    assert out[0].name == ''


def test_reduce_catalogs_skips_bins_below_obs_min_vmag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reduce_catalogs`` skips magnitude bins entirely below the obs floor.

    The star's ``vmag`` is chosen to fall strictly inside one bin so the
    catalog filter does not return it twice (catalog filtering is
    inclusive on both bin endpoints; values that fall on a bin boundary
    appear in two consecutive bins).
    """
    obs = FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(10, 10),
        ra_dec_limits_ext_rad=(0.0, 0.5, -0.1, 0.1),
        star_min_vmag=10.0,  # high floor: bins ending below 10 are skipped
        star_max_vmag=15.0,
    )
    star = make_star(unique_number=1, ra=0.1, dec=0.0, vmag=11.2)
    install_fake_catalogs(monkeypatch, ucac4=[star])
    out = reduce_catalogs(cast(Any, obs), DEFAULT_CONFIG)
    assert len(out) == 1


def test_reduce_catalogs_upgrades_pretty_name_from_lower_precedence_catalog(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A nameless higher-precedence star inherits the named lower-precedence pretty_name."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[
            make_star(unique_number=1, ra=0.10, dec=0.0, vmag=5.0, name=''),
        ],
        ybsc=[
            make_star(unique_number=2, ra=0.10, dec=0.0, vmag=5.0, name='Sirius', b_v=0.0),
        ],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    assert len(out) == 1
    survivor = out[0]
    assert survivor.catalog_name == 'ucac4'
    assert survivor.pretty_name == 'Sirius'


def test_stars_in_extfov_with_radec_movement_uses_movement_branch(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """Passing ``radec_movement`` exercises the smear-aware bracket projection."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0)],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
        radec_movement=(1e-5, 1e-5),
    )
    # The movement-aware branch projects start / end through tfrac
    # offsets; the resulting movement vector is non-zero per ``FakeObs``.
    assert len(out) == 1


def test_stars_in_extfov_returns_empty_when_every_star_drops_pre_projection(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A reduction whose every record is dropped lands in the empty-projection branch.

    Inject a star that the catalog filter passes (RA / DEC / vmag in
    range) but whose ``ra`` is set to ``None`` after filtering so the
    reduction's ``ra is None`` skip drops it before
    ``_project_stars_to_fov`` runs.  The projection helper sees an empty
    list and returns ``[]`` directly via the early-return branch.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.05, dec=0.05, vmag=5.0)],
    )
    cat = cast(Any, nav_catalog.get_ucac4_catalog())
    cat.stars[0].ra = None
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert out == []


def test_stars_in_extfov_defensive_none_filter_in_reduction_body(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """The production reduction's None-filter drops records the catalog yields.

    The shim catalog yields every record (including ones with None RA /
    DEC / vmag) so the production code's defensive
    ``if star.ra is None or ...: continue`` branch is reached.
    """
    bad = make_star(unique_number=2, ra=0.05, dec=0.05, vmag=5.0)
    bad.vmag = None  # pass through find_stars's yield-anyway branch
    good = make_star(unique_number=1, ra=0.05, dec=0.05, vmag=5.0)
    install_fake_catalogs(monkeypatch, ucac4=[good, bad])
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert [s.unique_number for s in out] == [1]


def test_stars_in_extfov_ybsc_skips_records_with_none_vmag(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """YBSC's per-record vmag-None skip is exercised by a None-vmag record."""
    bad = make_star(unique_number=2, ra=0.05, dec=0.05, vmag=5.0, b_v=0.65)
    bad.vmag = None
    install_fake_catalogs(monkeypatch, ybsc=[bad])
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ybsc',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert out == []


def test_stars_in_extfov_ybsc_uses_vmag_for_johnson_mag_b_when_no_b_v(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A YBSC record with ``b_v=None`` populates ``johnson_mag_b`` from ``vmag``."""
    install_fake_catalogs(
        monkeypatch,
        ybsc=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0, b_v=None)],
    )
    out = stars_in_extfov(
        cast(Any, fake_obs),
        DEFAULT_CONFIG,
        catalog_name='ybsc',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == 1
    assert out[0].johnson_mag_b == 5.0  # mirrors johnson_mag_v


def test_merge_catalogs_breaks_inner_loop_on_sorted_prefix() -> None:
    """A ``later`` star earlier (in dec_pm) than every ``earlier`` star hits the break.

    The inner loop relies on dec_pm-sorted ordering so it can short-circuit
    once ``prev.dec_pm - star.dec_pm > duplicate_radec``.  Configure
    earlier=[dec_pm=0.05] and later=[dec_pm=0.0] so the first prev
    is far above the star's dec_pm and the break fires.
    """
    earlier = [make_star(unique_number=1, ra=0.0, dec=0.05, vmag=5.0)]
    later = [make_star(unique_number=2, ra=0.0, dec=0.0, vmag=5.0)]
    earlier[0].dec_pm = 0.05
    earlier[0].ra_pm = 0.0
    later[0].dec_pm = 0.0
    later[0].ra_pm = 0.0
    duplicate_radec = math.radians(5.0 / 3600.0)
    out = _merge_catalogs(
        cast(list[MutableStar], earlier),
        cast(list[MutableStar], later),
        duplicate_radec=duplicate_radec,
        duplicate_vmag=2.0,
    )
    # No duplicate match -> both stars survive.
    assert {s.unique_number for s in out} == {1, 2}


def test_reduce_catalogs_collapses_saturated_ucac4_onto_ybsc_twin(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A saturated UCAC4 bright star collapses onto its native YBSC twin.

    UCAC4 reports the star three-plus magnitudes too faint (mirroring the
    Pleiades Eta Tau, catalogued near V6.7 against a true V2.9), so the
    merge kept both records; the reduction resolves the field to the single
    YBSC record, which carries the true magnitude and accurate astrometry.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=6.7)],
        ybsc=[make_star(unique_number=2, ra=0.10, dec=0.0, vmag=2.87, b_v=-0.09)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    assert len(out) == 1
    # The YBSC record (not the corrected UCAC4 record) must be the survivor.
    assert out[0].catalog_name == 'ybsc'
    assert out[0].vmag == pytest.approx(2.87)


def test_reduce_catalogs_corrects_sole_source_ucac4_in_place(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A saturated UCAC4 star with no surviving twin is corrected and flagged.

    The YBSC value is within the duplicate-magnitude threshold, so the
    merge already dropped the YBSC record as a duplicate; the UCAC4 record
    remains and adopts the YBSC magnitude, flagged as corrected.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=7.6)],
        ybsc=[make_star(unique_number=2, ra=0.10, dec=0.0, vmag=5.09, b_v=0.0)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    assert len(out) == 1
    assert out[0].catalog_name == 'ucac4'
    assert out[0].photometry_corrected is True
    assert out[0].vmag == pytest.approx(5.09)


def test_reduce_catalogs_flags_bright_star_without_ybsc_match(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A bright UCAC4 star with no co-located YBSC record is flagged saturated."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=4.0)],
        ybsc=[make_star(unique_number=2, ra=0.40, dec=0.05, vmag=3.0, b_v=0.0)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    corrected = next(s for s in out if s.catalog_name == 'ucac4')
    assert corrected.photometry_saturated is True


def test_reduce_catalogs_flags_bright_star_when_reference_field_empty(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A bright UCAC4 star is flagged even when the field has no reference at all.

    The reference catalogs are configured but return no in-field records, so
    the correction runs on an empty reference and flags the bright UCAC4
    record as saturated rather than silently trusting its aperture magnitude.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=4.0)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    corrected = next(s for s in out if s.catalog_name == 'ucac4')
    assert corrected.photometry_saturated is True


def test_reduce_catalogs_corrects_saturated_ucac4_against_tycho2(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A saturated UCAC4 star beyond YBSC reach is corrected against Tycho-2.

    YBSC is empty here (the star sits past YBSC completeness); Tycho-2
    supplies the true magnitude, which the UCAC4 record adopts.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=7.7)],
        tycho2=[make_star(unique_number=2, ra=0.10, dec=0.0, vmag=6.9)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    ucac4 = next(s for s in out if s.catalog_name == 'ucac4')
    assert ucac4.vmag == pytest.approx(6.9)


def test_reduce_catalogs_tycho2_correction_flags_corrected(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A UCAC4 star corrected against Tycho-2 carries the corrected flag."""
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=7.7)],
        tycho2=[make_star(unique_number=2, ra=0.10, dec=0.0, vmag=6.9)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    ucac4 = next(s for s in out if s.catalog_name == 'ucac4')
    assert ucac4.photometry_corrected is True


def test_reduce_catalogs_does_not_flag_genuine_faint_star(
    monkeypatch: pytest.MonkeyPatch, fake_obs: FakeObs
) -> None:
    """A genuine V7 to V8 star Tycho-2 covers is not flagged saturated.

    The UCAC4 reading agrees with Tycho-2, so the star is neither corrected
    nor flagged even though YBSC (empty here) does not reach it.
    """
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.10, dec=0.0, vmag=7.3)],
        tycho2=[make_star(unique_number=2, ra=0.10, dec=0.0, vmag=7.3)],
    )
    out = reduce_catalogs(cast(Any, fake_obs), DEFAULT_CONFIG)
    survivor = out[0]
    assert survivor.photometry_saturated is False


@pytest.mark.parametrize(
    ('overhang_px', 'expected_count'),
    [(-0.3, 1), (0.3, 0)],
    ids=['window-fits', 'window-spills'],
)
def test_edge_gate_keeps_a_star_whose_psf_window_fits(
    monkeypatch: pytest.MonkeyPatch, overhang_px: float, expected_count: int
) -> None:
    """The extfov edge gate is decided where the PSF window lives: array indices.

    A 100-row frame padded by 10 makes index 109 the last row an extfov array
    has, and a 5x5 window reaches two rows either side of its star.  The star
    is placed so that window ends just inside that last row, and then just
    past it.  The record states the position in oops uv, half a pixel above
    the index, so a gate comparing that uv against the index bound would
    reject the star whose window fits.
    """
    obs = FakeObs(
        data=np.zeros((100, 100), dtype=np.float64),
        extfov_margin_vu=(10, 10),
        ra_dec_limits_ext_rad=(0.0, 0.5, -0.1, 0.1),
        star_min_vmag=0.0,
        star_max_vmag=15.0,
    )
    psf_half_v = obs.star_psf_size(None)[0] // 2
    # The far edge of the window overhangs the last extfov row by
    # ``overhang_px``; the record states that star's index in uv.
    star_uv_v = obs.extfov_v_max - psf_half_v + overhang_px + STAR_UV_DATUM_PX
    obs.radec_to_uv = lambda _ra, _dec, _tfrac: (0.0, star_uv_v)
    install_fake_catalogs(
        monkeypatch,
        ucac4=[make_star(unique_number=1, ra=0.1, dec=0.0, vmag=5.0)],
    )
    out = stars_in_extfov(
        cast(Any, obs),
        DEFAULT_CONFIG,
        catalog_name='ucac4',
        mag_min=0.0,
        mag_max=15.0,
    )
    assert len(out) == expected_count
