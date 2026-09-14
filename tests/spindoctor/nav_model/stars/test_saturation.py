"""Tests for ``spindoctor.nav_model.stars.saturation``.

The bright-end correction replaces UCAC4's saturated aperture magnitude
with the true V from a YBSC or Tycho-2 reference wherever the catalogs
positionally match, and flags bright records with no reference in either
catalog so downstream consumers do not trust their magnitude.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
from tests.shims import make_star

from spindoctor.nav_model.stars.saturation import (
    SATURATION_MATCH_MAX_WIDEN_RADII,
    SATURATION_MATCH_WIDEN_PER_MAG,
    UCAC4_SATURATION_VMAG_LIMIT,
    _best_mag_aware_match,
    _nearest_within,
    _widened_radius,
    correct_saturated_vmags,
    correct_star_photometry,
    reference_photometry,
)
from spindoctor.support.types import MutableStar

# Two Pleiades bright members near their catalog positions (radians).
_ETA_TAU = (math.radians(56.8713), math.radians(24.1050))
_MEROPE = (math.radians(56.5817), math.radians(23.9483))
_MATCH_RAD = math.radians(5.0 / 3600.0)
_DVMAG = 3.0
_ORDER = ('ucac4', 'tycho2', 'ybsc')


def _star(**kwargs: object) -> MutableStar:
    """Build a ``MutableStar`` record from keyword overrides."""
    return cast(MutableStar, make_star(**kwargs))


def _ucac4(ra: float, dec: float, vmag: float) -> MutableStar:
    """A UCAC4 record with proper-motion position pinned to ``(ra, dec)``."""
    return _star(catalog_name='ucac4', ra_pm=ra, dec_pm=dec, vmag=vmag, dn=1.0)


def _ybsc(ra: float, dec: float, vmag: float, *, b_v: float = 0.0) -> MutableStar:
    """A YBSC reference record with Johnson photometry populated."""
    return _star(
        catalog_name='ybsc',
        ra_pm=ra,
        dec_pm=dec,
        vmag=vmag,
        b_v=b_v,
        johnson_mag_v=vmag,
        johnson_mag_b=vmag + b_v,
    )


def _tycho2(ra: float, dec: float, vmag: float, *, spectral_class: str = 'G0') -> MutableStar:
    """A Tycho-2 reference record: real V, no Johnson color (as reduced)."""
    return _star(
        catalog_name='tycho2',
        ra_pm=ra,
        dec_pm=dec,
        vmag=vmag,
        spectral_class=spectral_class,
        johnson_mag_v=None,
        johnson_mag_b=None,
    )


# A separation just past the base radius but inside the widened radius that a
# multi-magnitude saturation earns, used to probe the astrometry-disagreeing
# duplicate collapse.
_DISPLACED_RAD = math.radians(9.0 / 3600.0)


def test_saturation_limit_is_bright_end() -> None:
    """The documented limit sits in the bright regime UCAC4 mishandles."""
    assert UCAC4_SATURATION_VMAG_LIMIT == 8.0


def test_correct_star_photometry_replaces_saturated_vmag() -> None:
    """A saturated UCAC4 star adopts the matched YBSC magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87, b_v=-0.09)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(2.87)


def test_correct_star_photometry_sets_corrected_flag() -> None:
    """A corrected star is flagged so provenance is visible downstream."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_corrected is True


def test_correct_star_photometry_recomputes_dn() -> None:
    """The flux-derived DN follows the corrected magnitude, not the old one."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.dn == pytest.approx(2.512 ** -(2.87 - 4.0))


def test_correct_star_photometry_copies_johnson_mags() -> None:
    """Johnson B follows the YBSC color once corrected."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87, b_v=0.5)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_b == pytest.approx(3.37)


def test_correct_star_photometry_keeps_bv_consistent_without_johnson_b() -> None:
    """When the YBSC ref has no Johnson B, ``b_v`` stays consistent (zero)."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ref.johnson_mag_b = None
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.b_v == pytest.approx(0.0)


def test_correct_star_photometry_uses_vmag_when_ref_johnson_v_missing() -> None:
    """A YBSC ref with no Johnson V falls back to its V-magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ref.johnson_mag_v = None
    ref.johnson_mag_b = None
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_v == pytest.approx(2.87)


def test_correct_star_photometry_clears_faked_flag() -> None:
    """A corrected star no longer carries a faked-photometry mark."""
    star = _ucac4(*_ETA_TAU, 6.68)
    star.johnson_mag_faked = True
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_faked is False


def test_correct_star_photometry_flags_unmatched_bright_star() -> None:
    """A bright star absent from YBSC is flagged saturated, not corrected."""
    star = _ucac4(*_ETA_TAU, 6.68)
    far = _ybsc(math.radians(120.0), math.radians(-10.0), 3.0)
    correct_star_photometry(
        [star], [far], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is True


def test_correct_star_photometry_keeps_unmatched_bright_vmag() -> None:
    """An unmatched bright star keeps its (untrusted) catalog magnitude."""
    star = _ucac4(*_ETA_TAU, 6.68)
    far = _ybsc(math.radians(120.0), math.radians(-10.0), 3.0)
    correct_star_photometry(
        [star], [far], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(6.68)


def test_correct_star_photometry_ignores_faint_star() -> None:
    """A faint star is not a candidate even next to a bright YBSC star."""
    star = _ucac4(*_ETA_TAU, 11.0)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(11.0)


def test_correct_star_photometry_leaves_faint_star_unflagged() -> None:
    """A faint star gets neither the corrected nor the saturated flag."""
    star = _ucac4(*_ETA_TAU, 11.0)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is False


def test_correct_star_photometry_skips_ybsc_records() -> None:
    """A YBSC record already in the list is never treated as a candidate."""
    star = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [star], [star], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_corrected is False


def test_correct_star_photometry_respects_limit_boundary() -> None:
    """A star exactly at the limit is not corrected (strict inequality)."""
    star = _ucac4(*_ETA_TAU, 8.0)
    ref = _ybsc(*_ETA_TAU, 3.0)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(8.0)


def test_correct_star_photometry_collapses_saturated_ucac4_onto_ybsc_twin() -> None:
    """A saturated UCAC4 star collapses onto its native YBSC twin.

    The YBSC record read the star at its true magnitude, so it also placed
    it accurately: it wins and the saturated UCAC4 record is dropped.
    """
    ucac4 = _ucac4(*_ETA_TAU, 6.68)
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    out = correct_star_photometry(
        [ucac4, ybsc],
        [ybsc],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ybsc]


def test_correct_star_photometry_keeps_unconsumed_ybsc_star() -> None:
    """A YBSC bright star with no UCAC4 twin stays in the merged list."""
    ucac4 = _ucac4(*_ETA_TAU, 6.68)
    ybsc_match = _ybsc(*_ETA_TAU, 2.87)
    ybsc_solo = _ybsc(*_MEROPE, 4.18)
    out = correct_star_photometry(
        [ucac4, ybsc_match, ybsc_solo],
        [ybsc_match, ybsc_solo],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert ybsc_solo in out


def test_correct_star_photometry_collapses_revealed_tycho2_duplicate() -> None:
    """Correcting a saturated star collapses its co-located Tycho-2 twin.

    UCAC4 lists the star saturated, so the merge kept it alongside the
    Tycho-2 record at the same position (magnitudes disagreed).  Once
    corrected they match; the Tycho-2 record, which never needed
    correcting, keeps its accurate bright-star astrometry and survives.
    """
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=3.40)
    ref = _ybsc(*_ETA_TAU, 3.40)
    out = correct_star_photometry(
        [ucac4, tycho2],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [tycho2]


def test_correct_star_photometry_revealed_duplicate_inherits_name() -> None:
    """The surviving twin inherits the dropped record's catalog name."""
    ucac4 = _star(
        catalog_name='ucac4', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.88, name='27Tau', dn=1.0
    )
    tycho2 = _star(catalog_name='tycho2', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=3.40, name='')
    ref = _ybsc(*_ETA_TAU, 3.40)
    correct_star_photometry(
        [ucac4, tycho2],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert tycho2.name == '27Tau'


def test_correct_star_photometry_keeps_higher_precedence_when_both_corrected() -> None:
    """Between two corrected twins, the higher-precedence catalog survives.

    Two candidate catalogs both saturate the same star and are corrected
    against their own co-located references; the collapse keeps the
    higher-precedence catalog (``ucac4`` over the unranked ``gaia``).
    """
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    gaia = _star(catalog_name='gaia', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.9, dn=1.0)
    ref_a = _ybsc(*_ETA_TAU, 3.40)
    ref_b = _ybsc(_ETA_TAU[0], _ETA_TAU[1] + math.radians(0.3 / 3600.0), 3.41)
    out = correct_star_photometry(
        [ucac4, gaia],
        [ref_a, ref_b],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ucac4]


def test_correct_star_photometry_both_corrected_prefers_precedence_regardless_of_order() -> None:
    """A corrected higher-precedence twin survives even when listed second."""
    gaia = _star(catalog_name='gaia', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.9, dn=1.0)
    ucac4 = _ucac4(*_ETA_TAU, 6.88)
    ref_a = _ybsc(*_ETA_TAU, 3.40)
    ref_b = _ybsc(_ETA_TAU[0], _ETA_TAU[1] + math.radians(0.3 / 3600.0), 3.41)
    out = correct_star_photometry(
        [gaia, ucac4],
        [ref_a, ref_b],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ucac4]


def test_correct_star_photometry_ignores_none_vmag_neighbour() -> None:
    """A co-located record with no magnitude is never treated as a duplicate."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    ghost = _star(catalog_name='ucac4', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=None)
    out = correct_star_photometry(
        [star, ghost],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert ghost in out


def test_correct_star_photometry_keeps_corrected_star_with_distant_neighbour() -> None:
    """A corrected star with only a distant neighbor is kept as-is."""
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    far = _star(catalog_name='ucac4', ra_pm=_MEROPE[0], dec_pm=_MEROPE[1], vmag=9.0)
    out = correct_star_photometry(
        [star, far],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert star in out
    assert far in out


def test_correct_star_photometry_returns_same_list_when_nothing_corrected() -> None:
    """With no correction the original list object is returned unchanged."""
    star = _ucac4(*_ETA_TAU, 11.0)
    stars = [star]
    out = correct_star_photometry(
        stars, [], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert out is stars


def test_reference_photometry_skips_none_vmag() -> None:
    """Reference records with no magnitude are dropped from the arrays."""
    good = _ybsc(*_ETA_TAU, 2.87)
    bad = _ybsc(*_MEROPE, 4.18)
    bad.vmag = None
    ref_ra, _, _, ref_stars = reference_photometry([good, bad], _MATCH_RAD)
    assert ref_ra.shape == (1,)
    assert ref_stars == [good]


def test_reference_photometry_prefers_ybsc_over_colocated_tycho2() -> None:
    """A Tycho-2 record co-located with a YBSC record is dropped (YBSC wins)."""
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    tycho2 = _tycho2(*_ETA_TAU, 2.84)
    _, _, _, ref_stars = reference_photometry([ybsc, tycho2], _MATCH_RAD)
    assert ref_stars == [ybsc]


def test_reference_photometry_keeps_tycho2_beyond_ybsc() -> None:
    """A Tycho-2 record with no nearby YBSC record is kept in the reference."""
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    tycho2 = _tycho2(*_MEROPE, 7.2)
    _, _, ref_vmag, ref_stars = reference_photometry([ybsc, tycho2], _MATCH_RAD)
    assert tycho2 in ref_stars
    assert 7.2 in ref_vmag


def test_reference_photometry_skips_none_vmag_tycho2() -> None:
    """A Tycho-2 record with no magnitude is dropped before the YBSC check."""
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    tycho2 = _tycho2(*_MEROPE, 7.2)
    tycho2.vmag = None
    _, _, _, ref_stars = reference_photometry([ybsc, tycho2], _MATCH_RAD)
    assert tycho2 not in ref_stars


def test_nearest_within_returns_none_for_empty_reference() -> None:
    """An empty reference set yields no match."""
    empty = np.asarray([], dtype=np.float64)
    assert _nearest_within(0.0, 0.0, empty, empty, _MATCH_RAD) is None


def test_nearest_within_returns_none_beyond_radius() -> None:
    """A reference star just outside the radius is not matched."""
    ref_ra = np.asarray([math.radians(1.0)], dtype=np.float64)
    ref_dec = np.asarray([0.0], dtype=np.float64)
    assert _nearest_within(0.0, 0.0, ref_ra, ref_dec, _MATCH_RAD) is None


def test_nearest_within_matches_across_ra_seam() -> None:
    """A reference just across the RA 0/2*pi seam is matched, not missed."""
    ref_ra = np.asarray([2.0 * math.pi - 1e-6], dtype=np.float64)
    ref_dec = np.asarray([0.0], dtype=np.float64)
    assert _nearest_within(1e-6, 0.0, ref_ra, ref_dec, _MATCH_RAD) == 0


def test_correct_star_photometry_matches_across_ra_seam() -> None:
    """A saturated star is corrected against a YBSC twin across the RA seam."""
    star = _star(catalog_name='ucac4', ra_pm=1e-6, dec_pm=0.0, vmag=6.68, dn=1.0)
    ref = _star(
        catalog_name='ybsc',
        ra_pm=2.0 * math.pi - 1e-6,
        dec_pm=0.0,
        vmag=2.87,
        johnson_mag_v=2.87,
        johnson_mag_b=2.87,
    )
    correct_star_photometry(
        [star],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert star.photometry_corrected is True


def test_correct_saturated_vmags_adopts_ybsc_for_match() -> None:
    """A saturated catalog star reports the YBSC magnitude after correction."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87)]


def test_correct_saturated_vmags_keeps_faint_catalog_star() -> None:
    """A faint catalog star is never corrected, even beside a bright YBSC star.

    The co-located YBSC record is not re-added (it has a catalog
    counterpart), so the faint reading is the only surviving magnitude.
    """
    catalog = [(*_ETA_TAU, 11.0)]
    ybsc = [(*_ETA_TAU, 2.87)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(11.0)]


def test_correct_saturated_vmags_keeps_unmatched_bright_star() -> None:
    """A bright catalog star with no YBSC match keeps its magnitude."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(math.radians(120.0), math.radians(-10.0), 3.0)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == sorted([6.68, 3.0])


def test_correct_saturated_vmags_appends_ybsc_only_star() -> None:
    """A bright YBSC star the catalog missed is added to the result."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87), (*_MEROPE, 4.18)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87), pytest.approx(4.18)]


def test_correct_saturated_vmags_does_not_double_count_uncorrected_match() -> None:
    """A YBSC star co-located with an at-limit catalog star is not re-added.

    The catalog value sits at/above the limit (so it is left uncorrected),
    but its YBSC counterpart must not be appended a second time.
    """
    catalog = [(*_ETA_TAU, 8.5)]
    ybsc = [(*_ETA_TAU, 6.4)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(8.5)]


def test_correct_saturated_vmags_is_sorted() -> None:
    """The returned magnitudes are sorted ascending (brightest first)."""
    catalog = [(*_MEROPE, 6.9), (*_ETA_TAU, 6.68), (math.radians(57.0), math.radians(24.0), 10.2)]
    ybsc = [(*_ETA_TAU, 2.87), (*_MEROPE, 4.18)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == sorted(out)


# --------------------------------------------------------------------------
# Tycho-2 as an additional reference
# --------------------------------------------------------------------------


def test_correct_star_photometry_corrects_against_tycho2_reference() -> None:
    """A saturated UCAC4 star adopts a Tycho-2 reference V beyond YBSC reach."""
    star = _ucac4(*_ETA_TAU, 7.7)
    ref = _tycho2(*_ETA_TAU, 6.9)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.vmag == pytest.approx(6.9)


def test_correct_star_photometry_tycho2_correction_fakes_colour() -> None:
    """A Tycho-2 correction fakes the color from spectral class (faked flag set)."""
    star = _ucac4(*_ETA_TAU, 7.7)
    star.spectral_class = 'G0'
    ref = _tycho2(*_ETA_TAU, 6.9)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_faked is True


def test_correct_star_photometry_tycho2_correction_sets_johnson_v() -> None:
    """A Tycho-2 correction adopts Tycho-2's V as the Johnson V."""
    star = _ucac4(*_ETA_TAU, 7.7)
    ref = _tycho2(*_ETA_TAU, 6.9)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.johnson_mag_v == pytest.approx(6.9)


def test_correct_star_photometry_tycho2_reference_never_a_candidate() -> None:
    """A Tycho-2 record in the star list is never corrected against itself."""
    tycho2 = _tycho2(*_ETA_TAU, 6.9)
    correct_star_photometry(
        [tycho2], [tycho2], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert tycho2.photometry_corrected is False


def test_correct_star_photometry_genuine_faint_star_not_flagged() -> None:
    """A genuine V7 to V8 star matching Tycho-2 is not flagged saturated.

    The star is beyond YBSC completeness but agrees with its Tycho-2
    reference, so it is neither corrected nor flagged: using Tycho-2 as a
    reference is what keeps the flag honest.
    """
    star = _ucac4(*_ETA_TAU, 7.3)
    ref = _tycho2(*_ETA_TAU, 7.3)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is False


def test_correct_star_photometry_genuine_faint_star_keeps_vmag() -> None:
    """A genuine V7 to V8 star that agrees with Tycho-2 keeps its magnitude."""
    star = _ucac4(*_ETA_TAU, 7.3)
    ref = _tycho2(*_ETA_TAU, 7.3)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_corrected is False


# --------------------------------------------------------------------------
# Magnitude-aware widened match (issue #365 core bug)
# --------------------------------------------------------------------------


def test_correct_star_photometry_collapses_astrometry_disagreeing_duplicate() -> None:
    """A saturated record displaced beyond the base radius still collapses.

    The saturated UCAC4 position and its YBSC twin disagree by more than the
    base match radius; the magnitude-widened match corrects the UCAC4 record
    against the twin and then collapses the revealed duplicate onto it.
    """
    ucac4 = _ucac4(_ETA_TAU[0] + _DISPLACED_RAD, _ETA_TAU[1], 6.68)
    ybsc = _ybsc(*_ETA_TAU, 2.87)
    out = correct_star_photometry(
        [ucac4, ybsc],
        [ybsc],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert out == [ybsc]


def test_correct_star_photometry_no_widening_for_small_gap() -> None:
    """A well-behaved bright star displaced past the base radius is not matched.

    Its magnitude agrees with the reference, so the match radius is not
    widened and the displaced record stays flagged rather than corrected.
    """
    star = _ucac4(_ETA_TAU[0] + _DISPLACED_RAD, _ETA_TAU[1], 7.3)
    ref = _tycho2(*_ETA_TAU, 7.2)
    correct_star_photometry(
        [star], [ref], match_radius_rad=_MATCH_RAD, duplicate_vmag=_DVMAG, catalog_order=_ORDER
    )
    assert star.photometry_saturated is True


# --------------------------------------------------------------------------
# Same-reference guard (a reference corrects at most one record)
# --------------------------------------------------------------------------


def test_correct_star_photometry_keeps_neighbour_beyond_vmag_tolerance() -> None:
    """A co-located neighbor outside the magnitude tolerance is not collapsed.

    After correction the saturated record reads V2.87; a genuinely different
    star at the same position but far in magnitude is not its duplicate, so
    the collapse leaves it in place (the duplicate-magnitude guard holds even
    inside the widened collapse radius).
    """
    star = _ucac4(*_ETA_TAU, 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    other = _star(catalog_name='ucac4', ra_pm=_ETA_TAU[0], dec_pm=_ETA_TAU[1], vmag=6.5, dn=1.0)
    out = correct_star_photometry(
        [star, other],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert other in out


def test_correct_star_photometry_one_reference_corrects_one_record() -> None:
    """Two distinct saturated records cannot both consume one reference.

    Both UCAC4 records fall inside the widened radius of the single
    reference; the first consumes it and the second, left with no available
    reference, is flagged rather than corrected onto the same star.
    """
    near = _ucac4(*_ETA_TAU, 6.68)
    other = _ucac4(_ETA_TAU[0] + _DISPLACED_RAD, _ETA_TAU[1], 6.68)
    ref = _ybsc(*_ETA_TAU, 2.87)
    correct_star_photometry(
        [near, other],
        [ref],
        match_radius_rad=_MATCH_RAD,
        duplicate_vmag=_DVMAG,
        catalog_order=_ORDER,
    )
    assert near.photometry_corrected is True
    assert other.photometry_saturated is True


# --------------------------------------------------------------------------
# Helper unit tests (coverage)
# --------------------------------------------------------------------------


def test_widened_radius_grows_with_gap() -> None:
    """A larger magnitude gap earns a larger match radius."""
    small = _widened_radius(0.5, _MATCH_RAD)
    large = _widened_radius(3.0, _MATCH_RAD)
    assert large > small


def test_widened_radius_clamps_negative_gap() -> None:
    """A candidate brighter than the reference gets exactly the base radius."""
    assert _widened_radius(-2.0, _MATCH_RAD) == pytest.approx(_MATCH_RAD)


def test_widened_radius_caps_at_max() -> None:
    """The widened radius never exceeds the documented cap."""
    huge = _widened_radius(100.0, _MATCH_RAD)
    assert huge == pytest.approx(_MATCH_RAD * SATURATION_MATCH_MAX_WIDEN_RADII)


def test_widened_radius_uses_per_mag_slope() -> None:
    """The radius follows the per-magnitude slope below the cap."""
    got = _widened_radius(2.0, _MATCH_RAD)
    expected = _MATCH_RAD * (1.0 + SATURATION_MATCH_WIDEN_PER_MAG * 2.0)
    assert got == pytest.approx(expected)


def test_best_mag_aware_match_returns_none_for_empty_reference() -> None:
    """An empty reference set yields no match."""
    empty = np.asarray([], dtype=np.float64)
    assert _best_mag_aware_match(0.0, 0.0, 6.0, empty, empty, empty, _MATCH_RAD) is None


def test_best_mag_aware_match_respects_available_mask() -> None:
    """A reference masked unavailable is not matched even when in range."""
    ref_ra = np.asarray([_ETA_TAU[0]], dtype=np.float64)
    ref_dec = np.asarray([_ETA_TAU[1]], dtype=np.float64)
    ref_vmag = np.asarray([2.87], dtype=np.float64)
    available = np.asarray([False], dtype=np.bool_)
    idx = _best_mag_aware_match(
        _ETA_TAU[0], _ETA_TAU[1], 6.68, ref_ra, ref_dec, ref_vmag, _MATCH_RAD, available=available
    )
    assert idx is None


def test_best_mag_aware_match_matches_coincident_reference() -> None:
    """A coincident reference is returned over a fainter distant one."""
    ref_ra = np.asarray([_ETA_TAU[0], _MEROPE[0]], dtype=np.float64)
    ref_dec = np.asarray([_ETA_TAU[1], _MEROPE[1]], dtype=np.float64)
    ref_vmag = np.asarray([2.87, 4.18], dtype=np.float64)
    idx = _best_mag_aware_match(
        _ETA_TAU[0], _ETA_TAU[1], 6.68, ref_ra, ref_dec, ref_vmag, _MATCH_RAD
    )
    assert idx == 0


def test_best_mag_aware_match_prefers_brighter_true_twin_over_nearer_unrelated() -> None:
    """A nearer unrelated bright star does not capture the saturated candidate.

    The candidate reads V6.7 (a saturated bright star).  A V4.0 reference
    sits 4 arcsec away and the candidate's true V2.8 twin sits 8 arcsec
    away; both fall inside their own magnitude-widened reach.  A pure
    nearest-neighbour rule would correct the candidate to the nearer,
    unrelated V4.0 star; the brightness preference returns the V2.8 twin
    instead.
    """
    # Candidate at Eta Tau's true position; references offset east by 4 and 8
    # arcsec.  cos(dec) ~ 1 over this small field, so a pure RA offset gives
    # the intended separations to within the small-angle tolerance.
    arcsec = math.radians(1.0 / 3600.0)
    cand_ra, cand_dec = _ETA_TAU
    unrelated_ra = cand_ra + 4.0 * arcsec / math.cos(cand_dec)
    twin_ra = cand_ra + 8.0 * arcsec / math.cos(cand_dec)
    ref_ra = np.asarray([unrelated_ra, twin_ra], dtype=np.float64)
    ref_dec = np.asarray([cand_dec, cand_dec], dtype=np.float64)
    ref_vmag = np.asarray([4.0, 2.8], dtype=np.float64)
    idx = _best_mag_aware_match(cand_ra, cand_dec, 6.7, ref_ra, ref_dec, ref_vmag, _MATCH_RAD)
    assert idx == 1


# --------------------------------------------------------------------------
# Plain-magnitude path: Tycho-2 reference and widened match
# --------------------------------------------------------------------------


def test_correct_saturated_vmags_adopts_tycho2_for_match() -> None:
    """A saturated catalog star reports the Tycho-2 magnitude after correction."""
    catalog = [(*_ETA_TAU, 7.7)]
    out = correct_saturated_vmags(catalog, [], [(*_ETA_TAU, 6.9)], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(6.9)]


def test_correct_saturated_vmags_prefers_ybsc_over_tycho2() -> None:
    """A Tycho-2 triple co-located with a YBSC triple is not counted twice."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87)]
    tycho2 = [(*_ETA_TAU, 2.84)]
    out = correct_saturated_vmags(catalog, ybsc, tycho2, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87)]


def test_correct_saturated_vmags_collapses_displaced_double_count() -> None:
    """A displaced saturated star does not double count against its twin.

    The saturated catalog position and its YBSC twin disagree by more than
    the base radius; the magnitude-widened match collapses them so the field
    reports a single bright star, not two.
    """
    catalog = [(_ETA_TAU[0] + _DISPLACED_RAD, _ETA_TAU[1], 6.68)]
    ybsc = [(*_ETA_TAU, 2.87)]
    out = correct_saturated_vmags(catalog, ybsc, [], match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87)]


def test_correct_saturated_vmags_keeps_reading_for_small_gap() -> None:
    """A small-gap match keeps the catalog reading, matching the reduction.

    The catalog value agrees with the reference to within the no-op band, so
    it is not saturated: the screen keeps its own magnitude rather than
    adopting the reference (exactly as ``correct_star_photometry`` leaves the
    record untouched), and the consumed reference is not counted a second
    time.
    """
    catalog = [(*_ETA_TAU, 7.5)]
    tycho2 = [(*_ETA_TAU, 7.2)]
    out = correct_saturated_vmags(catalog, [], tycho2, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(7.5)]


def test_correct_saturated_vmags_appends_tycho2_only_star() -> None:
    """A Tycho-2 bright star the catalog missed is added to the result."""
    catalog = [(*_ETA_TAU, 6.68)]
    ybsc = [(*_ETA_TAU, 2.87)]
    tycho2 = [(*_MEROPE, 7.1)]
    out = correct_saturated_vmags(catalog, ybsc, tycho2, match_radius_rad=_MATCH_RAD)
    assert out == [pytest.approx(2.87), pytest.approx(7.1)]
