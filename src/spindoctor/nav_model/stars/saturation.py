"""UCAC4 bright-end photometric saturation correction against YBSC and Tycho-2.

UCAC4 aperture V-magnitudes saturate at the bright end: the reported
magnitude reads systematically too faint for bright stars.  A UCAC4-vs-
YBSC cross-match shows the residual (UCAC4 minus the true Johnson V) is a
few tenths of a magnitude near V8 and climbs to several magnitudes by
V3.  For the Pleiades, UCAC4 lists Eta Tau near V6.7 when the true value
is V2.9, and 27 Tau near V6.7 against a true V3.6.

Both the per-star detectability prediction and the navigable-content
screen read the catalog V-magnitude, so a field of two genuinely bright
stars plus a faint remainder is mistaken for a dozen comparably faint
stars: the two dominant anchors are dragged down into the faint
population and the pipeline cannot tell they are there.

Two catalogs supply the trusted brightness the correction reads against.
YBSC (the Yale Bright Star Catalog) carries real Johnson V and B
photometry but is complete only through about V6.5.  Tycho-2's
star-mapper photometry does not saturate at the bright end and is
complete through about V11: a UCAC4-vs-Tycho-2-vs-YBSC cross-match over
the Pleiades and Hyades shows Tycho-2 agreeing with YBSC to a few
hundredths of a magnitude (Eta Tau V2.84 vs YBSC V2.87, Theta-2 Tau V3.39
vs V3.40) where UCAC4 reads the same stars near V6.7.  The reference set
is the in-field union of the two: YBSC wherever it covers a star (it
carries a self-consistent Johnson V/B pair), and Tycho-2 elsewhere, which
reaches the V6.5 to V8 stars YBSC misses.  A star corrected against YBSC
inherits its Johnson pair; a star corrected against Tycho-2 adopts
Tycho-2's V and fakes its color from spectral class, because this
pipeline discards Tycho-2's own color.  Only UCAC4 saturates: Tycho-2
and YBSC records are references, never correction candidates, so a
reference is never corrected against itself.

Extending the reference to Tycho-2 also keeps the ``photometry_saturated``
flag honest.  A genuine V7 to V8 star absent from YBSC used to be flagged
saturated merely for sitting beyond YBSC completeness; Tycho-2 covers it,
so it matches a reference, agrees with it, and is left unflagged.  Only a
bright star with no reference in either catalog -- now genuinely rare --
is flagged, and its magnitude is treated as an untrusted lower bound on
brightness rather than a trustworthy reading.

Because UCAC4's saturated magnitude disagrees with the Tycho-2/YBSC value
by more than the merge's duplicate-magnitude tolerance, a bright star that
UCAC4 saturates is kept twice after the merge: a saturated UCAC4 record
plus its accurate Tycho-2 (or YBSC) twin.  Correcting the UCAC4 record
brings the two magnitudes back into agreement and exposes the duplicate,
which the final collapse resolves in favor of the twin that never needed
correcting -- the same saturation that corrupts UCAC4's photometry also
displaces its astrometry, so the catalog that read the star at its true
magnitude also placed it accurately.  That astrometric displacement is
what a fixed-radius position match misses: a badly saturated UCAC4 record
can sit farther from its true-magnitude twin than the base match radius,
so both the reference match and the duplicate collapse widen the radius in
proportion to the magnitude gap (a larger saturation error implies a
larger displacement).  A reference star is consumed by at most one
corrected record, so one true-magnitude star cannot absorb two distinct
saturated records.

The correction is exposed both as an in-place pass over ``MutableStar``
records (used by the catalog reduction, which feeds the detectability
model) and as a plain-magnitude helper (used by the navigable-content
screen), so every consumer of catalog brightness shares one policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from spindoctor.nav_model.stars.predicted_snr import SCLASS_TO_B_MINUS_V
from spindoctor.support.flux import clean_sclass

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from collections.abc import Sequence

    from spindoctor.support.types import MutableStar, NDArrayBoolType, NDArrayFloatType

__all__ = [
    'REFERENCE_CATALOGS',
    'SATURATION_CORRECTION_MIN_MAG',
    'SATURATION_MATCH_MAX_WIDEN_RADII',
    'SATURATION_MATCH_WIDEN_PER_MAG',
    'UCAC4_SATURATION_VMAG_LIMIT',
    'correct_saturated_vmags',
    'correct_star_photometry',
    'reference_photometry',
]


UCAC4_SATURATION_VMAG_LIMIT: float = 8.0
"""V-magnitude brighter than which UCAC4 aperture photometry is untrusted.

Chosen from a UCAC4-vs-YBSC cross-match: the residual (UCAC4 minus the
true Johnson V) is a few tenths of a magnitude near V8 and climbs to
about 4 magnitudes by V3, and UCAC4 itself documents its aperture
photometry as reliable only over roughly V9 to V14.  8.0 is a
conservative onset that captures every saturated Pleiades member
observed (reported by UCAC4 between V6.5 and V7.6) without reaching into
the regime where the aperture magnitude is trustworthy.
"""


SATURATION_CORRECTION_MIN_MAG: float = 0.5
"""Smallest catalog-versus-reference magnitude gap treated as real saturation.

A bright record whose magnitude already agrees with the reference to
within this tolerance is not saturated -- its photometry (and, for a
bright star, its astrometry) is trustworthy -- so it is left untouched and
unflagged.  Only a larger gap marks a genuinely saturated record to
correct and, absent any reference, to flag.  The value sits just above the
catalog-versus-reference scatter for well-behaved bright stars and well
below the multi-magnitude error of a truly saturated one.
"""


SATURATION_MATCH_WIDEN_PER_MAG: float = 0.5
"""Extra position-match reach, in base radii, per magnitude of saturation gap.

The same saturation that corrupts UCAC4's photometry also displaces its
astrometry, so a badly saturated record can sit farther from its
true-magnitude reference than the base match radius.  The match radius is
widened by this many base radii for each magnitude by which the record
reads fainter than a reference star, so the reference match and the
duplicate collapse still fire when astrometry disagrees.  Half a base
radius per magnitude keeps a well-behaved bright star (sub-magnitude gap)
close to the base radius while giving a multi-magnitude saturation the
several-arcsecond slack its displacement needs.
"""


SATURATION_MATCH_MAX_WIDEN_RADII: float = 3.0
"""Cap on the widened position-match radius, in base radii.

The widening is bounded to stay inside the arc-minute spacing of bright
stars: at the default 5 arcsec base radius the widest match is 15 arcsec,
comfortably below the separation of bright stars even in a dense field
such as the Pleiades.  Within the widened reach the match prefers the
brightest qualifying reference (see :func:`_best_mag_aware_match`), so a
nearer unrelated bright star does not capture a saturated candidate away
from its true-magnitude twin.
"""


REFERENCE_CATALOGS: frozenset[str] = frozenset({'ybsc', 'tycho2'})
"""Catalog names whose photometry is trusted at the bright end.

Records drawn from these catalogs serve as the correction reference and
are never treated as saturation-correction candidates, so a reference is
never corrected against itself.  Only UCAC4 (and any other catalog outside
this set) saturates.
"""


def _wrap_to_pi(delta: float) -> float:
    """Wrap an angle difference (radians) into ``[-pi, pi]``.

    Keeps RA separations correct across the ``0``/``2*pi`` seam; the wrap is
    the identity for ``|delta| < pi``.

    Parameters:
        delta: Angle difference in radians.

    Returns:
        The equivalent difference in ``[-pi, pi]``.
    """
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _separation2(
    ra: float,
    dec: float,
    ref_ra: NDArrayFloatType,
    ref_dec: NDArrayFloatType,
) -> NDArrayFloatType:
    """Return squared small-angle separations from ``(ra, dec)`` to references.

    Distance is the small-angle separation on the sky (RA scaled by
    ``cos(dec)``), accurate for the few-arcsecond match radii used here.
    The RA difference is wrapped into ``[-pi, pi]`` so a field straddling
    the ``0``/``2*pi`` seam still matches (the wrap is the identity away
    from the seam).

    Parameters:
        ra: Target right ascension in radians.
        dec: Target declination in radians.
        ref_ra: Reference right ascensions in radians.
        ref_dec: Reference declinations in radians.

    Returns:
        Array of squared separations (radians squared) parallel to the
        reference arrays.
    """
    d_dec = ref_dec - dec
    d_ra = (np.remainder(ref_ra - ra + math.pi, 2.0 * math.pi) - math.pi) * math.cos(dec)
    return cast('NDArrayFloatType', d_ra * d_ra + d_dec * d_dec)


def _nearest_within(
    ra: float,
    dec: float,
    ref_ra: NDArrayFloatType,
    ref_dec: NDArrayFloatType,
    radius_rad: float,
) -> int | None:
    """Return the index of the nearest reference star within ``radius_rad``.

    Parameters:
        ra: Target right ascension in radians.
        dec: Target declination in radians.
        ref_ra: Reference right ascensions in radians.
        ref_dec: Reference declinations in radians.
        radius_rad: Match radius in radians.

    Returns:
        Index into the reference arrays of the closest star within the
        radius, or ``None`` when the arrays are empty or nothing is close
        enough.
    """
    if ref_ra.size == 0:
        return None
    d2 = _separation2(ra, dec, ref_ra, ref_dec)
    idx = int(np.argmin(d2))
    if float(d2[idx]) <= radius_rad * radius_rad:
        return idx
    return None


def _widened_radius(mag_gap: float, base_radius: float) -> float:
    """Return the position-match radius widened for a magnitude gap.

    A record that reads ``mag_gap`` magnitudes fainter than a reference is
    treated as that much more likely to be astrometrically displaced by
    saturation, so its match radius grows with the gap up to the cap.

    Parameters:
        mag_gap: Magnitude by which the candidate reads fainter than the
            reference (clamped at zero for a candidate that is not fainter).
        base_radius: Base match radius in radians.

    Returns:
        The widened match radius in radians.
    """
    factor = 1.0 + SATURATION_MATCH_WIDEN_PER_MAG * max(mag_gap, 0.0)
    return base_radius * min(factor, SATURATION_MATCH_MAX_WIDEN_RADII)


def _best_mag_aware_match(
    ra: float,
    dec: float,
    vmag: float,
    ref_ra: NDArrayFloatType,
    ref_dec: NDArrayFloatType,
    ref_vmag: NDArrayFloatType,
    base_radius: float,
    *,
    available: NDArrayBoolType | None = None,
) -> int | None:
    """Return the reference within reach that best explains the saturation.

    For each reference star the allowed radius is the base radius widened
    by the amount the candidate reads fainter than that reference (see
    :func:`_widened_radius`), so a strongly saturated candidate collapses
    against a displaced true-magnitude reference while a well-behaved one
    stays near the base radius.  Among the references whose separation
    falls inside their own allowed radius, the brightest (largest
    magnitude gap to the candidate) wins, with the nearest breaking a
    brightness tie.

    Preferring the brightest qualifying reference over the merely nearest
    keeps a nearer unrelated bright field star from capturing a saturated
    candidate away from its true twin: UCAC4 saturation drives a bright
    star's reading systematically faint, so a faint reading is best
    explained by the brightest reference that can account for it, not by
    whichever bright star happens to sit closest.  Because a bright
    unrelated reference earns a wide reach of its own from its large
    magnitude gap, a pure nearest-neighbor rule would let it win over the
    true twin when it sits closer; the brightness preference does not.

    Parameters:
        ra: Candidate right ascension in radians.
        dec: Candidate declination in radians.
        vmag: Candidate V-magnitude.
        ref_ra: Reference right ascensions in radians.
        ref_dec: Reference declinations in radians.
        ref_vmag: Reference V-magnitudes, parallel to the position arrays.
        base_radius: Base match radius in radians.
        available: Optional boolean mask (parallel to the references)
            selecting references not yet consumed by another candidate; a
            reference with ``False`` is ignored.

    Returns:
        Index of the matched reference, or ``None`` when none qualifies.
    """
    if ref_ra.size == 0:
        return None
    d2 = _separation2(ra, dec, ref_ra, ref_dec)
    gap = np.maximum(vmag - ref_vmag, 0.0)
    widen = 1.0 + SATURATION_MATCH_WIDEN_PER_MAG * gap
    factor = np.minimum(widen, SATURATION_MATCH_MAX_WIDEN_RADII)
    reach = base_radius * factor
    ok = d2 <= reach * reach
    if available is not None:
        ok = ok & available
    if not bool(ok.any()):
        return None
    # Brightest (lowest ref_vmag) qualifying reference wins; separation and
    # then index break ties for a deterministic result.
    big = float(np.max(d2)) + 1.0
    ref_vmag_key = np.where(ok, ref_vmag, np.inf)
    d2_key = np.where(ok, d2, big)
    order = np.lexsort((np.arange(ref_ra.size), d2_key, ref_vmag_key))
    return int(order[0])


def reference_photometry(
    reference: Sequence[MutableStar],
    match_radius_rad: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType, list[MutableStar]]:
    """Build the trusted-photometry reference from YBSC and Tycho-2 records.

    YBSC records with a usable magnitude are kept unconditionally; a
    non-YBSC reference record (Tycho-2) is kept only where no YBSC record
    lies within ``match_radius_rad``, so YBSC wins wherever the two overlap
    and Tycho-2 supplies the bright stars beyond YBSC completeness.

    Parameters:
        reference: Reference records (the in-field YBSC and Tycho-2 sets),
            already reduced and FOV-projected so ``ra_pm`` / ``dec_pm`` /
            ``vmag`` are populated.
        match_radius_rad: Radius (radians) within which a Tycho-2 record is
            treated as the same star as a YBSC record and dropped.

    Returns:
        A ``(ref_ra, ref_dec, ref_vmag, ref_stars)`` tuple whose arrays are
        parallel to ``ref_stars``.
    """
    ybsc_ra: list[float] = []
    ybsc_dec: list[float] = []
    for star in reference:
        if star.vmag is None:
            continue
        if star.catalog_name.lower() == 'ybsc':
            ybsc_ra.append(float(star.ra_pm))
            ybsc_dec.append(float(star.dec_pm))
    ybsc_ra_arr = np.asarray(ybsc_ra, dtype=np.float64)
    ybsc_dec_arr = np.asarray(ybsc_dec, dtype=np.float64)

    ref_ra: list[float] = []
    ref_dec: list[float] = []
    ref_vmag: list[float] = []
    ref_stars: list[MutableStar] = []
    for star in reference:
        if star.vmag is None:
            continue
        if star.catalog_name.lower() != 'ybsc':
            # Tycho-2 (or any non-YBSC reference): keep it only where YBSC
            # does not already cover the same star, so YBSC takes precedence.
            near = _nearest_within(
                float(star.ra_pm), float(star.dec_pm), ybsc_ra_arr, ybsc_dec_arr, match_radius_rad
            )
            if near is not None:
                continue
        ref_ra.append(float(star.ra_pm))
        ref_dec.append(float(star.dec_pm))
        ref_vmag.append(float(star.vmag))
        ref_stars.append(star)
    return (
        np.asarray(ref_ra, dtype=np.float64),
        np.asarray(ref_dec, dtype=np.float64),
        np.asarray(ref_vmag, dtype=np.float64),
        ref_stars,
    )


def correct_star_photometry(
    stars: list[MutableStar],
    reference: Sequence[MutableStar],
    *,
    match_radius_rad: float,
    duplicate_vmag: float,
    catalog_order: Sequence[str],
    saturation_limit: float = UCAC4_SATURATION_VMAG_LIMIT,
    min_correction_mag: float = SATURATION_CORRECTION_MIN_MAG,
) -> list[MutableStar]:
    """Correct UCAC4 bright-end saturation against a YBSC/Tycho-2 reference.

    Each candidate record (one whose catalog is not in
    :data:`REFERENCE_CATALOGS`) brighter than ``saturation_limit`` that
    positionally matches a reference star and disagrees with it by more
    than ``min_correction_mag`` adopts that reference's photometry
    (``vmag``, the Johnson pair, ``b_v``, and a recomputed ``dn``) while
    keeping its own astrometry.  A match against YBSC propagates YBSC's
    self-consistent Johnson pair; a match against Tycho-2 adopts Tycho-2's
    V and fakes the color from the candidate's spectral class.  A matched
    record is flagged ``photometry_corrected``; a bright record with no
    reference in either catalog is flagged ``photometry_saturated`` so
    downstream consumers know its magnitude is an untrusted lower bound.
    A record that matches a reference and already agrees with it is not
    saturated, so it is left untouched and unflagged.

    The position match widens with the magnitude gap so a saturated record
    displaced beyond the base radius still matches its true-magnitude
    reference.  Each reference is consumed by at most one corrected record,
    so one true-magnitude star cannot correct two distinct records.

    Correcting a saturated magnitude can reveal a duplicate the catalog
    merge missed: the same physical star from two catalogs is only deduped
    when their magnitudes agree, so a star whose UCAC4 value was several
    magnitudes too faint survived alongside its Tycho-2 or YBSC twin.  Once
    corrected the two magnitudes match, so a final pass collapses the
    revealed duplicate -- widening the collapse radius by the same
    saturation gap -- and keeps the twin that never needed correcting,
    which read the star at its true magnitude and so placed it accurately.
    The survivor inherits a name from the dropped record.  A corrected
    record with no surviving twin is retained with its own astrometry,
    since no more accurate position is available.

    Parameters:
        stars: The merged, deduplicated star list.  Records are mutated in
            place.
        reference: The full in-field YBSC and Tycho-2 sets, used as the
            photometric reference (need not be a subset of ``stars``).
        match_radius_rad: Base position match radius in radians.
        duplicate_vmag: V-magnitude tolerance for judging two co-located
            records the same star once photometry is corrected.
        catalog_order: Catalog names in precedence order (most precise
            first); decides which record of a revealed duplicate survives.
        saturation_limit: V-magnitude brighter than which a candidate
            record is a saturation-correction candidate.
        min_correction_mag: Smallest candidate-versus-reference magnitude
            gap treated as saturation; a smaller gap is left untouched.

    Returns:
        The corrected star list with correction-revealed duplicates removed.
    """
    ref_ra, ref_dec, ref_vmag, ref_stars = reference_photometry(reference, match_radius_rad)
    consumed = np.zeros(len(ref_stars), dtype=bool)
    corrected: list[tuple[MutableStar, float]] = []
    for star in stars:
        if star.catalog_name.lower() in REFERENCE_CATALOGS:
            continue
        if star.vmag is None or float(star.vmag) >= saturation_limit:
            continue
        idx = _best_mag_aware_match(
            float(star.ra_pm),
            float(star.dec_pm),
            float(star.vmag),
            ref_ra,
            ref_dec,
            ref_vmag,
            match_radius_rad,
            available=~consumed,
        )
        if idx is None:
            # Bright per UCAC4 but with no reference in either catalog: the
            # aperture magnitude is an untrusted lower bound, so flag it.
            star.photometry_saturated = True
            continue
        consumed[idx] = True
        gap = abs(float(star.vmag) - float(ref_vmag[idx]))
        if gap < min_correction_mag:
            # Already agrees with the reference: not saturated, so leave it
            # be.  Consuming the reference keeps a second record from
            # claiming this same true-magnitude star.
            continue
        _apply_reference_photometry(star, ref_stars[idx])
        corrected.append((star, _widened_radius(gap, match_radius_rad)))
    if not corrected:
        return stars
    return _drop_revealed_duplicates(
        stars,
        corrected,
        duplicate_vmag=duplicate_vmag,
        catalog_order=catalog_order,
    )


def _drop_revealed_duplicates(
    stars: list[MutableStar],
    corrected: list[tuple[MutableStar, float]],
    *,
    duplicate_vmag: float,
    catalog_order: Sequence[str],
) -> list[MutableStar]:
    """Remove duplicates that a photometry correction exposed.

    For every corrected record, any other record within its
    (magnitude-widened) collapse radius and ``duplicate_vmag`` is the same
    physical star.  A twin that was not itself corrected wins (its native
    photometry and, for a bright star, its astrometry are trustworthy where
    UCAC4's saturated readings are not); between two corrected records the
    higher-precedence catalog wins.  The survivor inherits a name from the
    dropped record when it has none.  Only pairs involving a corrected
    record are considered, so records untouched by the correction keep the
    merge's original deduplication.

    Parameters:
        stars: The full star list (original order is preserved on return).
        corrected: ``(record, collapse_radius)`` pairs whose photometry was
            just corrected, each carrying its own widened collapse radius.
        duplicate_vmag: V-magnitude tolerance for a duplicate.
        catalog_order: Catalog names in precedence order (most precise
            first).

    Returns:
        ``stars`` with the dropped duplicates removed, order preserved.
    """
    priority = {name.lower(): i for i, name in enumerate(catalog_order)}
    fallback = len(catalog_order)

    def _rank(star: MutableStar) -> int:
        return priority.get(star.catalog_name.lower(), fallback)

    removed: set[int] = set()
    for star, collapse_radius in corrected:
        if id(star) in removed:
            continue
        r2 = collapse_radius * collapse_radius
        for other in stars:
            if other is star or id(other) in removed:
                continue
            if other.vmag is None or star.vmag is None:
                continue
            # Same seam-safe small-angle separation the reference match uses.
            d_dec = float(other.dec_pm) - float(star.dec_pm)
            d_ra = _wrap_to_pi(float(other.ra_pm) - float(star.ra_pm)) * math.cos(
                float(star.dec_pm)
            )
            if (
                d_ra * d_ra + d_dec * d_dec <= r2
                and abs(float(other.vmag) - float(star.vmag)) < duplicate_vmag
            ):
                if not other.photometry_corrected:
                    # ``other`` reads this bright star natively, so its
                    # astrometry is trustworthy where ``star``'s saturated
                    # UCAC4 position is not: keep ``other``.
                    winner, loser = other, star
                elif _rank(star) <= _rank(other):
                    winner, loser = star, other
                else:
                    winner, loser = other, star
                if (not winner.name) and loser.name:
                    winner.name = loser.name
                    winner.pretty_name = loser.pretty_name
                removed.add(id(loser))
                if loser is star:
                    break
    return [s for s in stars if id(s) not in removed]


def _apply_reference_photometry(star: MutableStar, ref: MutableStar) -> None:
    """Overwrite a star's photometry with a reference star's values.

    A YBSC reference supplies a self-consistent Johnson V/B pair, which is
    propagated directly.  A Tycho-2 reference (this pipeline discards its
    color) supplies only V; the color is faked from the corrected star's
    own spectral class, matching the catalog reduction's color-faking, and
    ``johnson_mag_faked`` is set so downstream code knows the color is
    synthetic.

    Parameters:
        star: Record to correct in place.
        ref: Reference star supplying the trusted photometry.
    """
    ref_vmag = cast(float, ref.vmag)
    star.vmag = ref_vmag
    if ref.catalog_name.lower() == 'ybsc' and ref.johnson_mag_v is not None:
        star.johnson_mag_v = ref.johnson_mag_v
        if ref.johnson_mag_b is not None:
            star.johnson_mag_b = ref.johnson_mag_b
            star.b_v = ref.johnson_mag_b - star.johnson_mag_v
        else:
            # No Johnson B from YBSC: treat the star as its own B (color 0)
            # so johnson_mag_b and b_v stay mutually consistent.
            star.johnson_mag_b = star.johnson_mag_v
            star.b_v = 0.0
        star.johnson_mag_faked = False
    else:
        # Tycho-2 reference (or a YBSC record lacking Johnson V): adopt V and
        # fake the color from the star's own spectral class.
        star.johnson_mag_v = ref_vmag
        star.johnson_mag_b = ref_vmag + SCLASS_TO_B_MINUS_V[clean_sclass(star.spectral_class)]
        star.b_v = star.johnson_mag_b - star.johnson_mag_v
        star.johnson_mag_faked = True
    # Standard flux-to-DN scaling, matching the catalog reduction's anchor.
    star.dn = float(2.512 ** -(ref_vmag - 4.0))
    star.photometry_corrected = True
    star.photometry_saturated = False


def correct_saturated_vmags(
    catalog_positions: Sequence[tuple[float, float, float]],
    ybsc_positions: Sequence[tuple[float, float, float]],
    tycho2_positions: Sequence[tuple[float, float, float]],
    *,
    match_radius_rad: float,
    saturation_limit: float = UCAC4_SATURATION_VMAG_LIMIT,
    min_correction_mag: float = SATURATION_CORRECTION_MIN_MAG,
) -> list[float]:
    """Return catalog V-magnitudes with UCAC4 bright-end saturation corrected.

    Each catalog star brighter than ``saturation_limit`` that positionally
    matches a reference bright star (YBSC or Tycho-2) and disagrees with it
    by more than ``min_correction_mag`` adopts the reference magnitude; the
    position match widens with the magnitude gap so a saturated star
    displaced beyond the base radius still matches.  A catalog star that
    matches a reference but already agrees with it is not saturated, so it
    keeps its own reading (the same no-op band as
    :func:`correct_star_photometry`, so the two paths share one policy).
    Each reference is consumed by at most one catalog star, so a reference
    is never counted twice.  Reference stars the catalog missed entirely
    are appended so the returned set reflects the field's true bright
    content.  Faint catalog stars and bright catalog stars with no
    reference keep their own magnitude.

    This is the magnitude-only counterpart of :func:`correct_star_photometry`
    for consumers (such as the navigable-content screen) that work with
    plain ``(ra, dec, vmag)`` triples rather than star records.  The
    reference is the union of YBSC and Tycho-2, with YBSC preferred where
    both cover a star.

    Parameters:
        catalog_positions: ``(ra, dec, vmag)`` triples (radians, radians,
            magnitude) for the catalog under correction (typically UCAC4).
        ybsc_positions: ``(ra, dec, vmag)`` triples for the YBSC bright
            stars in the same field.
        tycho2_positions: ``(ra, dec, vmag)`` triples for the Tycho-2 stars
            in the same field, complete well past YBSC into the V6.5 to V8
            range UCAC4 saturates.
        match_radius_rad: Base position match radius in radians.
        saturation_limit: V-magnitude brighter than which a catalog star is
            a saturation-correction candidate.
        min_correction_mag: Smallest candidate-versus-reference magnitude
            gap treated as saturation; a smaller gap keeps the catalog
            reading, matching :func:`correct_star_photometry`.

    Returns:
        Corrected V-magnitudes sorted ascending (brightest first).
    """
    ref_positions = _combined_reference_positions(
        ybsc_positions, tycho2_positions, match_radius_rad
    )
    ref_ra = np.asarray([p[0] for p in ref_positions], dtype=np.float64)
    ref_dec = np.asarray([p[1] for p in ref_positions], dtype=np.float64)
    ref_vmag = np.asarray([p[2] for p in ref_positions], dtype=np.float64)
    cat_ra = np.asarray([p[0] for p in catalog_positions], dtype=np.float64)
    cat_dec = np.asarray([p[1] for p in catalog_positions], dtype=np.float64)
    consumed = np.zeros(len(ref_positions), dtype=bool)
    out: list[float] = []
    for ra, dec, vmag in catalog_positions:
        if vmag >= saturation_limit:
            out.append(float(vmag))
            continue
        idx = _best_mag_aware_match(
            ra, dec, vmag, ref_ra, ref_dec, ref_vmag, match_radius_rad, available=~consumed
        )
        if idx is None:
            out.append(float(vmag))
            continue
        consumed[idx] = True
        if abs(float(vmag) - float(ref_vmag[idx])) < min_correction_mag:
            # Already agrees with the reference: not saturated, so keep the
            # catalog reading (the reference is still consumed above, so it
            # is not counted again).  This mirrors the no-op band in
            # correct_star_photometry so the screen and the reduction agree.
            out.append(float(vmag))
            continue
        out.append(float(ref_vmag[idx]))
    # Append reference stars with no catalog counterpart.  A reference a
    # saturated catalog star already matched (and consumed) is skipped so it
    # is never counted twice; a reference co-located with an at-limit or
    # faint catalog star is likewise skipped, since that catalog star is the
    # same physical object counted at its own (faint) reading.
    for i, (rra, rdec, rvmag) in enumerate(ref_positions):
        if consumed[i]:
            continue
        if _nearest_within(rra, rdec, cat_ra, cat_dec, match_radius_rad) is None:
            out.append(float(rvmag))
    return sorted(out)


def _combined_reference_positions(
    ybsc_positions: Sequence[tuple[float, float, float]],
    tycho2_positions: Sequence[tuple[float, float, float]],
    match_radius_rad: float,
) -> list[tuple[float, float, float]]:
    """Return the YBSC/Tycho-2 reference ``(ra, dec, vmag)`` triples.

    YBSC triples are kept unconditionally; a Tycho-2 triple is kept only
    where no YBSC triple lies within ``match_radius_rad``, so YBSC takes
    precedence wherever the two overlap and Tycho-2 supplies the stars
    beyond YBSC completeness.

    Parameters:
        ybsc_positions: YBSC ``(ra, dec, vmag)`` triples.
        tycho2_positions: Tycho-2 ``(ra, dec, vmag)`` triples.
        match_radius_rad: Radius (radians) within which a Tycho-2 triple is
            treated as the same star as a YBSC triple and dropped.

    Returns:
        The combined reference triples.
    """
    yb_ra = np.asarray([p[0] for p in ybsc_positions], dtype=np.float64)
    yb_dec = np.asarray([p[1] for p in ybsc_positions], dtype=np.float64)
    combined = list(ybsc_positions)
    for tra, tdec, tvmag in tycho2_positions:
        if _nearest_within(tra, tdec, yb_ra, yb_dec, match_radius_rad) is None:
            combined.append((tra, tdec, tvmag))
    return combined
