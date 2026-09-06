"""Star catalog reduction for the star NavModel.

The star pipeline pulls catalog records, then reduces them through:

1. **Stellar aberration.**  ``aberrate_star`` shifts a catalog RA/DEC into
   the spacecraft frame so the comparison against the image is fair.
2. **Proper motion.**  ``select_radec_list`` evaluates each star's
   proper-motion vector at ``obs.midtime`` so the predicted pixel
   reflects the star's position at the observation epoch.
3. **Multi-catalog precedence.**  ``reduce_catalogs`` walks the
   configured catalog order (``ucac4`` → ``tycho2`` → ``ybsc`` by
   default), pulls stars in incremental magnitude bins, and dedupes
   matching entries so the most precise catalog wins per star.
4. **FOV projection + edge culling.**  Stars too close to the extfov
   edge for their PSF support to fit, or whose smear motion would push
   them off the edge mid-exposure, are dropped.

The reduction is exposed as small free functions so ``NavModelStars``
composes them without subclassing.

Three module-level cached catalog instances avoid the expense of
re-loading UCAC4 / Tycho-2 / YBSC on every navigation; the getters
construct each catalog lazily on first call.
"""

from __future__ import annotations

import copy
import functools
import itertools
from typing import TYPE_CHECKING, cast

import numpy as np
import polymath
from oops import Event
from starcat import (
    SCLASS_TO_SURFACE_TEMP,
    SpiceStarCatalog,
    UCAC4StarCatalog,
    YBSCStarCatalog,
)

from spindoctor.nav_model.stars.predicted_snr import SCLASS_TO_B_MINUS_V
from spindoctor.nav_model.stars.saturation import (
    REFERENCE_CATALOGS as _PHOTOMETRY_REFERENCE_CATALOGS,
)
from spindoctor.nav_model.stars.saturation import (
    UCAC4_SATURATION_VMAG_LIMIT,
    correct_star_photometry,
)
from spindoctor.support.flux import clean_sclass

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.config import Config
    from spindoctor.obs import ObsSnapshot
    from spindoctor.support.types import MutableStar

__all__ = [
    'CATALOG_MAGNITUDE_BINS',
    'aberrate_star',
    'get_tycho2_catalog',
    'get_ucac4_catalog',
    'get_ybsc_catalog',
    'reduce_catalogs',
    'select_radec_list',
    'stars_in_extfov',
]


# Public so tests can pin against the same coarse magnitude grid the
# pipeline uses.  Stars are pulled bin by bin until the configured
# ``max_stars`` budget is hit; this avoids the worst case of pulling
# every dim star in a degree-square chunk of UCAC4.
CATALOG_MAGNITUDE_BINS: tuple[float, ...] = (
    0.0,
    8.0,
    9.0,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
)
"""Magnitude bin edges used to walk catalogs incrementally.

The bins are tighter near the threshold (10.0-13.0) where most catalog
stars in a typical Cassini-WAC FOV live, and looser at the bright and
faint ends.  The walk stops when the configured ``max_stars`` budget is
reached or the next bin's lower edge exceeds the obs's
``star_max_usable_vmag()``.
"""


@functools.lru_cache(maxsize=1)
def get_ucac4_catalog() -> UCAC4StarCatalog:
    """Return the UCAC4 catalog, lazily constructing on first call."""
    return UCAC4StarCatalog()


@functools.lru_cache(maxsize=1)
def get_tycho2_catalog() -> SpiceStarCatalog:
    """Return the Tycho-2 catalog, lazily constructing on first call."""
    return SpiceStarCatalog('tycho2')


@functools.lru_cache(maxsize=1)
def get_ybsc_catalog() -> YBSCStarCatalog:
    """Return the YBSC catalog, lazily constructing on first call."""
    return YBSCStarCatalog()


def aberrate_star(obs: ObsSnapshot, star: MutableStar) -> None:
    """Apply stellar aberration to a star in place.

    Builds an ``oops.Event`` at ``obs.midtime`` in the spacecraft frame,
    points its inverse-arrival vector at the catalog RA/DEC, and reads
    back the aberration-corrected RA/DEC from
    ``neg_arr_ap_j2000``.  ``star.ra`` and ``star.dec`` are overwritten.

    Parameters:
        obs: Observation snapshot supplying ``midtime``, ``path``, and
            ``frame``.
        star: Star record to mutate.  Both ``ra`` and ``dec`` are
            required; the catalog reduction drops records without them
            before it gets here.

    Raises:
        ValueError: If the record carries no RA or no DEC.
    """
    if star.ra is None or star.dec is None:
        raise ValueError(
            f'aberrate_star requires a star with RA and DEC, got ra={star.ra!r}, dec={star.dec!r}'
        )
    event = Event(
        obs.midtime,
        (polymath.Vector3.ZERO, polymath.Vector3.ZERO),
        obs.path,
        obs.frame,
    )
    event.neg_arr_j2000 = polymath.Vector3.from_ra_dec_length(
        star.ra, star.dec, 1.0, recursive=False
    )
    abb_ra, abb_dec, _ = event.neg_arr_ap_j2000.to_ra_dec_length(recursive=False)
    star.ra = abb_ra.vals
    star.dec = abb_dec.vals


def select_radec_list(
    stars: list[MutableStar],
    *,
    use_proper_motion: bool,
    midtime: float,
) -> list[tuple[float, float]]:
    """Return the per-star RA/DEC list, with optional proper-motion update.

    Parameters:
        stars: List of star records.
        use_proper_motion: When True, evaluate
            ``star.ra_dec_with_pm(midtime)`` for each entry; otherwise
            return the catalog ``(star.ra, star.dec)`` pair.
        midtime: Observation midtime in TDB seconds; ignored when
            ``use_proper_motion`` is False.

    Returns:
        Parallel ``[(ra, dec), ...]`` list in radians.
    """
    if use_proper_motion:
        return [cast(tuple[float, float], s.ra_dec_with_pm(midtime)) for s in stars]
    return [(cast(float, s.ra), cast(float, s.dec)) for s in stars]


def _find_stars_in_one_catalog(
    *,
    obs: ObsSnapshot,
    config: Config,
    catalog_name: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mag_min: float,
    mag_max: float,
    radec_movement: tuple[float, float] | None,
) -> list[MutableStar]:
    """Pull stars from a single catalog inside the (ra, dec, mag) box.

    Wraps the catalog-specific call signatures and patches up the per-star
    fields (``catalog_name``, ``pretty_name``, ``temperature``,
    ``johnson_mag_*``, ``conflicts`` flag, etc.) to match the reduced
    record shape used everywhere downstream.

    Parameters:
        obs: Observation snapshot (provides ``star_psf_size`` and
            extfov bounds).
        config: Project ``Config`` (provides
            ``stars.default_star_class`` and stellar-aberration toggle).
        catalog_name: One of ``'ucac4'``, ``'tycho2'``, ``'ybsc'``.
        ra_min, ra_max, dec_min, dec_max: Search box in radians.
        mag_min, mag_max: Catalog V-magnitude bounds.
        radec_movement: Optional ``(dra, ddec)`` vector representing half
            the per-exposure pointing motion in radians; passed through
            to the FOV projection so smear-affected stars near the edge
            are correctly culled.

    Returns:
        List of mutable star records with all derived fields set.
    """
    raw: list[MutableStar]
    if catalog_name == 'ucac4':
        raw = cast(
            list['MutableStar'],
            list(
                get_ucac4_catalog().find_stars(
                    allow_double=True,
                    allow_galaxy=False,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=dec_min,
                    dec_max=dec_max,
                    vmag_min=mag_min,
                    vmag_max=mag_max,
                )
            ),
        )
    elif catalog_name == 'tycho2':
        raw = cast(
            list['MutableStar'],
            list(
                get_tycho2_catalog().find_stars(
                    allow_double=True,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=dec_min,
                    dec_max=dec_max,
                    vmag_min=mag_min,
                    vmag_max=mag_max,
                )
            ),
        )
        for star in raw:
            star.johnson_mag_v = None
            star.johnson_mag_b = None
    elif catalog_name == 'ybsc':
        raw = cast(
            list['MutableStar'],
            list(
                get_ybsc_catalog().find_stars(
                    allow_double=True,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=dec_min,
                    dec_max=dec_max,
                    vmag_min=mag_min,
                    vmag_max=mag_max,
                )
            ),
        )
        for star in raw:
            if star.vmag is None:
                continue
            star.johnson_mag_v = star.vmag
            if star.b_v is None:
                star.johnson_mag_b = star.johnson_mag_v
            else:
                star.johnson_mag_b = star.vmag + star.b_v
    else:
        raise ValueError(f'Invalid star catalog: {catalog_name!r}')

    # Each star object is mutated heavily downstream; deep-copy now so a
    # second navigation against the same catalog gets fresh records.
    typed_raw: list[MutableStar] = [copy.deepcopy(s) for s in raw]
    stars_config = config.stars
    default_star_class = cast(str, stars_config.default_star_class)

    fixed: list[MutableStar] = []
    for star in typed_raw:
        if star.ra is None or star.dec is None or star.vmag is None:
            continue
        star.catalog_name = catalog_name
        star.pretty_name = str(star.unique_number)
        # The two catalogs disagree about this field and both are right: a YBSC
        # star carries a name that may be None, and a UCAC4 star has no name
        # attribute at all. getattr says that in one line. The AttributeError
        # this replaces said it too, but would have said the same thing about a
        # fault raised from inside a name property, which is not this.
        name = getattr(star, 'name', None)
        if name is None:
            star.name = ''
        elif name.strip():
            star.pretty_name = ' '.join(name.split())
        star.conflicts = ''
        star.temperature_faked = False
        star.johnson_mag_faked = False
        star.photometry_corrected = False
        star.photometry_saturated = False
        star.psf_size = obs.star_psf_size(star)
        if star.temperature is None:
            star.temperature_faked = True
            star.temperature = SCLASS_TO_SURFACE_TEMP[default_star_class]
            star.spectral_class = default_star_class
        if star.johnson_mag_v is None or star.johnson_mag_b is None:
            star.johnson_mag_v = star.vmag
            star.johnson_mag_b = star.vmag + SCLASS_TO_B_MINUS_V[clean_sclass(star.spectral_class)]
            star.johnson_mag_faked = True
        if stars_config.stellar_aberration:
            aberrate_star(obs, star)
        # Standard flux-to-DN scaling.  The 4.0 anchor is conventional; the
        # downstream SNR formula consumes ``star.dn`` as a relative DN value.
        star.dn = float(2.512 ** -(star.vmag - 4.0))
        fixed.append(star)

    return _project_stars_to_fov(
        obs=obs,
        config=config,
        stars=fixed,
        radec_movement=radec_movement,
    )


def _project_stars_to_fov(
    *,
    obs: ObsSnapshot,
    config: Config,
    stars: list[MutableStar],
    radec_movement: tuple[float, float] | None,
) -> list[MutableStar]:
    """Project stars into pixel coordinates and drop edge-clipped entries.

    Mutates each surviving star to populate ``u``, ``v``, ``move_u``,
    ``move_v``, ``ra_pm``, ``dec_pm``.  Stars whose PSF support would
    spill off the extfov are excluded; so are stars whose smear motion
    would push them off the extfov mid-exposure.

    Parameters:
        obs: Observation snapshot (provides FOV math + extfov bounds).
        config: Project ``Config`` (drives ``proper_motion`` toggle).
        stars: Stars after catalog-specific reduction.
        radec_movement: Optional half-exposure ``(dra, ddec)`` shift to
            evaluate edge-margin against.

    Returns:
        Filtered list with image-space fields populated.
    """
    if not stars:
        return []
    stars_config = config.stars

    ra_dec_pm_list = [cast(tuple[float, float], s.ra_dec_with_pm(obs.midtime)) for s in stars]
    if stars_config.proper_motion:
        ra_dec_list = ra_dec_pm_list
    else:
        ra_dec_list = [(cast(float, s.ra), cast(float, s.dec)) for s in stars]

    ra_arr = polymath.Scalar([entry[0] for entry in ra_dec_list])
    dec_arr = polymath.Scalar([entry[1] for entry in ra_dec_list])

    uv_mid = obs.uv_from_ra_and_dec(ra_arr, dec_arr, apparent=True)
    u_list, v_list = uv_mid.to_scalars()
    u_arr = u_list.vals
    v_arr = v_list.vals

    if radec_movement is not None:
        uv_start = obs.uv_from_ra_and_dec(
            ra_arr - radec_movement[0],
            dec_arr - radec_movement[1],
            apparent=True,
        )
        uv_end = obs.uv_from_ra_and_dec(
            ra_arr + radec_movement[0],
            dec_arr + radec_movement[1],
            apparent=True,
        )
    else:
        uv_start = obs.uv_from_ra_and_dec(ra_arr, dec_arr, tfrac=0.0, apparent=True)
        uv_end = obs.uv_from_ra_and_dec(ra_arr, dec_arr, tfrac=1.0, apparent=True)

    u_start_arr = uv_start.to_scalars()[0].vals
    v_start_arr = uv_start.to_scalars()[1].vals
    u_end_arr = uv_end.to_scalars()[0].vals
    v_end_arr = uv_end.to_scalars()[1].vals

    out: list[MutableStar] = []
    iter_args = zip(
        stars,
        ra_dec_pm_list,
        u_arr,
        v_arr,
        u_start_arr,
        v_start_arr,
        u_end_arr,
        v_end_arr,
        strict=True,
    )
    for star, (ra_pm, dec_pm), u, v, u_s, v_s, u_e, v_e in iter_args:
        psf_half_u = star.psf_size[1] // 2
        psf_half_v = star.psf_size[0] // 2
        if (
            u <= obs.extfov_u_min + psf_half_u
            or u >= obs.extfov_u_max - psf_half_u
            or v <= obs.extfov_v_min + psf_half_v
            or v >= obs.extfov_v_max - psf_half_v
            or u_s <= obs.extfov_u_min + psf_half_u
            or u_s >= obs.extfov_u_max - psf_half_u
            or v_s <= obs.extfov_v_min + psf_half_v
            or v_s >= obs.extfov_v_max - psf_half_v
            or u_e <= obs.extfov_u_min + psf_half_u
            or u_e >= obs.extfov_u_max - psf_half_u
            or v_e <= obs.extfov_v_min + psf_half_v
            or v_e >= obs.extfov_v_max - psf_half_v
        ):
            continue
        star.u = float(u)
        star.v = float(v)
        star.move_u = float(u_e - u_s)
        star.move_v = float(v_e - v_s)
        star.ra_pm = float(ra_pm)
        star.dec_pm = float(dec_pm)
        out.append(star)
    return out


def stars_in_extfov(
    obs: ObsSnapshot,
    config: Config,
    *,
    catalog_name: str,
    mag_min: float,
    mag_max: float,
    radec_movement: tuple[float, float] | None = None,
) -> list[MutableStar]:
    """Return all stars from one catalog that lie inside the extfov.

    Thin wrapper around ``_find_stars_in_one_catalog`` that pulls the
    extfov RA/DEC limits from ``obs`` so callers do not need to compute
    them.

    Parameters:
        obs: Observation snapshot.
        config: Project config.
        catalog_name: One of ``'ucac4'``, ``'tycho2'``, ``'ybsc'``.
        mag_min, mag_max: Catalog magnitude window.
        radec_movement: Optional half-exposure ``(dra, ddec)`` shift.

    Returns:
        Stars that fit inside the extfov with PSF support included.
    """
    ra_min, ra_max, dec_min, dec_max = obs.ra_dec_limits_ext()
    return _find_stars_in_one_catalog(
        obs=obs,
        config=config,
        catalog_name=catalog_name,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=dec_min,
        dec_max=dec_max,
        mag_min=mag_min,
        mag_max=mag_max,
        radec_movement=radec_movement,
    )


def reduce_catalogs(
    obs: ObsSnapshot,
    config: Config,
    *,
    radec_movement: tuple[float, float] | None = None,
) -> list[MutableStar]:
    """Walk every configured catalog, deduplicate, and return the merged list.

    Catalogs are walked in the order configured in
    ``config.stars.catalogs`` (default ``['ucac4', 'tycho2', 'ybsc']``).
    Within each catalog the search proceeds bin-by-bin against
    ``CATALOG_MAGNITUDE_BINS`` until the configured ``max_stars`` budget
    is hit.  Across catalogs, stars whose RA/DEC and magnitude match a
    star already kept from an earlier (more precise) catalog are
    dropped; the kept entry inherits any nicer ``name`` field the later
    catalog supplies.

    Parameters:
        obs: Observation snapshot supplying extfov RA/DEC limits.
        config: Project config carrying the catalog ordering and
            duplicate thresholds.
        radec_movement: Optional half-exposure ``(dra, ddec)`` shift.

    Returns:
        Merged star list with image-space fields populated, sorted by
        descending DN, capped at ``config.stars.max_stars``.
    """
    stars_config = config.stars
    max_stars = stars_config.max_stars
    mag_min_cap = obs.star_min_usable_vmag()
    mag_max_cap = obs.star_max_usable_vmag()

    duplicate_radec = float(np.radians(stars_config.duplicate_ra_dec_threshold_arcsec / 3600.0))
    duplicate_vmag = float(stars_config.duplicate_vmag_threshold)
    overlap_vmag = float(stars_config.overlapping_vmag_threshold)

    kept: list[MutableStar] = []
    photometry_reference: list[MutableStar] = []

    for catalog_name in stars_config.catalogs:
        next_per_catalog: list[MutableStar] = []
        for mag_min, mag_max in itertools.pairwise(CATALOG_MAGNITUDE_BINS):
            if mag_min > mag_max_cap:
                break
            if mag_max < mag_min_cap:
                continue
            mag_min_clamped = max(mag_min, mag_min_cap)
            mag_max_clamped = min(mag_max, mag_max_cap)
            chunk = stars_in_extfov(
                obs,
                config,
                catalog_name=catalog_name.lower(),
                mag_min=mag_min_clamped,
                mag_max=mag_max_clamped,
                radec_movement=radec_movement,
            )
            next_per_catalog.extend(chunk)
            if len(next_per_catalog) >= max_stars:
                break

        if catalog_name.lower() in _PHOTOMETRY_REFERENCE_CATALOGS:
            # Keep the full in-field YBSC and Tycho-2 sets as the photometric
            # reference even where the merge later drops individual entries as
            # duplicates; the bright-end correction reads from them below.
            photometry_reference += next_per_catalog

        kept = _merge_catalogs(
            kept,
            next_per_catalog,
            duplicate_radec=duplicate_radec,
            duplicate_vmag=duplicate_vmag,
        )

    # UCAC4 aperture photometry saturates at the bright end, so a merged
    # UCAC4 star can carry a magnitude several magnitudes too faint.  YBSC
    # and Tycho-2 are authoritative through the saturated regime (Tycho-2
    # reaches the V6.5 to V8 stars YBSC misses); correct the merged list
    # against them before overlaps and DN ordering are computed so both
    # depend on the true brightness.  Run the correction whenever a reference
    # catalog is configured, even when its in-field query returned nothing: a
    # bright UCAC4 record with no reference in either catalog must still be
    # flagged saturated rather than silently trusted.
    reference_configured = any(
        name.lower() in _PHOTOMETRY_REFERENCE_CATALOGS for name in stars_config.catalogs
    )
    if reference_configured:
        kept = correct_star_photometry(
            kept,
            photometry_reference,
            match_radius_rad=duplicate_radec,
            duplicate_vmag=duplicate_vmag,
            catalog_order=stars_config.catalogs,
            saturation_limit=UCAC4_SATURATION_VMAG_LIMIT,
        )

    _mark_visual_overlaps(kept, overlap_vmag=overlap_vmag)

    kept.sort(key=lambda s: s.dn, reverse=True)
    return kept[:max_stars]


def _merge_catalogs(
    earlier: list[MutableStar],
    later: list[MutableStar],
    *,
    duplicate_radec: float,
    duplicate_vmag: float,
) -> list[MutableStar]:
    """Merge ``later`` into ``earlier``, dropping duplicates.

    ``earlier`` wins on conflict; ``later`` only contributes a star
    whose proper-motion RA/DEC and V-magnitude are not within
    ``duplicate_radec`` (radians) and ``duplicate_vmag`` (mag) of any
    ``earlier`` entry.  When a duplicate is found and the earlier
    entry has no name but the later one does, the earlier entry's
    ``pretty_name`` is upgraded to the later catalog's name string.

    Parameters:
        earlier: Stars from a higher-precedence catalog (already
            reduced).  Modified in place to upgrade names.
        later: Stars from a lower-precedence catalog.
        duplicate_radec: Threshold in radians for matching RA and DEC.
        duplicate_vmag: Threshold in V-magnitude for matching brightness.

    Returns:
        Combined list (``earlier`` followed by the kept additions).
    """
    if not earlier:
        return list(later)
    earlier.sort(key=lambda s: s.dec_pm)
    later.sort(key=lambda s: s.dec_pm)
    additions: list[MutableStar] = []
    for star in later:
        is_dup = False
        for prev in earlier:
            if prev.dec_pm - star.dec_pm > duplicate_radec:
                # Sorted-prefix early-out.
                break
            if (
                abs(prev.dec_pm - star.dec_pm) < duplicate_radec
                and abs(prev.ra_pm - star.ra_pm) < duplicate_radec
                and abs(cast(float, prev.vmag) - cast(float, star.vmag)) < duplicate_vmag
            ):
                if (not prev.name) and star.name:
                    # The kept star has no catalog name but this duplicate does:
                    # adopt both ``name`` and ``pretty_name`` so the merged star
                    # carries the real identity (previously only ``pretty_name``
                    # was copied, leaving ``name`` empty and inconsistent with
                    # this very guard).
                    prev.name = star.name
                    prev.pretty_name = star.pretty_name
                is_dup = True
                break
        if not is_dup:
            additions.append(star)
    return earlier + additions


def _mark_visual_overlaps(stars: list[MutableStar], *, overlap_vmag: float) -> None:
    """Mark stars whose PSF supports overlap.

    Two-star overlap policy:

    * If both stars have similar magnitudes (``|delta_vmag| <
      overlap_vmag``), tag both with conflict ``'STAR'``: each one
      blinds the centroid fit for the other.
    * Otherwise, tag only the fainter one — the brighter star's PSF
      will dominate.

    Parameters:
        stars: All catalog stars after reduction.
        overlap_vmag: V-mag delta below which both stars are tagged.
    """
    n = len(stars)
    if n < 2:
        return
    # Sort by V (vertical pixel) so the inner loop can early-out once
    # consecutive entries are far enough apart in V to clear the overlap.
    stars.sort(key=lambda s: s.v, reverse=False)
    for i in range(n):
        for j in range(i + 1, n):
            si = stars[i]
            sj = stars[j]
            v_gap = (si.psf_size[0] + sj.psf_size[0]) / 2
            u_gap = (si.psf_size[1] + sj.psf_size[1]) / 2
            dv = sj.v - si.v
            if dv >= v_gap:
                # Stars are sorted by v ascending — once the v-gap is exceeded
                # for this pair, every later sj only moves further away.
                break
            if abs(dv) < v_gap and abs(si.u - sj.u) < u_gap:
                if cast(float, sj.vmag) - cast(float, si.vmag) < overlap_vmag:
                    si.conflicts = 'STAR'
                    sj.conflicts = 'STAR'
                else:
                    sj.conflicts = 'STAR'
