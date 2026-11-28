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
   matching entries so the most precise catalog wins per star.  A UCAC4
   star's own magnitude is UCAC's aperture magnitude, so each UCAC4 star
   is given a Johnson V magnitude and color first (see
   ``ucac4_johnson_photometry``).  Every star is then kept by its camera
   magnitude, its Johnson V corrected by the camera's color term (see
   ``star_camera_mag``), which is also what its predicted brightness
   (``dn``) follows.
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
    Star,
    UCAC4Star,
    UCAC4StarCatalog,
    YBSCStarCatalog,
)

from spindoctor.nav_model.stars.predicted_snr import SCLASS_TO_B_MINUS_V, star_camera_mag
from spindoctor.nav_model.stars.saturation import (
    REFERENCE_CATALOGS as _PHOTOMETRY_REFERENCE_CATALOGS,
)
from spindoctor.nav_model.stars.saturation import (
    UCAC4_SATURATION_VMAG_LIMIT,
    correct_star_photometry,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.flux import clean_sclass

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.config import Config
    from spindoctor.obs import ObsSnapshotInst
    from spindoctor.support.types import MutableStar

__all__ = [
    'CATALOG_MAGNITUDE_BINS',
    'aberrate_star',
    'catalog_query_margin',
    'get_tycho2_catalog',
    'get_ucac4_catalog',
    'get_ybsc_catalog',
    'reduce_catalogs',
    'select_radec_list',
    'stars_in_extfov',
    'ucac4_johnson_photometry',
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
faint ends.  They split the magnitude each catalog lists.  The walk stops
when the configured ``max_stars`` budget is reached or the next bin's
lower edge exceeds the obs's ``star_max_usable_vmag()`` plus the catalog's
query margin (see :func:`catalog_query_margin`).
"""


# The UCAC4 photometry constants below convert the catalog's own magnitude, the UCAC
# aperture magnitude (a single wide band, 579-642 nm), to Johnson V.  They were measured
# on the 47,087 UCAC4 stars to aperture magnitude 16 in the fields of the 98 Galileo and
# 341 LORRI star-field frames the star-limit measurement accepted (#1257), fitted on the
# 4,404 of them with APASS V 10-14, APASS V and B both measured, and 2MASS J and K both
# of grade A-D.  Each fit is a least-squares line with 3-sigma clipping.

UCAC4_APASS_MAX_SIGMA: float = 0.90
"""Largest APASS magnitude error, in magnitudes, that marks a measured APASS magnitude.

UCAC4 documents the valid range of an APASS error as -0.90 to +0.90 mag (a negative error
marks a single observation).  An error of exactly zero marks a Tycho-2 magnitude that
UCAC4 substituted for a missing APASS one, and an error outside the range is not a valid
error.  Neither is treated as an APASS measurement.
"""

# V - aperture = -0.0707 + 0.4893 (J - K), scatter 0.083 mag after clipping (4,102 stars
# kept); without a color the scatter is 0.156.  The residual stays within 0.025 mag of zero
# across J - K 0.0-1.5 and APASS V 10-14.
UCAC4_TWOMASS_V_ZERO_POINT: float = -0.0707
"""Johnson V minus the UCAC4 aperture magnitude of a star with 2MASS J - K of zero."""

UCAC4_TWOMASS_V_PER_J_MINUS_K: float = 0.4893
"""Change of Johnson V minus the UCAC4 aperture magnitude per magnitude of 2MASS J - K."""

# B - V = 0.0980 + 1.5294 (J - K), scatter 0.084 mag after clipping (4,162 stars kept).
UCAC4_TWOMASS_B_MINUS_V_ZERO_POINT: float = 0.0980
"""Johnson B - V of a star with 2MASS J - K of zero."""

UCAC4_TWOMASS_B_MINUS_V_PER_J_MINUS_K: float = 1.5294
"""Change of Johnson B - V per magnitude of 2MASS J - K."""

# The median of V - aperture over the same 4,404 stars is +0.161 mag; the robust spread
# about it is 0.156 mag, set by the color term the fallback cannot see.
UCAC4_UNCORRECTED_V_OFFSET: float = 0.161
"""Johnson V minus the UCAC4 aperture magnitude for a star with no usable color.

This is the uncorrected fallback, used only when a star has neither an APASS V magnitude
nor 2MASS J and K of grade A-D.  It removes the mean offset of the aperture magnitude
from Johnson V, but not the part that depends on the star's color.
"""

UCAC4_HALF_MILLIMAG: float = 0.0005
"""Half the step of UCAC4's magnitudes, which it lists in whole millimagnitudes."""

# Over the same fields, a star's APASS V is brighter than its aperture magnitude by more
# than 0.2 mag for 5% of the APASS stars fainter than aperture magnitude 10 (by more than
# 1 mag for 0.7%), and the 2MASS conversion brightens a star by at most 0.055 mag (99.9% of
# stars).  For the APASS stars more than 0.5 mag brighter, the 2MASS color puts V a median
# 0.28 mag fainter than the aperture magnitude, so most of that tail is APASS error.  At
# each frame's own limit and color term, the 439 frames' queries leave out 28 of their
# 69,636 stars in range; with no color term, 0.08% to 1% for limits from 11 to 14.5 (#1257).
UCAC4_QUERY_MARGIN_MAG: float = 0.2
"""Magnitudes by which a UCAC4 query reaches past the ends of the range it serves.

UCAC4 cuts a query on its aperture magnitude, and a star's Johnson V can be brighter or
fainter than that.  A UCAC4 query is widened by this margin, plus the color-term margin of
:func:`catalog_query_margin`, at both ends of the range, and the stars are then kept by
their camera magnitude.  A star is read only when its aperture magnitude lies inside the
widened bounds, so a star whose camera magnitude is in range is missed when its aperture
magnitude lies past them.  That can happen only near the faint limit, to a star whose
camera magnitude is brighter than its aperture magnitude by more than the whole margin,
which in practice means one with an APASS V.
"""

# Of the star records the star-limit measurement made on the star-field frames (#1257),
# 99.9% of those with a measured color have B - V from -0.18 to 3.15 in the LORRI fields,
# which lie near the galactic center where reddening is strong, and below 2.0 and 2.8 in
# the Galileo and Voyager fields.
STAR_QUERY_MAX_B_MINUS_V: float = 3.5
"""The largest ``|B - V|`` a catalog query allows for in a camera's color term.

A catalog cuts a query on the magnitude it lists, and a star's camera magnitude differs
from its Johnson V by the color term times its ``B - V``.  A query is widened by the
color term's size times this at both ends, and the stars are then kept by their camera
magnitude.
"""


def catalog_query_margin(catalog_name: str, color_term: float) -> float:
    """Return how far a catalog query reaches past the ends of the camera magnitudes it serves.

    Parameters:
        catalog_name: One of ``'ucac4'``, ``'tycho2'``, ``'ybsc'``.
        color_term: The camera's color term.

    Returns:
        :data:`STAR_QUERY_MAX_B_MINUS_V` times the size of the color term, plus
        :data:`UCAC4_QUERY_MARGIN_MAG` for UCAC4, which lists an aperture magnitude rather
        than Johnson V.
    """
    margin = abs(color_term) * STAR_QUERY_MAX_B_MINUS_V
    if catalog_name == 'ucac4':
        margin += UCAC4_QUERY_MARGIN_MAG
    return margin


def ucac4_johnson_photometry(star: UCAC4Star) -> tuple[float, float | None]:
    """Return a UCAC4 star's Johnson V magnitude and its B - V color.

    UCAC4 lists each star's UCAC aperture magnitude (``star.vmag``), measured in one wide
    band, which is not Johnson V.  Johnson V comes from the first of these sources that the
    star's record carries:

    1. APASS V, where APASS measured it: an error that is nonzero and within
       :data:`UCAC4_APASS_MAX_SIGMA`.
    2. The aperture magnitude converted with the 2MASS J - K color, where 2MASS J and K are
       both of grade A-D: ``V = aperture + UCAC4_TWOMASS_V_ZERO_POINT +
       UCAC4_TWOMASS_V_PER_J_MINUS_K * (J - K)``.
    3. The aperture magnitude plus :data:`UCAC4_UNCORRECTED_V_OFFSET`.

    The color is APASS B - V where APASS measured both V and B, else the 2MASS J - K color
    converted to B - V where J and K are both of grade A-D, else unknown.

    Parameters:
        star: A UCAC4 record that has an aperture magnitude.

    Returns:
        ``(v, b_minus_v)``: the Johnson V magnitude, and the B - V color or ``None`` when
        the record does not carry a color.

    Raises:
        ValueError: If the record has no aperture magnitude.
    """
    if star.vmag is None:
        raise ValueError(f'UCAC4 star {star.unique_number} has no aperture magnitude')
    apass_v = _measured_apass_mag(star.apass_mag_v, star.apass_mag_v_sigma)
    apass_b = _measured_apass_mag(star.apass_mag_b, star.apass_mag_b_sigma)
    j_minus_k = _twomass_j_minus_k(star)
    if apass_v is not None:
        v = apass_v
    elif j_minus_k is not None:
        v = star.vmag + UCAC4_TWOMASS_V_ZERO_POINT + UCAC4_TWOMASS_V_PER_J_MINUS_K * j_minus_k
    else:
        v = star.vmag + UCAC4_UNCORRECTED_V_OFFSET
    if apass_v is not None and apass_b is not None:
        return v, apass_b - apass_v
    if j_minus_k is not None:
        return v, (
            UCAC4_TWOMASS_B_MINUS_V_ZERO_POINT + UCAC4_TWOMASS_B_MINUS_V_PER_J_MINUS_K * j_minus_k
        )
    return v, None


def _measured_apass_mag(mag: float | None, sigma: float | None) -> float | None:
    """Return an APASS magnitude when APASS measured it, else ``None``.

    Parameters:
        mag: The magnitude UCAC4 lists in an APASS column.
        sigma: Its error.  Zero marks a substituted Tycho-2 magnitude.

    Returns:
        ``mag`` when its error is nonzero and within :data:`UCAC4_APASS_MAX_SIGMA`.
    """
    if mag is None or sigma is None or sigma == 0.0 or abs(sigma) > UCAC4_APASS_MAX_SIGMA:
        return None
    return mag


def _twomass_j_minus_k(star: UCAC4Star) -> float | None:
    """Return a UCAC4 star's 2MASS J - K color when both magnitudes are of grade A-D.

    Parameters:
        star: The UCAC4 record.

    Returns:
        J - K, or ``None`` when J or K is missing or of a grade below D.
    """
    if (
        star.twomass_mag_j is None
        or star.twomass_mag_k is None
        or star.twomass_ph_qual_j is None
        or star.twomass_ph_qual_k is None
        or star.twomass_ph_qual_j < UCAC4Star.UCAC4_2MASS_PH_QUAL_A
        or star.twomass_ph_qual_k < UCAC4Star.UCAC4_2MASS_PH_QUAL_A
    ):
        return None
    return star.twomass_mag_j - star.twomass_mag_k


def _set_ucac4_johnson_photometry(star: MutableStar, record: UCAC4Star) -> None:
    """Replace a UCAC4 star's aperture magnitude with its Johnson V and color.

    ``catalog_vmag`` keeps the aperture magnitude, ``vmag`` and ``johnson_mag_v`` take
    the Johnson V of :func:`ucac4_johnson_photometry`, and ``johnson_mag_b`` follows the
    color.  The spectral class and temperature are derived from that same color, so every
    color the star model reads has one source.  A star with no color is left with no
    Johnson B, spectral class, or temperature, which the catalog reduction then fills in
    from the default star class.

    Parameters:
        star: The record as the catalog reduction handles it, mutated in place.
        record: The same record as UCAC4 read it.
    """
    v, b_minus_v = ucac4_johnson_photometry(record)
    star.catalog_vmag = record.vmag
    star.vmag = v
    star.johnson_mag_v = v
    if b_minus_v is None:
        star.johnson_mag_b = None
        star.spectral_class = None
        star.temperature = None
        return
    star.johnson_mag_b = v + b_minus_v
    star.spectral_class = Star.sclass_from_bv(star.johnson_mag_b, v)
    star.temperature = Star.temperature_from_sclass(star.spectral_class)


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


def aberrate_star(obs: ObsSnapshotInst, star: MutableStar) -> None:
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
    # The event's arrival direction was set just above, so its apparent form exists.
    aberrated = cast(polymath.Vector3, event.neg_arr_ap_j2000)
    abb_ra, abb_dec, _ = aberrated.to_ra_dec_length(recursive=False)
    star.ra = float(abb_ra.vals)
    star.dec = float(abb_dec.vals)


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
    obs: ObsSnapshotInst,
    config: Config,
    catalog_name: str,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mag_min: float,
    mag_max: float,
    catalog_mag_min: float,
    catalog_mag_max: float,
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
        mag_min, mag_max: Camera magnitude bounds, inclusive (see
            :func:`~spindoctor.nav_model.stars.predicted_snr.star_camera_mag`).  A UCAC4
            star's camera magnitude starts from the Johnson V of
            :func:`ucac4_johnson_photometry`, not from its aperture magnitude.  A UCAC4
            star whose aperture magnitude is brighter than
            :data:`~spindoctor.nav_model.stars.saturation.UCAC4_SATURATION_VMAG_LIMIT` is
            kept whatever its camera magnitude, for the bright-end correction.
        catalog_mag_min, catalog_mag_max: Bounds, inclusive, on the magnitude the
            catalog lists, which the catalog query cuts on: Johnson V for Tycho-2 and
            YBSC, the aperture magnitude for UCAC4.
        radec_movement: Optional ``(dra, ddec)`` vector representing half
            the per-exposure pointing motion in radians; passed through
            to the FOV projection so smear-affected stars near the edge
            are correctly culled.

    Returns:
        List of mutable star records with all derived fields set.
    """
    raw: list[MutableStar]
    if catalog_name == 'ucac4':
        raw = []
        for record in get_ucac4_catalog().find_stars(
            allow_double=True,
            allow_galaxy=False,
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max,
            vmag_min=catalog_mag_min,
            vmag_max=catalog_mag_max,
        ):
            if record.vmag is None:
                continue
            star = cast('MutableStar', record)
            _set_ucac4_johnson_photometry(star, cast(UCAC4Star, record))
            raw.append(star)
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
                    vmag_min=catalog_mag_min,
                    vmag_max=catalog_mag_max,
                )
            ),
        )
        for star in raw:
            star.catalog_vmag = star.vmag
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
                    vmag_min=catalog_mag_min,
                    vmag_max=catalog_mag_max,
                )
            ),
        )
        for star in raw:
            star.catalog_vmag = star.vmag
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
    default_star_class = stars_config.default_star_class
    color_term = obs.star_color_term()

    fixed: list[MutableStar] = []
    for star in typed_raw:
        if star.ra is None or star.dec is None or star.vmag is None:
            continue
        star.temperature_faked = False
        star.johnson_mag_faked = False
        if star.temperature is None:
            star.temperature_faked = True
            star.temperature = SCLASS_TO_SURFACE_TEMP[default_star_class]
            star.spectral_class = default_star_class
        if star.johnson_mag_v is None or star.johnson_mag_b is None:
            star.johnson_mag_v = star.vmag
            star.johnson_mag_b = star.vmag + SCLASS_TO_B_MINUS_V[clean_sclass(star.spectral_class)]
            star.johnson_mag_faked = True
        # The star is kept by its magnitude in the camera's band, which needs its color.
        # A UCAC4 star brighter than the saturation limit has a magnitude that may read
        # too faint until the bright-end correction replaces it, so it is kept whatever
        # its camera magnitude and ``reduce_catalogs`` judges it after the correction.
        camera_mag = star_camera_mag(star, color_term)
        saturation_candidate = (
            catalog_name == 'ucac4' and cast(float, star.catalog_vmag) < UCAC4_SATURATION_VMAG_LIMIT
        )
        if not (mag_min <= camera_mag <= mag_max or saturation_candidate):
            continue
        star.catalog_name = catalog_name
        star.pretty_name = str(star.unique_number)
        # A YBSC star has a name that may be None; a UCAC4 star has no name
        # attribute.  Both hold it as a plain field, never as a property, so a
        # getattr with a default answers both and has no fault to hide.
        name = getattr(star, 'name', None)
        if name is None:
            star.name = ''
        elif name.strip():
            star.pretty_name = ' '.join(name.split())
        star.conflicts = ''
        star.photometry_corrected = False
        star.photometry_saturated = False
        # The catalog's records are starcat stars, whatever protocol they are handled by.
        star.psf_size = obs.star_psf_size(cast(Star, star))
        if stars_config.stellar_aberration:
            aberrate_star(obs, star)
        # Standard flux-to-DN scaling of the camera magnitude.  The 4.0 anchor is
        # conventional; the downstream SNR formula consumes ``star.dn`` as a relative
        # DN value.
        star.dn = float(2.512 ** -(camera_mag - 4.0))
        fixed.append(star)

    return _project_stars_to_fov(
        obs=obs,
        config=config,
        stars=fixed,
        radec_movement=radec_movement,
    )


def _project_stars_to_fov(
    *,
    obs: ObsSnapshotInst,
    config: Config,
    stars: list[MutableStar],
    radec_movement: tuple[float, float] | None,
) -> list[MutableStar]:
    """Project stars into pixel corner coordinates and drop edge-clipped entries.

    Mutates each surviving star to populate ``u``, ``v``, ``move_u``,
    ``move_v``, ``ra_pm``, ``dec_pm``.  ``u`` and ``v`` are whatever
    ``ObsSnapshotInst.uv_from_ra_and_dec`` returns, which is a pixel corner coordinate -- the
    convention :class:`~spindoctor.support.types.MutableStar` declares, half
    a pixel above the index of the pixel the star lands on.  Stars whose PSF
    support would spill off the extfov are excluded; so are stars whose
    smear motion would push them off the extfov mid-exposure.  The extfov
    bounds are whole-numbered, so they are converted before that
    comparison.

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
    u_arr = np.asarray(u_list.vals, dtype=np.float64)
    v_arr = np.asarray(v_list.vals, dtype=np.float64)

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

    u_start_arr = np.asarray(uv_start.to_scalars()[0].vals, dtype=np.float64)
    v_start_arr = np.asarray(uv_start.to_scalars()[1].vals, dtype=np.float64)
    u_end_arr = np.asarray(uv_end.to_scalars()[0].vals, dtype=np.float64)
    v_end_arr = np.asarray(uv_end.to_scalars()[1].vals, dtype=np.float64)

    # ``extfov_*_min`` / ``_max`` name the first and last pixel of the
    # padded frame as whole numbers; every position tested
    # below is pixel corner coordinates.  Convert the four bounds once here rather than the
    # six positions of every star, so the gate compares uv against uv and no
    # conversion is spelled out inside the loop.
    u_min_uv = obs.extfov_u_min + PIXEL_CENTER_TO_CORNER_PX
    u_max_uv = obs.extfov_u_max + PIXEL_CENTER_TO_CORNER_PX
    v_min_uv = obs.extfov_v_min + PIXEL_CENTER_TO_CORNER_PX
    v_max_uv = obs.extfov_v_max + PIXEL_CENTER_TO_CORNER_PX

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
            u <= u_min_uv + psf_half_u
            or u >= u_max_uv - psf_half_u
            or v <= v_min_uv + psf_half_v
            or v >= v_max_uv - psf_half_v
            or u_s <= u_min_uv + psf_half_u
            or u_s >= u_max_uv - psf_half_u
            or v_s <= v_min_uv + psf_half_v
            or v_s >= v_max_uv - psf_half_v
            or u_e <= u_min_uv + psf_half_u
            or u_e >= u_max_uv - psf_half_u
            or v_e <= v_min_uv + psf_half_v
            or v_e >= v_max_uv - psf_half_v
        ):
            continue
        # Straight from ``uv_from_ra_and_dec``: the record's declared
        # values are pixel-corner coordinates, so nothing is converted here.
        star.u = float(u)
        star.v = float(v)
        star.move_u = float(u_e - u_s)
        star.move_v = float(v_e - v_s)
        star.ra_pm = float(ra_pm)
        star.dec_pm = float(dec_pm)
        out.append(star)
    return out


def stars_in_extfov(
    obs: ObsSnapshotInst,
    config: Config,
    *,
    catalog_name: str,
    mag_min: float,
    mag_max: float,
    catalog_window: tuple[float, float] | None = None,
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
        mag_min, mag_max: Camera magnitude window, inclusive (see
            :func:`~spindoctor.nav_model.stars.predicted_snr.star_camera_mag`; for
            UCAC4 it starts from the Johnson V of :func:`ucac4_johnson_photometry`).
        catalog_window: The ``(min, max)`` magnitudes, inclusive, that the catalog lists
            to query (the aperture magnitude for UCAC4, Johnson V for Tycho-2 and
            YBSC); the stars it returns are then kept by their camera magnitude, but for
            the bright UCAC4 stars the bright-end correction may change (see
            :func:`_find_stars_in_one_catalog`).  By
            default it is the camera magnitude window widened at both ends by
            :func:`catalog_query_margin`.
        radec_movement: Optional half-exposure ``(dra, ddec)`` shift.

    Returns:
        Stars that fit inside the extfov with PSF support included.
    """
    ra_min, ra_max, dec_min, dec_max = obs.ra_dec_limits_ext()
    if catalog_window is None:
        margin = catalog_query_margin(catalog_name, obs.star_color_term())
        catalog_window = (mag_min - margin, mag_max + margin)
    catalog_mag_min, catalog_mag_max = catalog_window
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
        catalog_mag_min=catalog_mag_min,
        catalog_mag_max=catalog_mag_max,
        radec_movement=radec_movement,
    )


def reduce_catalogs(
    obs: ObsSnapshotInst,
    config: Config,
    *,
    radec_movement: tuple[float, float] | None = None,
) -> list[MutableStar]:
    """Walk every configured catalog, deduplicate, and return the merged list.

    Catalogs are walked in the order configured in
    ``config.stars.catalogs``.
    Within each catalog the search proceeds bin-by-bin against
    ``CATALOG_MAGNITUDE_BINS`` until the configured ``max_stars`` budget
    is hit.  The bins split the magnitude the catalog lists (UCAC4's
    aperture magnitude, Johnson V for the others), and every star a bin
    pulls is kept whose camera magnitude (see
    :func:`~spindoctor.nav_model.stars.predicted_snr.star_camera_mag`) lies
    within the observation's usable range (a UCAC4 star brighter than the
    saturation limit is judged after the bright-end correction instead); the
    first and last bins reach :func:`catalog_query_margin` past that range.  Across
    catalogs, stars whose RA/DEC and magnitude match a star already kept
    from an earlier (more precise) catalog are dropped; the kept entry
    inherits any nicer ``name`` field the later catalog supplies.

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
    color_term = obs.star_color_term()

    duplicate_radec = float(np.radians(stars_config.duplicate_ra_dec_threshold_arcsec / 3600.0))
    duplicate_vmag = stars_config.duplicate_vmag_threshold
    overlap_vmag = stars_config.overlapping_vmag_threshold

    kept: list[MutableStar] = []
    photometry_reference: list[MutableStar] = []

    for catalog_name in stars_config.catalogs:
        next_per_catalog: list[MutableStar] = []
        margin = catalog_query_margin(catalog_name, color_term)
        first_bin = True
        for mag_min, mag_max in itertools.pairwise(CATALOG_MAGNITUDE_BINS):
            if mag_min > mag_max_cap:
                break
            if mag_max < mag_min_cap:
                continue
            last_bin = mag_max >= mag_max_cap or mag_max == CATALOG_MAGNITUDE_BINS[-1]
            # A star's camera magnitude can sit in a different bin from the
            # magnitude its catalog lists, so the bins split the catalog's
            # magnitude and every star is kept by its camera magnitude over the
            # whole usable range.  Only the outermost bins reach past that range.
            # UCAC4 lists whole millimagnitudes and its query is inclusive, so an
            # interior bin stops half a millimagnitude short of its upper edge and
            # a star on the edge belongs to the next bin alone.
            low = mag_min_cap - margin if first_bin else mag_min
            high = mag_max_cap + margin if last_bin else mag_max
            if catalog_name == 'ucac4' and not last_bin:
                high -= UCAC4_HALF_MILLIMAG
            chunk = stars_in_extfov(
                obs,
                config,
                catalog_name=catalog_name,
                mag_min=mag_min_cap,
                mag_max=mag_max_cap,
                catalog_window=(low, high),
                radec_movement=radec_movement,
            )
            next_per_catalog.extend(chunk)
            first_bin = False
            # A bright UCAC4 star kept past the usable range for the correction does not
            # count against the budget, since the correction may yet drop it.
            n_usable = sum(
                1
                for star in next_per_catalog
                if mag_min_cap <= star_camera_mag(star, color_term) <= mag_max_cap
            )
            if n_usable >= max_stars or last_bin:
                break

        if catalog_name in _PHOTOMETRY_REFERENCE_CATALOGS:
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

    # UCAC4 aperture photometry saturates at the bright end, and so does the
    # APASS V that replaces it, so a merged UCAC4 star can carry a magnitude
    # several magnitudes too faint.  YBSC and Tycho-2 are authoritative through
    # the saturated regime (Tycho-2 reaches the V6.5 to V10 stars YBSC misses);
    # correct the merged list against them before overlaps and DN ordering are
    # computed so both depend on the true brightness.  Run the correction
    # whenever a reference catalog is configured, even when its in-field query
    # returned nothing: a bright UCAC4 record with no reference in either catalog
    # must still be flagged saturated rather than silently trusted.
    reference_configured = any(
        name in _PHOTOMETRY_REFERENCE_CATALOGS for name in stars_config.catalogs
    )
    if reference_configured:
        kept = correct_star_photometry(
            kept,
            photometry_reference,
            match_radius_rad=duplicate_radec,
            duplicate_vmag=duplicate_vmag,
            catalog_order=stars_config.catalogs,
            color_term=color_term,
            saturation_limit=UCAC4_SATURATION_VMAG_LIMIT,
        )
    # The bright UCAC4 stars kept past the usable range for the correction are judged
    # now, on their corrected camera magnitudes.  A star flagged saturated stays: its
    # magnitude is a lower bound on its brightness, and the star model exempts it.
    kept = [
        star
        for star in kept
        if star.photometry_saturated
        or mag_min_cap <= star_camera_mag(star, color_term) <= mag_max_cap
    ]

    _mark_visual_overlaps(kept, overlap_vmag=overlap_vmag, color_term=color_term)

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


def _mark_visual_overlaps(
    stars: list[MutableStar], *, overlap_vmag: float, color_term: float
) -> None:
    """Mark stars whose PSF supports overlap.

    Two-star overlap policy, on the stars' camera magnitudes (see
    :func:`~spindoctor.nav_model.stars.predicted_snr.star_camera_mag`):

    * If both stars have similar magnitudes (``|delta_mag| <
      overlap_vmag``), tag both with conflict ``'STAR'``: each one
      blinds the centroid fit for the other.
    * Otherwise, tag only the fainter one — the brighter star's PSF
      will dominate.

    Parameters:
        stars: All catalog stars after reduction.
        overlap_vmag: Magnitude delta below which both stars are tagged.
        color_term: The camera's color term.
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
                if star_camera_mag(sj, color_term) - star_camera_mag(si, color_term) < overlap_vmag:
                    si.conflicts = 'STAR'
                    sj.conflicts = 'STAR'
                else:
                    sj.conflicts = 'STAR'
