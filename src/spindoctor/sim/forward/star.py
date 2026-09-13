"""Image-side star rendering: catalog stars and the background sky field.

Stars are point sources, so they render into the detector-native point-source
plane (``SimFrame.point_e``) rather than the intensive signal plane: each star's
*total* flux is deposited as a sub-pixel-positioned point mass (a bilinear
splat), and the whole-scene optics PSF is the ONLY convolution it receives.
Pre-spreading a star and then convolving it would widen it by sqrt(2) and desync
its shape from the limb and ring-edge profiles.

The flux is normalized (not peak = f(vmag)): a star of magnitude ``vmag`` at
``exposure_sec`` deposits ``zero_point * 10**(-0.4 * vmag) * exposure_sec``, and
the PSF then dictates the peak.  The zero point is in electrons per second for a
CCD (the mass lands in the electron plane, added before Poisson) and in DN per
second for the Voyager vidicon (which has no electron domain).  The deposit
weight carries a factor ``oversample**2`` so the box-mean downsample conserves
the per-star detector-grid sum exactly.

The background sky field draws its star counts from a cumulative star-count law
``log10 N(<m) = a + b*m`` per square degree, scaled by the frame's field of
view, and renders every drawn star through the same flux/point-mass path (no
separate PSF sigma).

The rendered :class:`~spindoctor.support.types.MutableStar` records and the
``star_info`` hit-test entries are renderer *output* metadata (they carry the
planted offset, the rendered PSF sigma, and the deposited total flux); they stay
on the image side of the information boundary and are never handed to the
navigator-side models.
"""

import copy
import json
from functools import lru_cache
from typing import Any

import numpy as np

from spindoctor.sim.star_records import star_record_from_params
from spindoctor.support.types import STAR_UV_DATUM_PX, MutableStar, NDArrayFloatType

__all__ = [
    'faint_sky_cutoff_mag',
    'render_sky_counts',
    'render_stars',
    'total_flux_for_vmag',
]

# The brightest magnitude the sky-count inverse-CDF samples down from.  A star
# this bright is astronomically rare at the modelled counts (b > 0 weights the
# distribution toward faint stars), so the exact value only bounds the sampler.
_SKY_BRIGHT_CUTOFF_MAG = -2.0


def total_flux_for_vmag(vmag: float, *, zero_point: float, exposure_sec: float) -> float:
    """Return a star's deposited total flux for its magnitude and exposure.

    Parameters:
        vmag: The star's catalog magnitude.
        zero_point: Flux a magnitude-0 star deposits per second (electrons for a
            CCD, DN for the vidicon).
        exposure_sec: The scene exposure in seconds.

    Returns:
        The total flux to deposit as a point mass (before the ``os**2`` weight).
    """
    return float(float(zero_point) * 10.0 ** (-0.4 * float(vmag)) * float(exposure_sec))


def faint_sky_cutoff_mag(
    *, zero_point: float, exposure_sec: float, read_noise: float, psf_sigma: float
) -> float:
    """Return the faint magnitude below which a sky star is not worth rendering.

    A star falls below the sky background once its matched-filter signal drops to
    the read-noise floor integrated over its PSF core: with a Gaussian core of
    sigma ``psf_sigma`` the effective core area is ``2*pi*sigma**2`` pixels, so
    the noise over the core is ``read_noise * sqrt(2*pi*sigma**2)`` and the
    cutoff is where the total flux equals it.  Fainter draws add nothing above
    the noise, so the count integral is truncated there.

    Parameters:
        zero_point: Flux a magnitude-0 star deposits per second.
        exposure_sec: The scene exposure in seconds.
        read_noise: The camera's nominal per-pixel read noise (electrons for a
            CCD, DN for the vidicon), in the zero point's unit domain.
        psf_sigma: The scene PSF core sigma in detector pixels.

    Returns:
        The faint cutoff magnitude.
    """
    sigma = max(float(psf_sigma), 1e-3)
    noise = max(float(read_noise), 1.0) * float(np.sqrt(2.0 * np.pi * sigma**2))
    flux_ref = float(zero_point) * max(float(exposure_sec), 1e-9)
    if flux_ref <= 0.0 or noise <= 0.0:
        return 0.0
    return 2.5 * float(np.log10(flux_ref / noise))


def _deposit_point_mass(
    img: NDArrayFloatType,
    v: float,
    u: float,
    *,
    total: float,
    move_v: float = 0.0,
    move_u: float = 0.0,
) -> None:
    """Bilinearly deposit a point source's total flux at a sub-pixel position.

    The splat conserves the total and places the deposited centroid exactly at
    ``(v, u)`` (pixel-centre convention: integer index ``i`` is coordinate
    ``i``).  A non-zero motion vector distributes the mass evenly along the
    centred drift track (a per-star smear the whole-scene optics stage does not
    apply).

    Parameters:
        img: The (oversampled) point-source plane, modified in place.
        v: Centroid v coordinate on the grid of ``img``.
        u: Centroid u coordinate on the grid of ``img``.
        total: Total flux to deposit.
        move_v: Total drift along v over the exposure, in grid pixels.
        move_u: Total drift along u over the exposure, in grid pixels.
    """
    track = float(np.hypot(move_v, move_u))
    n_samples = 1 if track < 1e-9 else max(2, int(np.ceil(track)) + 1)
    offsets = [0.0] if n_samples == 1 else list(np.linspace(-0.5, 0.5, n_samples))
    weight = total / n_samples
    size_v, size_u = img.shape
    for t in offsets:
        pos_v = v + t * move_v
        pos_u = u + t * move_u
        v0 = int(np.floor(pos_v))
        u0 = int(np.floor(pos_u))
        fv = pos_v - v0
        fu = pos_u - u0
        for dv, du, frac in (
            (0, 0, (1.0 - fv) * (1.0 - fu)),
            (1, 0, fv * (1.0 - fu)),
            (0, 1, (1.0 - fv) * fu),
            (1, 1, fv * fu),
        ):
            vv = v0 + dv
            uu = u0 + du
            if 0 <= vv < size_v and 0 <= uu < size_u:
                img[vv, uu] += weight * frac


@lru_cache(maxsize=1)
def _render_stars_cached(
    size_v: int,
    size_u: int,
    stars_params_json: str,
    *,
    offset_v: float,
    offset_u: float,
    zero_point: float,
    exposure_sec: float,
    rendered_sigma: float,
    rotation_deg: float,
    oversample: int,
    catalog_scatter_px: float,
    catalog_scatter_seed: int,
) -> tuple[Any, ...]:
    """Internal cached function to compute star deposition."""
    stars_params = json.loads(stars_params_json)
    img = np.zeros((size_v, size_u), dtype=np.float64)
    sim_star_list: list[MutableStar] = []
    star_info: list[dict[str, Any]] = []

    # A camera roll rotates the whole frame about the boresight (image centre).
    # The rendered star position is the catalog position rotated by
    # ``rotation_deg`` about the centre, then translated by the planted offset.
    # The star record keeps its unrotated catalog (v, u) so the NavModel predicts
    # the unshifted geometry and a star technique recovers BOTH the rotation and
    # the translation.  The rotation matrix matches the navigator's
    # ``similarity_transform_fit`` convention (maps catalog -> detection in
    # ``(v, u)`` order), so the fitted angle equals ``rotation_deg``.
    theta = np.radians(rotation_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    roll_center_v = size_v / 2.0
    roll_center_u = size_u / 2.0
    # A point mass deposited at a detector coordinate scaled by ``os`` lands its
    # box-downsample centroid half a subsample low; the (os - 1) / 2 shift maps
    # it back so the deposited star centroids exactly at its catalog position.
    grid_shift = (oversample - 1) / 2.0
    # The hit-test half-window records the rendered star extent (a few sigma of
    # the scene PSF), in oversampled units; the downsample divides it back to
    # detector units.  The floor is one DETECTOR pixel (``oversample``
    # subsamples), matching the editor's one-pixel click-target floor: a
    # narrow-PSF star would otherwise record a sub-pixel window after the
    # downsample's divide and offer no whole-pixel hit target.
    psf_half = max(oversample, int(np.round(3.0 * rendered_sigma)))
    # A scene-level catalog-scatter sigma draws a per-star position error from a
    # seeded stream (byte-stable across processes), added to any explicit
    # per-star catalog_error.  The stream is consumed in star order so a scene's
    # scatter realization is reproducible; scenes with no scatter draw nothing.
    scatter_rng = np.random.default_rng(catalog_scatter_seed) if catalog_scatter_px > 0.0 else None

    for i, star_params in enumerate(stars_params):
        # The record is built by the builder shared with the navigator-side star
        # model, so both sides' catalog defaults match.  It reads only idealized
        # keys, so the record carries the catalog (v, u) and vmag -- never the
        # planted error that displaces the rendered star below.
        star = star_record_from_params(
            star_params, index=i, default_v=size_v / 2, default_u=size_u / 2
        )
        sim_star_list.append(star)

        # The record states the catalog position in oops uv; the deposit
        # below indexes an array, so the half-pixel datum comes back off
        # here.  The scene's pixel-space fields were scaled to this grid by
        # a pure multiply and the datum was added after that scaling, so
        # taking it off recovers the scaled index exactly, at any
        # oversampling.
        rel_v = star.v - STAR_UV_DATUM_PX - roll_center_v
        rel_u = star.u - STAR_UV_DATUM_PX - roll_center_u
        rot_v = cos_t * rel_v - sin_t * rel_u
        rot_u = sin_t * rel_v + cos_t * rel_u
        # The planted per-star catalog error (explicit plus the seeded scene
        # scatter draw) displaces the RENDERED star off the catalog position the
        # navigator predicts from; it is unrecoverable astrometric residual.
        err_v = float(star_params.get('catalog_error_v', 0.0))
        err_u = float(star_params.get('catalog_error_u', 0.0))
        if scatter_rng is not None:
            err_v += float(scatter_rng.normal(0.0, catalog_scatter_px))
            err_u += float(scatter_rng.normal(0.0, catalog_scatter_px))
        star_offset_v = roll_center_v + rot_v + offset_v + err_v
        star_offset_u = roll_center_u + rot_u + offset_u + err_u

        # A variable star renders at a brightness delta_mag off its catalog vmag
        # (positive = fainter); the catalog vmag stays what the navigator sees.
        vmag = star.vmag if star.vmag is not None else 8.0
        delta_mag = float(star_params.get('delta_mag', 0.0))
        total = total_flux_for_vmag(
            float(vmag) + delta_mag, zero_point=zero_point, exposure_sec=exposure_sec
        )
        _deposit_point_mass(
            img,
            star_offset_v + grid_shift,
            star_offset_u + grid_shift,
            total=total * oversample**2,
            move_v=star.move_v,
            move_u=star.move_u,
        )

        # An unresolved companion is a second point mass at sep_px / angle_deg,
        # companion.delta_mag fainter than the (already variable-adjusted)
        # primary.  Both convolve through the scene PSF, so the blended
        # photocenter sits off the catalog position by a magnitude-weighted
        # amount -- a physical catalog error, rendered here, never told to the
        # navigator.
        companion = star_params.get('companion')
        has_companion = isinstance(companion, dict) and float(companion.get('sep_px', 0.0)) > 0.0
        if has_companion:
            sep = float(companion.get('sep_px', 0.0))
            angle = np.radians(float(companion.get('angle_deg', 0.0)))
            comp_dmag = float(companion.get('delta_mag', 0.0))
            comp_total = total_flux_for_vmag(
                float(vmag) + delta_mag + comp_dmag,
                zero_point=zero_point,
                exposure_sec=exposure_sec,
            )
            _deposit_point_mass(
                img,
                star_offset_v + grid_shift + sep * float(np.cos(angle)),
                star_offset_u + grid_shift + sep * float(np.sin(angle)),
                total=comp_total * oversample**2,
                move_v=star.move_v,
                move_u=star.move_u,
            )

        # Truth metadata: the rendered-vs-catalog delta (position error and
        # magnitude offset), whether a companion was planted, and the realized
        # navigable flag.  The position keys are oversampled-grid pixels; the
        # downsample stage divides them back to detector pixels alongside
        # center_v/center_u.  These stay on the image side of the boundary.
        star_info.append(
            {
                'name': star.name,
                'center_v': star_offset_v,
                'center_u': star_offset_u,
                'sigma': rendered_sigma,
                'psf_half_v': psf_half,
                'psf_half_u': psf_half,
                'total_flux': total,
                'catalog_error_v': err_v,
                'catalog_error_u': err_u,
                'delta_mag': delta_mag,
                'has_companion': has_companion,
                'navigable': bool(star_params.get('navigable', True)),
            }
        )

    return (img, sim_star_list, star_info)


def render_stars(
    point_plane: NDArrayFloatType,
    stars_params: list[dict[str, Any]],
    offset_v: float,
    *,
    offset_u: float,
    zero_point: float,
    exposure_sec: float,
    rendered_sigma: float,
    rotation_deg: float = 0.0,
    oversample: int = 1,
    catalog_scatter_px: float = 0.0,
    catalog_scatter_seed: int = 0,
) -> tuple[NDArrayFloatType, list[MutableStar], list[dict[str, Any]]]:
    """Deposit catalog stars into the point-source plane. Returns (plane, list, info).

    Each star's total flux (``zero_point * 10**(-0.4 * vmag) * exposure_sec``) is
    deposited as a sub-pixel point mass carrying the ``oversample**2`` weight, so
    the box-mean downsample conserves the per-star sum and the whole-scene optics
    PSF is the star's only convolution.  A star may render off its catalog
    position (a per-star ``catalog_error_*`` plus the seeded scene scatter), at a
    variable brightness (``delta_mag``), and with an unresolved ``companion`` --
    all image-side truth the navigator is not told (see the star truth keys).

    Parameters:
        point_plane: The (oversampled) point-source plane, deposited into in
            place (electrons for a CCD, DN for the vidicon).
        stars_params: Per-star parameter dictionaries.
        offset_v: V offset (oversampled units) applied to every star.
        offset_u: U offset (oversampled units) applied to every star.
        zero_point: Flux a magnitude-0 star deposits per second.
        exposure_sec: The scene exposure in seconds.
        rendered_sigma: The scene PSF core sigma the stars render at (oversampled
            units), recorded in the hit-test metadata.
        rotation_deg: Camera-roll angle (degrees) applied about the image centre
            before the translation offset, modelling a pointing rotation the star
            techniques recover.
        oversample: The render-grid oversampling factor.
        catalog_scatter_px: Scene-level per-star position-scatter sigma
            (oversampled units); 0 disables it.
        catalog_scatter_seed: Seed for the per-star scatter stream.
    """
    size_v, size_u = point_plane.shape
    stars_params_json = json.dumps(stars_params, sort_keys=True)
    cached_img, cached_star_list, cached_star_info = _render_stars_cached(
        size_v,
        size_u,
        stars_params_json,
        offset_v=offset_v,
        offset_u=offset_u,
        zero_point=zero_point,
        exposure_sec=exposure_sec,
        rendered_sigma=rendered_sigma,
        rotation_deg=rotation_deg,
        oversample=oversample,
        catalog_scatter_px=catalog_scatter_px,
        catalog_scatter_seed=catalog_scatter_seed,
    )
    # Point deposits are added (never clipped) on the oversampled grid; the optics
    # PSF spreads them and the detector conversion clips after the downsample.
    point_plane += cached_img
    # The caller (and the downsample stage) rescale the hit-test entries and the
    # star records in place, so hand out fresh copies rather than the cached
    # objects.
    return (
        point_plane,
        [copy.copy(star) for star in cached_star_list],
        [dict(info) for info in cached_star_info],
    )


@lru_cache(maxsize=1)
def _render_sky_counts_cached(
    size_v: int,
    size_u: int,
    *,
    seed: int,
    a: float,
    b: float,
    density_factor: float,
    fov_deg2: float,
    faint_cutoff_mag: float,
    zero_point: float,
    exposure_sec: float,
    oversample: int,
) -> NDArrayFloatType:
    """Internal cached function to compute the background-sky star field."""
    plane = np.zeros((size_v, size_u), dtype=np.float64)
    # Cumulative count per square degree down to the faint cutoff, scaled to the
    # frame's field of view and the local-density multiplier.
    n_expected = 10.0 ** (a + b * faint_cutoff_mag) * density_factor * fov_deg2
    if n_expected <= 0.0:
        return plane
    rng = np.random.default_rng(seed)
    n_stars = int(rng.poisson(n_expected))
    if n_stars <= 0:
        return plane
    # Magnitudes from the implied differential distribution dN/dm ~ 10**(b*m)
    # between the bright bound and the faint cutoff: the inverse of the CDF that
    # is proportional to 10**(b*m).
    lo = 10.0 ** (b * _SKY_BRIGHT_CUTOFF_MAG)
    hi = 10.0 ** (b * faint_cutoff_mag)
    u_draw = rng.uniform(lo, hi, size=n_stars)
    mags = np.log10(u_draw) / b if b != 0.0 else np.full(n_stars, faint_cutoff_mag)
    vs = rng.uniform(0.0, float(size_v), size=n_stars)
    us = rng.uniform(0.0, float(size_u), size=n_stars)
    for mag, v, u in zip(mags.tolist(), vs.tolist(), us.tolist(), strict=True):
        total = total_flux_for_vmag(float(mag), zero_point=zero_point, exposure_sec=exposure_sec)
        _deposit_point_mass(plane, v, u, total=total * oversample**2)
    return plane


def render_sky_counts(
    point_plane: NDArrayFloatType,
    *,
    seed: int,
    a: float,
    b: float,
    density_factor: float,
    pixel_scale_arcsec: float,
    faint_cutoff_mag: float,
    zero_point: float,
    exposure_sec: float,
    diffuse_flux_per_px: float,
    oversample: int,
) -> None:
    """Render the background-sky star field (and optional diffuse floor) in place.

    The star count is drawn from ``log10 N(<m) = a + b*m`` per square degree,
    scaled by the frame's field of view (the detector pixel count times the
    angular pixel scale squared) and the ``density_factor`` multiplier, down to
    the faint cutoff.  Every star renders through the same flux/point-mass path
    as a catalog star, so the scene PSF shapes it.  A non-zero diffuse floor adds
    a flat pedestal (the box-mean downsample leaves a constant unchanged, so it
    is a per-detector-pixel level).

    Parameters:
        point_plane: The (oversampled) point-source plane, modified in place.
        seed: The stage sub-seed for the sky realization.
        a: Cumulative-count law intercept ``log10 N(<m)`` at ``m = 0``.
        b: Cumulative-count law slope per magnitude.
        density_factor: Multiplier on the counts (1 is mid galactic latitude).
        pixel_scale_arcsec: Angular pixel scale (arcsec / detector pixel).
        faint_cutoff_mag: The faint magnitude the count integral is truncated at.
        zero_point: Flux a magnitude-0 star deposits per second.
        exposure_sec: The scene exposure in seconds.
        diffuse_flux_per_px: A flat diffuse-sky pedestal per detector pixel (0
            disables it).
        oversample: The render-grid oversampling factor.
    """
    size_v, size_u = point_plane.shape
    det_v = size_v / oversample
    det_u = size_u / oversample
    deg_per_px = float(pixel_scale_arcsec) / 3600.0
    fov_deg2 = det_v * det_u * deg_per_px**2
    plane = _render_sky_counts_cached(
        size_v,
        size_u,
        seed=seed,
        a=float(a),
        b=float(b),
        density_factor=float(density_factor),
        fov_deg2=fov_deg2,
        faint_cutoff_mag=float(faint_cutoff_mag),
        zero_point=float(zero_point),
        exposure_sec=float(exposure_sec),
        oversample=int(oversample),
    )
    point_plane += plane
    if diffuse_flux_per_px > 0.0:
        # A flat pedestal survives the box-mean downsample as a constant, so the
        # commanded per-detector-pixel level is added directly on every subsample.
        point_plane += float(diffuse_flux_per_px)
