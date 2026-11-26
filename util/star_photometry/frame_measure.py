"""Measure one star-field frame: an independent pointing fix, then every catalog star at it.

This module is instrument-neutral.  A per-instrument program in this package supplies the
observation (read through the navigator's own reader, so star positions are the navigator's),
the image the photometry is measured on and the mapping into it, the frame's photometric
state (filter, gain state, readout mode, exposure), and the measurement parameters.  This
module queries the catalog deep, finds the pointing most catalog stars agree on without the
navigator's star techniques, and measures every catalog star at the pointing.

For each star the record holds what the limit and PSF fits need:

* detection statistics on the navigator's image, high-passed (peak SNR, pixels above three
  sigma);
* background-subtracted aperture DN on the photometric image in two apertures, the smaller for
  photometry and the larger for the aperture correction;
* the exclusion flags (saturated, blended, behind a body or ring, near an edge, smeared);
* for stars bright enough, a pixel-integrated Gaussian fit and encircled-energy radii.

:func:`run_frames` runs a frame list in a process pool, appending one JSON line per frame, and
skips frames an earlier run already wrote, so an interrupted run resumes.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import tempfile
import traceback
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
from oops import Meshgrid
from oops.backplane import Backplane
from util.star_photometry.star_field import (
    aperture_photometry,
    detect_peaks,
    detection_test,
    encircled_energy_radii,
    fit_gaussian_psf,
    highpass,
    robust_sigma,
)
from util.star_photometry.star_pointing import apply_fix, find_pointing

from spindoctor.config.config import Config
from spindoctor.nav_model.stars.catalog import reduce_catalogs
from spindoctor.obs import ObsSnapshotInst

__all__ = [
    'FrameInput',
    'MeasureParams',
    'deep_config',
    'measure_frame',
    'run_frames',
]

FloatArray = NDArray[np.float64]

#: Offset between the navigator's pixel-corner coordinates and array-index coordinates.
_CORNER_TO_INDEX = 0.5

#: Detection criterion: peak SNR at least this ...
DETECT_MIN_PEAK_SNR = 7.0

#: ... with at least this many pixels of the counting box above three sigma.
DETECT_MIN_PIXELS_ABOVE_3_SIGMA = 3


@dataclass(frozen=True)
class FrameInput:
    """One frame to measure and its photometric state, from the PDS3 index.

    Attributes:
        image_id: The image's name, unique within the instrument.
        path: The image the navigator reads.
        camera: The camera key (for example ``'NAC'``).
        filter_name: The filter as the index spells it.
        gain_state: The gain state as the index spells it.
        mode: The readout, summation, or binning mode.
        texp: Exposure in seconds.
        extra: Further index fields worth keeping with the record.
    """

    image_id: str
    path: str
    camera: str
    filter_name: str
    gain_state: str
    mode: str
    texp: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasureParams:
    """Measurement parameters for one camera and mode.

    Attributes:
        catalog_depth_mag: The catalog is queried to the navigator's current limit plus this.
        highpass_size: Median window of the high-pass filter (navigator-image pixels).
        smooth_sigma: Peak-finding smoothing sigma, about the star sigma (pixels).
        peak_threshold_snr: Peak-finding threshold in smoothed-image noise units.
        edge_px: Peaks and stars closer than this to a border are ignored.
        max_peaks: Peaks searched by the pointing vote.
        max_vote_stars: The brightest this-many unoccluded catalog stars vote.
        search_px: Largest pointing offset searched on each axis.
        vote_bin_px: Pointing vote bin size.
        agree_px: Agreement radius of the pointing fix.
        min_agree: Fewest agreeing stars an accepted fix needs.
        max_chance_probability: Largest chance probability an accepted fix may have.
        detect_search_px: Peak search radius of the detection test.
        noise_inner: Inner radius of the detection test's noise annulus.
        noise_outer: Outer radius of the detection test's noise annulus.
        box_half: Half-size of the detection test's counting box.
        aperture_px: Photometry aperture radius (photometric-image pixels).
        big_aperture_px: Aperture-correction aperture radius.
        sky_inner: Sky annulus inner radius.
        sky_outer: Sky annulus outer radius.
        blend_px: A star with a catalog neighbor within this distance (navigator pixels) ...
        blend_dmag: ... no more than this many magnitudes fainter than it is blended.
        saturation_dn: A photometric-image pixel at or above this is saturated.
        max_smear_px: Stars smeared further than this during the exposure are flagged.
        psf_min_snr: Stars at least this peak SNR, unsaturated and isolated, get a PSF fit.
        psf_half: Half-size of the PSF-fit cutout.
        ee_r_max: Outer radius of the encircled-energy profile.
        ee_n_bins: Radial bins of the encircled-energy profile.
        highpass_subsample: Pixel step of the high-pass filter's median (see
            :func:`util.star_photometry.star_field.highpass`).
        missing_value: A photometric-image pixel at or below this value is missing data (a
            lost line or a part of the frame not returned); a star with one in its large
            aperture or sky annulus is flagged ``no_data``.  None when every value is data.
    """

    catalog_depth_mag: float
    highpass_size: int
    smooth_sigma: float
    peak_threshold_snr: float
    edge_px: int
    max_peaks: int
    max_vote_stars: int
    search_px: float
    vote_bin_px: float
    agree_px: float
    min_agree: int
    max_chance_probability: float
    detect_search_px: float
    noise_inner: float
    noise_outer: float
    box_half: int
    aperture_px: float
    big_aperture_px: float
    sky_inner: float
    sky_outer: float
    blend_px: float
    blend_dmag: float
    saturation_dn: float
    max_smear_px: float
    psf_min_snr: float
    psf_half: int
    ee_r_max: float
    ee_n_bins: int
    highpass_subsample: int = 1
    missing_value: float | None = None


#: Maps navigator-image array-index positions to photometric-image positions.
PhotMap = Callable[[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


def deep_config(max_stars: int) -> Config:
    """Return the default configuration with the catalog's star budget raised.

    Parameters:
        max_stars: The catalog reduction's star budget.

    Returns:
        A loaded configuration whose ``stars.max_stars`` is ``max_stars``.
    """
    config = Config()
    config.read_config()
    with tempfile.TemporaryDirectory() as tmp:
        override = Path(tmp) / 'deep.yaml'
        override.write_text(f'stars:\n  max_stars: {max_stars}\n')
        config.update_config(override)
    return config


def _identity_map(v: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return positions unchanged.

    Parameters:
        v: Lines.
        u: Samples.

    Returns:
        ``(v, u)``.
    """
    return v, u


def measure_frame(
    obs: ObsSnapshotInst,
    frame: FrameInput,
    params: MeasureParams,
    *,
    config: Config,
    phot_image: FloatArray | None = None,
    phot_map: PhotMap | None = None,
) -> dict[str, Any]:
    """Measure one frame.

    Parameters:
        obs: The observation, read through the navigator's reader with ``config``.
        frame: The frame and its photometric state.
        params: The measurement parameters.
        config: The configuration (with a raised star budget, see :func:`deep_config`).
        phot_image: The image photometry is measured on; the navigator's image when None.
        phot_map: Maps navigator-image positions into ``phot_image``; identity when None.

    Returns:
        A JSON-serializable record: the frame, the navigator's current limit, the pointing
        fix, and one entry per catalog star measured (only when the fix was accepted).
    """
    nav = np.asarray(obs.data, dtype=np.float64)
    phot = nav if phot_image is None else np.asarray(phot_image, dtype=np.float64)
    to_phot = _identity_map if phot_map is None else phot_map
    current_limit = float(obs.star_max_usable_vmag())
    deep_limit = current_limit + params.catalog_depth_mag
    # reduce_catalogs caps its query at the observation's limit; raise it for the query only.
    navigator_limit = obs.star_max_usable_vmag
    obs.star_max_usable_vmag = lambda: deep_limit  # type: ignore[method-assign]
    try:
        stars = reduce_catalogs(obs, config)
    finally:
        obs.star_max_usable_vmag = navigator_limit  # type: ignore[method-assign]
    record: dict[str, Any] = {
        'frame': asdict(frame),
        'current_limit': current_limit,
        'deep_limit': deep_limit,
        'n_catalog': len(stars),
        'shape': list(nav.shape),
    }
    pv = np.array([float(s.v) - _CORNER_TO_INDEX for s in stars])
    pu = np.array([float(s.u) - _CORNER_TO_INDEX for s in stars])
    vmag = np.array([float(s.vmag) if s.vmag is not None else np.inf for s in stars])

    hp = highpass(nav, params.highpass_size, subsample=params.highpass_subsample)
    peaks = detect_peaks(
        hp,
        smooth_sigma=params.smooth_sigma,
        threshold_snr=params.peak_threshold_snr,
        edge=params.edge_px,
        max_peaks=params.max_peaks,
    )
    record['n_peaks'] = int(peaks.v.size)
    record['hp_noise'] = robust_sigma(hp.ravel())

    margin = params.search_px
    near = np.nonzero(
        (pv > -margin)
        & (pv < nav.shape[0] + margin)
        & (pu > -margin)
        & (pu < nav.shape[1] + margin)
    )[0]
    voters = near[np.argsort(vmag[near])][: params.max_vote_stars]
    center = ((nav.shape[0] - 1) / 2.0, (nav.shape[1] - 1) / 2.0)
    fix = find_pointing(
        pv[voters],
        pu[voters],
        peaks,
        center_vu=center,
        search_px=params.search_px,
        vote_bin_px=params.vote_bin_px,
        agree_px=params.agree_px,
        min_agree=params.min_agree,
        max_chance_probability=params.max_chance_probability,
    )
    fix_record = asdict(fix)
    agreeing_global = [int(voters[k]) for k in fix.agreeing]
    fix_record['agreeing'] = agreeing_global
    fix_record['agreeing_vmag'] = [float(vmag[k]) for k in agreeing_global]
    record['fix'] = fix_record
    if not fix.accepted:
        return record

    tv, tu = apply_fix(fix, pv, pu, center)
    inside = np.nonzero(
        (tv >= params.edge_px)
        & (tv < nav.shape[0] - params.edge_px)
        & (tu >= params.edge_px)
        & (tu < nav.shape[1] - params.edge_px)
    )[0]
    occluded = _occlusion_mask(obs, config, margin_px=params.search_px)
    for k in inside:
        if not stars[k].conflicts and occluded(float(pv[k]), float(pu[k])):
            stars[k].conflicts = 'BODY OR RING'
    phot_v, phot_u = to_phot(tv[inside], tu[inside])
    record['stars'] = [
        _measure_star(
            int(k),
            stars[k],
            float(tv[k]),
            float(tu[k]),
            float(phot_v[j]),
            float(phot_u[j]),
            nav=nav,
            hp=hp,
            phot=phot,
            all_v=tv,
            all_u=tu,
            vmag=vmag,
            agreeing=int(k) in set(agreeing_global),
            params=params,
        )
        for j, k in enumerate(inside)
    ]
    record['sky_sigma_phot'] = _median_sky_sigma(record['stars'])
    return record


#: Pixel step of the occlusion grid (the navigator's own check tests each star separately,
#: which costs a backplane per star; one coarse grid per frame serves every star).
_OCCLUSION_STEP_PX = 4


def _occlusion_mask(
    obs: ObsSnapshotInst, config: Config, *, margin_px: float
) -> Callable[[float, float], bool]:
    """Return a test of whether a predicted star position is behind a body or an opaque ring.

    The bodies are the closest planet and its configured satellites; the rings are the
    planet's ``stars.ring_occlusion_radii_km`` annuli, as in the navigator's own check.  The
    grid samples the field every ``_OCCLUSION_STEP_PX`` pixels, padded by ``margin_px`` so
    that stars predicted outside the field (which a pointing offset can bring inside) are
    covered, and is dilated by ``stars.body_conflict_margin``.  Positions are tested where the
    catalog predicts them: a pointing error moves a star and the body in front of it together.

    Parameters:
        obs: The observation.
        config: The configuration.
        margin_px: Padding of the grid beyond the field on every side.

    Returns:
        A function of a predicted array-index position ``(v, u)`` returning True when the
        star there is occluded.
    """
    planet = obs.closest_planet
    if planet is None:
        return lambda _v, _u: False
    step = _OCCLUSION_STEP_PX
    nv, nu = obs.data.shape
    origin = (-margin_px, -margin_px)
    limit = (nu + margin_px, nv + margin_px)
    meshgrid = Meshgrid.for_fov(obs.fov, origin=origin, limit=limit, undersample=step, swap=True)
    backplane = Backplane(obs, meshgrid)
    shape = meshgrid.shape
    mask = np.zeros(shape, dtype=bool)
    # Test only the bodies near the padded field: a backplane caches every array it computes,
    # and a planet with dozens of satellites would otherwise hold gigabytes per frame.
    # Every FOV the camera readers build is a FlatFOV or a subclass, which carries uv_scale.
    scale = float(np.mean(obs.fov.uv_scale.vals))  # type: ignore[attr-defined]
    nearby = obs.inventory([planet, *config.satellites(planet)], expand=margin_px * scale)
    for body in nearby:
        hit = backplane.where_intercepted(body)
        mask |= np.broadcast_to(np.asarray(hit.vals, dtype=bool), shape)
    annuli = config.stars.ring_occlusion_radii_km.get(planet.upper(), ())
    if config.stars.ring_occlusion_enabled and annuli:
        radii = backplane.ring_radius(f'{planet.lower()}:ring')
        values = np.broadcast_to(np.asarray(radii.vals, dtype=np.float64), shape)
        valid = ~np.broadcast_to(np.asarray(radii.mask, dtype=bool), shape)
        for inner, outer in annuli:
            mask |= valid & (values >= inner) & (values <= outer)
    grow = math.ceil(float(config.stars.body_conflict_margin) / step) + 1
    mask = scipy.ndimage.binary_dilation(mask, iterations=grow)

    def occluded(v: float, u: float) -> bool:
        iv = math.floor((v + _CORNER_TO_INDEX + margin_px) / step)
        iu = math.floor((u + _CORNER_TO_INDEX + margin_px) / step)
        if 0 <= iv < shape[0] and 0 <= iu < shape[1]:
            return bool(mask[iv, iu])
        return False

    return occluded


def _median_sky_sigma(star_records: list[dict[str, Any]]) -> float | None:
    """Return the median sky-annulus sigma of the measured stars, the frame's DN noise.

    Parameters:
        star_records: The per-star records.

    Returns:
        The median, or None when no star was measured.
    """
    values = [s['sky_sigma'] for s in star_records if s.get('sky_sigma') is not None]
    return float(np.median(values)) if values else None


def _measure_star(
    index: int,
    star: Any,
    v: float,
    u: float,
    phot_v: float,
    phot_u: float,
    *,
    nav: FloatArray,
    hp: FloatArray,
    phot: FloatArray,
    all_v: FloatArray,
    all_u: FloatArray,
    vmag: FloatArray,
    agreeing: bool,
    params: MeasureParams,
) -> dict[str, Any]:
    """Measure one catalog star at its fixed position.

    Parameters:
        index: The star's index in the frame's catalog list.
        star: The catalog star record.
        v: Fixed line in the navigator's image (array-index coordinates).
        u: Fixed sample in the navigator's image.
        phot_v: The same position in the photometric image.
        phot_u: The same position in the photometric image.
        nav: The navigator's image.
        hp: The navigator's image, high-passed.
        phot: The photometric image.
        all_v: Fixed lines of every catalog star.
        all_u: Fixed samples of every catalog star.
        vmag: V magnitudes of every catalog star.
        agreeing: Whether the star was one of the pointing fix's agreeing stars.
        params: Measurement parameters.

    Returns:
        The star's record.
    """
    dist = np.hypot(all_v - v, all_u - u)
    dist[index] = np.inf
    neighbors = (dist <= params.blend_px) & (vmag <= float(star.vmag) + params.blend_dmag)
    smear = math.hypot(float(star.move_v), float(star.move_u))
    out: dict[str, Any] = {
        'index': index,
        'vmag': float(star.vmag),
        'bmag': None if star.johnson_mag_faked else _float_or_none(star.johnson_mag_b),
        'catalog': str(star.catalog_name),
        'photometry_saturated': bool(star.photometry_saturated),
        'v': v,
        'u': u,
        'conflict': str(star.conflicts or ''),
        'blended': bool(np.any(neighbors)),
        'smear_px': smear,
        'smeared': smear > params.max_smear_px,
        'agreeing': agreeing,
    }
    det = detection_test(
        hp,
        v,
        u,
        search_px=params.detect_search_px,
        noise_inner=params.noise_inner,
        noise_outer=params.noise_outer,
        box_half=params.box_half,
    )
    if det is not None:
        snr, n3, peak_v, peak_u = det
        out.update(
            peak_snr=snr,
            n_above_3_sigma=n3,
            detected=bool(snr >= DETECT_MIN_PEAK_SNR and n3 >= DETECT_MIN_PIXELS_ABOVE_3_SIGMA),
            peak_v=peak_v,
            peak_u=peak_u,
        )
    small = aperture_photometry(
        phot,
        phot_v,
        phot_u,
        radius=params.aperture_px,
        sky_inner=params.sky_inner,
        sky_outer=params.sky_outer,
    )
    big = aperture_photometry(
        phot,
        phot_v,
        phot_u,
        radius=params.big_aperture_px,
        sky_inner=params.sky_inner,
        sky_outer=params.sky_outer,
    )
    if small is not None and big is not None:
        out.update(
            flux=small.flux,
            flux_sigma=small.flux_sigma,
            flux_big=big.flux,
            flux_big_sigma=big.flux_sigma,
            sky=small.sky,
            sky_sigma=small.sky_sigma,
            max_dn=big.max_value,
            saturated=bool(big.max_value >= params.saturation_dn),
            no_data=bool(
                params.missing_value is not None and big.min_value <= params.missing_value
            ),
        )
    clean = (
        det is not None
        and small is not None
        and out['peak_snr'] >= params.psf_min_snr
        and not out.get('saturated', True)
        and not out.get('no_data', False)
        and not out['blended']
        and not out['smeared']
        and out['conflict'] == ''
    )
    if clean:
        _add_psf(out, nav, params)
    return out


def _add_psf(out: dict[str, Any], nav: FloatArray, params: MeasureParams) -> None:
    """Fit the star's PSF on the navigator's image and add the fit to its record.

    Parameters:
        out: The star's record, which holds the detection peak; updated in place.
        nav: The navigator's image.
        params: Measurement parameters.
    """
    fit = fit_gaussian_psf(nav, out['peak_v'], out['peak_u'], half=params.psf_half)
    if fit is None:
        return
    ee50, ee80 = encircled_energy_radii(
        nav, fit.center_v, fit.center_u, r_max=params.ee_r_max, n_bins=params.ee_n_bins
    )
    out['psf'] = {**asdict(fit), 'ee50': ee50, 'ee80': ee80}


def _float_or_none(value: Any) -> float | None:
    """Return ``value`` as a float, or None when it is None or not finite.

    Parameters:
        value: The value.

    Returns:
        The float or None.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _done_ids(out_path: Path) -> set[str]:
    """Return the image IDs an earlier run already wrote to ``out_path``.

    Parameters:
        out_path: The JSON-lines output.

    Returns:
        The IDs.
    """
    if not out_path.exists():
        return set()
    with out_path.open() as fh:
        return {json.loads(line)['frame']['image_id'] for line in fh if line.strip()}


def _guarded(worker: Callable[[FrameInput], dict[str, Any]], frame: FrameInput) -> dict[str, Any]:
    """Run ``worker`` on one frame, turning an exception into an error record.

    Parameters:
        worker: The per-frame measurement.
        frame: The frame.

    Returns:
        The worker's record, or ``{'frame': ..., 'error': traceback}``.
    """
    try:
        return worker(frame)
    except Exception:  # A failing frame is recorded and the run continues.
        return {'frame': asdict(frame), 'error': traceback.format_exc()[-3000:]}


def run_frames(
    frames: Iterable[FrameInput],
    worker: Callable[[FrameInput], dict[str, Any]],
    out_path: Path,
    *,
    workers: int,
) -> int:
    """Measure frames in a process pool, appending one JSON line per frame to ``out_path``.

    Frames whose image ID ``out_path`` already holds are skipped.  ``worker`` must be a
    module-level function so the spawned workers can import it.

    Parameters:
        frames: The frames.
        worker: The per-frame measurement, returning a JSON-serializable record.
        out_path: The JSON-lines output, appended to.
        workers: Pool size; at most 4 when the images are on the holdings share.

    Returns:
        The number of frames measured by this call.
    """
    done = _done_ids(out_path)
    todo = [f for f in frames if f.image_id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context('spawn')
    count = 0
    with (
        context.Pool(workers, maxtasksperchild=20) as pool,
        out_path.open('a') as fh,
    ):
        jobs = [(worker, f) for f in todo]
        for result in pool.imap_unordered(_star_guarded, jobs, chunksize=1):
            fh.write(json.dumps(result) + '\n')
            fh.flush()
            count += 1
    return count


def _star_guarded(job: tuple[Callable[[FrameInput], dict[str, Any]], FrameInput]) -> dict[str, Any]:
    """Unpack a ``(worker, frame)`` job for :func:`_guarded`.

    Parameters:
        job: The pair.

    Returns:
        The frame's record.
    """
    return _guarded(job[0], job[1])
