"""Measure Voyager ISS star fields for the photometric star limit and the star PSF.

The navigator reads the geometrically corrected product (``*_GEOMED.IMG``, 1000 x 1000), which
the archive's pipeline has resampled and scaled to I/F, so a star's value there does not scale
with exposure.  The vidicon's own signal, which does, is the raw DN of the cleaned product
(``*_CLEANED.IMG``, 800 x 800: the raw image with the reseau marks, known blemishes, and spikes
removed, in the raw geometry).  So this program finds the pointing and measures detection and
the PSF on the navigator's image, and measures each star's DN on the cleaned image at the same
point, carried across by the archive's own tiepoints (``*_GEOMA.TAB``: GEOMED line and sample
to CALIB line and sample, the CALIB and CLEANED geometries being the same).

Frames come from the cumulative PDS3 index of each Voyager ISS volume set: GEOMED products
inside the epochs the local SPICE kernels cover, whose target is not a dark, calibration-lamp,
or plaque frame, and whose inventory does not list a planet (so the field holds stars and at most
moons), exposed at least half a second (NAC) or a tenth of a second (WAC).  Every camera,
filter, and exposure range present is sampled: within each group of encounter, camera, filter,
and exposure octave, up to ``--per-group`` frames evenly spaced in the index.  The vidicon's
gain state comes from ``GAIN_MODE`` (every frame in these volumes says ``LOW``) and its readout
from ``SCAN_MODE``.

The simulator's PSF kernels are measured on the raw product instead (``*_RAW.IMG``, 800 x 800
8-bit DN), because the archive's despike step, which makes the cleaned product, also cuts the
sharp cores of faint stars.  :func:`extract_raw_cutouts` takes a star-limit run's output and
cuts each fixed frame's isolated stars out of its raw image at the frame's own pointing,
carried across by the tiepoints, and refits each frame's translation and twist on the raw
stars (:func:`raw_star_cutouts`).  The kernel is then fitted with
:func:`util.star_photometry.star_field.fit_psf_kernel` and carried to GEOMED pixels with
:func:`util.star_photometry.star_psf_summary.resampled_kernel`.

Usage::

    python -m util.star_photometry.voyager_iss_star_limits OUT.jsonl [--per-group 5]
        [--workers 4] [--limit N] [--only ID ...]
    python -m util.star_photometry.voyager_iss_star_limits CUTS.npz --raw-cutouts-from OUT.jsonl
        [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.interpolate
from numpy.typing import NDArray
from util.star_photometry.frame_measure import (
    FrameInput,
    MeasureParams,
    deep_config,
    measure_frame,
    run_frames,
)
from util.star_photometry.star_field import StarCutout, fit_kernel_star
from vicar import VicarImage

from spindoctor.obs import ObsVoyagerISS

__all__ = [
    'PARAMS',
    'FrameTwist',
    'extract_raw_cutouts',
    'fit_frame_twist',
    'frame_raw_cutouts',
    'geoma_map',
    'geomed_stretch',
    'main',
    'measure',
    'raw_star_cutouts',
    'reseau_mask',
    'voyager_frames',
]

FloatArray = NDArray[np.float64]

#: A map from GEOMED to raw positions, ``(N, 2)`` array-index ``(v, u)`` in and out.
GeomaMap = Callable[[FloatArray], FloatArray]

#: The volume sets of the four Voyager ISS planetary encounters.
VOLUME_SETS = ('VGISS_5xxx', 'VGISS_6xxx', 'VGISS_7xxx', 'VGISS_8xxx')

#: Observation dates (inclusive ISO dates) the local SPICE kernels cover, from
#: ``util/cohort_curation/spice_coverage.json``.
SPICE_WINDOWS = (
    ('1979-01-31', '1979-04-09'),
    ('1979-06-04', '1979-08-29'),
    ('1980-08-06', '1980-11-19'),
    ('1981-06-07', '1981-09-21'),
    ('1985-08-21', '1986-02-13'),
    ('1988-11-12', '1989-10-01'),
)

#: Targets whose frames do not show a star field.
EXCLUDED_TARGETS = frozenset({'DARK', 'CAL LAMPS', 'PLAQUE'})

#: Inventory entries that mean a planet is in the field.
PLANETS = frozenset({'JUPITER', 'SATURN', 'URANUS', 'NEPTUNE'})

#: Shortest exposure sampled per camera (seconds): below it a star bright enough to show
#: (V 7-8 in the NAC) is rarely in the field (0.4 degree in the NAC), so the frame cannot be
#: fixed from stars.
MIN_EXPOSURE_S = {'NAC': 0.5, 'WAC': 0.1}

#: The 8-bit DN at which the vidicon's ADC saturates.
SATURATION_DN = 255.0

#: The catalog's star budget for the deep query.
DEEP_MAX_STARS = 3000

#: Columns of ``VGISS_n999_index.tab``.
_COL_VOLUME, _COL_PRODUCT, _COL_TARGET, _COL_TIME, _COL_INSTRUMENT = 0, 2, 6, 9, 11
_COL_SCAN, _COL_SHUTTER, _COL_GAIN, _COL_FILTER, _COL_EXPOSURE = 12, 13, 14, 16, 18

#: GEOMED product suffix, and the cleaned and tiepoint products beside it.
_GEOMED, _CLEANED, _GEOMA = '_GEOMED.IMG', '_CLEANED.IMG', '_GEOMA.TAB'

PARAMS = MeasureParams(
    catalog_depth_mag=4.0,
    highpass_size=21,
    smooth_sigma=1.0,
    peak_threshold_snr=5.0,
    edge_px=16,
    max_peaks=300,
    max_vote_stars=60,
    search_px=400.0,
    vote_bin_px=1.5,
    agree_px=1.5,
    min_agree=3,
    max_chance_probability=1e-3,
    detect_search_px=2.0,
    noise_inner=6.0,
    noise_outer=12.0,
    box_half=2,
    aperture_px=3.0,
    big_aperture_px=6.0,
    sky_inner=8.0,
    sky_outer=13.0,
    blend_px=10.0,
    blend_dmag=2.5,
    saturation_dn=SATURATION_DN,
    max_smear_px=1.0,
    psf_min_snr=20.0,
    psf_half=6,
    ee_r_max=8.0,
    ee_n_bins=16,
    highpass_subsample=2,
    missing_value=0.0,
)


def _in_spice_window(time: str) -> bool:
    """Return whether an index ``IMAGE_TIME`` lies in a SPICE-covered epoch.

    Parameters:
        time: The ISO time.

    Returns:
        True inside a window.
    """
    day = time[:10]
    return any(start <= day <= stop for start, stop in SPICE_WINDOWS)


def _exposure_octave(texp: float) -> int:
    """Return the base-2 octave of an exposure, the sampling group's exposure key.

    Parameters:
        texp: Exposure in seconds.

    Returns:
        ``floor(log2(texp))``, or -99 for a non-positive exposure.
    """
    return math.floor(math.log2(texp)) if texp > 0 else -99


def voyager_frames(holdings: Path, per_group: int) -> list[FrameInput]:
    """Return a stratified sample of Voyager ISS star-field frames from the PDS3 indexes.

    Parameters:
        holdings: The PDS3 holdings root (``$PDS3_HOLDINGS_DIR``).
        per_group: The most frames kept per encounter, camera, filter, and exposure
            octave.

    Returns:
        The sampled frames.
    """
    groups: dict[tuple[str, str, str, int], list[FrameInput]] = defaultdict(list)
    for volume_set in VOLUME_SETS:
        stem = volume_set.replace('xxx', '999')
        meta = holdings / 'metadata' / volume_set / stem
        bodies: dict[str, list[str]] = {}
        with (meta / f'{stem}_inventory.csv').open() as fh:
            for row in csv.reader(fh):
                bodies[Path(row[1].strip()).name.split('_')[0]] = [b.strip() for b in row[3:]]
        with (meta / f'{stem}_index.tab').open() as fh:
            for row in csv.reader(fh):
                product = row[_COL_PRODUCT].strip()
                if not product.endswith(_GEOMED):
                    continue
                if row[_COL_TARGET].strip() in EXCLUDED_TARGETS:
                    continue
                if not _in_spice_window(row[_COL_TIME]):
                    continue
                image_id = product.split('_')[0]
                if PLANETS & set(bodies.get(image_id, [])):
                    continue
                texp = float(row[_COL_EXPOSURE])
                camera = 'NAC' if row[_COL_INSTRUMENT].strip().startswith('NARROW') else 'WAC'
                if texp < MIN_EXPOSURE_S[camera]:
                    continue
                volume = row[_COL_VOLUME].strip()
                path = (
                    holdings
                    / 'volumes'
                    / volume_set
                    / volume
                    / 'DATA'
                    / f'{image_id[:6]}XX'
                    / product
                )
                frame = FrameInput(
                    image_id=image_id,
                    path=str(path),
                    camera=camera,
                    filter_name=row[_COL_FILTER].strip(),
                    gain_state=row[_COL_GAIN].strip(),
                    mode=row[_COL_SCAN].strip(),
                    texp=texp,
                    extra={
                        'volume': volume,
                        'target': row[_COL_TARGET].strip(),
                        'shutter': row[_COL_SHUTTER].strip(),
                        'moons': bodies.get(image_id, []),
                    },
                )
                key = (volume_set, camera, frame.filter_name, _exposure_octave(texp))
                groups[key].append(frame)
    sample: list[FrameInput] = []
    for key in sorted(groups):
        members = groups[key]
        if len(members) <= per_group:
            sample += members
            continue
        picks = np.linspace(0, len(members) - 1, per_group).round().astype(int)
        sample += [members[i] for i in sorted(set(picks.tolist()))]
    return sample


def geoma_map(table: Path) -> scipy.interpolate.LinearNDInterpolator:
    """Return the GEOMED-to-CALIB position map of a Voyager image from its tiepoint table.

    The table's rows are ``index, output line, output sample, input line, input sample`` in
    the one-based coordinates of VICAR (the first pixel's center is line 1, sample 1); output
    is the GEOMED image and input the CALIB image.  The map is piecewise linear over the
    tiepoints' triangulation.

    Parameters:
        table: The ``*_GEOMA.TAB`` file.

    Returns:
        An interpolator taking ``(line, sample)`` GEOMED array-index positions to CALIB
        array-index positions (NaN outside the tiepoint mesh).
    """
    rows = np.loadtxt(table, delimiter=',')
    points = np.unique(rows[:, 1:5], axis=0)
    return scipy.interpolate.LinearNDInterpolator(points[:, 0:2] - 1.0, points[:, 2:4] - 1.0)


#: The raw product and the reseau table beside a GEOMED product.
_RAW, _RESLOC = '_RAW.IMG', '_RESLOC.TAB'

#: The side of a raw frame (pixels) and its center in array-index coordinates, the point
#: a frame's twist turns about.
RAW_SIZE = 800
RAW_CENTER = (RAW_SIZE - 1) / 2.0

#: Half-sizes (pixels) of a star's raw cutout and of its GEOMED cutout.
RAW_CUTOUT_HALF = 13
GEOMED_CUTOUT_HALF = 14

#: A raw cutout keeps at least this many pixels (pixels) from the frame's edge.
RAW_EDGE_MARGIN_PX = 2

#: A star is isolated when no catalog star less than :data:`ISOLATION_DMAG` magnitudes
#: fainter lies within :data:`ISOLATION_RAW_PX` raw pixels of it.
ISOLATION_RAW_PX = 14.0
ISOLATION_DMAG = 3.0

#: Raw pixels within this distance of a reseau mark are left out of a cutout ...
RESEAU_MASK_PX = 3.5

#: ... and a star within this distance of one is not cut out.
RESEAU_EXCLUDE_PX = 6.0

#: Pixels within this distance (raw pixels) of a star's predicted center are its core,
#: where a value the archive's despike step changed is still the star's own.
CORE_RADIUS_RAW_PX = 2.5

#: The sky of a raw cutout is the pixels at least this far (pixels) from its center.
RAW_SKY_INNER_PX = 7.0

#: The least fraction of a raw cutout's pixels that must be usable.
MIN_VALID_FRACTION = 0.85

#: The least peak (DN above the sky) of a star that is cut out.
RAW_MIN_PEAK_DN = 8.0

#: The kernel and half-width that place each star's center before the frame's twist
#: is fitted.
PROVISIONAL_KERNEL = {'sigma_v': 0.8, 'sigma_u': 0.8, 'w': 0.0, 'r0': 2.0, 'n': 3.0}
PROVISIONAL_TRUNCATION_PX = 8

#: A star whose fitted center lies farther than this (raw pixels) from its frame's
#: translation and twist is left out of that fit.
TWIST_REJECT_PX = 1.0

#: How many times a frame's twist is refitted after rejecting stars.
TWIST_ITERATIONS = 6

#: Stars a frame needs for its twist to be fitted; with fewer only a translation is.
TWIST_MIN_STARS = 3

#: A gain so large that a star's shot noise is negligible, for the provisional centers.
_NO_SHOT_E_PER_DN = 1.0e6


def reseau_mask(reseau_vu: FloatArray, *, size: int, radius_px: float) -> NDArray[np.bool_]:
    """Return the pixels of a square frame within a distance of any reseau mark.

    Parameters:
        reseau_vu: The marks' ``(v, u)`` positions, one row per mark, in array-index
            coordinates.
        size: The frame's side, in pixels.
        radius_px: The distance, in pixels.

    Returns:
        A ``(size, size)`` mask, True near a mark.
    """
    mask = np.zeros((size, size), dtype=bool)
    reach = math.ceil(radius_px)
    for mv, mu in reseau_vu:
        if not (-reach <= mv <= size - 1 + reach and -reach <= mu <= size - 1 + reach):
            continue
        sv = slice(max(round(mv) - reach, 0), min(round(mv) + reach + 1, size))
        su = slice(max(round(mu) - reach, 0), min(round(mu) + reach + 1, size))
        gv, gu = np.mgrid[sv, su]
        mask[sv, su] |= np.hypot(gv - mv, gu - mu) <= radius_px
    return mask


def _raw_sky(cut: FloatArray, valid: NDArray[np.bool_]) -> tuple[float, float]:
    """Return a raw cutout's sky level and variance from its outer usable pixels.

    The variance is that of the pixels within three standard deviations (plus half a DN,
    for the rounding) of the median, iterated five times.

    Parameters:
        cut: The cutout.
        valid: Its usable pixels.

    Returns:
        ``(median, variance)`` in DN and DN squared.
    """
    gv, gu = np.mgrid[: cut.shape[0], : cut.shape[1]]
    middle = (cut.shape[0] - 1) / 2.0
    values = cut[(np.hypot(gv - middle, gu - middle) >= RAW_SKY_INNER_PX) & valid]
    median = float(np.median(values))
    keep = values
    for _ in range(5):
        keep = values[np.abs(values - median) < 3.0 * keep.std() + 0.5]
    return median, float(keep.var())


@dataclass(frozen=True)
class FrameTwist:
    """A frame's translation and twist, fitted to its stars' centers.

    Attributes:
        shift_v: The translation along lines (pixels).
        shift_u: The translation along samples (pixels).
        twist_rad: The rotation about the frame's center (radians, small).
        model: ``'translation_twist'``, ``'translation'``, or ``'none'`` when no star is
            left to fit.
        used: Which stars the final fit used.
    """

    shift_v: float
    shift_u: float
    twist_rad: float
    model: str
    used: NDArray[np.bool_]

    def apply(self, v: FloatArray, u: FloatArray, center: float) -> tuple[FloatArray, FloatArray]:
        """Return predicted positions moved by the translation and twist.

        Parameters:
            v: Predicted lines.
            u: Predicted samples.
            center: The center the twist turns about (the same for both axes).

        Returns:
            The moved ``(v, u)``.
        """
        dv, du = v - center, u - center
        return (
            dv + self.shift_v - self.twist_rad * du + center,
            du + self.shift_u + self.twist_rad * dv + center,
        )


def fit_frame_twist(
    predicted: FloatArray,
    measured: FloatArray,
    *,
    center: float,
    reject_px: float = TWIST_REJECT_PX,
) -> FrameTwist:
    """Fit a frame's translation and small twist to its stars' measured centers.

    The twist turns about ``center`` on both axes.  With at least
    :data:`TWIST_MIN_STARS` stars in the fit, the translation and twist are solved by
    linear least squares on the small-angle form; with fewer, the translation alone is
    the mean shift.  A star farther than ``reject_px`` from the fit is left out and the
    fit repeated, up to :data:`TWIST_ITERATIONS` times, until the stars used stop
    changing.

    Parameters:
        predicted: The stars' predicted ``(v, u)``, one row per star.
        measured: Their measured ``(v, u)``.
        center: The point the twist turns about.
        reject_px: The largest distance from the fit (pixels) a used star may lie.

    Returns:
        The fit.
    """
    pv, pu = predicted[:, 0] - center, predicted[:, 1] - center
    mv, mu = measured[:, 0] - center, measured[:, 1] - center
    used = np.ones(len(predicted), dtype=bool)
    twist = FrameTwist(0.0, 0.0, 0.0, 'none', ~used)
    for _ in range(TWIST_ITERATIONS):
        count = int(used.sum())
        if count == 0:
            return FrameTwist(0.0, 0.0, 0.0, 'none', used)
        if count >= TWIST_MIN_STARS:
            design = np.zeros((2 * count, 3))
            design[0::2, 0] = 1.0
            design[0::2, 2] = -pu[used]
            design[1::2, 1] = 1.0
            design[1::2, 2] = pv[used]
            rhs = np.empty(2 * count)
            rhs[0::2] = mv[used] - pv[used]
            rhs[1::2] = mu[used] - pu[used]
            solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
            twist = FrameTwist(
                float(solution[0]),
                float(solution[1]),
                float(solution[2]),
                'translation_twist',
                used,
            )
        else:
            twist = FrameTwist(
                float(np.mean(mv[used] - pv[used])),
                float(np.mean(mu[used] - pu[used])),
                0.0,
                'translation',
                used,
            )
        fv, fu = twist.apply(predicted[:, 0], predicted[:, 1], center)
        keep = np.hypot(measured[:, 0] - fv, measured[:, 1] - fu) < reject_px
        if np.array_equal(keep, used):
            break
        used = keep
    return twist


def geomed_stretch(mapping: GeomaMap, v: float, u: float) -> float:
    """Return the GEOMED pixels per raw pixel at a GEOMED position.

    The stretch is one over the square root of the area a GEOMED pixel covers in the raw
    frame, from the map's derivatives one pixel either side.

    Parameters:
        mapping: The GEOMED-to-raw map.
        v: The GEOMED line.
        u: The GEOMED sample.

    Returns:
        The stretch, or NaN where the map is undefined.
    """
    points = np.array([[v + 1, u], [v - 1, u], [v, u + 1], [v, u - 1]], dtype=np.float64)
    raw = mapping(points)
    jac_v = (raw[0] - raw[1]) / 2.0
    jac_u = (raw[2] - raw[3]) / 2.0
    area = abs(jac_v[0] * jac_u[1] - jac_v[1] * jac_u[0])
    return float(1.0 / math.sqrt(area)) if area > 0 else math.nan


def raw_star_cutouts(
    raw: FloatArray,
    cleaned: FloatArray,
    geomed: FloatArray,
    mapping: GeomaMap,
    stars: Sequence[dict[str, Any]],
    reseau_vu: FloatArray,
) -> list[dict[str, Any]]:
    """Cut a frame's isolated catalog stars out of its raw image, at the frame's own pointing.

    Each star detected on the GEOMED image (and neither blended nor in conflict) is
    carried to the raw geometry by ``mapping``.  A star is cut out when it is isolated
    (:data:`ISOLATION_RAW_PX`, :data:`ISOLATION_DMAG`), lies at least
    :data:`RESEAU_EXCLUDE_PX` from every reseau mark, its cutout lies inside the frame
    and is at least :data:`MIN_VALID_FRACTION` usable, and it peaks at least
    :data:`RAW_MIN_PEAK_DN` above the sky.  A pixel is unusable when it is zero, lies
    within :data:`RESEAU_MASK_PX` of a reseau mark, or differs between the raw and
    cleaned images outside the star's core: the archive's despike step changed it, so
    it held a spike.  Inside the core (:data:`CORE_RADIUS_RAW_PX`) the raw value is kept,
    because the despike step also cuts sharp star cores.

    Each star's center is fitted on its cutout with :data:`PROVISIONAL_KERNEL`, and the
    frame's translation and twist about the raw center are fitted to those centers
    (:func:`fit_frame_twist`).  Each star's record carries its raw cutout and usable
    pixels, its GEOMED cutout, the raw position of every GEOMED cutout pixel, its sky,
    its peak, how many of its core pixels the despike step changed and by how many DN,
    the local stretch (:func:`geomed_stretch`), and its place in the frame's fit.

    Parameters:
        raw: The raw image (``*_RAW.IMG``), 800 x 800 DN.
        cleaned: The cleaned image (``*_CLEANED.IMG``), in the raw geometry.
        geomed: The GEOMED image the stars were found on.
        mapping: The GEOMED-to-raw position map (:func:`geoma_map`).
        stars: The frame's catalog stars from its star-limit record, with ``v``, ``u``
            (GEOMED array-index coordinates), ``vmag``, and the ``detected``,
            ``blended``, and ``conflict`` flags.
        reseau_vu: The reseau marks' raw ``(v, u)``, one row per mark.

    Returns:
        One record per star cut out.
    """
    size = raw.shape[0]
    gv = np.array([s['v'] for s in stars], dtype=np.float64)
    gu = np.array([s['u'] for s in stars], dtype=np.float64)
    vmag = np.array([s['vmag'] for s in stars], dtype=np.float64)
    raw_vu = mapping(np.column_stack([gv, gu])) if len(stars) else np.zeros((0, 2))
    rv, ru = raw_vu[:, 0], raw_vu[:, 1]
    changed = raw != cleaned
    reseau = reseau_mask(reseau_vu, size=size, radius_px=RESEAU_MASK_PX)
    half, ghalf = RAW_CUTOUT_HALF, GEOMED_CUTOUT_HALF
    side = 2 * half + 1
    cv, cu = np.mgrid[:side, :side]
    out: list[dict[str, Any]] = []
    for k, star in enumerate(stars):
        if star.get('conflict') or star.get('blended') or not star.get('detected'):
            continue
        if not (math.isfinite(rv[k]) and math.isfinite(ru[k])):
            continue
        distance = np.hypot(rv - rv[k], ru - ru[k])
        distance[k] = np.inf
        if np.any((distance < ISOLATION_RAW_PX) & (vmag < vmag[k] + ISOLATION_DMAG)):
            continue
        if np.any(np.hypot(reseau_vu[:, 0] - rv[k], reseau_vu[:, 1] - ru[k]) < RESEAU_EXCLUDE_PX):
            continue
        iv, iu = round(rv[k]), round(ru[k])
        if (
            min(iv, iu) - half < RAW_EDGE_MARGIN_PX
            or max(iv, iu) + half >= size - RAW_EDGE_MARGIN_PX
        ):
            continue
        window = (slice(iv - half, iv + half + 1), slice(iu - half, iu + half + 1))
        cut = raw[window]
        core = np.hypot(cv - (rv[k] - iv + half), cu - (ru[k] - iu + half)) < CORE_RADIUS_RAW_PX
        despiked = changed[window]
        valid = (cut > 0) & ~reseau[window] & ~(despiked & ~core)
        if valid.mean() < MIN_VALID_FRACTION:
            continue
        sky, sky_var = _raw_sky(cut, valid)
        peak = float(cut[core].max() - sky)
        if peak < RAW_MIN_PEAK_DN:
            continue
        giv, giu = round(gv[k]), round(gu[k])
        if min(giv, giu) - ghalf < 0 or max(giv, giu) + ghalf >= geomed.shape[0]:
            continue
        gwindow = (slice(giv - ghalf, giv + ghalf + 1), slice(giu - ghalf, giu + ghalf + 1))
        mesh_v, mesh_u = np.mgrid[gwindow]
        geomed_raw_vu = mapping(np.column_stack([mesh_v.ravel(), mesh_u.ravel()]))
        out.append(
            {
                'k': k,
                'cut': cut,
                'valid': valid,
                'core_changed': int(np.count_nonzero(despiked & core)),
                'core_change_dn': float(np.sum((cleaned[window] - cut)[despiked & core])),
                'v0': iv - half,
                'u0': iu - half,
                'pred_v': float(rv[k]),
                'pred_u': float(ru[k]),
                'sky': sky,
                'sky_var': sky_var,
                'peak_dn': peak,
                'max_dn': float(cut[core].max()),
                'vmag': float(vmag[k]),
                'bmag': star.get('bmag'),
                'catalog': star.get('catalog'),
                'smear_geo_px': star.get('smear_px'),
                'stretch': geomed_stretch(mapping, float(gv[k]), float(gu[k])),
                'geo_cut': geomed[gwindow],
                'geo_v0': giv - ghalf,
                'geo_u0': giu - ghalf,
                'geo_v': float(gv[k]),
                'geo_u': float(gu[k]),
                'geo_rawmap': geomed_raw_vu.reshape(2 * ghalf + 1, 2 * ghalf + 1, 2),
            }
        )
    for c in out:
        fit = fit_kernel_star(
            StarCutout(
                image=c['cut'],
                valid=c['valid'],
                v=c['pred_v'] - c['v0'],
                u=c['pred_u'] - c['u0'],
                sky_var=max(c['sky_var'], 0.1),
                gain_e_per_dn=_NO_SHOT_E_PER_DN,
            ),
            PROVISIONAL_KERNEL,
            truncation_px=PROVISIONAL_TRUNCATION_PX,
        )
        c['fit_v'] = fit.center_v + c['v0']
        c['fit_u'] = fit.center_u + c['u0']
        c['fit_flux'] = fit.flux
    if not out:
        return out
    predicted = np.array([[c['pred_v'], c['pred_u']] for c in out])
    measured = np.array([[c['fit_v'], c['fit_u']] for c in out])
    twist = fit_frame_twist(predicted, measured, center=RAW_CENTER)
    tw_v, tw_u = twist.apply(predicted[:, 0], predicted[:, 1], RAW_CENTER)
    for j, c in enumerate(out):
        c['tw_v'] = float(tw_v[j])
        c['tw_u'] = float(tw_u[j])
        c['dev'] = float(math.hypot(c['fit_v'] - tw_v[j], c['fit_u'] - tw_u[j]))
        c['in_frame_fit'] = bool(twist.used[j])
        c['frame_model'] = twist.model
        c['frame_n_fit'] = int(twist.used.sum())
        c['frame_twist_deg'] = math.degrees(twist.twist_rad)
        c['frame_shift'] = (twist.shift_v, twist.shift_u)
    return out


def _image(path: str) -> FloatArray:
    """Return a VICAR image's array as floats.

    Parameters:
        path: The image file.

    Returns:
        The image.
    """
    image = VicarImage.from_file(path)
    assert isinstance(image, VicarImage)  # A tuple only when extraneous='include'.
    return np.asarray(image.data_2d, dtype=np.float64)


def frame_raw_cutouts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Read one frame's products and cut its stars out of the raw image.

    Parameters:
        record: The frame's star-limit record (from :func:`measure`), with its pointing
            fix accepted.

    Returns:
        The frame's star records (:func:`raw_star_cutouts`).
    """
    path = record['frame']['path']
    reseau = np.loadtxt(path.replace(_GEOMED, _RESLOC), delimiter=',')
    return raw_star_cutouts(
        _image(path.replace(_GEOMED, _RAW)),
        _image(path.replace(_GEOMED, _CLEANED)),
        _image(path),
        geoma_map(Path(path.replace(_GEOMED, _GEOMA))),
        record['stars'],
        reseau[:, 1:3] - 1.0,
    )


def _raw_cutouts_guarded(
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]] | str]:
    """Run :func:`frame_raw_cutouts`, returning a failure's last traceback line instead.

    Parameters:
        record: The frame's star-limit record.

    Returns:
        ``(record, stars)``, or ``(record, message)`` on a failure.
    """
    try:
        return record, frame_raw_cutouts(record)
    except Exception as exc:  # One frame's failure must not stop the run.
        return record, f'{type(exc).__name__}: {exc}'


#: The cutout arrays saved beside the star records.
_CUTOUT_ARRAYS = ('cut', 'valid', 'geo_cut', 'geo_rawmap')


def extract_raw_cutouts(records_path: Path, out_path: Path, *, workers: int) -> int:
    """Cut the stars of every fixed frame in a star-limit run out of its raw image.

    Each frame whose pointing fix was accepted is read once.  The cutouts are saved to a
    compressed ``.npz`` with one array per kind (``cut``, ``valid``, ``geo_cut``,
    ``geo_rawmap``), stacked over the stars, and ``meta``, a JSON list of each star's
    record with its frame's image ID, camera, spacecraft, filter, exposure, volume,
    path, and target.

    Parameters:
        records_path: The star-limit run's JSON-lines output.
        out_path: The ``.npz`` to write.
        workers: Pool size; at most 4 when the images are on the holdings share.

    Returns:
        How many stars were cut out.
    """
    seen: set[str] = set()
    records = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if 'error' in record or not record.get('fix', {}).get('accepted'):
            continue
        if record['frame']['image_id'] in seen:
            continue
        seen.add(record['frame']['image_id'])
        records.append(record)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[Any]] = {key: [] for key in _CUTOUT_ARRAYS}
    context = multiprocessing.get_context('spawn')
    with context.Pool(workers) as pool:
        for record, result in pool.imap_unordered(_raw_cutouts_guarded, records):
            frame = record['frame']
            if isinstance(result, str):
                print('error', frame['image_id'], result)
                continue
            for star in result:
                for key in _CUTOUT_ARRAYS:
                    arrays[key].append(star.pop(key))
                star.update(
                    image_id=frame['image_id'],
                    camera=frame['camera'],
                    spacecraft=record['spacecraft'],
                    filter=frame['filter_name'],
                    texp=frame['texp'],
                    volume=frame['extra']['volume'],
                    path=frame['path'],
                    target=frame['extra']['target'],
                )
                rows.append(star)
    saved: dict[str, Any] = {key: np.array(value) for key, value in arrays.items()}
    saved['meta'] = json.dumps(rows)
    np.savez_compressed(out_path, **saved)
    return len(rows)


def measure(frame: FrameInput) -> dict[str, Any]:
    """Measure one Voyager ISS frame (the pool worker).

    Parameters:
        frame: The frame.

    Returns:
        The frame's record.
    """
    config = deep_config(DEEP_MAX_STARS)
    obs = ObsVoyagerISS.from_file(frame.path, config=config)
    cleaned = VicarImage.from_file(frame.path.replace(_GEOMED, _CLEANED))
    assert isinstance(cleaned, VicarImage)  # A tuple only when extraneous='include'.
    mapping = geoma_map(Path(frame.path.replace(_GEOMED, _GEOMA)))

    def to_cleaned(v: FloatArray, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        out = mapping(np.column_stack([v, u]))
        return out[:, 0], out[:, 1]

    record = measure_frame(
        obs,
        frame,
        PARAMS,
        config=config,
        phot_image=np.asarray(cleaned.data_2d, dtype=np.float64),
        phot_map=to_cleaned,
    )
    record['obs_texp'] = float(obs.texp)
    record['obs_detector'] = str(obs.detector)
    record['spacecraft'] = obs.spacecraft_digit
    return record


def main() -> None:
    """Measure a stratified sample of Voyager ISS star fields, or cut their raw stars out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'out', type=Path, help='JSON-lines output (appended; resumes), or the cutouts .npz'
    )
    parser.add_argument(
        '--raw-cutouts-from',
        type=Path,
        default=None,
        help='Cut the stars of this star-limit output out of the raw frames into OUT',
    )
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Measure only the first N frames')
    parser.add_argument('--only', nargs='*', default=[], help='Measure only these image IDs')
    parser.add_argument('--list', action='store_true', help='Print the sample and stop')
    args = parser.parse_args()
    if args.raw_cutouts_from is not None:
        count = extract_raw_cutouts(args.raw_cutouts_from, args.out, workers=args.workers)
        print(count, 'stars cut out')
        return
    frames = voyager_frames(Path(os.environ['PDS3_HOLDINGS_DIR']), args.per_group)
    if args.only:
        frames = [f for f in frames if f.image_id in set(args.only)]
    if args.limit:
        frames = frames[: args.limit]
    if args.list:
        for f in frames:
            print(f.image_id, f.camera, f.filter_name, f.texp, f.mode, f.extra['target'])
        return
    run_frames(frames, measure, args.out, workers=args.workers)


if __name__ == '__main__':
    main()
