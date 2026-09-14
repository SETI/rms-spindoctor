"""Stage A cohort scan (plans/COHORT_CURATION_PLAN.md): candidate manifest
for the eight scene classes the image library lacked entirely, out of the
first-stage scene-class budget in that plan's appendix:

    body_irregular, faint_stars, negative_cases, ring_only_flat,
    ring_plus_body, scattered_light, stars_plus_body,
    two_bright_stars_no_body

Scans the local PDS geometry metadata only (no image access).  Star counts
use the UCAC4 catalog directly with the per-camera limiting-magnitude
formulas mirrored from spindoctor.obs.obs_inst_* (Pogson form:
anchor + ln(texp)/ln(2.512)).

Output (under _work/cohort_curation/, gitignored): candidates_batch001.yaml
(stratified, seeded sample) plus scan_counts.txt with raw per-class hit
counts.

Run:  venv/bin/python util/cohort_curation/scan_stage_a.py
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

import pdsmeta as pm

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'   # generated outputs (gitignored)

SEED = 20260708

RADII = json.loads((HERE / 'body_radii.json').read_text())['radius_km']

# Usable epochs per mission (see spice_coverage.json _provenance); frames
# outside these windows fail navigation with missing-SPICE errors, so they
# are excluded from candidacy for every scene class.
SPICE_COVERAGE = json.loads(
    (HERE / 'spice_coverage.json').read_text())['windows']


def epoch_covered(mission: str, image_time: str) -> bool:
    """True if the ISO image time falls inside a usable SPICE window.

    Parameters:
        mission: Mission key ('COISS', 'VGISS', 'GOSSI', 'NHLORRI').
        image_time: ISO date/time string from the index table; an empty
            or unknown value is treated as covered (Stage B triage
            catches any survivor).
    """
    if not image_time:
        return True
    windows = SPICE_COVERAGE.get(mission)
    if not windows:
        return True
    day = image_time[:10]
    return any(lo <= day <= hi for lo, hi in windows)

# Images already in the library (curated or uncurated) or offered in a
# prior batch manifest -- never re-offer.  Populated by load_existing_ids
# in main(); module-level so the scan functions can consult it.
EXISTING_IDS: set[str] = set()


def load_existing_ids(batch: int) -> set[str]:
    """IDs to exclude: library sidecar stems + prior-batch candidates.

    Parameters:
        batch: Current batch number; only manifests from earlier
            batches are excluded (a re-run of the current batch must
            not exclude its own previous output).
    """
    ids = {
        p.stem.replace('_CALIB', '')
        for p in (REPO / 'tests/integration/image_library').rglob('*.yaml')
    }
    for mp in sorted(OUT_DIR.glob('candidates_batch*.yaml')):
        try:
            prior_batch = int(mp.stem.rsplit('batch', 1)[1])
        except ValueError:
            continue
        if prior_batch >= batch:
            continue
        prior = yaml.safe_load(mp.read_text())
        for group in (prior.get('classes') or {}).values():
            ids |= {c['image_name'] for c in group}
    return ids

IRREGULARS = {'PHOEBE', 'HYPERION', 'JANUS', 'EPIMETHEUS', 'PROMETHEUS',
              'PANDORA', 'ATLAS', 'PAN', 'TELESTO', 'CALYPSO', 'HELENE'}

FOV_DEG = {('COISS', 'NAC'): 0.35, ('COISS', 'WAC'): 3.5,
           ('VGISS', 'NA'): 0.424, ('VGISS', 'WA'): 3.169,
           ('GOSSI', 'SSI'): 0.46, ('NHLORRI', 'LORRI'): 0.29}

FRAME_PX = 1024.0
# Edge sagitta (px) below which a ring edge reads as flat across the frame,
# making the scene rank-1 (the ring_only_flat class; see
# docs/dev_guide/dev_guide_image_library.rst for the class definitions).
FLAT_SAGITTA_PX = 0.5
FLAT_MIN_APPARENT_R_PX = FRAME_PX * FRAME_PX / (8.0 * FLAT_SAGITTA_PX)

# Minimum clearly-bright catalog stars a stars-only scattered_light surrogate
# must show to be autonomously navigable.  It mirrors the star-field pattern
# matcher's inlier floor (config_510_techniques.yaml pattern_match_min_inliers);
# with no resolved body or ring to fall back on, a frame carrying fewer stars
# than that floor cannot reach a star-field solve, and the two-star fallback
# needs a bounded pointing prior these surrogates lack.  See issue #238: the
# Galileo C00598xx quintet each showed only ~2 bright stars and failed wholesale
# with all_techniques_spurious.
STAR_FIELD_MIN_INLIERS = 6

LN_POGSON = math.log(2.512)


def maglim(mission: str, camera: str, texp_s: float) -> float:
    """Limiting vmag; mirrors spindoctor.obs.obs_inst_* star_max_usable_vmag."""
    anchors = {('COISS', 'NAC'): 10.5, ('COISS', 'WAC'): 10.7,
               ('GOSSI', 'SSI'): 10.3, ('NHLORRI', 'LORRI'): 11.7,
               ('VGISS', 'NA'): 8.3, ('VGISS', 'WA'): 5.9}
    anchor = anchors[(mission, camera)]
    if texp_s <= 0:
        return anchor
    if (mission, camera) == ('COISS', 'WAC'):
        return anchor + math.log(texp_s / 26.0) / LN_POGSON
    return anchor + math.log(texp_s) / LN_POGSON


# ---------------------------------------------------------------- ring edges

def load_saturn_ring_edges() -> dict[str, float]:
    """{feature_edge_name: a_km} for every mode-1 edge in config_310."""
    cfg = yaml.safe_load(
        (REPO / 'src/spindoctor/config_files/config_310_saturn_rings.yaml')
        .read_text())
    edges: dict[str, float] = {}
    feats = cfg['rings']['ring_features']['SATURN']['features']
    for fname, feat in feats.items():
        for side in ('inner_data', 'outer_data'):
            for mode in feat.get(side) or []:
                if mode.get('mode') == 1 and 'a' in mode:
                    edges[f'{fname}.{side.split("_")[0]}'] = mode['a']
    return edges


@functools.lru_cache(maxsize=1)
def ring_edges() -> dict[str, float]:
    """Cached ring-edge table; loaded lazily to avoid import-time I/O."""
    return load_saturn_ring_edges()


def edges_in_frame(rmin: float, rmax: float) -> list[str]:
    """Names of mode-1 ring edges strictly inside the radius range.

    Parameters:
        rmin: Minimum ring radius visible in the frame (km).
        rmax: Maximum ring radius visible in the frame (km).
    """
    margin = 0.02 * (rmax - rmin)
    return [name for name, a in ring_edges().items()
            if rmin + margin <= a <= rmax - margin]


# ---------------------------------------------------------------- star counts

_UCAC4 = None
_YBSC = None
_TYCHO2 = None

# Position match radius for cross-referencing UCAC4 against YBSC/Tycho-2;
# mirrors config_030_stars.yaml duplicate_ra_dec_threshold_arcsec.
_SATURATION_MATCH_RAD = math.radians(5.0 / 3600.0)


def _catalog_positions(catalog: Any, ra: float, ra_half: float, dec_min: float,
                       dec_max: float, vmag_max: float,
                       allow_double: bool) -> list[tuple[float, float, float]]:
    """Collect ``(ra, dec, vmag)`` triples from one catalog in an RA/DEC box.

    Handles RA wrap by splitting the box at 0/2pi when the pointing sits
    near the seam.

    Parameters:
        catalog: A starcat catalog instance exposing ``find_stars``.
        ra: Box-center right ascension in radians.
        ra_half: Half-width of the box in RA (radians, dec-corrected).
        dec_min: Lower declination bound in radians.
        dec_max: Upper declination bound in radians.
        vmag_max: Faint magnitude cutoff for the query.
        allow_double: Whether to include double/multiple-star records
            (required so YBSC returns bright multiples such as Eta Tau).

    Returns:
        ``(ra, dec, vmag)`` triples (radians, radians, magnitude) for every
        record in the box with all three fields present.
    """
    if ra - ra_half < 0 or ra + ra_half > 2 * math.pi:
        boxes = [(0.0, (ra + ra_half) % (2 * math.pi)),
                 ((ra - ra_half) % (2 * math.pi), 2 * math.pi)]
    else:
        boxes = [(ra - ra_half, ra + ra_half)]
    out: list[tuple[float, float, float]] = []
    for ra0, ra1 in boxes:
        for star in catalog.find_stars(ra_min=ra0, ra_max=ra1,
                                       dec_min=dec_min, dec_max=dec_max,
                                       vmag_max=vmag_max,
                                       allow_double=allow_double):
            v = star.vmag
            sra = star.ra
            sdec = star.dec
            if v is not None and sra is not None and sdec is not None:
                out.append((float(sra), float(sdec), float(v)))
    return out


def star_vmags(ra_deg: float, dec_deg: float, fov_deg: float,
               vmag_max: float) -> list[float]:
    """Sorted vmags of stars in a box around the pointing.

    UCAC4 supplies the field; its aperture photometry saturates at the
    bright end (it lists Pleiades members near V7 instead of V3), so the
    magnitudes are corrected against the YBSC/Tycho-2 reference via
    :func:`spindoctor.nav_model.stars.saturation.correct_saturated_vmags`
    before counting, so bright-star screens see true brightness.  Tycho-2
    reaches the V6.5 to V8 stars YBSC misses, so a genuine V7 to V8 star is
    counted at its true brightness rather than mistaken for saturation.

    Parameters:
        ra_deg: Pointing right ascension in degrees.
        dec_deg: Pointing declination in degrees.
        fov_deg: Full field of view in degrees; the search box is 0.75x
            this half-width to allow for pointing error.
        vmag_max: Faint magnitude cutoff for the catalog queries.

    Returns:
        Ascending (brightest first) V-magnitudes with UCAC4 bright-end
        saturation corrected against the YBSC/Tycho-2 reference.
    """
    global _UCAC4, _YBSC, _TYCHO2
    if _UCAC4 is None:
        from starcat import UCAC4StarCatalog
        _UCAC4 = UCAC4StarCatalog(
            os.environ.get('UCAC4_PATH',
                           '/data/external-data/star-catalogs/UCAC4'))
    if _YBSC is None:
        from starcat import YBSCStarCatalog
        _YBSC = YBSCStarCatalog(
            os.environ.get('YBSC_PATH',
                           '/data/external-data/star-catalogs/YBSC'))
    if _TYCHO2 is None:
        from starcat import SpiceStarCatalog
        _TYCHO2 = SpiceStarCatalog('tycho2')
    from spindoctor.nav_model.stars.saturation import correct_saturated_vmags

    half = math.radians(0.75 * fov_deg)      # frame + pointing-error margin
    dec = math.radians(dec_deg)
    ra = math.radians(ra_deg % 360.0)
    dec_min = max(dec - half, -math.pi / 2)
    dec_max = min(dec + half, math.pi / 2)
    cosd = max(math.cos(dec), 0.05)
    ra_half = half / cosd

    ucac4 = _catalog_positions(_UCAC4, ra, ra_half, dec_min, dec_max,
                               vmag_max, allow_double=True)
    ybsc = _catalog_positions(_YBSC, ra, ra_half, dec_min, dec_max,
                              vmag_max, allow_double=True)
    tycho2 = _catalog_positions(_TYCHO2, ra, ra_half, dec_min, dec_max,
                                vmag_max, allow_double=True)
    return correct_saturated_vmags(ucac4, ybsc, tycho2,
                                   match_radius_rad=_SATURATION_MATCH_RAD)


# ---------------------------------------------------------------- helpers

def app_diam_px(target: str, center_res: float | None) -> float | None:
    """Apparent diameter in pixels from the mean radius and resolution.

    Parameters:
        target: Body name (key into the oops-derived radius table).
        center_res: CENTER_RESOLUTION from the summary table (km/px).
    """
    r = RADII.get(target)
    if r is None or center_res is None or center_res <= 0:
        return None
    return 2.0 * r / center_res


def resolved(t: pm.SummaryTable, row: list[str]) -> bool:
    """True when the summary row carries surface lat/lon coverage."""
    return (t.num(row, 'MINIMUM_PLANETOCENTRIC_LATITUDE') is not None
            and t.num(row, 'MINIMUM_IAU_LONGITUDE') is not None)


def phase_bin(phase: float | None) -> str:
    """Coarse phase-angle bucket used as a stratification key."""
    if phase is None:
        return 'unknown'
    for hi, name in ((30, '<30'), (60, '30-60'), (90, '60-90'),
                     (120, '90-120')):
        if phase < hi:
            return name
    return '>120'


def res_decade(res: float | None) -> str:
    """Coarse resolution bucket (km decades) used as a strata key."""
    if res is None:
        return 'unknown'
    if res < 1:
        return '<1km'
    if res < 10:
        return '1-10km'
    return '>10km'


def cand(scene_class: str, volset: str, volume: str, *, filespec: str,
         mission: str, camera: str, dataset: str, strata: tuple,
         selection: dict, needs_visual: bool = False) -> dict:
    """Build one candidate-manifest record.

    Parameters:
        scene_class: Target scene class for the candidate.
        volset: Volume set (e.g. ``'COISS_2xxx'``).
        volume: Volume name (e.g. ``'COISS_2060'``).
        filespec: Label filespec relative to the volume directory.
        mission: Mission key (``'COISS'``/``'VGISS'``/``'GOSSI'``/
            ``'NHLORRI'``).
        camera: Camera name for the frame.
        dataset: sd_offset dataset name.
        strata: Stratification key parts (joined with ``' | '``).
        selection: Machine-readable record of why the frame qualified.
        needs_visual: True when class membership needs eyeball
            confirmation at review time.
    """
    stem = Path(filespec.strip()).stem
    if mission == 'GOSSI':
        img_name = stem
    elif mission == 'NHLORRI':
        img_name = stem.upper()[:14]     # lor_0003103486_0x630_sci
    else:
        img_name = stem.split('_')[0]
    return {
        'scene_class': scene_class,
        'image_name': img_name,
        'filespec': filespec.strip(),
        'volset': volset,
        'volume': volume,
        'mission': mission,
        'camera': camera,
        'dataset': dataset,
        'strata': ' | '.join(str(s) for s in strata),
        'needs_visual': needs_visual,
        'selection': selection,
    }


# ---------------------------------------------------------------- COISS scan

def scan_coiss() -> dict[str, list[dict]]:
    """Cassini ISS scan: body/ring geometry classes plus star pools."""
    out: dict[str, list[dict]] = defaultdict(list)
    star_frame_pool: list[dict] = []      # cheap-pass no-body frames
    body_star_pool: list[dict] = []       # cheap-pass one-body frames
    tiny_pool: list[dict] = []            # negative: tiny body frames

    # COISS_1xxx (Jupiter cruise) feeds the star/negative pools only: the
    # ring-edge catalog in config_310 is Saturn's, so ring classes are
    # restricted to COISS_2xxx.  main_ring_span is the radius window that
    # decides whether visible main rings are in frame.
    volsets = [
        dict(volset='COISS_2xxx', dataset='coiss_saturn',
             planet_kind='saturn_summary', ring_classes=True,
             main_ring_span=(70000.0, 145000.0)),
        dict(volset='COISS_1xxx', dataset='coiss_cruise',
             planet_kind='jupiter_summary', ring_classes=False,
             main_ring_span=(100000.0, 132000.0)),
    ]
    for vs in volsets:
        for volume in pm.volumes(vs['volset']):
            _scan_coiss_volume(vs, volume, out=out,
                               star_frame_pool=star_frame_pool,
                               body_star_pool=body_star_pool,
                               tiny_pool=tiny_pool)
    _coiss_star_passes(out, star_frame_pool=star_frame_pool,
                       body_star_pool=body_star_pool,
                       tiny_pool=tiny_pool)
    return out


def _scan_coiss_volume(vs: dict, volume: str, *,
                       out: dict[str, list[dict]],
                       star_frame_pool: list[dict],
                       body_star_pool: list[dict],
                       tiny_pool: list[dict]) -> None:
    """Scan one COISS volume, appending hits to ``out`` and the pools.

    Parameters:
        vs: Volume-set descriptor (volset, dataset, ring span, flags).
        volume: Volume name within the set.
        out: Per-class candidate lists (mutated).
        star_frame_pool: No-body frames for the star-count passes.
        body_star_pool: Single-body frames for stars_plus_body.
        tiny_pool: All-tiny-body frames for negative candidates.
    """
    volset, dataset = vs['volset'], vs['dataset']
    ring_classes = vs['ring_classes']
    span_lo, span_hi = vs['main_ring_span']

    moon = pm.load_table(volset, volume, 'moon_summary')
    ring = pm.load_table(volset, volume, 'ring_summary')
    sat = pm.load_table(volset, volume, vs['planet_kind'])
    idx = pm.load_table(volset, volume, 'index')
    if not (moon and idx):
        return

    # The index uses .IMG filespecs; the summary tables use .LBL.
    # Join on the extension-stripped path.
    def key(fs: str | None) -> str:
        return (fs or '').rsplit('.', 1)[0]

    moon_rows: dict[str, list[list[str]]] = defaultdict(list)
    for row in moon.rows:
        moon_rows[key(moon.get(row, 'FILE_SPECIFICATION_NAME'))].append(row)
    ring_rows = ({key(ring.get(r, 'FILE_SPECIFICATION_NAME')): r
                  for r in ring.rows} if ring else {})
    sat_rows = ({key(sat.get(r, 'FILE_SPECIFICATION_NAME')): r
                 for r in sat.rows} if sat else {})

    for irow in idx.rows:
        filespec = idx.get(irow, 'FILE_SPECIFICATION_NAME')
        if not filespec:
            continue
        filespec = key(filespec) + '.LBL'
        stem = Path(filespec).stem
        if stem.split('_')[0] in EXISTING_IDS or stem in EXISTING_IDS:
            continue
        camera = 'NAC' if stem.startswith('N') else 'WAC'
        texp_ms = idx.num(irow, 'EXPOSURE_DURATION')
        texp = texp_ms / 1000.0 if texp_ms else None
        filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
        time_full = idx.get(irow, 'IMAGE_TIME') or ''
        if not epoch_covered('COISS', time_full):
            continue
        time = time_full[:4]
        ra = idx.num(irow, 'RIGHT_ASCENSION')
        dec = idx.num(irow, 'DECLINATION')

        mrows = moon_rows.get(key(filespec), [])
        rrow = ring_rows.get(key(filespec))
        srow = sat_rows.get(key(filespec))
        planet_disc = srow is not None and resolved(sat, srow)

        res_moons = []      # (target, diam_px, row)
        all_diams = []
        for mr in mrows:
            tgt = (moon.get(mr, 'TARGET_NAME') or '').strip()
            d = app_diam_px(tgt, moon.num(mr, 'CENTER_RESOLUTION'))
            if d is not None:
                all_diams.append((tgt, d))
            if resolved(moon, mr) and d is not None:
                res_moons.append((tgt, d, mr))

        redges = []
        rmin = rmax = bobs = rres = None
        if rrow is not None:
            rmin = ring.num(rrow, 'MINIMUM_RING_RADIUS')
            rmax = ring.num(rrow, 'MAXIMUM_RING_RADIUS')
            bobs = ring.num(rrow, 'OBSERVER_RING_OPENING_ANGLE')
            rres = ring.num(rrow, 'FINEST_RING_INTERCEPT_RESOLUTION')
            if ring_classes and rmin is not None and rmax is not None:
                redges = edges_in_frame(rmin, rmax)
        # A ring_summary row exists for nearly every frame (the ring
        # PLANE crosses the FOV); visible main rings require the radius
        # range to overlap the main-ring span.
        rings_visible = (rmin is not None and rmax is not None
                         and rmin < span_hi and rmax > span_lo)

        # ---- body_irregular
        for tgt, d, mr in res_moons:
            if tgt in IRREGULARS and 150 <= d <= 800:
                ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
                out['body_irregular'].append(cand(
                    'body_irregular', volset, volume, filespec=filespec, mission='COISS',
                    camera=camera, dataset=dataset, strata=(tgt, phase_bin(ph)),
                    selection={'target': tgt, 'apparent_diameter_px': round(d, 1),
                     'center_phase_deg': ph, 'filter': filt,
                     'year': time}))

        # ---- ring_plus_body (skip near-edge-on rings: their edges collapse
        # toward a line and stop being an independent radial constraint)
        if (redges and res_moons and not planet_disc
                and bobs is not None and abs(bobs) >= 1.0):
            tgt, d, mr = max(res_moons, key=lambda x: x[1])
            if d >= 50:
                ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
                out['ring_plus_body'].append(cand(
                    'ring_plus_body', volset, volume, filespec=filespec, mission='COISS',
                    camera=camera, dataset=dataset,
                    strata=(tgt, phase_bin(ph), round(bobs or 0, -1)),
                    selection={'target': tgt, 'apparent_diameter_px': round(d, 1),
                     'ring_edges_in_frame': redges[:6],
                     'ring_opening_deg': bobs,
                     'center_phase_deg': ph, 'filter': filt,
                     'year': time}))

        # ---- ring_only_flat
        if (redges and not res_moons and not planet_disc
                and bobs is not None and abs(bobs) >= 1.0
                and rres is not None and rres > 0):
            best = None
            for name in redges:
                a = ring_edges()[name]
                r_app = a * abs(math.sin(math.radians(bobs))) / rres
                if best is None or r_app > best[1]:
                    best = (name, r_app)
            assert best is not None
            is_flat = best[1] >= FLAT_MIN_APPARENT_R_PX
            closeup = rres < 0.3
            if is_flat or closeup:
                out['ring_only_flat'].append(cand(
                    'ring_only_flat', volset, volume, filespec=filespec, mission='COISS',
                    camera=camera, dataset=dataset,
                    strata=(round(bobs, -1), res_decade(rres)),
                    selection={'ring_edges_in_frame': redges[:6],
                     'flattest_edge': best[0],
                     'apparent_curvature_radius_px': round(best[1]),
                     'criterion': 'sagitta' if is_flat else 'closeup',
                     'ring_opening_deg': bobs,
                     'finest_ring_intercept_res_km': rres,
                     'filter': filt, 'year': time}))
            # ---- ring_only_curved: clearly curved edge (sagitta > 2 px)
            worst = min(
                (ring_edges()[n] * abs(math.sin(math.radians(bobs))) / rres, n)
                for n in redges)
            if worst[0] <= FRAME_PX * FRAME_PX / 16.0 and rres >= 0.3:
                out['ring_only_curved'].append(cand(
                    'ring_only_curved', volset, volume, filespec=filespec, mission='COISS',
                    camera=camera, dataset=dataset,
                    strata=(round(bobs, -1), res_decade(rres)),
                    selection={'ring_edges_in_frame': redges[:6],
                     'most_curved_edge': worst[1],
                     'apparent_curvature_radius_px': round(worst[0]),
                     'ring_opening_deg': bobs,
                     'finest_ring_intercept_res_km': rres,
                     'filter': filt, 'year': time},
                    needs_visual=True))

        # ---- body geometry classes (single resolved moon)
        if len(res_moons) == 1 and not planet_disc:
            tgt, d, mr = res_moons[0]
            ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
            lat_lo = moon.num(mr, 'MINIMUM_PLANETOCENTRIC_LATITUDE')
            lat_hi = moon.num(mr, 'MAXIMUM_PLANETOCENTRIC_LATITUDE')
            lat_range = (lat_hi - lat_lo
                         if lat_lo is not None and lat_hi is not None
                         else None)
            regular = tgt not in IRREGULARS
            sel = {'target': tgt, 'apparent_diameter_px': round(d, 1),
                   'center_phase_deg': ph,
                   'lat_coverage_deg': (round(lat_range, 1)
                                        if lat_range is not None else None),
                   'filter': filt, 'year': time}
            if (regular and 700 <= d <= 950 and not rings_visible
                    and ph is not None and ph < 110
                    and lat_range is not None and lat_range > 160):
                out['body_full_fov'].append(cand(
                    'body_full_fov', volset, volume, filespec=filespec, mission='COISS',
                    camera=camera, dataset=dataset, strata=(tgt, phase_bin(ph)), selection=sel,
                    needs_visual=True))
            if 1080 <= d <= 1500 and not rings_visible:
                out['body_partial_overflow'].append(cand(
                    'body_partial_overflow', volset, volume, filespec=filespec,
                    mission='COISS', camera=camera, dataset=dataset, strata=(tgt, phase_bin(ph)), selection=sel,
                    needs_visual=True))
            # 1700 <= d <= 2200: body ~2x the frame, so roughly half
            # is off-screen with a limb arc in frame.  Batch-3 votes:
            # the m-voted exemplars sat at d=1840-1960; everything
            # d >= 2365 was rejected as 'extreme closeup filling fov'.
            if (1700 <= d <= 2200 and ph is not None and ph < 90
                    and lat_range is not None and 25 <= lat_range < 120):
                out['body_mostly_offscreen'].append(cand(
                    'body_mostly_offscreen', volset, volume, filespec=filespec,
                    mission='COISS', camera=camera, dataset=dataset, strata=(tgt, phase_bin(ph)), selection=sel,
                    needs_visual=True))
            if (regular and ph is not None and ph >= 95
                    and 100 <= d <= 700 and not rings_visible):
                out['high_phase_terminator'].append(cand(
                    'high_phase_terminator', volset, volume, filespec=filespec,
                    mission='COISS', camera=camera, dataset=dataset, strata=(tgt, phase_bin(ph)), selection=sel,
                    needs_visual=True))

        # ---- multi_body: >=2 separable resolved bodies
        big = [(t, d2, mr2) for t, d2, mr2 in res_moons if d2 >= 20]
        if len(big) >= 2 and not planet_disc:
            names = sorted(t for t, _, _ in big)
            dmax = max(d2 for _, d2, _ in big)
            out['multi_body'].append(cand(
                'multi_body', volset, volume, filespec=filespec, mission='COISS',
                camera=camera, dataset=dataset, strata=('+'.join(names[:3]),),
                selection={'targets_px': {t: round(d2, 1) for t, d2, _ in big},
                 'largest_px': round(dmax, 1),
                 'filter': filt, 'year': time},
                needs_visual=True))

        # ---- below_resolution_body: one 4-14 px body, nothing else
        smalls = [(t, d2) for t, d2 in all_diams if 4 <= d2 <= 14]
        others = [d2 for t, d2 in all_diams if (t, d2) not in smalls]
        if (len(smalls) == 1 and not res_moons and not rings_visible
                and not planet_disc and camera == 'NAC'
                and all(d2 < 3 for d2 in others)):
            out['below_resolution_body'].append(cand(
                'below_resolution_body', volset, volume, filespec=filespec,
                mission='COISS', camera=camera, dataset=dataset, strata=(smalls[0][0],),
                selection={'target': smalls[0][0],
                 'apparent_diameter_px': round(smalls[0][1], 2),
                 'texp_s': texp, 'filter': filt, 'year': time},
                needs_visual=True))

        # ---- pools for star-count classes (catalog query deferred)
        # No exposure floor: bright-pair star-cal frames are typically
        # SHORT exposures (low maglim leaves only the pair detectable).
        no_body = not mrows or all(not resolved(moon, mr) for mr in mrows)
        if (no_body and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and texp > 0
                and camera == 'NAC'):
            star_frame_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time))

        if (len(res_moons) == 1 and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and texp >= 0.5
                and camera == 'NAC'
                and 30 <= res_moons[0][1] <= 700):
            tgt, d, mr = res_moons[0]
            ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
            body_star_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time,
                     target=tgt, diam=d, phase=ph))

        # ---- negative: tiny distant body, no rings
        if (mrows and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and all_diams
                and all(d < 6 for _, d in all_diams)
                and not res_moons and camera == 'NAC'):
            tiny_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time,
                     bodies={t: round(d, 2) for t, d in all_diams}))


def _coiss_star_passes(out: dict[str, list[dict]], *,
                       star_frame_pool: list[dict],
                       body_star_pool: list[dict],
                       tiny_pool: list[dict]) -> None:
    """UCAC4 star-count passes over the pooled COISS frames.

    Parameters:
        out: Per-class candidate lists (mutated).
        star_frame_pool: No-body frames (star classes + negatives).
        body_star_pool: Single-body frames (stars_plus_body).
        tiny_pool: All-tiny-body frames (negatives).
    """
    rng = random.Random(SEED)

    def sample(pool: list[dict], n: int) -> list[dict]:
        pool = sorted(pool, key=lambda c: c['filespec'])
        rng.shuffle(pool)
        return pool[:n]

    for fr in sample(star_frame_pool, len(star_frame_pool)):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim + 2.0)
        n_bright = sum(1 for v in vm if v <= lim)
        sel = {'texp_s': fr['texp'], 'maglim': round(lim, 2),
               'star_vmags': [round(v, 2) for v in vm[:5]],
               'n_detectable': n_bright, 'filter': fr['filt'],
               'year': fr['year']}
        if n_bright == 2 and (len(vm) < 3 or vm[2] >= vm[1] + 1.5):
            out['two_bright_stars_no_body'].append(cand(
                'two_bright_stars_no_body', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=(fr['year'],), selection=sel))
        elif n_bright == 1 and (len(vm) < 2 or vm[1] >= vm[0] + 1.5):
            out['one_bright_star_no_body'].append(cand(
                'one_bright_star_no_body', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=(fr['year'],), selection=sel))
        elif n_bright >= 3:
            out['star_dominated'].append(cand(
                'star_dominated', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=(fr['year'], min(n_bright, 8)), selection=sel))
        elif n_bright == 0 and fr['texp'] <= 3.0 and len(vm) == 0:
            out['negative_cases'].append(cand(
                'negative_cases', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=('COISS', 'empty_sky'),
                selection={'type': 'empty_sky', 'texp_s': fr['texp'],
                 'maglim': round(lim, 2), 'n_stars_within_2mag': len(vm),
                 'filter': fr['filt'], 'year': fr['year']}))

    for fr in sample(body_star_pool, 400):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim)
        if len(vm) >= 3:
            out['stars_plus_body'].append(cand(
                'stars_plus_body', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=(fr['target'], phase_bin(fr['phase'])),
                selection={'target': fr['target'],
                 'apparent_diameter_px': round(fr['diam'], 1),
                 'n_detectable_stars': len(vm),
                 'star_vmags': [round(v, 2) for v in vm[:8]],
                 'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'center_phase_deg': fr['phase'], 'filter': fr['filt'],
                 'year': fr['year']}))

    for fr in sample(tiny_pool, 200):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim)
        if not vm:
            out['negative_cases'].append(cand(
                'negative_cases', fr['volset'], fr['volume'],
                filespec=fr['filespec'], mission='COISS', camera=fr['camera'], dataset=fr['dataset'],
                strata=('COISS', 'tiny_body'),
                selection={'type': 'tiny_body_no_stars', 'bodies_px': fr['bodies'],
                 'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'n_detectable_stars': 0, 'filter': fr['filt'],
                 'year': fr['year']}))



# ---------------------------------------------------------------- NH scan

def scan_nh() -> dict[str, list[dict]]:
    """New Horizons LORRI sky / calibration frames: the star classes.

    Only the calibrated (2001-series) volumes are scanned; frames with
    rows in any body-summary table are excluded, leaving sky and
    star-calibration pointings.  LORRI's deep limiting magnitude
    (11.7 anchor) makes it the best autonomous source for the
    one/two-bright-star and star_dominated classes.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    volset, dataset = 'NHxxLO_xxxx', 'nhlorri'
    cache: dict[tuple[float, float, float], list[float]] = {}

    for volume in pm.volumes(volset):
        if not volume.endswith('_2001'):
            continue
        sup = pm.load_table(volset, volume, 'supplemental_index')
        if not sup:
            continue
        body_specs: set[str] = set()
        for kind in ('moon_summary', 'jupiter_summary', 'pluto_summary',
                     'charon_summary', 'body_summary'):
            t = pm.load_table(volset, volume, kind)
            if t:
                body_specs |= {
                    (t.get(r, 'FILE_SPECIFICATION_NAME') or '')
                    .rsplit('.', 1)[0].strip().lower()
                    for r in t.rows}
        for row in sup.rows:
            filespec = (sup.get(row, 'FILE_SPECIFICATION_NAME')
                        or '').strip()
            if not filespec:
                continue
            img = Path(filespec).stem.upper()[:14]
            if img in EXISTING_IDS:
                continue
            if filespec.rsplit('.', 1)[0].lower() in body_specs:
                continue
            time_full = sup.get(row, 'START_TIME') or ''
            if not epoch_covered('NHLORRI', time_full):
                continue
            texp = sup.num(row, 'EXPOSURE_DURATION')   # seconds
            ra = sup.num(row, 'RIGHT_ASCENSION')
            dec = sup.num(row, 'DECLINATION')
            if texp is None or texp <= 0 or ra is None or dec is None:
                continue
            binning = (sup.get(row, 'BINNING_MODE') or '').strip()
            tgt = (sup.get(row, 'TARGET_NAME') or '').strip()
            lim = maglim('NHLORRI', 'LORRI', texp)
            ckey = (round(ra, 1), round(dec, 1), round(lim, 1))
            if ckey not in cache:
                cache[ckey] = star_vmags(
                    ra, dec, FOV_DEG[('NHLORRI', 'LORRI')], lim + 2.0)
            vm = cache[ckey]
            n_bright = sum(1 for v in vm if v <= lim)
            sel = {'texp_s': texp, 'maglim': round(lim, 2),
                   'star_vmags': [round(v, 2) for v in vm[:5]],
                   'n_detectable': n_bright, 'binning': binning,
                   'target': tgt, 'year': time_full[:4]}
            if n_bright == 2 and (len(vm) < 3 or vm[2] >= vm[1] + 1.5):
                out['two_bright_stars_no_body'].append(cand(
                    'two_bright_stars_no_body', volset, volume, filespec=filespec,
                    mission='NHLORRI', camera='LORRI', dataset=dataset,
                    strata=('NH', time_full[:4]), selection=sel))
            elif n_bright == 1 and (len(vm) < 2 or vm[1] >= vm[0] + 1.5):
                out['one_bright_star_no_body'].append(cand(
                    'one_bright_star_no_body', volset, volume, filespec=filespec,
                    mission='NHLORRI', camera='LORRI', dataset=dataset,
                    strata=('NH', time_full[:4]), selection=sel))
            elif n_bright >= 3:
                out['star_dominated'].append(cand(
                    'star_dominated', volset, volume, filespec=filespec,
                    mission='NHLORRI', camera='LORRI', dataset=dataset,
                    strata=('NH', time_full[:4], min(n_bright, 8)), selection=sel))
    return out


# ---------------------------------------------------------------- GO scan

def scan_go() -> dict[str, list[dict]]:
    """Galileo SSI scan: faint-star, empty-sky, and stray-light classes."""
    out: dict[str, list[dict]] = defaultdict(list)
    volset, dataset = 'GO_0xxx', 'gossi'
    rng = random.Random(SEED + 1)
    pool: list[dict] = []
    stray_pool: list[dict] = []

    for volume in pm.volumes(volset):
        body = pm.load_table(volset, volume, 'body_summary')
        sky = pm.load_table(volset, volume, 'sky_summary')
        ring = pm.load_table(volset, volume, 'ring_summary')
        idx = pm.load_table(volset, volume, 'index')
        if not (idx and sky):
            continue
        inv = pm.load_inventory(volset, volume)

        body_rows: dict[str, list[list[str]]] = defaultdict(list)
        if body:
            for row in body.rows:
                body_rows[body.get(row, 'FILE_SPECIFICATION_NAME')].append(row)
        ring_specs = ({ring.get(r, 'FILE_SPECIFICATION_NAME')
                       for r in ring.rows} if ring else set())
        sky_rows = {sky.get(r, 'FILE_SPECIFICATION_NAME'): r
                    for r in sky.rows}

        for irow in idx.rows:
            filespec = (idx.get(irow, 'FILE_SPECIFICATION_NAME') or '').strip()
            if not filespec:
                continue
            stem = Path(filespec).stem
            if stem in EXISTING_IDS:
                continue
            texp_ms = idx.num(irow, 'EXPOSURE_DURATION')
            texp = texp_ms / 1000.0 if texp_ms else None
            filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
            tgt = (idx.get(irow, 'TARGET_NAME') or '').strip()
            time_full = idx.get(irow, 'IMAGE_TIME') or ''
            if not epoch_covered('GOSSI', time_full):
                continue
            year = time_full[:4]

            brows = body_rows.get(filespec, [])
            res_bodies = [b for b in brows if resolved(body, b)]
            skyrow = sky_rows.get(filespec)

            no_body = not res_bodies
            if (no_body and filespec not in ring_specs and skyrow is not None
                    and texp is not None and texp > 0):
                ra0 = sky.num(skyrow, 'MINIMUM_RIGHT_ASCENSION')
                ra1 = sky.num(skyrow, 'MAXIMUM_RIGHT_ASCENSION')
                de0 = sky.num(skyrow, 'MINIMUM_DECLINATION')
                de1 = sky.num(skyrow, 'MAXIMUM_DECLINATION')
                if None not in (ra0, ra1, de0, de1):
                    pool.append(dict(
                        filespec=filespec, volume=volume, texp=texp,
                        ra=(ra0 + ra1) / 2 if ra0 <= ra1 else ra1,
                        dec=(de0 + de1) / 2, filt=filt, year=year, tgt=tgt))

            # stray-light candidates: Earth/Moon/Venus in inventory but
            # not resolved on disc (bright body near/behind the frame).
            # Batch-2 lesson (operator, 2026-07-09): a gradient alone is
            # useless -- the frame must ALSO contain navigable content,
            # so a pointing (for the star screen) is required here and
            # a detectable-star count is enforced at emit time.
            if (tgt in ('EARTH', 'MOON', 'VENUS')
                    and brows and not res_bodies
                    and texp is not None and texp >= 0.05
                    and skyrow is not None):
                ra0 = sky.num(skyrow, 'MINIMUM_RIGHT_ASCENSION')
                ra1 = sky.num(skyrow, 'MAXIMUM_RIGHT_ASCENSION')
                de0 = sky.num(skyrow, 'MINIMUM_DECLINATION')
                de1 = sky.num(skyrow, 'MAXIMUM_DECLINATION')
                if None not in (ra0, ra1, de0, de1):
                    stray_pool.append(dict(
                        filespec=filespec, volume=volume, texp=texp,
                        ra=(ra0 + ra1) / 2 if ra0 <= ra1 else ra1,
                        dec=(de0 + de1) / 2,
                        filt=filt, year=year, tgt=tgt))

    pool = sorted(pool, key=lambda c: c['filespec'])
    rng.shuffle(pool)
    for fr in pool[:400]:
        lim = maglim('GOSSI', 'SSI', fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('GOSSI', 'SSI')],
                        lim + 2.0)
        n_bright = sum(1 for v in vm if v <= lim - 0.3)
        n_faint = sum(1 for v in vm if lim - 0.3 < v <= lim + 2.0)
        if n_bright == 0 and n_faint >= 3:
            out['faint_stars'].append(cand(
                'faint_stars', volset, fr['volume'], filespec=fr['filespec'],
                mission='GOSSI', camera='SSI', dataset=dataset, strata=(fr['year'],),
                selection={'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'n_bright': 0, 'n_faint_within_2mag': n_faint,
                 'faintest_usable_vmags':
                     [round(v, 2) for v in vm[:6]],
                 'target': fr['tgt'], 'filter': fr['filt'],
                 'year': fr['year']}))
        elif n_bright == 0 and n_faint == 0 and fr['texp'] <= 0.5:
            out['negative_cases'].append(cand(
                'negative_cases', volset, fr['volume'], filespec=fr['filespec'],
                mission='GOSSI', camera='SSI', dataset=dataset, strata=('GOSSI', 'empty_sky'),
                selection={'type': 'empty_sky', 'texp_s': fr['texp'],
                 'maglim': round(lim, 2), 'n_stars_within_2mag': 0,
                 'target': fr['tgt'], 'filter': fr['filt'],
                 'year': fr['year']}))

    stray_pool = sorted(stray_pool, key=lambda c: c['filespec'])
    rng.shuffle(stray_pool)
    n_stray = 0
    for fr in stray_pool:
        if n_stray >= 60:
            break
        lim = maglim('GOSSI', 'SSI', fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('GOSSI', 'SSI')], lim)
        # navigable content requirement: batch-3 votes rejected frames
        # whose catalog stars sat at the detection limit ('just noise'),
        # so demand clearly-bright stars with margin under the glare.
        # The count floor mirrors the star-field matcher's inlier floor:
        # issue #238 showed that ~2 bright stars is not enough for an
        # autonomous solve on a stars-only surrogate (the Galileo
        # C00598xx quintet failed wholesale with all_techniques_spurious),
        # so require at least STAR_FIELD_MIN_INLIERS clearly-bright stars.
        # star_vmags corrects UCAC4's saturated bright-end photometry against
        # the YBSC/Tycho-2 reference (it otherwise lists Pleiades members near
        # V7 instead of V3), so this count reflects true bright content rather
        # than a lower bound.
        n_clear = sum(1 for v in vm if v <= lim - 1.5)
        if n_clear < STAR_FIELD_MIN_INLIERS:
            continue
        n_stray += 1
        out['scattered_light'].append(cand(
            'scattered_light', volset, fr['volume'], filespec=fr['filespec'],
            mission='GOSSI', camera='SSI', dataset=dataset, strata=('GOSSI', fr['year']),
            selection={'surrogate': 'bright body in inventory, unresolved on disc',
             'target': fr['tgt'], 'texp_s': fr['texp'],
             'n_detectable_stars': len(vm), 'maglim': round(lim, 2),
             'star_vmags': [round(v, 2) for v in vm[:6]],
             'filter': fr['filt'], 'year': fr['year']},
            needs_visual=True))

    return out


# ---------------------------------------------------------------- VGISS scan

def scan_vgiss() -> dict[str, list[dict]]:
    """Voyager: scattered-light surrogates (Saturn-encounter frames with
    nothing resolved) and short-exposure empty negatives.  The VGISS index
    has no RA/dec so star-count screens are not possible here."""
    out: dict[str, list[dict]] = defaultdict(list)
    rng = random.Random(SEED + 2)
    stray_pool: list[dict] = []
    neg_pool: list[dict] = []

    for volset in ('VGISS_6xxx',):
        for volume in pm.volumes(volset):
            moon = pm.load_table(volset, volume, 'moon_summary')
            sat = pm.load_table(volset, volume, 'saturn_summary')
            ring = pm.load_table(volset, volume, 'ring_summary')
            idx = pm.load_table(volset, volume, 'index')
            if not idx:
                continue

            moon_specs_resolved = set()
            if moon:
                for row in moon.rows:
                    if resolved(moon, row):
                        moon_specs_resolved.add(
                            moon.get(row, 'FILE_SPECIFICATION_NAME'))
            sat_specs_resolved = set()
            if sat:
                for row in sat.rows:
                    if resolved(sat, row):
                        sat_specs_resolved.add(
                            sat.get(row, 'FILE_SPECIFICATION_NAME'))
            ring_specs = ({ring.get(r, 'FILE_SPECIFICATION_NAME')
                           for r in ring.rows} if ring else set())

            for irow in idx.rows:
                product = (idx.get(irow, 'PRODUCT_TYPE') or '').strip()
                if product != 'CALIBRATED_IMAGE':
                    continue
                filespec = (idx.get(irow, 'FILE_SPECIFICATION_NAME')
                            or '').strip()
                # geometry tables reference the _RAW variant
                rawspec = filespec.replace('_CALIB', '_RAW')
                stem = Path(filespec).stem.split('_')[0]
                if stem in EXISTING_IDS:
                    continue
                texp_s = idx.num(irow, 'EXPOSURE_DURATION')
                filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
                tgt = (idx.get(irow, 'TARGET_NAME') or '').strip()
                time_full = idx.get(irow, 'IMAGE_TIME') or ''
                if not epoch_covered('VGISS', time_full):
                    continue
                year = time_full[:4]
                inst = (idx.get(irow, 'INSTRUMENT_NAME') or '')
                camera = 'WA' if 'WIDE' in inst else 'NA'

                nothing_resolved = (rawspec not in moon_specs_resolved
                                    and rawspec not in sat_specs_resolved
                                    and rawspec not in ring_specs)
                if texp_s is None:
                    continue
                rec = dict(filespec=filespec, volset=volset, volume=volume,
                           texp=texp_s, filt=filt, year=year, tgt=tgt,
                           camera=camera)
                # Batch-2 lesson: scattered_light frames must contain
                # navigable content; VGISS has no pointing columns for a
                # star screen, so require resolved rings or a resolved
                # limb alongside the (prescan-verified) glare.
                if (tgt == 'SATURN' and texp_s >= 1.0
                        and (rawspec in ring_specs
                             or rawspec in moon_specs_resolved
                             or rawspec in sat_specs_resolved)):
                    stray_pool.append(rec)
                if (nothing_resolved and texp_s <= 0.5
                        and tgt in ('DARK', 'SKY', 'STAR', 'SATURN')):
                    neg_pool.append(rec)

    stray_pool = sorted(stray_pool, key=lambda c: c['filespec'])
    rng.shuffle(stray_pool)
    for fr in stray_pool[:60]:
        out['scattered_light'].append(cand(
            'scattered_light', fr['volset'], fr['volume'], filespec=fr['filespec'],
            mission='VGISS', camera=fr['camera'], dataset='vgiss', strata=('VGISS', fr['year']),
            selection={'surrogate': 'Saturn-target frame, nothing resolved in FOV',
             'target': fr['tgt'], 'texp_s': fr['texp'],
             'filter': fr['filt'], 'year': fr['year']},
            needs_visual=True))

    neg_pool = sorted(neg_pool, key=lambda c: c['filespec'])
    rng.shuffle(neg_pool)
    for fr in neg_pool[:10]:
        out['negative_cases'].append(cand(
            'negative_cases', fr['volset'], fr['volume'], filespec=fr['filespec'],
            mission='VGISS', camera=fr['camera'], dataset='vgiss', strata=('VGISS', 'empty_short_exp'),
            selection={'type': 'empty_short_exposure', 'target': fr['tgt'],
             'texp_s': fr['texp'], 'filter': fr['filt'],
             'year': fr['year']},
            needs_visual=True))

    return out


# ---------------------------------------------------------------- sampling

QUOTAS_BY_BATCH: dict[int, dict[str, int]] = {
    # batch 1: the eight scene classes the library then lacked entirely
    # (first-stage budget, plans/COHORT_CURATION_PLAN.md appendix)
    1: {
        'body_irregular': 20,
        'ring_only_flat': 20,
        'ring_plus_body': 20,
        'stars_plus_body': 20,
        'two_bright_stars_no_body': 12,
        'faint_stars': 12,
        'scattered_light': 16,
        'negative_cases': 16,
    },
    # batch 2: every class still below its Part-10 minimum after the
    # batch-1 votes (2026-07-09), plus refreshed pools for the classes
    # that produced no usable exemplars (scattered_light, faint_stars,
    # two_bright) and the NH LORRI star classes.
    2: {
        'body_full_fov': 8,
        'body_partial_overflow': 8,
        'body_mostly_offscreen': 6,
        'multi_body': 8,
        'high_phase_terminator': 6,
        'below_resolution_body': 8,
        'ring_only_curved': 6,
        'star_dominated': 8,
        'one_bright_star_no_body': 8,
        'two_bright_stars_no_body': 10,
        'faint_stars': 10,
        'scattered_light': 12,
    },
    # batch 3: classes still empty after batch-2 votes; scattered_light
    # re-targeted with the navigable-content requirement
    3: {
        'scattered_light': 14,
        'body_mostly_offscreen': 8,
    },
    # batch 4: same two classes with batch-3 vote-driven refinements
    # (diameter cap + lat floor for offscreen; bright-star margin for
    # GOSSI scattered)
    4: {
        'scattered_light': 10,
        'body_mostly_offscreen': 8,
    },
}


def stratified_sample(cands: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin missions first, then strata within each mission.

    A flat strata round-robin starves low-volume missions: with tens of
    thousands of COISS hits spread over dozens of year strata, a small
    quota fills before the sorted key order ever reaches the NH or GO
    strata.  Balancing missions first guarantees each mission with hits
    contributes before any mission gets a second pick.
    """
    by_mission: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for c in cands:
        by_mission[c['mission']][c['strata']].append(c)
    for strata in by_mission.values():
        for group in strata.values():
            group.sort(key=lambda c: c['filespec'])
            rng.shuffle(group)
    mission_keys = sorted(by_mission)
    key_cycle = {m: sorted(by_mission[m]) for m in mission_keys}
    picked: list[dict] = []
    while len(picked) < n:
        progress = False
        for m in mission_keys:
            strata = by_mission[m]
            for k in key_cycle[m]:
                if strata[k]:
                    picked.append(strata[k].pop())
                    progress = True
                    # rotate this mission's strata so its next pick
                    # comes from a different stratum
                    key_cycle[m] = key_cycle[m][1:] + key_cycle[m][:1]
                    break
            if len(picked) >= n:
                break
        if not progress:
            break
    return picked


def main() -> None:
    """Run every mission scan and write the batch candidate manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=2,
                    help='batch number (selects quotas and output name)')
    args = ap.parse_args()
    if args.batch not in QUOTAS_BY_BATCH:
        ap.error(f'unsupported --batch {args.batch}; supported batches: '
                 f'{sorted(QUOTAS_BY_BATCH)}')
    quotas = QUOTAS_BY_BATCH[args.batch]

    global EXISTING_IDS
    EXISTING_IDS = load_existing_ids(args.batch)
    print(f'excluding {len(EXISTING_IDS)} library/prior-batch ids')

    results: dict[str, list[dict]] = defaultdict(list)
    for scan in (scan_coiss, scan_go, scan_vgiss, scan_nh):
        for cls, cands in scan().items():
            if cls in quotas:
                results[cls].extend(cands)

    # scattered_light: geometry cannot predict a visible gradient
    # (batch-1 lesson: 0/10 survived review), so score the actual
    # pixels and keep only frames with a strong low-order gradient.
    if args.batch >= 2 and results.get('scattered_light'):
        from scatter_prescan import prescan
        results['scattered_light'] = prescan(
            results['scattered_light'],
            keep=quotas.get('scattered_light', 12) * 2)

    # a frame can satisfy only one class: scarcest class first
    priority = sorted(results, key=lambda c: len(results[c]))
    seen: set[str] = set()
    for cls in priority:
        keep = []
        for c in results.get(cls, []):
            if c['filespec'] not in seen:
                seen.add(c['filespec'])
                keep.append(c)
        results[cls] = keep

    rng = random.Random(SEED + 3)
    manifest: dict = {
        'query': f'stage_a_batch{args.batch:03d}',
        'seed': SEED,
        'generated': datetime.date.today().isoformat(),
        'notes': 'Stage A candidates (plans/COHORT_CURATION_PLAN.md). '
                 'Selection criteria recorded per candidate under '
                 '"selection".',
        'classes': {},
    }
    counts_lines = []
    for cls in sorted(quotas):
        cands = results.get(cls, [])
        sample = stratified_sample(cands, quotas[cls], rng)
        manifest['classes'][cls] = sample
        counts_lines.append(f'{cls}: {len(cands)} hits -> {len(sample)} sampled')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'candidates_batch{args.batch:03d}.yaml'
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    counts = '\n'.join(counts_lines)
    (OUT_DIR / f'scan_counts_batch{args.batch:03d}.txt').write_text(counts + '\n')
    print(counts)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
