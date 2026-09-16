"""Supplementary star-visibility PNGs for review batches (Stage C).

The pipeline summary PNG uses a display stretch dominated by any bright
body in the frame, which crushes catalog stars into black even when they
are present in the data.  For the star-bearing scene classes this script
renders an additional PNG per image with:

- a hard asinh stretch (median/MAD based) that makes faint point
  sources visible, and
- a circle at each predicted catalog-star position from the navigation
  feature inventory (shifted by the proposed offset when one exists),
  labeled with the feature reliability.

Output: NNN_<image_name>_stars.png next to the originals in the batch
directory.  Votes and the original PNGs are untouched.

Run:  venv/bin/python util/cohort_curation/star_check_pngs.py --batch 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from vicar import VicarImage

from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'

# scattered_light is included for the hard stretch alone: the summary
# PNG's linear scale hides the measured low-order gradient.
STAR_CLASSES = {'stars_plus_body', 'two_bright_stars_no_body',
                'faint_stars', 'one_bright_star_no_body',
                'star_dominated', 'scattered_light'}

# extfov_margin_vu per instrument (src/spindoctor/config_files/
# config_4N0_inst_*.yaml), keyed by image size where it varies.
MARGINS = {
    ('COISS', 'NAC'): {1024: (50, 140), 512: (25, 50), 256: (13, 25)},
    ('COISS', 'WAC'): {1024: (5, 10), 512: (5, 10), 256: (5, 10)},
    ('VGISS', 'NA'): {1000: (400, 400)},
    ('VGISS', 'WA'): {1000: (400, 400)},
    ('GOSSI', 'SSI'): None,   # size-independent
    ('NHLORRI', 'LORRI'): {256: (15, 15), 512: (30, 30),
                           1024: (60, 60)},
}
MARGIN_FLAT = {('GOSSI', 'SSI'): (350, 350)}

# Stretch and overlay tuning.
MAD_TO_SIGMA = 1.4826            # normal-consistency factor for the MAD
STRETCH_SOFTNESS_SIGMA = 3.0     # asinh knee, in robust sigmas
CLIP_PERCENTILE = 99.9           # clip the brightest tail before scaling
U8_MAX = 255.0
CIRCLE_RADIUS_PX = 14


def margin_for(mission: str, camera: str, size: int) -> tuple[int, int]:
    """Extended-FOV margin (v, u) for one image.

    Parameters:
        mission: Mission key ('COISS', 'VGISS', 'GOSSI').
        camera: Camera name as recorded in the triage report.
        size: Image height in pixels (selects the size-dependent row).
    """
    if (mission, camera) in MARGIN_FLAT:
        return MARGIN_FLAT[(mission, camera)]
    table = MARGINS.get((mission, camera))
    if not table:
        return (0, 0)
    if size in table:
        return table[size]
    nearest = min(table, key=lambda s: abs(s - size))
    return table[nearest]


def stretch(data: np.ndarray) -> np.ndarray:
    """Median/MAD asinh stretch that keeps faint point sources visible.

    Parameters:
        data: 2-D image array in any physical units.
    """
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    sigma = MAD_TO_SIGMA * mad if mad > 0 else float(data.std()) or 1.0
    z = np.arcsinh((data - med) / (STRETCH_SOFTNESS_SIGMA * sigma))
    z = np.clip(z, 0.0, np.percentile(z, CLIP_PERCENTILE))
    top = z.max() or 1.0
    return (z / top * U8_MAX).astype(np.uint8)


def render_one(rec: dict, entry: dict, batch_dir: Path) -> str | None:
    """Render the star-check PNG for one review-batch entry.

    Parameters:
        rec: Triage-report record (metadata_path, mission, camera).
        entry: votes.yaml entry (seq, image_name, png).
        batch_dir: Review-batch directory receiving the output PNG.
    """
    mp = rec.get('metadata_path')
    if not mp:
        return None
    try:
        meta = json.loads(Path(mp).read_text())
    except FileNotFoundError:
        return None
    image_path = (meta.get('observation') or {}).get('image_path')
    if not image_path:
        return None
    nav = meta.get('navigation_result') or {}
    stars = [f for f in (nav.get('feature_inventory') or [])
             if f.get('feature_type') == 'STAR']

    # strict=False: Galileo SSI labels carry keywords longer than the
    # 32-character VICAR limit (UNEVEN_BIT_WEIGHT_CORRECTION_FLAG).
    # NH LORRI images are FITS, not VICAR; fall back to astropy.
    try:
        data = VicarImage.from_file(
            image_path, strict=False).data_2d.astype(np.float64)
    except Exception:
        try:
            from astropy.io import fits
            with fits.open(image_path) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
            if data.ndim == 3:
                data = data[0]
        except Exception:
            return None
    img = Image.fromarray(stretch(data)).convert('RGB')
    draw = ImageDraw.Draw(img)

    dv_du = nav.get('offset_px')
    mv, mu = margin_for(rec['mission'], rec['camera'], data.shape[0])
    color = (0, 255, 0) if dv_du else (255, 220, 0)
    r = CIRCLE_RADIUS_PX
    for f in stars:
        # The bbox is a half-open slice range, so v0 and v1 name the two
        # outer boundaries of the pixels it covers and their midpoint is a
        # pixel-corner position: a box covering rows 10 to 14 is [10, 15) and
        # its midpoint 12.5 is the centre of row 12.
        v0, u0, v1, u1 = f['bbox_extfov_vu']
        v = (v0 + v1) / 2.0 - mv
        u = (u0 + u1) / 2.0 - mu
        if dv_du:
            v += dv_du[0]
            u += dv_du[1]
        if not (0 <= v < data.shape[0] and 0 <= u < data.shape[1]):
            continue
        # Pillow addresses whole cells, so it reads a whole number as the
        # centre of a pixel; the half pixel comes off where the position
        # crosses into it.
        vc = v - PIXEL_CENTER_TO_CORNER_PX
        uc = u - PIXEL_CENTER_TO_CORNER_PX
        draw.ellipse([uc - r, vc - r, uc + r, vc + r], outline=color, width=2)
        rel = f.get('reliability')
        label = f'{float(rel):.2f}' if rel is not None else '?'
        draw.text((uc + r + 3, vc - 7), label, fill=color)
    if not stars:
        note = ('NO star features in nav metadata (stars gated or '
                'navigation errored); inspect for star dots manually')
    elif dv_du:
        note = 'circles at predicted star positions + proposed offset'
    else:
        note = ('circles at PREDICTED star positions (no offset; actual '
                'stars sit nearby, shifted by the true offset)')
    draw.text((8, 8), f'#{entry["seq"]:03d} {entry["image_name"]} '
                      f'hard stretch; {note}', fill=(255, 80, 80))

    out_name = f'{entry["seq"]:03d}_{entry["image_name"]}_stars.png'
    img.save(batch_dir / out_name)
    return out_name


def main() -> None:
    """Render star-check PNGs for every star-class entry of one batch.

    Loads the batch votes.yaml and triage report, filters entries whose
    scene class is in STAR_CLASSES, and writes NNN_<name>_stars.png
    next to the review PNGs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, required=True)
    args = ap.parse_args()

    batch_dir = REPO / '_work/cohort_review' / f'batch_{args.batch:03d}'
    votes = yaml.safe_load((batch_dir / 'votes.yaml').read_text())
    report_name = ('triage_report.yaml' if args.batch == 1
                   else f'triage_report_batch{args.batch:03d}.yaml')
    report = yaml.safe_load((OUT_DIR / report_name).read_text())
    byname = {r['image_name']: r for r in report['results']}

    n = 0
    for entry in votes['images']:
        if entry['scene_class'] not in STAR_CLASSES:
            continue
        rec = byname.get(entry['image_name'])
        if not rec:
            continue
        out = render_one(rec, entry, batch_dir)
        if out:
            print(f'wrote {out}')
            n += 1
    print(f'{n} star-check PNGs in {batch_dir}')


if __name__ == '__main__':
    main()
