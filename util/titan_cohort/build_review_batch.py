"""Build the operator overlay-review batch for the Titan cohort.

Picks a stratified sample of navigated cohort frames -- spread across filter
combinations and phase bins, which are the two axes the method's
filter-independence and working-phase-range claims rest on -- and writes, for
each, a downscaled summary PNG with the frame's identity, offset, confidence,
and gate outcome burned into a margin, plus a manifest CSV and a pre-filled
``votes.yaml``.

The operator's job per frame: look at whether the drawn envelope circle,
symmetry axis, and center cross land on the haze limb, and vote.  Nothing in
this repo fabricates those votes.

Run after a cohort collection pass::

    python util/titan_cohort/build_review_batch.py \\
        _work/titan_cohort/run5/rows.jsonl --campaign-dir _work/titan_cohort/run5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

OUT_DIR = HERE / 'review_batch'
PREVIEW_WIDTH_PX = 640
MARGIN_PX = 74
SELECTION_SEED = 20260726

# Frames seated in every batch regardless of stratum.  N1484962455_2 is the
# one cohort frame where the technique commits with no other technique on
# the frame to check it against, so an operator eye is the only evidence
# available for it.
ALWAYS_INCLUDE: tuple[str, ...] = ('N1484962455_2',)

# Phase bins the sample is spread over.  The lower edge is where the
# axis-degeneracy branch starts to matter and the upper edge is where the
# sunward limb has shrunk to a crescent, so a batch that misses either end
# cannot speak to the working range.
PHASE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 30.0),
    (30.0, 60.0),
    (60.0, 90.0),
    (90.0, 120.0),
    (120.0, 150.0),
    (150.0, 180.0),
)

VOTE_INSTRUCTIONS = (
    'One vote per frame. y = the drawn envelope circle, symmetry axis, and '
    'center cross sit on the haze limb at the stated offset. '
    'm = Titan is navigable in this frame but the overlay is misaligned; '
    'the frame is kept and routed to manual navigation. '
    'n = the frame should not have navigated at all (occulted, clipped, or '
    'defective) or the image is unusable. '
    'The overlay is drawn at the ENSEMBLE offset, so on a frame with other '
    'committed techniques a small misalignment may not be the haze fit; the '
    'manifest names the per-technique haze offset for comparison.'
)


def _load_font(size: int) -> Any:
    """A monospace font at the given size, falling back to PIL's default."""
    for name in ('DejaVuSansMono.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _phase_bin(phase_deg: float) -> tuple[float, float]:
    """The :data:`PHASE_BINS` entry containing ``phase_deg``."""
    for low, high in PHASE_BINS:
        if low <= phase_deg < high:
            return (low, high)
    return PHASE_BINS[-1]


def _titan_entry(row: dict[str, Any]) -> dict[str, Any] | None:
    """The committed haze-technique entry on a row, or None."""
    entry = row.get('techniques', {}).get('TitanHazeNav')
    if entry is None or entry.get('spurious') or entry.get('offset_px') is None:
        return None
    return entry


def select(
    rows: list[dict[str, Any]], *, count: int, include: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Pick ``count`` frames spread over filter combinations and phase bins.

    Round-robins over phase bins first and over filter combinations inside
    each bin, so neither the clear-filter majority nor the mid-phase
    majority can crowd out the ends of either axis: a single flat pass over
    strata sorted by phase would spend the whole batch before reaching the
    high-phase strata.  Draws inside a stratum come from a fixed seed, so
    the batch is reproducible.

    Parameters:
        rows: Cohort rows.
        count: Target batch size.
        include: Image ids to seat unconditionally, ahead of the stratified
            draw.  A frame the campaign could not resolve any other way --
            one that committed with no independent technique to check it --
            is worth a vote regardless of which stratum it lands in, and the
            stratified pass has no way to know that.

    Returns:
        The selected rows, ordered by image id.
    """
    rng = random.Random(SELECTION_SEED)
    strata: dict[tuple[float, float], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        entry = _titan_entry(row)
        if entry is None:
            continue
        phase = float((entry.get('diagnostics') or {}).get('phase_deg', 0.0))
        strata[_phase_bin(phase)][str(row.get('filters'))].append(row)
    for by_filter in strata.values():
        for members in by_filter.values():
            rng.shuffle(members)
    picked: list[dict[str, Any]] = []
    for image_id in include:
        for by_filter in strata.values():
            for members in by_filter.values():
                match = next((m for m in members if m['image_id'] == image_id), None)
                if match is not None:
                    members.remove(match)
                    picked.append(match)
                    break
            else:
                continue
            break
        else:
            print(f'{image_id}: not a committed frame in this run; not seated')
    bins = sorted(strata)
    cursors = dict.fromkeys(bins, 0)
    while len(picked) < count:
        added = False
        for bin_key in bins:
            by_filter = strata[bin_key]
            filters = sorted(by_filter)
            if not filters:
                continue
            # Advance around this bin's filters until one still has a frame.
            for _ in range(len(filters)):
                name = filters[cursors[bin_key] % len(filters)]
                cursors[bin_key] += 1
                if by_filter[name]:
                    picked.append(by_filter[name].pop())
                    added = True
                    break
            if len(picked) >= count:
                break
        if not added:
            break
    return sorted(picked, key=lambda r: r['image_id'])


def compose(row: dict[str, Any], png_path: Path, out_path: Path, sequence: int) -> None:
    """Write one review PNG: the summary image plus a burned-in text margin.

    Parameters:
        row: The frame's cohort row.
        png_path: The pipeline's summary PNG for the frame.
        out_path: Destination path.
        sequence: 1-based position in the batch.
    """
    font = _load_font(14)
    font_small = _load_font(12)
    if png_path.is_file():
        image = Image.open(png_path).convert('RGB')
        scale = PREVIEW_WIDTH_PX / image.width
        image = image.resize((PREVIEW_WIDTH_PX, max(1, round(image.height * scale))), Image.LANCZOS)
    else:
        image = Image.new('RGB', (PREVIEW_WIDTH_PX, PREVIEW_WIDTH_PX), (30, 30, 30))
        ImageDraw.Draw(image).text((20, 300), 'NO SUMMARY PNG', fill=(255, 80, 80), font=font)
    canvas = Image.new('RGB', (image.width, image.height + MARGIN_PX), (12, 12, 12))
    canvas.paste(image, (0, MARGIN_PX))
    draw = ImageDraw.Draw(canvas)
    entry = _titan_entry(row) or {}
    diagnostics = entry.get('diagnostics') or {}
    offset = entry.get('offset_px') or [float('nan'), float('nan')]
    draw.text(
        (8, 6),
        f'#{sequence:02d} {row["image_id"]} {row["camera"]} {row["filters"]}',
        fill=(255, 255, 255),
        font=font,
    )
    draw.text(
        (8, 26),
        f'haze offset ({offset[0]:+.2f}, {offset[1]:+.2f}) px  '
        f'conf {entry.get("confidence")}  phase {diagnostics.get("phase_deg")} deg',
        fill=(200, 200, 200),
        font=font_small,
    )
    draw.text(
        (8, 44),
        f'envelope {diagnostics.get("envelope_diameter_px")} px  '
        f'arc resid {diagnostics.get("arc_residual_rms_px")} px  '
        f'flags {";".join(row["flags"])}',
        fill=(180, 180, 180),
        font=font_small,
    )
    draw.text(
        (8, 60),
        f'ensemble {row.get("nav_status")}/{row.get("status_reason")} conf {row.get("confidence")}',
        fill=(160, 160, 160),
        font=font_small,
    )
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    """Select the batch, render its previews, and write manifest + votes."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument(
        '--include',
        default=','.join(ALWAYS_INCLUDE),
        help='comma-separated image ids to seat unconditionally',
    )
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    rows = []
    with args.rows.open() as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get('manifest'):
                rows.append(record)
    picked = select(
        rows,
        count=args.count,
        include=tuple(i.strip() for i in args.include.split(',') if i.strip()),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / 'manifest.csv'
    with manifest_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                'seq',
                'image_id',
                'camera',
                'filters',
                'phase_deg',
                'flags',
                'haze_dv_px',
                'haze_du_px',
                'haze_confidence',
                'envelope_diameter_px',
                'arc_residual_rms_px',
                'ensemble_status',
                'preview_png',
            ]
        )
        for index, row in enumerate(picked, start=1):
            entry = _titan_entry(row) or {}
            diagnostics = entry.get('diagnostics') or {}
            offset = entry.get('offset_px') or ['', '']
            name = f'{index:02d}_{row["image_id"]}.png'
            compose(
                row,
                args.campaign_dir / f'{row["image_id"]}_summary.png',
                args.out_dir / name,
                index,
            )
            writer.writerow(
                [
                    index,
                    row['image_id'],
                    row['camera'],
                    row['filters'],
                    diagnostics.get('phase_deg'),
                    ';'.join(row['flags']),
                    offset[0],
                    offset[1],
                    entry.get('confidence'),
                    diagnostics.get('envelope_diameter_px'),
                    diagnostics.get('arc_residual_rms_px'),
                    row.get('status_reason'),
                    name,
                ]
            )

    votes_path = args.out_dir / 'votes.yaml'
    lines = [
        '# Titan cohort overlay review -- OPERATOR VOTES PENDING.',
        '#',
        f'# {VOTE_INSTRUCTIONS}',
        '#',
        '# Nothing else in this repository writes to this file.',
        'votes:',
    ]
    for index, row in enumerate(picked, start=1):
        lines += [
            f'  - image_id: {row["image_id"]}',
            f'    preview: {index:02d}_{row["image_id"]}.png',
            '    vote: null',
            "    comment: ''",
        ]
    votes_path.write_text('\n'.join(lines) + '\n')
    print(f'{len(picked)} frames -> {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
