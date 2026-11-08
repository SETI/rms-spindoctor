"""Render-diff contact sheet for the simulator scene catalog.

Run as ``python -m tests.integration.render_contact_sheet`` (no holdings or
SPICE needed; everything renders in-process).  For every scene under
``tests/integration/sim_scenes/`` it produces a before / after / amplified
|difference| panel and composes one contact-sheet PNG per scene class under
``tests/integration/render_diffs/``.  The same run rewrites
``render_diffs/current/<scene_name>.png`` with each scene's current render.

The directory is a standing review artifact:

* ``render_diffs/current/`` holds one stretched grayscale PNG per scene: the
  catalog as this commit renders it.  These committed files are the *before*
  images of the next regeneration, so any PR that changes rendered output
  regenerates the sheets and the ``current/`` PNGs together, in that PR.
* ``render_diffs/sheet_<scene_class>.png`` are the review sheets.  Each row
  is one scene: the prior committed render, the current render, and their
  absolute difference amplified by ``DIFF_AMPLIFICATION`` (stated in the
  column header) so a sub-stretch change is visible.

Review criterion for a sheet: the scene still renders what it asks for --
same ingredients, same geometry, same planted truth; differences confined to
discretization and reseeding.  A scene that recovers its planted offset but
looks wrong is a conversion bug, and this review is what catches it.

The generator reads its *before* images from ``render_diffs/current/`` as
committed, so run it once per change; a second run would diff the new render
against itself (restore ``current/`` from git to redo a sheet).  The entry
point refuses to run when ``current/`` has uncommitted changes and no
``--before-dir`` was given, so a self-diff sheet cannot be published by
accident (``--force`` overrides).  Pass ``--before-dir`` to source the
before images from somewhere else (for example, a directory of renders
produced by another checkout).
"""

from __future__ import annotations

import argparse
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spindoctor.sim.png_export import stretch_to_uint8
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene, scene_class_for_path
from spindoctor.support.types import NDArrayUint8Type
from tests.integration.checkout_guard import require_checkout_package

__all__ = ['DIFF_AMPLIFICATION', 'encode_current_png', 'generate', 'main']

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'
_RENDER_DIFFS_DIR = Path(__file__).parent / 'render_diffs'
_CURRENT_DIR = _RENDER_DIFFS_DIR / 'current'

# The |difference| panels multiply the absolute difference of the two
# stretched uint8 renders by this factor (clipped at white), so a reseeded
# noise floor or a one-DN discretization shift is visible on the sheet.
DIFF_AMPLIFICATION = 4

# Sheet layout, in pixels.
_MARGIN = 10
_LABEL_H = 16
_HEADER_H = 34
_TEXT_FILL = 230
_PLACEHOLDER_FILL = 40


@dataclass
class _SceneRow:
    """One scene's three panels plus its row label."""

    name: str
    before: NDArrayUint8Type | None
    after: NDArrayUint8Type
    diff: NDArrayUint8Type | None


def encode_current_png(after: NDArrayUint8Type) -> bytes:
    """The exact PNG byte encoding of a scene's ``current/`` render.

    The single encoding path shared by :func:`generate` (which writes the
    committed files) and the render-diff currency test (which byte-compares
    fresh renders against them), so the two cannot drift apart.

    Parameters:
        after: The stretched uint8 grayscale render.

    Returns:
        The PNG file bytes.
    """
    buffer = io.BytesIO()
    Image.fromarray(after, mode='L').save(buffer, format='PNG')
    return buffer.getvalue()


def _load_before(before_dir: Path, scene_name: str) -> NDArrayUint8Type | None:
    """Load a scene's committed before render, or ``None`` if absent."""
    path = before_dir / f'{scene_name}.png'
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return np.asarray(im.convert('L'), dtype=np.uint8)


def _amplified_diff(
    before: NDArrayUint8Type | None, after: NDArrayUint8Type
) -> NDArrayUint8Type | None:
    """Amplified |before - after|, or ``None`` when the shapes do not match."""
    if before is None or before.shape != after.shape:
        return None
    diff = np.abs(before.astype(np.int32) - after.astype(np.int32)) * DIFF_AMPLIFICATION
    amplified: NDArrayUint8Type = np.clip(diff, 0, 255).astype(np.uint8)
    return amplified


def _placeholder(shape: tuple[int, int], text: str) -> NDArrayUint8Type:
    """A dark panel carrying a short explanatory label."""
    im = Image.new('L', (shape[1], shape[0]), color=_PLACEHOLDER_FILL)
    draw = ImageDraw.Draw(im)
    draw.text((6, shape[0] // 2 - 6), text, fill=_TEXT_FILL)
    return np.asarray(im, dtype=np.uint8)


def _compose_sheet(scene_class: str, rows: list[_SceneRow]) -> Image.Image:
    """Compose one scene class's rows into a labeled contact sheet."""
    cell_v = max(
        max(r.after.shape[0], (r.before.shape[0] if r.before is not None else 0)) for r in rows
    )
    cell_u = max(
        max(r.after.shape[1], (r.before.shape[1] if r.before is not None else 0)) for r in rows
    )
    sheet_w = _MARGIN + 3 * (cell_u + _MARGIN)
    sheet_h = _HEADER_H + len(rows) * (_LABEL_H + cell_v + _MARGIN)
    sheet = Image.new('L', (sheet_w, sheet_h), color=0)
    draw = ImageDraw.Draw(sheet)

    captions = ('before', 'after', f'|diff| x{DIFF_AMPLIFICATION}')
    draw.text((_MARGIN, 4), f'{scene_class}:', fill=_TEXT_FILL)
    for col, caption in enumerate(captions):
        x = _MARGIN + col * (cell_u + _MARGIN) + cell_u // 2 - 4 * len(caption) // 2
        draw.text((x, _HEADER_H - 14), caption, fill=_TEXT_FILL)

    y = _HEADER_H
    for row in rows:
        draw.text((_MARGIN, y + 2), row.name, fill=_TEXT_FILL)
        y += _LABEL_H
        panels = (
            row.before if row.before is not None else _placeholder(row.after.shape, 'no before'),
            row.after,
            row.diff if row.diff is not None else _placeholder(row.after.shape, 'diff n/a'),
        )
        for col, panel in enumerate(panels):
            x = _MARGIN + col * (cell_u + _MARGIN)
            sheet.paste(Image.fromarray(panel, mode='L'), (x, y))
        y += cell_v + _MARGIN
    return sheet


def generate(*, before_dir: Path | None = None) -> list[Path]:
    """Render the catalog, build the sheets, and rewrite ``current/``.

    Parameters:
        before_dir: Directory of before PNGs (one ``<scene_name>.png`` per
            scene); defaults to the committed ``render_diffs/current/``.

    Returns:
        The written paths (sheets first, then the per-scene current PNGs).
    """
    src = _CURRENT_DIR if before_dir is None else before_dir
    rows_by_class: dict[str, list[_SceneRow]] = {}
    for path in iter_scene_paths(_SCENES_ROOT):
        scene = load_sim_scene(path)
        # Read the before image FIRST: when src is current/, the write below
        # replaces it.
        before = _load_before(src, path.stem)
        img, _ = render_combined_model(scene)
        after = stretch_to_uint8(img)
        rows_by_class.setdefault(scene_class_for_path(path), []).append(
            _SceneRow(
                name=path.stem, before=before, after=after, diff=_amplified_diff(before, after)
            )
        )

    written: list[Path] = []
    _CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    for scene_class in sorted(rows_by_class):
        sheet = _compose_sheet(scene_class, rows_by_class[scene_class])
        sheet_path = _RENDER_DIFFS_DIR / f'sheet_{scene_class}.png'
        sheet.save(sheet_path)
        written.append(sheet_path)
    for rows in rows_by_class.values():
        for row in rows:
            out = _CURRENT_DIR / f'{row.name}.png'
            out.write_bytes(encode_current_png(row.after))
            written.append(out)
    return written


def _current_dir_is_dirty() -> bool:
    """True when ``render_diffs/current/`` has uncommitted changes in git."""
    result = subprocess.run(
        ['git', 'status', '--porcelain', str(_CURRENT_DIR)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    """Entry point: regenerate the contact sheets and the current renders."""
    parser = argparse.ArgumentParser(
        description='Render-diff contact sheet for the simulator scene catalog.'
    )
    parser.add_argument(
        '--before-dir',
        type=Path,
        default=None,
        help='directory of before PNGs (default: the committed render_diffs/current/)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='run even when render_diffs/current/ has uncommitted changes',
    )
    args = parser.parse_args(argv)
    if args.before_dir is None and not args.force and _current_dir_is_dirty():
        print(
            'render_diffs/current/ has uncommitted changes: a run now would diff\n'
            'the new renders against an already-regenerated "before" and publish\n'
            'a self-diff sheet.  Commit or restore current/ first (git restore),\n'
            'point --before-dir at the real before renders, or pass --force.'
        )
        return 1
    paths = generate(before_dir=args.before_dir)
    print(f'Wrote {len(paths)} render-diff file(s):')
    for path in paths:
        print(f'  {path.relative_to(Path(__file__).parent.parent.parent)}')
    return 0


if __name__ == '__main__':
    require_checkout_package(__file__)
    raise SystemExit(main())
