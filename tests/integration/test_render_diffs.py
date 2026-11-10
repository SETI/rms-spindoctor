"""Currency guard for the committed render-diff review artifacts.

``tests/integration/render_diffs/`` is a standing review artifact (see
``tests/integration/render_contact_sheet.py``): one committed
``current/<scene>.png`` per catalog scene and one ``sheet_<class>.png`` per
scene class.  This module enforces the regeneration rule mechanically, the
way ``test_sim_doc_images`` guards the documentation galleries:

* Every catalog scene's committed ``current/`` PNG, decoded, must equal a fresh
  render stretched through the generator's own path, pixel for pixel, and no
  ``current/`` PNG may outlive its scene.  The decoded pixels are compared, never
  the file bytes, so a change in the PNG encoder does not fail the check.
* Every scene class present in the catalog must have a committed sheet, and
  no sheet may outlive its class.

Noisy scenes are compared exactly too.  The detector draws each pixel's noise from
that pixel's own random values, so on the same NumPy a render reproduces exactly,
apart from a chance of about 1e-13 per pixel.  When NumPy changes a random stream,
the noise moves, and so do the star counts, the star positions, the cosmic rays,
and the hot pixels.  That fails this check and the simulator baselines alike, and
both are regenerated then, as they are after a deliberate change to what a scene
renders.

The sheets are checked for presence and coverage only, never for pixel
identity: a sheet's pixels legitimately encode the *previous* baseline's
before-panels (they show the last reviewed transition, which a fresh
regeneration would erase), and its text labels rasterize through PIL's
``ImageDraw`` default font, whose output is environment-dependent.  The
per-scene ``current/`` PNGs carry none of that -- they are pure stretched
renders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spindoctor.sim.png_export import stretch_to_uint8
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene, scene_class_for_path
from spindoctor.support.types import NDArrayUint8Type
from tests.integration.render_contact_sheet import encode_current_png

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_HERE = Path(__file__).parent
_SCENES_ROOT = _HERE / 'sim_scenes'
_RENDER_DIFFS_DIR = _HERE / 'render_diffs'
_CURRENT_DIR = _RENDER_DIFFS_DIR / 'current'
_REGEN_COMMAND = 'python -m tests.integration.render_contact_sheet'


def _scene_paths() -> list[Path]:
    """Every catalog scene path (the enforcement domain)."""
    return iter_scene_paths(_SCENES_ROOT)


def _decode_png(path: Path) -> NDArrayUint8Type:
    """The grayscale pixels of a committed PNG."""
    with Image.open(path) as image:
        return np.asarray(image.convert('L'), dtype=np.uint8)


def _mismatch(committed_path: Path, fresh: NDArrayUint8Type) -> str | None:
    """How a committed render differs from a fresh one, or None when they are equal.

    Parameters:
        committed_path: The committed ``current/`` PNG, which may be missing.
        fresh: The fresh render, stretched to display levels.

    Returns:
        None when the decoded PNG equals the fresh render pixel for pixel.  Otherwise
        a short description: that the PNG is missing, that the dimensions differ, or
        how many pixels differ and by how many display levels at most.
    """
    if not committed_path.is_file():
        return 'no committed PNG'
    committed = _decode_png(committed_path)
    if committed.shape != fresh.shape:
        return f'committed {committed.shape}, fresh {fresh.shape}'
    diff = np.abs(committed.astype(np.int16) - fresh.astype(np.int16))
    differing = int(np.count_nonzero(diff))
    if differing == 0:
        return None
    return f'{differing} of {diff.size} pixels differ, by up to {int(diff.max())} levels'


def test_every_scene_has_a_current_render(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every catalog scene has a committed ``current/`` PNG equal to a fresh render.

    A missing or stale PNG means a scene (or a rendering change) landed
    without regenerating the review artifacts; rerun the generator and
    review the sheet before committing.  Each stale scene's fresh render is
    written to a ``sim_render_diffs`` temporary directory, which CI uploads
    when the test fails.
    """
    fresh_dir = tmp_path_factory.mktemp('sim_render_diffs')
    stale: list[str] = []
    examined = 0
    for path in _scene_paths():
        examined += 1
        img, _ = render_combined_model(load_sim_scene(path))
        fresh = stretch_to_uint8(img)
        mismatch = _mismatch(_CURRENT_DIR / f'{path.stem}.png', fresh)
        if mismatch is not None:
            stale.append(f'{path.stem}: {mismatch}')
            (fresh_dir / f'{path.stem}.png').write_bytes(encode_current_png(fresh))
    assert examined > 0, 'no scenes were examined; the catalog is empty or unreachable'
    assert not stale, (
        f'{len(stale)} committed current render(s) missing or no longer matching a '
        f'fresh render: {stale}; the fresh renders are in {fresh_dir}.  Regenerate '
        f'with `{_REGEN_COMMAND}` and review the sheets before committing'
    )


def test_the_scene_catalog_is_not_empty() -> None:
    """The catalog these checks iterate actually contains scenes.

    Every check here reports what it found wrong while walking the catalog, so
    a catalog that yields nothing satisfies several of them without examining
    anything.  A moved or renamed scene root would otherwise turn the render
    guards off silently rather than failing.
    """
    assert _scene_paths() != []


def test_no_orphaned_current_renders() -> None:
    """No committed ``current/`` PNG has lost its catalog scene."""
    scene_names = {path.stem for path in _scene_paths()}
    orphans = sorted(png.name for png in _CURRENT_DIR.glob('*.png') if png.stem not in scene_names)
    assert not orphans, (
        f'committed current render(s) with no catalog scene: {orphans}; delete them '
        f'(or restore their scenes) and regenerate with `{_REGEN_COMMAND}`'
    )


def test_every_scene_class_has_a_sheet() -> None:
    """Every scene class in the catalog has a committed contact sheet.

    Sheet presence and class coverage are the enforced invariants; sheet
    bytes are not compared (see the module docstring for why byte identity
    is not meaningful for sheets).
    """
    classes = {scene_class_for_path(path) for path in _scene_paths()}
    assert classes, 'no scene classes were found; the catalog is empty or unreachable'
    missing = sorted(
        scene_class
        for scene_class in classes
        if not (_RENDER_DIFFS_DIR / f'sheet_{scene_class}.png').is_file()
    )
    assert not missing, (
        f'scene class(es) with no committed contact sheet: {missing}; regenerate '
        f'with `{_REGEN_COMMAND}`'
    )


def test_no_orphaned_sheets() -> None:
    """No committed sheet has lost its scene class."""
    classes = {scene_class_for_path(path) for path in _scene_paths()}
    orphans = sorted(
        sheet.name
        for sheet in _RENDER_DIFFS_DIR.glob('sheet_*.png')
        if sheet.stem.removeprefix('sheet_') not in classes
    )
    assert not orphans, (
        f'committed sheet(s) with no catalog scene class: {orphans}; delete them '
        f'(or restore their scenes) and regenerate with `{_REGEN_COMMAND}`'
    )
