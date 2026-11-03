"""Currency guard for the committed render-diff review artifacts.

``tests/integration/render_diffs/`` is a standing review artifact (see
``tests/integration/render_contact_sheet.py``): one committed
``current/<scene>.png`` per catalog scene and one ``sheet_<class>.png`` per
scene class.  This module enforces the regeneration rule mechanically, the
way ``test_sim_doc_images`` guards the documentation galleries:

* Every catalog scene's committed ``current/`` PNG must be byte-identical to
  a fresh render encoded through the generator's own PNG path, and no
  ``current/`` PNG may outlive its scene.
* Every scene class present in the catalog must have a committed sheet, and
  no sheet may outlive its class.

The sheets are checked for presence and coverage only, never for byte
identity: a sheet's pixels legitimately encode the *previous* baseline's
before-panels (they show the last reviewed transition, which a fresh
regeneration would erase), and its text labels rasterize through PIL's
``ImageDraw`` default font, whose output is environment-dependent.  The
per-scene ``current/`` PNGs carry none of that -- they are pure encoded
renders -- so byte identity is asserted exactly there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spindoctor.sim.png_export import stretch_to_uint8
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import iter_scene_paths, load_sim_scene, scene_class_for_path
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


def test_every_scene_has_a_current_render() -> None:
    """Every catalog scene has a committed, byte-current ``current/`` PNG.

    A missing or stale PNG means a scene (or a rendering change) landed
    without regenerating the review artifacts; rerun the generator and
    review the sheet before committing.
    """
    stale: list[str] = []
    examined = 0
    for path in _scene_paths():
        examined += 1
        committed = _CURRENT_DIR / f'{path.stem}.png'
        img, _ = render_combined_model(load_sim_scene(path))
        fresh = encode_current_png(stretch_to_uint8(img))
        if not committed.is_file() or committed.read_bytes() != fresh:
            stale.append(path.stem)
    assert examined > 0, 'no scenes were examined; the catalog is empty or unreachable'
    assert not stale, (
        f'{len(stale)} committed current render(s) missing or no longer matching a '
        f'fresh render: {stale}; regenerate with `{_REGEN_COMMAND}` and review the '
        f'sheets before committing'
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
