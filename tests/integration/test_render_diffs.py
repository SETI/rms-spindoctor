"""Currency guard for the committed render-diff review artifacts.

``tests/integration/render_diffs/`` is a standing review artifact (see
``tests/integration/render_contact_sheet.py``): one committed
``current/<scene>.png`` per catalog scene and one ``sheet_<class>.png`` per
scene class.  This module enforces the regeneration rule mechanically, the
way ``test_sim_doc_images`` guards the documentation galleries:

* Every catalog scene's committed ``current/`` PNG, decoded, must match a fresh
  render stretched through the generator's own path, and no ``current/`` PNG may
  outlive its scene.  The decoded pixels are compared, never the file bytes, so a
  change in the PNG encoder does not fail the check.
* Every scene class present in the catalog must have a committed sheet, and
  no sheet may outlive its class.

A scene whose detector adds no random value to each pixel must match pixel for
pixel.  A scene with shot noise, read noise, a per-pixel response pattern, or a
calibration residual is checked statistically: the two renders may differ by a
new noise realization, as they do when NumPy changes a random stream, but in each
16 by 16 block the mean difference must stay within what that block's own noise
explains.  A body or ring edge that moves, or a region whose brightness changes,
shifts the block means and fails.

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
from typing import Any

import numpy as np
import pytest
from PIL import Image

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.detector import resolve_detector_params
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

# The statistical check for a noisy scene compares the mean difference of each
# block of _BLOCK_PX by _BLOCK_PX pixels with that block's own noise.  The noise is
# the median absolute deviation of the difference, scaled to a standard deviation
# (1.4826 for a normal distribution), so the few pixels of a moved edge do not
# inflate it.  A block fails when its mean difference exceeds _BLOCK_SIGMAS
# standard errors of the mean plus _STRETCH_ALLOWANCE_LEVELS.  The allowance
# covers the display stretch: its black and white points are percentiles of the
# noisy frame, so a new noise realization moves every display level by a
# fraction of a level.  Eight standard errors leave room for noise that is
# heavier-tailed or more correlated than a normal distribution.
_BLOCK_PX = 16
_BLOCK_SIGMAS = 8.0
_STRETCH_ALLOWANCE_LEVELS = 1.0
_MAD_TO_SIGMA = 1.4826


def _scene_paths() -> list[Path]:
    """Every catalog scene path (the enforcement domain)."""
    return iter_scene_paths(_SCENES_ROOT)


def _draws_pixel_noise(scene: dict[str, Any]) -> bool:
    """Whether the scene adds a random value to every pixel.

    The CCD chain does so with shot noise, read noise, or a per-pixel response
    pattern; the vidicon chain with its line or pixel read noise; a calibrated
    frame of either with its per-pixel calibration residual; and the telemetry
    stage of any frame with its randomly missing pixels.
    """
    if _missing_data_rate(scene) > 0.0:
        return True
    dp = resolve_detector_params(scene)
    if dp.detector_model == 'vidicon':
        line_dn = float(dp.vidicon.get('read_noise_line_dn', 0.0))
        pixel_dn = float(dp.vidicon.get('read_noise_pixel_dn', 0.0))
        if line_dn > 0.0 or pixel_dn > 0.0:
            return True
    else:
        prnu_rms = float((dp.detector_modes.get('fixed_pattern') or {}).get('prnu_rms') or 0.0)
        if dp.poisson or dp.read_noise_e > 0.0 or prnu_rms > 0.0:
            return True
    return dp.data_units != 'raw_dn' and dp.calibration_pixel_residual_e > 0.0


def _missing_data_rate(scene: dict[str, Any]) -> float:
    """The fraction of pixels the telemetry stage marks missing at random.

    A null or absent scene rate takes the configuration default, as the telemetry
    stage does.
    """
    scene_rate = (scene.get('noise') or {}).get('missing_data_rate')
    if scene_rate is None:
        return float(DEFAULT_CONFIG.category('sim')['noise'].get('missing_data_rate', 0.0))
    return float(scene_rate)


def _decode_png(path: Path) -> NDArrayUint8Type:
    """The grayscale pixels of a committed PNG."""
    with Image.open(path) as image:
        return np.asarray(image.convert('L'), dtype=np.uint8)


def _blocks_beyond_noise(committed: NDArrayUint8Type, fresh: NDArrayUint8Type) -> int:
    """How many blocks differ by more than a new noise realization explains.

    Both arrays are stretched renders with the same dimensions.  See the constants
    above for the test each block must pass.
    """
    diff = fresh.astype(np.float64) - committed.astype(np.float64)
    size_v, size_u = diff.shape
    failing = 0
    for v0 in range(0, size_v, _BLOCK_PX):
        for u0 in range(0, size_u, _BLOCK_PX):
            block = diff[v0 : v0 + _BLOCK_PX, u0 : u0 + _BLOCK_PX]
            sigma = _MAD_TO_SIGMA * float(np.median(np.abs(block - np.median(block))))
            limit = _BLOCK_SIGMAS * sigma / np.sqrt(block.size) + _STRETCH_ALLOWANCE_LEVELS
            if abs(float(block.mean())) > limit:
                failing += 1
    return failing


def _is_current(
    scene: dict[str, Any], committed: NDArrayUint8Type, fresh: NDArrayUint8Type
) -> bool:
    """Whether a committed render matches a fresh one: exactly, or within the noise."""
    if committed.shape != fresh.shape:
        return False
    if not _draws_pixel_noise(scene):
        return bool(np.array_equal(committed, fresh))
    return _blocks_beyond_noise(committed, fresh) == 0


def test_every_scene_has_a_current_render(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every catalog scene has a committed, current ``current/`` PNG.

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
        committed = _CURRENT_DIR / f'{path.stem}.png'
        scene = load_sim_scene(path)
        img, _ = render_combined_model(scene)
        fresh = stretch_to_uint8(img)
        if not committed.is_file() or not _is_current(scene, _decode_png(committed), fresh):
            stale.append(path.stem)
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


def test_a_new_noise_realization_is_within_the_noise() -> None:
    """Two draws of the same noisy image pass the block check."""
    rng = np.random.default_rng(911)
    signal = np.tile(np.linspace(20.0, 230.0, 128), (128, 1))
    first = np.clip(np.rint(signal + rng.normal(0.0, 6.0, signal.shape)), 0, 255)
    second = np.clip(np.rint(signal + rng.normal(0.0, 6.0, signal.shape)), 0, 255)
    assert _blocks_beyond_noise(first.astype(np.uint8), second.astype(np.uint8)) == 0


def test_a_moved_edge_is_beyond_the_noise() -> None:
    """A bright edge moved by two pixels under the same noise fails the block check."""
    rng = np.random.default_rng(911)
    signal = np.full((128, 128), 30.0)
    signal[:, 60:] = 200.0
    moved = np.full((128, 128), 30.0)
    moved[:, 62:] = 200.0
    first = np.clip(np.rint(signal + rng.normal(0.0, 6.0, signal.shape)), 0, 255)
    second = np.clip(np.rint(moved + rng.normal(0.0, 6.0, signal.shape)), 0, 255)
    assert _blocks_beyond_noise(first.astype(np.uint8), second.astype(np.uint8)) > 0
