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

A scene that adds no random value to each pixel must match pixel for pixel.  A
scene with shot noise, read noise, a per-pixel response pattern, a calibration
residual, or randomly missing pixels is checked statistically, in 16 by 16 blocks of
the difference between the two renders; the comment above ``_BLOCK_PX`` gives the
statistics.  A new noise realization, as when NumPy changes a random stream, should
pass.  So should a frame in whole DN whose dark sky puts most pixels of a block on one display
level, and whose black and white points move by a whole DN between draws.

A change that is large next to the noise of the blocks it touches fails: a body or
ring edge that moves, or a region whose brightness changes across much of a block,
even when the region covers a third of the frame.  A change confined to a few pixels
can pass.  In a star field whose display stretch is set by the sky, a block's limit
is tens of display levels, so a star that disappears or moves passes.  A change that
the stretch takes out passes too, such as brightening a body that sets the white
point.  A change in where sparse artifacts land can fail: hot pixels, cosmic-ray
hits, and randomly missing pixels move a block's mean by a few levels where its
noise is low, so a NumPy change to how such events are placed can fail a scene that
a new noise draw alone would pass.

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
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint8Type
from tests.integration.render_contact_sheet import encode_current_png

pytestmark = [pytest.mark.integration, pytest.mark.sim_integration]

_HERE = Path(__file__).parent
_SCENES_ROOT = _HERE / 'sim_scenes'
_RENDER_DIFFS_DIR = _HERE / 'render_diffs'
_CURRENT_DIR = _RENDER_DIFFS_DIR / 'current'
_REGEN_COMMAND = 'python -m tests.integration.render_contact_sheet'

# The statistical check for a noisy scene works on blocks of _BLOCK_PX by _BLOCK_PX
# pixels of the difference between the two stretched renders.
#
# A block's noise is the median absolute deviation of its difference, scaled to a
# standard deviation (1.4826 for a normal distribution), so the few pixels of a moved
# edge do not inflate it.  In a frame in whole DN whose sky noise is under 1 DN, one DN
# spans tens of display levels and most sky pixels land on the same level in both
# renders, so a sky block's median absolute deviation is often 0.  Each block's
# standard error is therefore floored by a frame-wide one, taken from the differences
# between the residuals (below) of adjacent blocks: their scaled median absolute value
# divided by the square root of 2.  Adjacent blocks share any broad pattern, so a change
# that covers much of the frame does not raise its own limit.
#
# The display stretch takes its black and white points from percentiles of each noisy
# frame.  In a frame in whole DN, a new noise draw or a new bias pedestal can move a
# black or white point by a whole DN against the scene.  That moves every display
# level by an amount that changes linearly with the level, by up to tens of levels.
# The check removes it by fitting a straight line to each block's mean difference
# against the block's mean level.  The level is the average of the two renders, so the
# noise of either render does not tilt the line.  The line is fitted _TREND_PASSES
# times, each time to the blocks within their limits about the previous fit, so the
# blocks of a real change do not pull it; the first pass starts from the median
# difference.
#
# A block fails when its mean difference, less the line, exceeds _BLOCK_SIGMAS standard
# errors plus _STRETCH_ALLOWANCE_LEVELS.  Eight standard errors leave room for noise
# that is heavier-tailed or more correlated than a normal distribution, and the
# allowance covers rounding to whole display levels.
_BLOCK_PX = 16
_BLOCK_SIGMAS = 8.0
_STRETCH_ALLOWANCE_LEVELS = 1.0
_MAD_TO_SIGMA = 1.4826
_TREND_PASSES = 3


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
    committed_f = committed.astype(np.float64)
    fresh_f = fresh.astype(np.float64)
    diff = fresh_f - committed_f
    level = 0.5 * (fresh_f + committed_f)
    size_v, size_u = diff.shape
    grid = (-(-size_v // _BLOCK_PX), -(-size_u // _BLOCK_PX))
    mean_diff = np.empty(grid)
    mean_level = np.empty(grid)
    block_se = np.empty(grid)
    for bv in range(grid[0]):
        for bu in range(grid[1]):
            rows = slice(bv * _BLOCK_PX, (bv + 1) * _BLOCK_PX)
            cols = slice(bu * _BLOCK_PX, (bu + 1) * _BLOCK_PX)
            block = diff[rows, cols]
            mad = float(np.median(np.abs(block - np.median(block))))
            mean_diff[bv, bu] = block.mean()
            mean_level[bv, bu] = level[rows, cols].mean()
            block_se[bv, bu] = _MAD_TO_SIGMA * mad / np.sqrt(block.size)
    trend = np.full(grid, np.median(mean_diff))
    for _ in range(_TREND_PASSES):
        within = np.abs(mean_diff - trend) <= _block_limits(mean_diff - trend, block_se)
        trend = _fitted_line(mean_level, mean_diff, within)
    residual = mean_diff - trend
    return int(np.count_nonzero(np.abs(residual) > _block_limits(residual, block_se)))


def _block_limits(residual: NDArrayFloatType, block_se: NDArrayFloatType) -> NDArrayFloatType:
    """Each block's limit on its residual, with the frame-wide floor on its error.

    The floor is the scaled median absolute difference between the residuals of
    horizontally and vertically adjacent blocks, divided by the square root of 2.
    """
    steps = np.concatenate([np.diff(residual, axis=0).ravel(), np.diff(residual, axis=1).ravel()])
    frame_se = _MAD_TO_SIGMA * float(np.median(np.abs(steps))) / np.sqrt(2.0) if steps.size else 0.0
    limits: NDArrayFloatType = (
        _BLOCK_SIGMAS * np.maximum(block_se, frame_se) + _STRETCH_ALLOWANCE_LEVELS
    )
    return limits


def _fitted_line(
    level: NDArrayFloatType, diff: NDArrayFloatType, within: NDArrayBoolType
) -> NDArrayFloatType:
    """The least-squares line of difference against level over the blocks within.

    With fewer than two such blocks, or no spread in their levels, the line is flat
    at their median difference (0 when there are none).
    """
    x = level[within]
    y = diff[within]
    if x.size < 2 or np.ptp(x) == 0.0:
        return np.full(level.shape, float(np.median(y)) if y.size else 0.0)
    dx = x - x.mean()
    slope = float(np.dot(dx, y - y.mean()) / np.dot(dx, dx))
    line: NDArrayFloatType = y.mean() + slope * (level - x.mean())
    return line


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


def test_a_new_noise_realization_survives_the_stretch() -> None:
    """Two draws of a star field, each stretched for display, pass the block check.

    Six stars cover a third of a percent of the frame, so both the black and the
    white point fall in the sky's noise and move with each draw, as they do in a
    sky-dominated catalog scene.
    """
    rng = np.random.default_rng(911)
    signal = np.full((128, 128), 100.0)
    for v, u in ((20, 30), (40, 100), (64, 64), (90, 20), (100, 110), (115, 70)):
        signal[v - 1 : v + 2, u - 1 : u + 2] += 2000.0
    first = stretch_to_uint8(rng.poisson(signal).astype(np.float64))
    second = stretch_to_uint8(rng.poisson(signal).astype(np.float64))
    assert _blocks_beyond_noise(first, second) == 0


def _whole_dn_star_field(noise_seed: int, bias_dn: float) -> NDArrayFloatType:
    """A 512 by 512 frame in whole DN: 40 stars on a dark sky, 30 electrons per DN.

    The stars are the same in every call; ``noise_seed`` seeds the shot and read noise.
    """
    electrons = np.zeros((512, 512))
    for v, u in np.random.default_rng(911).integers(5, 507, (40, 2)):
        electrons[v - 1 : v + 2, u - 1 : u + 2] += 20000.0
    rng = np.random.default_rng(noise_seed)
    dn: NDArrayFloatType = np.rint(
        rng.poisson(electrons) / 30.0 + rng.normal(bias_dn, 0.4, electrons.shape)
    )
    return dn


def test_a_sky_in_whole_dn_survives_the_stretch() -> None:
    """Two draws of a star field in whole DN, each stretched, pass the block check.

    The sky holds a 20 DN bias and 0.4 DN of read noise, so most sky pixels take one
    of three values, one DN spans about 128 display levels, and most pixels of a sky
    block land on the same level in both draws.
    """
    first = stretch_to_uint8(_whole_dn_star_field(1, 20.0))
    second = stretch_to_uint8(_whole_dn_star_field(2, 20.0))
    assert _blocks_beyond_noise(first, second) == 0


def test_a_new_bias_pedestal_survives_the_stretch() -> None:
    """A star field in whole DN whose bias moves by 0.6 DN passes the block check.

    The black and white points fall on whole DN values while the sky moves by 0.6 DN,
    so every sky display level shifts.
    """
    first = stretch_to_uint8(_whole_dn_star_field(1, 20.0))
    second = stretch_to_uint8(_whole_dn_star_field(2, 20.6))
    assert _blocks_beyond_noise(first, second) == 0


def test_a_change_across_a_third_of_the_frame_is_beyond_the_noise() -> None:
    """Brightening the top third of a ramp by 20 percent fails the block check.

    The brightened rows hold the frame's brightest pixels, so the white point moves
    with them and every unchanged block shifts too.
    """
    rng = np.random.default_rng(911)
    ramp = np.tile(np.linspace(0.0, 3000.0, 128), (128, 1))
    brighter = ramp.copy()
    brighter[:43] *= 1.2
    first = stretch_to_uint8(rng.poisson(ramp).astype(np.float64))
    second = stretch_to_uint8(rng.poisson(brighter).astype(np.float64))
    assert _blocks_beyond_noise(first, second) > 0


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
