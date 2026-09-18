"""Annotated-summary-PNG rendering shared by the autonomous and manual paths.

The autonomous pipeline (``spindoctor.navigate_image_files.write_summary_png``)
and the manual-navigation dialog both produce a labeled overlay PNG of
the source image with each NavModel's annotation drawn on top.  The
rendering logic lives here so both code paths produce visually
identical PNGs from the same ``(obs, annotations, offset_px)`` triple.

Two features beyond the raw overlay are provided:

- A metadata text block in the least-crowded corner naming the image, its
  filter and exposure, whether navigation succeeded (and, if so, which
  techniques contributed), and the fused confidence.
- Per-box local contrast stretch: every star detection box carries a
  ``stretch_boxes`` region on its ``Annotation`` whose pixels are stretched
  against that box's own min/max so a faint star stays visible even when
  the rest of the frame is far brighter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.support.image import apply_linear_gamma_stretch
from spindoctor.support.types import NDArrayFloatType, NDArrayUint8Type

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from spindoctor.annotation import Annotations
    from spindoctor.obs import ObsSnapshot

__all__ = [
    'SummaryMetadata',
    'build_summary_metadata_lines',
    'grayscale_to_rgb_with_quantile_stretch',
    'render_annotated_summary_rgb',
]

# Inset of the metadata panel from the chosen image corner, in pixels.
_CORNER_MARGIN = 4


@dataclass(frozen=True)
class SummaryMetadata:
    """The header facts drawn as a text block on a summary PNG.

    Parameters:
        image_name: Basename of the source image.
        filter_name: Human-readable filter string (``''`` when the
            instrument carries no filter).
        exposure_s: Exposure time in seconds, or ``None`` when unknown.
        status: NavResult status (``'success'`` / ``'failed'`` /
            ``'conflicted'``).
        techniques: Technique names that contributed to a successful fuse,
            in display order; empty when navigation did not succeed.
        confidence: Fused confidence in ``[0, 1]``, or ``None`` when
            unavailable.
        confidence_rank: Five-bucket confidence tier string.
    """

    image_name: str
    filter_name: str
    exposure_s: float | None
    status: str
    techniques: tuple[str, ...]
    confidence: float | None
    confidence_rank: str


def build_summary_metadata_lines(metadata: SummaryMetadata) -> list[str]:
    """Assemble the text lines for a summary metadata block.

    Pure text assembly, kept apart from the pixel drawing so it can be
    asserted directly.  Absent fields (no filter, unknown exposure) are
    dropped rather than rendered blank.

    Parameters:
        metadata: The facts to present.

    Returns:
        One string per line, top to bottom.
    """
    lines = [metadata.image_name]
    if metadata.filter_name:
        lines.append(f'Filter: {metadata.filter_name}')
    if metadata.exposure_s is not None:
        lines.append(f'Exposure: {metadata.exposure_s:g} s')
    if metadata.status == 'success':
        techniques = ', '.join(metadata.techniques) if metadata.techniques else 'none'
        lines.append(f'Nav: success [{techniques}]')
    else:
        lines.append(f'Nav: {metadata.status}')
    confidence = 'n/a' if metadata.confidence is None else f'{metadata.confidence:.3f}'
    lines.append(f'Confidence: {confidence} ({metadata.confidence_rank})')
    return lines


def render_annotated_summary_rgb(
    obs: ObsSnapshot,
    annotations: Annotations,
    offset_px: tuple[float, float] = (0.0, 0.0),
    *,
    metadata: SummaryMetadata | None = None,
) -> NDArrayUint8Type:
    """Composite ``obs.data`` with ``annotations.combine`` at ``offset_px``.

    Builds a quantile-stretched grayscale background from the FOV image,
    stretches each annotation's star boxes locally, asks the annotations
    layer for its FOV-shaped RGB overlay at the requested offset, and
    replaces every pixel where the overlay carries any non-zero color
    channel.  When ``metadata`` is supplied a header text block is drawn in
    the least-crowded corner.  When the annotations collection is empty and
    no metadata is given the returned RGB is the source-image grayscale
    alone -- so the result is always a faithful record of what the
    navigator saw.

    Parameters:
        obs: Observation snapshot supplying the background image.
        annotations: Merged ``Annotations`` collection from every
            NavModel that contributed.
        offset_px: ``(dv, du)`` offset that shifts the overlay onto the
            best-fit pose.  The convention matches every other offset
            in the pipeline: predicted + offset = actual.
        metadata: Optional header facts; when given, a text block is drawn
            in the least-crowded corner.

    Returns:
        ``(H, W, 3)`` uint8 RGB array in FOV coordinates.
    """
    image_fov = np.asarray(obs.data, dtype=np.float64)
    rgb = grayscale_to_rgb_with_quantile_stretch(image_fov)
    _apply_local_stretch_boxes(rgb, image_fov, obs, annotations, offset_px)
    # ``combine`` collects the FOV-pixel bounding box of every label it draws so
    # the metadata block can be steered away from the other annotation text.
    text_bboxes: list[tuple[int, int, int, int]] = []
    overlay = annotations.combine(offset=offset_px, text_bboxes=text_bboxes)
    if overlay is not None:
        mask = overlay.any(axis=-1)
        rgb[mask] = overlay[mask]
    if metadata is not None:
        _draw_metadata_block(rgb, build_summary_metadata_lines(metadata), text_bboxes=text_bboxes)
    return rgb


def _apply_local_stretch_boxes(
    rgb: NDArrayUint8Type,
    image_fov: NDArrayFloatType,
    obs: ObsSnapshot,
    annotations: Annotations,
    offset_px: tuple[float, float],
) -> None:
    """Overwrite each annotation's stretch boxes with a locally-stretched patch.

    Parameters:
        rgb: FOV-shaped RGB background, modified in place.
        image_fov: The raw float FOV image the stretch reads from.
        obs: Observation snapshot (for the ext-FOV-to-FOV mapping).
        annotations: Annotation collection carrying ``stretch_boxes``.
        offset_px: ``(dv, du)`` offset applied to the overlay; the boxes
            follow the same shift so they align with the drawn outlines.
    """
    for annotation in annotations.annotations:
        for box in annotation.stretch_boxes:
            data_box = _extfov_box_to_data_slice(obs, box, offset_px)
            if data_box is None:
                continue
            patch = _local_stretch_patch(image_fov, data_box)
            if patch is None:
                continue
            v_lo, u_lo, v_hi, u_hi = data_box
            rgb[v_lo:v_hi, u_lo:u_hi] = patch


def _extfov_box_to_data_slice(
    obs: ObsSnapshot,
    box: tuple[int, int, int, int],
    offset_px: tuple[float, float],
) -> tuple[int, int, int, int] | None:
    """Map an ext-FOV ``(v_min, u_min, v_max, u_max)`` box to a FOV slice.

    Uses the same origin convention as ``ObsSnapshot.extract_offset_array``
    so a box aligns with the overlay drawn at the same offset.  Returns
    ``None`` when the box lands entirely outside the FOV.

    Parameters:
        obs: Observation snapshot supplying the ext-FOV margins and shape.
        box: ``(v_min, u_min, v_max, u_max)`` in ext-FOV coordinates; the
            max bounds are exclusive.
        offset_px: ``(dv, du)`` overlay offset.

    Returns:
        ``(v_lo, u_lo, v_hi, u_hi)`` FOV slice bounds (max exclusive),
        clipped to the FOV; ``None`` when empty.
    """
    v_min, u_min, v_max, u_max = box
    v0 = obs.extfov_margin_v - int(np.round(offset_px[0]))
    u0 = obs.extfov_margin_u - int(np.round(offset_px[1]))
    data_v, data_u = obs.data_shape_vu
    v_lo = max(0, v_min - v0)
    u_lo = max(0, u_min - u0)
    v_hi = min(data_v, v_max - v0)
    u_hi = min(data_u, u_max - u0)
    if v_hi <= v_lo or u_hi <= u_lo:
        return None
    return v_lo, u_lo, v_hi, u_hi


def _local_stretch_patch(
    image_fov: NDArrayFloatType,
    data_box: tuple[int, int, int, int],
) -> NDArrayUint8Type | None:
    """Maximally stretch one FOV box against its own finite min/max.

    Parameters:
        image_fov: The raw float FOV image.
        data_box: ``(v_lo, u_lo, v_hi, u_hi)`` FOV slice (max exclusive).

    Returns:
        An ``(h, w, 3)`` uint8 RGB patch, or ``None`` when the box holds no
        finite pixels or is perfectly flat (nothing to stretch).
    """
    v_lo, u_lo, v_hi, u_hi = data_box
    patch = image_fov[v_lo:v_hi, u_lo:u_hi]
    finite = np.isfinite(patch)
    if not finite.any():
        return None
    values = patch[finite]
    black = float(values.min())
    white = float(values.max())
    if white <= black:
        return None
    clean = np.where(finite, patch, black)
    stretched = apply_linear_gamma_stretch(clean, black=black, white=white, gamma=1.0)
    gray = (stretched * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _draw_metadata_block(
    rgb: NDArrayUint8Type,
    lines: list[str],
    *,
    text_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Draw a metadata text block in the least-crowded corner of ``rgb``.

    The corner is chosen by :func:`_least_crowded_corner`, which prefers a
    corner whose block region overlaps none of the other annotation labels and
    breaks that choice by image brightness.  A translucent dark panel is laid
    behind the text for legibility on any background.

    Parameters:
        rgb: FOV-shaped RGB image, modified in place.
        lines: Text lines to draw, top to bottom.
        text_bboxes: Drawn bounding boxes of the other annotation labels in
            FOV ``(v_min, u_min, v_max, u_max)`` pixel coordinates; the block
            is steered away from these.  ``None`` treats every corner as
            text-free (brightness alone decides).
    """
    if not lines:
        return
    height, width = rgb.shape[:2]
    font_size = max(11, min(width, height) // 55)
    font = _load_summary_font(font_size)

    image = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(image, mode='RGB')
    pad = max(4, font_size // 2)
    line_gap = max(2, font_size // 4)
    # Wrap any line wider than the panel so a long technique list never
    # overflows the frame (and never trips the fit guard on its own).
    max_text_w = width - 2 * _CORNER_MARGIN - 2 * pad
    wrapped = _wrap_lines_to_width(draw, lines, font, max_text_w)

    line_widths: list[int] = []
    line_height = 0
    for line in wrapped:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        line_widths.append(int(right - left))
        line_height = max(line_height, int(bottom - top))
    block_w = max(line_widths) + 2 * pad
    block_h = len(wrapped) * line_height + (len(wrapped) - 1) * line_gap + 2 * pad
    if block_w + 2 * _CORNER_MARGIN > width or block_h + 2 * _CORNER_MARGIN > height:
        # The frame is too small to carry the block without covering it; a
        # thumbnail-sized test image is the usual case.  Skip rather than
        # paint a panel over the whole image.
        return

    x0, y0 = _least_crowded_corner(rgb, block_w, block_h, text_bboxes=text_bboxes)

    panel = np.asarray(image).astype(np.float64)
    panel[y0 : y0 + block_h, x0 : x0 + block_w] *= 0.35
    image = Image.fromarray(panel.astype(np.uint8), mode='RGB')
    draw = ImageDraw.Draw(image, mode='RGB')

    text_y = y0 + pad
    for line in wrapped:
        draw.text((x0 + pad, text_y), line, fill=(255, 255, 0), font=font)
        text_y += line_height + line_gap

    rgb[...] = np.asarray(image)


def _wrap_lines_to_width(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Greedily wrap each line so none exceeds ``max_width`` pixels.

    Wrapping breaks on spaces; a single word wider than ``max_width`` is
    left intact (it will still be clipped by the panel rather than
    overflow the image, which the fit guard tolerates for a degenerate
    case).

    Parameters:
        draw: A drawing context for text measurement.
        lines: Source lines.
        font: Font the lines are measured and drawn in.
        max_width: Maximum text width in pixels.

    Returns:
        The wrapped lines, in order.
    """
    if max_width <= 0:
        return list(lines)
    wrapped: list[str] = []
    for line in lines:
        words = line.split(' ')
        current = ''
        for word in words:
            candidate = word if not current else f'{current} {word}'
            left, _top, right, _bottom = draw.textbbox((0, 0), candidate, font=font)
            if right - left <= max_width or not current:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def _least_crowded_corner(
    rgb: NDArrayUint8Type,
    block_w: int,
    block_h: int,
    *,
    text_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int]:
    """Return the top-left ``(x, y)`` of the corner best suited to the block.

    The metadata block should never land on top of another annotation label if
    that can be avoided, so a corner whose block region overlaps no annotation
    text bounding box is preferred outright.  Among the corners that clear the
    text -- or all four when none do -- the choice falls back to the existing
    image-brightness criterion (the darkest corner, where the fewest image
    features sit).  When every corner conflicts with text the one with the
    least text-overlap area wins, ties broken by brightness.

    Parameters:
        rgb: FOV-shaped RGB image.
        block_w: Block width in pixels.
        block_h: Block height in pixels.
        text_bboxes: Other annotation labels' drawn bounding boxes in FOV
            ``(v_min, u_min, v_max, u_max)`` pixel coordinates.  ``None`` or
            empty makes every corner text-free so brightness alone decides.

    Returns:
        ``(x0, y0)`` pixel origin for the block, inset a few pixels from
        the chosen corner.
    """
    height, width = rgb.shape[:2]
    margin = _CORNER_MARGIN
    luminance = rgb.mean(axis=-1)
    bboxes = text_bboxes or []
    x_left = max(0, min(margin, width - block_w))
    x_right = max(0, width - block_w - margin)
    y_top = max(0, min(margin, height - block_h))
    y_bottom = max(0, height - block_h - margin)
    candidates = [
        (x_left, y_top),
        (x_right, y_top),
        (x_left, y_bottom),
        (x_right, y_bottom),
    ]
    best_xy = candidates[0]
    best_key: tuple[int, float] | None = None
    for x0, y0 in candidates:
        block_rect = (y0, x0, y0 + block_h, x0 + block_w)
        overlap = _text_overlap_area(block_rect, bboxes)
        block_lum = float(luminance[y0 : y0 + block_h, x0 : x0 + block_w].mean())
        # Sort key: prefer zero overlap (never sit on a label), then the
        # darkest corner.  Both terms ascending, so ``min`` picks a text-free
        # corner first and the darkest of those to break the tie.
        key = (overlap, block_lum)
        if best_key is None or key < best_key:
            best_key = key
            best_xy = (x0, y0)
    return best_xy


def _text_overlap_area(
    block_rect: tuple[int, int, int, int],
    text_bboxes: list[tuple[int, int, int, int]],
) -> int:
    """Total overlap area (px^2) between a block rectangle and label boxes.

    Parameters:
        block_rect: The candidate block region in FOV
            ``(v_min, u_min, v_max, u_max)`` coordinates (max exclusive).
        text_bboxes: Annotation label boxes in the same coordinate convention.

    Returns:
        The summed intersection area over every label box; ``0`` when the
        block region touches no label.
    """
    v0b, u0b, v1b, u1b = block_rect
    total = 0
    for v0, u0, v1, u1 in text_bboxes:
        dv = min(v1b, v1) - max(v0b, v0)
        du = min(u1b, u1) - max(u0b, u0)
        if dv > 0 and du > 0:
            total += dv * du
    return total


def _load_summary_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace TrueType font, falling back to PIL's built-in bitmap.

    Parameters:
        size: Font size in points.

    Returns:
        A PIL font object; the bitmap default is returned when no
        configured TrueType font can be opened.
    """
    font_dir = str(DEFAULT_CONFIG.general.get('truetype_font_dir', '') or '')
    candidates = [
        os.path.join(font_dir, 'liberation2', 'LiberationMono-Bold.ttf'),
        os.path.join(font_dir, 'liberation', 'LiberationMono-Bold.ttf'),
        os.path.join(font_dir, 'dejavu', 'DejaVuSansMono-Bold.ttf'),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grayscale_to_rgb_with_quantile_stretch(image: NDArrayFloatType) -> NDArrayUint8Type:
    """Build a uint8 RGB grayscale background from a float image.

    The black point is fixed at the 0.001 quantile.  The white point
    adapts to the number of "bright" pixels in the image: the default
    0.999 quantile clips the top 0.1 % of pixels, but on an image with
    only a handful of bright outliers (a sparse star field over dark
    sky, a distant body against empty sky) that fixed clip count
    saturates every bright pixel to 255 even though the brightest is
    much brighter than the rest.

    The fix counts the bright outliers via a robust median + 15 * MAD
    threshold and clips at most half of them -- so the brightest few
    are saturated but the remaining bright pixels keep their relative
    brightness ordering.  When the image carries many bright pixels
    (a body filling the FOV, a busy ring scene) the original 0.1 %
    behavior dominates and nothing about the existing visualization
    changes.
    """
    finite = np.isfinite(image)
    if not finite.any():
        clean = np.zeros_like(image)
        black = 0.0
        white = 1.0
    else:
        clean = np.where(finite, image, 0.0)
        finite_values = image[finite]
        n_finite = int(finite_values.size)
        black = float(np.quantile(finite_values, 0.001))

        default_clip_count = max(1, round(n_finite * 0.001))
        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)))
        if mad > 0.0:
            # 15 * MAD ~ 10 * sigma for gaussian noise (MAD = 0.6745 *
            # sigma).  Even on a 1 M-pixel detector a 10-sigma threshold
            # catches no noise pixels (P > 10 sigma ~ 1.5e-23) so the
            # bright-pixel count reflects real outliers (stars, body
            # limbs, ring edges) without polluting the count with the
            # gaussian-noise tail.
            bright_threshold = median + 15.0 * mad
            n_bright = int(np.sum(finite_values > bright_threshold))
        else:
            n_bright = 0
        if n_bright == 0:
            clip_count = default_clip_count
        else:
            # Clip only the brightest 5 % of outliers -- the remaining
            # 95 % stretch across the visible 0..255 range and preserve
            # their relative brightness ordering.  Half-clipping
            # (n_bright // 2) was too aggressive for "few bright
            # pixels" scenes where the user wants to see the gradient
            # within the bright region (a sparse star field, a small
            # body against dark sky, a thin ring against empty sky):
            # half the brights still saturate to 255 and the visual is
            # over-exposed.  5 % keeps that count small (1 of 20)
            # while still saturating the very brightest pixel so the
            # overall stretch is anchored.
            clip_count = min(default_clip_count, max(1, n_bright // 20))

        clip_quantile = 1.0 - clip_count / n_finite
        white = float(np.quantile(finite_values, clip_quantile))
        if white <= black:
            white = float(np.nextafter(black, np.inf))

    stretched = apply_linear_gamma_stretch(clean, black=black, white=white, gamma=1.0)
    gray = (stretched * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
