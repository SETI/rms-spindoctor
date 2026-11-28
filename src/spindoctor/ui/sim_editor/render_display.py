"""Render, display, PSF preview, selection, and image-save panel.

The picture is drawn in two passes.  Each edit draws the simulator's fast preview
on the UI thread, which takes a fraction of a second.  Once the editor has been idle
for a moment, the full render runs on a worker thread
(:mod:`spindoctor.ui.sim_editor.full_render`).  When it finishes, it replaces the
preview, provided the scene has not changed since it started.  A new edit discards a
full render that is still running.  The status bar says which of the two images is
on screen.

Every render validates the scene first.  The panel stretches DN to a grayscale
pixmap (with an optional saturation overlay and visual aids), applies the zoom
scaling, hit-tests right-clicks to select and drag scene objects, renders the
instrument PSF inset, and saves the full render to PNG.  The status bar gives the
frame's units and range, the share of pixels at the missing-data marker, and the
navigator's verdict when it would not call the frame clean, because the display
stretch hides all of them: a ``vgiss`` frame in I/F looks the same as a
``coiss_nac`` frame in DN.  That summary, the saturation overlay, and a saved PNG
all come from the full render, never from the preview.

The picture is offset-free, but the summary reads the frame the navigator reads,
with the planted offset applied.  When a scene plants an offset, that frame is a
render of its own, which runs off the UI thread beside the full render
(:mod:`spindoctor.ui.sim_editor.verdict`).  Until the summary arrives, the status
bar says the verdict is pending.  When validation refuses a scene, the last valid
picture stays on screen and the status bar shows the reason in place of a verdict.
A preview that fails on a valid scene opens an error dialog.
"""

import copy
from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QLabel, QMessageBox, QVBoxLayout

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.config.sections_camera import CameraBlock
from spindoctor.sim.instruments.navigator_link import (
    resolve_sim_inst_config,
    scene_camera_block,
)
from spindoctor.sim.render.detector.params import resolve_detector_params
from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.fields import default_body_name, scene_value
from spindoctor.sim.scene.validate import validate_scene
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.ui.sim_editor.base import SimEditorBase
from spindoctor.ui.sim_editor.frame_status import frame_units_name, has_planted_offset
from spindoctor.ui.sim_editor.full_render import FullRender, render_full_frame

# The status bar's words for the image on screen.
PREVIEW_TEXT = 'Fast preview'
PREVIEW_FULL_RUNNING_TEXT = 'Fast preview (full render running)'
FULL_RENDER_TEXT = 'Full render'
# The saturation overlay's status while the fast preview is on screen: waiting for
# the full render, or after that render has failed.
SATURATION_WAITING_TEXT = 'Saturated: waiting for the full render'
SATURATION_FAILED_TEXT = 'Saturated: unknown, the full render failed'
# The units line's words, after the units name, once an offset-free scene's full
# render has failed: that render was the one its verdict would have come from.
NO_VERDICT_TEXT = 'no verdict, the full render failed'
# The most characters of a failure message the status bar shows.  A traceback's
# message can be long, and a permanent status-bar widget grows to fit its text.
FAILURE_TEXT_MAX_CHARS = 80


def _short_error(error: str) -> str:
    """Return the first line of a failure message, shortened to fit the status bar.

    Parameters:
        error: The failure message.

    Returns:
        Its first line, cut to :data:`FAILURE_TEXT_MAX_CHARS` characters, ending in
        ``...`` when cut.
    """
    lines = error.splitlines()
    line = lines[0] if lines else ''
    if len(line) <= FAILURE_TEXT_MAX_CHARS:
        return line
    return line[: FAILURE_TEXT_MAX_CHARS - 3] + '...'


def _dn_to_display_uint8(image: Any) -> Any:
    """Stretch a DN image to 8-bit grayscale for display, scaling by its peak.

    The renderer emits detector counts (DN), whose range depends on the signal
    full-scale and any cosmic-ray spikes, so a peak-relative stretch keeps the
    preview legible regardless of absolute DN.

    Parameters:
        image: The DN image array to stretch.

    Returns:
        A uint8 array in [0, 255].
    """
    arr = np.asarray(image, dtype=np.float64)
    peak = float(arr.max()) if arr.size else 0.0
    scale = 255.0 / peak if peak > 0 else 0.0
    return np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)


class RenderDisplayMixin(SimEditorBase):
    """Builds the preview panel and owns rendering, display, and selection."""

    def _build_psf_preview(self, gen_layout: QFormLayout) -> None:
        """Add the collapsible PSF-preview inset to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        # PSF preview (B5): a collapsible inset of the selected instrument's
        # star PSF, with its sigma / FWHM, updated when the instrument changes.
        self._psf_group = QGroupBox('PSF preview')
        self._psf_group.setCheckable(True)
        self._psf_group.setChecked(False)
        psf_layout = QVBoxLayout(self._psf_group)
        self._psf_image_label = QLabel()
        self._psf_image_label.setVisible(False)
        self._psf_info_label = QLabel()
        self._psf_info_label.setVisible(False)
        psf_layout.addWidget(self._psf_image_label)
        psf_layout.addWidget(self._psf_info_label)
        self._psf_group.toggled.connect(self._on_psf_group_toggled)
        gen_layout.addRow(self._psf_group)

    def _on_psf_group_toggled(self, checked: bool) -> None:
        """Show/hide the PSF inset and refresh it when expanded."""
        self._psf_image_label.setVisible(checked)
        self._psf_info_label.setVisible(checked)
        if checked:
            self._update_psf_preview()

    def _update_psf_preview(self) -> None:
        """Render the selected instrument's star PSF into the preview inset."""
        if not self._psf_group.isChecked():
            return
        inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, self.sim_params.get('instrument'))
        sigma = inst_config.star_psf_sigma
        size = 25
        coords = np.arange(size) - size // 2
        vv, uu = np.meshgrid(coords.astype(float), coords.astype(float), indexing='ij')
        patch = np.exp(-(vv**2 + uu**2) / (2.0 * sigma**2))
        patch_uint8 = np.ascontiguousarray((patch * 255.0).astype(np.uint8))
        qimage = QImage(
            patch_uint8.tobytes(), size, size, size, QImage.Format.Format_Grayscale8
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            96,
            96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._psf_image_label.setPixmap(pixmap)
        self._psf_info_label.setText(f'sigma = {sigma:.2f} px    FWHM = {2.3548 * sigma:.2f} px')

    # ---- Rendering ----
    def _update_image(self) -> None:
        """Draw the fast preview, refresh the display, and update the status bar.

        The scene is validated first, on the UI thread.  If validation refuses it,
        the last valid picture stays on screen and the status bar shows the reason in
        place of a verdict.  A valid scene renders its offset-free fast preview, which
        replaces the picture.  Any full render or verdict render still on its way is
        discarded, the status bar shows the units with ``verdict pending``, and the
        idle timer restarts.  When it fires, :meth:`_start_full_renders` runs.

        Any other failure, of the render or of the validator itself, opens an error
        dialog that reports it.
        """
        try:
            if self._report_refusal():
                return
            # The check above stored this scene in the shared validation cache, so
            # the render's own validation finds it there.
            img, meta = render_scene(self.sim_params, ignore_offset=True, preview=True)
            self._full_render_runner.cancel()
            self._verdict_runner.cancel()
            self._show_picture(img, meta, copy.deepcopy(self.sim_params), full_render=False)
            units = frame_units_name(self._inst_config(self.sim_params))
            self._units_label.setText(f'{units}: verdict pending')
            self._idle_timer.start()
        except Exception as e:
            self._stop_full_renders()
            self._units_label.setText('')
            QMessageBox.critical(self, 'Error', f'Failed to preview the scene:\n{e!s}')

    def _report_refusal(self) -> bool:
        """Validate the scene and, if validation refuses it, say why in the status bar.

        A refused scene starts no full render or verdict render, and any already on
        their way are dropped.  An exception other than
        :class:`~spindoctor.sim.scene.errors.SimSceneValidationError` propagates to
        the caller.

        Returns:
            True if validation refused the scene, False if it passed.
        """
        try:
            validate_scene(self.sim_params)
        except SimSceneValidationError as e:
            self._stop_full_renders()
            self._units_label.setText(f'scene would be refused: {e}')
            return True
        return False

    def _show_picture(
        self,
        image: Any,
        meta: dict[str, Any],
        scene: dict[str, Any],
        *,
        full_render: bool,
    ) -> None:
        """Put a rendered image on screen and say in the status bar which image it is.

        Parameters:
            image: The rendered image.
            meta: The renderer's output metadata for the image.
            scene: The editor's own copy of the scene that produced the image.
            full_render: True for the full render, False for the fast preview.
        """
        self._current_image = image
        self._last_meta = meta
        self._displayed_params = scene
        self._showing_full_render = full_render
        self._full_render_failed = False
        self._set_image_kind(FULL_RENDER_TEXT if full_render else PREVIEW_TEXT)
        self._display_image()

    def _set_image_kind(self, text: str, *, detail: str = '') -> None:
        """Say in the status bar which image is on screen.

        Parameters:
            text: The status bar's words for the image.
            detail: Longer text shown as the words' tooltip, such as the whole
                message of a failed full render.  Empty for no tooltip.
        """
        self._image_kind_label.setText(text)
        self._image_kind_label.setToolTip(detail)

    def _stop_full_renders(self) -> None:
        """Stop the idle timer and drop any full render or verdict render on its way."""
        self._idle_timer.stop()
        self._full_render_runner.cancel()
        self._verdict_runner.cancel()
        if not self._showing_full_render and self._current_image is not None:
            self._set_image_kind(PREVIEW_TEXT)

    def _start_full_renders(self) -> None:
        """Start the full render, and the verdict render if the scene plants an offset.

        The idle timer calls this once the editor has been idle for its delay.  If the
        scene on screen is no longer the edited scene, nothing starts, because a new
        preview is on its way and restarts the timer.  If the full render of the
        scene is already on screen, because saving a PNG drew it, only the verdict
        render starts.
        """
        if self._displayed_params != self.sim_params:
            return
        if has_planted_offset(self.sim_params):
            self._verdict_runner.request(self.sim_params)
        if self._showing_full_render:
            return
        self._full_render_runner.request(self.sim_params)
        self._set_image_kind(PREVIEW_FULL_RUNNING_TEXT)

    def _show_full_render(self, scene: dict[str, Any], result: FullRender) -> None:
        """Replace the preview with a finished full render, if its scene is current.

        Without a planted offset, the status bar then describes the full render.  With
        one, the verdict runner describes the navigated frame instead.

        Parameters:
            scene: The worker's copy of the scene it rendered.
            result: The full render.
        """
        if scene != self.sim_params:
            # The scene changed after the render started.  The edit's own preview
            # is on its way, or already on screen.
            self._set_image_kind(PREVIEW_TEXT)
            return
        self._show_picture(result.image, result.meta, scene, full_render=True)
        if not has_planted_offset(scene):
            self._units_label.setText(result.units_summary)

    def _show_full_render_failure(self, error: str) -> None:
        """Report a failed full render in the status bar.

        The preview stays on screen.  The status bar gives the first line of the
        error, shortened, with the whole message as its tooltip.  Without a planted
        offset, the units line can no longer expect a verdict, so it says the full
        render failed, and so does the saturation overlay's status.

        Parameters:
            error: ``'<ExceptionType>: <message>'`` from the failed render.
        """
        self._full_render_failed = True
        self._set_image_kind(
            f'{PREVIEW_TEXT} (full render failed: {_short_error(error)})', detail=error
        )
        if not has_planted_offset(self._displayed_params):
            units = frame_units_name(self._inst_config(self._displayed_params))
            self._units_label.setText(f'{units}: {NO_VERDICT_TEXT}')
        self._display_image()

    @staticmethod
    def _inst_config(scene: dict[str, Any]) -> CameraBlock:
        """Return the navigator block a scene's frame is read with.

        Parameters:
            scene: The scene mapping.

        Returns:
            The scene's navigator block.
        """
        return scene_camera_block(DEFAULT_CONFIG, scene)

    def _show_verdict(self, scene: dict[str, Any], summary: str) -> None:
        """Show the navigated frame's summary, if its scene is current.

        Parameters:
            scene: The worker's copy of the scene it rendered.
            summary: The status text for that scene's navigated frame.
        """
        if scene == self.sim_params:
            self._units_label.setText(summary)

    def _show_verdict_failure(self, error: str) -> None:
        """Report a failed navigated-frame render in the status bar.

        The status bar gives the first line of the error, shortened.  The whole
        message and its traceback go to standard error.

        Parameters:
            error: ``'<ExceptionType>: <message>'`` from the failed render.
        """
        self._units_label.setText(f'verdict render failed: {_short_error(error)}')

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Drop any edit or render still on its way, then close the window.

        Parameters:
            event: The close event.
        """
        self._updater.cancel()
        self._stop_full_renders()
        super().closeEvent(event)

    def _toggle_saturation_overlay(self, checked: bool) -> None:
        """Toggle the saturation overlay and re-display (no re-render)."""
        self._show_saturation_overlay = bool(checked)
        self._display_image()

    def _current_saturation_dn(self) -> float | None:
        """The ADC ceiling the displayed frame was clipped at, or None if it is not in DN."""
        dp = resolve_detector_params(self._displayed_params)
        return dp.saturation_dn if dp.data_units == 'raw_dn' else None

    def _display_image(self) -> None:
        """Compose the displayed pixmap from the current DN image and redraw.

        The marks for bodies, stars, and the ring center come from the scene that
        produced the picture, so they stay on the picture while an edit leaves the
        scene refused.  The saturation overlay is drawn only on the full render.
        While the fast preview is on screen, the overlay's status says it is waiting
        for the full render, or that the full render failed.
        """
        if self._current_image is None:
            return
        shown = self._displayed_params
        img_uint8 = _dn_to_display_uint8(self._current_image)
        height, width = img_uint8.shape
        saturation_dn = self._current_saturation_dn()
        overlay = self._show_saturation_overlay and saturation_dn is not None
        if overlay and self._showing_full_render:
            rgb = np.repeat(img_uint8[:, :, np.newaxis], 3, axis=2)
            sat_mask = self._current_image >= saturation_dn
            rgb[sat_mask] = (255, 0, 0)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            qimage = QImage(
                rgb.tobytes(),
                width,
                height,
                3 * width,
                QImage.Format.Format_RGB888,
            ).copy()
            self._saturation_label.setText(f'Saturated: {float(sat_mask.mean()) * 100:.2f}%')
        else:
            img_uint8 = np.ascontiguousarray(img_uint8.copy())
            qimage = QImage(
                img_uint8.tobytes(),
                width,
                height,
                width,
                QImage.Format.Format_Grayscale8,
            ).copy()
            if not overlay:
                self._saturation_label.setText('')
            elif self._full_render_failed:
                self._saturation_label.setText(SATURATION_FAILED_TEXT)
            else:
                self._saturation_label.setText(SATURATION_WAITING_TEXT)
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)
        if self._show_visual_aids:
            # A scene states every position as a pixel corner and a QPainter
            # coordinate is the same measure, so these marks take the scene value
            # itself: no conversion, and no rounding either, or the mark for a
            # center at 12.5 would be drawn at 12.
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            # Draw centers for bodies
            for b in shown.get('bodies') or []:
                center_x = float(scene_value(b, 'center_u', shown['size_u'] / 2.0))
                center_y = float(scene_value(b, 'center_v', shown['size_v'] / 2.0))
                painter.drawEllipse(QRectF(center_x - 4.0, center_y - 4.0, 8.0, 8.0))
            # Draw stars as small crosses
            pen = QPen(QColor(255, 255, 0), 1)
            painter.setPen(pen)
            for s in shown.get('stars') or []:
                v, u = self._star_position(s, scene=shown)
                painter.drawLine(QPointF(u - 4.0, v), QPointF(u + 4.0, v))
                painter.drawLine(QPointF(u, v - 4.0), QPointF(u, v + 4.0))
            # Draw the ring system's shared center as a small circle
            ring_system = shown.get('ring_system')
            if isinstance(ring_system, dict):
                pen = QPen(QColor(0, 255, 255), 2)
                painter.setPen(pen)
                geometry = ring_system.get('geometry') or {}
                # An absent center is the detector center, as the renderer reads it.
                center_u = float(scene_value(geometry, 'center_u', shown['size_u'] / 2.0))
                center_v = float(scene_value(geometry, 'center_v', shown['size_v'] / 2.0))
                painter.drawEllipse(QRectF(center_u - 4.0, center_v - 4.0, 8.0, 8.0))
        painter.end()
        self._base_pixmap = pixmap
        self._update_display()
        self._image_label.repaint()
        viewport = self._scroll_area.viewport()
        if viewport is not None:
            viewport.repaint()

    def _update_display(self) -> None:
        """Rescale the base pixmap to the current zoom and show it."""
        if self._base_pixmap is None:
            return
        scaled_width = int(self._base_pixmap.width() * self._zoom_factor)
        scaled_height = int(self._base_pixmap.height() * self._zoom_factor)
        transform_mode = (
            Qt.TransformationMode.FastTransformation
            if self._zoom_sharp
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled_pixmap = self._base_pixmap.scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            transform_mode,
        )
        self._image_label.setPixmap(scaled_pixmap)
        self._image_label.resize(scaled_width, scaled_height)

    # ---- Selection / drag-move ----
    def _select_model_at(self, img_v: float, img_u: float) -> None:
        """Select the topmost model at the given image coordinates based on range ordering.

        Checks bodies and rings together, sorted by range (near to far), and selects
        the first match (topmost object).

        Parameters:
            img_v: Cursor V position in pixel corner coordinates.
            img_u: Cursor U position in pixel corner coordinates.
        """
        # The masks belong to the displayed picture, so the hit test reads the
        # scene that produced it.
        shown = self._displayed_params
        if not shown:
            self._selected_model_key = None
            return
        # While an edit leaves the scene refused, the picture shows an earlier scene.
        # Its objects can sit at other indices in the edited scene, which a drag
        # changes, so a click leaves the selection empty until the scene validates.
        if shown != self.sim_params:
            self._selected_model_key = None
            return
        height = int(shown['size_v'])
        width = int(shown['size_u'])
        if not (0.0 <= img_v < height and 0.0 <= img_u < width):
            self._selected_model_key = None
            return
        # The mask pixel a click lands in is the one CONTAINING the pixel corner
        # position, which is its floor; rounding would credit the click to the
        # next pixel over for the whole upper half of every pixel.
        v_i = int(img_v)
        u_i = int(img_u)

        # Collect all objects (bodies and rings) with their ranges and masks
        objects: list[tuple[float, str, int, Any]] = []  # (range, kind, index, mask)

        # Add bodies
        body_masks = self._last_meta.get('body_masks', [])
        bodies = shown.get('bodies', [])
        inv = self._last_meta.get('inventory', {})
        if body_masks and bodies:
            # body_masks is in the original order of bodies_params, so match by index
            for idx, body in enumerate(bodies):
                if idx < len(body_masks):
                    body_name = str(scene_value(body, 'name', default_body_name(idx))).upper()
                    range_val = inv.get(body_name, {}).get('range', float('inf'))
                    objects.append((range_val, 'body', idx, body_masks[idx]))

        # Add the ring system.  range_km is the only depth key; a system
        # without one has no depth relation to bodies, so it hit-tests as
        # farthest here.  A hit selects the first feature's tab (the one
        # carrying the shared system controls).
        ring_masks = self._last_meta.get('ring_masks', [])
        ring_system = shown.get('ring_system')
        if ring_masks and isinstance(ring_system, dict) and self._ring_features():
            range_val = float(ring_system.get('range_km') or float('inf'))
            objects.append((range_val, 'ring', 0, ring_masks[0]))

        # Sort by range (near to far = ascending range)
        objects.sort(key=lambda x: x[0])

        # Check objects in order (near to far), select first match
        for _, kind, idx, mask in objects:
            if mask is not None and bool(mask[v_i, u_i]):
                self._selected_model_key = (kind, idx)
                tab_idx = self._find_tab_by_properties(kind, idx)
                if tab_idx is not None:
                    self._tabs.setCurrentIndex(tab_idx)
                return

        # Stars: evaluate PSF contribution approx via Gaussian envelope
        # Stars are always behind bodies and rings
        star_info = self._last_meta.get('star_info', [])
        if star_info:
            # A hit-test entry states the RENDERED position pixel centric, while
            # the cursor arrives pixel corner, so the cursor crosses to the
            # renderer's system once before the two are differenced.
            centric_v = img_v - PIXEL_CENTER_TO_CORNER_PX
            centric_u = img_u - PIXEL_CENTER_TO_CORNER_PX
            for j, info in enumerate(star_info):
                cv = info['center_v']
                cu = info['center_u']
                sigma = info['sigma']
                dv = centric_v - cv
                du = centric_u - cu
                r2 = dv * dv + du * du
                # Gaussian threshold ~ 3 sigma circle, floored so a
                # PSF-free star (recorded sigma 0, a 1-px spike) still
                # offers a clickable 1-px-radius target.
                if r2 <= max(3.0 * sigma, 1.0) ** 2:
                    self._selected_model_key = ('star', j)
                    # Switch to star tab by finding it by properties
                    tab_idx = self._find_tab_by_properties('star', j)
                    if tab_idx is not None:
                        self._tabs.setCurrentIndex(tab_idx)
                    return

        self._selected_model_key = None

    def _move_selected_by(self, img_v: float, img_u: float) -> None:
        """Drag the currently selected object to follow the cursor."""
        if self._last_drag_img_vu is None or self._selected_model_key is None:
            self._last_drag_img_vu = (img_v, img_u)
            return
        prev_v, prev_u = self._last_drag_img_vu
        dv = img_v - prev_v
        du = img_u - prev_u
        kind, idx = self._selected_model_key
        if kind == 'body' and 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['center_v'] = float(
                scene_value(
                    self.sim_params['bodies'][idx], 'center_v', self.sim_params['size_v'] / 2.0
                )
                + dv
            )
            self.sim_params['bodies'][idx]['center_u'] = float(
                scene_value(
                    self.sim_params['bodies'][idx], 'center_u', self.sim_params['size_u'] / 2.0
                )
                + du
            )
            # Sync the tab spin boxes for this body
            tab_idx = self._find_tab_by_properties('body', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    cv_spin = tab_w.center_v_spin  # type: ignore[attr-defined]
                    cu_spin = tab_w.center_u_spin  # type: ignore[attr-defined]
                    cv_spin.setValue(self.sim_params['bodies'][idx]['center_v'])
                    cu_spin.setValue(self.sim_params['bodies'][idx]['center_u'])
            self._updater.immediate_update()
        elif kind == 'ring' and isinstance(self.sim_params.get('ring_system'), dict):
            # Dragging any ring feature moves the whole system's shared center.
            geometry = self.sim_params['ring_system'].setdefault('geometry', {})
            center_v = scene_value(geometry, 'center_v', self.sim_params['size_v'] / 2.0)
            center_u = scene_value(geometry, 'center_u', self.sim_params['size_u'] / 2.0)
            geometry['center_v'] = float(center_v) + dv
            geometry['center_u'] = float(center_u) + du
            # Sync the shared center spin boxes (they live on the first
            # feature's tab).
            tab_idx = self._find_tab_by_properties('ring', 0)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    cv_spin = tab_w.center_v_spin  # type: ignore[attr-defined]
                    cu_spin = tab_w.center_u_spin  # type: ignore[attr-defined]
                    cv_spin.setValue(geometry['center_v'])
                    cu_spin.setValue(geometry['center_u'])
            self._updater.immediate_update()
        elif kind == 'star' and 0 <= idx < len(self.sim_params['stars']):
            star_v, star_u = self._star_position(self.sim_params['stars'][idx])
            self.sim_params['stars'][idx]['v'] = star_v + dv
            self.sim_params['stars'][idx]['u'] = star_u + du
            # Sync the tab spin boxes for this star
            tab_idx = self._find_tab_by_properties('star', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    v_spin = tab_w.v_spin  # type: ignore[attr-defined]
                    u_spin = tab_w.u_spin  # type: ignore[attr-defined]
                    v_spin.setValue(self.sim_params['stars'][idx]['v'])
                    u_spin.setValue(self.sim_params['stars'][idx]['u'])
            self._updater.immediate_update()
        else:
            raise AssertionError(f'Unknown kind: {kind}')
        self._last_drag_img_vu = (img_v, img_u)

    # ---- Save image ----
    def _full_render_image(self) -> Any:
        """Return the full render of the scene behind the picture on screen.

        If the preview is on screen, the full render is drawn now, on the UI thread.
        When the preview's scene is still the edited scene, the new render then
        replaces the preview, as a finished background render would, and the
        background render of that scene, if one is on its way, is dropped.

        Returns:
            The image on screen if it is the full render, and otherwise a new full
            render of the same scene.
        """
        if self._showing_full_render:
            return self._current_image
        scene = copy.deepcopy(self._displayed_params)
        full = render_full_frame(scene)
        if scene == self.sim_params:
            self._full_render_runner.cancel()
            self._show_full_render(scene, full)
        return full.image

    def _save_image(self) -> None:
        """Save the full render of the picture on screen to a PNG file.

        The PNG is the full render's stretched image, without the marks or the
        saturation overlay.  If the fast preview is on screen, the full render of the
        same scene is rendered now, on the UI thread, before the file is written, and
        shown (see :meth:`_full_render_image`).
        """
        if self._current_image is None:
            QMessageBox.warning(self, 'No Image', 'No image to save.')
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Image',
            'simulated_model.png',
            'PNG Images (*.png)',
        )
        if filename:
            try:
                from PIL import Image

                img_uint8 = _dn_to_display_uint8(self._full_render_image())
                Image.fromarray(img_uint8, mode='L').save(filename)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image:\n{e!s}')
