"""Render, display, PSF preview, selection, and image-save panel.

Drives the live preview: it calls the renderer, stretches DN to a grayscale
pixmap (with an optional saturation overlay and visual aids), applies the zoom
scaling, hit-tests right-clicks to select and drag scene objects, renders the
instrument PSF inset, and saves the current preview to PNG.
"""

from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QLabel, QMessageBox, QVBoxLayout

from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX


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
        sigma = float(inst_config.get('star_psf_sigma', 1.0))
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
        """Re-render the scene and refresh the display."""
        try:
            # Render image (caching is handled in render.py)
            img, meta = render_combined_model(self.sim_params, ignore_offset=True)
            self._current_image = img
            self._last_meta = meta
            self._display_image()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to render image:\n{e!s}')

    def _toggle_saturation_overlay(self, checked: bool) -> None:
        """Toggle the saturation overlay and re-display (no re-render)."""
        self._show_saturation_overlay = bool(checked)
        self._display_image()

    def _current_saturation_dn(self) -> float | None:
        """Saturation DN of the selected instrument, or None if it has none."""
        inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, self.sim_params.get('instrument'))
        if str(inst_config.get('data_units', 'raw_dn')) != 'raw_dn':
            return None
        noise = inst_config.get('noise') or {}
        if 'saturation_dn' not in noise:
            return None
        return float(noise['saturation_dn'])

    def _display_image(self) -> None:
        """Compose the preview pixmap from the current DN image and redraw."""
        if self._current_image is None:
            return
        img_uint8 = _dn_to_display_uint8(self._current_image)
        height, width = img_uint8.shape
        saturation_dn = self._current_saturation_dn()
        if self._show_saturation_overlay and saturation_dn is not None:
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
            self._saturation_label.setText('')
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)
        if self._show_visual_aids:
            # A scene states every position as a pixel corner and a QPainter
            # coordinate is the same measure, so these marks take the scene value
            # itself: no conversion, and no rounding either, or the mark for a
            # centre at 12.5 would be drawn at 12.
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            # Draw centers for bodies
            for b in self.sim_params.get('bodies', []):
                center_x = float(b.get('center_u', 0.0))
                center_y = float(b.get('center_v', 0.0))
                painter.drawEllipse(QRectF(center_x - 4.0, center_y - 4.0, 8.0, 8.0))
            # Draw stars as small crosses
            pen = QPen(QColor(255, 255, 0), 1)
            painter.setPen(pen)
            for s in self.sim_params.get('stars', []):
                u = float(s.get('u', 0.0))
                v = float(s.get('v', 0.0))
                painter.drawLine(QPointF(u - 4.0, v), QPointF(u + 4.0, v))
                painter.drawLine(QPointF(u, v - 4.0), QPointF(u, v + 4.0))
            # Draw the ring system's shared center as a small circle
            ring_system = self.sim_params.get('ring_system')
            if isinstance(ring_system, dict):
                pen = QPen(QColor(0, 255, 255), 2)
                painter.setPen(pen)
                geometry = ring_system.get('geometry') or {}
                center_u = float(geometry.get('center_u', 0.0))
                center_v = float(geometry.get('center_v', 0.0))
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
        height = int(self.sim_params['size_v'])
        width = int(self.sim_params['size_u'])
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
        bodies = self.sim_params.get('bodies', [])
        inv = self._last_meta.get('inventory', {})
        if body_masks and bodies:
            # body_masks is in the original order of bodies_params, so match by index
            for idx, body in enumerate(bodies):
                if idx < len(body_masks):
                    body_name = body.get('name', '').upper()
                    range_val = inv.get(body_name, {}).get('range', float('inf'))
                    objects.append((range_val, 'body', idx, body_masks[idx]))

        # Add the ring system.  range_km is the only depth key; a system
        # without one has no depth relation to bodies, so it hit-tests as
        # farthest here.  A hit selects the first feature's tab (the one
        # carrying the shared system controls).
        ring_masks = self._last_meta.get('ring_masks', [])
        ring_system = self.sim_params.get('ring_system')
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
                self.sim_params['bodies'][idx].get('center_v', 0.0) + dv
            )
            self.sim_params['bodies'][idx]['center_u'] = float(
                self.sim_params['bodies'][idx].get('center_u', 0.0) + du
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
            geometry['center_v'] = float(geometry.get('center_v', 0.0) + dv)
            geometry['center_u'] = float(geometry.get('center_u', 0.0) + du)
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
            self.sim_params['stars'][idx]['v'] = float(
                self.sim_params['stars'][idx].get('v', 0.0) + dv
            )
            self.sim_params['stars'][idx]['u'] = float(
                self.sim_params['stars'][idx].get('u', 0.0) + du
            )
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
    def _save_image(self) -> None:
        """Save the current preview to a PNG file."""
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

                img_uint8 = _dn_to_display_uint8(self._current_image)
                Image.fromarray(img_uint8, mode='L').save(filename)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image:\n{e!s}')
