#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any, cast

import cspyce
import numpy as np
from astropy.io import fits
from filecache import FCPath, FileCache
from PIL import Image
from PyQt6.QtCore import QObject, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from matplotlib import colormaps as mpl_colormaps

from spindoctor.config import (
    DEFAULT_CONFIG,
    Config,
    get_backplane_results_root,
    get_nav_results_root,
    image_scope,
    load_default_and_user_config,
)
from spindoctor.dataset import dataset_name_to_class, dataset_name_to_inst_name, dataset_names
from spindoctor.dataset.dataset import DataSet, ImageFiles
from spindoctor.obs import ObsSnapshot, inst_name_to_obs_class
from spindoctor.ui.common import ZoomPanController, build_stretch_controls


class _ImageLabel(QLabel):
    """Event-forwarding image label; behavior copied from create_simulated_body_model."""

    def __init__(
        self,
        parent: QWidget | None,
        on_press: Callable[[QMouseEvent], None],
        on_move: Callable[[QMouseEvent], None],
        on_release: Callable[[QMouseEvent], None],
        on_wheel: Callable[[QWheelEvent], None],
    ) -> None:
        super().__init__(parent)
        self._on_press = on_press
        self._on_move = on_move
        self._on_release = on_release
        self._on_wheel = on_wheel

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_press(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_release(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            self._on_wheel(event)


class _ParameterUpdater(QObject):
    update_requested = pyqtSignal()

    def __init__(self, delay_ms: int) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setInterval(delay_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_update)

    def request_update(self) -> None:
        self._timer.start()

    def immediate_update(self) -> None:
        self._timer.stop()
        self.update_requested.emit()

    def _emit_update(self) -> None:
        self.update_requested.emit()


def _apply_stretch_gamma(image: np.ndarray, black: float, white: float, gamma: float) -> np.ndarray:
    if white <= black:
        white = black + 1e-6
    scaled = np.clip((image - black) / (white - black), 0.0, 1.0)
    if gamma <= 0:
        gamma = 1.0
    scaled = np.power(scaled, 1.0 / gamma)
    return (scaled * 255.0).astype(np.uint8)


def _rad_to_deg_if_units(name: str, units: str | None, arr: np.ndarray) -> np.ndarray:
    if units and units.lower() == 'rad':
        return np.degrees(arr)
    # Heuristic: some HDU names encode angle but BUNIT may be missing
    lower = name.lower()
    if any(k in lower for k in ('longitude', 'latitude', 'incidence', 'emission', 'phase')):
        return np.degrees(arr)
    return arr


def _absolute_range_for(name: str, arr_deg_or_native: np.ndarray) -> tuple[float, float]:
    lname = name.lower()
    if 'longitude' in lname:
        return (0.0, 360.0)
    if 'latitude' in lname:
        return (-90.0, 90.0)
    if any(k in lname for k in ('incidence', 'emission', 'phase')):
        return (0.0, 180.0)
    if 'radius' in lname:
        vmin = 0.0
        vmax = float(np.nanmax(arr_deg_or_native))
        return (vmin, vmax if np.isfinite(vmax) else 1.0)
    if 'resolution' in lname:
        # fall back to observed min/max
        finite = arr_deg_or_native[np.isfinite(arr_deg_or_native)]
        if finite.size == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(finite)), float(np.nanmax(finite)))
    # default: observed min/max
    finite = arr_deg_or_native[np.isfinite(arr_deg_or_native)]
    if finite.size == 0:
        return (0.0, 1.0)
    return (float(np.nanmin(finite)), float(np.nanmax(finite)))


def _load_colormap(cmap_name: Any) -> Any:
    """Resolve a colormap by name."""
    if cmap_name is None:
        return None
    return mpl_colormaps.get(str(cmap_name))


def _alpha_blend_layer(
    dst_rgba: np.ndarray, src_rgb: np.ndarray, alpha_mask: np.ndarray
) -> np.ndarray:
    """Alpha-blend src_rgb over dst_rgba using per-pixel alpha in [0,1].

    dst_rgba is modified in-place and also returned.
    """
    dst_rgb = dst_rgba[..., :3].astype(np.float32)
    dst_a = dst_rgba[..., 3].astype(np.float32) / 255.0
    src_rgb_f = src_rgb.astype(np.float32)
    a = alpha_mask.astype(np.float32)
    out_a = a + dst_a * (1.0 - a)
    with np.errstate(invalid='ignore'):
        out_rgb = src_rgb_f * a[..., None] + dst_rgb * dst_a[..., None] * (1.0 - a[..., None])
        mask = out_a > 0
        out_rgb[mask] = out_rgb[mask] / out_a[mask, None]
        out_rgb[~mask] = 0.0
    dst_rgba[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    dst_rgba[..., 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return dst_rgba


class NavBackplaneViewer(QDialog):
    """Interactive backplane viewer with dataset-driven file discovery."""

    def __init__(
        self,
        *,
        dataset: DataSet,
        inst_name: str,
        image_groups: list[ImageFiles],
        nav_results_root: FCPath,
        backplane_results_root: FCPath,
        config: Config | None,
    ) -> None:
        super().__init__(None)
        self.setWindowTitle('Backplane Viewer')
        # Minimum size: scroll area (900x800) + right panel (~400) + margins
        self.setMinimumSize(1650, 950)

        self._dataset = dataset
        self._inst_name = inst_name
        self._nav_results_root = nav_results_root
        self._backplane_results_root = backplane_results_root
        self._config = config or DEFAULT_CONFIG

        self._obs_class = inst_name_to_obs_class(inst_name)

        # Enumerated images
        self._image_groups = image_groups
        self._current_index = 0

        # Image data
        self._img_float: np.ndarray | None = None
        self._summary_rgba: np.ndarray | None = None  # HxWx4 uint8
        self._fits_hdus: list[fits.ImageHDU | fits.PrimaryHDU] = []
        self._body_id_map: np.ndarray | None = None
        # one per HDU (excluding primary and BODY_ID_MAP)
        self._last_rgba: np.ndarray | None = None

        # Stretch controls
        self._black = 0.0
        self._white = 1.0
        self._gamma = 1.0
        self._stretch_min = 0.0
        self._stretch_max = 1.0

        # View state (copied math/state shape from create_simulated_body_model)
        self._zoom_factor = 1.0
        self._zoom_sharp = True

        self._updater = _ParameterUpdater(120)
        self._updater.update_requested.connect(self._compose_and_display)

        self._cmap_items: list[tuple[str, str]] = []
        self._stretch_controls: dict[str, Any] = {}

        self._build_ui()

        # Load first image group
        if self._image_groups:
            self._load_group(self._current_index)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left column: controls row + viewport + status
        left = QVBoxLayout()
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton('Prev Image')
        self._btn_next = QPushButton('Next Image')
        self._btn_zoom_out = QPushButton('Zoom -')
        self._btn_zoom_in = QPushButton('Zoom +')
        self._btn_reset = QPushButton('Reset View')
        self._btn_save = QPushButton('Save PNG')
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next.clicked.connect(self._next_image)
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        self._btn_reset.clicked.connect(self._reset_view)
        self._btn_save.clicked.connect(self._save_viewport_png)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        nav_row.addStretch()
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        nav_row.addWidget(self._btn_zoom_out)
        nav_row.addWidget(self._btn_zoom_in)
        nav_row.addWidget(self._zoom_sharp_check)
        nav_row.addWidget(self._btn_reset)
        nav_row.addWidget(self._btn_save)
        nav_row.addStretch()
        left.addLayout(nav_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setMinimumSize(900, 800)
        self._scroll.setStyleSheet('background-color: black;')
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = _ImageLabel(
            self, self._on_press, self._on_move, self._on_release, self._on_wheel
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet('background-color: black;')
        self._label.setMouseTracking(True)
        self._scroll.setWidget(self._label)
        left.addWidget(self._scroll)

        status = QStatusBar()
        self._status_label = QLabel('V, U: --, --  Value: --')
        self._zoom_label = QLabel('Zoom: 1.00x')
        status.addWidget(self._status_label)
        status.addPermanentWidget(self._zoom_label)
        left.addWidget(status)
        layout.addLayout(left, stretch=3)

        # Right column: sections (Image Stretch, Summary, Backplanes)
        right_widget = QWidget()
        right_widget.setMinimumWidth(380)  # Ensure controls panel doesn't get too narrow
        right = QVBoxLayout(right_widget)
        right.setSpacing(6)
        # Image Stretch group (built after image load)
        self._image_group = QGroupBox('Image Stretch')
        self._image_group.setContentsMargins(6, 14, 6, 6)
        self._image_form = QFormLayout(self._image_group)
        self._image_form.setSpacing(4)
        right.addWidget(self._image_group)
        # Show image toggle
        self._show_image_check = QCheckBox('Show image')
        self._show_image_check.setChecked(True)
        self._show_image_check.stateChanged.connect(lambda *_: self._compose_and_display())
        self._image_form.addRow(self._show_image_check)

        # Summary overlay group
        self._summary_group = QGroupBox('Summary Overlay')
        self._summary_group.setContentsMargins(6, 14, 6, 6)
        summary_form = QFormLayout(self._summary_group)
        summary_form.setSpacing(4)
        self._summary_enable = QCheckBox('Show summary overlay')
        self._summary_enable.stateChanged.connect(lambda *_: self._compose_and_display())
        summary_alpha_row = QHBoxLayout()
        summary_alpha_row.setSpacing(4)
        summary_alpha_row.setContentsMargins(0, 0, 0, 0)
        summary_alpha_min_label = QLabel('0.0')
        summary_alpha_min_label.setFixedWidth(30)
        summary_alpha_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        summary_alpha_max_label = QLabel('1.0')
        summary_alpha_max_label.setFixedWidth(35)
        summary_alpha_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._summary_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._summary_alpha_slider.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        self._summary_alpha_slider.setValue(500)  # 0.5
        self._summary_alpha_slider.valueChanged.connect(self._on_summary_alpha_changed)
        self._summary_alpha_spin = QDoubleSpinBox()
        self._summary_alpha_spin.setRange(0.0, 1.0)
        self._summary_alpha_spin.setDecimals(3)
        self._summary_alpha_spin.setSingleStep(0.01)
        self._summary_alpha_spin.setValue(0.5)
        self._summary_alpha_spin.valueChanged.connect(self._on_summary_alpha_spin_changed)
        summary_alpha_row.addWidget(summary_alpha_min_label)
        summary_alpha_row.addWidget(self._summary_alpha_slider, stretch=1)
        summary_alpha_row.addWidget(summary_alpha_max_label)
        summary_alpha_row.addWidget(self._summary_alpha_spin)
        summary_alpha_holder = QWidget()
        summary_alpha_holder.setLayout(summary_alpha_row)
        summary_form.addRow(self._summary_enable)
        summary_form.addRow('Alpha', summary_alpha_holder)
        right.addWidget(self._summary_group)

        # Backplanes - separate groups
        self._populate_cmaps_if_needed()
        alpha_min_width = 220

        # BODY_ID group
        self._bp_id_group = QGroupBox('Body ID')
        self._bp_id_group.setContentsMargins(6, 14, 6, 6)
        gid = QVBoxLayout(self._bp_id_group)
        gid.setSpacing(2)
        row_id1 = QHBoxLayout()
        row_id1.setSpacing(6)
        self._body_id_enable = QCheckBox('Show BODY_ID_MAP')
        self._body_id_enable.setChecked(False)
        self._body_id_enable.stateChanged.connect(lambda *_: self._compose_and_display())
        body_id_alpha_row = QHBoxLayout()
        body_id_alpha_row.setSpacing(4)
        body_id_alpha_row.setContentsMargins(0, 0, 0, 0)
        body_id_alpha_min_label = QLabel('0.0')
        body_id_alpha_min_label.setFixedWidth(30)
        body_id_alpha_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        body_id_alpha_max_label = QLabel('1.0')
        body_id_alpha_max_label.setFixedWidth(35)
        body_id_alpha_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._body_id_alpha = QSlider(Qt.Orientation.Horizontal)
        self._body_id_alpha.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        self._body_id_alpha.setValue(500)  # 0.5
        self._body_id_alpha.setMinimumWidth(alpha_min_width)
        self._body_id_alpha.valueChanged.connect(self._on_body_id_alpha_changed)
        self._body_id_alpha_spin = QDoubleSpinBox()
        self._body_id_alpha_spin.setRange(0.0, 1.0)
        self._body_id_alpha_spin.setDecimals(3)
        self._body_id_alpha_spin.setSingleStep(0.01)
        self._body_id_alpha_spin.setValue(0.5)
        self._body_id_alpha_spin.valueChanged.connect(self._on_body_id_alpha_spin_changed)
        body_id_alpha_row.addWidget(body_id_alpha_min_label)
        body_id_alpha_row.addWidget(self._body_id_alpha, stretch=1)
        body_id_alpha_row.addWidget(body_id_alpha_max_label)
        body_id_alpha_row.addWidget(self._body_id_alpha_spin)
        body_id_alpha_holder = QWidget()
        body_id_alpha_holder.setLayout(body_id_alpha_row)
        row_id1.addWidget(self._body_id_enable)
        row_id1.addWidget(QLabel('Alpha'))
        row_id1.addWidget(body_id_alpha_holder, stretch=1)
        row_id2 = QHBoxLayout()
        row_id2.setSpacing(6)
        self._body_id_cmap = QComboBox()
        for display, cmap_name in self._cmap_items:
            self._body_id_cmap.addItem(display, cmap_name)
        # Default to turbo if available
        for i in range(self._body_id_cmap.count()):
            if self._body_id_cmap.itemData(i) == 'turbo':
                self._body_id_cmap.setCurrentIndex(i)
                break
        self._body_id_cmap.currentIndexChanged.connect(lambda *_: self._compose_and_display())
        lbl_mode_id = QLabel('Mode:')
        lbl_mode_id.setMinimumWidth(48)
        row_id2.addWidget(lbl_mode_id)
        # Replace mode dropdown with radio buttons
        self._body_id_mode_rel = QRadioButton('Relative')
        self._body_id_mode_abs = QRadioButton('Absolute')
        self._body_id_mode_rel.setChecked(True)
        self._body_id_mode_rel.toggled.connect(lambda *_: self._compose_and_display())
        self._body_id_mode_abs.toggled.connect(lambda *_: self._compose_and_display())
        row_id2.addWidget(self._body_id_mode_rel)
        row_id2.addWidget(self._body_id_mode_abs)
        row_id2.addSpacing(12)
        lbl_cmap_id = QLabel('Colormap:')
        lbl_cmap_id.setMinimumWidth(70)
        row_id2.addWidget(lbl_cmap_id)
        row_id2.addWidget(self._body_id_cmap)
        row_id2.addStretch(1)
        row_id3 = QHBoxLayout()
        row_id3.setSpacing(6)
        self._body_id_val_label = QLabel('Object: --')
        row_id3.addWidget(self._body_id_val_label)
        gid.addLayout(row_id1)
        gid.addLayout(row_id2)
        gid.addLayout(row_id3)
        right.addWidget(self._bp_id_group)

        # Body backplane group
        self._bp_body_group = QGroupBox('Body Backplane')
        self._bp_body_group.setContentsMargins(6, 14, 6, 6)
        gbc = QVBoxLayout(self._bp_body_group)
        gbc.setSpacing(2)
        row_b0 = QHBoxLayout()
        row_b0.setSpacing(6)
        self._body_show = QCheckBox('Show body backplane')
        self._body_show.setChecked(False)
        self._body_show.stateChanged.connect(lambda *_: self._compose_and_display())
        row_b0.addWidget(self._body_show)
        row_b1 = QHBoxLayout()
        row_b1.setSpacing(6)
        self._body_combo = QComboBox()
        self._body_combo.addItem('None')
        self._body_combo.currentIndexChanged.connect(lambda *_: self._on_body_selection_changed())
        body_alpha_row = QHBoxLayout()
        body_alpha_row.setSpacing(4)
        body_alpha_row.setContentsMargins(0, 0, 0, 0)
        body_alpha_min_label = QLabel('0.0')
        body_alpha_min_label.setFixedWidth(30)
        body_alpha_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        body_alpha_max_label = QLabel('1.0')
        body_alpha_max_label.setFixedWidth(35)
        body_alpha_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._body_alpha = QSlider(Qt.Orientation.Horizontal)
        self._body_alpha.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        self._body_alpha.setValue(500)  # 0.5
        self._body_alpha.setMinimumWidth(alpha_min_width)
        self._body_alpha.valueChanged.connect(self._on_body_alpha_changed)
        self._body_alpha_spin = QDoubleSpinBox()
        self._body_alpha_spin.setRange(0.0, 1.0)
        self._body_alpha_spin.setDecimals(3)
        self._body_alpha_spin.setSingleStep(0.01)
        self._body_alpha_spin.setValue(0.5)
        self._body_alpha_spin.valueChanged.connect(self._on_body_alpha_spin_changed)
        body_alpha_row.addWidget(body_alpha_min_label)
        body_alpha_row.addWidget(self._body_alpha, stretch=1)
        body_alpha_row.addWidget(body_alpha_max_label)
        body_alpha_row.addWidget(self._body_alpha_spin)
        body_alpha_holder = QWidget()
        body_alpha_holder.setLayout(body_alpha_row)
        row_b1.addWidget(self._body_combo, stretch=1)
        row_b1.addWidget(QLabel('Alpha'))
        row_b1.addWidget(body_alpha_holder, stretch=1)
        row_b2 = QHBoxLayout()
        row_b2.setSpacing(6)
        # Body mode radio buttons
        self._body_mode_rel = QRadioButton('Relative')
        self._body_mode_abs = QRadioButton('Absolute')
        self._body_mode_rel.setChecked(True)
        self._body_mode_rel.toggled.connect(lambda *_: self._on_body_controls_changed())
        self._body_mode_abs.toggled.connect(lambda *_: self._on_body_controls_changed())
        self._body_cmap = QComboBox()
        for display, cmap_name in self._cmap_items:
            self._body_cmap.addItem(display, cmap_name)
        for i in range(self._body_cmap.count()):
            if self._body_cmap.itemData(i) == 'turbo':
                self._body_cmap.setCurrentIndex(i)
                break
        self._body_cmap.currentIndexChanged.connect(lambda *_: self._on_body_controls_changed())
        lbl_mode_b = QLabel('Mode')
        lbl_mode_b.setMinimumWidth(48)
        row_b2.addWidget(lbl_mode_b)
        row_b2.addWidget(self._body_mode_rel)
        row_b2.addWidget(self._body_mode_abs)
        row_b2.addSpacing(12)
        lbl_cmap_b = QLabel('Colormap')
        lbl_cmap_b.setMinimumWidth(70)
        row_b2.addWidget(lbl_cmap_b)
        row_b2.addWidget(self._body_cmap)
        row_b2.addStretch(1)
        row_b3 = QHBoxLayout()
        row_b3.setSpacing(6)
        self._body_val_label = QLabel('Value: --')
        row_b3.addWidget(self._body_val_label)
        gbc.addLayout(row_b0)
        gbc.addLayout(row_b1)
        gbc.addLayout(row_b2)
        gbc.addLayout(row_b3)
        right.addWidget(self._bp_body_group)

        # Ring backplane group
        self._bp_ring_group = QGroupBox('Ring Backplane')
        self._bp_ring_group.setContentsMargins(6, 14, 6, 6)
        grc = QVBoxLayout(self._bp_ring_group)
        grc.setSpacing(2)
        row_r0 = QHBoxLayout()
        row_r0.setSpacing(6)
        self._ring_show = QCheckBox('Show ring backplane')
        self._ring_show.setChecked(False)
        self._ring_show.stateChanged.connect(lambda *_: self._compose_and_display())
        row_r0.addWidget(self._ring_show)
        row_r1 = QHBoxLayout()
        row_r1.setSpacing(6)
        self._ring_combo = QComboBox()
        self._ring_combo.addItem('None')
        self._ring_combo.currentIndexChanged.connect(lambda *_: self._on_ring_selection_changed())
        ring_alpha_row = QHBoxLayout()
        ring_alpha_row.setSpacing(4)
        ring_alpha_row.setContentsMargins(0, 0, 0, 0)
        ring_alpha_min_label = QLabel('0.0')
        ring_alpha_min_label.setFixedWidth(30)
        ring_alpha_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        ring_alpha_max_label = QLabel('1.0')
        ring_alpha_max_label.setFixedWidth(35)
        ring_alpha_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._ring_alpha = QSlider(Qt.Orientation.Horizontal)
        self._ring_alpha.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        self._ring_alpha.setValue(500)  # 0.5
        self._ring_alpha.setMinimumWidth(alpha_min_width)
        self._ring_alpha.valueChanged.connect(self._on_ring_alpha_changed)
        self._ring_alpha_spin = QDoubleSpinBox()
        self._ring_alpha_spin.setRange(0.0, 1.0)
        self._ring_alpha_spin.setDecimals(3)
        self._ring_alpha_spin.setSingleStep(0.01)
        self._ring_alpha_spin.setValue(0.5)
        self._ring_alpha_spin.valueChanged.connect(self._on_ring_alpha_spin_changed)
        ring_alpha_row.addWidget(ring_alpha_min_label)
        ring_alpha_row.addWidget(self._ring_alpha, stretch=1)
        ring_alpha_row.addWidget(ring_alpha_max_label)
        ring_alpha_row.addWidget(self._ring_alpha_spin)
        ring_alpha_holder = QWidget()
        ring_alpha_holder.setLayout(ring_alpha_row)
        row_r1.addWidget(self._ring_combo, stretch=1)
        row_r1.addWidget(QLabel('Alpha'))
        row_r1.addWidget(ring_alpha_holder, stretch=1)
        row_r2 = QHBoxLayout()
        row_r2.setSpacing(6)
        # Ring mode radio buttons
        self._ring_mode_rel = QRadioButton('Relative')
        self._ring_mode_abs = QRadioButton('Absolute')
        self._ring_mode_rel.setChecked(True)
        self._ring_mode_rel.toggled.connect(lambda *_: self._on_ring_controls_changed())
        self._ring_mode_abs.toggled.connect(lambda *_: self._on_ring_controls_changed())
        self._ring_cmap = QComboBox()
        for display, cmap_name in self._cmap_items:
            self._ring_cmap.addItem(display, cmap_name)
        for i in range(self._ring_cmap.count()):
            if self._ring_cmap.itemData(i) == 'turbo':
                self._ring_cmap.setCurrentIndex(i)
                break
        self._ring_cmap.currentIndexChanged.connect(lambda *_: self._on_ring_controls_changed())
        lbl_mode_r = QLabel('Mode')
        lbl_mode_r.setMinimumWidth(48)
        row_r2.addWidget(lbl_mode_r)
        row_r2.addWidget(self._ring_mode_rel)
        row_r2.addWidget(self._ring_mode_abs)
        row_r2.addSpacing(12)
        lbl_cmap_r = QLabel('Colormap')
        lbl_cmap_r.setMinimumWidth(70)
        row_r2.addWidget(lbl_cmap_r)
        row_r2.addWidget(self._ring_cmap)
        row_r2.addStretch(1)
        row_r3 = QHBoxLayout()
        row_r3.setSpacing(6)
        self._ring_val_label = QLabel('Value: --')
        row_r3.addWidget(self._ring_val_label)
        grc.addLayout(row_r0)
        grc.addLayout(row_r1)
        grc.addLayout(row_r2)
        grc.addLayout(row_r3)
        right.addWidget(self._bp_ring_group)
        # All panes compact; put stretch at bottom to absorb extra space
        right.addStretch(1)
        # Exit button at bottom right
        exit_row = QHBoxLayout()
        exit_row.addStretch()
        exit_btn = QPushButton('Exit')
        exit_btn.clicked.connect(self.close)
        exit_row.addWidget(exit_btn)
        right.addLayout(exit_row)
        layout.addWidget(right_widget, stretch=2)
        # Initialize preferences and maps
        self._body_prefs: dict[str, dict[str, Any]] = {}
        self._ring_prefs: dict[str, dict[str, Any]] = {}
        self._bp_body_map: dict[str, tuple[np.ndarray, str]] = {}
        self._bp_ring_map: dict[str, tuple[np.ndarray, str]] = {}
        # Initialize zoom/pan controller
        self._zoom_ctl = ZoomPanController(
            label=self._label,
            scroll_area=self._scroll,
            get_zoom=lambda: self._zoom_factor,
            set_zoom=lambda z: setattr(self, '_zoom_factor', float(z)),
            update_display=self._compose_and_display,
            set_zoom_label_text=lambda s: self._zoom_label.setText(s),
        )

    # ---- Loading ----
    def _load_group(self, index: int) -> None:
        if index < 0 or index >= len(self._image_groups):
            return
        group = self._image_groups[index]
        if len(group.image_files) != 1:
            print(
                f'Expected single image per group; got {len(group.image_files)}',
                file=sys.stderr,
            )
            return
        image_file = group.image_files[0]
        image_path = image_file.image_file_path.absolute()
        results_stub = image_file.results_path_stub
        self._current_image_name = image_path.name
        self.setWindowTitle(f'Backplane Viewer - {self._current_image_name}')

        # Load observation (science image).  The observation classes are
        # image-scoped library code and log while reading, so the load runs
        # inside an image scope even though the viewer itself keeps no logger;
        # without one every load would report a scope violation per record.
        try:
            with image_scope():
                snapshot = self._obs_class.from_file(image_path)
            if not isinstance(snapshot, ObsSnapshot):
                raise ValueError('Expected ObsSnapshot')
        except Exception as e:
            print(f'Failed to read image {image_path}', file=sys.stderr)
            traceback.print_exc()
            self._status_label.setText(f'Image read error: {e}')
            return

        img = snapshot.data.astype(np.float64)
        self._img_float = img
        self._stretch_min = float(np.nanmin(img))
        self._stretch_max = float(np.nanmax(img))
        if not np.isfinite(self._stretch_min):
            self._stretch_min = 0.0
        if not np.isfinite(self._stretch_max) or self._stretch_max <= self._stretch_min:
            self._stretch_max = self._stretch_min + 1.0
        # Initialize stretch to full min/max
        self._black = self._stretch_min
        self._white = self._stretch_max
        self._gamma = 1.0
        # Build or update stretch controls
        # Build stretch controls on first load
        if not self._stretch_controls:
            self._stretch_controls = build_stretch_controls(
                self._image_form,
                img_min=self._stretch_min,
                img_max=self._stretch_max,
                black_init=self._black,
                white_init=self._white,
                gamma_init=self._gamma,
                on_black_changed=self._on_black_changed,
                on_white_changed=self._on_white_changed,
                on_gamma_changed=self._on_gamma_changed,
            )
        # Ensure slider mappings use current image bounds before setting values
        self._stretch_controls['set_range'](self._stretch_min, self._stretch_max)
        # Update displayed values
        self._stretch_controls['set_values'](self._black, self._white, self._gamma)

        # Optional summary PNG
        summary_png_file = self._nav_results_root / (results_stub + '_summary.png')
        self._summary_rgba = None
        if summary_png_file.exists():
            try:
                png_local = cast(str, summary_png_file.get_local_path())
                with Image.open(png_local, mode='r') as im:
                    rgba = im.convert('RGBA')
                    self._summary_rgba = np.array(rgba)
                if self._summary_rgba is not None:
                    print(
                        f'Loaded summary overlay "{png_local}" '
                        f'size={self._summary_rgba.shape[1]}x'
                        f'{self._summary_rgba.shape[0]}'
                    )
            except Exception:
                print(
                    f'Failed to load summary overlay from {summary_png_file}',
                    file=sys.stderr,
                )
                traceback.print_exc()
                self._summary_rgba = None
        # Enable/disable summary checkbox based on availability
        has_summary = self._summary_rgba is not None
        self._summary_enable.setEnabled(has_summary)
        if not has_summary:
            self._summary_enable.setChecked(False)

        # Load backplanes FITS
        fits_file = self._backplane_results_root / (results_stub + '_backplanes.fits')
        self._fits_hdus = []
        self._body_id_map = None
        self._bp_body_map.clear()
        self._bp_ring_map.clear()
        try:
            local_path = cast(str, fits_file.get_local_path())
            with fits.open(local_path) as hdul:
                self._fits_hdus = list(hdul)
                print(
                    f'Opened backplanes FITS file "{local_path}" with {len(self._fits_hdus)} HDUs'
                )
                # Parse HDUs: locate BODY_ID_MAP and backplanes
                for hdu in self._fits_hdus[1:]:
                    name = (hdu.name or '').upper()
                    if name == 'BODY_ID_MAP':
                        try:
                            self._body_id_map = np.asarray(hdu.data, dtype=np.int32)
                        except Exception:
                            print(
                                f'Failed to parse BODY_ID_MAP from HDU for {fits_file}',
                                file=sys.stderr,
                            )
                            traceback.print_exc()
                            self._body_id_map = None
                    else:
                        arr: np.ndarray | None = None
                        try:
                            arr = np.asarray(hdu.data, dtype=np.float64)
                        except Exception:
                            print(
                                f'Failed to parse {name} from HDU for {fits_file}',
                                file=sys.stderr,
                            )
                            traceback.print_exc()
                            arr = None
                        if arr is None:
                            continue
                        units = hdu.header.get('BUNIT', '')
                        if name.startswith('BODY_'):
                            self._bp_body_map[name] = (arr, units)
                        elif name.startswith('RING_'):
                            self._bp_ring_map[name] = (arr, units)
        except FileNotFoundError:
            # Show modal dialog for file not found
            QMessageBox.warning(
                self,
                'FITS File Not Found',
                f'Backplane FITS file not found:\n{fits_file}\n\n'
                'All backplane data will be unavailable.',
            )
            print(f'FITS file not found: {fits_file}', file=sys.stderr)
            self._fits_hdus = []
            self._bp_body_map.clear()
            self._bp_ring_map.clear()
            self._body_id_map = None
        except Exception:
            print('Failed to read FITS backplane file', file=sys.stderr)
            traceback.print_exc()
            self._fits_hdus = []
            self._bp_body_map.clear()
            self._bp_ring_map.clear()
            self._body_id_map = None

        # Populate dropdowns with names
        self._populate_backplane_combos()
        # Enable/disable body show checkbox based on availability
        has_bodies = bool(self._bp_body_map)
        self._body_show.setEnabled(has_bodies)
        if not has_bodies:
            self._body_show.setChecked(False)
        # Enable/disable ring show checkbox based on availability
        has_rings = bool(self._bp_ring_map)
        self._ring_show.setEnabled(has_rings)
        if not has_rings:
            self._ring_show.setChecked(False)
        # Enable/disable BODY_ID_MAP checkbox based on availability
        has_body_id = self._body_id_map is not None
        self._body_id_enable.setEnabled(has_body_id)
        if not has_body_id:
            self._body_id_enable.setChecked(False)

        # Compose first display
        self._reset_view()
        self._compose_and_display()

    def _save_viewport_png(self) -> None:
        # Render full-resolution RGBA ignoring zoom/pan
        rgba = self._render_full_rgba()
        if rgba is None:
            return
        h, w, _ = rgba.shape
        qimg = QImage(bytes(rgba.data), w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
        default_name = 'backplanes.png'
        try:
            if self._current_image_name:
                default_name = f'{self._current_image_name}_backplanes.png'
        except Exception:
            pass
        path, _ = QFileDialog.getSaveFileName(self, 'Save PNG', default_name, 'PNG Files (*.png)')
        if not path:
            return
        try:
            qimg.save(path, 'PNG')
        except Exception as e:
            QMessageBox.critical(self, 'Save Error', f'Failed to save PNG to:\n{path}\n\n{e}')

    def _render_full_rgba(self) -> np.ndarray | None:
        if self._img_float is None:
            return None
        h, w = self._img_float.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if self._show_image_check.isChecked():
            base_u8 = _apply_stretch_gamma(self._img_float, self._black, self._white, self._gamma)
            rgba[..., :3] = np.repeat(base_u8[..., None], 3, axis=2)
            rgba[..., 3] = 255
        else:
            rgba[..., 3] = 255
        # Apply summary overlay and backplanes using same logic as compose
        # Summary
        if self._summary_rgba is not None and self._summary_enable.isChecked():
            s = self._summary_rgba
            if s.shape[0] == h and s.shape[1] == w:
                alpha = float(self._summary_alpha_slider.value()) / 1000.0
                s_rgb = s[..., :3]
                s_a = (s[..., 3].astype(np.float32) / 255.0) * alpha
                rgba = _alpha_blend_layer(rgba, s_rgb, s_a)
        # BODY_ID overlay
        if self._body_id_map is not None and self._body_id_enable.isChecked():
            idmap = self._body_id_map
            valid = idmap > 0
            if np.any(valid):
                ids = idmap.astype(np.int32)
                if self._body_id_mode_abs.isChecked():
                    id_min = 0
                    id_max = int(np.max(ids))
                else:
                    id_vals = ids[valid]
                    id_min = int(np.min(id_vals))
                    id_max = int(np.max(id_vals))
                if id_max <= id_min:
                    id_max = id_min + 1
                norm = np.zeros_like(ids, dtype=np.float32)
                norm[valid] = (ids[valid] - id_min) / float(id_max - id_min)
                cmap_name = self._body_id_cmap.currentData()
                cmap_obj = _load_colormap(cmap_name)
                if cmap_obj is not None:
                    rgb = (cmap_obj(norm)[..., :3] * 255.0).astype(np.uint8)
                else:
                    val = (norm * 255.0).astype(np.uint8)
                    rgb = np.stack([val, val, val], axis=-1)
                a = np.zeros((h, w), dtype=np.float32)
                a[valid] = float(self._body_id_alpha.value()) / 1000.0
                rgba = _alpha_blend_layer(rgba, rgb, a)

        # Body and Ring backplanes (shared helper)
        # Body
        body_name = self._body_combo.currentText()
        if self._body_show.isChecked() and body_name in self._bp_body_map and body_name != 'None':
            arr, units = self._bp_body_map[body_name]
            if arr.shape == (h, w):
                valid = self._valid_pixels(arr)
                rgba = self._composite_scalar_layer(
                    rgba,
                    arr,
                    units,
                    body_name,
                    valid_mask=valid,
                    mode=('Absolute' if self._body_mode_abs.isChecked() else 'Relative'),
                    alpha=float(self._body_alpha.value()) / 1000.0,
                    cmap_name=self._body_cmap.currentData(),
                )
        # Ring
        ring_name = self._ring_combo.currentText()
        if self._ring_show.isChecked() and ring_name in self._bp_ring_map and ring_name != 'None':
            arr, units = self._bp_ring_map[ring_name]
            if arr.shape == (h, w):
                valid = self._valid_pixels(arr)
                rgba = self._composite_scalar_layer(
                    rgba,
                    arr,
                    units,
                    ring_name,
                    valid_mask=valid,
                    mode=('Absolute' if self._ring_mode_abs.isChecked() else 'Relative'),
                    alpha=float(self._ring_alpha.value()) / 1000.0,
                    cmap_name=self._ring_cmap.currentData(),
                )
        return rgba

    # ---- Helpers ----
    def _populate_cmaps_if_needed(self) -> None:
        if self._cmap_items:
            return

        self._cmap_items = [
            ('Grayscale', 'gray'),
            ('Viridis (Perceptual)', 'viridis'),
            ('Plasma (Perceptual)', 'plasma'),
            ('Inferno (Perceptual)', 'inferno'),
            ('Magma (Perceptual)', 'magma'),
            ('Cividis (Colorblind-safe)', 'cividis'),
            ('Turbo (Rainbow-like)', 'turbo'),
            ('Coolwarm (Diverging)', 'coolwarm'),
            ('Spectral (Diverging)', 'Spectral'),
            ('Terrain', 'terrain'),
            ('Ocean', 'ocean'),
        ]

    def _populate_backplane_combos(self) -> None:
        # Preserve selections
        prev_body = self._body_combo.currentText() if self._body_combo.count() > 0 else None
        prev_ring = self._ring_combo.currentText() if self._ring_combo.count() > 0 else None
        self._body_combo.blockSignals(True)
        self._ring_combo.blockSignals(True)
        self._body_combo.clear()
        self._ring_combo.clear()
        for name in sorted(self._bp_body_map.keys()):
            self._body_combo.addItem(name)
        for name in sorted(self._bp_ring_map.keys()):
            self._ring_combo.addItem(name)
        # Restore previous if present
        if prev_body is not None:
            idx = self._body_combo.findText(prev_body)
            if idx >= 0:
                self._body_combo.setCurrentIndex(idx)
        if prev_ring is not None:
            idx = self._ring_combo.findText(prev_ring)
            if idx >= 0:
                self._ring_combo.setCurrentIndex(idx)
        self._body_combo.blockSignals(False)
        self._ring_combo.blockSignals(False)

    def _valid_pixels(self, arr: np.ndarray) -> np.ndarray:
        """Return the mask of pixels a backplane array actually measured.

        A pixel is measured when it is finite and is not the configured masked
        value.  The array itself carries that, so one rule serves both the body
        and the ring planes; deriving validity from ``BODY_ID_MAP`` instead
        would mark every ring pixel invalid, because the merge writes no ID for
        pixels the ring system wins.

        Parameters:
            arr: The backplane array to test.

        Returns:
            A boolean array, True where the pixel holds a measurement.
        """
        masked_value = float(self._config.backplanes.masked_value)
        return np.asarray(np.isfinite(arr) & (arr != masked_value))

    def _composite_scalar_layer(
        self,
        rgba: np.ndarray,
        arr: np.ndarray,
        units: str,
        name: str,
        *,
        valid_mask: np.ndarray,
        mode: str,
        alpha: float,
        cmap_name: Any,
    ) -> np.ndarray:
        h, w, _ = rgba.shape
        arr_disp = _rad_to_deg_if_units(name, units, arr)
        if mode == 'Absolute':
            vmin, vmax = _absolute_range_for(name, arr_disp)
        else:
            finite_vals = arr_disp[valid_mask]
            if finite_vals.size == 0:
                vmin, vmax = (0.0, 1.0)
            else:
                vmin = float(np.nanmin(finite_vals))
                vmax = float(np.nanmax(finite_vals))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
        with np.errstate(invalid='ignore', divide='ignore'):
            norm = np.clip((arr_disp - vmin) / (vmax - vmin), 0.0, 1.0)
        # Colormap
        cmap_obj = _load_colormap(cmap_name)
        if cmap_obj is not None:
            rgb = (cmap_obj(norm)[..., :3] * 255.0).astype(np.uint8)
        else:
            val = (norm * 255.0).astype(np.uint8)
            rgb = np.stack([val, val, val], axis=-1)
        a = np.zeros((h, w), dtype=np.float32)
        a[valid_mask] = float(alpha)
        return _alpha_blend_layer(rgba, rgb, a)

    def _on_body_selection_changed(self) -> None:
        name = self._body_combo.currentText()
        if not name:
            self._compose_and_display()
            return
        # Load prefs if any
        prefs = self._body_prefs.get(name)
        if prefs:
            self._body_alpha.blockSignals(True)
            self._body_cmap.blockSignals(True)
            alpha_val = float(prefs.get('alpha', 0.5))
            self._body_alpha.setValue(int(alpha_val * 1000))
            self._body_alpha_spin.setValue(alpha_val)
            # Restore mode radios
            mode = str(prefs.get('mode', 'Relative'))
            if mode.lower().startswith('abs'):
                self._body_mode_abs.setChecked(True)
            else:
                self._body_mode_rel.setChecked(True)
            cmap = prefs.get('cmap')
            if cmap is not None:
                # Find by data
                for i in range(self._body_cmap.count()):
                    if self._body_cmap.itemData(i) == cmap:
                        self._body_cmap.setCurrentIndex(i)
                        break
            self._body_alpha.blockSignals(False)
            self._body_cmap.blockSignals(False)
        self._compose_and_display()

    def _on_ring_selection_changed(self) -> None:
        name = self._ring_combo.currentText()
        if not name:
            self._compose_and_display()
            return
        prefs = self._ring_prefs.get(name)
        if prefs:
            self._ring_alpha.blockSignals(True)
            self._ring_cmap.blockSignals(True)
            alpha_val = float(prefs.get('alpha', 0.5))
            self._ring_alpha.setValue(int(alpha_val * 1000))
            self._ring_alpha_spin.setValue(alpha_val)
            mode = str(prefs.get('mode', 'Relative'))
            if mode.lower().startswith('abs'):
                self._ring_mode_abs.setChecked(True)
            else:
                self._ring_mode_rel.setChecked(True)
            cmap = prefs.get('cmap')
            if cmap is not None:
                for i in range(self._ring_cmap.count()):
                    if self._ring_cmap.itemData(i) == cmap:
                        self._ring_cmap.setCurrentIndex(i)
                        break
            self._ring_alpha.blockSignals(False)
            self._ring_cmap.blockSignals(False)
        self._compose_and_display()

    def _on_body_controls_changed(self) -> None:
        name = self._body_combo.currentText()
        if name:
            self._body_prefs[name] = {
                'alpha': float(self._body_alpha.value()) / 1000.0,
                'mode': ('Absolute' if self._body_mode_abs.isChecked() else 'Relative'),
                'cmap': self._body_cmap.currentData(),
            }
        self._compose_and_display()

    def _on_ring_controls_changed(self) -> None:
        name = self._ring_combo.currentText()
        if name:
            self._ring_prefs[name] = {
                'alpha': float(self._ring_alpha.value()) / 1000.0,
                'mode': ('Absolute' if self._ring_mode_abs.isChecked() else 'Relative'),
                'cmap': self._ring_cmap.currentData(),
            }
        self._compose_and_display()

    def _on_summary_alpha_changed(self, value: int) -> None:
        self._summary_alpha_spin.blockSignals(True)
        self._summary_alpha_spin.setValue(value / 1000.0)
        self._summary_alpha_spin.blockSignals(False)
        self._compose_and_display()

    def _on_summary_alpha_spin_changed(self, value: float) -> None:
        self._summary_alpha_slider.blockSignals(True)
        self._summary_alpha_slider.setValue(int(value * 1000))
        self._summary_alpha_slider.blockSignals(False)
        self._compose_and_display()

    def _on_body_id_alpha_changed(self, value: int) -> None:
        self._body_id_alpha_spin.blockSignals(True)
        self._body_id_alpha_spin.setValue(value / 1000.0)
        self._body_id_alpha_spin.blockSignals(False)
        self._compose_and_display()

    def _on_body_id_alpha_spin_changed(self, value: float) -> None:
        self._body_id_alpha.blockSignals(True)
        self._body_id_alpha.setValue(int(value * 1000))
        self._body_id_alpha.blockSignals(False)
        self._compose_and_display()

    def _on_body_alpha_changed(self, value: int) -> None:
        self._body_alpha_spin.blockSignals(True)
        self._body_alpha_spin.setValue(value / 1000.0)
        self._body_alpha_spin.blockSignals(False)
        self._on_body_controls_changed()

    def _on_body_alpha_spin_changed(self, value: float) -> None:
        self._body_alpha.blockSignals(True)
        self._body_alpha.setValue(int(value * 1000))
        self._body_alpha.blockSignals(False)
        self._on_body_controls_changed()

    def _on_ring_alpha_changed(self, value: int) -> None:
        self._ring_alpha_spin.blockSignals(True)
        self._ring_alpha_spin.setValue(value / 1000.0)
        self._ring_alpha_spin.blockSignals(False)
        self._on_ring_controls_changed()

    def _on_ring_alpha_spin_changed(self, value: float) -> None:
        self._ring_alpha.blockSignals(True)
        self._ring_alpha.setValue(int(value * 1000))
        self._ring_alpha.blockSignals(False)
        self._on_ring_controls_changed()

    # ---- Event handlers: pan/zoom ----
    def _on_press(self, event: QMouseEvent) -> None:
        self._zoom_ctl.on_mouse_press(event)

    def _on_move(self, event: QMouseEvent) -> None:
        self._zoom_ctl.on_mouse_move(event)
        # Status update of value at cursor
        self._update_cursor_status(event.pos())

    def _on_release(self, _event: QMouseEvent) -> None:
        self._zoom_ctl.on_mouse_release(_event)

    def _on_wheel(self, event: QWheelEvent) -> None:
        self._zoom_ctl.on_wheel(event)

    def _zoom_in(self) -> None:
        # Centre-anchored zoom is provided by the controller (it wraps this
        # window's scroll area); delegate instead of re-implementing it.
        self._zoom_ctl.zoom_in_center()

    def _zoom_out(self) -> None:
        self._zoom_ctl.zoom_out_center()

    def _reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._zoom_label.setText('Zoom: 1.00x')
        self._compose_and_display()

    def _toggle_zoom_sharp(self, state: Any) -> None:
        self._zoom_sharp = state == int(cast(int, Qt.CheckState.Checked.value))
        self._compose_and_display()

    def _zoom_at_point(
        self,
        factor: float,
        viewport_x: int,
        viewport_y: int,
        scaled_x: float,
        scaled_y: float,
    ) -> None:
        # The ZoomPanController owns the zoom clamp and the no-op-on-unchanged
        # short-circuit (identical logic), so delegate directly rather than
        # re-deriving the clamped zoom here with duplicated limit constants.
        self._zoom_ctl.zoom_at_point(factor, viewport_x, viewport_y, scaled_x, scaled_y)

    # ---- Compose & Display ----
    def _compose_and_display(self) -> None:
        if self._img_float is None:
            return
        # Base grayscale
        h, w = self._img_float.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if self._show_image_check.isChecked():
            base_u8 = _apply_stretch_gamma(self._img_float, self._black, self._white, self._gamma)
            rgba[..., 0] = base_u8
            rgba[..., 1] = base_u8
            rgba[..., 2] = base_u8
            rgba[..., 3] = 255
        else:
            rgba[..., 3] = 255

        # Summary overlay (alpha from slider)

        # Summary overlay
        if self._summary_rgba is not None and self._summary_enable.isChecked():
            s = self._summary_rgba
            if s.shape[0] == h and s.shape[1] == w:
                alpha = float(self._summary_alpha_slider.value()) / 1000.0
                s_rgb = s[..., :3].astype(np.float32)
                s_a = (s[..., 3].astype(np.float32) / 255.0) * alpha
                # Use the shared blend so the on-screen and saved-PNG paths
                # (_render_full_rgba) cannot diverge (CODE-UI-001).
                rgba = _alpha_blend_layer(rgba, s_rgb, s_a)

        # BODY_ID overlay
        if self._body_id_map is not None and self._body_id_enable.isChecked():
            idmap = self._body_id_map
            valid = idmap > 0
            ids = idmap.astype(np.int32)
            if np.any(valid):
                mode_id = 'Absolute' if self._body_id_mode_abs.isChecked() else 'Relative'
                if mode_id == 'Absolute':
                    id_min = 0
                    id_max = int(np.max(ids))
                else:
                    # Relative: ignore zeros using valid mask
                    id_vals = ids[valid]
                    id_min = int(np.min(id_vals))
                    id_max = int(np.max(id_vals))
                if id_max <= id_min:
                    id_max = id_min + 1
                norm = np.zeros_like(ids, dtype=np.float32)
                norm[valid] = (ids[valid] - id_min) / float(id_max - id_min)
                cmap_name = self._body_id_cmap.currentData()
                # Resolve via the shared helper (same as the save path), instead
                # of the inline fallback that used the removed matplotlib
                # ``cm.get_cmap`` API (CODE-UI-001).
                cmap_obj = _load_colormap(cmap_name)
                if cmap_obj is not None:
                    rgb = (cmap_obj(norm)[..., :3] * 255.0).astype(np.uint8)
                else:
                    val = (norm * 255.0).astype(np.uint8)
                    rgb = np.stack([val, val, val], axis=-1)
                a = np.zeros((h, w), dtype=np.float32)
                a[valid] = float(self._body_id_alpha.value()) / 1000.0
                # Shared blend (CODE-UI-001): keep the on-screen and saved-PNG
                # composites identical.
                rgba = _alpha_blend_layer(rgba, rgb.astype(np.float32), a)

        # Selected Body backplane
        body_name = self._body_combo.currentText()
        if self._body_show.isChecked() and body_name in self._bp_body_map and body_name != 'None':
            arr, units = self._bp_body_map[body_name]
            if arr.shape == (h, w):
                valid = self._valid_pixels(arr)
                rgba = self._composite_scalar_layer(
                    rgba,
                    arr,
                    units,
                    body_name,
                    valid_mask=valid,
                    mode=('Absolute' if self._body_mode_abs.isChecked() else 'Relative'),
                    alpha=float(self._body_alpha.value()) / 1000.0,
                    cmap_name=self._body_cmap.currentData(),
                )

        # Selected Ring backplane
        ring_name = self._ring_combo.currentText()
        if self._ring_show.isChecked() and ring_name in self._bp_ring_map and ring_name != 'None':
            arr, units = self._bp_ring_map[ring_name]
            if arr.shape == (h, w):
                valid = self._valid_pixels(arr)
                rgba = self._composite_scalar_layer(
                    rgba,
                    arr,
                    units,
                    ring_name,
                    valid_mask=valid,
                    mode=('Absolute' if self._ring_mode_abs.isChecked() else 'Relative'),
                    alpha=float(self._ring_alpha.value()) / 1000.0,
                    cmap_name=self._ring_cmap.currentData(),
                )

        # Scale/translate for zoom/pan
        # Ensure buffer lifetime while QImage uses it
        self._last_rgba = rgba
        qimg = QImage(
            bytes(self._last_rgba.data),
            w,
            h,
            self._last_rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(qimg)
        if self._zoom_factor != 1.0:
            # Prefer smooth transform only when not sharp
            transform_mode = (
                Qt.TransformationMode.FastTransformation
                if self._zoom_sharp
                else Qt.TransformationMode.SmoothTransformation
            )
            pixmap = pixmap.scaled(
                int(w * self._zoom_factor),
                int(h * self._zoom_factor),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                transform_mode,
            )
        # Pan by setting margins via QLabel alignment offset approximation: use contents margins
        self._label.setPixmap(pixmap)
        # Ensure label matches pixmap size so it is not displayed tiny
        try:
            self._label.resize(pixmap.size())
            self._label.adjustSize()
        except Exception:
            pass
        self._zoom_label.setText(f'Zoom: {self._zoom_factor:.2f}x')

    # ---- Cursor sampling ----
    def _update_cursor_status(self, pos: QPoint) -> None:
        if self._img_float is None:
            return
        pixmap = self._label.pixmap()
        if pixmap is None:
            return
        # Approximate inverse of zoom/pan to get image coords
        # Since we didn't actually change widget offset, map pos to image by dividing by zoom
        u = int(pos.x() / max(self._zoom_factor, 1e-6))
        v = int(pos.y() / max(self._zoom_factor, 1e-6))
        h, w = self._img_float.shape
        if v < 0 or v >= h or u < 0 or u >= w:
            self._status_label.setText('V, U: --, --  Value: --')
            self._body_id_val_label.setText('Object: --')
            self._body_val_label.setText('Value: --')
            self._ring_val_label.setText('Value: --')
            return
        val = float(self._img_float[v, u])
        self._status_label.setText(f'V, U: {v}, {u}  Value: {val:.6g}')
        # Update BODY_ID value regardless of visibility
        if (
            self._body_id_map is not None
            and 0 <= v < self._body_id_map.shape[0]
            and 0 <= u < self._body_id_map.shape[1]
        ):
            bid = int(self._body_id_map[v, u])
            try:
                bid_name = cspyce.bodc2n(bid)
            except Exception:
                bid_name = 'unknown'
            self._body_id_val_label.setText(
                f'Object: {bid_name} ({bid})' if bid != 0 else 'Object: --'
            )
        else:
            self._body_id_val_label.setText('Object: --')
        # Update selected body/ring labels
        body_name = self._body_combo.currentText()
        if body_name in self._bp_body_map:
            arr, units = self._bp_body_map[body_name]
            if 0 <= v < arr.shape[0] and 0 <= u < arr.shape[1] and np.isfinite(arr[v, u]):
                val = _rad_to_deg_if_units(body_name, units, np.asarray(arr[v, u])).item()
                self._body_val_label.setText(f'Value: {val:.6g}')
            else:
                self._body_val_label.setText('Value: --')
        else:
            self._body_val_label.setText('Value: --')
        ring_name = self._ring_combo.currentText()
        if ring_name in self._bp_ring_map:
            arr, units = self._bp_ring_map[ring_name]
            if 0 <= v < arr.shape[0] and 0 <= u < arr.shape[1] and np.isfinite(arr[v, u]):
                val = _rad_to_deg_if_units(ring_name, units, np.asarray(arr[v, u])).item()
                self._ring_val_label.setText(f'Value: {val:.6g}')
            else:
                self._ring_val_label.setText('Value: --')
        else:
            self._ring_val_label.setText('Value: --')

    # ---- Controls ----
    def _on_black_changed(self, val: float) -> None:
        self._black = float(val)
        self._updater.request_update()

    def _on_white_changed(self, val: float) -> None:
        self._white = float(val)
        self._updater.request_update()

    def _on_gamma_changed(self, val: float) -> None:
        self._gamma = float(val)
        self._updater.request_update()

    def _prev_image(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._load_group(self._current_index)

    def _next_image(self) -> None:
        if self._current_index + 1 < len(self._image_groups):
            self._current_index += 1
            self._load_group(self._current_index)


# ---- CLI & main ----
def parse_args(command_list: list[str]) -> argparse.Namespace:
    if len(command_list) < 1:
        print('Usage: python3 sd_backplane_viewer.py <dataset_name> [args]')
        sys.exit(1)

    dataset_name = command_list[0].lower()
    if dataset_name not in dataset_names():
        print(f'Unknown dataset "{dataset_name}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python3 sd_backplane_viewer.py <dataset_name> [args]')
        sys.exit(1)

    try:
        dataset = dataset_name_to_class(dataset_name)()
    except KeyError:
        print(f'Unknown dataset "{dataset_name}"')
        print(f'Valid datasets are: {", ".join(dataset_names())}')
        print('Usage: python3 sd_backplane_viewer.py <dataset_name> [args]')
        sys.exit(1)

    cmdparser = argparse.ArgumentParser(description='Backplane Viewer GUI')

    env = cmdparser.add_argument_group('Environment')
    env.add_argument(
        '--config-file',
        action='append',
        default=None,
        help='Configuration override file(s) (default: ./nav_default_config.yaml)',
    )
    env.add_argument(
        '--pds3-holdings-root',
        type=str,
        default=None,
        help='Root directory of PDS3 holdings; overrides env/config',
    )
    env.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help='Root containing navigation outputs (metadata, summary.png)',
    )
    env.add_argument(
        '--backplane-results-root',
        type=str,
        default=None,
        help='Root containing backplane outputs (FITS, XML)',
    )

    # Dataset selection args
    dataset.add_selection_arguments(cmdparser)

    args = cmdparser.parse_args(command_list[1:])
    args._dataset_name = dataset_name
    args._dataset = dataset
    return args


def main() -> None:
    command_list = sys.argv[1:]
    arguments = parse_args(command_list)

    # Config via the shared loader (single source of truth for config / CLI /
    # env precedence), matching the other drivers.
    load_default_and_user_config(arguments, DEFAULT_CONFIG)

    # Roots
    nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    dataset: DataSet = arguments._dataset

    image_groups: list[ImageFiles] = list(dataset.yield_image_files_from_arguments(arguments))
    if not image_groups:
        print('No images matched selection.')
        sys.exit(1)

    inst_name = dataset_name_to_inst_name(arguments._dataset_name)

    # Ensure QApplication exists
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication([])
        created = True

    dlg = NavBackplaneViewer(
        dataset=dataset,
        inst_name=inst_name,
        image_groups=image_groups,
        nav_results_root=nav_results_root,
        backplane_results_root=backplane_results_root,
        config=DEFAULT_CONFIG,
    )
    dlg.show()
    rc = app.exec()
    if created:
        app.quit()
    sys.exit(rc)


if __name__ == '__main__':
    main()
