"""Interactive manual-navigation dialog (PyQt6).

Renders a composite of the predicted scene against the source image and
lets an operator pick a translation offset by hand.  An ``Auto`` button
runs the same masked-NCC pyramid that correlation-based techniques use,
so the operator can either accept the auto-pick or override it.

This module is GUI-only; no autonomous-pipeline code imports it.  It is
constructed by ``NavTechniqueManual`` (which is opted out of the
auto-discovery registry) on demand.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from filecache import FCPath
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from spindoctor._version import __version__ as _spindoctor_version
from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.obs import ObsSnapshot
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.correlate import masked_ncc, navigate_with_pyramid_kpeaks
from spindoctor.support.image import apply_linear_gamma_stretch
from spindoctor.support.summary_png import render_annotated_summary_rgb
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint8Type
from spindoctor.ui.common import (
    ZoomPanController,
    build_stretch_controls,
)
from spindoctor.ui.library_entry import build_sidecar_yaml, compute_image_url, infer_obs_metadata

# Correlation map popup (_CorrMapDialog) layout sizing
_CORR_MAP_MIN_WIDTH = 550
_CORR_MAP_MIN_HEIGHT = 520
_CORR_MAP_MARGIN = 4
_CORR_MAP_SPACING = 2

# Semi-transparent white tint when overlaying the binary model mask on RGB
_MASK_OVERLAY_ALPHA = 0.4


def _apply_stretch_gamma(
    image: NDArrayFloatType, black: float, white: float, gamma: float
) -> NDArrayUint8Type:
    """Apply black/white/gamma to a float image and return uint8 mono."""
    scaled = apply_linear_gamma_stretch(image, black=black, white=white, gamma=gamma)
    return cast(NDArrayUint8Type, (scaled * 255.0).astype(np.uint8))


def _bilinear_interpolate_fov(arr: NDArrayFloatType, img_v: float, img_u: float) -> float:
    """Bilinear sample a 2D FOV array at the pixel-corner position ``(img_v, img_u)``.

    The caller states the position the way a cursor and the geometry layer state
    one, in pixel corner coordinates, and the array is addressed pixel centric, so
    the half pixel between the two comes off here -- once, where the value crosses
    between the stages.  Reading a pixel corner position straight as an array
    coordinate would return the array half a pixel down and right of the position
    asked for.

    Parameters:
        arr: 2-D float image in FOV coordinates.
        img_v: V position in pixel corner coordinates.
        img_u: U position in pixel corner coordinates.

    Returns:
        Interpolated sample as a scalar ``float``.
    """
    h, w = arr.shape
    centric_v = img_v - PIXEL_CENTER_TO_CORNER_PX
    centric_u = img_u - PIXEL_CENTER_TO_CORNER_PX
    v0 = min(max(math.floor(centric_v), 0), h - 1)
    u0 = min(max(math.floor(centric_u), 0), w - 1)
    v1 = min(v0 + 1, h - 1)
    u1 = min(u0 + 1, w - 1)
    dv = min(max(centric_v - v0, 0.0), 1.0)
    du = min(max(centric_u - u0, 0.0), 1.0)
    return float(
        arr[v0, u0] * (1 - du) * (1 - dv)
        + arr[v0, u1] * du * (1 - dv)
        + arr[v1, u0] * (1 - du) * dv
        + arr[v1, u1] * du * dv
    )


def _bilinear_sample_periodic(arr: NDArrayFloatType, y: float, x: float) -> float:
    """Periodic bilinear sample on 2D array arr at float indices (y, x)."""
    h, w = arr.shape
    # Wrap
    x = x % w
    y = y % h
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    dx = x - x0
    dy = y - y0
    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    return float(
        v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) + v10 * (1 - dx) * dy + v11 * dx * dy
    )


class _CorrMapDialog(QDialog):
    """Modal popup showing the full 2-D normalized cross-correlation (NCC) surface.

    The NCC surface is rearranged via ``np.fft.fftshift`` for display. The global
    peak marker uses ``np.argmax`` on the NCC surface itself, restricted (if
    ``max_offset_vu`` is provided) to the physically plausible offset range, so
    it matches the peak that the pyramid auto-search would find. Axes are labeled
    in offset pixels (dU for columns, dV for rows). The offset supplied at
    construction time is overlaid as a red cross marker and displayed in the
    legend.
    """

    def __init__(
        self,
        *,
        corr_surface: NDArrayFloatType,
        dv: float,
        du: float,
        max_offset_vu: tuple[int, int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Correlation Map')
        self.setMinimumSize(_CORR_MAP_MIN_WIDTH, _CORR_MAP_MIN_HEIGHT)

        layout = QVBoxLayout(self)
        m = _CORR_MAP_MARGIN
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(_CORR_MAP_SPACING)

        h, w = corr_surface.shape
        shifted = np.fft.fftshift(corr_surface)

        # Map FFT index ranges to signed offset extents for imshow axes.
        du_min = -(w // 2)
        du_max = (w - 1) // 2
        dv_min = -(h // 2)
        dv_max = (h - 1) // 2

        # Locate the global peak from the NCC surface; restrict the search to
        # the same signed-offset window the pyramid uses so the highlighted peak
        # matches what an auto-navigate would pick.
        search = shifted
        if max_offset_vu is not None:
            search = shifted.copy()
            dv_range = np.arange(dv_min, dv_max + 1)
            du_range = np.arange(du_min, du_max + 1)
            search[np.abs(dv_range) > max_offset_vu[0], :] = -np.inf
            search[:, np.abs(du_range) > max_offset_vu[1]] = -np.inf
        peak_row, peak_col = np.unravel_index(np.argmax(search), search.shape)
        peak_dv = dv_min + int(peak_row)
        peak_du = du_min + int(peak_col)

        fig = Figure(figsize=(5.8, 5.0))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.10)
        canvas = FigureCanvasQTAgg(fig)  # type: ignore[no-untyped-call]
        toolbar = NavigationToolbar2QT(canvas, self)  # type: ignore[no-untyped-call]
        ax = fig.add_subplot(111)

        # extent=[left, right, bottom, top] with origin='upper' places dv_min at
        # the top and dv_max at the bottom, consistent with image-row convention.
        extent_arg: tuple[float, float, float, float] = (
            du_min - 0.5,
            du_max + 0.5,
            dv_max + 0.5,
            dv_min - 0.5,
        )
        im = ax.imshow(
            shifted,
            origin='upper',
            extent=extent_arg,
            aspect='equal',
            cmap='viridis',
            interpolation='nearest',
        )
        fig.colorbar(im, ax=ax, label='Normalized Cross-Correlation (NCC)')

        ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='white', linewidth=0.5, alpha=0.5)

        # Circle around peak; radius scaled to ~4 % of the shorter axis.
        radius = max(2, min(h, w) * 0.04)
        ax.add_patch(
            Circle(
                (peak_du, peak_dv),
                radius=radius,
                fill=False,
                edgecolor='lime',
                linewidth=1.5,
            )
        )
        ax.annotate(
            f'peak  dV={peak_dv:+d}, dU={peak_du:+d}',
            xy=(peak_du, peak_dv),
            xytext=(8, 8),
            textcoords='offset points',
            color='lime',
            fontsize=8,
        )

        ax.plot(
            du,
            dv,
            'r+',
            markersize=14,
            markeredgewidth=2,
            label=f'current  dV={dv:.3f}, dU={du:.3f}',
        )
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        ax.set_xlabel('dU (column offset, pixels)')
        ax.set_ylabel('dV (row offset, pixels)\n[positive = down]')
        ax.set_title('Normalized Cross-Correlation Map')

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)


class _ImageLabel(QLabel):
    """Image label that forwards input events to the dialog handlers."""

    def __init__(self, owner_dialog: ManualNavDialog) -> None:
        super().__init__()
        self._owner = owner_dialog

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_release(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            self._owner._on_wheel(event)


class ManualNavDialog(QDialog):
    """Manual navigation dialog for overlaying image and combined model.

    The viewport uses false color (image in red/blue, blend in green). Separate
    **Image Stretch** and **Model Stretch** groups each provide black/white/gamma
    and a display toggle. Model transparency and an optional binary mask overlay
    sit under model stretch.
    """

    def __init__(
        self,
        *,
        obs: ObsSnapshot,
        model_img_ext: NDArrayFloatType,
        model_mask_ext: NDArrayBoolType,
        config: Config | None,
        annotations: Annotations | None = None,
        constraint_normal_seed_deg: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the manual-navigation dialog.

        Parameters:
            obs: The observation snapshot under navigation.
            model_img_ext: Composite predicted-scene image in ext-FOV
                coordinates (the union of every NavModel template painted
                Z-buffer-style by subject range).
            model_mask_ext: Boolean mask matching ``model_img_ext``;
                True wherever the composite carries signal.
            config: Optional ``Config`` override.
            annotations: Merged ``Annotations`` collection (one entry
                per NavModel that contributed).  Drawn at the chosen
                ``(dv, du)`` into the labelled summary PNG written
                alongside a saved sidecar.  ``None`` (or empty) is
                permitted; the saved PNG then carries the source-image
                grayscale alone.
            constraint_normal_seed_deg: Optional seed for the
                constrained-drag normal angle (degrees in the ``(v, u)``
                frame, ``+v`` toward ``+u``).  Supplied by the caller when
                the scene's observable axis is known — e.g. the aggregate
                edge normal of an all-straight flat-ring scene — so the
                operator can toggle the constraint on without measuring
                the angle by eye.  ``None`` leaves the angle at 0.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Manual Navigation')
        self.setMinimumSize(1200, 800)

        self._obs = obs
        self._config = config
        self._annotations: Annotations = annotations if annotations is not None else Annotations()

        self.setWindowTitle(f'Manual Navigation - {obs.abspath.name}')

        # Image and model arrays
        self._img_fov = obs.data  # V x U, float64
        self._img_ext = obs.extdata  # for correlation
        if model_img_ext.shape != model_mask_ext.shape:
            raise ValueError(
                f'model_img_ext shape {model_img_ext.shape} does not match '
                f'model_mask_ext shape {model_mask_ext.shape}'
            )
        self._model_img_ext = model_img_ext
        self._model_mask_ext = model_mask_ext

        # Stretch/gamma parameters (image)
        self._image_black = float(np.quantile(self._img_fov, 0.001))
        self._image_white = float(np.quantile(self._img_fov, 0.999))
        if self._image_black >= self._image_white:
            self._image_white = self._image_black + 0.01
        self._image_gamma = 1.0
        # Model stretch (defaults from model at offset (0, 0))
        _m0 = np.asarray(
            self._obs.extract_offset_array(self._model_img_ext, (0.0, 0.0)),
            dtype=np.float64,
        )
        self._model_stretch_min = float(np.min(_m0))
        self._model_stretch_max = float(np.max(_m0))
        if self._model_stretch_min >= self._model_stretch_max:
            self._model_stretch_max = self._model_stretch_min + 1e-6
        self._model_black = float(np.quantile(_m0, 0.001))
        self._model_white = float(np.quantile(_m0, 0.999))
        if self._model_black >= self._model_white:
            self._model_white = self._model_black + 1e-6
        self._model_gamma = 1.0
        # 0 = fully opaque model in green blend, 1 = fully transparent (image only)
        self._model_transparency = 0.5
        self._show_image = True  # show observation in R/B (and base of G)
        self._show_model = True  # show model contribution in green blend
        self._show_mask = False  # overlay model mask as a binary white tint
        # For slider mapping (image)
        self._stretch_min = float(np.min(self._img_fov))
        self._stretch_max = float(np.max(self._img_fov))

        # Offsets (dv, du)
        self._dv = 0.0
        self._du = 0.0

        # Constrained-drag state (rank-1 scenes): when enabled, drags move
        # the offset only along the normal at ``_constraint_angle_deg``.
        self._constraint_seed_deg = constraint_normal_seed_deg

        # Zoom/pan state
        self._zoom = 1.0
        self._drag_start_pos: QPoint | None = None
        self._drag_mode: str | None = None  # 'offset' (right)
        self._drag_start_offset: tuple[float, float] | None = None
        # Zoom rendering mode
        self._zoom_sharp = True

        # Model slice in FOV coords (aligned with _img_fov); for status bar sampling
        self._model_slice_fov: NDArrayFloatType | None = None

        # Precompute correlation surface once for status bar display
        self._precompute_correlation_surface()

        # Build UI
        self._build_ui()
        self._refresh_overlay()

    # ---- Correlation helpers ----

    def _precompute_correlation_surface(self) -> None:
        """Compute masked NCC surface on padded arrays; reuse for sampling."""
        image = np.asarray(self._img_ext, dtype=np.float64)
        model = np.asarray(self._model_img_ext, dtype=np.float64)
        mask = np.asarray(self._model_mask_ext, dtype=bool)
        # Use the bi-directional (data-mask aware) NCC so the surface the user
        # sees -- and the peak the Auto button finds -- matches the
        # composite-template stream this dialog consumes from
        # ``compose_template_features``; without data_mask the zero-padded
        # extfov margin biases the peak toward |dV| = extfov_margin_v.
        data_mask = self._obs.extfov_data_sensor_mask()
        self._corr_surface, _ = masked_ncc(image, model, mask, data_mask=data_mask)
        self._corr_h, self._corr_w = self._corr_surface.shape

    def _offset_to_corr_indices(self, dv: float, du: float) -> tuple[float, float]:
        """Map signed (dv, du) to correlation surface indices for sampling."""
        # Same mapping as int_to_signed inverse: idx = s if s >= 0 else s + size
        y = dv if dv >= 0 else dv + self._corr_h
        x = du if du >= 0 else du + self._corr_w
        return y, x

    def _current_corr_value(self) -> float:
        y, x = self._offset_to_corr_indices(self._dv, self._du)
        return _bilinear_sample_periodic(self._corr_surface, y, x)

    # ---- UI construction ----

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left: image viewport with zoom controls
        left = QVBoxLayout()
        zoom_row = QHBoxLayout()
        self._btn_zoom_out = QPushButton('Zoom -')
        self._btn_zoom_in = QPushButton('Zoom +')
        self._btn_reset = QPushButton('Reset View')
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._btn_zoom_out.clicked.connect(self._zoom_out_center)
        self._btn_zoom_in.clicked.connect(self._zoom_in_center)
        self._btn_reset.clicked.connect(self._reset_view)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        # Prevent Enter from triggering zoom buttons; keep them out of focus chain
        for btn in (self._btn_zoom_out, self._btn_zoom_in, self._btn_reset):
            try:
                btn.setAutoDefault(False)
                btn.setDefault(False)
            except Exception:
                pass
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        zoom_row.addStretch()
        zoom_row.addWidget(self._btn_zoom_out)
        zoom_row.addWidget(self._btn_zoom_in)
        zoom_row.addWidget(self._zoom_sharp_check)
        zoom_row.addWidget(self._btn_reset)
        zoom_row.addStretch()
        left.addLayout(zoom_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setMinimumSize(700, 700)
        self._scroll.setStyleSheet('background-color: black;')
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = _ImageLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet('background-color: black;')
        self._label.setMouseTracking(True)
        self._scroll.setWidget(self._label)

        left.addWidget(self._scroll)
        layout.addLayout(left, stretch=2)

        # Status bar (within dialog)
        status = QStatusBar()
        self._status_label = QLabel('V, U: --, --  Image: --  Model: --  Correlation: --')
        self._zoom_label = QLabel('Zoom: 1.00x')
        status.addWidget(self._status_label)
        status.addPermanentWidget(self._zoom_label)
        left.addWidget(status)

        # Right: controls
        right = QVBoxLayout()
        # Image Stretch
        stretch_group = QGroupBox('Image Stretch')
        stretch_form = QFormLayout()
        self._check_show_image = QCheckBox('Display image')
        self._check_show_image.setChecked(True)
        self._check_show_image.setToolTip(
            'Show the stretched observation in red/blue channels (and as the base in green).'
        )
        self._check_show_image.stateChanged.connect(self._on_show_image_changed)
        stretch_form.addRow(self._check_show_image)
        controls = build_stretch_controls(
            stretch_form,
            img_min=self._stretch_min,
            img_max=self._stretch_max,
            black_init=self._image_black,
            white_init=self._image_white,
            gamma_init=self._image_gamma,
            on_black_changed=self._on_image_black_changed,
            on_white_changed=self._on_image_white_changed,
            on_gamma_changed=self._on_image_gamma_changed,
            value_label_min_width=52,
            slider_horizontal_stretch=1,
        )
        self._slider_image_black = controls['slider_black']
        self._slider_image_white = controls['slider_white']
        self._slider_image_gamma = controls['slider_gamma']
        self._lbl_image_black = controls['label_black']
        self._lbl_image_white = controls['label_white']
        self._lbl_image_gamma = controls['label_gamma']
        self._stretch_controls = controls
        self._btn_reset_stretch = QPushButton('Reset Image Stretch')
        self._btn_reset_stretch.clicked.connect(self._on_reset_stretch)
        reset_img_row = QHBoxLayout()
        reset_img_row.addStretch(1)
        reset_img_row.addWidget(self._btn_reset_stretch)
        reset_img_row.addStretch(1)
        reset_img_holder = QWidget()
        reset_img_holder.setLayout(reset_img_row)
        stretch_form.addRow(reset_img_holder)
        stretch_group.setLayout(stretch_form)
        right.addWidget(stretch_group)

        # Model Stretch (same controls as image + transparency)
        model_stretch_group = QGroupBox('Model Stretch')
        model_stretch_form = QFormLayout()
        self._check_show_model = QCheckBox('Display model')
        self._check_show_model.setChecked(True)
        self._check_show_model.setToolTip(
            'Show the stretched navigation model in the green channel (blended with the image).'
        )
        self._check_show_model.stateChanged.connect(self._on_show_model_changed)
        model_stretch_form.addRow(self._check_show_model)
        self._check_show_mask = QCheckBox('Display mask')
        self._check_show_mask.setChecked(False)
        self._check_show_mask.setToolTip(
            'Overlay the model mask (binary) as a semi-transparent white tint on the image.'
        )
        self._check_show_mask.stateChanged.connect(self._on_show_mask_changed)
        model_stretch_form.addRow(self._check_show_mask)
        model_controls = build_stretch_controls(
            model_stretch_form,
            img_min=self._model_stretch_min,
            img_max=self._model_stretch_max,
            black_init=self._model_black,
            white_init=self._model_white,
            gamma_init=self._model_gamma,
            on_black_changed=self._on_model_black_changed,
            on_white_changed=self._on_model_white_changed,
            on_gamma_changed=self._on_model_gamma_changed,
            value_label_min_width=52,
            slider_horizontal_stretch=1,
        )
        self._slider_model_black = model_controls['slider_black']
        self._slider_model_white = model_controls['slider_white']
        self._slider_model_gamma = model_controls['slider_gamma']
        self._lbl_model_black = model_controls['label_black']
        self._lbl_model_white = model_controls['label_white']
        self._lbl_model_gamma = model_controls['label_gamma']
        self._model_stretch_controls = model_controls
        self._slider_model_transparency = QSlider(Qt.Orientation.Horizontal)
        self._slider_model_transparency.setRange(0, 100)
        self._slider_model_transparency.setValue(round(self._model_transparency * 100))
        self._lbl_model_transparency = QLabel(f'{self._model_transparency:.2f}')
        self._lbl_model_transparency.setMinimumWidth(52)
        self._lbl_model_transparency.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._slider_model_transparency.valueChanged.connect(
            lambda v: self._on_model_transparency_changed(v / 100.0)
        )
        self._slider_model_transparency.setToolTip(
            '0 = fully opaque model, 1 = fully transparent (image only in the blend).'
        )
        row_a = QHBoxLayout()
        row_a.setSpacing(4)
        row_a.addWidget(self._slider_model_transparency, stretch=1)
        row_a.addWidget(self._lbl_model_transparency)
        model_stretch_form.addRow('Model transparency:', row_a)
        self._btn_reset_model_stretch = QPushButton('Reset Model Stretch')
        self._btn_reset_model_stretch.clicked.connect(self._on_reset_model_stretch)
        reset_model_row = QHBoxLayout()
        reset_model_row.addStretch(1)
        reset_model_row.addWidget(self._btn_reset_model_stretch)
        reset_model_row.addStretch(1)
        reset_model_holder = QWidget()
        reset_model_holder.setLayout(reset_model_row)
        model_stretch_form.addRow(reset_model_holder)
        model_stretch_group.setLayout(model_stretch_form)
        right.addWidget(model_stretch_group)

        # Offsets
        offset_group = QGroupBox('Offset (pixels)')
        offset_form = QFormLayout()
        # V and U with 0.001 precision
        self._spin_dv = QDoubleSpinBox()
        self._spin_du = QDoubleSpinBox()
        self._spin_dv.setDecimals(3)
        self._spin_du.setDecimals(3)
        # Bounds based on extfov margins
        self._spin_dv.setRange(-self._obs.extfov_margin_v, self._obs.extfov_margin_v)
        self._spin_du.setRange(-self._obs.extfov_margin_u, self._obs.extfov_margin_u)
        self._spin_dv.setSingleStep(0.1)
        self._spin_du.setSingleStep(0.1)
        self._spin_dv.setValue(self._dv)
        self._spin_du.setValue(self._du)
        self._spin_dv.valueChanged.connect(self._on_spin_dv)
        self._spin_du.valueChanged.connect(self._on_spin_du)
        offset_form.addRow('dV (rows):', self._spin_dv)
        offset_form.addRow('dU (cols):', self._spin_du)
        # Constrained drag (rank-1 scenes): restrict right-drag motion to the
        # normal direction so the operator cannot move the model along an
        # axis the scene does not constrain (a straight ring edge's tangent).
        self._chk_constrain = QCheckBox('Constrain drag to normal')
        self._chk_constrain.setChecked(False)
        self._chk_constrain.setToolTip(
            'Restrict right-drag to the normal direction below. Use for '
            'rank-1 scenes (straight ring edges) where only the edge-normal '
            'offset component is observable; the saved sidecar then carries '
            'a ground_truth.constraint block.'
        )
        self._spin_constraint_angle = QDoubleSpinBox()
        self._spin_constraint_angle.setDecimals(2)
        self._spin_constraint_angle.setRange(-180.0, 180.0)
        self._spin_constraint_angle.setSingleStep(1.0)
        self._spin_constraint_angle.setValue(
            self._constraint_seed_deg if self._constraint_seed_deg is not None else 0.0
        )
        self._spin_constraint_angle.setToolTip(
            'Normal direction in the (v, u) frame, measured from +v toward '
            '+u; the unit normal is (cos, sin). Seeded from the aggregate '
            'ring-edge normal when the scene is all-straight.'
        )
        offset_form.addRow(self._chk_constrain)
        offset_form.addRow('Normal angle (deg):', self._spin_constraint_angle)
        offset_group.setLayout(offset_form)
        right.addWidget(offset_group)

        # Buttons: Correlation Map, Auto, Save Library Entry, OK/Cancel
        btn_row = QHBoxLayout()
        self._btn_corr_map = QPushButton('Correlation Map...')
        self._btn_auto = QPushButton('Auto')
        self._btn_save_library = QPushButton('Save as Library Entry...')
        self._btn_ok = QPushButton('OK')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_corr_map.clicked.connect(self._on_show_corr_map)
        self._btn_auto.clicked.connect(self._on_auto)
        self._btn_save_library.clicked.connect(self._on_save_library_entry)
        self._btn_ok.clicked.connect(self.accept)
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_corr_map)
        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_save_library)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_cancel)
        right.addLayout(btn_row)
        right.addStretch(1)

        layout.addLayout(right, stretch=1)
        # Initialize zoom/pan controller for left-pan and wheel zoom
        self._zoom_ctl = ZoomPanController(
            label=self._label,
            scroll_area=self._scroll,
            get_zoom=lambda: self._zoom,
            set_zoom=lambda z: setattr(self, '_zoom', float(z)),
            update_display=self._update_display_only,
            set_zoom_label_text=lambda s: self._zoom_label.setText(s),
        )

    # ---- Event handlers ----

    @staticmethod
    def _is_checked(state: Any) -> bool:
        """Return True if ``state`` is the Qt Checked checkbox state."""
        return Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_image_black_changed(self, val: float) -> None:
        self._image_black = float(val)
        self._lbl_image_black.setText(f'{self._image_black:.5f}')
        self._refresh_overlay()

    def _on_image_white_changed(self, val: float) -> None:
        self._image_white = float(val)
        self._lbl_image_white.setText(f'{self._image_white:.5f}')
        self._refresh_overlay()

    def _on_image_gamma_changed(self, val: float) -> None:
        self._image_gamma = float(val)
        self._lbl_image_gamma.setText(f'{self._image_gamma:.5f}')
        self._refresh_overlay()

    def _on_model_black_changed(self, val: float) -> None:
        """Apply a new model black-point level from the Model Stretch black slider.

        Updates the in-memory model stretch lower bound used when compositing the
        navigation model into the overlay, keeps the model black value label in
        sync with the slider mapping, and redraws the RGB overlay.

        Parameters:
            val: Black level in data units after ``build_stretch_controls`` maps the
                slider position through ``from_slider`` (same linear range as
                ``_model_stretch_min`` / ``_model_stretch_max``).
        """
        self._model_black = float(val)
        self._lbl_model_black.setText(f'{self._model_black:.5f}')
        self._refresh_overlay()

    def _on_model_white_changed(self, val: float) -> None:
        """Apply a new model white-point level from the Model Stretch white slider.

        Updates the in-memory model stretch upper bound used when compositing the
        model, refreshes the model white value label, and redraws the overlay.

        Parameters:
            val: White level in data units from the slider mapping (see
                ``_on_model_black_changed``).
        """
        self._model_white = float(val)
        self._lbl_model_white.setText(f'{self._model_white:.5f}')
        self._refresh_overlay()

    def _on_model_gamma_changed(self, val: float) -> None:
        """Apply a new model gamma from the Model Stretch gamma slider.

        The shared stretch helper clamps gamma to at least ``0.10`` before this
        callback runs; this handler stores the value and redraws the overlay.

        Parameters:
            val: Gamma factor (``>= 0.10``) after ``build_stretch_controls`` gamma
                slot applies ``max(0.10, v / 100.0)``.
        """
        self._model_gamma = float(val)
        self._lbl_model_gamma.setText(f'{self._model_gamma:.5f}')
        self._refresh_overlay()

    def _on_model_transparency_changed(self, val: float) -> None:
        """Apply model transparency for the green-channel blend (0 opaque, 1 clear).

        Values are clipped to ``[0.0, 1.0]`` so the overlay stays well-defined even
        if the caller passes an out-of-range float.

        Parameters:
            val: Nominal transparency in ``[0, 1]``. The model transparency slider
                connects via ``lambda v: self._on_model_transparency_changed(v /
                100.0)``, so ``val`` is the slider position divided by ``100``.
        """
        self._model_transparency = float(np.clip(val, 0.0, 1.0))
        self._lbl_model_transparency.setText(f'{self._model_transparency:.2f}')
        self._refresh_overlay()

    def _on_show_image_changed(self, state: Any) -> None:
        """Toggle whether the stretched observation contributes to the overlay.

        Parameters:
            state: ``QCheckBox.stateChanged`` argument (typically an ``int`` Qt check
                state). Interpreted via ``_is_checked`` as ``True`` only for
                ``Qt.CheckState.Checked``.
        """
        self._show_image = self._is_checked(state)
        self._refresh_overlay()

    def _on_show_model_changed(self, state: Any) -> None:
        """Toggle model visibility and enablement of the transparency controls.

        When the model is hidden, the transparency slider and label are disabled so
        the user cannot adjust a blend that is not shown.

        Parameters:
            state: Same contract as ``_on_show_image_changed`` for the ``Display
                model`` checkbox.
        """
        self._show_model = self._is_checked(state)
        self._slider_model_transparency.setEnabled(self._show_model)
        self._lbl_model_transparency.setEnabled(self._show_model)
        self._refresh_overlay()

    def _on_show_mask_changed(self, state: Any) -> None:
        """Toggle overlay of the binary model mask on the composited image.

        When enabled, pixels where the model mask is ``True`` are highlighted with a
        semi-transparent white tint so the mask boundary is clearly visible against
        the false-color composite.

        Parameters:
            state: ``QCheckBox.stateChanged`` argument. Interpreted via
                ``_is_checked`` as ``True`` only for ``Qt.CheckState.Checked``.
        """
        self._show_mask = self._is_checked(state)
        self._refresh_overlay()

    def _on_reset_stretch(self) -> None:
        # Recompute defaults from current image (FOV only; extended margins are zero)
        self._image_black = float(np.quantile(self._img_fov, 0.001))
        self._image_white = float(np.quantile(self._img_fov, 0.999))
        if self._image_black >= self._image_white:
            self._image_white = self._image_black + 0.01
        self._image_gamma = 1.0
        # Update UI via common helper
        self._stretch_controls['set_values'](
            self._image_black, self._image_white, self._image_gamma
        )
        # Redraw
        self._refresh_overlay()

    def _on_reset_model_stretch(self) -> None:
        """Reset model black/white/gamma from quantiles of the current model slice."""
        ms = np.asarray(
            self._obs.extract_offset_array(self._model_img_ext, (self._dv, self._du)),
            dtype=np.float64,
        )
        self._model_black = float(np.quantile(ms, 0.001))
        self._model_white = float(np.quantile(ms, 0.999))
        if self._model_black >= self._model_white:
            self._model_white = self._model_black + 1e-6
        self._model_gamma = 1.0
        self._model_stretch_min = float(np.min(ms))
        self._model_stretch_max = float(np.max(ms))
        if self._model_stretch_min >= self._model_stretch_max:
            self._model_stretch_max = self._model_stretch_min + 1e-6
        self._model_stretch_controls['set_range'](self._model_stretch_min, self._model_stretch_max)
        self._model_stretch_controls['set_values'](
            self._model_black, self._model_white, self._model_gamma
        )
        self._refresh_overlay()

    def _on_spin_dv(self, val: float) -> None:
        self._dv = float(val)
        self._refresh_overlay()

    def _on_spin_du(self, val: float) -> None:
        self._du = float(val)
        self._refresh_overlay()

    def _on_show_corr_map(self) -> None:
        """Open a modal dialog displaying the NCC surface with the current offset marked.

        The correlation surface is fftshift-ed so zero offset is at the center.
        Axes are labeled in offset pixels: X = dU (cols), Y = dV (rows). The
        current ``(dV, dU)`` offset is overlaid as a red cross marker.
        """
        dlg = _CorrMapDialog(
            corr_surface=self._corr_surface,
            dv=self._dv,
            du=self._du,
            max_offset_vu=self._obs.extfov_margin_vu,
            parent=self,
        )
        dlg.exec()

    def _on_auto(self) -> None:
        # Auto-pick: run the same KPeaks pyramid NCC against the composite
        # template the dialog assembled via ``compose_template_features``.
        # The bi-directional ``data_mask`` keeps a body model that extends
        # into the extfov zero-padded margin from biasing the peak toward
        # |dV| = extfov_margin_v.
        up_factor = (
            getattr(self._config.offset, 'correlation_fft_upsample_factor', 128)
            if self._config
            else 128
        )
        res = navigate_with_pyramid_kpeaks(
            image=self._img_ext,
            model=self._model_img_ext,
            mask=self._model_mask_ext,
            upsample_factor=up_factor,
            max_offset_vu=self._obs.extfov_margin_vu,
            data_mask=self._obs.extfov_data_sensor_mask(),
            logger=None,
        )
        dv, du = float(res['offset'][0]), float(res['offset'][1])
        self._dv, self._du = dv, du
        self._spin_dv.blockSignals(True)
        self._spin_du.blockSignals(True)
        self._spin_dv.setValue(self._dv)
        self._spin_du.setValue(self._du)
        self._spin_dv.blockSignals(False)
        self._spin_du.blockSignals(False)
        self._refresh_overlay()

    def _on_save_library_entry(self) -> None:
        """Write the current dv/du as a fresh library sidecar.

        Pops a save-file dialog, defaults to ``<image_id>.yaml`` in
        whatever directory the operator was last in, and writes a sidecar
        seeded with the auto-fillable fields plus ``TODO`` placeholders
        for the operator-curated ones.  No round-trip into the image
        library is performed; the operator chooses the destination so an
        existing sidecar is not silently clobbered.

        After the YAML is written, the dialog also drops an annotated
        ``<stem>.png`` next to it: the same red-image / green-model
        composite the operator was looking at when they hit save.  The
        PNG is only an orientation aid (so a future reviewer skimming
        the sidecar directory can see the scene at a glance); it is not
        consumed by any test or downstream tool.
        """
        draft = infer_obs_metadata(self._obs)
        suggested_name = f'{draft.image_id or "TODO"}.yaml'
        path_str, _selected_filter = QFileDialog.getSaveFileName(
            self,
            'Save Library Sidecar',
            suggested_name,
            'Sidecar YAML (*.yaml);;All files (*)',
        )
        if not path_str:
            return
        image_url = compute_image_url(self._obs, self._config)
        constraint_angle: float | None = None
        if self._chk_constrain.isChecked():
            constraint_angle = float(self._spin_constraint_angle.value())
        try:
            yaml_text = build_sidecar_yaml(
                draft=draft,
                image_url=image_url,
                offset_dv_px=self._dv,
                offset_du_px=self._du,
                ui_version=_spindoctor_version,
                constraint_normal_angle_deg=constraint_angle,
            )
            FCPath(path_str).write_text(yaml_text)
        except OSError as exc:
            QMessageBox.critical(
                self,
                'Save failed',
                f'Could not write sidecar to {path_str}:\n{exc}',
            )
            return

        # Companion PNG: render the same labelled annotation overlay
        # (body / star / ring labels) the autonomous summary PNG would
        # produce, drawn at the chosen (dv, du) so the snapshot matches
        # the YAML's offset.  Falls back to the source-image grayscale
        # alone when no NavModel emitted any annotations.  PNG-write
        # failures are non-fatal — the YAML is the load-bearing
        # artefact.
        png_path = Path(path_str).with_suffix('.png')
        png_warning = ''
        try:
            rgb = render_annotated_summary_rgb(self._obs, self._annotations, (self._dv, self._du))
            Image.fromarray(rgb, mode='RGB').save(str(png_path), format='PNG')
        except (OSError, ValueError) as exc:
            png_warning = f'\n\n(Could not write companion PNG to {png_path}: {exc})'

        QMessageBox.information(
            self,
            'Sidecar saved',
            (
                f'Wrote sidecar to:\n{path_str}\n'
                f'Wrote summary PNG to:\n{png_path}\n\n'
                'Edit the TODO_REPLACE_* fields (scene_tags, primary_technique, '
                'notes, etc.) before committing.' + png_warning
            ),
        )

    # Mouse handling
    def _on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Use common zoom/pan controller for left-button pan
            self._zoom_ctl.on_mouse_press(event)
            self._label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_mode = 'offset'
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_offset = (self._dv, self._du)
            self._label.setCursor(Qt.CursorShape.SizeAllCursor)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_mode == 'offset' and self._drag_start_pos is not None:
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._drag_start_pos
            if self._drag_start_offset is not None:
                # Convert label-pixel delta to image pixels via zoom
                ddv = delta.y() / max(self._zoom, 1e-6)
                ddu = delta.x() / max(self._zoom, 1e-6)
                if self._chk_constrain.isChecked():
                    # Project the drag onto the constraint normal so the
                    # offset moves only along the observable axis.
                    angle = math.radians(self._spin_constraint_angle.value())
                    n_v = math.cos(angle)
                    n_u = math.sin(angle)
                    along = ddv * n_v + ddu * n_u
                    ddv = along * n_v
                    ddu = along * n_u
                dv = self._drag_start_offset[0] + ddv
                du = self._drag_start_offset[1] + ddu
                # Clamp within extfov bounds (minus 1 to keep slices valid after rounding)
                dv = float(
                    np.clip(
                        dv,
                        -self._obs.extfov_margin_v + 1,
                        self._obs.extfov_margin_v - 1,
                    )
                )
                du = float(
                    np.clip(
                        du,
                        -self._obs.extfov_margin_u + 1,
                        self._obs.extfov_margin_u - 1,
                    )
                )
                self._dv, self._du = dv, du
                # Update spin boxes without feedback loop
                self._spin_dv.blockSignals(True)
                self._spin_du.blockSignals(True)
                self._spin_dv.setValue(self._dv)
                self._spin_du.setValue(self._du)
                self._spin_dv.blockSignals(False)
                self._spin_du.blockSignals(False)
                self._refresh_overlay()
        else:
            # Delegate to common controller for hover/move and update status
            self._zoom_ctl.on_mouse_move(event)
            self._update_status_from_mouse(event)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_release(event)
            self._label.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start_pos = None
            self._drag_start_offset = None
            self._drag_mode = None
            self._label.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_wheel(self, event: QWheelEvent) -> None:
        # Ignore wheel-zoom if focus is on editable controls
        fw = self.focusWidget()
        if isinstance(fw, (QDoubleSpinBox, QSlider)):
            event.ignore()
            return
        # Delegate wheel zoom to common controller
        self._zoom_ctl.on_wheel(event)

    # ---- Zoom/pan helpers (parity with sim_body_gui) ----

    def _zoom_in_center(self) -> None:
        # Centre-anchored zoom is provided by the controller (it wraps this
        # window's scroll area); delegate instead of re-implementing it.
        self._zoom_ctl.zoom_in_center()

    def _zoom_out_center(self) -> None:
        self._zoom_ctl.zoom_out_center()

    def _zoom_at_point(
        self, factor: float, vx: int, vy: int, scaled_x: float, scaled_y: float
    ) -> None:
        if self._pixmap_base is None:
            return
        # The ZoomPanController owns the zoom clamp + no-op short-circuit, so
        # delegate directly instead of re-deriving the clamped zoom here.
        self._zoom_ctl.zoom_at_point(factor, vx, vy, scaled_x, scaled_y)

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._zoom_label.setText(f'zoom: {self._zoom:.2f}x')
        self._update_display_only()

    # ---- Rendering ----

    def _compose_overlay_rgb_fov(self) -> NDArrayUint8Type:
        """Build the FOV-shaped RGB overlay at the current ``(dv, du)``.

        Encapsulates the pixel logic shared between the on-screen pixmap
        and the on-disk summary PNG written alongside a saved sidecar:
        red + blue carry the stretched image, green carries the
        image-and-model blend, and the optional binary-mask overlay
        applies a semi-transparent white tint.

        Returns:
            ``(H, W, 3)`` uint8 array in FOV coordinates.
        """
        # Primary image (FOV) -> red channel (mono repeated into RGB then tinted)
        img_u8 = _apply_stretch_gamma(
            self._img_fov, self._image_black, self._image_white, self._image_gamma
        )
        h, w = img_u8.shape
        img_layer = img_u8 if self._show_image else np.zeros((h, w), dtype=np.uint8)
        # Extract model slice at current (dv, du)
        # Note: extract_offset_array will round inside; for display that's acceptable
        model_slice = self._obs.extract_offset_array(self._model_img_ext, (self._dv, self._du))
        self._model_slice_fov = np.asarray(model_slice, dtype=np.float64)
        model_u8 = _apply_stretch_gamma(
            self._model_slice_fov,
            self._model_black,
            self._model_white,
            self._model_gamma,
        )

        # Build RGB with red for image, green for model blend, blue for image
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = img_layer
        # Green: blend image and model when both visible; image only if model hidden;
        # model-tinted black if image hidden but model visible
        if self._show_model:
            img_f = img_layer.astype(np.float32)
            mod_f = model_u8.astype(np.float32)
            # transparency 0 => opaque model; 1 => model invisible (image only in green)
            t = self._model_transparency
            green = t * img_f + (1.0 - t) * mod_f
            rgb[:, :, 1] = np.clip(green, 0, 255).astype(np.uint8)
        else:
            rgb[:, :, 1] = img_layer
        rgb[:, :, 2] = img_layer

        # Overlay binary model mask as a semi-transparent white tint when enabled
        if self._show_mask:
            mask_raw = self._obs.extract_offset_array(self._model_mask_ext, (self._dv, self._du))
            mask_slice = np.asarray(mask_raw, dtype=np.float64) > 0.5
            for c in range(3):
                ch = rgb[:, :, c].astype(np.float32)
                ch[mask_slice] = (
                    ch[mask_slice] * (1.0 - _MASK_OVERLAY_ALPHA) + 255.0 * _MASK_OVERLAY_ALPHA
                )
                rgb[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        return rgb

    def _compose_overlay_pixmap(self) -> None:
        """Compose the RGB overlay pixmap based on current stretch/offset/alpha."""
        rgb = self._compose_overlay_rgb_fov()
        h, w = rgb.shape[:2]
        # Create QImage/QPixmap
        qimage = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap(w, h)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)
        painter.end()
        self._pixmap_base = pixmap

    def _update_display_only(self) -> None:
        """Update label to show scaled/panned image."""
        if self._pixmap_base is None:
            return
        scaled_w = int(self._pixmap_base.width() * self._zoom)
        scaled_h = int(self._pixmap_base.height() * self._zoom)
        transform_mode = (
            Qt.TransformationMode.FastTransformation
            if self._zoom_sharp
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled = self._pixmap_base.scaled(
            scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio, transform_mode
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled_w, scaled_h)
        # Update status bar with latest corr (no mouse move)
        self._status_label.setText(
            f'V, U: --, --  Image: --  Model: --  Correlation: {self._current_corr_value():.6f}'
        )

    def _refresh_overlay(self) -> None:
        """Rebuild overlay pixmap and update view."""
        self._compose_overlay_pixmap()
        self._update_display_only()

    def _update_status_from_mouse(self, event: QMouseEvent) -> None:
        # Position in label coordinates == scaled image coordinates.  A Qt event
        # position is pixel corner, and dividing by the zoom keeps it so, which is
        # what the V, U readout prints; the sample beside it is taken at that same
        # position, so the two describe one place in the frame.
        scaled_x = float(event.position().x())
        scaled_y = float(event.position().y())
        # Convert to original image coords
        img_u = scaled_x / max(self._zoom, 1e-6)
        img_v = scaled_y / max(self._zoom, 1e-6)
        h, w = self._img_fov.shape
        mod_arr = self._model_slice_fov
        if 0 <= img_v < h and 0 <= img_u < w and mod_arr is not None and mod_arr.shape == (h, w):
            val_img = _bilinear_interpolate_fov(self._img_fov, img_v, img_u)
            val_model = _bilinear_interpolate_fov(mod_arr, img_v, img_u)
            corr_val = self._current_corr_value()
            self._status_label.setText(
                f'V, U: {img_v:.2f}, {img_u:.2f}  '
                f'Image: {val_img:.6f}  Model: {val_model:.6f}  Correlation: {corr_val:.6f}'
            )
        else:
            self._status_label.setText(
                f'V, U: --, --  Image: --  Model: --  Correlation: {self._current_corr_value():.6f}'
            )

    # ---- Dialog control ----

    def run_modal(self) -> tuple[bool, tuple[float, float] | None, float | None]:
        """Run the dialog modally, creating a QApplication if necessary."""
        app_created = False
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
            app_created = True
        result = self.exec()
        accepted = result == QDialog.DialogCode.Accepted
        chosen = (self._dv, self._du) if accepted else None
        corr = self._current_corr_value() if accepted else None
        if app_created:
            # Do not quit an existing app
            app.quit()
        return accepted, chosen, corr

    # ---- Zoom options ----
    def _toggle_zoom_sharp(self, state: Any) -> None:
        """Turn sharp (pixel-crisp) zoom scaling on or off.

        When enabled, the image pixmap is rescaled with a mode that preserves sharp
        edges at high zoom; when disabled, a smoother filter is used. This affects
        only how the label pixmap is scaled, not stretch or offset state.

        Parameters:
            state: ``QCheckBox.stateChanged`` argument for the ``Sharp zoom``
                checkbox. ``True`` means ``Qt.CheckState.Checked`` per
                ``_is_checked``.
        """
        self._zoom_sharp = self._is_checked(state)
        self._update_display_only()

    # Internal buffers
    _pixmap_base: QPixmap | None = None
