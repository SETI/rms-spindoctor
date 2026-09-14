"""RingMosaicWindow: PyQt6 UI for browsing ring reprojections and ring mosaics.

Provides header toggles (radial profile, EW profile, axis ticks), a vertical
splitter (**radial | EW | image**), corotating EW bands (Ctrl+click), a Cursor
Info grid and Color By controls in the lower strip, and status-bar readouts.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.ma as ma
from matplotlib.figure import Figure
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QCursor, QFontMetrics, QKeyEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from spindoctor.config import image_scope
from spindoctor.support.time import et_to_utc
from spindoctor.ui.common import build_stretch_controls
from spindoctor.ui.mosaic_viewer.common import (
    RingDisplayData,
    _SyncedSlider,
    _ZoomSync,
    load_ring_file,
)
from spindoctor.ui.mosaic_viewer.histogram_stretch import HistogramStretchWidget
from spindoctor.ui.mosaic_viewer.matplotlib_qt import canvas_draw_idle, new_figure_canvas_qtagg
from spindoctor.ui.mosaic_viewer.photometric_display import compute_ring_display_image
from spindoctor.ui.mosaic_viewer.tiled_image_widget import (
    TiledImageWidget,
)

EW_PROFILE_LEFT_GUTTER_PX = 58

_EW_BAND_COLOR_CYCLE = (
    '#d62728',
    '#2ca02c',
    '#ff7f0e',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#bcbd22',
    '#17becf',
    '#8da0cb',
    '#ff9896',
)

STATUS_BAR_HINT = (
    'Mouse wheel zooms both axes (Shift+wheel: X only, Ctrl+wheel: Y only). '
    'Shift+Left to zoom to region, Left drag to pan, '
    'Ctrl+Left to set EW inner and outer radii, ESC to cancel.'
)

_COLORBY_REL_META_FIELD: dict[str, str] = {
    'rel_rad_res': 'mean_radial_resolution',
    'rel_ang_res': 'mean_angular_resolution',
    'rel_phase': 'mean_phase',
    'rel_emission': 'mean_emission',
}
_COLORBY_ABS_RANGE: dict[str, tuple[str, float, float]] = {
    'abs_phase': ('mean_phase', 0.0, 180.0),
    'abs_emission': ('mean_emission', 0.0, 90.0),
}


def _compute_ew(image_ma: ma.MaskedArray, radial_resolution_km: float) -> ma.MaskedArray:
    """Column-sum the sparse ring image and scale by radial bin width (km).

    Parameters:
        image_ma: 2-D masked ring image (radius x longitude).
        radial_resolution_km: Radial bin width in km.

    Returns:
        1-D masked array (length = number of longitude columns), EW brightness proxy.
    """
    return cast(ma.MaskedArray, ma.sum(image_ma, axis=0) * radial_resolution_km)


def _compute_ewmu(ew: ma.MaskedArray, emission_deg: ma.MaskedArray) -> ma.MaskedArray:
    """Weight EW columns by ``|cos(emission)|`` (mu factor).

    Parameters:
        ew: East-west integrated brightness per column (see :func:`_compute_ew`).
        emission_deg: Per-column mean emission angle in degrees (masked array).

    Returns:
        1-D masked ``ma.MaskedArray`` of ``ew * mu``. Masked emission angles stay
        masked (no fill to 0°); the result combines ``ew`` and ``emission_deg``
        masks so ``EW * mu`` does not invent unmasked values where emission is unknown.
    """
    # ``np.radians`` on ``ma.MaskedArray`` preserves ``emission_deg.mask`` (there is
    # no ``ma.radians``); ``ma.cos`` / ``np.abs`` keep the mask through ``mu``.
    emi_rad = np.radians(emission_deg)
    mu = cast(ma.MaskedArray, np.abs(ma.cos(emi_rad)))
    return cast(ma.MaskedArray, ma.asarray(ew) * mu)


def _mean_std_masked_1d(arr: ma.MaskedArray) -> tuple[float, float]:
    """Mean and standard deviation of unmasked samples in a 1-D masked array.

    Parameters:
        arr: Input masked vector.

    Returns:
        ``(mean, std)`` as floats; ``(0.0, 0.0)`` when there are no valid points.
    """
    valid = arr.compressed()
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.mean(valid)), float(np.std(valid))


def _ring_longitude_corotating(dd: RingDisplayData) -> bool:
    """True when the file was built with a ring orbit model (co-rotating longitude)."""
    return dd.orbit_model_name is not None


def _percentile_stretch(
    image_ma: ma.MaskedArray, lo_pct: float = 0.0, hi_pct: float = 98.0
) -> tuple[float, float]:
    """Percentile-based black/white levels for stretch controls.

    Parameters:
        image_ma: 2-D masked ring image; masked pixels ignored.
        lo_pct: Lower percentile for black level.
        hi_pct: Upper percentile for white level.

    Returns:
        ``(black, white)`` floats; widens ``white`` slightly if ``white <= black``.
    """
    valid = image_ma.compressed()
    if valid.size == 0:
        return 0.0, 1.0
    black = float(np.percentile(valid, lo_pct))
    white = float(np.percentile(valid, hi_pct))
    if white <= black:
        white = black + 1e-6
    return black, white


def _colorby_column(
    data_1d: np.ndarray | ma.MaskedArray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map a 1-D metadata column to RGB rows for the Color By overlay.

    Parameters:
        data_1d: Per-column scalar values (numpy or masked 1-D).
        vmin: Optional lower ramp bound; default uses ``nanmin``.
        vmax: Optional upper ramp bound; default uses ``nanmax``.

    Returns:
        ``np.ndarray`` of shape ``(len(data_1d), 3)``, float32 in ``[0, 1]``.
    """
    arr = np.asarray(data_1d, dtype=np.float64)
    if isinstance(data_1d, ma.MaskedArray):
        mask = ma.getmaskarray(data_1d)
        arr = np.where(mask, np.nan, arr)
    lo = float(np.nanmin(arr)) if vmin is None else vmin
    hi = float(np.nanmax(arr)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0).astype(np.float32)
    g = np.clip(np.where(t < 0.5, 2.0 * t, 2.0 - 2.0 * t), 0.0, 1.0).astype(np.float32)
    b = np.clip(0.5 - 1.5 * (t - 0.5 / 1.5), 0.0, 1.0).astype(np.float32)
    rgb = np.stack([r, g, b], axis=1)
    rgb[np.isnan(arr)] = 0.5
    return rgb


class RingMosaicWindow(QMainWindow):
    """PyQt6 viewer for ring reprojections and ring mosaics (``RingDisplayData``).

    Displays a sparse or dense ring image with optional EW/radial profile plots,
    stretch and zoom controls, cursor metadata, and color-by overlays. All UI
    updates run on the GUI thread; the window does not mutate caller-owned paths
    or file contents.

    Parameters:
        file_paths: Non-empty list of paths to ``RingReprojResult`` / ``RingMosaicData``
            files (see :func:`~spindoctor.ui.mosaic_viewer.common.load_ring_file`).
        initial_black: Optional fixed stretch black; ``None`` uses percentile stretch.
        initial_white: Optional fixed stretch white; ``None`` uses percentile stretch.
        initial_gamma: Initial gamma for the stretch controls (default ``0.5``).
        show_radii_km: Radii (km) for horizontal guide lines; ``None`` uses ``[]``.
        show_longitude_ticks: Pre-check longitude axis tick overlay.
        show_radius_ticks: Pre-check radius axis tick overlay.
        parent: Optional Qt parent widget.

    Behavior:
        Navigation uses ``_load_file`` and list/prev/next controls; closing the
        window releases child widgets and Matplotlib canvases normally. Signals
        from ``TiledImageWidget`` drive zoom readouts and cursor handling.

    Notes:
        EW matplotlib interaction and ``eventFilter`` resize handling expect the
        window to remain on the main Qt thread.
    """

    def __init__(
        self,
        file_paths: list[str],
        *,
        initial_black: float | None = None,
        initial_white: float | None = None,
        initial_gamma: float = 0.5,
        show_radii_km: list[float] | None = None,
        show_longitude_ticks: bool = False,
        show_radius_ticks: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Open the ring mosaic UI for ``file_paths``.

        Parameters:
            file_paths: At least one ring ``.npz`` / ``.fits`` path.
            initial_black: Optional stretch black override after load.
            initial_white: Optional stretch white override after load.
            initial_gamma: Initial display gamma (default ``0.5``).
            show_radii_km: Overlay radii in km; ``None`` becomes an empty list.
            show_longitude_ticks: Enable longitude tick overlay on first paint.
            show_radius_ticks: Enable radius tick overlay on first paint.
            parent: Optional Qt parent.

        Raises:
            ValueError: If ``file_paths`` is empty.
        """
        super().__init__(parent)
        if not file_paths:
            raise ValueError('file_paths must contain at least one path')
        self._file_paths = file_paths
        self._current_idx = 0
        self._display_data: RingDisplayData | None = None
        self._initial_black = initial_black
        self._initial_white = initial_white
        self._initial_gamma = initial_gamma
        self._show_radii_km = show_radii_km or []
        self._stretch_controls: dict[str, Any] = {}
        self._stretch_form: QFormLayout = QFormLayout()
        self._histogram_widget: HistogramStretchWidget | None = None
        self._chk_histogram_mode: QCheckBox | None = None
        self._default_gamma = initial_gamma
        self._ring_view_ma: ma.MaskedArray | None = None
        self._last_profile_lon_ix: int | None = None

        self._ew_phase = 0
        self._ew_first_py = 0.0
        self._ew_radial_ranges: list[tuple[int, int]] = []
        self._pending_fit = False
        self._radial_profile_line: Any = None
        self._colorby_alpha: float = 1.0

        self._ew_data: ma.MaskedArray | None = None
        self._ew_mu_data: ma.MaskedArray | None = None
        self._ew_mean = 0.0
        self._ew_std = 0.0
        self._ewmu_mean = 0.0
        self._ewmu_std = 0.0
        self._image_vmin = 0.0
        self._image_vmax = 1.0

        self._setup_ui()
        self._chk_lon_ticks.setChecked(show_longitude_ticks)
        self._chk_rad_ticks.setChecked(show_radius_ticks)
        self._load_file(0)

    def statusBar(self) -> QStatusBar:
        bar = super().statusBar()
        assert bar is not None
        return bar

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if (
            obj is getattr(self, '_ew_canvas', None)
            and event is not None
            and event.type() == QEvent.Type.Resize
        ):
            self._sync_ew_figure_margins()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if getattr(self, '_pending_fit', False):
            self._fit_zoom_to_window()
        if getattr(self, '_cor_wrap', None) is not None and self._cor_wrap.isVisible():
            self._sync_ew_figure_margins()
            self._sync_ew_xlim_from_mosaic()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape and self._ew_phase != 0:
            self._ew_phase = 0
            self.statusBar().showMessage(STATUS_BAR_HINT)
        super().keyPressEvent(event)

    def _safe_radial_canvas_draw(self) -> None:
        c = getattr(self, '_radial_canvas', None)
        if c is None:
            return
        try:
            if sip.isdeleted(c):
                return
        except Exception:
            pass
        try:
            c.draw()
        except RuntimeError:
            # Repaint failures here are recoverable and were filed at DEBUG;
            # with no logger to file them under, printing every one would turn
            # a silent diagnostic into permanent noise on a repaint loop.
            pass

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle('Ring Mosaic Viewer')
        self.resize(1400, 900)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        self._image_widget = TiledImageWidget()
        self._image_widget.mouse_moved.connect(self._on_mouse_moved)
        self._image_widget.zoom_changed.connect(self._on_zoom_changed)
        self._image_widget.zoom_changed.connect(self._sync_ew_xlim_from_mosaic)
        self._image_widget.horizontalScrollBar().valueChanged.connect(
            self._sync_ew_xlim_from_mosaic
        )
        self._image_widget.verticalScrollBar().rangeChanged.connect(self._sync_ew_xlim_from_mosaic)
        self._image_widget.ctrl_clicked.connect(self._on_ctrl_click)

        self._build_plot_panels(left_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([1150, 250])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter, stretch=1)
        main_layout.addWidget(self._build_control_panel())

        self.setCentralWidget(central)

        self._cursor_status_lbl = QLabel('')
        self._cursor_status_lbl.setStyleSheet('font-family: monospace;')
        self.statusBar().addPermanentWidget(self._cursor_status_lbl)
        self.statusBar().showMessage(STATUS_BAR_HINT)

    def _build_plot_panels(self, left_layout: QVBoxLayout) -> None:
        header = QWidget()
        hh = QHBoxLayout(header)
        hh.setContentsMargins(4, 2, 4, 0)
        hh.setSpacing(10)

        self._chk_lon_ticks = QCheckBox('Longitude axis ticks')
        self._chk_rad_ticks = QCheckBox('Radius axis ticks')
        self._chk_rad_profile = QCheckBox('Radial Profile')
        self._chk_corot_ew = QCheckBox('EW Profile')
        self._chk_corot_use_ewmu = QCheckBox('Use EW x mu')
        self._chk_lon_ticks.toggled.connect(self._sync_axis_tick_options)
        self._chk_rad_ticks.toggled.connect(self._sync_axis_tick_options)
        self._chk_rad_profile.toggled.connect(self._on_rad_profile_toggled)
        self._chk_corot_use_ewmu.toggled.connect(self._on_corot_ew_mode_changed)
        self._chk_corot_ew.toggled.connect(self._on_corot_ew_panel_toggled)
        self._chk_rad_profile.setChecked(False)
        self._chk_corot_ew.setChecked(False)
        self._chk_corot_use_ewmu.setChecked(False)

        for w in (self._chk_lon_ticks, self._chk_rad_ticks, self._chk_rad_profile):
            hh.addWidget(w)
        hh.addWidget(self._chk_corot_ew)
        self._btn_clear_ew_profile = QPushButton('Clear EW Profile')
        self._btn_clear_ew_profile.clicked.connect(self._on_clear_ew_profile)
        hh.addWidget(self._btn_clear_ew_profile)
        hh.addStretch()
        hh.addWidget(self._chk_corot_use_ewmu)
        left_layout.addWidget(header)

        self._rad_wrap = QWidget()
        rad_l = QVBoxLayout(self._rad_wrap)
        rad_l.setContentsMargins(0, 0, 0, 0)
        self._radial_fig = Figure(figsize=(12, 2.8), constrained_layout=True)
        self._radial_ax = self._radial_fig.add_subplot(111)
        self._radial_canvas = new_figure_canvas_qtagg(self._radial_fig)
        self._radial_canvas.setMinimumHeight(120)
        self._radial_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        rad_l.addWidget(self._radial_canvas)
        self._init_radial_axes()
        self._rad_wrap.setVisible(False)

        self._cor_wrap = QWidget()
        cor_l = QVBoxLayout(self._cor_wrap)
        cor_l.setContentsMargins(0, 0, 0, 0)
        self._ew_fig = Figure(figsize=(12, 3.6))
        self._ew_ax = self._ew_fig.add_subplot(111)
        self._ew_canvas = new_figure_canvas_qtagg(self._ew_fig)
        self._ew_canvas.setMinimumHeight(180)
        self._ew_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._ew_canvas.installEventFilter(self)
        cor_l.addWidget(self._ew_canvas)
        self._init_corot_ew_axes()
        self._cor_wrap.setVisible(False)

        self._plot_splitter = QSplitter(Qt.Orientation.Vertical)
        self._plot_splitter.setChildrenCollapsible(True)
        self._plot_splitter.addWidget(self._rad_wrap)
        self._plot_splitter.addWidget(self._cor_wrap)
        self._plot_splitter.addWidget(self._image_widget)
        self._plot_splitter.setStretchFactor(0, 0)
        self._plot_splitter.setStretchFactor(1, 0)
        self._plot_splitter.setStretchFactor(2, 1)
        self._plot_splitter.setSizes([0, 0, 600])
        left_layout.addWidget(self._plot_splitter, stretch=1)

    def _build_right_panel(self) -> QWidget:
        right = QWidget()
        right.setMinimumWidth(200)
        right.setMaximumWidth(300)
        v = QVBoxLayout(right)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton('< Prev')
        self._btn_next = QPushButton('Next >')
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        v.addLayout(nav_row)
        self._file_lbl = QLabel('')
        self._file_lbl.setWordWrap(True)
        v.addWidget(self._file_lbl)
        v.addWidget(QLabel('Files'))
        self._file_list = QListWidget()
        self._file_list.setMinimumHeight(140)
        self._file_list.currentRowChanged.connect(self._on_file_list_row_changed)
        v.addWidget(self._file_list, stretch=1)
        self._populate_file_list()
        overlay_box = QGroupBox('Overlays')
        overlay_v = QVBoxLayout(overlay_box)
        self._chk_show_radii = QCheckBox('Show radii')
        self._chk_show_radii.setChecked(bool(self._show_radii_km))
        self._chk_show_radii.toggled.connect(self._update_show_radii)
        overlay_v.addWidget(self._chk_show_radii)
        v.addWidget(overlay_box)
        return right

    def _populate_file_list(self) -> None:
        self._file_list.clear()
        for i, p in enumerate(self._file_paths):
            name = Path(p).name
            item = QListWidgetItem(f'{i + 1}. {name}')
            item.setToolTip(str(p))
            self._file_list.addItem(item)

    def _refresh_file_list_selection(self) -> None:
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(self._current_idx)
        cur = self._file_list.currentItem()
        if cur is not None:
            self._file_list.scrollToItem(cur)
        self._file_list.blockSignals(False)

    def _on_file_list_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._file_paths):
            return
        if row == self._current_idx:
            return
        self._load_file(row)

    def _build_control_panel(self) -> QWidget:
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 2, 4, 2)
        ctrl_layout.setSpacing(2)

        upper = QWidget()
        upper_h = QHBoxLayout(upper)
        upper_h.setContentsMargins(0, 0, 0, 0)
        upper_h.setSpacing(6)

        stretch_box = QGroupBox('Stretch')
        stretch_form = QFormLayout()
        stretch_form.setHorizontalSpacing(4)
        self._stretch_form = stretch_form
        self._stretch_controls = build_stretch_controls(
            stretch_form,
            img_min=0.0,
            img_max=1.0,
            black_init=0.0,
            white_init=1.0,
            gamma_init=self._initial_gamma,
            on_black_changed=lambda _v: self._apply_stretch(),
            on_white_changed=lambda _v: self._apply_stretch(),
            on_gamma_changed=lambda _v: self._apply_stretch(),
            slider_horizontal_stretch=1,
        )
        self._histogram_widget = HistogramStretchWidget(
            on_black_changed=self._on_histogram_black_changed,
            on_white_changed=self._on_histogram_white_changed,
        )
        self._histogram_widget.setVisible(False)
        stretch_left = QVBoxLayout()
        stretch_left.setContentsMargins(0, 0, 0, 0)
        stretch_left.setSpacing(2)
        stretch_left.addWidget(self._histogram_widget)
        stretch_left.addLayout(stretch_form)
        stretch_btn_col = QVBoxLayout()
        stretch_btn_col.setSpacing(4)
        btn_stretch_reset = QPushButton('Reset')
        btn_stretch_full = QPushButton('Full')
        btn_stretch_bright = QPushButton('Bright')
        btn_stretch_reset.clicked.connect(self._on_stretch_preset_reset)
        btn_stretch_full.clicked.connect(self._on_stretch_preset_full)
        btn_stretch_bright.clicked.connect(self._on_stretch_preset_bright)
        for b in (btn_stretch_reset, btn_stretch_full, btn_stretch_bright):
            b.setMaximumWidth(72)
            stretch_btn_col.addWidget(b)
        self._chk_histogram_mode = QCheckBox('Histogram')
        self._chk_histogram_mode.toggled.connect(self._on_histogram_mode_toggled)
        stretch_btn_col.addWidget(self._chk_histogram_mode)
        stretch_btn_col.addStretch()
        stretch_outer = QHBoxLayout()
        stretch_outer.setContentsMargins(4, 4, 4, 4)
        stretch_outer.setSpacing(8)
        stretch_outer.addLayout(stretch_left, stretch=1)
        stretch_outer.addLayout(stretch_btn_col)
        stretch_box.setLayout(stretch_outer)
        upper_h.addWidget(stretch_box, stretch=2)

        zoom_box = QGroupBox('Zoom')
        zoom_layout = QVBoxLayout(zoom_box)
        zoom_layout.setSpacing(2)

        def _make_zoom_row(label: str) -> tuple[QLineEdit, QSlider, QWidget]:
            le = QLineEdit()
            le.setMaximumWidth(55)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(1, 1000)
            row = QWidget()
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(label))
            rh.addWidget(le)
            rh.addWidget(sl, stretch=1)
            return le, sl, row

        self._xzoom_le, self._xzoom_sl, xz_row = _make_zoom_row('X:')
        self._yzoom_le, self._yzoom_sl, yz_row = _make_zoom_row('Y:')
        zoom_layout.addWidget(xz_row)
        zoom_layout.addWidget(yz_row)
        self._xzoom_sync = self._make_zoom_sync(self._xzoom_le, self._xzoom_sl, axis='x')
        self._yzoom_sync = self._make_zoom_sync(self._yzoom_le, self._yzoom_sl, axis='y')

        btn_row = QHBoxLayout()
        self._zoom_info_lbl = QLabel('1.00x / 1.00x')
        btn_zi = QPushButton('+')
        btn_zo = QPushButton('\N{MINUS SIGN}')  # U+2212 for symmetric glyph next to '+'
        btn_zr = QPushButton('Reset')
        btn_sf = QPushButton('Save FOV')
        btn_zi.setMaximumWidth(28)
        btn_zo.setMaximumWidth(28)
        btn_zi.clicked.connect(self._on_zoom_in)
        btn_zo.clicked.connect(self._on_zoom_out)
        btn_zr.clicked.connect(self._on_zoom_reset)
        btn_sf.clicked.connect(self._on_save_fov)
        btn_row.addWidget(self._zoom_info_lbl)
        btn_row.addWidget(btn_zi)
        btn_row.addWidget(btn_zo)
        btn_row.addWidget(btn_zr)
        btn_row.addStretch()
        btn_row.addWidget(btn_sf)
        zoom_layout.addLayout(btn_row)
        upper_h.addWidget(zoom_box, stretch=1)
        ctrl_layout.addWidget(upper)

        lower = QWidget()
        lower_h = QHBoxLayout(lower)
        lower_h.setContentsMargins(0, 0, 0, 0)
        lower_h.setSpacing(6)

        info_box = QGroupBox('Cursor Info')
        info_grid_widget = QWidget()
        info_grid = QGridLayout(info_grid_widget)
        info_grid.setHorizontalSpacing(10)
        info_grid.setVerticalSpacing(0)
        info_grid.setContentsMargins(2, 0, 2, 0)
        info_box_layout = QVBoxLayout(info_box)
        info_box_layout.setContentsMargins(4, 1, 4, 1)
        info_box_layout.setSpacing(0)
        info_box_layout.addWidget(info_grid_widget)

        info_columns: list[list[tuple[str, str]]] = [
            [
                ('orbit_model', 'Orbit model:'),
                ('corot', 'Co-rotating longitude:'),
                ('inert', 'Inertial longitude:'),
                ('abs_r', 'Radius (km):'),
                ('rel_r', 'Radial offset from orbit (km):'),
                ('core_r', 'Orbital model radius (km):'),
            ],
            [
                ('incidence', 'Incidence angle:'),
                ('phase', 'Phase angle:'),
                ('emission', 'Emission angle:'),
                ('rad_res', 'Radial resolution:'),
                ('long_res', 'Longitudinal resolution:'),
            ],
            [
                ('image', 'Source image:'),
                ('long_ew', 'EW at longitude:'),
                ('long_ewmu', 'EW\N{MULTIPLICATION SIGN}\N{GREEK SMALL LETTER MU} at longitude:'),
                ('full_ew', 'Full mosaic EW:'),
                ('full_ewmu', 'Full mosaic EW\N{MULTIPLICATION SIGN}\N{GREEK SMALL LETTER MU}:'),
            ],
        ]
        self._info: dict[str, QLabel] = {}
        self._info_name: dict[str, QLabel] = {}
        # First column carries the orbit model name (e.g.
        # ``F-RING-CORE-ALBERS-2007``) which is wider than a numeric value.
        name_w = 168
        val_w = (200, 118, 400)
        for col_idx, col in enumerate(info_columns):
            base = col_idx * 2
            for row_idx, (key, name) in enumerate(col):
                nl = QLabel(name)
                nl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                nl.setFixedWidth(name_w)
                vl = QLabel('---')
                vl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                vl.setFixedWidth(val_w[col_idx])
                info_grid.addWidget(nl, row_idx, base)
                info_grid.addWidget(vl, row_idx, base + 1)
                self._info[key] = vl
                if key in ('corot', 'inert', 'abs_r', 'rel_r', 'core_r'):
                    self._info_name[key] = nl
        lower_h.addWidget(info_box, stretch=1)

        colorby_box = QGroupBox('Color By')
        cb_grid = QGridLayout(colorby_box)
        cb_grid.setContentsMargins(4, 2, 4, 2)
        cb_grid.setHorizontalSpacing(12)
        cb_grid.setVerticalSpacing(2)
        self._colorby_group = QButtonGroup()
        colorby_rows: list[list[tuple[str, str]]] = []
        colorby_rows.append([('none', 'None'), ('image_no', 'Image number')])
        colorby_rows.extend(
            [
                [
                    ('rel_rad_res', 'Radial resolution (rel)'),
                    ('rel_ang_res', 'Longitudinal resolution (rel)'),
                ],
                [('abs_phase', 'Phase (abs)'), ('rel_phase', 'Phase (rel)')],
                [('abs_emission', 'Emission (abs)'), ('rel_emission', 'Emission (rel)')],
            ]
        )
        for row_idx, row in enumerate(colorby_rows):
            for col_idx, (key, label) in enumerate(row):
                btn = QRadioButton(label)
                btn.setProperty('colorby_key', key)
                self._colorby_group.addButton(btn)
                cb_grid.addWidget(btn, row_idx, col_idx)
                if key == 'none':
                    btn.setChecked(True)
        self._colorby_group.buttonClicked.connect(self._on_colorby_changed)
        alpha_row = QHBoxLayout()
        alpha_row.setContentsMargins(0, 2, 0, 0)
        alpha_row.addWidget(QLabel('Alpha:'))
        self._colorby_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._colorby_alpha_slider.setRange(0, 100)
        self._colorby_alpha_slider.setValue(100)
        self._colorby_alpha_slider.valueChanged.connect(self._on_colorby_alpha_changed)
        alpha_row.addWidget(self._colorby_alpha_slider)
        cb_grid.addLayout(alpha_row, len(colorby_rows), 0, 1, 2)
        lower_h.addWidget(colorby_box)

        photometry_box = QGroupBox('Photometric')
        ph_grid = QGridLayout(photometry_box)
        ph_grid.setContentsMargins(4, 2, 4, 2)
        ph_grid.setHorizontalSpacing(8)
        ph_grid.setVerticalSpacing(2)
        self._photometry_group = QButtonGroup()
        photometry_rows: list[list[tuple[str, str]]] = [
            [('as_saved', 'As saved'), ('intrinsic', 'Uncorrected')],
            [('lambert', 'Lambert'), ('lommel_seeliger', 'Lommel\N{EN DASH}Seeliger')],
            [('minnaert', 'Minnaert')],
        ]
        for row_idx, row in enumerate(photometry_rows):
            for col_idx, (key, label) in enumerate(row):
                btn = QRadioButton(label)
                btn.setProperty('photometry_key', key)
                self._photometry_group.addButton(btn)
                ph_grid.addWidget(btn, row_idx, col_idx)
                if key == 'as_saved':
                    btn.setChecked(True)
        self._photometry_group.buttonClicked.connect(self._on_photometry_changed)
        lower_h.addWidget(photometry_box)

        ctrl_layout.addWidget(lower)
        return ctrl

    def _make_zoom_sync(self, le: QLineEdit, sl: QSlider, axis: str) -> _SyncedSlider:
        def _on_change(zoom_val: float) -> None:
            iw = self._image_widget
            xz, yz = iw.get_zoom()
            if axis == 'x':
                iw.set_zoom(zoom_val, yz)
            else:
                iw.set_zoom(xz, zoom_val)

        sync = _ZoomSync(le, sl, 0.05, 100.0, '%.2f', on_change=_on_change)
        sync.set_value(1.0)
        return sync

    def _photometry_mode(self) -> str:
        btn = self._photometry_group.checkedButton()
        if btn is None:
            return 'as_saved'
        return str(btn.property('photometry_key'))

    def _sync_photometry_ui(self, dd: RingDisplayData) -> None:
        """Reset photometric display to file pixels when loading a new file."""
        _ = dd
        for b in self._photometry_group.buttons():
            if b.property('photometry_key') == 'as_saved':
                b.setChecked(True)
                break

    def _apply_ring_display_image(self, *, preserve_view: bool) -> None:
        dd = self._display_data
        if dd is None:
            return
        mode = self._photometry_mode()
        img = compute_ring_display_image(
            mode=mode,
            image_ma=dd.image_ma,
            photometric_model_name=dd.photometric_model_name,
            mean_phase_deg=dd.mean_phase,
            mean_emission_deg=dd.mean_emission,
            mean_incidence_deg=dd.mean_incidence_deg,
        )
        self._ring_view_ma = img
        lon_max = dd.longitude_extent_hi_deg
        if lon_max is None:
            lon_max = dd.longitude_column_origin_deg + float(
                dd.n_longitude * dd.longitude_resolution_deg
            )
        mid = (dd.radius_inner + dd.radius_outer) / 2.0
        corot = _ring_longitude_corotating(dd)
        self._image_widget.set_image(
            img,
            x_interval=dd.longitude_resolution_deg,
            y_interval=dd.radius_resolution_km,
            x_label=('Co-rotating longitude (°)' if corot else 'Inertial longitude (°)'),
            y_label=('Radial offset from orbit (km)' if corot else 'Radius (km)'),
            y_flip=True,
            x_axis_max=float(lon_max),
            x_origin_deg=float(dd.longitude_column_origin_deg),
            ring_radial_axis_absolute=not corot,
            ring_radial_mid_km=mid,
            ring_full_lon=True,
            preserve_view=preserve_view,
        )

    def _on_photometry_changed(self, _btn: Any = None) -> None:
        if self._display_data is None:
            return
        self._apply_ring_display_image(preserve_view=True)
        dd = self._display_data
        view = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        black, white = _percentile_stretch(view, 0.0, 98.0)
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_range'](dd.vmin, dd.vmax)
        self._set_stretch_levels(black, white, g)
        self._refresh_histogram_data()
        self._image_widget.set_stretch(black, white, g)
        self._recompute_ring_ew_from_view()
        self._info['full_ew'].setText(f'{self._ew_mean:.5f} ± {self._ew_std:.5f}')
        self._info['full_ewmu'].setText(f'{self._ewmu_mean:.5f} ± {self._ewmu_std:.5f}')
        self._replot_corot_ew_panel()
        self._on_colorby_changed(self._colorby_group.checkedButton())
        if self._chk_rad_profile.isChecked() and self._last_profile_lon_ix is not None:
            self._update_radial_profile_plot(self._last_profile_lon_ix)

    def _recompute_ring_ew_from_view(self) -> None:
        dd = self._display_data
        if dd is None:
            self._ew_data = None
            self._ew_mu_data = None
            return
        img = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        self._ew_data = _compute_ew(img, dd.radius_resolution_km)
        self._ew_mu_data = _compute_ewmu(self._ew_data, dd.mean_emission)
        self._ew_mean, self._ew_std = _mean_std_masked_1d(self._ew_data)
        self._ewmu_mean, self._ewmu_std = _mean_std_masked_1d(self._ew_mu_data)

    # ------------------------------------------------------------------
    #  EW and radial profile layout
    # ------------------------------------------------------------------

    def _ew_align_gutter_px(self) -> int:
        if getattr(self, '_cor_wrap', None) is not None and self._cor_wrap.isVisible():
            return EW_PROFILE_LEFT_GUTTER_PX
        return 0

    def _sync_ew_mosaic_layout(self) -> None:
        g = self._ew_align_gutter_px()
        self._image_widget.setViewportMargins(g, 0, 0, 0)
        self._sync_ew_figure_margins()

    def _sync_ew_figure_margins(self) -> None:
        if getattr(self, '_ew_canvas', None) is None:
            return
        w = max(1, self._ew_canvas.width())
        g = self._ew_align_gutter_px()
        right = 1.0 - 2.0 / w
        if g <= 0:
            self._ew_fig.subplots_adjust(left=0.09, right=right, top=0.93, bottom=0.18)
        else:
            self._ew_fig.subplots_adjust(left=g / w, right=right, top=0.93, bottom=0.18)
        canvas_draw_idle(self._ew_canvas)

    def _sync_ew_xlim_from_mosaic(self) -> None:
        if (
            getattr(self, '_cor_wrap', None) is None
            or not self._cor_wrap.isVisible()
            or self._display_data is None
        ):
            return
        dd = self._display_data
        iw = self._image_widget
        hv = iw.horizontalScrollBar().value()
        vw = iw.x_fov_span_px()
        xz, _ = iw.get_zoom()
        if vw <= 0 or xz <= 0:
            return
        # The image widget always builds a virtual full-circle canvas for
        # rings (``ring_full_lon=True``), so virtual column 0 is at longitude
        # 0 deg regardless of where the stored data starts. The viewport's left
        # and right edges sit at virtual columns ``hv / xz`` and
        # ``(hv + vw) / xz``; converting straight to longitude (without any
        # data-range clipping) lets the EW xlim track the viewport into
        # empty regions of the ring as the user pans, which is what the user
        # needs for inspecting gaps in the mosaic.
        res = dd.longitude_resolution_deg
        c0 = (hv / xz) * res
        c1 = ((hv + vw) / xz) * res
        if c1 <= c0:
            return
        self._ew_ax.set_xlim(c0, c1)
        canvas_draw_idle(self._ew_canvas)

    def _init_radial_axes(self) -> None:
        ax = self._radial_ax
        ax.set_xlabel('', fontsize=8)
        ax.set_ylabel('I/F', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.margins(y=0)

    def _init_corot_ew_axes(self) -> None:
        ax = self._ew_ax
        self._sync_corot_ew_xlabel()
        self._update_corot_ew_ylabel()
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    def _sync_corot_ew_xlabel(self) -> None:
        if getattr(self, '_ew_ax', None) is None:
            return
        dd = self._display_data
        if dd is None:
            self._ew_ax.set_xlabel('Longitude (°)', fontsize=8)
            return
        if _ring_longitude_corotating(dd):
            self._ew_ax.set_xlabel('Co-rotating longitude (°)', fontsize=8)
        else:
            self._ew_ax.set_xlabel('Inertial longitude (°)', fontsize=8)

    def _update_corot_ew_ylabel(self) -> None:
        if getattr(self, '_chk_corot_use_ewmu', None) is None:
            return
        self._ew_ax.set_ylabel(
            'EW\N{MULTIPLICATION SIGN}\N{GREEK SMALL LETTER MU} (km)'
            if self._chk_corot_use_ewmu.isChecked()
            else 'EW (km)',
            fontsize=8,
        )

    def _sync_axis_tick_options(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        mid = (dd.radius_inner + dd.radius_outer) / 2.0
        y_ticks_absolute = not _ring_longitude_corotating(dd)
        self._image_widget.set_axis_tick_options(
            self._chk_lon_ticks.isChecked(),
            self._chk_rad_ticks.isChecked(),
            mid,
            y_tick_labels_absolute=y_ticks_absolute,
        )

    def _sync_ring_cursor_row_labels(self, dd: RingDisplayData) -> None:
        # Labels are fixed; only the orbit model name and the per-cursor
        # values vary. Without an orbit model the orbit-relative rows
        # (co-rotating longitude, radial offset, orbital model radius)
        # show ``---`` because they cannot be computed.
        self._info['orbit_model'].setText(dd.orbit_model_name if dd.orbit_model_name else '---')
        self._info_name['corot'].setText('Co-rotating longitude:')
        self._info_name['inert'].setText('Inertial longitude:')
        self._info_name['abs_r'].setText('Radius (km):')
        self._info_name['rel_r'].setText('Radial offset from orbit (km):')
        self._info_name['core_r'].setText('Orbital model radius (km):')

    def _on_rad_profile_toggled(self, checked: bool) -> None:
        self._rad_wrap.setVisible(checked)
        if checked and self._display_data is not None:
            self._safe_radial_canvas_draw()
        self._balance_plot_splitter()

    def _on_corot_ew_panel_toggled(self, checked: bool) -> None:
        self._cor_wrap.setVisible(checked)
        if checked:
            self._update_corot_ew_ylabel()
            canvas_draw_idle(self._ew_canvas)
        self._balance_plot_splitter()
        self._sync_ew_mosaic_layout()
        if self._display_data is not None:
            self._fit_zoom_to_window()
        self._sync_ew_xlim_from_mosaic()

    def _balance_plot_splitter(self) -> None:
        sp = self._plot_splitter
        h = max(250, sp.height())
        rad_on = self._rad_wrap.isVisible()
        cor_on = self._cor_wrap.isVisible()
        if rad_on and cor_on:
            sp.setSizes([int(h * 0.24), int(h * 0.28), int(h * 0.48)])
        elif rad_on:
            sp.setSizes([int(h * 0.32), 0, int(h * 0.68)])
        elif cor_on:
            sp.setSizes([0, int(h * 0.35), int(h * 0.65)])
        else:
            sp.setSizes([0, 0, h])

    def _on_corot_ew_mode_changed(self) -> None:
        self._update_corot_ew_ylabel()
        if self._display_data is not None:
            self._replot_corot_ew_panel()

    def _on_clear_ew_profile(self) -> None:
        self._ew_radial_ranges.clear()
        self._ew_phase = 0
        if self._display_data is not None:
            self._replot_corot_ew_panel()
        else:
            self._reset_ew_plot()
        self.statusBar().showMessage(STATUS_BAR_HINT)

    def _reset_ew_plot(self) -> None:
        self._ew_ax.cla()
        self._init_corot_ew_axes()
        canvas_draw_idle(self._ew_canvas)

    def _column_band_ew(self, dd: RingDisplayData, arr_min: int, arr_max: int) -> ma.MaskedArray:
        img = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        return cast(
            ma.MaskedArray,
            ma.sum(img[arr_min : arr_max + 1, :], axis=0) * dd.radius_resolution_km,
        )

    def _column_band_ewmu(self, dd: RingDisplayData, arr_min: int, arr_max: int) -> ma.MaskedArray:
        ew = self._column_band_ew(dd, arr_min, arr_max)
        return _compute_ewmu(ew, dd.mean_emission)

    def _replot_corot_ew_panel(self) -> None:
        dd = self._display_data
        if dd is None or self._ew_data is None:
            return
        self._ew_ax.cla()
        self._init_corot_ew_axes()
        use_mu = self._chk_corot_use_ewmu.isChecked()
        longs = self._column_longitudes_deg(dd)
        if use_mu:
            yfull = self._ew_mu_data.filled(np.nan) if self._ew_mu_data is not None else np.nan
            stat = f'{self._ewmu_mean:.4f} ± {self._ewmu_std:.4f} km'
            tag = 'Full (EW\N{MULTIPLICATION SIGN}\N{GREEK SMALL LETTER MU})'
        else:
            yfull = self._ew_data.filled(np.nan)
            stat = f'{self._ew_mean:.4f} ± {self._ew_std:.4f} km'
            tag = 'Full (EW)'
        self._ew_ax.plot(longs, yfull, color='steelblue', lw=0.8, label=f'{tag}  {stat}')
        for i, (arr_min, arr_max) in enumerate(self._ew_radial_ranges):
            c = _EW_BAND_COLOR_CYCLE[i % len(_EW_BAND_COLOR_CYCLE)]
            self._draw_corot_band_curve(dd, arr_min, arr_max, use_mu, c)
        self._ew_ax.legend(fontsize=7, loc='upper right')
        self._sync_ew_xlim_from_mosaic()

    def _draw_corot_band_curve(
        self,
        dd: RingDisplayData,
        arr_min: int,
        arr_max: int,
        use_mu: bool,
        color: str,
    ) -> None:
        if use_mu:
            ew_data = self._column_band_ewmu(dd, arr_min, arr_max)
        else:
            ew_data = self._column_band_ew(dd, arr_min, arr_max)
        longs = self._column_longitudes_deg(dd)
        r_min = self._band_row_radius_km(dd, arr_min)
        r_max = self._band_row_radius_km(dd, arr_max)
        if _ring_longitude_corotating(dd):
            band = f'{r_min:+.0f} to {r_max:+.0f} km (radial offset from orbit)'
        else:
            band = f'{r_min:.0f} to {r_max:.0f} km (absolute radius)'
        valid = ew_data.compressed()
        ew_mean = float(np.mean(valid)) if valid.size > 0 else 0.0
        ew_std = float(np.std(valid)) if valid.size > 0 else 0.0
        self._ew_ax.plot(
            longs,
            ew_data.filled(np.nan),
            color=color,
            lw=0.8,
            label=f'{band}  {ew_mean:.4f} ± {ew_std:.4f}',
        )

    @staticmethod
    def _band_row_radius_km(dd: RingDisplayData, arr_row: float) -> float:
        """Radius (km) the equivalent-width band reports for array row ``arr_row``.

        The signed radial offset from the orbit model when the mosaic carries
        co-rotating longitudes, and the absolute radius when it carries inertial
        ones -- the same quantity the band legend prints, so a boundary quoted
        while the band is being selected and the legend of the band that results
        are one number.

        Parameters:
            dd: The loaded ring display data.
            arr_row: Array row (0 = inner), the row a display position was
                converted to.

        Returns:
            The radius or radial offset in km.
        """
        rel = (arr_row - (dd.n_radii - 1) / 2.0) * dd.radius_resolution_km
        if _ring_longitude_corotating(dd):
            return float(rel)
        return float(rel + (dd.radius_inner + dd.radius_outer) / 2.0)

    @staticmethod
    def _mosaic_radial_abs_km_bounds(dd: RingDisplayData) -> tuple[float, float]:
        rows = np.arange(dd.n_radii, dtype=np.float64)
        rel = (rows - (dd.n_radii - 1) / 2.0) * dd.radius_resolution_km
        mean_core = (dd.radius_inner + dd.radius_outer) / 2.0
        abs_r = rel + mean_core
        return float(np.min(abs_r)), float(np.max(abs_r))

    @staticmethod
    def _column_longitudes_deg(dd: RingDisplayData) -> np.ndarray:
        """Return a 1-D float64 array of longitude (deg) for each image column.

        Uses ``longitude_global_bins`` for non-contiguous sparse mosaics so that
        the mapping is bin-accurate rather than ``origin + ix * resolution``.
        """
        if dd.longitude_global_bins is not None:
            return dd.longitude_global_bins * dd.longitude_resolution_deg
        return (
            dd.longitude_column_origin_deg
            + np.arange(dd.n_longitude, dtype=np.float64) * dd.longitude_resolution_deg
        )

    @staticmethod
    def _corot_longitude_for_column(dd: RingDisplayData, ix: int) -> float:
        hi = dd.longitude_extent_hi_deg
        if hi is None:
            hi = dd.longitude_column_origin_deg + float(
                dd.n_longitude * dd.longitude_resolution_deg
            )
        lo = float(dd.longitude_column_origin_deg)
        if dd.longitude_global_bins is not None and 0 <= ix < len(dd.longitude_global_bins):
            lon = float(dd.longitude_global_bins[ix]) * dd.longitude_resolution_deg
        else:
            lon = lo + float(ix) * dd.longitude_resolution_deg
        return float(np.clip(lon, lo, hi))

    def _update_radial_profile_plot(self, ix: int) -> None:
        dd = self._display_data
        if dd is None or not self._chk_rad_profile.isChecked():
            return
        img = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        col = img[:, ix]
        arr_rows = np.arange(dd.n_radii, dtype=np.float64)
        rel = (arr_rows - (dd.n_radii - 1) / 2.0) * dd.radius_resolution_km
        y = np.asarray(col, dtype=np.float64)
        if hasattr(col, 'mask'):
            m = ma.getmaskarray(col)
            y = np.where(m, np.nan, y)
        mean_core = (dd.radius_inner + dd.radius_outer) / 2.0
        abs_lo, abs_hi = RingMosaicWindow._mosaic_radial_abs_km_bounds(dd)
        rel_lo = abs_lo - mean_core
        rel_hi = abs_hi - mean_core
        lon_deg = RingMosaicWindow._corot_longitude_for_column(dd, ix)
        corot = _ring_longitude_corotating(dd)
        if corot:
            x_plot = rel
            r_span = max(abs(rel_lo), abs(rel_hi), 1e-6)
            x_lo, x_hi = -r_span, r_span
            ax_xlabel = f'Radial offset from orbit at co-rotating longitude {lon_deg:.2f}° (km)'
        else:
            x_plot = rel + mean_core
            x_lo, x_hi = abs_lo, abs_hi
            if x_hi <= x_lo:
                x_hi = x_lo + 1e-6
            ax_xlabel = f'Absolute radius at inertial longitude {lon_deg:.2f}° (km)'
        y_lo, y_hi = self._image_vmin, self._image_vmax
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-6
        ax = self._radial_ax
        line = self._radial_profile_line
        need_new = line is None or getattr(line, 'axes', None) is None or line.axes is not ax
        if need_new:
            ax.clear()
            self._init_radial_axes()
            (self._radial_profile_line,) = ax.plot(x_plot, y, 'b-', lw=0.9)
        else:
            line.set_data(x_plot, y)
        ax.set_xlim(x_lo, x_hi)
        ax.margins(y=0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_autoscaley_on(False)
        ax.set_xlabel(ax_xlabel, fontsize=8)
        self._safe_radial_canvas_draw()

    def _add_ew_range_to_plot(self, py1: float, py2: float, dd: RingDisplayData) -> None:
        arr1 = self._image_widget.pixel_y_to_arr_row(py1)
        arr2 = self._image_widget.pixel_y_to_arr_row(py2)
        arr_min = max(0, min(arr1, arr2))
        arr_max = min(dd.n_radii - 1, max(arr1, arr2))
        self._ew_radial_ranges.append((arr_min, arr_max))
        self._replot_corot_ew_panel()
        if not self._chk_corot_ew.isChecked():
            self._chk_corot_ew.setChecked(True)

    # ------------------------------------------------------------------
    #  Load / navigation
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        self._load_file(self._current_idx - 1)

    def _on_next(self) -> None:
        self._load_file(self._current_idx + 1)

    def _load_file(self, idx: int) -> None:
        if not (0 <= idx < len(self._file_paths)):
            return
        path = self._file_paths[idx]
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            try:
                with image_scope():
                    dd = load_ring_file(path)
            except Exception as exc:
                print(f'Error loading {path}', file=sys.stderr)
                traceback.print_exc()
                self.statusBar().showMessage(f'Error loading {path}: {exc}', 5000)
                return

            self._current_idx = idx
            self._display_data = dd
            self.setWindowTitle(f'Ring Mosaic Viewer - {dd.title}')
            self._file_lbl.setText(f'{idx + 1} / {len(self._file_paths)}\n{dd.title}')
            self._btn_prev.setEnabled(idx > 0)
            self._btn_next.setEnabled(idx < len(self._file_paths) - 1)

            self._last_profile_lon_ix = None
            self._clear_info()
            self._sync_ring_cursor_row_labels(dd)
            self._sync_photometry_ui(dd)
            self._apply_ring_display_image(preserve_view=False)
            self._recompute_ring_ew_from_view()
            view = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
            valid_img = view.compressed()
            if valid_img.size > 0:
                self._image_vmin = float(np.min(valid_img))
                self._image_vmax = float(np.max(valid_img))
            else:
                self._image_vmin, self._image_vmax = 0.0, 1.0

            self._sync_axis_tick_options()

            black, white = _percentile_stretch(view, 0.0, 98.0)
            if self._initial_black is not None:
                black = self._initial_black
            if self._initial_white is not None:
                white = self._initial_white
            self._stretch_controls['set_range'](dd.vmin, dd.vmax)
            self._set_stretch_levels(black, white, self._initial_gamma)
            self._refresh_histogram_data()
            self._image_widget.set_stretch(black, white, self._initial_gamma)

            self._update_colorby_widgets_for_mosaic(dd.is_mosaic)
            self._on_colorby_changed(self._colorby_group.checkedButton())
            self._info['full_ew'].setText(f'{self._ew_mean:.5f} ± {self._ew_std:.5f}')
            self._info['full_ewmu'].setText(f'{self._ewmu_mean:.5f} ± {self._ewmu_std:.5f}')

            self._ew_phase = 0
            self._ew_radial_ranges.clear()
            self._rad_wrap.setVisible(self._chk_rad_profile.isChecked())
            self._cor_wrap.setVisible(self._chk_corot_ew.isChecked())
            self._balance_plot_splitter()
            self._sync_ew_mosaic_layout()
            self._fit_zoom_to_window()
            self._sync_zoom_ui()
            self._reset_ew_plot()
            self._replot_corot_ew_panel()
            self._radial_profile_line = None
            self._radial_ax.clear()
            self._init_radial_axes()
            self._safe_radial_canvas_draw()

            self._update_show_radii(self._chk_show_radii.isChecked())
            self.statusBar().showMessage(STATUS_BAR_HINT)
            self._refresh_file_list_selection()
        finally:
            QApplication.restoreOverrideCursor()

    def _update_colorby_widgets_for_mosaic(self, is_mosaic: bool) -> None:
        for btn in self._colorby_group.buttons():
            key = btn.property('colorby_key')
            if key == 'image_no':
                btn.setEnabled(is_mosaic)
                if not is_mosaic and btn.isChecked():
                    for b2 in self._colorby_group.buttons():
                        if b2.property('colorby_key') == 'none':
                            b2.setChecked(True)
                            break

    # ------------------------------------------------------------------
    #  Stretch / zoom
    # ------------------------------------------------------------------

    def _apply_stretch(self) -> None:
        b = self._stretch_controls['from_slider'](self._stretch_controls['slider_black'].value())
        w = self._stretch_controls['from_slider'](self._stretch_controls['slider_white'].value())
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._image_widget.set_stretch(b, w, g)

    def _set_stretch_levels(self, black: float, white: float, gamma: float) -> None:
        """Sync black/white/gamma into both the slider controls and the histogram."""
        self._stretch_controls['set_values'](black, white, gamma)
        if self._histogram_widget is not None:
            self._histogram_widget.set_values(black, white)

    def _on_histogram_black_changed(self, black: float) -> None:
        white = self._stretch_controls['from_slider'](
            self._stretch_controls['slider_white'].value()
        )
        gamma = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_values'](black, white, gamma)
        self._apply_stretch()

    def _on_histogram_white_changed(self, white: float) -> None:
        black = self._stretch_controls['from_slider'](
            self._stretch_controls['slider_black'].value()
        )
        gamma = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_values'](black, white, gamma)
        self._apply_stretch()

    def _on_histogram_mode_toggled(self, checked: bool) -> None:
        if self._histogram_widget is None or self._chk_histogram_mode is None:
            return
        # Slider rows are at indices 0 (Black) and 1 (White); Gamma at 2.
        self._stretch_form.setRowVisible(0, not checked)
        self._stretch_form.setRowVisible(1, not checked)
        self._histogram_widget.setVisible(checked)
        if checked:
            black = self._stretch_controls['from_slider'](
                self._stretch_controls['slider_black'].value()
            )
            white = self._stretch_controls['from_slider'](
                self._stretch_controls['slider_white'].value()
            )
            self._histogram_widget.set_values(black, white)
            self._refresh_histogram_data()

    def _refresh_histogram_data(self) -> None:
        if self._histogram_widget is None:
            return
        dd = self._display_data
        if dd is None:
            return
        view = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        self._histogram_widget.set_data(view)

    def _on_stretch_preset_reset(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        view = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        black, white = _percentile_stretch(view, 0.0, 98.0)
        self._set_stretch_levels(black, white, self._default_gamma)
        self._apply_stretch()

    def _on_stretch_preset_full(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        self._set_stretch_levels(dd.vmin, dd.vmax, self._default_gamma)
        self._image_widget.set_stretch(dd.vmin, dd.vmax, self._default_gamma)

    def _on_stretch_preset_bright(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        view = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        black, white = _percentile_stretch(view, 2.0, 98.0)
        self._set_stretch_levels(black, white, self._default_gamma)
        self._apply_stretch()

    def _fit_zoom_to_window(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        if dd.n_longitude == 0 or dd.n_radii == 0:
            self._pending_fit = False
            return
        vw = self._image_widget.viewport().width()
        vh = self._image_widget.viewport().height()
        if vw <= 0 or vh <= 0:
            self._pending_fit = True
            return
        self._pending_fit = False
        x_zoom = min(float(vw) / dd.n_longitude, 100.0)
        y_zoom = min(float(vh) / dd.n_radii, 100.0)
        self._image_widget.set_zoom(x_zoom, y_zoom)
        # Scroll to place the data at the left edge of the viewport (nop when
        # ring_x_col_offset is 0, but essential when the mosaic doesn't start at 0°).
        self._image_widget.scroll_to_pixel(dd.n_longitude / 2.0, dd.n_radii / 2.0)

    def _on_zoom_in(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz * 1.5, yz * 1.5)
        self._sync_zoom_ui()

    def _on_zoom_out(self) -> None:
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz / 1.5, yz / 1.5)
        self._sync_zoom_ui()

    def _on_zoom_reset(self) -> None:
        self._fit_zoom_to_window()

    def _on_zoom_changed(self, xz: float, yz: float) -> None:
        self._update_zoom_slider_ranges()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')
        self._sync_ew_xlim_from_mosaic()

    def _update_zoom_slider_ranges(self) -> None:
        min_x, min_y = self._image_widget.get_min_zoom()
        self._xzoom_sync.set_range(min_x, 100.0)
        self._yzoom_sync.set_range(min_y, 100.0)

    def _sync_zoom_ui(self) -> None:
        self._update_zoom_slider_ranges()
        xz, yz = self._image_widget.get_zoom()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _on_save_fov(self) -> None:
        dd = self._display_data
        if dd is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Field of View',
            f'{dd.title}_fov.png',
            'PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)',
        )
        if not path:
            return
        qimg = self._image_widget.render_viewport_to_image()
        if not qimg.save(path):
            QMessageBox.warning(self, 'Save FOV', f'Failed to save {path}')

    # ------------------------------------------------------------------
    #  Color-by
    # ------------------------------------------------------------------

    def _on_colorby_changed(self, btn: Any) -> None:
        if btn is None or self._display_data is None:
            self._image_widget.set_color_tint(None)
            return
        dd = self._display_data
        key = btn.property('colorby_key')
        col = self._compute_color_column(str(key), dd)
        if col is None:
            self._image_widget.set_color_tint(None)
            return
        # Broadcast per-column (n_cols, 3) tint to (n_rows, n_cols, 3)
        n_rows = dd.n_radii
        tint = np.ascontiguousarray(
            np.broadcast_to(col[np.newaxis, :, :], (n_rows, col.shape[0], 3)),
            dtype=np.float32,
        )
        self._image_widget.set_color_tint(self._tint_with_alpha(tint))

    def _on_colorby_alpha_changed(self, value: int) -> None:
        self._colorby_alpha = value / 100.0
        self._on_colorby_changed(self._colorby_group.checkedButton())

    def _tint_with_alpha(self, tint: np.ndarray | None) -> np.ndarray | None:
        if tint is None or self._colorby_alpha >= 1.0:
            return tint
        return (self._colorby_alpha * tint + (1.0 - self._colorby_alpha)).astype(np.float32)

    def _compute_color_column(self, key: str, dd: RingDisplayData) -> np.ndarray | None:
        if key == 'none':
            return None
        if key == 'image_no':
            if dd.image_number is None:
                return None
            img_no = dd.image_number.astype(float)
            valid = ma.array(img_no, mask=ma.getmaskarray(dd.image_number)).compressed()
            if valid.size == 0:
                return None
            return _colorby_column(
                ma.array(img_no, mask=ma.getmaskarray(dd.image_number)),
                vmin=float(valid.min()),
                vmax=float(valid.max()),
            )
        if key in _COLORBY_REL_META_FIELD:
            attr = _COLORBY_REL_META_FIELD[key]
            if not hasattr(dd, attr):
                return None
            meta_vals: Any = getattr(dd, attr)
            if isinstance(meta_vals, ma.MaskedArray):
                valid = meta_vals.compressed()
            else:
                valid = np.asarray(meta_vals).ravel()
                valid = valid[np.isfinite(valid)]
            if valid.size == 0:
                return None
            if isinstance(meta_vals, ma.MaskedArray):
                return _colorby_column(meta_vals, vmin=float(valid.min()), vmax=float(valid.max()))
            return _colorby_column(np.asarray(meta_vals, dtype=np.float64))
        if key in _COLORBY_ABS_RANGE:
            attr, lo, hi = _COLORBY_ABS_RANGE[key]
            if not hasattr(dd, attr):
                return None
            abs_vals = getattr(dd, attr)
            return _colorby_column(abs_vals, vmin=lo, vmax=hi)
        return None

    def _update_show_radii(self, checked: bool) -> None:
        dd = self._display_data
        if dd is None or not checked or not self._show_radii_km:
            self._image_widget.set_show_rows([])
            return
        if _ring_longitude_corotating(dd):
            # When an orbit model is set, the absolute radius at each row
            # varies with longitude (offset semantics), so a fixed absolute
            # radius cannot be drawn as a single guide row.
            self._image_widget.set_show_rows([])
            return
        mid = (dd.radius_inner + dd.radius_outer) / 2.0
        n_rows = dd.n_radii
        pixel_ys = []
        for r_km in self._show_radii_km:
            rel = r_km - mid
            py = float(n_rows - 1) / 2.0 - rel / dd.radius_resolution_km
            pixel_ys.append(round(py))
        self._image_widget.set_show_rows(pixel_ys)

    # ------------------------------------------------------------------
    #  Mouse / cursor info
    # ------------------------------------------------------------------

    @staticmethod
    def _format_ma_at_ix(arr: ma.MaskedArray, ix: int, fmt: str = '%.4f') -> str:
        v = arr[ix]
        if ma.is_masked(v):
            return '---'
        return fmt % float(v)

    @staticmethod
    def _fmt_deg(s: str) -> str:
        return f'{s}°' if s != '---' else '---'

    def _on_mouse_moved(self, px: float, py: float, in_bounds: bool) -> None:
        if not in_bounds or self._display_data is None:
            self._clear_info()
            return
        dd = self._display_data
        ix = self._image_widget.pixel_x_to_arr_col(px, py)
        ix = int(np.clip(ix, 0, dd.n_longitude - 1))
        self._last_profile_lon_ix = ix
        arr_row = self._image_widget.pixel_y_to_arr_row(py)
        arr_row = int(np.clip(arr_row, 0, dd.n_radii - 1))
        lon_deg, _y_axis = self._image_widget.pixel_to_physical(px, py)
        img = self._ring_view_ma if self._ring_view_ma is not None else dd.image_ma
        raw_val = img[arr_row, ix]
        if ma.is_masked(raw_val):
            value_str = f'{"masked":>11}'
        else:
            value_str = f'{float(raw_val):11.8f}'

        phase = self._format_ma_at_ix(dd.mean_phase, ix, '%.3f')
        emiss = self._format_ma_at_ix(dd.mean_emission, ix, '%.3f')
        rad_r = self._format_ma_at_ix(dd.mean_radial_resolution, ix, '%.3f')
        lng_r = self._format_ma_at_ix(dd.mean_angular_resolution, ix, '%.5f')
        # row_value matches dd.radius_inner / dd.radius_outer semantics:
        # absolute km when no orbit model; signed offset (km) from the
        # orbital radius at this (longitude, time) when one is set.
        row_value = dd.radius_inner + arr_row * dd.radius_resolution_km

        if dd.image_number is not None:
            img_idx_v = dd.image_number[ix]
            if not ma.is_masked(img_idx_v):
                idx = int(img_idx_v)
                names = dd.contributing_image_names
                if idx < len(names) and names[idx]:
                    img_name = f'{names[idx]} (#{idx})'
                else:
                    img_name = f'image #{idx}'
            else:
                img_name = '---'
        else:
            n0 = dd.contributing_image_names[0] if dd.contributing_image_names else ''
            img_name = f'{n0} (#0)' if n0 else dd.title

        if dd.observation_time_tdb is not None:
            tdb_v = dd.observation_time_tdb[ix]
            if not ma.is_masked(tdb_v):
                date_str = et_to_utc(float(tdb_v), digits=0)
            else:
                date_str = '---'
        else:
            date_str = '---'

        ew_v = self._ew_data[ix] if self._ew_data is not None else ma.masked
        ewmu_v = self._ew_mu_data[ix] if self._ew_mu_data is not None else ma.masked
        ew_str = (
            f'{float(ew_v):.5f}' if self._ew_data is not None and not ma.is_masked(ew_v) else '---'
        )
        ewmu_str = (
            f'{float(ewmu_v):.5f}'
            if self._ew_mu_data is not None and not ma.is_masked(ewmu_v)
            else '---'
        )

        inc_str = f'{dd.mean_incidence_deg:.3f}' if dd.mean_incidence_deg is not None else '---'

        x_str = f'{px:8.2f}'
        y_str = f'{py:7.2f}'
        self._cursor_status_lbl.setText(f'X: {x_str}  Y: {y_str}  Value: {value_str}')
        if _ring_longitude_corotating(dd):
            # row_value is the pixel's signed offset (km) from the orbital
            # radius at this (longitude, time); orbit model radius itself
            # varies with longitude/time, and the absolute radius at the
            # cursor is model_r + offset.
            inert_str = '---'
            core_r_str = '---'
            abs_r_str = '---'
            rel_r_str = f'{row_value:+.2f}'
            if dd.orbit_model is not None and dd.observation_time_tdb is not None:
                tdb_v = dd.observation_time_tdb[ix]
                if not ma.is_masked(tdb_v):
                    corot_rad = np.array([lon_deg * math.pi / 180.0])
                    inert_rad = dd.orbit_model.corotating_to_inertial(corot_rad, float(tdb_v))
                    inert_str = f'{math.degrees(float(inert_rad[0])):.4f}°'
                    model_r = float(dd.orbit_model.radius_at_longitude(inert_rad, float(tdb_v))[0])
                    core_r_str = f'{model_r:.2f}'
                    abs_r_str = f'{model_r + row_value:.2f}'
            self._info['abs_r'].setText(abs_r_str)
            self._info['rel_r'].setText(rel_r_str)
            self._info['core_r'].setText(core_r_str)
            self._info['corot'].setText(f'{lon_deg:.4f}°')
            self._info['inert'].setText(inert_str)
        else:
            # No orbit model: row_value is the absolute radius (km); the
            # orbit-relative quantities (offset, model radius) are not
            # defined.
            self._info['abs_r'].setText(f'{row_value:.2f}')
            self._info['rel_r'].setText('---')
            self._info['core_r'].setText('---')
            self._info['inert'].setText(f'{lon_deg:.4f}°')
            self._info['corot'].setText('---')
        self._info['incidence'].setText(self._fmt_deg(inc_str))
        self._info['phase'].setText(self._fmt_deg(phase))
        self._info['emission'].setText(self._fmt_deg(emiss))
        self._info['rad_res'].setText(f'{rad_r} km/px' if rad_r != '---' else '---')
        self._info['long_res'].setText(f'{lng_r} deg/px' if lng_r != '---' else '---')
        if img_name != '---' and date_str != '---':
            source_display = f'{img_name}  \N{MIDDLE DOT}  {date_str}'
        else:
            source_display = img_name
        img_lbl = self._info['image']
        iw = img_lbl.width()
        if iw > 0 and source_display != '---':
            fm = QFontMetrics(img_lbl.font())
            source_display = fm.elidedText(source_display, Qt.TextElideMode.ElideRight, iw)
        img_lbl.setText(source_display)
        self._info['long_ew'].setText(ew_str)
        self._info['long_ewmu'].setText(ewmu_str)

        if self._chk_rad_profile.isChecked():
            self._update_radial_profile_plot(ix)

    def _clear_info(self) -> None:
        self._cursor_status_lbl.setText('')
        _STATIC_INFO_KEYS = frozenset({'orbit_model', 'full_ew', 'full_ewmu'})
        for key, lbl in self._info.items():
            if key not in _STATIC_INFO_KEYS:
                lbl.setText('---')

    def _on_ctrl_click(self, px: float, py: float) -> None:
        dd = self._display_data
        if dd is None:
            return
        # Quote the boundary of the band that will actually be integrated: the
        # band is cut on the array row ``pixel_y_to_arr_row`` selects, so the
        # message reads that row's radius rather than the continuous coordinate
        # under the cursor, which lies anywhere within the row.
        arr_row = self._image_widget.pixel_y_to_arr_row(py)
        arr_row = int(np.clip(arr_row, 0, dd.n_radii - 1))
        r_val = self._band_row_radius_km(dd, arr_row)
        if _ring_longitude_corotating(dd):
            r_desc = 'radial offset from orbit'
        else:
            r_desc = 'absolute radius'
        if self._ew_phase == 0:
            self._ew_phase = 1
            self._ew_first_py = py
            msg = (
                f'Ctrl+click to select upper radial boundary '
                f'(lower: {r_desc}={r_val:.1f} km). ESC to cancel.'
            )
            self.statusBar().showMessage(msg)
        else:
            self._ew_phase = 0
            self._add_ew_range_to_plot(self._ew_first_py, py, dd)
            self.statusBar().showMessage(STATUS_BAR_HINT)
