"""BodyMosaicWindow: PyQt6 window for browsing body reprojections and mosaics."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    BodyDisplayData,
    _SyncedSlider,
    _ZoomSync,
    load_body_file,
)
from spindoctor.ui.mosaic_viewer.histogram_stretch import HistogramStretchWidget
from spindoctor.ui.mosaic_viewer.photometric_display import compute_body_display_image
from spindoctor.ui.mosaic_viewer.projections import ProjectionKind
from spindoctor.ui.mosaic_viewer.tiled_image_widget import (
    TiledImageWidget,
)

_STATUS_HINTS: dict[ProjectionKind, str] = {
    ProjectionKind.RECT: (
        'Mouse wheel zooms both axes (Shift+wheel: X only, Ctrl+wheel: Y only). '
        'Shift+Left to zoom to region, Left drag to pan.'
    ),
    ProjectionKind.POLAR_N: (
        'Polar North Stereographic. Left drag pans, Shift+Left zooms to region, wheel zooms.'
    ),
    ProjectionKind.POLAR_S: (
        'Polar South Stereographic. Left drag pans, Shift+Left zooms to region, wheel zooms.'
    ),
    ProjectionKind.MOLLWEIDE: (
        'Mollweide projection. Left drag pans, Shift+Left zooms to region, wheel zooms.'
    ),
    ProjectionKind.SPHERE_3D: (
        'Left drag rotates (yaw/pitch). Shift+Left drag pans. Wheel zooms. Reset Zoom fits sphere.'
    ),
}


def _percentile_stretch(
    image_ma: ma.MaskedArray, lo_pct: float = 0.0, hi_pct: float = 98.0
) -> tuple[float, float]:
    """Compute black/white stretch limits from valid pixels of a masked image.

    Parameters:
        image_ma: Input image; masked entries are ignored.
        lo_pct: Lower percentile for black (default ``0.0``).
        hi_pct: Upper percentile for white (default ``98.0``).

    Returns:
        ``(black, white)`` as ``tuple[float, float]``. If there are no valid pixels,
        returns ``(0.0, 1.0)``. If ``white <= black`` after percentiles, ``white`` is
        bumped to ``black + 1e-6``.
    """
    valid = image_ma.compressed()
    if valid.size == 0:
        return 0.0, 1.0
    black = float(np.percentile(valid, lo_pct))
    white = float(np.percentile(valid, hi_pct))
    if white <= black:
        white = black + 1e-6
    return black, white


def _colorby_tint(
    data: np.ndarray | ma.MaskedArray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Return per-pixel RGB tint (n_rows, n_cols, 3) float32 from a 2-D metadata array.

    Parameters:
        data: 2-D array (or masked array) of per-pixel scalar metadata values.
        vmin: Lower clamp for the color ramp; defaults to nanmin.
        vmax: Upper clamp for the color ramp; defaults to nanmax.

    Returns:
        Array of shape ``(n_rows, n_cols, 3)`` with float32 values in ``[0, 1]``.
        Masked / NaN pixels receive the neutral tint ``(0.5, 0.5, 0.5)``.
    """
    arr = np.asarray(data, dtype=np.float64)
    if isinstance(data, ma.MaskedArray):
        arr = np.where(ma.getmaskarray(data), np.nan, arr)
    n_rows, n_cols = arr.shape
    if not np.isfinite(arr).any():
        return np.full((n_rows, n_cols, 3), 0.5, dtype=np.float32)
    lo = float(np.nanmin(arr)) if vmin is None else vmin
    hi = float(np.nanmax(arr)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.5, 0.0, 1.0).astype(np.float32)
    g = np.clip(np.where(t < 0.5, 2.0 * t, 2.0 - 2.0 * t), 0.0, 1.0).astype(np.float32)
    b = np.clip(0.5 - 1.5 * (t - 1.0 / 3.0), 0.0, 1.0).astype(np.float32)
    rgb = np.stack([r, g, b], axis=2)  # (n_rows, n_cols, 3)
    rgb[np.isnan(arr)] = 0.5
    return rgb


class BodyMosaicWindow(QMainWindow):
    """Viewer window for a list of body reprojection / mosaic files."""

    def __init__(
        self,
        file_paths: list[str],
        *,
        initial_black: float | None = None,
        initial_white: float | None = None,
        initial_gamma: float = 0.5,
        show_parallels: bool = False,
        show_meridians: bool = False,
        show_lat_ticks: bool = False,
        show_lon_ticks: bool = False,
        initial_projection: ProjectionKind = ProjectionKind.RECT,
        parent: QWidget | None = None,
    ) -> None:
        """PyQt6 viewer for a list of body reprojection or mosaic files.

        Parameters:
            file_paths: Non-empty list of paths to body ``.npz`` / ``.fits`` data.
            initial_black: Optional fixed black level for stretch; ``None`` uses
                percentile stretch after load.
            initial_white: Optional fixed white level for stretch; ``None`` uses
                percentile stretch after load.
            initial_gamma: Initial display gamma (default ``0.5``).
            show_parallels: Pre-check graticule parallels overlay.
            show_meridians: Pre-check graticule meridians overlay.
            show_lat_ticks: Pre-check latitude axis ticks.
            show_lon_ticks: Pre-check longitude axis ticks.
            initial_projection: Initial map projection kind.
            parent: Optional Qt parent widget.

        Raises:
            ValueError: If ``file_paths`` is empty.
        """
        super().__init__(parent)
        if not file_paths:
            raise ValueError('file_paths must contain at least one path')
        self._file_paths = file_paths
        self._current_idx = 0
        self._display_data: BodyDisplayData | None = None
        self._initial_black = initial_black
        self._initial_white = initial_white
        self._initial_gamma = initial_gamma
        self._default_gamma = initial_gamma
        self._show_parallels = show_parallels
        self._show_meridians = show_meridians
        self._show_lat_ticks = show_lat_ticks
        self._show_lon_ticks = show_lon_ticks
        self._stretch_controls: dict[str, Any] = {}
        self._stretch_form: QFormLayout = QFormLayout()
        self._histogram_widget: HistogramStretchWidget | None = None
        self._chk_histogram_mode: QCheckBox | None = None
        self._pending_fit = False
        self._body_view_ma: ma.MaskedArray | None = None
        self._proj_kind: ProjectionKind = initial_projection
        self._colorby_alpha: float = 1.0
        self._setup_ui()
        self._chk_lat_ticks.setChecked(show_lat_ticks)
        self._chk_lon_ticks.setChecked(show_lon_ticks)
        # Sync the combo to the initial projection without triggering the slot yet
        self._proj_combo.blockSignals(True)
        idx = self._proj_combo.findData(initial_projection)
        if idx >= 0:
            self._proj_combo.setCurrentIndex(idx)
        self._proj_combo.blockSignals(False)
        self._load_file(0)
        # Apply non-RECT projection after the first file is loaded
        if self._proj_kind != ProjectionKind.RECT:
            self._on_projection_changed()

    def statusBar(self) -> QStatusBar:
        """Return this window's status bar as a non-optional ``QStatusBar``.

        Overrides ``QMainWindow.statusBar`` from Qt. Delegates to
        ``super().statusBar()`` and asserts the result is not ``None`` so callers
        may rely on a concrete ``QStatusBar`` (not ``Optional``); the first use
        in this class is from :meth:`_setup_ui`, which materializes the bar.

        Returns:
            The ``QStatusBar`` from the base implementation. The
            ``assert bar is not None`` narrows the type from ``QStatusBar | None``
            (Qt's return contract) to ``QStatusBar`` for static analysis and callers.
        """
        bar = super().statusBar()
        assert bar is not None
        return bar

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Qt resize hook: finish a deferred fit-to-window when the viewport gets size."""
        super().resizeEvent(event)
        if self._pending_fit:
            self._fit_zoom_to_window()

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the main window UI (layouts, header, projection, image, splitter, controls).

        Connects ``TiledImageWidget`` signals to ``_on_mouse_moved``, ``_on_zoom_changed``,
        and wires ``_proj_combo`` to ``_on_projection_changed``; shows default RECT status
        hint on the status bar.
        """
        self.setWindowTitle('Body Mosaic Viewer')
        self.resize(1400, 900)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        header = QWidget()
        hh = QHBoxLayout(header)
        hh.setContentsMargins(4, 2, 4, 0)
        hh.setSpacing(10)
        self._chk_lat_ticks = QCheckBox('Latitude axis ticks')
        self._chk_lon_ticks = QCheckBox('Longitude axis ticks')
        self._chk_lat_ticks.toggled.connect(self._update_axis_ticks)
        self._chk_lon_ticks.toggled.connect(self._update_axis_ticks)
        hh.addWidget(self._chk_lat_ticks)
        hh.addWidget(self._chk_lon_ticks)
        hh.addSpacing(16)
        self._chk_parallels = QCheckBox('Show parallels (lat)')
        self._chk_parallels.setChecked(self._show_parallels)
        self._chk_parallels.toggled.connect(self._update_overlays)
        self._chk_meridians = QCheckBox('Show meridians (lon)')
        self._chk_meridians.setChecked(self._show_meridians)
        self._chk_meridians.toggled.connect(self._update_overlays)
        hh.addWidget(self._chk_parallels)
        hh.addWidget(self._chk_meridians)
        hh.addSpacing(16)
        hh.addWidget(QLabel('Projection:'))
        self._proj_combo = QComboBox()
        for _kind, _label in [
            (ProjectionKind.RECT, 'Rectangular'),
            (ProjectionKind.POLAR_N, 'Polar North Stereographic'),
            (ProjectionKind.POLAR_S, 'Polar South Stereographic'),
            (ProjectionKind.MOLLWEIDE, 'Mollweide'),
            (ProjectionKind.SPHERE_3D, '3D Sphere'),
        ]:
            self._proj_combo.addItem(_label, _kind)
        self._proj_combo.currentIndexChanged.connect(self._on_projection_changed)
        hh.addWidget(self._proj_combo)
        hh.addStretch()
        left_layout.addWidget(header)

        self._image_widget = TiledImageWidget()
        self._image_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_widget.mouse_moved.connect(self._on_mouse_moved)
        self._image_widget.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self._image_widget, stretch=1)

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
        self.statusBar().showMessage(_STATUS_HINTS[ProjectionKind.RECT])

    def _build_right_panel(self) -> QWidget:
        """Build the right-hand file navigation panel (prev/next, list, labels)."""
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
        return right

    def _populate_file_list(self) -> None:
        """Fill the file list widget from ``self._file_paths``."""
        self._file_list.clear()
        for i, p in enumerate(self._file_paths):
            name = Path(p).name
            item = QListWidgetItem(f'{i + 1}. {name}')
            item.setToolTip(str(p))
            self._file_list.addItem(item)

    def _refresh_file_list_selection(self) -> None:
        """Highlight and scroll to ``self._current_idx`` in the file list without firing signals."""
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(self._current_idx)
        cur = self._file_list.currentItem()
        if cur is not None:
            self._file_list.scrollToItem(cur)
        self._file_list.blockSignals(False)

    def _on_file_list_row_changed(self, row: int) -> None:
        """Handle list selection change by loading the chosen file index when it changes."""
        if row < 0 or row >= len(self._file_paths):
            return
        if row == self._current_idx:
            return
        self._load_file(row)

    def _build_control_panel(self) -> QWidget:
        """Assemble the bottom control panel (stretch, zoom, cursor info, color-by, photometry)."""
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
                ('body', 'Body:'),
                ('lat', 'Latitude:'),
                ('lon', 'Longitude:'),
                ('latlon', 'Lat/lon mode:'),
            ],
            [
                ('phase', 'Phase angle:'),
                ('emission', 'Emission angle:'),
                ('incidence', 'Incidence angle:'),
                ('res', 'Resolution (km/px):'),
            ],
            [
                ('eff_res', 'Eff. resolution (km/px):'),
                ('ssl_lon', 'Sub-solar lon:'),
                ('ssl_lat', 'Sub-solar lat:'),
                ('subobs_lon', 'Sub-observer lon:'),
                ('subobs_lat', 'Sub-observer lat:'),
            ],
            [
                ('image', 'Source image:'),
            ],
        ]
        self._info: dict[str, QLabel] = {}
        name_w = 150
        val_w_default = 132
        val_w_image = 420
        for col_idx, col in enumerate(info_columns):
            base = col_idx * 2
            for row_idx, (key, name) in enumerate(col):
                nl = QLabel(name)
                nl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                nl.setFixedWidth(name_w)
                vl = QLabel('---')
                vl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                vw = val_w_image if col_idx == 3 else val_w_default
                vl.setFixedWidth(vw)
                info_grid.addWidget(nl, row_idx, base)
                info_grid.addWidget(vl, row_idx, base + 1)
                self._info[key] = vl
        info_grid.setColumnStretch(len(info_columns) * 2, 1)
        lower_h.addWidget(info_box, stretch=1)

        colorby_box = QGroupBox('Color By')
        cb_grid = QGridLayout(colorby_box)
        cb_grid.setContentsMargins(4, 2, 4, 2)
        cb_grid.setHorizontalSpacing(12)
        cb_grid.setVerticalSpacing(2)
        self._colorby_group = QButtonGroup()
        colorby_rows: list[list[tuple[str, str]]] = [
            [('none', 'None'), ('image_no', 'Image number')],
            [('res', 'Resolution (rel)'), ('eff_res', 'Eff. resolution (rel)')],
            [('abs_phase', 'Phase (abs)'), ('rel_phase', 'Phase (rel)')],
            [('abs_emission', 'Emission (abs)'), ('rel_emission', 'Emission (rel)')],
            [('abs_incidence', 'Incidence (abs)'), ('rel_incidence', 'Incidence (rel)')],
        ]
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
        """Create a zoom slider/edit pair synced to ``TiledImageWidget`` for one axis."""

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
        """Return the photometry radio key string (default ``as_saved`` if none checked)."""
        btn = self._photometry_group.checkedButton()
        if btn is None:
            return 'as_saved'
        return str(btn.property('photometry_key'))

    def _sync_photometry_ui(self, dd: BodyDisplayData) -> None:
        """Reset photometric display to file pixels when loading a new file (``as_saved``)."""
        _ = dd
        for b in self._photometry_group.buttons():
            if b.property('photometry_key') == 'as_saved':
                b.setChecked(True)
                break

    def _apply_body_display_image(self, *, preserve_view: bool) -> None:
        """Recompute photometry-corrected pixels and push them to the image widget."""
        dd = self._display_data
        if dd is None:
            return
        mode = self._photometry_mode()
        img = compute_body_display_image(
            mode=mode,
            image_ma=dd.image_ma,
            photometric_model_name=dd.photometric_model_name,
            phase_deg=dd.phase,
            emission_deg=dd.emission,
            incidence_deg=dd.incidence,
        )
        self._body_view_ma = img
        self._image_widget.set_image(
            img,
            x_interval=dd.lon_resolution_deg,
            y_interval=dd.lat_resolution_deg,
            x_label=f'Longitude ({dd.latlon_type}/{dd.lon_direction}) (°)',
            y_label='Latitude (°)',
            y_flip=False,
            body_full_sphere_canvas=True,
            body_lon_range_deg=dd.lon_range_deg,
            body_lat_range_deg=dd.lat_range_deg,
            preserve_view=preserve_view,
        )

    def _on_projection_changed(self, _idx: int = 0) -> None:
        """Switch the display projection and reset the view."""
        kind = self._proj_combo.currentData()
        if kind is None:
            return
        self._proj_kind = kind
        self._image_widget.set_projection(kind)
        self.statusBar().showMessage(_STATUS_HINTS.get(kind, ''))
        self._fit_zoom_to_window()
        self._sync_zoom_ui()

    def _on_photometry_changed(self, _btn: Any = None) -> None:
        """Refresh display image and restretch after the photometry mode radio changes."""
        if self._display_data is None:
            return
        self._apply_body_display_image(preserve_view=True)
        dd = self._display_data
        img = self._body_view_ma if self._body_view_ma is not None else dd.image_ma
        black, white = _percentile_stretch(img, 0.0, 98.0)
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_range'](dd.vmin, dd.vmax)
        self._set_stretch_levels(black, white, g)
        self._refresh_histogram_data()
        self._image_widget.set_stretch(black, white, g)
        self._on_colorby_changed(self._colorby_group.checkedButton())

    # ------------------------------------------------------------------
    #  File loading
    # ------------------------------------------------------------------

    def _on_prev(self) -> None:
        """Load the previous file in the list."""
        self._load_file(self._current_idx - 1)

    def _on_next(self) -> None:
        """Load the next file in the list."""
        self._load_file(self._current_idx + 1)

    def _load_file(self, idx: int) -> None:
        """Load body data at index ``idx``, refresh UI, overlays, and stretch.

        No-op if ``idx`` is out of range. On ``load_body_file`` failure, prints the
        exception and shows a status-bar message without changing the current file.
        """
        if not (0 <= idx < len(self._file_paths)):
            return
        path = self._file_paths[idx]
        try:
            with image_scope():
                dd = load_body_file(path)
        except Exception as exc:
            print(f'Error loading {path}', file=sys.stderr)
            traceback.print_exc()
            self.statusBar().showMessage(f'Error loading {path}: {exc}', 5000)
            return

        self._current_idx = idx
        self._display_data = dd
        self.setWindowTitle(f'Body Mosaic Viewer - {dd.title}  ({dd.body_name})')
        self._file_lbl.setText(f'{idx + 1} / {len(self._file_paths)}\n{dd.title}')
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._file_paths) - 1)

        self._sync_photometry_ui(dd)
        self._apply_body_display_image(preserve_view=False)
        self._update_axis_ticks()

        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 0.0, 98.0
        )
        if self._initial_black is not None:
            black = self._initial_black
        if self._initial_white is not None:
            white = self._initial_white
        self._stretch_controls['set_range'](dd.vmin, dd.vmax)
        self._set_stretch_levels(black, white, self._initial_gamma)
        self._refresh_histogram_data()
        self._image_widget.set_stretch(black, white, self._initial_gamma)

        self._clear_info()
        self._update_colorby_widgets_for_mosaic(dd.is_mosaic)
        self._on_colorby_changed(self._colorby_group.checkedButton())
        self._info['body'].setText(dd.body_name)
        self._info['latlon'].setText(f'{dd.latlon_type} / {dd.lon_direction}')

        self._update_overlays(self._chk_parallels.isChecked())
        self._fit_zoom_to_window()
        self._sync_zoom_ui()
        self.statusBar().showMessage(_STATUS_HINTS.get(self._proj_kind, ''))
        self._refresh_file_list_selection()

    # ------------------------------------------------------------------
    #  Stretch / zoom
    # ------------------------------------------------------------------

    def _apply_stretch(self) -> None:
        """Apply black/white/gamma from the stretch sliders to the image widget."""
        b = self._stretch_controls['from_slider'](self._stretch_controls['slider_black'].value())
        w = self._stretch_controls['from_slider'](self._stretch_controls['slider_white'].value())
        g = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._image_widget.set_stretch(b, w, g)

    def _set_stretch_levels(self, black: float, white: float, gamma: float) -> None:
        """Sync black/white/gamma into the slider controls and the histogram widget."""
        self._stretch_controls['set_values'](black, white, gamma)
        if self._histogram_widget is not None:
            self._histogram_widget.set_values(black, white)

    def _on_histogram_black_changed(self, black: float) -> None:
        """Histogram drag handler for the black indicator."""
        white = self._stretch_controls['from_slider'](
            self._stretch_controls['slider_white'].value()
        )
        gamma = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_values'](black, white, gamma)
        self._apply_stretch()

    def _on_histogram_white_changed(self, white: float) -> None:
        """Histogram drag handler for the white indicator."""
        black = self._stretch_controls['from_slider'](
            self._stretch_controls['slider_black'].value()
        )
        gamma = max(0.1, self._stretch_controls['slider_gamma'].value() / 100.0)
        self._stretch_controls['set_values'](black, white, gamma)
        self._apply_stretch()

    def _on_histogram_mode_toggled(self, checked: bool) -> None:
        """Swap between slider stretch UI and histogram-with-indicators UI."""
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
        """Push the active display image into the histogram widget."""
        if self._histogram_widget is None:
            return
        dd = self._display_data
        if dd is None:
            return
        view = self._body_view_ma if self._body_view_ma is not None else dd.image_ma
        self._histogram_widget.set_data(view)

    def _on_stretch_preset_reset(self) -> None:
        """Reset stretch to a 0-98 percentile of the current display image."""
        dd = self._display_data
        if dd is None:
            return
        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 0.0, 98.0
        )
        self._set_stretch_levels(black, white, self._default_gamma)
        self._apply_stretch()

    def _on_stretch_preset_full(self) -> None:
        """Stretch across the file's full ``vmin``-``vmax`` dynamic range."""
        dd = self._display_data
        if dd is None:
            return
        self._set_stretch_levels(dd.vmin, dd.vmax, self._default_gamma)
        self._image_widget.set_stretch(dd.vmin, dd.vmax, self._default_gamma)

    def _on_stretch_preset_bright(self) -> None:
        """Percentile stretch with a 2-98% window for a brighter default view."""
        dd = self._display_data
        if dd is None:
            return
        black, white = _percentile_stretch(
            self._body_view_ma if self._body_view_ma is not None else dd.image_ma, 2.0, 98.0
        )
        self._set_stretch_levels(black, white, self._default_gamma)
        self._apply_stretch()

    def _fit_zoom_to_window(self) -> None:
        """Fit zoom so the image fills the viewport (or defer if size not yet known)."""
        dd = self._display_data
        if dd is None:
            return
        vw = self._image_widget.viewport().width()
        vh = self._image_widget.viewport().height()
        if vw <= 0 or vh <= 0:
            self._pending_fit = True
            return
        self._pending_fit = False
        if self._proj_kind != ProjectionKind.RECT:
            self._image_widget.fit_projection_to_window()
            return
        if self._image_widget.is_body_full_sphere_canvas():
            n_rows, n_cols = self._image_widget.display_grid_shape()
        else:
            n_rows, n_cols = dd.image_ma.shape
        x_zoom = min(float(vw) / max(n_cols, 1), 100.0)
        y_zoom = min(float(vh) / max(n_rows, 1), 100.0)
        self._image_widget.set_zoom(x_zoom, y_zoom)

    def _on_zoom_in(self) -> None:
        """Multiply both zoom factors by 1.5 and refresh slider UI."""
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz * 1.5, yz * 1.5)
        self._sync_zoom_ui()

    def _on_zoom_out(self) -> None:
        """Divide both zoom factors by 1.5 and refresh slider UI."""
        xz, yz = self._image_widget.get_zoom()
        self._image_widget.set_zoom(xz / 1.5, yz / 1.5)
        self._sync_zoom_ui()

    def _on_zoom_reset(self) -> None:
        """Re-run fit-to-window and sync zoom readouts."""
        self._fit_zoom_to_window()
        self._sync_zoom_ui()

    def _on_zoom_changed(self, xz: float, yz: float) -> None:
        """Keep zoom sliders and the summary label in sync with widget zoom."""
        self._update_zoom_slider_ranges()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _update_zoom_slider_ranges(self) -> None:
        """Expand slider ranges to the widget's current minimum zoom limits."""
        min_x, min_y = self._image_widget.get_min_zoom()
        self._xzoom_sync.set_range(min_x, 100.0)
        self._yzoom_sync.set_range(min_y, 100.0)

    def _sync_zoom_ui(self) -> None:
        """Read zoom from the widget and push it into both axis controls."""
        self._update_zoom_slider_ranges()
        xz, yz = self._image_widget.get_zoom()
        self._xzoom_sync.set_value(xz)
        self._yzoom_sync.set_value(yz)
        self._zoom_info_lbl.setText(f'{xz:.2f}x / {yz:.2f}x')

    def _on_save_fov(self) -> None:
        """Prompt for a path and save the current viewport as PNG/JPEG."""
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
        """Apply or clear the metadata color tint on the image from the color-by choice."""
        if btn is None or self._display_data is None:
            self._image_widget.set_color_tint(None)
            return
        key = btn.property('colorby_key')
        tint = self._compute_color_tint(str(key), self._display_data)
        self._image_widget.set_color_tint(self._tint_with_alpha(tint))

    def _on_colorby_alpha_changed(self, value: int) -> None:
        """Update color-by tint blend strength and refresh the overlay."""
        self._colorby_alpha = value / 100.0
        self._on_colorby_changed(self._colorby_group.checkedButton())

    def _tint_with_alpha(self, tint: np.ndarray | None) -> np.ndarray | None:
        """Blend ``tint`` toward gray using ``self._colorby_alpha`` (or pass through)."""
        if tint is None or self._colorby_alpha >= 1.0:
            return tint
        return (self._colorby_alpha * tint + (1.0 - self._colorby_alpha)).astype(np.float32)

    def _compute_color_tint(self, key: str, dd: BodyDisplayData) -> np.ndarray | None:
        """Return per-pixel RGB tint (n_rows, n_cols, 3) float32, or None for grayscale.

        Parameters:
            key: Colorby radio button key string.
            dd: Current display data containing metadata arrays.

        Returns:
            Float32 array of shape ``(n_rows, n_cols, 3)`` with values in
            ``[0, 1]``, or ``None`` when no colorisation is requested.
        """
        if key == 'none':
            return None
        if key == 'image_no':
            if dd.image_number is None:
                return None
            vals = dd.image_number.astype(float)
            valid = ma.array(vals, mask=ma.getmaskarray(dd.image_number)).compressed()
            if valid.size == 0:
                return None
            return _colorby_tint(
                dd.image_number,
                vmin=float(valid.min()),
                vmax=float(valid.max()),
            )
        if key == 'res':
            return _colorby_tint(dd.resolution)
        if key == 'eff_res':
            return _colorby_tint(dd.eff_resolution)
        if key == 'abs_phase':
            return _colorby_tint(dd.phase, vmin=0.0, vmax=180.0)
        if key == 'rel_phase':
            return _colorby_tint(dd.phase)
        if key == 'abs_emission':
            return _colorby_tint(dd.emission, vmin=0.0, vmax=90.0)
        if key == 'rel_emission':
            return _colorby_tint(dd.emission)
        if key == 'abs_incidence':
            return _colorby_tint(dd.incidence, vmin=0.0, vmax=90.0)
        if key == 'rel_incidence':
            return _colorby_tint(dd.incidence)
        return None

    def _update_colorby_widgets_for_mosaic(self, is_mosaic: bool) -> None:
        """Enable/disable mosaic-only color-by options (e.g. image number)."""
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
    #  Overlays / axis ticks
    # ------------------------------------------------------------------

    def _update_overlays(self, _checked: bool = False) -> None:
        """Update the graticule overlay from the checkbox state.

        The body display is always built on the full-sphere canvas
        (:meth:`_apply_body_display_image` asks for one), which draws its own
        parallels and meridians in viewport coordinates, so the row / column
        overlays stay cleared.
        """
        dd = self._display_data
        show_parallels = dd is not None and self._chk_parallels.isChecked()
        show_meridians = dd is not None and self._chk_meridians.isChecked()
        self._image_widget.set_body_sphere_geo_overlays(show_parallels, show_meridians)
        self._image_widget.set_show_rows([])
        self._image_widget.set_show_cols([])

    def _update_axis_ticks(self, _checked: bool = False) -> None:
        """Push latitude/longitude tick visibility to the image widget."""
        dd = self._display_data
        if dd is None:
            return
        self._image_widget.set_axis_tick_options(
            show_x=self._chk_lon_ticks.isChecked(),
            show_y=self._chk_lat_ticks.isChecked(),
            y_tick_center=0.0,
            y_tick_labels_absolute=True,
        )

    # ------------------------------------------------------------------
    #  Cursor info
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_ma(arr: ma.MaskedArray, iy: int, ix: int, fmt: str) -> str:
        """Format one masked-array sample, or ``'---'`` when masked."""
        v = arr[iy, ix]
        if ma.is_masked(v):
            return '---'
        return fmt % float(v)

    @staticmethod
    def _fmt_deg_label(s: str) -> str:
        """Append a degree sign to a numeric angle string unless it is ``'---'``."""
        return f'{s}°' if s != '---' else '---'

    def _clear_info(self) -> None:
        """Clear the cursor status line and reset all info labels to placeholders."""
        self._cursor_status_lbl.setText('')
        for lbl in self._info.values():
            lbl.setText('---')

    def _on_mouse_moved(self, px: float, py: float, in_bounds: bool) -> None:
        """Handle pointer moves: map to lon/lat or pixel coords and refresh the info panel.

        Parameters:
            px: Pointer X in widget coordinates.
            py: Pointer Y in widget coordinates.
            in_bounds: False when the cursor leaves the image; clears info via
                ``_clear_info`` and returns early.

        Side effects:
            Updates ``self._cursor_status_lbl`` and ``self._info`` labels from
            ``self._display_data`` and ``self._proj_kind`` (RECT vs projected modes).
            No exceptions are raised; out-of-bounds or off-surface positions clear info.
        """
        if not in_bounds or self._display_data is None:
            self._clear_info()
            return
        dd = self._display_data
        view_ma = self._body_view_ma if self._body_view_ma is not None else dd.image_ma
        n_c = view_ma.shape[1]
        n_r = view_ma.shape[0]

        if self._proj_kind != ProjectionKind.RECT:
            # px/py are the cursor's own continuous viewport position; convert to
            # geographic without quantizing it to a whole screen pixel first.
            lon_deg, lat_deg, on_surface = self._image_widget.viewport_to_lonlat(px, py)
            if not on_surface:
                self._clear_info()
                return
            dc, dr, inside = self._image_widget.body_sphere_data_indices(lon_deg, lat_deg)
            coord_str = f'Lon: {lon_deg:9.4f}°  Lat: {lat_deg:8.4f}°'
        else:
            x_phys, y_phys = self._image_widget.pixel_to_physical(px, py)
            lon_deg = float(x_phys)
            lat_deg = float(y_phys)
            lon_min, lon_max = dd.lon_range_deg
            lat_min, _lat_max = dd.lat_range_deg
            d_lon = dd.lon_resolution_deg
            if self._image_widget.is_body_full_sphere_canvas():
                dc, dr, inside = self._image_widget.body_sphere_data_indices(lon_deg, lat_deg)
            else:
                # Each column and row is one point sample, so the column or row a
                # coordinate names is the one whose sample is nearest it.
                lm = lon_deg % 360.0
                dc = -1
                for cand in (lm, lm - 360.0, lm + 360.0):
                    if lon_min - 1e-9 <= cand <= lon_max + 1e-9:
                        dc = int(np.round((cand - lon_min) / d_lon))
                        break
                dr = int(np.round((lat_deg - lat_min) / dd.lat_resolution_deg))
                inside = 0 <= dc < n_c and 0 <= dr < n_r
            coord_str = f'X: {px:8.2f}  Y: {py:7.2f}'

        self._info['body'].setText(dd.body_name)
        self._info['latlon'].setText(f'{dd.latlon_type} / {dd.lon_direction}')

        ix = int(np.clip(dc, 0, n_c - 1))
        iy = int(np.clip(dr, 0, n_r - 1))

        raw_val = view_ma[iy, ix] if inside else ma.masked
        if ma.is_masked(raw_val):
            value_str = f'{"masked":>11}'
        else:
            value_str = f'{float(raw_val):11.8f}'

        time_s = '---'
        if inside:
            ph = self._fmt_ma(dd.phase, iy, ix, '%.3f')
            em = self._fmt_ma(dd.emission, iy, ix, '%.3f')
            inc = self._fmt_ma(dd.incidence, iy, ix, '%.3f')
            res = self._fmt_ma(dd.resolution, iy, ix, '%.4f')
            eff = self._fmt_ma(dd.eff_resolution, iy, ix, '%.4f')
            if dd.image_number is not None:
                img_v = dd.image_number[iy, ix]
                if not ma.is_masked(img_v):
                    idx = int(img_v)
                    names = dd.contributing_image_names
                    if idx < len(names) and names[idx]:
                        img_s = f'{names[idx]} (#{idx})'
                    else:
                        img_s = f'#{idx}'
                else:
                    img_s = '---'
            else:
                n0 = dd.contributing_image_names[0] if dd.contributing_image_names else ''
                img_s = f'{n0} (#0)' if n0 else dd.title
            # Sub-solar / sub-observer lon/lat: index by image_number for mosaics, or 0 for reproj
            ssl_arr_lon = dd.sub_solar_lon_per_image_deg
            ssl_arr_lat = dd.sub_solar_lat_per_image_deg
            sol_arr_lon = dd.sub_observer_lon_per_image_deg
            sol_arr_lat = dd.sub_observer_lat_per_image_deg
            if dd.image_number is not None and not ma.is_masked(dd.image_number[iy, ix]):
                geom_idx = int(dd.image_number[iy, ix])
            else:
                geom_idx = 0
            if geom_idx < len(ssl_arr_lon):
                ssl_lon_s = f'{ssl_arr_lon[geom_idx]:.4f}°'
                ssl_lat_s = f'{ssl_arr_lat[geom_idx]:.4f}°'
            else:
                ssl_lon_s = ssl_lat_s = '---'
            if geom_idx < len(sol_arr_lon):
                subobs_lon_s = f'{sol_arr_lon[geom_idx]:.4f}°'
                subobs_lat_s = f'{sol_arr_lat[geom_idx]:.4f}°'
            else:
                subobs_lon_s = subobs_lat_s = '---'
            if dd.observation_time_tdb is not None:
                tv = dd.observation_time_tdb[iy, ix]
                if not ma.is_masked(tv) and np.isfinite(float(tv)):
                    try:
                        time_s = et_to_utc(float(tv))
                    except (ValueError, OverflowError, RuntimeError, TypeError):
                        time_s = '---'
        else:
            ph = em = inc = res = eff = '---'
            img_s = ssl_lon_s = ssl_lat_s = subobs_lon_s = subobs_lat_s = '---'

        self._cursor_status_lbl.setText(f'{coord_str}  Value: {value_str}')
        self._info['lat'].setText(f'{lat_deg:.4f}°')
        self._info['lon'].setText(f'{lon_deg:.4f}°')
        self._info['phase'].setText(self._fmt_deg_label(ph))
        self._info['emission'].setText(self._fmt_deg_label(em))
        self._info['incidence'].setText(self._fmt_deg_label(inc))
        self._info['res'].setText(f'{res} km/px' if res != '---' else '---')
        self._info['eff_res'].setText(f'{eff} km/px' if eff != '---' else '---')
        self._info['ssl_lon'].setText(ssl_lon_s)
        self._info['ssl_lat'].setText(ssl_lat_s)
        self._info['subobs_lon'].setText(subobs_lon_s)
        self._info['subobs_lat'].setText(subobs_lat_s)
        if img_s != '---' and time_s != '---':
            source_display = f'{img_s}  \N{MIDDLE DOT}  {time_s}'
        else:
            source_display = img_s
        img_lbl = self._info['image']
        iw = img_lbl.width()
        if iw > 0 and source_display != '---':
            fm = QFontMetrics(img_lbl.font())
            source_display = fm.elidedText(source_display, Qt.TextElideMode.ElideRight, iw)
        img_lbl.setText(source_display)
