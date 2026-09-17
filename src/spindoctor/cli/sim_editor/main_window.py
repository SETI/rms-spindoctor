"""The simulated-image editor's main window.

``CreateSimulatedImageModel`` assembles the per-schema-block mixins into one
``QMainWindow``.  This module owns only the cross-cutting scaffolding: the data
model defaults, the top-level layout, the pan / zoom / status-bar interaction,
and the visual-aid toggles.  Each schema block's widgets, handlers, and (in
later phases) new control tabs live in their own mixin module.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.artifacts_tab import ArtifactsTabMixin
from spindoctor.cli.sim_editor.background_stars import BackgroundStarsMixin
from spindoctor.cli.sim_editor.base import SimEditorBase
from spindoctor.cli.sim_editor.body_appearance import BodyAppearanceMixin
from spindoctor.cli.sim_editor.body_atmosphere import BodyAtmosphereMixin
from spindoctor.cli.sim_editor.body_tab import BodyTabMixin
from spindoctor.cli.sim_editor.expected_outcome import ExpectedOutcomeMixin
from spindoctor.cli.sim_editor.global_fields import GlobalFieldsMixin
from spindoctor.cli.sim_editor.noise import NoiseMixin
from spindoctor.cli.sim_editor.optics_tab import OpticsTabMixin
from spindoctor.cli.sim_editor.render_display import RenderDisplayMixin
from spindoctor.cli.sim_editor.ring_advanced import RingAdvancedMixin
from spindoctor.cli.sim_editor.ring_tab import RingTabMixin
from spindoctor.cli.sim_editor.scene_io import SceneIoMixin
from spindoctor.cli.sim_editor.star_tab import StarTabMixin
from spindoctor.cli.sim_editor.stray_light import StrayLightMixin
from spindoctor.cli.sim_editor.tabs import TabsMixin
from spindoctor.cli.sim_editor.widgets import ImageLabel, ParameterUpdater
from spindoctor.ui.common import ZoomPanController


class CreateSimulatedImageModel(
    GlobalFieldsMixin,
    NoiseMixin,
    StrayLightMixin,
    OpticsTabMixin,
    ArtifactsTabMixin,
    BackgroundStarsMixin,
    ExpectedOutcomeMixin,
    BodyAppearanceMixin,
    BodyAtmosphereMixin,
    BodyTabMixin,
    RingAdvancedMixin,
    RingTabMixin,
    StarTabMixin,
    TabsMixin,
    RenderDisplayMixin,
    SceneIoMixin,
    SimEditorBase,
):
    """Interactive editor for a simulated-image scene with a live preview."""

    def __init__(self) -> None:
        """Initialize the data model, build the UI, and render the first frame."""
        super().__init__()
        # The docs call this tool the sd_create_simulated_image scene editor.
        self.setWindowTitle('SpinDoctor Scene Editor')
        self.setMinimumSize(1300, 850)

        # Data model mirrors JSON schema
        self.sim_params: dict[str, Any] = {
            'size_v': 512,
            'size_u': 512,
            'offset_v': 0.0,
            'offset_u': 0.0,
            'offset_rotation_deg': 0.0,
            'exposure_sec': 1.0,
            'random_seed': 42,
            'instrument': 'generic',
            'closest_planet': 'SATURN',
            'time': 0.0,
            'ring_epoch': 0.0,
            'noise': {
                'poisson': True,
                'read_noise_dn': 4.0,
                'cosmic_ray_rate_per_sec': 0.0,
                'missing_data_rate': 0.0,
            },
            'stars': [],
            'bodies': [],
        }

        # Render cache/meta
        self._current_image: np.ndarray | None = None
        self._last_meta: dict[str, Any] = {}
        self._base_pixmap = None

        # View state
        self._zoom_factor = 1.0
        self._right_drag_active = False
        # ('body'/'star'/'ring', index)
        self._selected_model_key = None
        self._last_drag_img_vu = None
        # Track last valid (non-"+") tab for cancel behavior
        self._last_valid_tab_index = 0  # Start with General tab

        self._show_visual_aids = True
        self._show_saturation_overlay = False
        self._zoom_sharp = True
        self._syncing = False

        self._updater = ParameterUpdater(140)
        self._updater.update_requested.connect(self._update_image)

        self._setup_ui()
        self._update_image()

    def _setup_ui(self) -> None:
        """Build the window layout and wire the General tab and action buttons."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left image with zoom/pan
        left = QVBoxLayout()
        zoom_row = QHBoxLayout()
        zoom_row.addStretch()
        self._zoom_out_btn = QPushButton('Zoom -')
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_row.addWidget(self._zoom_out_btn)
        self._zoom_in_btn = QPushButton('Zoom +')
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_row.addWidget(self._zoom_in_btn)
        self._reset_view_btn = QPushButton('Reset View')
        self._reset_view_btn.clicked.connect(self._reset_view)
        zoom_row.addWidget(self._reset_view_btn)
        zoom_row.addStretch()
        left.addLayout(zoom_row)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setMinimumSize(700, 700)
        self._scroll_area.setStyleSheet('background-color: #303000;')
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = ImageLabel(
            self,
            self._on_press,
            self._on_move,
            self._on_release,
            self._on_wheel,
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet('background-color: #800000;')
        self._image_label.setMouseTracking(True)
        self._scroll_area.setWidget(self._image_label)

        left.addWidget(self._scroll_area)
        main_layout.addLayout(left, stretch=2)

        # Status bar
        status_bar = QStatusBar()
        self._status_label = QLabel('V, U: --------, --------  Value: --')
        status_bar.addWidget(self._status_label)
        self._saturation_label = QLabel('')
        status_bar.addPermanentWidget(self._saturation_label)
        self._zoom_label = QLabel('Zoom: 1.00x')
        status_bar.addPermanentWidget(self._zoom_label)
        self.setStatusBar(status_bar)

        # Right tabs panel
        right = QVBoxLayout()

        # Warning label for range duplicates
        self._warning_label = QLabel('')
        self._warning_label.setStyleSheet('color: orange;')
        right.addWidget(self._warning_label)

        self._tabs = QTabWidget()
        self._tabs.setMovable(False)  # Prevent manual reordering
        # Connect tab bar click to detect clicks on "+" tab
        self._tabs.tabBarClicked.connect(self._on_tab_bar_clicked)
        # Track current tab changes to remember last valid tab
        self._tabs.currentChanged.connect(self._on_tab_changed)
        right.addWidget(self._tabs, stretch=1)

        # General tab (always first): each schema block contributes its rows.
        self._general_tab = QWidget()
        gen_layout = QFormLayout(self._general_tab)
        self._build_global_fields(gen_layout)
        self._build_noise_panel(gen_layout)
        self._build_psf_preview(gen_layout)
        self._build_background_stars_panel(gen_layout)
        self._build_expected_panel(gen_layout)

        # Add General tab first, then the fixed Optics and Artifacts tabs.
        self._tabs.addTab(self._general_tab, 'General')
        self._tabs.addTab(self._build_optics_tab(), 'Optics')
        self._tabs.addTab(self._build_artifacts_tab(), 'Artifacts')

        # Add "+" tab for adding new objects (fake tab - just header, no content, always last)
        self._add_tab_widget = QWidget()
        self._tabs.addTab(self._add_tab_widget, '+')

        # Ensure correct tab order
        self._ensure_tab_order()

        # Buttons row (no Add/Delete buttons - handled by tabs)
        btns = QHBoxLayout()
        btns.addStretch()

        self._save_img_btn = QPushButton('Save Image (PNG)')
        self._save_img_btn.clicked.connect(self._save_image)
        btns.addWidget(self._save_img_btn)

        self._save_scene_btn = QPushButton('Save Scene (YAML)')
        self._save_scene_btn.clicked.connect(self._save_scene)
        btns.addWidget(self._save_scene_btn)

        self._load_scene_btn = QPushButton('Load Scene (YAML)')
        self._load_scene_btn.clicked.connect(self._load_scene)
        btns.addWidget(self._load_scene_btn)

        right.addLayout(btns)

        # Visual options with Exit button on same line
        vis_row = QHBoxLayout()
        self._visual_aids_check = QCheckBox('Show Visual Aids')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._visual_aids_check.stateChanged.connect(self._toggle_visual_aids)
        vis_row.addWidget(self._visual_aids_check)
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        vis_row.addWidget(self._zoom_sharp_check)
        self._saturation_overlay_check = QCheckBox('Saturation overlay')
        self._saturation_overlay_check.setChecked(self._show_saturation_overlay)
        self._saturation_overlay_check.setToolTip(
            'Highlight pixels at or above the instrument saturation DN in red.'
        )
        self._saturation_overlay_check.toggled.connect(self._toggle_saturation_overlay)
        vis_row.addWidget(self._saturation_overlay_check)
        vis_row.addStretch()
        exit_btn = QPushButton('Exit')
        exit_btn.clicked.connect(self.close)
        vis_row.addWidget(exit_btn)
        right.addLayout(vis_row)

        main_layout.addLayout(right, stretch=1)
        # Initialize common zoom/pan controller for left-button pan and wheel zoom
        self._zoom_ctl = ZoomPanController(
            label=self._image_label,
            scroll_area=self._scroll_area,
            get_zoom=lambda: self._zoom_factor,
            set_zoom=lambda z: setattr(self, '_zoom_factor', float(z)),
            update_display=self._update_display,
            set_zoom_label_text=lambda s: self._zoom_label.setText(s),
        )

    # ---- Event handlers: pan/zoom ----
    def _on_press(self, event: QMouseEvent) -> None:
        """Start a left-button pan or right-button object selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_press(event)
            self._image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            # Select model at cursor
            img_v, img_u = self._label_pos_to_image_vu(event.position())
            self._select_model_at(img_v, img_u)
            self._right_drag_active = True
            self._last_drag_img_vu = (img_v, img_u)

    def _on_move(self, event: QMouseEvent) -> None:
        """Pan or drag the selected object and update the status bar."""
        self._zoom_ctl.on_mouse_move(event)
        # status
        self._update_status_bar(event.position())

        # Right-drag to move selected model
        if self._right_drag_active and self._selected_model_key is not None:
            img_v, img_u = self._label_pos_to_image_vu(event.position())
            self._move_selected_by(img_v, img_u)

    def _on_release(self, event: QMouseEvent) -> None:
        """End a left-button pan or right-button drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_release(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_drag_active = False
            self._last_drag_img_vu = None

    def _on_wheel(self, event: QWheelEvent) -> None:
        """Zoom the preview on a wheel event."""
        self._zoom_ctl.on_wheel(event)

    def _zoom_in(self) -> None:
        """Zoom in about the viewport center."""
        # The ZoomPanController's center-anchored zoom is identical to the
        # open-coded version (it wraps this window's scroll area + zoom state).
        if self._base_pixmap is not None:
            self._zoom_ctl.zoom_in_center()

    def _zoom_out(self) -> None:
        """Zoom out about the viewport center."""
        if self._base_pixmap is not None:
            self._zoom_ctl.zoom_out_center()

    def _zoom_at_point(
        self,
        factor: float,
        viewport_x: int,
        viewport_y: int,
        scaled_x: float,
        scaled_y: float,
    ) -> None:
        """Zoom by ``factor`` anchored at a viewport point."""
        if self._base_pixmap is None:
            return
        # The ZoomPanController owns the zoom clamp + no-op short-circuit, so
        # delegate directly instead of re-deriving the clamped zoom here.
        self._zoom_ctl.zoom_at_point(factor, viewport_x, viewport_y, scaled_x, scaled_y)

    def _reset_view(self) -> None:
        """Reset the zoom to 1x and refresh the display."""
        self._zoom_factor = 1.0
        self._zoom_label.setText(f'Zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _label_pos_to_image_vu(self, label_pos: QPointF) -> tuple[float, float]:
        """Convert a label position to an image (v, u) position.

        A Qt event position is already a continuous pixel corner coordinate, the
        same measure a scene states every position in, and the zoom divide keeps
        it continuous.  Taking the whole number first would quantize the result to
        multiples of ``1 / zoom``, and at zoom 1 no cursor position could then
        name a pixel's center at all.

        Parameters:
            label_pos: Cursor position in label coordinates.

        Returns:
            ``(v, u)`` in image pixel corner coordinates.
        """
        img_u = float(label_pos.x()) / self._zoom_factor
        img_v = float(label_pos.y()) / self._zoom_factor
        return img_v, img_u

    def _update_status_bar(self, label_pos: QPointF) -> None:
        """Update the status bar with the cursor's (v, u) and pixel value.

        The printed position is the continuous pixel corner one; the sampled pixel
        is the one containing it, which is its floor.

        Parameters:
            label_pos: Cursor position in label coordinates.
        """
        self._zoom_label.setText(f'Zoom: {self._zoom_factor:.2f}x')
        if self._current_image is None:
            self._status_label.setText('V, U: --------, --------  Value: --')
            return
        img_v, img_u = self._label_pos_to_image_vu(label_pos)
        height, width = self._current_image.shape
        if 0 <= img_v < height and 0 <= img_u < width:
            v0 = int(img_v)
            u0 = int(img_u)
            val = self._current_image[v0, u0]
            self._status_label.setText(f'V, U: {img_v:8.2f}, {img_u:8.2f}  Value: {val:9.6f}')
        else:
            self._status_label.setText('V, U: --------, --------  Value: --')

    # ---- Visual toggles ----
    def _toggle_visual_aids(self, state: Any) -> None:
        """Toggle the body/star/ring center overlays."""
        if isinstance(state, Qt.CheckState):
            self._show_visual_aids = state is Qt.CheckState.Checked
        elif isinstance(state, int):
            self._show_visual_aids = state == cast(int, Qt.CheckState.Checked.value)
        else:
            self._show_visual_aids = False
        if self._current_image is not None:
            self._base_pixmap = None
            self._display_image()

    def _toggle_zoom_sharp(self, state: Any) -> None:
        """Toggle nearest-neighbor (sharp) vs smooth zoom scaling."""
        self._zoom_sharp = state == int(cast(int, Qt.CheckState.Checked.value))
        self._update_display()


def build_arg_parser() -> argparse.ArgumentParser:
    """The scene editor's command-line parser.

    Kept separate from :func:`main` so ``--help`` and argument handling are
    testable without constructing a Qt application.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog='sd_create_simulated_image',
        description=(
            'SpinDoctor Scene Editor: an interactive editor for simulated-image '
            'scenes with a live preview.'
        ),
    )
    parser.add_argument(
        'scene',
        nargs='?',
        type=Path,
        default=None,
        help=(
            'optional scene YAML to open on launch (the same files the Load '
            'Scene button reads); omitted, the editor starts with a blank scene'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Launch the simulated-image scene editor as a standalone application.

    Parameters:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).  An
            optional positional scene path is loaded through the editor's
            Load Scene machinery after the window is built.
    """
    args = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv)
    window = CreateSimulatedImageModel()
    if args.scene is not None:
        window.load_scene_file(args.scene)
    window.show()
    sys.exit(app.exec())
