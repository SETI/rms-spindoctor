"""Shared state base class for the simulated-image editor mixins.

The editor is composed from one mixin per schema block (global fields, noise,
stray light, background stars, body / ring / star tabs, tab management,
render / display, scene I/O).  Every mixin needs the same widget references and
data model, so this base declares them once as annotations and every mixin
inherits it.  It also declares the handful of methods one mixin calls on
another; the owning mixin overrides each with the real implementation, so the
declarations here exist only to let mypy resolve the cross-mixin calls (they
are never invoked on the base itself).
"""

from typing import Any

import numpy as np
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.widgets import ImageLabel, ParameterUpdater
from spindoctor.ui.common import ZoomPanController


class SimEditorBase(QMainWindow):
    """Declares the editor's shared widgets, data model, and mixin seams."""

    # ---- Data model and render cache ----
    sim_params: dict[str, Any]
    _current_image: np.ndarray | None
    _last_meta: dict[str, Any]
    _base_pixmap: QPixmap | None

    # ---- View / interaction state ----
    _zoom_factor: float
    _right_drag_active: bool
    _selected_model_key: tuple[str, int] | None
    _last_drag_img_vu: tuple[float, float] | None
    _last_valid_tab_index: int
    _show_visual_aids: bool
    _show_saturation_overlay: bool
    _zoom_sharp: bool
    _syncing: bool
    _updater: ParameterUpdater
    _zoom_ctl: ZoomPanController

    # ---- Left image panel ----
    _scroll_area: QScrollArea
    _image_label: ImageLabel
    _status_label: QLabel
    _saturation_label: QLabel
    _zoom_label: QLabel

    # ---- Right tabs panel ----
    _warning_label: QLabel
    _tabs: QTabWidget
    _general_tab: QWidget
    _add_tab_widget: QWidget

    # ---- General tab: global fields ----
    _size_v_spin: QSpinBox
    _size_u_spin: QSpinBox
    _offset_v_spin: QDoubleSpinBox
    _offset_u_spin: QDoubleSpinBox
    _offset_rotation_spin: QDoubleSpinBox
    _exposure_spin: QDoubleSpinBox
    _random_seed_spin: QSpinBox
    _instrument_combo: QComboBox
    _fit_rotation_combo: QComboBox
    _midtime_edit: QLineEdit
    _closest_planet_combo: QComboBox
    _time_spin: QDoubleSpinBox
    _epoch_spin: QDoubleSpinBox

    # ---- General tab: noise panel ----
    _poisson_check: QCheckBox
    _read_noise_spin: QDoubleSpinBox
    _cosmic_ray_spin: QDoubleSpinBox
    _missing_data_spin: QDoubleSpinBox
    _bias_spin: QDoubleSpinBox
    _bloom_spin: QSpinBox
    _signal_frac_spin: QDoubleSpinBox
    _pixel_area_spin: QDoubleSpinBox

    # ---- Optics tab: stray-light panel ----
    _stray_amplitude_spin: QDoubleSpinBox
    _stray_direction_spin: QDoubleSpinBox
    _stray_model_combo: QComboBox
    _stray_center_check: QCheckBox
    _stray_center_v_spin: QDoubleSpinBox
    _stray_center_u_spin: QDoubleSpinBox

    # ---- Optics tab: fixed tab and its sub-block groups ----
    _optics_tab: QWidget
    _psf_optics_group: QGroupBox
    _psf_match_nav_check: QCheckBox
    _psf_sigma_v_spin: QDoubleSpinBox
    _psf_sigma_u_spin: QDoubleSpinBox
    _psf_w_spin: QDoubleSpinBox
    _psf_r0_spin: QDoubleSpinBox
    _psf_n_spin: QDoubleSpinBox
    _smear_group: QGroupBox
    _smear_rows_layout: QVBoxLayout
    _smear_rows: list[Any]
    _distortion_group: QGroupBox
    _distortion_k1_spin: QDoubleSpinBox
    _distortion_k2_spin: QDoubleSpinBox
    _distortion_center_v_spin: QDoubleSpinBox
    _distortion_center_u_spin: QDoubleSpinBox
    _distortion_nonradial_spin: QDoubleSpinBox
    _ghosts_group: QGroupBox
    _ghosts_rows_layout: QVBoxLayout
    _ghost_rows: list[Any]
    _stray_group: QGroupBox
    _oversample_check: QCheckBox
    _oversample_spin: QSpinBox
    _spk_error_group: QGroupBox
    _spk_dv_spin: QDoubleSpinBox
    _spk_du_spin: QDoubleSpinBox
    _spk_range_spin: QDoubleSpinBox

    # ---- Artifacts tab: fixed tab and its groups ----
    _artifacts_tab: QWidget
    _instrument_defaults_check: QCheckBox
    _adversarial_check: QCheckBox
    _detector_group: QGroupBox
    _detector_gain_state_spin: QSpinBox
    _detector_model_combo: QComboBox
    _detector_exposure_ref_spin: QDoubleSpinBox
    _detector_quantization_combo: QComboBox
    _mode_rows: dict[str, Any]

    # ---- General tab: PSF preview ----
    _psf_group: QGroupBox
    _psf_image_label: QLabel
    _psf_info_label: QLabel

    # ---- General tab: background sky (sky_counts) ----
    _sky_counts_check: QCheckBox
    _sky_density_slider: QSlider
    _sky_density_spin: QDoubleSpinBox
    _sky_a_spin: QDoubleSpinBox
    _sky_b_spin: QDoubleSpinBox
    _sky_diffuse_spin: QDoubleSpinBox
    _star_scatter_check: QCheckBox
    _star_scatter_spin: QDoubleSpinBox

    # ---- General tab: expected outcome (test-only) ----
    _expected_group: QGroupBox
    _expected_status_combo: QComboBox
    _expected_tier_combo: QComboBox
    _expected_reason_edit: QLineEdit

    # ---- Action buttons and visual toggles ----
    _save_img_btn: QPushButton
    _save_scene_btn: QPushButton
    _load_scene_btn: QPushButton
    _visual_aids_check: QCheckBox
    _zoom_sharp_check: QCheckBox
    _saturation_overlay_check: QCheckBox

    # ---- Cross-mixin seams (each owning mixin overrides its own) ----
    def _update_psf_preview(self) -> None:
        """Refresh the PSF preview inset (implemented in RenderDisplayMixin)."""
        raise NotImplementedError

    def _find_tab_by_properties(self, kind: str, data_index: int) -> int | None:
        """Locate a tab by kind and index (implemented in TabsMixin)."""
        raise NotImplementedError

    def _update_tab_titles(self) -> None:
        """Rebuild tabs to keep titles sorted (implemented in TabsMixin)."""
        raise NotImplementedError

    def _validate_ranges(self) -> None:
        """Warn on duplicate body ranges (implemented in TabsMixin)."""
        raise NotImplementedError

    def _delete_tab_by_index(self, kind: str, data_index: int) -> None:
        """Delete an object tab (implemented in TabsMixin)."""
        raise NotImplementedError

    def _rebuild_dynamic_tabs(self) -> None:
        """Rebuild the object tabs from the data model (TabsMixin)."""
        raise NotImplementedError

    def _build_body_tab(self, idx: int) -> QWidget:
        """Build a body tab widget (implemented in BodyTabMixin)."""
        raise NotImplementedError

    def _build_body_appearance_groups(self, w: QWidget, idx: int, layout: QVBoxLayout) -> None:
        """Add the appearance groups to a body tab (BodyAppearanceMixin)."""
        raise NotImplementedError

    def _sync_body_mesh_enabled(self, w: QWidget, is_mesh: bool) -> None:
        """Gate the mesh-extras group by shape (BodyAppearanceMixin)."""
        raise NotImplementedError

    def _body(self, idx: int) -> dict[str, Any] | None:
        """The body dict at ``idx``, or None (BodyAppearanceMixin)."""
        raise NotImplementedError

    def _body_tab_widget(self, idx: int) -> QWidget | None:
        """The built body tab widget at ``idx``, or None (BodyAppearanceMixin)."""
        raise NotImplementedError

    def _build_atmosphere_group(self, w: QWidget, idx: int, p: dict[str, Any]) -> QGroupBox:
        """Build the body atmosphere group (BodyAtmosphereMixin)."""
        raise NotImplementedError

    def _build_ring_tab(self, idx: int) -> QWidget:
        """Build a ring tab widget (implemented in RingTabMixin)."""
        raise NotImplementedError

    def _build_ring_advanced_groups(self, w: QWidget, idx: int, layout: QVBoxLayout) -> None:
        """Add the advanced ring groups to a ring tab (RingAdvancedMixin)."""
        raise NotImplementedError

    def _ring_features(self) -> list[dict[str, Any]]:
        """The live ring_system feature list (implemented in RingTabMixin)."""
        raise NotImplementedError

    def _build_star_tab(self, idx: int) -> QWidget:
        """Build a star tab widget (implemented in StarTabMixin)."""
        raise NotImplementedError

    def _noise_value(self, key: str, default: Any) -> Any:
        """Read from the noise block (implemented in NoiseMixin)."""
        raise NotImplementedError

    def _stray_value(self, key: str, default: Any) -> Any:
        """Read from the stray_light block (implemented in StrayLightMixin)."""
        raise NotImplementedError

    def _build_stray_panel(self, gen_layout: QFormLayout) -> None:
        """Populate a stray-light form (implemented in StrayLightMixin)."""
        raise NotImplementedError

    def _set_sky_widgets_enabled(self, enabled: bool) -> None:
        """Enable/disable the sky_counts widgets (BackgroundStarsMixin)."""
        raise NotImplementedError

    def _set_stray(self, key: str, value: Any) -> None:
        """Write into the stray_light block (implemented in StrayLightMixin)."""
        raise NotImplementedError

    def _on_stray_center(self, key: str, value: float) -> None:
        """Write a stray-light center key (implemented in StrayLightMixin)."""
        raise NotImplementedError

    def _stray_has_center(self) -> bool:
        """True when a stray-light center key is authored (StrayLightMixin)."""
        raise NotImplementedError

    def _stray_center_defaults(self) -> tuple[float, float]:
        """The effective stray-light center default (StrayLightMixin)."""
        raise NotImplementedError

    def _build_optics_tab(self) -> QWidget:
        """Build the Optics tab (implemented in OpticsTabMixin)."""
        raise NotImplementedError

    def _build_artifacts_tab(self) -> QWidget:
        """Build the Artifacts tab (implemented in ArtifactsTabMixin)."""
        raise NotImplementedError

    def _sync_optics_from_params(self) -> None:
        """Sync the Optics-tab widgets from sim_params (OpticsTabMixin)."""
        raise NotImplementedError

    def _sync_artifacts_from_params(self) -> None:
        """Sync the Artifacts-tab widgets from sim_params (ArtifactsTabMixin)."""
        raise NotImplementedError

    def _refresh_detector_catalog_defaults(self) -> None:
        """Refresh displayed detector catalog defaults (ArtifactsTabMixin)."""
        raise NotImplementedError

    def _refresh_artifact_mode_availability(self) -> None:
        """Refresh per-instrument mode-row availability (ArtifactsTabMixin)."""
        raise NotImplementedError
