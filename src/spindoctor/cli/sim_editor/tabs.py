"""Tab management for the simulated-image editor.

Owns the object-tab lifecycle: adding bodies / rings / stars via the ``+`` tab,
keeping ``General`` first and ``+`` last, rebuilding the dynamic tabs in sorted
order when the data model changes, deleting tabs, and the range-uniqueness
bookkeeping.
"""

import re
from typing import Any

from PyQt6.QtWidgets import QMessageBox, QWidget

from spindoctor.cli.sim_editor.base import SimEditorBase


class TabsMixin(SimEditorBase):
    """Builds and manages the dynamic object tabs."""

    def _ensure_tab_order(self) -> None:
        """Ensure General is first and '+' is last."""
        # Block signals to prevent tab change events during reordering
        self._tabs.blockSignals(True)

        general_idx = -1
        plus_idx = -1
        for i in range(self._tabs.count()):
            text = self._tabs.tabText(i)
            if text == 'General':
                general_idx = i
            elif text == '+':
                plus_idx = i

        # Remember current tab before reordering
        current_idx = self._tabs.currentIndex()
        current_widget = None
        if current_idx >= 0 and current_idx < self._tabs.count():
            current_widget = self._tabs.widget(current_idx)

        # Move General to first if needed
        if general_idx >= 0 and general_idx != 0:
            general_widget = self._tabs.widget(general_idx)
            self._tabs.removeTab(general_idx)
            self._tabs.insertTab(0, general_widget, 'General')
            # Recalculate plus_idx after removal
            for i in range(self._tabs.count()):
                if self._tabs.tabText(i) == '+':
                    plus_idx = i
                    break

        # Move "+" to last if needed
        if plus_idx >= 0 and plus_idx != self._tabs.count() - 1:
            plus_widget = self._tabs.widget(plus_idx)
            self._tabs.removeTab(plus_idx)
            self._tabs.addTab(plus_widget, '+')

        # Restore current tab if it still exists
        if current_widget is not None:
            for i in range(self._tabs.count()):
                if self._tabs.widget(i) == current_widget:
                    self._tabs.setCurrentIndex(i)
                    break

        # Unblock signals
        self._tabs.blockSignals(False)

    def _on_tab_changed(self, index: int) -> None:
        """Track tab changes and intercept switches to the '+' tab."""
        # Ignore invalid indices (can happen during tab rebuilding)
        if index < 0 or index >= self._tabs.count():
            return

        # If signals are blocked, we're in the middle of a programmatic change - don't intercept
        if self._tabs.signalsBlocked():
            # Still track valid tabs for future reference
            tab_text = self._tabs.tabText(index)
            if tab_text != '+':
                self._last_valid_tab_index = index
            return

        tab_text = self._tabs.tabText(index)

        # If switching to the "+" tab, intercept it
        if tab_text == '+':
            # Get the last valid tab index
            prev_tab = self._last_valid_tab_index
            # Ensure it's valid
            if (
                prev_tab < 0
                or prev_tab >= self._tabs.count()
                or self._tabs.tabText(prev_tab) == '+'
            ):
                # Fallback: find the last non-"+", non-General tab, or use General
                prev_tab = 0  # Default to General
                # Start from second-to-last, skip General
                for i in range(self._tabs.count() - 2, 0, -1):
                    if self._tabs.tabText(i) != '+':
                        prev_tab = i
                        break

            # Block signals to prevent recursion
            self._tabs.blockSignals(True)
            # Switch back to the previous tab immediately (before showing dialog)
            self._tabs.setCurrentIndex(prev_tab)
            self._tabs.blockSignals(False)

            # Now show the dialog
            result = self._add_tab_dialog()
            # If canceled, we've already switched back, so we're done
            # If successful, the new tab will be created and automatically selected
            if not result and (
                prev_tab >= 0
                and prev_tab < self._tabs.count()
                and self._tabs.tabText(prev_tab) != '+'
            ):
                # Make sure we're still on the previous tab (should already be, but be explicit)
                self._tabs.blockSignals(True)
                self._tabs.setCurrentIndex(prev_tab)
                self._tabs.blockSignals(False)
        else:
            # This is a valid tab, remember it
            self._last_valid_tab_index = index

    def _on_tab_bar_clicked(self, index: int) -> None:
        """Track tab-bar clicks; interception happens in _on_tab_changed."""
        # This is just for tracking - the actual interception happens in _on_tab_changed
        pass

    def _add_tab_dialog(self) -> bool:
        """Show dialog to add object. Returns True if object was added, False if canceled."""
        msg = QMessageBox(self)
        msg.setWindowTitle('Add object')
        msg.setText('Add what type of model?')
        body_btn = msg.addButton('Body', QMessageBox.ButtonRole.AcceptRole)
        ring_btn = msg.addButton('Ring', QMessageBox.ButtonRole.AcceptRole)
        star_btn = msg.addButton('Star', QMessageBox.ButtonRole.AcceptRole)
        msg.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == body_btn:
            self._add_body_tab()
            return True
        elif clicked == star_btn:
            self._add_star_tab()
            return True
        elif clicked == ring_btn:
            self._add_ring_tab()
            return True
        else:
            return False

    def _find_unique_name(self, base_name: str) -> str:
        """Find a unique name by incrementing the number suffix if needed.

        Checks bodies, stars, and ring_system features to ensure the name is
        unique.
        """
        # Collect all existing names (case-insensitive)
        existing_names = set()
        for body in self.sim_params.get('bodies', []):
            existing_names.add(body.get('name', '').lower())
        for star in self.sim_params.get('stars', []):
            existing_names.add(star.get('name', '').lower())
        for feature in self._ring_features():
            existing_names.add(feature.get('name', '').lower())

        # Try the base name first
        if base_name.lower() not in existing_names:
            return base_name

        # Extract base prefix and number if present
        match = re.match(r'^(.+?)(\d+)$', base_name)
        if match:
            prefix = match.group(1)
            start_num = int(match.group(2))
        else:
            # No number suffix, add one
            prefix = base_name
            start_num = 1

        # Increment until we find a unique name
        num = start_num + 1
        while True:
            candidate = f'{prefix}{num}'
            if candidate.lower() not in existing_names:
                return candidate
            num += 1

    def _add_body_tab(self, params: dict[str, Any] | None = None) -> None:
        """Append a body to the data model and select its new tab."""
        if params is None:
            default_name = f'Body{len(self.sim_params["bodies"]) + 1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                # The frame centre in the pixel corner coordinates a scene
                # states every position in -- the same point the ring system
                # defaults to and the same one the star record builder stands in
                # for an omitted star position, so a fresh scene is concentric.
                'center_v': self.sim_params['size_v'] / 2.0,
                'center_u': self.sim_params['size_u'] / 2.0,
                'range_km': self._find_unique_range(),
                'shape_model': 'ellipsoid',
                'axis1': 100.0,
                'axis2': 80.0,
                'axis3': 80.0,
                'mesh_lumpiness': 0.3,
                'mesh_seed': 0,
                'pose_euler_deg': [0.0, 0.0, 0.0],
                'rotation_z': 0.0,
                'rotation_tilt': 0.0,
                'illumination_angle': 0.0,
                'phase_angle': 0.0,
                'crater_fill': 0.0,
                'crater_min_radius': 0.05,
                'crater_max_radius': 0.25,
                'crater_power_law_exponent': 3.0,
                'crater_relief_scale': 0.6,
                'anti_aliasing': 0.5,
            }
        else:
            p = params
        idx = len(self.sim_params['bodies'])
        self.sim_params['bodies'].append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._validate_ranges()
        self._updater.request_update()

    def _add_ring_tab(self, params: dict[str, Any] | None = None) -> None:
        """Append a ring_system feature (creating the block) and select its tab."""
        ring_system = self.sim_params.setdefault(
            'ring_system',
            {
                'geometry': {
                    'center_v': self.sim_params['size_v'] / 2.0,
                    'center_u': self.sim_params['size_u'] / 2.0,
                    'opening_deg_obs': 90.0,
                    'opening_deg_sun': 90.0,
                    'node_deg': 0.0,
                },
                'features': [],
            },
        )
        features = ring_system.setdefault('features', [])
        if params is None:
            default_name = f'Ring{len(features) + 1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                'kind': 'ringlet',
                'tau': 1.0,
                'width': 20.0,
                'navigable': True,
                'orbit': {'a': 100.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0},
            }
        else:
            p = params
        idx = len(features)
        features.append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('ring', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._updater.request_update()

    def _add_star_tab(self, params: dict[str, Any] | None = None) -> None:
        """Append a star to the data model and select its new tab."""
        if params is None:
            default_name = f'Star{len(self.sim_params["stars"]) + 1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                # The frame centre, stated the way the star record builder
                # states the default it stands in for an omitted position.
                'v': self.sim_params['size_v'] / 2.0,
                'u': self.sim_params['size_u'] / 2.0,
                'vmag': 3.0,
                'spectral_class': 'G2',
                'psf_sigma': 1.0,
                'psf_size': [11, 11],
            }
        else:
            p = params
        idx = len(self.sim_params['stars'])
        self.sim_params['stars'].append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._updater.request_update()

    def _find_tab_by_properties(self, kind: str, data_index: int) -> int | None:
        """Find tab index by kind and data_index properties.

        Returns the tab index if found, None otherwise.
        """
        for tab_idx in range(self._tabs.count()):
            widget = self._tabs.widget(tab_idx)
            if widget is None:
                continue
            widget_kind = widget.property('kind')
            widget_data_index = widget.property('data_index')
            if (
                widget_kind == kind
                and widget_data_index is not None
                and widget_data_index == data_index
            ):
                return tab_idx
        return None

    def _delete_current_tab(self) -> None:
        """Delete the currently selected tab by looking up its data_index."""
        tab_idx = self._tabs.currentIndex()
        widget = self._tabs.widget(tab_idx)
        if widget is None:
            # No widget at this tab index, nothing to delete
            return
        data_index = widget.property('data_index')
        widget_kind = widget.property('kind')
        if data_index is None or widget_kind is None:
            # Widget doesn't have required properties (e.g., General or "+" tab)
            return
        self._delete_tab_by_index(widget_kind, data_index)

    def _delete_tab_by_index(self, kind: str, data_index: int) -> None:
        """Delete a tab by its kind ('body' or 'star') and data_index."""
        # Use the helper function to find the correct tab
        tab_idx = self._find_tab_by_properties(kind, data_index)
        if tab_idx is None:
            # Tab not found, nothing to delete
            return

        # Verify the widget matches what we expect
        widget = self._tabs.widget(tab_idx)
        if widget is None:
            return
        widget_kind = widget.property('kind')
        widget_data_index = widget.property('data_index')
        if widget_kind != kind or widget_data_index != data_index:
            # Safety check: widget doesn't match what we're looking for
            return

        # Delete from the correct list
        if kind == 'body':
            if 0 <= data_index < len(self.sim_params['bodies']):
                del self.sim_params['bodies'][data_index]
        elif kind == 'ring':
            features = self._ring_features()
            ring_system = self.sim_params.get('ring_system', {})
            # The system-level azimuthal / moonlets widgets live on the first
            # feature's tab, so deleting the last feature would strand those
            # authored blocks with no widget to reach them: refuse the
            # deletion and say why until the blocks are removed first.
            if len(features) == 1 and (ring_system.get('azimuthal') or ring_system.get('moonlets')):
                status_bar = self.statusBar()
                if status_bar is not None:
                    status_bar.showMessage(
                        'Cannot delete the last ring feature while system-level '
                        'azimuthal / moonlets blocks exist; disable those blocks first.',
                        8000,
                    )
                return
            if 0 <= data_index < len(features):
                del features[data_index]
            # Deleting the last feature retires the whole block: an empty
            # ring_system renders nothing and only clutters the saved scene.
            # (The guard above means no system-level blocks remain here.)
            if not features:
                self.sim_params.pop('ring_system', None)
        elif kind == 'star':
            if 0 <= data_index < len(self.sim_params['stars']):
                del self.sim_params['stars'][data_index]
        else:
            raise AssertionError(f'Unknown kind: {kind}')

        # Block signals before removing tab to prevent unwanted tab change events
        self._tabs.blockSignals(True)
        self._tabs.removeTab(tab_idx)
        self._tabs.blockSignals(False)

        # Rebuild tabs indices to align with lists
        self._rebuild_dynamic_tabs()
        self._ensure_tab_order()  # Ensure order is correct
        self._validate_ranges()
        self._updater.request_update()

    def _rebuild_dynamic_tabs(self) -> None:
        """Rebuild all object tabs from the data model in sorted order."""
        # Save General and "+" tab widgets
        general_widget = None
        plus_widget = None
        for i in range(self._tabs.count()):
            text = self._tabs.tabText(i)
            if text == 'General':
                general_widget = self._tabs.widget(i)
            elif text == '+':
                plus_widget = self._tabs.widget(i)

        # Remember current tab before rebuilding (if it's a valid tab)
        current_idx = self._tabs.currentIndex()
        target_tab_name = None
        if current_idx >= 0 and current_idx < self._tabs.count():
            current_text = self._tabs.tabText(current_idx)
            if current_text not in ('General', 'Optics', 'Artifacts', '+'):
                # Try to identify which body/star this was
                widget = self._tabs.widget(current_idx)
                if widget is not None:
                    widget_kind = widget.property('kind')
                    widget_data_index = widget.property('data_index')
                    if widget_kind == 'body' and widget_data_index is not None:
                        if 0 <= widget_data_index < len(self.sim_params['bodies']):
                            body_name = self.sim_params['bodies'][widget_data_index].get(
                                'name', f'Body{widget_data_index + 1}'
                            )
                            target_tab_name = body_name
                    elif widget_kind == 'ring' and widget_data_index is not None:
                        features = self._ring_features()
                        if 0 <= widget_data_index < len(features):
                            ring_name = features[widget_data_index].get(
                                'name', f'Ring{widget_data_index + 1}'
                            )
                            target_tab_name = ring_name
                    elif widget_kind == 'star' and widget_data_index is not None:
                        if 0 <= widget_data_index < len(self.sim_params['stars']):
                            star_name = self.sim_params['stars'][widget_data_index].get(
                                'name', f'Star{widget_data_index + 1}'
                            )
                            target_tab_name = star_name
                    else:
                        raise AssertionError(f'Unknown kind: {widget_kind}')

        # Block signals during rebuild to prevent tab change handler from firing
        self._tabs.blockSignals(True)

        # Remove all tabs
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)

        # Re-add in correct order: General first, then the fixed Optics and
        # Artifacts tabs, then bodies (sorted by range), then rings (sorted by
        # name), then stars (sorted by name), then "+"
        if general_widget is not None:
            self._tabs.addTab(general_widget, 'General')
        self._tabs.addTab(self._optics_tab, 'Optics')
        self._tabs.addTab(self._artifacts_tab, 'Artifacts')

        # Add body tabs (sorted by range)
        body_indices = list(range(len(self.sim_params['bodies'])))
        body_indices.sort(
            key=lambda i: (
                self.sim_params['bodies'][i].get('range_km', float('inf')),
                self.sim_params['bodies'][i].get('name', f'Body{i + 1}').lower(),
            )
        )
        for i in body_indices:
            tab = self._build_body_tab(i)
            tab_name = self.sim_params['bodies'][i].get('name', f'Body{i + 1}')
            self._tabs.addTab(tab, tab_name)

        # Add ring-feature tabs (in feature-list order: the first feature's
        # tab carries the shared system-level controls, so it stays first)
        ring_features = self._ring_features()
        for i in range(len(ring_features)):
            tab = self._build_ring_tab(i)
            tab_name = ring_features[i].get('name', f'Ring{i + 1}')
            self._tabs.addTab(tab, tab_name)

        # Add star tabs (sorted by name)
        star_indices = list(range(len(self.sim_params['stars'])))
        star_indices.sort(
            key=lambda i: self.sim_params['stars'][i].get('name', f'Star{i + 1}').lower()
        )
        for i in star_indices:
            tab = self._build_star_tab(i)
            tab_name = self.sim_params['stars'][i].get('name', f'Star{i + 1}')
            self._tabs.addTab(tab, tab_name)

        # Add "+" tab last
        if plus_widget is not None:
            self._tabs.addTab(plus_widget, '+')
        else:
            # Create if it doesn't exist
            self._add_tab_widget = QWidget()
            self._tabs.addTab(self._add_tab_widget, '+')

        # Restore the previously selected tab if it still exists
        if target_tab_name is not None:
            found = False
            for i in range(self._tabs.count()):
                if self._tabs.tabText(i) == target_tab_name:
                    self._tabs.setCurrentIndex(i)
                    self._last_valid_tab_index = i
                    found = True
                    break
            if not found:
                # Tab was deleted, default to General
                self._tabs.setCurrentIndex(0)
                self._last_valid_tab_index = 0
        else:
            # Default to General tab (index 0)
            self._tabs.setCurrentIndex(0)
            self._last_valid_tab_index = 0

        # Ensure we're on a valid tab (not "+") before unblocking signals
        current_idx = self._tabs.currentIndex()
        if (
            current_idx >= 0
            and current_idx < self._tabs.count()
            and self._tabs.tabText(current_idx) == '+'
        ):
            # Shouldn't happen, but be safe
            self._tabs.setCurrentIndex(0)
            self._last_valid_tab_index = 0

        # Unblock signals - this might emit currentChanged, but we're on General so it's safe
        self._tabs.blockSignals(False)

    def _update_tab_titles(self) -> None:
        """Rebuild tabs to maintain sorted order when names change."""
        # Rebuild tabs to maintain sorted order when names change
        # This ensures bodies and stars are always sorted by name
        self._rebuild_dynamic_tabs()

    def _find_unique_range(self) -> float:
        """Find a unique range value by incrementing from 1 until one doesn't exist."""
        existing_ranges = set()
        for body in self.sim_params.get('bodies', []):
            range_val = body.get('range_km')
            if range_val is not None:
                existing_ranges.add(float(range_val))
        ring_system = self.sim_params.get('ring_system')
        if isinstance(ring_system, dict) and ring_system.get('range_km') is not None:
            existing_ranges.add(float(ring_system['range_km']))

        # Start from 1 and increment until we find a unique range
        candidate = 1.0
        while candidate in existing_ranges:
            candidate += 1.0
        return candidate

    def _validate_ranges(self) -> None:
        """Check for duplicate body ranges and display a warning if found."""
        ranges = []
        for i in range(len(self.sim_params['bodies'])):
            range_val = self.sim_params['bodies'][i].get('range_km')
            if range_val is not None:
                ranges.append(float(range_val))
        duplicates = len(ranges) != len(set(ranges))
        self._warning_label.setText('Warning: duplicate body ranges' if duplicates else '')
