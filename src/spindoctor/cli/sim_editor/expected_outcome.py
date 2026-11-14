"""Expected-outcome panel for the General tab.

The scene-level ``expected`` block declares what the navigator should produce
for the scene: a ``status`` (``success`` / ``failed`` / ``conflicted``), an
optional ``status_reason`` token, and a ``confidence_tier``.  It is test-only
metadata -- the integration suite's assertion machinery reads it, and neither
the image-side renderer nor the navigator sees it.  The panel is a checkable
group: enabling it writes the block (with the required ``status``), unchecking
it removes the block entirely.  The confidence-tier ``(none)`` choice writes a
null tier (assert the status only).

A ``failed`` or ``conflicted`` status must pin the matching tier (the validator
enforces this at save time); the panel leaves the two combos independent, so an
inconsistent pair is the operator's to reconcile before saving.
"""

from typing import Any

from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLineEdit

from spindoctor.cli.sim_editor.base import SimEditorBase

# The tier combo's visible choices; ``(none)`` maps to a null tier.
_TIER_CHOICES: list[str] = ['high', 'medium', 'low', 'failed', 'conflicted', '(none)']


class ExpectedOutcomeMixin(SimEditorBase):
    """Builds and handles the test-only ``expected`` outcome block."""

    def _build_expected_panel(self, gen_layout: QFormLayout) -> None:
        """Add the Expected-outcome group to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        expected = self.sim_params.get('expected')
        has_expected = isinstance(expected, dict)
        block: dict[str, Any] = expected if isinstance(expected, dict) else {}
        # Keys the panel has no widgets for (the hand-authored known-error
        # check keys) survive an uncheck/recheck cycle through this stash.
        self._removed_expected_extras: dict[str, object] = {}

        self._expected_group = QGroupBox('Expected outcome (test-only)')
        self._expected_group.setCheckable(True)
        self._expected_group.setChecked(has_expected)
        self._expected_group.setToolTip(
            'The navigation outcome the integration suite asserts; neither the '
            'renderer nor the navigator reads it. Unchecked leaves the key absent.'
        )
        form = QFormLayout(self._expected_group)

        self._expected_status_combo = QComboBox()
        self._expected_status_combo.addItems(['success', 'failed', 'conflicted'])
        status_index = self._expected_status_combo.findText(str(block.get('status', 'success')))
        if status_index >= 0:
            self._expected_status_combo.setCurrentIndex(status_index)
        self._expected_status_combo.currentTextChanged.connect(self._on_expected_status)
        form.addRow('Status:', self._expected_status_combo)

        self._expected_tier_combo = QComboBox()
        self._expected_tier_combo.addItems(_TIER_CHOICES)
        tier = block.get('confidence_tier')
        tier_text = '(none)' if tier is None else str(tier)
        tier_index = self._expected_tier_combo.findText(tier_text)
        if tier_index >= 0:
            self._expected_tier_combo.setCurrentIndex(tier_index)
        self._expected_tier_combo.currentTextChanged.connect(self._on_expected_tier)
        form.addRow('Confidence tier:', self._expected_tier_combo)

        self._expected_reason_edit = QLineEdit(str(block.get('status_reason') or ''))
        self._expected_reason_edit.setToolTip(
            'Optional status-reason token (a NavStatusReason value); empty leaves the key absent.'
        )
        self._expected_reason_edit.textChanged.connect(self._on_expected_reason)
        form.addRow('Status reason:', self._expected_reason_edit)

        self._expected_group.toggled.connect(self._on_expected_toggled)
        gen_layout.addRow(self._expected_group)

    def _tier_from_text(self, text: str) -> str | None:
        """Map the tier combo text to its stored value (``(none)`` -> None)."""
        return None if text == '(none)' else text

    def _on_expected_toggled(self, checked: bool) -> None:
        """Insert or remove the expected block from the scene.

        Keys the panel has no widgets for (the hand-authored known-error
        check, ``known_offset_error_px`` / ``known_offset_error_tol_px``)
        are preserved across an uncheck/recheck cycle rather than being
        rebuilt away: the removed block is remembered and its extra keys
        are re-attached when the group is re-enabled.
        """
        if checked:
            block: dict[str, object] = dict(self._removed_expected_extras)
            block['status'] = self._expected_status_combo.currentText()
            block['confidence_tier'] = self._tier_from_text(self._expected_tier_combo.currentText())
            reason = self._expected_reason_edit.text().strip()
            if reason:
                block['status_reason'] = reason
            else:
                block.pop('status_reason', None)
            self.sim_params['expected'] = block
        else:
            removed = self.sim_params.pop('expected', None)
            if isinstance(removed, dict):
                self._removed_expected_extras = {
                    key: value
                    for key, value in removed.items()
                    if key not in ('status', 'confidence_tier', 'status_reason')
                }
        self._updater.request_update()

    def _on_expected_status(self, text: str) -> None:
        """Update the expected status when the block is present."""
        expected = self.sim_params.get('expected')
        if isinstance(expected, dict):
            expected['status'] = text
            self._updater.request_update()

    def _on_expected_tier(self, text: str) -> None:
        """Update the expected confidence tier when the block is present."""
        expected = self.sim_params.get('expected')
        if isinstance(expected, dict):
            expected['confidence_tier'] = self._tier_from_text(text)
            self._updater.request_update()

    def _on_expected_reason(self, text: str) -> None:
        """Set or clear the expected status-reason token when present."""
        expected = self.sim_params.get('expected')
        if isinstance(expected, dict):
            reason = text.strip()
            if reason:
                expected['status_reason'] = reason
            else:
                expected.pop('status_reason', None)
            self._updater.request_update()
