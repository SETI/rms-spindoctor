"""Percentile-table parameters in the ``sd_create_simulated_image`` editor.

A percentile table is a list of fractions.  Its row shows the table the renderer
resolves while the key is unset, and writes fractions once its box is checked.

Qt runs headless (offscreen platform).
"""

import os
from typing import Any

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

from spindoctor.sim.artifact_modes import ARTIFACT_MODES
from spindoctor.ui.sim_editor import CreateSimulatedImageModel

_MODE = 'compression_truncated_lines'
_KEY = 'lost_share_percentiles'


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image editor model."""
    return CreateSimulatedImageModel()


def test_an_unset_table_shows_the_resolved_table(model: Any) -> None:
    """An unset percentile table shows the registry's table in its spins."""
    row = model._mode_rows[_MODE]
    row.group.setChecked(True)
    shown = [spin.value() for spin in row._list_spins[_KEY]]
    assert shown == list(ARTIFACT_MODES[_MODE].param_map[_KEY].default)


def test_a_checked_table_writes_fractions(model: Any) -> None:
    """Checking the table's box writes its spins' fractions to the scene."""
    row = model._mode_rows[_MODE]
    row.group.setChecked(True)
    row._list_spins[_KEY][6].setValue(0.75)
    row._list_checks[_KEY].setChecked(True)
    assert model.sim_params['artifacts'][_MODE][_KEY] == [
        0.0,
        0.05,
        0.125,
        0.25,
        0.375,
        0.45,
        0.75,
    ]
