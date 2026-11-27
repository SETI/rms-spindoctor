"""Percentile-table and mean-count parameters in the ``sd_create_simulated_image`` editor.

A percentile table is a list of numbers: shares in [0, 1], or positive sizes.  Its row
shows the table the renderer resolves while the key is unset, and writes numbers once
its box is checked.  A mean count that includes one certain event cannot go below 1.

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

_MODE = 'downlink_gaps'


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image editor model."""
    return CreateSimulatedImageModel()


def test_an_unset_table_shows_the_resolved_table(model: Any) -> None:
    """An unset share table shows the registry's table in its spins."""
    row = model._mode_rows[_MODE]
    row.group.setChecked(True)
    shown = [spin.value() for spin in row._list_spins['head_kept_percentiles']]
    assert shown == list(ARTIFACT_MODES[_MODE].param_map['head_kept_percentiles'].default)


def test_a_checked_share_table_writes_fractions(model: Any) -> None:
    """Checking a share table's box writes its spins' fractions to the scene."""
    row = model._mode_rows[_MODE]
    row.group.setChecked(True)
    row._list_spins['head_kept_percentiles'][9].setValue(0.75)
    row._list_checks['head_kept_percentiles'].setChecked(True)
    written = model.sim_params['artifacts'][_MODE]['head_kept_percentiles']
    assert written == [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.75]


def test_a_checked_size_table_writes_numbers_above_one(model: Any) -> None:
    """A size table's spins reach past 1, and checking its box writes them."""
    row = model._mode_rows[_MODE]
    row.group.setChecked(True)
    row._list_spins['size_percentiles'][9].setValue(373.0)
    row._list_checks['size_percentiles'].setChecked(True)
    written = model.sim_params['artifacts'][_MODE]['size_percentiles']
    assert written == [8.0] * 9 + [373.0]


def test_the_mean_group_count_stops_at_one(model: Any) -> None:
    """The mean group count's spin cannot go below one group."""
    spin = model._mode_rows[_MODE]._scalar_widgets['mean_groups']
    spin.setValue(0.0)
    assert spin.value() == 1.0


def test_a_fraction_list_spin_steps_by_a_hundredth(model: Any) -> None:
    """One step up from an intermediate share lands on the next hundredth."""
    spin = model._mode_rows[_MODE]._list_spins['head_kept_percentiles'][9]
    spin.setValue(0.5)
    spin.stepUp()
    assert spin.value() == pytest.approx(0.51)


def test_a_fraction_scalar_spin_steps_by_a_hundredth(model: Any) -> None:
    """One step up from an intermediate scalar fraction lands on the next hundredth."""
    spin = model._mode_rows[_MODE]._scalar_widgets['probability']
    spin.setValue(0.5)
    spin.stepUp()
    assert spin.value() == pytest.approx(0.51)
