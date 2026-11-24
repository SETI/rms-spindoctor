"""Test that the manual navigation dialog's automatic match reads the upsample factor.

Qt runs headless (offscreen platform); the match is driven on a stand-in for the dialog, so
no window is built.
"""

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from tests.spindoctor.conftest import OVERRIDDEN_UPSAMPLE_FACTOR

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import PyQt6.QtWidgets  # noqa: F401
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6 not available: {exc}', allow_module_level=True)

import spindoctor.ui.manual_nav_dialog as dialog_module
from spindoctor.config import Config


def test_auto_match_reads_the_configured_upsample_factor(
    monkeypatch: pytest.MonkeyPatch,
    upsample_override_config: Config,
) -> None:
    """The automatic match correlates at the upsample factor the configuration names."""
    seen: list[int] = []

    def _record(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs['upsample_factor'])
        return {'offset': (1.0, 2.0)}

    monkeypatch.setattr(dialog_module, 'navigate_with_pyramid_kpeaks', _record)
    stand_in = SimpleNamespace(
        _config=upsample_override_config,
        _img_ext=np.zeros((8, 8)),
        _model_img_ext=np.zeros((8, 8)),
        _model_mask_ext=np.zeros((8, 8), dtype=bool),
        _obs=MagicMock(),
        _spin_dv=MagicMock(),
        _spin_du=MagicMock(),
        _refresh_overlay=MagicMock(),
    )
    dialog_module.ManualNavDialog._on_auto(stand_in)  # type: ignore[arg-type]
    assert seen == [OVERRIDDEN_UPSAMPLE_FACTOR]
