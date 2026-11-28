"""Tests of what the realism match reads for a Cassini ISS frame beyond its label.

The raw product is a stand-in, so nothing reads holdings.  CI does not run the file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import vicar
from util.sim_realism.sim_realism_frames_cassini import (
    cassini_clipped_pixels,
)

_CALIB = '/holdings/calibrated/COISS_2xxx/COISS_2009/data/1486900812/N1486991973_1_CALIB.IMG'
_RAW = '/holdings/volumes/COISS_2xxx/COISS_2009/data/1486900812/N1486991973_1.IMG'


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The paths of the raw products opened, which read DN 254, 255, 4094, and 4095."""
    paths: list[str] = []

    def vicar_image(path: Any) -> Any:
        paths.append(str(path))
        return SimpleNamespace(data_2d=np.array([[254, 255], [4094, 4095]]))

    monkeypatch.setattr(vicar, 'VicarImage', vicar_image)
    return paths


def test_the_raw_product_lies_beside_the_calibrated_one(opened: list[str]) -> None:
    """The raw product is under volumes/, with _CALIB dropped from its name."""
    cassini_clipped_pixels(_CALIB, 'TABLE')
    assert opened == [_RAW]


@pytest.mark.parametrize(
    ('conversion', 'clipped'),
    [('TABLE', [[False, True], [True, True]]), ('12BIT', [[False, False], [False, True]])],
    ids=['table', '12-bit'],
)
def test_the_full_scale_follows_the_conversion_type(
    opened: list[str], conversion: str, clipped: list[list[bool]]
) -> None:
    """A table frame clips at DN 255 and a 12-bit frame at DN 4095."""
    mask = cassini_clipped_pixels(_CALIB, conversion)
    assert mask is not None
    np.testing.assert_array_equal(mask, np.array(clipped))
