"""Fixtures shared by the simulator tests.

The base scenes themselves are plain functions in ``scenes.py`` beside this file, so a
module can also build parametrize lists and constants from them at import time.
Each fixture here hands its test a fresh scene the test may change freely.
"""

from typing import Any

import pytest
from tests.spindoctor.sim.scenes import body_scene, frame_scene


@pytest.fixture
def empty_frame() -> dict[str, Any]:
    """A 64 x 64 coiss_nac frame with random seed 1 and nothing in view."""
    return frame_scene()


@pytest.fixture
def lit_body_frame() -> dict[str, Any]:
    """The empty frame holding one fully lit spherical body of radius 10 at its center."""
    return body_scene()
