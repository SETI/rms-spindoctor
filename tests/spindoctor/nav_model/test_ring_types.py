"""Tests for ``RingBaseOrbitMode`` and ``RingPerturbationMode``."""

from dataclasses import FrozenInstanceError

import pytest

from spindoctor.nav_model.rings.ring_types import RingBaseOrbitMode, RingPerturbationMode


def test_base_orbit_mode_is_frozen() -> None:
    """``RingBaseOrbitMode`` rejects field assignment after construction."""
    m = RingBaseOrbitMode(a=1.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)
    with pytest.raises(AttributeError) as exc_info:
        m.a = 2.0  # type: ignore[misc]
    assert isinstance(exc_info.value, FrozenInstanceError)
    msg = str(exc_info.value)
    assert 'cannot assign' in msg
    assert "field 'a'" in msg


def test_ring_perturbation_mode_is_frozen() -> None:
    """``RingPerturbationMode`` rejects field assignment after construction."""
    p = RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    with pytest.raises(AttributeError) as exc_info:
        p.mode_num = 3  # type: ignore[misc]
    assert isinstance(exc_info.value, FrozenInstanceError)
    msg = str(exc_info.value)
    assert 'cannot assign' in msg
    assert "field 'mode_num'" in msg
