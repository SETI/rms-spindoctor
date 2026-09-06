"""Tests for ``spindoctor.nav_model.stars.smeared_psf``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.nav_model.stars.smeared_psf import (
    compute_smear_vector_px,
    movement_granularity_px,
    render_smeared_psf,
    smear_length_px,
)


def test_smear_length_px_pythagorean() -> None:
    """``smear_length_px`` returns ``hypot(move_v, move_u)``."""
    assert smear_length_px(3.0, 4.0) == 5.0


def test_smear_length_px_zero_amplitude() -> None:
    """A motionless exposure has zero smear length."""
    assert smear_length_px(0.0, 0.0) == 0.0


def test_movement_granularity_clamps_to_floor() -> None:
    """Sub-pixel smears use the 0.1 lower bound."""
    assert movement_granularity_px(0.05, 0.0, max_steps=50) == pytest.approx(0.1)


def test_movement_granularity_clamps_to_ceiling() -> None:
    """Smears longer than ``max_steps`` pixels saturate at 1.0 px per sample."""
    assert movement_granularity_px(200.0, 0.0, max_steps=50) == pytest.approx(1.0)


def test_movement_granularity_picks_max_axis() -> None:
    """The granularity is set by the larger of the two axis amplitudes."""
    assert movement_granularity_px(0.0, 5.0, max_steps=50) == pytest.approx(5.0 / 50.0)


def test_movement_granularity_rejects_zero_max_steps() -> None:
    """Zero or negative ``max_steps`` raises ``ValueError`` naming the value."""
    with pytest.raises(ValueError, match='max_steps must be > 0'):
        movement_granularity_px(1.0, 1.0, max_steps=0)


@dataclass
class _FakeStar:
    """Minimal star stand-in used by the smeared-PSF renderer."""

    psf_size: tuple[int, int]
    move_v: float
    move_u: float
    u: float
    v: float
    dn: float


class _FakePSF:
    """PSF stand-in returning a deterministic stamp via ``eval_rect``.

    The stamp shape is determined by ``rect_size``.  The contents encode
    ``offset`` so tests can verify the call signature.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def eval_rect(
        self,
        rect_size: tuple[int, int],
        *,
        offset: tuple[float, float],
        scale: float,
        movement: tuple[float, float],
        movement_granularity: float,
    ) -> np.ndarray:
        """Record arguments and return a constant stamp filled with ``scale``."""
        self.calls.append(
            {
                'rect_size': rect_size,
                'offset': offset,
                'scale': scale,
                'movement': movement,
                'movement_granularity': movement_granularity,
            }
        )
        out = np.full(rect_size, fill_value=scale, dtype=np.float64)
        return out


def test_render_smeared_psf_passes_movement_through_to_psf() -> None:
    """``render_smeared_psf`` forwards smear vector and granularity to ``eval_rect``."""
    psf = _FakePSF()
    star = _FakeStar(psf_size=(5, 5), move_v=1.5, move_u=2.5, u=10.4, v=20.7, dn=100.0)
    stamp = render_smeared_psf(psf, star=star, max_movement_steps=50)  # type: ignore[arg-type]
    assert len(psf.calls) == 1
    call = psf.calls[0]
    expected_half_u = (5 + round(2.5)) // 2
    expected_half_v = (5 + round(1.5)) // 2
    assert call['rect_size'] == (expected_half_v * 2 + 1, expected_half_u * 2 + 1)
    assert call['movement'] == (1.5, 2.5)
    assert call['scale'] == 100.0
    expected_offset = (round(20.7 - int(20.7), 6), round(10.4 - int(10.4), 6))
    actual_offset = (round(call['offset'][0], 6), round(call['offset'][1], 6))  # type: ignore[index]
    assert actual_offset == expected_offset
    # 2.5/50 = 0.05 falls below the 0.1 floor so the clamp produces 0.1.
    granularity = cast(float, call['movement_granularity'])
    assert math.isclose(granularity, 0.1, rel_tol=1e-12)
    assert stamp.dtype == np.float64


class _FakeUVResult:
    """Stand-in for ``oops``'s ``UV`` result exposing ``to_scalars()``."""

    def __init__(self, u: float, v: float) -> None:
        self._u = _Scalar(u)
        self._v = _Scalar(v)

    def to_scalars(self) -> tuple[_Scalar, _Scalar]:
        """Return ``(u_scalar, v_scalar)`` mirroring oops's UV API."""
        return self._u, self._v


@dataclass
class _Scalar:
    """Stand-in for ``polymath.Scalar`` exposing ``vals``."""

    vals: float


class _FakeObsForBracket:
    """Observation stand-in for the smear-bracket calculation.

    ``uv_from_ra_and_dec`` returns one ``UV`` value at ``tfrac=0`` and
    another at ``tfrac=1``, simulating the spacecraft pointing
    displacement during the exposure.
    """

    def __init__(self, *, du: float, dv: float) -> None:
        self._du = du
        self._dv = dv

    def center_ra_dec(self, *, apparent: bool = True) -> tuple[float, float]:
        """Return a constant sky direction.

        The projection below is planted, so which direction this is does not
        reach the answer; that it can be asked for does.

        Parameters:
            apparent: Whether to correct for aberration; ignored here.

        Returns:
            ``(ra, dec)`` in radians, always the origin of the sky frame.
        """
        del apparent
        return 0.0, 0.0

    def uv_from_ra_and_dec(
        self,
        ra: float,
        dec: float,
        *,
        tfrac: float,
        apparent: bool,
    ) -> _FakeUVResult:
        """Return a UV that depends linearly on ``tfrac``."""
        del ra, dec, apparent
        return _FakeUVResult(u=10.0 + self._du * tfrac, v=20.0 + self._dv * tfrac)


def test_compute_smear_vector_px_returns_bracket_difference() -> None:
    """The smear vector is ``(v_end - v_start, u_end - u_start)`` in pixels."""
    obs: Any = _FakeObsForBracket(du=2.5, dv=1.5)
    my, mx = compute_smear_vector_px(obs)
    assert my == pytest.approx(1.5)
    assert mx == pytest.approx(2.5)


def test_compute_smear_vector_px_zero_when_no_pointing_drift() -> None:
    """A static-pointing exposure produces a zero smear vector."""
    obs: Any = _FakeObsForBracket(du=0.0, dv=0.0)
    my, mx = compute_smear_vector_px(obs)
    assert my == 0.0
    assert mx == 0.0
