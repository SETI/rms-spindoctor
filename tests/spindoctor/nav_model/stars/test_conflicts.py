"""Tests for ``spindoctor.nav_model.stars.conflicts``."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
import pytest

import spindoctor.nav_model.stars.conflicts as conflicts_module
from spindoctor.nav_model.stars.conflicts import (
    _conflict_body_list,
    _ring_opaque_fraction,
    parse_ring_occlusion_annuli,
)


def test_parse_ring_occlusion_annuli_normalises_keys() -> None:
    """Planet keys are normalised to upper case in the returned mapping."""
    result = parse_ring_occlusion_annuli({'saturn': [[100.0, 200.0]]})
    assert list(result.keys()) == ['SATURN']
    assert result['SATURN'] == [(100.0, 200.0)]


def test_parse_ring_occlusion_annuli_returns_empty_for_none() -> None:
    """A ``None`` input is treated as the empty mapping."""
    assert parse_ring_occlusion_annuli(None) == {}


def test_parse_ring_occlusion_annuli_returns_empty_for_empty() -> None:
    """An empty mapping returns an empty mapping."""
    assert parse_ring_occlusion_annuli({}) == {}


def test_parse_ring_occlusion_annuli_rejects_non_sequence_pair() -> None:
    """A scalar where a length-2 sequence was expected raises ``ValueError``."""
    bad: Any = {'SATURN': [123.0]}
    with pytest.raises(ValueError, match='must be a length-2 sequence'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_wrong_length() -> None:
    """A length-3 sequence raises ``ValueError`` with the actual length."""
    with pytest.raises(ValueError, match='must have exactly 2 elements'):
        parse_ring_occlusion_annuli({'SATURN': [[1.0, 2.0, 3.0]]})


def test_parse_ring_occlusion_annuli_rejects_non_numeric() -> None:
    """A non-numeric inner radius raises ``ValueError`` naming it."""
    bad: Any = {'SATURN': [['inner', 200.0]]}
    with pytest.raises(ValueError, match='must be numeric'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_inner_ge_outer() -> None:
    """Degenerate annuli (inner >= outer) raise ``ValueError``."""
    with pytest.raises(ValueError, match=r'inner 100\.0 km >= outer 100\.0 km'):
        parse_ring_occlusion_annuli({'SATURN': [[100.0, 100.0]]})


def test_parse_ring_occlusion_annuli_rejects_bool() -> None:
    """A bool is rejected even though Python admits it as int."""
    bad: Any = {'SATURN': [[True, 200.0]]}
    with pytest.raises(ValueError, match='got bool'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_inf() -> None:
    """Non-finite radii raise ``ValueError`` naming the bad value."""
    with pytest.raises(ValueError, match='must be finite'):
        parse_ring_occlusion_annuli({'SATURN': [[float('inf'), 200.0]]})


class _FakeObsWithSaturn:
    """Stand-in for an observation whose closest planet is Saturn."""

    closest_planet = 'SATURN'


class _FakeObsNoPlanet:
    """Stand-in for an observation with no closest planet."""

    closest_planet: str | None = None


class _FakeSaturnConfig:
    """Stand-in for the project config exposing only ``satellites``."""

    @staticmethod
    def satellites(planet: str) -> list[str]:
        """Return Saturn's satellites in canonical order."""
        return ['MIMAS', 'TETHYS'] if planet == 'SATURN' else []


class _FakeEmptyConfig:
    """Stand-in for a config that knows about no satellites."""

    @staticmethod
    def satellites(planet: str) -> list[str]:
        """Return an empty list for any planet."""
        del planet
        return []


def test_conflict_body_list_includes_planet_and_satellites() -> None:
    """``_conflict_body_list`` returns the planet plus its satellites."""
    out = _conflict_body_list(cast(Any, _FakeObsWithSaturn()), cast(Any, _FakeSaturnConfig()))
    assert out == ['SATURN', 'MIMAS', 'TETHYS']


def test_conflict_body_list_returns_empty_when_no_closest_planet() -> None:
    """When ``obs.closest_planet`` is ``None`` the body list is empty."""
    out = _conflict_body_list(cast(Any, _FakeObsNoPlanet()), cast(Any, _FakeEmptyConfig()))
    assert out == []


# --- Per-pixel ring occlusion (issue #145) ---


def test_ring_opaque_fraction_counts_per_pixel_membership() -> None:
    """Half the window inside the annulus gives fraction 0.5."""
    radii = np.array([[100.0, 100.0], [200.0, 200.0]])
    valid = np.ones_like(radii, dtype=bool)
    fraction = _ring_opaque_fraction(radii, valid, [(150.0, 250.0)])
    assert fraction == pytest.approx(0.5)


def test_ring_opaque_fraction_ignores_invalid_pixels() -> None:
    """Masked pixels are excluded from both numerator and denominator."""
    radii = np.array([[100.0, 200.0], [200.0, 200.0]])
    valid = np.array([[True, True], [False, False]])
    fraction = _ring_opaque_fraction(radii, valid, [(150.0, 250.0)])
    assert fraction == pytest.approx(0.5)


def test_ring_opaque_fraction_zero_when_all_invalid() -> None:
    """An all-masked window yields fraction 0.0 (no occlusion verdict)."""
    radii = np.array([[100.0, 200.0]])
    valid = np.zeros_like(radii, dtype=bool)
    assert _ring_opaque_fraction(radii, valid, [(150.0, 250.0)]) == 0.0


def test_ring_opaque_fraction_unions_annuli() -> None:
    """A pixel inside any configured annulus counts as opaque."""
    radii = np.array([[100.0, 300.0, 500.0]])
    valid = np.ones_like(radii, dtype=bool)
    fraction = _ring_opaque_fraction(radii, valid, [(90.0, 110.0), (290.0, 310.0)])
    assert fraction == pytest.approx(2.0 / 3.0)


class _FakeScalar:
    """Stand-in for the oops ring-radius backplane scalar."""

    def __init__(self, vals: np.ndarray, mask: np.ndarray | bool = False) -> None:
        self.vals = vals
        self.mask = mask

    def is_all_masked(self) -> bool:
        """True when every window pixel is masked."""
        return bool(np.all(self.mask))


class _FakeBackplane:
    """Stand-in Backplane serving canned body and ring answers."""

    ring_radii: np.ndarray = np.zeros((1, 1))

    def __init__(self, obs: Any, meshgrid: Any) -> None:
        del obs, meshgrid

    def where_intercepted(self, body_name: str) -> np.ndarray:
        """No body intercepts anywhere in the window."""
        del body_name
        return np.zeros((2, 2), dtype=bool)

    def ring_radius(self, ring_target: str) -> _FakeScalar:
        """Serve the canned ring-radius window."""
        del ring_target
        return _FakeScalar(self.ring_radii)


class _FakeMeshgrid:
    """Stand-in for ``oops.Meshgrid``, keeping the window it was asked for.

    The fake backplane serves canned radii and does not need a grid, but the
    origin and limit are the one place the conflict check states where it is
    looking, so they are recorded for a test to read back.
    """

    windows: ClassVar[list[tuple[tuple[float, float], tuple[float, float]]]] = []

    @classmethod
    def for_fov(cls, fov: Any, *, origin: Any, limit: Any) -> None:
        """Record the window and accept the call.

        Parameters:
            fov: Field of view (unused).
            origin: ``(u, v)`` low corner of the conflict window.
            limit: ``(u, v)`` high corner of the conflict window.
        """
        del fov
        cls.windows.append((tuple(origin), tuple(limit)))
        return None


class _FakeStar:
    """Minimal star record for the conflict check."""

    def __init__(self) -> None:
        self.u = 10.0
        self.v = 10.0
        self.conflicts = ''


def _run_check_one_star(
    monkeypatch: pytest.MonkeyPatch,
    ring_radii: np.ndarray,
    *,
    min_opaque_fraction: float,
) -> _FakeStar:
    """Run ``_check_one_star`` against a canned ring-radius window."""
    monkeypatch.setattr(conflicts_module, 'Meshgrid', _FakeMeshgrid)
    monkeypatch.setattr(conflicts_module, 'Backplane', _FakeBackplane)
    monkeypatch.setattr(_FakeBackplane, 'ring_radii', ring_radii)
    star = _FakeStar()
    obs = _FakeObsWithSaturn()
    obs_any = cast(Any, obs)
    obs_any.fov = object()
    conflicts_module._check_one_star(
        obs=obs_any,
        star=cast(Any, star),
        body_list=['SATURN'],
        ring_annuli={'SATURN': [(74490.0, 91980.0)]},
        rings_can_conflict=True,
        body_conflict_margin=2.0,
        ring_min_opaque_fraction=min_opaque_fraction,
    )
    return star


def test_the_conflict_window_is_centred_on_the_star_record_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window brackets the uv the star record carries, unconverted.

    ``Meshgrid.for_fov`` takes the pixel corner coordinates a star record is
    written in, so the conflict window is the record's own numbers plus and
    minus the margin with no half pixel in between.  The anchor is the record
    the test wrote, not anything the checker computed, so a conversion applied
    here would move the window off the star it is meant to cover.

    Parameters:
        monkeypatch: Patches the module's oops Meshgrid / Backplane names.
    """
    _FakeMeshgrid.windows.clear()
    star = _run_check_one_star(
        monkeypatch,
        np.array([[60000.0, 70000.0], [95000.0, 100000.0]]),
        min_opaque_fraction=0.25,
    )
    assert len(_FakeMeshgrid.windows) == 1
    origin, limit = _FakeMeshgrid.windows[0]
    assert origin == (star.u - 2.0, star.v - 2.0)
    assert limit == (star.u + 2.0, star.v + 2.0)


def test_check_one_star_flags_star_straddling_annulus_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window straddling the annulus edge is judged per pixel, not by median.

    Two thirds of the window sit just outside the C-ring inner edge, so the
    median radius lands in free space and a collapsed-median test would clear
    the star -- but a third of its window is over opaque ring material.
    """
    ring_radii = np.array([[74000.0, 74000.0, 74000.0], [74500.0, 74500.0, 74000.0]])
    star = _run_check_one_star(monkeypatch, ring_radii, min_opaque_fraction=0.25)
    assert star.conflicts == 'RING: SATURN'


def test_check_one_star_clears_star_when_median_falls_in_thin_annulus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible star is kept when only a sliver of the window is opaque.

    The median radius of this window falls inside the annulus, so the
    collapsed-median test would reject the star even though only one of six
    pixels is actually over ring material.
    """
    ring_radii = np.array([[60000.0, 70000.0, 74500.0], [95000.0, 100000.0, 105000.0]])
    star = _run_check_one_star(monkeypatch, ring_radii, min_opaque_fraction=0.25)
    assert star.conflicts == ''
