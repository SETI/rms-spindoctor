"""Tests for ``spindoctor.nav_model.limb_placement``.

The geometry query is replaced by the analytic sphere stand-in from
``tests.shims``, so the tests pin the placement arithmetic and the coordinate
conventions without SPICE.  The sensor is not square and its margins differ by
axis, so a swapped axis moves a vertex off the sphere.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from tests.shims import FakeObs, sphere_limb_altitudes

import spindoctor.nav_model.limb_placement as limb_placement_module
from spindoctor.nav_model.limb_placement import place_on_visible_limb

_CENTER_VU = (40.0, 75.0)
_RADIUS_PX = 25.0
_KM_PER_PX = 4.0
_MARGIN_VU = (6, 13)


def _obs() -> FakeObs:
    """Return a 100 x 140 sensor with a (6, 13) extended-frame margin."""
    return FakeObs(data=np.zeros((100, 140)), extfov_margin_vu=_MARGIN_VU)


def _radial_vertices(angles_deg: list[float], radius_px: float) -> tuple[Any, Any]:
    """Return extended-frame vertices at ``radius_px`` from the sphere and radial normals.

    Parameters:
        angles_deg: Position angles of the vertices around the sphere.
        radius_px: Their distance from the sphere center.
    """
    ang = np.radians(np.asarray(angles_deg, dtype=np.float64))
    normals = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    # Sensor pixel-corner center -> extended-frame pixel-centric center.
    center = np.array(_CENTER_VU) + np.array(_MARGIN_VU) - 0.5
    return center + radius_px * normals, normals


@pytest.fixture
def sphere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer limb-altitude queries from the analytic sphere."""
    monkeypatch.setattr(
        limb_placement_module,
        'limb_altitudes_km',
        sphere_limb_altitudes(_CENTER_VU, _RADIUS_PX, _KM_PER_PX),
    )


@pytest.mark.usefixtures('sphere')
def test_vertex_moves_onto_the_silhouette() -> None:
    """A vertex 0.4 px inside the edge moves 0.4 px out along a radial normal."""
    vertices, normals = _radial_vertices([0.0, 90.0, 200.0], _RADIUS_PX - 0.4)
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', vertices, normals, altitude_km=0.0)
    assert placement.offset_px == pytest.approx([0.4, 0.4, 0.4])


@pytest.mark.usefixtures('sphere')
def test_vertex_moves_to_the_visible_limb_altitude() -> None:
    """An altitude of 8 km at 4 km/px adds two pixels to the move."""
    vertices, normals = _radial_vertices([30.0], _RADIUS_PX - 0.25)
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', vertices, normals, altitude_km=8.0)
    assert placement.offset_px == pytest.approx([2.25])


@pytest.mark.usefixtures('sphere')
def test_sensitivity_is_pixels_per_kilometer() -> None:
    """The vertex moves 1/(km per px) pixels per kilometer of altitude."""
    vertices, normals = _radial_vertices([120.0], _RADIUS_PX - 0.5)
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', vertices, normals, altitude_km=0.0)
    assert placement.px_per_km == pytest.approx([1.0 / _KM_PER_PX])


@pytest.mark.usefixtures('sphere')
def test_inward_normal_gives_no_position() -> None:
    """A normal pointing into the body gives the vertex no visible-limb position."""
    vertices, normals = _radial_vertices([0.0, 45.0], _RADIUS_PX - 0.5)
    normals[1] = -normals[1]
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', vertices, normals, altitude_km=0.0)
    assert placement.valid.tolist() == [True, False]


@pytest.mark.usefixtures('sphere')
def test_no_vertices_give_empty_placement() -> None:
    """An empty polyline gives empty arrays."""
    empty = np.empty((0, 2))
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', empty, empty, altitude_km=0.0)
    assert placement.offset_px.shape == (0,)


def test_masked_altitude_gives_no_position(monkeypatch: pytest.MonkeyPatch) -> None:
    """Where the geometry layer masks the altitude the vertex gets no position."""
    sphere = sphere_limb_altitudes(_CENTER_VU, _RADIUS_PX, _KM_PER_PX)

    def masked_second(obs: Any, body_name: str, uv: Any) -> Any:
        """Return the sphere's altitudes with the second vertex's masked."""
        values = sphere(obs, body_name, uv)
        values[1] = np.nan
        return values

    monkeypatch.setattr(limb_placement_module, 'limb_altitudes_km', masked_second)
    vertices, normals = _radial_vertices([0.0, 50.0, 100.0], _RADIUS_PX - 0.5)
    placement = place_on_visible_limb(cast(Any, _obs()), 'B', vertices, normals, altitude_km=0.0)
    assert placement.valid.tolist() == [True, False, True]
