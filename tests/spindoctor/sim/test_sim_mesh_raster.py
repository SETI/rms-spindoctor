"""Rasterization rules of the shared mesh renderer.

Each mesh here is built directly in pixel units (unit semi-axes, zero pose, no
supersampling, light from the observer), so a vertex sits exactly where its
coordinates say and a flat face's shade is the depth component of its unit normal.
"""

import math

import numpy as np
import pytest

from spindoctor.sim import mesh_geometry
from spindoctor.sim.mesh_geometry import Mesh, render_polyhedral_body
from spindoctor.support.types import NDArrayFloatType

# A fold: two faces sharing a vertical ridge through the pixel centers of column 10
# at depth 0, sloping away to feet at different depths, so both faces cover the
# ridge pixels at exactly the same depth but shade differently.  The right face
# (unit normal depth component sqrt(1/2)) is listed before the left one (2/sqrt(5)).
_FOLD_VERTICES = np.array(
    [
        [10.5, 2.5, 0.0],  # ridge top
        [10.5, 18.5, 0.0],  # ridge bottom
        [2.5, 10.5, -4.0],  # left foot
        [18.5, 10.5, -8.0],  # right foot
    ]
)
_FOLD = Mesh(vertices=_FOLD_VERTICES, faces=np.array([[0, 3, 1], [0, 1, 2]]))
_RIGHT_FACE_SHADE = math.sqrt(0.5)

# A triangle reaching past the top and left edges of a 16 x 16 frame; its long edge
# is the line u + v = 20 in pixel-corner coordinates.
_OVERHANG = np.array([[-10.0, -10.0, 0.0], [30.0, -10.0, 0.0], [-10.0, 30.0, 0.0]])


def _render(mesh: Mesh, *, size: int = 21, shading: str = 'flat') -> NDArrayFloatType:
    """Render ``mesh`` in pixel units with the light along the line of sight."""
    return render_polyhedral_body(
        size=(size, size),
        center=(0.0, 0.0),
        mesh=mesh,
        semi_axes_px=(1.0, 1.0, 1.0),
        phase_angle=0.0,
        anti_aliasing=0.0,
        shading=shading,
    )


def test_shared_edge_tie_goes_to_the_face_listed_first() -> None:
    """A ridge pixel both faces cover at equal depth takes the first face's shade."""
    image = _render(_FOLD)
    assert image[10, 10] == pytest.approx(_RIGHT_FACE_SHADE)


def test_shared_edge_tie_holds_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one box row per batch, the later face still cannot take a tied pixel."""
    monkeypatch.setattr(mesh_geometry, '_RASTER_BATCH_PIXELS', 1)
    image = _render(_FOLD)
    assert image[10, 10] == pytest.approx(_RIGHT_FACE_SHADE)


# A far face square to the line of sight (shade 1) and a smaller tilted face wholly
# in front of it (shade 2/sqrt(5)), overlapping at pixel (4, 4).
_FAR_AND_NEAR = np.array(
    [
        [0.0, 0.0, -10.0],
        [20.0, 0.0, -10.0],
        [0.0, 20.0, -10.0],
        [2.0, 2.0, 0.0],
        [10.0, 2.0, -4.0],
        [2.0, 10.0, 0.0],
    ]
)
_FAR_FACE = [0, 1, 2]
_NEAR_FACE = [3, 4, 5]


@pytest.mark.parametrize(
    ('faces', 'batch_pixels'),
    [
        ([_NEAR_FACE, _FAR_FACE], None),
        ([_FAR_FACE, _NEAR_FACE], None),
        ([_FAR_FACE, _NEAR_FACE], 1),
    ],
    ids=['near-listed-first', 'near-listed-second', 'near-second-one-row-per-batch'],
)
def test_nearer_face_wins_whatever_its_order(
    monkeypatch: pytest.MonkeyPatch, faces: list[list[int]], batch_pixels: int | None
) -> None:
    """The nearer face shows over the farther one in either order and batching."""
    if batch_pixels is not None:
        monkeypatch.setattr(mesh_geometry, '_RASTER_BATCH_PIXELS', batch_pixels)
    image = _render(Mesh(vertices=_FAR_AND_NEAR, faces=np.array(faces)))
    assert image[4, 4] == pytest.approx(2.0 / math.sqrt(5.0))


def test_triangle_crossing_the_frame_edge_covers_its_in_frame_pixels() -> None:
    """Exactly the in-frame pixel centers on or inside the triangle are covered."""
    image = _render(Mesh(vertices=_OVERHANG, faces=np.array([[0, 1, 2]])), size=16)
    v, u = np.mgrid[0:16, 0:16]
    assert np.array_equal(image > 0.0, (u + 0.5) + (v + 0.5) <= 20.0)


def test_back_facing_triangle_is_culled() -> None:
    """The same triangle wound away from the observer covers nothing."""
    image = _render(Mesh(vertices=_OVERHANG, faces=np.array([[0, 2, 1]])), size=16)
    assert not image.any()


def test_non_finite_center_is_rejected() -> None:
    """A NaN center raises rather than rendering a blank body."""
    with pytest.raises(ValueError, match=r'center must be finite; got \(nan, 0\.0\)'):
        render_polyhedral_body(
            size=(16, 16),
            center=(math.nan, 0.0),
            mesh=Mesh(vertices=_OVERHANG, faces=np.array([[0, 1, 2]])),
            semi_axes_px=(1.0, 1.0, 1.0),
        )


def test_gouraud_shades_a_shared_vertex_with_the_averaged_normal() -> None:
    """The ridge-top vertex pixel takes the area-weighted mean normal of both faces.

    The raw face normals are (-64, 0, 128) and (128, 0, 128), so the vertex normal
    is along (64, 0, 256); flat shading would give that pixel the first face's
    sqrt(1/2) instead.
    """
    image = _render(_FOLD, shading='gouraud')
    assert image[2, 10] == pytest.approx(256.0 / math.hypot(64.0, 256.0))
