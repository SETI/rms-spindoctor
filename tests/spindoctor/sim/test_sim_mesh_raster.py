"""Rasterization rules of the shared mesh renderer.

Most meshes here are built directly in pixel units (unit semi-axes, zero pose, no
supersampling, light from the observer), so a vertex sits exactly where its
coordinates say and a flat face's shade is the depth component of its unit normal.
"""

import math

import numpy as np
import pytest

from spindoctor.sim.shared import mesh as shared_mesh
from spindoctor.sim.shared.mesh import Mesh, MeshBodyRender, render_polyhedral_body
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


def _render_full(mesh: Mesh, *, size: int = 21, shading: str = 'flat') -> MeshBodyRender:
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


def _render(mesh: Mesh, *, size: int = 21, shading: str = 'flat') -> NDArrayFloatType:
    """The brightness of :func:`_render_full`."""
    return _render_full(mesh, size=size, shading=shading).intensity


def test_shared_edge_tie_goes_to_the_face_listed_first() -> None:
    """A ridge pixel both faces cover at equal depth takes the first face's shade."""
    image = _render(_FOLD)
    assert image[10, 10] == pytest.approx(_RIGHT_FACE_SHADE)


def test_shared_edge_tie_holds_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one box row per batch, the later face still cannot take a tied pixel."""
    monkeypatch.setattr(shared_mesh, '_RASTER_BATCH_PIXELS', 1)
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
        monkeypatch.setattr(shared_mesh, '_RASTER_BATCH_PIXELS', batch_pixels)
    image = _render(Mesh(vertices=_FAR_AND_NEAR, faces=np.array(faces)))
    assert image[4, 4] == pytest.approx(2.0 / math.sqrt(5.0))


def test_triangle_crossing_the_frame_edge_covers_its_in_frame_pixels() -> None:
    """Exactly the in-frame pixel centers on or inside the triangle are covered."""
    render = _render_full(Mesh(vertices=_OVERHANG, faces=np.array([[0, 1, 2]])), size=16)
    v, u = np.mgrid[0:16, 0:16]
    assert np.array_equal(render.coverage > 0.0, (u + 0.5) + (v + 0.5) <= 20.0)


def test_coverage_is_the_covered_fraction_of_each_pixel_however_dark() -> None:
    """The coverage counts the supersample centers a face covers, lit or not.

    Lit from behind, the triangle renders black, but its coverage is still 1 inside
    and the covered fraction of each pixel along its long edge.  A coverage read
    from its shading would be empty.
    """
    render = render_polyhedral_body(
        size=(16, 16),
        center=(0.0, 0.0),
        mesh=Mesh(vertices=_OVERHANG, faces=np.array([[0, 1, 2]])),
        semi_axes_px=(1.0, 1.0, 1.0),
        phase_angle=math.pi,
        anti_aliasing=1.0,
    )
    samples = (np.arange(64) + 0.5) / 4.0
    covered = (samples[:, None] + samples[None, :] <= 20.0).astype(np.float64)
    expected = covered.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    assert np.array_equal(render.coverage, expected)


@pytest.mark.parametrize(('phase_deg', 'lit'), [(80.0, True), (100.0, False)])
def test_face_is_lit_by_its_mean_angle_of_incidence(phase_deg: float, lit: bool) -> None:
    """A face square to the line of sight is lit at 80 deg phase and unlit at 100 deg."""
    render = render_polyhedral_body(
        size=(16, 16),
        center=(0.0, 0.0),
        mesh=Mesh(vertices=_OVERHANG, faces=np.array([[0, 1, 2]])),
        semi_axes_px=(1.0, 1.0, 1.0),
        phase_angle=math.radians(phase_deg),
        anti_aliasing=0.0,
        with_lit=True,
    )
    assert render.lit is not None
    assert np.array_equal(render.lit, np.full((16, 16), lit) & (render.coverage > 0.0))


def test_body_draws_the_same_pixels_on_frames_of_any_size() -> None:
    """A lumpy mesh at the default pose draws the same window on a 64 and a 192 px frame.

    At the default pose, with the body centered on a whole pixel, many supersample
    centers lie exactly on the edges between faces, so which face covers them depends
    on the last bit of each edge test.
    """
    mesh = shared_mesh.make_irregular_mesh(lumpiness=0.3)
    small, large = (
        render_polyhedral_body(
            size=(size, size),
            center=(size / 2.0, size / 2.0),
            mesh=mesh,
            semi_axes_px=(20.0, 20.0, 20.0),
            anti_aliasing=1.0,
        ).intensity
        for size in (64, 192)
    )
    assert np.array_equal(small, large[64:128, 64:128])


def test_back_facing_triangle_is_culled() -> None:
    """The same triangle wound away from the observer covers nothing."""
    render = _render_full(Mesh(vertices=_OVERHANG, faces=np.array([[0, 2, 1]])), size=16)
    assert not render.coverage.any()


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


def test_gouraud_blends_three_vertices_inside_a_face() -> None:
    """An interior pixel takes the barycentric blend of its face's vertex intensities.

    Pixel (10, 7) has its center at u = 7.5, v = 10.5 on the left face, with weight
    0.375 on the left foot and 0.3125 on each ridge vertex.  The ridge vertices carry
    the averaged normal's intensity and the foot carries the left face's own, 2/sqrt(5).
    """
    image = _render(_FOLD, shading='gouraud')
    ridge = 256.0 / math.hypot(64.0, 256.0)
    expected = 0.625 * ridge + 0.375 * 2.0 / math.sqrt(5.0)
    assert image[10, 7] == pytest.approx(expected)
