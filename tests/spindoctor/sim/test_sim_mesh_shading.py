"""Acceptance checks for the mesh-body truth upgrades.

Gouraud shading removes the facet-boundary shading discontinuities without
moving the silhouette; mesh detail octaves add high-frequency shape content
on top of an unchanged base shape; mesh-space limb relief perturbs the
silhouette with the same statistics the ellipsoid relief contract commands;
and the per-frame pose scatter moves the RENDERED pose by the recorded
drawn amount while the navigator's catalog view is unmoved.
"""

import math
from typing import Any, cast

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.forward.body_mesh import render_single_mesh_body
from spindoctor.sim.forward.body_spec import body_spec_from_params
from spindoctor.sim.mesh_geometry import (
    make_irregular_mesh,
    render_polyhedral_body,
)
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import build_nav_params
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_SIZE = 220
_CENTER = 110.0
_RADIUS = 80.0


def _render_sphere_mesh(shading: str) -> NDArrayFloatType:
    """A coarse UV sphere (big facets) rendered in the given shading mode."""
    mesh = make_irregular_mesh(n_lat=12, n_lon=24, lumpiness=0.0, seed=1)
    return render_polyhedral_body(
        size=(_SIZE, _SIZE),
        center=(_CENTER, _CENTER),
        mesh=mesh,
        semi_axes_px=(_RADIUS, _RADIUS, _RADIUS),
        illumination_angle=0.4,
        phase_angle=0.9,
        anti_aliasing=0.0,
        shading=shading,
    )


def _facet_discontinuity(img: NDArrayFloatType) -> float:
    """Mean absolute second difference of the shading over the disc interior."""
    interior: NDArrayBoolType = ndimage.binary_erosion(img > 0, iterations=3)
    d2v = np.abs(img[2:, :] - 2.0 * img[1:-1, :] + img[:-2, :])
    d2u = np.abs(img[:, 2:] - 2.0 * img[:, 1:-1] + img[:, :-2])
    return float(d2v[interior[1:-1, :]].mean() + d2u[interior[:, 1:-1]].mean())


def test_gouraud_removes_facet_shading_discontinuities() -> None:
    """The interior shading-discontinuity metric drops by a large factor."""
    flat = _facet_discontinuity(_render_sphere_mesh('flat'))
    gouraud = _facet_discontinuity(_render_sphere_mesh('gouraud'))
    assert flat > 10.0 * gouraud


def test_gouraud_leaves_the_silhouette_unchanged() -> None:
    """Shading mode only re-shades: the rasterized silhouette is identical."""
    flat = _render_sphere_mesh('flat')
    gouraud = _render_sphere_mesh('gouraud')
    assert np.array_equal(flat > 0.0, gouraud > 0.0)


def test_unknown_shading_mode_is_rejected() -> None:
    """The shared rasterizer rejects a shading mode it does not implement."""
    mesh = make_irregular_mesh(n_lat=8, n_lon=16, lumpiness=0.0, seed=1)
    with pytest.raises(ValueError, match="shading must be 'flat' or 'gouraud'"):
        render_polyhedral_body(
            size=(64, 64),
            center=(32.0, 32.0),
            mesh=mesh,
            semi_axes_px=(20.0, 20.0, 20.0),
            shading='phong',
        )


def test_detail_octaves_zero_reproduces_the_base_mesh() -> None:
    """detail_octaves = 0 is bit-exact with the base draw (octaves opt in)."""
    base = make_irregular_mesh(n_lat=16, n_lon=32, lumpiness=0.3, seed=7)
    explicit = make_irregular_mesh(n_lat=16, n_lon=32, lumpiness=0.3, seed=7, detail_octaves=0)
    assert np.array_equal(base.vertices, explicit.vertices)
    assert np.array_equal(base.faces, explicit.faces)


def _equator_radii(n_lat: int, n_lon: int, octaves: int) -> NDArrayFloatType:
    """Vertex radii around the mesh's equatorial ring."""
    mesh = make_irregular_mesh(
        n_lat=n_lat, n_lon=n_lon, lumpiness=0.3, seed=7, detail_octaves=octaves
    )
    ring = (n_lat // 2 - 1) * n_lon
    ring_verts = mesh.vertices[ring : ring + n_lon]
    return np.asarray(np.linalg.norm(ring_verts, axis=1), dtype=np.float64)


def test_detail_octaves_add_high_frequency_content_over_the_base() -> None:
    """Octaves raise the shape's high-frequency energy but keep the base shape."""
    smooth = _equator_radii(64, 128, 0)
    detailed = _equator_radii(64, 128, 2)
    # The base (low-frequency) shape is preserved underneath...
    assert float(np.corrcoef(smooth, detailed)[0, 1]) > 0.8

    # ...while the small-scale (second-difference) energy grows strongly.
    def roughness(radii: NDArrayFloatType) -> float:
        return float(np.mean(np.abs(np.diff(radii, n=2))))

    assert roughness(detailed) > 3.0 * roughness(smooth)


def _render_mesh_body(body: dict[str, Any]) -> NDArrayFloatType:
    """Render one mesh body through the forward path onto a black frame."""
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    render_single_mesh_body(
        img,
        body_spec_from_params(body, index=0),
        'LUMP',
        center_v=_CENTER,
        center_u=_CENTER,
        axis1=2.0 * _RADIUS,
        axis2=2.0 * _RADIUS,
        axis3=2.0 * _RADIUS,
        illumination_angle=0.0,
        phase_angle=math.radians(5.0),
        anti_aliasing=0.0,
        ref_center_v=_CENTER,
        ref_center_u=_CENTER,
        seed=99,
        body_index=0,
    )
    return img


_MESH_SPHERE: dict[str, Any] = {
    'name': 'LUMP',
    'shape_model': 'polyhedral_mesh',
    'mesh_lumpiness': 0.0,
    'mesh_n_lat': 48,
    'mesh_n_lon': 96,
    'mesh_seed': 5,
}


def _boundary_radii(img: NDArrayFloatType) -> NDArrayFloatType:
    """Silhouette radius along 256 rays from the body center."""
    mask = img > 0
    radii = []
    for angle in np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / 256.0):
        dv, du = math.cos(angle), math.sin(angle)
        r = 60.0
        while r < 105.0 and mask[int(_CENTER + r * dv), int(_CENTER + r * du)]:
            r += 0.05
        radii.append(r)
    return np.asarray(radii, dtype=np.float64)


def test_mesh_relief_perturbs_the_silhouette_like_the_ellipsoid_case() -> None:
    """The silhouette displacement RMS lands near the commanded rms * R."""
    rms = 0.02
    base = _render_mesh_body(_MESH_SPHERE)
    relief = _render_mesh_body({**_MESH_SPHERE, 'limb_relief_rms': rms})
    delta = _boundary_radii(relief) - _boundary_radii(base)
    commanded = rms * _RADIUS
    assert 0.5 * commanded < float(np.std(delta)) < 2.0 * commanded


def test_mesh_relief_off_is_bit_identical() -> None:
    """limb_relief_rms 0 renders the base mesh byte-for-byte."""
    base = _render_mesh_body(_MESH_SPHERE)
    explicit = _render_mesh_body({**_MESH_SPHERE, 'limb_relief_rms': 0.0})
    assert np.array_equal(base, explicit)


def _pose_scatter_scene(
    *, sigma_deg: float, pose_override: list[float] | None = None
) -> dict[str, Any]:
    """A one-mesh-body scene, optionally replacing the pose or adding scatter."""
    body: dict[str, Any] = {
        'name': 'HYPERION',
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': 0.25,
        'mesh_seed': 7,
        'pose_euler_deg': pose_override if pose_override is not None else [10.0, 35.0, 0.0],
        'center_v': 110.0,
        'center_u': 110.0,
        'axis1': 150.0,
        'axis2': 110.0,
        'axis3': 95.0,
        'illumination_angle': 25.0,
        'phase_angle': 30.0,
    }
    if sigma_deg > 0.0:
        body['pose_scatter'] = {'sigma_deg': sigma_deg}
    return {
        'instrument': 'coiss_nac',
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 42,
        'bodies': [body],
    }


def test_pose_scatter_moves_the_render_by_the_recorded_drawn_amount() -> None:
    """The scattered render equals a render at catalog pose + the drawn angles."""
    scattered_img, meta = render_combined_model(_pose_scatter_scene(sigma_deg=3.0))
    drawn = meta['bodies']['HYPERION']['pose_scatter_drawn_deg']
    assert len(drawn) == 3
    assert any(abs(d) > 1e-6 for d in drawn)
    equivalent_pose = [10.0 + drawn[0], 35.0 + drawn[1], 0.0 + drawn[2]]
    explicit_img, _ = render_combined_model(
        _pose_scatter_scene(sigma_deg=0.0, pose_override=equivalent_pose)
    )
    assert np.array_equal(scattered_img, explicit_img)


def test_pose_scatter_draw_is_returned_with_the_body_info() -> None:
    """The mesh renderer returns the drawn scatter for the render truth."""
    body: dict[str, Any] = {**_MESH_SPHERE, 'pose_scatter': {'sigma_deg': 3.0}}
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    _, info = render_single_mesh_body(
        img,
        body_spec_from_params(body, index=0),
        'LUMP',
        center_v=_CENTER,
        center_u=_CENTER,
        axis1=2.0 * _RADIUS,
        axis2=2.0 * _RADIUS,
        axis3=2.0 * _RADIUS,
        illumination_angle=0.0,
        phase_angle=math.radians(5.0),
        anti_aliasing=0.0,
        ref_center_v=_CENTER,
        ref_center_u=_CENTER,
        seed=99,
        body_index=0,
    )
    drawn = info['pose_scatter_drawn_deg']
    assert len(drawn) == 3
    assert any(abs(d) > 1e-6 for d in drawn)


def test_pose_scatter_changes_the_rendered_frame() -> None:
    """The scattered frame differs from the catalog-pose frame."""
    scattered_img, _ = render_combined_model(_pose_scatter_scene(sigma_deg=3.0))
    catalog_img, _ = render_combined_model(_pose_scatter_scene(sigma_deg=0.0))
    assert not np.array_equal(scattered_img, catalog_img)


def test_navigator_view_is_unmoved_by_pose_scatter_and_shading() -> None:
    """nav_params is identical with or without the mesh truth keys.

    The navigator predicts the catalog pose with flat shading by
    construction: pose_scatter and shading are truth keys the boundary
    filter strips, so the filtered views of a scattered/gouraud scene and
    the plain scene are equal and the predicted mesh cannot move.
    """
    plain = _pose_scatter_scene(sigma_deg=0.0)
    dressed = _pose_scatter_scene(sigma_deg=3.0)
    dressed['bodies'][0]['shading'] = 'gouraud'
    assert build_nav_params(dressed) == build_nav_params(plain)


def test_mesh_anti_aliasing_suppressed_on_oversampled_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mesh path consumes anti_aliasing only at oversample 1, like topo.

    An oversampled scene already resolves the limb on its own grid, so the
    dispatch hands the mesh rasterizer anti_aliasing 0 there; supersampling
    on top of the oversampled grid would only multiply the render cost.
    """
    from spindoctor.sim.forward import body_mesh
    from spindoctor.sim.forward.body import render_single_body

    captured: list[float] = []
    real = body_mesh.render_single_mesh_body

    def _capture(*args: Any, **kwargs: Any) -> Any:
        captured.append(float(kwargs['anti_aliasing']))
        return real(*args, **kwargs)

    monkeypatch.setattr(body_mesh, 'render_single_mesh_body', _capture)
    body_params: dict[str, Any] = {
        'name': 'LUMPY',
        'shape_model': 'polyhedral_mesh',
        'axis1': 24.0,
        'axis2': 20.0,
        'axis3': 18.0,
        'mesh_lumpiness': 0.3,
        'mesh_seed': 5,
        'anti_aliasing': 1.0,
    }
    img = np.zeros((64, 64), dtype=np.float64)
    render_single_body(
        img,
        body_spec_from_params(body_params, index=0),
        0.0,
        offset_u=0.0,
        ref_center_v=32.0,
        ref_center_u=32.0,
    )
    img_os = np.zeros((128, 128), dtype=np.float64)
    scaled = dict(body_params)
    for axis_key in ('axis1', 'axis2', 'axis3'):
        scaled[axis_key] = float(cast(float, scaled[axis_key])) * 2
    render_single_body(
        img_os,
        body_spec_from_params(scaled, index=0),
        0.0,
        offset_u=0.0,
        ref_center_v=64.0,
        ref_center_u=64.0,
        oversample=2,
    )
    assert captured[0] == 1.0
    assert captured[1] == 0.0
