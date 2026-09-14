"""Polyhedral irregular-body renderer and navigation-geometry override (B7).

Covers the standalone mesh renderer (shape, pose, shading, determinism), its
hook into the combined render path, and the ``nav_override`` channel that builds
the predicted body from a divergent geometry (the render-vs-navigation
separation the shape-mismatch and pose-disagreement scenarios rely on).
"""

from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_model.nav_model_body_simulated import NavModelBodySimulated
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.mesh_geometry import make_irregular_mesh, render_polyhedral_body
from spindoctor.sim.render import render_combined_model
from spindoctor.sim.scene import build_nav_params


def _boundary_radial_cv(mask: np.ndarray) -> float:
    """Coefficient of variation of boundary radius about the silhouette centroid.

    A round body has near-constant boundary radius (cv ~ 0); surface relief
    raises it.
    """
    boundary = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.nonzero(boundary)
    cv_ = float(ys.mean())
    cu_ = float(xs.mean())
    radii = np.hypot(ys - cv_, xs - cu_)
    return float(radii.std() / radii.mean())


def _render(lumpiness: float, *, pose: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Render a single equal-axis mesh body (a sphere when lumpiness is 0)."""
    mesh = make_irregular_mesh(lumpiness=lumpiness, seed=4)
    return render_polyhedral_body(
        size=(96, 96),
        center=(48.0, 48.0),
        mesh=mesh,
        semi_axes_px=(34.0, 34.0, 34.0),
        pose_euler_deg=pose,
        illumination_angle=0.0,
        phase_angle=np.radians(90.0),
        anti_aliasing=1.0,
    )


def test_mesh_is_outward_wound() -> None:
    """Every face normal points away from the body center."""
    mesh = make_irregular_mesh(lumpiness=0.3, seed=1)
    p0 = mesh.vertices[mesh.faces[:, 0]]
    p1 = mesh.vertices[mesh.faces[:, 1]]
    p2 = mesh.vertices[mesh.faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    assert np.all(np.sum(normals * centroids, axis=1) > 0.0)


def test_zero_lumpiness_is_nearly_round() -> None:
    """A zero-relief mesh projects to a nearly circular silhouette."""
    cv = _boundary_radial_cv(_render(0.0) > 0)
    assert cv < 0.05


def test_relief_increases_roughness() -> None:
    """Surface relief makes the silhouette less round than a sphere."""
    smooth = _boundary_radial_cv(_render(0.0) > 0)
    lumpy = _boundary_radial_cv(_render(0.45) > 0)
    assert lumpy > smooth * 2.0


def test_pose_changes_silhouette() -> None:
    """Rotating the body changes its projected silhouette."""
    a = _render(0.45, pose=(0.0, 0.0, 0.0)) > 0
    b = _render(0.45, pose=(0.0, 90.0, 30.0)) > 0
    assert int((a != b).sum()) > 50


def test_render_is_deterministic() -> None:
    """The same mesh and pose render byte-identically."""
    assert np.array_equal(_render(0.4), _render(0.4))


def test_shading_follows_illumination() -> None:
    """With light from the top, the lit (upper) half is brighter."""
    img = _render(0.2)
    half = img.shape[0] // 2
    assert float(img[:half, :].mean()) > float(img[half:, :].mean())


def _mesh_scene(**body: Any) -> dict[str, Any]:
    """A noiseless coiss scene with one mesh body."""
    base = {
        'name': 'B',
        'center_v': 48.0,
        'center_u': 48.0,
        'axis1': 60.0,
        'axis2': 46.0,
        'axis3': 46.0,
        'illumination_angle': 30.0,
        'phase_angle': 45.0,
        'shape_model': 'polyhedral_mesh',
        'mesh_lumpiness': 0.4,
        'mesh_seed': 2,
    }
    base.update(body)
    return {
        'size_v': 96,
        'size_u': 96,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [base],
    }


def test_combined_render_mesh_differs_from_ellipsoid() -> None:
    """A mesh body and an ellipsoid body of equal axes render different shapes."""
    mesh_img, _ = render_combined_model(_mesh_scene())
    ell_img, _ = render_combined_model(_mesh_scene(shape_model='ellipsoid'))
    assert int(((mesh_img > 0) != (ell_img > 0)).sum()) > 50


def test_combined_render_mesh_pose_changes_shape() -> None:
    """The body pose drives the rendered mesh silhouette in the combined path."""
    a, _ = render_combined_model(_mesh_scene())
    b, _ = render_combined_model(_mesh_scene(pose_euler_deg=[0.0, 90.0, 30.0]))
    assert int(((a > 0) != (b > 0)).sum()) > 50


def test_combined_render_mesh_is_deterministic() -> None:
    """Re-rendering the same mesh scene is byte-identical."""
    a, _ = render_combined_model(_mesh_scene())
    b, _ = render_combined_model(_mesh_scene())
    assert np.array_equal(a, b)


def test_nav_params_overlays_override() -> None:
    """nav_override keys win over the body params and the key itself is dropped."""
    nav = build_nav_params(
        {
            'bodies': [
                {
                    'mesh_lumpiness': 0.4,
                    'pose_euler_deg': [1.0, 2.0, 3.0],
                    'nav_override': {'mesh_lumpiness': 0.0, 'shape_model': 'ellipsoid'},
                }
            ]
        }
    )
    assert nav['bodies'] == [
        {
            'mesh_lumpiness': 0.0,
            'pose_euler_deg': [1.0, 2.0, 3.0],
            'shape_model': 'ellipsoid',
        }
    ]


def test_nav_params_passthrough_without_override() -> None:
    """Absent an override the predicted params equal the rendered params."""
    body = {'mesh_lumpiness': 0.4, 'mesh_seed': 2}
    assert build_nav_params({'bodies': [body]})['bodies'] == [body]


def _body_sim_model(body: dict[str, Any]) -> NavModelBodySimulated:
    """Build and render the simulated body NavModel for a one-body scene."""
    obs = ObsSim.from_file('/tmp/override.json', sim_params=_mesh_scene(**body))
    model = next(m for m in build_models_for_obs(obs) if isinstance(m, NavModelBodySimulated))
    model.create_model()
    return model


def test_nav_override_predicts_the_overridden_shape() -> None:
    """The predicted silhouette follows nav_override, not the rendered shape.

    A body rendered lumpy but predicted smooth (the B7 scenario-2 channel) yields
    the same predicted silhouette as a body that is smooth on both sides, and a
    different one from a body predicted lumpy.
    """
    overridden = _body_sim_model(
        {'mesh_lumpiness': 0.4, 'nav_override': {'mesh_lumpiness': 0.0}}
    )._body_mask
    smooth = _body_sim_model({'mesh_lumpiness': 0.0})._body_mask
    lumpy = _body_sim_model({'mesh_lumpiness': 0.4})._body_mask
    assert overridden is not None
    assert smooth is not None
    assert lumpy is not None
    assert np.array_equal(overridden, smooth)
    assert not np.array_equal(overridden, lumpy)
