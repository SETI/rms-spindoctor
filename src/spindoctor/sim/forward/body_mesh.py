"""Image-side polyhedral-mesh body rendering.

The mesh primitives live in the shared :mod:`spindoctor.sim.mesh_geometry`
module so the rendered mesh and the navigator's predicted mesh are the same
shape by construction.  On top of that shared geometry this image side adds
its truth-key upgrades, none of which the navigator's prediction consumes:

- ``shading`` selects the shared rasterizer's mode for the RENDERED image
  ('flat' default, 'gouraud' per-vertex smooth shading).  The rasterizer
  capability is shared; each side chooses its own mode, and the navigator's
  predicted mesh keeps flat shading because the key is truth-side.
- ``limb_relief_rms`` / ``limb_relief_corr_deg`` apply the same relief-field
  machinery the ellipsoid path uses, here as a per-vertex radial
  perturbation of the unit mesh (its own seeded 'relief' stream).  The
  field is sampled in body-fixed spherical coordinates -- mesh terrain is
  attached to the body and rotates with the pose -- so the commanded
  limb-slice statistics apply to whichever great circle the pose turns
  toward the observer.
- ``pose_scatter`` draws a seeded per-frame Gaussian perturbation (sigma_deg
  per Euler axis, its own 'pose_scatter' stream) added to the rendered
  pose only.  The navigator predicts the catalog pose, so the drawn
  rotation is a known-wrong rotation state; the draw is recorded in the
  render truth as ``pose_scatter_drawn_deg``.
"""

import dataclasses
from functools import lru_cache
from typing import Any

import numpy as np

from spindoctor.sim.forward.body import finish_single_body
from spindoctor.sim.forward.relief import ReliefField, synthesize_relief_field
from spindoctor.sim.mesh_geometry import (
    Mesh,
    MeshBodySpec,
    make_irregular_mesh,
    mesh_spec_from_params,
    render_polyhedral_body,
)
from spindoctor.sim.seeds import derive_effect_seed, stable_param_seed
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['render_single_mesh_body']


def _perturb_mesh_radially(mesh: Mesh, field: ReliefField) -> Mesh:
    """Scale each vertex radially by ``1 + h(lat, lon)`` of the relief field.

    Coordinates are body-fixed spherical: latitude from the body's xy plane
    toward +z, longitude around +z from +x.  The perturbation moves vertices
    along their radius only, so the mesh stays star-shaped and the face
    winding stays outward for the relief amplitudes the schema admits.

    Parameters:
        mesh: The unit-radius base mesh.
        field: The relief-field realization (fractional heights).

    Returns:
        A new perturbed :class:`Mesh` sharing the face array.
    """
    vertices = mesh.vertices
    radius = np.linalg.norm(vertices, axis=1)
    safe_radius = np.maximum(radius, 1e-12)
    lat = np.arcsin(np.clip(vertices[:, 2] / safe_radius, -1.0, 1.0))
    lon = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2.0 * np.pi)
    scale = 1.0 + field.sample(lat, lon)
    return Mesh(vertices=vertices * scale[:, np.newaxis], faces=mesh.faces)


@lru_cache(maxsize=30)
def _render_mesh_shape_cached(
    size_v: int,
    size_u: int,
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    spec: MeshBodySpec,
    illumination_angle: float,
    phase_angle: float,
    anti_aliasing: float,
    shading: str,
    relief_rms: float,
    relief_corr_deg: float,
    relief_seed: int,
) -> NDArrayFloatType:
    """Cache an irregular mesh body shape at the reference (image) center.

    The truth-side upgrades (shading mode, relief field) are part of the
    cache key; at their defaults ('flat', relief off) this renders exactly
    the shared ``render_mesh_body_image`` result the navigator predicts.
    """
    mesh = make_irregular_mesh(
        n_lat=spec.n_lat,
        n_lon=spec.n_lon,
        lumpiness=spec.lumpiness,
        seed=spec.seed,
        detail_octaves=spec.detail_octaves,
    )
    if relief_rms > 0.0:
        field = synthesize_relief_field(relief_rms, relief_corr_deg, relief_seed)
        if field.h_max > field.h_min:
            mesh = _perturb_mesh_radially(mesh, field)
    return render_polyhedral_body(
        size=(size_v, size_u),
        center=(size_v / 2.0, size_u / 2.0),
        mesh=mesh,
        semi_axes_px=(axis1 / 2.0, axis2 / 2.0, axis3 / 2.0),
        pose_euler_deg=spec.pose_euler_deg,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
        shading=shading,
    )


def _mesh_identity_seed(
    body_params: dict[str, Any],
    *,
    seed: int | None,
    body_index: int,
    body_name: str,
    axis1: float,
    axis2: float,
    axis3: float,
) -> int:
    """The per-body identity seed the mesh truth streams derive from.

    Mirrors the ellipsoid path's chain exactly: the scene sub-seed mixed
    with the body's stable identity when available, else the explicit
    per-body seed, else a geometry-derived process-stable seed.
    """
    if seed is not None:
        return derive_effect_seed(seed, f'body:{body_index}:{body_name}')
    if body_params.get('seed') is not None:
        return int(body_params['seed']) & 0x7FFFFFFF
    return stable_param_seed(axis1, axis2, axis3, body_index, body_name) & 0x7FFFFFFF


def render_single_mesh_body(
    img: NDArrayFloatType,
    body_params: dict[str, Any],
    body_name: str,
    *,
    center_v: float,
    center_u: float,
    axis1: float,
    axis2: float,
    axis3: float,
    illumination_angle: float,
    phase_angle: float,
    anti_aliasing: float,
    ref_center_v: float,
    ref_center_u: float,
    seed: int | None = None,
    body_index: int = 0,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render one polyhedral-mesh body into the image in place.

    Parameters:
        img: Image array to modify in-place.
        body_params: Body parameters dictionary (mesh keys are parsed by
            ``mesh_spec_from_params``; the truth keys ``shading``,
            ``limb_relief_*``, and ``pose_scatter`` are read here).  The
            mapping is never mutated: when a pose scatter is drawn, the
            returned body info's ``params`` is a copy of this mapping with
            ``pose_scatter_drawn_deg`` added (render truth metadata).
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        axis1: Full width of ellipsoidal-envelope axis 1 in pixels.
        axis2: Full width of ellipsoidal-envelope axis 2 in pixels.
        axis3: Full width of ellipsoidal-envelope axis 3 in pixels.
        illumination_angle: Image-plane light azimuth in radians.
        phase_angle: Phase angle in radians.
        anti_aliasing: Limb supersampling control.
        ref_center_v: Reference center V for shape caching.
        ref_center_u: Reference center U for shape caching.
        seed: Scene-level sub-seed the per-body truth streams derive from.
        body_index: Stable index of this body in the scene's body list.

    Returns:
        Tuple of (body_mask, body_info_dict) matching the ellipsoid path.
    """
    size_v, size_u = img.shape
    spec = mesh_spec_from_params(body_params)
    shading = str(body_params.get('shading', 'flat'))
    relief_rms = float(body_params.get('limb_relief_rms', 0.0))
    relief_corr_deg = float(body_params.get('limb_relief_corr_deg', 15.0))
    scatter = body_params.get('pose_scatter') or {}
    scatter_sigma_deg = float(scatter.get('sigma_deg', 0.0))

    # The relief terrain and the pose scatter each draw from their own named
    # stream of the per-body identity seed, mirroring the ellipsoid path, so
    # they are independent of each other and of any crater draws.  The drawn
    # scatter travels to the render truth through the returned body info's
    # params copy; body_params itself stays untouched (the caller's scene
    # mapping must remain schema-valid, and pose_scatter_drawn_deg is not a
    # schema key).
    truth_params = body_params
    relief_seed = 0
    if relief_rms > 0.0 or scatter_sigma_deg > 0.0:
        identity_seed = _mesh_identity_seed(
            body_params,
            seed=seed,
            body_index=body_index,
            body_name=body_name,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
        )
        if relief_rms > 0.0:
            relief_seed = derive_effect_seed(identity_seed, 'relief')
        if scatter_sigma_deg > 0.0:
            rng = np.random.default_rng(derive_effect_seed(identity_seed, 'pose_scatter'))
            drawn = rng.normal(0.0, scatter_sigma_deg, size=3)
            truth_params = {**body_params, 'pose_scatter_drawn_deg': [float(d) for d in drawn]}
            spec = dataclasses.replace(
                spec,
                pose_euler_deg=(
                    spec.pose_euler_deg[0] + float(drawn[0]),
                    spec.pose_euler_deg[1] + float(drawn[1]),
                    spec.pose_euler_deg[2] + float(drawn[2]),
                ),
            )

    body_shape = _render_mesh_shape_cached(
        size_v,
        size_u,
        axis1,
        axis2=axis2,
        axis3=axis3,
        spec=spec,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
        shading=shading,
        relief_rms=relief_rms,
        relief_corr_deg=relief_corr_deg,
        relief_seed=relief_seed,
    )
    # A mesh body's pose is an arbitrary 3D rotation, so fall back to the
    # bounding sphere of the ellipsoidal envelope for both image axes.
    mesh_half_extent = max(axis1, axis2, axis3) / 2.0
    return finish_single_body(
        img,
        body_shape,
        truth_params,
        body_name=body_name,
        center_v=center_v,
        center_u=center_u,
        half_extent_v=mesh_half_extent,
        half_extent_u=mesh_half_extent,
        ref_center_v=ref_center_v,
        ref_center_u=ref_center_u,
    )
