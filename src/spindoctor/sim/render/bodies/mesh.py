"""Image-side polyhedral-mesh body rendering.

The mesh primitives live in the shared :mod:`spindoctor.sim.shared.mesh`
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
  rotation is a known-wrong rotation state; the draw is returned with the
  body info as ``pose_scatter_drawn_deg``, and the scene radiance stage
  records it in the render truth.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from spindoctor.sim.render.bodies.placement import finish_single_body
from spindoctor.sim.render.bodies.relief import ReliefField, synthesize_relief_field
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.sim.shared.mesh import (
    Mesh,
    MeshBodySpec,
    make_irregular_mesh,
    render_polyhedral_body,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['clear_mesh_shape_caches', 'render_single_mesh_body']


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
    semi_a_px: float,
    *,
    semi_b_px: float,
    semi_c_px: float,
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
        semi_axes_px=(semi_a_px, semi_b_px, semi_c_px),
        pose_euler_deg=spec.pose_euler_deg,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
        shading=shading,
    )


def clear_mesh_shape_caches() -> None:
    """Drop the cached mesh body shapes (test/tooling helper)."""
    _render_mesh_shape_cached.cache_clear()


def render_single_mesh_body(
    img: NDArrayFloatType,
    body: BodySpec,
    body_name: str,
    *,
    center_v: float,
    center_u: float,
    semi_a_px: float,
    semi_b_px: float,
    semi_c_px: float,
    illumination_angle: float,
    phase_angle: float,
    anti_aliasing: float,
    ref_center_v: float,
    ref_center_u: float,
    body_streams_seed: int | None = None,
    body_index: int = 0,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Render one polyhedral-mesh body into the image in place.

    Parameters:
        img: Image array to modify in-place.
        body: The body, parsed by
            :func:`~spindoctor.sim.render.bodies.spec.body_spec_from_params`.
            Its mesh shape and pose, and the truth fields ``shading``,
            ``limb_relief_*``, and ``pose_scatter_sigma_deg``, are read here.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        semi_a_px: Semi-axis a of the ellipsoidal envelope in pixels.
        semi_b_px: Semi-axis b of the ellipsoidal envelope in pixels.
        semi_c_px: Semi-axis c of the ellipsoidal envelope in pixels.
        illumination_angle: Image-plane light azimuth in radians.
        phase_angle: Phase angle in radians.
        anti_aliasing: Limb supersampling control.
        ref_center_v: Reference center V for shape caching.
        ref_center_u: Reference center U for shape caching.
        body_streams_seed: The scene's seed for every body's random streams; this
            body's relief and pose-scatter streams derive from it, mixed with the
            body's index and name.
        body_index: Stable index of this body in the scene's body list.

    Returns:
        Tuple of (body_mask, body_info_dict) matching the ellipsoid path.
        When a pose scatter is drawn, the info also carries the drawn
        per-axis angles in degrees as ``pose_scatter_drawn_deg``.
    """
    size_v, size_u = img.shape
    streams = _draw_mesh_streams(
        body,
        body_name,
        body_streams_seed=body_streams_seed,
        body_index=body_index,
        widths=(2.0 * semi_a_px, 2.0 * semi_b_px, 2.0 * semi_c_px),
    )
    body_shape = _render_mesh_shape_cached(
        size_v,
        size_u,
        semi_a_px,
        semi_b_px=semi_b_px,
        semi_c_px=semi_c_px,
        spec=streams.mesh,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
        shading=body.shading,
        relief_rms=body.limb_relief_rms,
        relief_corr_deg=body.limb_relief_corr_deg,
        relief_seed=streams.relief_seed,
    )
    # A mesh body's pose is an arbitrary 3D rotation, so fall back to the
    # bounding sphere of the ellipsoidal envelope for both image axes.
    mesh_half_extent = max(semi_a_px, semi_b_px, semi_c_px)
    mask, body_info = finish_single_body(
        img,
        body_shape,
        body_name=body_name,
        center_v=center_v,
        center_u=center_u,
        half_extent_v=mesh_half_extent,
        half_extent_u=mesh_half_extent,
        ref_center_v=ref_center_v,
        ref_center_u=ref_center_u,
        range_km=body.range_km,
    )
    if streams.drawn_deg is not None:
        body_info['pose_scatter_drawn_deg'] = streams.drawn_deg
    return mask, body_info


@dataclass(frozen=True)
class _MeshStreams:
    """What a mesh body's random streams draw.

    Parameters:
        mesh: The mesh shape and the pose it is rendered at, the drawn pose scatter
            added.
        relief_seed: The seed of the relief terrain (0 for a body without relief).
        drawn_deg: The drawn pose scatter per Euler axis in degrees, or None when
            the body has no pose scatter.
    """

    mesh: MeshBodySpec
    relief_seed: int
    drawn_deg: list[float] | None


def _draw_mesh_streams(
    body: BodySpec,
    body_name: str,
    *,
    body_streams_seed: int | None,
    body_index: int,
    widths: tuple[float, float, float],
) -> _MeshStreams:
    """Seed a mesh body's relief terrain and draw its pose scatter.

    The relief terrain and the pose scatter each draw from their own named stream
    of the body's identity seed, as the ellipsoid path's streams do, so they are
    independent of each other.  The drawn scatter travels to the render truth
    through the returned body info.

    Parameters:
        body: The body (its mesh, relief, pose scatter, and seed are read).
        body_name: The upper-cased body name.
        body_streams_seed: The scene's seed for every body's random streams, or None.
        body_index: The body's position in the scene's body list.
        widths: The full widths of the ellipsoidal envelope, twice its semi-axes, in
            pixels.

    Returns:
        The rendered mesh, the relief seed, and the drawn pose scatter.
    """
    mesh = body.mesh
    if body.limb_relief_rms <= 0.0 and body.pose_scatter_sigma_deg <= 0.0:
        return _MeshStreams(mesh=mesh, relief_seed=0, drawn_deg=None)
    # The fallback digest hashes the full widths and keeps its low 31 bits, so a
    # body drawn without seeds keeps its terrain (#913).
    identity_seed = body_identity_seed(
        body_streams_seed,
        body,
        body_index,
        fallback_key=(*widths, body_index, body_name),
        mask_fallback=True,
    )
    relief_seed = 0
    if body.limb_relief_rms > 0.0:
        relief_seed = derive_effect_seed(identity_seed, 'relief')
    drawn_deg: list[float] | None = None
    if body.pose_scatter_sigma_deg > 0.0:
        rng = np.random.default_rng(derive_effect_seed(identity_seed, 'pose_scatter'))
        drawn = rng.normal(0.0, body.pose_scatter_sigma_deg, size=3)
        drawn_deg = [float(d) for d in drawn]
        mesh = replace(
            mesh,
            pose_euler_deg=(
                mesh.pose_euler_deg[0] + float(drawn[0]),
                mesh.pose_euler_deg[1] + float(drawn[1]),
                mesh.pose_euler_deg[2] + float(drawn[2]),
            ),
        )
    return _MeshStreams(mesh=mesh, relief_seed=relief_seed, drawn_deg=drawn_deg)
