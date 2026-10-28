"""Shared polyhedral-mesh geometry for irregular sim bodies.

The ellipsoid renderers cannot produce the non-ellipsoidal silhouette of an
irregular body (Hyperion, Phoebe).  Because ``oops`` will not gain DSK
support, the sim carries its own small renderer that projects a triangle mesh
through a scene-supplied pose and rasterizes the shaded silhouette.  It is
sim-only: the body's orientation is ground truth from the scene, not from
SPICE.

This module is deliberately shared between the image-side forward renderer
(``spindoctor.sim.forward.body_mesh``) and the navigator-side predicted-body
renderer (``spindoctor.nav_model.nav_model_body_simulated``): the mesh shape,
pose, and rasterization conventions are idealized information both sides may
know, and sharing one implementation guarantees that a scene's planted
geometry error (via ``nav_override``) is the only difference between the
rendered and the predicted silhouette.  Every function takes explicit
geometry arguments; none reads the scene mapping, so no truth-side
information can cross the boundary here (``mesh_spec_from_params`` parses a
body parameter mapping, but only its idealized mesh keys).

The output contract matches ``create_simulated_body``: a ``(size_v, size_u)``
float array in [0, 1], lit by the same Lambertian convention so a mesh body and
an ellipsoid body can be compared directly (the B7 shape-mismatch fixture).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.sim.ellipsoid_geometry import DARK_SIDE_ILLUM_STRENGTH
from spindoctor.sim.spherical_harmonics import (
    degree_mean_square,
    draw_sh_coefficients,
    evaluate_at_directions,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayComplexType, NDArrayFloatType, NDArrayIntType

# Lambertian floor for the visible-but-unlit side, shared with the ellipsoid
# shading convention in spindoctor.sim.ellipsoid_geometry so the two paths
# cannot desync.
_DARK_SIDE_ILLUM = DARK_SIDE_ILLUM_STRENGTH


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh in a unit-radius body-fixed frame.

    Parameters:
        vertices: ``(N, 3)`` vertex coordinates (x, y, z).
        faces: ``(M, 3)`` vertex indices, wound so face normals point outward.
    """

    vertices: NDArrayFloatType
    faces: NDArrayIntType


def _rotation_matrix(euler_deg: tuple[float, float, float]) -> NDArrayFloatType:
    """Build a rotation matrix from intrinsic X, Y, Z Euler angles in degrees."""
    ax, ay, az = (np.radians(a) for a in euler_deg)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return _as_float64(rz @ ry @ rx)


def _as_float64(arr: NDArrayFloatType) -> NDArrayFloatType:
    """Return ``arr`` as a contiguous float64 array (typing helper)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _orient_faces_outward(vertices: NDArrayFloatType, faces: NDArrayIntType) -> NDArrayIntType:
    """Flip any face whose normal points inward, so all normals point outward.

    For a star-shaped mesh the outward direction at a face is its centroid
    direction from the origin; a face whose geometric normal opposes that is
    rewound.
    """
    oriented = faces.copy()
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    inward = np.sum(normals * centroids, axis=1) < 0.0
    oriented[inward] = oriented[inward][:, ::-1]
    return oriented


# The irregular shape's relief: a random sum of real spherical harmonics of
# degrees 2 to 5 (index = degree).  Degrees 0 and 1 are zero by construction:
# degree 0 would only resize the body and degree 1 would only move it, and
# neither is irregularity.  The per-degree RMS shares match those of the
# (latitude, longitude) Fourier relief this replaced, measured over 100 seeds
# (#626), so the body keeps its balance of elongation (degree 2) and lopsided
# lumps (degrees 3 to 5).
_BASE_DEGREE_RMS = np.array([0.0, 0.0, 0.082, 0.092, 0.072, 0.040])
# Lowest degree that counts as non-ellipsoidal: degree 2 is an ellipsoid-like
# distortion, degrees 3 and up are what a best-fit ellipsoid leaves behind.
_NON_ELLIPSOIDAL_DEGREE = 3
# The relief's non-ellipsoidal (degree >= 3) RMS per unit lumpiness, as a
# fraction of the unit radius: 1/8, the median of the Fourier relief it
# replaced, so a scene's lumpiness keeps its irregularity (#626).
LUMPINESS_RMS_PER_UNIT = 0.125
# Radius floor, so an extreme lumpiness cannot fold the mesh through its center.
_MIN_RADIUS = 0.2


def _octave_degrees(octave: int) -> tuple[int, int]:
    """The inclusive harmonic-degree band of detail octave ``octave`` (>= 1)."""
    top = _BASE_DEGREE_RMS.size - 1
    return top * 2 ** (octave - 1) + 1, top * 2**octave


def _relief_coefficients(seed: int, detail_octaves: int) -> NDArrayComplexType:
    """Draw the relief's harmonic coefficients for one mesh seed.

    The base degrees are drawn with :data:`_BASE_DEGREE_RMS` as their expected
    per-degree RMS, then the whole draw is scaled so its degree >= 3 RMS is
    exactly :data:`LUMPINESS_RMS_PER_UNIT`.  Each detail octave is drawn with
    equal expected power in every degree of its band and scaled to an RMS of
    ``0.5**k`` times the base's degree >= 3 RMS.  Octaves draw from the
    generator after the base, so adding octaves never changes the base shape.

    Parameters:
        seed: The mesh seed.
        detail_octaves: Number of detail octaves (0 = base only).

    Returns:
        The complex coefficient array of :mod:`spindoctor.sim.spherical_harmonics`,
        sized for the highest degree drawn.
    """
    rng = np.random.default_rng(seed)
    base_sigma = _BASE_DEGREE_RMS * np.sqrt(
        4.0 * np.pi / (2.0 * np.arange(_BASE_DEGREE_RMS.size) + 1.0)
    )
    base = draw_sh_coefficients(rng, base_sigma)
    base_ge3 = math.sqrt(float(degree_mean_square(base)[_NON_ELLIPSOIDAL_DEGREE:].sum()))
    banks = [base * (LUMPINESS_RMS_PER_UNIT / base_ge3)]
    for octave in range(1, max(int(detail_octaves), 0) + 1):
        low, high = _octave_degrees(octave)
        degrees = np.arange(high + 1)
        sigma = np.where(degrees >= low, np.sqrt(4.0 * np.pi / (2.0 * degrees + 1.0)), 0.0)
        bank = draw_sh_coefficients(rng, sigma)
        rms = math.sqrt(float(degree_mean_square(bank).sum()))
        banks.append(bank * (0.5**octave * LUMPINESS_RMS_PER_UNIT / rms))
    size = banks[-1].shape[0]
    coeffs = np.zeros((size, size), dtype=np.complex128)
    for bank in banks:
        coeffs[: bank.shape[0], : bank.shape[0]] += bank
    return coeffs


def make_irregular_mesh(
    *,
    n_lat: int = 16,
    n_lon: int = 32,
    lumpiness: float = 0.3,
    seed: int = 0,
    detail_octaves: int = 0,
) -> Mesh:
    """Build a lumpy unit-radius mesh (a UV sphere with low-degree relief).

    The vertex radius is ``1 + lumpiness * h(u)``, floored at 0.2, where ``u`` is
    the vertex's unit direction and ``h`` is a random sum of real spherical
    harmonics of degrees 2 to 5.  Because ``h`` is a function on the sphere, the
    relief is smooth everywhere, poles included, and independent of the mesh's
    own (latitude, longitude) parametrization: refining ``n_lat`` / ``n_lon``
    samples the same surface more finely.  ``h`` has no degree 0 or 1 content,
    so the relief neither resizes the body (its mean radius stays 1) nor moves
    it (its first-order center shift is zero).  Each seed's draw is scaled so
    the non-ellipsoidal part (degrees 3 and up) has an RMS of exactly
    ``lumpiness / 8`` of the radius; the degree 2 part, an ellipsoid-like
    elongation, varies with the seed around two thirds of that.  The same seed
    always yields the same mesh, and ``lumpiness`` 0 gives the unit sphere.

    ``detail_octaves`` adds bands of higher degrees with geometric amplitude
    falloff: octave ``k`` covers degrees ``5 * 2**(k-1) + 1`` to ``5 * 2**k``
    at an RMS of ``0.5**k`` times the base's non-ellipsoidal RMS.  Octave draws
    consume the generator after the base draws, so ``detail_octaves = 0``
    reproduces the base mesh and raising the count never changes the base
    shape.  The mesh resolution must support the frequency content: ``n_lon``
    should exceed roughly four samples per cycle of the top degree
    (``20 * 2**detail_octaves``), and ``n_lat`` half that, or the extra
    degrees alias.

    Parameters:
        n_lat: Number of latitude bands (>= 2).
        n_lon: Number of longitude divisions (>= 3).
        lumpiness: Relief amplitude; the non-ellipsoidal RMS is ``lumpiness / 8``
            of the unit radius.
        seed: Seed selecting the relief.
        detail_octaves: Number of higher-degree detail bands (0 = base only).

    Returns:
        An outward-wound :class:`Mesh`.
    """
    # Unit directions of the ring vertices (row-major by ring, then longitude),
    # then the north and south poles.  The relief is evaluated on these
    # directions rather than on (theta, phi): a Fourier mode in the mesh's own
    # angles takes every value around a pole as it approaches it, which pinches
    # the poles into pits or spikes that refining the mesh only sharpens (#626).
    theta = np.pi * np.arange(1, n_lat) / n_lat
    phi = 2.0 * np.pi * np.arange(n_lon) / n_lon
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    ring_dirs = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    unit_dirs = np.concatenate([ring_dirs, poles])
    relief = evaluate_at_directions(_relief_coefficients(seed, detail_octaves), unit_dirs)
    radius = np.maximum(1.0 + lumpiness * relief, _MIN_RADIUS)
    vertices = _as_float64(unit_dirs * radius[:, None])
    north_idx = len(ring_dirs)
    south_idx = north_idx + 1

    faces: list[tuple[int, int, int]] = []

    def ring_vertex(ring: int, col: int) -> int:
        return ring * n_lon + (col % n_lon)

    for ring in range(n_lat - 2):
        for col in range(n_lon):
            a = ring_vertex(ring, col)
            b = ring_vertex(ring, col + 1)
            c = ring_vertex(ring + 1, col)
            d = ring_vertex(ring + 1, col + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))
    for col in range(n_lon):
        faces.append((north_idx, ring_vertex(0, col), ring_vertex(0, col + 1)))
        faces.append((south_idx, ring_vertex(n_lat - 2, col + 1), ring_vertex(n_lat - 2, col)))

    face_arr = np.array(faces, dtype=np.int64)
    return Mesh(vertices=vertices, faces=_orient_faces_outward(vertices, face_arr))


def _light_direction(illumination_angle: float, phase_angle: float) -> NDArrayFloatType:
    """Unit light direction in camera (u, v, depth) coordinates.

    Matches the convention in
    ``spindoctor.sim.ellipsoid_geometry.lambert_from_normals``: v increases
    downward, depth is positive toward the observer; phase 0 lights the visible
    face, phase pi back-lights it.
    """
    illum_u = np.sin(illumination_angle) * np.sin(phase_angle)
    illum_v = -np.cos(illumination_angle) * np.sin(phase_angle)
    illum_z = np.cos(phase_angle)
    light = np.array([illum_u, illum_v, illum_z], dtype=np.float64)
    norm = float(np.linalg.norm(light))
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _as_float64(light / norm)


def render_polyhedral_body(
    *,
    size: tuple[int, int],
    center: tuple[float, float],
    mesh: Mesh,
    semi_axes_px: tuple[float, float, float],
    pose_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    anti_aliasing: float = 1.0,
    shading: str = 'flat',
) -> NDArrayFloatType:
    """Render a mesh body to a [0, 1] shaded silhouette.

    Two shading modes share one rasterization (the same front-face set, the
    same z-buffer, the same silhouette): ``'flat'`` shades each face by its
    own normal, ``'gouraud'`` computes per-vertex normals as the
    area-weighted average of the adjacent face normals and interpolates the
    per-vertex intensities barycentrically across each face, removing the
    facet-boundary shading discontinuities.  The mode never moves the
    silhouette; each caller picks its own mode (the navigator's predicted
    mesh keeps flat shading unless told otherwise).

    Parameters:
        size: ``(size_v, size_u)`` output image size in pixels.
        center: ``(v, u)`` body center in pixels.
        mesh: The unit-radius body mesh.
        semi_axes_px: Per-axis ``(a, b, c)`` half-sizes in pixels applied to the
            mesh in its body frame before the pose rotation.
        pose_euler_deg: Body orientation as intrinsic X, Y, Z Euler angles, deg.
        illumination_angle: Image-plane light azimuth in radians (0 = top).
        phase_angle: Phase angle in radians (0 = fully lit, pi = back-lit).
        anti_aliasing: 0 disables supersampling; (0, 1] supersamples the limb.
        shading: ``'flat'`` (per-face) or ``'gouraud'`` (per-vertex
            interpolated).

    Returns:
        A ``(size_v, size_u)`` float array in [0, 1].

    Raises:
        ValueError: On an unknown shading mode.
    """
    if shading not in ('flat', 'gouraud'):
        raise ValueError(f"shading must be 'flat' or 'gouraud'; got {shading!r}")
    size_v, size_u = size
    aa_scale = int(1 + 3 * anti_aliasing) if anti_aliasing > 0 else 1
    height = size_v * aa_scale
    width = size_u * aa_scale

    # Match the ellipsoid renderer's axis convention: axis1 (semi_axes_px[0])
    # sets the vertical (v) extent and axis2 the horizontal (u) extent.  The
    # projection below sends body x -> u and body y -> v, so scale body x by
    # axis2 and body y by axis1.
    axis1_semi, axis2_semi, axis3_semi = np.array(semi_axes_px, dtype=np.float64) * aa_scale
    semi = np.array([axis2_semi, axis1_semi, axis3_semi])
    rot = _rotation_matrix(pose_euler_deg)
    verts_cam = (mesh.vertices * semi) @ rot.T
    px = center[1] * aa_scale + verts_cam[:, 0]
    py = center[0] * aa_scale + verts_cam[:, 1]
    depth = verts_cam[:, 2]

    p0 = verts_cam[mesh.faces[:, 0]]
    p1 = verts_cam[mesh.faces[:, 1]]
    p2 = verts_cam[mesh.faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norm_mag = np.linalg.norm(normals, axis=1)
    norm_mag = np.maximum(norm_mag, 1e-12)
    unit_normals = normals / norm_mag[:, None]
    light = _light_direction(illumination_angle, phase_angle)
    face_intensity = np.clip(unit_normals @ light, _DARK_SIDE_ILLUM, 1.0)

    vertex_intensity: NDArrayFloatType | None = None
    if shading == 'gouraud':
        # Per-vertex normals: the area-weighted average of the adjacent face
        # normals.  The raw cross products carry twice the face area as their
        # magnitude, so summing them per vertex IS the area weighting.
        vertex_normals = np.zeros_like(verts_cam)
        for corner in range(3):
            np.add.at(vertex_normals, mesh.faces[:, corner], normals)
        vn_mag = np.maximum(np.linalg.norm(vertex_normals, axis=1), 1e-12)
        vertex_normals = vertex_normals / vn_mag[:, None]
        vertex_intensity = np.clip(vertex_normals @ light, _DARK_SIDE_ILLUM, 1.0)

    front = (unit_normals[:, 2] > 0.0) & (norm_mag > 1e-9)

    z_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    intensity = np.zeros((height, width), dtype=np.float64)

    for face_idx in np.nonzero(front)[0]:
        tri = mesh.faces[face_idx]
        xs = px[tri]
        ys = py[tri]
        ds = depth[tri]
        min_x = max(int(np.floor(xs.min())), 0)
        max_x = min(int(np.ceil(xs.max())), width - 1)
        min_y = max(int(np.floor(ys.min())), 0)
        max_y = min(int(np.ceil(ys.max())), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(area) < 1e-9:
            continue
        # The face's box is named by array rows and columns and the projected
        # vertices are in the geometry layer's pixel corner coordinates, so the
        # half pixel goes on here.
        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1) + PIXEL_CENTER_TO_CORNER_PX,
            np.arange(min_y, max_y + 1) + PIXEL_CENTER_TO_CORNER_PX,
        )
        w0 = ((xs[1] - xs[0]) * (gy - ys[0]) - (ys[1] - ys[0]) * (gx - xs[0])) / area
        w1 = ((xs[2] - xs[1]) * (gy - ys[1]) - (ys[2] - ys[1]) * (gx - xs[1])) / area
        w2 = ((xs[0] - xs[2]) * (gy - ys[2]) - (ys[0] - ys[2]) * (gx - xs[2])) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        # Barycentric depth: weights are the opposite-vertex sub-triangle areas.
        depth_pix = w0 * ds[2] + w1 * ds[0] + w2 * ds[1]
        sub = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        win = inside & (depth_pix > sub)
        sub[win] = depth_pix[win]
        if vertex_intensity is None:
            intensity[min_y : max_y + 1, min_x : max_x + 1][win] = float(face_intensity[face_idx])
        else:
            # Gouraud: interpolate the vertex intensities with the same
            # barycentric weight-to-vertex pairing as the depth above.
            iv = vertex_intensity[tri]
            shade_pix = w0 * iv[2] + w1 * iv[0] + w2 * iv[1]
            intensity[min_y : max_y + 1, min_x : max_x + 1][win] = shade_pix[win]

    if aa_scale > 1:
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
    return _as_float64(np.clip(intensity, 0.0, 1.0))


@dataclass(frozen=True)
class MeshBodySpec:
    """The mesh shape and pose for an irregular body, read from scene params.

    Kept separate from the image so the same spec drives both the rendered data
    and the navigator's predicted silhouette (or, for a shape-mismatch fixture,
    two deliberately different specs).

    Parameters:
        lumpiness: Surface-relief amplitude; the non-ellipsoidal relief has an
            RMS of ``lumpiness / 8`` of the unit radius.
        n_lat: Mesh latitude bands.
        n_lon: Mesh longitude divisions.
        seed: Seed selecting which irregular shape is generated.
        pose_euler_deg: Body orientation as intrinsic X, Y, Z Euler angles, deg.
        detail_octaves: Bands of higher harmonic degrees added to the base
            relief (see :func:`make_irregular_mesh`); part of the published shape.
    """

    lumpiness: float = 0.3
    n_lat: int = 16
    n_lon: int = 32
    seed: int = 0
    pose_euler_deg: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    detail_octaves: int = 0


def mesh_spec_from_params(body_params: Mapping[str, Any]) -> MeshBodySpec:
    """Build a :class:`MeshBodySpec` from a body parameter mapping.

    The mesh seed and pose are explicit body parameters (not the scene's noise
    seed), so the same body params reproduce the same shape on both the render
    and prediction sides.

    Parameters:
        body_params: A body parameter mapping.

    Returns:
        The resolved :class:`MeshBodySpec`.

    Raises:
        ValueError: If ``pose_euler_deg`` does not hold exactly three angles.
    """
    pose = body_params.get('pose_euler_deg', (0.0, 0.0, 0.0))
    values = [float(x) for x in pose]
    if len(values) != 3:
        raise ValueError(f'pose_euler_deg must have 3 angles; got {pose!r}')
    return MeshBodySpec(
        lumpiness=float(body_params.get('mesh_lumpiness', 0.3)),
        n_lat=int(body_params.get('mesh_n_lat', 16)),
        n_lon=int(body_params.get('mesh_n_lon', 32)),
        seed=int(body_params.get('mesh_seed', 0)),
        pose_euler_deg=(values[0], values[1], values[2]),
        detail_octaves=int(body_params.get('mesh_detail_octaves', 0)),
    )


def render_mesh_body_image(
    *,
    size: tuple[int, int],
    center: tuple[float, float],
    semi_axes_px: tuple[float, float, float],
    spec: MeshBodySpec,
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    anti_aliasing: float = 1.0,
) -> NDArrayFloatType:
    """Render an irregular mesh body from a :class:`MeshBodySpec`.

    A thin convenience over ``make_irregular_mesh`` + ``render_polyhedral_body``
    so the render path and the navigator's prediction share one primitive.

    Parameters:
        size: ``(size_v, size_u)`` output image size in pixels.
        center: ``(v, u)`` body center in pixels.
        semi_axes_px: Per-axis ``(a, b, c)`` half-sizes in pixels.
        spec: The mesh shape and pose.
        illumination_angle: Image-plane light azimuth in radians.
        phase_angle: Phase angle in radians.
        anti_aliasing: Limb supersampling control.

    Returns:
        A ``(size_v, size_u)`` float array in [0, 1].
    """
    mesh = make_irregular_mesh(
        n_lat=spec.n_lat,
        n_lon=spec.n_lon,
        lumpiness=spec.lumpiness,
        seed=spec.seed,
        detail_octaves=spec.detail_octaves,
    )
    return render_polyhedral_body(
        size=size,
        center=center,
        mesh=mesh,
        semi_axes_px=semi_axes_px,
        pose_euler_deg=spec.pose_euler_deg,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
    )
