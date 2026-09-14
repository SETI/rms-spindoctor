"""Shared polyhedral-mesh geometry for irregular sim bodies.

The ellipsoid renderers cannot produce the non-ellipsoidal silhouette of an
irregular body (Hyperion, Phoebe).  Because ``oops`` will not gain DSK
support, the sim carries its own small renderer that projects a triangle mesh
through a scene-supplied pose and rasterises the shaded silhouette.  It is
sim-only: the body's orientation is ground truth from the scene, not from
SPICE.

This module is deliberately shared between the image-side forward renderer
(``spindoctor.sim.forward.body_mesh``) and the navigator-side predicted-body
renderer (``spindoctor.nav_model.nav_model_body_simulated``): the mesh shape,
pose, and rasterisation conventions are idealized information both sides may
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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.sim.ellipsoid_geometry import DARK_SIDE_ILLUM_STRENGTH
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

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


def make_irregular_mesh(
    *,
    n_lat: int = 16,
    n_lon: int = 32,
    lumpiness: float = 0.3,
    n_modes: int = 6,
    seed: int = 0,
    detail_octaves: int = 0,
) -> Mesh:
    """Build a lumpy unit-radius mesh (a UV sphere with low-frequency relief).

    The radius is modulated by a few random low-frequency angular modes, so the
    body is smoothly irregular rather than spiky.  The same seed always yields
    the same mesh.

    ``detail_octaves`` adds banks of higher-frequency modes with geometric
    amplitude falloff: octave ``k`` draws ``n_modes`` fresh modes with angular
    wavenumbers in ``[3 * 2**(k-1) + 1, 3 * 2**k]`` (the base modes occupy
    1..3) at amplitude ``0.5**k`` relative to the base bank.  Octave draws
    consume the generator after the base draws, so ``detail_octaves = 0``
    reproduces the base mesh bit-exactly and raising the count never changes
    the base shape.  The mesh resolution must support the frequency content:
    ``n_lat`` / ``n_lon`` should exceed roughly four samples per top-octave
    cycle (``12 * 2**detail_octaves``) or the extra modes alias.

    Parameters:
        n_lat: Number of latitude bands (>= 2).
        n_lon: Number of longitude divisions (>= 3).
        lumpiness: Relief amplitude as a fraction of the unit radius.
        n_modes: Number of random angular modes per bank.
        seed: Seed for the relief modes.
        detail_octaves: Number of higher-frequency mode banks (0 = base only).

    Returns:
        An outward-wound :class:`Mesh`.
    """
    rng = np.random.default_rng(seed)
    banks: list[tuple[NDArrayFloatType, NDArrayIntType, NDArrayIntType, NDArrayFloatType, float]]
    banks = []
    amps = rng.uniform(-1.0, 1.0, size=n_modes)
    m_lat = rng.integers(1, 4, size=n_modes)
    m_lon = rng.integers(1, 4, size=n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
    banks.append((amps, m_lat, m_lon, phases, 1.0))
    for octave in range(1, max(int(detail_octaves), 0) + 1):
        octave_amps = rng.uniform(-1.0, 1.0, size=n_modes)
        low = 3 * 2 ** (octave - 1) + 1
        high = 3 * 2**octave + 1
        octave_m_lat = rng.integers(low, high, size=n_modes)
        octave_m_lon = rng.integers(low, high, size=n_modes)
        octave_phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        banks.append((octave_amps, octave_m_lat, octave_m_lon, octave_phases, 0.5**octave))

    def radius(theta: float, phi: float) -> float:
        relief = 0.0
        for bank_amps, bank_m_lat, bank_m_lon, bank_phases, falloff in banks:
            relief += falloff * sum(
                bank_amps[k] * np.cos(bank_m_lat[k] * theta + bank_m_lon[k] * phi + bank_phases[k])
                for k in range(n_modes)
            )
        # relief spans about [-n_modes, n_modes]; normalise so lumpiness is the
        # fractional relief amplitude.
        return float(max(1.0 + lumpiness * relief / max(n_modes, 1), 0.2))

    verts: list[tuple[float, float, float]] = []
    ring_lats = [np.pi * i / n_lat for i in range(1, n_lat)]
    for theta in ring_lats:
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            r = radius(theta, phi)
            verts.append(
                (
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                )
            )
    north_idx = len(verts)
    rn = radius(0.0, 0.0)
    verts.append((0.0, 0.0, rn))
    south_idx = len(verts)
    rs = radius(np.pi, 0.0)
    verts.append((0.0, 0.0, -rs))

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

    vertices = np.array(verts, dtype=np.float64)
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
        center: ``(v, u)`` body centre in pixels.
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
        lumpiness: Surface-relief amplitude as a fraction of the unit radius.
        n_lat: Mesh latitude bands.
        n_lon: Mesh longitude divisions.
        seed: Seed selecting which irregular shape is generated.
        pose_euler_deg: Body orientation as intrinsic X, Y, Z Euler angles, deg.
        detail_octaves: Higher-frequency mode banks added to the base relief
            (see :func:`make_irregular_mesh`); part of the published shape.
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
        center: ``(v, u)`` body centre in pixels.
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
