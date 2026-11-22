"""Shared polyhedral-mesh geometry for irregular sim bodies.

The ellipsoid renderers cannot produce the non-ellipsoidal silhouette of an
irregular body (Hyperion, Phoebe).  Because ``oops`` will not gain DSK
support, the sim carries its own small renderer that projects a triangle mesh
through a scene-supplied pose and rasterizes the shaded silhouette.  It is
sim-only: the body's orientation is ground truth from the scene, not from
SPICE.

This module is deliberately shared between the image-side renderer
(``spindoctor.sim.render.bodies.mesh``) and the navigator-side predicted-body
renderer (``spindoctor.nav_model.simulated.body``): the mesh shape,
pose, and rasterization conventions are idealized information both sides may
know, and sharing one implementation guarantees that a scene's planted
geometry error (via ``nav_override``) is the only difference between the
rendered and the predicted silhouette.  Every function takes explicit
geometry arguments; none reads the scene mapping, so no truth-side
information can cross the boundary here (``mesh_spec_from_params`` parses a
body parameter mapping, but only its idealized mesh keys).

The output contract matches the ellipsoid renderers (``create_ellipsoid_body`` on
the image side, ``render_lambert_ellipsoid`` on the navigator side): a
``(size_v, size_u)`` float array in [0, 1], lit by the same Lambertian convention so
a mesh body and an ellipsoid body can be compared directly.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spindoctor.sim.scene.fields import scene_value
from spindoctor.sim.shared.ellipsoid import DARK_SIDE_ILLUM_STRENGTH
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.sim.shared.spherical_harmonics import (
    degree_mean_square,
    draw_sh_coefficients,
    evaluate_at_directions,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_matmul3
from spindoctor.support.types import NDArrayComplexType, NDArrayFloatType, NDArrayIntType

# Lambertian floor for the visible-but-unlit side, shared with the ellipsoid
# shading convention in spindoctor.sim.shared.ellipsoid so the two paths
# cannot desync.
_DARK_SIDE_ILLUM = DARK_SIDE_ILLUM_STRENGTH


@dataclass(frozen=True)
class PolyhedralBodyRender:
    """One mesh body render: its shaded radiance and its silhouette.

    Parameters:
        intensity: The shaded radiance in [0, 1], 0 outside the body.
        silhouette: The fraction of each pixel the body covers, in [0, 1].
    """

    intensity: NDArrayFloatType
    silhouette: NDArrayFloatType


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
    return stable_matmul3(stable_matmul3(rz, ry), rx)


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
        The complex coefficient array of :mod:`spindoctor.sim.shared.spherical_harmonics`,
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


# Most candidate pixels one rasterization batch tests.  This bounds only the
# per-candidate arrays (about twenty of this length, near 10 MB); larger batches
# measured no faster.  The rest of the working memory is not bounded by it: the
# per-(face, box row) arrays take about 130 bytes per row, so they grow with the total
# height of all the face boxes (the mesh's face count times their size on the grid),
# and the whole-grid buffers (depth, shade, and the int32 tie-break owner) take 20
# bytes per grid pixel.
_RASTER_BATCH_PIXELS = 2**16
# Sentinel for "no face yet" in the per-pixel tie-break buffer.
_NO_FACE = np.iinfo(np.int32).max


def _rasterize_faces(
    *,
    px: NDArrayFloatType,
    py: NDArrayFloatType,
    depth: NDArrayFloatType,
    faces: NDArrayIntType,
    face_intensity: NDArrayFloatType,
    vertex_intensity: NDArrayFloatType | None,
    grid_size: tuple[int, int],
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Z-buffer and shade a list of projected triangles.

    A face covers the pixels of its bounding box (clipped to the grid) whose centers
    have all three barycentric weights >= 0, and a face whose projected area is below
    1e-9 square pixels covers nothing.  At each pixel the covering face with the
    greatest interpolated depth wins, and among faces at exactly that depth the one
    listed first wins: the result of drawing the faces one at a time in list order
    with a strict ``>`` depth test.

    Parameters:
        px: ``(N,)`` vertex column coordinates in pixel-corner units.
        py: ``(N,)`` vertex row coordinates in pixel-corner units.
        depth: ``(N,)`` vertex depths, larger nearer the observer.
        faces: ``(M, 3)`` vertex indices of the faces to draw, in priority order.
        face_intensity: ``(M,)`` flat shade of each face.
        vertex_intensity: ``(N,)`` per-vertex shade interpolated across each face
            (Gouraud), or None to shade each face with ``face_intensity``.
        grid_size: ``(height, width)`` of the grid.

    Returns:
        The ``(height, width)`` depth buffer (``-inf`` where no face covers a pixel)
        and the shade buffer (0 there).
    """
    height, width = grid_size
    z_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    intensity = np.zeros((height, width), dtype=np.float64)
    xs = px[faces]
    ys = py[faces]
    ds = depth[faces]

    # Clip the boxes in floating point, before any integer conversion, so a vertex
    # far off the grid cannot overflow the cast.
    box_x0 = np.maximum(np.floor(xs.min(axis=1)), 0.0)
    box_x1 = np.minimum(np.ceil(xs.max(axis=1)), float(width - 1))
    box_y0 = np.maximum(np.floor(ys.min(axis=1)), 0.0)
    box_y1 = np.minimum(np.ceil(ys.max(axis=1)), float(height - 1))
    x0, x1, x2 = xs[:, 0], xs[:, 1], xs[:, 2]
    y0, y1, y2 = ys[:, 0], ys[:, 1], ys[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    drawn = (box_x0 <= box_x1) & (box_y0 <= box_y1) & (np.abs(area) >= 1e-9)
    keep = np.nonzero(drawn)[0]
    if keep.size == 0:
        return z_buffer, intensity

    # Per-face quantities, kept faces only, in priority order.  The weight of each
    # vertex is the edge function of the opposite edge over the area, written with
    # exactly the operations and operand order of the one-face formula so every
    # weight, depth, and shade is bit-identical to drawing that face alone.
    x0, x1, x2 = x0[keep], x1[keep], x2[keep]
    y0, y1, y2 = y0[keep], y1[keep], y2[keep]
    d0, d1, d2 = ds[keep, 0], ds[keep, 1], ds[keep, 2]
    area = area[keep]
    e0x, e0y = x1 - x0, y1 - y0
    e1x, e1y = x2 - x1, y2 - y1
    e2x, e2y = x0 - x2, y0 - y2
    col0 = box_x0[keep].astype(np.int64)
    row0 = box_y0[keep].astype(np.int64)
    n_cols = box_x1[keep].astype(np.int64) - col0 + 1
    n_rows = box_y1[keep].astype(np.int64) - row0 + 1
    flat_shade = face_intensity[keep]
    vertex_shades = None if vertex_intensity is None else vertex_intensity[faces[keep]]

    # One entry per (face, box row), carrying what the weight tests need.  Along a row
    # the pixel-center v is constant, so each weight's first product is computed once
    # per row.  The boxes are named by array rows and columns and the projected
    # vertices are in the geometry layer's pixel corner coordinates, so the half pixel
    # goes on to the row and column indices.
    row_face = np.repeat(np.arange(keep.size, dtype=np.int32), n_rows)
    row_first = np.cumsum(n_rows) - n_rows
    row_v = row0[row_face] + (np.arange(row_face.size) - np.repeat(row_first, n_rows))
    gy = row_v + PIXEL_CENTER_TO_CORNER_PX
    row_a0 = e0x[row_face] * (gy - y0[row_face])
    row_a1 = e1x[row_face] * (gy - y1[row_face])
    row_a2 = e2x[row_face] * (gy - y2[row_face])
    row_e0y, row_e1y, row_e2y = e0y[row_face], e1y[row_face], e2y[row_face]
    row_x0, row_x1, row_x2 = x0[row_face], x1[row_face], x2[row_face]
    row_area = area[row_face]
    row_col0 = col0[row_face]
    row_len = n_cols[row_face]
    row_end = np.cumsum(row_len)

    z_flat = z_buffer.reshape(-1)
    shade_flat = intensity.reshape(-1)
    owner = np.full(height * width, _NO_FACE, dtype=np.int32)
    # The rows are drawn in batches, in face order.  The winners within a batch are
    # resolved together, and a batch takes a pixel only from an earlier batch that
    # holds it at a strictly smaller depth, which keeps the one-at-a-time rule exact.
    start = 0
    while start < row_face.size:
        done = int(row_end[start - 1]) if start > 0 else 0
        stop = int(np.searchsorted(row_end, done + _RASTER_BATCH_PIXELS, side='right'))
        stop = max(stop, start + 1)
        rows = slice(start, stop)
        lens = row_len[rows]
        start = stop

        # Test the weights one at a time, dropping the pixels each one rejects.  The
        # first test runs on every box pixel, so its row values are expanded by
        # repetition rather than gathered.
        cand_row = np.repeat(np.arange(rows.start, rows.stop), lens)
        cand_u = np.repeat(row_col0[rows] - (np.cumsum(lens) - lens), lens)
        cand_u += np.arange(cand_u.size)
        gx = cand_u + PIXEL_CENTER_TO_CORNER_PX
        w0 = (
            np.repeat(row_a0[rows], lens)
            - np.repeat(row_e0y[rows], lens) * (gx - np.repeat(row_x0[rows], lens))
        ) / np.repeat(row_area[rows], lens)
        sel = np.nonzero(w0 >= 0)[0]
        cand_row, cand_u, w0 = cand_row[sel], cand_u[sel], w0[sel]
        gx = cand_u + PIXEL_CENTER_TO_CORNER_PX
        w1 = (row_a1[cand_row] - row_e1y[cand_row] * (gx - row_x1[cand_row])) / row_area[cand_row]
        sel = np.nonzero(w1 >= 0)[0]
        cand_row, cand_u, w0, w1 = cand_row[sel], cand_u[sel], w0[sel], w1[sel]
        gx = cand_u + PIXEL_CENTER_TO_CORNER_PX
        w2 = (row_a2[cand_row] - row_e2y[cand_row] * (gx - row_x2[cand_row])) / row_area[cand_row]
        sel = np.nonzero(w2 >= 0)[0]
        cand_row, w0, w1, w2 = cand_row[sel], w0[sel], w1[sel], w2[sel]
        cand_face = row_face[cand_row]
        cand_pix = row_v[cand_row] * width + cand_u[sel]
        cand_depth = w0 * d2[cand_face] + w1 * d0[cand_face] + w2 * d1[cand_face]

        # A candidate must beat what the earlier batches left at its pixel; within the
        # batch the greatest depth wins, and an exact depth tie goes to the face
        # listed first.
        sel = np.nonzero(cand_depth > z_flat[cand_pix])[0]
        cand_face, cand_pix, cand_depth = cand_face[sel], cand_pix[sel], cand_depth[sel]
        w0, w1, w2 = w0[sel], w1[sel], w2[sel]
        np.maximum.at(z_flat, cand_pix, cand_depth)
        sel = np.nonzero(cand_depth == z_flat[cand_pix])[0]
        cand_face, cand_pix, cand_depth = cand_face[sel], cand_pix[sel], cand_depth[sel]
        w0, w1, w2 = w0[sel], w1[sel], w2[sel]
        np.minimum.at(owner, cand_pix, cand_face)
        sel = np.nonzero(cand_face == owner[cand_pix])[0]
        owner[cand_pix] = _NO_FACE
        cand_face, cand_pix = cand_face[sel], cand_pix[sel]
        # Store the winner's own depth: of two tied zeros of opposite sign,
        # np.maximum.at may have kept the other one.  Only the depth buffer can see
        # this, never the shade, but it keeps that buffer bit-identical to drawing
        # the faces one at a time.
        z_flat[cand_pix] = cand_depth[sel]
        if vertex_shades is None:
            shade_flat[cand_pix] = flat_shade[cand_face]
        else:
            shade_flat[cand_pix] = (
                w0[sel] * vertex_shades[cand_face, 2]
                + w1[sel] * vertex_shades[cand_face, 0]
                + w2[sel] * vertex_shades[cand_face, 1]
            )
    return z_buffer, intensity


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

    The intensity of :func:`render_polyhedral_body_with_silhouette`, which documents
    the parameters, the shading modes, and the errors.

    Returns:
        A ``(size_v, size_u)`` float array in [0, 1].
    """
    return render_polyhedral_body_with_silhouette(
        size=size,
        center=center,
        mesh=mesh,
        semi_axes_px=semi_axes_px,
        pose_euler_deg=pose_euler_deg,
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
        anti_aliasing=anti_aliasing,
        shading=shading,
    ).intensity


def render_polyhedral_body_with_silhouette(
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
) -> PolyhedralBodyRender:
    """Render a mesh body's shaded radiance and the fraction of each pixel it covers.

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
        The :class:`PolyhedralBodyRender`, both arrays ``(size_v, size_u)``.  The
        silhouette is the fraction of each pixel's supersamples some face covers, so
        a pixel inside the body is exactly 1 whatever its shade.

    Raises:
        ValueError: On an unknown shading mode, or a center that is not finite.
    """
    if shading not in ('flat', 'gouraud'):
        raise ValueError(f"shading must be 'flat' or 'gouraud'; got {shading!r}")
    # A non-finite center would project every face off the grid and render a blank
    # body without complaint.
    if not (math.isfinite(center[0]) and math.isfinite(center[1])):
        raise ValueError(f'center must be finite; got {center!r}')
    size_v, size_u = size
    aa_scale = int(1 + 3 * anti_aliasing) if anti_aliasing > 0 else 1
    height = size_v * aa_scale
    width = size_u * aa_scale

    # Match the ellipsoid renderer's axis convention: semi-axis a (semi_axes_px[0])
    # sets the vertical (v) extent and semi-axis b the horizontal (u) extent.  The
    # projection below sends body x -> u and body y -> v, so scale body x by b and
    # body y by a.
    semi_a, semi_b, semi_c = np.array(semi_axes_px, dtype=np.float64) * aa_scale
    semi = np.array([semi_b, semi_a, semi_c])
    rot = _rotation_matrix(pose_euler_deg)
    verts_cam = stable_matmul3(mesh.vertices * semi, rot.T)
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
    # The shared light direction is ordered (v, u, depth); the camera frame here is
    # (u, v, depth).
    light_v, light_u, light_z = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )
    light = np.array([light_u, light_v, light_z], dtype=np.float64)
    face_intensity = np.clip(stable_matmul3(unit_normals, light), _DARK_SIDE_ILLUM, 1.0)

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
        vertex_intensity = np.clip(stable_matmul3(vertex_normals, light), _DARK_SIDE_ILLUM, 1.0)

    front = (unit_normals[:, 2] > 0.0) & (norm_mag > 1e-9)
    front_idx = np.nonzero(front)[0]
    z_buffer, intensity = _rasterize_faces(
        px=px,
        py=py,
        depth=depth,
        faces=mesh.faces[front_idx],
        face_intensity=face_intensity[front_idx],
        vertex_intensity=vertex_intensity,
        grid_size=(height, width),
    )
    silhouette = np.isfinite(z_buffer).astype(np.float64)

    if aa_scale > 1:
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
        silhouette = silhouette.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))
    return PolyhedralBodyRender(
        intensity=_as_float64(np.clip(intensity, 0.0, 1.0)), silhouette=silhouette
    )


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
    and prediction sides.  A key that is absent or null takes its default.

    Parameters:
        body_params: A body parameter mapping.

    Returns:
        The resolved :class:`MeshBodySpec`.

    Raises:
        ValueError: If ``pose_euler_deg`` does not hold exactly three angles.
    """
    pose = scene_value(body_params, 'pose_euler_deg', (0.0, 0.0, 0.0))
    values = [float(x) for x in pose]
    if len(values) != 3:
        raise ValueError(f'pose_euler_deg must have 3 angles; got {pose!r}')
    return MeshBodySpec(
        lumpiness=float(scene_value(body_params, 'mesh_lumpiness', 0.3)),
        n_lat=int(scene_value(body_params, 'mesh_n_lat', 16)),
        n_lon=int(scene_value(body_params, 'mesh_n_lon', 32)),
        seed=int(scene_value(body_params, 'mesh_seed', 0)),
        pose_euler_deg=(values[0], values[1], values[2]),
        detail_octaves=int(scene_value(body_params, 'mesh_detail_octaves', 0)),
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
