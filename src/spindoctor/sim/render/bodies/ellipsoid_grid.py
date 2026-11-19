"""The pixel grids the ellipsoid body renderer works on.

The renderer (:mod:`spindoctor.sim.render.bodies.ellipsoid`) lays a body out on
two grids: the *working grid* it draws the silhouette and shades on (the
oversampled grid, or the anti-aliasing supersample grid at oversample 1), and the
*detector grid*, one point per detector pixel, where surface detail is computed
before it is interpolated onto the working grid.  This module builds those grids,
crops them to the body, carries values from one to the other, and marks the
regions the renderer treats specially: the silhouette, the band along the limb,
and the cells along the day-night edge.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from spindoctor.sim.render.bodies.relief import ReliefField
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType

__all__ = [
    'BodyGrid',
    'bilinear_upsample',
    'body_box',
    'body_grid',
    'crop_grid',
    'cubic_upsample',
    'day_night_edge_cells',
    'expand_to_working_grid',
    'limb_band',
    'neighboring_cells',
    'silhouette_mask',
]

# Cells by which the day-night edge of a supersampled relief body is widened before
# its relief normals and shadows are computed on the working grid there.  Off the
# edge, interpolated tilts have less variance than the terrain, and the night-side
# clamp turns that into a darkening.  With the tilt interpolated by a cubic spline
# the margin matters little: on five test bodies it keeps the light's center
# within 0.0009 px of shading wholly on the working grid, against 0.0014 px with no
# margin, and it costs little.
_DAY_NIGHT_MARGIN_CELLS = 3
# Detector cells kept around the silhouette when a body is cropped to its box; one
# cell covers the bilinear interpolation's reach and one more the outline's.
_BOX_MARGIN_CELLS = 2


@dataclass
class BodyGrid:
    """One resolution level of the body geometry.

    Parameters:
        v_ctr: Centered (unrotated) v coordinate of each pixel.
        u_ctr: Centered (unrotated) u coordinate of each pixel.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        e2: Squared normalized ellipse radial function ``e(p)**2``.
        center: Body center (v, u) in this grid's pixels.
        semi_a: Semi-axis a at this grid's resolution, pixels.
        semi_b: Semi-axis b at this grid's resolution, pixels.
        semi_c: Depth semi-axis c at this grid's resolution, pixels.
    """

    v_ctr: NDArrayFloatType
    u_ctr: NDArrayFloatType
    v_rot: NDArrayFloatType
    u_rot: NDArrayFloatType
    e2: NDArrayFloatType
    center: tuple[float, float]
    semi_a: float
    semi_b: float
    semi_c: float


def body_grid(
    dims: tuple[int, int],
    center: tuple[float, float],
    semi_axes: tuple[float, float, float],
    *,
    cos_rz: float,
    sin_rz: float,
    cos_rt: float,
) -> BodyGrid:
    """Build the rotated-frame coordinates and radial function for one grid.

    The conventions (the grid built in pixel corner coordinates, rotation then
    tilt compression) match :func:`~spindoctor.sim.shared.ellipsoid.project_ellipsoid`
    exactly.

    Parameters:
        dims: Grid dimensions (rows, columns).
        center: Body center (v, u) in this grid's pixels.
        semi_axes: Semi-axes (a, b, c) in this grid's pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        cos_rt: Cosine of the tilt.

    Returns:
        The populated :class:`BodyGrid`.
    """
    size_v, size_u = dims
    semi_a, semi_b, semi_c = semi_axes
    # The grid starts as array rows and columns and the body center is stated
    # in pixel corner coordinates, so the half pixel goes on before the center
    # comes off.
    v_coords, u_coords = np.mgrid[0:size_v, 0:size_u].astype(float)
    v_coords += PIXEL_CENTER_TO_CORNER_PX - center[0]
    u_coords += PIXEL_CENTER_TO_CORNER_PX - center[1]
    v_rot1 = v_coords * cos_rz - u_coords * sin_rz
    u_rot1 = v_coords * sin_rz + u_coords * cos_rz
    v_rot = v_rot1 * cos_rt
    e2 = (v_rot / semi_a) ** 2 + (u_rot1 / semi_b) ** 2
    return BodyGrid(
        v_ctr=v_coords,
        u_ctr=u_coords,
        v_rot=v_rot,
        u_rot=u_rot1,
        e2=e2,
        center=center,
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_c,
    )


def bilinear_upsample(field: NDArrayFloatType, factor: int) -> NDArrayFloatType:
    """Bilinearly upsample a field by an integer factor (pixel-center aligned).

    Output sample ``i`` lands at ``(i + PIXEL_CENTER_TO_CORNER_PX) / factor -
    PIXEL_CENTER_TO_CORNER_PX`` in the input field's pixel centric coordinates:
    the output row is lifted to pixel corner coordinates, rescaled to the input
    grid, and lowered again for the array lookup.  Borders clamp to the nearest
    input sample.

    Parameters:
        field: The 2-D input field.
        factor: Integer upsampling factor (1 returns the input unchanged).

    Returns:
        The ``(rows * factor, cols * factor)`` upsampled field.
    """
    if factor == 1:
        return field
    n_v, n_u = field.shape

    def _axis(n: int) -> tuple[NDArrayIntType, NDArrayIntType, NDArrayFloatType]:
        centers = (
            np.arange(n * factor, dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX
        ) / factor - PIXEL_CENTER_TO_CORNER_PX
        low = np.clip(np.floor(centers).astype(np.intp), 0, n - 1)
        high = np.minimum(low + 1, n - 1)
        weight = np.clip(centers - low, 0.0, 1.0)
        return low, high, weight

    row_lo, row_hi, row_w = _axis(n_v)
    col_lo, col_hi, col_w = _axis(n_u)
    rows = field[row_lo, :] * (1.0 - row_w)[:, np.newaxis] + field[row_hi, :] * row_w[:, np.newaxis]
    return rows[:, col_lo] * (1.0 - col_w) + rows[:, col_hi] * col_w


def cubic_upsample(field: NDArrayFloatType, factor: int) -> NDArrayFloatType:
    """Upsample a field by an integer factor with a cubic spline (pixel-center aligned).

    Output samples land where :func:`bilinear_upsample` puts them, and borders
    continue the nearest input sample.

    Parameters:
        field: The 2-D input field.
        factor: Integer upsampling factor (1 returns the input unchanged).

    Returns:
        The ``(rows * factor, cols * factor)`` upsampled field.
    """
    if factor == 1:
        return field
    return np.asarray(
        ndimage.zoom(field, factor, order=3, grid_mode=True, mode='nearest'),
        dtype=np.float64,
    )


def body_box(detector_grid: BodyGrid, field: ReliefField | None) -> tuple[slice, slice] | None:
    """The detector rows and columns the body's silhouette can touch, plus a margin.

    The margin of two cells keeps every interpolation that reaches the silhouette
    inside the box, so shading the box alone changes no pixel the silhouette
    covers.

    Parameters:
        detector_grid: The detector-grid geometry.
        field: The relief realization, or None for the smooth limb.

    Returns:
        The ``(rows, cols)`` slices, or None when the body misses the frame.
    """
    delta_cap = 0.0 if field is None else max(abs(field.h_min), abs(field.h_max))
    # The reach is two detector pixels beyond the outline whatever the body's size;
    # a body smaller than a pixel still reaches the cells its silhouette touches.
    reach = 1.0 + delta_cap + 2.0 / max(min(detector_grid.semi_a, detector_grid.semi_b), 1e-9)
    inside = detector_grid.e2 < reach * reach
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    if rows.size == 0:
        return None
    n_rows, n_cols = inside.shape
    margin = _BOX_MARGIN_CELLS
    return (
        slice(max(int(rows[0]) - margin, 0), min(int(rows[-1]) + margin + 1, n_rows)),
        slice(max(int(cols[0]) - margin, 0), min(int(cols[-1]) + margin + 1, n_cols)),
    )


def crop_grid(grid: BodyGrid, rows: slice, cols: slice) -> BodyGrid:
    """The part of a grid inside a box; its coordinates keep the body-centered origin.

    Parameters:
        grid: The geometry to crop.
        rows: The box's rows.
        cols: The box's columns.

    Returns:
        The cropped :class:`BodyGrid`, whose arrays are views into ``grid``'s.
    """
    return replace(
        grid,
        v_ctr=grid.v_ctr[rows, cols],
        u_ctr=grid.u_ctr[rows, cols],
        v_rot=grid.v_rot[rows, cols],
        u_rot=grid.u_rot[rows, cols],
        e2=grid.e2[rows, cols],
    )


def limb_band(grid: BodyGrid, field: ReliefField, *, width_px: float) -> NDArrayBoolType:
    """The pixels within ``width_px`` of the relief outline's reach on either side.

    Parameters:
        grid: The geometry to select on.
        field: The relief realization; its extremes bound how far the outline
            departs from the smooth limb.
        width_px: The band's half-width beyond the outline's reach, in ``grid``
            pixels.

    Returns:
        The band mask on ``grid``.  When the band's inner edge reaches the body
        center, the band is the whole disc out to its outer edge, the center
        included.
    """
    delta_cap = max(abs(field.h_min), abs(field.h_max))
    margin = width_px / max(min(grid.semi_a, grid.semi_b), 1e-9)
    low = max(0.0, 1.0 - delta_cap - margin)
    high = 1.0 + delta_cap + margin
    band = grid.e2 < high * high
    if low > 0.0:
        band &= grid.e2 > low * low
    return band


def day_night_edge_cells(lit: NDArrayBoolType) -> NDArrayBoolType:
    """The detector cells along the day-night edge, widened by a margin.

    A cell is on the edge when it or one of its eight neighbors lies on the other
    side; the edge is then widened by three cells all round.

    Parameters:
        lit: Detector-grid mask of the cells whose surface faces the sun.

    Returns:
        The edge cells on the detector grid.
    """
    square = np.ones((3, 3), dtype=np.bool_)
    edge = ndimage.binary_dilation(lit, structure=square) & ~ndimage.binary_erosion(
        lit, structure=square, border_value=1
    )
    if _DAY_NIGHT_MARGIN_CELLS == 0:
        # scipy reads zero iterations as "repeat until nothing changes".
        return np.asarray(edge, dtype=np.bool_)
    widened = ndimage.binary_dilation(edge, structure=square, iterations=_DAY_NIGHT_MARGIN_CELLS)
    return np.asarray(widened, dtype=np.bool_)


def neighboring_cells(cells: NDArrayBoolType) -> NDArrayBoolType:
    """The marked detector cells and their eight neighbors.

    Parameters:
        cells: A detector-grid mask.

    Returns:
        The mask widened by one cell all round.
    """
    if not cells.any():
        return cells.copy()
    square = np.ones((3, 3), dtype=np.bool_)
    return np.asarray(ndimage.binary_dilation(cells, structure=square), dtype=np.bool_)


def expand_to_working_grid(cells: NDArrayBoolType, up_factor: int) -> NDArrayBoolType:
    """Mark every working-grid pixel of the marked detector cells.

    Parameters:
        cells: A detector-grid mask.
        up_factor: Pixels of the working grid per detector pixel.

    Returns:
        The mask on the working grid.
    """
    return np.repeat(np.repeat(cells, up_factor, axis=0), up_factor, axis=1)


def silhouette_mask(grid: BodyGrid, field: ReliefField | None) -> NDArrayFloatType:
    """The body's silhouette on the working grid, relief-perturbed when on.

    With relief, ``e_adj = e / (1 + delta(theta))`` places the limb at
    ``r_ellipse * (1 + delta)``, with ``delta`` the relief surface's outline
    envelope (:meth:`~spindoctor.sim.render.bodies.relief.ReliefField.outline_delta`);
    the perturbation only matters in a band ``|e - 1| <= |delta|_max`` around
    the smooth limb, so azimuths and the outline are evaluated on that band
    only.

    The relief azimuth ``phi`` is the elliptical parametric angle in the
    body's rotated frame, NOT the image azimuth about the body center: the
    relief field is attached to the body, so the outline, the shading, and
    the cast shadows read one consistent surface under any in-plane rotation.
    For a circular disc the two conventions coincide.

    A sample exactly on the smooth limb lies outside the smooth body.  The
    surface there faces sideways and shows no area, so the navigator's predicted
    body leaves the sample dark; leaving it out of the outline too keeps it dark
    here and keeps two bodies whose limbs touch there from both claiming it.

    Parameters:
        grid: The working-grid geometry.
        field: The relief realization, or None for the smooth limb.

    Returns:
        Float mask (1 inside the body, 0 outside).
    """
    mask = grid.e2 < 1.0
    if field is not None:
        delta_cap = max(abs(field.h_min), abs(field.h_max)) * 1.001 + 1e-12
        low = max(0.0, 1.0 - delta_cap)
        high = 1.0 + delta_cap
        band = (grid.e2 > low * low) & (grid.e2 < high * high)
        if band.any():
            e_band = np.sqrt(grid.e2[band])
            # Body-attached azimuth: the rotated-frame elliptical parametric
            # angle (identical to the image azimuth for a circular disc).
            phi = np.arctan2(grid.u_rot[band] / grid.semi_b, grid.v_rot[band] / grid.semi_a)
            delta = field.outline_delta(phi)
            mask[band] = e_band <= 1.0 + delta
    return mask.astype(np.float64)
