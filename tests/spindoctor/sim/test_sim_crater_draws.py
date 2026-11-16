"""Each crater draws from its own stream, so a geometry change stays local.

The craters on a simulated body are carved by one function that both body renderers
share.  Every crater takes its random values from a stream derived from the body's
crater seed and the crater's index, so neither a one-pixel change in the body's interior
nor a crater that carves nothing can shift the values of any other crater.
"""

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.forward import body as forward_body
from spindoctor.sim.forward.body import carve_crater_heights
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

_SIZE = (64, 64)
_CENTER = (32.3, 31.7)
_SEMI_B = 20.0
# Raising this semi-axis by one step in its last bit brings exactly one pixel inside the
# ellipse: this one.
_SEMI_A = 25.123667852666337
_CHANGED_PIXEL = (7, 28)
_N_CRATERS = 40
_R_MAX = 4.0
# A seed for which one crater covers the added pixel, no crater's center search reaches
# it, and the first two craters lie apart.
_SEED = 37

# A small body on a large grid, so that a crater's center search often takes several
# rounds of candidates; one stream shared by all craters would then shift every crater
# after one whose search changed.
_SMALL_SIZE = (256, 256)
_SMALL_CENTER = (128.3, 127.6)
_SMALL_SEMI_AXES = (6.2, 5.1)
_CLEARED_PIXEL = (122, 128)
_SMALL_N_CRATERS = 60
_SMALL_SEEDS = range(20)


def _ellipse(
    size: tuple[int, int], center: tuple[float, float], semi_a: float, semi_b: float
) -> tuple[NDArrayBoolType, NDArrayFloatType, NDArrayFloatType]:
    """The strict interior of an ellipse and the grid's centered pixel coordinates.

    Parameters:
        size: The grid's rows and columns.
        center: The ellipse center (v, u) in pixel corner coordinates.
        semi_a: The semi-axis along v, in pixels.
        semi_b: The semi-axis along u, in pixels.

    Returns:
        The interior mask and the centered v and u pixel corner coordinates.
    """
    v_coords, u_coords = np.mgrid[0 : size[0], 0 : size[1]].astype(float)
    v_coords += 0.5 - center[0]
    u_coords += 0.5 - center[1]
    mask = (v_coords / semi_a) ** 2 + (u_coords / semi_b) ** 2 < 1.0
    return mask, v_coords, u_coords


def _carve(
    mask: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    n_craters: int,
    crater_seed: int = _SEED,
    center: tuple[float, float] = _CENTER,
    r_min: float = 1.0,
    r_max: float = _R_MAX,
) -> NDArrayFloatType:
    """Carve test craters on a mask.

    Parameters:
        mask: The interior mask.
        v_coords: The centered v pixel coordinates.
        u_coords: The centered u pixel coordinates.
        n_craters: The number of craters to carve.
        crater_seed: The body's crater seed.
        center: The body center (v, u) in pixel corner coordinates.
        r_min: The smallest crater radius, in pixels.
        r_max: The largest crater radius, in pixels.

    Returns:
        The crater height field.
    """
    return carve_crater_heights(
        mask,
        v_coords,
        u_coords,
        crater_seed=crater_seed,
        n_craters=n_craters,
        R_min=r_min,
        R_max=r_max,
        crater_power_law_exponent=3.0,
        crater_relief_scale=0.6,
        work_center_v=center[0],
        work_center_u=center[1],
        aa_scale=1,
    )


def test_clearing_a_pixel_moves_only_craters_centered_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing one interior pixel moves only the craters whose center was that pixel.

    Every crater's center is recorded on the full mask and on the mask with one pixel
    cleared, over many seeds.  A crater may move only if its old center was the cleared
    pixel, and at least one crater must move, so the comparison is not empty.
    """
    centers: list[tuple[int, int]] = []
    draw_center = forward_body._draw_crater_center

    def recording_draw(mask: NDArrayBoolType, rng: np.random.Generator) -> tuple[int, int]:
        """Draw a crater center as the carver does and record it.

        Parameters:
            mask: The pixels a center may land on.
            rng: The crater's random stream.

        Returns:
            The center's array row and column.
        """
        center = draw_center(mask, rng)
        centers.append(center)
        return center

    monkeypatch.setattr(forward_body, '_draw_crater_center', recording_draw)
    mask_full, v_coords, u_coords = _ellipse(_SMALL_SIZE, _SMALL_CENTER, *_SMALL_SEMI_AXES)
    mask_cleared = mask_full.copy()
    mask_cleared[_CLEARED_PIXEL] = False
    assert mask_full[_CLEARED_PIXEL]

    moved: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for seed in _SMALL_SEEDS:
        center_lists = []
        for mask in (mask_full, mask_cleared):
            centers.clear()
            _carve(
                mask,
                v_coords,
                u_coords,
                n_craters=_SMALL_N_CRATERS,
                crater_seed=seed,
                center=_SMALL_CENTER,
                r_min=0.5,
                r_max=1.5,
            )
            center_lists.append(list(centers))
        moved += [(seed, old, new) for old, new in zip(*center_lists, strict=True) if old != new]
    assert moved
    assert [entry for entry in moved if entry[1] != _CLEARED_PIXEL] == []


def test_last_bit_change_moves_only_heights_near_the_changed_pixel() -> None:
    """A last-bit change in one semi-axis changes heights only around the pixel it adds.

    For this seed one crater covers the added pixel and no crater's center lands on it,
    so some heights change, and every changed height lies within a crater's reach (1.1
    times the largest radius) of that pixel.
    """
    mask_a, v_coords, u_coords = _ellipse(_SIZE, _CENTER, _SEMI_A, _SEMI_B)
    semi_a_b = float(np.nextafter(_SEMI_A, np.inf))
    mask_b, _, _ = _ellipse(_SIZE, _CENTER, semi_a_b, _SEMI_B)
    assert np.argwhere(mask_a != mask_b).tolist() == [list(_CHANGED_PIXEL)]

    height_a = _carve(mask_a, v_coords, u_coords, n_craters=_N_CRATERS)
    height_b = _carve(mask_b, v_coords, u_coords, n_craters=_N_CRATERS)
    changed = np.argwhere(height_a != height_b)
    assert changed.size
    distance = np.hypot(*(changed - np.array(_CHANGED_PIXEL)).T)
    assert np.all(distance <= 1.1 * _R_MAX)


def test_crater_that_carves_nothing_leaves_the_next_crater_unchanged() -> None:
    """A crater that carves nothing leaves the next crater's values unchanged.

    No valid body produces such a crater, since a crater always reaches its own center,
    so the test makes one: coordinates that are not numbers over the first crater's
    footprint put no pixel within its reach.  The second crater, which lies elsewhere
    for this seed, must still carve exactly what it carves when the first crater carves
    its full footprint.
    """
    mask, v_coords, u_coords = _ellipse(_SIZE, _CENTER, _SEMI_A, _SEMI_B)
    first = _carve(mask, v_coords, u_coords, n_craters=1)
    second = _carve(mask, v_coords, u_coords, n_craters=2) - first
    blind = ndimage.binary_dilation(first != 0.0)
    assert blind.any()
    assert not np.any(second[blind])

    v_blind = np.where(blind, np.nan, v_coords)
    assert np.array_equal(_carve(mask, v_blind, u_coords, n_craters=2), second)
