"""Each crater draws from its own stream, so a geometry change stays local.

The craters on a simulated body are carved by one function that both body renderers
share.  Every crater takes its random values from a stream derived from the body's
crater seed and the crater's index, so neither a one-pixel change in the body's interior
nor a crater that carves nothing can shift the values of any other crater.
"""

import numpy as np
from scipy import ndimage

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


def _grid(semi_a: float) -> tuple[NDArrayBoolType, NDArrayFloatType, NDArrayFloatType]:
    """The strict interior of the test ellipse and its centered pixel coordinates.

    Parameters:
        semi_a: The semi-axis along v, in pixels.

    Returns:
        The interior mask and the centered v and u pixel corner coordinates.
    """
    v_coords, u_coords = np.mgrid[0 : _SIZE[0], 0 : _SIZE[1]].astype(float)
    v_coords += 0.5 - _CENTER[0]
    u_coords += 0.5 - _CENTER[1]
    mask = (v_coords / semi_a) ** 2 + (u_coords / _SEMI_B) ** 2 < 1.0
    return mask, v_coords, u_coords


def _carve(
    mask: NDArrayBoolType,
    v_coords: NDArrayFloatType,
    u_coords: NDArrayFloatType,
    *,
    n_craters: int,
) -> NDArrayFloatType:
    """Carve the test craters on a mask.

    Parameters:
        mask: The interior mask.
        v_coords: The centered v pixel coordinates.
        u_coords: The centered u pixel coordinates.
        n_craters: The number of craters to carve.

    Returns:
        The crater height field.
    """
    return carve_crater_heights(
        mask,
        v_coords,
        u_coords,
        crater_seed=_SEED,
        n_craters=n_craters,
        R_min=1.0,
        R_max=_R_MAX,
        crater_power_law_exponent=3.0,
        crater_relief_scale=0.6,
        work_center_v=_CENTER[0],
        work_center_u=_CENTER[1],
        aa_scale=1,
    )


def test_last_bit_change_moves_only_heights_near_the_changed_pixel() -> None:
    """A last-bit change in one semi-axis changes heights only around the pixel it adds.

    No crater's center lands on the added pixel for this seed, so every crater keeps
    its values and the height field can differ only within a crater's reach (1.1 times
    the largest radius) of that pixel.
    """
    mask_a, v_coords, u_coords = _grid(_SEMI_A)
    mask_b, _, _ = _grid(float(np.nextafter(_SEMI_A, np.inf)))
    assert np.argwhere(mask_a != mask_b).tolist() == [list(_CHANGED_PIXEL)]

    height_a = _carve(mask_a, v_coords, u_coords, n_craters=_N_CRATERS)
    height_b = _carve(mask_b, v_coords, u_coords, n_craters=_N_CRATERS)
    changed = np.argwhere(height_a != height_b)
    distance = np.hypot(*(changed - np.array(_CHANGED_PIXEL)).T)
    assert np.all(distance <= 1.1 * _R_MAX)


def test_crater_with_empty_mask_leaves_the_next_crater_unchanged() -> None:
    """A crater that carves nothing leaves the next crater's values unchanged.

    Coordinates that are not numbers over the first crater's footprint put no pixel
    within its reach, so its mask is empty; the second crater, which lies elsewhere for
    this seed, must carve exactly what it carves when the first crater's mask is full.
    """
    mask, v_coords, u_coords = _grid(_SEMI_A)
    first = _carve(mask, v_coords, u_coords, n_craters=1)
    second = _carve(mask, v_coords, u_coords, n_craters=2) - first
    blind = ndimage.binary_dilation(first != 0.0)
    assert not np.any(second[blind])

    v_blind = np.where(blind, np.nan, v_coords)
    assert np.array_equal(_carve(mask, v_blind, u_coords, n_craters=2), second)
