"""Render the streak a star draws when it moves during the exposure.

Shared by the star model's streak-measurement tests and the star techniques' centroid
tests, which need the same smeared star: its light spread evenly along a segment, with a
Gaussian point-spread function across and along it.
"""

import math

import numpy as np

_SAMPLE_STEP_PX = 0.05
"""Spacing in pixels of the Gaussian samples the streak is summed from."""


def draw_streak(
    image: np.ndarray,
    center_vu: tuple[float, float],
    streak_vu: tuple[float, float],
    *,
    flux: float,
    sigma: float = 0.6,
) -> None:
    """Add a streak to ``image`` in place.

    Parameters:
        image: 2-D float array, indexed pixel-centric (pixel ``(i, j)`` is centered at
            ``(i, j)``).
        center_vu: ``(v, u)`` of the streak's center, the star at mid-exposure.
        streak_vu: ``(v, u)`` from the streak's start to its end, in pixels.
        flux: Total light the streak adds.
        sigma: Standard deviation in pixels of the Gaussian point-spread function.
    """
    length = math.hypot(streak_vu[0], streak_vu[1])
    n_samples = max(1, math.ceil(length / _SAMPLE_STEP_PX) + 1)
    vv, uu = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    light = np.zeros(image.shape, np.float64)
    for t in np.linspace(-0.5, 0.5, n_samples) if n_samples > 1 else [0.0]:
        sv = center_vu[0] + t * streak_vu[0]
        su = center_vu[1] + t * streak_vu[1]
        light += np.exp(-((vv - sv) ** 2 + (uu - su) ** 2) / (2.0 * sigma * sigma))
    image += flux * light / light.sum()
