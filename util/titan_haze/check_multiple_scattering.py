"""Check the renderer's haze reflectance against a Monte Carlo simulation.

Usage::

    python util/titan_haze/check_multiple_scattering.py [<photons>]

Sunlight falls straight down (``mu0 = 1``) on a plane-parallel haze, and photons are
followed through it until they leave the top, leave the bottom of a finite layer, or
are absorbed.  Absorption is carried as a weight, ``w`` per scattering.  The photons
leaving the top are binned by the cosine ``mu`` of their direction, which gives the
layer's I/F at the phase angle ``arccos(mu)``.  The script prints, beside each Monte
Carlo value, the value of
:func:`spindoctor.sim.render.bodies.haze_scattering.haze_reflectance` for the same
layer:

- a semi-infinite layer with Titan's haze at 650 nm (albedo 0.92, the two-term phase
  function ``g`` 0.886, ``g2`` -0.151, share 0.099);
- the same layer with isotropic scattering, a control on Hapke's closed-form
  H-function;
- finite layers of vertical optical depth 0.05, 0.3, 1, and 3 with Titan's haze, a
  check on the scaling of single and multiple scattering by the column.

With the default 2,000,000 photons the Monte Carlo values carry about 1% of noise.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from spindoctor.sim.render.bodies.haze_scattering import (
    haze_reflectance,
    mean_cosine,
)

# The bins of the emergent direction cosine.
_BINS = 20

# A photon whose weight falls below this is dropped.
_MIN_WEIGHT = 1e-4


@dataclass(frozen=True)
class Haze:
    """A haze's single-scattering albedo and two-term Henyey-Greenstein phase function.

    Parameters:
        albedo: The single-scattering albedo.
        g: The first term's asymmetry parameter.
        g2: The second term's asymmetry parameter.
        g2_fraction: The share of the scattered light in the second term.
    """

    albedo: float
    g: float
    g2: float
    g2_fraction: float

    def phase(self, cos_angle: float) -> float:
        """The phase function, averaging 1 over the sphere, at one scattering angle."""

        def hg(g: float) -> float:
            return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cos_angle) ** 1.5

        return (1.0 - self.g2_fraction) * hg(self.g) + self.g2_fraction * hg(self.g2)


TITAN_650 = Haze(albedo=0.92, g=0.886, g2=-0.151, g2_fraction=0.099)
ISOTROPIC = Haze(albedo=0.92, g=0.0, g2=0.0, g2_fraction=0.0)


def _sample_hg(g: float, u: np.ndarray) -> np.ndarray:
    """Cosines of the scattering angle drawn from a Henyey-Greenstein function."""
    if abs(g) < 1e-6:
        return 2.0 * u - 1.0
    s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
    return np.clip((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0)


def monte_carlo(
    haze: Haze, depth: float, *, photons: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The I/F of a layer lit from straight above, by emergent direction cosine.

    Parameters:
        haze: The haze.
        depth: The layer's vertical optical depth; ``math.inf`` for a semi-infinite one.
        photons: The number of photons.
        seed: The random seed.

    Returns:
        ``(mu, i_over_f)``: the bin centers and the I/F in each.
    """
    rng = np.random.default_rng(seed)
    # Direction cosines, z pointing down into the layer; z is the optical depth.
    dx = np.zeros(photons)
    dy = np.zeros(photons)
    dz = np.ones(photons)
    z = np.zeros(photons)
    weight = np.ones(photons)
    alive = np.ones(photons, dtype=bool)
    emergent = np.zeros(_BINS)
    while alive.any():
        idx = np.nonzero(alive)[0]
        z[idx] += dz[idx] * -np.log(rng.random(idx.size))
        out_top = z[idx] < 0.0
        top = idx[out_top]
        np.add.at(emergent, np.clip((-dz[top] * _BINS).astype(int), 0, _BINS - 1), weight[top])
        out_bottom = z[idx] > depth
        alive[idx[out_top | out_bottom]] = False
        idx = idx[~(out_top | out_bottom)]
        if idx.size == 0:
            break
        weight[idx] *= haze.albedo
        second = rng.random(idx.size) < haze.g2_fraction
        u = rng.random(idx.size)
        ct = np.where(second, _sample_hg(haze.g2, u), _sample_hg(haze.g, u))
        st = np.sqrt(1.0 - ct * ct)
        azimuth = 2.0 * math.pi * rng.random(idx.size)
        ox, oy, oz = dx[idx], dy[idx], dz[idx]
        sz = np.sqrt(np.maximum(1.0 - oz * oz, 0.0))
        vertical = sz < 1e-8
        safe = np.where(vertical, 1.0, sz)
        cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
        nx = np.where(vertical, st * cos_a, st * (ox * oz * cos_a - oy * sin_a) / safe + ox * ct)
        ny = np.where(vertical, st * sin_a, st * (oy * oz * cos_a + ox * sin_a) / safe + oy * ct)
        nz = np.where(vertical, np.sign(oz) * ct, -st * cos_a * sz + oz * ct)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        dx[idx], dy[idx], dz[idx] = nx / norm, ny / norm, nz / norm
        alive[idx[weight[idx] < _MIN_WEIGHT]] = False
    mu = (np.arange(_BINS) + 0.5) / _BINS
    # I/F = pi * I / (mu0 * pi * F): the emergent weight per unit solid angle, over the
    # incident flux per unit area, is weight / (2 pi mu dmu); mu0 is 1.
    i_over_f = (emergent / photons) / (2.0 * mu / _BINS)
    return mu, i_over_f


def renderer(haze: Haze, depth: float, mu: float) -> float:
    """The renderer's I/F for the same layer, lit from straight above, seen at ``mu``."""
    tau_los = 1e6 if math.isinf(depth) else depth / mu
    value = haze_reflectance(
        np.array(tau_los),
        np.array(mu),
        np.array(1.0),
        albedo=haze.albedo,
        g=mean_cosine(haze.g, haze.g2, haze.g2_fraction),
        phase_function=haze.phase(-mu),
    )
    return float(value)


def report(label: str, haze: Haze, depth: float, *, photons: int) -> None:
    """Print the Monte Carlo and renderer I/F at a few emergent cosines."""
    mu, i_over_f = monte_carlo(haze, depth, photons=photons, seed=1)
    print(label)
    for k in (4, 9, 14, 19):
        code = renderer(haze, depth, float(mu[k]))
        print(
            f'  mu {mu[k]:.3f}: Monte Carlo {i_over_f[k]:.4f}, renderer {code:.4f}, '
            f'ratio {code / i_over_f[k]:.2f}'
        )


def main() -> None:
    """Run every check."""
    photons = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    report("Semi-infinite, Titan's haze at 650 nm", TITAN_650, math.inf, photons=photons)
    report('Semi-infinite, isotropic', ISOTROPIC, math.inf, photons=photons)
    for depth in (0.05, 0.3, 1.0, 3.0):
        report(f"Vertical depth {depth}, Titan's haze", TITAN_650, depth, photons=photons)


if __name__ == '__main__':
    main()
