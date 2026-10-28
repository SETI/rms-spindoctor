"""Pick a mesh scene's ``mesh_seed`` by its silhouette, before any navigation.

A mesh body's ``mesh_seed`` names one realization of the relief, and a scene
whose verdict depends on the shape can land anywhere in the population's spread.
The shape-mismatch scene (``planted_offset_shapemismatch``) and its irregularity
sweep therefore take their seed by a rule fixed in advance, which this module
implements so a later change to the relief synthesis can apply it again:

    Among mesh seeds 0 to 12 (the population ``test_sim_sweeps`` navigates),
    take the seed whose silhouette at the sweep's extreme lumpiness, rendered at
    the scene's pose, axes, and tessellation, has the median mismatch with the
    predicted smooth (zero-lumpiness) silhouette.

The mismatch is the symmetric-difference area of the two silhouettes over the
predicted silhouette's perimeter, a mean edge displacement in pixels at the
scene's scale.  The silhouettes are rendered fully lit, without anti-aliasing,
at 4 times the scene's scale so the measure isn't limited by the pixel grid.  The
rule reads only the renderer: it never navigates, so it can't favor a seed for
its navigation outcome.  Ties in the mismatch go to the lower seed.

Run from a checkout, with that checkout's ``src`` on the path::

    PYTHONPATH=src python -m tests.integration.sim_mesh_seed_rule

It prints every seed's mismatch, sorted, the seed the rule picks, and the seed
the scene holds; ``test_sim_mesh_seed_rule`` fails when the two differ.  If the
choice changes, update the scene's ``mesh_seed`` and the numbers that its
comment, the sweep's comment, the irregularity tests, and the simulator report
quote.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.sim.mesh_geometry import MeshBodySpec, mesh_spec_from_params, render_mesh_body_image
from spindoctor.sim.scene import load_sim_scene
from spindoctor.support.types import NDArrayBoolType
from tests.integration.sim_sweep import load_sweep

_INTEGRATION_ROOT = Path(__file__).resolve().parent
SWEEP_PATH = _INTEGRATION_ROOT / 'sim_sweeps' / 'irregularity_shape_mismatch.yaml'
# The population test_sim_sweeps navigates.
SWEEP_SEEDS = range(13)
# Render scale of the silhouettes relative to the scene.
_SCALE = 4


@dataclass(frozen=True)
class SeedMismatch:
    """One seed's silhouette mismatch with the predicted smooth body.

    Attributes:
        seed: The mesh seed.
        mismatch_px: Symmetric-difference area over the predicted perimeter, in
            scene pixels.
    """

    seed: int
    mismatch_px: float


def _silhouette(
    body: Mapping[str, Any], size: tuple[int, int], lumpiness: float, seed: int
) -> NDArrayBoolType:
    """The body's silhouette at ``_SCALE`` times the scene's scale.

    Parameters:
        body: The scene's body parameters (pose, axes, tessellation, center).
        size: The scene's ``(size_v, size_u)``.
        lumpiness: The rendered ``mesh_lumpiness``.
        seed: The rendered ``mesh_seed``.

    Returns:
        A boolean mask, true where the body covers the pixel.
    """
    base = mesh_spec_from_params(body)
    spec = MeshBodySpec(
        lumpiness=lumpiness,
        n_lat=base.n_lat,
        n_lon=base.n_lon,
        seed=seed,
        pose_euler_deg=base.pose_euler_deg,
        detail_octaves=base.detail_octaves,
    )
    axes = (float(body['axis1']), float(body['axis2']), float(body['axis3']))
    image = render_mesh_body_image(
        size=(size[0] * _SCALE, size[1] * _SCALE),
        center=(float(body['center_v']) * _SCALE, float(body['center_u']) * _SCALE),
        semi_axes_px=(axes[0] / 2.0 * _SCALE, axes[1] / 2.0 * _SCALE, axes[2] / 2.0 * _SCALE),
        spec=spec,
        anti_aliasing=0.0,
    )
    return np.asarray(image > 0.0)


def _perimeter(mask: NDArrayBoolType) -> float:
    """Count the mask's pixels that touch its outside (its 4-connected edge)."""
    return float(np.count_nonzero(mask & ~ndimage.binary_erosion(mask)))


def rank_seeds(
    body: Mapping[str, Any],
    size: tuple[int, int],
    lumpiness: float,
    seeds: Sequence[int],
) -> list[SeedMismatch]:
    """Measure each seed's silhouette mismatch and sort by it, then by seed.

    Parameters:
        body: The scene's body parameters.
        size: The scene's ``(size_v, size_u)``.
        lumpiness: The lumpiness at which the seeds' silhouettes are compared.
        seeds: The candidate seeds.

    Returns:
        One entry per seed, the smallest mismatch first.
    """
    predicted = _silhouette(body, size, 0.0, 0)
    perimeter = _perimeter(predicted)
    rows = []
    for seed in seeds:
        rendered = _silhouette(body, size, lumpiness, seed)
        area = float(np.count_nonzero(rendered ^ predicted))
        rows.append(SeedMismatch(seed=seed, mismatch_px=area / perimeter / _SCALE))
    return sorted(rows, key=lambda row: (row.mismatch_px, row.seed))


def median_seed(ranked: Sequence[SeedMismatch]) -> int:
    """The seed at the middle rank of an odd-length ranking.

    Raises:
        ValueError: If the ranking has an even length, which has no single median.
    """
    if len(ranked) % 2 == 0:
        raise ValueError(f'the rule needs an odd number of seeds; got {len(ranked)}')
    return ranked[len(ranked) // 2].seed


@dataclass(frozen=True)
class SweepSeedRanking:
    """The seed rule applied to a sweep's base scene.

    Attributes:
        scene_name: The base scene's file stem.
        lumpiness: The sweep's extreme ``mesh_lumpiness``, at which seeds compare.
        scene_seed: The ``mesh_seed`` the base scene holds.
        ranked: Every candidate seed's mismatch, the smallest first.
        chosen_seed: The seed the rule picks, the median of ``ranked``.
    """

    scene_name: str
    lumpiness: float
    scene_seed: int
    ranked: tuple[SeedMismatch, ...]
    chosen_seed: int


def rank_sweep_seeds(
    sweep_path: Path = SWEEP_PATH, seeds: Sequence[int] = SWEEP_SEEDS
) -> SweepSeedRanking:
    """Apply the seed rule to a sweep's base scene and its first body.

    Parameters:
        sweep_path: The sweep whose base scene and extreme lumpiness to use.
        seeds: The candidate seeds (an odd count).

    Returns:
        The ranking, the seed it picks, and the seed the scene holds.
    """
    sweep = load_sweep(sweep_path)
    scene = load_sim_scene(sweep.base_scene)
    body = scene['bodies'][0]
    size = (int(scene['size_v']), int(scene['size_u']))
    lumpiness = max(float(value) for value in sweep.values)
    ranked = rank_seeds(body, size, lumpiness, seeds)
    return SweepSeedRanking(
        scene_name=sweep.base_scene.stem,
        lumpiness=lumpiness,
        scene_seed=int(body['mesh_seed']),
        ranked=tuple(ranked),
        chosen_seed=median_seed(ranked),
    )


def main() -> None:
    """Print the shape-mismatch scene's seed ranking and the seed the rule picks."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--sweep', type=Path, default=SWEEP_PATH, help='the irregularity sweep')
    args = parser.parse_args()
    result = rank_sweep_seeds(args.sweep)
    print(f'{result.scene_name}: silhouette mismatch at mesh_lumpiness {result.lumpiness} (px)')
    for rank, row in enumerate(result.ranked, start=1):
        print(f'  {rank:2d}. seed {row.seed:2d}  {row.mismatch_px:.3f}')
    print(f'rule picks seed {result.chosen_seed}; the scene holds seed {result.scene_seed}')


if __name__ == '__main__':
    main()
