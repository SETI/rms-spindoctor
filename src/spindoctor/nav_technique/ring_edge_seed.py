"""The basin check on ``RingEdgeNav``'s coarse seed.

``RingEdgeNav`` chooses its integer seed from scores that credit a model vertex
only where its rounded pixel is itself a detected edge pixel.  Each cataloged
edge lies a pixel or two from its rounded model vertices, by a different amount
for each edge, so a model of several edges scores only part of its strength at
the true shift while a shift that lays one model edge exactly on one strong
image step scores all of it.  The check here rescores the chosen seed crediting
each vertex with the strongest edge within a tolerance across it, and replaces
the seed when it is clearly weaker by that measure than the best alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from spindoctor.nav_technique.dt_fitting import CoarseSearchResult, TolerantEdgeStrength
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['SeedBasinCheck', 'check_seed_basin']


@dataclass(frozen=True)
class SeedBasinCheck:
    """Outcome of :func:`check_seed_basin`.

    Parameters:
        seed: The seed to use: the one checked, or the tolerant maximum.
        ratio: The checked seed's tolerant credit over the tolerant maximum's,
            on the vertices both keep on the sensor; above 1 when the seed earns
            more there, and ``1.0`` when the two coincide or are not compared.
    """

    seed: CoarseSearchResult
    ratio: float


def check_seed_basin(
    seed: CoarseSearchResult,
    *,
    edge_mask: NDArrayBoolType,
    gradient_vu: NDArrayFloatType,
    vertices: NDArrayFloatType,
    normals: NDArrayFloatType,
    search_window_vu: tuple[int, int],
    valid_mask: NDArrayBoolType,
    tolerance_px: int,
    keep_ratio: float,
    min_shared_vertices: int,
) -> SeedBasinCheck:
    """Keep ``seed`` unless the edges within the tolerance favor another alignment.

    :class:`~spindoctor.nav_technique.dt_fitting.TolerantEdgeStrength` finds the
    shift whose vertices lie on the strongest edges within ``tolerance_px``
    pixel steps across them.  The two shifts are then compared on the model
    vertices both keep on the sensor
    (:meth:`~spindoctor.nav_technique.dt_fitting.TolerantEdgeStrength.compare`):
    ``seed`` is kept when its credit there is at least ``keep_ratio`` of the
    maximum's, and the maximum replaces it otherwise.  A maximum that wins only
    by dragging more of the model onto the sensor, where any ring texture
    credits it, therefore does not displace the seed.

    Parameters:
        seed: The seed chosen from the exact scores.
        edge_mask: Binary image edge mask.
        gradient_vu: ``(H, W, 2)`` image gradient.
        vertices: ``(N, 2)`` model vertices.
        normals: ``(N, 2)`` model normals, either sign.
        search_window_vu: ``(margin_v, margin_u)`` search window.
        valid_mask: Pixels whose edges may count: the sensor interior.
        tolerance_px: Pixel steps searched on each side of a vertex.
        keep_ratio: Fraction of the tolerant maximum's credit the seed must reach.
        min_shared_vertices: Fewest vertices on the sensor at both shifts for the
            two to be compared.

    Returns:
        The seed to use and the checked seed's tolerant ratio.
    """
    tolerant = TolerantEdgeStrength(
        edge_mask,
        gradient_vu,
        vertices,
        normals,
        tolerance_px=tolerance_px,
        valid_mask=valid_mask,
    )
    basin = tolerant.search(search_window_vu)
    if basin.offset_vu == seed.offset_vu:
        return SeedBasinCheck(seed=seed, ratio=1.0)
    seed_strength, basin_strength, shared = tolerant.compare(seed.offset_vu, basin.offset_vu)
    if shared < min_shared_vertices or basin_strength <= 0.0:
        # Too few vertices are on the sensor at both shifts to compare them on
        # the same evidence; the seed the exact scores chose stands.
        return SeedBasinCheck(seed=seed, ratio=1.0)
    ratio = seed_strength / basin_strength
    if ratio >= keep_ratio:
        return SeedBasinCheck(seed=seed, ratio=ratio)
    # The coarse-acquisition gate reads the best exact placement in the basin.
    replacement = CoarseSearchResult(
        offset_vu=basin.offset_vu, score=tolerant.best_match_fraction(basin.offset_vu)
    )
    return SeedBasinCheck(seed=replacement, ratio=ratio)
