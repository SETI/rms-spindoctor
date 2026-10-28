"""The shape-mismatch scene's ``mesh_seed`` is the one its seed rule picks.

``planted_offset_shapemismatch`` and its irregularity sweep navigate one mesh
realization, chosen before any navigation by the rule in
:mod:`tests.integration.sim_mesh_seed_rule`: the median silhouette mismatch of
mesh seeds 0 to 12 at the sweep's extreme lumpiness.  A change to the relief
synthesis or to the mesh renderer can move that median.  This test then fails,
asking for the rule to be applied again and for the scene's seed and quoted
numbers to be updated, so the scene never keeps a seed picked under an older
relief.  It only renders silhouettes (about 5 s) and never navigates.
"""

from __future__ import annotations

import pytest

from tests.integration.sim_mesh_seed_rule import rank_sweep_seeds

pytestmark = pytest.mark.integration


def test_shape_mismatch_scene_uses_the_median_mismatch_seed() -> None:
    """The scene's ``mesh_seed`` is the median-mismatch seed of seeds 0 to 12."""
    ranking = rank_sweep_seeds()
    assert ranking.scene_seed == ranking.chosen_seed
