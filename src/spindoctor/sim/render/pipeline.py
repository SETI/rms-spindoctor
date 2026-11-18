"""The fixed-order stage pipeline of the renderer.

Stage order is physical and does not vary per scene: scene radiance is
composed, passes through the optics, is downsampled to the detector grid,
is read out through the detector model, and is then subjected to telemetry
loss.  A stage whose scene block is absent contributes nothing, so a
single-variable sweep can attribute error to exactly one effect.

Each stage draws from its own ``numpy.random.Generator`` seeded by
``derive_effect_seed(random_seed, '<stage-name>')``: stage noise
realizations are independent of which other stages are enabled, and adding
a stage later leaves existing stages' realizations unchanged.  Renaming a
stage reseeds its scenes and regenerates their baselines.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.sim.render.detector import apply_detector
from spindoctor.sim.render.optics.stage import apply_optics
from spindoctor.sim.render.radiance.compose import compose_scene_radiance
from spindoctor.sim.render.stages import SimFrame, Stage, downsample_to_detector
from spindoctor.sim.render.telemetry.stage import apply_telemetry
from spindoctor.sim.seeds import derive_effect_seed

__all__ = ['STAGES', 'run_pipeline']

# The registry: (stage name, stage callable) in execution order.  The name
# seeds the stage's RNG stream, so it is part of the determinism contract.
STAGES: tuple[tuple[str, Stage], ...] = (
    ('scene_radiance', compose_scene_radiance),
    ('optics', apply_optics),
    ('downsample', downsample_to_detector),
    ('detector', apply_detector),
    ('telemetry', apply_telemetry),
)


def run_pipeline(frame: SimFrame, params: Mapping[str, Any]) -> None:
    """Run every stage over ``frame`` in the fixed order, mutating it in place.

    Parameters:
        frame: The frame to render into.
        params: The full scene ``sim_params`` mapping (already stripped of
            the planted offset when rendering a GUI preview).
    """
    random_seed = int(params['random_seed'])
    for stage_name, stage in STAGES:
        rng = np.random.default_rng(derive_effect_seed(random_seed, stage_name))
        stage(frame, params=params, rng=rng)
