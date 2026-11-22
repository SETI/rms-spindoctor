"""The fixed-order stage pipeline of the renderer.

Stage order is physical and does not vary per scene: scene radiance is
composed, passes through the optics, is downsampled to the detector grid,
is read out through the detector model, and is then subjected to telemetry
loss.  A stage whose scene block is absent contributes nothing, so a
single-variable sweep can attribute error to exactly one effect.

The pipeline hands each stage only the frame and the scene.  Each stage seeds its own
random streams by name from the scene seed (see :mod:`spindoctor.sim.render.stages`).
"""

from collections.abc import Mapping
from typing import Any

from spindoctor.sim.render.detector import apply_detector
from spindoctor.sim.render.optics.stage import apply_optics
from spindoctor.sim.render.radiance.compose import compose_scene_radiance
from spindoctor.sim.render.stages import SimFrame, Stage, downsample_to_detector
from spindoctor.sim.render.telemetry.stage import apply_telemetry

__all__ = ['STAGES', 'run_pipeline']

# The registry: the stage callables in execution order.
STAGES: tuple[Stage, ...] = (
    compose_scene_radiance,
    apply_optics,
    downsample_to_detector,
    apply_detector,
    apply_telemetry,
)


def run_pipeline(frame: SimFrame, params: Mapping[str, Any]) -> None:
    """Run every stage over ``frame`` in the fixed order, mutating it in place.

    Parameters:
        frame: The frame to render into.
        params: The full scene ``sim_params`` mapping (already stripped of
            the planted offset when rendering a GUI preview).
    """
    for stage in STAGES:
        stage(frame, params=params)
