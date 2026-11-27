"""The telemetry stage: what transmission loses or mangles.

- :mod:`spindoctor.sim.render.telemetry.stage`: the stage, which applies the modes
  below in order.
- :mod:`spindoctor.sim.render.telemetry.loss`: the structured loss modes.
- :mod:`spindoctor.sim.render.telemetry.lossless`: the lossless line-pair coder whose
  budget the compression-truncation mode enforces.
- :mod:`spindoctor.sim.render.telemetry.artifacts`: the other telemetry artifact
  modes.
"""
