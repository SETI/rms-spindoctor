"""The optics stage: what the camera's optical path does to the scene.

- :mod:`spindoctor.sim.render.optics.stage`: the stage, which runs the sub-stages
  below in order.
- :mod:`spindoctor.sim.render.optics.smear`: motion smear.
- :mod:`spindoctor.sim.render.optics.distortion`: residual geometric distortion.
- :mod:`spindoctor.sim.render.optics.psf`: the whole-scene point-spread function.
- :mod:`spindoctor.sim.render.optics.ghosts`: ghost reflections.
- :mod:`spindoctor.sim.render.optics.lit_clusters`: grouping of the lit pixels of a
  render-grid plane.
"""
