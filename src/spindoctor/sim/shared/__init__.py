"""Conventions the renderer and the navigator-side simulated models share.

Both sides of the information boundary import these modules, so the image the
renderer draws and the prediction the navigator makes follow the same geometry, and
the planted scene error is the only error in a recovery measurement:

- :mod:`spindoctor.sim.shared.ellipsoid`: ellipsoid shading geometry.
- :mod:`spindoctor.sim.shared.lighting`: the light direction and the Henyey-Greenstein
  phase function.
- :mod:`spindoctor.sim.shared.mesh`: polyhedral-mesh geometry for irregular bodies.
- :mod:`spindoctor.sim.shared.ring`: ring-edge geometry.
- :mod:`spindoctor.sim.shared.stars`: catalog star records built from scene star
  parameters.
- :mod:`spindoctor.sim.shared.spherical_harmonics`: real spherical-harmonic
  synthesis.

A module here imports nothing from the simulator outside this package except
:mod:`spindoctor.sim.scene.fields`, so it never loads the renderer, the scene rules,
or the editor.  The package imports nothing (see :mod:`spindoctor.sim`).
"""
