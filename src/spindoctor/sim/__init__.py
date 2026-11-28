"""The image simulator: synthetic frames with known pointing.

The simulator renders a scene into the frame the navigator is handed, so the
navigator's recovery of the planted pointing can be measured.  Its subpackages are:

- :mod:`spindoctor.sim.scene`: the scene format (loading and saving scene files,
  validation, the information boundary, and the field helpers every scene reader
  shares).
- :mod:`spindoctor.sim.instruments`: the simulated instrument catalog and its
  per-mission configuration files.
- :mod:`spindoctor.sim.shared`: the geometry and star records that the renderer and
  the navigator-side simulated models both use.
- :mod:`spindoctor.sim.render`: the image side, which renders a validated scene
  into an image.

Three modules sit beside them: :mod:`spindoctor.sim.artifact_modes` (the
artifact-mode registry), :mod:`spindoctor.sim.seeds` (seed derivation for the
randomized effects), and :mod:`spindoctor.sim.png_export` (PNG export).

This package imports nothing and declares no ``__all__``.  Python runs a package's
``__init__`` before any of its submodules, so a re-export here would load the image
side whenever navigator-side code imports :mod:`spindoctor.sim.shared` or
:mod:`spindoctor.sim.scene.fields`, the only simulator modules navigator-side code
may import.  The ``__init__`` of each package on the way to those modules imports
nothing for the same reason.
"""
