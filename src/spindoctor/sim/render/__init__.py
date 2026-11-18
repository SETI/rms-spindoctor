"""The simulator's image side: renders a validated scene into an image.

This package owns everything that turns a validated ``sim_params`` scene
into the image the navigator is handed: scene-radiance composition, optics,
the detector model, and telemetry effects, run as an ordered pipeline of
stages over a :class:`~spindoctor.sim.render.stages.SimFrame`.

The entry point is :mod:`spindoctor.sim.render.driver`, a thin cached driver over
:func:`spindoctor.sim.render.pipeline.run_pipeline`.  The modules beside it hold the
pipeline (:mod:`~spindoctor.sim.render.pipeline`), the stage interface
(:mod:`~spindoctor.sim.render.stages`), and the navigation-feature loci that artifact
placement aims at (:mod:`~spindoctor.sim.render.feature_loci`).  The stages live in
subpackages:

- :mod:`spindoctor.sim.render.radiance`: the scene-radiance stage, which composes
  the noise-free signal image from the bodies, rings, and stars.
- :mod:`spindoctor.sim.render.bodies`: the body renderers (ellipsoid, topographic,
  and polyhedral mesh) with their photometry, relief, textures, and atmosphere.
- :mod:`spindoctor.sim.render.rings`: the optical-depth ring-system renderer.
- :mod:`spindoctor.sim.render.optics`: the optics stage (smear, distortion, PSF,
  and ghosts).
- :mod:`spindoctor.sim.render.detector`: the detector stage.
- :mod:`spindoctor.sim.render.telemetry`: the telemetry stage.

The navigator-side simulated NavModels never import from this package; they
consume only the navigator view (``obs.nav_params``) built at the
``ObsSim`` boundary.  Conventions shared with the navigator side live in
:mod:`spindoctor.sim.shared`.
"""
