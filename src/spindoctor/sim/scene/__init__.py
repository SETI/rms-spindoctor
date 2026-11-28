"""The sim scene format: files, the typed scene, validation, the boundary, and fields.

- :mod:`spindoctor.sim.scene.io`: load and save scene files, and walk the scene
  catalog.
- :mod:`spindoctor.sim.scene.validate`: validate a scene and build its typed form,
  through a cache that loading and rendering share.
- The typed scene: :mod:`spindoctor.sim.scene.idealized` (the keys a real pipeline
  could know), :mod:`spindoctor.sim.scene.truth` (the whole scene, which extends them
  with the truth keys the renderer reads, such as the planted offset and the detector and
  noise blocks, and with the test-only keys the integration tests read),
  :mod:`spindoctor.sim.scene.idealized_camera` (the ``instrument_config`` block),
  :mod:`spindoctor.sim.scene.truth_optics`, :mod:`spindoctor.sim.scene.truth_detector`,
  :mod:`spindoctor.sim.scene.truth_body` (a ``bodies`` entry), and
  :mod:`spindoctor.sim.scene.truth_atmosphere` (a body's haze).
  :mod:`spindoctor.sim.scene.declare`
  states how a field declares itself, and :mod:`spindoctor.sim.scene.reference` writes
  the schema reference page from the declarations.
- :mod:`spindoctor.sim.scene.boundary`: the key inventory, the information-boundary
  class of every key, and the navigator view of a scene.
- :mod:`spindoctor.sim.scene.fields`: the field helpers and tables every reader of
  a scene shares, the navigator-side simulated models included.
- :mod:`spindoctor.sim.scene.errors`: the error a scene that fails validation raises.
- :mod:`spindoctor.sim.scene.truth_echo`: the warning for a published instrument
  config that repeats the scene's planted truth.
- The rules the validator applies beside the types: the rules that read the scene's
  camera, in :mod:`spindoctor.sim.scene.rules_detector` and
  :mod:`spindoctor.sim.scene.rules_instrument_config`, the rules that reach past one
  body entry, in :mod:`spindoctor.sim.scene.rules_body`, and the rules of the blocks not
  typed yet, one module per block: :mod:`spindoctor.sim.scene.rules_common` (the
  primitives and the smaller blocks), :mod:`spindoctor.sim.scene.rules_ring`, and
  :mod:`spindoctor.sim.scene.rules_artifacts`.

The package imports nothing, so importing :mod:`spindoctor.sim.scene.fields` loads
neither the validator nor the instrument catalog (see :mod:`spindoctor.sim`).
"""
