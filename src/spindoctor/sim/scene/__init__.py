"""The sim scene format: files, validation, the information boundary, and fields.

- :mod:`spindoctor.sim.scene.io`: load and save scene files, and walk the scene
  catalog.
- :mod:`spindoctor.sim.scene.validate`: validate a scene, through a cache that
  loading and rendering share.
- :mod:`spindoctor.sim.scene.boundary`: the key inventory, the information-boundary
  class of every key, and the navigator's filtered view of a scene.
- :mod:`spindoctor.sim.scene.fields`: the field helpers and tables every reader of
  a scene shares, the navigator-side simulated models included.
- :mod:`spindoctor.sim.scene.truth_echo`: the warning for a published instrument
  config that repeats the scene's planted truth.
- The rules the validator applies, one module per block:
  :mod:`spindoctor.sim.scene.rules_common` (the primitives and the smaller blocks),
  :mod:`spindoctor.sim.scene.rules_body`, :mod:`spindoctor.sim.scene.rules_ring`,
  :mod:`spindoctor.sim.scene.rules_artifacts`, and
  :mod:`spindoctor.sim.scene.rules_instrument_config`.

The package imports nothing, so importing :mod:`spindoctor.sim.scene.fields` loads
neither the validator nor the instrument catalog (see :mod:`spindoctor.sim`).
"""
