"""The simulated instrument catalog.

Each simulated instrument is an entry in one of the per-mission YAML files under
``configs/``.  The modules that load and read them are:

- :mod:`spindoctor.sim.instruments.catalog`: load and merge the configuration files.
- :mod:`spindoctor.sim.instruments.schema`: the typed schema of one entry, with the
  rules that tie its parts together.
- :mod:`spindoctor.sim.instruments.schema_detector`: the schema of an entry's
  detector block and readout modes.
- :mod:`spindoctor.sim.instruments.resolvers`: every typed read of an entry.
- :mod:`spindoctor.sim.instruments.navigator_link`: the navigator block and
  detector an instrument emulates.
"""
