"""The simulated instrument catalog.

Each simulated instrument is an entry in one of the per-mission YAML files under
``configs/``.  The modules that load and read them are:

- :mod:`spindoctor.sim.instruments.catalog`: load and merge the configuration files.
- :mod:`spindoctor.sim.instruments.schema`: the schema of one entry.
- :mod:`spindoctor.sim.instruments.checks`: the cross-checks of one entry.
- :mod:`spindoctor.sim.instruments.resolvers`: every typed read of an entry.
- :mod:`spindoctor.sim.instruments.navigator_link`: the navigator config block and
  detector an instrument emulates.
"""
