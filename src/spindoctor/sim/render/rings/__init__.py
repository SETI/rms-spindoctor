"""The ring renderer of the scene-radiance stage.

:mod:`spindoctor.sim.render.rings.render` draws the optical-depth ring system.
:mod:`spindoctor.sim.render.rings.profiles` holds the features' radial profiles and cuts
pixel footprints at their boundaries, :mod:`spindoctor.sim.render.rings.footprint`
integrates those footprints, :mod:`spindoctor.sim.render.rings.photometry` holds the
single-scattering forms and their means, and :mod:`spindoctor.sim.render.rings.clutter`
holds the azimuthal structure, propellers, and moonlets.
"""
