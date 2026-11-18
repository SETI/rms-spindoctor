"""The body renderers of the scene-radiance stage.

- :mod:`spindoctor.sim.render.bodies.render`: the ellipsoid renderer and the dispatch
  among the render paths.
- :mod:`spindoctor.sim.render.bodies.topo`: the topographic ellipsoid renderer.
- :mod:`spindoctor.sim.render.bodies.mesh`: the polyhedral-mesh renderer.
- :mod:`spindoctor.sim.render.bodies.spec`: the typed, frozen form of one scene
  ``bodies`` entry.
- :mod:`spindoctor.sim.render.bodies.texture`: multiplicative surface textures.
- :mod:`spindoctor.sim.render.bodies.relief`: the limb and terminator topography
  field.
- :mod:`spindoctor.sim.render.bodies.photometry`: the surface-scattering laws.
- :mod:`spindoctor.sim.render.bodies.atmosphere`: the haze layer of a haze-limb body.
- :mod:`spindoctor.sim.render.bodies.haze`: the structure that breaks a haze
  layer's symmetry.
- :mod:`spindoctor.sim.render.bodies.translate`: sub-pixel translation of a
  render-grid plane.
"""
