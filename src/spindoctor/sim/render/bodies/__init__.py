"""The body renderers of the scene-radiance stage.

- :mod:`spindoctor.sim.render.bodies.render`: the placement of one body in the frame,
  and the crater carving.
- :mod:`spindoctor.sim.render.bodies.ellipsoid`: the ellipsoid renderer.
- :mod:`spindoctor.sim.render.bodies.ellipsoid_grid`: the pixel grids the ellipsoid
  renderer works on.
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
