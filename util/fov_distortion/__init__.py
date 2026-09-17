"""Star-field FOV distortion and camera-twist analysis.

Measures, per instrument and directly from star fields, the two components of
the camera-pointing error that survive the geometry oops already applies: the
rotational error (FOV twist) and the lateral residual distortion (the
field-position-dependent displacement left after the known distortion model is
removed).

The package is organized so the numerical core carries no navigation
dependency and is unit-tested on synthetic point clouds:

- ``decompose``: pure-numpy decomposition of a per-star residual field into a
  rigid twist (rotation + translation) plus a low-order radial distortion
  model.  No spindoctor imports.
- ``aggregate``: per-instrument aggregation of per-frame twists into a
  consistency verdict and a rotation-fitting recommendation.  No spindoctor
  imports.
- ``measure``: per-frame measurement -- loads an observation, runs star
  navigation for the translation prior, centroids every predictable catalog
  star, and hands the predicted / detected pairs to ``decompose``.  This is
  the only module that reaches into spindoctor and the navigation holdings.
- ``plots``: residual-quiver, twist-scatter, and radial-profile figures.
- ``config``: per-instrument analysis configuration loaded from the sidecar
  YAML files in ``configs/``.
- ``run``: the campaign driver.
"""

__all__: list[str] = []
