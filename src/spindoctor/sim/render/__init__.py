"""The simulator's image side: renders a validated scene into an image.

This package owns everything that turns a validated ``sim_params`` scene
into the image the navigator is handed: scene-radiance composition, optics,
the detector model, and telemetry effects, run as an ordered pipeline of
stages over a :class:`~spindoctor.sim.render.stages.SimFrame`.

Entry point: :func:`render_combined_model`, a thin cached driver
(:mod:`spindoctor.sim.render.driver`) over
:func:`spindoctor.sim.render.pipeline.run_pipeline`.  Callers outside this package
import the names in ``__all__`` from here.

The modules beside the driver hold the pipeline (:mod:`~spindoctor.sim.render.pipeline`),
the stage interface (:mod:`~spindoctor.sim.render.stages`), and the navigation-feature
loci that artifact placement aims at (:mod:`~spindoctor.sim.render.feature_loci`).
The stages live in subpackages:

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
consume only the filtered idealized view (``obs.nav_params``) built at the
``ObsSim`` boundary.  Conventions shared with the navigator side live in
:mod:`spindoctor.sim.shared`.

The names in ``__all__`` load the driver when one is first read, and importing the
package alone does not.  The scene validator reads constants from the body
renderers in :mod:`spindoctor.sim.render.bodies`, and the driver validates every
scene, so a package that imported the driver at once would make importing the
scene validator before the renderer fail with a circular import.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spindoctor.sim.render.driver import (
        clear_render_caches,
        render_combined_model,
        resolve_oversample,
    )

__all__ = [
    'clear_render_caches',
    'render_combined_model',
    'resolve_oversample',
]


def __getattr__(name: str) -> Any:
    """Return an exported name from the driver, loading the driver on first use.

    Parameters:
        name: The attribute being read from this package.

    Returns:
        The driver's attribute of that name, when ``name`` is in ``__all__``.

    Raises:
        AttributeError: If ``name`` is not in ``__all__``.
    """
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module('spindoctor.sim.render.driver'), name)
