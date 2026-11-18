"""NavModel — predicted-scene generators consumed by the orchestrator.

A NavModel is one of: ``stars``, a body, or a planet's rings.  Each model
renders the predicted scene from SPICE prediction (or operator-supplied
simulation parameters) and emits ``NavFeature`` instances ready for
technique consumption plus ``Annotations`` for the summary PNG.

Modules:

    ``nav_model``
        ``NavModel`` ABC.  Concrete subclasses implement ``create_model``,
        ``to_features``, and ``to_annotations``.
    ``nav_model_body``
        ``NavModelBody`` — catalog-driven body NavModel.
    ``nav_model_body_base``
        ``NavModelBodyBase`` — shared annotation helpers for body models.
    ``nav_model_rings``
        ``NavModelRings`` — catalog-driven ring NavModel.
    ``nav_model_rings_base``
        ``NavModelRingsBase`` — shared annotation helpers for ring models.
    ``nav_model_titan``
        ``NavModelTitan`` — haze-envelope model for a body whose opaque
        atmosphere hides its surface.
    ``simulated``
        The simulated sibling of each model above, built from the filtered
        navigator view of a synthetic frame's scene.
    ``titan_geometry``
        Observation-side geometry extraction feeding ``NavModelTitan``.
    ``stars``
        Catalog-driven star ``NavModel`` and supporting helpers.
"""

from spindoctor.nav_model.nav_model import NavModel, build_models_for_obs

# Each model registers when its module is imported, and build_models_for_obs builds
# models in registration order.  The blocks below fix that order: each model family
# registers its real model and then its simulated sibling.
from spindoctor.nav_model.simulated.stars import NavModelStarsSimulated
from spindoctor.nav_model.stars import NavModelStars

# isort: split
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_body_base import NavModelBodyBase
from spindoctor.nav_model.simulated.body import NavModelBodySimulated

# isort: split
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.nav_model_rings_base import NavModelRingsBase
from spindoctor.nav_model.simulated.rings import NavModelRingsSimulated

# isort: split
from spindoctor.nav_model.nav_model_titan import NavModelTitan
from spindoctor.nav_model.simulated.titan import NavModelTitanSimulated
from spindoctor.nav_model.titan_geometry import TitanGeometryInputs

__all__ = [
    'NavModel',
    'NavModelBody',
    'NavModelBodyBase',
    'NavModelBodySimulated',
    'NavModelRings',
    'NavModelRingsBase',
    'NavModelRingsSimulated',
    'NavModelStars',
    'NavModelStarsSimulated',
    'NavModelTitan',
    'NavModelTitanSimulated',
    'TitanGeometryInputs',
    'build_models_for_obs',
]
