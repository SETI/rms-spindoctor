"""The navigator's models of synthetic frames.

Each module here is the simulated sibling of one of the navigator's models.  It predicts
what a synthetic frame shows from the filtered navigator view of the scene
(``obs.nav_params``), never from the scene's truth or from the image side's renderer, and
it emits the same features and annotations its real sibling emits for a real frame.  Each
class sets ``inputs`` to ``'scene_view'``, the model source ``ObsSim`` supplies, so the
model builder makes these models for the observations that supply it and for no others.

Modules:

    ``body``
        ``NavModelBodySimulated``, the body model.
    ``rings``
        ``NavModelRingsSimulated``, one model per navigable ring feature.
    ``titan``
        ``NavModelTitanSimulated``, the haze-envelope model.
    ``stars``
        ``NavModelStarsSimulated``, the star model.
    ``predict_body``
        The predicted-body renderer the body and haze-envelope models use.
    ``predict_ring``
        The predicted-ring renderer the ring model uses.

The model classes are exported from :mod:`spindoctor.nav_model`, which imports them in
the order the model registry requires, so this package re-exports nothing.
"""
