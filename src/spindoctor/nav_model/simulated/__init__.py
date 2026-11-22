"""The navigator's models of synthetic frames.

Each module here is the simulated sibling of one of the navigator's models.  It predicts
what a synthetic frame shows from the filtered navigator view of the scene
(``obs.nav_params``), never from the scene's truth or from the image side's renderer, and
it emits the same features and annotations its real sibling emits for a real frame.  Each
class sets ``inputs`` to ``'scene_view'``, which matches ``ObsSim``'s ``model_source``,
so these models are built for simulated frames and never for real ones.

Modules:

    ``body``
        ``NavModelBodySimulated``, the body model.
    ``body_arcs``
        The body model's limb and terminator arcs.
    ``rings``
        ``NavModelRingsSimulated``, one model per navigable ring feature.
    ``titan``
        ``NavModelTitanSimulated``, the haze-envelope model.
    ``stars``
        ``NavModelStarsSimulated``, the star model.
    ``predicted_body``
        The predicted body image and silhouette the body and haze models draw.
    ``predicted_ring``
        The predicted-ring renderer the ring model uses.

The model classes are exported from :mod:`spindoctor.nav_model`, which imports them in
the order the model registry requires, so this package re-exports nothing.
"""
