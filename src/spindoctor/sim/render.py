"""Entry point for rendering a simulated scene: params in, image + meta out.

``render_combined_model`` is a thin, cached driver over the forward-model
stage pipeline (:mod:`spindoctor.sim.forward`): it encodes the scene as
deterministic sorted-key JSON, validates the scene, runs the pipeline on a fresh
:class:`~spindoctor.sim.forward.stages.SimFrame` (cached on a key built from
that JSON), and returns the rendered image with the renderer's output metadata.

Every render validates its scene with
:func:`~spindoctor.sim.scene.validate_sim_params`, so a scene built in memory
is held to the same rules as one read from a file.  The one route that skips
validation is :func:`render_unvalidated_preview`, which the scene editor's live
preview uses so the author can see a half-edited scene that would not yet pass.

A scene that passes validation is cached on its full JSON alone, so rendering one
scene many times validates it once, whatever label its callers give their errors.
That is not quite the render cache key: an offset-free render leaves the planted
pointing out of its key, and validation never does.  A refused scene is never
cached, and its error starts with the label of the caller that met it.

Callers are ``ObsSim``, the scene-editor GUI, and ``sim/png_export.py``
(which the doc-gallery and sweep runners drive).  The params-JSON caching
contract: scene parameters are JSON-serializable scalars/lists/maps, no
wall-clock or global RNG state is consulted, and two renders of the same
scene are bit-identical on one machine.

The scene editor renders on two threads at once (its preview on the UI thread,
the frame the navigator reads on a worker), which relies on three properties of
the render path: every render builds its own frame, parameter dict, and RNG
streams; the ``lru_cache`` tables stay coherent under concurrent calls; and no
caller writes into a cached value (this function hands out copies, and the stage
caches' callers only read theirs).  A change that breaks one of them breaks the
editor.
"""

import copy
import json
from functools import lru_cache
from typing import Any

from spindoctor.sim.forward.pipeline import run_pipeline
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.sim.scene import validate_sim_params
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.types import NDArrayFloatType

# The planted pointing keys an offset-free render zeroes, and so leaves out of
# its cache key: previews differing only in planted pointing share one render.
_PLANTED_POINTING_KEYS: tuple[str, ...] = ('offset_v', 'offset_u', 'offset_rotation_deg')


def clear_render_caches() -> None:
    """Drop every render-path lru_cache so RNG and shape paths re-run.

    Test and tooling helper: the render caches are parameter-keyed and never
    stale in production, but determinism tests must re-execute the cached
    paths to prove reproducibility.
    """
    from spindoctor.sim.forward import body, body_mesh, psf, relief, star

    _render_combined_model_cached.cache_clear()
    _validate_scene_json.cache_clear()
    star._render_stars_cached.cache_clear()
    star._render_sky_counts_cached.cache_clear()
    body._render_body_shape_cached.cache_clear()
    body._render_topo_shape_cached.cache_clear()
    body_mesh._render_mesh_shape_cached.cache_clear()
    relief.clear_relief_caches()
    psf.psf_kernel.cache_clear()


# Two entries: the scene editor renders a scene's offset-free preview and the
# frame the navigator reads (planted offset applied), and an offset-only edit
# then re-renders only the second.
@lru_cache(maxsize=2)
def _render_combined_model_cached(
    sim_params_json: str,
    *,
    ignore_offset: bool,
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Internal cached function to compute combined model rendering."""
    sim_params = json.loads(sim_params_json)
    if ignore_offset:
        # GUI preview mode: render the unshifted, unrolled geometry.
        for key in _PLANTED_POINTING_KEYS:
            sim_params[key] = 0.0
    size_v = int(sim_params['size_v'])
    size_u = int(sim_params['size_u'])

    # A whole-scene PSF is applied on an oversampled radiance grid so limb, ring
    # edge, and star profiles carry sub-detector-pixel structure into the
    # convolution; the box downsample after optics returns to the detector grid.
    # A scene with no optics block renders at oversample 1 (identical to a
    # frame with no sub-pixel optics to resolve).
    oversample = resolve_oversample(sim_params)
    frame = new_sim_frame(size_v, size_u, oversample=oversample)
    run_pipeline(frame, sim_params)
    return frame.signal, frame.truth


def resolve_oversample(sim_params: dict[str, Any]) -> int:
    """Return the radiance oversampling factor for a scene.

    An explicit ``oversample`` key wins.  Otherwise a scene with an active PSF
    -- an ``optics.psf`` block or the ``instrument_defaults`` opt-in, either of
    which puts a kernel on the frame -- oversamples 4x by default so the
    convolution resolves sub-pixel edge structure; a scene with no PSF renders on
    the detector grid at oversample 1.

    Parameters:
        sim_params: The full scene mapping.

    Returns:
        The oversampling factor (a positive integer >= 1).
    """
    explicit = sim_params.get('oversample')
    if explicit is not None:
        return max(1, int(explicit))
    optics = sim_params.get('optics')
    psf_active = isinstance(optics, dict) and isinstance(optics.get('psf'), dict)
    artifacts = sim_params.get('artifacts')
    instrument_defaults = isinstance(artifacts, dict) and bool(
        artifacts.get('instrument_defaults', False)
    )
    return 4 if (psf_active or instrument_defaults) else 1


# A handful of entries: a validation is cheap next to a render, and a scene
# that validated once validates again whenever it is rendered again.
@lru_cache(maxsize=8)
def _validate_scene_json(scene_json: str) -> None:
    """Validate the scene that ``scene_json`` encodes, once per scene.

    The mapping validated is decoded from the same JSON the render decodes, so
    validation checks what the pipeline will read.  Only a pass is cached: a refused
    scene raises every time, since ``lru_cache`` does not cache exceptions.  Its
    error carries the default label; :func:`_check_scene_json` relabels it.
    """
    validate_sim_params(json.loads(scene_json))


def _check_scene_json(scene_json: str, source: str) -> None:
    """Validate a scene through the validation cache, labeling a refusal ``source``.

    The label changes only the text of an error, so a pass is cached on the scene
    alone.  A refused scene is validated again under ``source`` so that its error
    starts with the caller's label.

    Parameters:
        scene_json: The scene as sorted-key JSON.
        source: The label validation errors start with.

    Raises:
        ~spindoctor.sim.scene_schema.SimSceneValidationError: If the scene fails
            validation.
    """
    try:
        _validate_scene_json(scene_json)
    except SimSceneValidationError:
        pass
    else:
        return
    # Raised outside the handler, so the unlabeled refusal is not chained onto it.
    validate_sim_params(json.loads(scene_json), source=source)


def validate_scene(sim_params: dict[str, Any], *, source: str = 'sim_params') -> None:
    """Validate a scene exactly as :func:`render_combined_model` does, without rendering.

    The check shares the render path's validation cache, so a scene checked here and
    then rendered is validated once, whatever ``source`` each call gives.  The scene
    editor calls it after each preview, so an author sees why a scene would be
    refused even when nothing else renders it.

    Parameters:
        sim_params: The parameters describing the simulated model.
        source: The label validation errors start with.

    Raises:
        ~spindoctor.sim.scene_schema.SimSceneValidationError: If the scene fails validation.
    """
    _check_scene_json(json.dumps(sim_params, sort_keys=True), source)


def render_combined_model(
    sim_params: dict[str, Any], *, ignore_offset: bool = False, source: str = 'sim_params'
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Validate a scene, then render it. Returns (img, meta).

    The scene is checked with :func:`~spindoctor.sim.scene.validate_sim_params`
    before anything renders, so an invalid scene raises the same error a scene
    file would raise when loaded.  A pass is cached on the scene's full JSON alone,
    so rendering one scene repeatedly validates it once, whatever ``source`` each
    render gives.

    Parameters:
        sim_params: The parameters describing the simulated model.
        ignore_offset: Render the unshifted, unrolled geometry, ignoring the
            planted offset and roll.  The frame the navigator reads always
            carries them; a display of the scene's geometry may not want them.
        source: The label validation errors start with, such as the path of
            the scene file the parameters came from.

    Returns:
        A tuple containing the image and metadata.  Metadata keys: ``stars``
        (rendered star records), ``bodies``, ``rings``, ``inventory``,
        ``star_info``, ``body_masks``, ``ring_masks``, ``order_near_to_far``,
        ``body_index_map``, ``body_mask_map``, and ``body_occlusion``
        (per-body mutual-event truth: ``visible_fraction`` and
        ``occluded_limb_arc_deg`` against every nearer body).

    Raises:
        ~spindoctor.sim.scene_schema.SimSceneValidationError: If the scene fails validation.
    """
    scene_json = json.dumps(sim_params, sort_keys=True)
    _check_scene_json(scene_json, source)
    return _render(sim_params, scene_json, ignore_offset=ignore_offset)


def render_unvalidated_preview(
    sim_params: dict[str, Any],
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render a scene's offset-free geometry without validating it.

    This is the scene editor's live preview.  While the author edits, the scene
    often fails validation for a moment, and the preview should still show it.
    Nothing that navigates or measures a frame may use this function; every
    such caller goes through :func:`render_combined_model`.  A scene that the
    renderer cannot read raises whatever error the renderer meets.

    Parameters:
        sim_params: The parameters describing the simulated model.

    Returns:
        The image and metadata, as :func:`render_combined_model` returns them
        with ``ignore_offset=True``.
    """
    return _render(sim_params, json.dumps(sim_params, sort_keys=True), ignore_offset=True)


def _render(
    sim_params: dict[str, Any], scene_json: str, *, ignore_offset: bool
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render a scene through the cache and hand back private copies.

    Parameters:
        sim_params: The scene mapping.
        scene_json: ``sim_params`` as sorted-key JSON.
        ignore_offset: Whether to zero the planted offset and roll.

    Returns:
        Copies of the cached image and metadata.
    """
    render_key = scene_json
    if ignore_offset:
        # The cached function zeroes the planted offset and roll, so all three
        # keys stay out of the cache key.
        render_key = json.dumps(
            {k: v for k, v in sim_params.items() if k not in _PLANTED_POINTING_KEYS},
            sort_keys=True,
        )
    cached_img, cached_meta = _render_combined_model_cached(render_key, ignore_offset=ignore_offset)
    # Return copies to avoid cache modification
    # Use deepcopy for meta to fully isolate nested mutable structures
    return cached_img.copy(), copy.deepcopy(cached_meta)
