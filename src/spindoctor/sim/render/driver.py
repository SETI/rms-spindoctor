"""Entry point for rendering a simulated scene: params in, image + meta out.

``render_scene`` is a thin, cached driver over the render pipeline
(:mod:`spindoctor.sim.render.pipeline`): it encodes the scene as
deterministic sorted-key JSON, validates the scene, runs the pipeline on a fresh
:class:`~spindoctor.sim.render.stages.SimFrame` (cached on a key built from
that JSON), and returns the rendered image with the renderer's output metadata.

Every render validates its scene with
:func:`~spindoctor.sim.scene.validate.validate_sim_params`, so a scene built in memory
is held to the same rules as one read from a file.  The scene editor's fast
preview is one such render: it validates its scene like every other caller.

Validation goes through :func:`~spindoctor.sim.scene.validate.validate_scene`, whose cache
loading a scene file shares, so a scene that is loaded and then rendered, or
rendered many times, is validated once.  Its key is not quite the render cache
key: an offset-free render leaves the planted pointing out of its key, and
validation never does.

Callers are ``ObsSim``, the scene-editor GUI, and ``sim/png_export.py``
(which the doc-gallery and sweep runners drive).  The params-JSON caching
contract: scene parameters are JSON-serializable scalars/lists/maps, no
wall-clock or global RNG state is consulted, and two renders of the same
scene are bit-identical on one machine.

The scene editor renders on up to three threads at once (its fast preview on the
UI thread, and its full render and the frame the navigator reads on two workers),
which relies on three properties of the render path: every render builds its own
frame, parameter dict, and RNG streams; the ``lru_cache`` tables stay coherent
under concurrent calls; and no caller writes into a cached value (this function
hands out copies, and the stage caches' callers only read theirs).  A change that
breaks one of them breaks the editor.
"""

import copy
import json
from functools import lru_cache
from typing import Any

from spindoctor.sim.instruments.navigator_link import PLANTED_POINTING_KEYS
from spindoctor.sim.render.optics.stage import instrument_defaults_on
from spindoctor.sim.render.pipeline import run_pipeline
from spindoctor.sim.render.stages import new_sim_frame
from spindoctor.sim.scene.validate import clear_validation_cache, validate_scene
from spindoctor.support.types import NDArrayFloatType


def clear_render_caches() -> None:
    """Drop every render-path lru_cache so RNG and shape paths re-run.

    Test and tooling helper: the render caches are parameter-keyed and never
    stale in production, but determinism tests must re-execute the cached
    paths to prove reproducibility.
    """
    from spindoctor.sim.render.bodies import mesh as bodies_mesh
    from spindoctor.sim.render.bodies import relief
    from spindoctor.sim.render.bodies import render as bodies_render
    from spindoctor.sim.render.optics import psf
    from spindoctor.sim.render.radiance import stars as radiance_stars

    _render_scene_cached.cache_clear()
    clear_validation_cache()
    radiance_stars._render_stars_cached.cache_clear()
    radiance_stars._render_sky_counts_cached.cache_clear()
    bodies_render._render_ellipsoid_shape_cached.cache_clear()
    bodies_mesh._render_mesh_shape_cached.cache_clear()
    relief.clear_relief_caches()
    psf.psf_kernel.cache_clear()


# Three entries: the scene editor renders a scene's fast preview, its full
# offset-free render, and the frame the navigator reads (planted offset applied).
# An offset-only edit then re-renders only the last.  The preview flag is part of
# the key, so a preview is never handed out for a full render.
@lru_cache(maxsize=3)
def _render_scene_cached(
    sim_params_json: str,
    *,
    ignore_offset: bool,
    preview: bool,
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render a scene from its sorted-key JSON through the pipeline, cached."""
    sim_params = json.loads(sim_params_json)
    if ignore_offset:
        # The editor's display: render the unshifted, unrolled geometry.
        for key in PLANTED_POINTING_KEYS:
            sim_params[key] = 0.0
    size_v = int(sim_params['size_v'])
    size_u = int(sim_params['size_u'])

    # A whole-scene PSF is applied on an oversampled radiance grid so limb, ring
    # edge, and star profiles carry sub-detector-pixel structure into the
    # convolution; the box downsample after optics returns to the detector grid.
    # A scene with no optics block renders at oversample 1 (identical to a
    # frame with no sub-pixel optics to resolve).  The editor's fast preview
    # always renders at oversample 1: drawing on a grid 4 times finer in each
    # direction costs 16 times the pixels, for detail its display can't show.
    oversample = 1 if preview else resolve_oversample(sim_params)
    frame = new_sim_frame(size_v, size_u, oversample=oversample, preview=preview)
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
    return 4 if (psf_active or instrument_defaults_on(sim_params)) else 1


def render_scene(
    sim_params: dict[str, Any],
    *,
    ignore_offset: bool = False,
    preview: bool = False,
    source: str = 'sim_params',
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Validate a scene, then render it. Returns (img, meta).

    The scene is checked with :func:`~spindoctor.sim.scene.validate.validate_sim_params`
    before anything renders, so an invalid scene raises the same error a scene
    file would raise when loaded.  Validation is cached on the scene's full JSON
    and ``source``, so rendering one scene repeatedly validates it once.

    Parameters:
        sim_params: The scene.
        ignore_offset: Render the unshifted, unrolled geometry, ignoring the
            planted offset and roll.  The frame the navigator reads always
            carries them; a display of the scene's geometry may not want them.
        preview: Render the scene editor's fast preview, which simplifies the full
            render in three ways: the radiance is drawn on the detector grid, with no
            oversampling; each body's limb is drawn without anti-aliasing; and the
            optics stage skips the distortion warp.  The PSF and every other stage
            still run.  The result is deterministic and has the full render's image
            shape, but it is an approximation.  Only the scene editor's on-screen
            preview may ask for it: every image that leaves the editor, and every
            test, sweep, and baseline, uses the full render.
        source: The label that begins each validation error message, such as
            the path of the scene file the parameters came from.

    Returns:
        A tuple containing the image and metadata.  Metadata keys: ``stars``
        (rendered star records), ``bodies``, ``rings``, ``inventory``,
        ``star_info``, ``body_masks``, ``ring_masks``, ``order_near_to_far``,
        ``body_index_map``, ``body_mask_map``, and ``body_occlusion``
        (per-body mutual-event truth: ``visible_fraction`` and
        ``occluded_limb_arc_deg`` against every nearer body).

    Raises:
        ~spindoctor.sim.scene.errors.SimSceneValidationError: If the scene fails
            validation.
    """
    validate_scene(sim_params, source=source)
    return _render(
        sim_params,
        json.dumps(sim_params, sort_keys=True),
        ignore_offset=ignore_offset,
        preview=preview,
    )


def _render(
    sim_params: dict[str, Any], scene_json: str, *, ignore_offset: bool, preview: bool
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render a scene through the cache and hand back private copies.

    Parameters:
        sim_params: The scene mapping.
        scene_json: ``sim_params`` as sorted-key JSON.
        ignore_offset: Whether to zero the planted offset and roll.
        preview: Whether to render the scene editor's fast preview.

    Returns:
        Copies of the cached image and metadata.
    """
    render_key = scene_json
    if ignore_offset:
        # The cached function zeroes the planted offset and roll, so all three
        # keys stay out of the cache key.
        render_key = json.dumps(
            {k: v for k, v in sim_params.items() if k not in PLANTED_POINTING_KEYS},
            sort_keys=True,
        )
    cached_img, cached_meta = _render_scene_cached(
        render_key, ignore_offset=ignore_offset, preview=preview
    )
    # Return copies to avoid cache modification
    # Use deepcopy for meta to fully isolate nested mutable structures
    return cached_img.copy(), copy.deepcopy(cached_meta)
