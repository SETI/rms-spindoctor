"""Entry point for rendering a simulated scene: params in, image + meta out.

``render_combined_model`` is a thin, cached driver over the forward-model
stage pipeline (:mod:`spindoctor.sim.forward`): it normalizes the scene
parameters into a deterministic JSON cache key, runs the pipeline on a fresh
:class:`~spindoctor.sim.forward.stages.SimFrame`, and returns the rendered
image with the renderer's output metadata.

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
from spindoctor.support.types import NDArrayFloatType


def clear_render_caches() -> None:
    """Drop every render-path lru_cache so RNG and shape paths re-run.

    Test and tooling helper: the render caches are parameter-keyed and never
    stale in production, but determinism tests must re-execute the cached
    paths to prove reproducibility.
    """
    from spindoctor.sim.forward import body, body_mesh, psf, relief, star

    _render_combined_model_cached.cache_clear()
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
        sim_params['offset_v'] = 0.0
        sim_params['offset_u'] = 0.0
        sim_params['offset_rotation_deg'] = 0.0
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


def render_combined_model(
    sim_params: dict[str, Any], *, ignore_offset: bool = False
) -> tuple[NDArrayFloatType, dict[str, Any]]:
    """Render stars then bodies from a full sim_params dict. Returns (img, meta).

    ignore_offset = True should be used when rendering the image in the GUI, but not
    when creating the simulated image to navigate.

    Parameters:
        sim_params: The parameters describing the simulated model.
        ignore_offset: Whether to ignore the offset.

    Returns:
        A tuple containing the image and metadata.  Metadata keys: ``stars``
        (rendered star records), ``bodies``, ``rings``, ``inventory``,
        ``star_info``, ``body_masks``, ``ring_masks``, ``order_near_to_far``,
        ``body_index_map``, ``body_mask_map``, and ``body_occlusion``
        (per-body mutual-event truth: ``visible_fraction`` and
        ``occluded_limb_arc_deg`` against every nearer body).
    """
    # Create cache key from parameters.  When ignore_offset is True the cached
    # function zeroes the planted offset AND roll, so all three keys are
    # excluded from the cache key (previews differing only in planted pointing
    # share one render).
    params_for_hash = dict(sim_params)
    if ignore_offset:
        params_for_hash = {
            k: v
            for k, v in params_for_hash.items()
            if k not in ('offset_v', 'offset_u', 'offset_rotation_deg')
        }
    sim_params_json = json.dumps(params_for_hash, sort_keys=True)
    cached_img, cached_meta = _render_combined_model_cached(
        sim_params_json, ignore_offset=ignore_offset
    )
    # Return copies to avoid cache modification
    # Use deepcopy for meta to fully isolate nested mutable structures
    return cached_img.copy(), copy.deepcopy(cached_meta)
