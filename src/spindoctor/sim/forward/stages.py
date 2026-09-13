"""The stage interface of the forward-model rendering pipeline.

A render is a fixed-order sequence of stages, each a callable matching the
:class:`Stage` protocol, mutating a :class:`SimFrame` in place.  The order
(scene radiance, optics, downsample, detector, telemetry) is physical: light
is composed, passes through the camera optics, lands on the detector grid,
is read out, and is then transmitted.

Each stage receives its own ``numpy.random.Generator`` seeded from the
scene's single ``random_seed`` via
``derive_effect_seed(random_seed, '<stage-name>')``, so one stage's noise
realization is independent of which other stages are enabled.  A stage name
is therefore part of its scenes' noise realization: renaming a stage reseeds
it and regenerates the affected baselines.

A scene with an active whole-scene PSF renders its radiance on an oversampled
grid (``oversample`` > 1) so the convolution resolves sub-detector-pixel edge
structure; the box downsample after optics returns the image and the
pixel-space truth metadata to the detector grid.  A scene with no optics block
renders at ``oversample`` 1, where the downsample is a no-op.

The ``signal`` plane carries normalized [0, ~1] intensive scene units through
the optics stage; the detector stage converts it to electrons through the
exposure and digitizes it to DN in place (the electron unit chain).  The
``point_e`` plane carries the detector-native point sources (stars): electrons
for a CCD, added into the electron image after the signal conversion and before
Poisson so they never pass through the intensive scale; DN for the Voyager
vidicon (which has no electron domain), added onto the converted signal before
the DN-domain noise.  Both planes share every optical transform (PSF, smear,
distortion, ghosts), so a star's shape tracks the limb and ring-edge profiles.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['SimFrame', 'Stage', 'downsample_to_detector', 'new_sim_frame']


@dataclass
class SimFrame:
    """The mutable image state threaded through the rendering stages.

    Parameters:
        signal: ``(V*os, U*os)`` float64 image of intensive scene signal
            (I/F-like normalized units): bodies, rings, and diffuse
            backgrounds.  The detector stage converts it to DN in place.
        point_e: ``(V*os, U*os)`` float64 image of detector-native point
            sources (stars), kept separate because one array cannot carry two
            unit systems through the detector stage's signal-to-electron
            conversion.  Electrons for a CCD, DN for the vidicon (see the module
            docstring); zeroed on a scene with no stars.
        oversample: Oversampling factor ``os >= 1``; the detector grid is
            ``(V, U)``.
        truth: Feature truth accumulated by the radiance stage (the rendered
            star records, body masks, inventory, and z-order maps).  This is
            renderer output metadata; none of it crosses the information
            boundary to the navigator side.  Pixel-space entries are carried on
            the oversampled grid and returned to the detector grid by the
            downsample stage.
    """

    signal: NDArrayFloatType
    point_e: NDArrayFloatType
    oversample: int = 1
    truth: dict[str, Any] = field(default_factory=dict)


class Stage(Protocol):
    """One rendering stage: a pure in-place transform of a :class:`SimFrame`.

    Parameters:
        frame: The frame to mutate.
        params: The full validated scene ``sim_params`` mapping.  A stage
            whose scene block is absent is disabled and contributes nothing
            (per-stage parameter blocks land with their phases).
        rng: The stage's own seeded random generator.
    """

    def __call__(
        self,
        frame: SimFrame,
        *,
        params: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> None: ...


def new_sim_frame(size_v: int, size_u: int, *, oversample: int = 1) -> SimFrame:
    """Allocate a zeroed :class:`SimFrame` for a ``(size_v, size_u)`` detector.

    Parameters:
        size_v: Detector-grid height in pixels.
        size_u: Detector-grid width in pixels.
        oversample: Oversampling factor of the radiance grid.

    Returns:
        A frame with zeroed ``signal`` and ``point_e`` planes.
    """
    shape = (size_v * oversample, size_u * oversample)
    return SimFrame(
        signal=np.zeros(shape, dtype=np.float64),
        point_e=np.zeros(shape, dtype=np.float64),
        oversample=oversample,
    )


# Truth-metadata keys carrying a per-pixel array on the render grid.  The
# masks classify pixels, so they are sampled at each detector pixel's central
# subsample rather than averaged (an averaged bool has no meaning).
_TRUTH_MASK_KEYS: tuple[str, ...] = ('body_masks', 'ring_masks')
# Truth-metadata inventory-bbox fields in pixel units (scaled back by 1/os);
# the 'range' entry is a physical distance and is left unscaled.
_INVENTORY_PIXEL_KEYS: tuple[str, ...] = (
    'v_min_unclipped',
    'v_max_unclipped',
    'u_min_unclipped',
    'u_max_unclipped',
    'v_pixel_size',
    'u_pixel_size',
)
_STAR_INFO_PIXEL_KEYS: tuple[str, ...] = (
    'center_v',
    'center_u',
    'sigma',
    'psf_half_v',
    'psf_half_u',
    'catalog_error_v',
    'catalog_error_u',
)


def _center_subsample(array: Any, os: int) -> Any:
    """Sample the central subsample of each ``os x os`` detector-pixel block.

    Classifying arrays (boolean masks, integer index maps) cannot be averaged,
    so each detector pixel takes the value of the subsample nearest its centre.

    Parameters:
        array: A ``(V*os, U*os)`` array on the render grid.
        os: The oversampling factor.

    Returns:
        The ``(V, U)`` detector-grid array.
    """
    size_v = array.shape[0] // os
    size_u = array.shape[1] // os
    mid = os // 2
    return array.reshape(size_v, os, size_u, os)[:, mid, :, mid]


def downsample_to_detector(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Box-downsample the oversampled planes to the detector grid.

    The box filter is a mean over the ``os**2`` subsamples, so the intensive
    ``signal`` passes through unchanged in level.  The pixel-space truth
    metadata (body/ring masks, the body index map, inventory bounding boxes,
    and star hit-test records) is returned to the detector grid alongside the
    image: classifying arrays are sampled at each detector pixel's central
    subsample, and pixel-unit scalars are divided by ``os``.  At
    ``oversample == 1`` this stage is a no-op.

    Parameters:
        frame: The frame to downsample in place.
        params: The scene mapping (unused; downsampling has no scene knobs).
        rng: The stage generator (unused; downsampling is deterministic).
    """
    del params, rng
    os = frame.oversample
    if os == 1:
        return
    size_v = frame.signal.shape[0] // os
    size_u = frame.signal.shape[1] // os
    frame.signal = frame.signal.reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    # The point-source plane carries extensive electron weights scaled by
    # os**2 at deposition, so the mean conserves the per-star electron sum.
    frame.point_e = frame.point_e.reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    frame.oversample = 1
    _downsample_truth(frame.truth, os)


def _downsample_truth(truth: dict[str, Any], os: int) -> None:
    """Return the pixel-space truth metadata to the detector grid in place."""
    index_map = truth.get('body_index_map')
    if index_map is not None:
        truth['body_index_map'] = _center_subsample(index_map, os)
    for key in _TRUTH_MASK_KEYS:
        masks = truth.get(key)
        if masks is not None:
            truth[key] = [_center_subsample(mask, os) for mask in masks]
    mask_map = truth.get('body_mask_map')
    if isinstance(mask_map, dict):
        truth['body_mask_map'] = {
            name: _center_subsample(mask, os) for name, mask in mask_map.items()
        }
    inventory = truth.get('inventory')
    if isinstance(inventory, dict):
        for item in inventory.values():
            for key in _INVENTORY_PIXEL_KEYS:
                if key in item:
                    item[key] = item[key] / os
    star_info = truth.get('star_info')
    if star_info is not None:
        for info in star_info:
            for key in _STAR_INFO_PIXEL_KEYS:
                if key in info:
                    info[key] = info[key] / os
    # The rendered star records carry the same pixel-space fields the scaled
    # scene entries did (catalog position, motion vector, PSF window), so they
    # return to detector units alongside the hit-test entries.  The records are
    # per-render copies (see render_stars), so the render cache stays on the
    # oversampled grid.
    #
    # A record's position is oops uv, measured from the grid's own corner, so
    # it scales by a pure divide: uv 0 is the corner on either grid.  At
    # oversample 1 this is the identity, as it must be.
    stars = truth.get('stars')
    if stars is not None:
        for star in stars:
            star.v = star.v / os
            star.u = star.u / os
            star.move_v = star.move_v / os
            star.move_u = star.move_u / os
            # psf_size was scaled by an exact integer multiply, so this is exact.
            star.psf_size = (star.psf_size[0] // os, star.psf_size[1] // os)
