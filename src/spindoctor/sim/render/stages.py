"""The stage interface of the render pipeline.

A render is a fixed-order sequence of stages, each a callable matching the
:class:`Stage` protocol, mutating a :class:`SimFrame` in place.  The order
(scene radiance, optics, downsample, detector, telemetry) is physical: light
is composed, passes through the camera optics, lands on the detector grid,
is read out, and is then transmitted.

A stage receives only the frame and the scene.  Each stage derives every random stream
it needs by name with :func:`~spindoctor.sim.seeds.derive_effect_seed`, from the scene's
single ``random_seed`` and a name that begins with the stage the effect belongs to: the
detector stage's shot and read noise draw from the stream ``detector``, its hot pixels
from ``detector/hot_pixels``, the radiance stage's background stars from
``scene_radiance/background_stars``, and so on.  The downsample stage draws nothing.  A
stream may in turn seed its own named sub-streams: ``scene_radiance/craters`` seeds one
``body:<index>:<name>`` stream for each body, which seeds that body's ``relief``,
``albedo``, and ``pose_scatter`` streams and one ``crater:<k>`` stream for each crater.
A scene may also set ``noise_seed``.  The detector's per-frame streams then derive from
it in place of ``random_seed``, under the same names, so the same scene and camera are
read out with a new draw of the noise.

The derivation is a stable digest, so a stream's seed is the same in every process and
on every run.  A stream's seed is computed from nothing but its name and the seed it is
derived from, so one stage's realization is the same whichever other stages are
enabled, and adding a stream never reseeds the others.  A stream name is part of the
noise realization of every scene that uses it.  Renaming one reseeds those scenes and
moves their committed renders, so a stream name must never change.

A scene with an active whole-scene PSF renders its radiance on an oversampled
grid (``oversample`` > 1) so the convolution resolves sub-detector-pixel edge
structure; the box downsample after optics returns the image and the
pixel-space truth metadata to the detector grid.  A scene with no PSF (no
``optics.psf`` block, and no catalog kernel from ``instrument_defaults``) renders
at ``oversample`` 1, where the downsample is a no-op.  An explicit ``oversample``
key sets the factor either way.

The ``signal`` plane carries normalized intensive scene units through the optics
stage, where 1 is a white disc lit face-on and a bright law, spot, or haze can carry it
above 1; the detector stage converts it to electrons through the
exposure and digitizes it to DN in place (the electron unit chain).  The
``point_e`` plane carries the detector-native point sources (stars): electrons
for a CCD, added into the electron image after the signal conversion and before
Poisson so they never pass through the intensive scale; DN for the Voyager
vidicon (which has no electron domain), added onto the converted signal before
the DN-domain noise.  Both planes share every optical transform (PSF, smear,
distortion, ghosts), so a star's shape tracks the limb and ring-edge profiles.

A camera's optics spread light from just outside its field of view into the pixels at
its edge, and spread the light at the edge outward.  The planes therefore cover more
than the frame: a margin on every side, as wide as the optics stage reaches (see
:func:`~spindoctor.sim.render.optics.stage.optics_margin_px`).  The radiance stage
draws the scene across the margin as well as the frame, the optics stage spreads light
across both, and the downsample stage cuts the frame out before it averages.  A scene
whose optics spread no light has no margin.  The truth metadata describes the frame
alone: the radiance stage cuts its per-pixel arrays to the frame and states every
position in frame pixels.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['SimFrame', 'Stage', 'crop_to_frame', 'downsample_to_detector', 'new_sim_frame']


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
        preview: Whether this is the scene editor's fast preview, which simplifies
            the full render in three ways.  The frame is built at oversample 1,
            whatever the scene asks for.  The radiance stage reads this flag and
            draws each body's limb without anti-aliasing, and the optics stage reads
            it and skips the distortion warp.  Nothing but the editor's preview sets
            it (see :func:`~spindoctor.sim.render.driver.render_scene`).
        margin: How many render-grid pixels the planes extend past the frame on each
            side, a whole number of detector pixels (see the module docstring).  The
            frame's first pixel is plane pixel ``(margin, margin)``.  The downsample
            stage cuts the frame out and sets it to 0.
    """

    signal: NDArrayFloatType
    point_e: NDArrayFloatType
    oversample: int = 1
    truth: dict[str, Any] = field(default_factory=dict)
    preview: bool = False
    margin: int = 0

    @property
    def frame_shape(self) -> tuple[int, int]:
        """The frame's rows and columns on the render grid, without the margin."""
        size_v, size_u = self.signal.shape
        return size_v - 2 * self.margin, size_u - 2 * self.margin

    @property
    def frame_box(self) -> tuple[slice, slice]:
        """The rows and columns of the planes that hold the frame."""
        size_v, size_u = self.frame_shape
        return (
            slice(self.margin, self.margin + size_v),
            slice(self.margin, self.margin + size_u),
        )


class Stage(Protocol):
    """One rendering stage: a transform that mutates a :class:`SimFrame` in place.

    A stage seeds its own random streams by name from the scene seed (see the module
    docstring).

    Parameters:
        frame: The frame to mutate.
        params: The full validated scene ``sim_params`` mapping.  A stage
            whose scene block is absent is disabled and contributes nothing.
    """

    def __call__(self, frame: SimFrame, *, params: Mapping[str, Any]) -> None: ...


def new_sim_frame(
    size_v: int,
    size_u: int,
    *,
    oversample: int = 1,
    preview: bool = False,
    margin_px: int = 0,
) -> SimFrame:
    """Allocate a zeroed :class:`SimFrame` for a ``(size_v, size_u)`` detector.

    Parameters:
        size_v: Detector-grid height in pixels.
        size_u: Detector-grid width in pixels.
        oversample: Oversampling factor of the radiance grid.
        preview: Whether the frame is the scene editor's fast preview.
        margin_px: How many detector pixels the planes extend past the frame on
            each side.

    Returns:
        A frame with zeroed ``signal`` and ``point_e`` planes.
    """
    margin = int(margin_px) * oversample
    shape = (size_v * oversample + 2 * margin, size_u * oversample + 2 * margin)
    return SimFrame(
        signal=np.zeros(shape, dtype=np.float64),
        point_e=np.zeros(shape, dtype=np.float64),
        oversample=oversample,
        preview=preview,
        margin=margin,
    )


def crop_to_frame(frame: SimFrame) -> None:
    """Cut the frame out of the planes, dropping the margin, in place.

    A frame without a margin is left as it is.

    Parameters:
        frame: The frame whose planes are cut.
    """
    if frame.margin == 0:
        return
    box = frame.frame_box
    frame.signal = np.ascontiguousarray(frame.signal[box])
    frame.point_e = np.ascontiguousarray(frame.point_e[box])
    frame.margin = 0


# Truth-metadata keys carrying a per-pixel array on the render grid.  The
# masks classify pixels, so they are reduced by how much of each detector
# pixel they cover rather than averaged (an averaged bool has no meaning).
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


def _covering_mask(mask: Any, os: int) -> Any:
    """Reduce a boolean render-grid mask to the detector grid by coverage.

    A detector pixel is true when at least half of the ``os * os`` subsamples of
    its block are true.  The count is compared as whole subsamples, so a block
    covered exactly half is true at every ``os``.  The rule is symmetric about
    the block, which is what keeps a reduced mask registered with the
    box-averaged image it describes: no subsample is preferred over the one
    mirroring it, so a straight edge sweeping across a block turns the pixel on
    as it passes the block's middle rather than a fixed fraction of a pixel to
    one side of it.

    Parameters:
        mask: A ``(V*os, U*os)`` boolean array on the render grid.
        os: The oversampling factor.

    Returns:
        The ``(V, U)`` detector-grid boolean array.
    """
    size_v = mask.shape[0] // os
    size_u = mask.shape[1] // os
    counts = np.count_nonzero(mask.reshape(size_v, os, size_u, os), axis=(1, 3))
    return 2 * counts >= os * os


def _covering_index_map(index_map: Any, os: int) -> Any:
    """Reduce the body index map to the detector grid by coverage.

    Each detector pixel takes the body covering the most subsamples of its
    block.  The labels run from 1 upward in near-to-far order with 0 for no
    body, so a tie between two bodies goes to the nearer one, and the no-body
    label wins only where it strictly covers more of the block than any single
    body does.  A block one body covers exactly half therefore comes out as that
    body, which is the answer :func:`_covering_mask` gives the same block.

    That agreement does not extend to every block, because the two answer
    different questions: this one asks which body covers the most of a block,
    and :func:`_covering_mask` asks whether one named body covers half of it.
    Where several bodies share a block and none reaches half, the block still
    has a most-covering body while every one of their masks is false.  Nor are
    the two the same quantity before either is reduced: a mask is a body's whole
    silhouette, while the index map holds only the nearest body at each
    subsample, so they already differ wherever one body hides another.  Read the
    index map for which body a detector pixel shows and a mask for how far one
    body reaches; a caller wanting them to mean the same thing wants the index
    map twice.

    Parameters:
        index_map: A ``(V*os, U*os)`` integer array on the render grid.
        os: The oversampling factor.

    Returns:
        The ``(V, U)`` detector-grid integer array.
    """
    size_v = index_map.shape[0] // os
    size_u = index_map.shape[1] // os
    dtype = index_map.dtype
    best_count = np.zeros((size_v, size_u), dtype=np.int64)
    best_label = np.zeros((size_v, size_u), dtype=dtype)
    covered = np.zeros((size_v, size_u), dtype=np.int64)
    for label in sorted(int(value) for value in np.unique(index_map) if int(value) != 0):
        counts = np.count_nonzero((index_map == label).reshape(size_v, os, size_u, os), axis=(1, 3))
        covered += counts
        wins = counts > best_count
        best_count = np.where(wins, counts, best_count)
        best_label = np.where(wins, np.asarray(label, dtype=dtype), best_label)
    empty = os * os - covered
    return np.where(empty > best_count, np.asarray(0, dtype=dtype), best_label)


def downsample_to_detector(frame: SimFrame, *, params: Mapping[str, Any]) -> None:
    """Cut the frame out of the planes, then box-downsample it to the detector grid.

    The margin the planes carry past the frame (see :class:`SimFrame`) is dropped
    first, so only the frame reaches the detector.  The box filter is a mean over
    the ``os**2`` subsamples, so the intensive ``signal`` passes through unchanged
    in level.  The pixel-space truth
    metadata (body/ring masks, the body index map, inventory bounding boxes,
    and star hit-test records) is returned to the detector grid alongside the
    image: classifying arrays are reduced by coverage (see
    :func:`_covering_mask` and :func:`_covering_index_map`), and pixel-unit
    scalars are divided by ``os``.  At ``oversample == 1`` the downsample is a
    no-op.

    Parameters:
        frame: The frame to downsample in place.
        params: The scene mapping.  Unused, since downsampling has no scene knobs; it
            is accepted because every stage takes the same arguments.
    """
    del params
    os = frame.oversample
    if os == 1:
        crop_to_frame(frame)
        return
    # The frame is averaged straight from the planes: a view of the frame's rows and
    # columns reshapes into blocks without a copy, and its mean sums each block in
    # the order a copied frame's would, so the result is the same to the last bit.
    box = frame.frame_box
    size_v = frame.frame_shape[0] // os
    size_u = frame.frame_shape[1] // os
    frame.signal = frame.signal[box].reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    # The point-source plane carries extensive electron weights scaled by
    # os**2 at deposition, so the mean conserves the per-star electron sum.
    frame.point_e = frame.point_e[box].reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    frame.margin = 0
    frame.oversample = 1
    _downsample_truth(frame.truth, os)


def _downsample_truth(truth: dict[str, Any], os: int) -> None:
    """Return the pixel-space truth metadata to the detector grid in place."""
    index_map = truth.get('body_index_map')
    if index_map is not None:
        truth['body_index_map'] = _covering_index_map(index_map, os)
    for key in _TRUTH_MASK_KEYS:
        masks = truth.get(key)
        if masks is not None:
            truth[key] = [_covering_mask(mask, os) for mask in masks]
    mask_map = truth.get('body_mask_map')
    if isinstance(mask_map, dict):
        truth['body_mask_map'] = {name: _covering_mask(mask, os) for name, mask in mask_map.items()}
    inventory = truth.get('inventory')
    if isinstance(inventory, dict):
        for item in inventory.values():
            for key in _INVENTORY_PIXEL_KEYS:
                if key in item:
                    item[key] = item[key] / os
    # Every hit-test entry scales by a pure divide.  The sizes and displacements
    # do so because that is what a length does.  ``center_v`` and ``center_u``
    # do so because the renderer states them as the DETECTOR pixel centric
    # position times ``os`` and lands the deposit on the oversampled grid with
    # its own half-subsample shift (see render_stars' grid_shift); reading them
    # as an oversampled-grid pixel centric position and taking a further half
    # pixel off would move every star by 0.5 * (1 - 1 / os) detector pixels.
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
    # A record's position is a pixel corner coordinate, measured from the grid's own corner, so
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
