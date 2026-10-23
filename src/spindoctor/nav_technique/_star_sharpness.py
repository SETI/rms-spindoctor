"""PSF-sharpness test separating compact cosmic-ray hits from stars.

A cosmic ray deposits its charge in one to a few pixels with nothing around them; a star's
light is spread by the instrument's point-spread function, and smear, saturation, and
defocus only spread it further.  No star can therefore hold more of its light in its
brightest pixels than the narrowest star the camera records.  :func:`psf_concentration`
bounds that share for such a PSF, and :func:`sharper_than_psf` tests a candidate detection
against the bound.  ``StarFieldFromCatalogNav`` sets such candidates aside before choosing
the brightest detections it matches against the catalog.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

__all__ = ['PsfConcentration', 'psf_concentration', 'sharper_than_psf']

_SHARPNESS_BOX_HALF_PX = 2
"""Half-width of the box whose flux the PSF-sharpness test divides among its pixels.

A 5x5 box holds all but a few percent of the light of a star whose point-spread function
is a pixel or less wide, the width at which a star and a cosmic ray are confused."""

_BACKGROUND_RING_WIDTH_PX = 2
"""Width of the ring of pixels around the box whose median is the background."""

_PEAK_SEARCH_HALF_PX = 1
"""Half-width of the search for a candidate's brightest pixel about its detection.

A star's brightest pixel touches the peak of its matched-filter response; a wider
search could settle on a brighter cosmic-ray hit beside the star and judge the star by
the hit."""

_PSF_PHASE_STEPS = 11
"""Sub-pixel phases sampled per axis, over one pixel, to bound a PSF's concentration."""


@dataclass(frozen=True)
class PsfConcentration:
    """The largest share of a star's light its brightest pixels can hold.

    Each share is of the light inside the ``(2 * _SHARPNESS_BOX_HALF_PX + 1)``-pixel box
    centered on the star's brightest pixel, maximized over where in its pixel the star
    falls.  Smear, saturation, and defocus only spread a star's light, so no star is more
    concentrated than the narrowest point-spread function its camera records allows.

    Parameters:
        one_px: Share of the brightest pixel.
        two_px: Share of the brightest pixel and the brightest of its eight neighbors
            together.
        two_by_two_px: Share of the brightest 2x2 block holding the brightest pixel.
    """

    one_px: float
    two_px: float
    two_by_two_px: float


def psf_concentration(psf: PSF) -> PsfConcentration:
    """Return how concentrated a pixel-integrated point-spread function can be.

    The PSF is evaluated, pixel-integrated, at ``_PSF_PHASE_STEPS`` sub-pixel phases per
    axis across one pixel, and each share of :class:`PsfConcentration` is the largest
    found at any phase.

    Parameters:
        psf: The instrument's star point-spread function.

    Returns:
        The largest shares of the box flux held by the brightest pixel, the brightest
        pixel with the brightest of its eight neighbors, and the brightest 2x2 block.
    """
    half = _SHARPNESS_BOX_HALF_PX
    size = 2 * (half + 1) + 1
    one = two = quad = 0.0
    for phase_v in np.linspace(0.0, 1.0, _PSF_PHASE_STEPS):
        for phase_u in np.linspace(0.0, 1.0, _PSF_PHASE_STEPS):
            stamp = np.asarray(
                psf.eval_rect((size, size), offset=(float(phase_v), float(phase_u))),
                np.float64,
            )
            shares = _box_shares(stamp, half)
            if shares is None:
                continue
            one = max(one, shares.one_px)
            two = max(two, shares.two_px)
            quad = max(quad, shares.two_by_two_px)
    return PsfConcentration(one_px=one, two_px=two, two_by_two_px=quad)


@dataclass(frozen=True)
class _BoxFlux:
    """Flux sums of a source's box about its brightest pixel (see :func:`_box_flux`)."""

    total: float
    one_px: float
    two_px: float
    two_by_two_px: float


def _box_flux(excess: NDArrayFloatType, peak_v: int, peak_u: int, half: int) -> _BoxFlux | None:
    """Return the flux sums of the box of half-width ``half`` about ``(peak_v, peak_u)``.

    Parameters:
        excess: 2-D array of the light above the background.
        peak_v: Row of the brightest pixel.
        peak_u: Column of the brightest pixel.
        half: Half-width of the box.

    Returns:
        The box's total (negative pixels counted as zero, so a dark line or column across
        the box does not shrink it), and the flux of the pixel at ``(peak_v, peak_u)``, of
        that pixel with the brightest of its eight neighbors, and of the brightest 2x2
        block holding that pixel; None when the box does not fit inside ``excess``.
    """
    h, w = excess.shape
    if not (half <= peak_v < h - half and half <= peak_u < w - half):
        return None
    box = excess[peak_v - half : peak_v + half + 1, peak_u - half : peak_u + half + 1]
    peak = float(excess[peak_v, peak_u])
    # The neighbor must touch the brightest pixel: a star's second brightest pixel
    # always does, while a cosmic-ray hit or hot pixel elsewhere in the box is no part
    # of the star's light and must not make the star look concentrated.  A corner
    # neighbor counts, so a hit crossing a pixel corner is caught; a star's brightest
    # neighbor shares an edge, so admitting corners does not loosen its bound.
    near = excess[peak_v - 1 : peak_v + 2, peak_u - 1 : peak_u + 2].copy()
    near[1, 1] = -np.inf
    neighbor = float(near.max())
    quad = max(
        float(excess[peak_v + dv : peak_v + dv + 2, peak_u + du : peak_u + du + 2].sum())
        for dv in (-1, 0)
        for du in (-1, 0)
    )
    return _BoxFlux(
        total=float(np.clip(box, 0.0, None).sum()),
        one_px=peak,
        two_px=peak + neighbor,
        two_by_two_px=quad,
    )


def _box_shares(stamp: NDArrayFloatType, half: int) -> PsfConcentration | None:
    """Return the shares of a noiseless stamp's box held by its brightest pixels.

    Parameters:
        stamp: A noiseless source on a zero background.
        half: Half-width of the box about the brightest pixel.

    Returns:
        The shares, or None when the box about the brightest pixel leaves the stamp or
        holds no light.
    """
    peak_v, peak_u = (int(x) for x in np.unravel_index(int(np.argmax(stamp)), stamp.shape))
    flux = _box_flux(stamp, peak_v, peak_u, half)
    if flux is None or flux.total <= 0.0:
        return None
    return PsfConcentration(
        one_px=flux.one_px / flux.total,
        two_px=flux.two_px / flux.total,
        two_by_two_px=flux.two_by_two_px / flux.total,
    )


def sharper_than_psf(
    image_ext: NDArrayFloatType,
    v: int,
    u: int,
    *,
    limits: PsfConcentration,
    noise_sigma: float,
    detection_sigma: float,
) -> bool:
    """Return whether a candidate's light is more concentrated than any star's can be.

    The box of :class:`PsfConcentration` is taken about the brightest pixel within
    ``_PEAK_SEARCH_HALF_PX`` of ``(v, u)``, above the median of the ring of pixels
    ``_BACKGROUND_RING_WIDTH_PX`` wide around it.  For each of the brightest pixel, the
    two brightest pixels, and the brightest 2x2 block, the test forms the light those
    pixels hold beyond the largest share of the box total that ``limits`` allows a star.
    The two brightest pixels are the brightest pixel and the brightest of its eight
    neighbors, so a second source elsewhere in the box does not count toward them.  A
    cosmic ray deposits its charge in a few pixels with nothing around them, so one of
    the three exceeds its share; a star's light falls off over its neighbors.  The
    candidate is too sharp when any excess is more than ``detection_sigma`` times its
    noise, the per-pixel ``noise_sigma`` propagated through the difference.  A candidate
    whose box or ring leaves the image is never judged too sharp.

    Parameters:
        image_ext: 2-D extfov image array.
        v: Row of the candidate's detection, the peak of its matched-filter response.
        u: Column of the candidate's detection.
        limits: The largest shares a star can hold, from :func:`psf_concentration`.
        noise_sigma: Robust per-pixel noise sigma in DN units.
        detection_sigma: Noise multiplier an excess must exceed.

    Returns:
        True when the candidate is too concentrated to be a star.
    """
    half = _SHARPNESS_BOX_HALF_PX
    reach = half + _BACKGROUND_RING_WIDTH_PX
    h, w = image_ext.shape
    seek = _PEAK_SEARCH_HALF_PX
    v_lo, u_lo = max(0, v - seek), max(0, u - seek)
    near = np.asarray(image_ext[v_lo : min(h, v + seek + 1), u_lo : min(w, u + seek + 1)])
    bv, bu = (int(x) for x in np.unravel_index(int(np.argmax(near)), near.shape))
    peak_v, peak_u = v_lo + bv, u_lo + bu
    if not (reach <= peak_v < h - reach and reach <= peak_u < w - reach):
        return False
    region = np.asarray(
        image_ext[peak_v - reach : peak_v + reach + 1, peak_u - reach : peak_u + reach + 1],
        np.float64,
    )
    inner = slice(reach - half, reach + half + 1)
    ring = np.ones(region.shape, dtype=bool)
    ring[inner, inner] = False
    background = float(np.median(region[ring]))
    flux = _box_flux(region[inner, inner] - background, half, half, half)
    if flux is None:
        return False
    n_box = (2 * half + 1) ** 2
    for n_px, share, held in (
        (1, limits.one_px, flux.one_px),
        (2, limits.two_px, flux.two_px),
        (4, limits.two_by_two_px, flux.two_by_two_px),
    ):
        noise = noise_sigma * math.sqrt(n_px * (1.0 - share) ** 2 + (n_box - n_px) * share**2)
        if held - share * flux.total > detection_sigma * noise:
            return True
    return False
