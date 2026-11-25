"""Tell the realism match how a real Cassini ISS frame was commanded.

The realism match (:mod:`tests.integration.sim_realism`) checks the simulator against
real images. For each real frame in the image-library cohort it renders a simulated
twin, then compares the two: sky noise, star and limb profiles, artifact rates, and so
on. The comparison is only fair if the twin went through the same camera settings as
the real frame. A frame taken through a narrowband filter collects far fewer electrons
than one taken through the clear filters, a higher gain state turns each electron into
more DN, and a frame sent home through the 12-to-8-bit downlink table has much coarser
DN steps than a 12-bit frame. If the twin ignored any of these, its noise would differ
from the real frame's for reasons that have nothing to do with how well the simulator
models the camera.

Those settings are recorded in each frame's label. This module reads three of them for
a Cassini frame:

* the filter pair, from the two filter wheels (``CL1+CL2``, ``CL1+GRN``, ...);
* the gain state;
* whether the frame was downlinked through the table (``DATA_CONVERSION_TYPE`` is
  ``TABLE``), in which case the twin is quantized with ``sqrt_lut``.

If the camera's config entry has no throughput for the filter pair, the twin renders at
the camera's nominal scale and a warning gives the frame's image ID. One odd filter
shouldn't stop the whole match.

It also says which pixels of a calibrated frame the camera clipped. Calibration
rescales every pixel, so a clipped disc no longer reads one exact level, but the raw
product beside it still does: 4095 for a 12-bit frame and 255 for a frame sent through
the table. The raw product lies under ``volumes/`` where the calibrated one lies under
``calibrated/``, with ``_CALIB`` dropped from its name.

Last, it gives the radii the real limbs' silhouettes are measured against for the
mid-sized moons. The Cassini kernels the frames load give older radii than the IAU's,
and for Tethys and Rhea the older ones are 1 to 3 km larger, which put a real limb up to
a pixel inside its silhouette.

Reading a label is mission-specific, so it lives here and not in the shared realism
code. When this module is imported, it registers its reader for the ``COISS`` cohort.
The shared code imports every ``sim_realism_frames_<mission>.py`` beside it and picks
the reader by the cohort's mission (see
:func:`tests.integration.sim_realism_support.frame_settings_reader`). A mission with
no such module renders its twins at the camera's default settings. Adding one for
another mission takes only a new file with the same pattern.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import vicar
from filecache import FCPath

from spindoctor.sim.instruments.resolvers import canonical_filter, cataloged_filters
from spindoctor.support.types import NDArrayBoolType
from tests.integration.sim_realism_support import FrameSettings, register_frame_settings_reader

__all__ = ['IAU_MOON_RADII_KM', 'cassini_clipped_pixels', 'cassini_frame_settings']

# The raw product's full scale, by the label's DATA_CONVERSION_TYPE.  A frame sent
# through the 12-to-8-bit table tops out at the table's last entry.  If a real gain-2
# frame fills its full well below the ADC limit (#854), a disc clipped there plateaus
# lower and is not caught.  On 2026-11-22 no NAC or WAC limb disc whose highest DN fell
# short of these levels had more than 4 pixels at that DN, so none plateaus there.
_RAW_FULL_SCALE_DN = {'12BIT': 4095, 'TABLE': 255}

# The radii (a, b, c) in km of the mid-sized moons from NAIF's pck00010.tpc and
# pck00011.tpc, the IAU values from Thomas (2010, Icarus 208, 395-401).  The frames load
# the Cassini PCK cpck15Dec2017_saturn_only.tpc after NAIF's, and it gives the Thomas et
# al. (2007, Icarus 190, 573-584) radii: Tethys 540.4 x 531.1 x 527.5 and Rhea 767.2 x
# 762.5 x 763.1.  Against those, four of the cohort's five real Tethys limbs lay 0.24%
# to 0.35% of their radius inside their matched limbs; against these, all five lie within
# 0.19 px of their matched limbs.  Only the realism match's silhouettes use
# these radii; real navigation still fits the Cassini PCK's (#1216).
IAU_MOON_RADII_KM: dict[str, tuple[float, float, float]] = {
    'MIMAS': (207.8, 196.7, 190.6),
    'ENCELADUS': (256.6, 251.4, 248.3),
    'TETHYS': (538.4, 528.3, 526.3),
    'DIONE': (563.4, 561.3, 559.6),
    'RHEA': (765.0, 763.1, 762.4),
}
"""Radii in km, longest first, of the moons whose real silhouettes take the IAU values."""


def cassini_clipped_pixels(image_path: Any, conversion: str | None) -> NDArrayBoolType | None:
    """Which pixels of a calibrated Cassini ISS frame read full scale in its raw product.

    Parameters:
        image_path: The calibrated product's path (``..._CALIB.IMG``), as an
            :class:`~filecache.FCPath` or a string.
        conversion: The frame's ``DATA_CONVERSION_TYPE``.

    Returns:
        A mask in the frame's shape, or None when the product is not a calibrated one
        or its conversion type has no single full scale.
    """
    path = FCPath(image_path)
    full_scale = _RAW_FULL_SCALE_DN.get(str(conversion))
    if '_CALIB' not in path.name or full_scale is None:
        return None
    # Built from the path's own parents so the raw path keeps its file cache and URL
    # translators.  The topmost calibrated/ directory becomes volumes/.
    raw_dir = path.parent
    calibrated = [p for p in path.parent.parents if p.name == 'calibrated']
    if calibrated:
        raw_dir = calibrated[-1].parent / 'volumes' / path.parent.relative_to(calibrated[-1])
    raw_path = raw_dir / path.name.replace('_CALIB', '')
    raw = np.asarray(vicar.VicarImage(raw_path).data_2d)
    return np.asarray(raw >= full_scale, dtype=bool)


def cassini_frame_settings(
    obs: Any, *, image_id: str, instrument: str, image_path: Any
) -> FrameSettings:
    """Read the filter, gain state, downlink table, and clipped pixels of a Cassini frame.

    Parameters:
        obs: The real observation, with the ``filter1``, ``filter2``, ``gain_mode``,
            and ``dict`` (the VICAR label) of a Cassini ISS frame.
        image_id: The frame's image ID, for the warning about an uncataloged filter.
        instrument: The sim instrument the frame is matched against.
        image_path: The product's resolved path, beside which the raw product of a
            calibrated frame lies.

    Returns:
        The frame's settings.  The filter is None when the instrument does not
        catalog the frame's combination (a warning gives the frame's image ID, and
        its matched scene renders at the nominal scale); the quantization is
        ``'sqrt_lut'`` for a frame downlinked through the 12-to-8-bit table, else
        None; the clipped pixels are those :func:`cassini_clipped_pixels` gives; the
        silhouette radii are :data:`IAU_MOON_RADII_KM`.
    """
    filter_name: str | None = f'{obs.filter1}+{obs.filter2}'
    if canonical_filter(instrument, str(filter_name)) not in cataloged_filters(instrument):
        # An uncataloged pair renders at the nominal scale rather than aborting
        # the whole match; the warning gives the frame's image ID.
        warnings.warn(
            f'{image_id}: filter {filter_name} is not cataloged for {instrument}; '
            'its matched scene uses the nominal scale',
            stacklevel=3,
        )
        filter_name = None
    conversion = obs.dict.get('DATA_CONVERSION_TYPE', None)
    quantization = 'sqrt_lut' if conversion == 'TABLE' else None
    return FrameSettings(
        filter_name=filter_name,
        gain_state=int(obs.gain_mode),
        quantization=quantization,
        clipped=cassini_clipped_pixels(image_path, conversion),
        silhouette_radii_km=IAU_MOON_RADII_KM,
    )


register_frame_settings_reader('COISS', cassini_frame_settings)
