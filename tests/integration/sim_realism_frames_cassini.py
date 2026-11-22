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

__all__ = ['cassini_clipped_pixels', 'cassini_frame_settings']

# The raw product's full scale, by the label's DATA_CONVERSION_TYPE.  A frame sent
# through the 12-to-8-bit table tops out at the table's last entry.
_RAW_FULL_SCALE_DN = {'12BIT': 4095, 'TABLE': 255}


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
    raw_path = FCPath(str(path.parent).replace('/calibrated/', '/volumes/', 1)) / (
        path.name.replace('_CALIB', '')
    )
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
        None; the clipped pixels are those :func:`cassini_clipped_pixels` gives.
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
    )


register_frame_settings_reader('COISS', cassini_frame_settings)
