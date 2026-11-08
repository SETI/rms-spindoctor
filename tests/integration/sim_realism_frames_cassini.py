"""Read a real Cassini ISS frame's commanded settings for its matched scene.

The realism match renders each cohort frame's matched scene through the frame's own
filter throughput, gain state, and downlink table, so the simulated DN-to-I/F scale
and quantization step are the real product's.  These are facts of the real frame's
label, which this module reads for a Cassini ISS frame and registers as the reader
of the ``COISS`` cohort's frames (see
:func:`tests.integration.sim_realism_support.register_frame_settings_reader`).
"""

from __future__ import annotations

import warnings
from typing import Any

from spindoctor.sim.forward.filter_catalog import canonical_filter, cataloged_filters
from tests.integration.sim_realism_support import FrameSettings, register_frame_settings_reader

__all__ = ['cassini_frame_settings']


def cassini_frame_settings(obs: Any, *, image_id: str, instrument: str) -> FrameSettings:
    """Read the filter, gain state, and downlink table of a Cassini ISS frame.

    Parameters:
        obs: The real observation, with the ``filter1``, ``filter2``, ``gain_mode``,
            and ``dict`` (the VICAR label) of a Cassini ISS frame.
        image_id: The frame's image ID, for the warning about an uncataloged filter.
        instrument: The sim instrument the frame is matched against.

    Returns:
        The frame's settings.  The filter is None when the instrument does not
        catalog the frame's combination (a warning names the frame, and its matched
        scene renders at the nominal scale); the quantization is ``'sqrt_lut'`` for
        a frame downlinked through the 12-to-8-bit table, else None.
    """
    filter_name: str | None = f'{obs.filter1}+{obs.filter2}'
    if canonical_filter(instrument, str(filter_name)) not in cataloged_filters(instrument):
        # An uncataloged pair renders at the nominal scale rather than aborting
        # the whole match; the warning names the frame.
        warnings.warn(
            f'{image_id}: filter {filter_name} is not cataloged for {instrument}; '
            'its matched scene uses the nominal scale',
            stacklevel=3,
        )
        filter_name = None
    quantization = 'sqrt_lut' if obs.dict.get('DATA_CONVERSION_TYPE', None) == 'TABLE' else None
    return FrameSettings(
        filter_name=filter_name, gain_state=int(obs.gain_mode), quantization=quantization
    )


register_frame_settings_reader('COISS', cassini_frame_settings)
