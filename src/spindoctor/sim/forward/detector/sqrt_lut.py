"""The 8-bit square-root companding table a camera downlinks its DN through.

A companding table sends each 12-bit DN as one of 256 codes and the ground
decodes each code back to a DN.  The decode curve is quadratic in the code, so
the encoding is a square root and the quantization step grows with the signal,
tracking the photon noise.  A camera's curve is a per-instrument catalog entry
(``sqrt_lut`` in ``DETECTOR_DEFAULTS``); a camera without one gets the plain
square-root curve scaled to its ADC ceiling.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['SQRT_LUT_CODES', 'SqrtLut', 'resolve_sqrt_lut']

# An 8-bit code word.
SQRT_LUT_CODES = 256


@dataclass(frozen=True)
class SqrtLut:
    """The decode curve of an 8-bit companding table.

    Code ``c`` decodes to ``linear_dn * c + quadratic_dn * c**2`` DN, rounded to
    whole DN when ``whole_dn`` is set.  The encoder sends each DN as the code
    whose decoded value is nearest.

    Parameters:
        linear_dn: The curve's linear coefficient, DN per code.
        quadratic_dn: The curve's quadratic coefficient, DN per code squared.
        whole_dn: Whether each decoded value is rounded to a whole DN.
    """

    linear_dn: float
    quadratic_dn: float
    whole_dn: bool

    def levels(self) -> NDArrayFloatType:
        """The distinct decoded DN values, in increasing order."""
        codes = np.arange(SQRT_LUT_CODES, dtype=np.float64)
        dn = self.linear_dn * codes + self.quadratic_dn * codes**2
        if self.whole_dn:
            dn = np.round(dn)
        levels: NDArrayFloatType = np.unique(dn)
        return levels


def resolve_sqrt_lut(entry: Mapping[str, Any] | None, *, saturation_dn: float) -> SqrtLut:
    """The companding curve a catalog entry names, or the plain square-root curve.

    Parameters:
        entry: The instrument's ``sqrt_lut`` catalog entry, with the keys
            ``linear_dn``, ``quadratic_dn``, and ``whole_dn``; None for an
            instrument without a table of its own.
        saturation_dn: The ADC ceiling.  Without an entry, the top code decodes
            to it and the curve is a pure square root: ``linear_dn`` 0,
            ``quadratic_dn`` ``saturation_dn / 255**2``, fractional DN kept.

    Returns:
        The resolved curve.

    Raises:
        KeyError: If ``entry`` lacks one of the three keys.
    """
    if entry is None:
        top_code = SQRT_LUT_CODES - 1
        return SqrtLut(
            linear_dn=0.0, quadratic_dn=max(saturation_dn, 1.0) / top_code**2, whole_dn=False
        )
    return SqrtLut(
        linear_dn=float(entry['linear_dn']),
        quadratic_dn=float(entry['quadratic_dn']),
        whole_dn=bool(entry['whole_dn']),
    )
