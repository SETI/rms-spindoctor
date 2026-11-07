"""The square-root companding table a camera downlinks its DN through.

A companding table sends each DN as one of a fixed number of codes (256 for an
8-bit code word) and the ground decodes each code back to a DN.  The decode
curve is quadratic in the code, so the encoding is a square root and the
quantization step grows with the signal, tracking the photon noise.  A camera's
curve is a per-instrument catalog entry (``sqrt_lut`` in ``DETECTOR_DEFAULTS``);
a camera without one gets the plain square-root curve scaled to its ADC ceiling.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['DEFAULT_SQRT_LUT_CODES', 'SqrtLut', 'resolve_sqrt_lut']

# The plain curve's table size: an 8-bit code word.
DEFAULT_SQRT_LUT_CODES = 256


@dataclass(frozen=True)
class SqrtLut:
    """The decode curve of a companding table.

    Code ``c``, from 0 to ``codes - 1``, decodes to
    ``linear_dn * c + quadratic_dn * c**2`` DN, rounded to whole DN when
    ``whole_dn`` is set.  The encoder sends each DN as the code whose decoded
    value is nearest.

    Parameters:
        codes: The number of codes the table sends, 256 for an 8-bit code word.
        linear_dn: The curve's linear coefficient, DN per code.
        quadratic_dn: The curve's quadratic coefficient, DN per code squared.
        whole_dn: Whether each decoded value is rounded to a whole DN.
    """

    codes: int
    linear_dn: float
    quadratic_dn: float
    whole_dn: bool

    def levels(self) -> NDArrayFloatType:
        """The distinct decoded DN values, in increasing order."""
        codes = np.arange(self.codes, dtype=np.float64)
        dn = self.linear_dn * codes + self.quadratic_dn * codes**2
        if self.whole_dn:
            dn = np.round(dn)
        levels: NDArrayFloatType = np.unique(dn)
        return levels


def resolve_sqrt_lut(entry: Mapping[str, Any] | None, *, saturation_dn: float) -> SqrtLut:
    """The companding curve a catalog entry names, or the plain square-root curve.

    Parameters:
        entry: The instrument's ``sqrt_lut`` catalog entry, with the keys
            ``codes``, ``linear_dn``, ``quadratic_dn``, and ``whole_dn``; None
            for an instrument without a table of its own.
        saturation_dn: The ADC ceiling.  Without an entry, the curve is a pure
            square root over 256 codes whose top code decodes to it:
            ``linear_dn`` 0, ``quadratic_dn`` ``saturation_dn / 255**2``,
            fractional DN kept.

    Returns:
        The resolved curve.

    Raises:
        KeyError: If ``entry`` lacks one of the four keys.
        ValueError: If ``codes`` isn't a whole number of at least 2, or if
            ``linear_dn`` or ``quadratic_dn`` isn't finite.
    """
    if entry is None:
        top_code = DEFAULT_SQRT_LUT_CODES - 1
        return SqrtLut(
            codes=DEFAULT_SQRT_LUT_CODES,
            linear_dn=0.0,
            quadratic_dn=saturation_dn / top_code**2,
            whole_dn=False,
        )
    raw_codes = float(entry['codes'])
    if not raw_codes.is_integer():
        raise ValueError(f'sqrt_lut codes must be a whole number, got {raw_codes}')
    codes = int(raw_codes)
    if codes < 2:
        raise ValueError(f'sqrt_lut codes must be at least 2, got {codes}')
    linear_dn = float(entry['linear_dn'])
    quadratic_dn = float(entry['quadratic_dn'])
    # A NaN coefficient would turn every finite pixel the table snaps into NaN.
    if not (math.isfinite(linear_dn) and math.isfinite(quadratic_dn)):
        raise ValueError(
            f'sqrt_lut coefficients must be finite, got linear_dn {linear_dn} '
            f'and quadratic_dn {quadratic_dn}'
        )
    return SqrtLut(
        codes=codes,
        linear_dn=linear_dn,
        quadratic_dn=quadratic_dn,
        whole_dn=bool(entry['whole_dn']),
    )
