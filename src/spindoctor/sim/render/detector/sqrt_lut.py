"""The companding table a camera downlinks its DN through.

A companding table sends each DN as one of a fixed number of codes (256 for an
8-bit code word) and the ground decodes each code back to a DN.  A parametric
table's decode curve is quadratic in the code, so the encoding is a square root
and the quantization step grows with the signal, tracking the photon noise; a
tabulated table lists each code's decoded DN, so it can describe a curve of any
shape.  A camera lists its tables under ``downlink_tables`` in its instrument
entry and sets one as the default, and a scene may select another with
``detector.downlink_table``.  If a camera lists no table, it gets the plain
square-root curve scaled to its ADC ceiling.
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.sim.instruments.schema import DownlinkTable
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'DEFAULT_SQRT_LUT_CODES',
    'DownlinkCurve',
    'LUTSqrt',
    'LUTTabulated',
    'check_curve_ceiling',
    'resolve_sqrt_lut',
]

# The plain curve's table size: an 8-bit code word.
DEFAULT_SQRT_LUT_CODES = 256


@dataclass(frozen=True)
class LUTSqrt:
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


@dataclass(frozen=True)
class LUTTabulated:
    """The decode curve of a companding table that lists each code's decoded DN.

    The encoder sends each DN as the code whose decoded value is nearest.

    Parameters:
        decoded_dn: The DN each code decodes to, in code order.
    """

    decoded_dn: tuple[float, ...]

    def levels(self) -> NDArrayFloatType:
        """The distinct decoded DN values, in increasing order."""
        levels: NDArrayFloatType = np.unique(np.asarray(self.decoded_dn, dtype=np.float64))
        return levels


# The decode curve of either kind of table.
DownlinkCurve = LUTSqrt | LUTTabulated


def resolve_sqrt_lut(table: DownlinkTable | None, *, saturation_dn: float) -> DownlinkCurve:
    """The companding curve of a downlink table, or the plain square-root curve.

    Parameters:
        table: A downlink table from the instrument's entry, parametric or tabulated;
            None for an instrument that lists no table.
        saturation_dn: The ADC ceiling.  Without a table, the curve is a pure square
            root over 256 codes whose top code decodes to it: ``linear_dn`` 0,
            ``quadratic_dn`` ``saturation_dn / 255**2``, fractional DN kept.

    Returns:
        The resolved curve.
    """
    if table is None:
        top_code = DEFAULT_SQRT_LUT_CODES - 1
        return LUTSqrt(
            codes=DEFAULT_SQRT_LUT_CODES,
            linear_dn=0.0,
            quadratic_dn=saturation_dn / top_code**2,
            whole_dn=False,
        )
    if table.decoded_dn is not None:
        return LUTTabulated(decoded_dn=tuple(table.decoded_dn))
    # The catalog requires all four keys of a table that is not tabulated.
    assert table.codes is not None
    assert table.linear_dn is not None
    assert table.quadratic_dn is not None
    assert table.whole_dn is not None
    return LUTSqrt(
        codes=table.codes,
        linear_dn=table.linear_dn,
        quadratic_dn=table.quadratic_dn,
        whole_dn=table.whole_dn,
    )


def check_curve_ceiling(curve: DownlinkCurve, *, saturation_dn: float) -> None:
    """Refuse a decode curve whose top decoded value lies above the ADC ceiling.

    The quantizer snaps each DN, already clipped at the ceiling, to the nearest
    decoded value, so a value above the ceiling would put DN in the product that
    the ADC can never produce.

    Parameters:
        curve: A camera's downlink table, resolved.
        saturation_dn: The ADC ceiling the frame's DN is clipped at.

    Raises:
        ValueError: If the curve's largest decoded value exceeds ``saturation_dn``.
    """
    top_dn = float(curve.levels()[-1])
    if top_dn > saturation_dn:
        raise ValueError(
            f'the downlink table decodes up to {top_dn:g} DN, above the ADC ceiling '
            f'(saturation_dn) of {saturation_dn:g} DN'
        )
