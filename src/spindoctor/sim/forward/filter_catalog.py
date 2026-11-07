"""Per-filter throughput of the simulated cameras that carry a filter wheel.

A scene's optional ``detector.filter`` key names the filter combination the
frame was shot through, written as the product label writes it (``'CL1+CL2'``
for the Cassini ISS wheel-1 and wheel-2 elements).  The catalog value is the
electrons per second one native pixel collects from a surface at I/F 1.0, so
the detector chain renders a signal of 1.0 at that rate and the calibrated
products divide by the same scale.  A scene that does not name a filter keeps
the nominal scale its detector entry derives (``signal_full_scale_frac *
full_well_e / exposure_ref_sec``).

The throughput is the only filter-dependent quantity the simulator models.
Haze, body photometry, star color, and the PSF render the same for every
filter.
"""

from __future__ import annotations

from spindoctor.sim.forward.artifact_modes import normalize_instrument

__all__ = [
    'FILTER_THROUGHPUT_E_PER_S_PER_IF',
    'canonical_filter',
    'filter_star_flux_scale',
    'resolve_filter_throughput',
]

# Electrons per second per native pixel at I/F 1.0, by filter combination,
# keyed by the raw camera name (the calibrated aliases share it).
#
# Measured 2026-11-06 from Cassini ISS raw/CALIB pairs.  CISSCAL writes the
# conversion it applied into every CALIB label (``RADIOMETRIC_CORRECTION_TEXT``:
# an efficiency factor and an absolute correction factor per filter
# combination), and the camera constant comes from fitting raw DN against
# CALIB I/F on the bright pixels of 12-bit frames, converted to electrons by
# the label's ``GAIN_CORRECTION``, divided by the exposure less
# ``EXPOSURE_OFFSET`` and by the summation factor.  The fits put the constant
# at 16,800 e-/s per I/F per unit efficiency for the NAC (31 frames, scatter
# about 15%) and 193,000 for the WAC (16 frames), a ratio of 11.5 that the
# cameras' apertures and pixel solid angles predict to within 30%.
#
# An entry tagged ``measured`` is the median of two or more 12-bit pair fits
# for that combination; every other entry is the camera constant times the
# label's efficiency and absolute correction.  Two measured filters depart
# from that model by more than the fit scatter: NAC UV3 (0.32x, two frames at
# two gains) and NAC IR3 (0.74x, three frames).  WAC VIO measures 0.67x on two
# frames of one sequence.  The 8-bit TABLE frames are excluded: their LUT
# encoding breaks the linear raw/CALIB fit.
#
# CISSCAL gives the polarizers the efficiency of the clear element they
# replace, so the real products carry the same DN-to-I/F scale with or
# without a polarizer; ``canonical_filter`` folds them onto that clear
# element.
FILTER_THROUGHPUT_E_PER_S_PER_IF: dict[str, dict[str, float]] = {
    'coiss_nac': {
        'BL1+CL2': 110000.0,
        'BL1+GRN': 5750.0,
        'CL1+BL2': 24700.0,
        'CL1+CB1': 62300.0,
        'CL1+CB2': 31300.0,
        'CL1+CB3': 6380.0,
        'CL1+CL2': 1450000.0,  # measured, 9 frames
        'CL1+GRN': 413000.0,  # measured, 8 frames
        'CL1+IR1': 471000.0,  # measured, 2 frames
        'CL1+IR3': 65300.0,  # measured, 3 frames
        'CL1+MT1': 11000.0,
        'CL1+MT2': 13400.0,
        'CL1+MT3': 13900.0,
        'CL1+UV3': 19600.0,  # measured, 2 frames
        'HAL+CL2': 28700.0,
        'IR2+CL2': 167000.0,
        'IR2+IR1': 56400.0,
        'IR2+IR3': 51400.0,
        'IR4+CL2': 11800.0,
        'IR4+IR3': 9630.0,
        'RED+CL2': 445000.0,
        'RED+GRN': 141000.0,
        'RED+IR1': 141000.0,
        'UV1+CL2': 13100.0,
        'UV2+CL2': 20800.0,
        'UV2+UV3': 6700.0,
    },
    'coiss_wac': {
        'CB2+CL2': 279000.0,
        'CB3+CL2': 11700.0,
        'CL1+BL1': 1150000.0,  # measured, 2 frames
        'CL1+CL2': 13500000.0,  # measured, 4 frames
        'CL1+GRN': 4470000.0,
        'CL1+HAL': 323000.0,
        'CL1+IR1': 4100000.0,
        'CL1+RED': 5760000.0,  # measured, 2 frames
        'CL1+VIO': 115000.0,  # measured, 2 frames
        'IR2+CL2': 695000.0,
        'IR2+IR1': 293000.0,
        'IR3+CL2': 216000.0,
        'IR4+CL2': 11200.0,
        'IR5+CL2': 1320.0,
        'MT2+CL2': 135000.0,
        'MT3+CL2': 39300.0,
    },
}

# The clear combination each camera's star zero point refers to: the
# zero points are sized from the published clear-filter limiting magnitudes.
_CLEAR_FILTER: dict[str, str] = {
    'coiss_nac': 'CL1+CL2',
    'coiss_wac': 'CL1+CL2',
}

# Polarizer elements and the clear element of the same wheel.  The NAC
# polarizers sit in wheel 1; the WAC polarizers sit in wheel 2.
_POLARIZER_CLEAR: dict[str, dict[str, str]] = {
    'coiss_nac': {'P0': 'CL1', 'P60': 'CL1', 'P120': 'CL1', 'IRP0': 'CL1'},
    'coiss_wac': {'IRP0': 'CL2', 'IRP90': 'CL2'},
}


class FilterCatalogError(ValueError):
    """A scene names a filter its camera does not catalog."""


def canonical_filter(instrument: str | None, filter_name: str) -> str:
    """Fold a filter combination onto its cataloged spelling.

    Each ``+``-separated element is stripped of surrounding whitespace and
    upper-cased, and a polarizer becomes the clear element of its wheel, so
    ``'p120+grn'`` reads as ``'CL1+GRN'`` on the Cassini NAC.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The filter combination as the scene writes it.

    Returns:
        The combination as the catalog keys it.
    """
    polarizers = _POLARIZER_CLEAR.get(normalize_instrument(instrument), {})
    elements = [element.strip().upper() for element in filter_name.split('+')]
    return '+'.join(polarizers.get(element, element) for element in elements)


def resolve_filter_throughput(instrument: str | None, filter_name: str) -> float:
    """Return a filter combination's throughput in e-/s per native pixel at I/F 1.0.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The filter combination, in any spelling
            :func:`canonical_filter` folds onto a catalog key.

    Returns:
        The throughput.

    Raises:
        FilterCatalogError: If the instrument has no filter table, or the
            combination is not in it.
    """
    table = FILTER_THROUGHPUT_E_PER_S_PER_IF.get(normalize_instrument(instrument))
    if table is None:
        cataloged = sorted(FILTER_THROUGHPUT_E_PER_S_PER_IF)
        raise FilterCatalogError(
            f'sim instrument {instrument!r} has no filter table; instruments with one: {cataloged}'
        )
    canonical = canonical_filter(instrument, filter_name)
    if canonical not in table:
        raise FilterCatalogError(
            f'filter {filter_name!r} is not cataloged for sim instrument {instrument!r}; '
            f'available filters: {sorted(table)}'
        )
    return table[canonical]


def filter_star_flux_scale(instrument: str | None, filter_name: str | None) -> float:
    """Return the factor a filter applies to the camera's star zero point.

    The zero point refers to the camera's clear combination, so a named filter
    scales it by that filter's throughput over the clear throughput.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The scene's ``detector.filter``, or ``None``.

    Returns:
        1.0 when the scene does not name a filter, else the throughput ratio.

    Raises:
        FilterCatalogError: As :func:`resolve_filter_throughput`.
    """
    if filter_name is None:
        return 1.0
    throughput = resolve_filter_throughput(instrument, filter_name)
    clear = _CLEAR_FILTER[normalize_instrument(instrument)]
    return throughput / resolve_filter_throughput(instrument, clear)
