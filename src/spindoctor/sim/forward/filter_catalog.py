"""Per-filter throughput of the simulated cameras that carry a filter wheel.

A scene's optional ``detector.filter`` key names the filter combination the
frame was shot through, written as the product label writes it, with the
elements joined by ``+`` (for example ``'CL1+CL2'`` for a two-wheel camera).
Each instrument's config file catalogs its combinations under ``filters`` (see
:mod:`spindoctor.sim.instrument_catalog`), along with the clear combination its
star zero point refers to and the clear element each polarizer folds onto.  The
catalog value is the electrons per second one native pixel collects from a
surface at I/F 1.0, so the detector chain renders a signal of 1.0 at that rate
and the calibrated products divide by the same scale.  A scene that does not name
a filter keeps the nominal scale its detector entry derives
(``signal_full_scale_frac * full_well_e / exposure_ref_sec``).

The throughput is the only filter-dependent quantity the simulator models.
Haze, body photometry, star color, and the PSF render the same for every
filter.
"""

from __future__ import annotations

from typing import Any

from spindoctor.sim.instrument_catalog import instrument_entry, load_instrument_catalog
from spindoctor.sim.instrument_checks import canonical_filter_elements

__all__ = [
    'FilterCatalogError',
    'canonical_filter',
    'cataloged_filters',
    'filter_star_flux_scale',
    'resolve_filter_throughput',
]


class FilterCatalogError(ValueError):
    """A scene names a filter its camera does not catalog."""


def _filters(instrument: str | None) -> dict[str, Any] | None:
    """The ``filters`` section of an instrument's config, or None."""
    filters: dict[str, Any] | None = instrument_entry(instrument).get('filters')
    return filters


def _require_filters(instrument: str | None) -> dict[str, Any]:
    """The ``filters`` section of an instrument's config, refusing one without it."""
    filters = _filters(instrument)
    if filters is None:
        catalog = load_instrument_catalog()
        cataloged = [
            name for name in catalog.instrument_names() if 'filters' in catalog.entry(name)
        ]
        raise FilterCatalogError(
            f'sim instrument {instrument!r} has no filter table; instruments with one: {cataloged}'
        )
    return filters


def canonical_filter(instrument: str | None, filter_name: str) -> str:
    """Fold a filter combination onto its cataloged spelling.

    Each ``+``-separated element is stripped of surrounding whitespace and
    upper-cased, and a polarizer the instrument's ``filters.polarizer_clear`` names
    becomes the clear element of its wheel, so ``'p120+grn'`` reads as
    ``'CL1+GRN'`` on a camera whose ``P120`` folds onto ``CL1``.  The catalog
    refuses a filter table or polarizer spelled any other way, since no scene
    spelling could reach it.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The filter combination as the scene writes it.

    Returns:
        The combination as the catalog keys it.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    filters = _filters(instrument) or {}
    polarizers = filters.get('polarizer_clear') or {}
    elements = canonical_filter_elements(filter_name)
    return '+'.join(polarizers.get(element, element) for element in elements)


def cataloged_filters(instrument: str | None) -> tuple[str, ...]:
    """The filter combinations an instrument catalogs, sorted.

    Parameters:
        instrument: The sim instrument name.

    Returns:
        The combinations, as the catalog keys them; empty for an instrument
        without a filter table.

    Raises:
        ValueError: If the instrument is not in the catalog.
    """
    filters = _filters(instrument)
    return () if filters is None else tuple(sorted(filters['throughput_e_per_s_per_if']))


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
        ValueError: If the instrument is not in the catalog.
    """
    table = _require_filters(instrument)['throughput_e_per_s_per_if']
    canonical = canonical_filter(instrument, filter_name)
    if canonical not in table:
        raise FilterCatalogError(
            f'filter {filter_name!r} is not cataloged for sim instrument {instrument!r}; '
            f'available filters: {sorted(table)}'
        )
    throughput: float = table[canonical]
    return throughput


def filter_star_flux_scale(instrument: str | None, filter_name: str | None) -> float:
    """Return the factor a filter applies to the camera's star zero point.

    The zero point refers to the camera's clear combination (``filters.clear``),
    so a named filter scales it by that filter's throughput over the clear
    throughput.

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
    clear = _require_filters(instrument)['clear']
    return throughput / resolve_filter_throughput(instrument, clear)
