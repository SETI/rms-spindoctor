"""Filter passbands of the simulated cameras, and the solar spectrum that weights them.

A camera's entry may set a passband table under ``filters.passbands``: a CSV file in
``spindoctor/sim/instruments/passbands/`` with a ``wavelength_nm`` column and one column
of system transmission (detector quantum efficiency times optics times both filters)
per filter combination the entry catalogs, on 2 nm bins.  ``solar_flux.csv`` beside them
holds the solar spectrum on the same bins.  ``util/passbands/build_passbands.py`` builds
the files from the camera team's published calibration tables.

A scene's filter selects its column, folded onto the catalog's spelling as
:func:`~spindoctor.sim.instruments.resolvers.canonical_filter` folds it, and a scene
that names no filter takes the camera's clear combination.  The light a filter passes
from a sunlit scene is weighted by the solar flux times the transmission
(:func:`solar_weighted_passband`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spindoctor.sim.instruments.catalog import instrument_entry
from spindoctor.sim.instruments.resolvers import FilterCatalogError, canonical_filter
from spindoctor.support.types import NDArrayFloatType

__all__ = ['PASSBAND_DIR', 'Passband', 'has_passbands', 'solar_weighted_passband']

PASSBAND_DIR = Path(__file__).resolve().parent / 'passbands'
"""The directory of the passband tables and the solar spectrum."""


@dataclass(frozen=True)
class Passband:
    """A filter combination's solar-weighted passband.

    Parameters:
        filter_name: The combination as the catalog keys it.
        wavelength_nm: The bin centers, nm.
        weight: The solar flux times the system transmission in each bin, normalized
            to sum to 1.
    """

    filter_name: str
    wavelength_nm: NDArrayFloatType
    weight: NDArrayFloatType


@functools.cache
def _table(name: str) -> dict[str, NDArrayFloatType]:
    """The columns of one CSV table in :data:`PASSBAND_DIR`, keyed by header."""
    path = PASSBAND_DIR / f'{name}.csv'
    header = path.read_text().split('\n', 1)[0].split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    return {column: data[:, index] for index, column in enumerate(header)}


def has_passbands(instrument: str | None) -> bool:
    """Whether the instrument's catalog entry names a passband table.

    Parameters:
        instrument: The sim instrument name.

    Returns:
        True when the entry's ``filters.passbands`` is set.
    """
    filters = instrument_entry(instrument).filters
    return filters is not None and filters.passbands is not None


def solar_weighted_passband(instrument: str | None, filter_name: str | None) -> Passband:
    """The solar-weighted passband of a scene's filter.

    Parameters:
        instrument: The sim instrument name.
        filter_name: The scene's ``detector.filter``, or None for the camera's clear
            combination.

    Returns:
        The passband, its weights summing to 1.

    Raises:
        FilterCatalogError: If the camera has no passband table, or the table has no
            column for the filter.
    """
    filters = instrument_entry(instrument).filters
    if filters is None or filters.passbands is None:
        raise FilterCatalogError(f'sim instrument {instrument!r} has no passband table')
    selected = filters.clear if filter_name is None else filter_name
    combination = canonical_filter(instrument, selected)
    table = _table(filters.passbands)
    if combination not in table:
        raise FilterCatalogError(
            f'the passband table of sim instrument {instrument!r} has no column for {combination!r}'
        )
    solar = _table('solar_flux')
    weight = table[combination] * solar['photons_per_cm2_s_angstrom']
    return Passband(
        filter_name=combination,
        wavelength_nm=table['wavelength_nm'].copy(),
        weight=weight / weight.sum(),
    )
