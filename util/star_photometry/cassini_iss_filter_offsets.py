"""Derive the Cassini ISS star-photometry table from the CISSCAL calibration files.

The Cassini star limiting magnitude (``ObsCassiniISS.star_max_usable_vmag``) is computed
photometrically: the integrated DN a star of a given V magnitude deposits through the
frame's filter pair in its exposure, against the minimum DN a star needs to be seen.  This
program regenerates the two inputs that come from the camera's calibration:

* the electron rate a V = 0 star of the default spectral class (G0) deposits through
  CL1+CL2, per camera; and
* for every filter pair, the magnitude offset of that pair relative to CL1+CL2 for the same
  star.

Both follow the photometric chain of the previous navigation software (``flux.py``,
``_compute_dn_from_spectrum`` and ``compute_dn_from_star``): a stellar photon spectrum
scaled to the star's V magnitude, multiplied by the CISSCAL system transmission
(quantum efficiency times optics times both filters) and the CISSCAL quantum-efficiency
correction, integrated over wavelength and multiplied by the collecting area, then
scaled by that software's per-filter correction for G-class stars measured on Cassini
star fields.  The stellar spectrum is a 5920 K Planck curve (the G0 surface temperature
the star catalogs use), normalized so its photon flux density at 550 nm is that of a
V = 0 star, 1.0e4 photons / cm^2 / s / nm.

The CISSCAL files are distributed with the Cassini ISS archive (the ``calib`` tree of
the calibration volume, ``efficiency/systrans/*_systrans.tab`` and
``correction/{nac,wac}_qe_correction.tab``).

Usage::

    python -m util.star_photometry.cassini_iss_filter_offsets CISSCAL_CALIB_DIR

prints the YAML block that ``config_400_inst_coiss.yaml`` carries.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spindoctor.support.filter_combo import canonicalize

__all__ = ['filter_pair_electron_rate', 'main', 'read_cisscal_table']

FloatArray = NDArray[np.float64]

#: Collecting area of each camera's optics (cm^2), from CISSCAL.
OPTICS_AREA_CM2 = {'NAC': 284.86, 'WAC': 29.43}

#: Surface temperature (K) of the default G0 star.
G0_TEMPERATURE_K = 5920.0

#: Photon flux density of a V = 0 star at 550 nm (photons / cm^2 / s / nm).
V0_PHOTON_FLUX_550NM = 1.0e4

#: Wavelength (nm) at which the stellar spectrum is normalized to the V magnitude.
V_REFERENCE_NM = 550.0

#: Planck constant (J s).
_PLANCK_J_S = 6.62607015e-34

#: Speed of light (m / s).
_LIGHT_SPEED_M_S = 2.99792458e8

#: Boltzmann constant (J / K).
_BOLTZMANN_J_K = 1.380649e-23

#: Meters per nanometer.
_METERS_PER_NM = 1.0e-9

#: The previous software's per-filter multiplicative correction for G-class stars,
#: measured on Cassini star fields (``flux.py``, ``STAR_FINE_CALIBRATION``), keyed by the
#: simplified filter name of ``_simple_filter_name``.  Filters it did not measure take 1.0.
G_CLASS_FINE_CORRECTION = {
    'NAC': {
        'CLEAR': 1.07,
        'UV3': 0.71,
        'BL2': 0.72,
        'BL1': 0.80,
        'GRN': 0.91,
        'CB1': 1.07,
        'MT1': 1.06,
        'RED': 1.08,
        'CB2': 1.11,
        'MT2': 1.16,
        'CB3': 1.33,
        'MT3': 1.26,
        'IR1': 1.18,
        'IR2': 1.22,
        'IR3': 2.04,
        'IR4': 1.32,
    },
    'WAC': {
        'CLEAR': 1.47,
        'VIO': 1.04,
        'BL1': 0.91,
        'GRN': 1.38,
        'RED': 1.59,
        'HAL': 1.47,
        'CB2': 1.72,
        'MT2': 1.45,
        'MT3': 1.49,
        'IR1': 2.68,
        'IR2': 1.69,
        'IR3': 2.24,
    },
}

_VISIBLE_POLARIZERS = frozenset({'P0', 'P60', 'P120'})


def read_cisscal_table(path: Path) -> FloatArray:
    """Return the numeric rows between ``\\begindata`` and ``\\enddata`` of a CISSCAL table.

    Parameters:
        path: The table file.

    Returns:
        A 2-D array, one row per data line.
    """
    rows: list[list[float]] = []
    in_data = False
    for line in path.read_text().splitlines():
        if line.startswith('\\begindata'):
            in_data = True
            continue
        if line.startswith('\\enddata'):
            break
        if in_data and line.strip():
            rows.append([float(x) for x in line.split()])
    return np.asarray(rows, dtype=np.float64)


def _planck_photons(wavelength_nm: FloatArray, temperature_k: float) -> FloatArray:
    """Return a Planck photon spectrum (arbitrary scale) at the given wavelengths."""
    wl_m = wavelength_nm * _METERS_PER_NM
    return 1.0 / (
        wl_m**4 * np.expm1(_PLANCK_J_S * _LIGHT_SPEED_M_S / (wl_m * _BOLTZMANN_J_K * temperature_k))
    )


def _simple_filter_name(filter1: str, filter2: str) -> str:
    """Return the name the previous software looked its correction up by.

    It names the pair CLEAR when both filters are clear, the other filter when one is
    clear, and otherwise the first filter; a visible polarizer (P0, P60, P120) with a
    clear second filter is P, which the table does not carry, and with any other second
    filter is that filter (``misc.simple_filter_name_f1f2`` with polarizers consolidated,
    then ``compute_dn_from_star``).
    """
    if filter1 in _VISIBLE_POLARIZERS:
        return 'P' if filter2 == 'CL2' else filter2
    if filter1 == 'CL1':
        return 'CLEAR' if filter2 == 'CL2' else filter2
    return filter1


def filter_pair_electron_rate(calib_dir: Path, camera: str, filter1: str, filter2: str) -> float:
    """Return the electrons per second a V = 0 G0 star deposits through a filter pair.

    Parameters:
        calib_dir: The CISSCAL ``calib`` directory.
        camera: ``'NAC'`` or ``'WAC'``.
        filter1: Filter-wheel-1 name.
        filter2: Filter-wheel-2 name.

    Returns:
        The electron rate, including the previous software's G-class correction.
    """
    name = f'iss{camera.lower()[:2]}{filter1.lower()}{filter2.lower()}_systrans.tab'
    systrans = read_cisscal_table(calib_dir / 'efficiency' / 'systrans' / name)
    qecorr = read_cisscal_table(calib_dir / 'correction' / f'{camera.lower()}_qe_correction.tab')
    wl = systrans[:, 0]
    trans = systrans[:, 1] * np.interp(wl, qecorr[:, 0], qecorr[:, 1])
    spectrum = _planck_photons(wl, G0_TEMPERATURE_K)
    spectrum *= V0_PHOTON_FLUX_550NM / float(
        _planck_photons(np.asarray([V_REFERENCE_NM]), G0_TEMPERATURE_K)[0]
    )
    rate = float(np.trapezoid(spectrum * trans, wl)) * OPTICS_AREA_CM2[camera]
    correction = G_CLASS_FINE_CORRECTION[camera].get(_simple_filter_name(filter1, filter2), 1.0)
    return rate * correction


def main() -> None:
    """Print, per camera, the electron rate and the filter-pair magnitude offsets as YAML."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('calib_dir', type=Path, help='The CISSCAL calib directory')
    args = parser.parse_args()
    systrans_dir = args.calib_dir / 'efficiency' / 'systrans'
    for camera in ('NAC', 'WAC'):
        prefix = f'iss{camera.lower()[:2]}'
        clear = filter_pair_electron_rate(args.calib_dir, camera, 'CL1', 'CL2')
        offsets: dict[str, float] = {}
        for path in sorted(systrans_dir.glob(f'{prefix}*_systrans.tab')):
            pair = path.name[len(prefix) : -len('_systrans.tab')].upper()
            for filter1, filter2 in _split_pairs(camera, pair):
                if read_cisscal_table(path).ndim != 2:
                    # CISSCAL ships an empty table for a pair that passes no light.
                    continue
                rate = filter_pair_electron_rate(args.calib_dir, camera, filter1, filter2)
                offsets[canonicalize([filter1, filter2])] = -2.5 * math.log10(rate / clear)
        print(f'  {camera.lower()}:')
        print(f'    flux_electrons_per_s_vmag0: {clear:.4g}')
        print('    filter_mag_offset:')
        for combo in sorted(offsets):
            print(f"      '{combo}': {round(offsets[combo], 2) + 0.0:.2f}")


_WHEEL1 = {
    'NAC': ('CL1', 'RED', 'BL1', 'UV2', 'UV1', 'IRP0', 'P120', 'P60', 'P0', 'HAL', 'IR4', 'IR2'),
    'WAC': ('CL1', 'IR3', 'IR4', 'IR5', 'CB3', 'MT3', 'CB2', 'MT2', 'IR2'),
}


def _split_pairs(camera: str, pair: str) -> list[tuple[str, str]]:
    """Split a CISSCAL file's concatenated filter names into (wheel 1, wheel 2)."""
    return [(f1, pair[len(f1) :]) for f1 in _WHEEL1[camera] if pair.startswith(f1)]


if __name__ == '__main__':
    main()
