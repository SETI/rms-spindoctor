"""Build the simulator's filter passband tables from the CISSCAL system transmissions.

Usage::

    python util/passbands/build_passbands.py <efficiency_dir> [<out_dir>]

``<efficiency_dir>`` is the ``calib/efficiency`` directory of the Cassini ISS
calibration volume in the PDS archive (``COISS_0011``).  Its ``systrans`` files give,
for each filter combination of each camera, the system transmission (CCD quantum
efficiency times optics times both filters) against wavelength, and ``solarflux.tab``
gives the solar spectrum (the ASTM E490 reference spectrum, in photons per cm^2 per s
per Angstrom at 1 AU).

Each table is averaged into 2 nm bins centered on 200, 202, ..., 1056 nm and written
next to the instrument catalog, one file per camera
(``src/spindoctor/sim/instruments/passbands/<camera>.csv``) with a column per filter
combination the catalog lists for that camera, and one file for the solar spectrum
(``solar_flux.csv``).  A combination whose polarizer the catalog folds onto a clear
element takes that clear element's table, whose solar-weighted mean wavelength lies
under 1 nm from the polarizer's own.  ``tests/integration/test_passband_tables.py``
rebuilds the tables from the archive and compares them with the committed files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The 2 nm bin centers every table is written on.
GRID_NM = np.arange(200.0, 1057.0, 2.0)

# The catalog's camera entries and the CISSCAL file prefix of each.
CAMERAS = {'coiss_nac': 'issna', 'coiss_wac': 'isswa'}

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / 'src/spindoctor/sim/instruments/passbands'


def read_table(path: Path) -> np.ndarray:
    """The numeric rows after a CISSCAL table's ``\\begindata`` line."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == '\\begindata') + 1
    rows = [line.split() for line in lines[start:] if line.strip()]
    return np.array([[float(x) for x in row] for row in rows])


def bin_average(wavelength_nm: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Average a finely sampled curve over 2 nm bins centered on :data:`GRID_NM`."""
    fine = np.arange(GRID_NM[0] - 1.0, GRID_NM[-1] + 1.0, 0.1) + 0.05
    sampled = np.interp(fine, wavelength_nm, values, left=0.0, right=0.0)
    return sampled.reshape(len(GRID_NM), -1).mean(axis=1)


def camera_columns(camera: str) -> list[str]:
    """The filter combinations the catalog lists for a camera."""
    sys.path.insert(0, str(REPO / 'src'))
    from spindoctor.sim.instruments.resolvers import cataloged_filters

    return list(cataloged_filters(camera))


def build(efficiency_dir: Path) -> dict[str, str]:
    """Build every table; returns the CSV text keyed by file name."""
    out: dict[str, str] = {}
    solar = read_table(efficiency_dir / 'solarflux.tab')
    solar_binned = bin_average(solar[:, 0] / 10.0, solar[:, 1])
    lines = ['wavelength_nm,photons_per_cm2_s_angstrom']
    lines += [f'{w:.0f},{s:.6e}' for w, s in zip(GRID_NM, solar_binned, strict=True)]
    out['solar_flux.csv'] = '\n'.join(lines) + '\n'
    for camera, prefix in CAMERAS.items():
        columns = camera_columns(camera)
        curves = []
        for combination in columns:
            first, second = combination.lower().split('+')
            table = read_table(
                efficiency_dir / 'systrans' / f'{prefix}{first}{second}_systrans.tab'
            )
            curves.append(bin_average(table[:, 0], table[:, 1]))
        lines = ['wavelength_nm,' + ','.join(columns)]
        for i, w in enumerate(GRID_NM):
            lines.append(f'{w:.0f},' + ','.join(f'{curve[i]:.6e}' for curve in curves))
        out[f'{camera}.csv'] = '\n'.join(lines) + '\n'
    return out


def main() -> None:
    """Build and write the tables."""
    efficiency_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in build(efficiency_dir).items():
        (out_dir / name).write_text(text)
        print(out_dir / name)


if __name__ == '__main__':
    main()
