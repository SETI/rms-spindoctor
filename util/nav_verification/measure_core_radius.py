"""Check a ring mosaic against the orbit model it was built on.

A mosaic built in a co-rotating frame is built on a model that says where the
ring core is, so the core should land at radius offset zero in every column.
Where it actually lands measures the navigation of the frames that contributed,
and needs nothing outside the mosaic itself, which is what makes this usable on
observations no other project has navigated.

Two different things are worth reading out of the result.  A displacement that
is the same all the way round is the navigation, the orbit model, or the epoch
being off together, and is a single number to chase.  A displacement that steps
between adjacent longitudes is two neighbouring frames navigated differently
from each other, because real ring structure varies smoothly with longitude and
a pointing error does not -- so a step is the signature of one bad frame, and
its longitude says which.

Adjacency is in longitude and not in storage.  A sparse mosaic keeps only the
longitudes it has data for, so two columns sitting side by side in the file can
be tens of degrees apart on the ring, and over tens of degrees the ring really
does change.  Comparing across a gap like that reports the gap as a bad frame,
so only columns one longitude bin apart are compared and the gaps are counted
and reported on their own.

The core is taken as the brightness-weighted centroid of the peak, with the
search restricted to a band around a robust first estimate so that a cosmic ray
or a field star cannot pull one column hundreds of kilometres away.  Single
column spikes are then removed with a median filter before steps are counted, so
what is reported is a discontinuity rather than one bad pixel.

Run after ``source /seti/newnav/setup.sh``, from the repository root::

    python util/nav_verification/measure_core_radius.py mosaic.fits
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

WINDOW_KM = 150.0
SEARCH_KM = 400.0
STEP_KM = 25.0


@dataclass(frozen=True)
class CoreProfile:
    """Where the ring core sits, column by column.

    Parameters:
        longitudes: The longitude each column stands for, in degrees.
        core: The core's radius in the mosaic's own radial coordinate, with NaN
            where a column had nothing to measure.
        bins: The longitude bin each column stands for, so that columns one
            longitude apart can be told from columns that merely sit side by
            side in the file.
        longitude_resolution_deg: The width of one bin.
        orbit_model: The orbit model the mosaic was built on, or None for a
            mosaic whose radial axis is an absolute radius rather than an offset
            from a model.
    """

    longitudes: np.ndarray
    core: np.ndarray
    bins: np.ndarray
    longitude_resolution_deg: float
    orbit_model: str | None


def _centroid(
    column: np.ndarray,
    radii: np.ndarray,
    *,
    low_km: float,
    high_km: float,
    half_window: int,
) -> float:
    """The brightness-weighted radius of the brightest feature in one column.

    Parameters:
        column: The column's brightness against radius, with gaps left as NaN.
        radii: The radius each row stands for, in kilometres.
        low_km: The innermost radius the peak may be found at.
        high_km: The outermost radius the peak may be found at.
        half_window: How many rows either side of the peak the centroid runs
            over.

    Returns:
        The centroid radius, or NaN when the column has nothing to measure.
    """
    band = (radii >= low_km) & (radii <= high_km)
    if not band.any():
        return math.nan
    inside = np.flatnonzero(band)
    within = column[inside]
    if not np.isfinite(within).any():
        return math.nan
    peak = inside[int(np.nanargmax(within))]
    low, high = max(0, peak - half_window), min(column.size, peak + half_window + 1)
    weights, radius = column[low:high], radii[low:high]
    usable = np.isfinite(weights) & (weights > 0)
    if usable.sum() < 3:
        return math.nan
    return float(np.sum(radius[usable] * weights[usable]) / np.sum(weights[usable]))


def _column_bins(header: fits.Header, antimask: np.ndarray, n_columns: int) -> np.ndarray:
    """The longitude bin each stored column stands for.

    A mosaic is saved in one of three shapes and they index longitude
    differently: the sparse form keeps one column per longitude it holds data
    for, the full form keeps every bin of the circle, and the bounded form keeps
    every bin between two longitudes.  Only the first is what a mosaic run
    writes, but all three are public and a caller can hand any of them here.

    Parameters:
        header: The mosaic's primary header.
        antimask: The longitude antimask, one entry per bin of the full circle.
        n_columns: How many columns the image has.

    Returns:
        The bin index of each column.

    Raises:
        ValueError: If the columns match none of the three shapes, so that no
            longitude can be assigned to them.
    """
    if not header.get('LONGITUDE_RANGE_NONE', False):
        start = round(float(header['LONGITUDE_RANGE_0']) / float(header['LONGITUDE_RESOLUTION']))
        return np.arange(start, start + n_columns)
    if n_columns == int(antimask.sum()):
        return np.flatnonzero(antimask)
    if n_columns == antimask.size:
        return np.arange(n_columns)
    raise ValueError(
        f'{n_columns} columns match neither the {int(antimask.sum())} longitudes this '
        f'mosaic holds data for nor the {antimask.size} bins of the full circle'
    )


def core_offsets(
    path: Path, *, window_km: float = WINDOW_KM, search_km: float = SEARCH_KM
) -> CoreProfile:
    """Where the ring core sits in every column of a mosaic.

    Parameters:
        path: The mosaic.
        window_km: How far either side of the peak the centroid runs.
        search_km: How far from the first estimate the core may be looked for.

    Returns:
        The profile, whose radial values are an offset from the orbit model when
        the mosaic was built on one and an absolute radius when it was not.
    """
    with fits.open(path) as mosaic:
        header = mosaic[0].header
        image = np.where(
            mosaic['IMG_MASK'].data.astype(bool), np.nan, mosaic['IMG'].data.astype(float)
        )
        antimask = mosaic['LONGITUDE_ANTIMASK'].data.astype(bool)
        radial_resolution = float(header['RADIUS_RESOLUTION'])
        radii = float(header['RADIUS_INNER']) + np.arange(image.shape[0]) * radial_resolution
        resolution_deg = math.degrees(float(header['LONGITUDE_RESOLUTION']))
        bins = _column_bins(header, antimask, image.shape[1])
        orbit_model = (
            None
            if header.get('ORBIT_MODEL_NAME_NONE', False)
            else str(header.get('ORBIT_MODEL_NAME', '')) or None
        )

    half_window = round(window_km / radial_resolution)
    rough = np.array(
        [
            _centroid(
                image[:, c], radii, low_km=radii[0], high_km=radii[-1], half_window=half_window
            )
            for c in range(image.shape[1])
        ]
    )
    centre = float(np.nanmedian(rough))
    core = np.array(
        [
            _centroid(
                image[:, c],
                radii,
                low_km=centre - search_km,
                high_km=centre + search_km,
                half_window=half_window,
            )
            for c in range(image.shape[1])
        ]
    )
    return CoreProfile(
        longitudes=bins * resolution_deg,
        core=core,
        bins=bins,
        longitude_resolution_deg=resolution_deg,
        orbit_model=orbit_model,
    )


def median_filter(values: np.ndarray, *, width: int = 5) -> np.ndarray:
    """A running median, so one bad column is not read as a step.

    Parameters:
        values: The series to filter.
        width: How many samples the median runs over.

    Returns:
        The filtered series, the same length as the input.
    """
    pad = width // 2
    padded = np.pad(values, pad, mode='edge')
    return np.array([np.nanmedian(padded[i : i + width]) for i in range(values.size)])


def steps_between_adjacent(
    bins: np.ndarray, values: np.ndarray, *, step_km: float = STEP_KM
) -> np.ndarray:
    """Which neighbouring pairs jump, counting only pairs that really neighbour.

    Two columns that sit side by side in a sparse mosaic can be tens of degrees
    apart on the ring, and the ring changes over tens of degrees, so a pair
    straddling a gap is not evidence of anything and is never counted.

    Parameters:
        bins: The longitude bin of each column, ascending.
        values: The measured value in each column.
        step_km: How large a change between two adjacent longitudes is a step.

    Returns:
        The index of the first column of each jumping pair.
    """
    adjacent = np.diff(bins) == 1
    change = np.abs(np.diff(values))
    return np.flatnonzero(adjacent & (change > step_km))


def report(profile: CoreProfile, *, step_km: float = STEP_KM) -> None:
    """Print where the core sits and where it jumps.

    Parameters:
        profile: The measured profile.
        step_km: How large a change between two adjacent longitudes is called a
            step.
    """
    measurable = np.isfinite(profile.core)
    found = profile.core[measurable]
    at = profile.longitudes[measurable]
    bins = profile.bins[measurable]
    print(f'columns with a measurable core: {measurable.sum()} of {profile.core.size}')
    if not measurable.any():
        return

    if profile.orbit_model is None:
        print('built on no orbit model, so the radial axis is an absolute radius and what')
        print('follows is a radius rather than an offset from a model')
        label = 'core radius'
    else:
        label = f'core radius offset from {profile.orbit_model}'

    print(
        f'{label}: median {np.median(found):+.1f} km, '
        f'mean {found.mean():+.1f}, std {found.std():.1f}'
    )
    print(
        f'  10th/90th percentile: {np.percentile(found, 10):+.1f} / '
        f'{np.percentile(found, 90):+.1f} km'
    )
    print(f'  full range: {found.min():+.1f} to {found.max():+.1f} km')

    smoothed = median_filter(found)
    spacing = np.diff(bins)
    adjacent = spacing == 1
    change = np.abs(np.diff(smoothed))
    if not adjacent.all():
        widest = float(spacing[~adjacent].max()) * profile.longitude_resolution_deg
        print(
            f'  longitude gaps: {int((~adjacent).sum())}, widest {widest:.2f} deg -- '
            f'not compared across'
        )
    if not adjacent.any():
        print('  no two measured columns are one longitude bin apart; nothing to compare')
        return

    neighbouring = change[adjacent]
    print(
        f'  change between adjacent longitudes (despiked): '
        f'median {np.median(neighbouring):.2f} km, '
        f'95th {np.percentile(neighbouring, 95):.2f}, max {neighbouring.max():.1f} km'
    )
    jumps = steps_between_adjacent(bins, smoothed, step_km=step_km)
    runs = int(np.sum(np.diff(jumps) > 1)) + 1 if jumps.size else 0
    print(
        f'  steps over {step_km:.0f} km between adjacent longitudes: {jumps.size}'
        + (f', in {runs} run(s)' if runs and runs != jumps.size else '')
    )
    for jump in jumps:
        print(
            f'    at {at[jump]:7.2f} deg: {smoothed[jump]:+7.1f} -> {smoothed[jump + 1]:+7.1f} km'
        )


def main() -> None:
    """Measure one mosaic's core placement and report it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mosaic', type=Path, help='the ring mosaic to measure')
    parser.add_argument(
        '--step-km',
        type=float,
        default=STEP_KM,
        help='how large a change between two adjacent longitudes is '
        f'called a step (default {STEP_KM})',
    )
    parser.add_argument(
        '--window-km',
        type=float,
        default=WINDOW_KM,
        help=f'how far either side of the peak the centroid runs (default {WINDOW_KM})',
    )
    parser.add_argument(
        '--search-km',
        type=float,
        default=SEARCH_KM,
        help=f'how far from the first estimate the core may be looked for (default {SEARCH_KM})',
    )
    args = parser.parse_args()

    report(
        core_offsets(args.mosaic, window_km=args.window_km, search_km=args.search_km),
        step_km=args.step_km,
    )


if __name__ == '__main__':
    main()
