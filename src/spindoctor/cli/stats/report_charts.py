"""Chart writers for the statistics report.

Every chart is rendered with the Agg backend into a PNG buffer and written
through ``FCPath``, so the same inputs give the same bytes and the output
directory can be a local path or any URL the ``filecache`` layer accepts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import numpy as np
from filecache import FCPath
from numpy.typing import NDArray

__all__ = [
    'instrument_color',
    'write_offset_heatmap',
    'write_offset_hist',
    'write_stacked_bar_chart',
    'write_stacked_value_hist',
]


# Fixed per-instrument chart colors so a stacked segment means the same
# instrument in every chart and across runs.
_INSTRUMENT_COLORS: dict[str, str] = {
    'coiss': '#4878d0',
    'vgiss': '#ee854a',
    'gossi': '#6acc64',
    'nhlorri': '#d65f5f',
    'sim': '#956cb4',
}
_FALLBACK_COLORS: tuple[str, ...] = ('#8c8c8c', '#797979', '#b3b3b3', '#5c5c5c')


def instrument_color(instrument: str, index: int) -> str:
    """The chart color for an instrument.

    Parameters:
        instrument: Instrument name.
        index: The instrument's position in the report's instrument list;
            picks a fallback color for unregistered instruments.

    Returns:
        A matplotlib color string.
    """
    known = _INSTRUMENT_COLORS.get(instrument.lower())
    if known is not None:
        return known
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def import_pyplot() -> Any:
    """Import matplotlib with the deterministic Agg backend and return pyplot.

    Returns:
        The ``matplotlib.pyplot`` module, with the backend forced to Agg
        so chart output is identical with or without a display.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, plt: Any, path: FCPath) -> None:
    """Render a figure to PNG bytes and write them through ``FCPath``.

    Rendering to a buffer first means the destination can be a local path
    or any URL the ``filecache`` layer accepts.

    Parameters:
        fig: Matplotlib figure to write.
        plt: The pyplot module (from :func:`import_pyplot`); the figure is
            closed here.
        path: Destination PNG path.
    """
    buffer = BytesIO()
    fig.savefig(buffer, dpi=100, format='png')
    plt.close(fig)
    path.write_bytes(buffer.getvalue())


def write_stacked_bar_chart(
    path: FCPath,
    labels: list[str],
    counts_by_instrument: dict[str, list[int]],
    instruments: list[str],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Write a horizontal stacked bar chart PNG (deterministic, Agg backend).

    Each bar is one category, segmented by instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        labels: One bar label per category, top to bottom.
        counts_by_instrument: Instrument name to one count per category,
            matching ``labels``; instruments absent here contribute zero.
        instruments: Instrument names, in stacking order (left to right).
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(9, max(2.0, 0.4 * len(labels) + 1.5)))
    positions = list(range(len(labels)))
    left = [0.0] * len(labels)
    for index, instrument in enumerate(instruments):
        counts = counts_by_instrument.get(instrument, [0] * len(labels))
        ax.barh(
            positions,
            counts,
            left=left,
            color=instrument_color(instrument, index),
            label=instrument,
        )
        left = [base + value for base, value in zip(left, counts, strict=True)]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if len(instruments) > 0:
        ax.legend(title='instrument', loc='lower right')
    fig.tight_layout()
    _save_figure(fig, plt, path)


def write_stacked_value_hist(
    path: FCPath,
    values_by_instrument: Mapping[str, Sequence[float]],
    instruments: list[str],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Write a single-panel stacked histogram PNG, segmented by instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        values_by_instrument: Instrument name to the values to histogram;
            all instruments share one set of bins.
        instruments: Instrument names, in stacking order.
        title: Chart title.
        xlabel: X-axis label.
    """
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(9, 4))
    series = [values_by_instrument.get(instrument, []) for instrument in instruments]
    if sum(len(values) for values in series) > 0:
        ax.hist(
            series,
            bins=40,
            stacked=True,
            color=[instrument_color(name, index) for index, name in enumerate(instruments)],
            label=instruments,
        )
        ax.legend(title='instrument', loc='upper right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('images')
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, plt, path)


def write_offset_hist(
    path: FCPath, dv: Sequence[float], du: Sequence[float], *, title: str
) -> None:
    """Write a two-panel V/U offset histogram PNG for one instrument.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        dv: Fused V-axis offsets (pixels) of successful images.
        du: Fused U-axis offsets (pixels) of successful images.
        title: Figure title (the instrument is named by the caller).
    """
    plt = import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label in ((axes[0], dv, 'dV (px)'), (axes[1], du, 'dU (px)')):
        if len(values) > 0:
            ax.hist(values, bins=40, color='#4878d0')
        ax.set_xlabel(label)
        ax.set_ylabel('images')
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, plt, path)


# The panel's half-width is this percentile of the correction length (with a
# margin), so a few large corrections fall outside the panel instead of shrinking
# the core to a few bins; the caption counts the ones that do.
_HEATMAP_LIMIT_PERCENTILE = 98.0
# Bins across each axis of the panel: about the square root of the image count,
# within these bounds, and odd so the predicted pointing sits in the middle of a
# bin. The bins are square.
_HEATMAP_MIN_BINS = 11
_HEATMAP_MAX_BINS = 61
# Light-to-dark single-hue ramp for the image count per bin; it starts dark
# enough that a bin holding one stray image still shows on white.
_HEATMAP_RAMP: tuple[str, ...] = (
    '#b7d3f6',
    '#86b6ef',
    '#5598e7',
    '#2a78d6',
    '#1c5cab',
    '#104281',
)
_HEATMAP_GRID = '#b9b8b3'
_HEATMAP_INK = '#52514e'


def _ring_radii(limit: float) -> list[float]:
    """Scale-ring radii for an offset heat map of the given half-width.

    The spacing is the smallest 1-2-5 step of at least a fifth of the half-width,
    so the panel carries two to four rings with round labels.

    Parameters:
        limit: The panel's half-width in pixels; positive.

    Returns:
        The ring radii in pixels, ascending, all inside ``limit``.
    """
    magnitude = 10.0 ** math.floor(math.log10(limit / 5.0))
    step = next(
        magnitude * factor for factor in (1.0, 2.0, 5.0, 10.0) if magnitude * factor >= limit / 5.0
    )
    return [step * multiple for multiple in range(1, 5) if step * multiple < limit]


def _heatmap_limit(offsets: NDArray[np.float64]) -> float:
    """The half-width of an offset heat map, in pixels.

    Parameters:
        offsets: The ``(N, 2)`` array of every ``(dv, du)`` correction; ``N > 0``.

    Returns:
        A little beyond the given percentile of the correction length, or 1 pixel
        when every correction is zero.
    """
    length = np.hypot(offsets[:, 0], offsets[:, 1])
    limit = 1.1 * float(np.percentile(length, _HEATMAP_LIMIT_PERCENTILE))
    return limit if limit > 0.0 else 1.0


def _heatmap_bins(count: int) -> int:
    """The number of bins across each axis of an offset heat map.

    Parameters:
        count: How many images the heat map counts.

    Returns:
        An odd bin count near ``sqrt(count)``, within ``_HEATMAP_MIN_BINS`` and
            ``_HEATMAP_MAX_BINS``.
    """
    bins = min(max(round(math.sqrt(count)), _HEATMAP_MIN_BINS), _HEATMAP_MAX_BINS)
    return bins if bins % 2 == 1 else bins + 1


def _heatmap_counts(
    offsets: NDArray[np.float64], limit: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Count the corrections into the square bins of an offset heat map.

    Parameters:
        offsets: The ``(N, 2)`` array of every ``(dv, du)`` correction.
        limit: The panel's half-width in pixels.

    Returns:
        The counts, with rows indexed by dV and columns by dU (the image's own
        layout, so row 0 is the most negative dV), and the bin edges shared by
        both axes. Corrections outside the panel are not counted.
    """
    edges = np.linspace(-limit, limit, _heatmap_bins(len(offsets)) + 1)
    counts, _, _ = np.histogram2d(offsets[:, 0], offsets[:, 1], bins=[edges, edges])
    return counts, edges


def _count_ticks(most: float) -> list[float]:
    """Colorbar ticks for image counts from 1 to ``most`` on a log scale.

    Parameters:
        most: The largest count on the colorbar; at least 1.

    Returns:
        The 1-2-5 values from 1 up to ``most``.
    """
    ticks: list[float] = []
    decade = 1.0
    while decade <= most:
        ticks += [decade * factor for factor in (1.0, 2.0, 5.0) if decade * factor <= most]
        decade *= 10.0
    return ticks


def _heatmap_caption(offsets: NDArray[np.float64], limit: float) -> str:
    """The caption above an offset heat map.

    It states the image count, the RMS correction length over every image, and
    how many corrections fall outside the panel.

    Parameters:
        offsets: The ``(N, 2)`` array of every ``(dv, du)`` correction.
        limit: The panel's half-width in pixels.

    Returns:
        The caption text.
    """
    total = len(offsets)
    rms = float(np.sqrt(np.mean(offsets[:, 0] ** 2 + offsets[:, 1] ** 2)))
    outside = int(np.count_nonzero(np.max(np.abs(offsets), axis=1) > limit))
    noun = 'image' if total == 1 else 'images'
    caption = f'{total:,} {noun}     RMS {rms:.2f} px'
    if outside > 0:
        caption += f'     {outside:,} outside the panel'
    return caption


def write_offset_heatmap(
    path: FCPath, dv: Sequence[float], du: Sequence[float], *, title: str
) -> bool:
    """Write the pointing-correction heat map PNG for one camera.

    Every successful image's fused ``(dv, du)`` correction is counted into a grid
    of square pixel bins centered on the predicted pointing, drawn in image
    orientation: dU to the right, dV down. Bin color is the image count on a log
    scale, empty bins are left blank, the axes are equal-aspect in pixels, and
    labeled rings give the scale. The caption states the image count, the RMS
    correction length, and how many corrections fall outside the panel.

    Parameters:
        path: Destination PNG path (local or ``filecache`` URL).
        dv: Fused V-axis offsets (pixels) of successful images.
        du: Fused U-axis offsets (pixels) of successful images, paired with
            ``dv`` by position.
        title: Figure title (the instrument and camera are named by the caller).

    Returns:
        True if a chart was written; False, writing nothing, when there are no
        offsets.
    """
    if len(dv) == 0:
        return False
    offsets = np.column_stack([np.asarray(dv, dtype=np.float64), np.asarray(du, dtype=np.float64)])
    limit = _heatmap_limit(offsets)
    counts, edges = _heatmap_counts(offsets, limit)
    bins = len(edges) - 1

    plt = import_pyplot()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.ticker import NullFormatter

    most = max(float(counts.max()), 2.0)
    cmap = LinearSegmentedColormap.from_list('offset_heatmap', list(_HEATMAP_RAMP))
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    mesh = ax.pcolormesh(
        edges,
        edges,
        np.ma.masked_equal(counts, 0.0),
        cmap=cmap,
        norm=LogNorm(vmin=1.0, vmax=most),
        zorder=2,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(limit, -limit)
    ax.set_aspect('equal')
    for radius in _ring_radii(limit):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec=_HEATMAP_GRID, lw=0.8, zorder=3))
        ax.text(
            -radius * 0.7071,
            -radius * 0.7071,
            f'{radius:g} px',
            color=_HEATMAP_INK,
            fontsize=8,
            ha='center',
            va='center',
            zorder=5,
            bbox={'fc': 'white', 'ec': 'none', 'pad': 1.2, 'alpha': 0.8},
        )
    ax.axhline(0, color=_HEATMAP_GRID, lw=0.8, zorder=3)
    ax.axvline(0, color=_HEATMAP_GRID, lw=0.8, zorder=3)
    ax.plot([0], [0], marker='+', ms=12, mew=1.8, color='#d65f5f', zorder=4)
    ax.set_xlabel('dU (px), right')
    ax.set_ylabel('dV (px), down')
    ax.set_title(_heatmap_caption(offsets, limit), fontsize=10, color=_HEATMAP_INK)
    bin_size = 2.0 * limit / bins
    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.8, ticks=_count_ticks(most), format='{x:g}')
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    size_text = f'{bin_size:.0f}' if bin_size >= 10.0 else f'{bin_size:.2g}'
    colorbar.set_label(f'images per {size_text} px bin')
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, plt, path)
    return True
