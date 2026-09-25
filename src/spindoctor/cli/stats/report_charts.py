"""Chart writers for the statistics report.

Every chart is rendered with the Agg backend into a PNG buffer and written
through ``FCPath``, so the same inputs give the same bytes and the output
directory can be a local path or any URL the ``filecache`` layer accepts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

from filecache import FCPath

__all__ = [
    'instrument_color',
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
