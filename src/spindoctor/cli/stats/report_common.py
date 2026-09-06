"""Shared state, accumulators, formatting, and chart helpers for the statistics report.

The report is one pass over the record seam followed by a formatting run over
what that pass accumulated, and this module holds the join between the two
halves.  :class:`ReportStatistics` is what a section's numbers are read off, and
:class:`ReportContext` is what carries it to each section builder alongside the
output directory and the drill-down options.

The accumulators are declared here rather than beside the pass that fills them
because :class:`ReportContext` names one and the section builders name a
context, so the pass is written on top of this module rather than under it.

Every number a section prints comes from the accumulators, and the accumulators
come from whichever storage answered the seam, so one report is written over a
results tree and over an index ingested from it.
"""

from __future__ import annotations

import heapq
import math
import re
import statistics
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Generic, Protocol, TypeVar

from filecache import FCPath

__all__ = [
    'BotsimFrame',
    'MemoryImage',
    'ReportContext',
    'ReportStatistics',
    'SuspectOffset',
    'TimedImage',
    'TopImages',
    'add_drilldown',
    'add_instrument_count_table',
    'count_pct',
    'fmt',
    'image_name_from_filename',
    'instrument_color',
    'offset_stats',
    'percentile',
    'safe_filename',
    'write_offset_hist',
    'write_stacked_bar_chart',
    'write_stacked_value_hist',
]


# A record carries ``observation.image_name``, which is the source file's
# basename (``N1454725799_1_CALIB.IMG``).  The dataset layer's notion of an
# image name is the shorter token its ``--image-filelist`` selection matches
# against (``N1454725799``), so every name the report prints or writes to a
# filelist goes through the per-instrument rule below.  An unregistered
# instrument falls back to the extension-stripped basename.
_IMAGE_NAME_RULES: dict[str, Callable[[str], str]] = {
    # [NW]dddddddddd[_d[_CALIB]]
    'coiss': lambda stem: stem.split('_', 1)[0],
    # Cddddddd[_GEOMED]
    'vgiss': lambda stem: stem.split('_', 1)[0],
    # Cdddddddddd[RS] (no suffix to strip)
    'gossi': lambda stem: stem,
    # lor_dddddddddd[_0xddd_sci]
    'nhlorri': lambda stem: stem[:14],
}


def image_name_from_filename(instrument: str, filename: str) -> str:
    """The dataset-level image name for a recorded image filename.

    Parameters:
        instrument: Registered instrument name the record carries
            (``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri``); an
            unregistered name only has its extension stripped.
        filename: The recorded image name (a basename, possibly with a
            directory prefix).

    Returns:
        The image name in the form ``--image-filelist`` selects on, e.g.
        ``N1454725799_1_CALIB.IMG`` yields ``N1454725799`` and
        ``lor_0003103486_0x630_sci.fit`` yields ``lor_0003103486``.
    """
    stem = filename.rsplit('/', 1)[-1]
    dot = stem.rfind('.')
    if dot > 0:
        stem = stem[:dot]
    rule = _IMAGE_NAME_RULES.get(instrument.lower())
    return rule(stem) if rule is not None else stem


@dataclass(frozen=True)
class SuspectOffset:
    """One successful image whose fused offset reaches near the search limit.

    Parameters:
        ratio: How far into the search box the offset reaches, as a fraction of
            the per-axis limit, over whichever axis reaches further.
        image_name: The recorded image filename.
        instrument: The image's instrument.
        offset_dv: The fused V-axis offset, in pixels.
        offset_du: The fused U-axis offset, in pixels.
        magnitude: The offset's length, in pixels.
        limit_text: The per-axis limit, rendered as the table cell shows it.
        root_url: The results root the image was read under.
        results_path_stub: The image's key under that root.
    """

    ratio: float
    image_name: str
    instrument: str
    offset_dv: float
    offset_du: float
    magnitude: float
    limit_text: str
    root_url: str
    results_path_stub: str

    @property
    def rank(self) -> tuple[float, str, str, str]:
        """Where this image sorts in the suspect table.

        Returns:
            Worst ratio first, then the image name, then the pair that keys the
            image.  The pair is part of the key because an image name is not
            unique across roots: two roots holding one basename would otherwise
            be separated by whatever order the records arrived in, which is an
            order no source promises.
        """
        return (-self.ratio, self.image_name, self.root_url, self.results_path_stub)


@dataclass(frozen=True)
class BotsimFrame:
    """One frame of a possible BOTSIM pair, as the pair check reads it.

    Parameters:
        image_name: The recorded image filename.
        root_url: The results root the image was read under.
        results_path_stub: The image's key under that root.
        status: The image's navigation status.
        offset_dv: The fused V-axis offset, or None when none was recorded.
        offset_du: The fused U-axis offset, or None when none was recorded.
    """

    image_name: str
    root_url: str
    results_path_stub: str
    status: str
    offset_dv: float | None
    offset_du: float | None

    @property
    def identity(self) -> tuple[str, str, str]:
        """What decides which frame stands for a clock count and a camera.

        Returns:
            The image name and the pair that keys the image.  Two images of one
            camera sharing a clock count is a tree nobody expects, and the one
            with the smallest identity is taken, so the answer does not depend
            on which of them the source happened to yield first.
        """
        return (self.image_name, self.root_url, self.results_path_stub)


@dataclass(frozen=True)
class TimedImage:
    """One image's run time, ordered as the slowest-image list orders it.

    Parameters:
        elapsed_s: What the run recorded for this image, in seconds.
        image_name: The recorded image filename.
        instrument: The image's instrument.
        root_url: The results root the image was read under.
        results_path_stub: The image's key under that root.
    """

    elapsed_s: float
    image_name: str
    instrument: str
    root_url: str
    results_path_stub: str

    @property
    def rank(self) -> tuple[float, str, str, str]:
        """Where this image sorts in the slowest-image list.

        Returns:
            Slowest first, then the image name, then the pair that keys the
            image, for the reason :attr:`SuspectOffset.rank` carries the pair.
        """
        return (-self.elapsed_s, self.image_name, self.root_url, self.results_path_stub)

    def __lt__(self, other: TimedImage) -> bool:
        """Compare two images the opposite way round from how they are listed.

        A heap hands back its smallest element, and what a bounded slowest-N
        wants to hand back is the one it would print last, so the comparison is
        inverted here rather than at every use of the heap.

        Parameters:
            other: The image to compare against.

        Returns:
            True when this image sorts *later* in the printed list.
        """
        return self.rank > other.rank


@dataclass(frozen=True)
class MemoryImage:
    """One image's peak memory, ordered as the hungriest-image list orders it.

    Parameters:
        peak_memory_gib: The largest resident size the run reached for this
            image, in GiB.
        image_name: The recorded image filename.
        instrument: The image's instrument.
        root_url: The results root the image was read under.
        results_path_stub: The image's key under that root.
    """

    peak_memory_gib: float
    image_name: str
    instrument: str
    root_url: str
    results_path_stub: str

    @property
    def rank(self) -> tuple[float, str, str, str]:
        """Where this image sorts in the hungriest-image list.

        Returns:
            Hungriest first, then the image name, then the pair that keys the
            image, for the reason :attr:`SuspectOffset.rank` carries the pair.
        """
        return (-self.peak_memory_gib, self.image_name, self.root_url, self.results_path_stub)

    def __lt__(self, other: MemoryImage) -> bool:
        """Compare two images the opposite way round from how they are listed.

        Parameters:
            other: The image to compare against.

        Returns:
            True when this image sorts *later* in the printed list.
        """
        return self.rank > other.rank


class _Ranked(Protocol):
    """What the bounded list needs of whatever it holds.

    The heap orders by comparison and the printed list orders by ``rank``, so
    an entry has to offer both, and offer them the opposite way round from each
    other for the reason :meth:`TimedImage.__lt__` gives.
    """

    @property
    def rank(self) -> tuple[float, str, str, str]:
        """Where the entry sorts in the printed list."""

    def __lt__(self, other: Any) -> bool:
        """Whether this entry sorts later in the printed list than another."""


_RankedT = TypeVar('_RankedT', bound=_Ranked)


class TopImages(Generic[_RankedT]):
    """The leading images of a pass by one measure, kept without holding the rest.

    A list of every image's value is retained anyway, packed, because the
    quantiles and the histogram need every value.  The names are not: the
    leading list is the only thing that wants them, it is bounded by
    ``--top-n``, and a name is the expensive half of a per-image retention.

    Parameters:
        limit: How many to keep.  Zero keeps none, which is what the default
            ``--top-n 0`` asks for, and the section then prints no list at all.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(0, limit)
        self._held: list[_RankedT] = []

    def add(self, image: _RankedT) -> None:
        """Offer one image to the list, keeping it only if it belongs there.

        Parameters:
            image: The image and what it took.
        """
        if self._limit == 0:
            return
        if len(self._held) < self._limit:
            heapq.heappush(self._held, image)
            return
        heapq.heappushpop(self._held, image)

    @property
    def entries(self) -> list[_RankedT]:
        """The images kept, in the order the report lists them.

        Returns:
            Leading first, ties broken by name and then by the pair that keys
            the image.
        """
        return sorted(self._held, key=lambda image: image.rank)


@dataclass
class ReportStatistics:
    """Every number the report prints, accumulated in one pass over the records.

    Each field is keyed exactly as the section that reads it groups: a counter
    where the section counts, a packed array where it needs every value for a
    mean, a median, a percentile or a histogram, and a bounded structure where
    it prints a top-N list.  Two retentions grow with the images rather than
    with a fixed space of keys, and each is described where it is declared: the
    suspect list, which is the one always-on section printing a row per image
    and which ``--top-n 0`` leaves uncapped, and the Cassini frame held per
    spacecraft-clock count for the BOTSIM pairing.  Everything else that names
    an image -- the four drill-down lists and the filelists -- is held only
    where one of those was asked for.

    The two-level counters carry the section's key and then the instrument,
    because every count table has one column per instrument and a total.

    Parameters:
        top_n: How many rows the capped tables print, and how many names a
            drill-down shows.
        retain_names: Whether any section will name images, which is true when
            examples or filelists were asked for.  Off, the four drill-down
            lists retain nothing at all.
        suspect_fraction: Fraction of the per-axis search limit at or beyond
            which a fused offset is screened as suspect.
        images_by_instrument: Selected images per instrument.
        unreadable_files: Files under the selected roots named like a
            navigation document that no record could be read out of.  Root
            scoped rather than selection scoped: such a file records no
            instrument, date or image number, so no filter can be applied to it.
        failed_images: Selected images whose status is not ``success``.
        csv_rows: Rows the CSV export wrote, for the line naming the file.
        first_image: Per instrument, the lowest ``(image_number, image_name,
            root_url, results_path_stub)`` of the images that carry a number.
        last_image: The same, with the number negated, so the entry held is the
            highest-numbered image and the lowest name among any that tie.
        first_et: Per instrument, the earliest recorded epoch.
        last_et: Per instrument, the latest recorded epoch.
        status_counts: Images per status.
        reason_counts: Images per ``(status, failure reason)``.
        failure_names: Per failure reason, the images that carried it; retained
            only under ``retain_names``.
        content_counts: Failed images per scene-content category.
        content_reason_counts: Failed images per ``(content, failure reason)``.
        content_names: Per content category, the failed images in it; retained
            only under ``retain_names``.
        body_failed: Failed images naming each ``(body, instrument)``.
        body_success: Successful images naming each ``(body, instrument)``.
        body_failed_names: Per ``(body, instrument)``, the failed images naming
            it; retained only under ``retain_names``.
        technique_images: Images each ``(technique, instrument)`` ran on.
        technique_good: How many of those the technique did not call spurious.
        technique_confidence: Per ``(technique, instrument)``, every confidence
            it reported, packed.  A confidence that was never recorded is no
            part of the population, so the mean skips it and a group that
            reported none has no mean rather than a zero.  The values are held
            rather than folded into a running total so that the mean is one
            call over the population, whose answer is the same whatever order
            the records arrived in.  An exact running total would be order-free
            too -- the non-overlapping partials an exact sum is made of are a
            couple of floats however many values go by -- but that is a small
            algorithm to write and a smaller one to get quietly wrong, against
            a packed array costing a few percent of the pass's peak.
        source_images: Images each ``(model, source, instrument)`` appears in,
            counted once per image however many feature types it supplied.
        source_features: The features and gated features it supplied, summed.
        offsets: Per ``(instrument, camera, image size)``, the fused offsets of
            the successful images, packed per axis.  The per-camera section
            pools these over the sizes rather than keeping a second copy.
        pair_deltas: Per ``(instrument, technique, technique)``, the distances
            between the two techniques' offsets on the images where both
            reported non-spuriously.
        rank_disagreement: Per ``(instrument, confidence tier)``, each image's
            largest such distance.
        tier_counts: Images per confidence tier.
        exclusion_counts: Images per set of techniques the ensemble excluded.
        exclusion_names: The images in each such set; retained only under
            ``retain_names``.
        screened: Successful images whose search limit could be resolved.
        suspect_counts: How many of those reached the limit.
        unresolved: Per reason, the images whose limit could not be resolved.
        suspects: Every suspect image, uncapped.
        botsim: Cassini images that name a clock count, by clock count and then
            by camera letter.
        elapsed_by_instrument: Per instrument, every recorded run time.
        slowest: The slowest images, bounded by ``top_n``.
        peak_memory_by_instrument: Per instrument, every recorded peak memory,
            in GB.
        hungriest: The images that reached the largest peaks, bounded by
            ``top_n``.
    """

    top_n: int = 0
    retain_names: bool = False
    suspect_fraction: float = 0.9

    images_by_instrument: dict[str, int] = field(default_factory=dict)
    unreadable_files: int = 0
    failed_images: int = 0
    csv_rows: int = 0

    first_image: dict[str, tuple[int, str, str, str]] = field(default_factory=dict)
    last_image: dict[str, tuple[int, str, str, str]] = field(default_factory=dict)
    first_et: dict[str, float] = field(default_factory=dict)
    last_et: dict[str, float] = field(default_factory=dict)

    status_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    reason_counts: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    failure_names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    content_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    content_reason_counts: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    content_names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    body_failed: dict[tuple[str, str], int] = field(default_factory=dict)
    body_success: dict[tuple[str, str], int] = field(default_factory=dict)
    body_failed_names: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    technique_images: dict[tuple[str, str], int] = field(default_factory=dict)
    technique_good: dict[tuple[str, str], int] = field(default_factory=dict)
    technique_confidence: dict[tuple[str, str], array[float]] = field(default_factory=dict)

    source_images: dict[tuple[str, str, str], int] = field(default_factory=dict)
    source_features: dict[tuple[str, str, str], list[int]] = field(default_factory=dict)

    offsets: dict[tuple[str, str, str], tuple[array[float], array[float]]] = field(
        default_factory=dict
    )

    pair_deltas: dict[tuple[str, str, str], array[float]] = field(default_factory=dict)
    rank_disagreement: dict[tuple[str, str], array[float]] = field(default_factory=dict)
    tier_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    exclusion_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    exclusion_names: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    screened: dict[str, int] = field(default_factory=dict)
    suspect_counts: dict[str, int] = field(default_factory=dict)
    unresolved: dict[str, int] = field(default_factory=dict)
    suspects: list[SuspectOffset] = field(default_factory=list)

    botsim: dict[str, dict[str, BotsimFrame]] = field(default_factory=dict)

    elapsed_by_instrument: dict[str, array[float]] = field(default_factory=dict)
    slowest: TopImages[TimedImage] = field(default_factory=lambda: TopImages(0))
    peak_memory_by_instrument: dict[str, array[float]] = field(default_factory=dict)
    hungriest: TopImages[MemoryImage] = field(default_factory=lambda: TopImages(0))

    def __post_init__(self) -> None:
        """Size the bounded heaps, which ``top_n`` bounds rather than the data."""
        self.slowest = TopImages(self.top_n)
        self.hungriest = TopImages(self.top_n)


@dataclass
class ReportContext:
    """Mutable state threaded through every report section builder.

    Parameters:
        output_dir: Directory receiving ``report.md``, charts, and the
            optional ``filelists/`` subdirectory; a local directory or
            any URL the ``filecache`` layer accepts.
        stats: What the pass over the records accumulated.  Every number a
            section prints is read off this and nothing else.
        lines: Markdown lines accumulated so far.
        top_n: When positive, categorical sections list up to this many
            example image names per category.
        filelists: When True, categorical sections write one plain-text
            file per category and instrument under ``filelists/``.
        suspect_fraction: Fraction of the per-axis search limit at or
            beyond which a fused offset is flagged as suspect.
        images_by_instrument: Selected image count per instrument, in
            instrument-name order; read off the accumulators.
        instruments: The selected instrument names, in name order.
        total_images: Total selected images across all instruments.
    """

    output_dir: FCPath
    stats: ReportStatistics
    lines: list[str] = field(default_factory=list)
    top_n: int = 0
    filelists: bool = False
    suspect_fraction: float = 0.9
    images_by_instrument: dict[str, int] = field(default_factory=dict)
    instruments: list[str] = field(default_factory=list)
    total_images: int = 0

    def __post_init__(self) -> None:
        """Read the per-instrument image counts the whole report reports against.

        In instrument-name order, because that is the order every count table
        lays its columns out in and the order the charts stack their segments
        in, and a pass over records promises no order of its own.
        """
        self.images_by_instrument = dict(sorted(self.stats.images_by_instrument.items()))
        self.instruments = list(self.images_by_instrument)
        self.total_images = sum(self.images_by_instrument.values())

    def write_filelist(self, stub: str, names: list[str]) -> str:
        """Write one image name per line into ``filelists/<stub>.txt``.

        The written file is directly consumable by the datasets'
        ``--image-filelist`` option: one image name per line, with a
        leading ``#`` comment line naming the category.

        Parameters:
            stub: Category identifier; sanitized into a filesystem-safe
                filename.
            names: Image names to write (written in the given order).

        Returns:
            The path of the written file, relative to ``output_dir``.
        """
        relative = f'filelists/{safe_filename(stub)}.txt'
        body = f'# {stub} ({len(names)} image(s))\n' + ''.join(f'{name}\n' for name in names)
        (self.output_dir / relative).write_text(body, encoding='utf-8')
        return relative


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a float for a Markdown table cell.

    Parameters:
        value: Value to format, or None.
        digits: Decimal places.

    Returns:
        The fixed-point string, or ``'-'`` for None.
    """
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def count_pct(count: int, total: int) -> str:
    """Format an image count as ``'5 (3.2%)'``.

    Parameters:
        count: The number of images.
        total: The number of images the percentage is taken against; a
            zero total renders as ``0.0%``.

    Returns:
        The count followed by its percentage of ``total``.
    """
    fraction = count / total if total > 0 else 0.0
    return f'{count} ({fraction * 100:.1f}%)'


def add_instrument_count_table(
    ctx: ReportContext,
    table_rows: list[tuple[list[str], dict[str, int]]],
    *,
    headers: list[str],
) -> None:
    """Append a count table with one column per instrument plus a total.

    Every count cell is rendered by :func:`count_pct`.  An instrument
    column's percentage is of that instrument's selected images; the total
    column's percentage is of all selected images.

    Parameters:
        ctx: Report context.
        table_rows: ``(leading_cells, counts_by_instrument)`` pairs in the
            order they should be listed; instruments absent from a row's
            mapping count zero.
        headers: Column headings for the leading cells (e.g.
            ``['status']``); the instrument and total headings are added
            here.
    """
    columns = [*headers, *ctx.instruments, 'total']
    ctx.lines += [
        '| ' + ' | '.join(columns) + ' |',
        '|' + '---|' * len(columns),
    ]
    for leading, counts in table_rows:
        cells = list(leading)
        for instrument in ctx.instruments:
            cells.append(count_pct(counts.get(instrument, 0), ctx.images_by_instrument[instrument]))
        cells.append(count_pct(sum(counts.values()), ctx.total_images))
        ctx.lines.append('| ' + ' | '.join(cells) + ' |')
    ctx.lines.append('')


def offset_stats(values: Sequence[float]) -> dict[str, float] | None:
    """Mean / median / stdev / min / max summary of a value list.

    Parameters:
        values: Values to summarize; may be empty.

    Returns:
        A dict with ``mean`` / ``median`` / ``stdev`` / ``min`` / ``max``
        keys (``stdev`` is 0.0 for a single value), or None when
        ``values`` is empty.
    """
    if len(values) == 0:
        return None
    return {
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty value list.

    Parameters:
        values: Non-empty list of values.
        fraction: Percentile as a fraction in ``[0, 1]`` (e.g. ``0.95``).

    Returns:
        The nearest-rank percentile value.
    """
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def safe_filename(stub: str) -> str:
    """Collapse a category label into a filesystem-safe filename stub.

    Parameters:
        stub: Arbitrary category label (may contain spaces, slashes,
            punctuation).

    Returns:
        The label with every run of unsafe characters replaced by ``_``
        and leading/trailing underscores stripped; ``'unnamed'`` when
        nothing survives.
    """
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', stub).strip('_')
    return cleaned or 'unnamed'


def add_drilldown(
    ctx: ReportContext,
    categories: list[tuple[str, list[tuple[str, str]]]],
    *,
    label: str,
    stub_prefix: str,
) -> None:
    """Append per-category example image names and write per-category filelists.

    Both the inline examples and the filelists are grouped by instrument:
    each category contributes one line (and one file) per instrument that
    has images in it.  Honors ``ctx.top_n`` (examples shown inline,
    alphabetically first N) and ``ctx.filelists`` (full alphabetical lists
    written to ``filelists/``, named ``<stub_prefix>_<category>_<inst>``).
    Does nothing when both are off.

    Parameters:
        ctx: Report context.
        categories: ``(category_label, entries)`` pairs in the order they
            should be listed, where each entry is an
            ``(instrument, image_filename)`` pair; filenames are converted
            to image names by :func:`image_name_from_filename` and sorted
            here.
        label: Human-readable singular noun for the category kind, used
            in the "Examples (up to N per ...)" line.
        stub_prefix: Filelist filename prefix; the category label and
            instrument are appended.
    """
    if ctx.top_n <= 0 and not ctx.filelists:
        return
    # (category, instrument) -> sorted image names, keeping the caller's
    # category order and instrument-name order within each category.
    grouped: list[tuple[str, str, list[str]]] = []
    for category, entries in categories:
        by_instrument: dict[str, list[str]] = {}
        for instrument, filename in entries:
            by_instrument.setdefault(instrument, []).append(
                image_name_from_filename(instrument, filename)
            )
        for instrument in sorted(by_instrument):
            grouped.append((category, instrument, sorted(by_instrument[instrument])))
    if len(grouped) == 0:
        return
    if ctx.top_n > 0:
        ctx.lines += [f'Examples (up to {ctx.top_n} per {label} and instrument):', '']
        for category, instrument, names in grouped:
            shown = ', '.join(names[: ctx.top_n])
            ctx.lines.append(f'- {category} / {instrument}: {shown}')
        ctx.lines.append('')
    if ctx.filelists:
        references = [
            ctx.write_filelist(f'{stub_prefix}_{category}_{instrument}', names)
            for category, instrument, names in grouped
        ]
        ctx.lines += [f'Full lists: {", ".join(references)}', '']


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
