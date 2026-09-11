"""Report sections: failure taxonomy, suspect offsets, BOTSIM, run time, CSV export.

Every section here formats what the pass over the records already counted.  None
of them reads a record, and none of them holds anything of its own beyond the
lines it appends, so the cost of a section is the size of its table rather than
the size of the tree.

The CSV export is the exception, because it is not a section: it writes one row
per image as the pass reads it, and only its closing line is a section.  It is
here rather than beside the pass because what a column of it holds is a decision
about the export, and the pass has no opinion about any of them.
"""

from __future__ import annotations

import contextlib
import csv
import heapq
import json
import math
import statistics
from array import array
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from filecache import FCPath

from spindoctor.cli.stats.report_common import (
    ReportContext,
    add_drilldown,
    add_instrument_count_table,
    count_pct,
    fmt,
    image_name_from_filename,
    percentile,
    write_stacked_value_hist,
)
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.nav_records import ImageFacts
from spindoctor.results_index import IMAGES

__all__ = [
    'CONTENT_CATEGORIES',
    'CSV_LINE_TERMINATOR',
    'EXPORT_COLUMNS',
    'IMAGE_COLUMNS',
    'MULTILINE_COLUMNS',
    'CsvExport',
    'add_botsim_section',
    'add_csv_export_section',
    'add_failure_taxonomy_section',
    'add_memory_section',
    'add_narrowing_section',
    'add_offset_by_group_section',
    'add_runtime_section',
    'add_suspect_offset_section',
    'content_category',
    'resolve_offset_limit',
    'source_kind',
]


# ---------------------------------------------------------------------------
# What the report was narrowed to
# ---------------------------------------------------------------------------

_DROPPED_ROOTS_PROSE = """\
The index holds no completed ingest of the roots named below, so under one of them the absence of a
row means nothing at all, and none of them is covered here: neither its images nor the files under
it that yielded no record. Ingest such a root and it joins the roots the filters above name; name it
with --root and the run is refused rather than quietly narrowed."""
"""The paragraph that explains what a root the report could not cover contributes.

Written with the wrapping it is printed with, and naming no root, so that the
prose is a constant and the roots are a line of its own after it.
"""


def add_narrowing_section(
    ctx: ReportContext, *, filters: Sequence[str], dropped_roots: Sequence[str]
) -> None:
    """Append what the report was narrowed to, and what it could not cover.

    A report that covered fewer roots than it was pointed at has to say so.  The
    roots it did cover are one of the filters, named as a filter, and the ones it
    dropped are named under the paragraph that says what dropping one costs.

    Parameters:
        ctx: Report context.
        filters: What narrowed the report, already rendered, in the order to
            print them; empty means nothing narrowed it.
        dropped_roots: The roots the source could not be bound to, in the order
            to name them; empty means it was bound to every root it was pointed
            at.
    """
    narrowing = ', '.join(filters) if len(filters) > 0 else 'none (every image read)'
    ctx.lines += [f'Filters: {narrowing}', '']
    if len(dropped_roots) == 0:
        return
    ctx.lines += [
        *_DROPPED_ROOTS_PROSE.splitlines(),
        '',
        f'Roots dropped: {", ".join(dropped_roots)}',
        '',
    ]


# ---------------------------------------------------------------------------
# Suspect-offset screen
# ---------------------------------------------------------------------------

# Config section per instrument holding ``extfov_margin_vu`` (the per-axis
# maximum expected pointing offset the search allows for).  Cassini ISS is
# handled separately because its sections are nested per detector and split
# between raw and CALIB blocks.
_INSTRUMENT_CONFIG_SECTIONS = {
    'gossi': 'galileo_ssi',
    'nhlorri': 'newhorizons_lorri',
    'sim': 'sim',
    'vgiss': 'voyager_iss',
}


def resolve_offset_limit(
    instrument: str,
    image_name: str,
    image_shape_v: int | None,
    *,
    config: Config | None = None,
) -> tuple[float, float] | str:
    """Resolve the per-axis maximum expected pointing offset for one image.

    The limit is the configured ``extfov_margin_vu`` search margin for the
    instrument (for Cassini ISS: per detector, chosen from the image-name
    prefix ``N``/``W`` and the raw-vs-CALIB config block; for margin tables
    keyed by image size, the recorded ``image_shape_v`` selects the entry).

    Parameters:
        instrument: Registered instrument name the record carries.
        image_name: Image name (used to pick the Cassini ISS detector and
            config block).
        image_shape_v: Recorded V-axis image size, or None when the
            record carries no shape.
        config: Configuration to read; None uses ``DEFAULT_CONFIG``.

    Returns:
        ``(limit_v, limit_u)`` in pixels, or a human-readable string
        explaining why the limit could not be resolved.
    """
    config = config or DEFAULT_CONFIG
    if instrument == 'coiss':
        section_name = 'cassini_iss_calib' if '_CALIB' in image_name.upper() else 'cassini_iss'
        detector = {'N': 'nac', 'W': 'wac'}.get(image_name.rsplit('/', 1)[-1][:1].upper())
        if detector is None:
            return 'cannot determine NAC/WAC from the image name'
        section = config.category(section_name)
        detector_config = section.get(detector) if section else None
        entry = detector_config.get('extfov_margin_vu') if detector_config else None
        source = f'{section_name}.{detector}'
    else:
        if instrument not in _INSTRUMENT_CONFIG_SECTIONS:
            return f'no configured search limit for instrument {instrument!r}'
        section_name = _INSTRUMENT_CONFIG_SECTIONS[instrument]
        section = config.category(section_name)
        entry = section.get('extfov_margin_vu') if section else None
        source = section_name
    if entry is None:
        return f'config section {source!r} has no extfov_margin_vu'
    if isinstance(entry, dict):
        if image_shape_v is None:
            return 'image shape not recorded'
        if image_shape_v not in entry:
            return f'{source!r} has no extfov_margin_vu entry for image size {image_shape_v}'
        entry = entry[image_shape_v]
    limit_v = float(entry[0])
    limit_u = float(entry[1])
    if limit_v <= 0.0 or limit_u <= 0.0:
        return f'{source!r} extfov_margin_vu is not positive'
    return limit_v, limit_u


def add_suspect_offset_section(ctx: ReportContext) -> None:
    """Flag successful images whose fused offset sits near the search limit.

    An offset at the edge of the search box is as likely to be a
    correlation artifact as a real pointing error, so any image with
    ``|dV|`` or ``|dU|`` at or beyond ``suspect_fraction`` times the
    per-axis limit is listed for operator review.

    This is the one always-on section that prints a row per image, and
    ``--top-n`` caps it only when it is given: the default lists every suspect,
    because an operator screening a run needs the whole list rather than the
    worst few of it.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    ctx.lines += [
        '## Suspect offsets (near the search limit)',
        '',
        f'Successful images whose fused offset reaches at least '
        f'{ctx.suspect_fraction:.2f} of the per-axis maximum expected pointing '
        'offset (the configured extfov search margin) on either axis.  These '
        'offsets may be correlation artifacts pinned to the search boundary.',
        '',
    ]
    suspects = sorted(stats.suspects, key=lambda suspect: suspect.rank)
    n_screened = sum(stats.screened.values())
    ctx.lines += [
        f'Suspect images: {count_pct(len(suspects), ctx.total_images)} of {n_screened} screened.',
        '',
    ]
    add_instrument_count_table(ctx, [(['suspect'], stats.suspect_counts)], headers=['category'])
    if len(suspects) > 0:
        shown = suspects[: ctx.top_n] if ctx.top_n > 0 else suspects
        ctx.lines += [
            '| image | instrument | dV | dU | magnitude | limit (v, u) |',
            '|---|---|---|---|---|---|',
        ]
        for suspect in shown:
            name = image_name_from_filename(suspect.instrument, suspect.image_name)
            ctx.lines.append(
                f'| {name} | {suspect.instrument} | {fmt(suspect.offset_dv)} '
                f'| {fmt(suspect.offset_du)} | {fmt(suspect.magnitude)} | {suspect.limit_text} |'
            )
        ctx.lines.append('')
        add_drilldown(
            ctx,
            [('suspect', [(suspect.instrument, suspect.image_name) for suspect in suspects])],
            label='category',
            stub_prefix='suspect_offsets',
        )
    if len(stats.unresolved) > 0:
        ctx.lines.append('Search limit could not be resolved for some images:')
        ctx.lines.append('')
        for reason in sorted(stats.unresolved):
            ctx.lines.append(f'- {reason} ({stats.unresolved[reason]} image(s))')
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# BOTSIM pair consistency (Cassini ISS)
# ---------------------------------------------------------------------------


def add_botsim_section(ctx: ReportContext) -> None:
    """Compare NAC and WAC offsets over simultaneously shuttered Cassini pairs.

    BOTSIM observations shutter both cameras at once, so the two frames
    share one spacecraft-clock count and see the same pointing.  One WAC
    pixel is ten NAC pixels; a consistent pair therefore satisfies
    ``NAC offset ~= 10 x WAC offset`` per axis, making the per-axis
    residual ``NAC - 10 x WAC`` an end-to-end accuracy check that needs
    no ground truth.

    Parameters:
        ctx: Report context.
    """
    pairs = {clock: frames for clock, frames in ctx.stats.botsim.items() if len(frames) == 2}
    residuals: list[tuple[float, str, str, str, float, float]] = []
    for clock in sorted(pairs):
        nac = pairs[clock]['N']
        wac = pairs[clock]['W']
        if nac.status != 'success' or wac.status != 'success':
            continue
        nac_dv, nac_du = nac.offset_dv, nac.offset_du
        wac_dv, wac_du = wac.offset_dv, wac.offset_du
        if nac_dv is None or nac_du is None or wac_dv is None or wac_du is None:
            continue
        residual_dv = nac_dv - 10.0 * wac_dv
        residual_du = nac_du - 10.0 * wac_du
        residuals.append(
            (
                math.hypot(residual_dv, residual_du),
                clock,
                nac.image_name,
                wac.image_name,
                residual_dv,
                residual_du,
            )
        )
    ctx.lines += [
        '## BOTSIM pair consistency (Cassini ISS)',
        '',
        'BOTSIM observations shutter the NAC and WAC simultaneously (the image',
        'names share one spacecraft-clock count).  One WAC pixel is ten NAC',
        'pixels, so a consistent pair satisfies NAC offset ~= 10 x WAC offset',
        'per axis.  Residuals below are NAC - 10 x WAC, in NAC pixels.',
        '',
        '| metric | value |',
        '|---|---|',
        f'| pairs identified | {len(pairs)} |',
        f'| pairs with both navigated | {len(residuals)} |',
    ]
    if len(residuals) > 0:
        magnitudes = [residual[0] for residual in residuals]
        ctx.lines += [
            f'| median residual (px) | {fmt(statistics.median(magnitudes))} |',
            f'| p95 residual (px) | {fmt(percentile(magnitudes, 0.95))} |',
        ]
    ctx.lines.append('')
    if len(residuals) > 0 and ctx.top_n > 0:
        # A clock count belongs to one pair, so the key is a total order over
        # the residuals and the worst few come off a bounded heap.  What that
        # bounds is the selection rather than the section: the residuals above
        # are built whatever --top-n says, because the median and the
        # percentile need every one of them.  What it saves is the copy a full
        # sort makes of them, which over a hundred thousand pairs is a few
        # kilobytes against ten megabytes.
        worst = heapq.nsmallest(
            ctx.top_n, residuals, key=lambda residual: (-residual[0], residual[1])
        )
        ctx.lines += [
            f'Worst {len(worst)} pair(s):',
            '',
            '| clock | NAC image | WAC image | residual dV | residual dU | residual |',
            '|---|---|---|---|---|---|',
        ]
        for magnitude, clock, nac_name, wac_name, residual_dv, residual_du in worst:
            nac_image = image_name_from_filename('coiss', nac_name)
            wac_image = image_name_from_filename('coiss', wac_name)
            ctx.lines.append(
                f'| {clock} | {nac_image} | {wac_image} | {fmt(residual_dv)} | '
                f'{fmt(residual_du)} | {fmt(magnitude)} |'
            )
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# Failure taxonomy by image content
# ---------------------------------------------------------------------------

CONTENT_CATEGORIES = (
    'stars-only',
    'single-body',
    'multi-body',
    'rings-only',
    'body+rings',
    'no-features',
)
"""Scene-content categories a failed image is classified into, in report order."""


def source_kind(source_model: str) -> str:
    """Coarse feature-source kind of a model name.

    Parameters:
        source_model: The recorded ``source_model`` of a feature source.

    Returns:
        ``'stars'``, ``'rings'`` or ``'body'``.
    """
    kind = source_model.split(':', 1)[0].lower()
    if kind in ('star', 'stars'):
        return 'stars'
    if kind.startswith('ring'):
        return 'rings'
    return 'body'


def content_category(entries: list[tuple[str, str]]) -> str:
    """Classify an image's ``(source_model, source_name)`` inventory.

    Parameters:
        entries: The image's feature sources, as model and source name.

    Returns:
        One of :data:`CONTENT_CATEGORIES`.
    """
    if len(entries) == 0:
        return 'no-features'
    body_names = {name for model, name in entries if source_kind(model) == 'body'}
    has_rings = any(source_kind(model) == 'rings' for model, _ in entries)
    if len(body_names) > 0 and has_rings:
        return 'body+rings'
    if has_rings:
        return 'rings-only'
    if len(body_names) == 1:
        return 'single-body'
    if len(body_names) > 1:
        return 'multi-body'
    return 'stars-only'


def add_failure_taxonomy_section(ctx: ReportContext) -> None:
    """Classify failed images by scene content and localize per-body problems.

    Content comes from the ``feature_sources`` inventory recorded per
    image; a body whose failure share is far above its peers points at a
    modeling problem for that body rather than a pipeline-wide issue.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    if stats.failed_images == 0:
        return
    populated = [category for category in CONTENT_CATEGORIES if category in stats.content_counts]
    ctx.lines += [
        '## Failure taxonomy by image content',
        '',
        'Failed images classified by what the feature inventory says was in',
        'the scene.',
        '',
    ]
    add_instrument_count_table(
        ctx,
        [([category], stats.content_counts[category]) for category in populated],
        headers=['content'],
    )
    ordered_reasons = sorted(
        stats.content_reason_counts,
        key=lambda key: (
            CONTENT_CATEGORIES.index(key[0]),
            -sum(stats.content_reason_counts[key].values()),
            key[1],
        ),
    )
    add_instrument_count_table(
        ctx,
        [
            ([category, reason], stats.content_reason_counts[(category, reason)])
            for category, reason in ordered_reasons
        ],
        headers=['content', 'reason'],
    )
    add_drilldown(
        ctx,
        [(category, stats.content_names.get(category, [])) for category in CONTENT_CATEGORIES],
        label='content category',
        stub_prefix='failed_content',
    )
    _add_per_body_shares(ctx)


def _add_per_body_shares(ctx: ReportContext) -> None:
    """Append the per-body failure shares over all images, failed and successful.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    bodies = set(stats.body_failed) | set(stats.body_success)
    if len(bodies) == 0:
        return
    ranked = sorted(
        bodies,
        key=lambda key: (-_failure_share(ctx, key), -stats.body_failed.get(key, 0), key),
    )
    ctx.lines += [
        '### Per-body failure shares',
        '',
        'How often each named body appears in failed versus successful',
        'images; a body with a high failure share is a modeling problem.',
        '',
        '| body | instrument | failed images | successful images | failure share |',
        '|---|---|---|---|---|',
    ]
    for key in ranked:
        body, instrument = key
        n_failed = stats.body_failed.get(key, 0)
        n_success = stats.body_success.get(key, 0)
        total = ctx.images_by_instrument[instrument]
        ctx.lines.append(
            f'| {body} | {instrument} | {count_pct(n_failed, total)} '
            f'| {count_pct(n_success, total)} | {_failure_share(ctx, key):.3f} |'
        )
    ctx.lines.append('')
    by_body: dict[str, list[tuple[str, str]]] = {}
    for body, instrument in ranked:
        names = stats.body_failed_names.get((body, instrument), [])
        by_body.setdefault(body, []).extend((instrument, name) for name in sorted(names))
    add_drilldown(
        ctx,
        [(body, entries) for body, entries in by_body.items() if len(entries) > 0],
        label='body',
        stub_prefix='failed_body',
    )


def _failure_share(ctx: ReportContext, key: tuple[str, str]) -> float:
    """The fraction of one body's images that failed.

    Parameters:
        ctx: Report context.
        key: The body and the instrument.

    Returns:
        Failed images over all images naming that body under that instrument.
        The key comes from the union of the two counters, so at least one of
        them is non-zero and the denominator is never zero.
    """
    n_failed = ctx.stats.body_failed.get(key, 0)
    n_success = ctx.stats.body_success.get(key, 0)
    return n_failed / (n_failed + n_success)


# ---------------------------------------------------------------------------
# Run-time statistics
# ---------------------------------------------------------------------------


def add_runtime_section(ctx: ReportContext) -> None:
    """Summarize per-image run times; skipped when no timing data exists.

    Every statistic in the table is a function of the set of run times and not
    of the sequence they arrived in: the total is summed exactly, the mean sums
    exactly, and the rest are extremes or sort what they are given.  The pooled
    row is therefore a plain concatenation of the per-instrument arrays.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    by_instrument = stats.elapsed_by_instrument
    if len(by_instrument) == 0:
        return
    ctx.lines += [
        '## Run-time statistics',
        '',
        '| instrument | images | total (s) | min (s) | max (s) | mean (s) | median (s) '
        '| stdev (s) |',
        '|---|---|---|---|---|---|---|---|',
    ]
    series: list[tuple[str, array[float], int]] = [
        (
            instrument,
            by_instrument.get(instrument, array('d')),
            ctx.images_by_instrument[instrument],
        )
        for instrument in ctx.instruments
    ]
    # The pooled row only says something new once more than one instrument
    # contributed to it.
    if len(ctx.instruments) > 1:
        pooled = array('d')
        for instrument in ctx.instruments:
            pooled.extend(by_instrument.get(instrument, array('d')))
        series.append(('(all)', pooled, ctx.total_images))
    for instrument, values, denominator in series:
        if len(values) == 0:
            continue
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        ctx.lines.append(
            f'| {instrument} | {count_pct(len(values), denominator)} | {fmt(math.fsum(values))} '
            f'| {fmt(min(values))} | {fmt(max(values))} | {fmt(statistics.fmean(values))} '
            f'| {fmt(statistics.median(values))} | {fmt(stdev)} |'
        )
    ctx.lines.append('')
    write_stacked_value_hist(
        ctx.output_dir / 'runtime_hist.png',
        by_instrument,
        ctx.instruments,
        title='Per-image run time',
        xlabel='elapsed (s)',
    )
    ctx.lines += ['![run time](runtime_hist.png)', '']
    slowest = stats.slowest.entries
    if ctx.top_n > 0 and len(slowest) > 0:
        ctx.lines += [
            f'Slowest {len(slowest)} image(s):',
            '',
            '| image | instrument | elapsed (s) |',
            '|---|---|---|',
        ]
        for image in slowest:
            name = image_name_from_filename(image.instrument, image.image_name)
            ctx.lines.append(f'| {name} | {image.instrument} | {fmt(image.elapsed_s)} |')
        ctx.lines.append('')


def add_memory_section(ctx: ReportContext) -> None:
    """Write the peak-memory table, its histogram and the hungriest images.

    The peak is the largest resident size a navigation reached, which is the
    figure an out-of-memory kill is decided against, so the maximum is the
    column that sizes a worker and the distribution says how much of the pass
    runs nowhere near it.

    Every statistic in the table is a function of the set of peaks and not of
    the sequence they arrived in, so the pooled row is a plain concatenation of
    the per-instrument arrays.  The image count is of images that recorded a
    peak, which is not every image: a run on a kernel publishing none records
    none.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    by_instrument = stats.peak_memory_by_instrument
    if len(by_instrument) == 0:
        return
    ctx.lines += [
        '## Peak-memory statistics',
        '',
        '| instrument | images | min (GiB) | max (GiB) | mean (GiB) | median (GiB) | stdev (GiB) |',
        '|---|---|---|---|---|---|---|',
    ]
    series: list[tuple[str, array[float], int]] = [
        (
            instrument,
            by_instrument.get(instrument, array('d')),
            ctx.images_by_instrument[instrument],
        )
        for instrument in ctx.instruments
    ]
    # The pooled row only says something new once more than one instrument
    # contributed to it.
    if len(ctx.instruments) > 1:
        pooled = array('d')
        for instrument in ctx.instruments:
            pooled.extend(by_instrument.get(instrument, array('d')))
        series.append(('(all)', pooled, ctx.total_images))
    for instrument, values, denominator in series:
        if len(values) == 0:
            continue
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        ctx.lines.append(
            f'| {instrument} | {count_pct(len(values), denominator)} '
            f'| {fmt(min(values))} | {fmt(max(values))} | {fmt(statistics.fmean(values))} '
            f'| {fmt(statistics.median(values))} | {fmt(stdev)} |'
        )
    ctx.lines.append('')
    write_stacked_value_hist(
        ctx.output_dir / 'memory_hist.png',
        by_instrument,
        ctx.instruments,
        title='Per-image peak memory',
        xlabel='peak resident size (GiB)',
    )
    ctx.lines += ['![peak memory](memory_hist.png)', '']
    hungriest = stats.hungriest.entries
    if ctx.top_n > 0 and len(hungriest) > 0:
        ctx.lines += [
            f'Hungriest {len(hungriest)} image(s):',
            '',
            '| image | instrument | peak (GiB) |',
            '|---|---|---|',
        ]
        for image in hungriest:
            name = image_name_from_filename(image.instrument, image.image_name)
            ctx.lines.append(f'| {name} | {image.instrument} | {fmt(image.peak_memory_gib)} |')
        ctx.lines.append('')


# ---------------------------------------------------------------------------
# Per-instrument / per-image-size offset breakdown
# ---------------------------------------------------------------------------


def add_offset_by_group_section(ctx: ReportContext) -> None:
    """Break the fused-offset statistics down by (instrument, camera, image size).

    Parameters:
        ctx: Report context.
    """
    groups = ctx.stats.offsets
    if len(groups) == 0:
        return
    ctx.lines += [
        '### By instrument, camera, and image size',
        '',
        '| instrument | camera | size (v x u) | images '
        '| dV mean | dV stdev | dV min | dV max '
        '| dU mean | dU stdev | dU min | dU max |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for instrument, camera, size in sorted(groups):
        dv, du = groups[(instrument, camera, size)]
        stdev_dv = statistics.stdev(dv) if len(dv) > 1 else 0.0
        stdev_du = statistics.stdev(du) if len(du) > 1 else 0.0
        images = count_pct(len(dv), ctx.images_by_instrument[instrument])
        ctx.lines.append(
            f'| {instrument} | {camera} | {size} | {images} '
            f'| {fmt(statistics.fmean(dv))} | {fmt(stdev_dv)} | {fmt(min(dv))} | {fmt(max(dv))} '
            f'| {fmt(statistics.fmean(du))} | {fmt(stdev_du)} | {fmt(min(du))} | {fmt(max(du))} |'
        )
    ctx.lines.append('')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


IMAGE_COLUMNS: tuple[str, ...] = tuple(IMAGES.columns.keys())
"""Every column of one image's facts, in the order the column set declares them."""

_SORT_COLUMN = 'results_path_stub'
"""The column the export leads with, so an operator can sort the file in a shell.

The rows are not sorted.  Sorting them means holding every one of them, which is
what a streaming write exists not to do, so the file leads with the column an
operator would sort on -- field 1 of a ``sort -t,``, once the header line has
been held out of the sort -- and the root each row came from stays a column of
its own further right.
"""

MULTILINE_COLUMNS = frozenset({'status_traceback'})
"""Columns held out of the export because their value spans lines.

A traceback is the one recorded value carrying newlines of its own.  Quoting it
into a field is valid CSV and unreadable by the tools this file is for: a row
would span as many lines as the traceback has frames, and every count, ``sort``
and ``cut`` an operator reaches for reads those lines as rows.  The value is in
the index, which is where a query for it belongs; this file stays one line per
image.
"""

_EXPORT_IMAGE_COLUMNS: tuple[str, ...] = (
    _SORT_COLUMN,
    *(
        column
        for column in IMAGE_COLUMNS
        if column != _SORT_COLUMN and column not in MULTILINE_COLUMNS
    ),
)
"""The image's own columns, in the order the export writes them."""

_AGGREGATE_COLUMNS: tuple[str, ...] = (
    'n_technique_rows',
    'n_feature_sources',
    'n_features',
    'n_gated',
)
"""What the export adds beside the image's own columns, counted off its children."""

EXPORT_COLUMNS: tuple[str, ...] = (*_EXPORT_IMAGE_COLUMNS, *_AGGREGATE_COLUMNS)
"""Every column of ``images.csv``, in the order the export writes them."""

CSV_LINE_TERMINATOR = '\n'
"""What ends a row of the CSV export, on every platform."""


def _csv_value(value: Any) -> Any:
    """Render one column value for the CSV.

    A structured column arrives as the container it holds -- a matrix, a list of
    names, a mapping of diagnostics -- whichever storage answered, so a CSV
    carrying a Python container's repr is one nothing else can read back.  Such a
    value goes out as JSON text and everything else goes out as it came.

    Parameters:
        value: The value the facts carry for this column.

    Returns:
        JSON text for a list or a dict, and the value itself for anything else.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


class CsvExport:
    """The flattened one-row-per-image export, written as the pass reads.

    Rows are written where they are read rather than collected and written at
    the end, so a report over an archive-scale root pays for the file rather
    than for a copy of it in memory.  What that costs is the row order: the seam
    promises none and the two storages find records in two orders, so the file
    says which images were exported rather than in what sequence.  The first
    column is what an operator would sort on, for exactly that reason.

    Parameters:
        path: Where to write, a local path or any URL the ``filecache`` layer
            accepts.
    """

    def __init__(self, path: FCPath) -> None:
        self._path = path
        self._stack = contextlib.ExitStack()
        self._writer: Any = None

    def __enter__(self) -> CsvExport:
        """Open the file and write its header.

        Returns:
            The export itself, which the pass hands each image to.
        """
        # Stated rather than left to the module default, which is CRLF: this
        # file is read back by whatever an operator points at it, and a line
        # ending that changes with a library default is a diff nobody asked for.
        handle = self._stack.enter_context(self._path.open('w', newline='', encoding='utf-8'))
        self._writer = csv.writer(handle, lineterminator=CSV_LINE_TERMINATOR)
        self._writer.writerow(EXPORT_COLUMNS)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file, whether the pass finished or failed.

        Parameters:
            exc_type: The exception's class, when the pass is leaving on one.
            exc: The exception, when the pass is leaving on one.
            traceback: Its traceback, when the pass is leaving on one.
        """
        self._stack.close()

    def add(self, facts: ImageFacts) -> None:
        """Write one image's row.

        Parameters:
            facts: What the image's record says, in the shape both storages
                answer in.
        """
        image = facts.image
        row = [_csv_value(image[column]) for column in _EXPORT_IMAGE_COLUMNS]
        row.append(len(facts.techniques))
        row.append(len(facts.feature_sources))
        row.append(sum(int(entry['n_features']) for entry in facts.feature_sources))
        row.append(sum(int(entry['n_gated']) for entry in facts.feature_sources))
        self._writer.writerow(row)


def add_csv_export_section(ctx: ReportContext) -> None:
    """Append the line naming the export the pass wrote.

    Parameters:
        ctx: Report context.
    """
    ctx.lines += [
        '## CSV export',
        '',
        f'One row per image: images.csv ({ctx.stats.csv_rows} row(s)).',
        '',
    ]
