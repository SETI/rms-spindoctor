"""Generate a deterministic statistics report (Markdown + charts) from the records.

One pass over the record seam answers every section.  The pass reads a results
tree or an ingested results index -- whichever the run was pointed at -- and
fills the accumulators in :mod:`spindoctor.cli.stats.report_accumulate`; the
sections here and in :mod:`spindoctor.cli.stats.report_sections` then turn those
into text.  A results index makes the pass cheaper and is required by none of
it.

The output is deterministic: the same records and the same options always
produce byte-identical Markdown, whichever storage answered, so a difference
between two runs is a difference in the data or a defect and never noise.  That
holds without the stream being ordered, because every section either counts,
reduces to a minimum, or sorts what it prints on a key that includes the pair
that identifies an image -- an image name alone is not unique across roots.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from array import array
from collections.abc import Sequence
from pathlib import Path

from filecache import FCPath

from spindoctor.cli.stats.report_accumulate import RangeFilters, accumulate_statistics
from spindoctor.cli.stats.report_common import (
    ReportContext,
    ReportStatistics,
    add_drilldown,
    add_instrument_count_table,
    count_pct,
    fmt,
    image_name_from_filename,
    offset_stats,
    percentile,
    safe_filename,
    write_offset_hist,
    write_stacked_bar_chart,
)
from spindoctor.cli.stats.report_sections import (
    CsvExport,
    add_botsim_section,
    add_csv_export_section,
    add_failure_taxonomy_section,
    add_narrowing_section,
    add_offset_by_group_section,
    add_runtime_section,
    add_suspect_offset_section,
)
from spindoctor.config import (
    DEFAULT_CONFIG,
    get_nav_results_root,
    get_results_index_db_url,
    get_results_tree_tuning,
    load_default_and_user_config,
)
from spindoctor.nav_records import (
    RecordSource,
    Selection,
    TreeRecordSource,
    TreeTuning,
    UnlistableDirectoryError,
    datetime_from_image_et,
    distinct_roots,
    image_number_from_name,
)
from spindoctor.results_index import (
    IndexRecordSource,
    RootNotIngestedError,
    ingested_roots,
    normalize_root_url,
    open_index_for_roots,
    unfinished_roots,
)

__all__ = ['build_report', 'main_report']

# Confidence tiers, in descending-confidence order, always reported.
_CONFIDENCE_TIERS: tuple[str, ...] = ('high', 'medium', 'low', 'failed', 'conflicted')

_NO_RECORD_PROSE = """\
A file named like a navigation document that no record could be read out of is counted here and
nowhere else: it records no instrument, no date and no image number, so this count covers the whole
of every selected root and none of the filters above narrows it. One kind of such file is counted
by a report over the documents and by no report from an index: one the storage could not deliver at
all. An ingest records no refusal for that, because a retrieval that failed once is worth trying
again rather than being remembered as a file that will not read."""
"""The paragraph that explains what the count of unreadable files covers.

Written with the wrapping it is printed with, and holding no number, so that the
prose of the section is a constant.  The count is a line of its own after it,
which keeps the width of every line here independent of how many digits the
count runs to.
"""


def _extreme_image_name(ctx: ReportContext, instrument: str, *, last: bool) -> str:
    """The lowest- or highest-numbered selected image name of one instrument.

    Parameters:
        ctx: Report context.
        instrument: Instrument name to restrict to.
        last: True for the highest-numbered image, False for the lowest.

    Returns:
        The image name, or ``'-'`` when the instrument has no selected
        image whose name contains a number.
    """
    held = ctx.stats.last_image if last else ctx.stats.first_image
    found = held.get(instrument)
    if found is None:
        return '-'
    return image_name_from_filename(instrument, found[1])


def _extreme_times(ctx: ReportContext, instrument: str) -> tuple[str, str]:
    """The earliest and latest epoch among one instrument's selected images.

    Computed over the images that have an epoch, independently of the
    image-name ordering: a single image with no recorded epoch at one end
    of the number range would otherwise hide the whole instrument's time
    span.

    Parameters:
        ctx: Report context.
        instrument: Instrument name to restrict to.

    Returns:
        ``(first, last)`` UTC timestamps to the second, each ``'-'`` when
        no selected image of that instrument has an epoch.
    """
    return (
        datetime_from_image_et(ctx.stats.first_et.get(instrument)) or '-',
        datetime_from_image_et(ctx.stats.last_et.get(instrument)) or '-',
    )


def _add_selection_section(ctx: ReportContext) -> None:
    """Append the per-instrument summary of what the filters selected.

    Image numbers are only comparable within one instrument, so the first
    and last image are reported per instrument and never pooled.  The image
    and time bounds are found independently, so the first image is not
    necessarily the one at the first available time.

    Parameters:
        ctx: Report context.
    """
    ctx.lines += ['## Images selected', '']
    if ctx.total_images == 0:
        ctx.lines += ['No images match the filters.', '']
        return
    ctx.lines += [
        '| instrument | images | first image | last image | first avail. date | last avail. date |',
        '|---|---|---|---|---|---|',
    ]
    for instrument in ctx.instruments:
        first_image = _extreme_image_name(ctx, instrument, last=False)
        last_image = _extreme_image_name(ctx, instrument, last=True)
        first_time, last_time = _extreme_times(ctx, instrument)
        images = count_pct(ctx.images_by_instrument[instrument], ctx.total_images)
        ctx.lines.append(
            f'| {instrument} | {images} | {first_image} | {last_image} '
            f'| {first_time} | {last_time} |'
        )
    ctx.lines += ['', f'Total images: {ctx.total_images}', '']


def _add_unreadable_files_section(ctx: ReportContext) -> None:
    """Append how many files under the selected roots yielded no record.

    Printed whether the count is zero or not.  A line that disappeared at zero
    could not be told apart by a reader from a report that never looked, and a
    summary that quietly covered less than the tree is worse than one that says
    how much less.

    Parameters:
        ctx: Report context.
    """
    ctx.lines += [
        '## Files that yielded no record',
        '',
        *_NO_RECORD_PROSE.splitlines(),
        '',
        f'Files that yielded no record: {ctx.stats.unreadable_files}',
        '',
    ]


def _add_status_sections(ctx: ReportContext) -> None:
    """Append the success/failure counts and the failure-reason breakdown.

    Parameters:
        ctx: Report context.
    """
    by_status = ctx.stats.status_counts
    statuses = sorted(by_status, key=lambda name: (name != 'success', name))
    ctx.lines += ['## Success / failure', '']
    add_instrument_count_table(
        ctx, [([status], by_status[status]) for status in statuses], headers=['status']
    )
    write_stacked_bar_chart(
        ctx.output_dir / 'status_counts.png',
        statuses,
        {
            instrument: [by_status[status].get(instrument, 0) for status in statuses]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Navigation status',
        xlabel='images',
    )
    ctx.lines += ['![status](status_counts.png)', '']

    # Every non-success status is a failure for reporting purposes, so
    # `error` images (SPICE and otherwise) appear here alongside `failed`.
    by_reason = ctx.stats.reason_counts
    if len(by_reason) == 0:
        return
    ordered = sorted(by_reason, key=lambda key: (-sum(by_reason[key].values()), key))
    ctx.lines += ['### Failure reasons', '']
    add_instrument_count_table(
        ctx,
        [([status, reason], by_reason[(status, reason)]) for status, reason in ordered],
        headers=['status', 'reason'],
    )
    names_by_reason = ctx.stats.failure_names
    add_drilldown(
        ctx,
        [
            (reason, names_by_reason.get(reason, []))
            for reason in dict.fromkeys(r for _s, r in ordered)
        ],
        label='reason',
        stub_prefix='failure_reason',
    )
    labels = [f'{status}/{reason}' for status, reason in ordered]
    write_stacked_bar_chart(
        ctx.output_dir / 'failure_reasons.png',
        labels,
        {
            instrument: [by_reason[key].get(instrument, 0) for key in ordered]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Failure reasons',
        xlabel='images',
    )
    ctx.lines += ['![failure reasons](failure_reasons.png)', '']


def _add_technique_usage_section(ctx: ReportContext) -> None:
    """Append per-technique image counts, spurious shares, and mean confidence.

    The mean is taken over the retained confidences with an exact sum, so it is
    the same number whatever order the records reached the pass in.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    if len(stats.technique_images) == 0:
        return
    empty: array[float] = array('d')
    images: dict[str, dict[str, int]] = {}
    for (name, instrument), count in stats.technique_images.items():
        images.setdefault(name, {})[instrument] = count
    techniques = sorted(images, key=lambda name: (-sum(images[name].values()), name))
    ctx.lines += ['## Technique usage', '', 'Images on which each technique ran.', '']
    add_instrument_count_table(
        ctx, [([name], images[name]) for name in techniques], headers=['technique']
    )
    ctx.lines += [
        '### Per-technique detail',
        '',
        '| technique | instrument | images | non-spurious | mean confidence |',
        '|---|---|---|---|---|',
    ]
    for name in techniques:
        for instrument in ctx.instruments:
            n_images = stats.technique_images.get((name, instrument))
            if n_images is None:
                continue
            n_good = stats.technique_good.get((name, instrument), 0)
            # Only the entries that recorded a confidence are in the population,
            # and a group where none did prints a dash rather than a zero.
            reported = stats.technique_confidence.get((name, instrument), empty)
            mean_conf = statistics.fmean(reported) if len(reported) > 0 else None
            images_cell = count_pct(n_images, ctx.images_by_instrument[instrument])
            ctx.lines.append(
                f'| {name} | {instrument} | {images_cell} '
                f'| {count_pct(n_good, n_images)} | {fmt(mean_conf)} |'
            )
    ctx.lines.append('')
    write_stacked_bar_chart(
        ctx.output_dir / 'technique_usage.png',
        techniques,
        {
            instrument: [images[name].get(instrument, 0) for name in techniques]
            for instrument in ctx.instruments
        },
        ctx.instruments,
        title='Images per technique',
        xlabel='images',
    )
    ctx.lines += ['![technique usage](technique_usage.png)', '']


def _add_source_usage_section(ctx: ReportContext) -> None:
    """Append the per-model / per-source feature-usage tables.

    Parameters:
        ctx: Report context.
    """
    stats = ctx.stats
    if len(stats.source_images) == 0:
        return
    images: dict[tuple[str, str], dict[str, int]] = {}
    for (model, name, instrument), count in stats.source_images.items():
        images.setdefault((model, name), {})[instrument] = count
    sources = sorted(images, key=lambda key: (key[0], -sum(images[key].values()), key[1]))
    ctx.lines += ['## Model and source usage', '', 'Images in which each source appears.', '']
    add_instrument_count_table(
        ctx,
        [([model, name], images[(model, name)]) for model, name in sources],
        headers=['model', 'source'],
    )
    ctx.lines += [
        '### Per-source feature counts',
        '',
        '| model | source | instrument | features | gated |',
        '|---|---|---|---|---|',
    ]
    for model, name in sources:
        for instrument in ctx.instruments:
            entry = stats.source_features.get((model, name, instrument))
            if entry is None:
                continue
            ctx.lines.append(f'| {model} | {name} | {instrument} | {entry[0]} | {entry[1]} |')
    ctx.lines.append('')


def _pooled_offsets(ctx: ReportContext) -> dict[tuple[str, str], tuple[array[float], array[float]]]:
    """The fused offsets of the successful images, by instrument and camera.

    The pass keys them by image size as well, because the finer breakdown needs
    that; this pools over the sizes rather than the pass keeping a second copy
    of every value.

    Parameters:
        ctx: Report context.

    Returns:
        Per ``(instrument, camera)``, the V-axis and U-axis offsets.
    """
    pooled: dict[tuple[str, str], tuple[array[float], array[float]]] = {}
    for key in sorted(ctx.stats.offsets):
        instrument, camera, _size = key
        values = ctx.stats.offsets[key]
        entry = pooled.setdefault((instrument, camera), (array('d'), array('d')))
        entry[0].extend(values[0])
        entry[1].extend(values[1])
    return pooled


def _add_offset_section(ctx: ReportContext) -> None:
    """Append the fused-offset statistics and a histogram per camera.

    Pointing error is a property of the camera, not of the spacecraft, so
    the distributions are grouped by ``(instrument, camera)`` and never
    pooled (a Cassini NAC pixel is a tenth of a WAC pixel, so pooling the
    two would describe neither).

    Parameters:
        ctx: Report context.
    """
    by_camera = _pooled_offsets(ctx)
    ctx.lines += [
        '## Offset statistics (successful images)',
        '',
        'Grouped by camera: pointing errors of different cameras are unrelated',
        'and are never pooled.  Percentages are of the instrument total.',
        '',
        '| instrument | camera | axis | images | mean | median | stdev | min | max |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for instrument, camera in sorted(by_camera):
        dv_values, du_values = by_camera[(instrument, camera)]
        for axis, values in (('dV', dv_values), ('dU', du_values)):
            stats = offset_stats(values)
            images = count_pct(len(values), ctx.images_by_instrument[instrument])
            if stats is None:
                ctx.lines.append(
                    f'| {instrument} | {camera} | {axis} | {images} | - | - | - | - | - |'
                )
            else:
                ctx.lines.append(
                    f'| {instrument} | {camera} | {axis} | {images} | {fmt(stats["mean"])} '
                    f'| {fmt(stats["median"])} | {fmt(stats["stdev"])} '
                    f'| {fmt(stats["min"])} | {fmt(stats["max"])} |'
                )
    ctx.lines.append('')
    for instrument, camera in sorted(by_camera):
        dv_values, du_values = by_camera[(instrument, camera)]
        chart = f'offsets_hist_{safe_filename(f"{instrument}_{camera}")}.png'
        write_offset_hist(
            ctx.output_dir / chart,
            dv_values,
            du_values,
            title=f'Fused offset distribution, {instrument} {camera} (successful images)',
        )
        ctx.lines += [f'![offsets {instrument} {camera}]({chart})', '']


def _add_agreement_sections(ctx: ReportContext) -> None:
    """Append the cross-technique agreement and confidence-calibration tables.

    Parameters:
        ctx: Report context.
    """
    per_pair = ctx.stats.pair_deltas
    per_image_rank = ctx.stats.rank_disagreement
    ctx.lines += [
        '## Cross-technique agreement',
        '',
        'Euclidean distance between per-technique offsets on images where both',
        'techniques produced non-spurious results.',
        '',
        '| instrument | technique pair | images | median (px) | p95 (px) |',
        '|---|---|---|---|---|',
    ]
    for instrument, technique_a, technique_b in sorted(per_pair):
        deltas = per_pair[(instrument, technique_a, technique_b)]
        ctx.lines.append(
            f'| {instrument} | {technique_a} vs {technique_b} '
            f'| {count_pct(len(deltas), ctx.images_by_instrument[instrument])} | '
            f'{fmt(statistics.median(deltas))} | {fmt(percentile(deltas, 0.95))} |'
        )
    ctx.lines.append('')

    by_tier: dict[str, dict[str, int]] = {tier: {} for tier in _CONFIDENCE_TIERS}
    for tier, counts in ctx.stats.tier_counts.items():
        by_tier.setdefault(tier, {}).update(counts)
    # The standard tiers always appear, in tier order, so an empty tier reads
    # as a real zero rather than a missing row.  Any unrecognized rank the
    # records carry is listed after them rather than dropped.
    tiers = [*_CONFIDENCE_TIERS, *sorted(set(by_tier) - set(_CONFIDENCE_TIERS))]
    ctx.lines += [
        '## Confidence calibration (agreement as accuracy proxy)',
        '',
        'For each confidence tier: how well the techniques that fed the fused',
        'offset agreed with one another.  Without ground truth, cross-technique',
        'agreement is the standing production check that confidence tiers are',
        'meaningful (the calibrated anchor is the simulated-scene campaign).',
        '',
    ]
    add_instrument_count_table(ctx, [([tier], by_tier[tier]) for tier in tiers], headers=['tier'])
    ctx.lines += [
        '| tier | instrument | images | with >=2 techniques | median max-disagreement (px) '
        '| p95 (px) |',
        '|---|---|---|---|---|---|',
    ]
    empty: array[float] = array('d')
    for tier in tiers:
        for instrument in ctx.instruments:
            count = by_tier[tier].get(instrument, 0)
            disagreements = per_image_rank.get((instrument, tier), empty)
            ctx.lines.append(
                f'| {tier} | {instrument} '
                f'| {count_pct(count, ctx.images_by_instrument[instrument])} '
                f'| {count_pct(len(disagreements), count)} | '
                f'{fmt(statistics.median(disagreements)) if disagreements else "-"} | '
                f'{fmt(percentile(disagreements, 0.95)) if disagreements else "-"} |'
            )
    ctx.lines.append('')
    if len(per_image_rank) > 0:
        write_stacked_bar_chart(
            ctx.output_dir / 'agreement_by_tier.png',
            tiers,
            {
                instrument: [len(per_image_rank.get((instrument, tier), empty)) for tier in tiers]
                for instrument in ctx.instruments
            },
            ctx.instruments,
            title='Images with cross-technique agreement data, by tier',
            xlabel='images',
        )
        ctx.lines += ['![agreement by tier](agreement_by_tier.png)', '']


def _add_exclusions_section(ctx: ReportContext) -> None:
    """Append the ensemble outlier-exclusion breakdown.

    Parameters:
        ctx: Report context.
    """
    by_exclusion = ctx.stats.exclusion_counts
    if len(by_exclusion) == 0:
        return
    ordered = sorted(by_exclusion, key=lambda key: (-sum(by_exclusion[key].values()), key))
    ctx.lines += ['## Ensemble outlier exclusions', '']
    add_instrument_count_table(
        ctx,
        [([label], by_exclusion[label]) for label in ordered],
        headers=['excluded techniques'],
    )
    add_drilldown(
        ctx,
        [(label, ctx.stats.exclusion_names.get(label, [])) for label in ordered],
        label='exclusion set',
        stub_prefix='excluded',
    )


def _image_bound(value: str | None, *, option: str) -> int | None:
    """Parse a ``--min-image`` / ``--max-image`` value into its numeric bound.

    Parameters:
        value: Image name (``N1454725799``) or bare number, or None.
        option: Option name for the error message.

    Returns:
        The integer bound, or None when ``value`` is None.

    Raises:
        ValueError: If the value contains no digits.
    """
    if value is None:
        return None
    number = image_number_from_name(value)
    if number is None:
        raise ValueError(f'{option} value {value!r} contains no digits')
    return number


def build_report(
    source: RecordSource,
    output_dir: str | Path | FCPath,
    *,
    instrument: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_image: str | None = None,
    max_image: str | None = None,
    roots: Sequence[str] = (),
    dropped_roots: Sequence[str] = (),
    top_n: int = 0,
    filelists: bool = False,
    suspect_fraction: float = 0.9,
    csv_export: bool = False,
) -> FCPath:
    """Read every record once and write ``report.md`` plus charts.

    The report is deterministic: the same records and options always produce
    byte-identical Markdown.  All filters combine and apply to every section,
    with one stated exception -- the count of files that yielded no record is of
    the whole of every selected root, because such a file records no instrument,
    date or image number for a filter to compare.

    Parameters:
        source: The record source to read, over a results tree or over an
            ingested results index.
        output_dir: Directory receiving ``report.md``, the PNG charts, and
            (with ``filelists`` / ``csv_export``) the ``filelists/``
            subdirectory and ``images.csv`` (created if missing).  A local
            directory or any URL the ``filecache`` layer accepts.
        instrument: Optional instrument filter (``coiss`` / ``vgiss`` /
            ``gossi`` / ``nhlorri``).
        start_date: Optional inclusive UTC start date (``YYYY-MM-DD``).
        end_date: Optional inclusive UTC end date (``YYYY-MM-DD``).
        min_image: Optional inclusive lower bound on the numeric portion
            of the image name; an image name (``N1454725799``) or a bare
            number.
        max_image: Optional inclusive upper bound on the numeric portion
            of the image name.
        roots: Optional normalized results-root URLs to restrict to, of the
            roots the source holds; empty reports over every one of them, which
            a report may legitimately do where a per-image lookup never does.
            Named in the report as the restriction they are.
        dropped_roots: Roots the caller was pointed at and could not bind the
            source to, which the report names as roots it covers nothing of.  A
            root with no completed ingest is one of these: nothing under it is
            reported, its files that yielded no record included, because a
            half-covered root is worse than an uncovered one.  A caller passing
            any of these passes the roots it did cover as ``roots``, so that the
            report says what it covered as well as what it did not.
        top_n: When positive, categorical sections list up to this many
            example image names per category, the suspect-offset and
            worst-BOTSIM-pair tables are capped at this many rows, and the
            slowest images are listed.
        filelists: When True, write one plain-text file per category and
            instrument (one image name per line, full list) under
            ``filelists/``, in the format ``--image-filelist`` reads.
        suspect_fraction: Fraction of the per-axis maximum expected
            pointing offset at or beyond which a fused offset is flagged
            as suspect.
        csv_export: When True, write the flattened one-row-per-image
            ``images.csv`` next to ``report.md``, one row per image as the pass
            reads it.

    Returns:
        The path of the written ``report.md``.

    Raises:
        ValueError: If ``min_image`` or ``max_image`` contains no digits, or if
            the source cannot honour the selection or cannot be read.
        UnlistableDirectoryError: If a selected root, or a directory under one,
            could not be listed.
    """
    output_path = FCPath(output_dir)
    min_image_num = _image_bound(min_image, option='min_image')
    max_image_num = _image_bound(max_image, option='max_image')
    selection = Selection(roots=tuple(roots), instrument=instrument)
    filters = RangeFilters(
        start_date=start_date,
        end_date=end_date,
        min_image_num=min_image_num,
        max_image_num=max_image_num,
    )
    stats = ReportStatistics(
        top_n=top_n,
        retain_names=top_n > 0 or filelists,
        suspect_fraction=suspect_fraction,
    )
    if csv_export:
        with CsvExport(output_path / 'images.csv') as export:
            accumulate_statistics(source, selection, stats, filters=filters, csv_export=export)
    else:
        accumulate_statistics(source, selection, stats, filters=filters)

    ctx = ReportContext(
        output_dir=output_path,
        stats=stats,
        top_n=top_n,
        filelists=filelists,
        suspect_fraction=suspect_fraction,
    )
    ctx.lines += ['# Navigation statistics report', '']
    active = [
        text
        for text in (
            f'instrument = {instrument}' if instrument is not None else None,
            f'from {start_date}' if start_date is not None else None,
            f'to {end_date}' if end_date is not None else None,
            f'image number >= {min_image_num}' if min_image_num is not None else None,
            f'image number <= {max_image_num}' if max_image_num is not None else None,
            f'root in {", ".join(roots)}' if roots else None,
        )
        if text is not None
    ]
    add_narrowing_section(ctx, filters=active, dropped_roots=dropped_roots)

    _add_selection_section(ctx)
    _add_unreadable_files_section(ctx)
    _add_status_sections(ctx)
    add_failure_taxonomy_section(ctx)
    _add_technique_usage_section(ctx)
    _add_source_usage_section(ctx)
    _add_offset_section(ctx)
    add_offset_by_group_section(ctx)
    add_suspect_offset_section(ctx)
    add_botsim_section(ctx)
    _add_agreement_sections(ctx)
    _add_exclusions_section(ctx)
    add_runtime_section(ctx)
    if csv_export:
        add_csv_export_section(ctx)

    report_path = output_path / 'report.md'
    report_path.write_text('\n'.join(ctx.lines) + '\n', encoding='utf-8')
    return report_path


def _to_stderr(message: str) -> None:
    """Print one diagnostic where this program's other diagnostics go.

    This program's output is terminal text for a person, so a refusal it caught
    rather than raised belongs on the stream its other refusals print to, and
    arrives there in the order it happened.

    Parameters:
        message: The line to print.
    """
    print(message, file=sys.stderr)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare every option this program accepts.

    Parameters:
        parser: The parser to declare them on.
    """
    parser.add_argument(
        '--config-file',
        action='append',
        default=None,
        help='The configuration file(s) to use to override default settings; may be '
        'specified multiple times. If not provided, attempts to load '
        './nav_default_config.yaml if present.',
    )
    parser.add_argument(
        '--results-index-db',
        default=None,
        metavar='URL',
        help='Connection URL of the results index written by sd_results_index '
        '(a sqlite: URL naming a local path, or a postgresql+psycopg: URL); '
        'overrides the environment.results_index_db configuration variable and '
        'NAV_RESULTS_INDEX_DB. Without one the navigation results tree is read '
        'instead, one document per image. Pass --results-index-db none to read the '
        'tree even where an index is configured',
    )
    parser.add_argument(
        '--nav-results-root',
        action='append',
        dest='nav_results_roots',
        default=None,
        metavar='ROOT',
        help='Root directory of a navigation results tree to report on (a local '
        'directory or any URL the filecache layer accepts); may be specified '
        'multiple times. Overrides NAV_RESULTS_ROOT and the nav_results_root '
        'configuration variable. Read only when no index is named, and then '
        'every document under it is read once',
    )
    parser.add_argument(
        '--root',
        action='append',
        default=None,
        metavar='ROOT',
        help='Restrict the report to one ingested navigation-results root; may '
        'be given more than once. Default: every root the index holds. Refused '
        'when no index is named, since there is then nothing to select among',
    )
    parser.add_argument(
        '--output-dir',
        default='nav_stats_report',
        help='Directory (local path or filecache URL) receiving report.md and charts '
        '(default: %(default)s)',
    )
    parser.add_argument(
        '--instrument',
        choices=['coiss', 'vgiss', 'gossi', 'nhlorri'],
        default=None,
        help='Restrict the report to one instrument',
    )
    parser.add_argument(
        '--start-date', default=None, metavar='YYYY-MM-DD', help='Inclusive UTC start date'
    )
    parser.add_argument(
        '--end-date', default=None, metavar='YYYY-MM-DD', help='Inclusive UTC end date'
    )
    parser.add_argument(
        '--min-image',
        default=None,
        metavar='NAME',
        help='Inclusive lower bound on the numeric portion of the image name '
        '(an image name like N1454725799 or a bare number)',
    )
    parser.add_argument(
        '--max-image',
        default=None,
        metavar='NAME',
        help='Inclusive upper bound on the numeric portion of the image name',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=0,
        metavar='N',
        help='List up to N example image names per category in categorical '
        'sections, cap the suspect-offset and worst-BOTSIM-pair tables at N '
        'rows, and list the N slowest images (default: 0 = off)',
    )
    parser.add_argument(
        '--filelists',
        action='store_true',
        default=False,
        help='Write one plain-text file per category and instrument (one '
        'image name per line, full list, readable by --image-filelist) into '
        'the filelists/ subdirectory of the output dir',
    )
    parser.add_argument(
        '--suspect-fraction',
        type=float,
        default=0.9,
        metavar='F',
        help='Flag successful offsets at or beyond this fraction of the '
        'per-axis maximum expected pointing offset (default: %(default)s)',
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        default=False,
        help='Write a flattened one-row-per-image images.csv next to report.md',
    )


def main_report(cmdline: list[str] | None = None) -> int:
    """Entry point for ``sd_stats_report``.

    This program keeps ``print()`` rather than a logger.  Its output *is*
    terminal text for a person reading a report, and a logger would wrap that
    in run-log machinery it has no use for.

    An index is optional here as it is everywhere else.  Named, its rows are
    read; unnamed, the navigation results tree is read, one document per image.
    Both answer the same seam, so over the records both of them can read they
    produce the same report; the count of files that yielded no record is the
    one exception, a file the storage could not deliver at all being counted
    from a tree and not from an index.

    Parameters:
        cmdline: Argument list; None uses ``sys.argv``.

    Returns:
        Process exit code: 0 on success, and 1 when a level names the index with
        an empty value, when the index that was named cannot be read, when it
        holds no completed ingest of anything, when no tree can be resolved to
        read instead, or when a tree cannot be read whole.

    Raises:
        SystemExit: With status 2, from the argument parser, for a command line
            it will not accept -- an unknown flag, an unparseable bound, a root
            that is not a location that can be read, a root the index holds no
            completed ingest of (a value the index rather than the parser
            rejects, reported the same way), or ``--root`` with no index to hold
            it against.
    """
    parser = argparse.ArgumentParser(
        description='Generate a navigation statistics report from a results tree or an index.'
    )
    _add_arguments(parser)
    arguments = parser.parse_args(cmdline)
    # The configuration is loaded the way every other program loads it, so a
    # machine's nav_default_config.yaml and a --config-file apply here as they
    # do everywhere else.  A refusal is printed where this program's other
    # refusals print, rather than raised.
    try:
        load_default_and_user_config(arguments, DEFAULT_CONFIG)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        _to_stderr(f'Invalid configuration: {exc}')
        return 1
    # Checked here, before any storage is opened, so that the one thing a
    # ValueError out of the pass can still mean is a storage that stopped
    # answering.  Left to the pass, an unparseable bound and an index that
    # cannot be read arrive as the same exception at the same place, and the
    # exit code the caller reads then says usage error for both.
    try:
        _image_bound(arguments.min_image, option='--min-image')
        _image_bound(arguments.max_image, option='--max-image')
    except ValueError as exc:
        parser.error(str(exc))

    # A level that names the index with an empty value refuses the run, and its
    # message names that level and says what to write.  Printed where this
    # program's other refusals print, and returned as a status rather than raised,
    # because a traceback would bury the one line that says what to change.
    try:
        url = get_results_index_db_url(arguments, DEFAULT_CONFIG)
    except ValueError as exc:
        _to_stderr(str(exc))
        return 1
    if url is None:
        # Resolved here, at the top, and passed down the way every program
        # passes it; the configuration was validated when it was loaded.
        return _report_over_a_tree(arguments, parser, get_results_tree_tuning(DEFAULT_CONFIG))
    return _report_from_an_index(url, arguments, parser)


def _report_written_from(
    source: RecordSource,
    arguments: argparse.Namespace,
    roots: Sequence[str],
    dropped_roots: Sequence[str] = (),
) -> FCPath:
    """Write the report from one open source, whichever storage it reads.

    Parameters:
        source: The open record source.
        arguments: The parsed command line.
        roots: Normalized roots to restrict the report to, and empty for every
            root the source holds.
        dropped_roots: Roots this run was pointed at that the source could not
            be bound to, which the report names as covering none of.

    Returns:
        Where the report was written.
    """
    return build_report(
        source,
        arguments.output_dir,
        instrument=arguments.instrument,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        min_image=arguments.min_image,
        max_image=arguments.max_image,
        roots=roots,
        dropped_roots=dropped_roots,
        top_n=arguments.top_n,
        filelists=arguments.filelists,
        suspect_fraction=arguments.suspect_fraction,
        csv_export=arguments.csv,
    )


def _report_from_an_index(
    url: str, arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """Report from an index somebody built, which is the cheap way to run twice.

    The roots read are the roots the index holds a completed ingest of, and no
    others.  Under a half-ingested root the absence of a row says nothing, so a
    report that counted one would be reading absence it has no license to read.

    Parameters:
        url: Connection URL of the results index.
        arguments: The parsed command line.
        parser: The parser, for the refusals it reports as usage errors.

    Returns:
        Process exit code: 0 on success, and 1 when the index cannot be opened,
        when it holds no completed ingest of any root, or when it stops
        answering while the pass streams from it.

    Raises:
        SystemExit: With status 2, from the parser, for a ``--root`` that is not
            a location or that the index holds no completed ingest of.
    """
    try:
        roots = [normalize_root_url(root) for root in arguments.root or []]
    except ValueError as exc:
        parser.error(f'a --root is not a location that can be read: {exc}')
    try:
        engine = open_index_for_roots(url, roots)
    except RootNotIngestedError as exc:
        # A value the operator typed, so it is reported as a usage error like
        # every other bad value on this command line, rather than as a run that
        # failed on something it found.
        parser.error(str(exc))
    except ValueError as exc:
        print(f'Cannot read the results index: {exc}', file=sys.stderr)
        return 1
    try:
        with engine.connect() as connection:
            # A run that named its roots was refused above unless every one of
            # them has a completed ingest, so it drops none; one that named
            # none is bound to the roots that have one, and names the rest as
            # roots it covers nothing of.
            held = list(roots) or ingested_roots(connection)
            dropped = [] if roots else unfinished_roots(connection)
    except Exception:
        engine.dispose()
        raise
    if len(held) == 0:
        engine.dispose()
        print(
            f'The results index {url} holds no completed ingest of any root, so there is '
            f'nothing it can be asked about. Run sd_results_index over a root first, or '
            f'report over the tree with --results-index-db none.',
            file=sys.stderr,
        )
        return 1
    # A run that dropped a root covers fewer roots than the index holds rows
    # for, so the roots it did cover are the restriction the report prints.
    covered = held if dropped else list(roots)
    try:
        with IndexRecordSource(engine, held, url, ()) as source:
            report_path = _report_written_from(source, arguments, covered, dropped)
    except ValueError as exc:
        # The stream issues its queries while the pass reads them, so an index
        # that stops answering fails here rather than where it was opened.  That
        # is a run that failed on what it found, not a command line to reject.
        print(f'Cannot read the results index: {exc}', file=sys.stderr)
        return 1
    print(f'Wrote {report_path}')
    return 0


def _report_over_a_tree(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, tuning: TreeTuning
) -> int:
    """Report over a results tree, reading one document per image.

    What it costs is one full read of every document under the roots, which is
    exactly the cost an index exists to remove.  That is the right trade for a
    local tree and one report, and the wrong one for a cloud root or a repeated
    report; the statistics guide says so beside the option.

    Parameters:
        arguments: The parsed command line.
        parser: The parser, for the refusals it reports as usage errors.
        tuning: How much of the pass over the tree runs at once, resolved by
            the entry point from the configuration.

    Returns:
        Process exit code: 0 on success, and 1 when no root can be resolved, or
        when a root cannot be read whole -- one that cannot be listed, and one
        that stops answering while the pass reads it.

    Raises:
        SystemExit: With status 2, from the parser, for a command line that names
            ``--root`` with no index to hold it against, or a root that is not a
            location that can be read.
    """
    if arguments.root:
        # Refused rather than read as a second spelling of --nav-results-root.
        # --root selects among the roots one index holds, and there is no index
        # here to hold anything.
        parser.error(
            '--root restricts a report to a root the index holds, and no index was named. '
            'Name the trees to report on with --nav-results-root, or name an index with '
            '--results-index-db.'
        )
    try:
        named = arguments.nav_results_roots or [get_nav_results_root(arguments, DEFAULT_CONFIG)]
    except ValueError as exc:
        print(
            f'No results index and no navigation results root: {exc}. Name a tree with '
            f'--nav-results-root, the nav_results_root configuration variable or '
            f'NAV_RESULTS_ROOT, or name an index with --results-index-db.',
            file=sys.stderr,
        )
        return 1
    try:
        roots = distinct_roots(named)
    except ValueError as exc:
        parser.error(f'a navigation results root is not a location that can be read: {exc}')
    print(f'Reading {", ".join(roots)}')
    try:
        with TreeRecordSource(roots, tuning=tuning) as source:
            report_path = _report_written_from(source, arguments, ())
    except (UnlistableDirectoryError, ValueError) as exc:
        # A root the walk could not read whole is a report that would cover less
        # than the tree and say nothing about it, and a tree that stops answering
        # part way through is the same report.  Both are runs that failed on what
        # they found rather than command lines to reject.
        print(f'Cannot read the navigation results tree: {exc}', file=sys.stderr)
        return 1
    print(f'Wrote {report_path}')
    return 0
