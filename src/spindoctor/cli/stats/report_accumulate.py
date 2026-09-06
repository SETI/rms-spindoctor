"""One pass over the navigation records, into everything the report says.

The report reads the record seam once, and every section's numbers fall out of
accumulators filled as the records go by.  That is what makes the storage
irrelevant: a stream of records comes out of a results tree as readily as out of
an ingested results index, and one report is written over either.

Nothing here can be answered by looking back over what the pass has seen.  A
count table divides by the per-instrument totals, and those are known only when
the pass ends, so nothing here formats a line: the accumulators are the whole
output of this module and :mod:`spindoctor.cli.stats.report_sections` turns them
into text afterwards.

Two of the report's filters are applied here rather than by the seam.  The date
bounds compare the calendar date a record's epoch renders to, and the image
bounds compare the number in its name; a selection carries neither, and widening
one to carry them would be widening it for one consumer.  Everything else the
report narrows by -- the roots and the mission -- is a selection field, and the
storage applies it where it is cheapest.

The stream also carries files no record could be read out of, and this counts
them.  That count is of the whole of every selected root: a refused file records
no mission, no date and no image number, so none of the filters above can be
applied to it, and the report says so where it prints the number.
"""

import math
import re
from array import array
from dataclasses import dataclass
from itertools import combinations
from typing import Any, TypeVar

from spindoctor.cli.stats.report_common import (
    BotsimFrame,
    MemoryImage,
    ReportStatistics,
    SuspectOffset,
    TimedImage,
)
from spindoctor.cli.stats.report_sections import (
    CsvExport,
    content_category,
    resolve_offset_limit,
    source_kind,
)
from spindoctor.nav_records import ImageFacts, RecordSource, Selection, UnreadableFile

__all__ = [
    'RangeFilters',
    'accumulate_statistics',
]

_Key = TypeVar('_Key')
"""Whatever a count table groups by: a name, or a pair of them."""

_BOTSIM_NAME_RE = re.compile(r'^([NW])(\d{10})')
"""What a Cassini image name has to look like to name a spacecraft-clock count."""


@dataclass(frozen=True)
class RangeFilters:
    """The two report filters no selection carries, applied to each record here.

    Both compare a value a record derives rather than one it records: the
    calendar date its epoch falls on, and the number inside its image name.  A
    record that derives neither is outside every bound, since an image that
    cannot be placed in time or in a numbering cannot be shown to be inside one.

    Parameters:
        start_date: Inclusive UTC start date (``YYYY-MM-DD``), or None.
        end_date: Inclusive UTC end date, or None.
        min_image_num: Inclusive lower bound on the number in the image name,
            or None.
        max_image_num: Inclusive upper bound on it, or None.
    """

    start_date: str | None = None
    end_date: str | None = None
    min_image_num: int | None = None
    max_image_num: int | None = None

    def keeps(self, image: dict[str, Any]) -> bool:
        """Whether one image's facts are inside every bound this places.

        Parameters:
            image: The image's own values, keyed by column name.

        Returns:
            True when the image is selected.  A bound that is placed and a
            value that is absent is not selected: an image whose epoch or whose
            name yielded no number cannot be shown to be inside the range.
        """
        date = image['image_date']
        if self.start_date is not None and (date is None or date < self.start_date):
            return False
        if self.end_date is not None and (date is None or date > self.end_date):
            return False
        number = image['image_number']
        if self.min_image_num is not None and (number is None or number < self.min_image_num):
            return False
        return not (
            self.max_image_num is not None and (number is None or number > self.max_image_num)
        )


def accumulate_statistics(
    source: RecordSource,
    selection: Selection,
    stats: ReportStatistics,
    *,
    filters: RangeFilters,
    csv_export: CsvExport | None = None,
) -> None:
    """Read every record the selection covers, into the report's accumulators.

    Parameters:
        source: The record source, over a results tree or over an index.
        selection: What to read: the roots and the mission the report was
            narrowed to.
        stats: The accumulators, filled in place.  Their ``top_n``,
            ``retain_names`` and ``suspect_fraction`` are read here and decide
            what is retained.
        filters: The date and image-number bounds, applied per record.
        csv_export: The open export to write each image into, or None when no
            CSV was asked for.

    Raises:
        ValueError: If the source cannot honour the selection, or cannot be
            read; raised as the stream is consumed, which is where a storage
            discovers it.
        UnlistableRootError: If a selected root could not be listed, or
            :class:`~spindoctor.nav_records.UnlistableDirectoryError` if a
            directory under one could not be.  A report covering less than the
            tree it names is worse than no report.
    """
    for facts in source.facts(selection):
        if isinstance(facts, UnreadableFile):
            stats.unreadable_files += 1
            continue
        if not filters.keeps(facts.image):
            continue
        _add_image(stats, facts)
        if csv_export is not None:
            csv_export.add(facts)
            stats.csv_rows += 1


def _add_image(stats: ReportStatistics, facts: ImageFacts) -> None:
    """Fold one image's facts into every accumulator that counts it.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
    """
    instrument = str(facts.image['instrument'])
    stats.images_by_instrument[instrument] = stats.images_by_instrument.get(instrument, 0) + 1
    _add_extremes(stats, facts, instrument)
    _add_status(stats, facts, instrument)
    _add_bodies(stats, facts, instrument)
    _add_techniques(stats, facts, instrument)
    _add_sources(stats, facts, instrument)
    _add_offsets(stats, facts, instrument)
    _add_agreement(stats, facts, instrument)
    _add_exclusions(stats, facts, instrument)
    _add_suspect(stats, facts, instrument)
    _add_botsim(stats, facts, instrument)
    _add_runtime(stats, facts, instrument)
    _add_peak_memory(stats, facts, instrument)


def _bump(counts: dict[_Key, dict[str, int]], key: _Key, instrument: str) -> None:
    """Count one image under a key, in the two-level shape a count table reads.

    Every count table the report prints has one column per instrument and a
    total column, so every counter behind one is keyed by the section's own key
    and then by instrument.

    Parameters:
        counts: The counter.
        key: The section's key, whatever shape that section groups by.
        instrument: The image's instrument.
    """
    bucket = counts.setdefault(key, {})
    bucket[instrument] = bucket.get(instrument, 0) + 1


def _add_extremes(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold the numbered ends and the timed ends of one instrument's images.

    Image numbers are only comparable within one instrument, and the two ends
    are found independently of the epochs beside them, so an image with no
    recorded epoch at one end of the number range does not hide the instrument's
    time span.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    number = image['image_number']
    if number is not None:
        name = str(image['image_name'])
        root_url = str(image['root_url'])
        stub = str(image['results_path_stub'])
        first = (int(number), name, root_url, stub)
        held_first = stats.first_image.get(instrument)
        if held_first is None or first < held_first:
            stats.first_image[instrument] = first
        # The other end is the highest number and, among any that tie, the
        # lowest name: a maximum on the number and a minimum on the tie-break,
        # rather than a maximum on the tuple.
        last = (-int(number), name, root_url, stub)
        held_last = stats.last_image.get(instrument)
        if held_last is None or last < held_last:
            stats.last_image[instrument] = last
    image_et = image['image_et']
    if image_et is None:
        return
    epoch = float(image_et)
    held_earliest = stats.first_et.get(instrument)
    if held_earliest is None or epoch < held_earliest:
        stats.first_et[instrument] = epoch
    held_latest = stats.last_et.get(instrument)
    if held_latest is None or epoch > held_latest:
        stats.last_et[instrument] = epoch


def _failure_reason(image: dict[str, Any]) -> str:
    """Which of the two reason vocabularies describes a non-success outcome.

    The navigator's own explanation when it ran, and the fatal error when it
    never got that far.  They are separate values holding separate vocabularies,
    and a report of failure reasons wants whichever one the document carried.

    Parameters:
        image: The image's own values.

    Returns:
        The reason, or ``'(none)'`` when the record carries neither.
    """
    reason = image['status_reason'] if image['status_reason'] is not None else image['status_error']
    return str(reason or '(none)')


def _add_status(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Count one image's outcome, and classify it when the outcome is not success.

    Every non-success status is a failure for reporting purposes, so an image
    that ended in a fatal error is classified beside one the navigator failed.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    status = str(image['status'])
    _bump(stats.status_counts, status, instrument)
    if status == 'success':
        return
    stats.failed_images += 1
    reason = _failure_reason(image)
    _bump(stats.reason_counts, (status, reason), instrument)
    category = content_category(
        [(str(entry['source_model']), str(entry['source_name'])) for entry in facts.feature_sources]
    )
    _bump(stats.content_counts, category, instrument)
    _bump(stats.content_reason_counts, (category, reason), instrument)
    if not stats.retain_names:
        return
    image_name = str(image['image_name'])
    stats.failure_names.setdefault(reason, []).append((instrument, image_name))
    stats.content_names.setdefault(category, []).append((instrument, image_name))


def _add_bodies(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Count one image against each named body its feature inventory holds.

    An image contributes several feature rows for one body -- one per feature
    type -- and this table counts images, so the names are collapsed to a set
    first.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    bodies = {
        str(entry['source_name'])
        for entry in facts.feature_sources
        if source_kind(str(entry['source_model'])) == 'body'
        and str(entry['source_name']) != '(none)'
    }
    if len(bodies) == 0:
        return
    failed = str(facts.image['status']) != 'success'
    counts = stats.body_failed if failed else stats.body_success
    for body in sorted(bodies):
        key = (body, instrument)
        counts[key] = counts.get(key, 0) + 1
        if failed and stats.retain_names:
            stats.body_failed_names.setdefault(key, []).append(str(facts.image['image_name']))


def _add_techniques(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Count the techniques that reported on one image, and what they reported.

    A technique reports once per image, so counting entries counts images.  The
    confidences are kept, one value per entry that recorded one, and the entries
    that recorded none are no part of that population: a technique that recorded
    none has no mean rather than a zero.  Keeping the values rather than a
    running total is what makes the mean the same number whatever order the
    records arrive in.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    for entry in facts.techniques:
        key = (str(entry['technique_name']), instrument)
        stats.technique_images[key] = stats.technique_images.get(key, 0) + 1
        if not entry['spurious']:
            stats.technique_good[key] = stats.technique_good.get(key, 0) + 1
        confidence = entry['confidence']
        if confidence is None:
            continue
        stats.technique_confidence.setdefault(key, array('d')).append(float(confidence))


def _add_sources(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Count one image against each model and source its features came from.

    The inventory holds one entry per feature type and source, and this table
    counts images, so the entries are collapsed to one per source first and the
    two feature counts are summed across the types before the image is counted.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    per_source: dict[tuple[str, str], list[int]] = {}
    for entry in facts.feature_sources:
        key = (str(entry['source_model']), str(entry['source_name']))
        tally = per_source.setdefault(key, [0, 0])
        tally[0] += int(entry['n_features'])
        tally[1] += int(entry['n_gated'])
    for (model, name), tally in per_source.items():
        key3 = (model, name, instrument)
        stats.source_images[key3] = stats.source_images.get(key3, 0) + 1
        totals = stats.source_features.setdefault(key3, [0, 0])
        totals[0] += tally[0]
        totals[1] += tally[1]


def _fused_offset(image: dict[str, Any]) -> tuple[float, float] | None:
    """The fused offset of a successful image, where there is one to screen.

    Parameters:
        image: The image's own values.

    Returns:
        The pair, or None when the image did not succeed or recorded no offset.
    """
    if str(image['status']) != 'success':
        return None
    offset_dv = image['offset_dv']
    offset_du = image['offset_du']
    if offset_dv is None or offset_du is None:
        return None
    return float(offset_dv), float(offset_du)


def _add_offsets(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold one successful image's fused offset under its camera and image size.

    Pointing error is a property of the camera, so the distributions are never
    pooled across cameras; the per-camera section pools these over the image
    sizes rather than keeping a second copy of every value.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    offset = _fused_offset(image)
    if offset is None:
        return
    camera = str(image['camera'] or '(unknown)')
    shape_v = image['image_shape_v']
    shape_u = image['image_shape_u']
    size = f'{shape_v}x{shape_u}' if shape_v is not None and shape_u is not None else '(none)'
    values = stats.offsets.setdefault((instrument, camera, size), (array('d'), array('d')))
    values[0].append(offset[0])
    values[1].append(offset[1])


def _add_agreement(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold how far one image's techniques disagreed with one another.

    Only the techniques that produced a non-spurious offset take part, and only
    an image where two of them did contributes anything.  An image with no
    confidence tier contributes to the per-pair distances and not to the
    per-tier ones, since there is no tier to attribute its disagreement to.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    rank = facts.image['confidence_rank']
    if rank is not None:
        _bump(stats.tier_counts, str(rank), instrument)
    offsets = sorted(
        (str(entry['technique_name']), float(entry['offset_dv']), float(entry['offset_du']))
        for entry in facts.techniques
        if not entry['spurious']
        and entry['offset_dv'] is not None
        and entry['offset_du'] is not None
    )
    if len(offsets) < 2:
        return
    largest = 0.0
    for first, second in combinations(offsets, 2):
        delta = math.hypot(first[1] - second[1], first[2] - second[2])
        pair = (instrument, min(first[0], second[0]), max(first[0], second[0]))
        stats.pair_deltas.setdefault(pair, array('d')).append(delta)
        largest = max(largest, delta)
    if rank is not None:
        stats.rank_disagreement.setdefault((instrument, str(rank)), array('d')).append(largest)


def _excluded_techniques(excluded: Any) -> list[str]:
    """The technique names one image's exclusion value holds.

    Parameters:
        excluded: The recorded exclusion set, as the facts carry it.

    Returns:
        The names, or an empty list when nothing was excluded or the value is
        no list of names.
    """
    if not isinstance(excluded, list):
        return []
    return [str(name) for name in excluded]


def _add_exclusions(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Count one image under the set of techniques the ensemble excluded.

    The names are sorted before they are joined into the label the section
    groups on.  A record holds them in whatever order it was written in, so two
    images that excluded the same techniques in two orders would otherwise be
    counted as two different exclusion sets.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    names = _excluded_techniques(facts.image['excluded_from_consensus'])
    if len(names) == 0:
        return
    label = ', '.join(sorted(names))
    _bump(stats.exclusion_counts, label, instrument)
    if stats.retain_names:
        stats.exclusion_names.setdefault(label, []).append(
            (instrument, str(facts.image['image_name']))
        )


def _add_suspect(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Screen one successful image's offset against the configured search limit.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    offset = _fused_offset(image)
    if offset is None:
        return
    image_name = str(image['image_name'])
    limit = resolve_offset_limit(instrument, image_name, image['image_shape_v'])
    if isinstance(limit, str):
        reason = f'{instrument}: {limit}'
        stats.unresolved[reason] = stats.unresolved.get(reason, 0) + 1
        return
    stats.screened[instrument] = stats.screened.get(instrument, 0) + 1
    limit_v, limit_u = limit
    ratio = max(abs(offset[0]) / limit_v, abs(offset[1]) / limit_u)
    if ratio < stats.suspect_fraction:
        return
    stats.suspect_counts[instrument] = stats.suspect_counts.get(instrument, 0) + 1
    stats.suspects.append(
        SuspectOffset(
            ratio=ratio,
            image_name=image_name,
            instrument=instrument,
            offset_dv=offset[0],
            offset_du=offset[1],
            magnitude=math.hypot(offset[0], offset[1]),
            limit_text=f'({limit_v:.1f}, {limit_u:.1f})',
            root_url=str(image['root_url']),
            results_path_stub=str(image['results_path_stub']),
        )
    )


def _add_botsim(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold one Cassini image under the spacecraft-clock count its name carries.

    Two images of one camera sharing a clock count is a tree nobody expects, and
    the one with the smallest identity is the one held, so which of them stands
    for the camera does not depend on the order the records arrived in.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    if instrument != 'coiss':
        return
    image = facts.image
    image_name = str(image['image_name'])
    match = _BOTSIM_NAME_RE.match(image_name.rsplit('/', 1)[-1].upper())
    if match is None:
        return
    camera, clock = match.group(1), match.group(2)
    frame = BotsimFrame(
        image_name=image_name,
        root_url=str(image['root_url']),
        results_path_stub=str(image['results_path_stub']),
        status=str(image['status']),
        offset_dv=image['offset_dv'],
        offset_du=image['offset_du'],
    )
    frames = stats.botsim.setdefault(clock, {})
    held = frames.get(camera)
    if held is None or frame.identity < held.identity:
        frames[camera] = frame


# 1024**3, so the unit is the gibibyte the value is actually divided into,
# and every label over these numbers says GiB.
_BYTES_PER_GIB = float(1024**3)


def _add_peak_memory(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold one image's peak memory, and offer it to the hungriest-image list.

    An image whose run recorded no peak contributes to neither, so the counts
    the section prints are of images that reported one rather than of images
    that ran.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    peak = image.get('peak_memory_bytes')
    if peak is None:
        return
    gb = float(peak) / _BYTES_PER_GIB
    stats.peak_memory_by_instrument.setdefault(instrument, array('d')).append(gb)
    stats.hungriest.add(
        MemoryImage(
            peak_memory_gib=gb,
            image_name=str(image['image_name']),
            instrument=instrument,
            root_url=str(image['root_url']),
            results_path_stub=str(image['results_path_stub']),
        )
    )


def _add_runtime(stats: ReportStatistics, facts: ImageFacts, instrument: str) -> None:
    """Hold one image's run time, and offer it to the slowest-image list.

    Parameters:
        stats: The accumulators.
        facts: What the image's record says.
        instrument: The image's instrument.
    """
    image = facts.image
    elapsed = image['elapsed_s']
    if elapsed is None:
        return
    stats.elapsed_by_instrument.setdefault(instrument, array('d')).append(float(elapsed))
    stats.slowest.add(
        TimedImage(
            elapsed_s=float(elapsed),
            image_name=str(image['image_name']),
            instrument=instrument,
            root_url=str(image['root_url']),
            results_path_stub=str(image['results_path_stub']),
        )
    )
