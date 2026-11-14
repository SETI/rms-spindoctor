"""The report of what became of every image a generator run considered.

A corrected C-kernel says where a camera was pointing; it does not say how well
that was measured, and it says nothing at all about the images that got no
segment.  Both are what this report is for.  One row per image, one file per
mission: the image's own measurement as the navigation recorded it, and then
either the corrected file carrying its segment or the reason it has none.  An
image appears exactly once, and never with neither.

Nothing here is a judgment.  The confidence, rank, status and status reason are
reported as recorded, unrounded where the pipeline left them unrounded and
rounded where it rounded them, because a consumer filtering low-confidence or
conflicted pointing out of a kernel has no other place to read them from -- the
generator itself applies no threshold.

The column set is expected to grow as consumers ask for more; it is written
with a header row so that a reader keys on names rather than positions.
"""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cspyce
from filecache import FCPath

from spindoctor.cli.ck.images import OmissionReason
from spindoctor.cli.ck.pointing import (
    read_optional_number,
    read_optional_pair,
    read_optional_text,
    read_section,
    read_text,
)

# The columns, in the order they are written.  This is version 1 of the set:
# a consumer should read by header name, since the set is expected to grow.
REPORT_COLUMNS: tuple[str, ...] = (
    'image_name',
    'utc',
    'et',
    'sclk',
    'offset_dv',
    'offset_du',
    'sigma_dv',
    'sigma_du',
    'confidence',
    'confidence_rank',
    'status',
    'status_reason',
    'source_bc',
    'omission_reason',
)

# The calendar format of the ``utc`` column: ISO calendar date and time, with
# milliseconds, which is enough to place an exposure whose shortest instance in
# the corpus is five of them.
_UTC_FORMAT = 'ISOC'
_UTC_DECIMALS = 3


@dataclass(frozen=True)
class ImageReportFacts:
    """What one image's navigation metadata says about the image itself.

    These are the facts a report row and a segment comment line both carry, so
    a reader of a kernel's comment area and a reader of the report see the same
    numbers.  Every field but the name and the status is optional, because an
    image that failed to load records neither an epoch nor a measurement.

    Parameters:
        image_name: Basename of the image.
        utc: Exposure midtime as a UTC calendar string, or ``None`` when no
            midtime was recorded.
        et: Exposure midtime, TDB seconds past J2000, or ``None``.
        sclk: Exposure midtime as the spacecraft clock string the pipeline
            recorded, or ``None``.
        offset_dv: Navigated offset along ``v``, in pixels, or ``None``.
        offset_du: Navigated offset along ``u``, in pixels, or ``None``.
        sigma_dv: Per-axis one-sigma uncertainty along ``v``, or ``None``.
        sigma_du: Per-axis one-sigma uncertainty along ``u``, or ``None``.
        confidence: The confidence recorded for the result, or ``None``.
        confidence_rank: The rank recorded for the result, or ``None``.
        status: The navigation status recorded for the image.
        status_reason: The status reason recorded for the result, or ``None``
            when the image has no result at all.
    """

    image_name: str
    utc: str | None
    et: float | None
    sclk: str | None
    offset_dv: float | None
    offset_du: float | None
    sigma_dv: float | None
    sigma_du: float | None
    confidence: float | None
    confidence_rank: str | None
    status: str
    status_reason: str | None

    def __post_init__(self) -> None:
        """Refuse an image with no name or no status.

        A row with no name could not be attributed to an image, and an empty
        status would render a cell indistinguishable from an unrecorded value.

        Raises:
            ValueError: if ``image_name`` or ``status`` is empty.
        """
        if len(self.image_name) == 0:
            raise ValueError('image_name is empty; a report row must name the image it is for')
        if len(self.status) == 0:
            raise ValueError(
                f'status is empty for {self.image_name}; a report row must say what became '
                f'of the image'
            )


def utc_for_et(et: float) -> str:
    """Return one epoch as the UTC calendar string the report carries.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The epoch as ``YYYY-MM-DDTHH:MM:SS.sss``.

    Raises:
        RuntimeError: if no leapseconds kernel is furnished, which is what
            converts TDB to UTC.
    """
    return str(cspyce.et2utc(et, _UTC_FORMAT, _UTC_DECIMALS))


def read_image_report_facts(metadata: dict[str, Any]) -> ImageReportFacts:
    """Read one image's reported facts out of its navigation metadata.

    The sources are exactly these, and deliberately not any equivalent-looking
    field beside them: ``observation.image_name`` and the top-level ``status``,
    ``offset`` (``[dv, du]``, unrounded) and ``confidence``; and from
    ``navigation_result``, ``sigma_px`` (rounded there, and reported as
    recorded), ``confidence_rank``, ``status_reason``, and the ``times`` block's
    ``midtime_et`` and ``sclk_midtime``.  The UTC column is that same midtime
    converted, so all three time columns name one instant.

    A document with no ``navigation_result`` -- what an image that failed to
    load records -- yields a row with a name and a status and nothing else,
    which is all there is to report for an image that measured nothing.

    Parameters:
        metadata: The image's full navigation metadata dict.

    Returns:
        The facts.

    Raises:
        ValueError: if the image name or the status is absent, if a numeric
            field holds a non-finite value, or if the offset or sigma does not
            hold exactly two values.
        TypeError: if a field is present but holds a value of the wrong kind.
        RuntimeError: if a midtime was recorded and no leapseconds kernel is
            furnished to express it as UTC.
    """
    observation = read_section(metadata, 'observation', 'metadata')
    result: dict[str, Any] = {}
    if 'navigation_result' in metadata:
        result = read_section(metadata, 'navigation_result', 'metadata')
    times: dict[str, Any] = {}
    if 'times' in result:
        times = read_section(result, 'times', 'navigation_result')
    et = read_optional_number(times, 'midtime_et', 'times')
    offset = read_optional_pair(metadata, 'offset', 'metadata')
    sigma = read_optional_pair(result, 'sigma_px', 'navigation_result')
    return ImageReportFacts(
        image_name=read_text(observation, 'image_name', 'observation'),
        utc=None if et is None else utc_for_et(et),
        et=et,
        sclk=read_optional_text(times, 'sclk_midtime', 'times'),
        offset_dv=None if offset is None else offset[0],
        offset_du=None if offset is None else offset[1],
        sigma_dv=None if sigma is None else sigma[0],
        sigma_du=None if sigma is None else sigma[1],
        confidence=read_optional_number(metadata, 'confidence', 'metadata'),
        confidence_rank=read_optional_text(result, 'confidence_rank', 'navigation_result'),
        status=read_text(metadata, 'status', 'metadata'),
        status_reason=read_optional_text(result, 'status_reason', 'navigation_result'),
    )


@dataclass(frozen=True)
class ReportRow:
    """One image's row: what it measured, and what became of it.

    Parameters:
        facts: What the image's metadata says about the image.
        source_bc: Basename of the corrected C-kernel carrying its segment, or
            ``None`` when it has none.
        omission_reason: Why it has none, or ``None`` when it has one.

    Raises:
        ValueError: if a source file is paired with a reason there is none, or
            if neither is present.
    """

    facts: ImageReportFacts
    source_bc: str | None
    omission_reason: OmissionReason | None

    def __post_init__(self) -> None:
        """Refuse a row that both names a source file and does not."""
        if (self.source_bc is None) == (self.omission_reason is None):
            raise ValueError(
                f'{self.facts.image_name} must be reported with either the file carrying its '
                f'segment or a reason it has none, not both and not neither'
            )
        if self.source_bc is not None and len(self.source_bc) == 0:
            raise ValueError(
                f'{self.facts.image_name} is reported with an empty source file name, which reads '
                f'in the report exactly like an image that was omitted'
            )

    @property
    def image_name(self) -> str:
        """Basename of the image this row is for."""
        return self.facts.image_name

    def values(self) -> tuple[str, ...]:
        """Return the row's cells, in :data:`REPORT_COLUMNS` order.

        Returns:
            One string per column.  A value the metadata did not record is an
            empty cell, which is how the report distinguishes "not measured"
            from a measurement that happened to be zero.
        """
        facts = self.facts
        return tuple(
            _cell(value)
            for value in (
                facts.image_name,
                facts.utc,
                facts.et,
                facts.sclk,
                facts.offset_dv,
                facts.offset_du,
                facts.sigma_dv,
                facts.sigma_du,
                facts.confidence,
                facts.confidence_rank,
                facts.status,
                facts.status_reason,
                self.source_bc,
                None if self.omission_reason is None else self.omission_reason.value,
            )
        )


def _cell(value: str | float | None) -> str:
    """Return one report cell as text.

    Parameters:
        value: The value to render, or ``None`` for a value the metadata did
            not record.

    Returns:
        The value as text, or the empty string.  A float is rendered with
        ``repr``, which round-trips exactly, because the offsets are recorded
        unrounded and a report that rounded them would disagree with the
        metadata it reports.
    """
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return repr(value)


def report_text(rows: Sequence[ReportRow]) -> str:
    """Render a run's report as CSV text.

    Parameters:
        rows: The rows, in the order they should appear.

    Returns:
        The CSV text, a header row followed by one row per image.

    Raises:
        ValueError: if two rows name the same image.  Every image the run
            considered appears exactly once, and a report holding one twice
            would be counted twice by every consumer of it.
    """
    names = [row.image_name for row in rows]
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f'the report holds these images more than once: {repeated}')
    buffer = io.StringIO()
    # Newlines are written by the csv module itself, per its documented usage.
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(REPORT_COLUMNS)
    for row in rows:
        writer.writerow(row.values())
    return buffer.getvalue()


def write_report(path: str | Path | FCPath, rows: Sequence[ReportRow]) -> None:
    """Write a run's report.

    Parameters:
        path: The file to write, local or remote.
        rows: The rows, in the order they should appear.

    Raises:
        ValueError: if two rows name the same image.
    """
    FCPath(path).write_text(report_text(rows))
