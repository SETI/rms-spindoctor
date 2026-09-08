"""The one correspondence between a row's columns and a record's fields.

Every consumer that reads the index instead of the documents asks the same thing
of it: give me back the navigation record, so that the code which classifies a
pointing, decides an eligibility or writes a kernel is the same code whichever
storage answered.  What stands between a row and that record is a mapping from
column to field, and this module is that mapping written once.

It was worth writing once.  Each consumer used to carry its own rebuild, and two
rebuilds of one row are two answers about what a document said: they agreed on
the day they were written and then each grew the rules the other did not.  A
consumer here names the columns it wants -- nobody selects forty columns to read
five -- and :func:`record_from_row` puts back whatever the row carries, so the
shapes cannot diverge and a column added for one consumer is available to the
next without a second rebuild being written.

The rules, all of which used to be stated twice
-----------------------------------------------

**A column the row does not carry is a field the record did not have.**  Every
consumer distinguishes an absent field from one holding null, so an absent value
is left out rather than written as null.

**``offset`` is the exception**, and is written as a key holding null when the
row carries no pair.  That is what the navigator writes for an image which
measured none, and it decides the name a shortfall is counted under.

**The status column is NOT NULL** and stands in for a document that named no
outcome with :data:`~spindoctor.support.nav_record.UNKNOWN_STATUS`.  That value
is rebuilt as the absent field it stands for, so a record naming no outcome is
read as one by :func:`~spindoctor.support.nav_record.record_status` on both
paths.  A document naming that same word for itself is rebuilt without the field
too, which is the one case the sentinel cannot tell apart.

**Half a pair is not a pair.**  Ingest stores a pair it could not read whole as
two NULLs, so a rebuilt pair needs both halves.

**A block is written only when something goes in it.**  A record with no
pointing block and one whose pointing block held nothing readable are the same
record to every consumer, and the block's presence is what several of them read
to tell a missing block from a missing attitude inside one.

What a rebuilt record cannot recover
------------------------------------

**A value the ingest could not store is rebuilt as absent.**  Ingest stores a
malformed offset, sigma or confidence as NULL, exactly as it stores an absent
one, so the row cannot say which it was and the rebuilt record reads as one that
recorded nothing there.  A consumer that refuses a malformed value outright when
it reads the document therefore reports the image as one that recorded no value
when it reads the row.  This is the one class of difference between the two
storages, it is a property of what a column can hold rather than of this module,
and each consumer's documentation says what it does about it.

Which columns are here, and which cannot be
-------------------------------------------

:data:`RECORD_FIELDS` holds every column that is a faithful copy of one field of
one document.  The rest of the table falls into two groups that a rebuild must
not invent a place for, and both are named here so that a column added to the
schema is deliberately sorted into one of the three rather than quietly missed:

* :data:`IDENTITY_COLUMNS` say where the document is, not what it said.
* :data:`DERIVED_COLUMNS` are computed by the ingest from a document rather than
  copied out of one field of it -- a date rendered from the recorded epoch, a
  number read out of an image name, a count of a list -- so no field of a record
  is what they came from.

The forward direction, document to row, is
:func:`~spindoctor.nav_records.facts.facts_from_document`, which reads each
field through the accessors in :mod:`spindoctor.support.nav_record` that the
consumers read it through.  The invariant that ties the two together is that a
record rebuilt from the columns classifies exactly as its document does.
"""

from dataclasses import dataclass
from typing import Any

import sqlalchemy

from spindoctor.support.nav_record import UNKNOWN_STATUS

__all__ = [
    'DERIVED_COLUMNS',
    'IDENTITY_COLUMNS',
    'RECORD_FIELDS',
    'RecordField',
    'record_from_row',
]


@dataclass(frozen=True)
class RecordField:
    """Where one column's value sits in the record a row is rebuilt into.

    Parameters:
        columns: The column the value comes from, or the two columns holding the
            halves of a recorded pair.
        block: The blocks the field sits inside, outermost first, and empty for a
            field of the record itself.
        field: The field's name inside that block.
        sentinel: A stored value standing for a field the document did not have,
            or None when every value the column can hold is one a document held.
        always_written: Whether the field is written even when the row carries no
            value for it.  One field is; see the module docstring.
    """

    columns: tuple[str, ...]
    block: tuple[str, ...]
    field: str
    sentinel: str | None = None
    always_written: bool = False


_OBSERVATION = ('observation',)
_RESULT = ('navigation_result',)
_CLASSIFIER = ('navigation_result', 'image_classifier')
_POINTING = ('navigation_result', 'pointing')
_PROVENANCE = ('navigation_result', 'provenance')
_TIMES = ('navigation_result', 'times')
_TIMING = ('timing',)

RECORD_FIELDS: tuple[RecordField, ...] = (
    RecordField(('image_name',), _OBSERVATION, 'image_name'),
    RecordField(('instrument',), _OBSERVATION, 'instrument'),
    RecordField(('camera',), _OBSERVATION, 'camera'),
    RecordField(('shutter_mode',), _OBSERVATION, 'shutter_mode'),
    RecordField(('image_path',), _OBSERVATION, 'image_path'),
    RecordField(('image_shape_v', 'image_shape_u'), _OBSERVATION, 'image_shape'),
    RecordField(('status',), (), 'status', sentinel=UNKNOWN_STATUS),
    RecordField(('status_error',), (), 'status_error'),
    RecordField(('status_traceback',), (), 'status_traceback'),
    RecordField(('offset_dv', 'offset_du'), (), 'offset', always_written=True),
    RecordField(('confidence',), (), 'confidence'),
    RecordField(('status_reason',), _RESULT, 'status_reason'),
    RecordField(('sigma_dv', 'sigma_du'), _RESULT, 'sigma_px'),
    RecordField(('covariance_px2',), _RESULT, 'covariance_px2'),
    RecordField(('sigma_along_unobservable_px',), _RESULT, 'sigma_along_unobservable_px'),
    RecordField(('rotation_deg',), _RESULT, 'rotation_deg'),
    RecordField(('sigma_rotation_deg',), _RESULT, 'sigma_rotation_deg'),
    RecordField(('confidence_rank',), _RESULT, 'confidence_rank'),
    RecordField(('excluded_from_consensus',), _RESULT, 'excluded_from_consensus'),
    RecordField(('image_class',), _CLASSIFIER, 'class'),
    RecordField(('noise_sigma',), _CLASSIFIER, 'noise_sigma'),
    RecordField(('image_et',), _PROVENANCE, 'image_et'),
    RecordField(('config_hash',), _PROVENANCE, 'config_hash'),
    RecordField(('git_sha',), _PROVENANCE, 'spindoctor_git_sha'),
    RecordField(('pipeline_run',), _PROVENANCE, 'pipeline_run_iso8601'),
    RecordField(('spice_kernels',), _PROVENANCE, 'spice_kernels'),
    RecordField(('start_et',), _TIMES, 'start_et'),
    RecordField(('stop_et',), _TIMES, 'stop_et'),
    RecordField(('midtime_et',), _TIMES, 'midtime_et'),
    RecordField(('exposure_s',), _TIMES, 'exposure_s'),
    RecordField(('sclk_start',), _TIMES, 'sclk_start'),
    RecordField(('sclk_midtime',), _TIMES, 'sclk_midtime'),
    RecordField(('sclk_stop',), _TIMES, 'sclk_stop'),
    RecordField(('camera_frame',), _POINTING, 'camera_frame'),
    RecordField(('camera_frame_id',), _POINTING, 'camera_frame_id'),
    RecordField(('ck_frame_id',), _POINTING, 'ck_frame_id'),
    RecordField(('cmatrix',), _POINTING, 'cmatrix'),
    RecordField(('cmatrix_original',), _POINTING, 'cmatrix_original'),
    RecordField(('run_start',), _TIMING, 'start_iso8601'),
    RecordField(('run_end',), _TIMING, 'end_iso8601'),
    RecordField(('elapsed_s',), _TIMING, 'elapsed_s'),
)
"""Every column that is a faithful copy of one field of one document.

In the order a rebuilt record carries them, which is the order the navigator
writes them, so that a rebuilt record reads like the document it stands for.
"""

IDENTITY_COLUMNS = frozenset(
    {
        'root_url',
        'results_path_stub',
        'source_file',
        'mtime_ns',
        'size_bytes',
    }
)
"""Columns saying where a document is and what the walk saw of it.

None of them is a field of the record.  The first two are its key, and a
consumer selects them to know which image answered; the last three are what lets
a later pass skip a document it has already read.
"""

DERIVED_COLUMNS = frozenset(
    {
        'subtree',
        'image_date',
        'image_number',
        'n_techniques',
    }
)
"""Columns the ingest computes from a document rather than copying out of it.

``subtree`` is the first path segment of the stub, which the walk knows and no
document states; ``image_date`` is rendered from the recorded epoch and
``image_number`` from the image name; and ``n_techniques`` counts a list.  A
rebuild that put any of them back would be inventing a field the document did
not have, so each is read as a column by whatever wants it and none is a record
field.
"""


def _pair_value(first: Any, second: Any) -> list[Any] | None:
    """Return a recorded pair, or None when the row does not carry one whole.

    Parameters:
        first: The first member's column value.
        second: The second member's column value.

    Returns:
        The pair, or None when either half is NULL.  Ingest stores a pair it
        could not read whole as two NULLs, so half a pair is not a pair.
    """
    if first is None or second is None:
        return None
    return [first, second]


def _place(record: dict[str, Any], block: tuple[str, ...], field: str, value: Any) -> None:
    """Write one field into the record, making the blocks above it as needed.

    A block therefore exists exactly when something was written into it, which
    is what lets a consumer read the presence of a block as a fact about the
    record rather than about this function.

    Parameters:
        record: The record being rebuilt, modified in place.
        block: The blocks the field sits inside, outermost first.
        field: The field's name inside that block.
        value: What to write.
    """
    holder = record
    for name in block:
        nested = holder.get(name)
        if not isinstance(nested, dict):
            nested = {}
            holder[name] = nested
        holder = nested
    holder[field] = value


def record_from_row(row: sqlalchemy.Row[Any]) -> dict[str, Any]:
    """Rebuild the navigation record one index row records.

    Only the columns the row carries are put back: a consumer selects the
    columns it reads, and the record it gets is the part of the document those
    columns hold.  A column the row carries that is no field of a record -- its
    key, the labels a lookup joins on -- is passed over rather than refused, so
    a consumer can select what it needs to identify the row alongside what it
    needs to read it.

    Parameters:
        row: One row of ``images``, carrying any subset of its columns.

    Returns:
        The rebuilt record.

    """
    values = row._mapping
    record: dict[str, Any] = {}
    for entry in RECORD_FIELDS:
        if not any(column in values for column in entry.columns):
            continue
        if len(entry.columns) == 1:
            value = values.get(entry.columns[0])
            if value is not None and value == entry.sentinel:
                continue
        else:
            first, second = entry.columns
            value = _pair_value(values.get(first), values.get(second))
        if value is None and not entry.always_written:
            continue
        _place(record, entry.block, entry.field, value)
    return record
