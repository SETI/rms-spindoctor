"""spindoctor.nav_records -- where a program gets its navigation records.

A navigation pass writes one JSON document per image, and every program
downstream of navigation reads what that document says.  The same record can be
kept two ways -- as the document itself, or as a row of an ingested results index
-- and no consumer cares which, so they all ask the same four questions through
one seam: one image by its stub, a stream of records, a stream of per-image
facts, and a listing of what is there.

This package is the half of that seam that needs no database.  It holds what a
record is, what a document is named and where one lives, what a caller is asking
for, the protocol both storages implement, and the implementation over the
documents themselves.  It also holds what one document says about its image, in
the shape both storages answer in, so that a consumer summarizing a whole run
reads one shape rather than one storage.  It deliberately imports no database
layer, because every navigation run imports the enumeration that reaches it and
most of those runs name no index.

Public API:

    NavRecord        -- one image's record, and the document it stands for
    UnreadableFile   -- a file a record could not be read out of, and why
    ListedRecord     -- one document a listing found, with its two metrics
    METADATA_SUFFIX  -- what a per-image navigation document is named
    document_path    -- where one stub's document lives, the one join
    stub_refusal           -- why a stub is not a key under a root
    subtree_refusal        -- why a subtree is not one directory under one
    stub_for_document      -- the stub of one document under a root
    read_document          -- read one document, refusing what is not one
    document_or_refusal    -- read one document, or say why it is not one
    NULL_BYTE_IN_PATH      -- why a stub naming a null byte may not be read
    ABSOLUTE_PATH_FRAGMENT -- why an absolute stub may not be read under a root
    PARENT_SEGMENT_IN_PATH -- why a stub naming a parent directory may not be
    NOT_A_SINGLE_COMPONENT -- why a subtree of more than one component may not
    COULD_NOT_RETRIEVE     -- why a file that never arrived is not a record
    UNREADABLE             -- why a file whose bytes would not read is not one
    NOT_VALID_JSON         -- why a file no JSON value came out of is not one
    NOT_A_JSON_OBJECT      -- why JSON of another kind is not one
    NAMES_NO_INSTRUMENT    -- why a document naming no mission is reported
    RECORDS_NO_MIDTIME     -- why a document naming no midtime is reported
    normalize_root_url     -- the one spelling of a results root
    distinct_roots         -- the named roots, normalized and de-duplicated
    Selection              -- which of a source's records a caller is asking for
    RecordSource           -- the protocol both storages implement
    selected_roots         -- the roots of a source one selection covers
    root_for_stub          -- the one root a bare stub is a key under
    root_for_stubs         -- the one root a selection of stubs names keys under
    refuse_what_a_listing_cannot_answer -- what no listing may be asked
    in_batches             -- how both storages batch what they ask for
    TreeRecordSource       -- that protocol over the documents
    UnlistableDirectoryError -- a directory under a root that ends a walk
    UnlistableRootError      -- a root that could not be listed at all
    TreeTuning               -- how much of a pass runs at once
    ImageFacts               -- what one document says about its image
    DocumentOrigin           -- where one document came from, and its metrics
    MetadataDocumentError    -- a document that is not a navigation result
    NOT_A_NAVIGATION_DOCUMENT -- why such a document is refused
    facts_from_document      -- one document into that shape
    subtree_of               -- the directory a stub is under, for both tables
    date_from_image_et       -- the calendar date a date filter compares
    datetime_from_image_et   -- the same instant, to the second
    image_number_from_name   -- the number an image-range filter compares

What the values of a record mean is
:mod:`spindoctor.support.nav_record`'s, and the storage that needs a database is
:mod:`spindoctor.results_index`'s.
"""

from spindoctor.nav_records.derived import (
    date_from_image_et,
    datetime_from_image_et,
    image_number_from_name,
)
from spindoctor.nav_records.document import (
    ABSOLUTE_PATH_FRAGMENT,
    COULD_NOT_RETRIEVE,
    METADATA_SUFFIX,
    NOT_A_JSON_OBJECT,
    NOT_A_SINGLE_COMPONENT,
    NOT_VALID_JSON,
    NULL_BYTE_IN_PATH,
    PARENT_SEGMENT_IN_PATH,
    UNREADABLE,
    document_or_refusal,
    document_path,
    read_document,
    stub_for_document,
    stub_refusal,
    subtree_refusal,
)
from spindoctor.nav_records.facts import (
    NOT_A_NAVIGATION_DOCUMENT,
    DocumentOrigin,
    ImageFacts,
    MetadataDocumentError,
    facts_from_document,
    subtree_of,
)
from spindoctor.nav_records.record import ListedRecord, NavRecord, UnreadableFile
from spindoctor.nav_records.roots import distinct_roots, normalize_root_url
from spindoctor.nav_records.selection import Selection
from spindoctor.nav_records.source import (
    RecordSource,
    in_batches,
    refuse_what_a_listing_cannot_answer,
    root_for_stub,
    root_for_stubs,
    selected_roots,
)
from spindoctor.nav_records.tree import (
    NAMES_NO_INSTRUMENT,
    RECORDS_NO_MIDTIME,
    TreeRecordSource,
)
from spindoctor.nav_records.tuning import TreeTuning
from spindoctor.nav_records.walk import UnlistableDirectoryError, UnlistableRootError

__all__ = [
    'ABSOLUTE_PATH_FRAGMENT',
    'COULD_NOT_RETRIEVE',
    'METADATA_SUFFIX',
    'NAMES_NO_INSTRUMENT',
    'NOT_A_JSON_OBJECT',
    'NOT_A_NAVIGATION_DOCUMENT',
    'NOT_A_SINGLE_COMPONENT',
    'NOT_VALID_JSON',
    'NULL_BYTE_IN_PATH',
    'PARENT_SEGMENT_IN_PATH',
    'RECORDS_NO_MIDTIME',
    'UNREADABLE',
    'DocumentOrigin',
    'ImageFacts',
    'ListedRecord',
    'MetadataDocumentError',
    'NavRecord',
    'RecordSource',
    'Selection',
    'TreeRecordSource',
    'TreeTuning',
    'UnlistableDirectoryError',
    'UnlistableRootError',
    'UnreadableFile',
    'date_from_image_et',
    'datetime_from_image_et',
    'distinct_roots',
    'document_or_refusal',
    'document_path',
    'facts_from_document',
    'image_number_from_name',
    'in_batches',
    'normalize_root_url',
    'read_document',
    'refuse_what_a_listing_cannot_answer',
    'root_for_stub',
    'root_for_stubs',
    'selected_roots',
    'stub_for_document',
    'stub_refusal',
    'subtree_of',
    'subtree_refusal',
]
