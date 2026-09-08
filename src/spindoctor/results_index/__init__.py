"""spindoctor.results_index -- an optional, rebuildable database over the results tree.

The navigation pipeline writes one ``_metadata.json`` document per image.  Programs
downstream of navigation need only a few fields from each, and reading a whole
document per image costs a round trip per image on a cloud root.  This package
defines a database whose rows carry those fields, built by one pass over the tree,
so a consumer answers its questions with a query instead of a walk.

The index is derived and disposable.  The documents remain the authoritative
record, no program requires an index, and deleting one costs nothing but the time
to rebuild it.  It is also a snapshot: it reflects the tree as of the last ingest,
with no staleness detection and no automatic refresh.

Public API:

    SCHEMA_VERSION   -- column-set version a database is stamped with
    STUBS_PER_STATEMENT -- how many stubs one statement names at a time
    METADATA         -- SQLAlchemy MetaData holding every table
    IMAGES           -- one row per image, keyed by (root_url, results_path_stub)
    TECHNIQUES       -- per-technique results for an image
    FEATURE_SOURCES  -- aggregated feature inventory for an image
    FAILED_FILES     -- files that are not current-schema navigation documents
    SCHEMA_META      -- the single row stamping the schema version
    INGEST_RUNS      -- one row per ingest pass over one root
    UNKNOWN_STATUS   -- the status a document that named no outcome is recorded with
    open_index       -- the opener every reader and writer goes through
    open_database    -- the opener the drop uses, which stops before the gate
    masked_url       -- a connection URL with any password in it hidden
    TableContents    -- one table of the index, and how many rows it holds
    IndexContents    -- what of the index a database holds, before a drop
    index_contents   -- read that, so a drop can say what it is about to remove
    index_table_names -- every table the index owns, in drop order
    drop_index_tables -- remove those tables, and nothing else
    normalize_root_url    -- the one spelling of a results root
    ingested_roots        -- the roots a completed ingest covered
    require_ingested_roots -- refuse to read absence from a root nobody ingested
    RootNotIngestedError  -- what that refusal is, for a caller that reports it its own way
    open_index_for_roots  -- open an index, refusing a root it has not ingested
    newest_finish_time    -- when the newest pass over a root finished
    snapshot_finish_time  -- the same, for a caller holding no connection
    unfinished_roots      -- the roots whose newest pass never finished
    reporting_a_failed_read -- the one translation of a database failure into a refusal
    RECORD_FIELDS         -- the one correspondence between columns and record fields
    RecordField           -- one column's place in the record a row is rebuilt into
    record_from_row       -- rebuild the record a row holds, through that mapping
    IndexRecordSource     -- the record seam over the index
    open_record_source    -- open whichever half of that seam a run resolved

Reading a document rather than a row needs no database, so the seam itself --
what a record is, what a document is named, where one lives under a root, how
one is read, what a caller is asking for, and the half that reads documents --
is :mod:`spindoctor.nav_records`'s.
"""

from spindoctor.results_index.drop import (
    IndexContents,
    TableContents,
    drop_index_tables,
    index_contents,
    index_table_names,
)
from spindoctor.results_index.engine import (
    STUBS_PER_STATEMENT,
    open_database,
    open_index,
    reporting_a_failed_read,
)
from spindoctor.results_index.masking import masked_url
from spindoctor.results_index.open_source import open_record_source
from spindoctor.results_index.rebuild import RECORD_FIELDS, RecordField, record_from_row
from spindoctor.results_index.record_source import IndexRecordSource
from spindoctor.results_index.roots import (
    RootNotIngestedError,
    ingested_roots,
    newest_finish_time,
    normalize_root_url,
    open_index_for_roots,
    require_ingested_roots,
    snapshot_finish_time,
    unfinished_roots,
)
from spindoctor.results_index.schema import (
    FAILED_FILES,
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    METADATA,
    SCHEMA_META,
    SCHEMA_VERSION,
    TECHNIQUES,
    UNKNOWN_STATUS,
)

__all__ = [
    'FAILED_FILES',
    'FEATURE_SOURCES',
    'IMAGES',
    'INGEST_RUNS',
    'METADATA',
    'RECORD_FIELDS',
    'SCHEMA_META',
    'SCHEMA_VERSION',
    'STUBS_PER_STATEMENT',
    'TECHNIQUES',
    'UNKNOWN_STATUS',
    'IndexContents',
    'IndexRecordSource',
    'RecordField',
    'RootNotIngestedError',
    'TableContents',
    'drop_index_tables',
    'index_contents',
    'index_table_names',
    'ingested_roots',
    'masked_url',
    'newest_finish_time',
    'normalize_root_url',
    'open_database',
    'open_index',
    'open_index_for_roots',
    'open_record_source',
    'record_from_row',
    'reporting_a_failed_read',
    'require_ingested_roots',
    'snapshot_finish_time',
    'unfinished_roots',
]
