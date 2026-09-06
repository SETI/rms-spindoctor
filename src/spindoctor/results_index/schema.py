"""SQLAlchemy Core schema for the navigation results index.

The index is a database whose rows are derived from the navigation results tree
by a separate ingest step.  It is not authoritative: the per-image
``_metadata.json`` documents are, and the index can be deleted and rebuilt from
them at any time.  Consumers that only need a few fields per image read one row
instead of one document.

Keying
------

The primary key of ``images`` is the pair ``(root_url, results_path_stub)``.
``results_path_stub`` is the volume-and-filespec fragment every consumer already
uses to address a result (for example
``COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB``); it is the only
identifier unique across volumes, because two volumes may hold images with the
same basename.  ``root_url`` is the ingested results root in normalized form, so
one database can serve several roots without a consumer seeing another root's
rows.  The child tables ``techniques`` and ``feature_sources`` key on the same
pair and cascade on delete, so re-ingesting an image replaces its children.  Each
of them also declares the tuple that identifies one of its rows -- the technique
name, or the feature type and its source -- unique within an image, so a repeated
or retried insert is refused rather than silently doubling every count read from
it.

``results_path_stub`` carries a non-unique index of its own, for lookups that
already know the root, and ``image_date`` and ``instrument`` carry one each,
because the report groups and filters on both across a whole root.  Both
``images`` and ``failed_files`` carry ``(root_url, subtree)``, because a
selection restricted to some of a root's top-level directories filters on that
pair in both tables and the primary key's second column is the stub.

``failed_files`` keys on the same pair and holds the files that are not
current-schema navigation documents at all.  It is a table of its own rather
than a marked ``images`` row precisely because absence of an ``images`` row is
what a consumer reads as "this image was never navigated": a file with no usable
data must leave that answer alone while still recording enough for the next
ingest to skip it.  It also carries the one fact the walk knows about a file
whatever the file says -- the subtree it lives under -- because a selection
filter asks about the file rather than about its contents, and a refused file
is one the tree still holds.

Types
-----

Every pixel, ET and sigma column is ``Double`` rather than ``Float``, because a
dialect is free to map ``Float`` to single precision and the stored offset would
then lose digits the document recorded.  Booleans are ``Boolean``, never an
integer flag, so that a PostgreSQL backend rejects integer arithmetic on them.
Structured values -- a covariance matrix, a rotation, a list of names -- are
``JSON``.

What a stored number reads back as
----------------------------------

A ``Double`` column returns every finite double exactly, on both backends, with
one exception: SQLite does not distinguish negative zero from positive zero, so
a ``-0.0`` written there reads back ``0.0``.  PostgreSQL returns it as it was
written.  No other value of any magnitude changes on either backend.

A JSON column returns exactly what was written -- sign of zero, magnitude and
all -- only where it is declared :data:`_EXACT_JSON`, whose value travels as the
JSON text the driver wrote and is parsed back by the same rules that wrote it.
The :data:`_QUERYABLE_JSON` declaration is ``jsonb`` on PostgreSQL, whose number
type is ``numeric``: a ``-0.0`` reads back ``0.0``, and a float of large
magnitude reads back as a Python ``int`` of the same value rather than as a
float.  So a JSON column that can hold a number is :data:`_EXACT_JSON` and a
column holding text alone is :data:`_QUERYABLE_JSON`, and the choice is pinned
per column by the schema tests.

Negative zero is not a curiosity in these columns.  The navigator rounds a
covariance to four decimals, so any term of smaller magnitude than that and
negative sign is written as ``-0.0``; and a C-kernel is built from ``cmatrix``
and ``cmatrix_original``, so one written from an index has to hold what one
written from the same documents holds.

A JSON column distinguishes *absent* from *empty*.  An absent value is SQL NULL,
so ``WHERE cmatrix IS NOT NULL`` finds the rows that carry a matrix; without
that declaration the column would hold the JSON value ``null``, which is a value
and satisfies ``IS NOT NULL`` on every row ever written.  An empty list or empty
object is a statement rather than an absence -- nothing was excluded from the
consensus, the technique named no sources, it reported no diagnostics -- and is
stored as the empty container it is.

Versioning
----------

``schema_meta`` holds a single row carrying :data:`SCHEMA_VERSION`.  There are no
migrations: ingest is cheap relative to navigation and entirely reproducible from
the tree, so the remedy for a column set an index does not hold is to delete the
database and re-ingest.
"""

from typing import Any

import sqlalchemy
from sqlalchemy.dialects import postgresql

# The ``status`` column is NOT NULL, so a document whose top-level ``status``
# is absent, null, empty or not a string is recorded as naming no outcome.  The
# value it is recorded as is the one every reader of that document reports for
# it, and it is taken from there rather than restated here, so the column cannot
# come to say something the readers do not.
from spindoctor.support.nav_record import UNKNOWN_STATUS

__all__ = [
    'FAILED_FILES',
    'FEATURE_SOURCES',
    'IMAGES',
    'INGEST_RUNS',
    'METADATA',
    'SCHEMA_META',
    'SCHEMA_VERSION',
    'TECHNIQUES',
    'UNKNOWN_STATUS',
]

SCHEMA_VERSION = 1
"""Column-set version of the index.

Stamped into every index at creation and compared on every later open: a
database stamped with a different version is refused rather than migrated,
naming both versions, and the remedy it prescribes is to empty the database and
ingest the tree again.
"""


# TEXT on SQLite, json on PostgreSQL.  The value is stored as the JSON text the
# driver wrote and parsed back by the same rules, so every number in it reads
# back as the number that was written.  This is the declaration a column holding
# a float takes, whatever else it holds beside it.  none_as_null makes an absent
# value SQL NULL rather than the JSON value null, which is what any IS NULL test
# over these columns reads, and what an exported CSV leaves as an empty cell.
_EXACT_JSON = sqlalchemy.JSON(none_as_null=True)

# TEXT on SQLite, jsonb on PostgreSQL.  jsonb is the type PostgreSQL's array and
# object accessors operate on, so a direct-SQL query against the index can reach
# inside these values without a cast.  Its number type is numeric rather than a
# double, so this declaration is for a column holding text and nothing else: it
# would return a stored -0.0 as 0.0 and a large-magnitude float as an integer.
_QUERYABLE_JSON = sqlalchemy.JSON(none_as_null=True).with_variant(
    postgresql.JSONB(none_as_null=True), 'postgresql'
)

METADATA = sqlalchemy.MetaData()
"""Container for every table of the index; the source of all DDL."""


def _image_key_columns() -> tuple[sqlalchemy.Column[Any], sqlalchemy.Column[Any]]:
    """Return a fresh pair of columns naming the image a child row belongs to.

    A ``Column`` object binds to exactly one table, so each child table needs its
    own pair rather than a shared constant.

    Returns:
        The ``root_url`` and ``results_path_stub`` columns, both NOT NULL.
    """
    return (
        sqlalchemy.Column('root_url', sqlalchemy.Text, nullable=False),
        sqlalchemy.Column('results_path_stub', sqlalchemy.Text, nullable=False),
    )


def _image_foreign_key() -> sqlalchemy.ForeignKeyConstraint:
    """Return a fresh composite foreign key from a child table to ``images``.

    Deleting an image row deletes its children, which is what makes the
    delete-then-insert upsert of one image atomic and complete.

    Returns:
        The constraint, which the caller passes to its ``Table``.
    """
    return sqlalchemy.ForeignKeyConstraint(
        ['root_url', 'results_path_stub'],
        ['images.root_url', 'images.results_path_stub'],
        ondelete='CASCADE',
    )


IMAGES = sqlalchemy.Table(
    'images',
    METADATA,
    # Identity.  Derived from the file's own location under the ingest root, not
    # from the document, so it is exact by construction.
    sqlalchemy.Column('root_url', sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column('results_path_stub', sqlalchemy.Text, primary_key=True),
    # First path segment of the stub, which is the top-level directory of the
    # results tree the image sits under; NULL when the stub has no separator,
    # which is what the simulated dataset's bare scene basenames produce.
    sqlalchemy.Column('subtree', sqlalchemy.Text),
    # Observation.
    sqlalchemy.Column('image_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('instrument', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('camera', sqlalchemy.Text),
    # What the exposure was commanded as, where the dataset records one.  Two
    # cameras exposed together share one bus attitude that cannot honor two
    # different corrections, and this is what says an exposure was one of them.
    sqlalchemy.Column('shutter_mode', sqlalchemy.Text),
    sqlalchemy.Column('image_path', sqlalchemy.Text),
    # A copy of the epoch the document recorded as provenance, which is the
    # observation's midtime.  The report aggregates it into the time span it
    # gives per instrument; a date bound compares the calendar date beside it
    # rather than this column.  NULL for an image that never loaded, since the
    # epoch comes from the observation the load never built.
    sqlalchemy.Column('image_et', sqlalchemy.Double),
    sqlalchemy.Column('image_date', sqlalchemy.Text),
    # Outcome.  status_error and status_reason are different vocabularies:
    # status_error is what the SPICE-error selection filter matches verbatim,
    # status_reason is the navigator's explanation of a non-success outcome.
    sqlalchemy.Column('status', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('status_error', sqlalchemy.Text),
    # The traceback the document recorded for that error, copied whole.  A
    # sweep asking the index which images failed and why has no cheap way to
    # reach the per-image log beside each document, and the failures worth
    # chasing are the ones a message alone does not place.
    sqlalchemy.Column('status_traceback', sqlalchemy.Text),
    sqlalchemy.Column('status_reason', sqlalchemy.Text),
    # The authoritative offset every consumer applies, stored unrounded.
    sqlalchemy.Column('offset_dv', sqlalchemy.Double),
    sqlalchemy.Column('offset_du', sqlalchemy.Double),
    sqlalchemy.Column('sigma_dv', sqlalchemy.Double),
    sqlalchemy.Column('sigma_du', sqlalchemy.Double),
    # The fused covariance as the document wrote it, square and row-major.  A
    # twist-fitted result records 3x3, whose rotation row and column carry the
    # offset-to-rotation cross terms; sigma_rotation_deg is one number and
    # cannot stand in for them, so the matrix is stored whole rather than
    # reduced to the offset block.
    sqlalchemy.Column('covariance_px2', _EXACT_JSON),
    sqlalchemy.Column('sigma_along_unobservable_px', sqlalchemy.Double),
    sqlalchemy.Column('rotation_deg', sqlalchemy.Double),
    sqlalchemy.Column('sigma_rotation_deg', sqlalchemy.Double),
    # Quality.
    sqlalchemy.Column('confidence', sqlalchemy.Double),
    sqlalchemy.Column('confidence_rank', sqlalchemy.Text),
    sqlalchemy.Column('n_techniques', sqlalchemy.Integer, nullable=False),
    # In the order the document wrote them, whatever order that is.  A recorded
    # list is stored as recorded, so a consumer comparing this column against
    # the document it came from finds the same list; one wanting some other
    # order applies it.
    sqlalchemy.Column('excluded_from_consensus', _QUERYABLE_JSON),
    sqlalchemy.Column('image_class', sqlalchemy.Text),
    sqlalchemy.Column('noise_sigma', sqlalchemy.Double),
    sqlalchemy.Column('image_shape_v', sqlalchemy.Integer),
    sqlalchemy.Column('image_shape_u', sqlalchemy.Integer),
    # Run provenance.
    sqlalchemy.Column('run_start', sqlalchemy.Text),
    sqlalchemy.Column('run_end', sqlalchemy.Text),
    sqlalchemy.Column('elapsed_s', sqlalchemy.Double),
    # The largest resident size the navigating process reached, which is the
    # figure an out-of-memory kill is decided against.  Null where the kernel
    # publishes no peak, which is every kernel but Linux's.
    sqlalchemy.Column('peak_memory_bytes', sqlalchemy.BigInteger),
    sqlalchemy.Column('config_hash', sqlalchemy.Text),
    sqlalchemy.Column('git_sha', sqlalchemy.Text),
    sqlalchemy.Column('pipeline_run', sqlalchemy.Text),
    # The kernels the run recorded, as basenames in the order recorded.  A
    # correction is an overlay on one original kernel, and this is what says
    # which originals the attitude it corrects was measured against.  An empty
    # list is a statement -- the run named none -- and NULL is a document with
    # no provenance block at all.
    sqlalchemy.Column('spice_kernels', _QUERYABLE_JSON),
    # The numeric portion of the image name, so a range filter compares against a
    # column instead of calling a function.  BigInteger because an instrument
    # naming scheme is free to run past the 32-bit range a dialect may impose.
    sqlalchemy.Column('image_number', sqlalchemy.BigInteger),
    # Corrected-pointing fields, carrying the names and shapes their producer
    # specifies.  They are absent from a document with no navigation result at
    # all, and cmatrix is additionally absent where the navigation fitted a
    # camera rotation, so both cases are stored as NULL.
    sqlalchemy.Column('start_et', sqlalchemy.Double),
    sqlalchemy.Column('stop_et', sqlalchemy.Double),
    # Stored as the navigator recorded it, never recomputed from the shutter
    # epochs beside it.  It is the epoch a recorded attitude belongs to, and a
    # reader gates it against the observation's own midtime to a microsecond, so
    # what the index has to carry is the value that was written rather than one
    # that happens to agree with it for a particular producer's arithmetic.
    sqlalchemy.Column('midtime_et', sqlalchemy.Double),
    sqlalchemy.Column('exposure_s', sqlalchemy.Double),
    sqlalchemy.Column('sclk_start', sqlalchemy.Text),
    sqlalchemy.Column('sclk_midtime', sqlalchemy.Text),
    sqlalchemy.Column('sclk_stop', sqlalchemy.Text),
    # The frame the recorded attitude is expressed in, by name and by id.  The
    # name is what a kernel writer looks up in the frame kernels it furnishes;
    # readers that gate an attitude against the observation take the frame
    # identity from the observation instead and never consult the name.  Both
    # identifiers are read back by every consumer that rebuilds a pointing.
    sqlalchemy.Column('camera_frame', sqlalchemy.Text),
    sqlalchemy.Column('camera_frame_id', sqlalchemy.Integer),
    sqlalchemy.Column('ck_frame_id', sqlalchemy.Integer),
    sqlalchemy.Column('cmatrix', _EXACT_JSON),
    sqlalchemy.Column('cmatrix_original', _EXACT_JSON),
    # File provenance.  mtime_ns and size_bytes drive the incremental skip, and
    # both need 64 bits: a nanosecond epoch alone is far past the 32-bit range.
    sqlalchemy.Column('source_file', sqlalchemy.Text),
    sqlalchemy.Column('mtime_ns', sqlalchemy.BigInteger),
    sqlalchemy.Column('size_bytes', sqlalchemy.BigInteger),
    sqlalchemy.Index('ix_images_results_path_stub', 'results_path_stub'),
    # The key's second column is the stub, so a selection restricted to some
    # subtrees has no ordered path to its rows and reads every row under the
    # root to find them.  Both tables carry the pair, because a selection asks
    # both of them the same question and a shortfall is counted from the half
    # that has no images row.
    sqlalchemy.Index('ix_images_root_url_subtree', 'root_url', 'subtree'),
    sqlalchemy.Index('ix_images_image_date', 'image_date'),
    sqlalchemy.Index('ix_images_instrument', 'instrument'),
)
"""One row per navigated image, keyed by ``(root_url, results_path_stub)``."""

TECHNIQUES = sqlalchemy.Table(
    'techniques',
    METADATA,
    *_image_key_columns(),
    sqlalchemy.Column('technique_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('offset_dv', sqlalchemy.Double),
    sqlalchemy.Column('offset_du', sqlalchemy.Double),
    # As on images, and for the same reason: the technique's own covariance as
    # it was written, square and row-major.  Storing the per-axis sigmas instead
    # would lose the correlation between the axes, which is what the off-diagonal
    # terms state and which no pair of sigmas can carry; a reader that wants the
    # sigmas takes the square roots of the diagonal.
    sqlalchemy.Column('covariance_px2', _EXACT_JSON),
    sqlalchemy.Column('confidence', sqlalchemy.Double),
    sqlalchemy.Column('spurious', sqlalchemy.Boolean, nullable=False),
    sqlalchemy.Column('at_edge', sqlalchemy.Boolean, nullable=False),
    sqlalchemy.Column('source_names', _QUERYABLE_JSON),
    sqlalchemy.Column('diagnostics', _EXACT_JSON),
    _image_foreign_key(),
    # One row per technique per image, enforced rather than assumed: a technique
    # reports once for an image, and a retried or duplicated ingest that inserted
    # a second row would change every count and average read from this table
    # without replacing anything.  The constraint's index also serves the lookup
    # by image, whose columns are its leading pair.
    sqlalchemy.UniqueConstraint(
        'root_url', 'results_path_stub', 'technique_name', name='uq_techniques_image_technique'
    ),
)
"""One row per technique that produced a result for an image."""

FEATURE_SOURCES = sqlalchemy.Table(
    'feature_sources',
    METADATA,
    *_image_key_columns(),
    sqlalchemy.Column('feature_type', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('source_model', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('source_name', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('n_features', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column('n_gated', sqlalchemy.Integer, nullable=False),
    _image_foreign_key(),
    # The inventory is aggregated by exactly this tuple, so a second row for one
    # of them is a contradiction rather than more detail.  As on techniques, the
    # constraint's index leads with the image key and serves the lookup by image.
    sqlalchemy.UniqueConstraint(
        'root_url',
        'results_path_stub',
        'feature_type',
        'source_model',
        'source_name',
        name='uq_feature_sources_image_source',
    ),
)
"""Feature inventory of an image, aggregated per feature type and source."""

FAILED_FILES = sqlalchemy.Table(
    'failed_files',
    METADATA,
    # Deliberately not a child of images and deliberately not a row in it: a
    # file this table names produced no usable data, and "no images row" is
    # exactly what every consumer reads as "this image was never navigated".
    # A row here would be a row there, and would answer that question wrongly.
    sqlalchemy.Column('root_url', sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column('results_path_stub', sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column('reason', sqlalchemy.Text, nullable=False),
    # First path segment of the stub, derived exactly as the images column is,
    # so that a selection restricted to some subtrees is one restriction in one
    # query rather than a second pass over every refusal the root holds.
    sqlalchemy.Column('subtree', sqlalchemy.Text),
    # The same two metrics images carries, for the same purpose: a file that
    # was refused and has not changed since is not read again.  Without them a
    # tree whose non-navigation files outnumber its results pays to re-read
    # every one of them on every pass, forever.
    sqlalchemy.Column('mtime_ns', sqlalchemy.BigInteger),
    sqlalchemy.Column('size_bytes', sqlalchemy.BigInteger),
    sqlalchemy.Index('ix_failed_files_root_url_subtree', 'root_url', 'subtree'),
)
"""One row per file that could not be read as a current-schema document."""

SCHEMA_META = sqlalchemy.Table(
    'schema_meta',
    METADATA,
    # A constant primary key with a check constraint: the table describes the
    # database, so a second row would describe a contradiction.
    sqlalchemy.Column('singleton', sqlalchemy.Integer, primary_key=True, autoincrement=False),
    sqlalchemy.Column('schema_version', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column('created_utc', sqlalchemy.Text, nullable=False),
    sqlalchemy.CheckConstraint('singleton = 1', name='ck_schema_meta_singleton'),
)
"""Single row stamping the database with the column-set version that wrote it."""

INGEST_RUNS = sqlalchemy.Table(
    'ingest_runs',
    METADATA,
    sqlalchemy.Column('run_id', sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column('root_url', sqlalchemy.Text, nullable=False),
    sqlalchemy.Column('started_utc', sqlalchemy.Text, nullable=False),
    # NULL while the run is in flight.  A root whose newest row has no finish
    # time, or has no row at all, has not been ingested, and a consumer must say
    # so rather than read absence as "nothing was navigated".
    sqlalchemy.Column('finished_utc', sqlalchemy.Text),
    sqlalchemy.Column('files_seen', sqlalchemy.Integer),
    sqlalchemy.Column('files_ingested', sqlalchemy.Integer),
    sqlalchemy.Column('files_skipped', sqlalchemy.Integer),
    sqlalchemy.Column('files_failed', sqlalchemy.Integer),
    # Rows whose document is no longer in the tree, deleted by this run.
    sqlalchemy.Column('files_removed', sqlalchemy.Integer),
    sqlalchemy.Column('schema_version', sqlalchemy.Integer, nullable=False),
    sqlalchemy.Index('ix_ingest_runs_root_url', 'root_url'),
)
"""One row per ingest pass over one root, recording what it covered."""
