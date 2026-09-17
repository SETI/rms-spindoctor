# SpinDoctor Results Index Plan

*Implementation plan for an optional, rebuildable database index over the
navigation results tree, so that programs which need a few fields per image
stop reading one JSON document per image. Written to be executed by an
implementing model with no briefing beyond `/seti/newnav/CLAUDE.md` and the
repository itself. Conventions from `CLAUDE.md` and `.cursor/rules/` apply
throughout: line length 100, mypy strict, pdslogger-only logging,
Google-style docstrings with `Parameters:`, Conventional Commits, one logical
change per commit, modules under 1000 lines, no issue numbers in docstrings
or `.rst` files (only in `#` comments).*

Integration branch: `rf_results_index`, cut from `main`. Each phase below
lands as its own pull request targeting that branch, so each gets an
independent review pass before the branch merges to `main`.

---

## 1. Purpose and scope

The navigation pipeline writes one `_metadata.json` per image under
`nav_results_root`. Every program downstream of navigation then reads one of
those documents per image it processes. On a local disk that is cheap. On a
cloud root it is one paid round trip per image per program, and a
Cassini-scale run is order 400,000 images. The curation triage tool is the
extreme case: up to ten full-tree recursive globs per candidate frame.

The fields those programs actually consume are narrow: `status`,
`status_error`, the top-level `offset`, and a handful of quality numbers. One
pass that reads every document once can answer all of those questions from
then on. That pass is the index this plan builds.

**In scope:** a stub-keyed schema over the results tree; a SQLAlchemy Core
data-access layer with a connection-URL backend selection (SQLite and
PostgreSQL); incremental ingest with per-root bookkeeping; a cloud-task
ingest driver; a `--results-index-db` opt-in on consuming programs; and the
conversion of the statistics programs, the backplane driver, reprojection
offset lookup, and the `ResultsFilter` selection filters to read through the
index when it is given.

**Out of scope, deliberately:**

- Replacing the JSON documents. They remain the authoritative record. The
  index is derived and disposable; deleting it costs nothing but the time to
  rebuild.
- Making any program require a database. Every program, `sd_stats_report`
  included, must run correctly with no index at all, and that path stays the
  default.
- Writing the index from the navigation pipeline. Navigation writes JSON;
  ingest reads it. No navigation run opens a database at all unless it was
  given a `--results-index-db` URL to answer its selection from.
- Automatic ingest. No batch driver invokes ingest as a side effect; an
  operator runs it.
- Bundle generation (`spindoctor.cli.pds4.bundle_data`), which serializes the
  entire navigation document into the PDS4 supplemental product, and
  `sd_consolidate_metadata`, which copies raw file bytes. Neither is served
  by a column schema; both keep reading files. Section 7 records the
  extension that would serve bundle generation.
- The curation triage tool (`util/cohort_curation/triage_stage_b.py`). It
  consumes five fields the schema does not carry (`missing_frac`,
  `saturation_frac`, classifier flags, the per-technique list, and result
  file paths), reads two roots, and keys by bare image name rather than by
  stub. Serving it means widening the schema for one out-of-package tool;
  that is a follow-up (section 7), not a phase.
- Ingesting documents written by an older metadata schema. Ingest reads the
  current schema and reports anything else as an error for that file.
- MySQL. PostgreSQL is the only server backend; MySQL was considered and
  rejected.

---

## 2. Target design

### 2.1 What the index is

A database whose rows are derived from the results tree by a separate ingest
step. A consumer given `--results-index-db` answers its questions with SQL instead
of file reads. A consumer given no such option behaves exactly as it does
today, reading files. `sd_stats_report` is answered by whichever storage it
is pointed at: one pass of accumulators over the per-image facts the record
seam yields, so an index makes the same report cheaper rather than possible.

The index is a **snapshot**. It reflects the tree as of the last ingest over
the roots it covers. Consumers trust its contents as given: there is no
staleness detection, no re-verification against the tree, and no automatic
refresh. An operator who navigates more images and wants them visible runs
ingest again. This is a deliberate simplification, and it must be stated
plainly in the user guide, because the failure mode it permits -- a consumer
silently working from stale rows -- is otherwise surprising.

### 2.2 Keying

The primary key of `images` is `(root_url, results_path_stub)`.

`results_path_stub` is the volume-and-filespec path fragment every consumer
already uses to address a result, carried on `ImageFile.results_path_stub`.
A real example: `COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB`
(the PDS3 datasets build it as `{volume}/{filespec}` with only the final
extension stripped, so a `_CALIB` suffix survives). It is the only
identifier unique across volumes; the simulated dataset produces a bare
scene basename with no path separator at all, which is a valid stub.

`results_path_stub` does not appear in the metadata document. Ingest derives
it from the file's own location: the path of a metadata file relative to the
ingest root, with the `_metadata.json` suffix removed. This is exact by
construction, because that is precisely how the navigator chose the path it
wrote to.

**The ingest root is the `nav_results_root`, not a free-form path.**
`sd_results_index` resolves its roots the way every consumer does (the
command line, then `config.environment.nav_results_root`, then
`NAV_RESULTS_ROOT`); an operator who ingests a subdirectory of a results
root would produce stubs no consumer's lookup can match, so the root
identity is part of the contract, not a convenience.

`root_url` is the ingested root in **normalized** form, and every consumer
normalizes its own resolved `nav_results_root` the same way before
comparing: `FCPath(root).expanduser().resolve().as_posix()` with any trailing
`/` removed, so a `~`, a `..` or a symbolic link is one spelling of the place
it names. Resolving there is what lets everything downstream stop thinking
about paths: joining a validated key onto a canonical root is one answer for
every caller. A consumer whose normalized root has no completed `ingest_runs` row
(section 2.7) fails with a message naming its root and the roots the index
does contain -- absence of a row must never be read as "nothing was
navigated".

Two derived columns avoid string surgery in queries: `subtree` -- the first
path segment of the stub, or NULL when the stub contains no `/` (the
simulated dataset) -- and `image_number`, computed by the existing
`image_number_from_name` logic at ingest time. `results_path_stub` alone
carries a non-unique index. Every consumer lookup filters on
`root_url = <its normalized root>`; `sd_stats_report` gains `--root`
(repeatable, default: every root the index holds a completed ingest of, and
refused when no index was named to hold it against), since a report
legitimately spans roots where a pipeline lookup never does.

`images` carries two further non-unique indexes, `image_date` and
`instrument`, which the statistics report groups and filters on across a whole
root. They are carried over from the indexes `cli/stats/schema.py` already
declares, so the report does not lose an index in the move.

The child tables key on the same pair, and each additionally declares the tuple
that identifies one of its rows unique within an image:
`(root_url, results_path_stub, technique_name)` on `techniques` and
`(root_url, results_path_stub, feature_type, source_model, source_name)` on
`feature_sources`. A technique reports once for an image and the feature
inventory is aggregated by exactly that tuple, so a second row is a
contradiction rather than more detail, and a retried or duplicated ingest that
inserted one would change every count and average read from the table without
replacing anything. Each constraint's index leads with the image key, so it is
also the index a lookup by image uses and no separate one is declared.

### 2.3 Columns

The `images` table:

| Column | Source in the document | Notes |
|---|---|---|
| `root_url`, `results_path_stub` | derived (section 2.2) | primary key, NOT NULL |
| `subtree` | derived (section 2.2) | NULL for a stub with no separator |
| `image_name`, `instrument` | `observation.*` | NOT NULL; ingest rejects a document lacking either |
| `camera` | `observation.camera` | present whenever the dataset index supplied it, including for an image that never loaded; NULL only when no camera column exists for the dataset |
| `image_path` | `observation.image_path` | |
| `image_et` | `navigation_result.provenance.image_et` | the observation's midtime, aggregated into the time span the report gives per instrument; a date bound compares `image_date`, not this column; NULL for an image that never loaded, which built no observation |
| `image_date` | the UTC date rendered from `image_et` | NULL wherever the epoch is |
| `status` | top-level `status` | NOT NULL; `'unknown'` when the document names no outcome, never a value taken from another field |
| `status_error` | top-level `status_error` | stored **verbatim and separately** from `status_reason` |
| `status_reason` | `navigation_result.status_reason` | the navigator's vocabulary, distinct from `status_error` |
| `offset_dv`, `offset_du` | **top-level `offset`**, unrounded | the authoritative number every consumer applies |
| `sigma_dv`, `sigma_du` | `navigation_result.sigma_px` | |
| `covariance_px2` | `navigation_result.covariance_px2` | the matrix whole, square and row-major: a twist-fitted result records 3x3, whose rotation row and column carry the offset-to-rotation cross terms that `sigma_rotation_deg` cannot stand in for. A reader wanting the per-axis sigmas takes the square roots of the diagonal |
| `sigma_along_unobservable_px` | `navigation_result` | |
| `rotation_deg`, `sigma_rotation_deg` | `navigation_result` | present only where twist was fitted |
| `confidence`, `confidence_rank` | top-level `confidence`, `navigation_result.confidence_rank` | |
| `n_techniques`, `excluded_from_consensus` | `navigation_result` | `n_techniques` NOT NULL |
| `image_class`, `noise_sigma` | `navigation_result.image_classifier` | |
| `image_shape_v`, `image_shape_u` | `observation.image_shape` | |
| `run_start`, `run_end`, `elapsed_s` | `timing` | |
| `config_hash`, `git_sha`, `pipeline_run` | `navigation_result.provenance` | |
| `image_number` | derived from `image_name` | replaces the SQL function (section 2.5) |
| `start_et`, `stop_et`, `midtime_et`, `exposure_s` | `navigation_result.times.*` | NULL for a document with no navigation result (see below); `midtime_et` is stored as recorded, never re-derived, because the reader's midtime gate holds it to 1e-6 s |
| `sclk_start`, `sclk_midtime`, `sclk_stop` | `navigation_result.times.*` | TEXT |
| `camera_frame_id`, `ck_frame_id` | `navigation_result.pointing.*` | INTEGER |
| `cmatrix`, `cmatrix_original` | `navigation_result.pointing.*` | JSON (nine floats, row-major); `cmatrix` NULL where the navigation fitted a camera rotation |
| `source_file` | the metadata file's URL | provenance |
| `mtime_ns`, `size_bytes` | the metadata file's listing entry | incremental ingest (section 2.7) |

The `times` and `pointing` columns are the corrected-attitude fields the
navigator records (`plans/CK_KERNEL_PLAN.md` section 2.3), and they carry
that section's names and shapes exactly. They are absent from a document
that has no navigation result at all -- an image that failed to load -- and
`cmatrix` is additionally absent where the navigation fitted a camera
rotation, so both cases ingest as NULL.

Types: every pixel, ET and sigma column is declared
`sqlalchemy.Double` (not bare `Float`, which a dialect may map to single
precision); booleans are `sqlalchemy.Boolean`; the covariance matrices, the two
C-matrices, `excluded_from_consensus`, `spice_kernels` and the child tables'
`source_names` / `diagnostics` are `sqlalchemy.JSON`. Which of the two JSON
declarations a column takes depends on what it holds: one that can hold a number
takes the plain declaration -- SQLite TEXT, PostgreSQL `json` -- whose value
travels as the text the driver wrote, so every number in it reads back as the
number written; one holding text alone takes the PostgreSQL `jsonb` variant,
whose array and object accessors a direct-SQL query reaches inside without a
cast and whose `numeric` number type would return a stored `-0.0` as `0.0` and a
large-magnitude float as an integer. `mtime_ns`, `size_bytes` and `image_number`
are `sqlalchemy.BigInteger`: a nanosecond epoch is far past the 32-bit range a
dialect is free to give a plain `Integer`, and an image-numbering scheme is
free to run past it too.

**Absent is not empty, in every JSON column.** The JSON type is declared
`none_as_null=True` (and its PostgreSQL variant likewise), so a Python `None`
is stored as SQL NULL rather than as the JSON value `null`. Without that,
`WHERE cmatrix IS NOT NULL` matches every row ever written and the CSV export
carries the literal text `null` in the cell. Which columns are ever absent is
a property of the mapping rather than of the type: `cmatrix`,
`cmatrix_original` and both `covariance_px2` columns are, and are NULL.
`excluded_from_consensus`, `source_names` and `diagnostics` are not -- an empty
list or object there is a statement (nothing was excluded, the technique named
no source, it reported no diagnostics), and is stored as the empty container.

**Precision.** The top-level `offset` is stored as written, with no
rounding. (It is written unrounded by `navigate_image_files`; the rounded
`navigation_result.offset_px` is the curated display value and is not what
the index stores.) The acceptance test asserts a bit-exact round trip on a
value with 15 significant digits.

**`status_error` and `status_reason` are different vocabularies** and are
stored in different columns. `status_error` takes values like
`missing_spice_data` and is what the SPICE-error selection filter matches
verbatim; `status_reason` is the navigator's explanation of a non-success
outcome. The current ingest merges them into one column; the report queries
that read the merged column are rewritten in the same phase (section 4,
Phase 2) as `COALESCE(status_reason, status_error)`, which reproduces
today's report exactly.

The `techniques` table carries one row per technique that reported: its
`offset_dv` / `offset_du`, its `covariance_px2` whole (a reader wanting the
per-axis sigmas takes the square roots of the diagonal, and no pair of sigmas
carries the correlation between the axes), its `confidence`, its `spurious` and
`at_edge` flags, the `source_names` parsed out of its feature ids, and its
`diagnostics`. The `feature_sources` table carries `feature_type`,
`source_model`, `source_name` and the `n_features` / `n_gated` counts. Both are
keyed on `(root_url, results_path_stub)` and constrained to one row per logical
key (section 2.2). The feature inventory is aggregated by `(feature_type,
source_model, source_name)`; per-feature `feature_id` and `gated` detail is not
retained.

### 2.4 Schema version, creation, and no migrations

The database carries a `schema_meta` table with a single row holding
`schema_version` (an integer) and `created_utc`. The single row is enforced by
a constant primary key (`singleton`) with a `CHECK (singleton = 1)`
constraint, so a second row is a database error rather than an ambiguity the
version gate has to resolve.

Every failure `open_index` reports is a `ValueError` carrying the URL, so a
consumer that wants to report the cause rather than crash catches one type.
The exceptions a database driver raises are translated into that one --
an unparseable URL, an unknown URL scheme, an absent driver, and the ordinary
operational failures of a server that will not accept the connection -- each
keeping the driver's own exception as its `__cause__`. Without the
translation the most common operational failure of all, a PostgreSQL server
that is down or misconfigured, would reach a consumer as a SQLAlchemy
traceback naming neither the URL nor which of the three resolution levels
supplied it.

The translation is a catch-all rather than a list of expected types. A
dialect coerces its own connect arguments and reports a bad one as a bare
`ValueError` naming nothing: a non-numeric port, a non-numeric `timeout`. To
a caller, such an escape is indistinguishable from the guarantee being
broken, so everything that escapes the builder is translated, and only the
messages this layer writes itself pass through untouched.

**The URL a message names is rendered with its credentials masked.** These
messages are written to run logs and pasted into bug reports, and a database
password belongs in neither. Everything else about the URL survives, because
naming the URL is what tells a reader which of the three resolution levels
supplied the value.

The rule lives in `spindoctor/results_index/masking.py`, beside the opener
rather than inside it, because it has a second caller: a run log records the
command line it was given, and one of those words may be an index URL, so
`support/command_line.py` masks the arguments through the same rule the opener
masks its messages through.

One structural rule does this for every URL, not only for the ones the parser
rejects. A parsed URL renders itself with its password hidden but renders a
`?password=` query parameter verbatim, and that parameter authenticates exactly
as the authority form does; adopting the parser for the URLs it accepts would
adopt its blind spot with it. The rule is stated rather than pattern-matched:

1. If the scheme -- the text before the first `:`, with any `+driver` suffix
   and surrounding space removed -- is `sqlite`, the string is returned
   unchanged. A SQLite URL names a local path, which has no credentials and is
   free to carry the colons, at-signs and question marks that would otherwise
   read as some.
2. The authority begins after the scheme's `:` **only when a `/` follows that
   colon**, and then after **every** slash of that run, however many there are.
   Without that slash, `postgresql:svc:pw@host/db` and `svc:pw@host/db` are the
   same shape, and reading the leading word as a scheme leaves the second one's
   password visible; it is read as a user name in both, which hides the first
   one's user name along with its password. That is one word of a message about
   a URL no driver would have accepted; the other reading loses a working
   password. One slash and two are what a hand-edited setting arrives as, three
   is `postgresql:///db` omitting the host to name a local socket, and a rule
   that stopped counting at two left the authority start on a slash -- which
   reads as a path beginning before any password and returns such a URL whole.
3. Within the authority the user name runs to the first `:`; only a `:`
   introduces a password. The credentials end at the **last** `@` in the
   string. A user name is free to carry one -- `user@servername` is the login
   form of a managed server -- and so is a password: `p@ssword`, `p@ss/word`
   and `pw:with@both` are all things an operator types. Every narrower bound
   stops inside a password that carries the character it stops at: ending at
   the `@` before the first `/` leaves the tail of `p@ss/word` in the message,
   and ending at the last `@` before a `#` leaves the tail of `pw@part#rest`.
   The last `@` is the only bound that cannot stop early, because the span it
   produces contains every other candidate span. What it costs is over-masking
   a URL whose credentials end sooner and whose tail carries an `@` -- a
   fragment such as `...?password=x#note@host` is masked to its last character
   -- which is a mangled message about a URL no driver accepts, against a
   working password in a run log. If the first `/` precedes the first `:`, the
   authority ended before any colon and there is no password.
4. `host:5432/path@name` is genuinely ambiguous -- a host with a port and a
   path, or a user name with a password that carries a slash. It is read as
   credentials, which is how the URL parser itself reads it: for the spelling
   of that shape a parser accepts, this rule and `render_as_string()` hide the
   same characters, and a test asserts that agreement over a hand-picked set of
   URLs both can read, which is the only comparison available and cannot reach
   the unparseable shapes where this rule is the sole defense. Reading it as a
   port instead -- deciding by whether the text before the slash is digits --
   leaves a password that opens with digits, `123/secret`, visible in full,
   which is the failure this rule exists to prevent; the cost of the reading
   taken is a mangled host and database name in a message about a URL that was
   already unusable.
5. A query parameter whose name carries `password`, `passwd`, `pwd`, `secret`,
   `token` or `credential` has its value replaced. A driver accepts any
   parameter its library knows, so the name is matched by what it contains
   rather than against a fixed list: over-hiding a setting whose name says
   credential costs a word of a message, and under-hiding one puts a working
   password in a run log. One parameter is separated from the next by `&` or by
   `;`, both of which libpq and the drivers built on it accept, so both are
   split on and each is put back as it was written.
6. Only credential material is replaced. Nothing outside it is touched.

The rule is asked about a **corpus** rather than a list of remembered shapes:
every combination of scheme, of the slashes after it, of credentials, and of
what a password may contain, asserted in both directions -- the secret is gone,
and the result is exactly the URL with its credentials replaced.

Each dimension of that corpus is **covered rather than sampled**, which is not
the same thing and is where the leaks kept living. A corpus carrying no slash,
one and two stopped one value short of the three-slash spelling; a corpus
varying one special character of a password at a time could not reach a
password carrying an at-sign *and* a slash. So the slash count runs from none
to four, and the passwords carry every ordered pair of the characters that mean
something to a URL -- order included, since `p@ss/word` and `pw/part@rest` stop
a rule reading by eye at different places. Tests assert that the dimensions are
still crossed, because a corpus quietly narrowed proves less than it says.

The rule is a named function of the Core layer rather than a private helper of
the opener, because a run log records the command line a program was given and
one of those words can be a connection URL. Which words those are is decided by
`masked_command_line` in `spindoctor/support/command_line.py`, which names the
connection-URL options in one place and applies the rule to each of their
values. `log_run_environment` masks every command line it records through it, so
`sd_offset`, `sd_consolidate_metadata`, `sd_mosaic`, `sd_create_ck` and the
per-image log of `navigate_image_files` are covered by it; `sd_results_index`,
which logs its arguments itself, calls the same function. Every spelling
argparse accepts is masked: the value as a separate word, the value joined to
the option by `=`, and either of those under a distinguishing abbreviation of
the option's name.

**A results root is never masked.** It is not a connection URL, it has no
credentials to hide, and it is the one string an operator reads a run log to
correct, so the ambiguity rule 4 accepts would corrupt it for nothing. Every
message that names both -- `require_ingested_roots`'s refusal above all -- masks
the index URL and prints the roots as they were given.

The engine factory is the only opener:

```text
open_index(url: str, *, create: bool = False) -> Engine
```

- `create=True` -- used only by `sd_results_index`'s interactive path and the
  cloud-task enqueuer -- creates the tables and the `schema_meta` row when
  they are absent, and otherwise validates the version.
- `create=False` -- every consumer -- raises when the database or its
  `schema_meta` row does not exist, with a message saying to run
  `sd_results_index` first. A consumer pointed at a nonexistent SQLite path
  fails; it does not create an empty database, which is a deliberate change
  from the current opener's silent-create behavior.
- Either way, a `schema_version` that does not match the version the code
  was built for raises, naming both versions and instructing the reader to
  empty the database with `sd_results_index drop` and re-ingest
  (section 2.11). The stamped version is read and checked
  before anything is written, so a refused `create=True` open creates no table
  and writes no row; creating this version's tables inside a database
  stamped with another version would leave a mixture no single version number
  describes. (The database file is not literally untouched: the connect-time
  journal-mode selection rewrites a rollback-journal database's header before
  the gate is reached. What the gate guarantees is the schema and the rows,
  which is what a rebuild would otherwise have to undo.)

There are no migrations. Ingest is cheap relative to navigation and entirely
reproducible from the tree, so rebuilding is always available and always
correct. The gate replaces a column-set comparison, which detects a changed
column set but not a changed meaning of an unchanged column.

**Development-stage decision: `SCHEMA_VERSION` and `COLUMN_SET_VERSION` are held at 1 for the duration of the record-source work.** No index is carried from one column set to the next while that work is under way -- there is none in use, and nothing on these branches builds one it then keeps -- so a column change does not raise the number and no one drops and re-ingests anything. The number resumes incrementing on a column-set change once the work reaches `main`, where the index is in real use and the stamp does its normal job.

This pin is recorded here and nowhere else. The user guide and the dev guide describe the finished system: an index carries a schema version, an index whose stamp is not the version the code reads is refused, there are no migrations, rebuilding is the remedy, and a column change raises the number. A guide that narrated the pin would be describing a state its readers will not be in.

### 2.5 Backend selection

All database access goes through **SQLAlchemy Core** -- not the ORM. The
tables are flat and the queries are explicit; an object model would add a
layer without removing one. Existing report SQL moves to `sqlalchemy.text()`
with named binds: the `where_clause` helper's contract changes from
`(fragment with ?, list)` to `(fragment with :name binds, dict)`, and every
call site follows.

The backend is chosen by connection URL:

```text
sqlite:////data/nav-results/index.sqlite3
postgresql+psycopg://user@host/spindoctor
```

A `sqlite:` URL names a **local filesystem path** and is the one path in
this system that does not go through `FCPath` -- SQLAlchemy opens it with
the C library directly. A SQLite URL whose path is on a filesystem that
cannot honor its locking (probed at open with a `BEGIN IMMEDIATE` /
rollback) is refused, naming PostgreSQL as the cross-machine option.

That URL carries **no query string**. The driver derives the filename it
opens from the whole URL, so `.../index.sqlite3?mode=ro` passes an existence
check on `index.sqlite3` and then creates a second file whose name contains
the query -- neither the database the operator named nor an index anything
can read. Such a URL is refused, naming the URL and saying that a SQLite
index URL is a plain local path.

**A read-only database is a separate case from a locking failure**, and the
two are told apart rather than merged. A read-only mount honors locking
perfectly and a consumer cannot corrupt anything, so refusing one and
blaming its filesystem is a false diagnosis of a deployment the index is
meant to support -- an archived copy, or a file shipped to workers on
read-only media.

**Whether ingest can write is asked of the filesystem, not of SQLite**:
`os.access` on the file and on its parent directory, before the engine is
built. SQLite does not answer the question at open. A write-ahead-logged
database -- the shape this opener always leaves behind, having selected that
journal mode -- accepts both the journal-mode selection, which its header
already records, and the write lock `BEGIN IMMEDIATE` takes, on a file it
will never write; it refuses only the first real write, in the middle of an
ingest. A rollback-journal database refuses the journal-mode selection
instead, so no single SQLite answer covers both shapes. The directory is
asked about with the file, because SQLite writes the write-ahead log and its
shared-memory index beside the database, and a writable file in a directory
that permits nothing is still a database ingest cannot write. So:

- `create=False` **accepts** a read-only database and reads it.
- `create=True` asks the filesystem first, and a database or directory that
  `os.access` reports unwritable is refused there, before an engine is built,
  with a message naming read-only -- of the file, or of its directory -- as the
  cause and saying to ingest a writable copy, not the filesystem-locking
  message. A path the filesystem permits still passes through the engine build
  and the open probe, and a refusal from either is diagnosed by what the driver
  said, including the `SQLITE_READONLY*` a rollback-journal database gives to
  the journal-mode selection.
- A genuine `SQLITE_BUSY` or `SQLITE_IOERR` refuses in both modes, with the
  filesystem-and-PostgreSQL message. A refusal carrying no result code at all
  is neither, and is reported as what SQLite said with no remedy prescribed for
  it: the locking remedy is a deployment rebuild, and answering an
  unclassifiable failure with one is a false diagnosis.

The connect-time journal-mode selection still has to tolerate a refusal (any
`SQLITE_READONLY*` result code), because every connection to a read-only
rollback-journal database would otherwise fail; the refusal is ignored rather
than raised, and nothing is inferred from its absence.

**A probe failure is classified by SQLite's own result code before a message
is chosen.** One exception type covers several unrelated causes, and only one
of them is a reason to move the index to a server. `SQLITE_NOTADB` says the
file is not a SQLite database. `SQLITE_CANTOPEN` names the path and says
which of its causes applies: a directory that does not exist, a path that is
not a file, or a file this user cannot open -- prescribing PostgreSQL for an
operator whose results directory has not been created yet is a wrong remedy
for the most common first-run error there is. Only `SQLITE_BUSY`,
`SQLITE_IOERR` and a code that names no cause at all keep the
filesystem-and-PostgreSQL message. Every other code -- a full disk, a corrupt
file -- is reported as what SQLite said, naming the path and the code and
prescribing nothing, because inventing a remedy for an unclassified failure is
exactly how the wrong one gets prescribed. Codes are matched by prefix, since
SQLite refines several of them into extended forms naming the same cause more
precisely.

One read-only database cannot be read at all: SQLite reads a write-ahead-logged
database through a shared-memory index it creates beside the file, so a
write-ahead-logged copy in a directory that permits no writes is unreadable
by construction. That is refused with a message saying exactly that and to
copy the file somewhere writable, rather than letting a consumer's first
query fail with a write error on a read.

`sqlalchemy` becomes a runtime dependency in `[project] dependencies`. The
PostgreSQL driver ships as an optional extra, `rms-spindoctor[postgres]`,
declaring `psycopg[binary]`; a PostgreSQL URL with the driver absent must
fail with a message naming the extra to install, not with an import
traceback. SQLAlchemy's own internal use of stdlib `logging` is left
untouched and unconfigured; nothing in `spindoctor.results_index` may call
`logging.getLogger` or enable engine echo.

Constructs that are SQLite-only and must not survive into the Core layer:

- **`image_number` as a Python UDF.** `register_image_number_function`
  (defined in `report_common.py`, registered by `report.py`) installs a
  Python callable that queries then call in `WHERE` clauses. It becomes the
  ingested `image_number` column; the queries compare against the column.
- **`TOTAL(...)`** (in `report_sections.py`) becomes
  `COALESCE(SUM(...), 0)`.
- **Integer arithmetic on booleans.** `t.spurious = 0` becomes
  `NOT t.spurious`; `SUM(1 - t.spurious)` becomes
  `SUM(CASE WHEN t.spurious THEN 0 ELSE 1 END)`. Both current spellings are
  type errors under a native PostgreSQL boolean.
- **`executescript` and `PRAGMA`.** DDL is emitted by SQLAlchemy metadata.
  `PRAGMA foreign_keys = ON`, WAL, and `busy_timeout` become SQLite-dialect
  connect-time events.

### 2.6 Configuration and the command-line surface

A new `environment.results_index_db` configuration key holds the connection URL.
Resolution follows the pattern the other roots use, in
`spindoctor/config/config_helper.py`: the command-line value, then
`config.environment.results_index_db`, then the `NAV_RESULTS_INDEX_DB` environment
variable. Absence is not an error -- it means "no index", the default mode
of every program. Add `get_results_index_db_url(arguments, config) -> str | None`
beside the existing getters.

Every consuming program accepts `--results-index-db URL`, and also
`--results-index-db none`: the literal sentinel `none` resolves to no index,
overriding the configuration key and the environment variable. Without an
explicit opt-out, an exported `NAV_RESULTS_INDEX_DB` would make file-mode runs
impossible on that machine. The sentinel is recognized at whichever level
supplied the value, so a configuration file or an exported variable can opt out
the same way; the spaces around it are not part of it, and it is otherwise
matched as the exact string, so a URL that merely contains the word is still a
URL.

A value that is empty, or nothing but spaces, is refused with a `ValueError`,
which is the family a bad configuration value raises in here. It does not fall
through to the next level and it is not answered as "no index": `none` is the
deliberate spelling of that and is honored at all three levels, so a machine that
means to run without an index already has a way to say so, and an empty value is
a typo, a script that computed nothing, or a variable left half unset. A warning
would be effectively silence in a batch log, and the run would then read a
different source than the operator believes for as long as it lasts, which on a
cloud root is hours. Failing every run on a machine configured that way is the
desired outcome, not a hazard: one `unset` fixes it, and it is found on the first
run rather than after a long batch has quietly read the tree. The refusal names
the level that carries the value -- `--results-index-db`, the `environment.results_index_db`
configuration variable, or `NAV_RESULTS_INDEX_DB` -- and says to write `none` for no
index or a connection URL for one, which is what makes it a one-line fix.

What the refusal does *not* say is what this run would otherwise have done,
because one resolver serves the programs that read the tree -- `sd_offset` and
the report among them -- and `sd_results_index`, which writes an index and has
nowhere to put its rows without one. Each caller reports the refusal the way it
reports every other misconfiguration: the dataset layer re-raises it as
`SelectionError` so `sd_offset` prints it and exits 1; `sd_stats_report` prints
it on the stream its other refusals print to and returns 1; `sd_results_index`
logs it fatal and exits 1; the three cloud-task workers return it as
`unusable_results_index_db` with the message attached; and `sd_backplanes`, `sd_mosaic`
and `sd_create_ck` let it out of `main` exactly as they let out an index that
will not open and a malformed time bound.

The codebase convention for an argument of this kind is that each program
defines its own, as it does for the results roots: the reprojection family
shares `add_common_env_args` in `spindoctor/cli/reproj/args.py`, and that is
the only grouping.

**Declaring the option is what makes a program index-backed, and nothing else
is.** This is the rule that lets section 1 and this section both hold. Section 1
puts bundle generation and `sd_consolidate_metadata` out of scope and says both
keep reading files; this section says every consuming program accepts
`--results-index-db`. A program that inherited a resolved URL from the configuration
or the environment would satisfy neither: the out-of-scope programs would stop
reading files on a machine that exports `NAV_RESULTS_INDEX_DB`, and they would have no
command line to say no on. So resolution is gated on the declaration. A program
that declares the option resolves a URL through the three levels above, in that
order; a program that does not declare it resolves nothing, whatever the machine
exports, and passes no URL at all. A shared enumerator therefore reads the
option from the arguments it was handed rather than resolving one for every
caller, and a library caller that names no index gets none.

A program that resolves a URL and cannot open it fails immediately with that
error; it does not silently fall back to reading files. Falling back would
turn a misconfigured run into a slow, silently different one.

The `environment` section joins `HASH_EXCLUDED_SECTIONS` in
`spindoctor/config/config.py`, alongside `logging`: it holds deployment
locations (roots, and now a database URL), none of which can change a
navigation result, and a provenance hash that shifts when an operator moves
a results directory answers its question wrongly. This changes the stamped
`config_hash` once for operators who set `environment` keys in
configuration files; the change is provenance-only and is called out in the
PR body.

### 2.7 Ingest

`sd_results_index` walks each root for `*_metadata.json`, reads each
document, and upserts its rows.

**One walk feeds everything.** The recursive listing collects the
`*_metadata.json` names in a single pass -- the seam's walk, which every reader
of a results tree shares -- and each metadata file's `mtime_ns` and
`size_bytes` come from the same listing entries. Every other file the listing
reports -- the `*_summary.png` beside each document, and
whatever else an operator has left under the root -- is passed over and enters
no tally: not the count of files seen, not the count of directories the walk
missed. There is no per-file `stat` call and no per-file `exists` call; a
backend whose listing does not supply size and mtime degrades to `--force`
behavior for that root, with a logged warning.

**Incremental.** A file whose `(mtime_ns, size_bytes)` matches the stored pair
is skipped without being read, and those two metrics are the whole of the
comparison, for a refused file exactly as for an ingested one. `--force`
re-reads everything.

**A refused file is bookkeeping, not a row.** A file that is not a
current-schema navigation document is recorded in a `failed_files` table --
`root_url`, `results_path_stub`, `reason`, `subtree`, `mtime_ns`, `size_bytes`
-- and is skipped on the next pass on the same evidence as an ingested one, so
a tree whose non-navigation files outnumber its results does not pay to
download and parse every one of them on every run. It is a table of its own
rather than a marked `images` row: absence of an `images` row is what every
consumer reads as "this image was never navigated", and a file with no usable
data must leave that answer alone. The one column beyond the bookkeeping is the
one fact the walk knows about a file whatever the file turned out to contain:
which subtree it lies under. A selection filter asks about the file rather than
about its contents, so a refused document answers that exactly as an ingested
one does, and it has to be a column because otherwise a one-subtree enumeration
fetches every refusal in the root. `--force` re-reads a refused file too. A
document that ingested on an earlier pass and no longer reads has its `images`
row deleted as the refusal is written, since a row nothing backs would answer
for an image nothing produced; and a file that was refused and now reads has
its refusal deleted as its rows are written.

**What a pass refused is not what the root holds refusals for.** A refused file is recorded and then skipped unchanged by every pass after it, so the pass's own tally counts it once and never again, and a second pass over the same tree refuses nothing and tallies nothing. That is what the tally means and it is reported as such: what this pass read. `--force` is what puts the reasons and the example files back into it, and the root's whole set of refusals is one query over `failed_files` away at any time. No pass reports a standing count of them, because there is nothing about them for a consumer to act on: a file of either refusal family yields no per-image facts, so no report summarizes it and no error filter selects its image, whichever storage answers. The reason says what to fix rather than what a consumer will be short by.

**A root the walk cannot list is not an empty root.** The walk reports whether
the root itself could be listed. When it could not -- a mistyped root, an
unmounted share -- the run's `ingest_runs` row keeps its NULL finish time, so
every consumer treats the root as one nobody has ingested rather than one that
holds nothing, and no row of it is touched. A root that exists and is genuinely
empty completes normally.

**Every way a directory refuses to be listed is one way.** The walk treats any
`OSError` as "could not be listed": not there, not a directory, unreadable by
this user, a share that stopped answering. A permission error is the commonest
of them on a shared tree, and telling them apart would only decide which of them
ends the pass, since all of them mean the walk can see no result file there,
which is not the same as there being none.

**A directory under a root that will not list ends the pass.** The walk raises
where it meets the directory; nothing catches it before the driver's console
entry point, which reports the directory as a fatal error and exits 1. The root
it was under keeps its NULL finish time, every root named after it on the same
command line is left without a run at all, and both are therefore roots every
consumer refuses. Absence of an `images` row is the load-bearing claim of the
whole design -- every consumer reads it as "this image was never navigated" --
and under a directory nobody enumerated that reading is simply false; a pass
that completed around one stamped the root as ingested and made that reading an
answer, permanently, since a pass with no evidence about the stubs it did not
see must also remove no row. Every completed run is now a run that listed its
whole root, which is what lets the prune act on every one of them.

It is raised **at discovery, in the walk**, and not where the prune would
otherwise be skipped. The walk happens before any document is read, so a pass
stopped there throws away a listing; a pass stopped at prune time throws away
every retrieval of an archive-scale root, which is hours of work to reach the
same conclusion. The cost of the rule is that a transient failure -- a share
that stops answering for a moment, a permission fixed a minute later -- ends a
run instead of degrading it. That is the trade, taken deliberately: an ingest is
reproducible from the tree and cheap to repeat, and the answer the alternative
leaves behind is one no later pass corrects.

**A directory already walked is not walked again, and is not a gap.** The walk
records each local directory's device and inode as it enters it and skips one it
has already listed. A link from a subdirectory back to an ancestor otherwise
writes the same document under a new stub at every level, until the filesystem's
own limit on link traversal stops it -- forty-one rows for one document, forty
of them answering for images at paths no consumer will ask about, and the count
of navigated images wrong by all of them. Declining the second path is not the
same as failing to reach it: every document under that directory is in the
listing already, under the path the walk met first, and the stubs the second
path would produce name a directory no consumer's lookup spells. So the walk
logs it and goes on, the root counts as wholly listed, and the pass completes
and prunes. The identity is taken only for a local directory: a cloud location
has no links to go round in, and asking a bucket about a prefix is a paid round
trip per directory per run.

**Rows of documents that have left the tree are removed.** Presence has to mean
what absence means, so the stubs recorded for a root that this walk did not
find are deleted, and the count is reported and recorded in `ingest_runs` as
`files_removed`. The delete cascades to the child tables. This is sound only on
the evidence of a **listing of the whole root**: the prune reads the walk's own
listing and refuses one of a root the walk could not list. A listing missing a
directory under the root never reaches it, because such a walk raises instead of
returning. Section 2.8 states the consequence for a worker that covers a share
of a root.

**`--no-prune` keeps them, and is documented as a correctness relaxation rather than a speed feature.** Presence stops implying the document is still there: a row outlives its document, so a consumer asking whether an image has been navigated is answered yes for one whose result the tree no longer holds, and `--has-offset-file` hands such an image to a downstream stage. Absence is untouched, because skipping a delete adds no row -- `--has-no-offset-file`, "this image was never navigated" and `require_ingested_roots` all keep exactly the meaning they had. That is what makes the flag offerable at all, and it holds only because the ingest is whole-root: there is no subtree ingest, so a pass that reaches the prune listed everything.

**What it saves is a query and the deletes, never the walk, and which of those depends on the mode.** The rows the index already holds for a root have two readers -- the skip rule in `_files_to_read` and the prune -- and one predicate, `_reads_recorded_rows`, decides whether to run the query on the strength of either wanting it. So an ordinary pass under `--no-prune` still runs it for the skip rule and stops only under `--force` as well; a fan-out, whose workers skip from the metrics their own shares carry, reads it for the prune alone and therefore stops running it either way. The closing summary says the pass left those rows in place, in the position the count of removals occupies otherwise, so a run log read later says which guarantee the index under it was built with. The flag belongs to `ingest` and `divide` alone: `complete` removes no row, and `drop` reads no tree; nothing new is stored for it, because a field with no reader is not stored.

**Per-root bookkeeping.** An `ingest_runs` table records, per root:
`root_url`, `started_utc`, `finished_utc` (NULL while running),
`files_seen` / `files_ingested` / `files_skipped` / `files_failed` /
`files_removed`, and `schema_version`, under a surrogate
`run_id` primary key
(a root legitimately has many runs, and a consumer reads the newest). The row
is written at
start and updated at completion, in both the interactive and cloud paths. A
consumer treats a root whose newest row has `finished_utc IS NULL` -- or no
row at all -- as not ingested, and fails with a message saying so.

**The normalized root is what is walked.** `normalize_root_url` renders the
root absolute once, and the walk and the retrievals are handed that rendering
rather than the string as typed. A relative root is a documented spelling of
the option, and the storage layer refuses a relative local URL outright; walking
the typed form would also record a relative `source_file` beside an absolute
`root_url` for the same file.

**Batched retrieval.** Metadata files are retrieved in batches through
`FCPath.retrieve()` with `exception_on_fail=False`. The current code reads via
`get_local_path()`, which does not download on a cloud root and must be
replaced. Batch and chunk sizes are constants, independent of one another:
`RETRIEVE_BATCH_SIZE = 64` in `spindoctor/nav_records/tree.py`, so that the
seam and the ingest retrieve in the same groups, and
`INGEST_COMMIT_CHUNK_SIZE = 512` in the ingest, which is the only thing that
writes.

**Chunked transactions.** Commit every `INGEST_COMMIT_CHUNK_SIZE` images.
A crash costs one chunk, and concurrent writers are not serialized behind a
run-length lock.

**Per-image atomicity.** The delete-then-insert upsert for one image and its
child rows happens inside one transaction, so concurrent workers can never
interleave halves of one image's rows.

**A row the database refuses ends the pass, naming the file.** A document can
read cleanly and still not go in -- an identifier too large for a `bigint`, a
value a backend's type will not hold -- and the failure arrives from the driver
at the insert rather than from any check the converter makes. A chunk whose
write fails is rolled back and written again one image at a time, so every
document of it that will go in is kept and a rerun after the fix reads only what
is left; the first one that will not go in raises, and the root's ingest run
keeps its NULL finish time, so every consumer refuses the index rather than
reading the absence of that image's row as an answer. Charging it to the file
and going on was the alternative and is worse than the failure it hides: the
image is then in neither table, which every consumer reads as an image nothing
navigated, and the run stamps itself finished over it. What has to be fixed is
the writer or the column set rather than the file -- a value one backend holds
and another refuses is a property of the schema -- so the message carries the
file and the driver's own sentence. A connection the driver reports as
invalidated ends the pass the same way and for the same reason.

**Names inside a document are checked with its shapes.** A `per_technique` entry
carries a `technique_name` and no two entries carry the same one, because that
name is half the primary key of `techniques`; an entry with none has no identity
and two entries with one name have the same one. A document that breaks either
is refused as a document of another shape. Standing a nameless entry in under a
placeholder name manufactured the collision out of the absence, and numbering
duplicates apart would put a technique nobody ran into the operator's report.

**The driver's exit status says whether the pass completed**, not what it found:
0 when every named root was walked, whatever mix of documents was read, skipped
and refused, and 1 when the run could not complete -- no index or no root
resolvable, the index unopenable, a root that could not be listed, or a
directory under one that could not be. A status
read from a count of ingested documents flips between two passes over one
unchanged tree, since what one pass ingests or refuses the next one skips, and a
scheduled run would then see a failure once and never again. **The driver always
exits rather than raising.** The pass charges every failure it enumerates to one
file or one root; anything still escaping is a failure nobody enumerated, and a
console entry point owes its caller a message and a status for one rather than a
traceback. The roots such a run never reached keep their NULL finish times, so
no consumer reads absence under them as an answer.

### 2.8 Cloud-task ingest

`sd_results_index_cloud_tasks` mirrors the structure of the existing
cloud-task drivers (`spindoctor/cli/sd_backplanes_cloud_tasks.py` is the
reference): a `process_task(task_id, task_data, worker_data)` returning
`(retry, result)`, a `Worker` started from `async_main`, and cloud-task
logging isolation. Like every cloud-task driver, it adds **no** logging
arguments (and is added to the `_CLOUD_TASK_DRIVERS` list in
`tests/spindoctor/cli/test_logging_argument_surface.py`, whose test asserts
exactly that). A cloud ingest worker has no run log and no per-image scope;
its outcome -- counts of ingested, skipped, and failed files, with the
failing files named -- is **returned in the task result**, which the
enqueuer aggregates. The result carries the per-reason tally of section 2.7
beside the names, with one example file per reason, because the enqueuer's
closing summary is the only place a divided ingest can report why files were
refused and a bare count of refusals reads the same whether a tree holds many
documents that were never navigation results or the ingest went wrong.

A task carries the files of its share as the enqueuer's own walk reported them
-- each with its stub, `mtime_ns` and `size_bytes`, plus the run identifier,
the root, `force`, and whether the listing reported metrics at all -- and the
worker ingests exactly those. It stats nothing and checks for nothing: the one
listing of the pass is the enqueuer's, and everything a worker would otherwise
ask the tree travels with the task. The worker applies the incremental skip of
section 2.7 to its own share, against what the index records for its own stubs,
so a retried task reads no document at all and costs a lookup over its own
stubs instead. An entry carrying neither metric is read rather than compared,
whatever the task claims about its listing: that claim is the enqueuer's and
travels beside the entries rather than on them, so the two can disagree.

**A worker never prunes.** Removing the rows of documents that have left the
tree (section 2.7) is licensed by a complete listing of the root, and a worker
holding a share of one has no evidence about the stubs outside its share:
pruning on it would delete its peers' rows. The prune takes the walk's listing
and raises unless that listing covers the whole root, so the restriction is a
property of the seam rather than a rule a worker has to remember. Nothing hands
a worker a listing at all, so there is nothing for it to offer.

**The enqueuer prunes at fan-out, not at completion.** Fan-out is the one moment
of the pass that holds a complete listing, and it is the listing the shares were
cut from, so the prune is licensed exactly as the single-process one is. It
cannot race the workers either: every stub a worker writes is one the listing
held and the prune deletes only stubs it did not hold, so the two sets are
disjoint by construction however the workers are scheduled. Pruning at
completion instead would mean listing the whole root a second time -- the most
expensive thing an ingest does, and a paid round trip per directory on a cloud
root -- to act on evidence no share ever came from. The window between the two
is not a hazard: the run is unfinished throughout, so no consumer reads the root
either way. `--no-prune` therefore belongs to the fan-out too, and is refused at
completion, which removes nothing; it is also the one mode where the flag saves
the query over the root's rows outright, since the prune is that query's only
reader here.

**Concurrency:**

- **Workers on one machine, SQLite backend.** They open the same local file.
  WAL, `busy_timeout`, and short transactions (section 2.7) make multiple
  local writer processes safe. There is one file, so there is no merge step
  and none must be written.
- **Workers across machines.** A shared SQLite file is not an option. They
  connect to PostgreSQL as ordinary clients. This is the case the backend
  abstraction exists for.

**Schema creation is not per-task.** The enqueuing program opens with
`create=True` before fan-out and writes the `ingest_runs` start row; workers
open with `create=False` and fail if `schema_meta` is absent. Per-task
counts return in task results and are aggregated and written to
`ingest_runs` by the enqueuer at completion; workers never touch
`ingest_runs`.

**The enqueuer is `sd_results_index` in two further subcommands**, since the
fan-out resolves the same roots and the same index URL as the pass it replaces:
`divide --tasks-file` lists, prunes, records what the walk found on each run row,
and writes the shares out; `complete --events-log` reads the `cloud_tasks` event
log, adds the tallies up, and stamps the runs. The two are separate
subcommands. Completion opens with `create=False`: the runs it means to
finish are in the index the fan-out wrote, and creating an empty one would report
every root as never fanned out.

**A run is stamped only when its shares account for the whole listing.** The
fan-out records `files_seen` on the run row, because no worker sees more than a
share; completion sums `files_ingested + files_skipped + files_failed` over that
run's own shares and refuses to stamp a run the sum does not match exactly. A
task that failed, timed out, or was never run read none of its documents, and a
run stamped without them tells every consumer that absence of their rows means
those images were never navigated -- the one claim the run bookkeeping exists to
license. A worker that reported an error rather than a share is likewise a
shortfall, so an unopenable index or a malformed task leaves the root unfinished
rather than silently shrinking it. An account that runs *past* the listing is
refused from the other side of the same rule: each task counts once, so the sum
can only exceed the listing on a report that is not this run's, and a run is not
stamped on an account that cannot be right.

Three rules are what make that sum mean what it says.

**Each task's report counts once**, taken under the `task_id` its event carries,
which the fan-out mints uniquely per share. A queue redelivers a task whenever
it could not see the delivery acknowledged, and an operator re-runs a task file
after a partial failure; a retried task reads nothing and reports its share a
second time, as skipped. Added twice, that report covers for a share that never
ran at all: over- and under-accounting cancel, the sum reaches the number the
walk found, and the run is stamped with its documents unread. The later report
of a task supersedes the earlier one, since a task that failed and was re-run
reports its failure first. A result carrying no task identity cannot be told
from a repeat of another and so counts toward no run.

**A share counts toward a run only when it names that run's root**, which is why
the worker returns the root beside the run identifier and completion compares
both. The identifier is a surrogate that starts again at 1 in a fresh index --
which is exactly what the remedy for a schema-version mismatch produces, and
what a mistyped `--results-index-db` names -- so a task file that outlived the index it
was cut from carries the run number of whatever was built next. Its shares then
add up to that run's listing while their rows sit under a different root, and a
run stamped on them is a root with nothing under it: every consumer reads absence
there as "this image was never navigated". A result naming a run being completed
but another root is counted and reported rather than credited, and is told apart
from one belonging to a fan-out nobody here is completing.

**A run whose listing was never recorded is never stamped**, whatever its shares
say. No files seen is not zero files seen: zero is what a root that was listed
and holds nothing records, and only zero can be accounted for by no shares at
all. A root the walk could not list keeps a run with no `files_seen`, exactly as
section 2.7 requires, and so does a pass that died between starting its run and
listing its root; reading either as zero completes a root nobody ever listed and
hands every consumer a tree of images to read as never navigated.

### 2.9 Consumers

Each consumer gains one code path behind the resolved URL; its existing path
is untouched and remains the default.

**`sd_stats_report`** (with ingest, Phase 2). Moves onto the Core layer and
the versioned schema. Its report must not change: the failure-reason tables
that read the previously merged reason column move to
`COALESCE(status_reason, status_error)`; every join and correlated subquery
(including the four in the CSV export) moves to the composite key; the CSV
export's `images.csv` columns become the section 2.3 set in schema order,
`root_url` through `size_bytes` included, and the user-guide column list is
updated to match. `--db` is **removed** from both statistics programs and
replaced by `--results-index-db` (a URL, not a bare path); the four documented
invocations in `docs/user_guide/user_guide_statistics.rst` change with it,
and that page's "plain SQLite" framing is rewritten around the URL scheme
with each direct-SQL example shown in both dialect spellings where they
differ (`json_each` vs `jsonb_array_elements_text`).

It reads through the record seam rather than the database. One pass of
accumulators over `facts(selection)` answers every section, so the report runs
over a results tree or an index with no query of its own: no SQL, no connection
and no `where_clause` lives in the report modules. Over a tree that means no
database is opened at all; over an index it means one is read, through
`IndexRecordSource`, which is the seam's reader rather than the report's.
`--nav-results-root` and its ladder name the trees to read; `--root` selects
among the roots one index holds, and is refused rather than read as a second
spelling of it. With
no `--root`, an index-backed report covers the roots whose newest ingest run
completed rather than every row `images` happens to hold, since under a
half-ingested root a report reads absence it has no license to read.
`images.csv` is written a row at a time and is therefore unsorted, with
`results_path_stub` as its first column so an operator sorts it in the shell.
A file that yielded no record is counted and the count is printed whether it
is zero or not; it is scoped to the whole of every selected root rather than
to the selection, because a refused file carries no instrument, date or image
number for a filter to compare.

**`sd_backplanes`.** Reads `status`, `status_error`, and the top-level
`offset` for one stub. With an index it reads one row. A stub absent from
the index raises, exactly as a missing metadata file raises in the file
path -- the caller reports it -- with a message naming the stub and the
index URL.

**Where a consumer's records come from.** `spindoctor/cli/reproj/offsets.py`
holds the highest-volume reader in the system. Since the C-matrix reader
switch its classifier, `select_pointing(nav_metadata)`, takes a parsed record
and returns a `PointingSelection` (mechanism plus reason) rather than a bare
offset, and its reason vocabulary includes the pointing-ladder rows
(`no_pointing_block`, `no_cmatrix_rotation_fitted`, `missing_offset_key`,
`malformed_pointing`, the gate reasons, `pool_already_corrected`) beside the
original offset reasons.

The seam is made explicit rather than overloaded, in
`spindoctor/cli/reproj/pointing_source.py`: a `PointingSource` protocol with
two implementations,

```text
FilePointingSource(nav_results_root)        # today's body, unchanged
IndexPointingSource(engine, root_url)       # one SELECT per lookup
```

Each answers two questions -- `read_record(image_file)`, which the backplane
stage asks because it reads the record's status before it decides there is
work to do, and `load_pointing(image_file)`, which returns the existing
`PointingSelection`. `build_pointing_source` chooses between them from the
resolved URL, and drivers pass the source where they pass `nav_results_root`
today (`generate_backplanes_image_files` and the mosaic pass take the source
object in place of the root-plus-implicit-reader they take now).

**The index-backed source classifies nothing itself.** It rebuilds the shape
of the document from the row and calls the same `select_pointing`, so there is
one ladder rather than two that could drift apart. Every field that ladder
reads is a column: `status`, `status_error`, `offset_dv` / `offset_du`, the
`times` block including `midtime_et` (section 2.3), and the `pointing` block's
`cmatrix`, `cmatrix_original`, `camera_frame_id` and `ck_frame_id`. The
`pointing` block's `camera_frame` name is a column that this rebuild does not
read: the frame identity a recorded attitude is gated against is taken from the
observation, so these two readers consult the name nowhere. The kernel writer
does consult it, which is why the column exists.

**Ingest fills those columns through the readers' own functions.** The domain
of every value a consumer classifies a record from lives in
`spindoctor/support/nav_record.py` -- `record_status`, `record_status_error`,
`record_offset`, `record_rotation_matrix` and `finite_float` -- and both the
readers and `facts_from_document` call it. The invariant is that every value a
reader can use is stored, in the form the reader reads it as, and nothing else
is stored, so a
record rebuilt from the columns classifies exactly as its document does. A
second set of rules in the store, even one that agreed the day it was written,
is a second reader of the record: that is how a three-element `offset` came to
be truncated on the way in and refused on the way out, and how a rotation
written as a 3x3 nesting came to be applied through a document and not through
its row.

Two distinctions the rebuild has to preserve are keyed deliberately. A record
that fitted a camera rotation and a record with no pointing block at all both
leave `cmatrix` NULL, so the `pointing` block is emitted whenever *any* of its
four columns is set rather than only when the corrected attitude is: a row with
none of them had no pointing block, and a row missing only the corrected
attitude fitted a rotation. And a NULL offset is rendered as an `offset` key
holding null rather than as an absent key. Which of them it is rendered as
makes no difference to any consumer: ingest stores an absent, null, malformed
and non-finite offset alike as NULL, none of the four supplies a pointing, and
no consumer branches on whether the key is present.

Every other field the row does not carry is rendered as a field the document
did not have, not as one holding null, so the record reads as the document it
came from. `status` is NOT NULL and records a document naming no outcome as
`unknown`, so that value is rendered back as the absent field it stands for; a
document naming that same word for itself is rendered without the field too,
and both are read as naming no outcome by the one function every consumer reads
the field through, so the sentinel colliding with a value a document could
carry changes nothing anybody reports. What makes the rebuild faithful at all
is that ingest stores the document's own top-level `status` and nothing
standing in for it: a column that borrowed the copy inside `navigation_result`
would answer `success` for a document that never said so, and the ladder's
first question is exactly that field.

The reason vocabulary maps as follows, and the mapping table belongs in the
`pointing_source` module docstring:

| File-path reason | Index-path equivalent |
|---|---|
| `no_metadata` | no row for the stub under this root |
| `navigation_did_not_succeed` | row with `status != 'success'` |
| `null_offset` | row with `status = 'success'` and `offset_dv` or `offset_du` NULL |
| `no_pointing_block` | row with no pointing column set |
| `no_cmatrix_rotation_fitted` | row with a pointing column set and `cmatrix` NULL |
| `malformed_pointing` | row carrying a `cmatrix` the validator refuses, or one with no `midtime_et` |
| `pool_already_corrected` and the gate reasons | identical: they are decided when the selection is applied, from the three recorded values both paths carry |
| `unusable_metadata_path` | unreachable: a stub is a key, not a path |
| `unreadable_metadata`, `invalid_json`, `metadata_not_an_object` | unreachable: ingest already refused such a file, so it has no record row and a refusal row instead, and the lookup fails the image rather than classifying it |
| `missing_offset_key`, `invalid_offset_type`, `non_finite_offset`, `malformed_offset` | reported as `null_offset`: one column pair holds all five ways an offset can supply no pair, and none of them supplies a pointing |

**A document the ingest refused is not a record the index can classify at all**, and it must not be read as an image nothing navigated. Ingest writes such a file to `failed_files` and not to `images`, for every reason `facts_from_document` refuses one: no `observation.instrument`, no `image_name`, a declared container of another shape, a duplicated `technique_name`, a file that is not JSON or not an object, and anything else the converter cannot read whole. The document itself is often a perfectly readable navigation record with a status, an offset and a corrected attitude, so a lookup that saw no `images` row and reported `no_metadata` would reproject that image corrected through the tree and uncorrected through the index, and would skip it in `sd_backplanes` while the tree built its product. So the lookup asks `failed_files` whenever it finds no `images` row, and a stub recorded there **fails that image**, naming the stub, the index and the recorded reason. Reading the document instead was considered and rejected: it would make `--results-index-db` mean a different thing per image, and one round trip per image is the cost the index exists to remove. Failing one image does not fail the run -- both consumers contain a per-image failure -- and the remedy the message names is to fix the document and re-ingest, or to run without an index. The one refusal ingest deliberately records nowhere is a file it could not retrieve, and an image whose document failed that way still reads as one nothing navigated; that is the first member of the Phase 5 enumeration, and it is the same fact here.

**The rule the seam is held to: a record the two storages *classify* differently may differ in the reason and in nothing else.** The reason is a name a run-level tally counts under; the mechanism, the matrices, the midtime and the offset are what a product is built from. A difference in any of those is a defect in the reader or in what ingest stores, not an entry for the list. The list itself is derived by measurement rather than by argument: both sources are driven over every shape a record's fields can take -- absent, null, wrong type, over-long, non-finite, boolean, nested, ragged, and an integer too large for a float -- and what survives defines it.

Nothing survives. The navigator's own types settle it: a `success` record carries an offset of two floats, a recorded attitude is validated as a proper rotation of finite numbers before it is written, and a pointing block always carries its baseline and both frame identities as integers. So for every record the navigator wrote and ingest stored, everything a product is built from -- the mechanism, the matrices, the midtime, the offset -- is identical in the two paths, and so is every field the readers report about it, the reason among them. The index path logs the same per-image `IMAGE_LOGGER` warnings, with the same message shapes; where a message names the storage that was searched, it names the one that actually was.

**`ResultsFilter`.** All six filters are answered through the record seam, by two of its questions, so one implementation serves both storages. A listing says which images have a document and costs no document read; a stream of per-image facts says what each document records, which answers the error filters and settles presence along with them. `open_record_source` decides which storage answers: a run naming an index reads rows, a run naming none reads the documents. Because the facts are the values the index holds in its columns, a document is narrowed on exactly what a row is narrowed on, and the two agree by construction rather than by two pieces of code that agree today. The constructor keeps every contradictory-pair rejection. A file the ingest refused is still a file the walk finds, so a listing of the index reads `failed_files` alongside `images`, and that table carries the subtree for the same reason.

**The listing is asked in one of two ways, and which one is decided by what the run selects.** A run selecting images that *have* a document -- `--has-offset-file`, and every error filter, since each of those requires one to exist -- lists the selected subtrees once at construction, so `passes(stub)` is a set lookup and the same listing supplies the count the run log reports. A run selecting images that have *none* -- `--has-no-offset-file`, which contradicts all five of the others and is therefore always alone -- names its candidate images instead, a batch at a time, exactly as the error filters name theirs. It keeps the images the root holds nothing for, so every entry a listing of a whole volume produced would be one to reject from, and a run whose other constraints name ten images would pay for a volume to answer about ten. Such a run applies no subtree restriction and needs none: the candidates come from the volumes the enumeration walked. It reports the age of an index at construction, because that is what an index owes a run whatever it goes on to find, and reports what it asked about and how much of it was already navigated when it is closed.

**An error filter reading the tree hands each kept image the record its own document was read out of, so the enumeration's read and the per-image read are one.** `ImageFacts` carries `record`, filled by `TreeRecordSource`, which holds one by the time it has any facts at all, and left None by `IndexRecordSource`, where a record is rebuilt from the column set its consumer declares rather than from the columns the facts hold. `ResultsFilter.filter_batch` puts it on `ImageFile.nav_record`, and `FilePointingSource`, the backplane stage and `cli/pds4/bundle_data.py` answer from it. Against an index the two reads want different columns and stay separate, which is the settled decision rather than an omission. A second read of the same document would not be a `filecache` cache hit either: the dispatch programs build their results root through `FileCache(None).new_path`, an anonymous cache deleted when the process ends, while the seam reads through the persistent `_filecache_global`, so on a cloud results root a second read is a second download. What travels is bounded by what holds the image: an enumeration yielding one image at a time holds at most one filter batch of records (64), and `--choose-random-images` holds at most the requested sample count plus the remainder of the last batch, at about 40 to 50 KB a record. A run given no error filter reads no document at enumeration time and carries nothing, so nothing here changes what it costs.

**What the index answers differently is what one ingest pass could read and record, never a property of the query.** It is enumerated in section 4's Phase 5 entry, restated as acceptance criterion 1, repeated in the `results_filter` module docstring and stated for an operator in the navigation guide's account of `--results-index-db`; each member has a test of its own, and a member found later is added in all four of those places in one commit. The enumeration-list guard reads the four and compares them; it is not itself one of them. The enumeration is maintained rather than audited closed: it names what is known to differ, and a divergence nobody has found yet is not evidence that none exists. The guide is one of the four because the person a silently short selection is served to reads the guide and not the module: a divergence enumerated only where a maintainer meets it leaves a user with a selection that looks like the one they asked for.

**The filter vocabulary is six flags, and they conjoin.** `--has-offset-file`
and `--has-no-offset-file` ask whether the document exists;
`--has-offset-error`, `--has-no-offset-error`, `--has-offset-spice-error` and
`--has-offset-nonspice-error` ask what it records. Every error filter needs a
document to read, so each one folds presence in, `--has-no-offset-file`
contradicts all four, and `--has-no-offset-error` contradicts the three that
name an error. That makes `--has-offset-file --has-no-offset-error` how a run
asks for "the images this root holds a navigated result for, whatever it
concluded" -- the selection that excludes the frames whose navigation died
before it could read the image, which is a document with a fatal `status` and
no summary picture beside it.

**That pair is not the selection a picture-presence filter made, and the navigation guide says what it really selects.** The two differ in both directions, and neither is a defect of this vocabulary. A `*_summary.png` with no document beside it -- a PDS4 browse product, a document deleted from under its picture -- was a file a picture-presence filter selected, and this pair does not, because it asks a question about a document and there is none. A document no per-image facts come out of is one this pair drops where a picture-presence filter kept it, since the refusal is a fact about reading the document and the picture beside it is untouched by it. No image a navigation run wrote both files for falls in the first gap, since a picture is drawn only beside a document; the second is bounded by what the per-image shape accepts rather than by what a navigator wrote, and on a root whose documents carry less than that shape requires it is most of the root. So the claim the guide makes is the narrow one: the pair selects every image whose document yields per-image facts recording no fatal `status`, and it selects the same images whichever storage answers.

Because the flags conjoin, a selection that is a union of two of them is two
runs rather than one, and one such union is worth naming: "never navigated, or
navigated and errored" is `--has-no-offset-file` and `--has-offset-error` run
separately, and no single flag combination expresses it. The vocabulary is
deliberately a conjunction of questions about one document, not an expression
language over the tree.

`ResultsFilter` lives in `spindoctor.dataset` and imports `open_record_source` at the top of its module like any other import. Keeping SQLAlchemy out of `spindoctor.dataset` was measured and protects nothing: importing that package already loads oops (+1695 modules, 0.790s) where `import sqlalchemy` adds 83 modules and 0.053s, on a path that then does SPICE and image work taking seconds per image, and `sqlalchemy>=2.0` is a hard runtime dependency rather than an optional extra. The database line that is load-bearing runs elsewhere, and section 4's Phase 7 entry states it: `spindoctor.nav_records` imports no database layer, because it is the storage-free half of the seam and every reader of a results tree goes through it whether or not its run can open an index. A subprocess probe pins that one, and the dev guide records it.

### 2.10 Logging

`sd_results_index` gains a program identity and a logger. It becomes
infrastructure other programs depend on, and a partial or failed ingest must
appear in a run log rather than only in an exit code.

Concretely:

- Add `SD_RESULTS_INDEX = 'sd_results_index'` to
  `spindoctor/config/program_names.py` and `PROGRAM_NAMES`; both drivers
  declare `PROGRAM_NAME`.
- The interactive driver adds
  `add_logging_arguments(parser, has_image_logger=False)` (ingest processes
  documents, not images) and wraps its configuration load in
  `reporting_logging_errors()` with the exact
  `with reporting_logging_errors():` / `load_default_and_user_config`
  adjacency -- a source-scanning test asserts that literal pattern. The
  cloud-task driver adds no logging arguments (section 2.8).
- `print()` calls in ingest become `MAIN_LOGGER` calls.
- Collateral that asserts the old exclusion, all updated in the same
  commit: `sd_results_index` moves from `_WITHOUT_LOGGER` to
  `_WITH_ANY_LOGGER` in
  `tests/spindoctor/cli/test_logging_argument_surface.py`; the program
  table in `docs/user_guide/user_guide_logging.rst`; the `program_names.py`
  module docstring; and `.cursor/rules/logging_nav.mdc` section 1, whose
  statistics-programs sentence is replaced so that it names
  `sd_stats_report` alone as the print-only statistics program (the
  license for `sd_stats_report`'s `print()` must survive the edit).

`sd_stats_report` keeps `print()`. Its output *is* terminal text for a human
reading a report; a logger would wrap that in machinery it does not need.
The docstring says so, so the split reads as a decision rather than an
oversight.


### 2.11 Dropping an index

Emptying an index is the counterpart to starting a results tree over, and the
version gate in section 2.4 makes it structural rather than convenient: a
version bump is deliberately not migrated, and the remedy that gate prescribes
is delete-and-re-ingest. With no command behind it that remedy would be `rm` on
SQLite and, on PostgreSQL, `psql` with the right connection string and a
knowledge of which tables SpinDoctor owns in a database that may hold somebody
else's; `sd_results_index drop` is one command for it on either backend.

`sd_results_index drop` removes the index's tables from whatever backend
the URL names and **stops** -- it reads no results root and ingests no
document. Dropping is a deliberate act rather than the first step of a long
pass, and a command that did both would make a mistyped URL expensive twice
over; dropping and re-ingesting in one command is available as two commands. It
resolves no results root at all, so a machine holding the index and not the tree
can still drop.

`--yes` drops without asking. Without it the tables are listed with their row
counts, the schema and the schema version, the question is put to standard
output by `input`, and anything but `y` or `yes` -- compared after the line is
stripped and lower-cased -- leaves the index alone and exits 1. A standard input
that has nothing to read -- at its end, closed, or absent, which is what a
scheduled run has -- is a refusal rather than consent, and the refusal names
`--yes`. Ctrl-C at the question is a refusal too, and prints the line every
other refusal prints rather than a traceback. Every message names the index URL
through `masked_url`.

The Core layer holds the operation, in `spindoctor/results_index/drop.py`:
`index_contents(engine)` reads what of the index a database holds and where, and
`drop_index_tables(engine, contents)` removes exactly that. The confirmation,
the messages and the exit status are the CLI's, in `spindoctor/cli/results_index/drop.py`.

- **A name is not evidence.** The six names are `images`, `techniques`,
  `feature_sources`, `failed_files`, `schema_meta` and `ingest_runs`, which are
  among the commonest table names there are, and a shared PostgreSQL server
  holds databases SpinDoctor did not create. So nothing is dropped from a
  database until that database has proved it holds an index of ours, and the
  proof is the index's own stamp: a `schema_meta` carrying the columns
  `singleton` and `schema_version`. Two columns rather than the whole set,
  because a stamp left by a schema whose columns differed is one of the states
  the drop exists for; two rather than one, because a version column alone is
  what any migration table carries. A database with tables of these names and no
  such stamp is **refused** and its tables are named -- #487's "refuses
  rather than half-completes when the URL names something that is not a
  SpinDoctor index" -- since nothing distinguishes somebody else's `images` from
  the remains of an index whose stamp has gone. The cost is named rather than
  hidden: a creation interrupted before it wrote `schema_meta` is a database the
  drop declines, and its tables go by hand or with the file.
- **The evidence names the schema, and the schema is named in every
  statement.** A server resolves a bare table name through a search path, and
  six bare names resolved one at a time is how a drop comes to span two schemas
  -- destroying a table in the first while leaving the index's own standing in
  the second, committing, and exiting 0. So the stamp is located once (through
  `pg_table_is_visible`, which is the server's own resolution rule) and every
  statement afterwards -- the presence checks, the row counts, the `DROP TABLE`
  -- names that schema explicitly. A table of one of these names in any other
  schema is never reached. On SQLite the schema is `main`, the one namespace a
  database file has, and naming it is what keeps the two backends on one code
  path.
- **The tables come from `METADATA` by name**, one `DROP TABLE` each. Never
  `DROP SCHEMA`, never a wildcard, nothing discovered by pattern. A hand-written
  list would be right on the day it was written; a test reads the statements the
  drop issues and asserts they are exactly those names, each qualified by the
  one schema.
- **The six names in a stamped schema are six tables SpinDoctor created**,
  because an ingest refuses to build an index in a schema holding anything it
  did not create (below). That is what turns "the tables the drop removes" from
  a claim about names into a claim about provenance, and it is what the user
  guide's "nothing else in that schema is touched" rests on.
- **The reading is what is dropped.** `drop_index_tables` takes the
  `IndexContents` the operator was shown rather than reading the database again,
  so the tables that go are the tables that were named in the question. Between
  two readings the answer is free to change, and a destructive command must not
  act on a list nobody saw.
- **`open_database`** is the opener, beside `open_index` in `engine.py`. It
  applies every URL diagnosis, SQLite probe and masking rule `open_index` does
  and stops before the version gate, because a database the gate refuses is
  precisely what a drop is pointed at. It connects all the same: reading the
  stamp is what reaches the server for the other opener, and an engine handed
  back unconnected would turn an absent database or a refused password into
  whatever statement ran first. Nothing is read from the database through it, so
  no column the gate protects is ever touched. The three accesses differ along
  four axes -- creating, writing, must-exist, gated -- and an `_Access` record
  spells each combination out rather than inferring one from another; it carries
  the read-only and absent-path remedies too, and what a failure calls the thing
  it could not open, since a drop is pointed at databases that are not indexes.
- **A database that is not there is refused**, on both backends alike: a
  PostgreSQL database that does not exist is refused by the server, and a
  SQLite path that does not exist gets the same answer rather than being
  created. A database that *is* there and holds none of these tables is not
  written at all and says so, and exits 0: an index already gone is the state
  asked for, and an idempotent drop has to be visibly idempotent. The two
  backends give the same status for the same state of the database, which is
  what "works on both backends from the same flag" means for a teardown script.
- **The transaction carries the guarantee, on both backends.** PostgreSQL rolls
  DDL back with everything else. SQLite's own DDL is transactional as well; what
  is not is its Python driver, which opens a transaction when it sees an INSERT,
  UPDATE or DELETE and for nothing else, so a run of `DROP TABLE` statements
  would otherwise be issued in autocommit and stand one at a time. The drop
  therefore issues `BEGIN IMMEDIATE` itself as its first statement, which also
  takes the write lock before anything has been dropped. Ctrl-C, a table that
  will not drop and a lost connection each leave the database exactly as it was.
- **`schema_meta` is dropped first**, before the tables it stamps. With the
  transaction covering both backends this is the second line rather than the
  first: what it forbids is the one state that must never be reached, a stamp
  standing over tables that have gone, which the gate reads as healthy and
  inside which every consumer's first query fails. The opposite state -- tables
  standing with no stamp -- is not a safe resting place either, since a creating
  open adopts it and the incremental skip then never re-reads the documents
  whose rows survived; that it cannot arise is a property of the transaction,
  not of the order.
- **A dropped index and one that never existed are the same thing to every
  consumer.** Both are "not ingested". There are five opener call sites:
  `cli/stats/report.py`, `sd_results_index.py`, `sd_results_index_cloud_tasks.py`
  and `results_index/record_source.py` through `open_index`, and
  `cli/results_index/drop.py` through `open_database`. `open_index` refuses both naming
  `sd_results_index`, so the report exits 1 on both, the cloud-task worker
  returns `index_unopenable` on both, a completion exits 1 on both, and an
  enumeration -- which reaches the opener through `open_record_source` --
  refuses both; a creating ingest builds one over either. On PostgreSQL they are
  literally the same database, and a test compares the two refusals character
  for character over one URL. On SQLite the
  emptied file remains and the drop deliberately does not delete it, so that one
  flag means one thing on both backends; deleting the file removes the database
  rather than the index, which reads the same to every consumer but which a
  later `sd_results_index drop` refuses rather than reporting as nothing to do.
- **A failure is named as what it was.** A drop can fail for a lock somebody
  holds, a view or another object depending on one of these tables, an account
  that does not own one of them, or a table that went between the reading and
  the drop, and the remedies are different in every case. The CLI reads the
  database's own code -- SQLSTATE from PostgreSQL, the result-code name from
  SQLite -- and answers it from a table; a code not in that table is reported as
  the database worded it, with no cause invented for it. Naming the wrong cause
  in a destructive command's failure sends whoever reads it to grant a privilege
  over a lock.

Two questions the drop answers by reporting rather than by refusing.

**An unfinished ingest run does not stop a drop.** Such a run is either a pass
writing the index at this moment or one that died, and nothing recorded in the
index tells the two apart: there is no heartbeat and no process to ask. A pass
that died is also the commonest reason to want a drop, so refusing on that
evidence would withhold the command from the case that needs it most, to guard a
case the confirmation already guards. The count is therefore named in the
summary, before the question, so the person about to end a live pass is told
while there is still an answer to give. What a drop under a live pass costs is
that pass, which fails on a table that has gone; no reader is affected, since an
unfinished run already reads as "not ingested" before and after. The count is
asked only of a database stamped with this version, because the question is
phrased in a column.

**Another process holding the database does not stop it either.** Neither
backend can be asked honestly: SQLite's readers take no lock to observe under
write-ahead logging, and a PostgreSQL role need not be allowed to read the
server's activity view, so a "nobody is using it" answer would be a guess. What
can be done is to make the attempt fail rather than hang, and to leave nothing
half-finished when it does. `DROP_LOCK_TIMEOUT_MS` (30 s, matching
`SQLITE_BUSY_TIMEOUT_MS`) is issued as `SET LOCAL lock_timeout` -- local, so the
bound belongs to that transaction and does not ride a pooled connection into
whatever runs on it next -- inside the drop's transaction **and inside the
reading that precedes the question**, because counting a table's rows takes a
lock too and an unbounded reading hangs the command before anybody has been
asked anything. On SQLite the busy timeout already bounds the same wait, and
`BEGIN IMMEDIATE` is where it is met. The database itself decides, per table,
instead of a guess deciding beforehand.

**What makes the drop's promise true is a rule on the creating open.** A drop
that will not destroy a table on the strength of its name must not become
willing because a stamp was found beside it, and `MetaData.create_all` defaults
to `checkfirst=True`, so without a rule an ordinary ingest adopts an existing
table of one of these names and stamps the schema over it. From then on nothing
distinguishes that table from one SpinDoctor created. So an ingest never stamps
a schema that already holds tables SpinDoctor did not create. The creating open
resolves one schema -- the one a `schema_meta` of ours was found in, or, where
there is none, the one an unqualified `CREATE TABLE` lands in -- and answers
four ways:

- The schema holds nothing: create the tables and stamp it.
- The schema carries a stamp of SpinDoctor's: it is the index's own whatever
  version that stamp names, and the version gate decides. This is the case
  `drop` exists for, so it must keep working for a stamp of any version;
  the evidence is deliberately the two marks rather than the column set, since a
  stamp of an older version has our tables with other columns.
- The schema holds a table of one of the index's own names with no such stamp:
  refuse. A name is not evidence, and a stamp written beside a stranger's table
  would make it the index's for every later reading.
- The schema holds any table the index does not own, stamped or not: refuse. The
  index and its consumers own the schema they live in, so a foreign table means
  the URL, or the search path behind it, names somewhere it should not.

A refusal names the schema, the tables it found and the URL with its password
hidden, creates nothing, stamps nothing and leaves the schema exactly as it was;
the exit status is 1. The DDL is issued against that one schema by name, so a
table of one of these names in another schema the search path reaches is neither
adopted nor built around: the index is created whole in one schema, which is the
schema the drop later removes it from. Only that schema is examined -- the rest
of the database belongs to whoever made it and is neither read nor named.

**What is left for #501.** The DDL names its schema; the queries beside it --
the ingest's writes, `record_source.py`, `roots.py` and the report's raw SQL --
still name their tables bare and let the server resolve them through the search
path. Nothing can be adopted or destroyed through that any more, since the index is
built whole in one schema and the drop removes it from that same one; what
remains is that a table of one of these names created in an earlier schema of
the path *after* the index was built would shadow the index's own for those
queries. Binding `METADATA` to the resolved schema for every statement closes
it, and it is a change to every query rather than to one command.

---

## 3. Current state

What exists, and what each phase has to change.

**`spindoctor/cli/stats/schema.py`** defines three tables via a `_SCHEMA`
string applied with `executescript`. `images` is keyed on `image_name`
alone, so two images with the same basename in different volumes silently
overwrite each other. `open_stats_db` compares `PRAGMA table_info(images)`
against a frozen column set and raises with delete-and-re-ingest
instructions on a mismatch -- and silently creates a fresh database at a
nonexistent path, which section 2.4 deliberately changes for consumers.
`upsert_image` issues DELETE + INSERT with transaction management left to
the caller.

**`spindoctor/cli/stats/ingest.py`** flattens a document in
`rows_from_metadata`. It reads the rounded `navigation_result.offset_px`
rather than the top-level `offset`; merges `status_reason` and
`status_error` into one column; and drops `rotation_deg`,
`sigma_rotation_deg`, `covariance_px2`, and `sigma_along_unobservable_px`.
(The feature inventory is not dropped: it is aggregated into
`feature_sources`, and stays that way.) `ingest_metadata_files` wraps the
entire multi-root scan in a single transaction, has no incremental logic,
and reads files via `get_local_path()`, which does not download on a cloud
root. `main_ingest` takes `--db` defaulting to `nav_stats.sqlite3` in the
working directory, bypassing the configuration system, and reports via
`print()`.

**`spindoctor/cli/stats/report_common.py`** defines the `image_number`
Python UDF and a `where_clause` helper returning `?`-style fragments;
`report.py` registers the UDF and, with `report_sections.py`, issues the
`status_reason` taxonomy queries, the boolean integer arithmetic, the
`TOTAL(...)` aggregate, and `image_name`-keyed joins that all change per
sections 2.3 and 2.5. `report.py` is 782 lines; if the migration pushes it
past the 1000-line cap it splits into a package in the same commit.

**`spindoctor/dataset/results_filter.py`** implements the selection filters
with two distinct modes: presence/error filters walk the results tree once
per selected volume; absence-only filters skip the walk and use batched
`exists()` calls driven from `dataset_pds3.py`'s 64-image batching loop.
`_metadata_matches` requires `status == 'error'` and compares `status_error`
against `missing_spice_data` verbatim -- the behavior the split columns
preserve. Phase 5 makes this one implementation over the index, and the step
after Phase 7 makes it one implementation over the seam; the Phase 7 paragraph
below records where it ends up.

**`spindoctor/cli/backplanes/backplanes.py`** reads one record per image by
stub through a `PointingSource`; a stub nothing recorded raises and the caller
reports it. (The module had no docstring; Phase 4 adds one.)
**`spindoctor/cli/reproj/offsets.py`** classifies one parsed record and returns
`PointingSelection`; its reason vocabulary is section 2.9's plus the
pointing-ladder reasons the C-matrix reader switch added, and
**`spindoctor/cli/reproj/pointing_source.py`** holds the two storages that
record can come from, with the reason-mapping table in its module docstring.
**`spindoctor/cli/pds4/bundle_data.py`** serializes the whole document into
the supplemental product, which is why it is out of scope.

SQLAlchemy is not currently installed or declared.

Phase 7 changed where the paragraphs above get their records, and the steps
after it read that rather than this. **`spindoctor/nav_records/`** holds the
record types, the document rules, root identity, `Selection`, the
`RecordSource` protocol and `TreeRecordSource`, and imports no database layer;
**`spindoctor/results_index/record_source.py`** holds `IndexRecordSource` and
`open_record_source`. There is one walk of a results tree, one spelling of the
`_metadata.json` suffix and one rule about what makes a stub a key.
`spindoctor/support/nav_document.py` and `spindoctor/cli/stats/ingest/walk.py`
are gone. The enumeration is on the seam too: `spindoctor/dataset/results_filter.py`
carries no walk, no parser, no `exists()` probe and no copy of the suffix, and
answers all six of its flags from `listing()` and `facts()`, whichever storage
`open_record_source` resolved. The two are asked at different moments, because
an error filter reads a document and the documents worth reading are the
candidate images, which the enumeration's other constraints decide: the listing
of the selected subtrees is taken once and settles presence and absence, and the
facts are asked of each batch of candidates by name. Asked of the subtrees
instead, a run whose other constraints keep one image in a hundred would read
every document under them and discard almost all of it, which on a cloud results
root is one paid download apiece. What is left off the seam is the ingest's
retrieve-and-parse loop. The statistics report has moved:
`spindoctor/cli/stats/report_accumulate.py` fills one set of accumulators from
`facts(selection)`, and `report.py` and `report_sections.py` format every
section out of those rather than issuing a statement apiece.

The flattening the second paragraph above describes has moved as well.
**`spindoctor/nav_records/facts.py`** holds `facts_from_document`, which turns
one document into the `images` row and its two lists of child rows, and
`spindoctor/nav_records/derived.py` holds the values derived from a document's
own fields. The ingest, the index rebuild and both halves of the seam call that
one function, so the per-image shape a consumer reads is built once whichever
storage answered; `spindoctor/cli/stats/ingest_rows.py` and
`spindoctor/cli/stats/classify.py` are gone.

---

## 4. Implementation phases

Each phase is one pull request against `rf_results_index`, and each must
leave `main`-equivalent behavior intact for every program not given a
results-index-db URL.

### Phase 1 — Core layer, schema, and version gate

Introduce `spindoctor/results_index/` as a library package (not under
`cli`; library consumers use it). Define the SQLAlchemy Core metadata for
`images`, `techniques`, `feature_sources`, `failed_files`, `schema_meta` and
`ingest_runs` per sections 2.2-2.4; `open_index` with the `create` flag and
the version gate; the SQLite dialect events (WAL, `busy_timeout`, foreign
keys, the lockability probe); and the missing-driver message for PostgreSQL
URLs.

Declare `sqlalchemy` in `[project] dependencies` and the `postgres` extra in
`[project.optional-dependencies]`. Add `environment` to
`HASH_EXCLUDED_SECTIONS` with its rationale comment and test.

Tests: schema creation against SQLite; `create=False` raising on a missing
database, a missing `schema_meta` row, and a version mismatch (each message
asserted); a refused `create=True` open creating no table; the
`ValueError` guarantee on each route a driver exception takes (an absent
driver, an unparseable URL, an unknown URL scheme, a non-numeric port that
reaches the caller as a bare `ValueError`, a server that refuses the
connection, and an unexpected failure inside the engine factory), with the
driver's exception kept as the `__cause__` on each; a password absent from
every one of those messages while the rest of the URL survives, including on
the `schema_meta` and version gates and against a server that rejects the
password (`postgres` tier); the structural masking rule as a table of URLs and
exactly what masking each produces -- a slashed password, an at-sign in the
user name, a leading space, a hyphenated scheme, a two-line copy, and the
non-credential strings it must leave alone (a port with a later at-sign, a
user name with no password, a SQLite path, a scheme alone, the empty string)
-- with every credential-bearing row driven through `open_index` itself and
not only through the helper; the missing-driver message, with the driver
hidden by an import hook rather than by its happening to be absent from the
environment; the read-only cases of section 2.5 (accepted by a consumer,
refused by an ingest, and a write-ahead-logged copy that cannot be read),
each parametrized over both journal modes, plus an ingest into a read-only
directory, alongside a genuine lock failure in both modes; the result-code
classification of a probe failure (a file that is not a database, a path that
is a directory, a directory that does not exist, a file this user cannot
open, an exception carrying no result code at all, one carrying a code that is
not text, and an extended form of each classified family, since matching by
equality rather than by prefix loses the family's remedy); the connect handler
re-raising a driver error that is not a read-only refusal; a `sqlite:` URL with
a query string refused without leaving a file behind; the connect-time
settings read from a second connection held open beside the first, so a
one-shot application fails the test; a refused open disposing the pool it
built; Double-precision round-trip of a 15-significant-digit value; boolean
round-trip; a duplicate child row refused on both backends; the config-hash
exclusion.

The engine tests are three files, because one would run past the 1000-line
module cap: the opener's contract (`test_engine.py`), what it does with a
SQLite file (`test_engine_sqlite.py`), and how it names a URL without naming
its password (`test_masking.py`).

### Phase 2 — Ingest and reporting onto the index

One PR, because the report reads the table ingest writes and neither is
usable mid-cutover.

Ingest: rewrite the document flattening to the section 2.3 column set and
`ingest_metadata_files` per section 2.7 (single walk feeding presence, stat
and file list; batched retrieval replacing `get_local_path()`; incremental
skip; chunked transactions; per-image atomicity; `ingest_runs`). Root
resolution and normalization per section 2.2. The driver gains the
configuration surface (`--results-index-db`, `environment.results_index_db`,
`NAV_RESULTS_INDEX_DB`, the `none` sentinel), the logging surface, and the program
identity, with every collateral edit of section 2.10.

Reporting: move `sd_stats_report` onto the Core layer per sections 2.5 and
2.9 -- named binds, composite-key joins, `COALESCE` for both the reason
taxonomy and `TOTAL`, boolean spellings, the column-backed `image_number`,
`--root`, and `--db` removed from both programs.

The source scan that enforces criterion 10 reads only
`src/spindoctor/results_index/`, which is the whole of the Core layer while
that is all there is. The statistics programs move onto the Core layer here,
so this phase widens the scan's `_SOURCE_ROOT` to cover
`src/spindoctor/cli/stats/` as well, and updates the test that pins which
modules the scan reaches. Left alone, criterion 10 would report green over a
package that no longer holds the queries it exists to check.

Tests: a two-volume tree with colliding basenames producing two rows; a
bare-basename stub with NULL `subtree`; the unrounded offset; the separated
status vocabularies; an unchanged file skipped on a second ingest (proven by
counting retrievals *and* asserting no per-file stat call) and a touched one
re-read; `--force`; a chunk boundary crossed mid-run; `ingest_runs` written
at start and completed at end; a malformed document counted as an error
without aborting; a cloud-style root actually downloading (the test fails if
`retrieve()` reverts to `get_local_path()`). For the report: byte-identical
report and CSV output (modulo the documented new CSV columns) against the
same fixture tree ingested by the old and new ingest, which is what proves
the `COALESCE` rewrite; and a PostgreSQL run of the same assertions under
the `postgres` marker.

Details settled during execution, none of them a change of intent:

- **The root normalization lives in `spindoctor/results_index/roots.py`**,
  alongside the `ingest_runs` reads that decide whether a root has been
  ingested at all. Section 2.2 asks every consumer to spell a root the same
  way and to refuse a root with no completed run; both are one function
  there rather than one per consumer. `sd_stats_report` already uses the
  refusal for `--root`; the pipeline consumers of Phase 4 use the same one.
- **The old ingest's output is frozen rather than re-derived.** The
  byte-identical comparison needs the previous implementation, which this
  phase deletes, so the fixture tree and the `report.md` / `images.csv` the
  previous implementation produced from it are committed as test data in
  their own commit, ahead of the rewrite.
- **The CSV's two feature-count aggregates become integers.** `TOTAL(...)`
  always returned a float, including a `0.0` for an image with no features;
  `COALESCE(SUM(...), 0)` returns the count. The regression test asserts the
  numbers are equal rather than the text, and asserts it for those two
  columns alone.
- **The CSV's `status_reason` column no longer carries a fatal error.** That
  is the merge the split columns exist to undo, and `status_error` is beside
  it now. The regression test asserts that the pair reproduces what the one
  column held.
- **`sd_results_index` and the statistics package leave the print-only list**
  in `tests/spindoctor/config/test_logging_static_invariants.py`, which
  section 2.10's collateral list does not name. The list is narrowed to the
  report modules, which keep `print()`.
- **The criterion-10 scan skips Markdown table rows.** The report writes its
  tables as literal rows, one of them headed `total (s)`, which the `TOTAL(`
  pattern reads as the SQLite aggregate. Nothing beginning with a table
  separator is SQL, and a test pins that the exclusion does not reach a
  statement -- against a widened exclusion as well as a blanked one, since a
  pattern that still excludes something is what would quietly empty the scan.
- **The column set changed, so the schema version was raised to 4 here.** The
  JSON columns gained `none_as_null` (section 2.3), `ingest_runs` gained
  `files_removed` and then `directories_missed`, and `failed_files` was added
  (section 2.7). There are no migrations, so this is one version bump covering
  all four. The last of them arrived after the version had already been raised
  once in this phase, and it was raised again rather than reused: an index built
  from an earlier state of this phase would otherwise pass the version gate and
  then fail on a column that is not there, which is exactly what the gate exists
  to prevent. Phase 5 raises it again, to 5, on the same reasoning, and the
  column set changed three times more after it -- the summary-PNG flag left both
  file tables; `directories_missed` left `ingest_runs` when a walk that cannot
  list a directory began to stop rather than complete; and `images` gained
  `shutter_mode`, `spice_kernels` and `camera_frame` when the C-kernel writer
  began reading its records from the index -- and `volume` became `subtree` in
  both tables that carry it, which is what makes the current version 9.
- **The CSV export states its line terminator.** `csv.writer` defaults to CRLF;
  the export now names LF. The frozen `images.csv` blobs are LF, so what the
  export writes matches them byte for byte, which the previous implementation's
  output did not -- the blobs were normalized when they were frozen and the
  commit that froze them did not say so. The regression comparisons read fields
  rather than lines and would not notice either way, so the terminator has a
  test of its own reading the file as bytes.
- **Masking became one rule over a corpus, and stopped touching results
  roots.** Section 2.4 above is the rule as it now stands: one structural pass
  covering the query-parameter form of a password as well as the authority
  form, applied to every URL rather than only to the ones the parser rejects,
  and asserted over a generated corpus instead of a list of remembered shapes.
  `require_ingested_roots` masks the index URL itself, so a consumer cannot
  reintroduce the leak by forgetting; the roots in that message, and the results
  roots in the ingest driver's log, are printed exactly as they were given.
- **Section 2.4 rule 4 decides the ambiguous shape the other way round.** The
  base plan read `host:5432/path@name` by whether the text before the slash is
  digits -- "a port is digits and a password is not". It is now read as
  credentials outright. This is the one plan edit of this phase that reverses a
  decision rather than adding or tightening a requirement, so it is listed here
  on its own: digits-decide leaves a password that opens with digits,
  `123/secret`, visible in full, and the cost of the reading taken -- a mangled
  host and database name -- no longer reaches anything an operator needs, since
  a results root is never passed through masking at all. The URL parser reads
  that shape the same way, so for every spelling of it a parser accepts, this
  rule and `render_as_string()` hide the same characters.
- **The credentials end at the last `@` of the string** (section 2.4 rule 3),
  not at the last one before the first `/` and not at the last one before a
  `#`. Both narrower bounds stop inside a password that carries the character
  they stop at, and each was a leak. The accepted cost is over-masking a URL
  whose tail carries an at-sign after the real credentials, which is a fragment
  on a connection URL and therefore a URL no driver would have taken.
- **A failed write costs one file, not the run.** Section 2.7's isolation of a
  chunk whose write fails. SQLAlchemy savepoints were tried first and are not
  usable here: pysqlite's transaction handling commits a released savepoint
  independently of the enclosing transaction, which silently breaks the chunk
  boundary. Rolling the chunk back and rewriting it one image at a time costs
  nothing when nothing fails.
- **The ingest driver's exit status reports completion rather than counts**
  (section 2.7), because the count-based status flipped between two passes over
  one unchanged tree. It also always exits rather than raising, which is what
  its own documented contract said and what a traceback out of a console entry
  point breaks.
- **The walk is handed the normalized root** (section 2.7). Absolutizing the
  root for the key while walking the string as typed made a relative root -- a
  documented spelling -- a traceback out of the driver, with the run row already
  written and its finish time left NULL.
- **A directory the walk cannot list ends the pass** (section 2.7), and the
  walk skips a directory it has already listed rather than descending into it
  again. The first is what keeps "absence means never navigated" honest for the
  consumers of Phase 4: no run that completes has a gap in it. The second stops
  a link back into a tree from writing one document as forty-one rows, and is
  not a gap, so it stops nothing. (Both arrived here as a count of missed
  directories on the run row, reported by every consumer; the count went with
  the change that made the first of them stop the pass, which left it always
  zero.)
- **Ingest is a package, on the treatment section 3 names for the report.**
  Everything section 2.7 asks of one pass carries `spindoctor/cli/results_index`
  past the 1000-line cap, so it is `ingest/` split along the stages a pass runs
  through: `counts` (the tally the summary is read from), `walk` (the single
  listing of a root), `store` (what the index already holds, and how rows go
  back in), `chunks` (batched retrieval, reading a document, the per-chunk
  write), `runs` (the record that makes absence of a row readable), and
  `driver` (the pass itself). The package re-exports the whole surface, so
  every consumer imports the names it always did from
  `spindoctor.cli.results_index`. `report.py` stays inside the cap and stays one
  module. The source scan of criterion 10 finds the split modules for itself,
  and the floor it asserts its own reach against names them.

### Phase 3 — Cloud-task ingest

`sd_results_index_cloud_tasks` per section 2.8, entry point in
`pyproject.toml`, with the enqueuer's two modes on `sd_results_index`.

Tests: enqueuer creates the schema and workers refuse to; concurrent local
SQLite workers produce the same rows as a serial ingest; per-task counts
aggregate into `ingest_runs`; the worker writes zero bytes to stdout and
stderr from SpinDoctor code, asserted at file-descriptor level as the
existing cloud-task silence tests do; the driver appears in
`_CLOUD_TASK_DRIVERS` and passes the no-logging-flags assertion.

Details settled during execution, none of them a change of intent:

- **The enqueuer prunes at fan-out**, which is section 2.8's open question
  decided. The reasoning is in section 2.8 above; the short form is that fan-out
  is the only moment of the pass holding the complete listing the prune is
  licensed by, and it is the listing the shares were cut from, so the prune and
  the workers' writes cannot touch the same stub. Within one fan-out this is an
  ordering guarantee rather than only a set argument: the prune runs before the
  task descriptions are built, so no worker of that pass exists while it runs.
  Two limits are worth recording, and both are **documented limits** rather
  than open questions: each is stated in the statistics guide, and each was
  decided rather than left. The disjointness is a claim about **one** fan-out:
  two overlapping fan-outs against one root can leave a stale row, when a worker
  of the first writes a stub after the second's snapshot of what is recorded and
  before its delete, for a document that left the tree between the two listings.
  It is narrow, both runs are unfinished while the window is open, and the next
  pass removes the row; refusing or warning about a fan-out over a root whose
  newest run is unfinished was considered and not done. And the prune is
  destructive before any document has been read, so a fan-out that is abandoned
  shrinks the index -- but only by rows whose documents have genuinely left the
  tree, and the run is unfinished throughout, so no consumer reads the root
  either way; what it costs an operator is a full re-ingest of that root, which
  is the sentence the guide gives them.
- **Each task's report is counted once, under its `task_id`**; a share counts
  toward a run only when it names that run's root; the account must match the
  listing exactly rather than merely reach it; and a run whose listing was never
  recorded is never stamped. All four are in section 2.8 above. Summing files
  alone lets a share reported twice cover for a share that never ran; crediting
  by run identifier alone lets a task file that outlived its index stamp a root
  with nothing under it, since the identifier restarts at 1 in a fresh one;
  accepting an account that runs past the listing stamps a run on evidence that
  cannot be right; and reading an unrecorded listing as zero files stamps a root
  nobody listed. Each one hands consumers a tree of images to read as never
  navigated, which is the claim the run bookkeeping exists to license.
- **Two spellings of one root are one root**, at the fan-out and at the
  completion. Listed twice, every document is handed out in two shares and read
  twice, the first of the two runs is left unfinished for good, and the
  completion stamps the newer run and then reports the root it has just finished
  as one nobody divided up. Every mode reads its roots through one helper,
  `distinct_roots`, which the driver applies once before any of them, so the
  roots a run opens by naming are the roots it works over rather than the words
  typed: a run that named a root two ways and accounted for it once read as a
  root having gone missing between the two messages. Applying it there also
  charges a spelling that is not a location to the root, rather than leaving it
  to the catch-all as a failure nobody enumerated. Which spellings those are is
  decided in `normalize_root_url`, so every program that reads a root refuses
  the same ones: what no storage layer can render absolute, what carries a null
  byte -- which renders and then fails at the first listing, charged to a
  directory rather than to the word that caused it -- and an empty one, which
  renders as the working directory and would otherwise walk it, write its
  documents under a root nobody named, and report a completed pass.
- **A count no share could report is not a share's tally, and neither is a
  sum.** `_share_tally` bounds the magnitude of each count as well as its type
  and its sign, and the completion holds the running total of a run's shares to
  the same bound. What reaches the run row is that total, and its columns hold
  32 bits on the narrowest supported backend, so bounding each count alone still
  leaves two accepted lines to overflow one between them -- ending the whole
  completion in the database driver's own error, for corrupt or foreign lines of
  a concatenated event log, which is exactly the input class the guards either
  side of it exist for. A result that would put its run past the bound costs its
  own line, like every other result nobody can read, and the run comes up short
  and is named.
- **Each of `_share_tally`'s guards is pinned by what breaking it costs.** The
  run identifier's type, the root's type and its normalization, and each count's
  type, sign and magnitude are separately tested, and the tests assert the
  consequence rather than the refusal: a fractional or Boolean count accounts
  for a listing exactly and stamps a run whose documents were never read, a
  Boolean run identifier is credited to run 1 because `hash(True) == hash(1)`,
  a NaN count writes SQL NULL where the run row records how far the pass got,
  and a root nothing can render absolute ends the whole completion in an
  exception nobody enumerated. The sum's bound is pinned on the `postgres` tier, since it
  is the backend whose columns the bound comes from.
- **The seam lives in `spindoctor/cli/results_index/tasks.py`**, beside the pass
  it divides: fan-out, one share, and the completion that adds them up are the
  same three stages `driver.py` runs in one process, and both read the same
  driver, store and chunk modules. The package re-exports them, so the drivers
  import from `spindoctor.cli.results_index` as they do everything else.
- **Two Phase 2 helpers were widened rather than copied.** `_files_to_read`
  takes the files, what the index records about them and the metrics flag
  instead of a whole `_RootListing`, so a share selects by exactly the rule a
  root does and there is nothing listing-shaped for a worker to reach for; a
  file the flag claims metrics for while carrying none is read rather than
  compared, since a share's claim travels beside the entries it describes and
  can disagree with them, where a whole-root walk reports metrics for every
  file or for none; `_recorded_files` takes
  an optional set of stubs, so a share reads what the index holds about its own
  files rather than about every row of an archive-scale root.
- **A share names every file it could not read**, which a pass over a whole
  root does not: the fan-out bounds a share, and a worker has no run log to name
  them in instead. The whole-root pass keeps one example per reason, as before.
  The share's result carries that per-reason tally as well as the names, and the
  completion folds it into the summary it writes, so a divided ingest reports
  why files were refused exactly as a single-process pass does. The names stay
  in the event log: a summary that listed several hundred thousand of them would
  read as a broken ingest rather than as the ordinary thing it is, which is why
  the whole-root pass keeps one example per reason in the first place.
- **A run row carries what the fan-out found before it is finished.**
  `files_seen` and `files_removed` are written at fan-out with the finish time
  left NULL, because nothing later in the pass can find them out again and the
  completion step must not have to list the root to learn them.
- **`sd_results_index` and `sd_results_index_cloud_tasks` joined the program
  identity tests** in `tests/spindoctor/config/test_logging_keys.py`, which named
  neither. The interactive driver has declared `PROGRAM_NAME` since Phase 2 and
  section 2.10's collateral list did not reach that file.
- **The developer guide's script table gained the statistics family.** It is
  headed as the full set of `[project.scripts]` and named none of them, so
  adding one program to it meant naming its siblings too.
- **Every root-keyed delete is exercised with a second root present.** A
  fixture holding one root cannot tell a query keyed by the pair from one keyed
  by the stub alone, which is how a root-blind query ships. Both arms of the
  share's own lookup, both deletes of an image write, both of a refusal, and
  both of the prune are each pinned by a test that fails when its root half is
  dropped (`tests/spindoctor/cli/results_index/test_ingest_two_roots.py`, and the two
  share-lookup tests beside it). What each break costs differs -- a navigated
  image made invisible under another root, a refusal cleared so its file is
  downloaded again on every pass -- and the tests are named for it.
- **The suite resolves no results index it did not name.** A URL comes from an
  argument, the `environment.results_index_db` configuration variable, or
  `NAV_RESULTS_INDEX_DB`, and a test of the no-index path names none of them. Both
  ambient levels are closed in `tests/conftest.py` rather than by a line each
  test author has to remember: the variable is unset and the working directory
  is one holding no `nav_default_config.yaml`, for the whole session and again
  around each test. The session half is what covers a fixture of a broader
  scope, which is built before any per-test closure could run and is exactly the
  kind that ingests a tree or builds a report. Run from a directory that names a
  live index, the suite had opened it -- for SQLite, a write-lock probe against
  a file an ingest may be holding. Two things are left to the test author and
  documented as such: a subprocess given a working directory of its own resolves
  its configuration there, and the directory the suite runs from is shared by
  every test of the worker, so a test that writes a file into it is failed on
  the way out rather than leaving a configuration for every later test in that
  worker to resolve through.

### Phase 4 — Backplanes and reprojection consume the index

The `PointingSource` protocol and both implementations; `--results-index-db` (and
the `none` sentinel) on `sd_backplanes`, `sd_mosaic`, and their cloud-task
variants; the backplane single-row read with the missing-stub raise; the
root-url comparison failure of section 2.2.

Unit tests at the `PointingSelection` level over one fixture tree read both
ways: each reachable reason with and without an index, including `null_offset`
from a success-with-NULL row, `no_pointing_block` and
`no_cmatrix_rotation_fitted` from the two shapes of NULL `cmatrix`, and
`malformed_pointing` from a stored matrix the validator refuses; the
unreachable-reason remapping of section 2.9 asserted (a malformed-offset
document yields `malformed_offset` via files and `null_offset` via the index);
the recorded C-matrix and midtime round-tripping bit for bit on both backends;
the same `IMAGE_LOGGER` warnings in the index path; a program handed an
unopenable URL failing rather than falling back; a consumer refusing a root
with no completed `ingest_runs` row; a mode that reads no navigation record --
a dry run, a mosaic pass that skips reprojection -- not failing on an index it
never asks; a recorded zero -- an offset of two zeros, a midtime at the J2000
epoch -- read as the value it is rather than as an absence, since that is what
separates asking whether a column holds a value from asking whether the value
is true; each credential-masking call on the consumer path asserted on real
bytes over a server URL, because a `sqlite:` URL is returned by the masking rule
unchanged and can hold nothing to it; each cloud-task driver's worker data
carried across a real spawn context, with a task run in the child, because a
source that cannot cross that boundary fails every task of a run before the
child starts; and a two-root fixture whose second root differs in the value
under test, asserted from both sides so that no single row can satisfy both.
Integration tests (marked `integration`): identical backplane and mosaic
products for the same real, really-navigated, really-ingested images with and
without an index, asserted exactly on the outputs.

Details settled during execution:

- **The plan's Phase 4 text described an API that no longer existed.** The
  C-matrix reader switch replaced `load_offset_if_any` / `OffsetLookup` with
  `select_pointing` / `load_pointing_if_any` / `PointingSelection` and a wider
  reason vocabulary. Section 2.9 above is rewritten against the code that
  exists; the requirement is unchanged, and the mapping table gained the rows
  the new vocabulary needs.
- **`midtime_et` was missing from the schema and is added here.** The
  navigator writes a seven-key `times` block and ingest stored six of them.
  The one it dropped is the value `_parse_pointing_values` requires and
  `apply_cmatrix_to_obs` gates against the observation's own midtime at 1e-6 s,
  so without the column every index-backed record would degrade to the pixel
  offset while the same record read from its document applied the C-matrix --
  silently, and with different products. The column set changed, so the schema
  version is 6.
- **The recorded midtime is stored, never recomputed.** The midpoint of the
  recorded shutter epochs reproduces it for the frames measured, but that is a
  property of one producer's arithmetic rather than of the record, and a gate
  at 1e-6 s leaves no room to be nearly right. A test plants a record where the
  two differ and asserts the recorded value comes back.
- **The seam carries the whole record, not only the pointing.** The backplane
  stage reads a record's status before it decides there is work to do, so
  `PointingSource` answers `read_record` as well as `load_pointing`. That is
  also where the missing-stub raise lives, since it is the question whose
  answer must be an exception.
- **The pointing block is rebuilt from any of its four columns**, not from
  `cmatrix_original` alone. Both keys tell a fitted-rotation result from a
  record with no pointing block, and the wider one additionally reproduces the
  file path's `malformed_pointing` for a record whose baseline ingest could not
  store while its corrected attitude survived.
- **The store reads a record through the readers' own functions.** Every column
  a consumer classifies a record from is filled by
  `spindoctor/support/nav_record.py`, which is where the readers read the same
  fields. A second set of coercions in the store, agreeing when written, is
  a second reader of the record and drifts: it truncated a three-element
  `offset` the classifier refuses whole, refused a numeric-string pair the
  classifier converts and applies, and stored a rotation only in the flat form
  while the classifier read the 3x3 nesting too. Each of those made the same
  document supply different pointing depending on which storage a run read.
  Sharing the domain is what makes "products are identical with and without an
  index" a property of the code rather than a claim about it.
- **A NULL offset is rebuilt as a null value rather than an absent key**, so
  the index reports `null_offset` where the file path can report any of five
  reasons. Which of the two the rebuild picks makes no difference: none of the
  five supplies a pointing, and no consumer branches on the key's presence. The
  backplane stage's refusal of a success-status record carrying no `offset`
  field is removed for exactly that reason -- it was a branch on a distinction
  no column can carry, so it refused through a document what it processed
  through a row. That record is now a recorded no-answer like a null offset:
  reported, counted under `missing_offset_key`, and processed on whatever
  pointing the rest of it supplies.
- **The sentinel for a document naming no outcome is not made collision-proof.**
  `UNKNOWN_STATUS` is the word `unknown`, which a hand-built document could name
  for itself; no string sentinel can be proof against that, and only a nullable
  column would be, which is a schema version bump and a full re-ingest. It is
  not needed: both readers report the field through one function that defaults
  a record naming no outcome to that word, so a document naming it and a
  document naming nothing are reported alike, and the collision changes nothing
  anybody reads.
- **Reading a record refuses a document that is valid JSON and not an object.**
  The backplane stage cast the parsed document to a mapping and reached an
  attribute error from the middle of a batch run; the seam has to return a
  record, so it refuses one that is not, naming the file.
- **`sd_backplanes` contains a failure to its image** (#491). The driver caught
  `FileNotFoundError` only, so anything else one image raised ended the run and
  discarded every image after it -- including the index read failure this phase
  introduced, whose own contract says a caller reports it against one image.
  Backplane generation is per-image work with no cross-image state, so the
  driver reports the failure against the image, counts it, and goes on, closing
  the pass with a done/skipped/failed summary. The run's log carries the
  traceback as well as the image and the message, because the failures the
  broad catch exists for include ones raised before the image has a log of its
  own -- the pointing lookup, the observation setup -- and for those the run log
  is the only account there is.
- **The backplanes driver logs its command line through
  `log_run_environment`** rather than directly, which is the one place a
  connection URL on a command line is masked and is what every other driver
  already does.
- **Each cloud task builds its own source and closes it**, in both workers.
  `cloud_tasks` runs every task in a process it spawns for that task and hands
  it the worker's shared data by serializing it, so a source built at worker
  startup reaches no task at all: a `SQLAlchemy` engine cannot be serialized,
  and the reprojection worker's whole index-backed mode failed in the parent
  before a single task ran. What crosses is the worker's parsed command line,
  which is what the root and the index are named by, so that is what a task
  builds from. A named index that cannot be opened returns
  `unusable_results_index_db` in the task result, which is how a worker reports every
  other environment failure and is what keeps a task that could not obtain its
  source from reprojecting its whole batch on uncorrected pointing and
  reporting a clean one. Pinned by a test that stands up a real spawn context
  with the shared data each driver's own startup built, and runs a task in the
  child.
- **The shared test helpers that drive a program's own parser** live in
  `tests/spindoctor/cli/conftest.py`, which is the narrowest scope covering
  every module that asserts a command-line surface, and is where the project's
  own testing rule puts shared test helpers.
- **`status` is the document's own top-level field and nothing standing in for
  it.** Ingest coalesced the top-level `status` with the copy inside
  `navigation_result` before falling back to `unknown`, which made a
  classification decision at ingest time: for a document naming no outcome of
  its own beside a nested `success`, a record rebuilt from the row selected the
  recorded C-matrix where the same record read as a file supplied no pointing
  at all -- a product difference, and the sharpest possible falsification of
  "the index-backed source classifies nothing itself". The fix belongs at
  ingest, because a rebuild cannot reproduce a field the column already merged
  away. The column now holds `'success'` exactly when the document's top-level
  field does, which is the whole of what the ladder asks it; a document naming
  no outcome still records `unknown`, which is what section 2.3 always said the
  column held and what the user guide documents. The consequence for the
  selection filters is that such a document no longer matches an error filter
  under an index either, so it no longer diverges from the tree path, and the
  divergence Phase 5 recorded for it is removed from that list.
- **A field the row does not carry is rebuilt as an absent field, not a null
  one**, `offset` excepted. The backplane stage reads `status_error` with a
  default, so a rebuilt record carrying the key with a null value reported
  `None` where the document reported `unknown` -- on every failed and
  conflicted record, which write no such field at all. The `status` column is
  NOT NULL, so its `unknown` is rendered back as the absent field it stands
  for. `offset` keeps its null-valued key for the reason below.
- **A `status: success` record carrying no `offset` field builds the same
  product in both storages.** The backplane stage used to refuse it through a
  document while the index, which cannot tell an absent pair from a null one,
  applied the recorded C-matrix and wrote the product. The refusal was the odd
  one out rather than the index: a null, malformed or non-finite offset already
  reached geometry on whatever pointing the rest of the record supplied, so the
  stage now treats an absent offset the same way, degrades on the reason the
  classifier names, and reports `uncorrected_pointing` when nothing else
  supplies a pointing either. `NavResult` forbids a success with no offset, so
  no navigation writes the shape at all. The rebuild still renders the pair as
  a null-valued key rather than an absent one, which is what makes
  `null_offset` the row's reason for all five ways an offset can supply no
  pair. Pinned by a test on both sides.
- **A `pointing` block carrying only fields the index has no column for is
  classified differently**, and is listed with the other stated differences. A
  block holding only `camera_frame` leaves no trace in the row, so the index
  reads `no_pointing_block` where the document reads
  `no_cmatrix_rotation_fitted`. Same mechanism, same product, different tally.
  Every block a navigation writes carries the baseline and both frame
  identities.
- **Both of a file-backed source's methods apply the same path rule.**
  `read_record` joined the stub onto the root while `load_pointing` put it
  through the rule that refuses null bytes, absolute fragments and `..`
  segments. One class applying two rules depending on which method was called is
  the shape a traversal gets through; the rule is now shared, and a stub that
  is not a key under the root reads as no record for that image.
- **The message for a record nobody holds names the storage that was
  searched.** It named `nav_results_root` in both paths, which is untrue of an
  index and hides the likelier of the two causes: an index is a snapshot of its
  last ingest, so an absent row can mean the image was navigated since. One
  message shape, one value naming what was searched, in both paths.
- **Nothing from the database layer escapes the pointing seam either.** The
  translation Phase 5 wrote for the selection seam is now shared rather than
  copied, so a lost connection or an unreadable table is reported against the
  image as a refusal naming the index, not as the database layer's own
  exception type in the middle of a per-image catch-all.
- **The modes that read no navigation record open no index.** `sd_backplanes
  --dry-run` and `sd_mosaic --skip-reproject` and `--dry-run` looked nothing
  up and yet failed on an index that would not open, which on a machine
  exporting `NAV_RESULTS_INDEX_DB` breaks invocations that worked before the variable
  was set. Fail-early stays for every mode that does read a record.
- **One lookup asks both tables** (section 2.9). Reading absence from `images` alone reported every document
  the ingest refused as an image nothing navigated, which on a real results root
  is hundreds of navigated images: their products were built on their recorded
  offsets through the tree and on uncorrected pointing through the index, and
  `sd_backplanes` skipped them under an index and built them without one. The
  index knows better -- `failed_files` holds a row for each -- and Phase 5's
  selection already reads that table for exactly this reason, so one index was
  answering "does a record exist for this stub?" two ways in two consumers.
  Both of the seam's methods now consult it and fail the image. `roots.py` says
  what is now true: a completed ingest makes absence from *both* tables
  meaningful, never absence from `images` alone. Both halves are one query --
  the root and stub are selected as a row of their own and each table is
  outer-joined onto it -- rather than a record lookup followed by a refusal
  lookup: an image with no record is the common case on a partially navigated
  root, so it is exactly that image that would have paid the second round trip
  the index exists to remove.
- **The store assembles a recorded rotation through the reader's own function.**
  `_cmatrix_or_none` re-decided "are these nine real finite numbers" with a
  per-entry rule while the reader assembled an array, so the two accepted
  different sets: a `cmatrix` written as nine one-element rows was a valid
  rotation to the reader and NULL to the column, and the same document was
  reprojected by frame replacement through the tree and by `OffsetFOV` through
  the index. `nav_record` now owns the whole question -- the two written shapes,
  the assembly, the real-number dtypes and finiteness -- and returns the 3x3
  matrix both the reader and the store use. `cmatrix.py` takes the dtype set
  from there rather than defining a second one. What is left to
  `validated_record_rotation` is the question only it can answer: whether the
  matrix is a proper rotation.
- **A recorded number too large to be a float is a value a reader cannot use,
  not an exception.** `finite_float` promised `None` and raised `OverflowError`
  on a JSON integer of several hundred digits, which JSON permits and no float
  holds. It cost the image through the documents and the whole document through
  ingest -- so the image then read as one nothing navigated, on a record whose
  every other field was fine. `finite_float` and `record_offset` refuse it, and
  the recorded midtime is read through `finite_float` rather than through a
  second copy of the same three checks.
- **`status_error` is stored through the function its readers read it with.**
  It was the one field a column filled by a rule of its own; the two agreed for
  every value, which is exactly the shape of the defect this module was created
  to remove. The column is now NULL wherever `record_status_error` reports the
  record as naming no error, the literal word `unknown` included, matching what
  the NOT NULL `status` column does with the same word.
- **A boolean is refused wherever a recorded rotation puts one.** The refusal
  was written on the nine entries and read their Python type, which is a
  container for every nesting deeper than the two a record is written in. So
  `[[true],[0.0],[0],[0],[1],[0],[0],[0],[1]]` assembled into a float array,
  read as the identity, and was selected and applied as a corrected attitude by
  both storages -- the case three docstrings and a test name said could not
  happen. Each entry is now judged by the array it makes, so a `True` among
  numbers is refused however deeply it is written.
- **The accepted domain of a recorded rotation is stated where it is decided,
  and is wider than the two shapes a producer writes.** A recorded value is one
  3x3 matrix of finite real numbers in any nesting an array library reconciles
  into that shape: nine values, a 3x3 nesting of them, and nine rows of one.
  That is deliberate rather than incidental -- the value denotes the same
  rotation however it is bracketed, and a reader that refused one denoting
  shape while applying another would classify a record by its typography -- and
  it is what the reader and the store both apply, so no shape is a rotation to
  one and nothing to the other. Three statements of the class the two storages
  count differently said "neither nine values nor a 3x3 nesting", which put
  nine rows of one in a class it is not in; all three now name it, and the
  drift test asserts on the shape that separates the two wordings.
- **A recorded value that is present and false is present.** The rebuild's two
  presence guards are `is not None` and were held by no test: a recorded offset
  of two zeros and a midtime at the J2000 epoch are the separating inputs, and
  they separate in the product -- the first becomes no offset at all, the
  second a pointing block with no epoch to gate against. Both are now fixture
  records the two storages are compared over.
- **Every masking call on the consumer path is pinned in the tier that runs.**
  A `sqlite:` URL is returned by the masking rule unchanged, so no test built on
  one can hold anything to masking; the calls that name a server URL -- the
  source's own name for its index, the run log's `Results index:` line in both
  consuming programs, and the translation of a read the database would not
  answer -- are pinned on real bytes, with a password carrying every character
  that delimits a URL and a user name carrying an at-sign. None of the three
  needs a database.
- **A cloud backplane task returns every way its image can fail.** It reported
  an unusable index and let a per-image failure out as an unhandled exception,
  which the framework logs as a traceback with no reason an enqueuer's tally
  can count and, under `--retry-on-exception`, retries for a refusal that will
  refuse identically. Nothing recorded the image is a skip named
  `no_navigation_record` and everything else is `backplanes_failed`, both
  returned rather than raised.
- **The unpinned guarantees are pinned.** Reverting the backplane stage's two
  reads through `nav_record`, `select_pointing`'s read of the same field,
  `_pair`'s refusal of a sequence that is not a pair, the guard that turns nine
  entries of mismatched shapes into a malformed record, or `close()`'s disposal
  of the engine, each left the whole suite green. Each now fails a test, and
  each test was verified by making the reversion.

### Phase 5 — Selection filters

`ResultsFilter` answers all six flags through the record seam, per section 2.9.

Tests: for every filter flag, the answer over the documents against the answer over the index, over a fixture tree whose malformed-metadata images are files the walk finds and the ingest refuses, so that the equivalence covers the refusal table; every contradictory-pair rejection, including the four `--has-no-offset-error` adds; the command-line surface of every program that declares `--results-index-db` and of every program section 1 keeps reading files; an exported URL answering an enumeration for the first and not for the second; a selected subtree the results root holds no directory for, which contributes nothing, against one that is there and will not be listed, which ends the run; and the enumeration-list guard that binds the four statements of what the index answers differently to each other.

Details settled during execution, none of them a change of intent:

- **A run selecting images that have a document hands itself a plain set.** The constructor opens the source, reads the one stream it needs, and keeps a frozen set of stubs, so `passes(stub)` costs a hash lookup; where no further question is coming it closes the source at once rather than holding a connection pool across an enumeration. The stream is read inside the same guard as the open, because a source reading rows runs its query as the caller reads it: a storage that stops answering surfaces at the boundary that translates it rather than above.
- **A document recording no fatal error is a filter of its own.** `--has-no-offset-error` is the negation the other three lacked, and it reads the same per-image facts they read: one more condition on the status those facts carry. A file no facts came out of matches it no more than it matches the other three, since such a file records neither an error nor the absence of one. It contradicts the three that name an error and, like them, contradicts `--has-no-offset-file`, because it asks what a document records and a document that does not exist records nothing.
- **A refusal says which exclusion it means.** Two flags that exclude each other
  and nothing else are named as mutually exclusive. One flag that excludes
  several which are satisfiable together -- `--has-no-offset-file` against the
  four that read a document, `--has-no-offset-error` against the three that name
  an error -- is named against the ones it excludes instead, because "mutually
  exclusive" over a set claims an exclusion between every pair in it, and
  `--has-offset-error` with `--has-offset-spice-error` is a pair the constructor
  accepts. Every flag of the combination is still named, which is what a user
  needs in order to know which one to drop; what changes is the claim made about
  them. A test holds every "mutually exclusive" clause any combination can
  produce to a pair the constructor really does refuse.
- **Presence is read from `failed_files` as well as `images`.** A `*_metadata.json` the ingest refused is a file the walk finds, so without the refusal table criterion 1's malformed-metadata image would be present in the tree and absent in the index, for `--has-offset-file` and `--has-no-offset-file` alike. It is a listing of the index that has to say so, which is why the seam's listing reads both tables.
- **`failed_files` carries the subtree**, which is a column-set change and so a
  schema version bump, to 5. It is a fact of the walk rather than of the
  document, so it is as knowable for a file nothing could be read from as for
  one that ingested, and a selection filter asks about the file and not about
  its contents. Without it, a one-subtree enumeration fetches every refusal the
  root holds. The incremental skip compares a refusal's metrics exactly as it
  compares an image's. That skip reads the refusal table for the root it is
  walking, and the read is exercised with a second root holding a copy of the
  same tree, which is what a mirror or a restored backup produces: the same
  stubs at the same lengths and the same times, so a refusal read without its
  root makes a pass decline to read a file it has never seen and write no row
  at all for it -- neither an image row nor a refusal, which every consumer
  reads as an image nobody navigated.
- **What the index answers differently, as far as it is known.** Each member is
  stated in the module docstring and in the navigation guide's account of
  `--results-index-db`, each has a test of its own, and a member found later is added
  here, in the docstring, in the guide, and in a test, in the same commit. Each
  member carries a phrase that identifies it, and a test binds that phrase to
  the entry carrying it and to that entry's place in the list, then compares the
  four lists entry for entry. So an entry deleted, an entry that states two
  members because a neighbor absorbed the deleted one's phrase, an entry that
  states none, a member added to one list and not the others, and a list whose
  members come in another order all fail. What that test cannot check is whether
  an entry tells the truth about the member it states: an entry that keeps its
  identifying phrase and claims the opposite of the member passes it, and only
  reading the paragraph, or the member's own behavioral test, catches that. The
  list is maintained rather than closed: it is what execution and code reading
  have found, and a divergence nobody has found yet would be a defect of this
  list rather than a departure from it.
  1. A file that exists and **has no row at all** reads as absent, which is what
     the absence filters read as "this image was never navigated". One pass ends
     that way: a file the pass could not retrieve. That is deliberate -- a
     recorded row would be skipped for as long as the file did not change, and
     the next pass would never retry it. Two failures are not a second and a
     third: a walk that cannot list a directory raises where it meets it, and a
     document whose rows the database would not store raises where it is written
     (section 2.7), so no root a consumer reads has a completed pass that skipped
     either.
  2. A document **rewritten in place, keeping the length and the modification
     time it had before,** is skipped by the incremental comparison
     (`_is_unchanged`, which has only `(mtime_ns, size_bytes)` to go on), so its
     row goes on recording what the document before it said and an error filter
     answers from that. A tree restored by a copy that
     preserves times, a document patched and stamped back from a sibling, and a
     backend reporting one modification time for two writes all produce it; an
     ordinary re-navigation writes a different length at a later time and does
     not. It is documented rather than fixed because the only thing that
     distinguishes such a file from the one already read is its content, and
     retrieving every document to find out is exactly the cost the skip exists
     to avoid -- a content digest would be paid on every file of every pass to
     catch a case a times-preserving restore produces. `--force` is the remedy
     and is what the documentation points at. This one is not the snapshot's
     age: a pass that finished a second ago answers from the document before
     the rewrite.
- **The answer says how old it is, and what that does not cover.**
  `ingest_runs.finished_utc` is read by `snapshot_finish_time`, which opens the
  index for that one question and closes it again, and `ResultsFilter` reports
  it with the count of what the index holds. The index detects no change since
  that moment, and a URL resolved from the environment means an operator may not
  know which pass is answering, so the moment travels with the answer rather than
  with whoever exported the variable. Outside the enumeration above, the age is
  what decides whether the answer is the answer the tree would give; both
  members above survive a pass that finished a second ago, which is why each is enumerated
  rather than left to be read off the stamp.
- **The subtree restriction is one restriction, made once, above both storages.** It is a field of `Selection`, so a walk narrows to the directories it names and a query restricts both its arms by it. A stub with no subtree above it is matched by neither arm, because SQL's `IN` is false for NULL -- which is also how a bare scene name falls outside a walk of the selected volumes' directories. The subtrees are asked about one at a time rather than all at once, so that a subtree the results root holds no directory for costs only itself: a single request ends at the first subtree it cannot read, and on an enumeration over volumes nobody has navigated yet that would be every volume after the first.
- **The URL reaches the filter through the dataset layer, and only from a
  program that declares the option.** `_yield_image_files_index` takes a
  `results_index_db_url` keyword; when its caller passes none it resolves one through
  `get_results_index_db_url` and its `none` sentinel, but only when the arguments it
  was handed carry a `results_index_db` attribute, which is what declaring
  `--results-index-db` supplies. That is section 2.6's rule, and it is what keeps
  section 1's out-of-scope programs reading files: `sd_create_bundle`,
  `sd_consolidate_metadata` and `sd_backplane_viewer` all enumerate with the
  selection flags, and none of them declares the option or resolves a URL.
  `sd_offset` declares it in this phase, because this phase is what makes it a
  consuming program. Phase 4 adds it to `sd_backplanes` and `sd_mosaic`, which
  this phase therefore leaves alone.
- **`sd_offset` reports a refused selection rather than tracing back.** The
  selection arguments are finally read while images are enumerated, so that is
  where a contradictory pair, or an index that cannot be opened, cannot be read,
  or does not cover this root, is first diagnosed. Each already carries a message
  saying what to change, and an index URL can carry a database password, so the
  enumeration is wrapped once and the message is reported through `MAIN_LOGGER`
  with an exit status. The refusal is a `ValueError` subclass of its own,
  `SelectionError`, raised by `ResultsFilter` for the flags and at its seam
  boundary for everything the index refuses with: catching plain `ValueError`
  around a whole enumeration would report a bad volume name,
  a value a label would not yield, or a caller error as advice about what to
  change, and would swallow the traceback that says where it is.
- **Nothing from the database layer escapes the seam into an enumeration.** `open_index` makes every way of failing to open the index a `ValueError`; the queries after it are outside that guarantee, and a table the account may not read, a partially restored database, or a connection lost between the open and the query would otherwise reach an enumeration as `sqlalchemy.exc`'s own types, which no caller should have to name in an `except` clause to report a bad `--results-index-db`. The seam translates them, masked URL and driver message included, and `ResultsFilter` turns the result into `SelectionError` at its own boundary, so that a program reporting a refused selection catches those and not every other `ValueError` an enumeration can raise.

### Phase 6 — Documentation

New `docs/user_guide/user_guide_results_index.rst` (added to the
`user_guide.rst` toctree): what the index is and that it is optional; the
snapshot semantics and staleness; the URL forms, the `postgres` extra, and
the SQLite-is-local rule; the ingest workflow, including that it is never
automatic; root normalization and the not-ingested failure; the version gate
and delete-and-re-ingest; which programs consume the index, that
`sd_stats_report` requires it, and which programs deliberately do not use
it. `docs/api_reference/api_results_index.rst` -- created with the package in
Phase 1, so the branch never carries an undocumented public package -- gains
the modules the later phases add. Updates: `user_guide_statistics.rst` (per section 2.9),
`user_guide_logging.rst` (program table), `introduction_configuration.rst`
(`environment.results_index_db`), and a dev-guide section covering the Core
layer, the concurrency model, the database line the seam is split along, and
how to add a column (raise the schema version, rebuild every index). No issue
numbers in any of it.

Details settled during execution, none of them a change of intent:

- **The dev-guide section is a chapter of its own,**
  `docs/dev_guide/dev_guide_results_index.rst`, rather than a section inside an
  existing chapter. The dev guide is
  organized one chapter per subsystem, and the four subjects this phase names --
  the Core layer, the concurrency model, the import exception and the column
  recipe -- are that subsystem's contracts. It sits between the C-kernel and the
  PDS4 chapters, where the subsystem sits in the pipeline.
- **`docs/api_reference/api_results_index.rst` needed nothing.** Phase 1 created
  it with the package, and each later phase added its module to it in the commit
  that added the module, which is what the phase asked for; the check here found
  all eight names already present. Nothing was added for
  `spindoctor.cli.results_index`: whether the `spindoctor.cli` subpackages belong
  in the API reference at all is an open question (#443), and answering it for
  one package in a documentation phase would settle it by accident.
- **`user_guide_logging.rst` needed nothing either.** The program table gained
  the three statistics programs when each was written, since that table is
  validated against the program registry.
- **The two guides divide by audience rather than by subject.** The new user-guide
  chapter is the index as a whole -- what it promises, which programs read it,
  when to rebuild it -- and the statistics chapter stays the reference for the
  two programs and the schema, each pointing at the other. Writing the whole
  subject twice was the alternative and is how the two come to disagree.
- **The nitpicky build was run** (`sphinx -n`) and the two chapters this phase
  adds produce no warnings of their own. The repository-wide nitpick backlog is
  untouched and remains #438's.

### Phase 7 — One record seam, over both storages

`spindoctor/nav_records/` plus an index-backed half, answering the questions the
programs actually ask: `record(stub)` for one image, `records(selection)`
yielding a stream, `facts(selection)` yielding every field of every image, and
`listing(selection)` yielding keys with the size and modification time of each.
Above the seam, one implementation of everything; below it, batching.
`volume -> subtree` folded in, since the schema version moves either way.

**A listing answers a selection that names stubs, and picks its own way to.** A
stub is the identity of a file rather than something the file says, so it is not
a restriction a listing has to refuse, and it is the question a caller
enumerating candidate images asks. The index answers it with one keyed query per
batch. The tree answers it by checking the named files on a local root and by
walking the directories they lie in on a remote one, and decides on
`FCPath.is_local()` rather than on a ratio, because what differs is what one
call costs: a syscall against a paid round trip. Measured on a local root over a
volume of fifty thousand documents, checking beats walking by 600x or better at
ten files named, by 200x at a hundred, by 24x at a thousand and by 2.6x at ten
thousand, and loses by about half at every document in the volume; about seventy
per cent of the check's cost is the storage layer's per-path machinery rather
than the syscall. A walk made to answer one batch answers every later batch of
the same run, keyed by root and by the directory walked, and is released when
the source is closed. The choice lives inside the seam: a caller has one way to
ask what a root holds, and two shapes in the callers would be two answers to
keep true of each other. An entry a check produced carries neither metric and
says so through `ListedRecord.has_metrics`, since a stand-in for either would
make a changed document look unchanged.

**The package is split along the database line, and the line is forced.**
`import spindoctor.nav_records` must not import SQLAlchemy: it is the
storage-free half of the seam, so a program that can open no index still reads
its records through it, and a database layer imported anywhere under it would be
acquired by every one of them. So `spindoctor/nav_records/` -- the record types,
the document rules, root identity, `Selection`, the `RecordSource` protocol and
the tree backend -- imports no database layer at all.
`spindoctor/results_index/record_source.py` holds `IndexRecordSource` and
`open_record_source`, which returns the tree backend when no index URL was
resolved and the index backend when one was. A subprocess probe pins the rule,
and also asserts that the walk itself loaded, since a guarantee about a package
that imported almost nothing is a guarantee about nothing.

**`Selection` is where a caller is refused.** It is frozen, and its constructor
checks everything it carries: each subtree one directory immediately under a
root, each stub a key rather than a path -- no absolute fragment, no `..`, no
null byte -- and each time bound a finite number with the start no later than
the stop. That is what makes the two backends refuse alike. A walk and a query
cannot be made to refuse identically by writing the same refusal twice; left to
a backend, an unusable value is refused in that storage's own terms, and an
inverted range simply selects nothing, which a run cannot tell from a clean pass
over a quiet span. `normalize_root_url` is the same rule for the root, one place
above everything: `expanduser().resolve()`, so a `~`, a `..` or a symbolic link
is one spelling of the place it names, and a spelling that is not a location --
an empty one, or one carrying a null byte -- is refused where it is spelled.
Once the root is canonical and the key is
validated, `root / f'{stub}{METADATA_SUFFIX}'` is the same answer for every call
in the seam and no reader needs a path rule of its own.

**No `order` parameter, and its absence is a decision.** Neither storage can
promise a total order over a stream without giving up the streaming: a walk
cannot know an image's epoch before it has read the document, and a server sorts
text under its own collation. Each backend yields in the order it finds records,
documents that order, and a caller wanting a total order calls `sorted()`. The
parameter arrives with the consumer that can use it (#513).

**Two walks are deleted, and one of them was a defect.**
`spindoctor/support/nav_document.py` goes entirely: its `read_documents` walked
with `rglob`, which skips a directory it cannot list and reports a clean run --
measured at 1 record of 2, silently, against the ingest's hard refusal over the
same tree. `spindoctor/cli/stats/ingest/walk.py` goes too, its strict walk
having moved into `nav_records/tree.py` where every reader of a tree inherits
it; the ingest's `_listing_of_root` is what is left, and it is the accounting a
pass keeps around a listing rather than a listing of its own.
`resolved_document_path` goes with the module, replaced by the canonical root
and the validated key. `INGEST_RETRIEVE_BATCH_SIZE` goes, replaced by
`RETRIEVE_BATCH_SIZE` in the seam, so the seam and the ingest retrieve in the
same groups. `_midtime_of` moves out of `cli/ck/inputs.py` into
`support/nav_record.py` as `record_midtime_et`, so the seam's time filter and
the kernel writer read one rule.

**A directory nobody can list ends a walk; a root nobody can list is charged to
that root.** `UnlistableRootError` is a subclass of `UnlistableDirectoryError`,
which is what lets the ingest catch the second to keep its per-root accounting
while every other consumer lets either end its run. Both carry the directory
they are about, so a caller can tell the directory it asked about from one
further down: a listing of named stubs answers "no document" for a directory the
root does not hold, which is what a check of a file under it answers, and still
refuses one further down, where the stubs it did find would be read as the whole
of a directory it did list.

Tests: the three calls against both backends over one fixture tree and the index
ingested from it, with the declared disagreements between them each pinned
rather than left to be discovered; every `Selection` refusal, each asserted on
its message; the strict walk's refusals, including the directory that ends a
walk, the root that does not, and the directory already listed under another
name; a two-root fixture whose
second root differs in the value under test for every query the seam writes or
touches, on the SQLite tier and on the `postgres` tier for both the images and
the refusals arm; and the two import probes above. Adversarial review
reconstructed the deleted ingest walk from git and ran 33 refusal scenarios
against both, finding every refusal identical down to the log message.

Details settled during execution, none of them a change of intent:

- **The ingest keeps its own retrieve-and-parse loop.** Only its *discovery*
  moves onto the seam. `cli/results_index/chunks.py` owns the refusal vocabulary
  stored in `failed_files.reason` and the "could not be retrieved" / "read but
  refused" split, which is bookkeeping this index makes and the seam does not;
  folding the rest in is #514.
- **The roots bind to the source, not to the selection.**
  `open_record_source(roots, ...)` binds them, a selection naming none covers
  every root the source holds, and `record(stub)` refuses on a source holding
  more than one, naming them: a bare stub does not say which root it belongs to,
  and an index-backed source checks each root's ingest bookkeeping once at
  opening rather than per query.
- **The prune's license became a type rather than a check.** One function builds
  the listing the prune takes, it builds one only for a root it listed entirely,
  and the listing carries the root it was of -- so a share of a root, a partial
  listing, and a prune of one root on the evidence of another are all
  unrepresentable. The runtime `ValueError` that used to refuse them went with
  the condition.
- **A stub both tables record is one file, read as the record it is.** The
  ingest writes the two tables independently and a queue-divided pass can leave
  a stale refusal beside a record, so all three index-backed calls prefer the
  record: the per-image lookup reads both halves of the key, the stream naming
  its own stubs puts the records in last, and the whole-root stream and the
  listing exclude a refusal whose key also carries a record.
- **Nothing from the database layer escapes this seam either.** A selection the
  index cannot answer is refused by `Selection` before a statement is built, and
  a query that fails is reported as a `ValueError` naming the index, so a
  consumer that imports no database layer never has to name one of its types.
- **The logging invariant is amended rather than weakened.** `nav_records` joins
  the packages that construct no logger of their own, `pdslogger` leaves the
  forbidden import set for those two packages so a caller can lend one, and a
  static check forbids every log call in either package except the one line the
  walk owes an operator, which says it declined to descend a directory it had
  already listed under another name. The stdlib and SQLAlchemy bans are
  untouched. A logger a caller hands in is by construction one the caller
  configured; a logger a data-access layer reaches for is not.
- **The C-kernel writer's no-oops guarantee is deleted**, and its import probe
  with it. A kernel's portability is a property of the file rather than of the
  process that wrote it; recomputation is caught directly by the round trip that
  furnishes the written kernel and compares SPICE's answer to the recorded
  matrix; `import oops` furnishes nothing into the SPICE pool, so there is no
  contamination hazard; and `sd_create_ck` the program loads oops anyway through
  its own logging and configuration, so nothing was protected at runtime. The
  `nav_records` probe keeps its no-SQLAlchemy half, which is load-bearing, and
  drops its oops half.

Behavior changes review must see:

- **The kernel writer refuses a root it cannot list whole**, where it used to
  write a short mission and report a clean run. It also refuses a root it cannot
  list at all, where `rglob` yielded nothing and reported the same clean run. It
  inherits the walk's already-listed-under-another-name rule too. `sd_create_ck`
  reports both as a fatal line and exit 1, before any kernel, meta-kernel or
  report is written.
- **Every existing index must be dropped and rebuilt**, for the `volume ->
  subtree` rename and the schema version it carries.

---

## 5. Acceptance criteria

Every criterion names what establishes it: a test as `path::function`, or a
command line and what its output says. The default tier is what a plain
`pytest -n 4 --dist=loadfile` runs; the `postgres` tier needs a server named by
`SPINDOCTOR_TEST_POSTGRES_URL` and is selected with `-m postgres`; the
integration tier needs the holdings and a local SPICE tree and is selected with
`-m integration`. A criterion that does not hold says so.

1. `sd_backplanes`, `sd_mosaic`, and the `ResultsFilter`-driven selections
   produce identical products and identical selections for the same inputs
   with and without an index, over a fixture tree exercising success,
   failure, error, missing-metadata and malformed-metadata images. Asserted
   by tests (default tier at the `PointingSelection`/selection level;
   integration tier on written products, where the evidence below records that
   the criterion is not met). "Identical" binds returned values, written
   products, and the reachable-reason warnings -- not incidental log text.
   The criterion binds every image whose document the ingest could read.

   Three carve-outs. The first is the reason vocabulary section 2.9 maps, whose
   unreachable rows are a stated behavioral difference in the name and not in
   the product.

   The second is **a document the ingest refused**, which is the one input the
   product programs do not answer identically for. They do not answer
   differently either: the image fails, naming itself, the index and the
   recorded reason, and the pass goes on. That is a refusal rather than a
   divergence, and it is what the criterion requires here -- an image whose
   corrected product one storage builds and the other cannot must never be
   built either way in silence. Section 2.9 states the rule, the
   `pointing_source` module docstring states it, the reprojection and backplane
   guides state it, and tests pin both directions: the refusal, and the image
   nothing navigated that must still read as such.

   The third is, **for the selections**, what section 4's Phase 5 entry
   enumerates, restated here member for member and in its order, so that a
   reader of this criterion sees the list rather than a sample of it:

   1. a file that has no row at all in the index, because no pass could read or
      record it;
   2. a document rewritten in place with the length and the modification time it
      had before, whose row goes on recording what the document before it said.

   Each carve-out is stated in the plan, in the module docstring, in the
   navigation guide, and in a test, and one found later is added to all four in
   one commit; a test binds each member's identifying phrase to the entry
   carrying it and compares the four lists entry for entry, so a member dropped
   from one of them, added to one and not the others, or stated out of order
   fails. It matches text, so it cannot tell whether an entry says something
   true about its member; that is what each member's own behavioral test is for.
   Neither carve-out is asserted to be complete: a divergence outside them is a
   defect of the enumeration, to be fixed or enumerated, and not a license to
   differ.

   `sd_stats_report`'s form of this criterion is that one report comes out of a
   results tree and out of an index ingested from that tree, byte for byte, and
   that both reproduce the frozen output. `images.csv` is held to the same rows
   rather than to the same bytes: the export writes each row where it reads it,
   so its line order is the order the storage found the images in and is not a
   contract.

   Evidence, default tier:

   - the selection matrix, every flag over the one fixture tree, each held to a
     stated answer --
     `tests/spindoctor/dataset/test_results_filter_index.py::test_the_tree_answers_each_filter`
     and `::test_the_index_answers_each_filter_the_same_way`, with
     `::test_the_index_path_reads_no_file_at_all`;
   - the `PointingSelection` level --
     `tests/spindoctor/cli/reproj/test_pointing_source.py::test_both_paths_select_the_same_mechanism`,
     `::test_both_paths_report_the_same_reason`,
     `::test_both_paths_carry_the_same_fallback_offset` and
     `::test_the_image_log_says_the_same_thing_either_way`;
   - the second carve-out, both directions --
     `tests/spindoctor/cli/reproj/test_pointing_source.py::test_a_document_the_ingest_refused_is_not_reported_as_an_unnavigated_image`
     and `::test_an_image_nothing_navigated_is_still_reported_as_unnavigated`;
   - the third carve-out, member by member --
     `tests/spindoctor/dataset/test_results_filter_index_divergence.py::test_a_file_the_pass_could_not_retrieve_reads_as_absent`
     and
     `::test_a_document_rewritten_in_place_reads_as_its_previous_self_in_the_index`;
   - the four lists compared entry for entry --
     `tests/spindoctor/results_index/test_enumeration_lists.py::test_every_list_states_each_member_in_an_entry_of_its_own`
     and `::test_no_entry_of_a_list_states_anything_outside_the_enumeration`,
     each parametrized over the four ids `module-docstring`,
     `navigation-guide`, `plan-phase-5` and `plan-criterion-1`, so this
     criterion's own list is one of the four the guard binds together;
   - `sd_stats_report` --
     `tests/spindoctor/cli/stats/test_report_over_a_tree.py::test_the_report_is_byte_identical_to_one_from_an_index`,
     `::test_the_tree_reproduces_the_frozen_report` and
     `::test_the_csv_export_holds_the_same_rows`, with
     `::test_the_report_is_not_empty` as the control an equality needs -- two
     empty reports compare equal -- and
     `tests/spindoctor/cli/stats/test_report_regression.py::test_the_report_is_byte_identical`
     for the frozen output itself.

   **The written-product half is not evidenced.**
   `tests/integration/test_results_index_consumers.py` is what states it, over
   real library frames navigated for real and ingested by the real ingest. Run
   with the holdings and a local SPICE tree, its Galileo frame's eight
   assertions pass and its Cassini ring frame's thirteen error in setup: that
   frame reaches `no_features_extracted` and writes no successful document, so
   the comparison has nothing to read. Every written-product assertion -- the
   backplane arrays, the ring reprojections and the ring mosaics -- belongs to
   that frame, so no product is compared today. The failure is the image
   library's rather than the index's; it reproduces on `main` for the same
   frame, and it is tracked as its own issue.
2. No pipeline program requires an index, and `import spindoctor.nav_records`
   -- the storage-free half of the record seam, which every reader of a results
   tree goes through -- does not import `sqlalchemy`. Asserted by a subprocess
   probe, which also asserts that the walk itself loaded, since a guarantee
   about a package that imported almost nothing is a guarantee about nothing.
   The claim is about what that package's own import pulls in; nothing here
   forbids a caller from importing a database layer.

   Evidence, default tier:
   `tests/spindoctor/nav_records/test_imports.py::test_the_record_seam_loads_no_database_layer`,
   with `::test_the_probe_really_imported_the_walk` and
   `::test_the_probe_imported_the_package_under_test` pinning that the probe
   answered for this checkout's walk rather than for an empty import of an
   installed copy.

   That no program requires an index is asserted twice over. Narrowly, by
   `tests/spindoctor/cli/test_results_index_db_argument_surface.py::test_no_option_and_no_variable_means_no_index`,
   where an argument, the configuration variable and the environment variable
   all being absent resolves to no index rather than to an error, and by
   `tests/spindoctor/cli/reproj/test_pointing_source.py::test_no_url_reads_documents`,
   where the same program then reads the tree. Broadly, by `tests/conftest.py`'s
   autouse `no_ambient_results_index_for_the_session` and
   `no_ambient_results_index`, which unset `NAV_RESULTS_INDEX_DB` and run from a
   directory holding no `nav_default_config.yaml`: the whole default tier,
   every pipeline program's tests included, runs with no index resolvable
   anywhere.
3. A program given an unopenable, nonexistent, or version-mismatched index,
   or a root the index has not fully ingested, fails with a message naming
   the cause; it does not fall back to reading files and does not create an
   empty database.

   Evidence, default tier:
   `tests/spindoctor/results_index/test_engine.py::test_a_consumer_refuses_a_database_that_does_not_exist`,
   `::test_a_consumer_does_not_create_the_database_it_refused`,
   `::test_a_database_stamped_with_another_version_is_refused` and
   `::test_the_version_message_names_the_version_this_code_reads`;
   `tests/spindoctor/results_index/test_roots.py::test_the_refusal_masks_its_index_and_leaves_its_roots_alone`,
   which is the refusal itself, naming the root and the masked index; and at
   the program level
   `tests/spindoctor/cli/reproj/test_pointing_source.py::test_an_unopenable_url_fails_rather_than_falling_back`
   with `::test_an_unopenable_url_leaves_no_database_behind`,
   `::test_a_root_nobody_ingested_is_refused` and
   `::test_the_refusal_names_the_roots_the_index_does_hold` for a root the
   index never heard of, and `::test_a_run_that_died_halfway_leaves_a_root_unreadable`
   for one whose newest ingest never finished, which is what "not fully
   ingested" means: the run's rows are whatever the dead pass wrote, so absence
   under that root answers nothing.
4. Two images with the same basename in different volumes produce two rows
   and are independently retrievable; a multi-root index serves each
   consumer only rows from its own normalized root.

   Evidence, default tier:
   `tests/spindoctor/cli/results_index/test_ingest.py::test_two_volumes_with_one_basename_produce_two_rows`
   and `::test_each_colliding_image_is_independently_retrievable`;
   `tests/spindoctor/cli/reproj/test_pointing_source.py::test_a_source_reads_its_own_roots_offset`,
   `::test_the_other_source_reads_the_other_roots_offset` and
   `::test_a_stub_recorded_only_under_another_root_is_absent`. The last three
   run over a two-root fixture whose second root differs in exactly the value
   under test, so one row cannot satisfy both directions. Dropping the root
   half of the join in `IndexRecordSource._row` fails two of the three --
   `::test_the_other_source_reads_the_other_roots_offset` reads the first
   root's offset for the second, and `::test_a_stub_recorded_only_under_another_root_is_absent`
   finds a record where there is none.
5. The stored offset round-trips the document's top-level `offset`
   bit-exactly at 15 significant digits, and `status_error` is retrievable
   verbatim.

   Evidence, default tier:
   `tests/spindoctor/cli/results_index/test_ingest.py::test_an_offset_survives_the_database_bit_for_bit`,
   which ingests `3.14159265358979` and reads the same value back out of
   `images.offset_dv`; and
   `tests/spindoctor/results_index/test_schema.py::test_status_error_is_retrievable_verbatim`
   with `::test_status_reason_is_stored_separately_from_status_error`, which is
   what keeps the two vocabularies from merging into one column again.
6. A second ingest over an unchanged tree reads no metadata file and issues
   no per-file stat or exists call, proven by counting.

   Evidence, default tier:
   `tests/spindoctor/cli/results_index/test_ingest.py::test_an_unchanged_file_is_not_read_again`
   wraps `FCPath.retrieve` and requires the list of retrievals the second pass
   made to be empty; `::test_an_unchanged_file_is_not_stat_ed_either` wraps
   `FCPath.stat`, replaces `FCPath.exists` with a call that raises, and requires
   no recorded path to be a metadata file, having first required the pass to
   report one file skipped; and
   `::test_a_first_ingest_asks_about_no_single_file_either` holds the first pass
   to the same rule, so the second pass's silence is not a listing that asked
   about every file once already.
7. Ingest on a cloud-style root downloads its files; the test fails if the
   download call is reverted to `get_local_path()`.

   Evidence, default tier:
   `tests/spindoctor/cli/results_index/test_ingest.py::test_a_cloud_style_root_downloads_its_files`
   and `::test_a_cloud_style_root_reads_the_downloaded_document`. The fixture
   root holds a file of the right name, size and modification time whose
   contents are not the document, and the retrieval is redirected to the
   directory holding the real ones, so a caller that named a file instead of
   retrieving it reads the placeholder. Reverting
   `spindoctor/cli/results_index/chunks.py` to `get_local_path()` fails both,
   the first on `files_ingested == 0` and the second on an empty row set.
8. Concurrent local SQLite ingest workers produce the same rows as one
   serial ingest over the same input.

   Evidence, default tier:
   `tests/spindoctor/cli/results_index/test_ingest_cloud_tasks.py::test_concurrent_shares_write_the_rows_a_single_pass_writes`
   runs six shares of twelve documents through four threads, each share with an
   engine of its own, and compares every `images` row against a serial
   `ingest_metadata_files` over the same tree;
   `::test_concurrent_shares_lose_no_feature_row` covers the child rows, which
   are written second and are what an interleaved image would lose.
9. `sd_results_index_cloud_tasks` writes zero bytes to stdout and stderr
   from SpinDoctor code under a worker subprocess, and its counts arrive in
   the task result.

   Evidence, default tier:
   `tests/spindoctor/cli/test_cloud_task_silence.py::test_an_ingest_task_writes_nothing_to_stdout`
   and `::test_an_ingest_task_writes_nothing_to_stderr`, measured on the
   descriptors of a real subprocess that calls `logging.basicConfig` first, so
   the root handler a worker installs is present; and
   `::test_an_ingest_task_still_reports_what_it_ingested`,
   `::test_an_ingest_task_still_names_what_it_refused` and
   `::test_an_ingest_task_still_reports_what_it_skipped` for the counts, which
   are the whole of what an ingest worker says.
10. No SQLite-only construct remains in any query: no UDF registration, no
    `TOTAL(`, no `executescript`, no `PRAGMA` outside a dialect event, no
    integer comparison or arithmetic against a Boolean column. Enforced by a
    source-scanning test whose roots are named as directories rather than as
    module lists, so a module added to any of them is covered without being
    named. Three roots, each for a reason of its own:
    `src/spindoctor/results_index/`, the Core layer, which holds the schema,
    the opener and the root bookkeeping; `src/spindoctor/cli/results_index/`,
    the programs that write and empty an index, which is where the statements
    that fill and prune it live; and `src/spindoctor/cli/stats/`, the report,
    which issues no statement of its own -- it formats one pass of accumulators
    fed from the record seam -- and is scanned so that a statement written there
    is read rather than missed. A scan aimed at one package while the queries
    live in another reports green without reading them.

    Evidence, default tier: the five parametrized checks of
    `tests/spindoctor/results_index/test_source_constructs.py` --
    `test_no_module_registers_a_python_function_as_sql`,
    `test_no_module_uses_the_total_aggregate`,
    `test_no_module_runs_a_statement_script`,
    `test_every_pragma_lives_inside_a_connect_event` and
    `test_no_module_compares_a_boolean_column_against_an_integer` -- each run
    once per module found under the three roots. What keeps an empty scan from
    passing is `::test_the_scan_actually_reads_some_modules`, which asserts a
    floor of named modules from all three roots;
    `::test_the_scan_reaches_the_statistics_programs` and
    `::test_the_scan_reaches_the_programs_that_write_an_index`, which pin those
    two roots as directories, since the floor is matched on base names and
    several of them have a namesake in the Core layer; and
    `::test_the_boolean_scan_knows_which_columns_are_boolean`, which reads the
    Boolean column names off the schema.
11. The suite's index tests pass against PostgreSQL under a `postgres`
    marker: registered in `[tool.pytest.ini_options].markers`, excluded by
    default via `addopts` (`-m "not integration and not postgres"`), given a
    `scripts/run-all-checks.sh` flag alongside `-i` (`-P` / `--postgres`), and
    named in the dev guide as a locally-runnable tier if CI has no service
    container.

    Evidence: `pytest -m postgres -n 4 --dist=loadfile -rs` with
    `SPINDOCTOR_TEST_POSTGRES_URL` exported, which selects the tier and reports
    every test passed and none skipped. The four surface facts are the
    `markers` and `addopts` tables of `[tool.pytest.ini_options]` in
    `pyproject.toml`, the `-P|--postgres` case in `scripts/run-all-checks.sh`
    -- which refuses `-P` outright when no server is named, since a tier that
    skipped every test would report as a pass -- and the tier's account in
    `docs/dev_guide/dev_guide_testing.rst`.
12. `ruff check`, `ruff format --check`, `mypy --strict`, `sphinx-build -W`
    and `pymarkdown scan` all pass; suite coverage stays at or above 90%.

    The five gates pass. Evidence, each from the repository root:
    `ruff check src tests`, `ruff format --check src tests`, `mypy src tests`,
    `rm -rf docs/_build && sphinx-build -W -b html docs docs/_build`, and
    `pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md`. Each exits 0
    and reports no finding; `scripts/run-all-checks.sh` runs all five together
    with the suite. `plans/` is deliberately outside the pymarkdown scan and is
    not clean under it.

    **The coverage half does not hold.** `pytest --cov` runs, so the number is
    measurable rather than unknown, and over the tier CI measures --
    the default one, `-m "not integration and not postgres"` --
    `pytest --cov -n 4 --dist=loadfile` reports **79%** with branch coverage on
    and 81% of statements. Adding the `postgres` tier moves neither figure.
    Nothing enforces the floor either: `.coveragerc` sets no `fail_under` and
    `codecov.yml` sets both targets to 0, so the figure sits below 90 without
    failing anything. The shortfall is the PyQt6 code: `src/spindoctor/ui/` and
    `src/spindoctor/cli/sd_backplane_viewer.py` account for about 4,470 of the
    7,511 unexecuted statements between them. Raising the number, or ratifying
    a lower floor and enforcing that one, is tracked as its own issue.

---

## 6. Risks and constraints

**The snapshot can be stale, by design.** A consumer reading an index that
predates the newest navigation run gets the older answer with no warning.
The mitigation is documentation and the `ingest_runs` timestamps, not
machinery. Anyone tempted to add staleness detection later should note that
it reintroduces per-image tree access, which is the cost this work removes.

**Absence is ambiguous unless bookkeeping is right.** "No row for this stub"
means "not navigated" only if the root was fully ingested. That is what
`ingest_runs` is for, and it is load-bearing rather than cosmetic -- hence
the hard failure on a root with no completed row, and the root
normalization rule, without which the comparison generates false failures.

**One writer assumption in SQLite.** WAL plus short transactions makes
multiple local writers safe; it does not make a shared network filesystem
safe. The cross-machine case is PostgreSQL, and the lockability probe
refuses the SQLite URL rather than corrupting the file.

**A required dependency.** `sqlalchemy>=2.0` is a runtime dependency rather
than an optional extra. The statistics programs and `results_index` import it
whether or not an index is used, and so does `spindoctor.dataset`, which answers
a selection through the seam. That last one was measured rather than assumed:
importing `spindoctor.dataset` already loads oops (+1695 modules, 0.790s) where
`import sqlalchemy` adds 83 modules and 0.053s, on a path that then does SPICE
and image work taking seconds per image. The one package that must not acquire
it is `spindoctor.nav_records`, because it is the storage-free half of the seam
and has to work where no index exists -- criterion 2 pins that.

**Schema changes cost every user a rebuild.** That is the accepted trade for
having no migration machinery, and the reason the schema carries the
corrected-pointing columns in the shapes their producer writes rather than
leaving them for a later migration.

**The report is the regression surface.** Phase 2 touches every query the
statistics user guide documents. The old-vs-new byte-identical report test
is the only thing standing between "migrated" and "silently different";
treat a diff there as a defect, not as an acceptable drift.

---

## 7. Follow-ups

File as tracking issues alongside the implementation issue:

- **A document column for bundle generation** (#464). Storing the raw metadata JSON
  per image (SQLite TEXT, PostgreSQL `jsonb`) is the only way bundle
  generation stops reading one document per image, since it needs the whole
  document. It roughly doubles the index size, which weakens shipping the
  SQLite file to workers, so it likely belongs in a separate optional
  database rather than the main index.
- **Serving the curation triage tool** (#465). Needs `missing_frac`,
  `saturation_frac`, classifier flags, the per-technique list, result file
  paths, a name-to-stub lookup, and a second (rescue) root -- a schema
  widening for one out-of-package tool, wanted only if triage's ten
  rglobs-per-frame remain a practical pain after the pipeline consumers are
  converted.
- **`sd_create_ck` reads one document per image** (#507), **closed.** It reads one
  mission's records in bulk now, and it reads them through the seam every program
  reads records through, so converting it unified the back end rather than adding
  a second one: `spindoctor/results_index/record_source.py` answers both shapes --
  one image by its stub, one mission in bulk -- over either storage, and
  `spindoctor/results_index/rebuild.py` holds the one column-to-field
  correspondence both shapes rebuild through. The program declares
  `--results-index-db` like every other index-backed one. Three of the fields it reads
  were not columns and were added with it -- `observation.shutter_mode`, which is
  what detects a simultaneous exposure;
  `navigation_result.provenance.spice_kernels`, which assigns a correction to the
  original it overlays; and `navigation_result.pointing.camera_frame`, which the
  earlier phases left out on the grounds that no reader consulted it. That is
  what raised the schema version to 8.
- **Two rebuilds of one row were two answers**, **closed with the phase above.**
  Each consumer carried its own row-to-record rebuild, its own open-and-check
  ceremony and its own account of how the two storages could differ. They agreed
  when they were written and then each grew a rule the other did not: the
  `ORDER BY` that was right under SQLite's collation and wrong under a
  PostgreSQL locale collation could only exist in the consumer that read in
  bulk. There is now one rebuild, one `open_index_for_roots`, and one place that
  states what the two storages may differ about; every column of `images` is
  asserted to be a record field, part of a row's identity or a value the ingest
  derived, so a column added for one consumer cannot read as absent to the rest.
- **Shipping the index to cloud workers** (#466). Publishing the SQLite file to the
  results bucket and having each worker download it once is the alternative
  to a PostgreSQL instance, and needs a documented workflow either way.
- **A `--since` selector for ingest** (#467). The stat-pair skip makes a re-scan
  cheap in reads but not in listings; a time-bounded scan would cut the
  listing too.
- **Two overlapping ingest passes over one root can leave a stale row** (#479),
  **closed as a documented limit.** A worker of the first writes a stub after
  the second has read what is recorded and before its delete, for a document
  that left the tree between the two listings. Narrow, self-healing on the next
  pass, and invisible to consumers while it is open, since both runs are
  unfinished. Refusing or warning about a fan-out over a root whose newest run
  is unfinished was considered and not done; the statistics guide says so in
  those terms, so that a later reader sees a choice rather than a question
  nobody reached.
- **A hand-written ingest task file can name a stub outside its root** (#489).
  A task file is an operator-visible artifact, and the worker accepts any string
  as a stub, so a hand-edited one reads a document outside the root it names and
  reports it under that root; a share repeated under a second task identity is
  counted twice, which is the one route left to an account that reaches its
  listing with a share unrun. Neither is reachable from a fan-out, and the root
  half of the key was closed for the same threat model; what is undecided is
  whether the worker should validate the stub domain as the log-path and offset
  readers already validate theirs.
- **An abandoned fan-out has already removed rows** (#480), **closed as a
  documented limit.** The prune runs before any document is read, so a pass that
  is given up on after step 1 has shrunk the index. Only rows whose documents
  have genuinely left the tree go, and the run is unfinished throughout, so
  nothing valid is lost and no consumer reads the root. The statistics guide
  states the operator-visible consequence, which is that abandoning a fan-out
  costs a full re-ingest of that root.
- **One unlistable directory stopped the prune for the whole root** (#481),
  **closed.** A pass that could not list one directory completed all the same
  and removed no row anywhere under its root, so a deleted document went on
  reading as present across any number of finished passes. The walk now raises
  where it meets such a directory: the pass ends, the root keeps no finish time,
  and every run that does complete listed its whole root and pruned. A directory
  reached a second way is no longer counted as a gap either, since its documents
  are in the listing under the path met first. `ingest_runs.directories_missed`
  and everything that reported it went with the change, which is what raised the
  schema version to 7.
- **A document rewritten in place with the same length and modification time is
  never read again** (#488). Those two metrics are everything a listing supplies,
  so `_is_unchanged` cannot tell such a file from the one already read, and its
  row goes on recording what the document before it said however many passes
  complete. Phase 5 enumerates it and tests both directions of it, and `--force`
  is the remedy; distinguishing it without one means either a cheap identity the
  storage layer already has (an object-store ETag) or a content digest paid for
  by retrieving every document on every pass, which is the cost the skip exists
  to avoid.
- **The ingest still has its own retrieve-and-parse loop beside the seam's**
  (#514). Only discovery moved onto the seam. `cli/results_index/chunks.py` reads
  and parses what the listing found, because it owns the `failed_files.reason`
  vocabulary and the "could not be retrieved" / "read but refused" split, which
  is bookkeeping the index makes and no other consumer wants. Folding it in
  means the seam yielding a refusal reason precise enough for the table.
- **A `Selection` order parameter** (#513). Neither backend can promise a total
  order over a stream today, so a caller wanting one calls `sorted()`. The
  parameter arrives with the consumer that can use it -- the kernel writer,
  reading a time-ordered stream out of the index rather than holding a mission.
- **A cloud-share ingest can write another root's document into this root's
  rows** (#515). Pre-existing, and not closed by validating a `Selection`:
  `_share_from_task` reads the stub straight out of the task JSON and never
  builds one. The fix is to call `stub_refusal` in its existing per-entry loop.
  It is the same threat model as #489 and the two should be answered together.
- **`test_a_share_only_writes_its_own_root` cannot fail** (#516). Pre-existing.
  Its docstring names the defect it misses and it asserts on the row a
  root-blind write would keep, so the mutation it exists to catch leaves it
  green.
- **S3 directory listings are not paginated** (SETI/rms-filecache#65), which is
  a dependency rather than a follow-up. `FileCacheSourceS3.iterdir_metadata`
  issues one `list_objects_v2` and reads that single response, so a prefix
  holding more than a thousand objects lists short and says nothing. The seam
  assumes `iterdir_metadata` returns a complete listing or raises, and there is
  deliberately no workaround here and no mention of it in the guides: a
  mitigation in this repository would have to be undone when the storage layer
  is fixed, and a listing that is complete or raises is the only contract a
  completeness guarantee can be built on. GS auto-paginates and is unaffected.
- **The lockability probe takes a write lock on a consumer's open** (#462).
  Section 2.5 has it refuse in both modes, so a consumer opening a SQLite index
  while an ingest holds a write transaction waits out the busy timeout and can
  then fail with the filesystem-and-PostgreSQL message though nothing is wrong. It
  is the plan's own rule and is left as written here. Cloud-task ingest makes
  the collision routine rather than occasional: a worker opens the index once
  per task, so a local SQLite run of a thousand tasks takes a thousand write-lock
  probes against peers holding chunk transactions. Nothing here measures what
  those probes wait, and no test does; the question of whether a `create=False`
  open should probe with a read instead is tracked on its issue, with what this
  work changed about it recorded there.

---

## 8. Execution protocol

1. Branch `rf_results_index` off current `main`; one commit series per
   phase.
2. Per phase: dispatch an **implementer subagent** (Opus-class) whose prompt
   embeds that phase's section of this plan verbatim plus sections 1-3, so
   the subagent needs no other briefing and does not have to locate this
   file. Then dispatch an **independent, fresh-context adversarial
   reviewer** (also Opus-class) with the diff, the same plan sections, and
   instructions to (a) verify each normative statement of section 2 against
   the code line by line, (b) run the phase's tests plus
   `ruff check src tests`, `ruff format --check src tests`, and
   `mypy src tests`, (c) confirm the no-index path is genuinely unchanged,
   and (d) hunt for convention violations and unstated deviations from this
   plan. Fix rounds until the review is clean; the controller, not the
   implementer, judges cleanliness.
3. The reviewer must verify each guarantee by **breaking the source and
   confirming a test fails**. A test that passes against a deliberately
   broken implementation is a defect in the test, and is reported as one.
4. Deviations discovered mid-phase -- an API that does not exist, a wrong
   assumption about a consumer -- are recorded in the phase commit message
   and reconciled into this plan file in the same commit, so the document
   the next reviewer holds is never stale. Scope changes go to the operator
   instead.
5. Final sweep before the pull request to `main`:
   `./scripts/run-all-checks.sh -i` plus the `postgres` tier, and a run of
   every consuming program over a real fixture tree with and without an
   index, with every output difference accounted for.
6. One pull request to `main`: summary, phase map, evidence, `Closes` per
   issue, and the plan and guide reconciliation included.
