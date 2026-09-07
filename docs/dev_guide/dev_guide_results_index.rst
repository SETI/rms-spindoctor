=================
The Results Index
=================

Overview
========

The results index is an optional database derived from the navigation results
tree. One ingest pass reads every ``_metadata.json`` document under a results
root and writes one row per image; consumers that need a few fields per image
then read a row instead of downloading a document. It is not authoritative:
the documents are, and the index can be dropped and rebuilt at any time.

:doc:`/user_guide/user_guide_results_index` is the operator's account of it ---
when to build one, how programs are pointed at one, what it promises. This
chapter is about the code: how the layer is put together, what it guarantees
under concurrency, and what changing it costs.

The code divides in two. The library package
:mod:`spindoctor.results_index` owns the schema, the opener and the queries
consumers issue; it is imported by anything that reads an index. The ingest
lives under the command-line package, in
``spindoctor/cli/results_index/``, because writing the index is a program an
operator runs rather than an API a consumer calls.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Responsibility
   * - :mod:`spindoctor.results_index.schema`
     - Every table, its columns and constraints, and ``SCHEMA_VERSION``.
   * - :mod:`spindoctor.results_index.engine`
     - :func:`~spindoctor.results_index.open_index`: backend selection, SQLite
       settings, the version gate.
       :func:`~spindoctor.results_index.open_database` is the one opener that
       stops before the gate.
   * - :mod:`spindoctor.results_index.scope`
     - Which schema of a database the index lives in, decided from the stamp.
   * - :mod:`spindoctor.results_index.masking`
     - The one rule that hides a password in a connection URL.
   * - :mod:`spindoctor.results_index.roots`
     - The per-root ingest bookkeeping that makes absence of a row readable,
       and the opener that checks it. Root normalization is a rule about
       identity rather than about a database, so it lives in
       :mod:`spindoctor.nav_records` and is re-exported from here.
   * - :mod:`spindoctor.results_index.rebuild`
     - The one correspondence between a row's columns and a record's fields, and
       the rebuild that reads it.
   * - :mod:`spindoctor.results_index.record_source`
     - The record seam's index-backed half, and
       :func:`~spindoctor.results_index.open_record_source`, which decides
       which half a run gets --- including the half that answers an
       enumeration's selection filters.
   * - :mod:`spindoctor.results_index.drop`
     - Reading what a database holds, and removing the index's own tables.
   * - :mod:`spindoctor.nav_records`
     - Not part of this package: the record seam's database-free half. What a
       record is, what a document is named and where one lives, what a
       selection is, the protocol both storages implement, the implementation
       over the documents themselves, and
       :func:`~spindoctor.nav_records.facts.facts_from_document`, which turns
       one document into the per-image shape both storages answer in. Reading a
       document needs no database, so every reader shares it whether or not its
       program can read an index.
   * - :mod:`spindoctor.dataset.results_filter`
     - Not part of this package either: the six results-based selection
       filters, answered through the seam. A listing of the selected subtrees
       settles which images have a document; the per-image facts of a batch of
       candidates settle what each one records. One implementation therefore
       serves both storages.
   * - ``spindoctor.cli.results_index``
     - The pass itself: walk, select, read, write, prune, complete --- in one
       process or divided into queue tasks.

The Core layer
==============

The index is written against **SQLAlchemy Core**, not the ORM. Tables are
``sqlalchemy.Table`` objects declared in one module against one
``sqlalchemy.MetaData``, and every read is a ``select()`` whose result the
caller consumes as rows. There is no mapped class, no session, no identity map
and no lazy loading.

That choice follows from what the index is for. Each consumer asks one narrow
question per run --- which stubs exist under this root, what does this stub
record --- and the answer is a set or a handful of scalars, never an object
graph to be navigated. An ORM would buy change tracking and relationship
loading that nothing here uses, and cost a mapping layer between the columns
and the rows that ingest writes from documents.

Two properties hold everywhere in the layer, and both are load-bearing:

**Every row is keyed by the pair** ``(root_url, results_path_stub)``. One
database serves several results roots, and two volumes hold images with the
same basename, so neither half identifies a row on its own. A query that
filters on the stub alone passes every test written against a single-root
fixture and answers another root's rows in production. Any new query needs a
fixture holding two roots whose second root *differs* in the value under test.

**Absence of a row is only meaningful once the root is known to be ingested.**
"No row for this stub" means "this image was never navigated" only after a
completed pass over that root; otherwise it means nothing at all.
:func:`~spindoctor.results_index.require_ingested_roots` is what every consumer
asks before it reads absence as an answer, and a completed run is necessary but
not sufficient: a document the ingest refused has a row in ``failed_files`` and
none in ``images``, so both tables are read before absence is reported.

One seam for records
====================

Several programs read what a navigation pass wrote, and each of them can be
pointed at either storage. What decides whether they agree is that none of them
reads a storage itself: they all go through
:class:`~spindoctor.nav_records.RecordSource`, and each storage implements it.
``sd_results_index`` is the one that reads only documents, since documents are
what it builds the index out of: it discovers them through the seam's listing
and keeps a reading loop of its own, which owns the refusal vocabulary the
index stores.

.. mermaid::

   classDiagram
      direction LR

      class RecordSource {
          <<protocol>>
          +record(stub) NavRecord
          +records(selection) Iterator[NavRecord | UnreadableFile]
          +facts(selection) Iterator[ImageFacts | UnreadableFile]
          +listing(selection) Iterator[ListedRecord]
          +describe() str
          +close()
          +\_\_enter\_\_() RecordSource
          +\_\_exit\_\_(exc_type, exc, traceback)
      }

      class TreeRecordSource {
          +\_\_init\_\_(roots, *, logger=None)
          +roots: tuple[str, ...]
      }

      class IndexRecordSource {
          +\_\_init\_\_(engine, roots, url, columns)
          +roots: tuple[str, ...]
      }

      class ResultsFilter {
          +needs_batch_filtering: bool
          +passes(results_path_stub) bool
          +filter_batch(image_files) list[ImageFile]
          +close()
      }

      class Selection {
          <<frozen dataclass>>
          +roots: tuple[str, ...]
          +subtrees: tuple[str, ...]
          +stubs: tuple[str, ...]
          +instrument: str | None
          +start_et: float | None
          +stop_et: float | None
          +bounded_in_time: bool
      }

      class NavRecord {
          <<frozen dataclass>>
          +path: FCPath
          +stub: str
          +metadata: dict[str, Any]
      }

      class ImageFacts {
          <<frozen dataclass>>
          +image: dict[str, Any]
          +techniques: list[dict[str, Any]]
          +feature_sources: list[dict[str, Any]]
      }

      class ListedRecord {
          <<frozen dataclass>>
          +stub: str
          +path: FCPath
          +mtime_ns: int | None
          +size_bytes: int | None
          +has_metrics: bool
      }

      class UnreadableFile {
          <<frozen dataclass>>
          +path: FCPath
          +stub: str
          +reason: str
      }

      RecordSource <|.. TreeRecordSource : reads documents
      RecordSource <|.. IndexRecordSource : reads rows

      ResultsFilter ..> RecordSource : lists once, then asks per batch
      ResultsFilter ..> Selection : writes

      RecordSource ..> Selection : asked in terms of
      RecordSource ..> NavRecord : record(), records()
      RecordSource ..> ImageFacts : facts()
      RecordSource ..> ListedRecord : listing()
      RecordSource ..> UnreadableFile : records(), facts()

:class:`~spindoctor.nav_records.RecordSource` is the whole of the contract: the
four questions, the two calls that hold a run together rather than answering
anything, and the context-manager pair, which is on the protocol rather than
left to each implementation because a stream may hold a connection that a caller
walking away mid-loop must still release.
:class:`~spindoctor.nav_records.TreeRecordSource` answers it out of the
documents and is built from nothing but the roots and a logger.
:class:`~spindoctor.results_index.IndexRecordSource` answers it out of the rows
and is built from an open engine, the index URL its messages name, and the
columns a consumer's records are rebuilt from --- the one asymmetry between the
two that a consumer has to know about, taken up under *The rules that hold at
the seam* below. Both expose the roots they hold, because a stub is a key under
one of them and a source holding more than one root cannot answer for a bare
stub.

The values on either side of the seam are the same values.
:class:`~spindoctor.nav_records.Selection` is what every question is asked in
terms of. :class:`~spindoctor.nav_records.NavRecord`,
:class:`~spindoctor.nav_records.ImageFacts` and
:class:`~spindoctor.nav_records.ListedRecord` are what the three streaming
questions yield, and :class:`~spindoctor.nav_records.UnreadableFile` is what
arrives in a stream in place of a record or a fact set when a file yielded
neither. Each is taken in turn in the sections below.

:class:`~spindoctor.dataset.results_filter.ResultsFilter` stands here for every
consumer, and is drawn because it is the one that asks at two different moments:
it writes a :class:`~spindoctor.nav_records.Selection` naming subtrees when it is
built and one naming stubs for each batch of candidates afterwards. Which of its
flags is answered at which of those two moments, and what each costs, is
*What each flag costs* below.

Four questions, because the programs ask four things
----------------------------------------------------

:meth:`~spindoctor.nav_records.RecordSource.record` takes one stub and returns
one record. It is the shape a per-image loop asks in --- the backplane and
mosaic stages reading the pointing an image is built with --- so it is the call
an implementation answers in a single round trip. A stub is a key under a root,
so a source holding more than one root refuses it, naming them.

:meth:`~spindoctor.nav_records.RecordSource.records` takes a selection and
yields the records it covers, one at a time. It is what a program that
summarizes or sweeps asks once per run --- the kernel writer reading a
mission. It also takes an explicit list of stubs, which is what a queue task
carries --- a worker must read exactly the files it was given, and read them in
batches, so looping the per-image call would cost it a round trip apiece on the
storage that batching exists for.

:meth:`~spindoctor.nav_records.RecordSource.facts` takes a selection and yields
what every image it covers says about itself --- the image's own values, one
entry per technique that reported, and the aggregated inventory of the features
the models offered. It is what the statistics report asks for, being the one
program that reads every field of every image: it makes a single pass of
accumulators over this stream and formats every section of its output from
them, so a report over a tree and a report over an index are one
implementation of every statistic rather than two obliged to agree forever. A
record cannot carry those fields: it is defined as looking like the document it
stands for, so the index rebuilds one out of the columns its consumer selected
and invents no field the document did not have, and no record carries a
per-technique or per-feature row at all. The facts are the whole row --- what
the index column set holds about an image, in the shape both storages hold it
in --- so the columns a consumer named narrow its records and never its facts.
A field of a document that no column holds is in neither storage's facts.

:meth:`~spindoctor.nav_records.RecordSource.listing` takes a selection and
yields what is there --- each stub, where it lives, and the size and
modification time that say whether it has changed --- without opening a single
document. On a cloud root one directory listing returns up to a thousand entries
with their metrics, against one round trip per document, which is what the
ingest's discovery and its unchanged-file skip are both built on. What a listing
cannot do is answer anything a document says, so a selection restricting on a
mission or a span is refused rather than partly honored.

**A selection naming stubs is not such a restriction.** A stub is the identity
of a file rather than something the file says, so a listing answers it, and that
is the question a caller enumerating candidate images asks: which of the ones
this run might still keep has a document. The index answers it with one keyed
query per batch. The tree has two ways to answer it and picks between them on
whether the root is local, because what decides is not a ratio of files named to
documents held but what one call costs. On a local root a check is a system
call, and checking the named files beats walking their directories at every
ratio worth having: ten files of a fifty-thousand-document volume by three
orders of magnitude, a fifth of the volume by two and a half times, and only at
something near the whole of it does the walk come back ahead. On a cloud root a check is a
paid round trip per file against one per directory for a thousand entries, so
the walk wins above roughly a thousandth of the root --- and one walk made for
one batch answers every later batch of the same run, since a run asks in
batches. The choice lives inside the seam, so a caller has one way to ask what a
root holds; two shapes in the callers would be two answers to keep true of each
other.

**On a cloud root, both halves of a pass are latency rather than bandwidth.** A
listing is one round trip and a document is another, and neither moves enough
bytes to matter: a navigation document is a few kilobytes. Both halves
therefore run in parallel, which is what makes a pass over a cloud root routine
rather than something to plan around. The walk lists several directories at
once, taking a bounded slice off a frontier each round so it still streams and
still holds only part of the tree; retrieval fetches several documents at once,
with a batch large enough to keep that pool full. A local root pays neither
latency, and neither costs it anything.

**How much of it runs at once is not a property of this program.** It belongs
to a machine, its link to the root, and what the service will do concurrently:
where the useful value stops rising is where that particular round trip becomes
what is left, and another provider or another link puts it somewhere else. So
the numbers are configuration rather than constants, and they travel through
the code one way. :class:`~spindoctor.nav_records.TreeTuning` carries them and
owns their defaults; the ``results_tree`` section of the configuration is where
a machine overrides them, shipped with the same values so that an operator can
see what they are; and :func:`~spindoctor.config.get_results_tree_tuning` is
the one door a program reads the section through. A program resolves the
tuning once, in its entry point, and passes it down to whatever reads the tree
--- :class:`~spindoctor.nav_records.TreeRecordSource`,
:func:`~spindoctor.results_index.open_record_source`, the ingest --- the way it
passes its logger. The library reads no configuration itself, and a call that
names no tuning gets the library's defaults.

The section is validated where every program loads its configuration, in
:func:`~spindoctor.config.load_default_and_user_config`, so a value no pass can
run at --- a count that is not a positive integer, a setting the section does
not have, a round of work smaller than the pool it feeds --- is refused before
the program has written anything, naming the section and the setting. The
section is also one of the three
:data:`~spindoctor.config.config.HASH_EXCLUDED_SECTIONS`: how many requests a
pass makes at once cannot change what any document says, so tuning it must not
re-stamp a run's results as differently configured.

**What a check cannot report is the size and the modification time.** Those come
from a directory entry, and an entry a check produced carries neither and says
so through :attr:`~spindoctor.nav_records.ListedRecord.has_metrics`. A consumer
that decides whether a document has changed reads that rather than a stand-in
value, which would make a changed document look unchanged.

Two more methods hold the run together rather than answering a question:
``describe()`` says which storage answered, for the run log, and ``close()``
releases what the source opened. A stream may hold a connection or a cursor, so
a caller that walks away mid-loop must still release it: a source is a context
manager, and a run writes ``with open_record_source(...) as source:``.

**A stream yields, and returns no list.** A caller that wants one writes
``list(...)`` and owns that decision. Nothing is accumulated on a caller's
behalf, so a program sweeping a mission holds one record at a time rather than
the mission --- and a program wanting many summaries of one stream is forced to
compute them in one pass, which is the constraint that makes the stream worth
having. Nothing promises an order either: a walk cannot know an image's epoch
before it has read the document, and a database sorting text sorts it under the
server's own collation, so each implementation yields in the order it finds
records, says what that order is, and a caller needing a total order calls
``sorted()`` and pays for it knowingly.

**A failure arrives from** ``next()``. A file that could not be read is yielded
into the stream as an :class:`~spindoctor.nav_records.UnreadableFile` rather
than raised, so one of them costs itself and not the rest of the pass. What
becomes of it is the caller's: the ingest records it in ``failed_files``, and
the statistics report counts them and prints the count, so a summary says how
much of a root it could not read rather than quietly covering less. A refusal
that ends a pass --- a directory nobody can list, an index that stops
answering --- surfaces in the middle of the caller's loop, so a program using
the stream finishes its pass before it writes its output.

Two backends, and why the package is split
------------------------------------------

:class:`~spindoctor.nav_records.TreeRecordSource` answers from the documents:
it walks the tree a bounded slice of directories at a time, carrying each
entry's metrics out of the listing, and retrieves documents in batches
underneath a stream that yields them one at a time.
:class:`~spindoctor.results_index.IndexRecordSource` answers from the rows,
streamed in server-side chunks. The per-image lookup and the listing
are one query each; a stream of records is two, the images and the files the
ingest refused; and :meth:`~spindoctor.nav_records.RecordSource.facts` is four,
because the per-technique and per-feature rows are merged onto the images stream
by key. That merge is the one place a statement here sorts --- its three
streams order on the key, so each image's child rows arrive next to it --- and
it is safe for the one reason a text sort ever is: the three orders are one
server's, read from one snapshot, and nothing compares them to an order
computed anywhere else.
:func:`~spindoctor.results_index.open_record_source` is what a program calls; it
returns the first when the run names no index and the second when it names one.

**The seam is split along the database line, and the line is not a matter of
taste.** ``import spindoctor.nav_records`` must not import SQLAlchemy. It is
the storage-free half of the seam: reading a results tree requires no index, so
a program that can open none still reads its records through this package, and
a database layer imported anywhere under it would be acquired by every one of
them. So :mod:`spindoctor.nav_records` --- what a record is, what a document is
named and where one lives, what a selection is, the protocol, the implementation
over the documents, and the per-image shape built from one document --- imports
no database layer, and a subprocess test pins that. The half that reads rows
needs SQLAlchemy by definition, so it lives in :mod:`spindoctor.results_index`
alongside the schema it reads, and so does the factory that chooses between the
two.

An inline import could not have arranged this. The rule is about a whole
package rather than about one call site, so every module under
:mod:`spindoctor.nav_records` would need the same guard and any one of them
could defeat it. The subprocess is likewise not a nicety: by the time any test
in the session runs, something has already imported SQLAlchemy, so the same
assertion inside the process would pass whatever the packages did. The probe
imports :mod:`spindoctor.nav_records` in a fresh interpreter, asserts that no
module of SQLAlchemy loaded, and asserts as well that the walk itself loaded,
since a guarantee about a package that imported almost nothing is a guarantee
about nothing.

What a caller asks for
----------------------

:class:`~spindoctor.nav_records.Selection` is the one value every stream is
asked in terms of: which of the source's roots to read, which top-level
directories to descend, which stubs to read outright, one mission, and a span of
exposure midtimes. Every field narrows and the fields combine, and a field left
at its default narrows nothing, so the empty selection covers everything the
source holds. Each backend applies whichever restrictions it can answer
cheaply --- a walk by looking at fewer directories, a query in its ``WHERE``
clause --- and the answer is the same either way.

It is frozen, and everything it carries is checked in its constructor: a subtree
is one directory immediately under a root, a stub is a key rather than a path,
and a time bound is a finite number with the start no later than the stop.
**That is what makes the two backends refuse alike.** A walk and a query cannot
be made to refuse identically by writing the same refusal twice; they refuse
identically because there is one place a selection can be wrong and neither of
them is reached from it. Left to a backend, an unusable value is refused in that
storage's own terms --- one raises out of a query builder in a language its
caller cannot name, another yields nothing, and an inverted range selects
nothing at all, which a run cannot tell from a clean pass over a quiet span.

The rules that hold at the seam
-------------------------------

**One piece of code, because two would be two answers.** A consumer carrying its
own row-to-record rebuild, its own open-and-check ceremony and its own account
of how the two storages differ is a second reader of the record: two rebuilds
agree the day they are written and then each grows a rule the other does not.
The defect that makes the point is an ``ORDER BY`` on text --- correct under
SQLite's default collation, wrong under a PostgreSQL locale collation, and
invisible to a consumer that reads one row at a time.

**A consumer names the columns it reads.** A row is only cheaper than a document
while it carries less, so nobody selects forty columns to read five. The columns
are declared beside the consumer --- ``_ROW_COLUMNS`` for the reprojection and
backplane stages, ``RECORD_COLUMNS`` for the kernel writer --- and a test holds
each list to the fields the rebuild knows a place for, since a column selected
that no field is rebuilt from is paid for on every read and then dropped. The
reprojection list is held to the other direction too: each of its columns is
dropped in turn and what those readers are then given has to differ, since a
column whose absence no reader notices is paid for on every read and then
ignored.

**The rebuild is a table, not a function per consumer.**
:data:`~spindoctor.results_index.rebuild.RECORD_FIELDS` maps each column to its
place in the record. Every column of ``images`` is deliberately one of three
things --- a record field, part of a row's identity, or a value the ingest
computed rather than copied --- and a test asserts the three are a partition of
the table, so a column added to the schema and to the ingest and to nobody's
consumer is caught rather than read as absent by everything.

**A record read from a document carries every field it has; one rebuilt from a
row carries what its consumer selected.** That is the one difference a consumer
has to know about, and it is why the columns are pinned rather than assumed.

**A stream of records narrows on the document; a stream of facts narrows on the
facts.** A record is the document, so a walk reads the mission and the exposure
midtime out of the document it has just parsed: one it can attribute to another
mission, or place outside the span, is passed over, and one it cannot place at
all is reported rather than dropped from every run there is. The facts are the
row an ingest of the same file writes, so a walk narrowing them compares the
fields the row holds, which are the fields a query narrows on --- and a file
that yields no facts is an unreadable file under every selection, because
nothing was read out of it for a filter to compare. That is what makes the two
storages cover the same images under one filter by construction rather than by
testing: under a filter on the facts there is no value one of them reads and the
other does not.

**A key is checked where it is written, not where it is read.** A results root
is normalized to one absolute, resolved spelling by
:func:`~spindoctor.nav_records.normalize_root_url` the moment a program is
handed one, and a stub and a subtree are checked as keys by
:class:`~spindoctor.nav_records.Selection` the moment a caller writes one: a
stub names no absolute path, no parent directory and no null byte, and a subtree
is one directory immediately under a root. Everything below that point is a
join, with one answer, so no reader carries a rule of its own about what a join
may have produced --- which is what a walk and a query could not have been made
to agree about by writing the same rule twice.

:func:`~spindoctor.results_index.roots.open_index_for_roots` is where the index
side's opening ceremony lives: it opens an index, refuses a root the index has
no completed ingest of, and disposes of the engine before the refusal leaves it.
Written out per call site, that sequence loses a step at a time --- an engine
left undisposed on a refusal, a root checked after the first query rather than
before it.

Which index, and when a results root is read at all
===================================================

Two different things are called an index in this codebase, and an enumeration
consults them for different questions.

The **dataset index file** is the archive's own catalog of a volume, the
``<VOL>_index.tab`` under ``metadata/``. It answers *what images exist*, it is
read once per selected volume, and every enumeration reads it. There is no mode
that walks the holdings tree instead: ``--img-name`` and ``--image-filespec-csv``
narrow within the rows the index file produced rather than replacing it as the
source.

The **results index** is this chapter's subject. It answers *what has already
been navigated*, which is a question an enumeration asks only when it is told
to. :class:`~spindoctor.dataset.results_filter.ResultsFilter` is constructed
only when one of the six image-selection flags is given
(``--has-offset-file``, ``--has-no-offset-file``, ``--has-offset-error``,
``--has-no-offset-error``, ``--has-offset-spice-error``,
``--has-offset-nonspice-error``). With none of them, no filter is built, the
navigation results root is never resolved, and nothing under it is opened or
listed.

Given one of those flags, which storage answers is a third and independent
choice, made by
:func:`~spindoctor.results_index.open_record_source`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Condition
     - What answers the flag
   * - No image-selection flag
     - Nothing. The results root is not read.
   * - A flag, and a results index URL resolves
     - The index, by query.
   * - A flag, and no index URL resolves
     - The results tree, by listing and by reading documents.

A results index URL resolves only for a program that **declares**
``--results-index-db``. The test is whether the parsed arguments carry a
``results_index_db`` attribute, which :mod:`argparse` supplies with its default
as soon as the option is added, so an operator need type nothing: a declaring
program still resolves the URL through the command line, then the
``environment.results_index_db`` configuration variable, then
``NAV_RESULTS_INDEX_DB``. A program that declares no such option has no
attribute to find, so an exported variable cannot quietly change what its
selection means. ``--results-index-db none`` is the deliberate spelling of "no
index", honored at every level, so a machine that exports one can still be told
to read the tree.

What each flag costs
--------------------

A filter asks the seam once when it is built and once per batch of candidates
the enumeration offers.

``--has-offset-file`` is settled outright at construction, by a
:meth:`~spindoctor.nav_records.source.RecordSource.listing` of the selected
subtrees. A listing opens no document: it is a directory enumeration, so the
cost is one listing call per directory rather than one read per image.
:meth:`~spindoctor.dataset.results_filter.ResultsFilter.passes` is then a set
lookup, which is what lets an enumeration offering a million rows reject most
of them without constructing anything.

The four error filters are settled in two stages, because each of them asks what
a document records and so requires one to exist. The construction listing
settles that half and no more: an error filter is folded into the same presence
question, so ``passes`` rejects an image the results root holds no document for
and no later stage ever names it. What the document *records* is settled per
batch, by :meth:`~spindoctor.nav_records.source.RecordSource.facts` over the
candidates
:meth:`~spindoctor.dataset.results_filter.ResultsFilter.filter_batch` is handed.
That call does read documents --- but only the candidates the other selection
constraints kept, never every document under the volume, and never one for an
image the listing has already excluded.

``--has-no-offset-file`` is asked about the candidates instead. Its answer set
is one it only ever rejects from, so listing a volume to build it would be work
with no reader. It is also structurally alone: the constructor refuses every
combination that pairs it with another results flag.

A listing that names the images it asks about is answered the way the root
answers most cheaply, and the source decides. A local root checks the named
images, because a local check is a system call. A remote root walks the named
subtrees once and answers every batch of that scan from the one walk, because a
remote check is a request per image while a listing returns about a thousand
entries with their metrics for the price of one. An entry produced by a check
carries neither of its two metrics and reports
:attr:`~spindoctor.nav_records.record.ListedRecord.has_metrics` false: a size
and a modification time come from a directory entry, and a stand-in would be
read as the file's own and make a changed document look unchanged.

What a filter hands on
----------------------

An error filter reading the tree hands each kept image the record its own
document was read out of.
:attr:`~spindoctor.nav_records.facts.ImageFacts.record` is that record: filled
by :class:`~spindoctor.nav_records.TreeRecordSource`, which holds one by the
time it has any facts at all, and left None by
:class:`~spindoctor.results_index.record_source.IndexRecordSource`, where a
record is rebuilt from the column set its consumer declares rather than from the
columns the facts hold.
:meth:`~spindoctor.dataset.results_filter.ResultsFilter.filter_batch` puts it on
:attr:`~spindoctor.dataset.dataset.ImageFile.nav_record`, and
:class:`~spindoctor.cli.reproj.pointing_source.FilePointingSource`,
:mod:`spindoctor.cli.backplanes` and
:mod:`spindoctor.cli.pds4.bundle_data` answer from it, so a program that selects
on what a document records reads that document once.
:class:`~spindoctor.cli.reproj.pointing_source.IndexPointingSource` reads its
row.

A second read of the same document would not be a ``filecache`` cache hit. The
dispatch programs build their results root through ``FileCache(None).new_path``,
an anonymous cache in a directory of its own that is deleted when the process
ends, while the record seam reads through the persistent ``_filecache_global``.
The two never see each other's downloads, so on a cloud results root a second
read is a second download. Pointing the programs at the global cache instead is
not the alternative: a Cassini-scale run would leave several hundred thousand
documents in a cache nothing empties.

What this bounds, and what it does not: an enumeration that yields its images
one at a time holds at most one filter batch of records,
:data:`~spindoctor.dataset.results_filter.RESULTS_FILTER_BATCH_SIZE` (64),
however many images the run covers. One that pools its candidates and samples
from them --- ``--choose-random-images`` --- holds the records of every image it
has kept, which the requested sample count bounds: it returns at that many
yields, so it holds at most that many plus the remainder of the batch the last
one came out of. Measured over 99 navigation documents from two results trees, a
parsed record is about 40 to 50 KB, so the sequential bound is a few megabytes
and the sampling bound is about 50 MB per thousand images sampled.

Opening an index
================

:func:`~spindoctor.results_index.open_index` is the opener every reader and
writer goes through. It selects the backend from the URL, applies the SQLite
settings the concurrency model depends on, and refuses a database whose stamped
schema version is not the one the code reads.

Three rules about that opener are worth knowing before changing it:

- **Every refusal is a** :exc:`ValueError` **naming the URL**, including the
  ones a database driver raises, so a caller that reports failures catches one
  type rather than the driver's exception hierarchy.
- **Every message masks the URL's password**, and so does anything the failure
  underneath it quoted back. These messages reach run logs, cloud-task results
  and terminals.
- **The index owns the schema it lives in.** A creating open refuses a schema
  holding any table it did not create, and refuses one holding a table of the
  index's own names that no stamp of SpinDoctor's stands over --- ``images`` is
  among the commonest table names there are, and a stamp written beside
  somebody else's table would make it the index's for every later reading,
  including the drop's. Which schema that is comes from the stamp:
  :mod:`~spindoctor.results_index.scope` resolves ``schema_meta``, identifies
  it by two marks rather than by today's column set, and the drop resolves it
  the same way, so what one builds is what the other removes.

:func:`~spindoctor.results_index.open_database` is the exception that proves
the gate: dropping the tables is the remedy the gate's own message prescribes,
so the drop has to work on a database the gate refuses.

The concurrency model
=====================

Several processes write one index and many read it. What makes that safe is a
small number of decisions, each of which a change can break silently.

**One image is written whole or not at all.** Its ``images`` row and its
``techniques`` and ``feature_sources`` rows are written inside one transaction,
after a delete that cascades to the children, so a concurrent reader never sees
half an image and re-ingesting one replaces its children rather than doubling
them.

**Writes are chunked, not batched into one transaction.** ``sd_results_index``
commits a chunk of images at a time, which bounds both what a crash costs and
how long a writer holds its lock. A chunk whose write fails is rewritten one
image at a time, so a single unstorable document costs itself rather than its
chunk.

A chunk is retrieved in batches, so a chunk smaller than a retrieval batch
would cap the batch at itself and throw away the download concurrency the batch
is sized for. The chunk is therefore stated as a multiple of the batch ---
``results_tree.ingest_commit_batches``, read off the same
:class:`~spindoctor.nav_records.TreeTuning` as the batch --- rather than as a
count of its own. A fixed chunk beside a configurable batch would let an
operator raise the batch past it and get a quieter, slower pass with nothing to
say why; expressed as a multiple, the two move together and the pool stays full
across a whole chunk while a transaction stays something a crash can afford to
repeat.

**Every statement that names stubs names at most**
:data:`~spindoctor.results_index.engine.STUBS_PER_STATEMENT` **of them.** Each stub is
a bind parameter and every backend caps how many one statement may carry, so
the lookup of what a share already recorded, the streams that answer about a
named set of images, and the prune's deletes all batch on that one constant. It
is a property of the SQL dialects underneath and not a tuning: nothing about
how a tree is walked or retrieved bears on it, which is why it lives beside the
engine rather than in the ``results_tree`` section.

**On SQLite, several local writers are an ordinary case.** The opener turns on
write-ahead logging and sets ``busy_timeout`` to ``SQLITE_BUSY_TIMEOUT_MS``, so
a competing writer waits rather than failing; with short transactions the
contention is brief. A ``sqlite:`` URL names a local path, which is what
confines this to one machine. Several machines share an index only through
PostgreSQL.

**Only a pass holding a listing of a whole root may delete a row.** Absence of
a row is an answer, so presence has to mean the tree still holds the document,
which is why each pass removes the rows of documents its walk did not find. A
worker handed a share of a root has no evidence about the stubs outside its
share and would delete its peers' rows, so nothing hands a worker a listing. The
license is a type rather than a check: one function builds the listing the prune
takes, it builds one only for a root it listed entirely, and it carries the root
it listed --- so a share of a root, a partial listing and a prune of one root on
the evidence of another are all unrepresentable rather than refused. In a
queue-divided pass the fan-out is the one step that sees a whole root, and it
prunes before it cuts the shares --- so the prune and the workers' writes
cannot touch the same stub.

**A pass may be told to keep them, and only presence is weakened by it.**
``ingest_metadata_files`` and ``fan_out_ingest_tasks`` both take ``prune``, and
``sd_results_index ingest --no-prune`` is what sets it. A row then outlives its
document and presence of one stops implying the tree still holds the result;
absence is untouched, because skipping a delete adds no row, so every rule above
that reads absence is unaffected. The relaxation is bounded by there being no
ingest of part of a root: a pass that reaches the prune listed the whole of one,
so the rows it declines to remove are exactly the rows it would have removed.

Which query that saves differs between the two passes, and the rule is written
once, in ``spindoctor.cli.results_index.driver._reads_recorded_rows``. The rows
the index already holds have two readers --- the skip rule and the prune --- and
the query runs when either wants them. So an ordinary pass under ``--no-prune``
still runs it for the skip rule and stops only under ``--force`` as well, while
a fan-out, whose workers skip from the metrics their own shares carry, stops
running it either way.

**A root is unreadable until its run is stamped.** The run row is written
before the walk and its finish time is left NULL until the pass completes, so
every consumer treats a root that is being ingested as one nobody has ingested.
A pass that cannot list a directory under its root stops rather than
completing, so a run that carries a finish time listed the whole of its root.
That rule belongs to the walk rather than to the ingest: the walk is
:class:`~spindoctor.nav_records.TreeRecordSource`'s, so a kernel-writing or
reporting run over the same tree stops at the same directory. A root that
cannot be listed **at all** raises
:exc:`~spindoctor.nav_records.UnlistableRootError` instead, which the ingest
catches to charge a mistyped root to that root and go on to the next one, and
which every other consumer lets end its run.

Adding a column
===============

There are no migrations. Any change to the column set of any table --- or to
the constraints over it --- means every existing index is rebuilt, and raising
:data:`~spindoctor.results_index.schema.SCHEMA_VERSION` is what makes that happen: an
index stamped with the earlier number is refused at open, naming both versions,
and its holder empties it and ingests the tree again.

1. Add the column to its table in
   :mod:`spindoctor.results_index.schema`. Use ``Double`` rather than ``Float``
   for anything that must round-trip a document's value, ``Boolean`` rather
   than an integer flag, and JSON for a structured value. Which of the two JSON
   declarations depends on what the value holds: a structure that can hold a
   number takes the one that is plain ``json`` on PostgreSQL, because ``jsonb``
   stores numbers as ``numeric`` and returns a stored negative zero as zero and
   a large-magnitude float as an integer; one holding text alone takes the
   ``jsonb`` variant, whose array and object accessors a direct-SQL query
   reaches inside without a cast.
2. **Increment** :data:`~spindoctor.results_index.schema.SCHEMA_VERSION` in the same
   commit. Increment it again for a second change in the same branch rather
   than reusing the bump: an index built from the intermediate state would
   otherwise pass the gate and then fail on a column that is not there.
3. Fill it in :mod:`spindoctor.nav_records.facts`, reading the document
   through the accessors in :mod:`spindoctor.support.nav_record` that the
   consumers read it through. The invariant is that a record rebuilt from the
   columns classifies exactly as its document does; a second set of rules in
   the store is a second reader of the record, and the two drift.
4. Sort it into one of the three groups in
   :mod:`spindoctor.results_index.rebuild`. A column copied out of one field of
   one document is an entry in
   :data:`~spindoctor.results_index.rebuild.RECORD_FIELDS` naming where in a
   record that field sits; one that says where the document is belongs to
   :data:`~spindoctor.results_index.rebuild.IDENTITY_COLUMNS`; one the ingest
   computes rather than copies --- a date rendered from the recorded epoch, a
   count of a list --- belongs to
   :data:`~spindoctor.results_index.rebuild.DERIVED_COLUMNS`, because no field
   of a record is what it came from. The
   three are asserted to be a partition of the table, so a column left out of
   all three fails rather than reading as absent from every record.
5. Read it wherever it is consumed, by adding it to that consumer's column list,
   and extend the reason vocabulary if the consumer classifies on it. A consumer
   that does not select a column reads its field as absent.
6. Update the column-set lists in
   ``tests/spindoctor/results_index/test_schema.py``: the per-table list, which
   pins each table's columns, their types, their nullability and their order,
   and --- for a JSON column --- the list of JSON columns, which pins the type
   each backend emits for it and that an absent value is stored as SQL NULL. A
   JSON column missing from the second list fails the test that holds the two
   against the schema, and ``COLUMN_SET_VERSION`` beside them takes the same
   number the schema was stamped with: a version compared only against itself
   agrees with every value it could be given, which is why the number is
   written down beside the columns as well as in the schema.
7. Update the schema tables in
   :doc:`/user_guide/user_guide_results_index`, which document the index for
   somebody writing SQL against it.

Dropping a column is the same list, and the same version bump.

The edits, in the order the list gives them:

.. code-block:: python

    # spindoctor/results_index/schema.py
    IMAGES = sqlalchemy.Table(
        'images',
        METADATA,
        ...,
        # What the column carries, and what a NULL in it means.
        sqlalchemy.Column('shutter_mode', sqlalchemy.Text),
    )

    SCHEMA_VERSION = 9  # incremented by this change

    # spindoctor/nav_records/facts.py, inside facts_from_document
    image_row: dict[str, Any] = {
        ...,
        'shutter_mode': _str_or_none(observation.get('shutter_mode')),
    }

    # spindoctor/results_index/rebuild.py, inside RECORD_FIELDS
    RecordField(('shutter_mode',), _OBSERVATION, 'shutter_mode'),

    # The consumer's own column list, which is what decides that it is read
    RECORD_COLUMNS = (
        ...,
        IMAGES.c.shutter_mode,
    )

    # The consumer, reading the field the column was rebuilt into
    mode = record['observation'].get('shutter_mode')

    # tests/spindoctor/results_index/test_schema.py
    IMAGES_COLUMNS: tuple[tuple[str, ColumnType, bool], ...] = (
        ...,
        ('shutter_mode', sqlalchemy.Text, True),  # name, type, nullable
    )

    # And, for a JSON column only, the declaration each backend emits
    JSON_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ...,
        ('images.spice_kernels', 'JSON', 'JSONB'),  # column, SQLite, PostgreSQL
    )

    COLUMN_SET_VERSION = 9  # the same number, written down beside the columns

The value is read out of the document with the coercion its column's meaning
calls for, and never coerced past what the document said: ``str(None)`` is
``'None'``, which would be stored as a value a document never recorded.

Testing
=======

The suite opens no index it was not handed, and the ``postgres`` tier is
deselected unless a server URL is exported --- so anything pinned only there is
unpinned in practice. See :doc:`dev_guide_testing` for both.

Two failure modes recur in this subsystem specifically, and both have cost real
time:

- **Root-blindness.** A query that filters on the stub and forgets the root.
  Every such defect passes a single-root fixture. Write the two-root fixture.
- **A test that cannot fail.** An exit status a catch-all also produces; an
  absence asserted against output that could not have contained it. When a test
  asserts that something is *not* reported, assert also that the thing it
  should report instead is.

API reference
=============

:doc:`/api_reference/api_results_index`
