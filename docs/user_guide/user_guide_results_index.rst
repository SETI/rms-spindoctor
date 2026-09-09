=================
The Results Index
=================

Overview
========

Navigation writes one ``_metadata.json`` document per image under
``nav_results_root``. Every program downstream of navigation then reads one of
those documents per image it processes. On a local disk that is cheap. On a
cloud root it is one paid round trip per image per program, and a
Cassini-scale run is of the order of 400,000 images.

The fields those programs actually consume are narrow: whether the image was
navigated, what fatal error it recorded, the pixel offset, the corrected
attitude and a handful of quality numbers. The results index holds them, and it
is an index in the sense a PDS index table is: one row per product, a few fields
chosen out of each, derived from the products themselves and read to select
among them and to summarize them. One pass reads every document once and writes
one row per image; from then on a consumer answers its questions with a query
instead of a download.

**The documents remain the authoritative record.** The results index is derived
and disposable: nothing in it cannot be rebuilt from the tree, and deleting it
costs the time of one ingest and nothing else. Navigation never writes to it, so
nothing in the pipeline that produces results depends on a database being
reachable.

This chapter is the results index: when to build one, how programs are pointed
at it, what it does and does not promise, how to build, share, rebuild and query
one, and the tables it holds. :doc:`user_guide_statistics` documents
``sd_stats_report``, which turns a results index into the navigation statistics
report.

Nothing requires a results index
================================

Every program runs with no index at all, and that is the default. A run that
names no index reads the results tree exactly as it always has; there is no
fallback path to get wrong, because reading files *is* the ordinary path.

``sd_stats_report`` is the one worth a word before it is pointed at a tree. It
reads every document under every root it is given, on every run --- which is
exactly the cost a results index exists to remove. For a local tree and a single
report that is the right trade; for a cloud root, or a report you will run
again, build an index first. :doc:`user_guide_statistics` says so beside the
option.

**A program becomes index-backed by declaring** ``--results-index-db``, never by
inheriting an exported ``NAV_RESULTS_INDEX_DB``. A program whose selection is
meant to read files does not quietly change what it reads because a variable was
exported for another program. A declaring program honors all three levels of the
ladder below; a program that declares nothing reads the tree whatever is set in
the environment.

Naming a results index
======================

Every index-backed program resolves its URL the same way, in this order:

1. the ``--results-index-db URL`` command-line option;
2. the ``environment.results_index_db`` configuration variable;
3. the ``NAV_RESULTS_INDEX_DB`` environment variable.

The literal value ``none`` at any level means "no index", and overrides a URL
set at a lower one. That is how a single run is made to read the tree on a
machine that has a results index configured:

.. code-block:: bash

    sd_backplanes ... --results-index-db none

A value that is empty, or nothing but spaces, is refused at whichever level
carries it. ``none`` is how a level says "no index" deliberately, so an empty
value is a typo, a script that computed nothing, or a variable left half unset;
read as naming no index it would send a whole batch to the results tree, which
answers a different question and takes hours longer to answer it on a cloud
root. Every run on a machine configured that way therefore stops, on the first
run rather than after the batch, and says which of the three levels to fix::

    NAV_RESULTS_INDEX_DB is set to an empty value, which is neither a
    connection URL nor the way to name no index. Write none to name no index,
    or a connection URL to name one.

Two URL forms are supported::

    sqlite:////data/nav-offset-results/index.sqlite3
    postgresql+psycopg://user@dbhost/spindoctor

A ``sqlite:`` URL names a **local filesystem path**, spelled with four slashes
for an absolute one. It is the one location in this system that is not
cloud-capable: the SQLite library opens the file directly, so a network
filesystem that cannot honor its locking is refused when the results index is
opened rather than corrupted later. The URL carries no query string, since the
driver would then open a file named after the query.

A ``postgresql+psycopg:`` URL names a server, and needs the driver:

.. code-block:: bash

    pip install "rms-spindoctor[postgres]"

A results index URL can carry a password, and these URLs are written to run
logs, printed in refusals and returned in cloud-task results. Every message that
names one masks it, so a log records ``postgresql+psycopg://user:***@dbhost``
and the command line it was typed on is masked the same way.

Which programs read a results index
===================================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Program
     - What the results index answers
   * - ``sd_results_index``
     - Writes it. The only program that does.
   * - ``sd_results_index_cloud_tasks``
     - Writes one share of it, as a queue worker.
   * - ``sd_stats_report``
     - Every section of the report. Given no index it reads the results tree
       instead, and over the records both storages can read the report is the
       same either way. The count of files that yielded no record is the one
       exception: a file the storage could not deliver at all is counted from a
       tree and not from an index, because the ingest deliberately records no
       refusal for a retrieval that failed once.
   * - ``sd_offset``
     - The results-based selection filters (``--has-offset-file``,
       ``--has-no-offset-file``, the error filters). Given no index, a run
       selecting images that have a document lists each selected volume, a run
       selecting images that have none asks about its candidate images by name,
       and the documents of the candidate images say what each one records.
   * - ``sd_backplanes``, ``sd_backplanes_cloud_tasks``
     - Each image's recorded status and pointing, which the stage reads before
       it decides there is work to do.
   * - ``sd_mosaic``, ``sd_mosaic_cloud_tasks``
     - The pointing each contributing image is reprojected with.
   * - ``sd_create_ck``
     - One mission's records, read in bulk rather than one document per image:
       which images carry a corrected attitude, the attitude itself, the exposure
       epochs, and the kernels the run recorded.

Every other program reads the tree. Two do so for a reason that will not
change: ``sd_create_bundle`` serializes the whole navigation document into the
PDS4 supplemental product, and ``sd_consolidate_metadata`` copies raw file
bytes. Neither is served by a set of columns.

A results index is a snapshot
=============================

The results index reflects the tree as of the last ingest over the roots it
covers. There is no staleness detection, no re-verification against the tree and
no automatic refresh. A consumer trusts what it holds.

That has two operator-visible consequences, and they run in opposite
directions:

- An image navigated since the last pass is one the results index does not
  hold, so ``--has-no-offset-file`` selects it again and a downstream stage
  reports it as an image nothing navigated.
- A result file deleted since the last pass is one the index still holds, so
  ``--has-offset-file`` hands on an image whose document is gone.

Both are answered by running ``sd_results_index ingest`` again, which is cheap
over a tree that has barely changed: a document whose size and modification time still
match is not read at all. Re-ingesting before a run that depends on the answer
is an ordinary thing to do. Every run that answers from a results index reports
when the pass that filled it finished and how long ago that was, so the age of
the answer is in the run log beside the answer.

Where the two disagree for reasons the age does not explain --- a file the
results index has no row at all for, a document rewritten in place --- the cases
are enumerated in :doc:`user_guide_navigation` under the selection filters, and
the reason vocabulary the backplane and reprojection stages report is in
:doc:`user_guide_backplanes` and :doc:`user_guide_reprojection`.

Tuning a pass for your machine and storage
------------------------------------------

A pass over a results tree does two things: it lists directories to find out
what is there, and it reads the documents it found. Against remote storage
both are dominated by waiting. A listing is one request and a document is
another, and a navigation document is only a few kilobytes, so a pass done one
request at a time spends almost all of its time waiting and almost none of it
moving data or using the processor.

Both halves therefore run several requests at a time, and the ``results_tree``
configuration section is where you say how many. You do not need to set any of
it: the defaults are chosen to be sensible.

**If your results tree is a local directory, these settings do not apply to
it.** A listing of a local directory and a read of a local file cost no round
trip, so the thread and batch settings have nothing to overlap and no useful
effect there; leave them at their defaults. The one exception is
``ingest_commit_batches``, which is about the database transactions an ingest
writes rather than about the tree, and so applies whichever storage the tree
is on.

.. code-block:: yaml

   results_tree:
     walk_threads: 32
     walk_directories_at_once: 256
     retrieve_threads: 64
     retrieve_batch_size: 1024
     ingest_commit_batches: 2

The section applies to every program that reads a results tree, not only to
``sd_results_index``: a statistics report or a C-kernel run over a tree, and a
navigation, backplane or bundle run whose selection names one of the
``--has-offset-file`` family of filters, all read the tree the same way and all
honor it. Only the last setting is particular to the ingest.

**A note on processor cores.** These are not settings that divide work between
cores. The threads they create spend their time waiting for storage to answer,
not computing, so it is normal and correct for them to outnumber your cores
several times over -- the defaults above will happily run on a four-core
machine. Adding cores is not by itself a reason to raise them. What decides
the useful value is your network and what your storage service will do at
once, and past that point extra requests queue, or the service refuses them,
and the pass gets slower rather than faster.

**A value no pass can run at is refused when the configuration is loaded**,
naming the setting, before the program has written anything: a count that is
not a positive whole number, a setting name the section does not have, or a
round of work smaller than the pool it feeds, which would leave threads idle at
every setting rather than merely running slowly.

The settings
~~~~~~~~~~~~

``walk_threads`` (default 32)
   How many directories are listed at the same time. Raise it if your tree has
   many directories and the finding stage is slow; lower it if your storage
   service complains about the request rate. It costs almost no memory -- a
   thread waiting on a request holds very little -- so memory is not the
   reason to keep it small.

``walk_directories_at_once`` (default 256)
   **This one is a memory limit, not a speed control.** Listing happens in
   rounds, and a round holds the full contents of every directory it is
   listing before moving on. Without a limit, a round over a wide tree would
   try to hold that entire level of the tree at once. This caps how many
   directories a round covers, and therefore how much a round holds.

   Raising it does not make a pass faster once there is enough work to keep
   ``walk_threads`` busy; it only increases the peak memory of the pass. Lower
   it on a machine short of memory, or if you have an unusually deep or wide
   tree with very large directories. Raise it only if you have plenty of
   memory and are certain the finding stage is starved for work. It must be at
   least ``walk_threads``, so that every thread has a directory to list, and
   is best kept comfortably above it; lower the two together on a machine that
   needs a small round.

``retrieve_threads`` (default 64)
   How many documents are downloaded at the same time. This is the setting
   that most affects how long a pass takes against remote storage, and the
   first one to change if reading is slow. Raise it on a fast, high-latency
   link; lower it if your provider throttles you, returns errors under load,
   or if you are sharing the connection and want the pass to be less greedy.
   Downloaded documents are written to the local file cache rather than held
   in memory, so this costs disk space in the cache rather than RAM.

``retrieve_batch_size`` (default 1024)
   How many documents are handed to the downloader at a time. Its job is to
   keep the download threads busy: a batch has to be comfortably larger than
   ``retrieve_threads``, or threads sit idle at the end of every batch waiting
   for the next one to start. If you change ``retrieve_threads``, change this
   with it and keep it several times larger.

   A batch smaller than ``retrieve_threads`` cannot fill the download pool, so
   a run configured that way is refused when the configuration is loaded,
   rather than running slowly for a reason you would have to go looking for.

``ingest_commit_batches`` (default 2)
   How many retrieval batches ``sd_results_index`` writes into the index in
   one database transaction. No other program reads it. A transaction is what
   a crash costs, so lower it to lose less work if an ingest dies part way;
   raise it for fewer, larger transactions. It is a multiple of
   ``retrieve_batch_size`` rather than a count of its own, so raising the batch
   can never leave a transaction smaller than the batch it is retrieved in.

Choosing values for your machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Short of memory.* Lower ``walk_directories_at_once`` first, with
``walk_threads`` lowered to match -- it is the setting that bounds how much the
walk holds while it lists, and halving it roughly halves that. An ingest also
keeps one small entry per document for the whole root while it works, which no
setting changes. The others cost little memory at any setting.

*Plenty of memory, slow remote storage.* Leave
``walk_directories_at_once`` alone and raise ``retrieve_threads``, with
``retrieve_batch_size`` raised alongside it. Raise them together in steps and
watch whether the pass actually gets faster; when it stops improving, you have
reached what your link or your provider will give you, and going further tends
to make things worse.

*A shared or metered connection.* Lower ``retrieve_threads`` and
``walk_threads``. The pass takes longer and leaves more of the connection for
everything else.

*A results tree on a local directory.* Leave the thread and batch settings
alone: they exist for remote storage and have nothing to do here. Only
``ingest_commit_batches`` is worth a thought, and only for what an ingest that
dies part way through should cost.

How a results index is asked for
================================

``sd_results_index`` takes the action as a subcommand:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Subcommand
     - What it does
   * - ``ingest``
     - Walks each named results root and reads the documents under it into the
       results index.
   * - ``divide``
     - Lists each root once, removes the rows whose documents have left it, and
       writes out the shares a queue of workers reads.
   * - ``complete``
     - Adds up what those workers did and stamps each root's ingest as
       finished.
   * - ``drop``
     - Removes the results index's own tables from the database and stops.

Each subcommand takes the options it acts on and no others, so a command line
asking for two different things is refused with a usage message naming the
option instead of being run as one of them. ``--results-index-db``,
``--config-file`` and the logging options belong to all four;
``--nav-results-root`` to the three that read a tree; ``--force`` and
``--no-prune`` to ``ingest`` and ``divide``; and ``--yes`` to ``drop``. Such a
refusal exits 2 and does nothing at all --- no database is opened and no tree is
walked.

Building a results index
========================

.. code-block:: bash

    sd_results_index ingest --nav-results-root /data/nav-offset-results \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3

``sd_results_index ingest`` walks each navigation-results root recursively for
``*_metadata.json`` files (the documents
:func:`~spindoctor.navigate_image_files.navigate_image_files` writes under
``nav_results_root``) and loads them into the results index. Roots may be local
directories or any URL the project's ``filecache`` layer accepts, so
cloud-hosted results ingest the same way as local ones.

One results index holds as many roots as you ingest into it, and each consumer
is answered only from the rows recorded under the root it was pointed at. Roots
are compared in one normalized spelling -- absolute and resolved, with any
trailing separator removed -- so a root named relatively on one run and
absolutely on the next, written with a ``~`` or a ``..`` in it, or reached
through a symbolic link on one machine and at its own location on another, is
one root.

**The roots are navigation-results roots**, resolved the way every other
program resolves one: ``--nav-results-root`` (repeatable), then the
``environment.nav_results_root`` configuration variable, then the
``NAV_RESULTS_ROOT`` environment variable. Each row records the root it came
from and its path under that root, and every consumer looks a row up by that
pair. Pointing ingest at a subdirectory of a results root would produce
identifiers no consumer's lookup can match. A root that is not a location is
refused before anything is walked, and an empty one is such a root: with
``--nav-results-root "$ROOT"`` and ``ROOT`` unset the program stops rather than
ingesting whatever directory it was started from under a name nobody chose.

**An ingest builds the results index in one schema, and only in a schema of its
own.** The tables go into the schema the index resolves to: the one holding a
``schema_meta`` stamp SpinDoctor wrote, or, where the database carries no such
stamp, the one an unqualified ``CREATE TABLE`` lands in (``main`` on SQLite, the
first schema of the search path on PostgreSQL). That schema is built in when it
holds nothing at all, and gone on with when it already carries SpinDoctor's
stamp, whatever schema version that stamp names. Anything else is **refused**,
before a table is created or a stamp is written:

- A schema holding a table of one of the results index's own names --
  ``images``, ``techniques``, ``feature_sources``, ``failed_files``,
  ``schema_meta`` or ``ingest_runs`` -- that no stamp of SpinDoctor's stands
  over. Those are among the commonest table names there are, so a table called
  ``images`` is not evidence of anything, and a stamp written beside somebody
  else's table would make it SpinDoctor's for every later reading, including the
  drop's.
- A schema holding any table SpinDoctor does not own, stamped or not. A results
  index owns the schema it lives in, so a table nobody here created in it means
  the URL, or the search path behind it, names a database or a schema other than
  the one intended.

The refusal names the schema, the tables it stopped on and the results index URL
with its password hidden, and exits 1; nothing is created, nothing is stamped,
and that schema is exactly as it was. The remedy is to check the URL, or to name
a schema of the index's own -- on PostgreSQL, by appending
``?options=-csearch_path=schemaname`` to the URL. Other schemas of the same
database are neither read nor named, so one server holds an index beside
whatever else it holds.

**Ingestion is incremental.** One recursive listing per root collects the
metadata documents and carries each file's size and modification time along
with it. Every other file under the root -- the summary PNG beside a document,
and whatever else is there -- is passed over and counted nowhere. A file whose
recorded size and modification time still match the listing is not read at all,
so a second pass over an unchanged root costs one listing and nothing else.
This holds for files that could not be ingested as well: a file that is not a
navigation document is recorded as such, with the same two metrics, and is
skipped for as long as it does not change. ``--force`` re-reads everything; so
does a storage backend whose listing reports neither size nor modification
time, which ingest warns about rather than silently skipping.

**Those two metrics are everything a listing supplies about a file.** A document
rewritten in place that kept both of them is therefore one the ingest cannot
tell from the document it already read: it is skipped, and its row goes on
recording what the earlier document said, however many passes complete
afterwards. A tree restored by a copy that preserves modification times, a
document patched and then stamped back from a sibling, and a backend reporting
one modification time for two writes of equal length all produce that; an
ordinary re-navigation writes a different length at a later time and does not.
``sd_results_index ingest --force`` re-reads every document and is what puts
such a row right. Reading each file to find out whether it needs reading is the retrieval
the skip exists to avoid, which is why the remedy is a flag rather than a finer
comparison.

**A recorded refusal outlives the version of SpinDoctor that made it.** The
record says what was wrong with the file, not which build read it, so a document
refused by one build is skipped by every later one, including a build that has
since learned to read it. After upgrading SpinDoctor, run
``sd_results_index ingest --force`` once over each root to re-offer everything
it refused.

**Ingestion is idempotent.** The results index holds one row per image, and
re-ingesting the same or an updated document replaces that image's row and its
child rows rather than duplicating them.

**A document that leaves the tree takes its row with it.** No row for an image
is what every consumer reads as "this image was never navigated", so a row is
only allowed to survive while the document behind it does. Each pass deletes
the rows of one root whose documents the walk no longer found, and reports how
many. That is done on the strength of a complete listing of the root and no
other: a root the walk could not list at all -- a mistyped path, an unmounted
share -- has nothing removed, and its ingest run is deliberately left
unfinished, so every consumer reports it as a root nobody has ingested rather
than answering "not navigated" for every image under it. A root that exists and
is empty completes normally.

**``--no-prune`` keeps those rows, and is a correctness relaxation rather than
a speed feature.** Under it a row may outlive its document, so presence of a row
stops meaning that the tree still holds the result it stands for: a consumer
asking whether an image has been navigated is answered yes for one whose result
has been deleted, and ``--has-offset-file`` hands such an image on to a
downstream stage that then finds nothing to read. What is *not* affected is
absence, because skipping a delete adds no row: "this image was never
navigated", ``--has-no-offset-file``, and the refusal of a root nobody has
ingested all keep exactly the meaning they had. That is what makes the flag safe
enough to offer, and it holds only because a pass covers a whole root -- there is
no way to ingest part of one, so a pass that gets as far as removing rows listed
everything.

**What it saves is the deletes and, in some modes, a query, and never any part
of the walk.** The listing is discovery and happens either way. The deletes are
what the flag always saves, since removing nothing is the whole of what it asks
for. The query -- one statement over every row the root holds -- is saved only
where nothing else wants the answer:

* An ordinary pass issues that query once and reads the answer twice over: to
  decide which documents to skip as unchanged, and to decide which rows to
  remove. Only the second of those goes with the flag, so ``--no-prune`` alone
  saves the deletes and still runs the query. ``--no-prune --force`` saves the
  query as well, since a forced pass reads every document regardless and so has
  nothing to skip.
* ``sd_results_index divide`` issues it for the removals alone; each worker
  decides what to skip from the metrics its own share carries. So ``--no-prune``
  saves the query there whether or not ``--force`` is given.

The pass says in its closing summary that it left those rows alone, in place of
the count it would otherwise report, so a run log read later says which
guarantee the results index under it was built with. Running the ingest again
without the flag is what puts a root right.

``--no-prune`` belongs to ``ingest`` and ``divide``, and neither of the other two
subcommands offers it: ``complete`` removes no row --- whether a fan-out's rows
went was settled at the ``divide`` that listed the root --- and ``drop`` reads no
tree at all.

**A root the results index has no completed ingest of is refused, not
answered.** Absence of a row would otherwise read as "this image was never
navigated", so a consumer pointed at an index that has never covered its root
fails with a message naming the root and the roots the index does hold. A pass
that is still running, or one that stopped, leaves its root in exactly that
state until it is completed. A consumer that names no root of its own -- the
statistics report is the one that may -- is bound to the roots that do have a
completed ingest and names the others as roots it covered nothing of, which is
the same rule stated about a set of roots rather than about one.

**Ingestion is never automatic.** No batch driver runs it as a side effect. The
index is a snapshot of the tree as of the last ingest: there is no staleness
detection and no automatic refresh, so an operator who navigates more images and
wants them visible runs ingest again.

Every ingestible document must carry ``observation.image_name`` and
``observation.instrument``, every container it declares -- ``observation``,
``navigation_result`` and the objects and lists inside it, ``timing`` -- must
hold what the schema says, and each of its ``per_technique`` entries must carry
a ``technique_name`` of its own, since that name is what the results index
stores the entry under (the pipeline writes all of this in every metadata
document). A results tree also holds ``*_metadata.json`` files that are not
per-image navigation documents at all; each is counted as an error for its own
file, the run continues, and the closing summary tallies the failures by reason
and names one file per reason, so several hundred files that were never
navigation results read as exactly that rather than as a broken ingest.

A file that could not be retrieved at all is the one failure the pass records
nowhere. It is counted in the pass's own tally and no ``failed_files`` row is
written for it, deliberately: a retrieval that failed once is worth trying again
on the next pass, where a file that was read and refused will be refused again
for as long as it does not change. So a report over the documents counts such a
file among the files that yielded no record and a report from a results index
does not, and running the ingest again is what closes the gap.

That tally is what this pass read, and not what the root holds. A refused file
is recorded in ``failed_files``, and every pass after it skips the file
unchanged rather than reading it again, so a second pass over the same tree
refuses nothing and tallies nothing and a summary of zero refusals is a summary
of what changed. ``sd_results_index ingest --force`` reads every document again,
which puts the reasons and the example files back into the summary; the root's whole
set of them is otherwise a query over ``failed_files`` away.

**A directory under a root that the walk cannot list stops the ingest.** One
this user may not read, or on a share that stopped answering, is reported as an
error naming the directory, and the pass ends there: the root it was under gets
no finish time, no root named after it on the same command line is walked, and
``sd_results_index ingest`` exits 1. Fix what stopped the walk and run it
again.

**A document the results index will not store stops it too.** A file that read
as a navigation result and whose rows the database then refused is a defect in
SpinDoctor -- in what it writes, or in the columns it writes into -- rather than
anything about the file, and the next document of that shape would be refused
the same way. The pass ends there, naming the file and what the database said,
the root gets no finish time, and ``sd_results_index ingest`` exits 1. Every
document
written before it stays in, so a rerun after the fix reads only what is left.
The alternative -- counting the file and carrying on -- put it in neither
``images`` nor ``failed_files`` under a run stamped finished, which made an
image nothing had ever navigated and an image SpinDoctor could not store look
exactly alike.

**A root that cannot be listed does not stop it.** That is the case above --
the mistyped path, the unmounted share -- and it is charged to the root it is
about: that root is reported, left unfinished and ingested not at all, every
other root named on the command line is still walked, and the status is 1 at the
end. The difference is what the walk knows. A root nobody can list is a root
this pass has said nothing about, and the next root has nothing to do with it; a
directory nobody can list sits inside a root the pass is otherwise about to
declare it has read.

The alternative was tried and is worse. A pass that finished around the gap
completed, stamped its root as ingested, and left every image under that
directory reading as one nothing had ever navigated -- and, because such a pass
must not remove rows on evidence it does not have, left every document deleted
since the pass before it reading as present, across any number of later passes
that completed the same way. Stopping costs an ingest, which is cheap and
repeatable; finishing cost a wrong answer that no later pass corrected.

**The cost of that is a transient failure ending a long pass.** A share that
stops answering for a moment, or a permission fixed a minute later, now ends the
run instead of degrading it. That is the trade, made deliberately: the walk
happens before any document is read, so what a stopped pass throws away is the
listing rather than hours of retrieval, and the pass can simply be run again.

**A directory reached a second way is not a gap and does not stop anything.** A
link pointing back up into the tree, or a volume reachable under two names,
brings the walk to a directory it has already listed; it says so, declines to
list it twice, and goes on. The documents under it are already in the listing,
under the path the walk met first, and walking it again would only write them a
second time under identifiers no consumer asks about.

**Do not put symbolic links inside a results tree.** Which of the two paths to
such a directory the walk meets first is whichever the directory listings
returned first, and that is not defined. Every document under it is therefore
recorded under one of two identifiers, either of them; a later pass can meet the
other one first, at which point the identifiers the earlier pass recorded name
documents that are no longer in the tree, and the pass removes those rows and
writes the other set. Two passes over one unchanged tree can each undo the
other, and nothing about either pass looks wrong: both complete, both stamp the
root, and both report that they removed the rows of documents that had left it.

**A results root that is itself a link is a different matter and is handled.**
Every program resolves the root it is given to the location that root names
before it reads anything, so ``--nav-results-root /data/latest`` pointing at
``/data/results-2026`` is the same root as naming ``/data/results-2026``
outright: one set of rows, and either spelling finds them. What is undefined is
a link *inside* the tree being walked, which is what the paragraph above is
about.

**The exit status says whether the pass completed, not what it found.**
``sd_results_index ingest`` exits 0 when every named root was walked, whatever
mix of documents was read, skipped and refused, and 1 when the run could not
complete: no index or no results root could be resolved, a named root is not a
location that can be read, the results index could not be opened, a root could
not be listed at all, or a directory under one could not be listed, which stops
the pass where it is found. A scheduled invocation therefore reads the same
status from the same tree every time, and a status of 1 always means something
needs fixing rather than that a tree happens to hold no results. A command line
no subcommand takes exits 2 instead, with a usage message naming the option, and
nothing is opened or walked.

Ingesting over a queue of workers
=================================

A root of a few hundred thousand documents is one listing followed by that many
independent reads, so the reads can be spread over a queue. Three steps do it,
and the middle one is where the work happens:

.. code-block:: bash

    # 1. List each root, remove the rows of documents that have left it, and
    #    write out the shares.
    sd_results_index divide --nav-results-root /data/nav-offset-results \
        --results-index-db postgresql+psycopg://user@dbhost/spindoctor \
        --tasks-file ingest_tasks.json

    # 2. Run the workers over those tasks, however the queue is driven. Each
    #    worker writes its results into an event log.
    sd_results_index_cloud_tasks \
        --results-index-db postgresql+psycopg://user@dbhost/spindoctor ...

    # 3. Add the workers' tallies up and record them against each root.
    sd_results_index complete --nav-results-root /data/nav-offset-results \
        --results-index-db postgresql+psycopg://user@dbhost/spindoctor \
        --events-log events.log

**Workers on one machine can share a SQLite index**; workers on several cannot.
A ``sqlite:`` URL names a local file, so a run spread across machines connects
to PostgreSQL instead. Several worker processes on one machine writing to one
local file is supported, and needs no merge step -- there is one file.

**Only step 1 creates the results index.** A worker opens an index that already
exists and fails if it does not, because a worker that created one would answer
a mistyped URL by building an empty index beside the real one, and every
consumer would then read absence of a row as "this image was never navigated".

**Only step 1 removes a row.** Deleting the rows of documents that have left the
tree is allowed on the strength of a complete listing of the root, and step 1 is
where the one listing of the pass happens; a worker holds a share and knows
nothing about the stubs outside it, so a worker that removed rows would remove
its peers'. Nothing a worker is about to write can be removed in step 1 either:
every file a worker is handed came from that listing, and only stubs the listing
did **not** hold are removed. ``--no-prune`` belongs to step 1 for the same
reason, and it is the one place in a queue-divided pass where the query over the
root's rows goes with the removals: step 1 reads them for nothing else, since
each worker skips from the metrics its own share carries.

**A root is not readable until step 3.** Its ingest run stays unfinished from
step 1 onwards, so every consumer reports it as a root nobody has ingested while
the workers are still writing -- rather than answering from whichever shares
have landed.

**Abandoning a fan-out costs a full re-ingest of that root.** Step 1 removes the
rows of documents that have left the tree before any document is read, so a pass
whose tasks are never queued, or that is given up on, has already shrunk the
results index. Nothing incorrect is lost -- the rows removed are exactly those
whose documents the listing did not find -- and the root stays unfinished
throughout, so no consumer reads a wrong answer from it. What it costs is the
rest of the root's content in the index, which comes back by running the three
steps through to the end, or an ordinary ``sd_results_index ingest`` over the
root.

**Two passes over one root at the same time are a documented limit.** Nothing
refuses a fan-out over a root whose newest run is still unfinished, and two that
overlap can leave behind one row whose document has gone: a worker of the first
writes a stub after the second has read what the results index holds and before
it deletes. The window is narrow, both runs are unfinished while it is open --
so no consumer reads the root during it -- and the next pass over the root
removes the row. Leaving it alone is a decision rather than an oversight:
refusing or warning about a concurrent pass was considered and not done, and the
case for either would have to be made afresh. Run one pass over a root at a
time.

**Step 3 refuses to finish a root its tasks did not cover.** Step 1 records how
many files it found; step 3 adds up how many the tasks ingested, skipped and
refused, counting each task's report once. If the tasks account for fewer files
than the listing found -- a task that failed, timed out, or was never run --
the root is named, its run is left unfinished, and ``sd_results_index complete``
exits 1. Re-run the outstanding tasks and run step 3 again over a log holding the
re-run results; a task re-run over a share it already ingested reads nothing,
because its files match what the results index records, so it reports them as
skipped. A task that reports twice is still one task: the later report stands
in for the earlier one, and a share reported twice never covers for a share
that never ran. An account that runs past the listing is refused the same way:
with each task counted once the sum can only exceed the listing on a report
belonging somewhere else.

**Step 3 counts a task's result only for the root it was written under.** A
result names the run it belongs to and the root it wrote its rows under, and
both have to match. A run number is only unique inside the results index that
minted it, so a task file run after its index was deleted and rebuilt -- the
remedy for a schema-version mismatch -- names a run of whatever was built next.
Its shares add up correctly and their rows are somewhere else entirely, and a
root stamped on them would hold nothing at all. Such results are counted and
named in the summary rather than credited; the root they were meant for is left
unfinished, and step 1 over that root is what starts it again.

**Step 3 needs every task's result in the log it reads.** It reads one event log
and counts what is in it, so a root whose tasks are spread over several logs is
completed from the concatenation of them::

    cat worker-*.events.log > all-events.log

A task appearing in more than one of them is counted once, under the last report
of it in the file. Order therefore decides between two reports of one task that
disagree -- a task that failed on one worker and succeeded on another -- so a
concatenation that puts the failure last leaves the root unfinished, which the
same log concatenated the other way completes. Both outcomes are safe; neither
stamps a root whose documents were not read. A log naming only some of a root's
tasks leaves that root unfinished, which is the same outcome as tasks that never
ran and is corrected the same way.

**Step 3 refuses a root whose listing was never recorded.** A root that step 1
could not list -- mistyped, or an unmounted share -- gets no tasks and no record
of what it holds, and step 3 will not finish it: there is nothing for its tasks
to be measured against, and a root completed on that basis would report every
image under it as never navigated. Correct the root and run step 1 again.

``--force`` belongs to step 1, and ``complete`` does not offer it, because step 3
reads no document. A pass whose shares must ignore what the results index records
is one whose ``divide`` was run with ``--force``. ``--no-prune`` is absent from
step 3 on the same reasoning from the other side: step 3 removes no row, so
whether a root keeps the rows of documents that have left it was settled at step
1.

The tasks file is a JSON array in the shape a ``cloud_tasks`` queue loads. Each
entry has a ``task_id`` and a ``data`` object carrying ``run_id`` (the ingest
run the share belongs to), ``root_url`` (the normalized results root),
``force``, ``has_file_metrics`` (whether the listing reported a size and
modification time for every file), and ``files`` -- one object per document,
with its ``results_path_stub``, ``mtime_ns`` and ``size_bytes``. Every one of
those comes from the single listing, so no worker stats a file or checks for
one.

Step 3 reads the ``cloud_tasks`` event log, which is JSON Lines with one event
per line; the ``task_completed`` events carry what each worker returned, under
the ``task_id`` the task ran as. Lines that are not events are counted and
reported rather than refused, because an event log being appended to while it is
read ends in a partial line.

The closing summary of step 3 is the summary a single-process ingest writes:
files seen, ingested, skipped and refused, with the refusals tallied by reason
and one example file per reason. The reasons come back in the task results,
since a worker has no run log to write them in. Every file a share could not
read is named in its task result too, and the ones refused for something about
the document are recorded in the results index's ``failed_files`` table as well.

Rebuilding a results index
==========================

The results index carries the version of the column set that wrote it. Opening
one stamped with a different version fails, naming both versions: there are no
migrations, because ingest is cheap relative to navigation and entirely
reproducible from the tree, so the remedy for a version bump is to empty the
database and read the tree again.

.. code-block:: bash

    sd_results_index drop \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3
    sd_results_index ingest --nav-results-root /data/nav-offset-results \
        --results-index-db sqlite:////data/nav-offset-results/index.sqlite3

Under ``drop`` the exit status says whether the results index is gone: 0
when the tables went, and 0 again when the database held none of them, since an
index that is already absent is the state the command was asked for. It is 1
when the database could not be opened or read, when it holds tables of those
names that nothing proves are the index's, when a table would not drop, and when
whoever was asked answered anything but yes.

Starting a results tree over -- delete the results, navigate again, ingest again
-- has a counterpart on the results index, and it is a subcommand of the same
program:

.. code-block:: bash

    sd_results_index drop \
        --results-index-db postgresql+psycopg://user@dbhost/spindoctor

**It drops and stops.** No results root is read and no document is ingested, so
dropping is a deliberate act rather than the opening move of a long pass, and a
mistyped URL costs one command. It needs no ``--nav-results-root``: a drop is
about the database alone, and works on a machine that has the results index and
not the tree.

**It removes SpinDoctor's own tables and nothing else** -- ``images``,
``techniques``, ``feature_sources``, ``failed_files``, ``schema_meta`` and
``ingest_runs``, named one at a time. No schema is dropped and nothing is
matched by pattern. What makes those six SpinDoctor's own rather than six names
is the rule the ingest follows: it refuses to build a results index in a schema
holding any table it did not create, so a schema carrying SpinDoctor's stamp
holds this index and nothing else. No other table of that schema, and no other
schema of that database, is read or written, so an index shares a PostgreSQL
server, and a database, with whatever else lives in the other schemas.

**It drops only from a database that proves it holds a results index.** Those
six are among the commonest table names there are, so a table called ``images``
is not evidence of anything and is never removed for its name alone. What is
evidence is the index's own stamp: a ``schema_meta`` table carrying the columns
SpinDoctor's stamp carries. A database with no such stamp is refused, exits 1,
and has the tables that stopped it named -- because nothing distinguishes
somebody else's ``images`` from what is left of an index whose stamp has gone,
and a destructive command may not decide that on your behalf. Such tables are
removed by hand, or with the SQLite file that holds them.

**It drops from one schema: the one that stamp was found in.** A server resolves
an unqualified table name through a search path that may cross several schemas,
so the stamp is looked for once and every statement of the drop then names the
schema it was found in. A table of one of these names in any other schema
belongs to whoever put it there and is not touched. The schema is named in the
confirmation and in the run log. On SQLite it is always ``main``, the one
namespace a database file has.

**It confirms first.** The run log lists the tables with their row counts and
names the schema and the schema version, and the question -- which names the
index, its schema, how many tables and rows go with it, and any ingest run that
has not finished -- is written to standard output, which is where ``input``
writes a prompt. The answer is read without regard to case or surrounding space,
so ``y``, ``Y``, ``yes`` and ``YES`` all mean yes; anything else, Ctrl-C
included, leaves the results index alone and exits 1. ``--yes`` drops without
asking, for a run with nobody at the terminal -- and is required for one,
because a standard input with nothing to read is treated as a refusal rather
than as consent. Every refusal, the question itself, and the first line of the
account name the index URL with its password hidden; the lines that continue
that account carry the schema and the counts rather than repeating the URL.

**It does one thing, so it cannot be asked for two.** ``drop`` offers neither
``--force``, ``--no-prune`` and ``--nav-results-root`` nor either of the two
cloud-task paths: a drop reads no document and walks no tree, so each of those
was meant for a different subcommand. Typing one is a usage error naming the
option, before anything is opened and with a status of 2. A results root reaching
the program from ``NAV_RESULTS_ROOT`` or from the configuration is a machine's
standing setting rather than a request, and is simply unused. ``--yes`` is
offered by ``drop`` alone for the same reason from the other side: it answers a
question only the drop asks.

**It works on the databases nothing else will open**, which is the point: an
index stamped with a schema version this build does not read, or one whose stamp
holds something no version number could, is exactly what the drop is pointed at.
It also works on one whose columns are not this build's, since a stamp says which
version wrote a database rather than that nothing has happened to it since: the
account then leaves out whatever could not be read -- the schema version, the
count of unfinished ingest runs -- and drops the tables all the same. The drop
never refuses a database the other programs open.

**Dropping twice is not an error.** A database holding none of these tables is
not written at all, and says so; a results index that is already gone is the
state the command was asked for, so it exits 0. What that answer was established
over is what the connection reaches: an index in a schema outside this URL's
search path, or in one this account may not look into, reads the same way as one
that is not there, and the message says so rather than claiming the database
holds none. A database that is not there at all is a different answer: the
server refuses a PostgreSQL database it does not have, and a SQLite path that is
not there gets the same refusal rather than being created, so both exit 1 and
neither leaves an empty database behind.

**What is left behind is a database, not a hole.** Every consumer reads a
dropped index exactly as it reads one nobody has ever ingested into -- "not
ingested", with a message naming ``sd_results_index`` -- and the next
``sd_results_index ingest`` builds it again from the metadata documents. On
PostgreSQL the two states are literally the same database. On SQLite the file
itself remains, empty, and the drop deliberately does not delete it, so that one
subcommand means one thing on both backends. Deleting the file instead removes
the database rather than the results index, which every consumer reads the same
way but which a later ``sd_results_index drop`` refuses rather than reporting as
nothing to do.

**An interruption costs nothing.** The whole drop is one transaction on both
backends: PostgreSQL rolls DDL back with everything else, and on SQLite -- whose
driver would otherwise commit each ``DROP TABLE`` on its own -- the drop opens
its transaction itself, which SQLite's own transactional DDL then honors. Ctrl-C
partway through, a table that will not drop, a lost connection: each leaves the
database exactly as it was, still readable by every consumer, and the drop is
run again when the cause is dealt with. Ctrl-C is reported as the refusal it is
wherever it lands -- while the database is being opened, while what it holds is
being read, or during the drop -- naming which step stopped and exiting 1,
rather than printing a traceback.

Two things it does **not** refuse:

* **An ingest run that has not finished.** Such a run is either a pass writing
  the results index at this moment or one that died, and nothing recorded in the
  index tells the two apart. A pass that died is also the commonest reason to
  want a drop, so the count is reported in the confirmation rather than acted
  on. Dropping under a live pass ends that pass, which fails on a table that has
  gone; no reader is affected, because an unfinished run already reads as "not
  ingested" both before and after.
* **Another process holding the database.** Neither backend can be asked that
  question honestly, so the attempt is what answers it. Both the reading that
  precedes the question and the drop itself wait a bounded time for each table
  -- ``lock_timeout`` on PostgreSQL, the busy timeout on SQLite -- and then give
  up, and what they give up on is put back whole by the transaction around it.
  A failure names what the database said the cause was: a lock somebody holds, a
  view or another object depending on one of these tables, an account that does
  not own one of them, or, where the database gave no code this recognizes, its
  own words with no cause invented for them.

Sharing a results index
=======================

Several worker processes **on one machine** can write one SQLite index; there
is one file and no merge step, which is what the queue workflow above rests on.
Workers on several machines cannot, because a ``sqlite:`` URL names a local
path: a run spread across machines writes to PostgreSQL.

For reading, the same rule decides. A SQLite index serves every consumer that
can open the file it names; consumers on other machines need either their own
copy of the file or a PostgreSQL server they can all reach.

The results index schema
========================

Opening the results index directly is a supported way to answer questions the
standard report does not: the ``sqlite3`` command-line shell or ``psql``,
Python's :mod:`sqlite3` module or ``psycopg``, pandas, or a GUI browser. Six
tables hold the data.

An image is identified by the pair ``(root_url, results_path_stub)``:
``root_url`` is the navigation-results root in normalized form, and
``results_path_stub`` is the image's path under it with the ``_metadata.json``
suffix removed, for example
``COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB``. Two volumes may
hold images with the same basename, so the pair rather than the name is what
keys a row. ``techniques`` and ``feature_sources`` reference ``images`` by that
pair with ``ON DELETE CASCADE``.

``images`` -- one row per image:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``
     - TEXT
     - Navigation-results root this image was ingested from, normalized to
       one absolute, resolved form with no trailing separator. Half of the
       primary key.
   * - ``results_path_stub``
     - TEXT
     - The image's path under that root, without the ``_metadata.json``
       suffix. The other half of the primary key.
   * - ``subtree``
     - TEXT
     - First path segment of the stub, i.e. the top-level directory of the
       results tree the image sits under, e.g. ``COISS_2001``. NULL for a stub
       with no separator, which is what the simulated dataset produces.
   * - ``image_name``
     - TEXT
     - Image filename, e.g. ``N1454725799_1_CALIB.IMG``.
   * - ``instrument``
     - TEXT
     - ``coiss`` / ``vgiss`` / ``gossi`` / ``nhlorri`` / ``sim``, from the
       metadata document's ``observation.instrument`` field (required; a
       document without it is skipped as an ingest error).
   * - ``camera``
     - TEXT
     - The camera that took the image (``NAC`` / ``WAC`` / ``SSI`` /
       ``LORRI``), from the metadata document's ``observation.camera``
       field. Offset statistics group by this: pointing error belongs to
       the camera, not the spacecraft. NULL when the document carries no
       camera; it is never inferred from the image name.
   * - ``shutter_mode``
     - TEXT
     - How the exposure was commanded, from ``observation.shutter_mode``.
       NULL for a dataset that records none. Two cameras exposed together
       share one bus attitude that cannot honor two different corrections,
       and this is what tells the kernel writer an exposure was one of a pair.
   * - ``image_path``
     - TEXT
     - Where the run read the source image from at navigate time: its URL for
       remote holdings, its absolute path for local ones.
   * - ``image_et``
     - DOUBLE
     - Observation midtime as the navigation recorded it, TDB seconds past
       J2000, from ``navigation_result.provenance.image_et``. The report
       aggregates it into the time span it reports per instrument; a date bound
       compares ``image_date`` rather than this column. NULL for an image that
       never loaded, which built no observation and so has no provenance block,
       so an image whose navigation died for want of a SPICE kernel is placed
       nowhere in time and is passed over by any date bound and by the reported
       time span.
   * - ``image_date``
     - TEXT
     - UTC calendar date ``YYYY-MM-DD`` derived from ``image_et``; drives
       the ``--start-date`` / ``--end-date`` report filters.
   * - ``status``
     - TEXT
     - Navigation outcome: ``success``, ``failed``, ``conflicted``, or
       ``error`` (the last from image-load failures). Read from the document's
       own top-level ``status`` field and from nowhere else; ``unknown`` when
       that field is absent, empty, or not a string, so a document that named
       no outcome is never recorded as having named one.
   * - ``status_error``
     - TEXT
     - The fatal error that ended the run, e.g. ``missing_spice_data``.
       Stored verbatim: the SPICE-error image selection matches this token
       exactly.
   * - ``status_traceback``
     - TEXT
     - The traceback of that error, copied from the document whole, newlines
       and all. NULL for an image whose document recorded none, which is every
       image that navigated. It is what lets a query over a whole mission's
       failures say where each one arose without opening a per-image log.
   * - ``status_reason``
     - TEXT
     - The navigator's own explanation of the outcome: successes hold ``ok``
       or ``rank_1_only``, failures hold the failure reason (e.g.
       ``no_features_extracted``). A different vocabulary from
       ``status_error``, in its own column; a failure is described by
       whichever of the two the document carried.
   * - ``offset_dv``, ``offset_du``
     - DOUBLE
     - Fused pointing offset in pixels (V then U), stored exactly as the
       document's top-level ``offset`` holds it. This is the number every
       consumer applies. NULL when navigation found no offset.
   * - ``sigma_dv``, ``sigma_du``
     - DOUBLE
     - Per-axis 1-sigma uncertainty of the fused offset, pixels.
   * - ``covariance_px2``
     - JSON
     - The fused covariance in pixels squared, square and row-major, as the
       document wrote it: ``[[vv, vu], [vu, uu]]``, or 3x3 for a twist-fitted
       result whose third row and column are the rotation's. Stored whole,
       because the offset-to-rotation cross terms in that third row and column
       are stated nowhere else. NULL where the recorded value is not a square
       matrix of real numbers.
   * - ``sigma_along_unobservable_px``
     - DOUBLE
     - Uncertainty along the direction the scene could not constrain.
   * - ``rotation_deg``, ``sigma_rotation_deg``
     - DOUBLE
     - Fitted camera twist and its uncertainty, present only where twist was
       fitted.
   * - ``confidence``
     - DOUBLE
     - Fused confidence in ``[0, 1]``, from the document's top-level
       ``confidence``.
   * - ``confidence_rank``
     - TEXT
     - Confidence tier label assigned by the ensemble.
   * - ``n_techniques``
     - INTEGER
     - Number of per-technique results recorded for the image.
   * - ``excluded_from_consensus``
     - JSON
     - Technique names the ensemble excluded as outliers (``[]`` when none),
       in the order the document recorded them: a recorded list is stored as
       recorded. Sort them in the query where some other order is wanted.
   * - ``image_class``
     - TEXT
     - Image-classifier verdict (e.g. ``clean``).
   * - ``noise_sigma``
     - DOUBLE
     - Image-classifier noise estimate.
   * - ``image_shape_v``, ``image_shape_u``
     - INTEGER
     - Pixel dimensions of the image data (V then U), from the metadata
       document's ``observation.image_shape`` field; NULL when the image
       never loaded.
   * - ``run_start``, ``run_end``
     - TEXT
     - UTC ISO8601 run start and end of the navigation of this image, from
       the metadata document's ``timing`` section; start is captured before
       the image load and end after navigation (or at error time).
   * - ``elapsed_s``
     - DOUBLE
     - Wall-clock seconds between ``run_start`` and ``run_end``.
   * - ``peak_memory_bytes``
     - BIGINT
     - Largest resident size the navigating process reached while navigating
       this image, from the metadata document's ``timing`` section. This is
       the figure an out-of-memory kill is decided against, so it is what
       sizes a worker: for a process that handled several images it is
       measured from a floor including what the earlier ones left resident,
       which is the memory the worker had to hold at that moment. NULL where
       the run recorded none: a system whose kernel publishes no peak, or one
       that would not let the mark be reset ahead of the image.
   * - ``config_hash``
     - TEXT
     - sha256 of the fully-resolved configuration used for the run.
   * - ``git_sha``
     - TEXT
     - Short git SHA of the navigating code (``-dirty`` suffix when the
       tree had uncommitted changes).
   * - ``pipeline_run``
     - TEXT
     - UTC ISO8601 timestamp of the navigation run.
   * - ``spice_kernels``
     - JSON
     - The kernel basenames the run recorded, in the order recorded. A
       corrected C-kernel overlays one original, and these are what say which
       originals an attitude was measured against. An empty list is a run that
       recorded none; NULL is a document with no provenance block, and also a
       block holding anything but a list of names, which its readers refuse.
   * - ``image_number``
     - BIGINT
     - Numeric portion of the image name (the first digit run in the
       basename). What the ``--min-image`` / ``--max-image`` filters compare.
   * - ``start_et``, ``stop_et``, ``midtime_et``, ``exposure_s``
     - DOUBLE
     - Shutter open and close epochs, the exposure midtime, and the exposure
       between them. ``midtime_et`` is stored as the navigator recorded it,
       because a reader that applies a recorded attitude checks it against the
       observation's own midtime to a microsecond.
   * - ``sclk_start``, ``sclk_midtime``, ``sclk_stop``
     - TEXT
     - The same three instants as spacecraft-clock strings.
   * - ``camera_frame``
     - TEXT
     - SPICE name of the frame a recorded attitude is expressed in, from
       ``navigation_result.pointing.camera_frame``. A kernel writer looks it
       up among the frame kernels it furnishes; a reader that gates an attitude
       against the observation takes the frame identity from the observation
       and never consults this name.
   * - ``camera_frame_id``, ``ck_frame_id``
     - INTEGER
     - SPICE frame identifiers of the camera and of the C-kernel a corrected
       attitude targets. Unlike the name above, both are read back by every
       consumer that rebuilds a pointing block.
   * - ``cmatrix``, ``cmatrix_original``
     - JSON
     - Corrected and as-flown camera attitude, nine floats row-major.
       ``cmatrix`` is absent where the navigation fitted a camera rotation.
       Absent means SQL NULL here and in every JSON column, so
       ``WHERE cmatrix IS NOT NULL`` selects the images that carry one; an
       empty list or object, where one appears, is a value rather than an
       absence.
   * - ``source_file``
     - TEXT
     - Path or URL of the ingested metadata document.
   * - ``mtime_ns``, ``size_bytes``
     - BIGINT
     - Modification time and size of that document as the ingest listing
       reported them. A later pass compares these to decide whether the
       document has to be read again.

``techniques`` -- one row per technique result per image:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``, ``results_path_stub``
     - TEXT
     - Foreign key into ``images``. Unique together with
       ``technique_name``: a technique reports once for an image.
   * - ``technique_name``
     - TEXT
     - Technique class name (e.g. ``BodyLimbNav``, ``StarUniqueMatchNav``).
   * - ``offset_dv``, ``offset_du``
     - DOUBLE
     - The technique's own offset estimate, pixels.
   * - ``covariance_px2``
     - JSON
     - The technique's own covariance in pixels squared, square and row-major,
       exactly as on ``images``. The per-axis 1-sigma pair is the square root of
       its diagonal; the off-diagonal terms state the correlation between the
       axes, which no pair of sigmas carries.
   * - ``confidence``
     - DOUBLE
     - The technique's calibrated confidence in ``[0, 1]``.
   * - ``spurious``
     - BOOLEAN
     - True when the technique flagged its own result as spurious.
   * - ``at_edge``
     - BOOLEAN
     - True when the fit landed at the edge of its search space.
   * - ``source_names``
     - JSON
     - Body / ring / catalog names the technique used.
   * - ``diagnostics``
     - JSON
     - The technique's full diagnostics dataclass as a JSON object.

``feature_sources`` -- per image, feature counts grouped by source:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Column
     - Type
     - Meaning
   * - ``root_url``, ``results_path_stub``
     - TEXT
     - Foreign key into ``images``. Unique together with ``feature_type``,
       ``source_model`` and ``source_name``.
   * - ``feature_type``
     - TEXT
     - Feature type (e.g. ``BODY_DISC``, ``STAR``, ``RING_EDGE``).
   * - ``source_model``
     - TEXT
     - :class:`~spindoctor.nav_model.NavModel` family that produced the
       features (``body``, ``rings``, ``stars``, ``titan``).
   * - ``source_name``
     - TEXT
     - Body, ring, or catalog name (e.g. ``IAPETUS``, ``UCAC4``).
   * - ``n_features``
     - INTEGER
     - Features of this type/source extracted for the image.
   * - ``n_gated``
     - INTEGER
     - How many of them the reliability gate removed.

``ingest_runs`` records one row per ingest pass over one root: the root, when
the pass started and finished (``finished_utc`` is NULL while it is running),
how many files it saw, ingested, skipped as unchanged, could not read and
removed, and the schema version it wrote. A root whose newest row has no finish
time, or which has no row at all, has not been fully ingested, and a consumer
says so rather than reading absence of rows as "nothing was navigated". A row
that does carry a finish time covers the whole of its root, because a pass that
could not list a directory stops rather than finishing.

``failed_files`` records one row per file that is not a current-schema
navigation document: the root and stub that identify it, the subtree it is
under, the reason it was refused, and the size and modification time it had when
it was read. It is what lets a second pass skip it. It is deliberately not an
``images`` row, because a file with no usable data must not answer the question
``images`` exists to answer.

``reason`` names one of two things. ``unreadable``, ``not valid JSON`` and ``not
a JSON object`` each name a file from which no JSON object was produced. A
reason beginning ``not a current-schema navigation document`` names a JSON
object that was produced and that this schema will not take, and says in
parentheses which field said so. Every one of the reasons is the reason a reader
of the results tree gives for the same file, so the two never describe one
file's fault two different ways, and a file refused for any of them yields no
per-image facts to either storage: it counts as a file that exists, for the
presence filters and for nothing else.

``schema_meta`` holds a single row stamping the database with the column-set
version that created it. A results index whose stamp is not the version this
build reads is refused at open, naming both; the remedy is the rebuild described
under `Rebuilding a results index`_.

Indexes exist on ``images(results_path_stub)``, ``images(image_date)``,
``images(instrument)``, and ``ingest_runs(root_url)``, plus the uniqueness
constraints on the two child tables.

Querying a results index directly
=================================

Success rate per instrument:

.. code-block:: sql

    SELECT instrument,
           COUNT(*) AS images,
           AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate
    FROM images
    GROUP BY instrument;

The ten largest fused offsets, with their confidence tier:

.. code-block:: sql

    SELECT image_name, offset_dv, offset_du, confidence_rank
    FROM images
    WHERE status = 'success'
    ORDER BY GREATEST(ABS(offset_dv), ABS(offset_du)) DESC
    LIMIT 10;

SQLite has no ``GREATEST``; its two-argument ``MAX`` does the same thing:

.. code-block:: sql

    ORDER BY MAX(ABS(offset_dv), ABS(offset_du)) DESC

The failure reason of every non-successful image, whichever of the two
vocabularies described it:

.. code-block:: sql

    SELECT COALESCE(status_reason, status_error) AS reason,
           instrument,
           COUNT(*) AS images
    FROM images
    WHERE status != 'success'
    GROUP BY reason, instrument
    ORDER BY images DESC;

JSON columns unpack with each backend's own JSON functions -- for example,
counting how often each technique was excluded from the consensus. On PostgreSQL
the columns holding lists of names (``excluded_from_consensus``,
``spice_kernels``, ``techniques.source_names``) are ``jsonb`` and take the
``jsonb_`` functions; the columns that can hold numbers (``covariance_px2``,
``cmatrix``, ``cmatrix_original``, ``techniques.diagnostics``) are ``json``,
which returns every stored number exactly and takes the ``json_`` functions.
Cast with ``::jsonb`` where a ``jsonb_`` function is wanted over one of those.
On SQLite:

.. code-block:: sql

    SELECT excluded.value AS technique, COUNT(*) AS images
    FROM images, json_each(images.excluded_from_consensus) AS excluded
    GROUP BY excluded.value
    ORDER BY COUNT(*) DESC;

and on PostgreSQL, where the column is ``jsonb``:

.. code-block:: sql

    SELECT technique, COUNT(*) AS images
    FROM images,
         jsonb_array_elements_text(images.excluded_from_consensus) AS technique
    GROUP BY technique
    ORDER BY COUNT(*) DESC;

A child table joins to its image on the pair that keys one:

.. code-block:: sql

    SELECT t.technique_name, i.instrument, AVG(t.confidence)
    FROM techniques t
    JOIN images i
      ON i.root_url = t.root_url
     AND i.results_path_stub = t.results_path_stub
    WHERE NOT t.spurious
    GROUP BY t.technique_name, i.instrument;

The results index loads straight into pandas from either backend:

.. code-block:: python

    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine('sqlite:////data/nav-offset-results/index.sqlite3')
    images = pd.read_sql_query('SELECT * FROM images', engine)
    per_technique = pd.read_sql_query(
        'SELECT t.*, i.instrument, i.image_date '
        'FROM techniques t JOIN images i USING (root_url, results_path_stub)',
        engine,
    )
    # Median per-technique confidence by instrument:
    print(per_technique.groupby(['instrument', 'technique_name'])['confidence'].median())

Where to look next
==================

- :doc:`user_guide_statistics` --- the report a results index is read into:
  ``sd_stats_report``'s options, its filters, and every section it writes.
- :doc:`user_guide_navigation` --- the selection filters, and what an
  index-answered selection holds that a tree-walked one does not.
- :doc:`user_guide_backplanes` and :doc:`user_guide_reprojection` --- what each
  stage reads per image, and what it reports when the index cannot answer.
- :doc:`/introduction_configuration` --- the configuration file and the
  environment variables the URL is resolved from.
