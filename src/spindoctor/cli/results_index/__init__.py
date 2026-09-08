"""Read a navigation results tree into the results index.

One pass over a results root reads every ``*_metadata.json`` document under it
and writes one ``images`` row per document, with its per-technique and feature
inventory rows beside it.  Nothing else reads the tree afterwards: a consumer
that needs a few fields per image reads a row.

A pass may also be divided into cloud tasks: one program lists each root and
hands out shares of it, a worker ingests a share, and the shares' tallies are
added up to complete the run.  A worker removes no row, because nothing hands
it the complete listing that alone licenses one.

Public API:

    INGEST_TASK_SHARE_SIZE     -- metadata files handed to one cloud task
    IngestCounts               -- what one pass did
    ingest_metadata_files      -- the pass itself, over one or more roots
    FanOut                     -- the tasks one root divides into
    fan_out_ingest_tasks       -- list each root once and divide it up
    ingest_task_share          -- ingest one task's share of a root
    TaskResult                 -- what one task returned, and which task it was
    TaskResults                -- what a worker event log holds
    task_results_from_event_log -- read the workers' return values back
    TaskCompletion             -- what adding the shares up did
    complete_ingest_tasks      -- add them up and stamp the runs

The implementation is split by stage, and this module re-exports the whole
surface so a consumer imports from ``spindoctor.cli.results_index`` whichever
stage a name lives in:

* :mod:`~spindoctor.cli.results_index.counts` -- the tally a pass keeps and the
  summary it is read from.
* :mod:`~spindoctor.cli.results_index.store` -- what the index already holds
  about a root, and how rows go back into it.
* :mod:`~spindoctor.cli.results_index.chunks` -- batched retrieval, reading a
  document into rows, and the per-chunk write.
* :mod:`~spindoctor.cli.results_index.runs` -- the record of one pass over one
  root, which is what makes absence of a row readable.
* :mod:`~spindoctor.cli.results_index.driver` -- the pass itself: list, select,
  ingest, prune, complete.  The listing is
  :class:`~spindoctor.nav_records.TreeRecordSource`'s, collected whole here
  because the prune is licensed by holding all of it.
* :mod:`~spindoctor.cli.results_index.tasks` -- the same pass divided into cloud
  tasks: fan out, ingest a share, add the shares up.

Emptying an index is the package's other command and is no part of that
surface: :mod:`~spindoctor.cli.results_index.drop` removes the index's own
tables from the database a URL names and walks no tree, so the program's
dispatch module reaches it directly.
"""

from spindoctor.cli.results_index.counts import IngestCounts
from spindoctor.cli.results_index.driver import ingest_metadata_files
from spindoctor.cli.results_index.store import UnwritableRowError
from spindoctor.cli.results_index.tasks import (
    INGEST_TASK_SHARE_SIZE,
    FanOut,
    TaskCompletion,
    TaskResult,
    TaskResults,
    complete_ingest_tasks,
    fan_out_ingest_tasks,
    ingest_task_share,
    task_results_from_event_log,
)

__all__ = [
    'INGEST_TASK_SHARE_SIZE',
    'FanOut',
    'IngestCounts',
    'TaskCompletion',
    'TaskResult',
    'TaskResults',
    'UnwritableRowError',
    'complete_ingest_tasks',
    'fan_out_ingest_tasks',
    'ingest_metadata_files',
    'ingest_task_share',
    'task_results_from_event_log',
]
