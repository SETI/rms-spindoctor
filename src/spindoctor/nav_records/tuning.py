"""How much of a pass over a results tree runs at once.

A pass over a cloud root is latency rather than bandwidth: a listing is one
round trip, a document is another, and a navigation document is a few
kilobytes.  So the walk lists several directories at once and retrieval fetches
several documents at once, and this is how many of each.  A tree on a local
directory pays no round trip for either, so there the thread and batch settings
have no useful effect; only the transaction size, which is about the index an
ingest writes rather than the tree it reads, applies to every storage.

The useful values belong to a machine, its link to the root and what the
service will do concurrently, not to this program, so they are configuration:
the ``results_tree`` section carries them, and the configuration parses that
section into a :class:`TreeTuning` when it loads.  Every value lives in the
packaged YAML, so the type has no defaults.  A program reads the parsed section
once, as ``config.results_tree``, and passes it down.  This package reads no
configuration itself.
"""

from dataclasses import dataclass

from spindoctor.schema import SCHEMA_CONFIG, PositiveInt, RuleError

__all__ = ['TreeTuning']


@dataclass(frozen=True)
class TreeTuning:
    """How much of a pass over a results tree runs at once.

    This is the ``results_tree`` configuration section, which the configuration parses
    into this type when it loads.  Every setting is a positive integer.

    Parameters:
        walk_threads: How many directories are listed at the same time.
        walk_directories_at_once: How many directories one round of the walk
            lists before it moves on.  A round holds every entry of every
            directory it lists, so this bounds what a round holds rather than
            how fast it runs.
        retrieve_threads: How many documents are downloaded at the same time.
        retrieve_batch_size: How many documents are handed to the downloader
            at a time.
        ingest_commit_batches: How many retrieval batches one database
            transaction of an ingest covers.  Read by the results-index ingest
            alone; a reader of the tree writes nothing.

    Raises:
        RuleError: If a round of work is smaller than the pool it feeds:
            ``walk_directories_at_once`` below ``walk_threads``, or
            ``retrieve_batch_size`` below ``retrieve_threads``.  Threads would then
            be idle at every setting rather than running slowly.  The problem is
            put on the round's key.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    walk_threads: PositiveInt
    walk_directories_at_once: PositiveInt
    retrieve_threads: PositiveInt
    retrieve_batch_size: PositiveInt
    ingest_commit_batches: PositiveInt

    def __post_init__(self) -> None:
        """Refuse a round of work smaller than the pool it feeds."""
        for pool, work in (
            ('walk_threads', 'walk_directories_at_once'),
            ('retrieve_threads', 'retrieve_batch_size'),
        ):
            if getattr(self, work) < getattr(self, pool):
                raise RuleError(
                    f'must be at least {pool}, or the pool never fills; '
                    f'got {getattr(self, work)} against {getattr(self, pool)} threads',
                    at=work,
                    reads=(pool, work),
                )

    @property
    def ingest_commit_chunk_size(self) -> int:
        """How many documents one ingest transaction covers.

        A multiple of the retrieval batch rather than a count of its own, so a
        transaction is never smaller than the batch it is retrieved in: that
        would cap the batch at the transaction and quietly undo the setting.

        Returns:
            The retrieval batch size times the number of batches a transaction
            covers.
        """
        return self.retrieve_batch_size * self.ingest_commit_batches
