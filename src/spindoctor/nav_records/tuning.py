"""How much of a pass over a results tree runs at once.

A pass over a cloud root is latency rather than bandwidth: a listing is one
round trip, a document is another, and a navigation document is a few
kilobytes.  So the walk lists several directories at once and retrieval fetches
several documents at once, and this is how many of each.

The useful values belong to a machine, its link to the root and what the
service will do concurrently, not to this program, so they are configuration:
the ``results_tree`` section carries them, and the defaults here are what that
section ships with.  A program resolves the section once, through
:func:`spindoctor.config.get_results_tree_tuning`, and passes the result down
the way it passes its logger.  This package reads no configuration itself.
"""

from dataclasses import dataclass, fields

__all__ = ['TreeTuning']


@dataclass(frozen=True)
class TreeTuning:
    """How much of a pass over a results tree runs at once.

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
    """

    walk_threads: int = 32
    walk_directories_at_once: int = 256
    retrieve_threads: int = 64
    retrieve_batch_size: int = 1024
    ingest_commit_batches: int = 2

    def __post_init__(self) -> None:
        """Refuse a value no pass can run at.

        Raises:
            ValueError: If a setting is not a positive integer, or if a round
                of work is smaller than the pool it feeds, which leaves threads
                idle at every setting rather than running slowly.  The message
                names the setting.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{field.name} must be a positive integer; got {value!r}')
        for pool, work in (
            ('walk_threads', 'walk_directories_at_once'),
            ('retrieve_threads', 'retrieve_batch_size'),
        ):
            if getattr(self, work) < getattr(self, pool):
                raise ValueError(
                    f'{work} must be at least {pool}, or the pool never fills; '
                    f'got {getattr(self, work)} against {getattr(self, pool)} threads'
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
